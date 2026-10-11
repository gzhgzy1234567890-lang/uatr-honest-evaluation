#!/usr/bin/env python3
"""W3 runner — one benchmark configuration, appended to results.csv.

Why one configuration per process
----------------------------------
A Kaggle session gives 12 h and the weekly budget is 30 GPU-hours.  One
invocation therefore runs exactly ONE (corpus, representation, backbone,
protocol, seed, method) tuple, appends a row to results.csv, and exits.  Nothing
is recomputed, and an interrupted sweep resumes by re-running the notebook: the
configurations whose rows already exist are skipped by the driver.

Every run also writes per-clip test predictions, carrying the recording id, the
vessel id and the source duration of each clip.  Without those files a
recording-level bootstrap confidence interval cannot be computed later, and a
single point estimate is exactly the practice this paper argues against.

The three identity levels
-------------------------
    clip        one 5 s row
    recording   (mmsi, AIS timestamp) — one acoustic session of one ship
    vessel      mmsi — the ship itself

Protocols differ ONLY in which of these is the unit of the split.  The model,
optimiser, schedule, epoch count and seed handling are identical across all of
them, because that identity is the only reason a difference between protocols is
interpretable at all.

Usage
-----
    python w3_run.py --data /kaggle/input/under_water_features \
        --corpus oceanship --rep logmel --backbone cnn_small \
        --protocol vessel_disjoint --seed 0

    python w3_run.py ... --method irm --domain date        # C4 invariant training
    python w3_run.py ... --shuffle-labels                  # required control
    python w3_run.py ... --train-corpus oceanship --test-corpus deepship --vocab common4
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover
    raise SystemExit("this runner needs torch")


# --------------------------------------------------------------------------- #
# torch version shims
#
# Kaggle ships torch 2.10, where the modern names exist.  A local CPU box may
# still have 1.9, where they do not.  Keeping the runner runnable on both is what
# makes "smoke it locally before spending GPU hours" possible, and that habit has
# already paid for itself several times in this project.
# --------------------------------------------------------------------------- #
def autocast(enabled: bool):
    if enabled and hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", enabled=True)
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cpu", enabled=False)
    return torch.cuda.amp.autocast(enabled=enabled)  # torch < 1.10


class _NoScaler:
    """Stand-in for GradScaler when running on CPU."""

    def scale(self, x):
        return x

    def step(self, opt):
        opt.step()

    def update(self):
        pass


def grad_scaler(enabled: bool):
    if not enabled:
        return _NoScaler()
    try:
        return torch.amp.GradScaler(enabled=True)  # torch >= 1.10
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=True)


# --------------------------------------------------------------------------- #
# corpus constants — the only place corpus-specific knowledge lives
# --------------------------------------------------------------------------- #
CORPORA = ("oceanship", "deepship", "shipsear")

LABEL_COL = {"oceanship": "label", "deepship": "class_dir", "shipsear": "class_dir"}
REC_COL = {"oceanship": "rec_key", "deepship": "rec_key", "shipsear": "group"}
VESSEL_COL = {"oceanship": "mmsi", "deepship": "ship_name", "shipsear": "group"}

# Protocol names differ per corpus.  Oceanship carries a real ship identifier so
# it gets the vessel-disjoint protocol; DeepShip's vessel id (ship_name) exists in
# the index but its frozen split file only ships the three original protocols, so
# `vessel_disjoint` is accepted for DeepShip as well and requires an extended
# split file built by w1_02b_deepship_vessel_splits.py and passed via --split-file.
# Hard-coding a single name makes the other corpus return zero rows — silently.
PROTOCOLS = {
    "oceanship": ("clip_random", "rec_random", "vessel_disjoint"),
    "deepship": ("clip_random", "rec_random", "temporal", "vessel_disjoint"),
    "shipsear": ("group_disjoint",),
}
SPLIT_FILE = {"oceanship": "oceanship_splits.csv.gz", "deepship": "deepship_splits.csv.gz"}
SPLIT_KEY = {"oceanship": ["clip_id"], "deepship": ["rec_key", "clip_idx"]}

# the four classes DeepShip and Oceanship have in common
COMMON4 = ("Cargo", "Passenger", "Tanker", "Tug")

# ShipsEar's class directories are 0..4 and their meaning is not stored in the
# files, so it comes from the literature.  Independently cross-checked: our copy
# holds 369/301/843/486/224 clips per class, matching MGFGNet (Front. Mar. Sci.
# 2023) Table 3 class for class.
#   0 -> fishing boats, trawlers, mussel boats, tugboats, dredgers
#   1 -> motoboats, pilot boats, sailboats
#   2 -> passenger ferries
#   3 -> ocean liners, Ro-Ro vessels
#   4 -> background noise (not a vessel at all)
SHIPSEAR_CLASS_NAMES = {
    "0": "Fishing/Trawler/Tug",
    "1": "Motor/Pilot/Sail",
    "2": "Passenger",
    "3": "OceanLiner/RoRo",
    "4": "BackgroundNoise",
}
# Only 2 and 0 have a defensible counterpart in the other corpora, and even
# class 0 is a coarse bucket.  The approximation is a printed warning and a
# `vocab_approx` column, never a silent assumption.
SHIPSEAR_TO_COMMON4 = {"2": "Passenger", "0": "Tug"}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def open_text(path: Path):
    """Open a possibly-gzipped text file by inspecting its magic bytes.

    Kaggle expands `.csv.gz` to `.csv` when it ingests a dataset, so a
    name-based decision is wrong in one of the two cases.  The first two bytes
    never lie.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(2)
    if head == b"\x1f\x8b":
        return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")
    return open(path, "r", newline="", encoding="utf-8", errors="replace")


def any_text_file(cands) -> Path | None:
    """First existing candidate, with `.gz` and the expanded spelling both counting."""
    for c in cands:
        if c is None:
            continue
        c = Path(c)
        for p in (c, Path(str(c)[:-3]) if str(c).endswith(".gz") else None):
            if p is not None and p.exists():
                return p
    return None


def resolve_features_root(p: Path) -> Path:
    """The mounted dataset may hold the feature tree directly or one level down."""
    p = Path(p)
    for cand in (p, *sorted(p.glob("*/")), *sorted(p.glob("*/*/"))):
        if (cand / "oceanship" / "index.csv").exists():
            return cand
    for hit in list(p.glob("**/oceanship/index.csv"))[:1]:
        return hit.parent.parent
    raise SystemExit(
        f"no feature tree under {p}: expected <root>/oceanship/index.csv. "
        f"top level: {[c.name for c in sorted(p.glob('*'))][:20]}"
    )


def protocol_for(corpus, protocol, train_corpus=None, test_corpus=None) -> str:
    """The protocol name a run will actually be recorded under.

    A cross-corpus run ignores `--protocol` entirely, so a driver that computes
    run ids from the requested protocol would never recognise a completed
    cross-corpus run and would re-run it every session.  Both the runner and the
    drivers resolve the name through here.
    """
    if train_corpus and test_corpus:
        return f"cross_{train_corpus}to{test_corpus}"
    return protocol


def run_id_for(corpus, rep, backbone, protocol, seed, domain="none", method="erm",
               shuffled=False, tag="", split_seed=None, init_seed=None) -> str:
    """The canonical run id.

    Lives here, not in the sweep drivers, so a driver can tell whether a
    configuration is already done without re-implementing the naming rule — a
    duplicated rule that drifts is how a sweep silently re-runs everything.

    When the split draw and the initialisation are decoupled (neither equals
    `seed`), the id carries both explicitly so the same split with different
    inits — and the same init with different splits — are distinct runs and the
    dedup logic can tell them apart.
    """
    suffix = ""
    if domain != "none":
        suffix += f"__{domain}"
    if method != "erm":
        suffix += f"__{method}"
    if shuffled:
        suffix += "__shuf"
    if tag:
        suffix += f"__{tag}"
    s = f"s{seed}"
    if split_seed is not None and init_seed is not None and \
            (split_seed != seed or init_seed != seed):
        s += f"_sp{split_seed}_in{init_seed}"
    return f"{corpus}__{rep}__{backbone}__{protocol}__{s}{suffix}"


def stable_int(*parts) -> int:
    """Deterministic seed.  Never the builtin hash(): it is salted per process."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


# --------------------------------------------------------------------------- #
# metadata / splits
# --------------------------------------------------------------------------- #
def load_index(root: Path, corpus: str) -> list[dict]:
    with (root / corpus / "index.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{corpus}/index.csv is empty")
    return rows


def load_native_split_file(root: Path, corpus: str, protocol: str, seed: int,
                           split_override: str = ""):
    """Read one (protocol, seed) out of the W1 split file.

    Filtering while reading is not an optimisation, it is the only correct
    option.  The file holds 3 protocols x 5 seeds = 15 rows per clip, so a dict
    keyed on the clip identity alone keeps just the LAST row for each clip and
    silently returns nothing for every other protocol and seed.  On the real
    Oceanship file that would mean 14 of the 15 protocols quietly producing an
    empty split, with no error anywhere.

    `split_override` points at a split file outside the frozen feature dataset
    (e.g. DeepShip vessel_disjoint appended by w1_02b into /kaggle/working).

    Returns (assign, keys, seen) where `seen` counts the (protocol, seed) pairs
    actually present, so a wrong pair is reported rather than ignored.
    """
    name = SPLIT_FILE.get(corpus)
    if name is None:
        return None
    cands = [Path(split_override)] if split_override else [root / "splits" / name]
    path = any_text_file(cands)
    if path is None:
        if split_override:
            raise SystemExit(f"--split-file {split_override!r} does not exist")
        return None
    keys = SPLIT_KEY[corpus]
    assign: dict[tuple, str] = {}
    seen: Counter = Counter()
    with open_text(path) as fh:
        for r in csv.DictReader(fh):
            seen[(r["protocol"], int(r["seed"]))] += 1
            if r["protocol"] == protocol and int(r["seed"]) == int(seed):
                assign[tuple(r[k] for k in keys)] = r["split"]
    return assign, keys, seen


def split_lookup(assign: dict, corpus: str, row: dict):
    return assign.get(tuple(row[k] for k in SPLIT_KEY[corpus]))


def group_split(rows: list[dict], corpus: str, seed: int) -> tuple[dict[str, str], list[str]]:
    """70/15/15 over identity groups, stratified by class; built here, not shipped.

    ShipsEar is the only user: its public copy stores clips under 12 *ship-type*
    groups with the original 90 recordings' provenance erased, so a group is the
    finest unit available.  A class with fewer than three groups is assigned
    wholly to train — it cannot fill three splits, and letting it produce an
    empty test set is exactly the kind of quiet flaw this paper is about.  The
    dropped classes are returned and reported, never hidden.
    """
    rec_col = REC_COL[corpus]
    grp_class: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        grp_class[r[rec_col]][r[LABEL_COL[corpus]]] += 1

    by_class: dict[str, list[str]] = defaultdict(list)
    for g, c in grp_class.items():
        by_class[c.most_common(1)[0][0]].append(g)

    rng = random.Random(stable_int("group", corpus, seed))
    assign: dict[str, str] = {}
    dropped: list[str] = []
    for cls, groups in sorted(by_class.items()):
        groups = sorted(groups)
        rng.shuffle(groups)
        n = len(groups)
        if n < 3:
            dropped.append(f"{cls}:{n}")
            for g in groups:
                assign[g] = "train"
            continue
        n_train = min(int(round(n * 0.70)), n - 2)
        n_val = max(1, min(int(round(n * 0.15)), n - n_train - 1))
        for i, g in enumerate(groups):
            assign[g] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return assign, dropped


def record_split(rows: list[dict], corpus: str, seed: int) -> dict[str, str]:
    """70/15 holdout by recording, stratified by class — used by the cross-corpus runs."""
    rec_col = REC_COL[corpus]
    rec_class: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        rec_class[r[rec_col]][r[LABEL_COL[corpus]]] += 1
    by_class: dict[str, list[str]] = defaultdict(list)
    for k, c in rec_class.items():
        by_class[c.most_common(1)[0][0]].append(k)
    rng = random.Random(stable_int("rec70", corpus, seed))
    assign: dict[str, str] = {}
    for cls, keys in sorted(by_class.items()):
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        if n < 3:
            for k in keys:
                assign[k] = "train"
            continue
        n_train = min(int(round(n * 0.70)), n - 2)
        n_val = max(1, min(int(round(n * 0.15)), n - n_train - 1))
        for i, k in enumerate(keys):
            assign[k] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return assign


# --------------------------------------------------------------------------- #
# label vocabulary and domain labels
# --------------------------------------------------------------------------- #
def label_for(corpus: str, row: dict, vocab: str) -> str | None:
    raw = row.get(LABEL_COL[corpus], "") or ""
    if not raw:
        return None
    if vocab == "native":
        return SHIPSEAR_CLASS_NAMES.get(raw, raw) if corpus == "shipsear" else raw
    if corpus == "shipsear":
        return SHIPSEAR_TO_COMMON4.get(raw)
    return raw if raw in COMMON4 else None


DATE_RE = re.compile(r"(\d{4})[-/]?(\d{2})")


def domain_for(corpus: str, row: dict, spec: str, dur_bounds) -> str:
    if spec == "none":
        return "d0"
    if spec == "official":
        return row.get("official_split") or "unknown"
    if spec == "date":
        text = row.get(REC_COL[corpus], "") or ""
        m = DATE_RE.search(text)
        if m:
            return f"{m.group(1)}-{m.group(2)}"
        for chunk in re.split(r"[^0-9]", text):
            if len(chunk) >= 9 and chunk.isdigit() and 1e9 < int(chunk[:10]) < 2e9:
                return time.strftime("%Y-%m", time.gmtime(int(chunk[:10])))
        d = row.get("date", "")
        return d[:7] if d else "unknown"
    if spec == "dur":
        d = float(row.get("src_dur_s") or 0.0)
        for i, edge in enumerate(dur_bounds or []):
            if d <= edge:
                return f"q{i}"
        return f"q{len(dur_bounds or [])}"
    if spec == "vessel_hash":
        # NEGATIVE CONTROL: real partitions of the training set that carry no
        # acoustic meaning.  An invariance gain must NOT appear here; if it does,
        # the gain came from the regulariser rather than from the domain
        # structure, which would invalidate the claim.
        key = row.get(VESSEL_COL[corpus]) or row.get(REC_COL[corpus]) or "?"
        h = int(hashlib.md5(str(key).encode()).hexdigest()[:8], 16)
        return f"h{h % 8}"
    raise SystemExit(f"unknown --domain {spec!r}")


def duration_quartiles(rows: list[dict]) -> list[float]:
    d = np.array([float(r.get("src_dur_s") or 0.0) for r in rows])
    return [float(x) for x in np.quantile(d, [0.25, 0.50, 0.75])] if len(d) else [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class ClipDataset(Dataset):
    """Serves (1, 128, 313) float32 straight out of the memory-mapped cache.

    Reading from the memmap per item keeps RSS flat: the Oceanship array alone is
    4.3 GB per representation, and both are mounted at once.

    `norm` fixes a real failure mode.  The arrays are dB relative to each clip's
    peak, floored at -80, so the first convolution sees values around -40 with a
    spread of 20.  BatchNorm sits *after* that convolution, so it cannot rescue
    the input scale, and a kaiming-initialised kernel produces near-zero
    activations for the first several epochs.  The measured symptom was a training
    loss that would not go below 1.6 against a random-guess value of log(15)=2.71
    — the network could not fit 37,000 samples at all.

        affine   (x + 80) / 80            -> [0, 1], fixed, leak-free, no state
        dataset  (x - mu) / sd            -> statistics from the TRAIN rows only
        none     raw dB                   -> what the first W3-A run used
    """

    def __init__(self, arr, rows: np.ndarray, labels: np.ndarray, domains,
                 norm: str = "affine", mu: float = 0.0, sd: float = 1.0):
        self.arr, self.rows, self.labels, self.domains = arr, rows, labels, domains
        self.norm, self.mu, self.sd = norm, float(mu), float(sd)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = int(self.rows[i])
        x = np.asarray(self.arr[r], dtype=np.float32)
        if self.norm == "affine":
            x = (x + 80.0) / 80.0
        elif self.norm == "dataset":
            x = (x - self.mu) / self.sd
        t = torch.from_numpy(x).unsqueeze(0)
        if self.domains is None:
            return t, int(self.labels[i])
        return t, int(self.labels[i]), int(self.domains[i])


def dataset_stats(arr, rows: np.ndarray, cap: int = 2000) -> tuple[float, float]:
    """Mean and sd over a bounded sample of TRAIN rows (never the test rows)."""
    if len(rows) == 0:
        return 0.0, 1.0
    idx = np.linspace(0, len(rows) - 1, min(cap, len(rows))).astype(int)
    xs = np.stack([np.asarray(arr[int(rows[i])], dtype=np.float32) for i in idx])
    return float(xs.mean()), float(xs.std() + 1e-6)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
class ConvNet(nn.Module):
    """Compact 4-block CNN; `width` is the knob.

    The parameter count is the point.  Soti et al. (2026) report that an
    11.17M-parameter ResNet18 gave NO improvement over a 157K compact CNN once
    the split was honest.  Width 24 lands near 137K, so `cnn_small` is the direct
    counterpart of their model and the ladder around it either reproduces or
    contradicts the claim.
    """

    def __init__(self, n_classes: int, width: int = 24, in_ch: int = 1):
        super().__init__()

        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, padding=1, bias=False),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_ch, width), block(width, width * 2),
            block(width * 2, width * 4), block(width * 4, width * 4),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(width * 4, n_classes)
        self.emb_dim = width * 4

    def forward_features(self, x):
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x):
        return self.head(self.drop(self.forward_features(x)))


class LinearProbe(nn.Module):
    """Logistic regression on the flattened time-frequency map, trained on GPU.

    Cheapest baseline in the literature and the sanity floor: if a linear model
    already separates the classes, the task was never the hard part — the
    protocol was.
    """

    def __init__(self, n_classes: int, dim: int = 128 * 313):
        super().__init__()
        self.fc = nn.Linear(dim, n_classes)
        self.emb_dim = dim

    def forward_features(self, x):
        return x.flatten(1)

    def forward(self, x):
        return self.fc(x.flatten(1))


WIDTHS = {"cnn_tiny": 16, "cnn_small": 24, "cnn_wide": 48}
BACKBONES = ["logreg", "cnn_tiny", "cnn_small", "cnn_wide", "resnet18", "efficientnet_b0"]


def build_model(backbone: str, n_classes: int, in_ch: int = 1):
    """Return (model, pretrained?)."""
    if backbone in WIDTHS:
        return ConvNet(n_classes, WIDTHS[backbone], in_ch), False
    if backbone == "logreg":
        return LinearProbe(n_classes), False
    if backbone in ("resnet18", "efficientnet_b0"):
        import torchvision.models as tvm
        for weights in ("IMAGENET1K_V1", None):
            try:
                if backbone == "resnet18":
                    m = tvm.resnet18(weights=weights)
                    m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
                    m.fc = nn.Linear(m.fc.in_features, n_classes)
                    m.emb_dim = m.fc.in_features
                else:
                    m = tvm.efficientnet_b0(weights=weights)
                    m.features[0][0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
                    m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
                    m.emb_dim = m.classifier[1].in_features
                return m, weights is not None
            except Exception as exc:
                if weights is None:
                    raise
                print(f"[warn] pretrained weights unavailable ({exc}); random init")
        raise SystemExit("unreachable")
    raise ValueError(f"unknown backbone {backbone!r}")


def head_of(model):
    """The final linear layer, whatever backbone this is."""
    for name in ("head", "fc"):
        if hasattr(model, name):
            return getattr(model, name)
    if hasattr(model, "classifier"):
        for layer in model.classifier:
            if isinstance(layer, nn.Linear):
                return layer
    raise SystemExit("cannot locate the classifier head")


def features_of(model, x):
    """Penultimate representation — used by the invariances and the diagnostics."""
    if hasattr(model, "forward_features"):
        return model.forward_features(x)
    if hasattr(model, "layer4"):  # resnet
        h = model.maxpool(model.relu(model.bn1(model.conv1(x))))
        h = model.layer4(model.layer3(model.layer2(model.layer1(h))))
        return model.avgpool(h).flatten(1)
    if hasattr(model, "features"):  # efficientnet
        return model.avgpool(model.features(x)).flatten(1)
    raise SystemExit("cannot extract features from this model")


def emb_dim_of(model) -> int:
    if hasattr(model, "emb_dim"):
        return int(model.emb_dim)
    return int(head_of(model).in_features)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def evaluate(model, loader, device, n_classes, want_emb=False):
    model.eval()
    preds, trues, probs, embs = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            with autocast(device.type == "cuda"):
                logits = model(x)
            probs.append(torch.softmax(logits.float(), 1).cpu().numpy())
            preds.append(logits.argmax(1).cpu().numpy())
            trues.append(batch[1].numpy())
            if want_emb:
                embs.append(features_of(model, x).float().cpu().numpy())
    p = np.concatenate(preds) if preds else np.zeros(0, dtype=int)
    t = np.concatenate(trues) if trues else np.zeros(0, dtype=int)
    return {
        "acc": float((p == t).mean()) if len(t) else 0.0,
        "macro_f1": macro_f1(t, p, n_classes),
        "y_true": t, "y_pred": p,
        "proba": np.concatenate(probs) if probs else np.zeros((0, n_classes), np.float32),
        "emb": np.concatenate(embs) if embs else None,
    }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, opt, sched, scaler, device, args, dom_head, n_domains,
                    epoch):
    model.train()
    if dom_head is not None:
        dom_head.train()
    head = head_of(model)
    total = seen = 0.0

    # Double back-propagation under fp16 autocast is not numerically safe, so the
    # gradient-penalty methods run in fp32.  They are ~2x slower and it does not
    # matter: they are 12 runs, not 30.
    amp = (device.type == "cuda") and args.method in ("erm", "dann")
    lam = 0.0
    if args.method == "dann" and n_domains > 1:
        p = (epoch - 1) / max(args.epochs - 1, 1)
        lam = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        y = batch[1].to(device, non_blocking=True)
        d = batch[2].to(device, non_blocking=True) if len(batch) > 2 else None
        opt.zero_grad(set_to_none=True)

        with autocast(amp):
            if args.method in ("irm", "vrex") and d is not None and n_domains > 1:
                phi = features_of(model, x)
                risks, scales = [], []
                for g in torch.unique(d):
                    m = d == g
                    if int(m.sum()) < 2:
                        continue
                    scale = torch.zeros((), device=device).requires_grad_(True)
                    risks.append(F.cross_entropy(head(phi[m] * scale), y[m]))
                    scales.append(scale)
                if risks:
                    loss_main = torch.stack(risks).mean()
                    if args.method == "irm":
                        grads = torch.autograd.grad(loss_main, scales, create_graph=True,
                                                    allow_unused=True)
                        pen = sum((g.float() ** 2).sum() for g in grads if g is not None)
                    else:
                        r = torch.stack(risks)
                        pen = r.var(unbiased=False)
                    loss = loss_main + args.dg_weight * pen
                else:
                    loss = F.cross_entropy(model(x), y)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                if args.method == "dann" and dom_head is not None and d is not None:
                    loss = loss + args.dg_weight * F.cross_entropy(
                        dom_head(GradReverse.apply(features_of(model, x), lam)), d)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        # The scheduler is stepped per OPTIMISER STEP, not per epoch: lr_at() takes
        # a global step index, and `warm`/`total_steps` are both counted in steps.
        #
        # Leaving this line out is silent and catastrophic.  LambdaLR returns
        # lr_at(0) forever, and with a warm-up configured lr_at(0) is
        # 1/(warmup * steps_per_epoch) — about 5.7e-7 for a 1e-3 request on 37,626
        # training clips.  Every configuration in the W3-A2 recipe check therefore
        # ran at effectively zero learning rate and plateaued at 0.27-0.36 training
        # accuracy, which read as "the representation is too weak" when the truth
        # was that the network never left its initialisation.
        sched.step()
        total += float(loss.detach()) * len(y)
        seen += len(y)
    return total / max(seen, 1)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Run one UATR benchmark configuration.")
    ap.add_argument("--data", type=Path, required=True,
                    help="mounted under_water_features dataset")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/w3-runs"))
    ap.add_argument("--corpus", default="oceanship", choices=list(CORPORA))
    ap.add_argument("--train-corpus", default=None, choices=list(CORPORA))
    ap.add_argument("--test-corpus", default=None, choices=list(CORPORA))
    ap.add_argument("--rep", default="logmel", choices=["logmel", "spec"])
    ap.add_argument("--backbone", default="cnn_small", choices=BACKBONES)
    ap.add_argument("--protocol", default="vessel_disjoint")
    ap.add_argument("--split-file", default="",
                    help="override the frozen split file (e.g. a DeepShip file with "
                         "vessel_disjoint rows appended by w1_02b); default: "
                         "<data>/splits/<corpus>_splits.csv.gz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-seed", type=int, default=None,
                    help="seed of the split draw; default: --seed.  Decouple it from "
                         "--seed to measure split-draw variance separately from "
                         "initialisation noise (the resolvability experiment)")
    ap.add_argument("--init-seed", type=int, default=None,
                    help="seed for weight init and data order; default: --seed")
    ap.add_argument("--vocab", default="native", choices=["native", "common4"])
    ap.add_argument("--method", default="erm", choices=["erm", "irm", "vrex", "dann"])
    ap.add_argument("--domain", default="none",
                    choices=["none", "official", "date", "dur", "vessel_hash"])
    ap.add_argument("--dg-weight", type=float, default=1.0)
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="control run: permute the TRAIN labels, keep everything else")
    ap.add_argument("--embeddings", action="store_true",
                    help="also dump penultimate features for the diagnostics notebook")
    ap.add_argument("--emb-train-cap", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=3,
                    help="linear warm-up epochs before the cosine decay")
    ap.add_argument("--norm", default="affine", choices=["affine", "dataset", "none"],
                    help="input scaling; the arrays are raw dB in [-80, 0] and an "
                         "unscaled first convolution learns very slowly")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=0, help="subsample train (smoke test)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on a small subset")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.max_train = args.max_train or 600
        # A two-epoch run cannot finish a three-epoch warm-up, so without this the
        # probe would spend its whole budget at lr ~ 0 and the loss line would say
        # nothing about whether the wiring can learn at all.
        args.warmup = 0

    # By default one seed does double duty (split draw + initialisation), which is
    # what every W3-A..E run used.  The resolvability experiment needs the two
    # roles separable: split-draw variance and init variance are different
    # quantities, and conflating them is exactly the confound the experiment
    # exists to remove.
    split_seed = args.seed if args.split_seed is None else args.split_seed
    init_seed = args.seed if args.init_seed is None else args.init_seed

    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = resolve_features_root(args.data)
    cross = bool(args.train_corpus and args.test_corpus)
    corpus = args.train_corpus if cross else args.corpus
    frc_approx = (args.vocab == "common4" and corpus == "shipsear") or \
                 (cross and args.test_corpus == "shipsear")

    print("=" * 88)
    print(f"W3 run   data={root}   device={device}")
    print(f"seeds      : run={args.seed}  split={split_seed}  init={init_seed}"
          + ("   (split and init DECOUPLED)" if (split_seed, init_seed) != (args.seed, args.seed)
             else ""))
    print("=" * 88)

    idx = load_index(root, corpus)
    dur_bounds = duration_quartiles(idx) if args.domain == "dur" else None
    test_idx = load_index(root, args.test_corpus) if cross else idx

    if cross:
        proto = protocol_for(corpus, args.protocol, args.train_corpus, args.test_corpus)
    else:
        proto = args.protocol
        if proto not in PROTOCOLS[corpus]:
            raise SystemExit(
                f"corpus {corpus} supports {PROTOCOLS[corpus]}, not {proto!r}. "
                f"(Oceanship has a ship id so it gets vessel_disjoint; DeepShip has "
                f"a temporal protocol instead.)"
            )

    def usable(corpus_name, rows):
        out = []
        for i, r in enumerate(rows):
            lb = label_for(corpus_name, r, args.vocab)
            if lb is not None:
                out.append((i, r, lb))
        return out

    tr_all = usable(corpus, idx)
    te_all = usable(args.test_corpus if cross else corpus, test_idx)
    if not tr_all:
        raise SystemExit(f"no usable rows for {corpus} under vocab {args.vocab!r}")
    print(f"  corpus      : {corpus}" + (f"  -> test on {args.test_corpus}" if cross else ""))
    if frc_approx:
        print("  [note] ShipsEar classes are mapped to the shared vocabulary by "
              "literature convention (class 2 -> Passenger, 0 -> Tug); this is an "
              "approximation and is flagged in results.csv as vocab_approx=1")

    # ------------------------------------------------------------------ splits
    dropped: list[str] = []
    if cross:
        rec_assign = record_split(idx, corpus, split_seed)
        rec_col = REC_COL[corpus]
        train_rows = [i for i, r, _ in tr_all if rec_assign.get(r[rec_col]) == "train"]
        val_rows = [i for i, r, _ in tr_all if rec_assign.get(r[rec_col]) == "val"]
        test_rows = [i for i, _, _ in te_all]
        test_meta = te_all
        split_note = (f"recording-level 70/15 holdout of {corpus}; ALL of "
                      f"{args.test_corpus} held out")
    else:
        sf = load_native_split_file(root, corpus, proto, split_seed,
                                    getattr(args, "split_file", ""))
        if sf is None:
            gen, dropped = group_split(idx, corpus, split_seed)
            rec_col = REC_COL[corpus]
            row_split = {i: gen.get(r[rec_col]) for i, r, _ in tr_all}
            split_note = (f"split generated here ({corpus} has no W1 split file), "
                          f"split_seed={split_seed}")
            if dropped:
                split_note += f"; classes with <3 groups forced into train: {dropped}"
        else:
            assign, _, seen = sf
            if not assign:
                raise SystemExit(
                    f"{corpus} split file holds no rows for protocol={proto!r} "
                    f"seed={split_seed}. It contains protocols "
                    f"{sorted({p for p, _ in seen})} with seeds "
                    f"{sorted({s for _, s in seen})}."
                )
            row_split = {i: split_lookup(assign, corpus, r) for i, r, _ in tr_all}
            split_note = (f"frozen W1 split  protocol={proto}  split_seed={split_seed}  "
                          f"({len(assign):,} rows read)")
        train_rows = [i for i, s in row_split.items() if s == "train"]
        val_rows = [i for i, s in row_split.items() if s == "val"]
        test_rows = [i for i, s in row_split.items() if s == "test"]
        test_meta = [(i, r, lb) for i, r, lb in tr_all if row_split.get(i) == "test"]

    if not train_rows or not test_rows:
        raise SystemExit(
            f"protocol {proto!r} gave train={len(train_rows)} test={len(test_rows)}. "
            f"available: {PROTOCOLS[corpus]}"
        )
    if args.max_train and len(train_rows) > args.max_train:
        rng = np.random.RandomState(init_seed)
        train_rows = list(rng.choice(train_rows, args.max_train, replace=False))

    # ------------------------------------------------------- label vocabulary
    # Built AFTER the split filter, not from the whole index.  The Oceanship
    # index carries 15 classes but the frozen split covers 53,754 clips = the 12
    # that have enough vessels for a three-way vessel-disjoint split.  Sizing the
    # head from the unfiltered index therefore added three classes that have no
    # training row at all: wasted capacity, and a macro-F1 denominator that the
    # model can never move.  The first W3-A run did exactly that.
    if cross:
        kept = {lb for _, _, lb in tr_all} | {lb for _, _, lb in te_all}
    else:
        kept = {lb for i, _, lb in tr_all if row_split.get(i) is not None}
    classes = sorted(kept)
    class_to_i = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    in_index = len({lb for _, _, lb in tr_all})
    print(f"  vocab       : {args.vocab}  ({n_classes} classes) {classes}")
    if in_index != n_classes:
        print(f"  [filter] the index holds {in_index} classes; {in_index - n_classes} "
              f"have no row in this split and are excluded from the label space")

    # rows outside the split (and therefore outside the label space) are dropped
    # here: they are never trained or evaluated on, and indexing class_to_i with
    # one of their labels would raise
    y_all = {i: class_to_i[lb] for i, _, lb in tr_all if lb in class_to_i}
    y_test = np.array([class_to_i[lb] for _, _, lb in test_meta], dtype=np.int64)

    def maps(corpus_name: str, rows: list[dict]):
        """Row index -> (recording, vessel, group, source duration).

        Kept per corpus on purpose: in a cross-corpus run the train and test row
        indices come from two different indexes, so one shared dict silently
        returns the wrong recording id — or raises KeyError deep inside the leak
        check.
        """
        rec = {i: r.get(REC_COL[corpus_name], "") for i, r in enumerate(rows)}
        # a clip with no vessel identity is its own vessel: two clips of one
        # recording can then never straddle the split in the leak check, and an
        # empty string (DeepShip's 48 AIS-unjoined recordings) can never be read
        # as one shared "vessel" spanning train and test
        ves = {i: (r.get(VESSEL_COL[corpus_name], "")
                   or f"rec::{r.get(REC_COL[corpus_name], '')}")
               for i, r in enumerate(rows)}
        grp = {i: r.get("group", "") for i, r in enumerate(rows)}
        dur = {i: float(r.get("src_dur_s") or 0.0) for i, r in enumerate(rows)}
        return rec, ves, grp, dur

    rec_all, ves_all, grp_all, dur_all = maps(corpus, idx)
    rec_te, ves_te, grp_te, dur_te = maps(args.test_corpus if cross else corpus, test_idx)

    # two more metadata columns the diagnostics notebook probes for domain
    # separability; carried through the embedding dump so w3_diag.py never has to
    # guess how to recover them from an identifier string
    off_all = {i: r.get("official_split", "") for i, r, _ in tr_all}
    dat_all = {i: r.get("date", "") for i, r, _ in tr_all}
    off_te = {i: r.get("official_split", "") for i, r, _ in te_all}
    dat_te = {i: r.get("date", "") for i, r, _ in te_all}

    # ----------------------------------------------------------------- domain
    dom_names: list[str] = []
    dom_raw: dict[int, str] = {}
    if args.domain != "none":
        dom_raw = {i: domain_for(corpus, r, args.domain, dur_bounds) for i, r, _ in tr_all}
        dom_names = sorted(set(dom_raw.values()))
        dom_all = {i: dom_names.index(v) for i, v in dom_raw.items()}
        counts = Counter(dom_all.get(i) for i in train_rows)
        print(f"  domain      : {args.domain} -> {len(dom_names)} groups "
              f"{ {dom_names[j]: n for j, n in sorted(counts.items())} }")
    else:
        dom_all = {}
    tr_dom = (np.array([dom_all.get(i, 0) for i in train_rows], dtype=np.int64)
              if dom_names else None)

    # ---------------------------------------------------------- label shuffle
    y_train = np.array([y_all[i] for i in train_rows], dtype=np.int64)
    if args.shuffle_labels:
        # label permutation is training-side randomness, so it follows the init seed
        perm = np.random.RandomState(stable_int("shuffle", init_seed, proto) % (2 ** 31)
                                     ).permutation(len(y_train))
        y_train = y_train[perm]
        print("  [control] TRAIN LABELS SHUFFLED — test accuracy must collapse to chance")

    # ------------------------------------------------------------------- load
    rep_file = root / corpus / f"{args.rep}.npy"
    if not rep_file.exists():
        raise SystemExit(f"{rep_file} missing")
    arr = np.load(rep_file, mmap_mode="r")
    print(f"  array       : {rep_file.name}  {arr.shape} {arr.dtype}")
    print(f"  rows        : train={len(train_rows):,} val={len(val_rows):,} test={len(test_rows):,}")
    print(f"  {split_note}")

    train_recs, test_recs = {rec_all[i] for i in train_rows}, {rec_te[i] for i in test_rows}
    train_ves, test_ves = {ves_all[i] for i in train_rows}, {ves_te[i] for i in test_rows}
    leak_rec, leak_ves = len(test_recs & train_recs), len(test_ves & train_ves)
    print(f"  leak check  : test recordings shared with train {leak_rec}/{len(test_recs)}; "
          f"vessels {leak_ves}/{len(test_ves)}")

    dl = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    y_val = np.array([y_all[i] for i in val_rows], dtype=np.int64)
    mu, sd = dataset_stats(arr, np.array(train_rows)) if args.norm == "dataset" else (0.0, 1.0)
    if args.norm == "dataset":
        print(f"  norm        : dataset  mu={mu:.3f} sd={sd:.3f}  (train rows only)")
    else:
        print(f"  norm        : {args.norm}")
    nd = dict(norm=args.norm, mu=mu, sd=sd)
    tr_ds = ClipDataset(arr, np.array(train_rows), y_train, tr_dom, **nd)
    va_ds = ClipDataset(arr, np.array(val_rows), y_val,
                        np.array([dom_all.get(i, 0) for i in val_rows], dtype=np.int64)
                        if dom_names else None, **nd)
    te_ds = ClipDataset(arr, np.array(test_rows), y_test, None, **nd)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, **dl)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size * 2, shuffle=False, **dl)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size * 2, shuffle=False, **dl)
    tr_eval_ld = DataLoader(tr_ds, batch_size=args.batch_size * 2, shuffle=False, **dl)

    # ------------------------------------------------------------------ model
    model, pretrained = build_model(args.backbone, n_classes)
    model = model.to(device)
    n_dom = len(dom_names) if dom_names else 1

    dom_head = None
    if args.method == "dann" and n_dom > 1:
        # built BEFORE the optimiser so its parameters are actually trained
        dom_head = nn.Sequential(nn.Linear(emb_dim_of(model), 128), nn.ReLU(inplace=True),
                                 nn.Linear(128, n_dom)).to(device)

    params = list(model.parameters()) + (list(dom_head.parameters()) if dom_head else [])
    n_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"  model       : {args.backbone} params={n_params:,} pretrained={pretrained} "
          f"method={args.method}{' +domain_head' if dom_head else ''}")
    print(f"  recipe      : epochs={args.epochs} lr={args.lr:g} warmup={args.warmup} "
          f"batch={args.batch_size} wd={args.weight_decay}")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(tr_ld), 1)
    warm = max(args.warmup, 0) * steps_per_epoch
    total_steps = max(args.epochs * steps_per_epoch, warm + 1)

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / max(warm, 1)
        p = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = grad_scaler(device.type == "cuda")
    print(f"  schedule    : peak lr={args.lr:g} at step {warm:,}, "
          f"decays to 0 over {total_steps - warm:,} more steps "
          f"({steps_per_epoch:,} steps/epoch)")

    t0 = time.time()
    best_val, best_state, tr_acc = -1.0, None, float("nan")
    lr_seen: list[float] = []
    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, tr_ld, opt, sched, scaler, device, args,
                               dom_head, n_dom, ep)
        v = evaluate(model, va_ld, device, n_classes)
        # training accuracy is the underfitting alarm: if it is not far above the
        # validation accuracy the model is too weak or too briefly trained for the
        # protocol comparison to mean anything
        tr = evaluate(model, tr_eval_ld, device, n_classes)
        tr_acc = tr["acc"]
        cur_lr = float(opt.param_groups[0]["lr"])
        lr_seen.append(cur_lr)
        marker = ""
        if v["macro_f1"] > best_val:
            best_val, marker = v["macro_f1"], " *"
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        gap = tr_acc - v["acc"]
        print(f"  ep {ep:>3} lr={cur_lr:.2e} loss={loss:.4f} train_acc={tr_acc:.4f} "
              f"val_acc={v['acc']:.4f} val_f1={v['macro_f1']:.4f} gap={gap:+.4f}{marker}",
              flush=True)
    wall = time.time() - t0

    # A schedule that never moves is not a schedule.  This fired silently once and
    # cost a six-hour sweep, so it is an error now, not a comment.
    if len(set(lr_seen)) == 1 and args.epochs > 1:
        raise SystemExit(
            f"the learning-rate schedule never changed (stuck at {lr_seen[0]:.2e} for "
            f"{args.epochs} epochs) - the scheduler is not being stepped")

    if best_state is not None:
        model.load_state_dict(best_state)
    te = evaluate(model, te_ld, device, n_classes, want_emb=args.embeddings)
    tr_final = evaluate(model, tr_eval_ld, device, n_classes)
    n_pred_classes = int(np.unique(te["y_pred"]).size)

    # ---------------------------------------------------------------- record
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictions").mkdir(exist_ok=True)
    run_id = run_id_for(corpus, args.rep, args.backbone, proto, args.seed,
                        args.domain, args.method, args.shuffle_labels, args.tag,
                        split_seed=split_seed, init_seed=init_seed)

    np.savez_compressed(
        args.out / "predictions" / f"{run_id}.npz",
        y_true=te["y_true"], y_pred=te["y_pred"],
        rec_key=np.array([rec_te[i] for i in test_rows]),
        vessel=np.array([ves_te[i] for i in test_rows]),
        group=np.array([grp_te[i] for i in test_rows]),
        dur=np.array([dur_te[i] for i in test_rows], dtype=np.float32),
        classes=np.array(classes),
        proba=te["proba"].astype(np.float16),
    )
    if args.embeddings and te["emb"] is not None:
        (args.out / "embeddings").mkdir(exist_ok=True)
        rng = np.random.RandomState(init_seed)
        cap = rng.choice(len(train_rows), min(args.emb_train_cap, len(train_rows)), replace=False)
        tr_sub = np.array(train_rows)[cap]
        sub_ld = DataLoader(ClipDataset(arr, tr_sub, y_train[cap], None),
                            batch_size=args.batch_size * 2, shuffle=False, **dl)
        tr_emb = evaluate(model, sub_ld, device, n_classes, want_emb=True)["emb"]
        np.savez_compressed(
            args.out / "embeddings" / f"{run_id}.npz",
            test_emb=te["emb"].astype(np.float16), test_y=te["y_true"],
            test_rec=np.array([rec_te[i] for i in test_rows]),
            test_vessel=np.array([ves_te[i] for i in test_rows]),
            train_emb=tr_emb.astype(np.float16), train_y=y_train[cap],
            train_rec=np.array([rec_all[i] for i in tr_sub]),
            train_vessel=np.array([ves_all[i] for i in tr_sub]),
            train_official=np.array([off_all[i] for i in tr_sub]),
            train_date=np.array([dat_all[i] for i in tr_sub]),
            test_official=np.array([off_te[i] for i in test_rows]),
            test_date=np.array([dat_te[i] for i in test_rows]),
            test_dur=np.array([dur_te[i] for i in test_rows], dtype=np.float32),
            train_dur=np.array([dur_all[i] for i in tr_sub], dtype=np.float32),
            classes=np.array(classes), shuffled=int(args.shuffle_labels),
            seed=args.seed, rep=args.rep, backbone=args.backbone,
            corpus=corpus, protocol=proto,
        )

    row = {
        "run_id": run_id, "corpus": corpus, "rep": args.rep, "backbone": args.backbone,
        "protocol": proto, "seed": args.seed, "method": args.method, "domain": args.domain,
        "dg_weight": args.dg_weight, "vocab": args.vocab,
        "vocab_approx": int(frc_approx), "shuffled": int(args.shuffle_labels),
        "n_train": len(train_rows), "n_val": len(val_rows), "n_test": len(test_rows),
        "n_classes": n_classes, "n_params": n_params, "n_domains": n_dom,
        "norm": args.norm, "lr": args.lr, "warmup": args.warmup,
        "lr_last": round(float(lr_seen[-1]), 8),
        "train_acc": round(float(tr_final["acc"]), 4),
        "train_val_gap": round(float(tr_final["acc"]) - float(te["acc"]), 4),
        "n_pred_classes": n_pred_classes,
        "n_test_recordings": len(test_recs), "n_test_vessels": len(test_ves),
        "leak_rec_pct": round(100 * leak_rec / max(len(test_recs), 1), 2),
        "leak_vessel_pct": round(100 * leak_ves / max(len(test_ves), 1), 2),
        "best_val_f1": round(float(best_val), 4),
        "test_acc": round(float(te["acc"]), 4),
        "test_macro_f1": round(float(te["macro_f1"]), 4),
        "chance_acc": round(1.0 / n_classes, 4),
        "epochs": args.epochs, "wall_min": round(wall / 60, 2),
    }
    for c, i in class_to_i.items():
        m = te["y_true"] == i
        row[f"acc_{c}"] = round(float((te["y_pred"][m] == i).mean()), 4) if m.any() else None
        row[f"n_{c}"] = int(m.sum())
    # Only present when decoupled, and only at the very END of the row: results.csv
    # is appended across sessions, and a mid-row extra column would shift every
    # later value relative to the header written by earlier sessions.
    if (split_seed, init_seed) != (args.seed, args.seed):
        row["split_seed"] = split_seed
        row["init_seed"] = init_seed

    res = args.out / "results.csv"
    header = not res.exists()
    with res.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if header:
            w.writeheader()
        w.writerow(row)

    print()
    print(json.dumps(row, indent=2, ensure_ascii=False))
    print(f"\nappended to {res}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
