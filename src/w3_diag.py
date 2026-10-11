#!/usr/bin/env python3
"""W3 diagnostics — the checks a reviewer is entitled to ask for.

Everything here runs on frozen embeddings dumped by `w3_run.py --embeddings`, so
none of it needs a GPU and all of it is cheap.

The four questions
------------------
1. How much *recording identity* survives in the representation?  (P8)
   Hummel et al. (2026, Ecological Informatics 95 103709) make this mandatory
   reporting for passive-acoustic deep learning: if a linear read-out of the
   frozen features can name the recording a clip came from, then any downstream
   accuracy is partly session recall rather than ship recognition.  Reported as a
   nearest-centroid balanced accuracy plus normalised mutual information against
   the true recording id — and the same pipeline run on shuffled keys, so the
   reader can see what "no signal" looks like on this data size.

2. How separable are the *domain* labels we intend to be invariant to?  (P6)
   An invariance objective can only help if the domains are distinguishable in
   the features.  If `date` is already invisible, a gain from IRM/VREx on it
   would be unexplained and should be treated as a red flag, not a result.

3. Is *duration* a shortcut?  (P5)
   Short clips are padded rather than dropped, and duration is class-correlated
   (Oceanship Passenger median 3.67 s against Tug 5.00 s).  A model can therefore
   read the class off the amount of padding.  Two proxies are measured: the
   stored source duration, and the fraction of time frames sitting on the
   log-mel floor — the latter being what the network sees in the image with no
   acoustic understanding at all.  Both are scored out-of-fold.

4. Does the label-shuffling control hold?  (P7)
   Summarised from the run rows.

Usage
-----
    python w3_diag.py --data /kaggle/input/under_water_features \
                      --runs /kaggle/working/w3-runs \
                      --out /kaggle/working/w3-tables
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

LABEL_COL = {"oceanship": "label", "deepship": "class_dir", "shipsear": "class_dir"}
REC_COL = {"oceanship": "rec_key", "deepship": "rec_key", "shipsear": "group"}
DATE_RE = re.compile(r"(\d{4})[-/]?(\d{2})")


# --------------------------------------------------------------------------- #
def open_text(path: Path):
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(2)
    if head == b"\x1f\x8b":
        return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")
    return open(path, "r", newline="", encoding="utf-8", errors="replace")


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0].astype(np.float64)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def mutual_information(a: np.ndarray, b: np.ndarray) -> float:
    """I(A;B) in nats, straight from the contingency table."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(table, (ia, ib), 1)
    n = table.sum()
    if n == 0:
        return 0.0
    pij = table / n
    pi, pj = pij.sum(1, keepdims=True), pij.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = pij * (np.log(pij) - np.log(pi) - np.log(pj))
    return float(np.nansum(terms))


def nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information in [0, 1] (geometric-mean normalisation)."""
    i = mutual_information(a, b)
    ha = entropy(np.bincount(np.unique(a, return_inverse=True)[1]))
    hb = entropy(np.bincount(np.unique(b, return_inverse=True)[1]))
    if ha <= 0 or hb <= 0:
        return 0.0
    return float(i / np.sqrt(ha * hb))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = [float((y_pred[y_true == c] == c).mean()) for c in np.unique(y_true)]
    return float(np.mean(r)) if r else 0.0


def centroid_predict(Xt, yt, Xq, min_per_class=2, max_classes=0):
    """Nearest class-centroid read-out on L2-normalised features.

    Centroid matching rather than multinomial logistic regression on purpose: the
    recording probe has thousands of classes, where a multinomial fit would be
    both slow and numerically unhappy.  Cosine-to-centroid is a linear read-out
    of the same features and is what makes this tractable at Oceanship's scale.
    """
    def norm(a):
        return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-8)

    yt = np.asarray(yt)
    Xt, Xq = norm(np.asarray(Xt, dtype=np.float32)), norm(np.asarray(Xq, dtype=np.float32))
    counts = Counter(yt.tolist())
    keep = [c for c, n in counts.items() if n >= min_per_class]
    if max_classes and len(keep) > max_classes:
        keep = sorted(keep, key=lambda c: -counts[c])[:max_classes]
    if len(keep) < 2:
        return None, None
    keep_arr = np.array(keep)
    cen = np.stack([Xt[yt == c].mean(0) for c in keep])
    cen = norm(cen)
    pred = keep_arr[np.argmax(Xq @ cen.T, axis=1)]
    return pred, keep_arr


def out_of_fold_1d(x: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    """Balanced accuracy of a nearest-class-mean rule in ONE feature, out of fold.

    Resubstitution would be optimistic here, and this probe exists precisely to
    bound how much a shortcut is worth, so it has to be scored honestly.
    """
    rng = np.random.default_rng(0)
    order = rng.permutation(len(x))
    parts = np.array_split(order, folds)
    pred = np.empty(len(x), dtype=object)
    for i in range(folds):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(folds) if j != i])
        means = {c: float(x[tr][y[tr] == c].mean()) for c in np.unique(y[tr])}
        cs = sorted(means)
        mu = np.array([means[c] for c in cs])
        pred[te] = [cs[j] for j in np.argmin(np.abs(x[te][:, None] - mu[None, :]), axis=1)]
    return balanced_accuracy(y, pred.astype(y.dtype))


def duration_bucket(x: np.ndarray, step: float = 0.1) -> np.ndarray:
    return np.round(np.asarray(x, dtype=np.float64) / step).astype(np.int64)


def date_key_of(rec_key: str) -> str:
    m = DATE_RE.search(str(rec_key) or "")
    return f"{m.group(1)}-{m.group(2)}" if m else "unknown"


def top_buckets(arr: np.ndarray, k: int = 4) -> np.ndarray:
    top = [c for c, _ in Counter(arr.tolist()).most_common(k)]
    return np.where(np.isin(arr, top), arr, "other")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="W3 diagnostics on frozen embeddings.")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", type=Path, nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/w3-tables"))
    ap.add_argument("--max-runs", type=int, default=6)
    ap.add_argument("--max-classes", type=int, default=3000,
                    help="cap the recording probe's class count (bounds runtime)")
    args = ap.parse_args()

    files: list[Path] = []
    for root in [Path(r) for r in (args.runs or ["/kaggle/working/w3-runs"])]:
        for depth in range(4):
            files += sorted(root.glob("/".join(["*"] * depth + ["embeddings", "*.npz"])))
    uniq, seen = [], set()
    for p in files:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    if not uniq:
        raise SystemExit(
            "no embedding .npz found. Run w3_run.py with --embeddings, or point "
            "--runs at a directory containing an embeddings/ folder."
        )
    uniq = uniq[: args.max_runs]

    (args.out / "tables").mkdir(parents=True, exist_ok=True)
    L: list[str] = []
    add = L.append
    add("=" * 100)
    add("W3 diagnostics")
    add("=" * 100)
    add(f"  {len(uniq)} embedding files from {args.runs or ['/kaggle/working/w3-runs']}")
    summary: dict = {}

    for f in uniq:
        d = np.load(f, allow_pickle=True)
        corpus = str(d["corpus"]) if "corpus" in d else "oceanship"
        add("")
        add("=" * 100)
        add(f"run: {corpus} / {d['rep']} / {d['backbone']} / {d['protocol']} / s{d['seed']}")
        add("=" * 100)
        tr_e = d["train_emb"].astype(np.float32)
        te_e = d["test_emb"].astype(np.float32)
        te_y = np.asarray(d["test_y"])
        chance_cls = 1.0 / max(len(set(te_y.tolist())), 1)
        add(f"  embeddings: train {tr_e.shape}  test {te_e.shape}   "
            f"test classes {len(set(te_y.tolist()))} (chance {chance_cls:.4f})")
        rng = np.random.default_rng(0)

        # ------------------------------------------------ the class read-out
        pred, _ = centroid_predict(tr_e, d["train_y"], te_e)
        if pred is not None:
            add(f"  class read-out        balanced acc = {balanced_accuracy(te_y, pred):.4f}"
                f"   NMI = {nmi(te_y, pred):.4f}")

        # --------------- P8 recording / vessel identity in the representation
        add("")
        add("  P8  identity surviving in the frozen representation "
            "(mandatory reporting, Hummel et al. 2026)")
        for name, tk, qk in (("recording", "rec", "rec"), ("vessel", "vessel", "vessel")):
            tr_key = np.asarray([str(x) for x in d[f"train_{tk}"]])
            te_key = np.asarray([str(x) for x in d[f"test_{tk}"]])
            pred, keep = centroid_predict(tr_e, tr_key, te_e, max_classes=args.max_classes)
            if pred is None:
                add(f"      {name:<10} too few classes with >=2 training clips")
                continue
            m = np.isin(te_key, keep)
            ba, mi = balanced_accuracy(te_key[m], pred[m]), nmi(te_key[m], pred[m])
            pred_s, keep_s = centroid_predict(tr_e, rng.permutation(tr_key), te_e,
                                              max_classes=args.max_classes)
            ms = np.isin(te_key, keep_s)
            ba_s = balanced_accuracy(te_key[ms], pred_s[ms])
            add(f"      {name:<10} classes={len(keep):>6,}  covered test clips={int(m.sum()):>7,}"
                f"   balanced acc = {ba:.4f}   NMI = {mi:.4f}")
            add(f"      {'':<10} shuffled-key control: balanced acc = {ba_s:.4f}"
                f"   (the floor this data size can reach by chance)")
            summary[f"{f.stem}|{name}_balanced_acc"] = round(ba, 4)
            summary[f"{f.stem}|{name}_nmi"] = round(mi, 4)
            summary[f"{f.stem}|{name}_shuffled_balanced_acc"] = round(ba_s, 4)

        # -------------------------------------------------- P6 domain labels
        add("")
        add("  P6  are the domain labels we want invariance to actually visible?")
        candidates = []
        if "train_official" in d:
            t = np.asarray([str(x) for x in d["train_official"]])
            q = np.asarray([str(x) for x in d["test_official"]])
            if len(set(t.tolist())) > 1:
                candidates.append(("official split", t, q))
        if "train_date" in d:
            t = np.asarray([str(x) for x in d["train_date"]])
            q = np.asarray([str(x) for x in d["test_date"]])
            if len(set(t.tolist())) > 1:
                candidates.append(("date (index column)", t, q))
        t = np.asarray([date_key_of(x) for x in d["train_rec"]])
        q = np.asarray([date_key_of(x) for x in d["test_rec"]])
        if len(set(t.tolist())) > 1:
            candidates.append(("date (from rec_key)", t, q))
        if not candidates:
            add("      no usable domain label for this corpus")
        for name, tk, qk in candidates:
            pred, keep = centroid_predict(tr_e, tk, te_e, min_per_class=2)
            if pred is None:
                add(f"      {name:<22} too few groups")
                continue
            m = np.isin(qk, keep)
            ba = balanced_accuracy(qk[m], pred[m])
            ch = 1.0 / max(len(set(qk[m].tolist())), 1)
            add(f"      {name:<22} groups={len(keep):<3} balanced acc = {ba:.4f}  "
                f"NMI = {nmi(qk[m], pred[m]):.4f}  (chance {ch:.4f})")
            summary[f"{f.stem}|domain_{name.split()[0]}_balanced_acc"] = round(ba, 4)

    # --------------------------------------------------- P5 duration shortcut
    add("")
    add("=" * 100)
    add("P5  duration / padding as a shortcut")
    add("=" * 100)
    add("")
    add("  Short clips are padded, not dropped, and duration is class-correlated, so a")
    add("  model can read the class off the amount of padding.  Both probes are scored")
    add("  out-of-fold; `chance` is 1/K.")
    add("")
    add(f"  {'corpus':<10}{'proxy':<26}{'K':>4}{'balanced acc':>14}{'NMI':>9}{'chance':>9}")
    for corpus in ("oceanship", "deepship", "shipsear"):
        idx_path = args.data / corpus / "index.csv"
        if not idx_path.exists():
            continue
        with idx_path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        y = np.array([r[LABEL_COL[corpus]] for r in rows])
        K = len(set(y.tolist()))
        dur = np.array([float(r.get("src_dur_s") or 0.0) for r in rows])
        add(f"  {corpus:<10}{'src_dur_s alone':<26}{K:>4}"
            f"{out_of_fold_1d(dur, y):>14.4f}{nmi(y, duration_bucket(dur)):>9.4f}{1/K:>9.4f}")
        summary[f"{corpus}|duration_balanced_acc"] = round(out_of_fold_1d(dur, y), 4)

        arr_p = args.data / corpus / "logmel.npy"
        if arr_p.exists():
            arr = np.load(arr_p, mmap_mode="r")
            n = min(4000, len(rows))
            idx = np.linspace(0, len(rows) - 1, n).astype(int)
            frac = np.empty(n, dtype=np.float32)
            for j, k in enumerate(idx):
                col = np.asarray(arr[int(k)], dtype=np.float32).mean(0)
                frac[j] = float((col <= -79.0).mean())
            y_s = y[idx]
            add(f"  {corpus:<10}{'floor-frame fraction':<26}{K:>4}"
                f"{out_of_fold_1d(frac, y_s):>14.4f}"
                f"{nmi(y_s, duration_bucket(frac)):>9.4f}{1/K:>9.4f}")
            summary[f"{corpus}|floorfrac_balanced_acc"] = round(
                out_of_fold_1d(frac, y_s), 4)
            del arr

    # --------------------------------------------------------- P7 summary
    add("")
    add("=" * 100)
    add("P7  label-shuffling control")
    add("=" * 100)
    add("")
    n_shuf = 0
    for f in uniq:
        d = np.load(f, allow_pickle=True)
        if int(d["shuffled"]) if "shuffled" in d else 0:
            n_shuf += 1
            add(f"  {f.stem} is a shuffled control — compare its test accuracy "
                f"against chance in w3_tables.txt (table T9)")
    if not n_shuf:
        add("  no shuffled control run present; run w3_run.py --shuffle-labels")
    add("")
    add("  A shuffled run that sits at chance is the evidence that neither the")
    add("  protocol nor the padding is handing the model the answer.")

    text = "\n".join(L)
    print(text)
    (args.out / "tables" / "w3_diagnostics.txt").write_text(text, encoding="utf-8")
    (args.out / "tables" / "w3_diagnostics.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwritten: {args.out}/tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
