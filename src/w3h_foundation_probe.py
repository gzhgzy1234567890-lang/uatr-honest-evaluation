# =========================================================================== #
# W3-H foundation-encoder probe: is REPRESENTATION the bottleneck of the
# honest-protocol collapse?
#
# Context (W3-F): under vessel_disjoint, our small CNN collapses OceanShip's
# 12 classes onto 3 (Towing/Tug/Cargo); the tail's mean top-2 hit rate is
# 0.114 (~ chance); no prior rebalancing can recover it (offline ceiling
# +2.0 pp macro-F1); confidence does not track correctness.  All three dead
# method routes failed for the SAME reason: the information is not in OUR
# features.  The one intervention that diagnosis points to is a richer
# representation.  This probe tests the cheapest version of that: a frozen
# AudioSet-pretrained transformer (AST) on the SAME precomputed 128-mel
# features, scored with a linear probe under the SAME frozen splits.
#
# PRE-REGISTERED verdict (written before running, do not move the goalposts):
#   tail6 = {Fishing, Passenger, Pleasure Craft, Sailing,
#            Search and Rescue vessel, Tanker}   (the W3-F tail)
#   reference (our cnn_small, rec2, vessel_disjoint, 5 seeds):
#       acc 0.3183   macro-F1 0.108   tail mean top-2 0.114
#   GO   : tail mean top-2 >= 0.30  -> representation was the bottleneck;
#          the method paper opens (representation processing / distillation)
#   NO-GO: tail mean top-2 <= 0.15  -> even a pretrained AudioSet encoder
#          does not encode the tail; audit paper with an upgraded claim
#   else : PARTIAL - per-class breakdown decides which classes recover
#
# Input : uwatr-features dataset (oceanship/logmel.npy + index.csv +
#         splits/oceanship_splits.csv) - the SAME dataset as every W3 run.
# Output: /kaggle/working/w3h_ast_emb.npy (53,770 x 768 fp16, ~80 MB)
#         /kaggle/working/w3h_foundation.json
# Run   : GPU session (T4/P100, ~30-60 min total).  Do a SHORT run first:
#         set MAX_CLIPS = 2000 below (or --max-clips 2000) to validate the
#         wiring in ~3 minutes, then set it back to 0 for the full run.
# =========================================================================== #
import argparse
import glob
import gzip
import json
import os
import sys
import time

import numpy as np

T0 = time.time()
REF = dict(acc=0.3183, macro_f1=0.108, tail_top2=0.114)
TAIL6 = {"Fishing", "Passenger", "Pleasure Craft", "Sailing",
         "Search and Rescue vessel", "Tanker"}
GO_T2, NOGO_T2 = 0.30, 0.15
CLASSES = ["Anti-pollution equipment", "Cargo", "Dredging", "Fishing",
           "Passenger", "Pleasure Craft", "Port Tender", "Sailing",
           "Search and Rescue vessel", "Tanker", "Towing", "Tug"]


def tick(msg):
    print(f"  [{time.time() - T0:7.1f}s] {msg}", flush=True)


# ------------------------------------------------------------------- args -- #
ap = argparse.ArgumentParser()
ap.add_argument("--max-clips", type=int, default=0,
                help="0 = all 53,770; set 2000 for a quick wiring check")
ap.add_argument("--features-root", default="",
                help="override auto-discovery of the uwatr-features directory")
ap.add_argument("--model", default="MIT/ast-finetuned-audioset-10-10-0.4593",
                help="HF repo id OR a local directory with config.json + "
                     "model.safetensors (use a local dir if the hub is "
                     "unreachable from your network)")
args, _ = ap.parse_known_args()
MAX_CLIPS = args.max_clips

# ------------------------------------------------------------- locate data -- #
root = args.features_root or None
if root is None:
    for cand in sorted(glob.glob("/kaggle/input/**/uwatr-features",
                                 recursive=True)):
        if os.path.isfile(os.path.join(cand, "oceanship", "logmel.npy")):
            root = cand
            break
if root is None:
    raise SystemExit("uwatr-features (with oceanship/logmel.npy) not found "
                     "under /kaggle/input - is the dataset attached?")
FEAT = os.path.join(root, "oceanship", "logmel.npy")
print("=" * 88)
print(f"W3-H foundation probe   features: {FEAT}")
print("=" * 88)

import csv

with open(os.path.join(root, "oceanship", "index.csv"), encoding="utf-8",
          newline="") as fh:
    idx_rows = list(csv.DictReader(fh))
idx = {r["clip_id"]: r for r in idx_rows}
row_of = {r["clip_id"]: i for i, r in enumerate(idx_rows)}
split_path = None
for p in (os.path.join(root, "splits", "oceanship_splits.csv"),
          os.path.join(root, "splits", "oceanship_splits.csv.gz")):
    if os.path.isfile(p):
        split_path = p
        break
if split_path is None:
    raise SystemExit("oceanship_splits not found in the dataset")

opener = gzip.open if split_path.endswith(".gz") else open
splits = {s: {} for s in range(5)}          # seed -> clip_id -> split
with opener(split_path, "rt", newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        s = int(r["seed"])
        if s < 5 and r["protocol"] == "vessel_disjoint":
            splits[s][r["clip_id"]] = r["split"]
for s in range(5):
    if not splits[s]:
        raise SystemExit(f"vessel_disjoint seed {s} has no rows in {split_path} "
                         f"- if the dataset was refreshed to 10 seeds this "
                         f"should not happen; check the file")
tick(f"index {len(idx):,} clips; splits 5 seeds loaded")

arr = np.load(FEAT, mmap_mode="r")
N = arr.shape[0]
n_use = N if MAX_CLIPS == 0 else min(N, MAX_CLIPS)
tick(f"array {arr.shape} {arr.dtype}; using {n_use:,} clips")

# ------------------------------------------------------------------- model -- #
import torch

from transformers import ASTModel

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type != "cuda":
    print("  [warn] no GPU visible - extraction will be ~10x slower")
model = ASTModel.from_pretrained(args.model)
model = model.eval().to(dev)
EMB = model.config.hidden_size
FREQ, FRAMES = 128, 313


def patch_grid(m, freq, frames):
    """Empirical patch grid: run the patch conv on zeros and read the shape."""
    holder = getattr(m.embeddings, "patch_embeddings", m.embeddings)
    conv = getattr(holder, "projection", None) or getattr(holder, "conv_proj")
    with torch.no_grad():
        out = conv(torch.zeros(1, 1, freq, frames, device=dev))
    return out.shape[2], out.shape[3]


def adapt_pos_enc(m):
    """Interpolate AST's positional embeddings from the trained grid to ours.

    Table layout (HF AST): [cls] + patches(oh x ow) + [distillation token(s)].
    Only the patch block is resampled; the special tokens stay untouched.
    """
    pe = m.embeddings.position_embeddings.data          # (1, 1+P+extra, D)
    D, n_pos = pe.shape[-1], pe.shape[1]
    max_len = int(getattr(m.config, "max_length", 1024))
    oh, ow = patch_grid(m, FREQ, max_len)
    gh, gw = patch_grid(m, FREQ, FRAMES)
    n_patch = oh * ow
    if not (0 <= n_pos - n_patch <= 2):
        raise RuntimeError(f"pos table {n_pos} vs patch grid {oh}x{ow}"
                           f"={n_patch}; layout unknown")
    head = pe[:, :1]                                    # cls
    tail = pe[:, 1 + n_patch:]                          # distill / extras
    grid = pe[:, 1:1 + n_patch].reshape(1, oh, ow, D).permute(0, 3, 1, 2)
    new = torch.nn.functional.interpolate(grid, size=(gh, gw), mode="bilinear",
                                          align_corners=False)
    new = new.permute(0, 2, 3, 1).reshape(1, gh * gw, D)
    m.embeddings.position_embeddings = torch.nn.Parameter(
        torch.cat([head, new, tail], dim=1))
    tick(f"positional embeddings interpolated {oh}x{ow} -> {gh}x{gw}")


@torch.no_grad()
def encode(x, mode="direct"):
    """x: float32 (B, 128, 313) already per-clip standardized.  Returns (B, D).

    Fallback chain: direct forward -> positional-embedding surgery -> resample
    the input to the native 128x1024 (time-stretch; frequency content is
    unchanged, acceptable for a probe on stationary ship noise).
    """
    if mode == "resample":
        x = torch.nn.functional.interpolate(
            torch.from_numpy(x).unsqueeze(1).to(dev), size=(FREQ, 1024),
            mode="bilinear", align_corners=False).squeeze(1)
    else:
        x = torch.from_numpy(x).to(dev)
    out = model(x).last_hidden_state
    return out.mean(dim=1).float().cpu().numpy()        # time-pool


_encode_mode = "direct"


def encode_auto(x):
    global _encode_mode
    if _encode_mode != "direct":
        return encode(x, mode=_encode_mode)
    try:
        return encode(x, mode="direct")
    except Exception as exc:
        print(f"  [fallback 1/2] direct forward failed: {str(exc)[:140]}")
        try:
            adapt_pos_enc(model)
            _ = encode(x[:2], mode="direct")
            _encode_mode = "direct"
            tick("positional surgery works - continuing with native 313 frames")
            return encode(x, mode="direct")
        except Exception as exc2:
            print(f"  [fallback 2/2] pos surgery failed: {str(exc2)[:140]}")
            _ = encode(x[:2], mode="resample")
            _encode_mode = "resample"
            tick("input resample to 128x1024 works - continuing (time-stretched)")
    return encode(x, mode=_encode_mode)


def standardize(batch):
    mu = batch.mean(axis=(1, 2), keepdims=True)
    sd = batch.std(axis=(1, 2), keepdims=True) + 1e-6
    return np.clip((batch - mu) / sd, -10.0, 10.0)


# ------------------------------------------------- extraction (with smoke) -- #
smoke_n = min(n_use, 128)
_ = encode_auto(standardize(
    np.array(arr[:smoke_n], dtype=np.float32)))         # wiring check
tick(f"smoke batch of {smoke_n} encoded OK (mode={_encode_mode})")

EMB_CANDIDATES = ["/kaggle/working", os.path.dirname(os.path.abspath(FEAT)),
                  os.getcwd()]
EMB_DIR = next((c for c in EMB_CANDIDATES if c and os.path.isdir(c)
                and os.access(c, os.W_OK)), os.getcwd())
EMB_FILE = os.path.join(EMB_DIR, "w3h_ast_emb.npy")
emb = np.zeros((N, EMB), dtype=np.float16)
B = 64
t0 = time.time()
done = 0
for s in range(0, n_use, B):
    e = min(s + B, n_use)
    x = standardize(np.array(arr[s:e], dtype=np.float32))
    emb[s:e] = encode_auto(x).astype(np.float16)
    done = e
    if done % (B * 50) == 0 or done == n_use:
        rate = done / (time.time() - t0)
        eta = (n_use - done) / max(rate, 1e-9)
        tick(f"encoded {done:,}/{n_use:,}  ({rate:.0f}/s, ETA {eta/60:.0f} min)")
np.save(EMB_FILE, emb[:n_use])
tick(f"embeddings saved -> {EMB_FILE}")

# ------------------------------------------------------ linear probes 5 seed -- #
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

X = emb[:n_use].astype(np.float32)
ids = list(row_of.keys())[:n_use]
labels = {cid: idx[cid]["label"] for cid in ids if cid in idx}
keep_ids = [c for c in ids if c in labels and c in splits[0]]
kept = sorted({labels[c] for c in keep_ids})
c2i = {c: i for i, c in enumerate(kept)}
print(f"  classes ({len(kept)}): {kept}")

accs, f1s, percls = [], [], {}
for s in range(5):
    tr = [c for c in keep_ids if splits[s][c] == "train"]
    te = [c for c in keep_ids if splits[s][c] == "test"]
    Xtr = X[[row_of[c] for c in tr]]
    ytr = np.array([c2i[labels[c]] for c in tr])
    Xte = X[[row_of[c] for c in te]]
    yte = np.array([c2i[labels[c]] for c in te])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit((Xtr - mu) / sd, ytr)
    P = clf.predict_proba((Xte - mu) / sd)
    yp = P.argmax(1)
    accs.append(float((yp == yte).mean()))
    f1s.append(float(f1_score(yte, yp, average="macro")))
    order = np.argsort(-P, axis=1)
    for c in kept:
        i = c2i[c]
        m = yte == i
        if not m.any():
            continue
        r = float((yp[m] == i).mean())
        t2 = float((order[m][:, :2] == i).any(1).mean())
        d = percls.setdefault(c, dict(rec=[], t2=[]))
        d["rec"].append(r)
        d["t2"].append(t2)
    tick(f"seed {s}: acc {accs[-1]:.4f}  macro-F1 {f1s[-1]:.4f}")

print()
print("=" * 88)
print("per-class, frozen AST embeddings + linear probe (mean over 5 seeds)")
print("=" * 88)
print(f"  {'class':<28}{'recall':>8}{'top2':>8}{'cnn_small ref':>15}")
print("  " + "-" * 62)
REF_REC = {"Cargo": 0.584, "Tug": 0.367, "Towing": 0.722, "Fishing": 0.022,
           "Pleasure Craft": 0.019, "Port Tender": 0.007}
for c in CLASSES:
    if c not in percls:
        continue
    r = float(np.mean(percls[c]["rec"]))
    t2 = float(np.mean(percls[c]["t2"]))
    ref = REF_REC.get(c)
    ref_txt = f"{ref:>15.3f}" if ref is not None else f"{'-':>15}"
    tail_mark = "  <- tail6" if c in TAIL6 else ""
    print(f"  {c:<28}{r:>8.3f}{t2:>8.3f}{ref_txt}{tail_mark}")

tail_t2 = float(np.mean([np.mean(percls[c]["t2"]) for c in TAIL6
                         if c in percls]))
tail_rec = float(np.mean([np.mean(percls[c]["rec"]) for c in TAIL6
                          if c in percls]))
print()
print("=" * 88)
print("VERDICT (pre-registered: GO >= 0.30, NO-GO <= 0.15 tail mean top-2)")
print("=" * 88)
print(f"  cnn_small reference : acc {REF['acc']:.4f}  macro-F1 {REF['macro_f1']:.3f}"
      f"  tail top-2 {REF['tail_top2']:.3f}")
print(f"  frozen AST + probe  : acc {np.mean(accs):.4f}+-{np.std(accs):.4f}"
      f"  macro-F1 {np.mean(f1s):.3f}  tail top-2 {tail_t2:.3f}"
      f"  tail recall {tail_rec:.3f}")
if tail_t2 >= GO_T2:
    verdict = "GO"
    print("  -> GO: representation WAS the bottleneck.  The method paper is")
    print("     open: representation processing (fine-tune / distillation /")
    print("     feature selection) on top of pretrained encoders, evaluated")
    print("     with the paired-draw honest protocol.")
elif tail_t2 <= NOGO_T2:
    verdict = "NO-GO"
    print("  -> NO-GO: even a pretrained AudioSet encoder does not encode the")
    print("     tail.  The collapse is corpus-intrinsic; commit to the audit")
    print("     paper with the upgraded claim (capacity, pretraining: neither")
    print("     fixes it), and target 3-region journals realistically.")
else:
    verdict = "PARTIAL"
    print("  -> PARTIAL: some tail classes recover, some do not.  The paper")
    print("     reports which and why (compare vessel counts and confusion")
    print("     structure); the method section covers the recoverable ones.")

out = dict(kind="w3h-foundation-probe", encoder=f"AST frozen ({args.model})",
           n_clips=int(n_use), classes=kept,
           acc_mean=float(np.mean(accs)), acc_sd=float(np.std(accs)),
           macro_f1_mean=float(np.mean(f1s)),
           tail_top2=tail_t2, tail_recall=tail_rec, verdict=verdict,
           per_class={c: dict(recall=float(np.mean(v["rec"])),
                              top2=float(np.mean(v["t2"])))
                      for c, v in percls.items()})
for cand in ("/kaggle/working", os.getcwd()):
    if cand and os.path.isdir(cand):
        with open(os.path.join(cand, "w3h_foundation.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)
        print()
        tick(f"summary -> {os.path.join(cand, 'w3h_foundation.json')}")
        break
