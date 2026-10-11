# =========================================================================== #
# W3-F selective probe: is a SELECTIVE (abstaining) classifier viable under
# the honest vessel-disjoint protocol?
#
# Context (from w3f_mechanism_probe): under honest evaluation OceanShip's
# 12-class problem collapses onto 3 acoustically separable classes
# (Towing / Tug / Cargo); no prior rebalancing can recover the tail
# (offline ceiling +2.0 pp macro-F1).  If that is the truth about the
# corpus, then an HONEST system should ABSTAIN on clips it cannot resolve.
# This probe measures, from the saved per-clip probabilities, whether the
# model's confidence separates answerable from unanswerable clips:
#
#   route B alive : accuracy at 50% coverage is far above the full-coverage
#                   accuracy AND tail-class clips are mostly abstained
#                   -> "Selective UATR" method paper (conformal-style wrapper)
#   route B dead  : the curve is flat (confidence does not track correctness)
#                   -> audit/measurement paper
#
# Same input as the mechanism probe: predictions/*.npz from the W3-A output.
# CPU only, ~2 minutes.
# =========================================================================== #
import glob
import json
import os
import time

import numpy as np

T0 = time.time()


def tick(msg):
    print(f"  [{time.time() - T0:6.1f}s] {msg}", flush=True)


root = None
args = sys_argv = __import__("sys").argv[1:]
if "--root" in args:
    root = args[args.index("--root") + 1]
if root is None:
    for cand in sorted(glob.glob("/kaggle/input/*")):
        hits = [h for h in glob.glob(os.path.join(cand, "**", "predictions"),
                                     recursive=True)
                if glob.glob(os.path.join(h, "oceanship__logmel__cnn_small__*.npz"))]
        if hits:
            root = hits[0]
            break
if root is None:
    raise SystemExit("no predictions/ folder found under /kaggle/input")
print("=" * 88)
print(f"W3-F selective probe   predictions root: {root}")
print("=" * 88)


def load_group(protocol):
    out = []
    for s in range(5):
        hits = glob.glob(os.path.join(
            root, f"oceanship__logmel__cnn_small__{protocol}__s{s}*.npz"))
        if hits:
            d = np.load(hits[0], allow_pickle=False)
            out.append(dict(classes=[str(c) for c in d["classes"]],
                            y_true=d["y_true"].astype(np.int64),
                            y_pred=d["y_pred"].astype(np.int64),
                            proba=d["proba"].astype(np.float64)))
    return out


GROUPS = {}
for proto in ("vessel_disjoint", "clip_random"):
    GROUPS[proto] = load_group(proto)
    tick(f"{proto:<16} {len(GROUPS[proto])} runs loaded")
VD = GROUPS["vessel_disjoint"]
if not VD:
    raise SystemExit("no vessel_disjoint runs found")

classes = VD[0]["classes"]
k = len(classes)


def scores_of(proba):
    srt = np.sort(proba, axis=1)
    return {
        "max-prob": proba.max(1),
        "margin": srt[:, -1] - srt[:, -2],
        "-entropy": -np.sum(-proba * np.log(np.clip(proba, 1e-12, None)), axis=1),
    }


def acc_at_coverage(y_true, y_pred, conf, cov):
    n = len(conf)
    keep = int(round(n * cov))
    idx = np.argsort(-conf)[:keep]
    return float((y_pred[idx] == y_true[idx]).mean())


COVS = (1.00, 0.90, 0.75, 0.50, 0.30)

report = {}
for proto, G in GROUPS.items():
    if not G:
        continue
    y_true = np.concatenate([g["y_true"] for g in G])
    y_pred = np.concatenate([g["y_pred"] for g in G])
    proba = np.concatenate([g["proba"] for g in G])
    SC = scores_of(proba)

    # per-class recall to identify the head classes this model actually answers
    rec = {c: float((y_pred[y_true == i] == i).mean())
           for i, c in enumerate(classes) if (y_true == i).any()}
    heads = sorted(rec, key=lambda c: -rec[c])[:3]
    head_idx = [classes.index(c) for c in heads]
    tail_idx = [i for i in range(k) if i not in head_idx]
    is_head_true = np.isin(y_true, head_idx)

    # a "head-only" system: answer only when the argmax is a head class
    head_ans = np.isin(y_pred, head_idx)
    head_only_acc = float((y_pred[head_ans] == y_true[head_ans]).mean())
    head_only_cov = float(head_ans.mean())
    tail_ans = head_ans[~is_head_true].mean() if (~is_head_true).any() else float("nan")

    print()
    print("=" * 88)
    print(f"{proto}  (pooled, {len(y_true):,} clips)   "
          f"full-coverage acc = {float((y_pred == y_true).mean()):.4f}")
    print(f"  head classes (recall top-3): {heads}")
    print(f"  head-only system (answer iff argmax in head): "
          f"accuracy {head_only_acc:.4f} at coverage {100*head_only_cov:.1f}%   "
          f"(tail clips answered: {100*tail_ans:.1f}%)")
    print()
    print(f"  {'score':<10}" + "".join(f"acc@{int(c*100)}%".rjust(9) for c in COVS))
    print("  " + "-" * (10 + 9 * len(COVS)))
    rows = {}
    for name, conf in SC.items():
        r = [acc_at_coverage(y_true, y_pred, conf, c) for c in COVS]
        rows[name] = r
        print(f"  {name:<10}" + "".join(f"{a:>9.4f}" for a in r))
    report[proto] = dict(heads=heads, head_only=dict(acc=head_only_acc,
                                                     cov=head_only_cov,
                                                     tail_answered=float(tail_ans)),
                         curves=rows,
                         full_acc=float((y_pred == y_true).mean()))

# ------------------------------------------------------------------ verdict #
vd = report.get("vessel_disjoint")
if vd:
    best50 = max(r[3] for r in vd["curves"].values())     # cov 0.50
    best30 = max(r[4] for r in vd["curves"].values())     # cov 0.30
    full = vd["full_acc"]
    print()
    print("=" * 88)
    print("VERDICT - is the selective (route B) paper alive?")
    print("=" * 88)
    print(f"  full coverage accuracy      : {full:.4f}")
    print(f"  best acc at 50% coverage    : {best50:.4f}   "
          f"(gain {100*(best50-full):+.1f} pp)")
    print(f"  best acc at 30% coverage    : {best30:.4f}   "
          f"(gain {100*(best30-full):+.1f} pp)")
    print(f"  tail clips answered by head-only system: "
          f"{100*vd['head_only']['tail_answered']:.1f}%")
    if best50 >= 0.65 and vd["head_only"]["tail_answered"] < 0.35:
        print("  -> ROUTE B ALIVE: confidence separates answerable from")
        print("     unanswerable clips.  A selective / conformal UATR paper is")
        print("     a real method contribution: model-agnostic wrapper,")
        print("     evaluated with the paired-draw protocol.")
    elif best50 >= 0.45:
        print("  -> PARTIAL: some selectivity exists but the operating points")
        print("     are weak; route B is a stretch, route A (audit paper) is")
        print("     the safer frame with the selective curve as one section.")
    else:
        print("  -> ROUTE B DEAD: the curve is flat - confidence does not track")
        print("     correctness.  Commit to the audit/measurement paper (A):")
        print("     protocol ladder + effective class structure + recipe.")

for cand in ("/kaggle/working", os.path.dirname(root), os.getcwd()):
    if cand and os.path.isdir(cand):
        with open(os.path.join(cand, "w3f_selective.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=float)
        print()
        tick(f"summary written to {os.path.join(cand, 'w3f_selective.json')}")
        break
