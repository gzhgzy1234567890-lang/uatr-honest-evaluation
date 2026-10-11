# ============================================================================================== #
# W3-E — recording-cluster bootstrap confidence intervals + paired protocol tests                 #
#                                                                                                #
# Input : any mounted Kaggle dataset containing  w3-runs/predictions/*.npz  (written by           #
#         w3_run.py: per-clip y_true / y_pred / rec_key / vessel) — e.g. W3_data, plus any        #
#         later result folders you saved (W3K_data, w3k2-runs ...).                                #
# Output: per-run recording-cluster bootstrap 95% CI of test accuracy; per-protocol summary        #
#         (mean of draws, CI on the mean, draw-to-draw SD with CI); paired per-seed differences    #
#         between protocols with a paired t-test.  Everything is printed as a ready-to-paste       #
#         table and saved to w3e_bootstrap.json.                                                   #
#                                                                                                #
# CPU only, minutes.  Attach: the dataset(s) that hold your predictions folders.                  #
# ============================================================================================== #

import argparse
import collections
import glob
import json
import os
import pathlib
import re
import statistics as st

import numpy as np

T0 = __import__("time").time()


def tick(msg):
    print(f"  [{__import__('time').time() - T0:6.1f}s] {msg}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--n-boot", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--min-clips", type=int, default=50,
                help="skip runs with fewer test clips (fixture data)")
ap.add_argument("--input-root", action="append", default=None,
                help="where mounted datasets live (repeatable: session output in "
                     "/kaggle/working can be scanned alongside /kaggle/input)")
ap.add_argument("--out", default="", help="output dir (default: first writable of /kaggle/working, cwd)")
args, _ = ap.parse_known_args()
OUT = args.out or next((c for c in ("/kaggle/working", os.getcwd()) if os.path.isdir(c)), ".")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------- discovery ---- #
# '_'-prefixed directories are scratch -- but judged RELATIVE to each input root
# (an absolute-path check also kills a fixture root that starts with '_').
input_roots = args.input_root or ["/kaggle/input"]
files = []
for root in input_roots:
    for f in glob.glob(os.path.join(root, "**", "predictions", "*.npz"), recursive=True):
        rel = pathlib.PurePath(f).relative_to(root)
        if any(p.startswith("_") for p in rel.parts[:-2]):
            continue
        files.append(f)
if not files:
    raise SystemExit("no predictions/*.npz found under the given input roots -- attach the "
                     "dataset(s) that hold your w3-runs / w3k-runs / w3m-runs output")
roots = sorted(set(os.path.dirname(f) for f in files))
tick(f"prediction roots: {len(roots)}")
for r in roots:
    tick(f"  {r}")

rng = np.random.RandomState(args.seed)
B = args.n_boot


def cluster_boot_ci(correct_by_rec, n_boot=B):
    """Recording-cluster bootstrap: resample recordings with replacement; a recording
    sampled twice contributes its clips twice.  Returns (point, lo, hi).  Fully vectorized."""
    recs = list(correct_by_rec)
    sums = np.array([sum(correct_by_rec[r]) for r in recs], dtype=np.int64)
    lens = np.array([len(correct_by_rec[r]) for r in recs], dtype=np.int64)
    R = len(recs)
    point = float(sums.sum() / max(lens.sum(), 1))
    idx = rng.randint(0, R, size=(n_boot, R))
    totals = sums[idx].sum(axis=1)
    counts = lens[idx].sum(axis=1)
    boots = totals / np.maximum(counts, 1)
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ------------------------------------------------- load runs and bootstrap -- #
# Cells are keyed WITH the recipe/allocator tag: pooling a clip-mass remedy run
# with a plain random draw of the same protocol mixes two different variance
# regimes into one "draw SD" (exactly what the first Kaggle pass did -- six
# tags silently averaged, probes included).
runs = {}
cells = collections.defaultdict(list)
n_probe = 0
for root in roots:
    for f in sorted(glob.glob(os.path.join(root, "*.npz"))):
        rid = os.path.splitext(os.path.basename(f))[0]
        d = np.load(f, allow_pickle=True)
        y_true, y_pred = d["y_true"], d["y_pred"]
        if len(y_true) < args.min_clips:      # fixture-size runs are not paper data
            continue
        correct = (y_pred == y_true)
        by_rec = collections.defaultdict(list)
        for c, r in zip(correct, d["rec_key"]):
            by_rec[str(r)].append(int(c))
        pt, lo, hi = cluster_boot_ci(by_rec)
        m = re.match(r"(.+?)__(logmel|spec)__(\w+?)__(.+?)__s(\d+)(?:__(.+))?$", rid)
        if m is None:
            tick(f"  [skip] unparseable run id: {rid}")
            continue
        corpus, rep, backbone, proto, seed, tag = (m.group(1), m.group(2), m.group(3),
                                                   m.group(4), int(m.group(5)),
                                                   m.group(6) or "")
        if "probe" in tag.lower():
            n_probe += 1
            continue
        runs[rid] = dict(corpus=corpus, rep=rep, backbone=backbone, proto=proto,
                         seed=seed, tag=tag, point=pt, lo=lo, hi=hi,
                         n_clips=len(y_true), n_recs=len(by_rec))
if n_probe:
    tick(f"[skip] {n_probe} probe-tagged runs excluded from the paper numbers")
tick(f"{len(runs)} usable runs bootstrapped ({B} replicates each)")

# ------------------------------------------------------ per-protocol table -- #
cells = collections.defaultdict(list)
for rid, r in runs.items():
    key = (r["corpus"], r["rep"], r["backbone"], r["proto"], r["tag"])
    cells[key].append(r)

print()
print("=" * 100)
print("per-protocol summary: mean of draws, recording-cluster bootstrap CI of a single draw,")
print("t-interval of the draw mean, draw-to-draw SD")
print("=" * 100)
summary = {}
for key in sorted(cells):
    rs = sorted(cells[key], key=lambda r: r["seed"])
    pts = [r["point"] for r in rs]
    label = (f"{key[0]}/{key[2]}/{key[3]}/{key[4]}"
             + (f" [{key[1]}]" if key[1] != "logmel" else ""))
    if len(pts) > 1:
        sd = 100 * st.stdev(pts)
        sem = st.stdev(pts) / len(pts) ** 0.5
        try:
            from scipy import stats as sps
            t = sps.t.interval(0.95, len(pts) - 1, loc=st.mean(pts),
                               scale=sem if sem > 0 else 1e-12)
        except Exception:
            t = (st.mean(pts) - 1.96 * sem, st.mean(pts) + 1.96 * sem)
        mean_ci = (f"mean {st.mean(pts):.4f}  t95 [{t[0]:.4f},{t[1]:.4f}]  "
                   f"drawSD {sd:.2f}pp")
    else:
        sd = float("nan")
        mean_ci = f"mean {pts[0]:.4f}  (1 draw)"
    ciw = [r["hi"] - r["lo"] for r in rs]
    print(f"  {label:<52} n={len(rs)}  {mean_ci}  "
          f"draw-CI width {st.mean(ciw) * 100:.1f}pp")
    summary["/".join(key)] = dict(
        n=len(rs), draws={r["seed"]: dict(point=r["point"], lo=r["lo"], hi=r["hi"])
                          for r in rs},
        mean=float(st.mean(pts)) if pts else None,
        draw_sd_pp=sd if sd == sd else None,
        ci_width_pp=float(st.mean(ciw)) if ciw else None)

# ------------------------------------------------- paired protocol tests ---- #
print()
print("=" * 100)
print("paired per-seed protocol differences (same split seed = same draw), paired t-test")
print("=" * 100)
by_key = collections.defaultdict(dict)
for key in cells:
    corpus, rep, backbone, proto, tag = key
    for r in cells[key]:
        by_key[(corpus, rep, backbone, tag)][(proto, r["seed"])] = r["point"]
pairs = {}
for (corpus, rep, backbone, tag), m in sorted(by_key.items()):
    protos = sorted({p for p, _ in m})
    for i in range(len(protos)):
        for j in range(i + 1, len(protos)):
            pa, pb = protos[i], protos[j]
            shared = sorted({s for p, s in m if p == pa} & {s for p, s in m if p == pb})
            if len(shared) < 3:
                continue
            da = [m[(pa, s)] for s in shared]
            db = [m[(pb, s)] for s in shared]
            diff = [100 * (a - b) for a, b in zip(da, db)]
            try:
                from scipy import stats as sps
                t, p = sps.ttest_rel(da, db)
            except Exception:
                t, p = float("nan"), float("nan")
            print(f"  {corpus}/{backbone}/{tag}: {pa} - {pb}  over seeds {shared}")
            print(f"      paired diff {st.mean(diff):+.1f} pp (SD {st.stdev(diff):.1f}), "
                  f"paired t p = {p:.3f}")
            pairs[f"{corpus}/{backbone}/{tag}/{pa}-{pb}"] = dict(
                seeds=shared, mean_diff_pp=st.mean(diff), sd_pp=st.stdev(diff), p=float(p))

# ------------------------------------------------------------------ save ---- #
with open(os.path.join(OUT, "w3e_bootstrap.json"), "w", encoding="utf-8") as fh:
    json.dump(dict(kind="w3e-bootstrap", n_boot=B, summary=summary, paired=pairs),
              fh, indent=2, default=float)
print()
tick("summary -> w3e_bootstrap.json")
print("  -> Output tab -> Create Dataset to keep it")
