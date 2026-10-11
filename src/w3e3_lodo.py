# ============================================================================================== #
# W3-E3 — leave-one-draw-out (LODO) composition prediction: is the per-draw accuracy predictable   #
#         OUT of sample, not just in-sample?                                                       #
#                                                                                                  #
# Review Major 1: the composition->accuracy regression and the variance share it explains were      #
# measured on the same draws. Here, for every draw d we fit the composition regression on all       #
# OTHER draws and predict d's accuracy from its composition alone. Reported per draw: observed,     #
# predicted, absolute error; overall: RMSE, MAE, Pearson r(predicted, observed) with permutation    #
# p, and the LODO R^2 next to the in-sample R^2.                                                    #
#                                                                                                  #
# CPU only.  Mount the datasets that hold the predictions of the cell you want audited              #
# (default: oceanship/cnn_small/vessel_disjoint/rec2, the 25-draw pool).                            #
# ============================================================================================== #

import argparse
import collections
import glob
import os
import pathlib
import re
import statistics as st

import numpy as np

T0 = __import__("time").time()


def tick(msg):
    print(f"  [{__import__('time').time() - T0:6.1f}s] {msg}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="oceanship")
ap.add_argument("--rep", default="logmel")
ap.add_argument("--backbone", default="cnn_small")
ap.add_argument("--protocol", default="vessel_disjoint")
ap.add_argument("--tag", default="rec2")
ap.add_argument("--min-clips", type=int, default=50)
ap.add_argument("--min-draws", type=int, default=14,
                help="LODO needs n > #composition predictors + 1 to be meaningful")
ap.add_argument("--n-perm", type=int, default=5000)
ap.add_argument("--input-root", action="append", default=None)
ap.add_argument("--out", default="")
args, _ = ap.parse_known_args()
OUT = args.out or next((c for c in ("/kaggle/working", os.getcwd()) if os.path.isdir(c)), ".")
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(0)

input_roots = args.input_root or ["/kaggle/input"]
files = []
for root in input_roots:
    for f in glob.glob(os.path.join(root, "**", "predictions", "*.npz"), recursive=True):
        rel = pathlib.PurePath(f).relative_to(root)
        if any(p.startswith("_") for p in rel.parts[:-2]):
            continue
        files.append(f)
if not files:
    raise SystemExit("no predictions/*.npz found under the given input roots")
tick(f"{len(files)} prediction files under {len(input_roots)} input root(s)")

RID = re.compile(r"(.+?)__(logmel|spec)__(\w+?)__(.+?)__s(\d+)(?:__(.+))?$")
target = (args.corpus, args.rep, args.backbone, args.protocol, args.tag)
loaded = {}
skipped_probe = 0
for f in sorted(files):
    rid = os.path.splitext(os.path.basename(f))[0]
    m = RID.match(rid)
    if m is None:
        continue
    if "probe" in (m.group(6) or "").lower():
        skipped_probe += 1
        continue
    key = (m.group(1), m.group(2), m.group(3), m.group(4), (m.group(6) or ""))
    if key != target:
        continue
    d = np.load(f, allow_pickle=True)
    y_true = np.asarray(d["y_true"])
    if len(y_true) < args.min_clips:
        continue
    y_pred = np.asarray(d["y_pred"])
    classes, counts = np.unique(y_true, return_counts=True)
    loaded[rid] = dict(seed=int(m.group(5)),
                       acc=float(np.mean(y_pred == y_true)),
                       comp=(classes.tolist(), (counts / counts.sum()).tolist()))
if skipped_probe:
    tick(f"[skip] {skipped_probe} probe-tagged run ids excluded")
tick(f"{len(loaded)} draws in cell {args.corpus}/{args.backbone}/{args.protocol}/{args.tag}")

draws = sorted(loaded.values(), key=lambda d: d["seed"])
n = len(draws)
if n == 0:
    raise SystemExit(f"no draws found for cell {args.corpus}/{args.rep}/{args.backbone}/"
                     f"{args.protocol}/{args.tag} -- mount the dataset that holds them "
                     "(e.g. w3n4-data for the 25-draw OceanShip pool)")
all_classes = sorted({c for d in draws for c in d["comp"][0]})
K = len(all_classes)
spaces = {tuple(d["comp"][0]) for d in draws}
if len(spaces) > 1:
    raise SystemExit(f"mixed label spaces ({len(spaces)} variants) in the target cell -- "
                     "composition alignment by index would be meaningless; fix the pool first")
p = K - 1                                   # one share dropped (shares sum to 1)
X = np.zeros((n, K))
for i, d in enumerate(draws):
    for c, s in zip(d["comp"][0], d["comp"][1]):
        X[i, all_classes.index(c)] = s
Xd = np.column_stack([np.ones(n), X[:, 1:]])
y = np.array([d["acc"] for d in draws])
seeds = [d["seed"] for d in draws]

print()
print(f"cell: {args.corpus}/{args.rep}/{args.backbone}/{args.protocol}/{args.tag}")
print(f"  n={n} draws (seeds {seeds[0]}..{seeds[-1]}), {K} classes, "
      f"total test SD {100 * st.stdev(y):.2f} pp")
if n <= p + 1:
    raise SystemExit(f"n={n} <= predictors+1={p + 1}: the regression is saturated; "
                     "collect more draws before auditing out-of-sample prediction")
if n < args.min_draws:
    raise SystemExit(f"n={n} < --min-draws={args.min_draws}: LODO would be unstable")


def ols_fit(Xm, yv):
    beta, _, _, _ = np.linalg.lstsq(Xm, yv, rcond=None)
    return beta


# in-sample reference
beta_in = ols_fit(Xd, y)
pred_in = Xd @ beta_in
ss_res = float(((y - pred_in) ** 2).sum())
ss_tot = float(((y - y.mean()) ** 2).sum())
r2_in = 1.0 - ss_res / ss_tot

# leave-one-draw-out
pred_lodo = np.empty(n)
for i in range(n):
    mask = np.ones(n, dtype=bool)
    mask[i] = False
    beta = ols_fit(Xd[mask], y[mask])
    pred_lodo[i] = Xd[i] @ beta

err = y - pred_lodo
rmse = float(np.sqrt((err ** 2).mean()))
mae = float(np.abs(err).mean())
r = float(np.corrcoef(pred_lodo, y)[0, 1])
r2_lodo = 1.0 - float((err ** 2).sum()) / ss_tot
hits = 0
for _ in range(args.n_perm):
    yp = rng.permutation(y)
    rp = float(np.corrcoef(pred_lodo, yp)[0, 1])
    if abs(rp) >= abs(r):
        hits += 1
perm_p = (hits + 1) / (args.n_perm + 1)

print()
print("per-draw leave-one-draw-out prediction (fit on the other draws, predict this one):")
print(f"  {'seed':>5}{'observed':>10}{'predicted':>11}{'error':>9}")
for i in np.argsort(np.abs(err))[::-1]:
    print(f"  {seeds[i]:>5}{y[i]:>10.4f}{pred_lodo[i]:>11.4f}{err[i]:>+9.4f}")
print()
print(f"  in-sample R^2       : {r2_in:.3f}")
print(f"  LODO R^2            : {r2_lodo:.3f}")
print(f"  LODO RMSE / MAE     : {100 * rmse:.2f} pp / {100 * mae:.2f} pp")
print(f"  Pearson r(pred,obs) : {r:.3f}   permutation p = {perm_p:.4f}")
print()
print("reading: a LODO R^2 close to the in-sample value means the composition model predicts")
print("accuracies it has never seen; a large gap means the share rides on in-sample fitting.")

out_json = os.path.join(OUT, "w3e3_lodo.json")
import json  # noqa: E402
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(dict(kind="w3e3-lodo", cell="/".join(map(str, target)), n=n,
                   in_sample_r2=r2_in, lodo_r2=r2_lodo, rmse_pp=100 * rmse,
                   mae_pp=100 * mae, pearson_r=r, perm_p=perm_p,
                   seeds=seeds, observed=y.tolist(), predicted=pred_lodo.tolist()),
              fh, indent=1)
tick(f"summary -> {out_json}")
