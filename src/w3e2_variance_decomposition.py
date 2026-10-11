# ============================================================================================== #
# W3-E2 — variance decomposition on ALL draws: how much of the honest protocol's draw variance    #
#         is test-set class composition?                                                          #
#                                                                                                 #
# Why: the paper's headline "85% of the vessel-disjoint draw variance is composition" was computed #
#      on the FIRST 5 draws.  The Kaggle side has since accumulated ~25 independent vessel draws  #
#      (w3-data / w3k-data / w3k2-data).  This script recomputes the decomposition on every draw  #
#      that exists, without retraining anything.                                                  #
#                                                                                                 #
# Method (per corpus/rep/backbone/protocol cell with enough draws):                               #
#      acc_d   = clip accuracy of draw d                                                          #
#      shares_d[k] = fraction of draw d's test clips whose true class is k                        #
#      OLS  acc ~ shares (one class dropped to avoid collinearity)                                #
#      explained share = R^2 of that regression (composition variance / total variance)           #
#      permutation p = P(R^2_perm >= R^2_obs) over shuffling the acc vector (5000 perms)          #
# Draws fewer than --min-draws, or n_draws <= n_predictors+1, are reported as "n/a" honestly --    #
# a saturated regression would fake an R^2 of 1.                                                  #
#                                                                                                 #
# CPU only, minutes.  Attach the dataset(s) that hold your predictions folders.                    #
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
ap.add_argument("--min-clips", type=int, default=50,
                help="skip runs with fewer test clips (fixture data)")
ap.add_argument("--min-draws", type=int, default=8,
                help="cells with fewer draws get no regression (report n/a)")
ap.add_argument("--n-perm", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--input-root", action="append", default=None,
                help="where mounted datasets live (repeatable: session output in "
                     "/kaggle/working can be scanned alongside /kaggle/input)")
ap.add_argument("--out", default="", help="output dir (default: first writable of /kaggle/working, cwd)")
args, _ = ap.parse_known_args()
OUT = args.out or next((c for c in ("/kaggle/working", os.getcwd()) if os.path.isdir(c)), ".")
os.makedirs(OUT, exist_ok=True)
rng = np.random.RandomState(args.seed)

# ------------------------------------------------------------- discovery ---- #
# '_'-prefixed directories are scratch -- but judged RELATIVE to each input root:
# checking absolute paths would also kill a fixture root that itself starts
# with '_' (a real bug the local fixture caught).
input_roots = args.input_root or ["/kaggle/input"]
all_npz = []
for root in input_roots:
    for f in glob.glob(os.path.join(root, "**", "predictions", "*.npz"), recursive=True):
        rel = pathlib.PurePath(f).relative_to(root)
        if any(p.startswith("_") for p in rel.parts[:-2]):
            continue
        all_npz.append(f)
if not all_npz:
    raise SystemExit("no predictions/*.npz found -- attach the dataset(s) that hold "
                     "your w3-runs / w3k-runs / w3m-runs output")
roots = sorted(set(os.path.dirname(f) for f in all_npz))
tick(f"prediction roots: {len(roots)}")
for r in roots:
    tick(f"  {r}")

# ---------------------------------------------------------------- loading --- #
# Cells are keyed WITH the recipe/allocator tag.  The first Kaggle run pooled
# six tags into one cell and the decomposition silently mixed between-scheme
# variance into the residual -- clipmass runs (the variance-reducing remedy)
# sat in the same regression as plain random draws.  Different tags are
# different protocols as far as draw variance is concerned.
cells = collections.defaultdict(list)
RID = re.compile(r"(.+?)__(logmel|spec)__(\w+?)__(.+?)__s(\d+)(?:__(.+))?$")
skipped_tags = collections.Counter()
loaded = {}                           # dedup by run_id: when the same run exists in a
for root in roots:                    # mounted dataset AND the session output, the
    for f in sorted(glob.glob(os.path.join(root, "*.npz"))):   # last copy scanned wins
        rid = os.path.splitext(os.path.basename(f))[0]
        m = RID.match(rid)
        if m is None:
            tick(f"  [skip] unparseable run id: {rid}")
            continue
        corpus, rep, backbone, proto = m.group(1), m.group(2), m.group(3), m.group(4)
        seed, tag = int(m.group(5)), (m.group(6) or "")
        if "probe" in tag.lower():
            skipped_tags[rid] += 1
            continue
        d = np.load(f, allow_pickle=True)
        y_true = np.asarray(d["y_true"])
        if len(y_true) < args.min_clips:
            continue
        y_pred = np.asarray(d["y_pred"])
        acc = float(np.mean(y_pred == y_true))
        classes, counts = np.unique(y_true, return_counts=True)
        loaded[rid] = dict(seed=seed, acc=acc, n=len(y_true), tag=tag,
                           corpus=corpus, rep=rep, backbone=backbone, proto=proto,
                           comp=(classes.tolist(), (counts / counts.sum()).tolist()))
if skipped_tags:
    tick(f"[skip] {len(skipped_tags)} probe-tagged run ids excluded from the paper numbers")
for rid, d in loaded.items():
    cells[(d["corpus"], d["rep"], d["backbone"], d["proto"], d["tag"])].append(d)
tick(f"{len(loaded)} usable runs loaded (deduplicated by run_id)")


def ols_r2(X, y):
    """R^2 of OLS y ~ X (X already full-rank by construction)."""
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ------------------------------------------------------ per-cell analysis --- #
print()
print("=" * 104)
print("variance decomposition: how much of the draw-to-draw variance is test-set class composition?")
print("  cells with n < min-draws, or n <= #predictors+1, are reported n/a (a saturated")
print("  regression would fake R^2 = 1)")
print("=" * 104)
hdr = (f"  {'cell':<52}{'n':>4}{'SD(pp)':>8}{'expl%':>8}{'permP':>8}  note")
print(hdr)

summary = {}
for key in sorted(cells):
    draws = sorted(cells[key], key=lambda d: d["seed"])
    accs = np.array([d["acc"] for d in draws])
    n = len(draws)
    sd_pp = 100 * st.stdev(accs) if n > 1 else 0.0
    label = (f"{key[0]}/{key[2]}/{key[3]}/{key[4]}"
             + (f" [{key[1]}]" if key[1] != "logmel" else ""))

    # union class space, aligned composition matrix
    all_classes = sorted({c for d in draws for c in d["comp"][0]})
    K = len(all_classes)
    X = np.zeros((n, K))
    for i, d in enumerate(draws):
        for c, s in zip(d["comp"][0], d["comp"][1]):
            X[i, all_classes.index(c)] = s
    Xd = X[:, 1:]                      # drop one share: shares sum to 1
    p = Xd.shape[1]

    note = ""
    # label-space guard: composition vectors are aligned by CLASS INDEX, which is
    # only meaningful if every draw in the cell shares one label space.  The
    # first w3n pass mixed 12- and 15-class draws and produced a fake 97.4%
    # explained share through exactly this hole.
    spaces = {tuple(d["comp"][0]) for d in draws}
    if len(spaces) > 1:
        note = f"mixed label spaces ({len(spaces)} variants): excluded"
        r2, perm_p, expl = None, None, None
    elif n < args.min_draws:
        note = f"n<{args.min_draws}: no regression"
        r2, perm_p, expl = None, None, None
    elif n <= p + 1:
        note = f"n={n} <= predictors+1={p + 1}: saturated"
        r2, perm_p, expl = None, None, None
    else:
        Xd1 = np.column_stack([np.ones(n), Xd])
        r2 = ols_r2(Xd1, accs)
        expl = max(0.0, min(1.0, r2)) * 100.0
        hits = 0
        for _ in range(args.n_perm):
            yp = rng.permutation(accs)
            if ols_r2(Xd1, yp) >= r2:
                hits += 1
        perm_p = (hits + 1) / (args.n_perm + 1)

    r2s = "n/a" if r2 is None else f"{expl:7.1f}"
    pps = "n/a" if perm_p is None else f"{perm_p:7.4f}"
    tags = dict(collections.Counter(d["tag"] for d in draws))
    tag_s = f"  tags: {tags}" if len(tags) > 1 else ""
    print(f"  {label:<52}{n:>4}{sd_pp:>8.2f}{r2s:>8}{pps:>8}  {note}{tag_s}")

    summary["/".join(key) + ("" if key[1] == "logmel" else f" [{key[1]}]")] = dict(
        n=n, total_sd_pp=sd_pp,
        classes=all_classes,
        comp_shares=[d["comp"][1] for d in draws],
        accs=accs.tolist(),
        tags=dict(collections.Counter(d["tag"] for d in draws)),
        explained_pct=None if r2 is None else expl,
        perm_p=None if perm_p is None else perm_p,
        note=note,
    )

# --------------------------------------------------- what drives it (top) --- #
print()
print("which classes drive the predictable part (univariate |r| with acc, cells with n>=8):")
for key in sorted(cells):
    draws = sorted(cells[key], key=lambda d: d["seed"])
    n = len(draws)
    if n < args.min_draws:
        continue
    if len({tuple(d["comp"][0]) for d in draws}) > 1:
        continue          # mixed label space: index alignment is meaningless
    accs = np.array([d["acc"] for d in draws])
    all_classes = sorted({c for d in draws for c in d["comp"][0]})
    rows = []
    for k, cls in enumerate(all_classes):
        xs = np.array([d["comp"][1][d["comp"][0].index(cls)] if cls in d["comp"][0] else 0.0
                       for d in draws])
        if xs.std() == 0:
            continue
        r = float(np.corrcoef(xs, accs)[0, 1])
        rows.append((abs(r), r, cls))
    rows.sort(reverse=True)
    label = (f"{key[0]}/{key[2]}/{key[3]}/{key[4]}"
             + (f" [{key[1]}]" if key[1] != "logmel" else ""))
    tops = ", ".join(f"{cls}({r:+.2f})" for _, r, cls in rows[:3])
    print(f"  {label:<52} {tops}")

out_json = os.path.join(OUT, "w3e2_variance.json")
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(dict(kind="w3e2-variance-decomposition", n_perm=args.n_perm,
                   summary=summary), fh, indent=1)
tick(f"summary -> {out_json}")
print()
print("next: Output tab -> Create Dataset to keep it, then paste both tables back.")
