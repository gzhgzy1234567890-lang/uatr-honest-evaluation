# ============================================================================================== #
# W3-K — does the clip-mass remedy generalize across MODEL FAMILIES?                             #
#                                                                                                #
# W3-I showed on cnn_small: clip-mass stratified allocation collapses the draw-to-draw sd of    #
# honest vessel_disjoint evaluation from 4.69 pp to 1.24 pp (sd ratio 0.26), and the composition #
# model predicted all three schemes' sd before training.  A reviewer's next question: "does     #
# this hold for other models, or is it an artifact of one tiny CNN?"                             #
#                                                                                                #
# This script repeats the W3-I comparison for two other model families at the extremes of the    #
# capacity ladder:                                                                               #
#   logreg    - linear probe (no learned representation)                                         #
#   cnn_wide  - a wider from-scratch CNN (~10x cnn_small params)                                 #
# for each:  random draws (frozen vessel_disjoint seeds) vs clip-mass stratified draws, same     #
# rec2 recipe.  Pre-registered verdict, PER BACKBONE:                                            #
#   sd(clipmass) <= 0.6 x sd(random)  ->  the remedy generalizes to this model family.           #
#                                                                                                #
# pred.sd is computed per backbone from reference per-class recalls pooled over THAT backbone's  #
# random draws (in-sample for the recalls, out-of-sample for the clipmass draws being predicted) #
# - so "prediction" here tests the composition model, not unseen classes.                        #
#                                                                                                #
# Budget: cnn_wide dominates (~20-25 min/run) -> 5 random + 6 clipmass draws ~ 4-5 h on P100.    #
# logreg adds ~1 h.  Session-budget guard skips runs that would not finish; rerun resumes.       #
#                                                                                                #
# Attach: under-water-features + workspace (ships w3_run.py).  W3I_data not required.            #
# Smoke:  --quick (2 epochs, 1 draw per scheme per backbone, minutes).                           #
# ============================================================================================== #

import argparse
import collections
import csv
import glob
import gzip
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

T0 = time.time()
RECIPE = ["--epochs", "24", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]


def tick(msg):
    print(f"  [{time.time() - T0:7.1f}s] {msg}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--backbones", default="logreg,cnn_wide")
ap.add_argument("--random-seeds", default="0,1,2,3,4",
                help="frozen vessel_disjoint seeds for the random baseline")
ap.add_argument("--clipmass-draws", type=int, default=6)
ap.add_argument("--budget-min", type=float, default=330.0,
                help="stop starting new runs after this many minutes")
ap.add_argument("--features-root", default="")
ap.add_argument("--w3run", default="", help="path to w3_run.py (else autodetect)")
ap.add_argument("--work", default="/kaggle/working")
ap.add_argument("--quick", action="store_true",
                help="2 epochs, 1 draw per scheme per backbone - wiring check only")
args, _ = ap.parse_known_args()

BACKBONES = [b.strip() for b in args.backbones.split(",")]
RAND_SEEDS = [int(s) for s in args.random_seeds.split(",")]
CM_DRAWS = 1 if args.quick else args.clipmass_draws
if args.quick:
    RAND_SEEDS = RAND_SEEDS[:1]
BUDGET_MIN = args.budget_min
WORK = args.work
if args.quick:
    RECIPE = ["--epochs", "2", "--norm", "affine", "--lr", "0.003", "--warmup", "0"]

# ------------------------------------------------------------- discovery ---- #
feat = args.features_root or None
if feat is None:
    for cand in sorted(glob.glob("/kaggle/input/**/uwatr-features", recursive=True)):
        if os.path.isfile(os.path.join(cand, "oceanship", "index.csv")):
            feat = cand
            break
if feat is None:
    raise SystemExit("uwatr-features dataset not found - attach it and rerun")

w3run_src = args.w3run or None
if w3run_src is None:
    for cand in sorted(glob.glob("/kaggle/input/**/w3_run.py", recursive=True)):
        if "w3-runs" not in cand:
            w3run_src = cand
            break
if w3run_src is None:
    raise SystemExit("w3_run.py not found under /kaggle/input - attach workspace")
src_txt = open(w3run_src, encoding="utf-8").read()
if "lr_last" not in src_txt:
    raise SystemExit(f"{w3run_src} predates the lr-schedule fix (no lr_last)")
scripts = os.path.join(WORK, "scripts")
os.makedirs(scripts, exist_ok=True)
RUNNER = os.path.join(scripts, "w3_run.py")
shutil.copy(w3run_src, RUNNER)
sys.path.insert(0, scripts)
import w3_run                                     # noqa: E402  (run_id_for)

# ------------------------------------------------- index + bench universe --- #
with open(os.path.join(feat, "oceanship", "index.csv"), encoding="utf-8",
          newline="") as fh:
    idx_rows = list(csv.DictReader(fh))
mm_of = {r["clip_id"]: r["mmsi"] for r in idx_rows}
label_of_clip = {r["clip_id"]: r["label"] for r in idx_rows}
ves_clips = collections.defaultdict(dict)
for r in idx_rows:
    ves_clips[r["label"]][r["mmsi"]] = ves_clips[r["label"]].get(r["mmsi"], 0) + 1


def bench_universe():
    for name in ("oceanship_splits.csv.gz", "oceanship_splits.csv"):
        p = os.path.join(feat, "splits", name)
        if os.path.exists(p):
            op = gzip.open if name.endswith(".gz") else open
            with op(p, "rt", newline="", encoding="utf-8") as fh:
                cls = {label_of_clip[r["clip_id"]] for r in csv.DictReader(fh)
                       if r["clip_id"] in label_of_clip}
            if cls:
                return cls
    raise SystemExit("cannot derive the 12-class universe from the frozen splits")


BENCH = sorted(bench_universe())
BENCH_SET = set(BENCH)
VES_BENCH = {c: ves_clips[c] for c in BENCH}
tick(f"index {len(idx_rows):,} clips; benchmark universe {len(BENCH)} classes")

ALL_CLIPS = [r["clip_id"] for r in idx_rows]


def hash_order(items, seed, salt):
    return sorted(items, key=lambda m: hashlib.sha256(
        f"{seed}|{salt}|{m}".encode()).digest())


def draw_clipmass(seed):
    """Per benchmark class: hash-ordered vessels accumulate CLIPS to the class
    quota (15% test, then 15% val); remainder train.  Same as W3-I."""
    a = {}
    used = set()
    for share, name in ((0.15, "test"), (0.15, "val")):
        for c in BENCH:
            vs = VES_BENCH[c]
            remaining = [m for m in vs if m not in used]
            total = sum(vs[m] for m in remaining)
            quota = share * total
            acc = 0
            for m in hash_order(remaining, seed, name + c):
                if acc >= quota:
                    break
                a[m] = name
                used.add(m)
                acc += vs[m]
    for c in BENCH:
        for m in VES_BENCH[c]:
            a.setdefault(m, "train")
    # expand, excluding non-benchmark labels riding on shared vessels
    return {cid: a[mm_of[cid]] for cid in ALL_CLIPS
            if label_of_clip[cid] in BENCH_SET and mm_of[cid] in a}


# ------------------------------------------------ fake root for clipmass ---- #
cm_root = os.path.join(WORK, "feat_clipmass")
os.makedirs(os.path.join(cm_root, "splits"), exist_ok=True)
os.makedirs(os.path.join(cm_root, "oceanship"), exist_ok=True)
for fn in ("logmel.npy", "index.csv"):
    real = os.path.abspath(os.path.join(feat, "oceanship", fn))
    dst = os.path.join(cm_root, "oceanship", fn)
    if not os.path.exists(dst):
        try:
            os.symlink(real, dst)
        except OSError:
            shutil.copy(real, dst)
rows = []
for s in range(CM_DRAWS):
    a = draw_clipmass(s)
    bad = {label_of_clip[cid] for cid in a} - BENCH_SET
    if bad:
        raise SystemExit(f"clipmass s{s}: non-bench classes leaked: {sorted(bad)}")
    for cid, spl in a.items():
        rows.append({"clip_id": cid, "protocol": "vessel_disjoint",
                     "seed": s, "split": spl})
with gzip.open(os.path.join(cm_root, "splits", "oceanship_splits.csv.gz"),
               "wt", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["clip_id", "protocol", "seed", "split"])
    w.writeheader()
    w.writerows(rows)
tick(f"clipmass splits written: {CM_DRAWS} draws, {len(rows):,} rows")

# ------------------------------------------------------------- sweep -------- #
RUNS = os.path.join(WORK, "w3k-runs")
os.makedirs(RUNS, exist_ok=True)


def done_ids():
    ids = set()
    p = os.path.join(RUNS, "results.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8", newline="") as fh:
            ids |= {r["run_id"] for r in csv.DictReader(fh)}
    return ids


def one_run(root, backbone, seed, tag):
    cmd = [sys.executable, RUNNER,
           "--data", root, "--out", RUNS,
           "--corpus", "oceanship", "--protocol", "vessel_disjoint",
           "--seed", str(seed), "--rep", "logmel", "--backbone", backbone,
           "--num-workers", "2", "--tag", tag] + RECIPE
    return subprocess.run(cmd).returncode


plan = []
for bb in BACKBONES:
    for s in RAND_SEEDS:
        plan.append((feat, bb, s, "genrand"))
    for s in range(CM_DRAWS):
        plan.append((cm_root, bb, s, "clipmass"))

have = done_ids()
ran = skipped = 0
for root, bb, s, tag in plan:
    rid = w3_run.run_id_for("oceanship", "logmel", bb, "vessel_disjoint", s,
                            "none", "erm", False, tag)
    if rid in have:
        skipped += 1
        continue
    el = (time.time() - T0) / 60
    if el > BUDGET_MIN:
        print(f"[stop] budget {BUDGET_MIN} min reached; rerun to resume")
        break
    tick(f"[{ran + skipped + 1}/{len(plan)}] {rid}")
    if one_run(root, bb, s, tag) != 0:
        print(f"  [warn] {rid} failed - see its log above")
    ran += 1
tick(f"sweep: {ran} runs, {skipped} skipped")

# ------------------------------------------------------------ analysis ------ #
rows = []
p = os.path.join(RUNS, "results.csv")
if os.path.exists(p):
    with open(p, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

groups = collections.defaultdict(lambda: collections.defaultdict(list))
percls_ref = {}
for r in rows:
    rid = r["run_id"]
    if "vessel_disjoint" not in rid:
        continue
    bb = r.get("backbone", "")
    if bb not in BACKBONES:
        continue
    scheme = "clipmass" if rid.endswith("__clipmass") else \
        ("genrand" if rid.endswith("__genrand") else None)
    if scheme is None:
        continue
    groups[bb][scheme].append(float(r["test_acc"]))
    if scheme == "genrand":
        d = percls_ref.setdefault(bb, {"num": collections.Counter(),
                                       "den": collections.Counter()})
        for k, v in r.items():
            if k.startswith("acc_") and ("n_" + k[4:]) in r:
                if not v:                 # class absent from this test set (null)
                    continue
                d["num"][k[4:]] += float(v) * int(r["n_" + k[4:]])
                d["den"][k[4:]] += int(r["n_" + k[4:]])

print()
print("=" * 88)
print("does the clip-mass remedy generalize across model families?")
print("=" * 88)
print(f"  {'backbone':<12}{'scheme':<10}{'n':>3}{'mean':>9}{'sd':>8}"
      f"{'min':>8}{'max':>8}{'pred.sd':>9}")
print("  " + "-" * 70)
summary = {}
for bb in BACKBONES:
    ref = {}
    if bb in percls_ref:
        d = percls_ref[bb]
        ref = {c: d["num"][c] / d["den"][c] for c in d["den"] if d["den"][c]}
    for scheme in ("genrand", "clipmass"):
        accs = groups[bb].get(scheme, [])
        if not accs:
            continue
        sd = 100 * statistics.stdev(accs) if len(accs) > 1 else 0.0
        pred = ""
        if scheme == "clipmass" and ref:
            masses = []
            for s in range(CM_DRAWS):
                by = collections.Counter()
                a = draw_clipmass(s)
                for cid in a:
                    if a[cid] == "test":
                        by[label_of_clip[cid]] += 1
                tot = sum(by.values())
                masses.append(sum(by[c] * ref.get(c, 0.0) for c in by)
                              / max(tot, 1))
            pred = f"{100 * statistics.pstdev(masses):>8.2f}"
        summary.setdefault(bb, {})[scheme] = dict(
            n=len(accs), mean=statistics.mean(accs), sd=sd,
            min=min(accs), max=max(accs))
        print(f"  {bb:<12}{scheme:<10}{len(accs):>3}{statistics.mean(accs):>9.4f}"
              f"{sd:>7.2f}p{min(accs):>8.4f}{max(accs):>8.4f}{pred:>9}")
    print()

print("  pre-registered verdict per backbone (sd ratio clipmass/genrand <= 0.6):")
for bb in BACKBONES:
    g = summary.get(bb, {})
    if "genrand" in g and "clipmass" in g and g["genrand"]["n"] > 1:
        ratio = g["clipmass"]["sd"] / max(g["genrand"]["sd"], 1e-9)
        ok = ratio <= 0.6
        print(f"    {bb:<12} sd ratio {ratio:.2f}  ->  "
              f"{'GENERALIZES' if ok else 'does NOT generalize'}")

for cand in (WORK, os.getcwd()):
    if cand and os.path.isdir(cand):
        with open(os.path.join(cand, "w3k_gencheck.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(dict(kind="w3k-gencheck", backbones=BACKBONES,
                           summary=summary, draws=dict(rand=RAND_SEEDS,
                                                       clipmass=CM_DRAWS),
                           recipe=RECIPE), fh, indent=2, default=float)
print()
tick("summary -> w3k_gencheck.json")
print("  -> Output tab -> Create Dataset to keep the results")
