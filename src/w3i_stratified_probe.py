# =========================================================================== #
# W3-I stratified-draw validation: does composition-stratified group
# allocation really shrink the draw-to-draw variance of the honest protocol?
#
# Prediction to validate (from the W3-F variance decomposition):
#   vessel_disjoint test accuracy varies across split draws with sd 4.69 pp,
#   85% of which comes from WHICH vessels land in test (class composition,
#   computable from metadata before training).  If we allocate vessels to
#   test/val/train so that each class's CLIP MASS is stratified, the
#   composition term should collapse and the observed draw sd should drop
#   to ~ the init/optimisation floor (~1-2 pp).
#
# Three stratifiers x 6 draws, real training, identical rec2 recipe:
#   clipmass   - per class, hash-ordered vessels accumulate clips up to the
#                class quota (15% test, then 15% val)      <- our fix
#   countstrat - per class, balanced VESSEL COUNTS (the intuitive version;
#                predicted to FAIL because clips/vessel varies 80x)
#   sgkf       - sklearn StratifiedGroupKFold (the standard tool baseline)
#   random     - the 5 rec2 vessel_disjoint runs already on disk (baseline)
#
# PRE-REGISTERED verdict: confirmed if sd(clipmass) <= 0.6 x sd(random).
#
# Runs: 18 x ~14 min ~= 4.2 h GPU (resumable; completed run_ids are skipped).
# The runner (w3_run.py) is copied from the mounted workspace dataset.
# =========================================================================== #
import argparse
import collections
import glob
import gzip
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
RECIPE = ["--epochs", "24", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]
DRAWS = 6
BUDGET_MIN = 420


def tick(msg):
    print(f"  [{time.time() - T0:7.1f}s] {msg}", flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--features-root", default="")
ap.add_argument("--w3run", default="", help="path to w3_run.py (else autodetect)")
ap.add_argument("--work", default="/kaggle/working")
ap.add_argument("--quick", action="store_true",
                help="2 draws x 2 epochs, wiring check only")
ap.add_argument("--rec2", default="",
                help="path to a results.csv holding the rec2 baseline rows "
                     "(else autodetected under /kaggle/input)")
args, _ = ap.parse_known_args()
WORK = args.work
if args.quick:
    DRAWS = 2
    RECIPE = ["--epochs", "2", "--norm", "affine", "--lr", "0.003", "--warmup", "0"]
QUICK = args.quick

# ------------------------------------------------------------- discovery --- #
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
    raise SystemExit("w3_run.py not found under /kaggle/input - attach the "
                     "workspace dataset (it ships the scripts)")
src_txt = open(w3run_src, encoding="utf-8").read()
if "lr_last" not in src_txt:
    raise SystemExit(f"{w3run_src} predates the lr-schedule fix (no lr_last). "
                     f"Update the workspace dataset to the current upload first.")
scripts = os.path.join(WORK, "scripts")
os.makedirs(scripts, exist_ok=True)
RUNNER = os.path.join(scripts, "w3_run.py")
shutil.copy(w3run_src, RUNNER)
sys.path.insert(0, scripts)
import w3_run                                    # noqa: E402  (run_id_for)

# rec2 baseline rows (the 5 random vessel_disjoint draws, already trained)
rec2_csv = None
if args.rec2:
    if not os.path.isfile(args.rec2):
        raise SystemExit(f"--rec2 {args.rec2} not found")
    rec2_csv = args.rec2
else:
    for cand in sorted(glob.glob("/kaggle/input/**/w3-runs/results.csv",
                                 recursive=True)):
        txt = open(cand, encoding="utf-8", errors="replace").read()
        if "vessel_disjoint" in txt and "rec2" in txt:
            rec2_csv = cand
            break
tick(f"features: {feat}")
tick(f"runner  : {w3run_src}")
tick(f"rec2    : {rec2_csv}")

# ------------------------------------------------- load index / inventory -- #
import csv

with open(os.path.join(feat, "oceanship", "index.csv"), encoding="utf-8",
          newline="") as fh:
    idx_rows = list(csv.DictReader(fh))
mm_of = {r["clip_id"]: r["mmsi"] for r in idx_rows}
label_of_mmsi, label_of_clip = {}, {}
ves_clips = {}
for r in idx_rows:
    m, c, lb = r["mmsi"], r["clip_id"], r["label"]
    label_of_mmsi[m] = lb
    label_of_clip[c] = lb
    ves_clips.setdefault(lb, {}).setdefault(m, 0)
    ves_clips[lb][m] += 1
CLASSES = sorted(ves_clips)
tick(f"index {len(idx_rows):,} clips, {len(CLASSES)} classes, "
     f"{len(label_of_mmsi):,} vessels")

# The rec2 baseline is a 12-class benchmark: the frozen split file simply does
# not contain the 3 rare classes (Diving ship / Military ship / Spare), and the
# runner drops clips that no split row covers.  The stratified draws MUST live
# in the same universe - taking all 15 classes here would change the label
# space (K=15 vs 12) and make every comparison against random(rec2) invalid.
# Source of truth: the classes actually covered by the frozen splits.
def _bench_universe():
    for name in ("oceanship_splits.csv.gz", "oceanship_splits.csv"):
        p = os.path.join(feat, "splits", name)
        if os.path.exists(p):
            opener = gzip.open if name.endswith(".gz") else open
            with opener(p, "rt", newline="", encoding="utf-8") as fh:
                cls = {label_of_clip[r["clip_id"]] for r in csv.DictReader(fh)
                       if r["clip_id"] in label_of_clip}
            if cls:
                return cls
    raise SystemExit("cannot derive the 12-class benchmark universe from the "
                     "frozen splits; mount the feature dataset with splits/")


BENCH = sorted(_bench_universe())
BENCH_SET = set(BENCH)
VES_BENCH = {c: ves_clips[c] for c in BENCH}
CLIPS_BENCH = {cid for cid, lb in label_of_clip.items() if lb in ves_clips
               and lb in VES_BENCH}
if len(BENCH) != len(CLASSES):
    tick(f"benchmark universe: {len(BENCH)} classes "
         f"(dropped: {sorted(set(CLASSES) - set(BENCH))})")


def hash_order(items, seed, salt):
    return sorted(items, key=lambda m: hashlib.sha256(
        f"{seed}|{salt}|{m}".encode()).digest())


ALL_CLIPS = [r["clip_id"] for r in idx_rows]


def _expand(vessel_assign):
    """Vessel-level assignment -> clip-level (the splits file is clip-keyed).

    Two exclusion rules, both matching how the frozen rec2 splits define the
    12-class universe:  (1) clips whose LABEL is outside BENCH get no row -
    necessary because a few real vessels carry both a benchmark label and a
    rare label (5 of 892 MMSI), and the rare-label clips must not ride into
    the test set on their vessel's assignment;  (2) vessels outside the
    benchmark classes (Diving ship / Military ship / Spare) get no row.
    """
    return {cid: vessel_assign[mm_of[cid]] for cid in ALL_CLIPS
            if label_of_clip[cid] in BENCH_SET and mm_of[cid] in vessel_assign}


def draw_clipmass(seed):
    """Per class: hash-ordered vessels accumulate CLIPS to the class quota
    (15% test, then 15% val); remainder train."""
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
    return _expand(a)


def draw_countstrat(seed):
    """Per class: balanced VESSEL COUNTS (the intuitive stratification that
    the clip-mass analysis predicts will FAIL - clips/vessel varies 80x)."""
    a = {}
    for c in BENCH:
        vs = hash_order(list(VES_BENCH[c]), seed, c)
        k = max(1, round(0.15 * len(vs)))
        for m in vs[:k]:
            a[m] = "test"
        for m in vs[k:2 * k]:
            a[m] = "val"
    for c in BENCH:
        for m in VES_BENCH[c]:
            a.setdefault(m, "train")
    return _expand(a)


def draw_sgkf(seed):
    """sklearn StratifiedGroupKFold on clips (groups = vessels), 7 folds:
    fold 0 = test, fold 1 = val, rest = train.  The standard-tool baseline."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    clip_ids = sorted(CLIPS_BENCH)
    y = np.array([BENCH.index(label_of_clip[c]) for c in clip_ids])
    g = np.array([hash(mm_of[c]) % (2 ** 31) for c in clip_ids])
    a = {}
    for f, (tr, te) in enumerate(StratifiedGroupKFold(
            n_splits=7, shuffle=True, random_state=seed).split(
            np.zeros(len(clip_ids)), y, g)):
        name = "test" if f == 0 else ("val" if f == 1 else "train")
        for i in te:
            a[clip_ids[i]] = name
    return a


GENERATORS = {"clipmass": draw_clipmass, "countstrat": draw_countstrat,
              "sgkf": draw_sgkf}

# -------------------------------------- write one fake feature root each --- #
import hashlib

real_logmel = os.path.abspath(os.path.join(feat, "oceanship", "logmel.npy"))
real_index = os.path.abspath(os.path.join(feat, "oceanship", "index.csv"))
FAKE = {}
for name, gen in GENERATORS.items():
    root = os.path.join(WORK, f"feat_{name}")
    os.makedirs(os.path.join(root, "splits"), exist_ok=True)
    dst_log = os.path.join(root, "oceanship", "logmel.npy")
    os.makedirs(os.path.join(root, "oceanship"), exist_ok=True)
    for real, dst in ((real_logmel, dst_log),
                      (real_index, os.path.join(root, "oceanship", "index.csv"))):
        if os.path.exists(dst):
            continue
        try:
            os.symlink(real, dst)
        except OSError:                       # Windows / no-privilege fallback
            shutil.copy(real, dst)
    rows = []
    for s in range(DRAWS):
        a = gen(s)
        bad = {label_of_clip[cid] for cid in a} - set(BENCH)
        if bad:
            raise SystemExit(f"{name} s{s}: classes outside the benchmark "
                             f"universe leaked into the split: {sorted(bad)}")
        for cid, spl in a.items():
            rows.append({"clip_id": cid, "protocol": "vessel_disjoint",
                         "seed": s, "split": spl})
    sp = os.path.join(root, "splits", "oceanship_splits.csv.gz")
    with gzip.open(sp, "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "protocol", "seed", "split"])
        w.writeheader()
        w.writerows(rows)
    FAKE[name] = root
    tick(f"{name:<10} {DRAWS} draws written ({len(rows):,} rows) -> {root}")

# --------------------------------------------------------- sweep driver --- #
RUNS = os.path.join(WORK, "w3i-runs")
os.makedirs(RUNS, exist_ok=True)


def done_ids():
    ids = set()
    for f in glob.glob(os.path.join(RUNS, "results.csv")) + \
             ([rec2_csv] if rec2_csv else []):
        try:
            with open(f, encoding="utf-8", newline="") as fh:
                ids |= {r["run_id"] for r in csv.DictReader(fh)}
        except Exception as exc:
            print(f"    [resume] {f} unreadable ({exc})")
    return ids


def one_run(strat, seed, smoke=False):
    cmd = [sys.executable, RUNNER,
           "--data", FAKE[strat], "--out",
           (os.path.join(WORK, "_probe") if smoke else RUNS),
           "--corpus", "oceanship", "--protocol", "vessel_disjoint",
           "--seed", str(seed), "--rep", "logmel", "--backbone", "cnn_small",
           "--num-workers", "2", "--tag",
           ("stratprobe" if smoke else f"{strat}")] + RECIPE
    if smoke:
        cmd.append("--smoke")
    r = subprocess.run(cmd)
    return r.returncode


# wiring check first - one minute, before committing four hours
if one_run("clipmass", 0, smoke=True) != 0:
    raise SystemExit("smoke run failed - fix the wiring before the sweep")
tick("smoke run OK")

if not QUICK:
    t0 = time.time()
    ran = skipped = 0
    have = done_ids()
    for strat in GENERATORS:
        for s in range(DRAWS):
            rid = w3_run.run_id_for("oceanship", "logmel", "cnn_small",
                                    "vessel_disjoint", s,
                                    "none", "erm", False, strat)
            if rid in have:
                skipped += 1
                continue
            el = (time.time() - t0) / 60
            if el > BUDGET_MIN:
                print(f"[stop] budget {BUDGET_MIN} min reached; rerun to resume")
                break
            tick(f"[{ran + skipped + 1}] {rid}")
            one_run(strat, s)
            ran += 1
    tick(f"sweep: {ran} runs, {skipped} skipped")
else:
    tick("QUICK mode: running 2 tiny draws per stratifier for the wiring check")
    for strat in GENERATORS:
        for s in range(DRAWS):
            if one_run(strat, s) != 0:
                print(f"  [warn] {strat} s{s} failed - see its log above")

# -------------------------------------------------------------- analysis --- #
rows = []
for f in glob.glob(os.path.join(RUNS, "results.csv")) + \
        ([rec2_csv] if rec2_csv else []):
    with open(f, encoding="utf-8", newline="") as fh:
        rows += [r for r in csv.DictReader(fh)]

# pooled per-class recall of the trained model, from the rec2 random draws
import statistics as st

pooled_num, pooled_den = {}, {}
for r in rows:
    if r["protocol"] != "vessel_disjoint" or "rec2" not in r["run_id"]:
        continue
    for k, v in r.items():
        if k.startswith("acc_") and ("n_" + k[4:]) in r:
            pooled_num[k[4:]] = pooled_num.get(k[4:], 0.0) + float(v) * int(r["n_" + k[4:]])
            pooled_den[k[4:]] = pooled_den.get(k[4:], 0) + int(r["n_" + k[4:]])
RECALL = {c: pooled_num[c] / pooled_den[c] for c in pooled_den if pooled_den[c]}


def pred_acc(assign):
    by = collections.Counter()
    for cid, spl in assign.items():
        if spl == "test":
            by[label_of_clip[cid]] += 1
    tot = sum(by.values())
    return sum(by[c] * RECALL.get(c, 0.0) for c in by) / max(tot, 1)


groups = {}
if rec2_csv:
    rnd = [float(r["test_acc"]) for r in rows
           if r["protocol"] == "vessel_disjoint" and "rec2" in r["run_id"]
           and r.get("rep") == "logmel" and "cnn_small" in r["run_id"]]
    if rnd:
        groups["random(rec2)"] = rnd
for strat in GENERATORS:
    sel = [r for r in rows if f"__{strat}" in r["run_id"]]
    accs = [float(r["test_acc"]) for r in sel]
    if accs:
        ks = {sum(1 for k in r if k.startswith("acc_")) for r in sel}
        if ks != {len(BENCH)}:
            print(f"  [warn] {strat}: label-space sizes {sorted(ks)} != "
                  f"{len(BENCH)} - stale results.csv from a pre-fix run?")
        groups[strat] = accs

print()
print("=" * 88)
print("draw-to-draw spread of vessel_disjoint test accuracy, by allocation scheme")
print("=" * 88)
print(f"  {'allocation':<16}{'n':>3}{'mean':>9}{'sd':>8}{'min':>8}{'max':>8}"
      f"{'pred.sd':>9}")
print("  " + "-" * 64)
summary = {}
for name, accs in groups.items():
    sd = 100 * st.stdev(accs) if len(accs) > 1 else 0.0
    pred = ""
    if name in GENERATORS and RECALL:
        ps = 100 * st.pstdev([pred_acc(GENERATORS[name](s)) for s in range(DRAWS)])
        pred = f"{ps:>8.2f}"
    summary[name] = dict(n=len(accs), mean=float(st.mean(accs)),
                         sd=sd, min=min(accs), max=max(accs))
    print(f"  {name:<16}{len(accs):>3}{st.mean(accs):>9.4f}{sd:>7.2f}p"
          f"{min(accs):>8.4f}{max(accs):>8.4f}{pred:>9}")
print()
if "random(rec2)" in groups and "clipmass" in groups:
    sr = summary["random(rec2)"]["sd"]
    sc = summary["clipmass"]["sd"]
    print(f"  sd ratio clipmass/random = {sc / max(sr, 1e-9):.2f}   "
          f"(prediction: <= 0.6;  metadata-only estimate was 1.74/3.86 = 0.45)")
    print(f"  draws needed for a 2 pp claim: "
          f"{2 * 7.849 * sr * sr / 4:.0f} (random) -> "
          f"{2 * 7.849 * sc * sc / 4:.0f} (clipmass)")
    if sc <= 0.6 * sr:
        print("  -> CONFIRMED: composition stratification collapses the draw")
        print("     variance on the real model, close to the init noise floor.")
        print("     The fix works; the paper's remedy section stands.")
    else:
        print("  -> NOT CONFIRMED: the real-model sd did not shrink as the")
        print("     composition model predicted. Investigate before writing")
        print("     (candidates: init noise larger than measured, or the")
        print("     per-class recall table not transferable to new draws).")

for cand in (WORK, os.getcwd()):
    if cand and os.path.isdir(cand):
        with open(os.path.join(cand, "w3i_stratified.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(dict(kind="w3i-stratified", groups=summary,
                           recall=RECALL, recipe=RECIPE, draws=DRAWS),
                      fh, indent=2, default=float)
        print()
        tick(f"summary -> {os.path.join(cand, 'w3i_stratified.json')}")
        break
