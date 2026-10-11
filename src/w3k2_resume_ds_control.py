# ============================================================================================== #
# W3-K2 — resume W3-K (cnn_wide clipmass seeds 3-5) + DeepShip no-op control                      #
#                                                                                                #
# Part 1 (resume): the W3-K sweep was cut by the session budget after 19 of 22 runs; cnn_wide    #
# clipmass has only seeds 0-2.  Run-ids are deduplicated against the mounted W3K_data dataset,   #
# so rerunning this script executes ONLY the 3 missing runs (~1 h).                              #
#                                                                                                #
# Part 2 (control): DeepShip rec_random as a pre-registered NO-OP control.  DeepShip has 4       #
# balanced classes and ~600 recordings, so a random draw's test composition barely moves - the   #
# composition model PREDICTS little draw variance and therefore no benefit from stratification.  #
# Confirming that turns the remedy from "we tried it on one corpus" into "the recipe acts        #
# exactly where, and only where, the composition model says there is something to fix."          #
#                                                                                                #
# Pre-registered verdict (Part 2):                                                               #
#   no-op PREDICTED  : pred.sd(genrand) < 1.5 pp                                                 #
#   no-op OBSERVED   : sd(clipmass) within [0.6, 1.67] x sd(genrand)                             #
#   both hold        -> control confirms the composition model in the null direction.            #
#   If pred.sd(genrand) is large instead, the control prediction failed - report honestly.       #
#                                                                                                #
# Attach: under-water-features + workspace + W3K_data.  ~3 h total (1 h resume + 2 h control).   #
# Smoke:  --quick --parts control (minutes, on the fixture).                                     #
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
ap.add_argument("--parts", default="both", choices=["resume", "control", "both"])
ap.add_argument("--random-seeds", default="0,1,2,3,4")
ap.add_argument("--clipmass-draws", type=int, default=6)
ap.add_argument("--budget-min", type=float, default=300.0)
ap.add_argument("--features-root", default="")
ap.add_argument("--w3run", default="")
ap.add_argument("--work", default="/kaggle/working")
ap.add_argument("--quick", action="store_true")
args, _ = ap.parse_known_args()

RAND_SEEDS = [int(s) for s in args.random_seeds.split(",")]
CM_DRAWS = 1 if args.quick else args.clipmass_draws
if args.quick:
    RAND_SEEDS = RAND_SEEDS[:1]
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
    raise SystemExit("w3_run.py not found - attach the workspace dataset")
if "lr_last" not in open(w3run_src, encoding="utf-8").read():
    raise SystemExit(f"{w3run_src} predates the lr-schedule fix")
scripts = os.path.join(WORK, "scripts")
os.makedirs(scripts, exist_ok=True)
RUNNER = os.path.join(scripts, "w3_run.py")
shutil.copy(w3run_src, RUNNER)
sys.path.insert(0, scripts)
import w3_run                                     # noqa: E402

RUNS = os.path.join(WORK, "w3k2-runs")
os.makedirs(RUNS, exist_ok=True)


def done_ids():
    ids = set()
    cands = [os.path.join(RUNS, "results.csv")] + \
        glob.glob("/kaggle/input/**/w3k-runs/results.csv", recursive=True)
    for p in cands:
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8", newline="") as fh:
                    ids |= {r["run_id"] for r in csv.DictReader(fh)}
            except Exception as exc:
                print(f"    [resume] {p} unreadable ({exc})")
    return ids


def one_run(root, corpus, backbone, seed, tag):
    cmd = [sys.executable, RUNNER, "--data", root, "--out", RUNS,
           "--corpus", corpus, "--protocol", "vessel_disjoint" if corpus == "oceanship"
           else "rec_random",
           "--seed", str(seed), "--rep", "logmel", "--backbone", backbone,
           "--num-workers", "2", "--tag", tag] + RECIPE
    return subprocess.run(cmd).returncode


def hash_order(items, seed, salt):
    return sorted(items, key=lambda m: hashlib.sha256(
        f"{seed}|{salt}|{m}".encode()).digest())


# ============================ PART 1: resume W3-K =========================== #
if args.parts in ("resume", "both"):
    # oceanship bench universe + clip-mass generator (identical to W3-K/W3-I)
    with open(os.path.join(feat, "oceanship", "index.csv"), encoding="utf-8",
              newline="") as fh:
        oidx = list(csv.DictReader(fh))
    o_mm = {r["clip_id"]: r["mmsi"] for r in oidx}
    o_lab = {r["clip_id"]: r["label"] for r in oidx}
    o_ves = collections.defaultdict(dict)
    for r in oidx:
        o_ves[r["label"]][r["mmsi"]] = o_ves[r["label"]].get(r["mmsi"], 0) + 1

    def bench_universe():
        for name in ("oceanship_splits.csv.gz", "oceanship_splits.csv"):
            p = os.path.join(feat, "splits", name)
            if os.path.exists(p):
                op = gzip.open if name.endswith(".gz") else open
                with op(p, "rt", newline="", encoding="utf-8") as fh:
                    cls = {o_lab[r["clip_id"]] for r in csv.DictReader(fh)
                           if r["clip_id"] in o_lab}
                if cls:
                    return cls
        raise SystemExit("cannot derive the oceanship 12-class universe")

    BENCH = sorted(bench_universe())
    BENCH_SET = set(BENCH)
    VES_BENCH = {c: o_ves[c] for c in BENCH}

    def draw_clipmass_oc(seed):
        a, used = {}, set()
        for share, name in ((0.15, "test"), (0.15, "val")):
            for c in BENCH:
                vs = VES_BENCH[c]
                remaining = [m for m in vs if m not in used]
                total = sum(vs[m] for m in remaining)
                quota, acc = share * total, 0
                for m in hash_order(remaining, seed, name + c):
                    if acc >= quota:
                        break
                    a[m] = name
                    used.add(m)
                    acc += vs[m]
        for c in BENCH:
            for m in VES_BENCH[c]:
                a.setdefault(m, "train")
        return {cid: a[o_mm[cid]] for cid, _ in ((r["clip_id"], 0) for r in oidx)
                if o_mm[cid] in a and o_lab[cid] in BENCH_SET}

    cm_root = os.path.join(WORK, "feat_clipmass_oc")
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
        a = draw_clipmass_oc(s)
        bad = {o_lab[cid] for cid in a} - BENCH_SET
        if bad:
            raise SystemExit(f"clipmass s{s}: non-bench leak {sorted(bad)}")
        for cid, spl in a.items():
            rows.append({"clip_id": cid, "protocol": "vessel_disjoint",
                         "seed": s, "split": spl})
    with gzip.open(os.path.join(cm_root, "splits", "oceanship_splits.csv.gz"),
                   "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["clip_id", "protocol", "seed", "split"])
        w.writeheader()
        w.writerows(rows)
    tick(f"[resume] oceanship clipmass splits ready ({CM_DRAWS} draws)")

    have = done_ids()
    ran = skipped = 0
    for bb in ("logreg", "cnn_wide"):
        for s in RAND_SEEDS:
            plan = (feat, bb, s, "genrand")
            rid = w3_run.run_id_for("oceanship", "logmel", bb, "vessel_disjoint",
                                    s, "none", "erm", False, "genrand")
            if rid in have:
                continue
            if (time.time() - T0) / 60 > args.budget_min:
                print("[stop] budget reached mid-resume; rerun to resume")
                break
            tick(f"[resume] {rid}")
            one_run(plan[0], "oceanship", bb, s, "genrand")
            ran += 1
        for s in range(CM_DRAWS):
            rid = w3_run.run_id_for("oceanship", "logmel", bb, "vessel_disjoint",
                                    s, "none", "erm", False, "clipmass")
            if rid in have:
                continue
            if (time.time() - T0) / 60 > args.budget_min:
                print("[stop] budget reached mid-resume; rerun to resume")
                break
            tick(f"[resume] {rid}")
            one_run(cm_root, "oceanship", bb, s, "clipmass")
            ran += 1
    tick(f"[resume] {ran} runs executed, rest dedup-skipped (W3K_data + this session)")

# ============================ PART 2: DeepShip control ====================== #
summary = {"kind": "w3k2-deepship-control", "schemes": {}}
if args.parts in ("control", "both"):
    with open(os.path.join(feat, "deepship", "index.csv"), encoding="utf-8",
              newline="") as fh:
        didx = list(csv.DictReader(fh))
    d_lab = {r["clip_id"]: r["class_dir"] for r in didx}
    rec_clips = collections.defaultdict(int)
    rec_cls = {}
    for r in didx:
        rec_clips[r["rec_key"]] += 1
        rec_cls[r["rec_key"]] = r["class_dir"]
    d_cls = sorted({r["class_dir"] for r in didx})
    tick(f"[control] deepship index {len(didx):,} clips, {len(rec_clips):,} "
         f"recordings, {len(d_cls)} classes")

    # frozen rec_random seeds actually available
    sp_path = None
    for name in ("deepship_splits.csv.gz", "deepship_splits.csv",
                 "splits.csv.gz"):
        p = os.path.join(feat, "splits", name)
        if os.path.exists(p):
            sp_path = p
            break
    if sp_path is None:
        raise SystemExit("no deepship splits in the feature dataset")
    op = gzip.open if sp_path.endswith(".gz") else open
    avail = set()
    with op(sp_path, "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["protocol"] == "rec_random":
                avail.add(int(r["seed"]))
    rand_seeds = [s for s in RAND_SEEDS if s in avail]
    tick(f"[control] frozen rec_random seeds available: {sorted(avail)} "
         f"-> using {rand_seeds}")

    # stratified draws: recording-level, per-class clip-mass quotas
    def draw_clipmass_ds(seed):
        a = {}
        used = set()
        for share, name in ((0.15, "test"), (0.15, "val")):
            for c in d_cls:
                recs = [k for k, v in rec_cls.items()
                        if v == c and k not in used]
                total = sum(rec_clips[k] for k in recs)
                quota, acc = share * total, 0
                for k in hash_order(recs, seed, name + c):
                    if acc >= quota:
                        break
                    a[k] = name
                    used.add(k)
                    acc += rec_clips[k]
        for k in rec_cls:
            a.setdefault(k, "train")
        return a

    ds_root = os.path.join(WORK, "feat_clipmass_ds")
    os.makedirs(os.path.join(ds_root, "splits"), exist_ok=True)
    os.makedirs(os.path.join(ds_root, "deepship"), exist_ok=True)
    # w3_run's resolve_features_root refuses a root without <root>/oceanship/,
    # so the deepship-only fake root also carries an oceanship symlink (unused)
    for corp in ("deepship", "oceanship"):
        os.makedirs(os.path.join(ds_root, corp), exist_ok=True)
        for fn in ("logmel.npy", "index.csv"):
            real = os.path.abspath(os.path.join(feat, corp, fn))
            dst = os.path.join(ds_root, corp, fn)
            if not os.path.exists(dst):
                try:
                    os.symlink(real, dst)
                except OSError:
                    shutil.copy(real, dst)
    rows = []
    for s in range(CM_DRAWS):
        a = draw_clipmass_ds(s)
        for r in didx:
            rows.append({"protocol": "rec_random", "seed": s,
                         "rec_key": r["rec_key"], "clip_idx": r["clip_idx"],
                         "split": a[r["rec_key"]]})
    with gzip.open(os.path.join(ds_root, "splits", "deepship_splits.csv.gz"),
                   "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["protocol", "seed", "rec_key",
                                           "clip_idx", "split"])
        w.writeheader()
        w.writerows(rows)
    tick(f"[control] stratified deepship splits written ({CM_DRAWS} draws, "
         f"{len(rows):,} rows)")

    have = done_ids()
    ran = skipped = 0
    for s in rand_seeds:
        rid = w3_run.run_id_for("deepship", "logmel", "cnn_small", "rec_random",
                                s, "none", "erm", False, "genrand")
        if rid in have:
            continue
        if (time.time() - T0) / 60 > args.budget_min:
            print("[stop] budget reached; rerun to resume")
            break
        tick(f"[control] {rid}")
        if one_run(feat, "deepship", "cnn_small", s, "genrand") != 0:
            print(f"  [warn] {rid} FAILED - see its log above")
        ran += 1
    for s in range(CM_DRAWS):
        rid = w3_run.run_id_for("deepship", "logmel", "cnn_small", "rec_random",
                                s, "none", "erm", False, "dsclipmass")
        if rid in have:
            continue
        if (time.time() - T0) / 60 > args.budget_min:
            print("[stop] budget reached; rerun to resume")
            break
        tick(f"[control] {rid}")
        if one_run(ds_root, "deepship", "cnn_small", s, "dsclipmass") != 0:
            print(f"  [warn] {rid} FAILED - see its log above")
        ran += 1
    tick(f"[control] {ran} runs executed")

    # ------------------------------------------------------- analysis ---- #
    rows = []
    p = os.path.join(RUNS, "results.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    schemes = collections.defaultdict(list)
    ref = {"num": collections.Counter(), "den": collections.Counter()}
    for r in rows:
        rid = r["run_id"]
        if "deepship" not in rid or "rec_random" not in rid:
            continue
        acc = r.get("test_acc")
        if not acc:
            continue
        if rid.endswith("__genrand"):
            schemes["genrand"].append(float(acc))
            for k, v in r.items():
                if k.startswith("acc_") and ("n_" + k[4:]) in r and v:
                    ref["num"][k[4:]] += float(v) * int(r["n_" + k[4:]])
                    ref["den"][k[4:]] += int(r["n_" + k[4:]])
        elif rid.endswith("__dsclipmass"):
            schemes["dsclipmass"].append(float(acc))

    recalls = {c: ref["num"][c] / ref["den"][c] for c in ref["den"]
               if ref["den"][c]}

    def pred_sd(scheme):
        vals = []
        for s in range(CM_DRAWS):
            a = draw_clipmass_ds(s)
            by = collections.Counter()
            for k, spl in a.items():
                if spl == "test":
                    by[rec_cls[k]] += rec_clips[k]
            tot = sum(by.values())
            vals.append(sum(by[c] * recalls.get(c, 0.0) for c in by)
                        / max(tot, 1))
        return 100 * statistics.pstdev(vals) if len(vals) > 1 else 0.0

    print()
    print("=" * 88)
    print("DeepShip rec_random no-op control (4 balanced classes - little to fix)")
    print("=" * 88)
    print(f"  {'scheme':<12}{'n':>3}{'mean':>9}{'sd':>8}{'min':>8}{'max':>8}{'pred.sd':>9}")
    print("  " + "-" * 60)
    for scheme in ("genrand", "dsclipmass"):
        accs = schemes.get(scheme, [])
        if not accs:
            continue
        sd = 100 * statistics.stdev(accs) if len(accs) > 1 else 0.0
        pred = f"{pred_sd(scheme):>8.2f}" if scheme == "dsclipmass" else \
            f"{pred_sd(0):>8.2f}" if recalls else ""
        summary["schemes"][scheme] = dict(
            n=len(accs), mean=statistics.mean(accs), sd=sd,
            min=min(accs), max=max(accs))
        print(f"  {scheme:<12}{len(accs):>3}{statistics.mean(accs):>9.4f}"
              f"{sd:>7.2f}p{min(accs):>8.4f}{max(accs):>8.4f}{pred:>9}")

    g, c = summary["schemes"].get("genrand"), summary["schemes"].get("dsclipmass")
    print()
    if g and c and g["n"] > 1:
        pred_rand_sd = pred_sd(0)
        ratio = c["sd"] / max(g["sd"], 1e-9)
        predicted_noop = pred_rand_sd < 1.5
        observed_noop = 0.6 <= ratio <= 1.67
        print(f"  predicted no-op (pred.sd(genrand) = {pred_rand_sd:.2f} pp < 1.5): "
              f"{predicted_noop}")
        print(f"  observed no-op (sd ratio {ratio:.2f} within [0.6, 1.67]):    "
              f"{observed_noop}")
        if predicted_noop and observed_noop:
            print("  -> CONTROL CONFIRMED: where the composition model predicts")
            print("     little composition variance, stratification is a no-op -")
            print("     the recipe acts exactly where, and only where, it should.")
        elif not predicted_noop:
            print("  -> the composition model predicted variance to fix here;")
            print("     report what stratification did with it (honest either way).")
        else:
            print("  -> mismatch: observed spread differs from the no-op band;")
            print("     investigate before writing.")
        summary["verdict"] = dict(pred_sd_genrand=pred_rand_sd, ratio=ratio,
                                  predicted_noop=predicted_noop,
                                  observed_noop=observed_noop)

    for cand in (WORK, os.getcwd()):
        if cand and os.path.isdir(cand):
            with open(os.path.join(cand, "w3k2_control.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, default=float)
            break
    print()
    tick("summary -> w3k2_control.json")
    print("  -> Output tab -> Create Dataset to keep w3k2-runs + the summary")
