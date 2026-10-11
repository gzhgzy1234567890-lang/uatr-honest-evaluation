#!/usr/bin/env python3
"""Generate the five W3 Kaggle notebooks from the scripts in src/.

Why generate rather than hand-write
-----------------------------------
Each notebook must embed the runner it uses, and the embedded copy has to be
byte-identical to `src/w3_run.py` — otherwise the notebook and the repository
drift and a run cannot be traced back to the code that produced it.  Generating
them, with an assertion on the embedded text, makes that impossible.

Usage
-----
    python notebooks/_build_w3_notebooks.py
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks"


# --------------------------------------------------------------------------- #
def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def writefile(name: str, text: str) -> dict:
    assert "%%writefile" not in text, name
    return code(f"%%writefile /kaggle/working/scripts/{name}\n" + text)


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


# --------------------------------------------------------------------------- #
# shared cells
# --------------------------------------------------------------------------- #
SETUP = r'''# --------------------------------------------------------------------------- #
# 1. locate the feature dataset
#
# Kaggle mounts a dataset by slug, in one of two layouts, and it expands
# archives on ingest. So rather than hard-coding a path we look for the one
# thing that must exist: <root>/oceanship/index.csv. If detection ever fails,
# copy the real path out of the listing below into MANUAL and re-run.
# --------------------------------------------------------------------------- #
import csv, json, os, shutil, subprocess, sys, time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
RUNS = WORK / "w3-runs"
SCRIPTS = WORK / "scripts"
MANUAL = {}          # e.g. MANUAL["features"] = "/kaggle/input/under-water-features/uwatr-features"

T0 = time.time()
def tick(msg):
    print(f"  [{time.time() - T0:7.1f}s] {msg}", flush=True)


def candidates():
    out = []
    for depth in range(1, 5):
        for p in sorted(INPUT.glob("/".join(["*"] * depth))):
            if p.is_dir() and (p / "oceanship" / "index.csv").exists():
                out.append(p)
    return sorted(set(out), key=lambda p: len(p.parts))


MANUAL_FEATURES = MANUAL.get("features")
FEAT = Path(MANUAL_FEATURES) if MANUAL_FEATURES else None
if FEAT is None:
    cands = candidates()
    FEAT = cands[0] if cands else None

print("=" * 88)
print("inputs")
print("=" * 88)
for depth in range(1, 3):
    for p in sorted(INPUT.glob("/".join(["*"] * depth))):
        if p.is_dir():
            kids = sorted(x.name for x in p.iterdir())[:6]
            print(f"  {p}")
            print(f"      {kids}")

if FEAT is None:
    print()
    print("  no feature tree found. A mounted dataset must contain")
    print("  <root>/oceanship/index.csv, <root>/deepship/index.csv, <root>/shipsear/index.csv,")
    print("  /splits. Copy the correct path into MANUAL at the top of this cell and re-run.")
    raise SystemExit("under_water_features not found")

print()
print(f"  features root : {FEAT}")
need = [FEAT / c / "index.csv" for c in ("oceanship", "deepship", "shipsear")]
need += [FEAT / c / f"{r}.npy" for c in ("oceanship", "deepship", "shipsear")
         for r in ("logmel", "spec")]
missing = [str(p.relative_to(FEAT)) for p in need if not p.exists()]
if missing:
    print(f"  MISSING: {missing}")
    raise SystemExit("feature dataset incomplete")

for c in ("oceanship", "deepship", "shipsear"):
    n = sum(1 for _ in open(FEAT / c / "index.csv", encoding="utf-8")) - 1
    gb = sum((FEAT / c / f"{r}.npy").stat().st_size for r in ("logmel", "spec")) / 1e9
    print(f"  {c:<10} {n:>7,} clips   {gb:>6.2f} GB of arrays")
print(f"  splits     : {sorted(p.name for p in (FEAT / 'splits').glob('*'))[:8]}")

mf = FEAT / "manifest.json"
if mf.exists():
    m = json.loads(mf.read_text(encoding="utf-8"))
    print(f"  manifest   : { {k: v.get('rows') for k, v in m.items() if isinstance(v, dict)} }")

SCRIPTS.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)
try:
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu only"
    print(f"  torch      : {torch.__version__}  cuda={torch.cuda.is_available()}  {gpu}")
except ImportError:
    # W3-A3 is CPU-only and must still run where torch is absent
    print("  torch      : not installed (fine for the CPU-only notebooks)")
print(f"  free space : {shutil.disk_usage(WORK).free / 1e9:.1f} GB")
tick("setup done")
'''

DRIVER = r'''# --------------------------------------------------------------------------- #
# the sweep driver
#
# One configuration per subprocess, so a crash cannot take the session down and a
# re-run picks up exactly where the last one stopped. "Already done" is decided
# from results.csv — which is read from /kaggle/working AND from anything mounted
# under /kaggle/input, so saving a previous session's output as a dataset makes
# the next session resume automatically.
#
# A directory whose name starts with an underscore is SCRATCH and is never
# scanned: the --smoke probe writes a row under the same run id as the real run of
# the same configuration, which in the first W3-A session made the driver skip the
# real 24-epoch `vessel_disjoint seed 0` and record the 2-epoch probe instead.
# --------------------------------------------------------------------------- #
import csv as _csv
sys.path.insert(0, str(SCRIPTS))
import w3_run                      # the module defines run_id_for(); no rule duplication

BUDGET_MIN = %(budget)d
RUNNER = str(SCRIPTS / "w3_run.py")
BASE = ["--data", str(FEAT), "--out", str(RUNS), "--num-workers", "2"]

# The training recipe is part of a run's identity, and it lives in exactly two
# places: here, and in the run id via RECIPE_TAG. Bump RECIPE_TAG whenever RECIPE
# changes, or a re-run with a different recipe will be skipped as "already done".
# Per-config `extra` flags come after RECIPE, and argparse takes the last value,
# so a single config can override any of these.
RECIPE_TAG = %(recipe_tag)r
RECIPE = %(recipe)s


def cfg_tag(c):
    t = c.get("tag", "")
    return f"{RECIPE_TAG}_{t}" if t else RECIPE_TAG


def done_ids():
    ids = set()
    roots = [WORK] + ([INPUT] if INPUT.is_dir() else [])
    # the base is carried alongside each file.  Reading it from the loop variable
    # after the collection loop has finished gives the LAST root for every file,
    # so a file under /kaggle/working would be tested against /kaggle/input, fail
    # the relative_to and then be rejected by the scratch rule below — which is
    # how the resume scan came back empty.
    found = []
    for base in roots:
        for depth in range(4):
            found += [(base, f) for f in sorted(
                base.glob("/".join(["*"] * depth + ["results.csv"])))]
    for base, f in found:
        try:
            rel = f.relative_to(base)
        except ValueError:
            rel = f
        if any(str(p).startswith("_") for p in rel.parts):
            print(f"    [resume] skipping scratch {f}")
            continue
        try:
            with f.open(encoding="utf-8", newline="") as fh:
                got = list(_csv.DictReader(fh))
            ids |= {x["run_id"] for x in got}
            print(f"    [resume] {f} -> {len(got)} rows")
        except Exception as exc:
            print(f"    [resume] {f} unreadable ({exc})")
    return ids


def rid(c):
    corpus = c.get("corpus", "oceanship")
    extra = list(c.get("extra", []))
    tc = extra[extra.index("--test-corpus") + 1] if "--test-corpus" in extra else None
    proto = w3_run.protocol_for(corpus, c["protocol"], corpus if tc else None, tc)
    return w3_run.run_id_for(corpus, c["rep"], c["backbone"], proto, c["seed"],
                             c.get("domain", "none"), c.get("method", "erm"),
                             c.get("shuffled", False), cfg_tag(c))


def sweep(configs):
    have = done_ids()
    todo = [c for c in configs if rid(c) not in have]
    print(f"[state] {len(configs) - len(todo)} of {len(configs)} already complete")
    print(f"[recipe] tag={RECIPE_TAG}  {' '.join(RECIPE)}")
    if not todo:
        print("nothing to do")
        return
    t0 = time.time()
    ran = 0
    for c in todo:
        el = (time.time() - t0) / 60
        if el > BUDGET_MIN:
            print(f"\n[stop] budget of {BUDGET_MIN} min reached after {ran} runs. "
                  f"Re-run this cell in the next session to continue.")
            break
        cmd = [sys.executable, RUNNER] + BASE + [
            "--corpus", c.get("corpus", "oceanship"), "--rep", c["rep"],
            "--backbone", c["backbone"], "--protocol", c["protocol"],
            "--seed", str(c["seed"]), "--method", c.get("method", "erm"),
            "--domain", c.get("domain", "none"), "--vocab", c.get("vocab", "native"),
            "--tag", cfg_tag(c),
        ] + RECIPE + (["--shuffle-labels"] if c.get("shuffled") else []) + list(c.get("extra", []))
        print(f"\n{'=' * 88}\n[{ran + 1}/{len(todo)}] {rid(c)}   elapsed {el:.1f} min"
              f"\n{'=' * 88}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode:
            print(f"[warn] exit {r.returncode}; leaving it pending and continuing", flush=True)
        ran += 1
    print(f"\n[session] {ran} runs in {(time.time() - t0) / 60:.1f} min")


CONFIGS = %(configs)s
print(f"[plan] {len(CONFIGS)} configurations in this notebook")
sweep(CONFIGS)
'''

REPORT = r'''# --------------------------------------------------------------------------- #
# what came out of this notebook
# --------------------------------------------------------------------------- #
import csv, glob, json
from pathlib import Path

res = RUNS / "results.csv"
if not res.exists():
    raise SystemExit("no results yet")

with res.open(encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))
print(f"{len(rows)} runs in {res}")
print(f"predictions : {len(list((RUNS / 'predictions').glob('*.npz')))} files")

STD = ("clip_random", "rec_random", "vessel_disjoint", "temporal", "group_disjoint")
print()
print(f"  {'run_id':<58}{'acc':>8}{'F1':>8}{'leak%':>8}{'min':>7}")
for r in sorted(rows, key=lambda x: x["run_id"]):
    print(f"  {r['run_id']:<58}{float(r['test_acc']):>8.4f}"
          f"{float(r['test_macro_f1']):>8.4f}{float(r['leak_rec_pct']):>8.1f}"
          f"{float(r['wall_min']):>7.1f}")

print()
print("=" * 88)
print("the headline number of this notebook")
print("=" * 88)
import collections
by = collections.defaultdict(list)
for r in rows:
    if r["method"] == "erm" and r["domain"] == "none" and r["shuffled"] in ("0", 0):
        by[(r["corpus"], r["rep"], r["backbone"], r["protocol"])].append(float(r["test_acc"]))
for k in sorted(by):
    v = by[k]
    print(f"  {k[0]:<10}{k[1]:<7}{k[2]:<14}{k[3]:<17} n={len(v):>2}  "
          f"mean {sum(v) / len(v):.4f}  per seed {[round(x, 3) for x in v]}")

total = sum(f.stat().st_size for f in RUNS.rglob("*") if f.is_file())
print()
print(f"total output: {total / 1e6:.1f} MB under {RUNS}")
print()
print("next: Output tab -> Create Dataset (name it after this notebook) so the next")
print("      notebook can mount these runs and so w3_agg can see them all at once.")
'''

SAVE_A = r'''# --------------------------------------------------------------------------- #
# carry the runs forward as a dataset
#
# Kaggle's Output tab -> "Create Dataset" is the official path from a rendered
# notebook, and it is what makes the next session resume instead of starting over.
# This cell just makes sure the folder is clean and self-describing first.
# --------------------------------------------------------------------------- #
import json, shutil
from pathlib import Path

# the scripts are already embedded in every notebook, so they need not be part of
# the run artifact; the run rows and the per-clip predictions must be.
res_csv = RUNS / "results.csv"
meta = {
    "kind": "w3-runs",
    "notebook": "W3-%(tag)s",
    "recipe_tag": RECIPE_TAG,
    "recipe": RECIPE,
    # W3-A3 is a CPU probe and writes no rows, so the count is conditional
    "runs": (sum(1 for _ in open(res_csv, encoding="utf-8")) - 1) if res_csv.exists() else 0,
    "has_predictions": (RUNS / "predictions").is_dir(),
    "has_embeddings": (RUNS / "embeddings").is_dir(),
    "note": "per-clip predictions carry rec_key / vessel / group / src duration, "
            "which is what makes the recording-level bootstrap possible",
}
(RUNS / "RUNS_INFO.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=2))
print()
print(f"  {RUNS}")
for sub in sorted(RUNS.iterdir()):
    if sub.is_dir():
        n = len(list(sub.glob("*")))
        print(f"    {sub.name:<16} {n:>5} files")
print()
print("  -> Output tab -> Create Dataset -> include the whole 'w3-runs' folder")
'''

SAVE_PROBE = r'''# --------------------------------------------------------------------------- #
# keep the numbers and the descriptors
#
# W3-A3 writes no runs, so there is no results.csv to carry forward.  What is
# worth keeping is the plain record of the measurements - a later session can
# re-run the whole analysis from these numbers - and the descriptors themselves,
# so the nearest-neighbour work does not have to read 14 GB of float16 again.
# --------------------------------------------------------------------------- #
import json

PROBE = WORK / "w3a3"
PROBE.mkdir(parents=True, exist_ok=True)

record = {
    "kind": "w3a3-leak-probe",
    "notebook": "W3-A3",
    "descriptor": "mean-pooled log-mel 16x16 (256 dims), per corpus",
    "metadata_ceiling": {
        f"{c}|{p}": {"vessel": CEIL[(c, p)][0], "recording": CEIL[(c, p)][1]}
        for (c, p) in CEIL},
    "knn": {f"{c}|{p}": v for (c, p), v in KNN.items()},
    "note": "CPU only, no runs written; descriptors saved as "
            "leakprobe_<corpus>16.npy so the analysis can be repeated without "
            "re-reading the feature arrays",
}
(PROBE / "w3a3_probe.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record, indent=2, default=float)[:1200])
print()
print("  -> Output tab -> Create Dataset if you want the descriptors kept")
'''

REPORT_DIAG = r'''# --------------------------------------------------------------------------- #
# is the model actually learning?  Read this before trusting any accuracy.
#
# The first W3-A session finished 32 configurations and looked healthy, but the
# training loss never went below 1.6 against a random-guess value of log(K), and
# only three classes were ever predicted.  A protocol difference measured on an
# underfitted model is not interpretable, so these columns are the gate.
# --------------------------------------------------------------------------- #
import csv
from pathlib import Path

res = RUNS / "results.csv"
if not res.exists():
    raise SystemExit("no results yet")
with res.open(encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))

def f(r, k, d=float("nan")):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return d

print(f"{'run_id':<52}{'K':>4}{'train':>7}{'test':>7}{'gap':>8}{'F1':>7}"
      f"{'predK':>6}{'chance':>8}{'min':>7}")
for r in sorted(rows, key=lambda x: x["run_id"]):
    print(f"  {r['run_id']:<50}{int(f(r, 'n_classes', 0)):>4}"
          f"{f(r, 'train_acc'):>7.4f}{f(r, 'test_acc'):>7.4f}"
          f"{f(r, 'train_val_gap'):>8.4f}{f(r, 'test_macro_f1'):>7.4f}"
          f"{int(f(r, 'n_pred_classes', 0)):>6}{f(r, 'chance_acc'):>8.4f}"
          f"{f(r, 'wall_min'):>7.1f}")

print()
print("how to read it")
print("  train  - training accuracy of the selected checkpoint. If this is near the")
print("           chance column the model has not fitted the training set at all and")
print("           NOTHING downstream is worth interpreting.")
print("  gap    - train minus test. Large means overfitting; near zero with a low")
print("           train value means underfitting.")
print("  predK  - how many of the K classes the model ever predicts on the test set.")
print("           A value far below K means the predictions have collapsed onto a")
print("           few classes, which is what a too-large first-layer input scale or a")
print("           too-short schedule looks like.")
'''


# --------------------------------------------------------------------------- #
# notebook definitions
# --------------------------------------------------------------------------- #
# The training recipe, in one place. It enters every run id through RECIPE_TAG, so
# changing either is an intentional act with a visible consequence rather than a
# silent overwrite of earlier results.
RECIPE_MAIN = ["--epochs", "24", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]
# Tag history: rec1 rows came from a run whose LR scheduler was never stepped, so
# their accuracies describe an untrained network.  rec2 marks the first rows that
# were produced at the learning rate the recipe actually asks for.
# rc  rows: same defect.  rc2 is the fixed recipe check.


def drv(budget, tag, configs, recipe=None):
    return DRIVER % {"budget": budget, "recipe_tag": tag,
                     "recipe": repr(recipe or RECIPE_MAIN),
                     "configs": repr(configs)}


def nb_w3a():
    configs = []
    for rep in ("logmel", "spec"):
        for proto in ("clip_random", "rec_random", "vessel_disjoint"):
            for seed in range(5):
                configs.append({"corpus": "oceanship", "rep": rep, "backbone": "cnn_small",
                                "protocol": proto, "seed": seed})
    for proto in ("clip_random", "rec_random", "vessel_disjoint"):
        configs.append({"corpus": "oceanship", "rep": "logmel", "backbone": "logreg",
                        "protocol": proto, "seed": 0})
    return notebook([
        md("""# W3-A — the headline: does the split unit decide the accuracy?

This notebook produces the paper's **Table 2** and **Figure 2**. One model, one
training recipe, one set of hyper-parameters — the only thing that changes is
which 5 s clips end up on which side of the split:

| protocol | unit of the split | what it measures |
|---|---|---|
| `clip_random` | the 5 s clip | what the literature reports |
| `rec_random` | the acoustic session (mmsi + AIS timestamp) | the level nobody reports |
| `vessel_disjoint` | the ship (mmsi) | the only level at which the model has not met the ship |

Nothing else differs: same optimiser, same schedule, same 24 epochs, same seed
handling. That identity is the only reason a difference between the three rows is
interpretable at all.

**Scope** —30 runs + 3 linear-probe floors.

| | |
|---|---|
| corpus | Oceanship (53,754 clips / 12 ship-type classes, the vessel-disjoint subset) |
| model | `cnn_small`, a 137K-parameter 4-block CNN |
| representations | `logmel` and `spec` — both, on purpose (see below) |
| protocols | `clip_random`, `rec_random`, `vessel_disjoint` |
| seeds | 0— |

**Why two representations.** If the collapse to the strict protocol appears only
in log-mel, that is an artefact of the feature engineering. If it appears in both,
it is a property of the data. The paper cannot claim the second without measuring
the first.

**Accelerator** — any GPU. Expected wall clock: ~5–7 h for the 30 CNN runs, less
if you land on a P100. The driver stops cleanly at 600 minutes; re-run the cell
(or the whole notebook) and it continues where it stopped.

**What you do after it finishes** — Output tab → Create Dataset, so W3-B… can
mount these runs.""") ,
        code(SETUP),
        md("""## the run matrix

`cnn_small` × {logmel, spec} × {clip_random, rec_random, vessel_disjoint} × 5 seeds
= 30 runs, plus three `logreg` floors (a linear read-out on the flattened
time-frequency map — if *that* separates the classes, the task was never the hard
part).

60 minutes of GPU per session is plenty for a probe; the real sweep is cell 5."""),
        writefile("w3_run.py", (SRC / "w3_run.py").read_text(encoding="utf-8")),
        code(r'''# --------------------------------------------------------------------------- #
# 4. wire check — one minute of compute, before committing five hours
#
# --smoke caps it at 2 epochs on 600 training clips and prints every resolved
# path, protocol, domain count and leak rate. If anything about the data is
# wrong, it shows up here rather than at hour four.
# --------------------------------------------------------------------------- #
cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
       "--data", str(FEAT), "--out", str(WORK / "_probe"),
       "--corpus", "oceanship", "--protocol", "vessel_disjoint", "--seed", "0",
       "--rep", "logmel", "--backbone", "cnn_small", "--smoke", "--num-workers", "0",
       "--tag", "probe"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print()
print("  if that printed rows=train/val/test and a leak rate, the wiring is correct.")
print("  note: the probe writes to _probe/ with tag 'probe' — a leading underscore is")
print("        excluded from the resume scan, so it can never mask the real run.")
tick("probe done")'''),
        md("""## the sweep

One subprocess per configuration, appending a row to `results.csv` and a
per-clip prediction file. Nothing is recomputed: a configuration whose row
already exists is skipped, and the row set is read from `/kaggle/input` as well
as `/kaggle/working`, so mounting a previous session's output resumes it.

The per-clip predictions are not optional — they carry the recording id and the
vessel id of every test clip, which is what makes the recording-level bootstrap
confidence interval possible in W3-E. A point estimate without it is exactly the
practice this paper argues against."""),
        code(drv(600, "rec2", configs)),
        code(REPORT),
        code(REPORT_DIAG),
        code(SAVE_A % {"tag": "A"}),
    ])


def nb_w3b():
    configs = []
    for backbone in ("logreg", "cnn_tiny", "cnn_wide"):
        for proto in ("clip_random", "rec_random", "vessel_disjoint"):
            for seed in range(3):
                configs.append({"corpus": "oceanship", "rep": "logmel", "backbone": backbone,
                                "protocol": proto, "seed": seed})
    for backbone in ("resnet18", "efficientnet_b0"):
        for proto in ("clip_random", "vessel_disjoint"):
            for seed in range(2):
                configs.append({"corpus": "oceanship", "rep": "logmel", "backbone": backbone,
                                "protocol": proto, "seed": seed})
    for backbone in ("resnet18", "efficientnet_b0"):
        configs.append({"corpus": "oceanship", "rep": "spec", "backbone": backbone,
                        "protocol": "vessel_disjoint", "seed": 0})
    return notebook([
        md("""# W3-B — baselines and model scale

This notebook produces the paper's **Table 4** and **Figure 4**, and it is where
two published claims get tested rather than cited.

**Claim 1** — Soti et al. (arXiv:2609.29179) report that an 11.17M-parameter
ResNet18 gave **no** improvement over a ~157K compact CNN once the split was
honest (macro F1 0.7226). We run the ladder around that parameter count on the
same split with the same recipe:

| backbone | parameters | role |
|---|---|---|
| `logreg` | 481K | linear read-out of the flattened map — the floor |
| `cnn_tiny` | ~61K | below the reference |
| `cnn_small` | ~137K | the compact reference (already run in W3-A) |
| `cnn_wide` | ~543K | above it |
| `resnet18` | 11.2M | ImageNet-pretrained, the claim under test |
| `efficientnet_b0` | 5.3M | a second pretrained architecture |

**Claim 2** — Sci Rep 2025 reports 72.38% with an STFT front end and 71.20% with
Mel, a 1.2-point difference for the same network. We run `resnet18` on both
representations to see whether a representation effect of that size survives an
honest protocol.

**Cost.** The compact ladder is cheap; `resnet18` is roughly five times a
`cnn_small` epoch. The driver stops at 600 minutes and resumes on re-run.

**Accelerator** — GPU. Expected: ~5— h.

**Afterwards** — Output → Create Dataset.""") ,
        code(SETUP),
        md("""## what runs

* compact ladder (`logreg`, `cnn_tiny`, `cnn_wide`) × 3 protocols × 3 seeds = 27
* pretrained pair (`resnet18`, `efficientnet_b0`) × 2 protocols × 2 seeds = 8
* representation check: pretrained pair on `spec`, strict protocol = 2

The pretrained pair is deliberately not run on all 15 cells: the question is
whether scale helps *at all* under an honest split, and that is answered by the
strict protocol. Running it on `clip_random` as well is the control that shows
what a pretrained ImageNet backbone can reach when the split is not honest."""),
        writefile("w3_run.py", (SRC / "w3_run.py").read_text(encoding="utf-8")),
        code(r'''# wire check, including the two pretrained backbones — they download weights,
# and an offline runtime falls back to random init with a warning. Better to
# learn that here than after four hours.
for backbone in ("cnn_tiny", "resnet18"):
    cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
           "--data", str(FEAT), "--out", str(WORK / "_probe"),
           "--corpus", "oceanship", "--protocol", "vessel_disjoint", "--seed", "0",
           "--rep", "logmel", "--backbone", backbone, "--smoke", "--num-workers", "0",
           "--tag", "probe"]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
tick("probe done")'''),
        code(drv(600, "rec2", configs)),
        code(REPORT),
        code(REPORT_DIAG),
        code(SAVE_A % {"tag": "B"}),
    ])


def nb_w3c():
    emb = [
        {"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "clip_random", "seed": 0, "extra": ["--embeddings"]},
        {"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "vessel_disjoint", "seed": 0, "extra": ["--embeddings"]},
        {"corpus": "deepship", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "rec_random", "seed": 0, "extra": ["--embeddings"]},
        {"corpus": "shipsear", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "group_disjoint", "seed": 0, "extra": ["--embeddings"]},
        {"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "vessel_disjoint", "seed": 0, "shuffled": True},
        {"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "clip_random", "seed": 0, "shuffled": True},
    ]
    return notebook([
        md("""# W3-C — diagnostics: the checks a reviewer is entitled to ask for

This notebook produces the paper's **Table 7** and answers four questions that
the strict protocol alone does not.

### P8 · How much *recording identity* survives in the representation?

Hummel et al. (2026, Ecological Informatics 95 103709) make this **mandatory**
reporting for passive-acoustic deep learning: if a linear read-out of the frozen
features can name the recording a clip came from, then part of any downstream
accuracy is session recall, not ship recognition. We report a nearest-centroid
balanced accuracy and the normalised mutual information against the true
recording id — plus the same pipeline on **shuffled keys**, so the reader can see
what chance looks like at this data size. Without that control the number is
uninterpretable.

### P6 · Are the domain labels we want invariance to actually visible?

If `date` or the author's own split is already invisible in the features, then a
gain from IRM/VREx on it would be unexplained, and should be treated as a red
flag rather than a result. So we measure the separability *before* training the
invariance methods in W3-D.

### P5 · Is duration a shortcut?

Short clips are padded rather than dropped, and duration is class-correlated
(Oceanship Passenger median 3.67 s against Tug 5.00 s). A classifier that sees
only *how much of the clip is padding* therefore predicts the class for free.
Two proxies are scored out-of-fold: the stored source duration, and the fraction
of time frames resting on the log-mel floor — the latter being exactly what the
network reads off the image with no acoustic understanding at all.

### P7 · Does the label-shuffling control hold?

Two runs with the **training labels permuted** and everything else identical.
A model that scores above chance here is not learning the task.

**Accelerator** — GPU, but only for six short runs. Expected: ~1.5— h total.""") ,
        code(SETUP),
        md("""## what runs

Four training runs with `--embeddings` (which dumps the penultimate features for
the test set and a 20,000-clip sample of the training set), and two
label-shuffled controls. Then the diagnostics run on CPU in a couple of minutes.

The negative-control reading is the important one: `vessel_disjoint` should give
a recording-identity probe near the shuffled floor, because no test recording was
seen in training. On `clip_random` the same probe should be far above it. That
contrast, not the absolute value, is what shows the representation is carrying
session identity."""),
        writefile("w3_run.py", (SRC / "w3_run.py").read_text(encoding="utf-8")),
        writefile("w3_diag.py", (SRC / "w3_diag.py").read_text(encoding="utf-8")),
        code(drv(420, "rec2", emb)),
        code(r'''# --------------------------------------------------------------------------- #
# diagnostics: recording/vessel identity probe, domain separability, and the
# duration shortcut. CPU only, a couple of minutes.
# --------------------------------------------------------------------------- #
cmd = [sys.executable, str(SCRIPTS / "w3_diag.py"),
       "--data", str(FEAT), "--runs", str(RUNS),
       "--out", str(RUNS / "tables"), "--max-runs", "8", "--max-classes", "3000"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print()
print("the full report is at", RUNS / "tables" / "w3_diagnostics.txt")'''),
        code(REPORT),
        code(REPORT_DIAG),
        code(SAVE_A % {"tag": "C"}),
    ])


def nb_w3d():
    configs = []
    for domain in ("date", "official"):
        for method in ("irm", "vrex", "dann"):
            for seed in range(2):
                configs.append({"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
                                "protocol": "vessel_disjoint", "seed": seed,
                                "method": method, "domain": domain})
    for seed in range(2):
        configs.append({"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
                        "protocol": "vessel_disjoint", "seed": seed,
                        "method": "irm", "domain": "vessel_hash"})
        configs.append({"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
                        "protocol": "vessel_disjoint", "seed": seed,
                        "method": "vrex", "domain": "dur"})
    for seed in range(2):
        configs.append({"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
                        "protocol": "vessel_disjoint", "seed": seed,
                        "domain": "date"})            # ERM with the labels present = control
    for seed in range(2):
        for tc in ("deepship", "shipsear"):
            configs.append({"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
                            "protocol": "vessel_disjoint", "seed": seed,
                            "extra": ["--train-corpus", "oceanship", "--test-corpus", tc,
                                      "--vocab", "common4"],
                            "tag": f"cross{tc}"})
    return notebook([
        md("""# W3-D — can the gap be narrowed? and how far does it transfer?

This notebook produces **Table 5** (methods) and **Table 6** (cross-corpus), and
it is the part that offers a remedy rather than only a diagnosis.

## C4 — domain generalisation, not domain adaptation

Nobody who reports the strict-protocol collapse offers a fix: Soti et al. stop at
macro F1 0.7226, and LHK-Net list the honest split as *future work*. That is the
gap this notebook works in.

The setting is deliberately **DG, not UDA**: the target domain is **never
observed**, not even unlabelled. UDA (Yang et al. 2024, Applied Acoustics 216
109725 and Ocean Engineering 119137) assumes unlabelled target data is available
at training time, which is exactly what a deployed passive-acoustic classifier
does not have. The distinction has to be stated in the paper, and this table is
where it is demonstrated rather than asserted.

| method | objective |
|---|---|
| `erm` | cross-entropy only — the reference, from W3-A |
| `irm` | + gradient penalty on per-domain risk (IRMv1) |
| `vrex` | + variance of per-domain risk |
| `dann` | + adversarial domain classifier behind a gradient reversal layer |

Domain labels come from the index metadata, so no extra annotation is used:
`date` (from the recording timestamp) and `official` (the authors' own
split). `dur` is a duration quartile. All are printed with their group sizes at
run time.

### The negative control is the point

`vessel_hash` partitions the training set by `md5(mmsi) mod 8` — real, disjoint,
and acoustically **meaningless**. An invariance gain there would mean the
regulariser did the work, not the domain structure, and the claim would be void.
Most DG papers do not run this control; the paper's argument rests on it.

*IRM and VREx run in fp32* (double back-propagation is not safe under fp16
autocast), so they are about twice as slow per epoch. Budget is 600 minutes.

## Cross-corpus

Train on Oceanship, test on DeepShip or ShipsEar, restricted to the four classes
they share (Cargo, Passenger, Tanker, Tug). The target corpus is **entirely**
unseen. ShipsEar's mapping to that vocabulary is a literature convention and is
flagged as `vocab_approx=1` in the results — an approximation that is disclosed
is worth more than a rounder number that is not."""),
        code(SETUP),
        md("""## what runs

* `irm` / `vrex` / `dann` × {`date`, `official`} × 2 seeds = 12
* negative control: `irm` on `vessel_hash`, `vrex` on `dur` × 2 seeds = 4
* `erm` with the domain labels present but unused × 2 = 2
  (this is the control that shows the domain information alone changes nothing)
* cross-corpus: Oceanship → DeepShip and → ShipsEar × 2 seeds = 4

The ERM references for `vessel_disjoint` come from W3-A; if that notebook's runs
are mounted, this one will report the deltas directly."""),
        writefile("w3_run.py", (SRC / "w3_run.py").read_text(encoding="utf-8")),
        code(r'''# wire check: the invariance path must actually reach the loss. The printed
# "domain -> N groups" line is the evidence; if it says 1 group, the penalty is
# a no-op and the run would be indistinguishable from ERM.
for method, domain in (("irm", "date"), ("dann", "official")):
    cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
           "--data", str(FEAT), "--out", str(WORK / "_probe"),
           "--corpus", "oceanship", "--protocol", "vessel_disjoint", "--seed", "0",
           "--rep", "logmel", "--backbone", "cnn_small", "--smoke", "--num-workers", "0",
           "--method", method, "--domain", domain, "--tag", "probe"]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
print()
print("  check: the 'domain' line shows more than one group, and a DANN run reports")
print("  'params=... +domain_head'.")
tick("probe done")'''),
        code(drv(600, "rec2", configs)),
        code(REPORT),
        code(REPORT_DIAG),
        code(SAVE_A % {"tag": "D"}),
    ])


def nb_w3e():
    return notebook([
        md("""# W3-E — tables and figures

**No GPU.** This notebook reads every run it can find and produces the paper's
tables and figures in minutes.

Run rows are collected from `/kaggle/working` **and** from anything mounted under
`/kaggle/input`, so if you saved W3-A''s outputs as datasets they are picked up
automatically; if you did not, it still reports everything currently in the
session.

## What it computes

**Recording-level bootstrap confidence intervals.** Test clips are not
independent: one recording contributes dozens of clips sharing a channel, a
background-noise field and an operating condition. Resampling clips would treat
fifty clips from one ship as fifty independent pieces of evidence and report an
interval that is far too narrow. We resample **recordings** with replacement,
1,000 times.

**The leak decomposition.** For each leaky protocol the test set is split into
clips whose ship was also seen in training and clips whose was not, and the
metric is reported on each part. This is the mechanism behind the headline gap:
it shows how much of a 99% number is recognition and how much is recall of a ship
the model already met.

**Everything else**: the inflation gap, per-class accuracy, per-seed spread (the
test-set size is a random variable —6,442 to 8,531 clips across the five
vessel-disjoint seeds), the parameter-count ladder, the domain-generalisation
table with its negative control, cross-corpus results, and the label-shuffling
control."""),
        code(SETUP),
        code(r'''# --------------------------------------------------------------------------- #
# find every run directory this session can see
# --------------------------------------------------------------------------- #
import csv as _csv
from pathlib import Path

def find_runs():
    dirs = []
    for base in (WORK, INPUT):
        if not base.is_dir():
            continue
        for depth in range(4):
            for p in sorted(base.glob("/".join(["*"] * depth + ["results.csv"]))):
                # scratch (leading underscore below the search root) is never a
                # run: this is where the --smoke probe lives, and its row carries
                # the same run id as the real configuration
                try:
                    rel = p.relative_to(base)
                except ValueError:
                    rel = p
                if any(str(part).startswith("_") for part in rel.parts):
                    print(f"  [skip] scratch {p}")
                    continue
                if p.parent not in dirs:
                    dirs.append(p.parent)
    return dirs

RUN_DIRS = find_runs()
if not RUN_DIRS:
    raise SystemExit(
        "no results.csv found anywhere. Run W3-A first, or mount its output as a "
        "dataset (Output tab -> Create Dataset) so it lands under /kaggle/input."
    )
print("run directories:")
tot = 0
for d in RUN_DIRS:
    n = sum(1 for _ in open(d / "results.csv", encoding="utf-8")) - 1
    tot += n
    print(f"  {d}   {n} runs")
print(f"  total {tot} run rows")
tick("inputs found")'''),
        writefile("w3_agg.py", (SRC / "w3_agg.py").read_text(encoding="utf-8")),
        writefile("w3_diag.py", (SRC / "w3_diag.py").read_text(encoding="utf-8")),
        code(r'''# --------------------------------------------------------------------------- #
# the tables and the figures
# --------------------------------------------------------------------------- #
TABLES = WORK / "w3-tables"
cmd = [sys.executable, str(SCRIPTS / "w3_agg.py"), "--data", str(FEAT),
       "--out", str(TABLES), "--runs"] + [str(d) for d in RUN_DIRS]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)'''),
        code(r'''# diagnostics, if any embedding dumps were produced by W3-C
emb = []
for d in RUN_DIRS:
    for depth in range(3):
        emb += sorted(d.glob("/".join(["*"] * depth + ["embeddings"])))
if emb:
    cmd = [sys.executable, str(SCRIPTS / "w3_diag.py"), "--data", str(FEAT),
           "--runs", str(RUN_DIRS[0]), "--out", str(TABLES)]
    for d in RUN_DIRS[1:]:
        cmd += ["--runs", str(d)]
    print("$ " + " ".join(cmd[:8]) + " ...", flush=True)
    subprocess.run(cmd, check=False)
else:
    print("no embeddings found — run W3-C to produce them")'''),
        code(r'''# --------------------------------------------------------------------------- #
# the deliverables, rendered here so the notebook is readable on its own
# --------------------------------------------------------------------------- #
for name in ("w3_tables.txt", "w3_diagnostics.txt"):
    p = TABLES / "tables" / name
    if not p.exists():
        continue
    print("\n" + "#" * 100)
    print(f"# {name}")
    print("#" * 100)
    print(p.read_text(encoding="utf-8"))

figs = sorted((TABLES / "figures").glob("*.png")) if (TABLES / "figures").is_dir() else []
print(f"\nfigures: {[f.name for f in figs]}")
print(f"tables : {sorted(p.name for p in (TABLES / 'tables').glob('*.csv'))}")
print()
print("next: Output tab -> Create Dataset for the tables, or download them directly.")'''),
        md("""## reading the output

`tables/w3_tables.txt` is the paper draft in plain text: T1 leakage audit, T2
headline, T3 inflation gap, T4 leak decomposition, T5 per-seed spread, T6
parameter ladder, T7 domain generalisation, T8 cross-corpus, T9 label-shuffling.

`tables/all_runs.csv` is every run row, the raw material for anything not
tabulated here. `tables/bootstrap_ci.csv` holds the recording-level intervals,
and `tables/leak_scope.csv` the seen/unseen decomposition.

`figures/` holds F2 (protocol gap), F3 (leak decomposition) and F4 (scale curve).
F1 — the three-level leakage ladder — is drawn by hand from the frozen leak
audit, since it is a diagram of the dataset rather than of a run."""),
    ])


def nb_w3a2():
    """Recipe check: can the model fit the training set at all?"""
    R = ["--epochs", "60", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]
    configs = [
        # ---- the fitting question, over (length x learning rate) --------------
        {"tag": "f24_1e3", "corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "clip_random", "seed": 0,
         "extra": ["--epochs", "24", "--norm", "affine", "--lr", "0.001", "--warmup", "3"]},
        {"tag": "f24_3e3", "corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "clip_random", "seed": 0,
         "extra": ["--epochs", "24", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]},
        {"tag": "f60_3e3", "corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "clip_random", "seed": 0,
         "extra": ["--epochs", "60", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]},
        # ---- the pipeline check: 512 clips, 200 epochs, seconds of GPU -------
        # A working pipeline memorises 512 clips.  If this cannot, the fault is
        # upstream of the recipe and no amount of GPU time will fix it.
        {"tag": "ovf512", "corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "clip_random", "seed": 0,
         "extra": ["--max-train", "512", "--epochs", "200", "--norm", "affine",
                   "--lr", "0.003", "--warmup", "5"]},
        # ---- capacity, now at a learning rate that actually trains -----------
        {"tag": "res30", "corpus": "oceanship", "rep": "logmel", "backbone": "resnet18",
         "protocol": "clip_random", "seed": 0,
         "extra": ["--epochs", "30", "--norm", "affine", "--lr", "0.001", "--warmup", "3"]},
        # ---- the honest headline number, early -------------------------------
        {"tag": "v60_3e3", "corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
         "protocol": "vessel_disjoint", "seed": 0,
         "extra": ["--epochs", "60", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]},
    ]
    return notebook([
        md("""# W3-A2 — recipe check: can the model fit the training set at all?

**Run this before re-running anything else.** It costs about two hours and it
decides whether the fifteen-hour benchmark is worth spending.

## What the previous run of this notebook actually found

Seven configurations, 217 minutes, no errors — and every one of them invalid, for
a reason that was invisible in the log at the time:

| config | recipe | train acc | test acc | classes predicted |
|---|---|---|---|---|
| `rc_base24` | 24 ep, raw dB, lr 3e-4, **warmup 0** | 0.359 | 0.347 | 5 of 12 |
| `rc_aff24` | 24 ep, `[0,1]`, lr 1e-3 and warm-up | 0.273 | 0.275 | 3 of 12 |
| `rc_aff60` | 60 ep, `[0,1]`, lr 1e-3 and warm-up | 0.288 | 0.287 | 3 of 12 |
| `rc_ds60` | 60 ep, standardised, lr 1e-3 and warm-up | 0.286 | 0.287 | 3 of 12 |
| `rc_res30` | resnet18 11.2M, 30 ep, lr 5e-4 and warm-up | 0.303 | 0.294 | 9 of 12 |

More capacity did not help, more epochs did not help, and the *better* input
scaling did **worse** than the raw dB. That pattern had one candidate explanation —
"the representation is too weak" — and it was wrong.

**The scheduler was never stepped.** `LambdaLR` was constructed and never
advanced, so the learning rate stayed at `lr_at(0)` for the whole run. With a
warm-up configured, `lr_at(0) = 1 / (warmup * steps_per_epoch)`:

| config | requested lr | effective lr |
|---|---|---|
| `rc_aff24` / `rc_aff60` / `rc_ds60` (warm-up 3, 588 steps/epoch) | 1e-3 | **5.7e-7** |
| `rc_res30` (warm-up 3) | 5e-4 | **2.8e-7** |
| `rc_base24` (**warm-up 0**) | 3e-4 | **3e-4** — why it did best |

Every configuration except `rc_base24` was frozen at its initialisation, which is
exactly what a plateau that ignores capacity, normalisation and epoch count looks
like. The one configuration that ran at a real learning rate was the one that
scored highest.

Fixed in `w3_run.py`: the scheduler is stepped after every optimiser step, the
live learning rate is printed on every epoch line, `lr_last` is recorded in
`results.csv`, and **a schedule that does not move is now a hard error** rather
than a silent one.

## Why not just re-run the benchmark

Because the model may still not fit. One more measurement settles it, and the
second configuration below makes that measurement in seconds of GPU: **a working
pipeline memorises 512 clips.** If `ovf512` does not reach a high training
accuracy, the fault is upstream of the recipe and no amount of GPU time helps.

Note that this paragraph is the only part of the original notebook that survived
contact with the data: **the leakage is real and fully exploitable.** Predicting
each test clip's majority training label for its own vessel — no audio at all —
scores **99.44%** on `clip_random` seed 0 and still **98.16%** on `rec_random`,
where the recordings are disjoint but the vessels are not. So the ceiling is not
0.35; the plateau was the optimiser.

## The decision rule

Read the table in the last cell.

* **`ovf512` train acc below ~0.9** → stop. The pipeline, not the recipe, is at
  fault; report this rather than spending GPU time.
* **`f24_*` already fits the training set** → 24 epochs stands and W3-A costs about
  6 GPU-hours.
* **`f60_3e3` clearly beats `f24_3e3`** → set `RECIPE` in the benchmark notebooks
  to 60 epochs; W3-A then costs ~15 GPU-hours and the budget guard splits it over
  sessions.
* **`f24_3e3` clearly beats `f24_1e3`** → 1e-3 was still too small; keep 3e-3.
* **`res30` far above the `f*` rows** → capacity is binding after all, and the
  headline should use the pretrained backbone with the compact CNN as the
  low-capacity end of the scale ladder.
* **`v60_3e3`** is the first honest strict-protocol number — the one the paper is
  about. Read it against `f60_3e3`.

## Cost

About two hours. `cnn_small` at 24 epochs is ~12 min and at 60 epochs ~30 min;
`resnet18` is roughly five times a `cnn_small` epoch; `ovf512` is under a minute.

These configurations carry the recipe tag `rc2`; the earlier, invalid `rc` rows
stay in `results.csv` but nothing here reads them."""),
        code(SETUP),
        writefile("w3_run.py", (SRC / "w3_run.py").read_text(encoding="utf-8")),
        md("""## what runs

| tag | backbone | protocol | recipe | asks |
|---|---|---|---|---|
| `f24_1e3` | cnn_small | clip_random | 24 ep, `[0,1]`, lr 1e-3, warm-up 3 | the recipe as previously written — is 24 epochs enough? |
| `f24_3e3` | cnn_small | clip_random | 24 ep, `[0,1]`, lr 3e-3, warm-up 3 | was 1e-3 itself still too small? |
| `f60_3e3` | cnn_small | clip_random | 60 ep, `[0,1]`, lr 3e-3, warm-up 3 | does the schedule want more epochs? |
| `ovf512` | cnn_small | clip_random | **512 clips**, 200 ep, lr 3e-3 | **can this pipeline memorise at all?** |
| `res30` | resnet18 11.2M | clip_random | 30 ep, `[0,1]`, lr 1e-3, warm-up 3 | capacity, now at a lr that trains |
| `v60_3e3` | cnn_small | vessel_disjoint | 60 ep, `[0,1]`, lr 3e-3, warm-up 3 | the honest strict-protocol number, early |

Every row records `train_acc`, `train_val_gap`, `n_pred_classes` and `lr_last`, so
both underfitting and a stalled schedule are visible in `results.csv`."""),
        code(r'''# wire check: the recipe flags must actually reach the runner. Look for the
# "recipe" and "norm" lines in the header, and the new train_acc column.
cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
       "--data", str(FEAT), "--out", str(WORK / "_probe"),
       "--corpus", "oceanship", "--protocol", "vessel_disjoint", "--seed", "0",
       "--rep", "logmel", "--backbone", "cnn_small", "--smoke", "--num-workers", "0",
       "--norm", "affine", "--lr", "0.001", "--tag", "probe"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
tick("probe done")'''),
        code(drv(300, "rc2", configs, recipe=R)),
        code(REPORT),
        code(r'''# --------------------------------------------------------------------------- #
# the verdict, spelled out
# --------------------------------------------------------------------------- #
import csv
from pathlib import Path

with (RUNS / "results.csv").open(encoding="utf-8", newline="") as fh:
    rows = list(csv.DictReader(fh))
# this notebook's runs carry the rc2 recipe tag; the earlier rc rows were produced
# with a scheduler that never advanced and are not read here
by = {r["run_id"].split("__")[-1]: r for r in rows if "__rc2_" in r["run_id"]}

def g(tag, key, d=float("nan")):
    r = by.get(tag)
    if not r:
        return d
    try:
        return float(r[key])
    except (KeyError, TypeError, ValueError):
        return d

TAGS = ("rc2_ovf512", "rc2_f24_1e3", "rc2_f24_3e3", "rc2_f60_3e3", "rc2_res30", "rc2_v60_3e3")
print(f"  {'config':<12}{'K':>4}{'lr set':>9}{'ep':>4}{'train':>8}{'test':>8}{'gap':>9}{'predK':>7}")
for t in TAGS:
    print(f"  {t.replace('rc2_', ''):<12}{int(g(t, 'n_classes', 0)):>4}"
          f"{g(t, 'lr'):>9.4f}{int(g(t, 'epochs', 0)):>4}"
          f"{g(t, 'train_acc'):>8.4f}{g(t, 'test_acc'):>8.4f}"
          f"{g(t, 'train_val_gap'):>9.4f}{int(g(t, 'n_pred_classes', 0)):>7}")

ovf = g("rc2_ovf512", "train_acc")
print()
print("  decision rule")
print("    ovf512 train < 0.90            -> STOP: the pipeline, not the recipe")
print("    f24_*    fits the train set    -> 24 epochs stands (W3-A ~6 GPU-h)")
print("    f60_3e3  > f24_3e3 clearly     -> set RECIPE to 60 epochs (W3-A ~15 GPU-h)")
print("    f24_3e3  > f24_1e3 clearly     -> keep lr 3e-3")
print("    res30    >> the f*  rows       -> capacity binds; use the pretrained backbone")
print("    v60_3e3                        -> the first honest strict-protocol number")
if ovf == ovf and ovf < 0.90:          # not NaN and below the bar
    print()
    print(f"  !! ovf512 reached only {ovf:.3f} training accuracy on 512 clips.")
    print("     Stop and investigate the input pipeline before any further GPU spend.")
print()
print(f"  schedule check: lr_last should be ~0 for a completed cosine, not lr/large.")
for t in TAGS:
    print(f"    {t.replace('rc2_', ''):<12} lr={g(t, 'lr'):.4f} lr_last={g(t, 'lr_last'):.2e}")
print()
print("  whatever you conclude, copy the winning flag set into RECIPE_MAIN at the top of")
print("  this file's builder and bump the tag in drv() so the old rows are not reused.")'''),
        code(SAVE_A % {"tag": "A2"}),
    ])


# --------------------------------------------------------------------------- #
# W3-A3 — where does the missing accuracy go?  (see PROBE_A below)
# --------------------------------------------------------------------------- #

# W3-A4 - the DeepShip reference point.
#
# W3-A3 settled what the leak is worth: the recording identity IS in the features
# (its 1-NN rate is 180x the chance rate) but it is weak - only 5.68% of test
# clips have their own recording as the nearest training clip - so the 99.44%
# metadata ceiling is largely NOT reachable from the audio.
#
# That leaves the other half of the question: is 0.37 on 12-class OceanShip
# simply what this pipeline does, or is the pipeline weak?  DeepShip is the
# calibration, because it is the corpus the literature quotes: LHK-Net reports
# val 99.06% / test 72.27% on DeepShip's FOUR classes.  If the same recipe
# reproduces ~0.7 there, the pipeline is sound and 12 classes is just harder.
# If it lands near 0.4, the pipeline is the problem and no amount of OceanShip
# tuning will fix it.
# --------------------------------------------------------------------------- #
CEILING = r'''# --------------------------------------------------------------------------- #
# 2. the gate: the metadata ceiling for BOTH corpora, and a wiring check
#
# The ceiling is what split-unit choice is worth.  Reading it next to the
# achieved accuracy is the whole argument, so it is computed for DeepShip too -
# and DeepShip is the corpus that has never actually been run, so the same cell
# does the wiring check.
# --------------------------------------------------------------------------- #
import collections
import pandas as pd

# corpus -> (label column, vessel column, recording column, split join keys)
SPEC = {
    "oceanship": ("label", "mmsi", "rec_key", ["clip_id"]),
    "deepship": ("class_dir", "ship_name", "rec_key", ["rec_key", "clip_idx"]),
}


def load(corpus):
    idx = pd.read_csv(FEAT / corpus / "index.csv")
    for name in (f"{corpus}_splits.csv.gz", f"{corpus}_splits.csv"):
        p = FEAT / "splits" / name
        if p.exists():
            return idx, pd.read_csv(p), p.name
    raise SystemExit(f"no {corpus} splits in the feature dataset")


def ceiling(idx, sp, proto, seed, corpus, key_col):
    """Accuracy of predicting a test clip's majority TRAIN label for its identity."""
    lab, ves, rec, keys = SPEC[corpus]
    sel = sp[(sp.protocol == proto) & (sp.seed == int(seed))]
    m = idx.merge(sel[keys + ["split"]], on=keys, how="inner")
    tr, te = m[m.split == "train"], m[m.split == "test"]
    maj = {}
    for k, v in zip(tr[key_col], tr[lab]):
        maj.setdefault(k, collections.Counter())[v] += 1
    maj = {k: c.most_common(1)[0][0] for k, c in maj.items()}
    cov = sum(1 for k in te[key_col] if k in maj)
    hit = sum(maj[k] == v for k, v in zip(te[key_col], te[lab]) if k in maj)
    base = collections.Counter(te[lab]).most_common(1)[0]
    return (len(tr), len(te), cov / max(len(te), 1), hit / max(len(te), 1),
            base[1] / max(len(te), 1))


print("=" * 88)
print("the metadata ceiling - what the split unit is worth, per corpus")
print("=" * 88)
for corpus in ("deepship", "oceanship"):
    IDX, SP, sname = load(corpus)
    lab, ves, rec, keys = SPEC[corpus]
    print(f"  {corpus:<10} splits={sname}  classes={IDX[lab].nunique()}  "
          f"protocols={sorted(SP.protocol.unique())}  seeds={sorted(SP.seed.unique())}")
    for proto in sorted(SP.protocol.unique()):
        ntr, nte, cov, acc, base = ceiling(IDX, SP, proto, 0, corpus, ves)
        _, _, covr, accr, _ = ceiling(IDX, SP, proto, 0, corpus, rec)
        print(f"      {proto:<15} train {ntr:>6,}  test {nte:>6,}   "
              f"vessel ceiling {100*acc:6.2f}% (cov {100*cov:5.1f}%)   "
              f"recording {100*accr:6.2f}%   best constant {100*base:5.2f}%")
print()
print("  literature anchor, DeepShip 4 classes: LHK-Net val 99.06% / test 72.27%")
tick("ceilings computed")

# wiring check on the corpus that has never been run
for proto in ("clip_random", "temporal"):
    cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
           "--data", str(FEAT), "--out", str(WORK / "_probe_ds"),
           "--corpus", "deepship", "--protocol", proto, "--seed", "0",
           "--rep", "logmel", "--backbone", "cnn_small", "--smoke",
           "--num-workers", "0", "--tag", "probe"]
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
tick("probe done")
'''

COMPARE = r'''# --------------------------------------------------------------------------- #
# 3. DeepShip next to OceanShip, and next to the literature
# --------------------------------------------------------------------------- #
import csv

with (RUNS / "results.csv").open(encoding="utf-8", newline="") as fh:
    ROWS = list(csv.DictReader(fh))


def row_for(rid):
    for r in ROWS:
        if r["run_id"].endswith(rid):
            return r
    return None


def f(r, k, d=float("nan")):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return d


print("=" * 88)
print("DeepShip (4 classes) - the calibration")
print("=" * 88)
print(f"  {'protocol':<16}{'seed':>5}{'test':>9}{'train':>9}{'predK':>7}")
DS = {}
for proto in ("clip_random", "rec_random", "temporal"):
    accs = []
    for s in (0, 1, 2):
        r = row_for(f"deepship__logmel__cnn_small__{proto}__s{s}__ds1")
        if r is None:
            print(f"  {proto:<16}{s:>5}   (not run - re-run the sweep cell to resume)")
            continue
        print(f"  {proto:<16}{s:>5}{f(r, 'test_acc'):>9.4f}{f(r, 'train_acc'):>9.4f}"
              f"{int(f(r, 'n_pred_classes', 0)):>7}")
        accs.append(f(r, "test_acc"))
    DS[proto] = sum(accs) / len(accs) if accs else float("nan")
    if accs:
        print(f"  {'':<16}{'mean':>5}{DS[proto]:>9.4f}")

print()
print("=" * 88)
print("OceanShip (12 classes) - the same recipe")
print("=" * 88)
for proto in ("clip_random", "vessel_disjoint"):
    r = row_for(f"oceanship__logmel__cnn_small__{proto}__s0__ds1")
    if r is None:
        print(f"  {proto:<20} (not run)")
        continue
    print(f"  {proto:<20} test {f(r, 'test_acc'):.4f}   train {f(r, 'train_acc'):.4f}"
          f"   predK {int(f(r, 'n_pred_classes', 0))}")

print()
print("  the reading")
print("    DeepShip ~0.70  -> the pipeline reproduces the literature; the 12-class")
print("                       OceanShip numbers stand as a harder-task result.")
print("    DeepShip ~0.40  -> the pipeline is the problem, not the task; fix the")
print("                       representation before spending more on W3-A.")
print("    clip_random minus temporal on DeepShip is the counterpart of the 9.6 pp")
print("    OceanShip gap, and makes the effect look general rather than corpus-bespoke.")
print()
print("  literature: LHK-Net DeepShip val 99.06% / test 72.27%")
if DS.get("temporal") == DS.get("temporal"):
    near = abs(DS["temporal"] - 0.7227) < 0.10
    print(f"  this run  : DeepShip temporal {DS['temporal']:.4f}   "
          f"({'order of the literature' if near else 'well below the literature'})")
tick("comparison done")
'''


# --------------------------------------------------------------------------- #
# W3-A3 - where does the missing accuracy go?  (cells PROBE_A .. SAVE_PROBE)
#
# W3-A2 showed the model CAN fit: 1.0000 training accuracy on 512 clips, 0.9999
# for resnet18 on the full 37,626.  But its accuracy on the leaky clip-level
# split saturates at ~0.37, while predicting each clip's vessel's majority
# training label - no audio at all - scores 99.44%.  So the question is no
# longer whether the model can learn; it is whether the recording identity that
# the leak depends on is actually in these features.  That costs no GPU.
# --------------------------------------------------------------------------- #
P_DESC = r'''# --------------------------------------------------------------------------- #
# 2. compact descriptors, one per corpus
#
# A 128x313 map is 40,040 numbers per clip; all-pairs distances need something
# smaller.  Mean-pooling to 16x16 keeps the coarse time-frequency shape - which
# is where a recording signature would live - and turns the nearest-neighbour
# search into a single matrix product.
# --------------------------------------------------------------------------- #
import collections
import numpy as np
import pandas as pd

P_CORPUS = {
    # corpus    : label col,  vessel col,  recording col, split join keys, stream?
    #
    # `stream` says whether the index carries a POSITION IN THE AUDIO STREAM.
    # The index column clip_idx is not it: the builder writes it as a row
    # counter inside a write batch, so it is a position only by accident for
    # corpora that are processed recording by recording.  DeepShip qualifies
    # (checked below); OceanShip does not, and pretending otherwise would
    # invent an adjacency that the corpus does not have.
    "deepship":  dict(label="class_dir", vessel="ship_name", rec="rec_key",
                      keys=("rec_key", "clip_idx"), stream=True),
    "oceanship": dict(label="label", vessel="mmsi", rec="rec_key",
                      keys=("clip_id",), stream=False),
}

IDX, DES = {}, {}
for corpus, spec in P_CORPUS.items():
    idx = pd.read_csv(FEAT / corpus / "index.csv")
    IDX[corpus] = idx
    arr = np.load(FEAT / corpus / "logmel.npy", mmap_mode="r")
    n_mels, n_fr = arr.shape[1], arr.shape[2]
    pad = int(np.ceil(n_fr / 16.0) * 16)
    out = np.zeros((len(idx), 256), dtype=np.float32)
    CH = 2048
    for s in range(0, len(idx), CH):
        e = min(s + CH, len(idx))
        x = np.asarray(arr[s:e], dtype=np.float32)
        x = x.reshape(len(x), 16, n_mels // 16, n_fr).mean(2)
        if pad > n_fr:
            x = np.pad(x, ((0, 0), (0, 0), (0, pad - n_fr)), mode="edge")
        out[s:e] = x.reshape(len(x), 16, 16, pad // 16).mean(3).reshape(len(x), -1)
    DES[corpus] = out
    print(f"  {corpus:<10} {tuple(arr.shape)} {arr.dtype} -> {out.shape}   "
          f"{out.min():.1f}..{out.max():.1f}")
    np.save(WORK / f"leakprobe_{corpus}16.npy", out)

# a clip's join key, as a plain string, for both index and split table
def keys_of(corpus: str, frame) -> list:
    k = P_CORPUS[corpus]["keys"]
    if len(k) == 1:
        return frame[k[0]].astype(str).tolist()
    return [f"{a}|{b}" for a, b in zip(*(frame[c].astype(str) for c in k))]


MET = {}
for corpus in P_CORPUS:
    spec = P_CORPUS[corpus]
    ks = keys_of(corpus, IDX[corpus])
    MET[corpus] = {
        "label": dict(zip(ks, IDX[corpus][spec["label"]].astype(str))),
        "vessel": dict(zip(ks, IDX[corpus][spec["vessel"]].astype(str))),
        "rec": dict(zip(ks, IDX[corpus][spec["rec"]].astype(str))),
        "dur": dict(zip(ks, IDX[corpus]["src_dur_s"].astype(float))),
        "row": {k: i for i, k in enumerate(ks)},
    }
    for extra in ("clip_idx",):
        if extra in IDX[corpus]:
            MET[corpus][extra] = dict(
                zip(ks, IDX[corpus][extra].astype(int)))
tick("descriptors built")
'''

P_SPLITS = r'''# --------------------------------------------------------------------------- #
# 3. the splits, and the metadata ceiling on each of them
#
# The ceiling is what the split unit is worth: predict a test clip's majority
# TRAIN label for its own recording / vessel, with no audio at all.  Reading it
# next to the achieved accuracy is the whole argument, so it is computed for
# every protocol of both corpora.
# --------------------------------------------------------------------------- #
def split_table(corpus: str):
    for name in (f"{corpus}_splits.csv.gz", f"{corpus}_splits.csv"):
        p = FEAT / "splits" / name
        if p.exists():
            return pd.read_csv(p)
    raise SystemExit(f"no {corpus} splits in the feature dataset")


SP = {}
for corpus in P_CORPUS:
    sp = split_table(corpus)
    sp = sp.assign(_k=keys_of(corpus, sp))
    SP[corpus] = sp
    print(f"  {corpus:<10} {len(sp):,} rows  protocols={sorted(sp.protocol.unique())}"
          f"  seeds={sorted(sp.seed.unique())}")


def sides(corpus: str, proto: str, seed: int):
    sp = SP[corpus]
    sel = sp[(sp.protocol == proto) & (sp.seed == int(seed))]
    have = MET[corpus]["row"]
    tr = [k for k in sel.loc[sel.split == "train", "_k"] if k in have]
    te = [k for k in sel.loc[sel.split == "test", "_k"] if k in have]
    return tr, te


def ceiling(corpus: str, tr: list, te: list, field: str):
    lab, who = MET[corpus]["label"], MET[corpus][field]
    maj = {}
    for k in tr:
        maj.setdefault(who[k], collections.Counter())[lab[k]] += 1
    maj = {k: v.most_common(1)[0][0] for k, v in maj.items()}
    cov = sum(1 for k in te if who[k] in maj)
    hit = sum(1 for k in te if maj.get(who[k]) == lab[k])
    return cov / max(len(te), 1), hit / max(len(te), 1)


PROTOCOLS = {c: [p for p in ("clip_random", "rec_random", "temporal",
                            "vessel_disjoint") if p in set(SP[c].protocol)]
             for c in P_CORPUS}
print()
print("=" * 88)
print("the metadata ceiling, on the very splits the models are scored on")
print("=" * 88)
CEIL = {}
for corpus in P_CORPUS:
    print(f"  {corpus}")
    for proto in PROTOCOLS[corpus]:
        tr, te = sides(corpus, proto, 0)
        if not te:
            continue
        cv, av = ceiling(corpus, tr, te, "vessel")
        cr, ar = ceiling(corpus, tr, te, "rec")
        maj = collections.Counter(MET[corpus]["label"][k] for k in tr).most_common(1)[0]
        CEIL[(corpus, proto)] = (av, ar)
        print(f"      {proto:<17} train {len(tr):>6,}  test {len(te):>6,}   "
              f"vessel {100*av:6.2f}% (cov {100*cv:5.1f}%)   "
              f"recording {100*ar:6.2f}%   best constant {100*maj[1]/len(tr):5.2f}%")
tick("ceilings computed")
'''

P_KNN = r'''# --------------------------------------------------------------------------- #
# 4. is the identity IN the features?  the same measurement, both corpora
#
# For every test clip, find its single nearest training clip in the 256-d
# descriptor and ask three questions: is the label right (is the leak usable),
# and does the neighbour share the recording / the vessel (is identity present)?
#
# Every rate is printed against its own null - the probability that the same
# event happens for a training clip picked at random.  OceanShip has 10,168
# recordings, so 1% is enormous; DeepShip has 609, so it is not.
# --------------------------------------------------------------------------- #
def distance_matrix(corpus: str, tr: list, te: list):
    rtr = np.array([MET[corpus]["row"][k] for k in tr])
    rte = np.array([MET[corpus]["row"][k] for k in te])
    T = DES[corpus][rtr]
    X = DES[corpus][rte]
    mu, sd = T.mean(0), T.std(0) + 1e-6
    T = np.ascontiguousarray((T - mu) / sd, dtype=np.float32)
    X = np.ascontiguousarray((X - mu) / sd, dtype=np.float32)
    tn = (T ** 2).sum(1)[None, :]
    nn = np.empty(len(X), dtype=np.int32)
    nnd = np.empty(len(X), dtype=np.float32)
    B = 512
    for s in range(0, len(X), B):
        e = min(s + B, len(X))
        d2 = (X[s:e] ** 2).sum(1)[:, None] + tn - 2.0 * (X[s:e] @ T.T)
        j = np.argmin(d2, 1)
        nn[s:e] = j
        nnd[s:e] = d2[np.arange(e - s), j]
    return T, X, nn, nnd


def same_null(corpus: str, tr: list, te: list, field: str) -> float:
    who = MET[corpus][field]
    cnt = collections.Counter(who[k] for k in tr)
    return sum(cnt[who[k]] for k in te) / len(te) / len(tr)


KNN = {}
print("=" * 88)
print("1-NN: does a test clip's nearest TRAINING clip share its identity?")
print("=" * 88)
for corpus in P_CORPUS:
    for proto in PROTOCOLS[corpus]:
        tr, te = sides(corpus, proto, 0)
        if not te:
            continue
        T, X, nn, nnd = distance_matrix(corpus, tr, te)
        lab, rec, ves = (MET[corpus]["label"], MET[corpus]["rec"],
                         MET[corpus]["vessel"])
        hit = np.mean([lab[tr[j]] == lab[c] for j, c in zip(nn, te)])
        sr = np.mean([rec[tr[j]] == rec[c] for j, c in zip(nn, te)])
        sv = np.mean([ves[tr[j]] == ves[c] for j, c in zip(nn, te)])
        nr = same_null(corpus, tr, te, "rec")
        nv = same_null(corpus, tr, te, "vessel")
        best = collections.Counter(lab[k] for k in tr).most_common(1)[0][1] / len(tr)
        rec_ = dict(proto=proto, seed=0, label_acc=float(hit), same_rec=float(sr),
                    same_ves=float(sv), null_rec=float(nr), null_ves=float(nv),
                    best_constant=float(best), n_train=len(tr), n_test=len(te))
        KNN[(corpus, proto)] = rec_
        print(f"  [{corpus} {proto} s0]  train {len(tr):,}  test {len(te):,}")
        print(f"      1-NN label accuracy        {100*hit:6.2f}%   "
              f"(best constant {100*best:5.2f}%)")
        print(f"      nearest neighbour shares the RECORDING {100*sr:6.2f}%   "
              f"(chance {100*nr:8.4f}%  ->  {sr/max(nr,1e-12):8.1f}x)")
        print(f"      nearest neighbour shares the VESSEL    {100*sv:6.2f}%   "
              f"(chance {100*nv:8.4f}%  ->  {sv/max(nv,1e-12):8.1f}x)")
tick("1-NN done")
'''

P_ADJ = r'''# --------------------------------------------------------------------------- #
# 5. WHY one corpus leaks and the other does not: adjacency in the raw stream
#
# DeepShip tiles each long recording into back-to-back 5 s windows, so a test
# clip's stream neighbours are near-duplicates of it.  OceanShip cuts short
# annotated segments out of a continuous hydrophone stream at arbitrary times,
# so two clips of one recording are unrelated audio.
#
# This cell checks that directly on the audio: among the test clips whose
# nearest training clip IS from their own recording, how far away in the stream
# is it?  If the answer is "one window", the model is doing retrieval.
# --------------------------------------------------------------------------- #
print("=" * 88)
print("where the nearest same-recording neighbour sits in the stream")
print("=" * 88)
for corpus in P_CORPUS:
    if not P_CORPUS[corpus]["stream"]:
        print(f"  {corpus:<10} the index carries no point in the audio stream -")
        print(f"  {'':<10} clip_idx in the feature index is a build counter, and the")
        print(f"  {'':<10} trailing integer of clip_id is not chronological either")
        print(f"  {'':<10} (median gap 0 between consecutive ids within a recording).")
        print(f"  {'':<10} This absence IS the finding: clips of one recording are")
        print(f"  {'':<10} arbitrary moments, so nothing about them is a neighbour.")
        print()
        continue
    ci = MET[corpus]["clip_idx"]
    # wiring check: for a corpus that IS tiled, clip_idx must run 0..N-1 per rec
    byrec = collections.defaultdict(list)
    for k, i in ci.items():
        byrec[MET[corpus]["rec"][k]].append(i)
    tiled = sum(1 for v in byrec.values() if sorted(v) == list(range(len(v))))
    print(f"  {corpus:<10} index clip_idx runs 0..N-1 per recording for "
          f"{tiled}/{len(byrec)} recordings")
    for proto in PROTOCOLS[corpus]:
        tr, te = sides(corpus, proto, 0)
        if not te:
            continue
        rec = MET[corpus]["rec"]
        _, _, nn, _ = distance_matrix(corpus, tr, te)
        delta = [abs(ci[c] - ci[tr[j]]) for j, c in zip(nn, te)
                 if rec[tr[j]] == rec[c]]
        if not delta:
            print(f"      {proto:<17} no same-recording nearest neighbour at all "
                  f"({len(te):,} test clips)")
            continue
        d = np.array(delta)
        print(f"      {proto:<17} {len(d):,} test clips with a same-recording "
              f"nearest neighbour")
        print(f"          |delta clip_idx|  = 1 {100*np.mean(d == 1):6.1f}%   "
              f"<= 2 {100*np.mean(d <= 2):6.1f}%   <= 5 {100*np.mean(d <= 5):6.1f}%"
              f"   median {np.median(d):.0f}")
        print(f"          a neighbour one window away is the SAME audio event")
    print()
tick("adjacency done")
'''

P_REST = r'''# --------------------------------------------------------------------------- #
# 6. the two explanations that are not the model
#
# (a) separation: are two clips of one recording closer than two clips of
#     different recordings?  ratio << 1 means identity is present.
# (b) duration: the clips were zero-padded to 5 s and the padding is not evenly
#     spread across classes, so a model can read the silence.
# (c) a linear probe on the same descriptor, as the cheap supervised baseline.
# --------------------------------------------------------------------------- #
for corpus in P_CORPUS:
    for proto in PROTOCOLS[corpus]:
        tr, te = sides(corpus, proto, 0)
        if not te:
            continue
        T, X, nn, nnd = distance_matrix(corpus, tr, te)
        rec = MET[corpus]["rec"]
        same = np.array([rec[tr[j]] == rec[c] for j, c in zip(nn, te)])
        if same.any() and (~same).any():
            print(f"  [{corpus} {proto}] nearest-neighbour distance, normalised")
            print(f"      same recording     median {nnd[same].mean():.1f}")
            print(f"      different record.  median {nnd[~same].mean():.1f}")
            print(f"      ratio {nnd[same].mean()/nnd[~same].mean():.3f}   "
                  f"(marginally closer -> identity present)")
        else:
            print(f"  [{corpus} {proto}] no same-recording neighbours at all: "
                  f"the protocol removed them")
tick("separability done")
'''

P_DURLIN = r'''# --------------------------------------------------------------------------- #
# 7. the scalar baseline, and a linear probe on the same descriptor
#
# Not every baseline needs audio.  DeepShip's src_dur_s is the SOURCE FILE's
# length, and the builder cuts one file into its 5 s windows - so every clip of a
# recording carries the same number, and one metadata column is then a recording
# fingerprint.  OceanShip's src_dur_s is the clip's OWN length, which varies
# inside a recording, so there is nothing to fingerprint.
#
# Both readings matter: the 1-NN asks whether duration points at the recording,
# the histogram asks whether it points at the class.
# --------------------------------------------------------------------------- #
DUR1NN = {}
for corpus in P_CORPUS:
    dur, lab = MET[corpus]["dur"], MET[corpus]["label"]
    for proto in PROTOCOLS[corpus]:
        tr, te = sides(corpus, proto, 0)
        if not tr or not te:
            continue
        cl = np.array([dur[k] for k in tr], dtype=np.float32)
        xl = np.array([dur[k] for k in te], dtype=np.float32)
        yl = np.array([lab[k] for k in tr])
        pred = np.empty(len(xl), dtype=object)
        for s0 in range(0, len(xl), 512):
            e = min(s0 + 512, len(xl))
            j = np.argmin(np.abs(xl[s0:e, None] - cl[None, :]), 1)
            pred[s0:e] = yl[j]
        DUR1NN[(corpus, proto)] = float((pred == np.array([lab[k] for k in te])).mean())

print("=" * 88)
print("1-NN on src_dur_s alone - one metadata column, no audio, no training")
print("=" * 88)
for corpus in P_CORPUS:
    lab = MET[corpus]["label"]
    for proto in PROTOCOLS[corpus]:
        if (corpus, proto) not in DUR1NN:
            continue
        tr, te = sides(corpus, proto, 0)
        best = collections.Counter(lab[k] for k in tr).most_common(1)[0][1] / len(tr)
        print(f"  {corpus:<10} {proto:<17} {100*DUR1NN[(corpus, proto)]:6.2f}%"
              f"   (best constant {100*best:5.2f}%)")
print()
print("  duration histogram -> majority label  (does duration carry the CLASS?)")
for corpus in P_CORPUS:
    proto = PROTOCOLS[corpus][0]
    tr, te = sides(corpus, proto, 0)
    lab, dur = MET[corpus]["label"], MET[corpus]["dur"]
    cl = np.array([dur[k] for k in tr], dtype=np.float32)
    edges = np.quantile(cl, np.linspace(0, 1, 21))
    edges[-1] += 1e-6
    maj = {}
    for b in range(20):
        sel = [tr[i] for i in np.where((cl >= edges[b]) & (cl < edges[b + 1]))[0]]
        if sel:
            maj[b] = collections.Counter(lab[c] for c in sel).most_common(1)[0][0]
    hit = sum(maj.get(int(np.clip(np.searchsorted(edges, dur[c]) - 1, 0, 19))) == lab[c]
              for c in te)
    print(f"    {corpus:<10} {proto:<17} {100*hit/len(te):6.2f}%")
print()

# --------------------------------------------------------------------------- #
# 8. the supervised floor on the same descriptor: nearest class mean, then a
#    ridge-regularised linear probe (closed form, so no GPU is involved)
# --------------------------------------------------------------------------- #
for corpus in P_CORPUS:
    proto = PROTOCOLS[corpus][0]
    tr, te = sides(corpus, proto, 0)
    lab = MET[corpus]["label"]
    rows_tr = np.array([MET[corpus]["row"][k] for k in tr])
    rows_te = np.array([MET[corpus]["row"][k] for k in te])
    # NB: not T0 - that name belongs to the timer set up in the setup cell
    Td, Xd = DES[corpus][rows_tr], DES[corpus][rows_te]
    mu, sd = Td.mean(0), Td.std(0) + 1e-6
    T = np.ascontiguousarray((Td - mu) / sd, dtype=np.float32)
    X = np.ascontiguousarray((Xd - mu) / sd, dtype=np.float32)
    classes = sorted({lab[c] for c in tr})
    ci = {k: i for i, k in enumerate(classes)}
    Y = np.array([ci[lab[c]] for c in tr])
    Xy = np.array([ci.get(lab[c], -1) for c in te])
    M = np.stack([T[Y == k].mean(0) for k in range(len(classes))])
    ncm = np.mean(np.argmin(((X[:, None, :] - M[None, :, :]) ** 2).sum(2), 1) == Xy)
    Yh = np.eye(len(classes), dtype=np.float32)[Y]
    W = np.linalg.solve(T.T @ T + 1e-2 * np.eye(T.shape[1], dtype=np.float32),
                        T.T @ Yh)
    lin = np.mean((X @ W).argmax(1) == Xy)
    print(f"  [{corpus} {proto}]  nearest class mean (256-d) {100*ncm:6.2f}%   "
          f"linear probe (ridge) {100*lin:6.2f}%   ({len(classes)} classes)")
tick("scalar + linear probe done")
'''

P_VERDICT = r'''# --------------------------------------------------------------------------- #
# 9. the table this whole notebook exists to produce
#
# Model accuracies are entered by hand from the run logs (results.csv on Kaggle
# keeps them; this notebook is CPU-only and deliberately reads no runs).  Update
# them when a sweep is re-run - the point is the ORDER, not the third decimal.
# --------------------------------------------------------------------------- #
MODEL = {
    # (corpus, protocol): test accuracy of logmel/cnn_small at the same recipe
    ("deepship", "clip_random"): 0.9219,
    ("deepship", "rec_random"): 0.6842,
    ("deepship", "temporal"): 0.6182,
    ("oceanship", "clip_random"): 0.3649,
    ("oceanship", "vessel_disjoint"): 0.2926,
    # OceanShip rec_random has not been run at this recipe: it prints as "-"
    # rather than being guessed.  W3-A fills it in for all five seeds.
}

print("=" * 88)
print("the money table: what the split hands over, and what the model converts")
print("=" * 88)
print()
print(f"  {'corpus':<11}{'protocol':<17}{'meta ceiling':>13}{'1 scalar':>10}"
      f"{'1-NN label':>12}{'1-NN same rec':>14}{'chance':>9}{'model':>8}")
print("  " + "-" * 92)
for corpus in P_CORPUS:
    for proto in PROTOCOLS[corpus]:
        if (corpus, proto) not in KNN:
            continue
        k = KNN[(corpus, proto)]
        m = MODEL.get((corpus, proto))
        s = DUR1NN.get((corpus, proto))
        s_txt = f"{100*s:>9.2f}%" if s is not None else f"{'-':>10}"
        m_txt = f"{m:>8.4f}" if m is not None else f"{'-':>8}"
        print(f"  {corpus:<11}{proto:<17}{100*CEIL[(corpus, proto)][1]:>12.2f}%"
              f"{s_txt}{100*k['label_acc']:>12.2f}%{100*k['same_rec']:>13.2f}%"
              f"{100*k['null_rec']:>8.3f}%{m_txt}")
    print()

print("  how to read the two blocks")
print()
print("    deepship, clip_random   every clip is one of ~51 back-to-back 5 s")
print("                            windows tiled out of a single file, so this")
print("                            split leaves ~90% of test clips with a")
print("                            near-duplicate in train.  ONE metadata column")
print("                            (source-file length) already gets 78% with no")
print("                            audio and no training; the full 256-d")
print("                            descriptor gets 83%; the trained model 92%.")
print("                            That ordering is the whole argument: the")
print("                            headline number is a retrieval task.")
print()
print("    deepship, rec_/temporal  the scalar collapses to the best-constant")
print("                            level, and so does the same-recording rate -")
print("                            which is what a split is supposed to do.")
print()
print("    oceanship, *            the recording ceiling is 90.8% on clip_random")
print("                            and the vessel ceiling 99.5%, yet the 1-NN")
print("                            same-recording rate is a few percent, and the")
print("                            scalar is no better than a constant.  Clips of")
print("                            one recording are arbitrary moments of a")
print("                            hydrophone stream, so the split hands over an")
print("                            identity the ACOUSTICS never expose.")
print()
print("  which is why the same code, model and recipe produce a 30.4 pp gap on one")
print("  corpus and a 7.2 pp gap on the other.  'We used a random split' is not a")
print("  specification: the size of the leak is set by how the corpus cuts its")
print("  recordings into clips, and no dataset paper states it.")
tick("verdict done")
'''


def nb_w3a4():
    """The DeepShip reference point: calibrate the pipeline against the literature."""
    R = ["--epochs", "24", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]
    configs = []
    for proto in ("clip_random", "rec_random", "temporal"):
        for seed in (0, 1, 2):
            configs.append({"corpus": "deepship", "rep": "logmel", "backbone": "cnn_small",
                            "protocol": proto, "seed": seed})
    # the OceanShip reference points at the same recipe, for the side-by-side
    for proto in ("clip_random", "vessel_disjoint"):
        configs.append({"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
                        "protocol": proto, "seed": 0})
    return notebook([
        md("""# W3-A4 — the DeepShip reference point

**Run this before spending anything more on OceanShip.** About two hours of GPU.

## The problem this notebook exists to solve

W3-A3 answered the leakage question and left a calibration question behind.

**What W3-A3 found.** The metadata ceiling on `clip_random` is **99.44%** — but
that number is metadata, not acoustics. The nearest-neighbour probe put the
acoustic version of it at **5.68%** of test clips whose single nearest training
clip is from their own recording. Against a properly computed null of
**0.0315%** that is 180x chance, so the identity *is* in the features — but it is
weak and diffuse, not the fingerprint that a 99% clip-level exploit needs.
Every cheap probe lands in the same band: 1-NN 25.48%, ridge on 256 dims 27.44%,
duration alone 22.81%, best constant 22.05%. The trained CNN gets 36.79%.

**So the numbers are internally consistent — and still 40 points below the
literature.** LHK-Net reports DeepShip val 99.06% / test **72.27%**. Either the
pipeline is weak, or 12 classes is a much harder task than 4 and nobody has said
so. Those two explanations need different papers, and this notebook separates
them, because **DeepShip is the corpus the quoted numbers come from**.

## What it runs

| corpus | classes | protocols | seeds | runs |
|---|---|---|---|---|
| **DeepShip** | 4 | `clip_random`, `rec_random`, `temporal` | 0, 1, 2 | 9 |
| OceanShip | 12 | `clip_random`, `vessel_disjoint` | 0 | 2 |

Same recipe as the `rc2` recipe check: `cnn_small`, logmel, 24 epochs, lr 3e-3,
3-epoch warm-up, `[0,1]` input. The recipe tag here is `ds1` so these rows can
never be confused with the `rec2` benchmark rows.

The gate cell prints the **metadata ceiling for both corpora** — the first time
DeepShip's leak is quantified the same way as OceanShip's — and runs the wiring
check on DeepShip, which has never been executed before.

## How to read the result

* **DeepShip ≈ 0.70** → the pipeline reproduces the literature. The 12-class
  OceanShip result is a harder-task result and stands as it is.
* **DeepShip ≈ 0.40** → the pipeline is the problem, not the task, and no
  OceanShip tuning will fix it. Fix the representation first.
* Whichever way it goes, **`clip_random` minus `temporal` on DeepShip** is the
  counterpart of the 9.6 pp OceanShip gap. The paper needs both to claim the
  effect is general rather than bespoke to one corpus.

Cost: DeepShip rows are ~33,770, about the size of OceanShip's, so ~11 runs at
24 epochs is roughly two hours — one session."""),
        code(SETUP),
        writefile("w3_run.py", (SRC / "w3_run.py").read_text(encoding="utf-8")),
        md("""## the gate

Before the sweep: the ceiling for both corpora, and the first real execution of
the DeepShip path (it has only ever been exercised on the miniature fixture)."""),
        code(CEILING),
        code(drv(240, 'ds1', configs, recipe=R)),
        code(REPORT),
        code(COMPARE),
        code(SAVE_A % {"tag": "A4"}),
    ])


def nb_w3a3():
    """The leak probe, on both corpora.  CPU only, no GPU budget."""
    return notebook([
        md("""# W3-A3 — what the split actually hands over

**CPU only, a few minutes, no GPU.** The measurement that turns the leakage
claim from "there is a leak" into "here is the leak, and here is why its size
depends on the corpus".

## Why it now runs on two corpora

W3-A4 ran the same recipe on DeepShip (4 classes) and OceanShip (12) and got
**0.9219** and **0.3649** on their clip-random splits. Same code, same model,
same hyper-parameters, two corpora that are both ONC Strait of Georgia
hydrophone data. The difference cannot be the split *name*.

`src/w1_05_recording_structure.py` (pure metadata, no audio) found the reason:

| | DeepShip | OceanShip |
|---|---|---|
| recordings | 609 | 10,168 |
| clips/recording, median | **51** | **3** |
| clip index is exactly 0..N-1 | **609/609** | no such index |
| (N x 5 s) / recording length | **0.994** | n/a |
| test clips with an adjacent clip in train (clip_random) | **89.6%** | 5.68% (measured, below) |

DeepShip tiles each long recording into back-to-back 5 s windows, so clip *i* and
clip *i+1* are the same audio event. OceanShip cuts short annotated segments out
of a continuous stream at arbitrary times, so two clips of one recording are
unrelated. A random split is therefore catastrophic on one corpus and nearly
inert on the other — and no dataset paper says which kind it published.

## What this notebook measures, per corpus

| cell | question |
|---|---|
| 3 | the metadata ceiling on the very splits the models were scored on |
| 4 | **1-NN**: nearest training clip — right label? same recording? same vessel? |
| 5 | among same-recording neighbours, **how far away in the stream** are they |
| 6 | are two clips of one recording closer together than two of different ones |
| 7 | duration alone, nearest class mean, linear probe |
| 8 | the money table |

Cell 8 prints the table this project now hangs on."""),
        code(SETUP),
        md("""## 1. descriptors, both corpora

Mean-pool each `(128, 313)` log-mel map to `16 x 16 = 256` numbers: coarse
enough that an all-pairs search is one matrix product, fine enough to keep the
time-frequency shape a recording signature would live in."""),
        code(P_DESC),
        md("""## 2. the ceiling on each split

The ceiling is what the *split unit* is worth — predicting a test clip's
majority **training** label for its own recording or vessel, with no audio at
all. It is the target a leak-exploiting model would chase, and it has to be read
per corpus, because DeepShip's vessel ceiling is not OceanShip's."""),
        code(P_SPLITS),
        md("""## 3. is the identity in the features?

The decisive measurement. Each rate is printed next to its own null — the chance
that a training clip picked at random would show the same property. OceanShip has
10,168 recordings, so a 1% rate is already enormous; DeepShip has 609."""),
        code(P_KNN),
        md("""## 4. why: where the nearest neighbour sits in the stream

If the nearest same-recording training clip is the **adjacent 5 s window**, the
network is not recognising ship types at all — it is retrieving a near-duplicate.
This cell measures that distance directly, and reports the case where it cannot
be measured at all as the finding it is."""),
        code(P_ADJ),
        md("""## 5. the other two explanations

Separation first (are same-recording pairs actually closer?), then duration —
the clips were zero-padded to 5 s and the padding is not evenly spread across
classes, so a model can read the silence."""),
        code(P_REST),
        code(P_DURLIN),
        code(P_VERDICT),
        md("""## 6. keep the descriptors

`/kaggle/working/leakprobe_<corpus>16.npy` are 256-d for every clip of both
corpora — a few tens of MB. Create a dataset from the output if you want to
re-run this analysis without rebuilding them."""),
        code(SAVE_PROBE),
    ])


NOTEBOOKS = {
    "kaggle_w3a4_deepship_reference.ipynb": nb_w3a4,
    "kaggle_w3a3_leak_probe.ipynb": nb_w3a3,
    "kaggle_w3a2_recipe_check.ipynb": nb_w3a2,
    "kaggle_w3a_headline.ipynb": nb_w3a,
    "kaggle_w3b_baselines_scale.ipynb": nb_w3b,
    "kaggle_w3c_diagnostics.ipynb": nb_w3c,
    "kaggle_w3d_domain_generalization.ipynb": nb_w3d,
    "kaggle_w3e_aggregate.ipynb": nb_w3e,
}


def main() -> int:
    for name, build in NOTEBOOKS.items():
        nb = build()
        p = OUT / name
        p.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
        print(f"  {name:<42} {len(nb['cells']):>3} cells ({n_code} code)  {p.stat().st_size:>8,} B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

