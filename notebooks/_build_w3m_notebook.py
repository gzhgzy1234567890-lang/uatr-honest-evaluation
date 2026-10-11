#!/usr/bin/env python3
"""Generate the W3-M pair of Kaggle notebooks.

kaggle_w3e2_variance.ipynb      CPU  - variance decomposition on ALL accumulated draws
kaggle_w3m_deepship_vessel.ipynb GPU - DeepShip TRUE vessel-disjoint (10 draws)

Both embed their src/ scripts byte-identically (asserted), same discipline as
_build_w3_notebooks.py.

Usage:  python notebooks/_build_w3m_notebook.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from _build_w3_notebooks import (  # noqa: E402
    DRIVER, REPORT, SETUP, SRC, code, md, notebook, writefile,
)

OUT = ROOT / "notebooks"

RUNDIR = r'''# --------------------------------------------------------------------------- #
# this notebook writes to its OWN run directory and its OWN split file:
#   - w3m-runs/  so the w3e aggregation picks it up as one more prediction root
#     without mixing tags with the OceanShip sweep
#   - splits/deepship_splits.csv.gz under /kaggle/working -- the frozen feature
#     dataset is read-only, and the vessel_disjoint protocol does not exist in it
# --------------------------------------------------------------------------- #
RUNS = WORK / "w3m-runs"
RUNS.mkdir(parents=True, exist_ok=True)
SPLIT_OUT = WORK / "splits" / "deepship_splits.csv.gz"
SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
tick(f"run dir   : {RUNS}")
tick(f"split out : {SPLIT_OUT}")
'''

BUILD_SPLITS = r'''# --------------------------------------------------------------------------- #
# build the extended DeepShip split file (frozen rows + vessel_disjoint x 10)
#
# preconditions printed explicitly so a failure on Kaggle is self-diagnosing:
#   - FEAT/deepship/index.csv must carry a ship_name column (the vessel identity)
#   - the frozen deepship_splits.csv.gz is copied through untouched if present
# --------------------------------------------------------------------------- #
idx = FEAT / "deepship" / "index.csv"
import csv as _csv
with open(idx, encoding="utf-8", newline="") as fh:
    cols = _csv.DictReader(fh).fieldnames
print(f"deepship index columns: {cols}")
if "ship_name" not in cols:
    raise SystemExit("deepship/index.csv has no ship_name column -- vessel grouping "
                     "is impossible; check the feature dataset version")

base = FEAT / "splits" / "deepship_splits.csv.gz"
sys.path.insert(0, str(SCRIPTS))
import w3_run
base = w3_run.any_text_file([base])          # ingest may expand x.csv.gz -> x.csv
base_arg = ["--base", str(base)] if base else []
print(f"frozen split file : {base}  (found: {base is not None})")

cmd = [sys.executable, str(SCRIPTS / "w1_02b_deepship_vessel_splits.py"),
       "--index", str(idx), "--out", str(SPLIT_OUT), "--n-seeds", "10"] + base_arg
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
assert SPLIT_OUT.exists(), "split builder did not write the output file"
tick("extended split file ready")
'''

PROBE = r'''# --------------------------------------------------------------------------- #
# wire check -- one minute of compute before committing three hours
# --------------------------------------------------------------------------- #
cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
       "--data", str(FEAT), "--out", str(WORK / "_probe"),
       "--corpus", "deepship", "--protocol", "vessel_disjoint", "--seed", "0",
       "--rep", "logmel", "--backbone", "cnn_small", "--smoke", "--num-workers", "0",
       "--split-file", str(SPLIT_OUT), "--tag", "probe"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print()
print("  if that printed rows=train/val/test, vessel-overlap 0.0% and a leak rate,")
print("  the vessel-disjoint DeepShip wiring is correct.")
tick("probe done")'''


def driver_with_split(budget: int, tag: str, configs, recipe=None) -> str:
    """DRIVER template + --split-file appended to BASE (anchor-checked splice)."""
    anchor = 'BASE = ["--data", str(FEAT), "--out", str(RUNS), "--num-workers", "2"]'
    assert anchor in DRIVER, "DRIVER anchor drifted -- re-inspect _build_w3_notebooks.py"
    text = DRIVER.replace(
        anchor,
        anchor + ' + ["--split-file", str(SPLIT_OUT)]   # vessel_disjoint lives outside the frozen dataset',
    )
    recipe = recipe or ["--epochs", "24", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]
    return text % {"budget": budget, "recipe_tag": tag,
                   "recipe": repr(recipe), "configs": repr(configs)}


def build_w3m() -> None:
    configs = [{"corpus": "deepship", "rep": "logmel", "backbone": "cnn_small",
                "protocol": "vessel_disjoint", "seed": s} for s in range(10)]
    nb = notebook([
        md("""# W3-M — DeepShip TRUE vessel-disjoint (the missing third leg)

The paper's DeepShip honest protocol is *temporal*, which still shares
40–60% of ships across the split. This notebook runs the protocol the
OceanShip ladder uses — **no ship on both sides** — on DeepShip, for 10
independent draws:

| | |
|---|---|
| corpus | DeepShip (4 classes, 242 vessels, 609 recordings) |
| model / recipe | `cnn_small`, the frozen `rec2` recipe (24 epochs, lr 3e-3) |
| protocol | `vessel_disjoint`, grouped by `ship_name`, stratified per class |
| draws | 10 (seeds 0–9; seeds 0–4 pair with the existing clip/rec/temporal draws) |

**What it adds to the paper** — the fourth row of the DeepShip ladder
(Table II / Fig. 2): does DeepShip, whose recordings are *not* cut into
near-duplicate windows the way OceanShip's are, also collapse under
identity-disjoint evaluation — or does it stay flat, as the null control
predicts?

**Attach**: `under_water_features` (the feature dataset). Nothing else —
the extended split file is generated in-cell from `deepship/index.csv`.

**Accelerator**: any GPU. 10 runs ≈ 2.5–3.5 h; the driver stops cleanly at
the budget and resumes on re-run.

**After it finishes** — Output tab → Create Dataset (e.g. `w3m-data`), so
`w3e_bootstrap` and `w3e2_variance` pick the runs up."""),
        code(SETUP),
        code(RUNDIR),
        writefile("w3_run.py", (SRC / "w3_run.py").read_text(encoding="utf-8")),
        writefile("w1_02b_deepship_vessel_splits.py",
                  (SRC / "w1_02b_deepship_vessel_splits.py").read_text(encoding="utf-8")),
        code(BUILD_SPLITS),
        code(PROBE),
        md("""## the sweep

10 × (DeepShip, log-mel, `cnn_small`, `vessel_disjoint`, seeds 0–9).
Every run's leak check must print **vessel overlap 0%** — that print is the
protocol's whole point."""),
        code(driver_with_split(240, "rec2", configs)),
        code(REPORT),
        md("""## next

1. Output tab → **Create Dataset** (suggest `w3m-data`).
2. Re-run `w3e_bootstrap` and `w3e2_variance` with this dataset attached —
   the DeepShip ladder row and its CI update automatically."""),
    ])
    path = OUT / "kaggle_w3m_deepship_vessel.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path}  ({path.stat().st_size:,} bytes, {len(nb['cells'])} cells)")


E2_ENV = r'''# --------------------------------------------------------------------------- #
# W3-E2 runs on CPU and needs no feature dataset -- only the predictions folders
# your earlier notebooks saved as datasets (W3_data, W3K_data, w3k2-data, w3m-data…)
# --------------------------------------------------------------------------- #
import os, subprocess, sys, time
from pathlib import Path

INPUT = Path("/kaggle/input")
OUT = Path("/kaggle/working")
(OUT / "scripts").mkdir(parents=True, exist_ok=True)
T0 = time.time()
def tick(msg):
    print(f"  [{time.time() - T0:6.1f}s] {msg}", flush=True)

print("mounted datasets:")
for p in sorted(INPUT.glob("*")):
    if p.is_dir():
        print(f"  {p}")
tick("env ready")'''

E2_RUN = r'''# --------------------------------------------------------------------------- #
# run the decomposition.  --min-clips 50 keeps fixture-sized runs out of the
# paper numbers; lower it only if you are testing on a tiny fixture.
# --------------------------------------------------------------------------- #
cmd = [sys.executable, "/kaggle/working/scripts/w3e2_variance_decomposition.py",
       "--input-root", "/kaggle/input", "--out", "/kaggle/working",
       "--min-clips", "50"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
tick("done -- w3e2_variance.json is in the Output tab")'''


def build_w3e2() -> None:
    nb = notebook([
        md("""# W3-E2 — variance decomposition on ALL draws (CPU, minutes)

The paper's headline "**85% of the vessel-disjoint draw variance is test-set
class composition**" was computed on the first 5 draws. You now have ~25
independent vessel draws (w3-data, w3k-data, w3k2-data). This notebook
re-runs the decomposition on **every draw that exists** — no GPU, no
retraining: it reads each run's per-clip predictions and asks, per protocol,
how much of the draw-to-draw accuracy variance a linear model on the test
set's class composition explains.

| output | what it is |
|---|---|
| per-cell table | n draws, total SD, **composition-explained %**, permutation p |
| top-class table | which classes' shares correlate with accuracy (the mechanism) |
| `w3e2_variance.json` | everything above, machine-readable |

Honesty guard: cells with fewer draws than predictors (e.g. `clip_random`,
n=5, 12 classes) are reported **n/a** — a saturated regression would fake
R² = 1.

**Attach**: the dataset(s) holding your `predictions/` folders.

**After it finishes** — Output tab → Create Dataset, then paste the two
tables back; the paper's Table IV gets a 25-draw column."""),
        code(E2_ENV),
        writefile("w3e2_variance_decomposition.py",
                  (SRC / "w3e2_variance_decomposition.py").read_text(encoding="utf-8")),
        code(E2_RUN),
        md("""## next

Paste both printed tables back into the working session. If the 25-draw
explained share confirms ~85% (or higher), the paper's headline survives its
strongest test; if it moves, we update Table IV and the abstract number
together — never one without the other."""),
    ])
    path = OUT / "kaggle_w3e2_variance.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path}  ({path.stat().st_size:,} bytes, {len(nb['cells'])} cells)")


def build_w3n() -> None:
    """W3-N — 20 more OceanShip random vessel-disjoint draws (seeds 5-24).

    Review Major 1: the 85% composition share rests on five pre-registered
    draws; a 25-draw same-scheme pool makes the independent re-estimate valid
    (n=25 > 11 predictors + 1).
    """
    configs = [{"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
                "protocol": "vessel_disjoint", "seed": s} for s in range(5, 25)]
    nb = notebook([
        md("""# W3-N — 20 more OceanShip random vessel-disjoint draws (seeds 5–24)

The paper's headline "**85% of the honest-protocol draw variance is test-set
class composition**" is estimated on the five pre-registered draws. This
notebook extends the *same scheme* (plain random vessel-disjoint,
`cnn_small`, log-mel, `rec2` recipe) to 25 draws, so the decomposition can be
re-estimated independently: n=25 > 11 composition predictors + 1.

| | |
|---|---|
| corpus | OceanShip (12 classes, 892 vessels) |
| protocol | `vessel_disjoint`, grouped by MMSI, stratified per class |
| draws | seeds 5–24 (the frozen file's 0–4 are kept verbatim and merged in) |
| model / recipe | `cnn_small`, `rec2` (24 epochs, lr 3e-3) |

**Attach**: `under_water_features` only. The extended split file is generated
in-cell from `oceanship/index.csv` + the frozen split file.

**Accelerator**: any GPU. 20 runs ≈ 3.5–4.5 h; the driver stops cleanly at the
budget and resumes on re-run (run ids `…__s5__rec2` … `…__s24__rec2`).

**After it finishes** — Output tab → Create Dataset (`w3n-data`), then re-run
`w3e_bootstrap` and `w3e2_variance`: the OceanShip `vessel_disjoint/rec2` cell
grows to n=25 and the decomposition gets its independent re-estimate."""),
        code(SETUP),
        code(r'''# --------------------------------------------------------------------------- #
# this notebook writes to its OWN run directory and its OWN split file
# --------------------------------------------------------------------------- #
RUNS = WORK / "w3n-runs"
RUNS.mkdir(parents=True, exist_ok=True)
SPLIT_OUT = WORK / "splits" / "oceanship_splits.csv.gz"
SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
tick(f"run dir   : {RUNS}")
tick(f"split out : {SPLIT_OUT}")'''),
        writefile("w3_run.py", (SRC / "w3_run.py").read_text(encoding="utf-8")),
        writefile("w1_02c_oceanship_vessel_splits.py",
                  (SRC / "w1_02c_oceanship_vessel_splits.py").read_text(encoding="utf-8")),
        code(r'''# --------------------------------------------------------------------------- #
# build the extended OceanShip split file: frozen rows + vessel_disjoint seeds 5-24
# preconditions printed so a Kaggle failure is self-diagnosing.
# NOTE: Kaggle's dataset ingest expands archives (x.csv.gz -> x.csv), so the
# frozen file is resolved through the runner's any_text_file helper, which
# accepts both spellings -- the first w3n session died on exactly this.
# --------------------------------------------------------------------------- #
idx = FEAT / "oceanship" / "index.csv"
base = FEAT / "splits" / "oceanship_splits.csv.gz"
import csv as _csv
with open(idx, encoding="utf-8", newline="") as fh:
    cols = _csv.DictReader(fh).fieldnames
print(f"oceanship index columns: {cols}")
for need in ("clip_id", "rec_key", "mmsi", "label"):
    if need not in cols:
        raise SystemExit(f"oceanship/index.csv has no {need!r} column")
sys.path.insert(0, str(SCRIPTS))
import w3_run
base_resolved = w3_run.any_text_file([base])
if base_resolved is None:
    raise SystemExit(f"frozen split file not found near {base} -- it carries seeds 0-4 "
                     "and must be merged, not regenerated")
base = base_resolved
print(f"frozen split file : {base}")
cmd = [sys.executable, str(SCRIPTS / "w1_02c_oceanship_vessel_splits.py"),
       "--index", str(idx), "--base", str(base), "--out", str(SPLIT_OUT),
       "--first-seed", "5", "--n-seeds", "20"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
assert SPLIT_OUT.exists(), "split builder did not write the output file"
tick("extended split file ready")'''),
        code(r'''# --------------------------------------------------------------------------- #
# wire check -- one minute of compute before committing four hours
# --------------------------------------------------------------------------- #
cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
       "--data", str(FEAT), "--out", str(WORK / "_probe"),
       "--corpus", "oceanship", "--protocol", "vessel_disjoint", "--seed", "5",
       "--rep", "logmel", "--backbone", "cnn_small", "--smoke", "--num-workers", "0",
       "--split-file", str(SPLIT_OUT), "--tag", "probe"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
print()
print("  if that printed rows=train/val/test and a leak rate, the wiring is correct.")
tick("probe done")'''),
        md("""## the sweep

20 × (OceanShip, log-mel, `cnn_small`, `vessel_disjoint`, seeds 5–24).
Every run must print **vessel overlap 0**."""),
        code(driver_with_split(230, "rec2", configs)),
        code(REPORT),
        md("""## next

1. Output tab → **Create Dataset** (`w3n-data`).
2. Re-run `w3e_bootstrap` and `w3e2_variance` with it attached: the
   OceanShip `vessel_disjoint/rec2` cell reaches n=25 and the 85% headline
   gets its independent re-estimate."""),
    ])
    path = OUT / "kaggle_w3n_ocean_random.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path}  ({path.stat().st_size:,} bytes, {len(nb['cells'])} cells)")


AGG_CELLS = r'''# --------------------------------------------------------------------------- #
# aggregate EVERYTHING: the session's own runs (working) + every mounted dataset
# (input).  Both aggregation scripts accept repeatable --input-root flags, so
# the two trees are scanned together and run ids are deduplicated.
# --------------------------------------------------------------------------- #
agg = [sys.executable, str(SCRIPTS / "w3e_bootstrap_ci.py"),
       "--input-root", str(INPUT), "--input-root", str(WORK),
       "--out", str(WORK)]
print("$ " + " ".join(agg), flush=True)
subprocess.run(agg, check=True)
tick("w3e bootstrap done")

agg2 = [sys.executable, str(SCRIPTS / "w3e2_variance_decomposition.py"),
        "--input-root", str(INPUT), "--input-root", str(WORK),
        "--out", str(WORK)]
print("$ " + " ".join(agg2), flush=True)
subprocess.run(agg2, check=True)
tick("w3e2 decomposition done")
'''


def build_w3all() -> None:
    """W3-ALL — one notebook: finish the OceanShip random draws (w3n), then
    aggregate everything (w3e + w3e2).  Mount the datasets you want aggregated
    (w3-data, w3k, w3k2, w3m, w3n2, w3e_data ...) plus under_water_features."""
    configs = [{"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
                "protocol": "vessel_disjoint", "seed": s} for s in range(5, 25)]
    nb = notebook([
        md("""# W3-ALL — finish the draws, then aggregate everything (one notebook)

Three stages in one session, no dataset juggling:

1. **sweep** — the remaining OceanShip random vessel-disjoint draws
   (seeds 5–24, `cnn_small`, log-mel, `rec2`); the resume scan skips whatever
   `w3n2-data` already holds, so this finishes the set;
2. **w3e bootstrap** — per-tag draw statistics + paired protocol tests over
   ALL mounted datasets + this session's output;
3. **w3e2 decomposition** — composition share of draw variance per cell
   (the OceanShip `vessel_disjoint/rec2` cell unlocks at n≥14, the target is
   n=25).

**Attach**: `under_water_features` + every results dataset you want in the
aggregate (`w3-data`, `w3k-data`, `w3k2-data`, `w3m-data`, `w3n2-data`, and
`w3e_data` if it holds the DeepShip reference runs). Do **not** mount any
dataset that still holds the discarded 15-class w3n-v1 runs.

**Accelerator**: any GPU.  The driver stops cleanly at its budget and resumes;
the aggregation runs afterwards in the same session and needs only CPU.

**After it finishes** — Output tab → Create Dataset (this session's new draws,
suggest `w3n3-data`), and paste the two aggregation tables back."""),
        code(SETUP),
        code(r'''# --------------------------------------------------------------------------- #
# this notebook writes to its OWN run directory and its OWN split file
# --------------------------------------------------------------------------- #
RUNS = WORK / "w3n-runs"
RUNS.mkdir(parents=True, exist_ok=True)
SPLIT_OUT = WORK / "splits" / "oceanship_splits.csv.gz"
SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
tick(f"run dir   : {RUNS}")
tick(f"split out : {SPLIT_OUT}")'''),
        writefile("w3_run.py", (SRC / "w3_run.py").read_text(encoding="utf-8")),
        writefile("w1_02c_oceanship_vessel_splits.py",
                  (SRC / "w1_02c_oceanship_vessel_splits.py").read_text(encoding="utf-8")),
        code(r'''# --------------------------------------------------------------------------- #
# build the extended OceanShip split file: frozen rows + vessel_disjoint seeds 5-24
# the frozen file is resolved through any_text_file (Kaggle ingest may expand
# archives: x.csv.gz -> x.csv) and the new draws are COVERAGE-LOCKED to the
# frozen clip set -- a 15-class test vocabulary would break composition
# alignment across draws (the fake 97.4% of the first w3n pass).
# --------------------------------------------------------------------------- #
idx = FEAT / "oceanship" / "index.csv"
import csv as _csv
with open(idx, encoding="utf-8", newline="") as fh:
    cols = _csv.DictReader(fh).fieldnames
print(f"oceanship index columns: {cols}")
for need in ("clip_id", "rec_key", "mmsi", "label"):
    if need not in cols:
        raise SystemExit(f"oceanship/index.csv has no {need!r} column")
sys.path.insert(0, str(SCRIPTS))
import w3_run
base_resolved = w3_run.any_text_file([FEAT / "splits" / "oceanship_splits.csv.gz"])
if base_resolved is None:
    raise SystemExit("frozen oceanship split file not found -- it carries seeds 0-4 "
                     "and must be merged, not regenerated")
base = base_resolved
print(f"frozen split file : {base}")
cmd = [sys.executable, str(SCRIPTS / "w1_02c_oceanship_vessel_splits.py"),
       "--index", str(idx), "--base", str(base), "--out", str(SPLIT_OUT),
       "--first-seed", "5", "--n-seeds", "20"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
assert SPLIT_OUT.exists(), "split builder did not write the output file"
tick("extended split file ready")'''),
        code(r'''# --------------------------------------------------------------------------- #
# wire check -- one minute of compute before committing hours
# --------------------------------------------------------------------------- #
cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
       "--data", str(FEAT), "--out", str(WORK / "_probe"),
       "--corpus", "oceanship", "--protocol", "vessel_disjoint", "--seed", "5",
       "--rep", "logmel", "--backbone", "cnn_small", "--smoke", "--num-workers", "0",
       "--split-file", str(SPLIT_OUT), "--tag", "probe"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
tick("probe done")'''),
        md("""## stage 1 — the sweep (seeds 5–24; resume skips what exists)"""),
        code(driver_with_split(450, "rec2", configs)),
        md("""## stage 2/3 — aggregate everything"""),
        writefile("w3e_bootstrap_ci.py",
                  (SRC / "w3e_bootstrap_ci.py").read_text(encoding="utf-8")),
        writefile("w3e2_variance_decomposition.py",
                  (SRC / "w3e2_variance_decomposition.py").read_text(encoding="utf-8")),
        code(AGG_CELLS),
        md("""## next

1. Output tab → **Create Dataset** for this session's new draws
   (`w3n3-data`), so future sessions resume from the full set;
2. Paste the **w3e per-tag table** and the **w3e2 decomposition table** back.
   The OceanShip `vessel_disjoint/rec2` row is the paper's independent
   re-estimate of the 85% headline."""),
    ])
    path = OUT / "kaggle_w3all.ipynb"
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {path}  ({path.stat().st_size:,} bytes, {len(nb['cells'])} cells)")


if __name__ == "__main__":
    build_w3m()
    build_w3e2()
    build_w3n()
    build_w3all()
