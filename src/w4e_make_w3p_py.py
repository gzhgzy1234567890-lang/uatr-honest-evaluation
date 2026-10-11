"""W4-E — generate kaggle_w3p_flat.py: the REVIEW-RESPONSE run as one paste-able
python file (interactive, live output).

Stages
------
M4  DeepShip vessel-disjoint WITHOUT the 48 AIS-unjoined recordings (singleton
    exclusion sensitivity control): 10 draws, tag `rec2ns`, coverage = named
    vessels only.  Compare its mean against the 10-draw full-corpus result
    (0.6241) -- a shift beyond ~2 SD would indicate real residual leakage.
M2  aggregation (w3e) over all mounted datasets: if any mounted dataset holds
    the DeepShip clip_random/rec_random/temporal reference runs (ds1), the
    paired clip-vessel test prints automatically.  Mount `w3e_data` to check.
M1  w3e3 leave-one-draw-out composition prediction on the OceanShip
    vessel_disjoint/rec2 25-draw pool.

Usage:  python src/w4e_make_w3p_py.py
Output: notebooks/kaggle_w3p_flat.py
"""
from __future__ import annotations

import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "kaggle_w3p_flat.py"

SCRIPTS = ("w3_run.py", "w1_02b_deepship_vessel_splits.py",
           "w3e_bootstrap_ci.py", "w3e2_variance_decomposition.py", "w3e3_lodo.py")

STAGE = '''
# --------------------------------------------------------------------------- #
# stage 0 - locate the feature dataset
# --------------------------------------------------------------------------- #
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SCRIPTS = WORK / "scripts"
RUNS = WORK / "w3p-runs"
SPLIT_OUT = WORK / "splits" / "deepship_splits_nosing.csv.gz"
BUDGET_MIN = %(budget)d
RECIPE = ["--epochs", "24", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]
RECIPE_TAG = "rec2ns"          # singleton-excluded draws -- never pooled with rec2
for d in (SCRIPTS, RUNS, SPLIT_OUT.parent):
    d.mkdir(parents=True, exist_ok=True)

print("=" * 88)
print("mounted datasets (w3e_data must be mounted if it holds the DeepShip ds1")
print("reference runs -- the paired clip-vessel test needs them)")
print("=" * 88)
for depth in range(1, 3):
    for p in sorted(INPUT.glob("/".join(["*"] * depth))):
        if p.is_dir():
            print(f"  {p}")


def candidates():
    found = []
    for depth in range(1, 5):
        for p in sorted(INPUT.glob("/".join(["*"] * depth))):
            if p.is_dir() and (p / "oceanship" / "index.csv").exists():
                found.append(p)
    return sorted(set(found), key=lambda p: len(p.parts))


FEAT = candidates()
FEAT = FEAT[0] if FEAT else None
if FEAT is None:
    raise SystemExit("no feature tree found -- mount under_water_features")
print(f"features root : {FEAT}")

sys.path.insert(0, str(SCRIPTS))

# --------------------------------------------------------------------------- #
# stage 1 - materialize scripts
# --------------------------------------------------------------------------- #
for name, blob in EMBEDDED.items():
    target = SCRIPTS / name
    target.write_text(base64.b64decode(blob).decode("utf-8"), encoding="utf-8")
    tick(f"scripts/{name}  written ({target.stat().st_size:,} bytes)")

# --------------------------------------------------------------------------- #
# stage 2 - M4 split file: vessel-disjoint, singletons EXCLUDED, seeds 0-9
# --------------------------------------------------------------------------- #
import w3_run as _wr
base = _wr.any_text_file([FEAT / "splits" / "deepship_splits.csv.gz"])
if base is None:
    raise SystemExit("frozen deepship split file not found (needed for the coverage audit)")
cmd = [sys.executable, str(SCRIPTS / "w1_02b_deepship_vessel_splits.py"),
       "--index", str(FEAT / "deepship" / "index.csv"), "--base", str(base),
       "--out", str(SPLIT_OUT), "--n-seeds", "10", "--exclude-unnamed"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
assert SPLIT_OUT.exists()
tick("singleton-excluded split file ready")

# --------------------------------------------------------------------------- #
# stage 3 - wire check
# --------------------------------------------------------------------------- #
cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
       "--data", str(FEAT), "--out", str(WORK / "_probe"),
       "--corpus", "deepship", "--protocol", "vessel_disjoint", "--seed", "0",
       "--rep", "logmel", "--backbone", "cnn_small", "--smoke", "--num-workers", "0",
       "--split-file", str(SPLIT_OUT), "--tag", "probe"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
tick("probe done")

# --------------------------------------------------------------------------- #
# stage 4 - M4 sweep: 10 draws (resume skips completed seeds), tag rec2ns
# --------------------------------------------------------------------------- #
import w3_run

CONFIGS = [{"seed": s} for s in range(10)]


def run_id_for(seed):
    return w3_run.run_id_for("deepship", "logmel", "cnn_small", "vessel_disjoint",
                             seed, "none", "erm", False, RECIPE_TAG)


def done_ids():
    import csv as _csv
    ids = set()
    for res in [RUNS / "results.csv"] + sorted(INPUT.rglob("results.csv")):
        if not res.is_file():
            continue
        try:
            rel = res.relative_to(INPUT)
        except ValueError:
            rel = res.relative_to(WORK)
        if any(str(part).startswith("_") for part in rel.parts):
            continue
        try:
            with res.open(encoding="utf-8", newline="") as fh:
                ids |= {x["run_id"] for x in _csv.DictReader(fh)}
        except Exception as exc:
            print(f"    [resume] {res} unreadable ({exc})")
    return ids


have = done_ids()
todo = [c for c in CONFIGS if run_id_for(c["seed"]) not in have]
print(f"[state] {len(CONFIGS) - len(todo)} of {len(CONFIGS)} already complete "
      f"(mount the w3p session's output dataset so seeds 0-2 count as done)")
t0 = time.time()
for i, c in enumerate(todo):
    el = (time.time() - t0) / 60
    if el > BUDGET_MIN:
        print(f"[stop] budget of {BUDGET_MIN} min reached; re-run this cell to continue.")
        break
    cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
           "--data", str(FEAT), "--out", str(RUNS), "--num-workers", "2",
           "--corpus", "deepship", "--rep", "logmel", "--backbone", "cnn_small",
           "--protocol", "vessel_disjoint", "--seed", str(c["seed"]),
           "--method", "erm", "--domain", "none", "--vocab", "native",
           "--tag", RECIPE_TAG, "--split-file", str(SPLIT_OUT)] + RECIPE
    print(f"\\n{'=' * 88}\\n[{i + 1}/{len(todo)}] {run_id_for(c['seed'])}   elapsed {el:.1f} min"
          f"\\n{'=' * 88}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        print(f"[warn] exit {r.returncode}; leaving it pending", flush=True)
tick("M4 sweep done")

# --------------------------------------------------------------------------- #
# stage 5 - aggregate: w3e (paired tests incl. M2) + w3e2 + M1 LODO
# --------------------------------------------------------------------------- #
for script in ("w3e_bootstrap_ci.py", "w3e2_variance_decomposition.py"):
    cmd = [sys.executable, str(SCRIPTS / script),
           "--input-root", str(INPUT), "--input-root", str(WORK), "--out", str(WORK)]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    tick(f"{script} done")

cmd = [sys.executable, str(SCRIPTS / "w3e3_lodo.py"),
       "--input-root", str(INPUT), "--input-root", str(WORK), "--out", str(WORK)]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
tick("w3e3 LODO done")

print()
print("=" * 88)
print("ALL STAGES DONE -- what to paste back")
print("=" * 88)
print("  M4  : the w3e row  deepship/cnn_small/vessel_disjoint/rec2ns  (n=3) --")
print("        compare its mean to the full-corpus 0.6241 (a shift beyond ~2 SD")
print("        of the 10-draw mean, ~1.7 pp, would indicate real residual leakage)")
print("  M2  : any 'deepship/.../rec2: clip_random - vessel_disjoint' paired line")
print("        (requires the ds1 reference runs to be mounted, e.g. in w3e_data)")
print("  M1  : the w3e3 LODO block (in-sample R^2 vs LODO R^2, RMSE, r, perm p)")
'''


def main():
    blobs = {}
    for name in SCRIPTS:
        src = (SRC / name).read_text(encoding="utf-8")
        blobs[name] = base64.b64encode(src.encode("utf-8")).decode("ascii")
    emb_lines = ["EMBEDDED = {"]
    for name, blob in blobs.items():
        emb_lines.append(f"    {name!r}:")
        for i in range(0, len(blob), 96):
            emb_lines.append(f"        {blob[i:i + 96]!r}")
        emb_lines[-1] += ","
    emb_lines.append("}")
    emb_block = "\n".join(emb_lines)

    header = (
        "# W3-P review-response driver - paste this ENTIRE file into ONE Kaggle cell\n"
        "# (interactive mode: live output).  Mount before running:\n"
        "#   under_water_features + w3-data + w3m-data + w3n4-data (the clean 25-draw\n"
        "#   pool) + w3k_data + w3k2_data + w3e_data (ds1 check for the paired test)\n"
        "# DO NOT mount any w3n-v1/v2/v3 dataset (invalid draws, same run ids).\n"
        "import base64, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "T0 = time.time()\n"
        "def tick(msg):\n"
        "    print(f\"  [{time.time() - T0:7.1f}s] {msg}\", flush=True)\n\n"
    )
    text = header + emb_block + "\n" + STAGE % {"budget": 250}
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    for name, blob in blobs.items():
        assert base64.b64decode(blob).decode("utf-8") == (SRC / name).read_text(encoding="utf-8"), name
    print("embedded copies verified byte-exact:", ", ".join(SCRIPTS))


if __name__ == "__main__":
    main()

