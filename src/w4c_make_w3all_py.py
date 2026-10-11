"""W4-C — generate kaggle_w3all_flat.py: the whole W3-ALL pipeline as ONE python
file to paste into a single interactive Kaggle cell (live output, no black box).

The four src scripts are embedded as base64 constants (byte-exact, no quoting
issues) and materialized to /kaggle/working/scripts at runtime.

Usage:  python src/w4c_make_w3all_py.py
Output: notebooks/kaggle_w3all_flat.py
"""
from __future__ import annotations

import base64
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "kaggle_w3all_flat.py"

SCRIPTS = ("w3_run.py", "w1_02c_oceanship_vessel_splits.py",
           "w3e_bootstrap_ci.py", "w3e2_variance_decomposition.py")

STAGE = '''
# --------------------------------------------------------------------------- #
# stage 0 - locate the feature dataset (same discipline as the notebooks)
# --------------------------------------------------------------------------- #
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SCRIPTS = WORK / "scripts"
RUNS = WORK / "w3n-runs"
SPLIT_OUT = WORK / "splits" / "oceanship_splits.csv.gz"
BUDGET_MIN = %(budget)d          # sweep budget; raise if a session survives longer
RECIPE = ["--epochs", "24", "--norm", "affine", "--lr", "0.003", "--warmup", "3"]
RECIPE_TAG = "rec2"

for d in (SCRIPTS, RUNS, SPLIT_OUT.parent):
    d.mkdir(parents=True, exist_ok=True)

print("=" * 88)
print("mounted datasets")
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
    raise SystemExit("no feature tree found -- mount under_water_features "
                     "(a dir containing oceanship/index.csv)")
print(f"features root : {FEAT}")

sys.path.insert(0, str(SCRIPTS))

# --------------------------------------------------------------------------- #
# stage 1 - materialize the scripts (base64, byte-exact)
# --------------------------------------------------------------------------- #
for name, blob in EMBEDDED.items():
    target = SCRIPTS / name
    target.write_text(base64.b64decode(blob).decode("utf-8"), encoding="utf-8")
    tick(f"scripts/{name}  written ({target.stat().st_size:,} bytes)")

# --------------------------------------------------------------------------- #
# stage 2 - build the coverage-locked OceanShip split file (seeds 5-24)
# --------------------------------------------------------------------------- #
import w3_run as _wr
base = _wr.any_text_file([FEAT / "splits" / "oceanship_splits.csv.gz"])
if base is None:
    raise SystemExit("frozen oceanship split file not found -- it carries seeds 0-4 "
                     "and must be merged, not regenerated")
print(f"frozen split file : {base}")
cmd = [sys.executable, str(SCRIPTS / "w1_02c_oceanship_vessel_splits.py"),
       "--index", str(FEAT / "oceanship" / "index.csv"), "--base", str(base),
       "--out", str(SPLIT_OUT), "--first-seed", "5", "--n-seeds", "20"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
assert SPLIT_OUT.exists()
tick("extended split file ready")

# --------------------------------------------------------------------------- #
# stage 3 - wire check
# --------------------------------------------------------------------------- #
cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
       "--data", str(FEAT), "--out", str(WORK / "_probe"),
       "--corpus", "oceanship", "--protocol", "vessel_disjoint", "--seed", "5",
       "--rep", "logmel", "--backbone", "cnn_small", "--smoke", "--num-workers", "0",
       "--split-file", str(SPLIT_OUT), "--tag", "probe"]
print("$ " + " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)
tick("probe done")

# --------------------------------------------------------------------------- #
# stage 4 - the sweep (resume: every results.csv under WORK and INPUT counts)
# --------------------------------------------------------------------------- #
import w3_run

CONFIGS = [{"corpus": "oceanship", "rep": "logmel", "backbone": "cnn_small",
            "protocol": "vessel_disjoint", "seed": s} for s in range(5, 25)]


def run_id_for(c):
    rid = w3_run.run_id_for(c["corpus"], c["rep"], c["backbone"], c["protocol"],
                            c["seed"], "none", "erm", False, RECIPE_TAG)
    return rid


def done_ids():
    import csv as _csv
    ids = set()
    roots = [WORK] + [p for p in INPUT.rglob("results.csv")]
    seen_csv = set()
    for res in [WORK / "w3n-runs" / "results.csv"] + sorted(INPUT.rglob("results.csv")):
        if res in seen_csv or not res.is_file():
            continue
        seen_csv.add(res)
        rel = res.relative_to(INPUT) if res.is_relative_to(INPUT) else res.relative_to(WORK)
        if any(str(part).startswith("_") for part in rel.parts):
            continue
        try:
            with res.open(encoding="utf-8", newline="") as fh:
                got = list(_csv.DictReader(fh))
            ids |= {x["run_id"] for x in got}
            print(f"    [resume] {res} -> {len(got)} rows")
        except Exception as exc:
            print(f"    [resume] {res} unreadable ({exc})")
    return ids


have = done_ids()
todo = [c for c in CONFIGS if run_id_for(c) not in have]
print(f"[state] {len(CONFIGS) - len(todo)} of {len(CONFIGS)} already complete")
t0 = time.time()
for i, c in enumerate(todo):
    el = (time.time() - t0) / 60
    if el > BUDGET_MIN:
        print(f"[stop] budget of {BUDGET_MIN} min reached after {i} runs; "
              f"re-run this cell to continue.")
        break
    cmd = [sys.executable, str(SCRIPTS / "w3_run.py"),
           "--data", str(FEAT), "--out", str(RUNS), "--num-workers", "2",
           "--corpus", c["corpus"], "--rep", c["rep"], "--backbone", c["backbone"],
           "--protocol", c["protocol"], "--seed", str(c["seed"]),
           "--method", "erm", "--domain", "none", "--vocab", "native",
           "--tag", RECIPE_TAG, "--split-file", str(SPLIT_OUT)] + RECIPE
    print(f"\\n{'=' * 88}\\n[{i + 1}/{len(todo)}] {run_id_for(c)}   elapsed {el:.1f} min"
          f"\\n{'=' * 88}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        print(f"[warn] exit {r.returncode}; leaving it pending and continuing", flush=True)
tick("sweep stage done")

# --------------------------------------------------------------------------- #
# stage 5 - aggregate EVERYTHING (mounted datasets + this session's output)
# --------------------------------------------------------------------------- #
for script in ("w3e_bootstrap_ci.py", "w3e2_variance_decomposition.py"):
    cmd = [sys.executable, str(SCRIPTS / script),
           "--input-root", str(INPUT), "--input-root", str(WORK),
           "--out", str(WORK)]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    tick(f"{script} done")

print()
print("=" * 88)
print("ALL STAGES DONE")
print("=" * 88)
print("next: Output tab -> Create Dataset (keep w3n-runs), then paste the two")
print("      aggregation tables above back into the working session.")
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
        "# W3-ALL flat driver - paste this ENTIRE file into ONE Kaggle cell\n"
        "# (interactive mode recommended: you get live output instead of the\n"
        "# commit-mode black box).  Mount before running:\n"
        "#   under_water_features + w3-data + w3k_data + w3k2_data + w3m-data\n"
        "#   + w3n2-data + w3e_data (w3e_data optional; harmless if unrelated)\n"
        "# DO NOT mount any dataset that still holds the discarded 15-class\n"
        "# w3n-v1 draws.  The sweep resumes: completed seeds are skipped.\n"
        "import base64, csv, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "T0 = time.time()\n"
        "def tick(msg):\n"
        "    print(f\"  [{time.time() - T0:7.1f}s] {msg}\", flush=True)\n\n"
    )
    text = header + emb_block + "\n" + STAGE % {"budget": 450}
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size:,} bytes)")
    # sanity: the embedded blobs decode back to the src files
    for name, blob in blobs.items():
        back = base64.b64decode(blob).decode("utf-8")
        assert back == (SRC / name).read_text(encoding="utf-8"), name
    print("embedded copies verified byte-exact:", ", ".join(SCRIPTS))


if __name__ == "__main__":
    main()
