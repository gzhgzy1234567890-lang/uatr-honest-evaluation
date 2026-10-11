#!/usr/bin/env python3
"""W3-00  Sweep driver: run every pending configuration, resuming across sessions.

Why a driver instead of a shell loop
------------------------------------
A Kaggle GPU session dies at 12 h, and a full sweep is bigger than that.  This
driver reads results.csv, works out which (representation, backbone, protocol,
seed) tuples are still missing, runs them in priority order, and stops cleanly
before the session limit.  Re-running it in the next session picks up exactly
where it left off - no configuration is ever run twice.

Priority order is chosen so the paper's headline result arrives first:
  1  cnn_small + logmel        -> the protocol gap, the core figure
  2  logreg    + logmel        -> a cheap sanity floor
  3  cnn_small + spec          -> is the effect representation-specific?
  4  resnet18  + logmel        -> does a bigger model help? (the negative result)
  5  the rest                  -> stretch goals

Usage
-----
    python w3_00_sweep.py --cache /kaggle/input/uwatr-logmel \
        --splits /kaggle/input/uwatr-protocol-v1/splits/splits.csv.gz \
        --out /kaggle/working --budget-min 600
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

PROTOCOLS = ("clip_random", "rec_random", "temporal")

# (representation, backbone) in priority order
PRIORITY = [
    ("logmel", "cnn_small"),
    ("logmel", "logreg"),
    ("spec", "cnn_small"),
    ("logmel", "resnet18"),
    ("spec", "logreg"),
    ("logmel", "efficientnet_b0"),
    ("spec", "resnet18"),
    ("logmel", "ast"),
]


def done_runs(results: Path) -> set[tuple[str, str, str, int]]:
    if not results.exists():
        return set()
    out = set()
    with results.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.add((r["rep"], r["backbone"], r["protocol"], int(r["seed"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all pending benchmark configurations.")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--runner", type=Path, default=Path(__file__).with_name("w3_01_run_benchmark.py"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--budget-min", type=float, default=600, help="stop after this many minutes")
    ap.add_argument("--limit", type=int, default=0, help="run at most N configs this session")
    args = ap.parse_args()

    results = args.out / "results.csv"
    already = done_runs(results)
    print(f"[state] {len(already)} configurations already complete")

    plan: list[tuple[str, str, str, int]] = []
    for rep, backbone in PRIORITY:
        for protocol in PROTOCOLS:
            for seed in range(args.seeds):
                if (rep, backbone, protocol, seed) not in already:
                    plan.append((rep, backbone, protocol, seed))

    total_expected = len(PRIORITY) * len(PROTOCOLS) * args.seeds
    print(f"[plan] {len(plan)} pending of {total_expected} total")
    if not plan:
        print("nothing to do")
        return 0

    t0 = time.time()
    ran = 0
    for rep, backbone, protocol, seed in plan:
        elapsed_min = (time.time() - t0) / 60
        if elapsed_min > args.budget_min:
            print(f"[stop] budget reached after {ran} runs ({elapsed_min:.1f} min)")
            break
        if args.limit and ran >= args.limit:
            print(f"[stop] per-session limit of {args.limit} runs reached")
            break

        cmd = [
            sys.executable, str(args.runner),
            "--cache", str(args.cache), "--splits", str(args.splits), "--out", str(args.out),
            "--rep", rep, "--backbone", backbone, "--protocol", protocol, "--seed", str(seed),
            "--epochs", str(args.epochs),
        ]
        print(f"\n{'='*78}\n[{ran+1}/{len(plan)}] {rep} / {backbone} / {protocol} / seed {seed}"
              f"   (elapsed {elapsed_min:.1f} min)\n{'='*78}", flush=True)
        r = subprocess.run(cmd)
        if r.returncode != 0:
            # a single failing config must not kill the sweep; it stays pending and
            # will be retried next session
            print(f"[warn] run failed with code {r.returncode}; continuing", flush=True)
        ran += 1

    print(f"\n[session done] ran {ran} configurations in {(time.time()-t0)/60:.1f} min")
    remaining = len(plan) - ran
    if remaining:
        print(f"[next session] {remaining} still pending - just re-run this script")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
