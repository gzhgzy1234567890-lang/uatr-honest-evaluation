#!/usr/bin/env python3
"""W3-02  Aggregate benchmark runs into the paper's tables, with honest error bars.

Two things this file exists to get right:

1. Recording-level bootstrap confidence intervals.
   Test clips are not independent observations.  A single recording can contribute
   dozens of clips that share its channel, its background noise and its operating
   condition, so resampling clips would badly understate the uncertainty - it
   would treat 50 clips from one ship as 50 independent pieces of evidence.

   We therefore bootstrap over RECORDINGS: draw recordings with replacement,
   gather all their test clips, recompute the metric, repeat B times.  This is the
   interval that a reviewer should ask for, and it is precisely what a recent
   DeepShip benchmark paper explicitly listed as future work.

2. Never reporting an accuracy without its protocol.
   Every table row carries the protocol and the measured leak rate, because a
   DeepShip number quoted without them is meaningless - the same model differs by
   roughly 27 points between protocols.

Usage
-----
    python w3_02_aggregate.py --results /kaggle/working/results.csv \
                              --predictions /kaggle/working/predictions \
                              --out /kaggle/working/tables
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PROTOCOL_ORDER = ("clip_random", "rec_random", "temporal")
B_BOOT = 1000
CI = 95.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def bootstrap_over_recordings(
    y_true: np.ndarray, y_pred: np.ndarray, rec_key: np.ndarray, n_classes: int,
    b: int = B_BOOT, seed: int = 0,
) -> dict:
    """Resample recordings (clusters) with replacement; return metric CIs."""
    rng = np.random.default_rng(seed)
    # group clip indices by recording once
    uniq, inv = np.unique(rec_key, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uniq))]
    n_rec = len(groups)

    accs, f1s = np.empty(b), np.empty(b)
    for i in range(b):
        pick = rng.integers(0, n_rec, n_rec)
        idx = np.concatenate([groups[j] for j in pick])
        accs[i] = (y_pred[idx] == y_true[idx]).mean()
        f1s[i] = macro_f1(y_true[idx], y_pred[idx], n_classes)

    lo, hi = (100 - CI) / 2, 100 - (100 - CI) / 2
    return {
        "acc_lo": float(np.percentile(accs, lo)),
        "acc_hi": float(np.percentile(accs, hi)),
        "f1_lo": float(np.percentile(f1s, lo)),
        "f1_hi": float(np.percentile(f1s, hi)),
        "n_recordings": int(n_rec),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate benchmark results.")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--b", type=int, default=B_BOOT)
    args = ap.parse_args()

    with args.results.open(encoding="utf-8") as fh:
        runs = list(csv.DictReader(fh))
    print(f"[input] {len(runs)} runs")

    # ------------------------------------------------------- per-run bootstrap CI
    ci_rows = []
    for r in runs:
        npz_path = args.predictions / f"{r['run_id']}.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        n_classes = int(r["n_classes"])
        ci = bootstrap_over_recordings(
            d["y_true"], d["y_pred"], d["rec_key"], n_classes, b=args.b, seed=int(r["seed"])
        )
        ci_rows.append({"run_id": r["run_id"], **ci})

    args.out.mkdir(parents=True, exist_ok=True)
    if ci_rows:
        with (args.out / "bootstrap_ci.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ci_rows[0].keys()))
            w.writeheader()
            w.writerows(ci_rows)
        ci_by_run = {c["run_id"]: c for c in ci_rows}
    else:
        ci_by_run = {}

    # ---------------------------------------------------------------- grouping
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["protocol"], r["rep"], r["backbone"])].append(r)

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("Benchmark results")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"bootstrap: {args.b} resamples over RECORDINGS (clusters), {CI:.0f}% interval")
    lines.append("")

    # ------------------------------------------------- the headline comparison
    lines.append("=" * 100)
    lines.append("Headline: the same model, three protocols")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"  {'representation':<12}{'backbone':<17}{'protocol':<14}{'acc mean':>10}{'acc sd':>9}"
                 f"{'95% CI (boot)':>20}{'leak%':>8}{'n test recs':>13}")
    for rep in sorted({r["rep"] for r in runs}):
        for backbone in sorted({r["backbone"] for r in runs}):
            for protocol in PROTOCOL_ORDER:
                rs = groups.get((protocol, rep, backbone))
                if not rs:
                    continue
                accs = np.array([float(x["test_acc"]) for x in rs])
                cis = [ci_by_run[x["run_id"]] for x in rs if x["run_id"] in ci_by_run]
                if cis:
                    lo = np.mean([c["acc_lo"] for c in cis])
                    hi = np.mean([c["acc_hi"] for c in cis])
                    ci_str = f"[{lo:.3f}, {hi:.3f}]"
                    nrecs = int(np.mean([c["n_recordings"] for c in cis]))
                else:
                    ci_str, nrecs = "-", 0
                lines.append(
                    f"  {rep:<12}{backbone:<17}{protocol:<14}{accs.mean():>10.4f}{accs.std(ddof=0):>9.4f}"
                    f"{ci_str:>20}{np.mean([float(x['leak_pct']) for x in rs]):>8.1f}{nrecs:>13}"
                )
        lines.append("")

    # ------------------------------------------------------------- the gap table
    lines.append("=" * 100)
    lines.append("The inflation gap: accuracy lost when the protocol is made honest")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"  {'representation':<12}{'backbone':<17}{'clip_random':>13}{'rec_random':>13}"
                 f"{'temporal':>11}{'P1-P2':>9}{'P1-P3':>9}")
    for rep in sorted({r["rep"] for r in runs}):
        for backbone in sorted({r["backbone"] for r in runs}):
            def mean_acc(p: str):
                rs = groups.get((p, rep, backbone))
                return float(np.mean([float(x["test_acc"]) for x in rs])) if rs else None
            a1, a2, a3 = mean_acc("clip_random"), mean_acc("rec_random"), mean_acc("temporal")
            if a1 is None:
                continue
            f = lambda v: f"{v:.4f}" if v is not None else "-"
            d2 = f"{a1-a2:.4f}" if a2 is not None else "-"
            d3 = f"{a1-a3:.4f}" if a3 is not None else "-"
            lines.append(f"  {rep:<12}{backbone:<17}{f(a1):>13}{f(a2):>13}{f(a3):>11}{d2:>9}{d3:>9}")
    lines.append("")

    # ---------------------------------------------------------- per-class detail
    lines.append("=" * 100)
    lines.append("Per-class accuracy (mean over seeds) - leakage is not uniform across classes")
    lines.append("=" * 100)
    lines.append("")
    class_cols = sorted({k[len("acc_"):] for k in runs[0] if k.startswith("acc_")})
    lines.append(f"  {'protocol':<14}{'rep':<12}{'backbone':<17}" + "".join(f"{c:>12}" for c in class_cols))
    for protocol in PROTOCOL_ORDER:
        for rep in sorted({r["rep"] for r in runs}):
            for backbone in sorted({r["backbone"] for r in runs}):
                rs = groups.get((protocol, rep, backbone))
                if not rs:
                    continue
                vals = []
                for c in class_cols:
                    v = [float(x[f"acc_{c}"]) for x in rs if x.get(f"acc_{c}")]
                    vals.append(f"{np.mean(v):>12.4f}" if v else f"{'-':>12}")
                lines.append(f"  {protocol:<14}{rep:<12}{backbone:<17}" + "".join(vals))
    lines.append("")

    # ---------------------------------------------------------------- parametric
    lines.append("=" * 100)
    lines.append("Model size vs strict-protocol accuracy")
    lines.append("=" * 100)
    lines.append("")
    lines.append("  Reproduces or contradicts the reported finding that an 11.17M-parameter")
    lines.append("  ResNet18 gives no gain over a ~150K compact CNN once the split is honest.")
    lines.append("")

    def mean_acc_for(protocol: str, backbone: str, rep: str | None = None) -> float:
        rs = [
            r for r in runs
            if r["protocol"] == protocol and r["backbone"] == backbone
            and (rep is None or r["rep"] == rep)
        ]
        return float(np.mean([float(r["test_acc"]) for r in rs])) if rs else float("nan")

    lines.append(f"  {'backbone':<17}{'params':>14}{'rec_random acc':>18}{'temporal acc':>15}")
    for backbone in sorted({r["backbone"] for r in runs}):
        params = [int(r["n_params"]) for r in runs if r["backbone"] == backbone]
        lines.append(
            f"  {backbone:<17}{max(params) if params else 0:>14,}"
            f"{mean_acc_for('rec_random', backbone):>18.4f}"
            f"{mean_acc_for('temporal', backbone):>15.4f}"
        )
    lines.append("")

    text = "\n".join(lines)
    print(text)
    (args.out / "benchmark_tables.txt").write_text(text, encoding="utf-8")
    print(f"\nwritten: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
