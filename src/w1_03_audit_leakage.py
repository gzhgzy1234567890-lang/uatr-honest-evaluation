#!/usr/bin/env python3
"""W1-03  Audit the evaluation protocols for recording-level leakage.

This produces the headline numbers of the paper's premise.  It needs only the
split files plus the recording table, so it runs in seconds and touches no audio.

What is measured
----------------
1. leakage rate
   The share of TEST clips whose source recording also contributes TRAIN clips.
   Under clip_random this is near 100%: the recorder's channel response, the
   background noise and the ship's operating condition are all present on both
   sides of the border, so the test set is largely a memorisation check.
   Under rec_random and temporal it is exactly 0% by construction.

2. recording overlap
   How many recordings appear in more than one split.  Zero is the target.

3. effective test diversity  <- the statistic worth putting in the abstract
   How many DISTINCT RECORDINGS the test split actually covers, versus how many
   clips it contains.  A test split with 5,000 clips drawn from 30 recordings
   offers far less independent evidence than 5,000 clips from 5,000 recordings.
   We report the ratio (clips per distinct recording) as an interpretability
   yardstick for any DeepShip number.

4. class-conditional leakage
   Leakage is rarely uniform across classes, and the classes differ in both size
   and recording length, so an aggregate number can hide the worst case.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

PROTOCOLS = ("clip_random", "rec_random", "temporal")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit splits for recording-level leakage.")
    ap.add_argument("--splits", type=Path, required=True, help="splits.csv.gz from w1_02")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    # (protocol, seed) -> split -> set/list of rec_keys and clip counts
    recs_in_split: dict[tuple, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    clip_count: dict[tuple, Counter] = defaultdict(Counter)
    rec_clip_count: dict[tuple, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    class_of: dict[str, str] = {}

    with gzip.open(args.splits, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["protocol"], int(row["seed"]))
            rk, sp = row["rec_key"], row["split"]
            recs_in_split[key][sp].add(rk)
            clip_count[key][sp] += 1
            rec_clip_count[key][rk][sp] += 1
            class_of.setdefault(rk, rk.split("/")[0])

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Protocol leakage audit")
    lines.append("=" * 80)
    lines.append("")
    lines.append("A test clip is 'leaked' if its source recording also contributes training clips.")
    lines.append("Under the leakage regime the model has already seen that recorder's channel,")
    lines.append("that day's background noise and that ship's operating condition.")
    lines.append("")

    table: list[dict] = []

    for protocol in PROTOCOLS:
        seeds = sorted(s for (p, s) in recs_in_split if p == protocol)
        lines.append("-" * 80)
        lines.append(f"{protocol}")
        lines.append("-" * 80)

        agg_leak = []
        agg_overlap = []
        agg_eff = []
        for seed in seeds:
            key = (protocol, seed)
            tr = recs_in_split[key]["train"]
            te = recs_in_split[key]["test"]
            va = recs_in_split[key]["val"]

            test_recs = {rk for rk in te}
            leaked_recs = test_recs & tr
            leaked_clips = sum(
                n for rk, c in rec_clip_count[key].items() if rk in leaked_recs for n in [c["test"]]
            )
            total_test_clips = clip_count[key]["test"]
            leak_rate = 100 * leaked_clips / total_test_clips if total_test_clips else 0.0

            overlap = len(tr & te) + len(tr & va) + len(va & te)
            clips_per_rec = total_test_clips / len(test_recs) if test_recs else 0.0

            agg_leak.append(leak_rate)
            agg_overlap.append(overlap)
            agg_eff.append(clips_per_rec)

            if seed == seeds[0]:
                lines.append(
                    f"  seed {seed}: test clips={total_test_clips:,}  distinct test recordings={len(test_recs):,}"
                )
                lines.append(
                    f"            leaked test clips={leaked_clips:,} ({leak_rate:.1f}%)  "
                    f"recordings crossing splits={overlap}"
                )
                lines.append(
                    f"            clips per distinct test recording = {clips_per_rec:.1f}"
                )

        mean = lambda v: sum(v) / len(v) if v else 0.0
        lines.append(
            f"  MEAN over {len(seeds)} seeds: leaked test clips {mean(agg_leak):.1f}%   "
            f"crossing recordings {mean(agg_overlap):.1f}   "
            f"clips per distinct test recording {mean(agg_eff):.1f}"
        )
        lines.append("")
        table.append(
            {
                "protocol": protocol,
                "leaked_test_clips_pct": round(mean(agg_leak), 2),
                "recordings_crossing_splits": round(mean(agg_overlap), 2),
                "clips_per_distinct_test_recording": round(mean(agg_eff), 2),
            }
        )

    # ------------------------------------------------ class-conditional leakage
    lines.append("=" * 80)
    lines.append("class-conditional leakage (clip_random, mean over seeds)")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'class':<11}{'test clips':>12}{'leaked %':>11}{'clips/rec':>11}")
    classes = sorted({class_of[rk] for k in recs_in_split.values() for rk in k.get("test", set())})
    for cls in classes:
        rates, per_rec, sizes = [], [], []
        for protocol in ("clip_random", "rec_random", "temporal"):
            if protocol != "clip_random":
                continue
            for seed in sorted(s for (p, s) in recs_in_split if p == protocol):
                key = (protocol, seed)
                tr = recs_in_split[key]["train"]
                te = {rk for rk in recs_in_split[key]["test"] if class_of[rk] == cls}
                if not te:
                    continue
                leaked = te & tr
                n_test = sum(rec_clip_count[key][rk]["test"] for rk in te)
                n_leak = sum(rec_clip_count[key][rk]["test"] for rk in leaked)
                rates.append(100 * n_leak / n_test if n_test else 0.0)
                per_rec.append(n_test / len(te) if te else 0.0)
                sizes.append(n_test)
        if sizes:
            lines.append(
                f"  {cls:<11}{sum(sizes)/len(sizes):>12,.0f}{sum(rates)/len(rates):>11.1f}"
                f"{sum(per_rec)/len(per_rec):>11.1f}"
            )
    lines.append("")

    # ------------------------------------------------------------- the takeaway
    p1 = next(t for t in table if t["protocol"] == "clip_random")
    p2 = next(t for t in table if t["protocol"] == "rec_random")
    lines.append("=" * 80)
    lines.append("Interpretation")
    lines.append("=" * 80)
    lines.append("")
    lines.append(
        f"  clip_random : {p1['leaked_test_clips_pct']:.1f}% of test clips come from a recording that"
    )
    lines.append(
        f"                the model already trained on; the test split draws on only"
    )
    lines.append(
        f"                {p1['clips_per_distinct_test_recording']:.0f} clip-equivalents per distinct recording."
    )
    lines.append(
        f"  rec_random  : {p2['leaked_test_clips_pct']:.1f}% leaked - clean by construction."
    )
    lines.append("")
    lines.append("  Any accuracy produced under clip_random is therefore an upper bound of")
    lines.append("  unknown tightness, not an estimate of recognition performance. This is why")
    lines.append("  the field reports 99% while the honest number sits near 72%.")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "w1_03_leakage_audit.txt").write_text(text, encoding="utf-8")
    with (args.out / "w1_03_leakage_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
