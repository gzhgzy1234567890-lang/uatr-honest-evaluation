#!/usr/bin/env python3
"""W1-02c — append NEW OceanShip vessel-disjoint draws (seeds 5..24) to the split file.

Why
---
The frozen OceanShip split file ships vessel_disjoint for seeds 0-4 only.  The
paper's variance decomposition (85% composition) rests on those five draws; the
review needs an independent re-estimate on a larger same-scheme pool.  This
script generates plain random vessel-disjoint draws for seeds 5-24 and merges
them with the frozen rows.

Discipline
----------
- Grouping unit is `mmsi`, the column the runner's leak check uses
  (VESSEL_COL["oceanship"]); one rec_key must map to exactly one mmsi.
- **Coverage lock**: new draws cover EXACTLY the frozen file's clip set.  The
  frozen file evaluates 12 of OceanShip's 15 classes (53,754 of 53,770 clips);
  emitting rows for the full index would give the new draws a 15-class test
  vocabulary while the old draws have 12 -- the class-index space then differs
  across draws and every composition alignment by index breaks (this produced
  a fake 97.4% explained share in the first w3n pass).
- Vessel-level stratified split per class, 70/15/15, stable_seed derived from
  "{protocol}|{seed_index}"; seeds 5-24 only, so the frozen seeds 0-4 are
  never redefined.
- Classes with <3 vessels among the covered clips are forced wholly into train
  and reported -- never silently dropped.
- The frozen rows are copied through untouched; a duplicate (protocol, seed)
  pair in the base file is an error, not a silent overwrite.

Output: one gzipped csv (protocol, seed, clip_id, split) + a .report.txt.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def stable_seed(protocol: str, seed_index: int) -> int:
    digest = hashlib.sha256(f"{protocol}|{seed_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def main() -> int:
    ap = argparse.ArgumentParser(description="Append OceanShip vessel_disjoint draws.")
    ap.add_argument("--index", required=True,
                    help="oceanship/index.csv (needs clip_id, rec_key, mmsi, label)")
    ap.add_argument("--base", required=True,
                    help="frozen oceanship_splits file (.csv.gz or expanded .csv); "
                         "rows copied through untouched, clip set locked to it")
    ap.add_argument("--out", required=True, help="output .csv.gz path")
    ap.add_argument("--first-seed", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=20,
                    help="draws generated: seeds first-seed .. first-seed+n-seeds-1")
    ap.add_argument("--protocol", default="vessel_disjoint")
    args = ap.parse_args()

    # ------------------------------------------------------- base (frozen) ---
    opener = gzip.open if str(args.base).endswith(".gz") else open
    with opener(args.base, "rt", newline="", encoding="utf-8") as fh:
        base_rows = list(csv.DictReader(fh))
    if not base_rows:
        raise SystemExit(f"{args.base} is empty")
    print(f"[base ] {len(base_rows):,} frozen rows from {args.base}")
    seen_pair = {(r["protocol"], int(r["seed"])) for r in base_rows}
    clash = [p for p in seen_pair
             if p[0] == args.protocol and args.first_seed <= p[1] < args.first_seed + args.n_seeds]
    if clash:
        raise SystemExit(f"frozen file already holds {args.protocol} seeds "
                         f"{sorted(s for _, s in clash)} -- refusing to redefine them")
    base_clips = {r["clip_id"] for r in base_rows}

    # ------------------------------------------------------------------ index
    with open(args.index, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{args.index} is empty")
    for col in ("clip_id", "rec_key", "mmsi", "label"):
        if col not in rows[0]:
            raise SystemExit(f"{args.index} has no {col!r} column -- cannot group by vessel")

    rec_meta: dict[str, tuple[str, str]] = {}
    for r in rows:
        if r["clip_id"] not in base_clips:
            continue                      # coverage lock: only frozen-covered clips
        meta = (r["mmsi"].strip(), r["label"].strip())
        prev = rec_meta.setdefault(r["rec_key"], meta)
        if prev != meta:
            raise SystemExit(
                f"rec_key {r['rec_key']!r} maps to vessels {prev} and {meta} -- "
                "a recording must belong to exactly one vessel")
    print(f"[index] {len(rows):,} clips total; {len(rec_meta):,} recordings inside the "
          f"frozen clip set ({len(base_clips):,} clips)")

    vessel_of: dict[str, str] = {rec: v for rec, (v, _) in rec_meta.items()}
    vessel_class: dict[str, str] = {}
    for v, c in rec_meta.values():
        vessel_class[v] = c
    print(f"[index] {len(vessel_class)} vessels cover the frozen clip set")

    rec_clips: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["rec_key"] in rec_meta:
            rec_clips[r["rec_key"]].append(r["clip_id"])
    covered = sum(len(v) for v in rec_clips.values())
    if covered != len(base_clips):
        raise SystemExit(f"coverage mismatch: index covers {covered:,} of {len(base_clips):,} "
                         "frozen clips -- index and split file disagree, refusing to proceed")

    # ------------------------------------------------- vessel-level assignment
    by_class: dict[str, list[str]] = defaultdict(list)
    for v, c in vessel_class.items():
        by_class[c].append(v)

    new_rows: list[tuple[str, int, str, str]] = []
    summary: list[dict] = []
    dropped_all: list[str] = []
    clip_rec: dict[str, str] = {r["clip_id"]: r["rec_key"] for r in rows
                                if r["rec_key"] in rec_meta}
    seed_test_clips: dict[int, frozenset] = {}
    for seed in range(args.first_seed, args.first_seed + args.n_seeds):
        rng = random.Random(stable_seed(args.protocol, seed))
        assign: dict[str, str] = {}
        dropped: list[str] = []
        for cls, vs in sorted(by_class.items()):
            vs = sorted(vs)
            rng.shuffle(vs)               # THE random draw: without this line every
            n = len(vs)                   # seed produces the SAME deterministic split
            if n < 3:
                dropped.append(f"{cls}:{n}")
                for v in vs:
                    assign[v] = "train"
                continue
            n_train = int(round(n * TRAIN_FRAC))
            n_val = int(round(n * VAL_FRAC))
            if n >= 3 and n - n_train - n_val < 1:
                n_val = max(0, n - n_train - 1)
            for i, v in enumerate(vs):
                assign[v] = "train" if i < n_train else (
                    "val" if i < n_train + n_val else "test")
        dropped_all = dropped
        for rec, v in vessel_of.items():
            s = assign[v]
            for clip in rec_clips[rec]:
                new_rows.append((args.protocol, seed, clip, s))

        # per-seed summary + the two leak checks that must both be zero
        cnt: Counter = Counter()
        rec_seen: dict[str, set] = defaultdict(set)
        ves_seen: dict[str, set] = defaultdict(set)
        for p, sd, clip, s in new_rows:
            if sd != seed:
                continue
            cnt[s] += 1
            rec = clip_rec[clip]
            rec_seen[s].add(rec)
            ves_seen[s].add(vessel_of[rec])
        rec_overlap = len(set(rec_seen["train"]) & set(rec_seen["test"]))
        ves_overlap = len(set(ves_seen["train"]) & set(ves_seen["test"]))
        summary.append({"seed": seed, "clips": dict(cnt),
                        "vessels": {s: len(ves_seen[s]) for s in ("train", "val", "test")},
                        "rec_overlap": rec_overlap, "vessel_overlap": ves_overlap})
        if rec_overlap or ves_overlap:
            raise SystemExit(f"seed {seed}: overlap rec={rec_overlap} vessel={ves_overlap} "
                             "-- refusing to write a leaky split")
        # determinism self-check: two seeds producing the same test set means the
        # shuffle is broken (this exact regression shipped once and burned 5h of
        # GPU on 16 copies of one non-random split)
        test_set = frozenset(c for p, sd, c, s in new_rows
                             if sd == seed and s == "test")
        for prev_seed, prev in seed_test_clips.items():
            if prev == test_set:
                raise SystemExit(f"seed {seed} reproduces seed {prev_seed}'s test set "
                                 "verbatim -- the draw is not random; refusing to write")
        seed_test_clips[seed] = test_set

    # ------------------------------------------------------------------ write
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["protocol", "seed", "clip_id", "split"])
        for r in base_rows:
            w.writerow([r["protocol"], r["seed"], r["clip_id"], r["split"]])
        for row in new_rows:
            w.writerow(row)
    print(f"[out  ] {len(base_rows) + len(new_rows):,} rows -> {out}")

    lines = ["=" * 78, "OceanShip vessel_disjoint extension (w1_02c)", "=" * 78,
             f"index          : {args.index}",
             f"coverage       : {len(base_clips):,} frozen clips, "
             f"{len(vessel_class)} vessels",
             f"seeds          : {args.first_seed}..{args.first_seed + args.n_seeds - 1}",
             f"dropped (<3 vessels/class): {dropped_all or 'none'}", ""]
    for s in summary:
        lines.append(f"  seed {s['seed']}: clips {s['clips']}  vessels {s['vessels']}  "
                     f"rec_overlap={s['rec_overlap']} vessel_overlap={s['vessel_overlap']}")
    lines += ["", f"written: {out}", "next: pass this file to the runner via --split-file"]
    text = "\n".join(lines)
    print()
    print(text)
    Path(str(out).rsplit(".csv.gz", 1)[0] + ".report.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
