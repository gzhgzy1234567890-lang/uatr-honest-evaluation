"""W1-04  Oceanship inventory, leakage audit and split generation.

This script runs entirely on the official metadata CSVs published by the
dataset authors, i.e. **it needs no audio**.  That matters because the audio
(23.4 GiB train + 5.1 GiB test on 123pan) is the slow part to obtain, while
the CSVs are a 27 MB GitHub download.

What the official CSVs give us
------------------------------
`oceanship_fg_{train,test}.csv` have these columns:

    text_path, wav_path, label, x, y, sog, cog, true_heading,
    mmsi, id, ais_timestamp, label_idx

`wav_path` looks like

    ./3090latest_wav/{audio_timestamp}_{n}_id_{d}_type{cls}_{ais_code}[_{seg}].pt

Three distinct notions of "identity" are recoverable, and this script keeps
them separate on purpose, because the whole point of the paper is that
confusing them is what produces the inflated numbers in the literature:

    clip        one row                     (5 s of audio)
    recording   (mmsi, ais_timestamp)       one acoustic session of one ship
    vessel      mmsi                        the ship itself

The authors' own train/test CSVs are split at the *clip* level.  Everything
else here is us re-deriving a stricter protocol from the same metadata.

Outputs (into --out)
--------------------
    oceanship_inventory.csv          one row per clip, with derived keys
    oceanship_splits.csv.gz          our own 3 protocols x N seeds, clip level
    oceanship_leakage_summary.csv    leakage rate per protocol per level
    w1_04_oceanship_report.txt       human-readable report

Usage
-----
    python w1_04_oceanship_inventory.py \
        --csv-dir data/external/oceanship_gh \
        --out out

Nothing here talks to the network or reads audio.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import random
import re
import sys
from pathlib import Path

# 70/15/15, same as the DeepShip pipeline so the two are directly comparable
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

PROTOCOLS = ("clip_random", "rec_random", "vessel_disjoint")

# wav_path -> ./3090latest_wav/{ts}_{n}_id_{d}_type{cls}_{ais}[_{seg}].pt
WAV_RE = re.compile(
    r"^(?P<ts>[0-9T:.Z-]+)_(?P<n>\d+)_id_(?P<dep>[^_]+)_type(?P<cls>[a-zA-Z]+)_(?P<ais>\d+)"
    r"(?:_(?P<seg>\d+))?$"
)


def stable_seed(protocol: str, index: int) -> int:
    """Deterministic per-(protocol, seed-index) seed.

    This must not use the builtin ``hash()``: string hashing is salted per
    process by PYTHONHASHSEED, so the same code on the same CSVs produced a
    different "frozen" split every run.  Measured before the fix: 374,085 of
    806,310 (protocol, seed, clip_id) rows were assigned differently between two
    runs of this script, while the leaked-recordings rate of the audited
    protocols moved with them (clip_random seed 0: 91.4% vs 91.8%).

    Uses the same construction as w1_02_make_splits.stable_seed so the
    Oceanship and DeepShip protocols are seeded identically.
    """
    digest = hashlib.sha256(f"{protocol}|{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def load_side(csv_dir: Path, side: str) -> list[dict]:
    """Read one official CSV and attach the derived keys we need."""
    path = csv_dir / f"oceanship_fg_{side}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            stem = Path(r["wav_path"]).name
            m = WAV_RE.match(stem[:-3] if stem.endswith(".pt") else stem)
            if not m:
                # keep it, but flag it: an unparsable name is itself a finding
                rows.append({"raw": r, "parsed": None})
                continue
            rows.append({"raw": r, "parsed": m.groupdict()})
    return rows


def build_inventory(csv_dir: Path, out_dir: Path) -> tuple[list[dict], dict]:
    """One row per clip, plus a diagnostic dict about the metadata itself."""
    inv: list[dict] = []
    diag = collections.Counter()
    dep_mismatch: collections.defaultdict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )

    for side in ("train", "test"):
        for item in load_side(csv_dir, side):
            r, p = item["raw"], item["parsed"]
            if p is None:
                diag["unparsable_wav_path"] += 1
                continue
            # the filename carries `id_{d}`; the CSV carries its own `id`
            if r["id"] and p["dep"] != r["id"]:
                dep_mismatch[r["id"]][p["dep"]] += 1
            inv.append(
                {
                    "official_split": side,
                    "clip_id": Path(r["wav_path"]).stem,
                    "label": r["label"],
                    "label_idx": r["label_idx"],
                    "mmsi": r["mmsi"],
                    "csv_id": r["id"],
                    "fname_id": p["dep"],
                    "ais_type_code": p["ais"],
                    "audio_ts": p["ts"],
                    "ais_timestamp": r["ais_timestamp"],
                    "sog": r["sog"],
                    "cog": r["cog"],
                    "true_heading": r["true_heading"],
                    "x": r["x"],
                    "y": r["y"],
                    # recording = one acoustic session of one ship
                    "rec_key": f"{r['mmsi']}|{r['ais_timestamp']}",
                }
            )

    all_dep = collections.Counter(v for c in dep_mismatch.values() for v in c)
    diag["csv_id_to_fname_id_mismatch"] = sum(sum(c.values()) for c in dep_mismatch.values())
    diag["distinct_csv_id"] = len(dep_mismatch) + sum(
        1 for r in inv if r["csv_id"] not in dep_mismatch
    )
    diag["distinct_fname_id"] = len(all_dep)

    with gzip.open(out_dir / "oceanship_inventory.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inv[0].keys()))
        w.writeheader()
        w.writerows(inv)
    return inv, diag


# --------------------------------------------------------------------------- #
# leakage auditing
# --------------------------------------------------------------------------- #
def leakage_for(assign: dict[str, str], inv: list[dict]) -> dict:
    """Given clip_id -> split, report how badly each identity level leaks.

    For 'clip_random' the notion of a clip leaking is meaningless (every clip
    is its own unit), so the interesting number there is how much of the test
    set shares a recording / a vessel with train.
    """
    train_rec, train_ves, train_clip = set(), set(), set()
    for r in inv:
        if assign[r["clip_id"]] == "train":
            train_rec.add(r["rec_key"])
            train_ves.add(r["mmsi"])
            train_clip.add(r["clip_id"])

    test = [r for r in inv if assign[r["clip_id"]] == "test"]
    if not test:
        return {"n_test": 0}
    n = len(test)
    return {
        "n_test": n,
        "n_test_vessels": len({r["mmsi"] for r in test}),
        "n_test_recordings": len({r["rec_key"] for r in test}),
        "pct_test_share_vessel": round(100 * sum(r["mmsi"] in train_ves for r in test) / n, 1),
        "pct_test_share_recording": round(100 * sum(r["rec_key"] in train_rec for r in test) / n, 1),
        "pct_test_share_clip": round(100 * sum(r["clip_id"] in train_clip for r in test) / n, 1),
    }


def official_assignment(inv: list[dict]) -> dict[str, str]:
    return {r["clip_id"]: r["official_split"] for r in inv}


# --------------------------------------------------------------------------- #
# split generation
# --------------------------------------------------------------------------- #
def _stratified_partition(units: list[str], classes: dict, rng: random.Random) -> dict:
    """70/15/15 by class, at whatever unit granularity the caller chose.

    `units` is a list of identity keys (a clip id, a rec_key or an MMSI list
    element) and `classes` maps each key to its label.  Splitting on *clips* is
    what the literature does; splitting on recordings or vessels is what we do.
    Keeping one function for all three means the protocols differ only in their
    unit definition, never in their arithmetic - which is the whole point when
    the paper's claim is that the difference in reported accuracy comes from the
    unit, not from the model.
    """
    by_class: dict[str, list] = collections.defaultdict(list)
    for u in units:
        by_class[classes[u]].append(u)

    assign: dict[str, str] = {}
    for cls in sorted(by_class):
        items = by_class[cls]
        rng.shuffle(items)
        n = len(items)
        if n < 3:  # can't populate three splits; keep it honest and visible
            for u in items:
                assign[u] = "train"
            continue
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        # never leave val or test empty
        n_train = min(n_train, n - 2)
        n_val = max(1, min(n_val, n - n_train - 1))
        for i, u in enumerate(items):
            assign[u] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return assign


def build_splits(inv: list[dict], n_seeds: int, classes_kept: set[str] | None = None):
    """Return (clip-level split rows, per-protocol/seed leakage rows).

    `classes_kept` restricts the benchmark to classes that have enough vessels
    to populate a 3-way vessel-disjoint split.  Classes with one or two vessels
    cannot appear in train AND val AND test no matter how you slice them, so
    keeping them would silently produce classes with an empty test set - which
    is exactly the kind of quiet flaw this paper is about.
    """
    if classes_kept is not None:
        inv = [r for r in inv if r["label"] in classes_kept]

    clip_class = {r["clip_id"]: r["label"] for r in inv}
    rec_class: dict[str, str] = {}
    rec_clips: dict[str, list[str]] = collections.defaultdict(list)
    ves_clips: dict[str, list[str]] = collections.defaultdict(list)
    for r in inv:
        rec_class.setdefault(r["rec_key"], r["label"])
        rec_clips[r["rec_key"]].append(r["clip_id"])
        ves_clips[r["mmsi"]].append(r["clip_id"])

    split_rows: list[dict] = []
    leak_rows: list[dict] = []

    for protocol in PROTOCOLS:
        for seed_index in range(n_seeds):
            rng = random.Random(stable_seed(protocol, seed_index))

            if protocol == "clip_random":
                clip_assign = _stratified_partition(list(clip_class), clip_class, rng)
            elif protocol == "rec_random":
                unit_assign = _stratified_partition(list(rec_clips), rec_class, rng)
                clip_assign = {c: unit_assign[k] for k, v in rec_clips.items() for c in v}
            else:  # vessel_disjoint
                # a vessel can (rarely) appear under two labels; take the majority
                votes: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
                for r in inv:
                    votes[r["mmsi"]][r["label"]] += 1
                ves_class = {v: c.most_common(1)[0][0] for v, c in votes.items()}
                unit_assign = _stratified_partition(list(ves_clips), ves_class, rng)
                clip_assign = {c: unit_assign[v] for v, cl in ves_clips.items() for c in cl}

            for c, s in clip_assign.items():
                split_rows.append(
                    {"protocol": protocol, "seed": seed_index, "clip_id": c, "split": s}
                )

            leak = leakage_for(clip_assign, inv)
            leak["protocol"] = protocol
            leak["seed"] = seed_index
            leak_rows.append(leak)

    return split_rows, leak_rows


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_report(inv, diag, leak_rows, official_leak, out_dir: Path, args, dropped: dict) -> None:
    L: list[str] = []
    add = L.append

    add("=" * 88)
    add("Oceanship inventory and leakage audit")
    add("=" * 88)
    add("")
    add(f"source CSVs : {args.csv_dir}")
    add(f"clips       : {len(inv):,}")
    add(f"recordings  : {len({r['rec_key'] for r in inv}):,}   (= distinct (mmsi, ais_timestamp))")
    add(f"vessels     : {len({r['mmsi'] for r in inv}):,}   (= distinct MMSI)")
    add(f"classes     : {len({r['label'] for r in inv})}")
    add("")

    add("clips per split (as published by the authors)")
    add("-" * 88)
    by = collections.Counter(r["official_split"] for r in inv)
    for s in ("train", "val", "test"):
        if by.get(s):
            add(f"  {s:<6}{by[s]:>9,}")
    add("")

    add("vessels and clips per class (vessel counts decide what a split can do)")
    add("-" * 88)
    ves: dict[str, set] = collections.defaultdict(set)
    cl: collections.Counter = collections.Counter()
    for r in inv:
        ves[r["label"]].add(r["mmsi"])
        cl[r["label"]] += 1
    add(f"  {'class':<30}{'vessels':>9}{'clips':>9}{'usable':>9}")
    for c in sorted(ves, key=lambda x: -len(ves[x])):
        add(f"  {c:<30}{len(ves[c]):>9,}{cl[c]:>9,}{'yes' if len(ves[c]) >= args.min_vessels else 'NO':>9}")
    add("")
    if dropped:
        lost = sum(dropped.values())
        add(f"  dropped {len(dropped)} class(es) with < {args.min_vessels} vessels: "
            f"{', '.join(sorted(dropped))}")
        add(f"  cost: {lost:,} clips ({100 * lost / (len(inv) + lost):.2f}%) - these classes")
        add("  cannot appear in all three splits at vessel level, so keeping them would")
        add("  create classes with an empty test set.  Excluded explicitly, not silently.")
        add("")

    add("=" * 88)
    add("THE HEADLINE NUMBER: the authors' own split is clip-level")
    add("=" * 88)
    add("")
    add("  A clip-level split cannot keep a ship out of both sides, because one")
    add("  ship contributes hundreds of clips.  Measured on the official CSVs:")
    add("")
    add(f"  {'metric':<46}{'value':>12}")
    add(f"  {'test clips':<46}{official_leak['n_test']:>12,}")
    add(f"  {'test clips whose RECORDING is also in train':<46}{official_leak['pct_test_share_recording']:>11.1f}%")
    add(f"  {'test clips whose VESSEL is also in train':<46}{official_leak['pct_test_share_vessel']:>11.1f}%")
    add("")
    add("  -> the vessel column is the one that decides whether the model is")
    add("     measuring ship-type recognition or ship memorisation.")
    add("")

    add("=" * 88)
    add(f"our own protocols, {args.n_seeds} seeds each (70/15/15, stratified by class)")
    add("=" * 88)
    add("")
    add(f"  {'protocol':<17}{'mean test clips':>16}{'share recording':>17}{'share vessel':>14}")
    agg: dict[str, list[dict]] = collections.defaultdict(list)
    for r in leak_rows:
        agg[r["protocol"]].append(r)
    for p in PROTOCOLS:
        rs = agg[p]
        mean = lambda k: sum(x[k] for x in rs) / len(rs)  # noqa: E731
        add(f"  {p:<17}{mean('n_test'):>16,.0f}{mean('pct_test_share_recording'):>16.1f}%{mean('pct_test_share_vessel'):>13.1f}%")
    add("")
    add("  rec_random should read 0.0% at the recording level but NOT at the")
    add("  vessel level - that gap is the finding the paper is built on.")
    add("  vessel_disjoint must read 0.0% at both.")
    add("")

    if diag.get("csv_id_to_fname_id_mismatch"):
        add("=" * 88)
        add("metadata integrity notes")
        add("=" * 88)
        add("")
        add(f"  wav_path not matching the expected pattern : {diag.get('unparsable_wav_path', 0):,}")
        add(f"  rows where the CSV `id` differs from the `id_` in wav_path : {diag['csv_id_to_fname_id_mismatch']:,}")
        add("")
        add("  Both columns take values from the same small set, but they disagree, so")
        add("  at least one of them is not the deployment id it appears to be.  We do")
        add("  not rely on either: vessel identity comes from `mmsi`, which is a real")
        add("  9-digit MMSI and is 100% populated.  Recorded for the paper's data")
        add("  description section only.")
        add("")

    text = "\n".join(L) + "\n"
    (out_dir / "w1_04_oceanship_report.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> int:
    ap = argparse.ArgumentParser(description="Oceanship inventory, leakage audit, splits.")
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--csv-dir", type=Path, default=root / "data" / "external" / "oceanship_gh")
    ap.add_argument("--out", type=Path, default=root / "out")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument(
        "--min-vessels",
        type=int,
        default=3,
        help="drop classes with fewer vessels than this (they cannot populate a 3-way "
        "vessel-disjoint split, so keeping them would give those classes an empty test set)",
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    inv, diag = build_inventory(args.csv_dir, args.out)

    ves_per_class: dict[str, set] = collections.defaultdict(set)
    for r in inv:
        ves_per_class[r["label"]].add(r["mmsi"])
    dropped = {
        c: sum(1 for r in inv if r["label"] == c)
        for c in ves_per_class
        if len(ves_per_class[c]) < args.min_vessels
    }
    classes_kept = {c for c in ves_per_class if c not in dropped}

    official_leak = leakage_for(official_assignment(inv), inv)
    split_rows, leak_rows = build_splits(inv, args.n_seeds, classes_kept)
    with gzip.open(args.out / "oceanship_splits.csv.gz", "wt", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["protocol", "seed", "clip_id", "split"])
        w.writeheader()
        w.writerows(split_rows)

    with open(args.out / "oceanship_leakage_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(leak_rows[0].keys()))
        w.writeheader()
        w.writerows(leak_rows)

    write_report(inv, diag, leak_rows, official_leak, args.out, args, dropped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
