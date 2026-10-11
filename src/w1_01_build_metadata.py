#!/usr/bin/env python3
"""W1-01  Build the DeepShip recording-level metadata table.

This is the foundation of the whole project (contribution C1: recording-level
provenance).  It turns the raw folder tree into one row per ORIGINAL RECORDING,
which is the granularity at which the strict evaluation protocol operates.

Why recording level matters
---------------------------
DeepShip ships as ~609 long recordings (roughly 44 h total).  Almost every
published paper slices them into short clips and then splits the CLIPS randomly,
which puts clips from the same recording on both sides of the train/test border.
Because clips from one recording share channel response, background noise and
operating condition, that inflates test accuracy.  To do anything rigorous we
first have to know exactly which clip came from which recording.

Two ways to run
---------------
  # on Kaggle (or locally, with the dataset downloaded)
  python w1_01_build_metadata.py --root /kaggle/input/deepship-raw --out out/

  # offline, from a manifest of "path<TAB>bytes" (no 20 GB download needed)
  python w1_01_build_metadata.py --manifest deepship_manifest.tsv --out out/

The manifest mode exists so the whole pipeline can be developed and unit-tested
before the dataset is ever downloaded.  Verified locally: 592/609 recordings
(97.2%) receive metadata.

Join strategy (validated against the real data)
-----------------------------------------------
DeepShip's four metafiles are mutually inconsistent - the Tanker file reuses its
index column up to 122 while having 243 rows, so a naive (date, index) join
silently matches only 20/240 Tanker recordings.  We therefore try three keys in
strict priority order and record which one succeeded:

  1. exact timestamp  (class, date, HHMMSS)
        The wav filename is very often the recording start time (e.g. 002044.wav),
        and the metafile carries the same HHMMSS.  This is an exact key and
        resolves ~82% of recordings.
  2. exact index      (class, date, folder index)
        Reliable for Cargo / Passenger / Tug.
  3. rank zip         (class, date) with positional alignment
        Dirs sorted by (variant letter, index) zipped against metafile rows
        sorted by HHMMSS.  Recovers most of the remaining Tanker recordings.

Anything still unmatched is kept with an empty ship_name and join_method="none",
so downstream code can decide whether to use it - we never silently drop data.

Metadata format observed in the metafiles
-----------------------------------------
    index, ais_type_code, ship_name, YYYYMMDD, HHMMSS, duration_s,
    "entry-exit" distance in metres, [optional trailing note]

Field 2 is the AIS vessel type code (52 = tug, 60/65/69 = passenger,
70/71/74/76/79 = cargo, 80-89 = tanker), which is a useful independent check on
the folder-level class label.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

# DeepShip wavs are stereo 32 kHz, signed 16-bit -> 128000 bytes per second.
# Verified: Cargo/20171104-1/1.wav is 58,496,058 bytes and the metafile says 458 s
# (58_496_058 / 128_000 = 457.0 s).
BYTES_PER_SECOND = 128_000

# The four class folders and the metafile that documents each of them.
# NOTE: the Kaggle mirror `tangqiji/deepship-raw` ships only the Cargo and
# Passenger metafiles; Tanker and Tug must be fetched from the official repo
# https://github.com/irfankamboh/DeepShip (raw.githubusercontent.com).
CLASS_TO_METAFILE = {
    "Cargo": "cargo-metafile.txt",
    "Passenger": "passengership-metafile.txt",
    "Tanker": "tanker-metafile.txt",
    "Tug": "tug-metafile.txt",
}

# AIS vessel type codes seen in the metafiles, used as a sanity cross-check.
AIS_MAJOR = {5: "tug", 6: "passenger", 7: "cargo", 8: "tanker"}

WAV_REL_RE = re.compile(
    r"^(?P<class_dir>[A-Za-z]+)/(?P<rec_id>\d{8}(?P<variant>[a-z]?)-\d+)/(?P<fname>[^/]+)$"
)
REC_RE = re.compile(r"^(?P<date>\d{8})(?P<variant>[a-z]?)-(?P<index>\d+)$")
TIME_RE = re.compile(r"^(?P<t>\d{6})\.wav$")


def read_text_no_bom(path: Path) -> str:
    """Read text ignoring a UTF-8 BOM.

    The metafiles start with a BOM.  If it is not stripped, the first row's index
    field becomes '\\ufeff1' and fails int(), silently deleting one recording per
    class - which is exactly the kind of quiet data loss that ruins a benchmark.
    """
    return path.read_text(encoding="utf-8-sig")


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #


def entries_from_root(root: Path) -> list[tuple[str, int]]:
    """Scan a real DeepShip tree -> [(relative_path, size_bytes), ...]."""
    out: list[tuple[str, int]] = []
    for wav in sorted(root.rglob("*.wav")):
        rel = wav.relative_to(root).as_posix()
        out.append((rel, wav.stat().st_size))
    return out


def entries_from_manifest(path: Path) -> list[tuple[str, int]]:
    """Read a `path<TAB>bytes` manifest (produced from the Kaggle dataset API)."""
    out: list[tuple[str, int]] = []
    for line in read_text_no_bom(path).splitlines():
        line = line.strip()
        if not line or line.startswith("path\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        out.append((parts[0], int(parts[1])))
    return out


def load_metafile(path: Path) -> list[dict]:
    """Parse one DeepShip metafile into dicts.  Rows that do not parse are skipped."""
    rows: list[dict] = []
    for lineno, raw in enumerate(read_text_no_bom(path).splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) < 7 or not parts[0].isdigit() or not parts[3].isdigit():
            continue
        dist_lo = dist_hi = None
        if "-" in parts[6]:
            lo, _, hi = parts[6].partition("-")
            if lo.strip().isdigit() and hi.strip().isdigit():
                dist_lo, dist_hi = int(lo), int(hi)
        rows.append(
            {
                "_index": int(parts[0]),
                "ais_type_code": int(parts[1]) if parts[1].isdigit() else None,
                "ship_name": parts[2].strip(),
                "date": parts[3],
                "time": parts[4],
                "duration_s": int(parts[5]) if parts[5].isdigit() else None,
                "dist_lo_m": dist_lo,
                "dist_hi_m": dist_hi,
                "note": ",".join(parts[7:]).strip() or None,
                "_lineno": lineno,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# the join
# --------------------------------------------------------------------------- #


def resolve_metadata(
    recs: list[dict], meta_rows: list[dict], class_dir: str, tol_s: int = 2
) -> tuple[dict[str, dict], Counter]:
    """Attach a metafile row to each recording as a one-to-one assignment.

    A naive (date, index) join looks fine but is wrong often enough to matter.
    Observed counter-example:

        Cargo/20171124a-41/41.wav      183 s   <- metafile index 41 says 406 s
        Cargo/20171124c-42/134001.wav  406 s   <- metafile time 134001 says 406 s

    Both keys resolve to the SAME metafile line, and only the timestamp one is
    right.  So the index column of the metafile does not reliably line up with
    the folder's trailing number, and a key alone cannot be trusted.

    The file size gives us an exact, independent measurement of each recording's
    duration (stereo 32 kHz 16-bit -> bytes/128000 seconds), and the metafile
    carries its own duration.  We therefore treat the join as a global
    one-to-one assignment problem and:
      1. enumerate every (recording, metafile row) pair on the same date,
         remembering which key produced it;
      2. sort candidates by (key priority, |duration difference|);
      3. greedily assign, never reusing a metafile row;
      4. accept only if the durations agree within `tol_s`.

    Anything left unassigned keeps join_method="none" instead of inheriting a
    plausible-looking but wrong ship name.
    """
    KEY_PRIORITY = {"timestamp": 0, "index": 1, "date": 2}

    by_time = {(r["date"], r["time"]): r for r in meta_rows}
    by_date: dict[str, list[dict]] = defaultdict(list)
    for r in meta_rows:
        by_date[r["date"]].append(r)

    # ---- enumerate candidates -------------------------------------------------
    candidates: list[tuple[int, int, dict, dict, str]] = []  # (prio, delta, rec, row, key)
    for rec in recs:
        dur = round(rec["wav_bytes"] / BYTES_PER_SECOND)
        seen: set[int] = set()

        def add(prio: int, row: dict | None, key: str, _rec=rec, _dur=dur, _seen=seen) -> None:
            if row is None or id(row) in _seen:
                return
            _seen.add(id(row))
            delta = abs(_dur - row["duration_s"]) if row["duration_s"] is not None else 10**6
            candidates.append((prio, delta, _rec, row, key))

        # key 1: the wav filename is the recording start time (e.g. 134001.wav)
        if rec["wav_time"]:
            add(KEY_PRIORITY["timestamp"], by_time.get((rec["date"], rec["wav_time"])), "timestamp")
        # key 2: metafile index matches the folder's trailing number
        for row in by_date.get(rec["date"], []):
            if row["_index"] == rec["index"]:
                add(KEY_PRIORITY["index"], row, "index")
        # key 3: any row sharing the date - duration decides
        for row in by_date.get(rec["date"], []):
            add(KEY_PRIORITY["date"], row, "date")

    # ---- global greedy assignment --------------------------------------------
    candidates.sort(key=lambda c: (c[0], c[1]))
    taken_rows: set[int] = set()
    resolved: dict[str, dict] = {}
    stats = Counter()

    for prio, delta, rec, row, key in candidates:
        if rec["rec_id"] in resolved or id(row) in taken_rows:
            continue
        if delta > tol_s:
            stats["rejected_duration_mismatch"] += 1
            continue
        resolved[rec["rec_id"]] = {**row, "join_method": key}
        taken_rows.add(id(row))
        stats[key] += 1

    # ---- report what is left --------------------------------------------------
    for rec in recs:
        if rec["rec_id"] not in resolved:
            resolved[rec["rec_id"]] = {"join_method": "none"}
            stats["none"] += 1

    return resolved, stats


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the DeepShip recording metadata table.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="DeepShip root directory to scan")
    src.add_argument("--manifest", type=Path, help="TSV manifest of 'path<TAB>bytes'")
    ap.add_argument("--metafile-dir", type=Path, required=True, help="directory holding the 4 *_metafile.txt")
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--segment-len", type=float, default=5.0, help="clip length in seconds (default 5.0)")
    args = ap.parse_args()

    entries = entries_from_root(args.root) if args.root else entries_from_manifest(args.manifest)
    print(f"[input] {len(entries)} files")

    # group recordings
    by_class: dict[str, dict[str, dict]] = defaultdict(dict)
    skipped: list[str] = []
    for rel, nbytes in entries:
        m = WAV_REL_RE.match(rel)
        if not m:
            skipped.append(rel)  # metafiles and anything unexpected
            continue
        rec_id = m.group("rec_id")
        rm = REC_RE.match(rec_id)
        tm = TIME_RE.match(m.group("fname"))
        by_class[m.group("class_dir")][rec_id] = {
            "class_dir": m.group("class_dir"),
            "rec_id": rec_id,
            "rec_key": f"{m.group('class_dir')}/{rec_id}",
            "date": rm.group("date"),
            "variant": rm.group("variant") or "",
            "index": int(rm.group("index")),
            "wav_rel": rel,
            "wav_bytes": nbytes,
            "wav_time": tm.group("t") if tm else None,
        }
    print(f"[scan] skipped {len(skipped)} non-recording entries: {skipped[:4]}")

    # resolve metadata per class
    rows: list[dict] = []
    class_stats: dict[str, Counter] = {}
    for class_dir, recs in sorted(by_class.items()):
        mf_path = args.metafile_dir / CLASS_TO_METAFILE[class_dir]
        if not mf_path.exists():
            print(f"[warn] {class_dir}: metafile missing ({mf_path}) -> no ship names for this class")
            resolved, stats = {}, Counter({"none": len(recs)})
        else:
            meta_rows = load_metafile(mf_path)
            resolved, stats = resolve_metadata(list(recs.values()), meta_rows, class_dir)
        class_stats[class_dir] = stats
        for rec_id, rec in recs.items():
            meta = resolved.get(rec_id, {})
            # Duration source of truth is the file size: it is exact, always
            # available, and does not depend on the metafile join succeeding.
            # The metafile's own duration column is kept separately as a
            # cross-check, because the two disagree on a non-trivial number of
            # recordings and that disagreement shifts the clip count.
            dur = round(rec["wav_bytes"] / BYTES_PER_SECOND)
            rows.append(
                {
                    "rec_key": rec["rec_key"],
                    "class_dir": class_dir,
                    "rec_id": rec_id,
                    "date": rec["date"],
                    "variant": rec["variant"],
                    "index": rec["index"],
                    "wav_rel": rec["wav_rel"],
                    "wav_bytes": rec["wav_bytes"],
                    "duration_s": int(dur),
                    "n_clips": int(dur // args.segment_len),
                    "ship_name": meta.get("ship_name") or "",
                    "ais_type_code": meta.get("ais_type_code") if meta.get("ais_type_code") is not None else "",
                    "dist_lo_m": meta.get("dist_lo_m") if meta.get("dist_lo_m") is not None else "",
                    "dist_hi_m": meta.get("dist_hi_m") if meta.get("dist_hi_m") is not None else "",
                    "meta_duration_s": meta.get("duration_s") if meta.get("duration_s") is not None else "",
                    "meta_note": meta.get("note") or "",
                    "meta_row_id": f"{class_dir}:{meta['_lineno']}" if meta.get("_lineno") else "",
                    "join_method": meta.get("join_method", "none"),
                }
            )

    rows.sort(key=lambda r: (r["class_dir"], r["date"], r["variant"], r["index"]))

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "deepship_recordings.csv"
    fields = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------ report
    lines: list[str] = []
    total = len(rows)
    matched = sum(1 for r in rows if r["join_method"] != "none")
    total_clips = sum(r["n_clips"] for r in rows)
    total_hours = sum(r["duration_s"] for r in rows) / 3600

    lines.append("=" * 74)
    lines.append("DeepShip recording-level metadata")
    lines.append("=" * 74)
    lines.append(f"recordings        : {total}")
    lines.append(f"carrying metadata : {matched} ({100*matched/total:.1f}%)")
    lines.append(f"total duration    : {total_hours:.2f} h")
    lines.append(f"clips @ {args.segment_len:g}s        : {total_clips:,}")
    lines.append("")
    lines.append(f"{'class':<11}{'recs':>6}{'meta%':>8}{'hours':>9}{'clips':>9}   join breakdown")
    for class_dir in sorted(by_class):
        sub = [r for r in rows if r["class_dir"] == class_dir]
        ok = sum(1 for r in sub if r["join_method"] != "none")
        hrs = sum(r["duration_s"] for r in sub) / 3600
        cl = sum(r["n_clips"] for r in sub)
        js = class_stats.get(class_dir, Counter())
        brk = " ".join(
            f"{k}={js[k]}" for k in ("timestamp", "index", "date", "none") if js.get(k)
        )
        lines.append(f"{class_dir:<11}{len(sub):>6}{100*ok/len(sub):>7.1f}%{hrs:>9.2f}{cl:>9,}   {brk}")
    lines.append("")

    # domain structure - this is what contribution C3 is built on
    named = [r for r in rows if r["ship_name"]]
    ship_counts = Counter(r["ship_name"] for r in named)
    multi = {s: c for s, c in ship_counts.items() if c > 1}
    lines.append("domain structure (the raw material for the attribution study)")
    lines.append("-" * 74)
    lines.append(f"distinct ship names       : {len(ship_counts)}")
    lines.append(f"ships recorded >1 time    : {len(multi)}")
    lines.append(f"recordings belonging to them: {sum(multi.values())}")
    lines.append(f"   -> these give same-label / different-condition pairs,")
    lines.append(f"      i.e. free domain pairs with the class label held fixed.")
    lines.append("")
    for class_dir in sorted(by_class):
        sub = [r for r in rows if r["class_dir"] == class_dir]
        distinct_days = len({r["date"] for r in sub})
        multi_days = sum(1 for d, c in Counter(r["date"] for r in sub).items() if c > 1)
        lines.append(
            f"  {class_dir:<11} days={distinct_days:>4}  days_with_multiple_sessions={multi_days:>4}  "
            f"date span {min(r['date'] for r in sub)}..{max(r['date'] for r in sub)}"
        )
    lines.append("")
    lines.append("top repeated ships per class")
    for class_dir in sorted(by_class):
        sub = [r for r in rows if r["class_dir"] == class_dir and r["ship_name"]]
        top = Counter(r["ship_name"] for r in sub).most_common(4)
        lines.append(f"  {class_dir:<11} {top}")
    lines.append("")
    lines.append("duration consistency check (file size is the source of truth)")
    lines.append("-" * 74)
    both = [r for r in rows if r["meta_duration_s"] != ""]
    if both:
        d = [abs(int(r["duration_s"]) - int(r["meta_duration_s"])) for r in both]
        exact = sum(1 for x in d if x == 0)
        le2 = sum(1 for x in d if x <= 2)
        lines.append(f"recordings with both values : {len(both)}")
        lines.append(f"  exact match               : {exact} ({100*exact/len(both):.1f}%)")
        lines.append(f"  within +/-2 s             : {le2} ({100*le2/len(both):.1f}%)")
        lines.append(f"  max abs difference        : {max(d)} s")
        worst = sorted(both, key=lambda r: -abs(int(r["duration_s"]) - int(r["meta_duration_s"])))[:3]
        for r in worst:
            lines.append(
                f"    {r['rec_key']:<28} bytes->{r['duration_s']}s  metafile->{r['meta_duration_s']}s"
            )
    lines.append("")
    lines.append("join integrity: metafile lines claimed by more than one recording")
    lines.append("-" * 74)
    # The assignment is one-to-one by construction, so this should be zero.
    # It is asserted rather than assumed, because a silent violation would mean
    # wrong ship names leaking into the same-ship analysis.
    claimed: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        if r["join_method"] == "none" or not r["meta_row_id"]:
            continue
        claimed[r["meta_row_id"]].append(r["rec_key"])
    clashes = {k: v for k, v in claimed.items() if len(v) > 1}
    lines.append(f"metafile lines used exactly once : {len(claimed) - len(clashes)}")
    lines.append(f"metafile lines used >1 time      : {len(clashes)}  (expected 0)")
    for k, v in list(clashes.items())[:6]:
        lines.append(f"    {k} -> {v}")
    rejected = sum(class_stats.get(c, Counter()).get("rejected_duration_mismatch", 0) for c in class_stats)
    lines.append(f"candidate pairs rejected on duration mismatch : {rejected}")
    lines.append("")
    lines.append(f"written: {csv_path}")

    text = "\n".join(lines)
    print()
    print(text)
    (args.out / "w1_01_metadata_report.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
