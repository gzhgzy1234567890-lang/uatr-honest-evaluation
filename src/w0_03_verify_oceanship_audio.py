"""W0-03  Verify the Oceanship audio you downloaded against the official CSVs.

Why this exists
---------------
The Oceanship release has a documented history of partial or mismatched
downloads.  Open (and unanswered) reports on the official repo include
"why do I only have 12 of the classes", "the files in the link are invalid",
and "train parts 6-9 may not have been uploaded, the count does not match the
paper".  None of those were ever answered by the authors.

So rather than trusting the download, we check it against the one thing the
authors did publish cleanly: the per-clip CSV manifests.  Those give an exact
expected list of 45,699 train + 8,071 test clips, each with its class and its
MMSI.  Diffing your extracted files against that list turns "did the download
work?" into a yes/no, and tells you exactly which classes and which vessels
you are missing.

Matching rule
-------------
The CSV `wav_path` column points at the authors' own preprocessing output
(`./3090latest_wav/<stem>.pt`), which is not what is inside the download.  The
audio files share the stem, so we match on **filename stem**, ignoring the
directory and the extension.  That is robust to however the zips nest.

Usage
-----
    python w0_03_verify_oceanship_audio.py \
        --audio-root E:/oceanship/extracted \
        --csv-dir    data/external/oceanship_gh \
        --out        out

Exit code is 0 when every expected clip is present, 1 otherwise, so it can be
used as a gate before uploading to Kaggle.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

AUDIO_SUFFIXES = {".wav", ".WAV", ".flac", ".pt", ".npy"}


def expected_clips(csv_dir: Path) -> dict[str, dict]:
    """stem -> {split, label, mmsi, rec_key} for every clip the authors list."""
    want: dict[str, dict] = {}
    for side in ("train", "test"):
        path = csv_dir / f"oceanship_fg_{side}.csv"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        with open(path, newline="", encoding="utf-8", errors="replace") as fh:
            for r in csv.DictReader(fh):
                stem = Path(r["wav_path"]).stem
                want[stem] = {
                    "official_split": side,
                    "label": r["label"],
                    "mmsi": r["mmsi"],
                    "rec_key": f"{r['mmsi']}|{r['ais_timestamp']}",
                }
    return want


def found_clips(audio_root: Path) -> dict[str, int]:
    """stem -> byte size, for every audio-looking file under the root."""
    found: dict[str, int] = {}
    for p in audio_root.rglob("*"):
        if p.is_file() and p.suffix in AUDIO_SUFFIXES:
            found.setdefault(p.stem, p.stat().st_size)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description="Diff downloaded Oceanship audio against the official CSVs.")
    root = Path(__file__).resolve().parent.parent
    ap.add_argument("--audio-root", type=Path, required=True, help="where you extracted the zips")
    ap.add_argument("--csv-dir", type=Path, default=root / "data" / "external" / "oceanship_gh")
    ap.add_argument("--out", type=Path, default=root / "out")
    ap.add_argument("--report-missing", type=int, default=25, help="how many missing stems to print per class")
    args = ap.parse_args()

    if not args.audio_root.exists():
        raise SystemExit(f"audio root not found: {args.audio_root}")
    args.out.mkdir(parents=True, exist_ok=True)

    want = expected_clips(args.csv_dir)
    got = found_clips(args.audio_root)

    missing = {k: v for k, v in want.items() if k not in got}
    extra = [k for k in got if k not in want]

    L: list[str] = []
    add = L.append
    add("=" * 88)
    add("Oceanship audio verification against the official CSVs")
    add("=" * 88)
    add("")
    add(f"audio root      : {args.audio_root}")
    add(f"csv dir         : {args.csv_dir}")
    add(f"expected clips  : {len(want):,}")
    add(f"files found     : {len(got):,}")
    add(f"matched         : {len(want) - len(missing):,}  ({100 * (len(want) - len(missing)) / len(want):.2f}%)")
    add(f"missing         : {len(missing):,}")
    add(f"unrecognised    : {len(extra):,}   (files not named in any official CSV)")
    add("")

    if not missing and not extra:
        add("RESULT: complete and exact.  Safe to upload.")
    else:
        add("RESULT: incomplete.  Read the breakdown below before uploading.")
    add("")

    # -- what is missing, grouped the way that actually matters -------------
    if missing:
        add("missing clips by official split")
        add("-" * 88)
        for s in ("train", "test"):
            n = sum(1 for v in missing.values() if v["official_split"] == s)
            tot = sum(1 for v in want.values() if v["official_split"] == s)
            add(f"  {s:<6}{n:>8,} / {tot:>8,} missing")
        add("")

        add("missing clips by class  (a whole row missing is the failure mode to look for)")
        add("-" * 88)
        tot_cls = collections.Counter(v["label"] for v in want.values())
        miss_cls = collections.Counter(v["label"] for v in missing.values())
        add(f"  {'class':<30}{'expected':>10}{'missing':>10}{'kept':>9}")
        for c, n in tot_cls.most_common():
            m = miss_cls.get(c, 0)
            mark = "  <-- ENTIRELY MISSING" if m == n else ""
            add(f"  {c:<30}{n:>10,}{m:>10,}{n - m:>9,}{mark}")
        add("")

        add("missing clips by vessel (MMSI) - the number that decides whether vessel-disjoint splits still hold")
        add("-" * 88)
        tot_mm = {v["mmsi"] for v in want.values()}
        got_mm = {want[k]["mmsi"] for k in want if k in got}
        lost = tot_mm - got_mm
        add(f"  vessels expected        : {len(tot_mm):,}")
        add(f"  vessels with >=1 clip   : {len(got_mm):,}  ({100 * len(got_mm) / len(tot_mm):.1f}%)")
        add(f"  vessels with NO clips   : {len(lost):,}")
        if lost:
            add(f"    e.g. {sorted(lost)[:12]}")
        add("")
        add(f"  Detail of the first {args.report_missing} missing stems per class:")
        shown: collections.Counter = collections.Counter()
        for k, v in list(missing.items()):
            if shown[v["label"]] < 3:
                add(f"    {v['label']:<28} {k}")
                shown[v["label"]] += 1
        add("")

    if extra:
        add("unrecognised files (first 15 of them)")
        add("-" * 88)
        for k in extra[:15]:
            add(f"  {k}")
        add("")
        add("  These may be background-class clips, or a newer release than the CSVs we")
        add("  hold.  Worth a look before treating them as junk.")
        add("")

    # -- machine-readable summary for the pipeline --------------------------
    summary = {
        "expected": len(want),
        "found": len(got),
        "matched": len(want) - len(missing),
        "missing": len(missing),
        "unrecognised": len(extra),
        "vessels_expected": len({v["mmsi"] for v in want.values()}),
        "vessels_with_clips": len({want[k]["mmsi"] for k in want if k in got}),
        "classes_fully_missing": sorted(
            c for c, n in collections.Counter(v["label"] for v in want.values()).items()
            if collections.Counter(v["label"] for v in missing.values()).get(c, 0) == n
        ),
        "complete": not missing,
    }
    (args.out / "oceanship_audio_verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    # a keep-list, so the pipeline can restrict itself to what actually arrived
    with open(args.out / "oceanship_clips_present.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["stem", "official_split", "label", "mmsi", "rec_key", "bytes"])
        for k, v in want.items():
            if k in got:
                w.writerow([k, v["official_split"], v["label"], v["mmsi"], v["rec_key"], got[k]])

    text = "\n".join(L) + "\n"
    (args.out / "w0_03_oceanship_audio_verification.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
