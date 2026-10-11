"""W0-07  Extract the author's Baidu release straight out of its raw split chunks.

Why not just use 7-Zip / WinRAR
-------------------------------
The Baidu folder contains 3090latest_wav00 .. 3090latest_wav05.  Those are NOT
six archives.  They are ONE gzip stream that Baidu sliced into fixed-size byte
chunks, so:

  * only the first chunk carries the gzip magic (1F 8B), and
  * every other chunk starts with arbitrary bytes.

A normal extraction tool opens each file on its own, sees no recognisable
container in chunks 01..05, and refuses.  Even for chunk 00 it would stop around
4.4 GiB because the gzip stream simply has not ended there.

Standard fixes and why this script exists
----------------------------------------
  * renaming to .001 .002 ... sometimes makes 7-Zip treat them as one volume,
    but it also depends on the tool version and on the archive inside being
    supported - fragile, and it still leaves you with a second 26 GiB copy;
  * `copy /b part00 + part01 + ... merged.tar.gz` works and is the most robust
    fallback, but it needs another 18 GiB free for the intermediate file before
    anything is extracted.

This script concatenates the chunks in memory, gunzips and untars in a single
pass, and writes only the wav files.  No intermediate archive, so peak disk
usage is just the extracted audio.

Important: you do NOT need to extract before uploading.  Kaggle Datasets store
files as-is and w2_02_cache_oceanship.py reads the chunks directly, so the
upload path is: drop the six files into a Dataset unchanged.  Extract only if
you want to look at / re-process the wavs locally.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tarfile
import time

# Reuse the chunk-group detection and the multi-part stream reader instead of
# re-deriving the naming rules in a second place.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from w2_02_cache_oceanship import MultiPartReader, find_chunk_groups  # noqa: E402

EXPECTED_CLIPS = 53_770


def safe_target(dst: pathlib.Path, name: str) -> pathlib.Path | None:
    """Join a tar member name onto dst, refusing anything that escapes dst."""
    parts = [p for p in pathlib.PurePosixPath(name).parts if p not in ("", ".", "/")]
    if any(p == ".." for p in parts):
        return None
    return dst.joinpath(*parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=pathlib.Path, required=True,
                    help="folder holding the chunks, e.g. G:\\baidu\\oceanship")
    ap.add_argument("--dst", type=pathlib.Path, required=True,
                    help="where to write the wav files")
    ap.add_argument("--strip", type=int, default=0,
                    help="leading path components to drop (1 removes the "
                         "'3090latest_wav/' top folder)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N files (smoke test)")
    ap.add_argument("--ext", default=".wav", help="only extract this extension")
    args = ap.parse_args()

    groups = find_chunk_groups(args.src)
    if not groups:
        print(f"no split-chunk groups found under {args.src}")
        print("expected files named <base>00, <base>01, ... in one folder")
        return 1

    args.dst.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("extract the split Baidu release")
    print("=" * 84)
    print(f"  source : {args.src}")
    print(f"  target : {args.dst}")
    print()

    grand = {"files": 0, "wav": 0, "bytes": 0, "skipped": 0, "other": 0}
    t0 = time.time()

    for base, parts in groups:
        size = sum(p.stat().st_size for p in parts)
        print(f"[group] {base}: {len(parts)} chunks, {size/1024**3:.2f} GiB")
        for i, p in enumerate(parts):
            print(f"          {i:02d}  {p.name:<28} {p.stat().st_size:>16,} B")
        print()

        reader = MultiPartReader(parts)
        # "r|*" = sequential stream, transparent compression detection.  It must
        # be sequential: a multi-part gzip cannot be seeked.
        try:
            tf = tarfile.open(fileobj=reader, mode="r|*")
        except Exception as exc:
            print(f"  cannot open the concatenated stream: {type(exc).__name__}: {exc}")
            print("  -> the chunk set is probably incomplete or one chunk is corrupt;")
            print("     run w0_05_check_gz_complete.py to find which chunk breaks.")
            return 1

        with tf:
            for member in tf:
                if not member.isfile():
                    continue
                grand["files"] += 1
                if args.ext and not member.name.endswith(args.ext):
                    grand["other"] += 1
                    continue

                target = safe_target(args.dst, member.name)
                if target is None:
                    print(f"  [skip] unsafe path in archive: {member.name}")
                    grand["skipped"] += 1
                    continue
                if args.strip:
                    rel = pathlib.PurePosixPath(member.name)
                    target = safe_target(args.dst, str(pathlib.PurePosixPath(*rel.parts[args.strip:])))

                if target.exists() and target.stat().st_size == member.size:
                    grand["wav"] += 1          # already done, keep it idempotent
                    continue

                target.parent.mkdir(parents=True, exist_ok=True)
                fh = tf.extractfile(member)
                if fh is None:
                    grand["skipped"] += 1
                    continue
                with open(target, "wb") as out:
                    while True:
                        blk = fh.read(1 << 20)
                        if not blk:
                            break
                        out.write(blk)

                grand["wav"] += 1
                grand["bytes"] += member.size

                if grand["wav"] % 5000 == 0:
                    el = time.time() - t0
                    rate = grand["wav"] / el if el else 0
                    print(f"  ... {grand['wav']:,} files, {grand['bytes']/1024**3:.2f} GiB, "
                          f"{rate:.0f} files/s, {el/60:.1f} min", flush=True)

                if args.limit and grand["wav"] >= args.limit:
                    break

        if args.limit and grand["wav"] >= args.limit:
            break

    el = time.time() - t0
    print()
    print("=" * 84)
    print("done")
    print("=" * 84)
    print(f"  archive members       : {grand['files']:,}")
    print(f"  wav files written     : {grand['wav']:,}")
    print(f"  bytes written         : {grand['bytes']:,} B = {grand['bytes']/1024**3:.2f} GiB")
    print(f"  skipped (unsafe)      : {grand['skipped']:,}")
    print(f"  other extensions      : {grand['other']:,}")
    print(f"  elapsed               : {el/60:.1f} min ({grand['wav']/el if el else 0:.0f} files/s)")
    print()
    if grand["wav"] >= EXPECTED_CLIPS:
        print(f"  {grand['wav']:,} >= the official FG count ({EXPECTED_CLIPS:,}): extraction complete.")
    elif grand["wav"]:
        print(f"  {grand['wav']:,} of the official {EXPECTED_CLIPS:,} clips. "
              f"That is {100*grand['wav']/EXPECTED_CLIPS:.1f}% - fine for a --limit run,")
        print("  but a full run should reach 53,770.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
