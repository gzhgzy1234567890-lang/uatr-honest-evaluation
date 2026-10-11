"""Merge the Baidu-split chunks into the single archive the author published.

Baidu's downloader cut `3090latest_wav.tar.gz` into 3 GiB chunks named
3090latest_wav00 .. 3090latest_wav05.  Only chunk 00 carries the gzip magic
(1f 8b) - the rest are raw byte continuations - so merging is a plain
concatenation in numeric order and nothing else.

Why merge at all, rather than uploading the chunks as they are?  Because a
.tar.gz can only be read as a stream: reaching member n means inflating
everything before it.  Given one file, `w2_02_cache_oceanship.py` streams it in
a single pass and never needs random access.  Given six chunks, the reader
would have to know how to stitch them - doable, but there is no reason to add
that complexity when `cat` costs one local copy.

The merged file is uploaded to Kaggle unchanged.  No extraction, here or there:
a Kaggle Dataset keeps files as they are, and /kaggle/working is only 20 GB, so
the ~28 GiB of wavs inside would not fit anyway.

Usage
-----
    python w0_06_merge_baidu_parts.py --src "G:\\baidu\\oceanship" \\
        --out "E:\\oceanship\\3090latest_wav.tar.gz"

    # then, to prove the result is complete before spending hours uploading:
    python w0_05_check_gz_complete.py --path "E:\\oceanship\\3090latest_wav.tar.gz"
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
import time

PART_RE = re.compile(r"^(?P<base>.+?)(?P<idx>\d{2})$")
BUF = 16 << 20


def collect(src: pathlib.Path) -> tuple[str, list[pathlib.Path]]:
    """Group the split parts by base name and return them in numeric order."""
    groups: dict[str, list[tuple[int, pathlib.Path]]] = {}
    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        m = PART_RE.match(p.name)
        if not m:
            continue
        groups.setdefault(m.group("base"), []).append((int(m.group("idx")), p))

    if not groups:
        raise SystemExit(f"no NN-suffixed split parts found in {src}")

    # the biggest group is the archive; anything else is a stray
    base, items = max(groups.items(), key=lambda kv: sum(p.stat().st_size for _, p in kv[1]))
    if len(groups) > 1:
        others = [b for b in groups if b != base]
        print(f"[warn] ignoring other part groups: {others}", flush=True)

    items.sort(key=lambda t: t[0])
    idxs = [i for i, _ in items]
    gaps = [i for i in range(idxs[0], idxs[-1] + 1) if i not in idxs]
    if gaps:
        print(f"[warn] part numbers are not contiguous; missing: "
              f"{[f'{base}{g:02d}' for g in gaps]}", flush=True)
        print("       a gap means the merged file WILL be corrupt", flush=True)
    return base, [p for _, p in items]


def main() -> int:
    ap = argparse.ArgumentParser(description="Concatenate the Baidu split parts.")
    ap.add_argument("--src", type=pathlib.Path, required=True, help="folder holding the chunks")
    ap.add_argument("--out", type=pathlib.Path, default=None, help="output file, e.g. E:\\oceanship\\3090latest_wav.tar.gz")
    ap.add_argument("--force", action="store_true", help="overwrite the output if it already exists")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"not a directory: {args.src}")

    base, parts = collect(args.src)
    out = args.out or (args.src / f"{base}.tar.gz")
    total = sum(p.stat().st_size for p in parts)

    print(f"base name : {base}")
    print(f"parts     : {len(parts)}")
    for p in parts:
        print(f"   {p.name:<26}{p.stat().st_size:>16,} B")
    print(f"total     : {total:,} B = {total / 1024**3:.2f} GiB")
    print(f"output    : {out}")
    print()

    if out.exists() and not args.force:
        have = out.stat().st_size
        if have == total:
            print("output already exists with the right size - nothing to do.")
            return 0
        raise SystemExit(f"{out} exists ({have:,} B) and differs; re-run with --force")

    free = None
    drive = pathlib.Path(out.anchor or out.parent)
    try:
        import shutil

        free = shutil.disk_usage(str(out.parent)).free
    except Exception:
        pass
    if free is not None and free < total * 1.05:
        raise SystemExit(f"only {free/1024**3:.1f} GiB free at {out.parent}, need "
                         f"{total/1024**3:.1f} GiB - pick another drive")

    t0 = time.time()
    written = 0
    with open(out, "wb") as dst:
        for p in parts:
            with open(p, "rb") as src:
                while True:
                    b = src.read(BUF)
                    if not b:
                        break
                    dst.write(b)
                    written += len(b)
            el = time.time() - t0
            print(f"  + {p.name:<26} written {written/1024**3:>6.2f} GiB  {el:>5.0f}s", flush=True)

    el = time.time() - t0
    print()
    print(f"merged    : {written:,} B = {written/1024**3:.2f} GiB in {el:.0f}s")
    if written != total:
        print("!! size mismatch - the merge is wrong", file=sys.stderr)
        return 1
    print()
    print("next: prove it is complete BEFORE uploading (18 GiB up is not cheap):")
    print(f'  python src/w0_05_check_gz_complete.py --path "{out}"')
    print()
    print("then upload that single file to Kaggle as-is.  No extraction - "
          "w2_02_cache_oceanship.py reads the wavs straight out of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
