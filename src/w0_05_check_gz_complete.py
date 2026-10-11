"""Decide whether the Baidu .tar.gz chunks are complete or truncated.

Walks the whole concatenated stream with a raw zlib inflater, discarding the
output, and reports exactly how far it gets.  A gzip stream that ends properly
sets `eof` and leaves nothing buffered; anything else means the download is
short or corrupt, which is the difference between "merge and upload" and
"download it again".
"""

from __future__ import annotations

import argparse
import pathlib
import re
import time
import zlib

# default source: the folder the Baidu downloader wrote into
DEFAULT_SRC = pathlib.Path(r"G:\baidu\oceanship")
PART_RE = re.compile(r"^(?P<base>.+?)(?P<idx>\d{2})$")
CHUNK = 4 << 20


def resolve_inputs(path: pathlib.Path | None) -> list[pathlib.Path]:
    """Accept either the merged .tar.gz or the folder of chunks."""
    if path is None:
        path = DEFAULT_SRC
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise SystemExit(f"not found: {path}")
    # prefer one merged file if it is already there
    merged = [p for p in sorted(path.iterdir()) if p.is_file() and p.name.endswith((".tar.gz", ".tgz"))]
    if merged:
        return merged[:1]
    groups: dict[str, list[tuple[int, pathlib.Path]]] = {}
    for p in sorted(path.iterdir()):
        if p.is_file():
            m = PART_RE.match(p.name)
            if m:
                groups.setdefault(m.group("base"), []).append((int(m.group("idx")), p))
    if not groups:
        raise SystemExit(f"neither a .tar.gz nor NN-suffixed chunks found in {path}")
    base, items = max(groups.items(), key=lambda kv: sum(q.stat().st_size for _, q in kv[1]))
    items.sort(key=lambda t: t[0])
    idxs = [i for i, _ in items]
    gaps = [i for i in range(idxs[0], idxs[-1] + 1) if i not in idxs]
    if gaps:
        print(f"[warn] missing part numbers: {[f'{base}{g:02d}' for g in gaps]}", flush=True)
    return [p for _, p in items]


def main() -> int:
    ap = argparse.ArgumentParser(description="Is this gzip stream complete?")
    ap.add_argument("--path", type=pathlib.Path, default=None,
                    help="a merged .tar.gz, or the folder holding the chunks (default: G:\\baidu\\oceanship)")
    args = ap.parse_args()

    PARTS = resolve_inputs(args.path)
    total_in = sum(p.stat().st_size for p in PARTS)
    print("=" * 78)
    print("raw zlib walk over the input")
    print("=" * 78)
    print(f"  input        : {PARTS[0] if len(PARTS) == 1 else f'{len(PARTS)} chunks in {PARTS[0].parent}'}")
    print(f"  size         : {total_in:,} B = {total_in/1024**3:.2f} GiB")

    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    consumed = 0
    produced = 0
    t0 = time.time()
    err = ""
    last_ok: pathlib.Path | None = None
    current: pathlib.Path | None = None
    try:
        for p in PARTS:
            current = p
            with open(p, "rb") as fh:
                while True:
                    data = fh.read(CHUNK)
                    if not data:
                        break
                    consumed += len(data)
                    out = d.decompress(data)
                    produced += len(out)
                    while d.unconsumed_tail:
                        out = d.decompress(d.unconsumed_tail)
                        produced += len(out)
            last_ok = p
            el = time.time() - t0
            print(f"  after {p.name}: consumed {consumed/1024**3:.2f} GiB  "
                  f"produced {produced/1024**3:.2f} GiB  eof={d.eof}  {el:.0f}s", flush=True)
    except zlib.error as exc:
        err = str(exc)

    el = time.time() - t0
    print()
    print("=" * 78)
    print("verdict")
    print("=" * 78)
    print(f"  compressed consumed : {consumed:,} B of {total_in:,}  ({100*consumed/total_in:.1f}%)")
    print(f"  uncompressed out    : {produced:,} B = {produced/1024**3:.2f} GiB")
    print(f"  eof reached         : {d.eof}")
    print(f"  zlib error          : {err or '(none)'}")
    print(f"  elapsed             : {el:.0f}s")
    print()
    if d.eof:
        print("  -> stream terminated cleanly: the chunks are a COMPLETE archive.")
        print("     Merge them and upload; nothing needs re-downloading.")
    else:
        print("  -> stream did NOT terminate. The archive is incomplete or damaged.")
        if len(PARTS) > 1 and current is not None:
            print()
            print("  fault localisation (the chunks are a plain byte split, so the")
            print("  stream breaks inside whichever chunk was being read):")
            print(f"     read cleanly up to : {last_ok.name if last_ok else '(none)'}")
            print(f"     breaks inside      : {current.name}")
            if current is last_ok and last_ok is not None:
                print("     (all earlier chunks are fine)")
            print()
            print(f"     -> re-download only {current.name}.  Delete it first so the")
            print("        client cannot resume onto a bad file, then re-run this check.")
        else:
            print("     -> re-download the archive.")
    return 0 if d.eof else 1


if __name__ == "__main__":
    raise SystemExit(main())
