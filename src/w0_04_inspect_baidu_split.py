"""Stream a Baidu-split .tar.gz without writing anything to disk.

Baidu's downloader cut the author's archive into 3 GiB chunks named
3090latest_wav00 .. 3090latest_wav05.  Only chunk 00 carries the gzip magic
(1f 8b); the rest are raw continuations.  So the six files are ONE gzip stream,
and that stream holds a tar of 3090latest_wav/*.wav.

Rather than merge 18 GB to disk, this concatenates the chunks as a read-only
stream and lets tarfile walk the members, reporting what is actually inside and
whether the gzip stream terminates cleanly (i.e. whether the download is
complete).
"""

from __future__ import annotations

import collections
import io
import pathlib
import sys
import tarfile
import time
import zlib

SRC = pathlib.Path(r"G:\baidu\oceanship")
PARTS = sorted(SRC.glob("3090latest_wav[0-9][0-9]"))


class MultiPartReader(io.RawIOBase):
    """Expose the Baidu chunks as one read-only stream, no decompression here.

    Only the concatenation lives here; tarfile does the gzip.  That keeps this
    class trivial and means a truncated download surfaces as an EOFError from
    gzip, which is exactly the signal we want.
    """

    def __init__(self, parts: list[pathlib.Path]) -> None:
        super().__init__()
        self._parts = list(parts)
        self._fh = None
        self.bytes_read = 0
        self.parts_opened = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:
        while True:
            if self._fh is None:
                if not self._parts:
                    return 0
                self._fh = open(self._parts.pop(0), "rb")
                self.parts_opened += 1
            n = self._fh.readinto(buf)
            if n:
                self.bytes_read += n
                return n
            self._fh.close()
            self._fh = None


def main() -> int:
    if not PARTS:
        print(f"no chunks found in {SRC}", file=sys.stderr)
        return 1

    print("=" * 84)
    print("Baidu split .tar.gz -- streaming inspection (nothing written to disk)")
    print("=" * 84)
    total_in = sum(p.stat().st_size for p in PARTS)
    for p in PARTS:
        print(f"  {p.name:<24}{p.stat().st_size:>16,} B")
    print(f"  {'TOTAL (compressed)':<24}{total_in:>16,} B = {total_in/1024**3:.2f} GiB")
    print()

    reader = MultiPartReader(PARTS)
    t0 = time.time()
    n_members = n_wav = 0
    uncompressed = 0
    tops = collections.Counter()
    first: list[str] = []
    last: list[str] = []
    odd: list[str] = []
    complete = False
    failure = ""

    try:
        with tarfile.open(fileobj=reader, mode="r|gz") as tf:
            for m in tf:
                n_members += 1
                if n_members % 20000 == 0:
                    el = time.time() - t0
                    print(f"  ... {n_members:,} members, {reader.bytes_read/1024**3:.2f} GiB read, "
                          f"{el:.0f}s", flush=True)
                if not m.isfile():
                    continue
                parts = pathlib.Path(m.name).parts
                tops[parts[0] if len(parts) > 1 else "(root)"] += 1
                if m.name.lower().endswith(".wav"):
                    n_wav += 1
                    uncompressed += m.size
                    if len(first) < 3:
                        first.append(f"{m.name}  ({m.size:,} B)")
                    last.append(f"{m.name}  ({m.size:,} B)")
                    if len(last) > 3:
                        last.pop(0)
                elif len(odd) < 8:
                    odd.append(m.name)
        complete = True
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        print(f"\n  !! stopped early: {failure}")
        print("     a truncated download typically fails exactly here")

    el = time.time() - t0
    print()
    print("=" * 84)
    print("result")
    print("=" * 84)
    print(f"  members                 : {n_members:,}")
    print(f"  .wav files              : {n_wav:,}")
    print(f"  uncompressed wav bytes  : {uncompressed:,} B = {uncompressed/1024**3:.2f} GiB")
    print(f"  compressed read         : {reader.bytes_read:,} B of {total_in:,} "
          f"({100*reader.bytes_read/total_in:.1f}%)")
    print(f"  chunks opened           : {reader.parts_opened} of {len(PARTS)}")
    print(f"  stream ended cleanly    : {complete}"
          + (f"   (failure: {failure})" if failure else "   -> download is complete"))
    print(f"  elapsed                 : {el:.0f}s")
    print()
    print(f"  top-level dirs          : {dict(tops)}")
    if odd:
        print(f"  non-wav members         : {odd}")
    print()
    print("  first wav members:")
    for s in first:
        print("    ", s)
    print("  last wav members:")
    for s in last:
        print("    ", s)
    print()
    print("  reference: the FG release is 53,770 clips (45,699 train + 8,071 test).")
    print("  Compare the .wav count above against that to judge completeness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
