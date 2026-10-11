"""W2-02  Cache Oceanship representations, reading straight out of the archives.

Why this exists (and why it is not w2_01)
-----------------------------------------
Oceanship is distributed as archives, not as a folder of wavs - 42 zips on
123pan, or a tar on the author's Baidu share.  You can upload those archives to
Kaggle exactly as downloaded, with no local extraction, and this script reads
the wavs from *inside* them.  That matters for two concrete reasons:

  * Kaggle Datasets store files as-is and do NOT unpack on mount, and
  * /kaggle/working is only 20 GB, so extracting 23 GiB of train audio inside
    the notebook is not possible anyway.

Reading an archive member gives a seekable file object, and libsndfile accepts
one, so no temporary file is ever needed.  Both containers are supported:

    .zip                      what the 123pan share gives you
    .tar / .tar.gz / .tgz     what the Baidu share gives you
                              (do NOT upload the raw .aa .ab ... parts; merge
                              them into one .tar locally first - the script
                              prints the exact command if it spots them)

Two commands, in this order
---------------------------
1. Ask what you actually downloaded (no STFT, seconds):

       python w2_02_cache_oceanship.py --probe-only \
           --audio-root /kaggle/input/oceanship-audio \
           --csv-dir    /kaggle/input/oceanship-metadata

   This prints the real sample rate, channel count and duration distribution.
   Oceanship's own issue tracker has an unanswered report that "the downloaded
   audio is not fixed in duration", so measure rather than assume.

2. Cache the representations:

       python w2_02_cache_oceanship.py \
           --audio-root /kaggle/input/oceanship-audio \
           --csv-dir    /kaggle/input/oceanship-metadata \
           --out        /kaggle/working/cache

Outputs match w2_01 exactly (logmel.npy / spec.npy / index.csv, fp16,
(128, 313) per clip), so every downstream script - w3_00, w3_01, w3_02 - works
unchanged on either dataset.

Budget warning: Oceanship has 53,770 clips against DeepShip's 33,710, so the
two-representation cache lands near 9-10 GB rather than 5.4 GB.  That still fits
the 20 GB working directory but leaves little headroom, so --representations
lets you write only one of them when you are short on space.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gc
import io
import os
import re
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from w2_01_cache_features import (  # noqa: E402
    BATCH, N_FRAMES, N_MELS, SEG_S, TOP_DB,
    build_filterbanks, stft_power_batch, to_db,
)

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    print("ERROR: needs soundfile (pip install soundfile)", file=sys.stderr)
    raise

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover
    resample_poly = None

TGT_SR = 16_000
SEG_LEN = int(TGT_SR * SEG_S)  # 80_000

# Anatomy of a file name, from the CSV and from the archives:
#
#   ./3090latest_wav/20200715T183608.623Z_2202_id_24_typecargo_37_0.pt
#    ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^ ^^^^ ^^^^^ ^^^^^^^^^^ ^^ ^
#    dir: author's      audio ts      seq  hydro-  PLACEHOLDER AIS seg
#    preprocessing                          phone   (see below) code idx
#    output, named after                    id
#    the GPU he used
#
# Three traps, all verified against the 53,770-row manifest:
#
#   * `type...` is always literally "typecargo" - all 53,770 file names say it,
#     while the CSV has 15 distinct classes.  It is a hard-coded placeholder.
#     The real class lives in the AIS code that follows, and each AIS code maps
#     to exactly one class (52->Tug, 37->Pleasure Craft, 70->Cargo, ...).
#   * `id_N` is the hydrophone, not the vessel: only three values (5, 24, 19)
#     occur.  Worse, it disagrees with the CSV `id` column in 45,511 of 53,770
#     rows, so we trust neither - vessel identity comes from `mmsi`.
#   * the trailing segment index is present for 32,025 clips and absent for
#     21,745, so names have either 6 or 7 fields.  Hence the optional group.
#
# We match on the stem alone: directory names and extensions differ between the
# manifest (.pt from the author's preprocessing) and the actual download (.wav),
# so neither can be relied on.
STEM_RE = re.compile(r"^(?P<stem>[0-9T:.Z-]+_\d+_id_[^_]+_type[a-zA-Z]+_\d+(?:_\d+)?)$")


ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz")
# the author's Baidu release is a tar.gz split into bytes of exactly 3 GiB, named
# 3090latest_wav00 .. 05.  Only chunk 00 carries the gzip magic, so the set has
# to be read as one stream - which we do, rather than making the user merge 18 GB
# locally and upload a duplicate copy.
SPLIT_PART_RE = re.compile(r"^(?P<base>.+?)(?P<idx>\d{2})$")
CHUNK_RE = re.compile(r"^[0-9a-zA-Z]{2}$")
# legacy naming (.aa .ab ...) that some mirrors use for the same thing
LEGACY_PART_RE = re.compile(r"\.(a[a-z])$", re.IGNORECASE)


class MultiPartReader(io.RawIOBase):
    """Expose several split chunks as a single read-only stream.

    Used for the Baidu release.  Only concatenation happens here; tarfile does
    the gzip on top, so a truncated set surfaces as a clean gzip error.
    """

    def __init__(self, parts: list[Path]) -> None:
        super().__init__()
        self._parts = list(parts)
        self._fh = None
        self.bytes_read = 0
        self.parts_opened = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:  # type: ignore[override]
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


def find_chunk_groups(audio_root: Path) -> list[tuple[str, list[Path]]]:
    """Detect Baidu-style split sets: <base>00, <base>01, ... in one folder.

    Returns [(base_name, [parts in numeric order]), ...] and keeps only groups
    with at least two members, so a single oddly-named file is not mistaken for
    a split set.
    """
    groups: dict[tuple[Path, str], list[tuple[int, Path]]] = {}
    for p in sorted(audio_root.rglob("*")):
        if not p.is_file():
            continue
        m = SPLIT_PART_RE.match(p.name)
        if not m or len(m.group("idx")) != 2:
            continue
        if p.suffix and p.suffix.lower() in ARCHIVE_SUFFIXES:
            continue
        groups.setdefault((p.parent, m.group("base")), []).append((int(m.group("idx")), p))

    out: list[tuple[str, list[Path]]] = []
    for (parent, base), items in groups.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda t: t[0])
        idxs = [i for i, _ in items]
        gaps = [i for i in range(idxs[0], idxs[-1] + 1) if i not in idxs]
        if gaps:
            print(f"[warn] {base}: missing parts {[f'{base}{g:02d}' for g in gaps]}", flush=True)
        out.append((base, [p for _, p in items]))
    return out


# --------------------------------------------------------------------------- #
# locating the audio
# --------------------------------------------------------------------------- #
def index_audio(audio_root: Path, verbose: bool = True) -> tuple[dict[str, tuple], list[Path]]:
    """stem -> ('zip'|'tar'|'file', archive_path, member_or_None).

    Scans every archive once and builds an in-memory index, so the per-clip read
    later is a dict lookup rather than a scan.  Both zip and tar are supported
    because the two official mirrors ship different formats: 123pan gives .zip,
    the author's Baidu release is a .tar split into .aa .ab ... parts.
    """
    idx: dict[str, tuple] = {}
    archives: list[Path] = []
    loose = 0
    stray_candidates: list[Path] = []

    for p in sorted(audio_root.rglob("*")):
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.endswith(ARCHIVE_SUFFIXES):
            archives.append(p)
        elif SPLIT_PART_RE.search(low):
            stray_candidates.append(p)
        elif p.suffix.lower() in (".wav", ".flac"):
            idx.setdefault(p.stem, ("file", p, None))
            loose += 1

    # A bare "ends in two digits" test matches almost any filename, so only call
    # something a split set when >=2 files actually share a base name.  Without
    # this guard a cluttered folder produced ~300 bogus "split parts" warnings.
    _groups: dict[tuple, list[Path]] = {}
    for p in stray_candidates:
        m = SPLIT_PART_RE.match(p.name)
        if m:
            _groups.setdefault((p.parent, m.group("base")), []).append(p)
    stray_parts = [q.name for v in _groups.values() if len(v) >= 2 for q in v]

    if verbose and stray_parts:
        print(f"[index] FOUND {len(stray_parts)} SPLIT PARTS (.aa/.ab/...), not a complete archive.", flush=True)
        print(f"        e.g. {stray_parts[:4]}", flush=True)
        print("        Either concatenate them into one .tar, or just point --audio-root", flush=True)
        print("        at the folder that holds them and let this script stitch the stream:", flush=True)
        print("          PowerShell merge (only if you insist on a single file):", flush=True)
        print("            $parts = Get-ChildItem 'E:\\oceanship\\*' -File | Where-Object { $_.Name -match '\\.a[a-z]$|\\d{2}$' } | Sort-Object Name", flush=True)
        print("            $out = [IO.File]::Create('E:\\oceanship\\oceanship.tar')", flush=True)
        print("            foreach ($p in $parts) { $in=[IO.File]::OpenRead($p.FullName); $in.CopyTo($out); $in.Close() }", flush=True)
        print("            $out.Close()", flush=True)
        print("", flush=True)

    if verbose and archives:
        print(f"[index] {len(archives)} archives, {loose} loose audio files", flush=True)

    for ai, ap in enumerate(archives, 1):
        n_before = len(idx)
        low = ap.name.lower()
        try:
            if low.endswith(".zip"):
                with zipfile.ZipFile(ap) as zf:
                    for name in zf.namelist():
                        if name.endswith("/"):
                            continue
                        stem = Path(name).stem
                        if STEM_RE.match(stem):
                            idx.setdefault(stem, ("zip", ap, name))
            else:
                with tarfile.open(ap, "r:*") as tf:
                    for member in tf:
                        if not member.isfile():
                            continue
                        stem = Path(member.name).stem
                        if STEM_RE.match(stem):
                            idx.setdefault(stem, ("tar", ap, member.name))
        except Exception as exc:
            print(f"[index] WARNING could not read {ap.name}: {exc}", flush=True)
        if verbose:
            print(f"  [{ai}/{len(archives)}] {ap.name}  +{len(idx)-n_before:,}  total={len(idx):,}", flush=True)

    return idx, archives


# Opened containers are reused instead of reopened per clip.
#
# This is not a micro-optimisation.  The official zip holds 53,770 members, so
# its central directory is ~6 MB.  Constructing ZipFile(path) re-reads and
# re-parses that directory every single time, which measured out at 3.9 clips/s
# - i.e. 231 minutes for a full pass, against ~10 minutes on the streaming path.
# Keeping one handle per archive turns each clip read into a plain seek.
_CONTAINERS: dict[tuple, object] = {}


def _container(kind: str, path):
    """Return a cached open archive for `path`, opening it on first use."""
    key = (kind, str(path))
    holder = _CONTAINERS.get(key)
    if holder is None:
        holder = zipfile.ZipFile(path) if kind == "zip" else tarfile.open(path, "r:*")
        _CONTAINERS[key] = holder
    return holder


def close_containers() -> None:
    """Release every cached archive.  Call at the end of a run."""
    while _CONTAINERS:
        _, holder = _CONTAINERS.popitem()
        try:
            holder.close()
        except Exception:
            pass


class _MemberHandle:
    """A member's file object; closing it leaves the container open for reuse."""

    def __init__(self, fh):
        self._fh = fh

    def read(self, *a):
        return self._fh.read(*a)

    def seek(self, *a):
        return self._fh.seek(*a)

    def tell(self):
        return self._fh.tell()

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_audio(entry: tuple):
    """Return a file object for one clip, whichever container it lives in.

    Only the member handle is meant to be closed by the caller - the archive
    behind it stays open so the next clip is cheap.
    """
    kind, a, b = entry
    if kind == "file":
        return open(a, "rb")
    holder = _container(kind, a)
    if kind == "zip":
        return _MemberHandle(holder.open(b))
    return _MemberHandle(holder.extractfile(b))


def _decode(fh) -> tuple[np.ndarray, int, int]:
    """Decode a wav from any file-like object to mono float32 at TGT_SR.

    Returns (x, source_sample_rate, source_channels).  Falls back to an
    in-memory buffer if libsndfile refuses the stream object, which keeps this
    working across soundfile versions and across both containers.
    """
    try:
        x, sr = sf.read(fh, dtype="float32", always_2d=True)
    except Exception:
        data = fh.read()
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if isinstance(x, tuple):  # defensive: some backends return (data, sr)
        x, sr = x
    ch = x.shape[1] if x.ndim > 1 else 1
    if x.ndim > 1 and ch > 1:
        x = x.mean(axis=1)
    elif x.ndim > 1:
        x = x[:, 0]

    if sr != TGT_SR:
        if resample_poly is None:
            raise RuntimeError("need scipy for resampling")
        from math import gcd

        g = gcd(int(sr), TGT_SR)
        x = resample_poly(x, TGT_SR // g, int(sr) // g).astype(np.float32)
    return np.ascontiguousarray(x, dtype=np.float32), int(sr), ch


def read_clip(entry: tuple) -> tuple[np.ndarray, int, int]:
    """Read one clip via the random-access index (zip / plain tar / loose file)."""
    fh = open_audio(entry)
    try:
        return _decode(fh)
    finally:
        fh.close()


def fit_length(x: np.ndarray, min_keep_s: float = 1.0) -> list[np.ndarray]:
    """Split/resize one clip into whole SEG_LEN chunks.

    Long clips become several chunks (correct - they simply are longer than the
    5 s the papers assume).

    Short clips are PADDED, not dropped, and this was a measured decision: on a
    300-clip sample of the real release, durations ran 2.01-5.00 s with 12.3%
    below 2.5 s and a median of 4.45 s.  Dropping everything under half a window
    would have silently removed an eighth of the corpus, and if the short clips
    cluster in particular classes that is a class-balance bug, not a tidy-up.

    The cost of padding is a confound: a 2 s clip padded to 5 s is 60% silence,
    so duration could become a class cue.  We therefore record the true duration
    per clip in index.csv, so the cache carries the information needed to run a
    with/without-short-clips ablation later.  Below `min_keep_s` the clip really
    is too little signal to be worth padding and is dropped.
    """
    if len(x) >= SEG_LEN:
        n = len(x) // SEG_LEN
        return list(x[: n * SEG_LEN].reshape(n, SEG_LEN))
    if len(x) >= int(TGT_SR * min_keep_s):
        return [np.pad(x, (0, SEG_LEN - len(x)))]
    return []


# --------------------------------------------------------------------------- #
# probe
# --------------------------------------------------------------------------- #
def probe(audio_root: Path, csv_dir: Path, out: Path, sample: int) -> int:
    idx, archives = index_audio(audio_root)

    want: dict[str, dict] = {}
    for side in ("train", "test"):
        with open(csv_dir / f"oceanship_fg_{side}.csv", newline="", encoding="utf-8",
                  errors="replace") as fh:
            for r in csv.DictReader(fh):
                want[Path(r["wav_path"]).stem] = {"split": side, "label": r["label"]}

    present = [s for s in want if s in idx]
    missing = [s for s in want if s not in idx]
    extra = [s for s in idx if s not in want]

    L: list[str] = []
    add = L.append
    add("=" * 80)
    add("Oceanship audio probe")
    add("=" * 80)
    add(f"audio root       : {audio_root}")
    add(f"archives         : {len(archives)}")
    add(f"audio entries    : {len(idx):,}")
    add(f"expected (CSV)   : {len(want):,}")
    add(f"matched stems    : {len(present):,}  ({100*len(present)/max(len(want),1):.2f}%)")
    add(f"missing stems    : {len(missing):,}")
    add(f"unexpected stems : {len(extra):,}")
    add("")

    if len(archives) == 0 and not idx:
        add("Nothing found.  Check --audio-root: it should contain the .zip files")
        add("(or, if they were already extracted, the folder holding the .wav files).")
        text = "\n".join(L) + "\n"
        print(text)
        return 1

    # Which release is this?  The two differ in a way that decides whether the
    # whole vessel-level analysis is even possible, so say it out loud.
    # Verified against the official CSVs: FG is an exact subset of Full, and
    #   FG   = 53,770 clips, every one with an MMSI            (3090latest_wav)
    #   CG   = 49,153 clips, no MMSI anywhere                  (v100_preprocessed_...)
    #   Full = FG + CG = 97,664 unique clips, and its CSV has ONLY
    #          wav_path/label/text_path - no mmsi column at all
    # So Full is strictly worse for this study: the 49,153 extra clips cannot be
    # placed in a vessel-disjoint split.
    FG_TOTAL, FULL_TOTAL = 53_770, 97_664
    n = len(idx)
    add("which release is this?")
    add("-" * 80)
    if n >= FG_TOTAL * 1.5:
        add(f"  {n:,} audio entries -> the FULL release (FG + CG).")
        add(f"  Full = {FULL_TOTAL:,} clips, but its CSV carries no mmsi column, so the")
        add(f"  {FULL_TOTAL - FG_TOTAL:,} CG clips cannot go into a vessel-disjoint split.")
        add(f"  The FG subset inside ({FG_TOTAL:,} clips) is still perfectly usable - it is")
        add("  exactly the set the FG manifest describes.  Point --audio-root at just the")
        add("  FG archives, or pass --present-list to keep only FG clips.")
    elif n >= FG_TOTAL * 0.95:
        add(f"  {n:,} audio entries -> the FG release, complete.")
        add(f"  Matches the official FG manifests ({FG_TOTAL:,} clips).  Every clip has an")
        add("  MMSI, so all three split protocols are available.  This is the one to use.")
    elif n >= FG_TOTAL * 0.30:
        add(f"  {n:,} audio entries -> FG-shaped, but only {100*n/FG_TOTAL:.0f}% of it.")
        add("  Probably a partial download.  The per-class breakdown below shows what is")
        add("  missing; whatever is present can still be used (see --present-list).")
    else:
        add(f"  {n:,} audio entries -> far below either release ({FG_TOTAL:,} FG / {FULL_TOTAL:,} Full).")
        add("  Either the download is very partial, or --audio-root points at the wrong")
        add("  folder.  Check the archive list above.")
    add("")

    # sample a spread rather than the first N, so the probe is not just one zip
    step = max(1, len(present) // max(sample, 1))
    picks = present[::step][:sample]

    add(f"reading {len(picks)} clips spread across the corpus ...")
    dur, srs, chs, errs = [], collections.Counter(), collections.Counter(), 0
    for s in picks:
        try:
            x, sr, ch = read_clip(idx[s])
        except Exception as exc:
            errs += 1
            if errs <= 3:
                add(f"    READ FAIL {s}: {exc}")
            continue
        srs[sr] += 1
        chs[ch] += 1
        dur.append(len(x) / TGT_SR)
    add("")

    if srs:
        add(f"source sample rates : {dict(srs.most_common())}")
        add(f"source channels     : {dict(chs.most_common())}")
    if dur:
        d = np.array(dur)
        add("duration after resampling to 16 kHz (seconds)")
        add(f"  min {d.min():.2f}   p25 {np.percentile(d,25):.2f}   median {np.median(d):.2f}"
            f"   p75 {np.percentile(d,75):.2f}   max {d.max():.2f}")
        add(f"  < 2.5 s (would be dropped)      : {(d < 2.5).sum():,} / {len(d):,}")
        add(f"  > 5 s   (becomes 2+ clips)      : {(d > 5).sum():,} / {len(d):,}")
        add("")
        add("  If most clips are already ~5 s, the corpus matches what the papers assume")
        add("  and nothing needs rethinking.  If they are wildly variable, that is worth")
        add("  a sentence in the data description - the issue tracker already flags it.")
    if errs:
        add("")
        add(f"  unreadable clips in sample: {errs}")
    add("")
    add("next: drop --probe-only and add --out to build the cache.")

    text = "\n".join(L) + "\n"
    print(text)
    out.mkdir(parents=True, exist_ok=True)
    (out / "w2_02_oceanship_probe.txt").write_text(text, encoding="utf-8")
    return 0


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #
def cache(audio_root: Path, csv_dir: Path, out: Path, device: str,
          limit: int, reps: str, present_list: Path | None) -> int:
    """Dispatch to the streaming path when the archives are compressed.

    This matters more than it looks.  A .zip and a plain .tar both support
    random access, so we can index once and then pull members in any order.  A
    .tar.gz does NOT: reaching member n means decompressing every byte before
    it.  Doing that per clip would mean 53,770 re-scans of a 18 GB stream.
    Baidu ships the FG release as exactly that - one .tar.gz cut into 3 GiB
    chunks - so compressed input has to be handled as a single sequential pass.
    """
    archives = [
        p for p in sorted(audio_root.rglob("*"))
        if p.is_file() and p.name.lower().endswith(ARCHIVE_SUFFIXES)
    ]
    compressed = [
        p for p in archives
        if p.name.lower().endswith((".gz", ".tgz", ".bz2", ".xz"))
    ]
    groups = find_chunk_groups(audio_root)

    if groups:
        # Baidu-style split set: stitch the chunks into one stream, no local merge
        sources: list[tuple[str, object]] = []
        for base, parts in groups:
            total = sum(p.stat().st_size for p in parts)
            print(f"[route] split set '{base}': {len(parts)} chunks, {total/1024**3:.2f} GiB "
                  f"-> stitched stream (no local merge needed)", flush=True)
            sources.append((f"{base} (chunks {parts[0].name}..{parts[-1].name})",
                            MultiPartReader(parts)))
        if compressed or archives:
            print(f"[note] ignoring {len(archives)} plain archive(s); using the split set", flush=True)
        return cache_streaming(audio_root, sources, csv_dir, out, device, limit, reps, present_list)

    if compressed:
        print(f"[route] {len(compressed)} compressed archive(s) -> single sequential pass", flush=True)
        return cache_streaming(audio_root, [(p.name, p) for p in archives],
                               csv_dir, out, device, limit, reps, present_list)
    print(f"[route] {len(archives)} random-access archive(s) -> indexed path", flush=True)
    return cache_random(audio_root, csv_dir, out, device, limit, reps, present_list)


def _load_wanted(csv_dir: Path, present_list: Path | None) -> list[dict]:
    rows: list[dict] = []
    for side in ("train", "test"):
        with open(csv_dir / f"oceanship_fg_{side}.csv", newline="", encoding="utf-8",
                  errors="replace") as fh:
            for r in csv.DictReader(fh):
                rows.append({
                    "stem": Path(r["wav_path"]).stem,
                    "split": side,
                    "label": r["label"],
                    "mmsi": r["mmsi"],
                    "rec_key": f"{r['mmsi']}|{r['ais_timestamp']}",
                })
    if present_list and present_list.exists():
        keep = {l.split(",")[0] for l in present_list.read_text(encoding="utf-8").splitlines()[1:]}
        rows = [r for r in rows if r["stem"] in keep]
        print(f"[filter] restricted to {len(rows):,} clips listed in {present_list.name}", flush=True)
    return rows


def cache_streaming(audio_root: Path, sources: list[tuple[str, object]], csv_dir: Path,
                    out: Path, device: str, limit: int, reps: str,
                    present_list: Path | None) -> int:
    """One pass over each source; no random access required.

    A source is (label, path_or_fileobj).  A plain path is opened as an archive;
    a file object is used directly, which is how the stitched Baidu chunks get
    in without ever being merged on disk.
    """
    rows = _load_wanted(csv_dir, present_list)
    meta = {r["stem"]: r for r in rows}
    want = set(meta)
    print(f"[plan] {len(want):,} wanted clips from {len(sources)} source(s); "
          f"device={device}; reps={reps}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    mel_fb, lin_fb = build_filterbanks()
    cap = 16384
    logmel = np.lib.format.open_memmap(out / "logmel.npy", mode="w+", dtype=np.float16,
                                       shape=(cap, N_MELS, N_FRAMES)) if "mel" in reps else None
    spec = np.lib.format.open_memmap(out / "spec.npy", mode="w+", dtype=np.float16,
                                     shape=(cap, N_MELS, N_FRAMES)) if "spec" in reps else None

    idx_fh = (out / "index.csv").open("w", newline="", encoding="utf-8")
    idx_w = csv.writer(idx_fh)
    idx_w.writerow(["row", "clip_id", "rec_key", "clip_idx", "class_dir", "date", "split",
                    "mmsi", "src_dur_s"])

    row = 0
    seen = 0
    done: set[str] = set()
    failed: list[tuple[str, str]] = []
    skipped_short = 0
    dur_by_class: dict[str, list[float]] = collections.defaultdict(list)
    t0 = time.time()

    for ai, (label, src) in enumerate(sources, 1):
        print(f"  [{ai}/{len(sources)}] {label}", flush=True)
        try:
            if hasattr(src, "readinto"):
                tf = tarfile.open(fileobj=src, mode="r|*")  # type: ignore[arg-type]
            else:
                tf = tarfile.open(src, "r|*")  # type: ignore[arg-type]
        except Exception as exc:
            print(f"     cannot open as a tar stream: {exc}", flush=True)
            continue
        with tf:
            for m in tf:
                if not m.isfile():
                    continue
                stem = Path(m.name).stem
                if stem not in want or stem in done:
                    continue
                info = meta[stem]
                done.add(stem)
                seen += 1
                try:
                    buf = io.BytesIO()
                    fh = tf.extractfile(m)
                    if fh is None:
                        raise RuntimeError("extractfile returned None")
                    while True:
                        b = fh.read(1 << 20)
                        if not b:
                            break
                        buf.write(b)
                    buf.seek(0)
                    x, _, _ = _decode(buf)
                except Exception as exc:
                    failed.append((stem, str(exc)))
                    continue
                chunks = fit_length(x)
                if not chunks:
                    skipped_short += 1
                    continue
                dur_by_class[info["label"]].append(len(x) / TGT_SR)

                if row + len(chunks) > cap:
                    cap = int(cap * 1.6) + 4096
                    for name, arr in (("logmel.npy", logmel), ("spec.npy", spec)):
                        if arr is None:
                            continue
                        grown = np.lib.format.open_memmap(out / name, mode="w+", dtype=np.float16,
                                                          shape=(cap, N_MELS, N_FRAMES))
                        grown[:row] = np.asarray(arr[:row])
                        if name == "logmel.npy":
                            logmel = grown
                        else:
                            spec = grown

                for b0 in range(0, len(chunks), BATCH):
                    block = np.stack(chunks[b0 : b0 + BATCH])
                    power = stft_power_batch(block, device)
                    if logmel is not None:
                        logmel[row : row + len(block)] = to_db(mel_fb @ power).astype(np.float16)
                    if spec is not None:
                        spec[row : row + len(block)] = to_db(lin_fb @ power).astype(np.float16)
                    for j in range(len(block)):
                        idx_w.writerow([row + j, f"{stem}#{b0+j}", info["rec_key"], b0 + j,
                                        info["label"], "", info["split"], info["mmsi"],
                                        round(len(x) / TGT_SR, 3)])
                    row += len(block)

                if seen % 500 == 0:
                    el = time.time() - t0
                    print(f"     matched {seen:,}/{len(want):,}  rows={row:,}  "
                          f"{seen/el if el else 0:.1f} clip/s", flush=True)
                if limit and seen >= limit:
                    print(f"     --limit {limit} reached", flush=True)
                    break
        if limit and seen >= limit:
            break

    for arr in (logmel, spec):
        if arr is not None:
            arr.flush()
    idx_fh.close()

    if row < cap:
        logmel = spec = None
        gc.collect()
        for name in ("logmel.npy", "spec.npy"):
            path = out / name
            if not path.exists():
                continue
            arr = np.load(path, mmap_mode="r")
            trimmed = np.asarray(arr[:row]).astype(np.float16)
            del arr
            gc.collect()
            tmp = out / (path.stem + ".tmp.npy")
            np.save(tmp, trimmed)
            os.replace(tmp, path)
            print(f"[trim] {name} -> {trimmed.shape}", flush=True)

    gb = row * N_MELS * N_FRAMES * 2 / 1e9
    lines = [
        "=" * 80,
        f"cached rows        : {row:,} from {seen:,} matched clips",
        f"shape              : ({row}, {N_MELS}, {N_FRAMES}) fp16",
        f"wanted / matched   : {len(want):,} / {seen:,}  "
        f"({100*seen/max(len(want),1):.1f}% found in the archives)",
        f"dropped (too short): {skipped_short:,}",
        f"read failures      : {len(failed):,}",
    ]
    lines += [f"    {s}: {w}" for s, w in failed[:10]]
    lines += [f"approx size        : {gb:.2f} GB", f"elapsed            : {(time.time()-t0)/60:.1f} min"]
    if dur_by_class:
        import statistics

        all_d = [d for v in dur_by_class.values() for d in v]
        short = sum(1 for d in all_d if d < 4.0)
        lines += [
            "",
            "source clip duration (before padding), seconds",
            f"  overall: median {statistics.median(all_d):.2f}  min {min(all_d):.2f}  "
            f"max {max(all_d):.2f}  n={len(all_d):,}",
            f"  under 4.0 s (padded by more than 20%): {short:,} "
            f"({100*short/len(all_d):.1f}%)",
            "  per class - a class concentrating in the short tail is a balance risk:",
        ]
        for c in sorted(dur_by_class, key=lambda k: statistics.median(dur_by_class[k])):
            v = dur_by_class[c]
            lines.append(f"    {c:<30} median {statistics.median(v):.2f}s  "
                         f"min {min(v):.2f}  n={len(v):,}")
    text = "\n".join(lines)
    print()
    print(text)
    (out / "w2_02_cache_report.txt").write_text(text, encoding="utf-8")
    if gb > 14:
        print(f"\nNOTE: {gb:.1f} GB is close to the 20 GB working-directory limit. "
              f"Consider --representations mel.", flush=True)
    return 0


def cache_random(audio_root: Path, csv_dir: Path, out: Path, device: str,
                 limit: int, reps: str, present_list: Path | None) -> int:
    idx, _ = index_audio(audio_root)

    rows: list[dict] = []
    for side in ("train", "test"):
        with open(csv_dir / f"oceanship_fg_{side}.csv", newline="", encoding="utf-8",
                  errors="replace") as fh:
            for r in csv.DictReader(fh):
                rows.append({
                    "stem": Path(r["wav_path"]).stem,
                    "split": side,
                    "label": r["label"],
                    "mmsi": r["mmsi"],
                    "rec_key": f"{r['mmsi']}|{r['ais_timestamp']}",
                })

    if present_list and present_list.exists():
        keep = {l.split(",")[0] for l in present_list.read_text(encoding="utf-8").splitlines()[1:]}
        rows = [r for r in rows if r["stem"] in keep]
        print(f"[filter] restricted to {len(rows):,} clips listed in {present_list.name}", flush=True)

    rows = [r for r in rows if r["stem"] in idx]
    # one canonical order, so a row index is a stable id across runs
    rows.sort(key=lambda r: (r["label"], r["mmsi"], r["rec_key"], r["stem"]))
    if limit:
        rows = rows[:limit]

    print(f"[plan] {len(rows):,} clips to cache; device={device}; reps={reps}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    mel_fb, lin_fb = build_filterbanks()

    # a clip may expand to several 5 s windows, so the row count is not known
    # up front; grow the memmap in blocks instead of pre-allocating wrongly
    def new_memmap(name: str, n: int):
        return np.lib.format.open_memmap(out / name, mode="w+", dtype=np.float16,
                                         shape=(n, N_MELS, N_FRAMES))

    cap = max(len(rows), 1024)
    logmel = new_memmap("logmel.npy", cap) if "mel" in reps else None
    spec = new_memmap("spec.npy", cap) if "spec" in reps else None

    idx_fh = (out / "index.csv").open("w", newline="", encoding="utf-8")
    idx_w = csv.writer(idx_fh)
    idx_w.writerow(["row", "clip_id", "rec_key", "clip_idx", "class_dir", "date", "split",
                    "mmsi", "src_dur_s"])

    row = 0
    failed: list[tuple[str, str]] = []
    skipped_short = 0
    t0 = time.time()

    for i, r in enumerate(rows, 1):
        try:
            x, _, _ = read_clip(idx[r["stem"]])
        except Exception as exc:
            failed.append((r["stem"], str(exc)))
            continue
        chunks = fit_length(x)
        if not chunks:
            skipped_short += 1
            continue
        for b0 in range(0, len(chunks), BATCH):
            block = np.stack(chunks[b0 : b0 + BATCH])
            if row + len(block) > cap:
                cap = int(cap * 1.6) + 1024
                for name, arr in (("logmel.npy", logmel), ("spec.npy", spec)):
                    if arr is None:
                        continue
                    grown = new_memmap(name, cap)
                    grown[:row] = np.asarray(arr[:row])
                    if name == "logmel.npy":
                        logmel = grown
                    else:
                        spec = grown
            power = stft_power_batch(block, device)
            if logmel is not None:
                logmel[row : row + len(block)] = to_db(mel_fb @ power).astype(np.float16)
            if spec is not None:
                spec[row : row + len(block)] = to_db(lin_fb @ power).astype(np.float16)
            for j in range(len(block)):
                idx_w.writerow([row + j, f"{r['stem']}#{b0+j}", r["rec_key"], b0 + j,
                                r["label"], "", r["split"], r["mmsi"],
                                round(len(x) / TGT_SR, 3)])
            row += len(block)

        if i % 200 == 0 or i == len(rows):
            el = time.time() - t0
            rate = i / el if el else 0
            print(f"  [{i}/{len(rows)}] rows={row:,}  {rate:.1f} clip/s  "
                  f"eta {(len(rows)-i)/rate/60 if rate else 0:.1f} min", flush=True)

    for arr in (logmel, spec):
        if arr is not None:
            arr.flush()
    idx_fh.close()

    if row < cap:
        # Release the memmaps before touching the files again: a file that still
        # has an open mapping cannot be overwritten in place (Windows refuses
        # outright).  Drop the references, collect, then swap in a trimmed copy
        # via os.replace so a crash mid-write cannot leave a half-file behind.
        logmel = spec = None
        gc.collect()
        for name in ("logmel.npy", "spec.npy"):
            path = out / name
            if not path.exists():
                continue
            arr = np.load(path, mmap_mode="r")
            trimmed = np.asarray(arr[:row]).astype(np.float16)
            del arr
            gc.collect()
            # np.save appends .npy when the name lacks it, so spell it out and
            # the replace target actually exists
            tmp = out / (path.stem + ".tmp.npy")
            np.save(tmp, trimmed)
            os.replace(tmp, path)
            print(f"[trim] {name} -> {trimmed.shape}", flush=True)

    gb = row * N_MELS * N_FRAMES * 2 / 1e9
    lines = [
        "=" * 80,
        f"cached rows        : {row:,} from {len(rows):,} clips",
        f"shape              : ({row}, {N_MELS}, {N_FRAMES}) fp16 x {len(reps.split('+'))}",
        f"dropped (too short): {skipped_short:,}",
        f"read failures      : {len(failed):,}",
    ]
    lines += [f"    {s}: {w}" for s, w in failed[:10]]
    lines += [f"approx size        : {gb:.2f} GB", f"elapsed            : {(time.time()-t0)/60:.1f} min"]
    text = "\n".join(lines)
    print()
    print(text)
    (out / "w2_02_cache_report.txt").write_text(text, encoding="utf-8")

    if gb > 14:
        print(f"\nNOTE: {gb:.1f} GB is close to the 20 GB working-directory limit. "
              f"Consider --representations mel.", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache Oceanship features directly from the zips.")
    ap.add_argument("--audio-root", type=Path, required=True)
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/cache"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (smoke test)")
    ap.add_argument("--representations", default="mel+spec", choices=["mel", "spec", "mel+spec"])
    ap.add_argument("--present-list", type=Path, default=None,
                    help="out/oceanship_clips_present.csv from w0_03, to cache only what verified")
    ap.add_argument("--probe-only", action="store_true", help="report what you downloaded, no STFT")
    ap.add_argument("--sample", type=int, default=200, help="clips to sample in probe mode")
    args = ap.parse_args()

    if args.device == "cpu":
        try:
            import torch

            if torch.cuda.is_available():
                args.device = "cuda"
        except ImportError:
            pass

    if args.probe_only:
        return probe(args.audio_root, args.csv_dir, args.out, args.sample)
    return cache(args.audio_root, args.csv_dir, args.out, args.device,
                 args.limit, args.representations, args.present_list)


if __name__ == "__main__":
    raise SystemExit(main())
