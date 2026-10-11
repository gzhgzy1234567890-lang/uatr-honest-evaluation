#!/usr/bin/env python3
"""W2-03  Build the frozen feature dataset for every UATR corpus in one pass.

Three corpora, one representation space
---------------------------------------
Oceanship, DeepShip and ShipsEar differ in container, sample rate and labelling
scheme, but they have to land in the *same* feature space, otherwise the
cross-corpus experiments compare pipelines instead of data.  So there is exactly
one set of STFT / filterbank / dB constants in this file and every job uses it.

Output per corpus (row-aligned, so a row index is a stable clip id):

    logmel.npy   (N, 128, 313) fp16   log-mel, dB re clip peak, floored at -80
    spec.npy     (N, 128, 313) fp16   linear frequency, 128 pooled bands
    index.csv    row -> identity + grouping keys + measured source duration

Two representations on purpose: if the strict-protocol collapse shows up in mel
only, that is a feature-engineering artefact; if it shows up in both, it is a
property of the data.  The paper needs the second statement, so the ablation has
to be built in from the start.

Why a measurement pass before writing
-------------------------------------
The cache is frozen and mounted read-only by every later run, so its shape has
to be right the first time.  A measurement pass reads only wav headers and gives
the exact row count (clips longer than 5 s become several rows) plus the true
duration distribution -- which matters, because duration is itself class
correlated in Oceanship (Passenger median 3.67 s against Tug 5.00 s).  Without
an exact count we would have to allocate a guess and then rewrite an 8 GB
memmap just to trim it.

Why the duration is written into index.csv
------------------------------------------
Short clips are padded rather than dropped: dropping everything under half a
window would silently remove an eighth of Oceanship and, if the short clips
cluster in particular classes, that is a class-balance bug rather than a tidy-up.
The cost of padding is a confound (a 2 s clip padded to 5 s is 60% silence), so
the measured duration goes into the index and the ablation is one join away.

Resampling note: everything is resampled through one polyphase routine
(scipy.signal.resample_poly).  The earlier per-corpus scripts used decimate() for
DeepShip and resample_poly() for Oceanship, which are not bit-identical; a single
routine for all three corpora is what makes rows from different corpora
comparable.

Usage
-----
    # what would it do? (headers only, seconds)
    python w2_build_feature_dataset.py --job oceanship --probe \\
        --oceanship-audio /kaggle/input/oceanship \\
        --oceanship-inventory .../oceanship_inventory.csv.gz --out /kaggle/working/uwatr-features

    # build everything
    python w2_build_feature_dataset.py --job all --out /kaggle/working/uwatr-features ...

    # half-size run (mel only) if the working directory is tight
    ... --reps mel
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    print("ERROR: needs soundfile", file=sys.stderr)
    raise

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover
    resample_poly = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


# --------------------------------------------------------------------------- #
# frozen representation configuration
#
# These constants define the public artifact.  Do not tune them per corpus, and
# do not tune them to chase accuracy: a cache that changes between runs makes
# every ablated number unreproducible, which is the exact failure mode this
# project exists to complain about.
# --------------------------------------------------------------------------- #
TGT_SR = 16_000
SEG_S = 5.0
SEG_LEN = int(TGT_SR * SEG_S)          # 80_000 samples
N_FFT = 1024
HOP = 256
N_BINS = N_FFT // 2 + 1                # 513
N_MELS = 128
N_FRAMES = 1 + SEG_LEN // HOP          # 313
MEL_FMIN, MEL_FMAX = 20.0, 8000.0
TOP_DB = 80.0
MIN_KEEP_S = 1.0                       # below this there is too little signal to pad
BATCH = 64

BYTES_PER_ROW = N_MELS * N_FRAMES * 2  # fp16


# --------------------------------------------------------------------------- #
# shared DSP
# --------------------------------------------------------------------------- #
def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_filterbanks() -> tuple[np.ndarray, np.ndarray]:
    """Return (mel_fb, linear_fb), each (128, 513).

    Written by hand rather than taken from librosa so the frozen cache does not
    depend on whichever librosa the current Kaggle image happens to ship.
    """
    freqs = np.linspace(0.0, TGT_SR / 2, N_BINS)

    mel_pts = np.linspace(_hz_to_mel(np.array([MEL_FMIN]))[0],
                          _hz_to_mel(np.array([MEL_FMAX]))[0], N_MELS + 2)
    hz_pts = _mel_to_hz(mel_pts)
    mel_fb = np.zeros((N_MELS, N_BINS), dtype=np.float32)
    for i in range(N_MELS):
        lo, mid, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        for j, f in enumerate(freqs):
            if lo <= f <= mid and mid > lo:
                mel_fb[i, j] = (f - lo) / (mid - lo)
            elif mid < f <= hi and hi > mid:
                mel_fb[i, j] = (hi - f) / (hi - mid)
    mel_fb *= (2.0 / (hz_pts[2:N_MELS + 2] - hz_pts[:N_MELS]))[:, None]

    # linear: average-pool 513 bins into 128 groups so both representations share
    # a shape and the ablation compares like with like
    lin_fb = np.zeros((N_MELS, N_BINS), dtype=np.float32)
    edges = np.linspace(0, N_BINS, N_MELS + 1).astype(int)
    for i in range(N_MELS):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        lin_fb[i, a:b] = 1.0 / (b - a)
    return mel_fb, lin_fb


def stft_power_batch(clips: np.ndarray, device: str) -> np.ndarray:
    """(B, SEG_LEN) -> (B, 513, N_FRAMES) power.

    Both branches must agree frame for frame: the cache is a public artifact, so
    a CPU build has to be comparable with a GPU build.  torch.stft(center=True)
    pads by n_fft//2 with 'reflect' and yields 1 + SEG_LEN//HOP = 313 frames;
    scipy with boundary=None/padded=False would yield 309, so the scipy branch
    pads identically first.  Getting this wrong once produced a 309-frame CPU
    cache and a 313-frame GPU cache from the same code.
    """
    win = np.hanning(N_FFT).astype(np.float32)
    if torch is not None and device != "cpu":
        t = torch.from_numpy(clips).to(device)
        w = torch.from_numpy(win).to(device)
        spec = torch.stft(t, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                          window=w, center=True, return_complex=True, pad_mode="reflect")
        return spec.abs().pow(2)[:, :, :N_FRAMES].cpu().numpy()

    from scipy.signal import stft as sp_stft

    pad = N_FFT // 2
    out = np.empty((clips.shape[0], N_BINS, N_FRAMES), dtype=np.float32)
    for i, c in enumerate(clips):
        c = np.pad(c, (pad, pad), mode="reflect")
        _, _, z = sp_stft(c, fs=TGT_SR, nperseg=N_FFT, noverlap=N_FFT - HOP,
                          window=win, boundary=None, padded=False)
        out[i] = (np.abs(z) ** 2)[:, :N_FRAMES]
    return out


def to_db(power: np.ndarray) -> np.ndarray:
    """Power -> dB relative to each clip's own peak, floored at -TOP_DB.

    Relative rather than absolute removes recorder gain, an instrument artifact,
    while keeping the within-clip dynamic range that carries the ship signature.
    Absolute level, which the attribution study needs as a range proxy, stays in
    the metadata.  Output range is exactly [-TOP_DB, 0].
    """
    peak = np.maximum(power.max(axis=(1, 2), keepdims=True), 1e-12)
    db = 10.0 * np.log10(np.maximum(power, peak * 1e-12) / peak)
    return np.maximum(db, -TOP_DB).astype(np.float32)


def to_mono_16k(x: np.ndarray, sr: int) -> np.ndarray:
    """Downmix to mono and resample to TGT_SR with one polyphase routine."""
    if x.ndim > 1 and x.shape[1] > 1:
        x = x.mean(axis=1)
    elif x.ndim > 1:
        x = x[:, 0]
    if sr == TGT_SR:
        return np.ascontiguousarray(x, dtype=np.float32)
    if resample_poly is None:
        raise RuntimeError("need scipy for resampling")
    from math import gcd
    g = gcd(int(sr), TGT_SR)
    return resample_poly(x, TGT_SR // g, int(sr) // g).astype(np.float32)


def fit_length(x: np.ndarray) -> list[np.ndarray]:
    """One clip -> whole 5 s rows.  Long clips split, short clips are padded."""
    if len(x) >= SEG_LEN:
        n = len(x) // SEG_LEN
        return list(x[: n * SEG_LEN].reshape(n, SEG_LEN))
    if len(x) >= int(MIN_KEEP_S * TGT_SR):
        return [np.pad(x, (0, SEG_LEN - len(x)))]
    return []


def info_any(fh):
    """sf.info on a stream that may not be seekable.

    A zip member handle is a forward-only stream: libsndfile's virtual IO wants
    seek/tell, so on a zip the direct call can fail.  Buffering the member is
    cheap (about 320 KB) and keeps one code path for every container.
    """
    try:
        return sf.info(fh)
    except Exception:
        return sf.info(io.BytesIO(fh.read()))


def read_any(fh) -> tuple[np.ndarray, int]:
    """sf.read with the same non-seekable fallback."""
    try:
        return sf.read(fh, dtype="float32", always_2d=True)
    except Exception:
        return sf.read(io.BytesIO(fh.read()), dtype="float32", always_2d=True)


# --------------------------------------------------------------------------- #
# audio sources: one interface over zip / loose wav / gzip-split tar
# --------------------------------------------------------------------------- #
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tgz", ".tar.gz", ".tar.bz2", ".tar.xz")
CHUNK_RE = re.compile(r"^(?P<base>.+?)(?P<idx>\d{2})$")

# Compression magics.  Needed because splitting detection is name-based and
# Oceanship file names END IN A TWO-DIGIT AIS CODE: on a folder of loose wavs,
# "..._typecargo_70" and "..._typecargo_79" share a base and look exactly like a
# split set.  Requiring a real archive header on the first part kills that
# false positive, which otherwise sends the whole run down the streaming path
# and fails inside tarfile.
_MAGICS = (b"\x1f\x8b", b"PK\x03\x04", b"PK\x05\x06", b"7z\xbc\xaf\x27\x1c",
           b"BZh", b"\xfd7zXZ")


def _looks_like_archive(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
    except OSError:
        return False
    return any(head.startswith(m) for m in _MAGICS)


def is_gzip(path: Path) -> bool:
    try:
        with open(path, "rb") as fh:
            return fh.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def open_text(path: Path):
    """Read a text file that may or may not be gzipped, whatever it is named.

    Kaggle extracts archives when it ingests a dataset: a folder uploaded as
    `uwatr-workspace.zip` arrives one level deeper, `ShipsEar.zip` arrives as
    `shipsear/ShipsEar/`, and `oceanship_inventory.csv.gz` can arrive as a plain
    `oceanship_inventory.csv`.  Sniff the magic bytes rather than trusting the
    name, so every combination works.
    """
    if is_gzip(path):
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return open(path, "rt", newline="", encoding="utf-8")


def find_rel(workspace: Path, rel: str) -> Path | None:
    """Locate `rel` in a workspace, tolerating the un-archived form.

    Tries the shipped name, then the name Kaggle's extractor produces, then any
    file with the same leading stem in the same folder.
    """
    names = [rel]
    if rel.endswith(".gz"):
        names.append(rel[:-3])
    elif rel.endswith((".zip", ".tar")):
        names.append(str(Path(rel).with_suffix("")))
    for name in names:
        p = workspace / name
        if p.exists():
            return p
    for name in names:
        parent, base = workspace / Path(name).parent, Path(name).name.split(".")[0]
        if not parent.is_dir():
            continue
        for hit in sorted(parent.glob(base + "*")):
            if hit.exists():
                return hit
    return None


def copy_as(src: Path, dst: Path) -> None:
    """Copy into the published artifact under the documented name.

    If the artifact documents `x.csv.gz` and the input arrived already
    expanded, re-compress on the way in.  Downstream notebooks then always see
    the one documented spelling, whatever Kaggle did to the upload.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.name.endswith(".gz") and not is_gzip(src):
        with open(src, "rb") as fi, gzip.open(dst, "wb") as fo:
            shutil.copyfileobj(fi, fo)
    else:
        shutil.copy2(src, dst)


class MultiPartReader(io.RawIOBase):
    """Several split chunks as one read-only stream.

    Only concatenation happens here; tarfile does the gunzip on top, so a
    truncated set surfaces as a clean gzip error rather than silent garbage.
    """

    def __init__(self, parts: list[Path]) -> None:
        super().__init__()
        self._parts = list(parts)
        self._fh = None
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buf) -> int:  # type: ignore[override]
        while True:
            if self._fh is None:
                if not self._parts:
                    return 0
                self._fh = open(self._parts.pop(0), "rb")
            n = self._fh.readinto(buf)
            if n:
                self.bytes_read += n
                return n
            self._fh.close()
            self._fh = None


class Source:
    """Base: records() gives the canonical order, open_audio() gives bytes."""

    label = "source"
    streamable = False

    def records(self) -> list[dict]:
        raise NotImplementedError

    def open_audio(self, rec: dict):
        raise NotImplementedError


class ZipSource(Source):
    """A .zip holding every clip.  Random access, and the container is cached.

    Reopening the ZipFile per clip is the single most expensive mistake available
    here: the official archive has 53,770 members, so its central directory is
    about 6 MB and re-parsing it per clip measured out at 3.9 clips/s against
    100 clips/s with one cached handle.

    key="stem" is right when clip identity is the file name somewhere else
    (Oceanship).  key="member" is right when the path itself carries the label
    (ShipsEar, whose stems repeat across groups).
    """

    def __init__(self, path: Path, meta: dict[str, dict], stem_re: re.Pattern | None,
                 key: str = "stem"):
        self.path = path
        self.meta = meta
        self.key = key
        self._zf: zipfile.ZipFile | None = None
        self._members: dict[str, str] = {}
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                stem = Path(info.filename).stem
                if stem_re is not None and not stem_re.match(stem):
                    continue
                k = info.filename if key == "member" else stem
                self._members.setdefault(k, info.filename)

    def _zip(self) -> zipfile.ZipFile:
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.path)
        return self._zf

    def records(self) -> list[dict]:
        rows = []
        for k, member in self._members.items():
            m = self.meta.get(k)
            if m is not None:
                rows.append({"key": k, "member": member, **m})
        return rows

    def open_audio(self, rec: dict):
        return self._zip().open(rec["member"])

    def close(self) -> None:
        if self._zf is not None:
            self._zf.close()
            self._zf = None


class DirSource(Source):
    """Loose wav files on disk (the extracted form of a release).

    `files` maps the key used by the record table straight to a path, which is
    what DeepShip needs: its identity is the dataset-relative path, not the stem.
    """

    def __init__(self, root: Path, meta: dict[str, dict], stem_re: re.Pattern | None,
                 files: dict[str, Path] | None = None):
        self.root = root
        self.meta = meta
        self._files: dict[str, Path] = dict(files) if files is not None else {}
        if files is None:
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.suffix.lower() in (".wav", ".flac"):
                    stem = p.stem
                    if stem_re is not None and not stem_re.match(stem):
                        continue
                    self._files.setdefault(stem, p)

    def records(self) -> list[dict]:
        rows = []
        for k, path in self._files.items():
            m = self.meta.get(k)
            if m is not None:
                rows.append({"key": k, "path": str(path), **m})
        return rows

    def open_audio(self, rec: dict):
        return open(rec["path"], "rb")


class StreamSource(Source):
    """A gzip-compressed tar split into byte chunks: one sequential pass only.

    Random access is impossible here, so this source is used when no .zip is
    present.  It is why measure() and write() are separable: this one only ever
    gets the write pass, with a generous allocation.
    """

    streamable = True

    def __init__(self, groups: list[tuple[str, list[Path]]], meta: dict[str, dict],
                 stem_re: re.Pattern | None):
        self.groups = groups
        self.meta = meta
        self.stem_re = stem_re

    def records(self) -> list[dict]:
        return [{"key": k, **v} for k, v in self.meta.items()]

    def iter_members(self):
        for base, parts in self.groups:
            reader = MultiPartReader(parts)
            with tarfile.open(fileobj=reader, mode="r|*") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    stem = Path(member.name).stem
                    if self.stem_re is not None and not self.stem_re.match(stem):
                        continue
                    fh = tf.extractfile(member)
                    if fh is not None:
                        yield stem, fh


def resolve_source(root: Path, meta: dict[str, dict], stem_re: re.Pattern | None) -> Source:
    """Prefer random access, fall back to streaming.

    Accepts the folder that holds the audio or the archive itself, since a
    dataset can be mounted either way and both are natural to pass in.

    Everything is decided from ONE walk.  The obvious implementation - look for
    a zip, then look for split parts, then list the wavs - walks the tree three
    times, and when the release is the extracted form that is three passes over
    53,770 files on a mounted dataset.
    """
    if root.is_file():
        if root.name.lower().endswith(".zip"):
            return ZipSource(root, meta, stem_re)
        if _looks_like_archive(root):
            return StreamSource([(root.name, [root])], meta, stem_re)
        root = root.parent

    zips: list[Path] = []
    loose: dict[str, Path] = {}
    parts: list[Path] = []
    # os.walk rather than Path.rglob: rglob builds a Path object per entry and
    # stats every one to decide if it is a directory.  On the extracted release
    # that is 53,770 stats plus 53,770 objects, which measured in the minutes on
    # a mounted dataset.  os.walk hands back names that are already known to be
    # files, so there is nothing to stat and nothing to allocate for the ones we
    # skip.
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        base = Path(dirpath)
        for name in sorted(filenames):
            low = name.lower()
            if low.endswith(".zip"):
                zips.append(base / name)
            elif low.endswith(".wav") or low.endswith(".flac"):
                stem = name.rsplit(".", 1)[0]
                if stem_re is None or stem_re.match(stem):
                    loose.setdefault(stem, base / name)
            elif CHUNK_RE.match(name):
                parts.append(base / name)

    if zips:
        zips.sort(key=lambda p: -p.stat().st_size)
        return ZipSource(zips[0], meta, stem_re)

    groups: dict[tuple, list[tuple[int, Path]]] = {}
    for p in parts:
        m = CHUNK_RE.match(p.name)
        if m:
            groups.setdefault((p.parent, m.group("base")), []).append((int(m.group("idx")), p))
    sets = []
    for (_, base), items in groups.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda t: t[0])
        first = items[0][1]
        if _looks_like_archive(first):
            sets.append((base, [q for _, q in items]))
    if sets:
        return StreamSource(sets, meta, stem_re)

    return DirSource(root, meta, stem_re, files=loose)


# --------------------------------------------------------------------------- #
# per-corpus record tables
# --------------------------------------------------------------------------- #
OCEANSHIP_STEM = re.compile(
    r"^(?P<stem>[0-9T:.Z-]+_\d+_id_[^_]+_type[a-zA-Z]+_\d+(?:_\d+)?)$")


def oceanship_meta(inventory: Path) -> dict[str, dict]:
    """Oceanship: identity and grouping come from the official CSV-derived inventory."""
    out: dict[str, dict] = {}
    with open_text(inventory) as fh:
        for r in csv.DictReader(fh):
            out[r["clip_id"]] = {
                "clip_id": r["clip_id"],
                "rec_key": r["rec_key"],
                "label": r["label"],
                "label_idx": r["label_idx"],
                "mmsi": r["mmsi"],
                "official_split": r["official_split"],
                "ais_type_code": r["ais_type_code"],
            }
    return out


def deepship_meta(recordings: Path) -> dict[str, dict]:
    """DeepShip: identity comes from the reconstructed recording table.

    wav_rel is relative to the dataset root, e.g. Cargo/20171104-1/1.wav.
    """
    out: dict[str, dict] = {}
    with recordings.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rel = r["wav_rel"].replace("\\", "/")
            out[rel] = {
                "clip_id": rel,
                "rec_key": r["rec_key"],
                "class_dir": r["class_dir"],
                "date": r["date"],
                "ship_name": r.get("ship_name", ""),
                "join_method": r.get("join_method", ""),
            }
    return out


SHIPSEAR_MEMBER = re.compile(r"^[^/]+/(?P<class_dir>[^/]+)/(?P<group>[^/]+)/(?P<stem>[^/]+)$")

# `shipsear_5s_16k/<class_dir>/<group>/<file>.wav` is the shape inside the zip;
# matching on the last four components means the extracted tree and the archive
# produce the same clip_id.
SHIPSEAR_TAIL = 4


def shipsear_meta(zip_path: Path) -> dict[str, dict]:
    """ShipsEar: the slice keeps class and group in the path, nothing else.

    The shipped *_list*.txt files are not used: their train/test split puts the
    same vessel on both sides (measured: 445/445 test clips contaminated in all
    six splits), so it is a leakage demo, not a protocol.
    """
    out: dict[str, dict] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".wav"):
                continue
            m = SHIPSEAR_MEMBER.match(info.filename.replace("\\", "/"))
            if not m:
                continue
            out[info.filename] = {
                "clip_id": info.filename,
                "member": info.filename,
                "class_dir": m.group("class_dir"),
                "group": m.group("group"),
            }
    return out


def shipsear_meta_dir(root: Path) -> tuple[dict[str, dict], dict[str, Path]]:
    """The same records, read from the tree Kaggle extracted the zip into.

    Kaggle expands archives when a dataset is ingested, so the upload arrives as
    `shipsear/ShipsEar/shipsear_5s_16k/<class_dir>/<group>/<file>.wav`.  Keying
    on the last four path components reproduces the zip member name exactly, so
    `index.csv` is identical either way and a later switch cannot silently
    renumber the corpus.
    """
    out: dict[str, dict] = {}
    files: dict[str, Path] = {}
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() != ".wav":
            continue
        parts = p.relative_to(root).parts
        if len(parts) < SHIPSEAR_TAIL:
            continue
        key = "/".join(parts[-SHIPSEAR_TAIL:])
        m = SHIPSEAR_MEMBER.match(key)
        if not m:
            continue
        out.setdefault(key, {
            "clip_id": key,
            "member": key,
            "class_dir": m.group("class_dir"),
            "group": m.group("group"),
        })
        files.setdefault(key, p)
    return out, files


# --------------------------------------------------------------------------- #
# the two passes
# --------------------------------------------------------------------------- #
def _name_of(rec: dict) -> str:
    return str(rec.get("clip_id") or rec.get("key") or "?")


def _plan_row(rec: dict, info) -> dict:
    """One measured clip: how many 5 s rows it needs, and its real properties."""
    dur = info.frames / float(info.samplerate)
    n = len(fit_length(np.zeros(int(dur * TGT_SR) + 1, dtype=np.float32)))
    return {"rec": rec, "n_rows": n, "src_sr": info.samplerate,
            "src_ch": info.channels, "src_dur_s": round(dur, 3)}


def measure(src: Source, records: list[dict], limit: int) -> tuple[list[dict], list[dict]]:
    """Read headers only.  Returns (plan, failures).

    This is what makes the allocation exact, and what produces the duration
    distribution the data-description section needs.  Guessing instead would mean
    writing an 8 GB memmap and then rewriting it just to trim, and the trim copy
    does not fit alongside the array it is trimming.
    """
    if limit:
        records = records[:limit]
    plan: list[dict] = []
    failed: list[tuple[str, str]] = []
    t0 = time.time()
    for i, rec in enumerate(records, 1):
        try:
            fh = src.open_audio(rec)
            try:
                info = info_any(fh)
            finally:
                fh.close()
        except Exception as exc:
            failed.append((_name_of(rec), f"{type(exc).__name__}: {exc}"))
            continue
        plan.append(_plan_row(rec, info))
        if i % 5000 == 0:
            print(f"    measured {i:,}/{len(records):,}  {time.time()-t0:.0f}s", flush=True)
    print(f"  [measure] {len(plan):,} clips -> {sum(p['n_rows'] for p in plan):,} rows "
          f"in {time.time()-t0:.0f}s, {len(failed)} unreadable", flush=True)
    return plan, failed


def measure_stream(src: "StreamSource", records: list[dict], limit: int):
    """The same plan for an archive that can only be read front to back.

    Costs one extra decompression pass, roughly a few minutes, and buys an exact
    allocation.  Plan order is archive order and the write pass walks the archive
    the same way, so the two stay aligned row for row.
    """
    by_key = {r["key"]: r for r in records}
    plan: list[dict] = []
    failed: list[tuple[str, str]] = []
    t0 = time.time()
    for key, fh in src.iter_members():
        rec = by_key.get(key)
        if rec is None:
            continue
        try:
            info = info_any(fh)
        except Exception as exc:
            failed.append((key, f"{type(exc).__name__}: {exc}"))
            continue
        plan.append(_plan_row(rec, info))
        n = len(plan)
        if n % 5000 == 0:
            print(f"    measured {n:,}  {time.time()-t0:.0f}s", flush=True)
        if limit and n >= limit:
            break
    print(f"  [measure] {len(plan):,} clips -> {sum(p['n_rows'] for p in plan):,} rows "
          f"in {time.time()-t0:.0f}s, {len(failed)} unreadable", flush=True)
    return plan, failed


def write(job: str, src: Source, plan: list[dict], out: Path, reps: str,
          device: str, failed: list[tuple[str, str]],
          stream=None) -> dict:
    """The transform pass.  One memmap per representation, row aligned.

    `stream` is a zero-argument callable returning an iterator of (item, handle).
    A random-access source opens by record; a streamed archive re-walks itself in
    the order the plan was built in.

    A clip that measured fine but fails to decode here has its rows filled with
    the dB floor instead of shifting every later row.  Row alignment is the
    contract the splits and the metadata joins rest on, and every fill is counted
    and reported, so this is loud rather than silent.
    """
    want_mel = "mel" in reps
    want_spec = "spec" in reps
    n_rows = sum(p["n_rows"] for p in plan)
    shape = (n_rows, N_MELS, N_FRAMES)

    out.mkdir(parents=True, exist_ok=True)
    mel_fb, lin_fb = build_filterbanks()

    logmel = (np.lib.format.open_memmap(out / "logmel.npy", mode="w+",
                                        dtype=np.float16, shape=shape) if want_mel else None)
    spec = (np.lib.format.open_memmap(out / "spec.npy", mode="w+",
                                      dtype=np.float16, shape=shape) if want_spec else None)

    idx_fh = (out / "index.csv").open("w", newline="", encoding="utf-8")
    idx_w = csv.writer(idx_fh)
    idx_w.writerow(IDX_COLUMNS[job])

    floor = np.full((N_MELS, N_FRAMES), -TOP_DB, dtype=np.float16)
    row = 0
    filled = 0
    skipped_short = 0
    done = 0
    t0 = time.time()

    def fill_floor(item: dict) -> None:
        nonlocal row, filled
        k = item["n_rows"]
        if logmel is not None:
            logmel[row: row + k] = floor
        if spec is not None:
            spec[row: row + k] = floor
        for j in range(k):
            idx_w.writerow(idx_row(job, item["rec"], item, j, row + j))
        row += k
        filled += k

    def by_record():
        for item in plan:
            yield item, src.open_audio(item["rec"])

    def by_archive():
        want = {p["rec"]["key"]: p for p in plan}
        for key, fh in src.iter_members():
            item = want.get(key)
            if item is not None:
                yield item, fh

    source = stream() if stream is not None else by_record()
    consumed: set[int] = set()

    for i, (item, fh) in enumerate(source, 1):
        rec = item["rec"]
        try:
            x, sr = read_any(fh)
        except Exception as exc:
            failed.append((_name_of(rec), f"{type(exc).__name__}: {exc}"))
            x = None
        finally:
            try:
                fh.close()
            except Exception:
                pass
        consumed.add(id(item))

        if x is None:
            fill_floor(item)
            done += 1
            continue

        chunks = fit_length(to_mono_16k(x, sr))
        if not chunks:
            skipped_short += 1
            done += 1
            continue
        block = np.stack(chunks)

        for b0 in range(0, len(block), BATCH):
            sub = block[b0: b0 + BATCH]
            power = stft_power_batch(sub, device)
            if logmel is not None:
                logmel[row: row + len(sub)] = to_db(mel_fb @ power).astype(np.float16)
            if spec is not None:
                spec[row: row + len(sub)] = to_db(lin_fb @ power).astype(np.float16)
            for j in range(len(sub)):
                idx_w.writerow(idx_row(job, rec, item, b0 + j, row + j))
            row += len(sub)
        done += 1

        if done % 2000 == 0 or done == len(plan):
            el = time.time() - t0
            print(f"    [{done:,}/{len(plan):,}] rows={row:,}  {done/el:.1f} clip/s  "
                  f"eta {(len(plan)-done)/max(done/el,1e-9)/60:.1f} min", flush=True)

    # anything the plan expected but the walk never produced
    for item in plan:
        if id(item) not in consumed:
            failed.append((_name_of(item["rec"]), "not present in the archive"))
            fill_floor(item)

    for arr in (logmel, spec):
        if arr is not None:
            arr.flush()
    idx_fh.close()
    if row != n_rows:  # can only happen if the plan changed mid-run
        raise RuntimeError(f"row count drifted: wrote {row:,}, planned {n_rows:,}")
    return {"rows": row, "planned": n_rows, "skipped_short": skipped_short,
            "floor_filled": filled, "failed": failed,
            "elapsed_min": (time.time() - t0) / 60}


IDX_COLUMNS = {
    "oceanship": ["row", "clip_id", "clip_idx", "rec_key", "label", "label_idx",
                  "mmsi", "official_split", "ais_type_code", "src_sr", "src_ch", "src_dur_s"],
    "deepship": ["row", "clip_id", "clip_idx", "rec_key", "class_dir", "date",
                 "ship_name", "join_method", "src_sr", "src_ch", "src_dur_s"],
    "shipsear": ["row", "clip_id", "clip_idx", "class_dir", "group",
                 "src_sr", "src_ch", "src_dur_s"],
}


def idx_row(job: str, rec: dict, item: dict, clip_idx: int, row: int) -> list:
    base = [row, rec.get("clip_id", rec.get("key")), clip_idx]
    tail = [item["src_sr"], item["src_ch"], item["src_dur_s"]]
    if job == "oceanship":
        return base + [rec["rec_key"], rec["label"], rec["label_idx"], rec["mmsi"],
                       rec["official_split"], rec["ais_type_code"]] + tail
    if job == "deepship":
        return base + [rec["rec_key"], rec["class_dir"], rec["date"],
                       rec.get("ship_name", ""), rec.get("join_method", "")] + tail
    return base + [rec["class_dir"], rec["group"]] + tail


# --------------------------------------------------------------------------- #
# job drivers
#
# Measuring and writing are separate steps on purpose.  Measuring costs a full
# header pass over every clip, so it happens ONCE and its plan is handed to the
# write step.  It is also the only way the space guard can mean anything: the
# total is knowable before the first byte is written, and nowhere else.
#
# v4 fix: the guard used to be evaluated after the jobs ran, i.e. after ~14 GB
# had already been written, so it could only ever fire when it was too late to
# matter - it aborted on a full disk and left the artifact with no manifest.
# --------------------------------------------------------------------------- #
def prepare_job(job: str, src: Source, limit: int) -> dict:
    """Measure one job.  Writes nothing, returns the plan to hand to commit_job."""
    print()
    print("=" * 88)
    print(f"job: {job}   ({src.label})")
    print("=" * 88)
    records = src.records()
    print(f"  records available: {len(records):,}", flush=True)
    if not records:
        print("  NOTHING TO DO - the source was found but holds no matching clips.")
        return {"job": job, "src": src, "plan": [], "failed": [], "stream": None,
                "planned": 0, "error": "no records"}

    if getattr(src, "streamable", False):
        plan, failed = measure_stream(src, records, limit)
        stream = lambda: _archive_items(src, plan)  # noqa: E731
    else:
        plan, failed = measure(src, records, limit)
        stream = None

    if not plan:
        print("  NOTHING TO DO - no readable headers.")
        return {"job": job, "src": src, "plan": [], "failed": failed, "stream": None,
                "planned": 0, "error": "no readable audio"}

    report_plan(job, plan)
    return {"job": job, "src": src, "plan": plan, "failed": failed, "stream": stream,
            "planned": sum(p["n_rows"] for p in plan)}


def commit_job(prep: dict, out: Path, reps: str, device: str) -> dict:
    """Write one already-measured job and report it."""
    job, plan = prep["job"], prep["plan"]
    if not plan:
        return {"rows": 0, "planned": 0, "failed": prep["failed"],
                "error": prep.get("error", "nothing to do")}

    stats = write(job, prep["src"], plan, out, reps, device, prep["failed"],
                  stream=prep["stream"])

    lines = [
        "=" * 78,
        f"{job} feature cache",
        "=" * 78,
        f"clips planned   : {len(plan):,}",
        f"rows written    : {stats['rows']:,}  (planned {stats['planned']:,})",
        f"shape           : ({stats['rows']}, {N_MELS}, {N_FRAMES}) fp16 x {len(reps.split('+'))}",
        f"representations : {reps}",
        f"dropped short   : {stats['skipped_short']:,} clips under {MIN_KEEP_S:g}s",
        f"floor-filled    : {stats.get('floor_filled', 0):,} rows (unreadable audio)",
        f"failed          : {len(stats['failed'])}",
    ]
    lines += [f"    {k}: {v}" for k, v in stats["failed"][:10]]
    lines += [f"elapsed         : {stats['elapsed_min']:.1f} min", f"written         : {out}"]
    text = "\n".join(lines)
    print()
    print(text)
    (out / f"w2_build_{job}_report.txt").write_text(text, encoding="utf-8")
    return stats


def _archive_items(src: Source, plan: list[dict]):
    """Walk a streamed archive, yielding only the plan entries, in archive order."""
    want = {p["rec"]["key"]: p for p in plan}
    for key, fh in src.iter_members():
        item = want.get(key)
        if item is not None:
            yield item, fh


def report_plan(job: str, plan: list[dict]) -> None:
    """Print the class row counts and the duration distribution per class."""
    by_class: dict[str, list[float]] = collections.defaultdict(list)
    n_rows: dict[str, int] = collections.defaultdict(int)
    key = {"oceanship": "label", "deepship": "class_dir", "shipsear": "class_dir"}[job]
    for p in plan:
        c = str(p["rec"].get(key) or p["rec"].get("group", "?"))
        by_class[c].append(p["src_dur_s"])
        n_rows[c] += p["n_rows"]
    print(f"  {'class':<28}{'clips':>8}{'rows':>8}{'med dur s':>11}{'min':>7}{'max':>7}")
    for c in sorted(by_class, key=lambda x: -len(by_class[x])):
        d = np.array(by_class[c])
        print(f"  {c:<28}{len(d):>8,}{n_rows[c]:>8,}{np.median(d):>11.2f}{d.min():>7.2f}{d.max():>7.2f}")
    d = np.array([p["src_dur_s"] for p in plan])
    print(f"  {'(all)':<28}{len(d):>8,}{sum(n_rows.values()):>8,}"
          f"{np.median(d):>11.2f}{d.min():>7.2f}{d.max():>7.2f}")
    print(f"  classes: {len(by_class)}   short (<{MIN_KEEP_S:g}s, dropped): {(d < MIN_KEEP_S).sum():,}")
    if job == "shipsear":
        g = collections.Counter(str(p["rec"].get("group", "?")) for p in plan)
        print(f"  groups ({len(g)}): {dict(sorted(g.items()))}")
    sr = collections.Counter(p["src_sr"] for p in plan)
    ch = collections.Counter(p["src_ch"] for p in plan)
    print(f"  source sample rates: {dict(sr.most_common(4))}")
    print(f"  source channels    : {dict(ch.most_common(4))}")


def estimate_gb(job: str, n_rows: int, reps: str) -> float:
    return n_rows * BYTES_PER_ROW * len(reps.split("+")) / 1e9


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #
README = """# uwatr-features

Frozen acoustic representations for three underwater acoustic corpora, built by
`build_features.py` (also in this folder) so that every later experiment reuses
one feature space instead of re-deriving it.

## Layout

    build_features.py        the exact script that produced this
    manifest.json            counts, shapes, per-split row counts, config, hashes
    oceanship/{logmel.npy,spec.npy,index.csv}
    deepship/{logmel.npy,spec.npy,index.csv}
    shipsear/{logmel.npy,spec.npy,index.csv}
    splits/                  frozen protocol splits + leakage audit CSVs

## Representation (identical for all three corpora)

    16 kHz mono, 5 s rows, 80,000 samples
    log-mel  (N, 128, 313) fp16, 20-8000 Hz, Slaney-style area normalisation
    linear   (N, 128, 313) fp16, 513 bins average-pooled into 128 bands
    both in dB relative to the clip peak, floored at -80 dB

Short clips are padded, not dropped, and the *measured* source duration of every
row is in `index.csv` as `src_dur_s`.  That column exists so a
with/without-short-clips ablation is one join away: duration is class-correlated
(Oceanship Passenger median 3.67 s against Tug 5.00 s), so padding without
recording it would hand the model a silent shortcut.

## Joining a split

    oceanship  index.clip_id                  -> splits/oceanship_splits.csv.gz  (clip_id)
    deepship   index.(rec_key, clip_idx)      -> splits/deepship_splits.csv.gz   (rec_key, clip_idx)

`splits/` carries three protocols x five seeds each: `clip_random`,
`rec_random`, `vessel_disjoint`, plus the per-protocol leakage summary.  Use
`vessel_disjoint` for anything that claims to measure generalisation.

## Mounting

    /kaggle/input/<this-dataset>/oceanship/logmel.npy     np.load(..., mmap_mode="r")
    /kaggle/input/<this-dataset>/oceanship/index.csv
    /kaggle/input/<this-dataset>/splits/oceanship_splits.csv.gz

## Whatever the uploader did to the archive

Kaggle expands archives when it ingests a dataset.  A folder uploaded as
`uwatr-workspace.zip` arrives one level deeper, `ShipsEar.zip` arrives as
`shipsear/ShipsEar/shipsear_5s_16k/<class>/<group>/<file>.wav`, and
`oceanship_inventory.csv.gz` can arrive as a plain `.csv`.  Every input path here
accepts either form, the extracted ShipsEar tree produces the *same* `clip_id`s
as the archive (both match on the last four path components), and the published
artifact always uses the documented name and compression so nothing downstream
has to guess.
"""


def assemble(out: Path, workspace: Path | None, summaries: dict) -> None:
    """Copy the frozen splits in, then write manifest.json and README.md."""
    (out / "splits").mkdir(parents=True, exist_ok=True)
    copied, absent = [], []
    if workspace is not None:
        # (shipped name, name in the artifact).  The artifact always uses the
        # shipped spelling: Kaggle may have expanded the upload, and downstream
        # notebooks should not have to guess.
        wanted = (("splits/oceanship_splits.csv.gz", "oceanship_splits.csv.gz"),
                  ("splits/oceanship_leakage_summary.csv", "oceanship_leakage_summary.csv"),
                  ("splits/splits.csv.gz", "deepship_splits.csv.gz"),
                  ("splits/split_summary.csv", "deepship_split_summary.csv"),
                  ("metadata/oceanship_inventory.csv.gz", "oceanship_inventory.csv.gz"))
        for rel, name in wanted:
            src = find_rel(workspace, rel)
            if src is None:
                absent.append(rel)
                continue
            copy_as(src, out / "splits" / name)
            copied.append(name)
    if not copied:
        print("[assemble] WARNING: no workspace dataset found; splits/ will be empty")
    elif absent:
        print(f"[assemble] WARNING: not found in the workspace: {absent}")

    manifest = {"config": {"tgt_sr": TGT_SR, "seg_s": SEG_S, "n_fft": N_FFT, "hop": HOP,
                           "n_mels": N_MELS, "n_frames": N_FRAMES, "top_db": TOP_DB,
                           "min_keep_s": MIN_KEEP_S},
                "jobs": {}, "splits_copied": copied}
    for job, s in summaries.items():
        jd = out / job
        entry = {"rows": s.get("rows", 0), "planned": s.get("planned"),
                 "skipped_short": s.get("skipped_short", 0),
                 "n_failed": len(s.get("failed", [])),
                 "elapsed_min": round(s.get("elapsed_min", 0.0), 1),
                 "arrays": {}}
        for name in ("logmel.npy", "spec.npy"):
            p = jd / name
            if p.exists():
                a = np.load(p, mmap_mode="r")
                entry["arrays"][name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                                         "gb": round(p.stat().st_size / 1e9, 3)}
                del a
        idx = jd / "index.csv"
        if idx.exists():
            with idx.open(encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            entry["index_rows"] = len(rows)
            if job == "oceanship" and rows:
                entry["classes"] = len({r["label"] for r in rows})
                entry["vessels"] = len({r["mmsi"] for r in rows})
            elif job == "deepship" and rows:
                entry["classes"] = len({r["class_dir"] for r in rows})
            elif job == "shipsear" and rows:
                entry["classes"] = len({r["class_dir"] for r in rows})
        manifest["jobs"][job] = entry

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "README.md").write_text(README, encoding="utf-8")

    print()
    print("=" * 88)
    print("output")
    print("=" * 88)
    total = 0
    for p in sorted(out.rglob("*")):
        if p.is_file():
            total += p.stat().st_size
            print(f"  {p.stat().st_size:>15,}  {p.relative_to(out)}")
    print(f"  {'-' * 15}")
    print(f"  {total:>15,} total = {total/1e9:.2f} GB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", default="all",
                    choices=["oceanship", "deepship", "shipsear", "all"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workspace", type=Path, default=None,
                    help="mounted uwatr-workspace dataset (for splits + inventory)")
    ap.add_argument("--oceanship-audio", type=Path, default=None)
    ap.add_argument("--oceanship-inventory", type=Path, default=None,
                    help="clip_id/label/mmsi table; .csv.gz or an expanded .csv")
    ap.add_argument("--deepship-root", type=Path, default=None)
    ap.add_argument("--deepship-recordings", type=Path, default=None)
    ap.add_argument("--shipsear-zip", type=Path, default=None,
                    help="ShipsEar.zip, or the ShipsEar/ folder Kaggle extracted it into")
    ap.add_argument("--reps", default="mel+spec", choices=["mel", "spec", "mel+spec"])
    ap.add_argument("--device", default="cuda" if (torch and torch.cuda.is_available()) else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="first N clips per job (smoke test)")
    ap.add_argument("--probe", action="store_true",
                    help="headers only: report counts and durations, write nothing")
    ap.add_argument("--skip-check", action="store_true", help="do not enforce the space guard")
    args = ap.parse_args()

    print("=" * 88)
    print("UATR frozen feature dataset")
    print("=" * 88)
    print(f"  out        : {args.out}")
    print(f"  reps       : {args.reps}   device: {args.device}")
    print(f"  torch      : {'yes ' + torch.__version__ if torch else 'NOT AVAILABLE (cpu only)'}")
    print(f"  resample   : {'scipy resample_poly' if resample_poly else 'MISSING - pip install scipy'}")

    jobs = ["oceanship", "deepship", "shipsear"] if args.job == "all" else [args.job]
    summaries: dict[str, dict] = {}

    ws = args.workspace
    inv = args.oceanship_inventory or (find_rel(ws, "metadata/oceanship_inventory.csv.gz")
                                       if ws else None)
    recs = args.deepship_recordings or (find_rel(ws, "deepship/deepship_recordings.csv")
                                        if ws else None)
    sea = args.shipsear_zip or (find_rel(ws, "shipsear/ShipsEar.zip") if ws else None)
    if sea is None and ws is not None:
        # extracted by Kaggle: shipsear/ShipsEar/ instead of shipsear/ShipsEar.zip
        for cand in sorted((ws / "shipsear").glob("ShipsEar*")) if (ws / "shipsear").is_dir() else []:
            if cand.is_dir():
                sea = cand
                break

    # ------------------------------------------------------------------ #
    # phase 1: resolve a source per job and MEASURE it.  Nothing is written.
    # ------------------------------------------------------------------ #
    prepared: list[dict] = []

    if "oceanship" in jobs:
        if args.oceanship_audio and inv and Path(inv).exists():
            meta = oceanship_meta(Path(inv))
            print(f"  oceanship  : {len(meta):,} inventory rows, audio root {args.oceanship_audio}")
            src = resolve_source(Path(args.oceanship_audio), meta, OCEANSHIP_STEM)
            src.label = f"{type(src).__name__} {getattr(src, 'path', getattr(src, 'root', ''))}"
            prepared.append(prepare_job("oceanship", src, args.limit))
        else:
            print("  oceanship  : SKIPPED (need --oceanship-audio and the inventory csv)")

    if "deepship" in jobs:
        if args.deepship_root and recs and Path(recs).exists():
            meta = deepship_meta(Path(recs))
            root = Path(args.deepship_root)
            # the dataset may be mounted with an extra folder level; find the one
            # that actually holds Cargo/ or Passenger/
            if not any((root / c).is_dir() for c in ("Cargo", "Passenger", "Tanker", "Tug")):
                for cand in sorted(root.rglob("*")):
                    if cand.is_dir() and (cand / "Cargo").is_dir():
                        root = cand
                        break
            print(f"  deepship   : {len(meta):,} recordings, resolved root {root}")
            present = {rel: (root / rel) for rel in meta if (root / rel).exists()}
            print(f"  deepship   : {len(present):,} of {len(meta):,} wav files present")
            src = DirSource(root, meta, None, files=present)
            src.label = f"DirSource {root}"
            prepared.append(prepare_job("deepship", src, args.limit))
        else:
            print("  deepship   : SKIPPED (need --deepship-root and the recordings csv)")

    if "shipsear" in jobs:
        if sea and Path(sea).exists():
            sea = Path(sea)
            if sea.is_dir():
                # Kaggle expanded the upload: read the extracted tree instead of
                # the archive, and it is faster because nothing is decompressed.
                meta, files = shipsear_meta_dir(sea)
                print(f"  shipsear   : {len(meta):,} clips in the extracted tree "
                      f"{sea.name}/  (no archive present)")
                src = DirSource(sea, meta, None, files=files)
            else:
                meta = shipsear_meta(sea)
                print(f"  shipsear   : {len(meta):,} clips in {sea.name}")
                src = ZipSource(sea, meta, None, key="member")
            src.label = f"{type(src).__name__} {sea.name}"
            prepared.append(prepare_job("shipsear", src, args.limit))
        else:
            print("  shipsear   : SKIPPED (need --shipsear-zip, a .zip or the "
                  "extracted ShipsEar/ folder)")

    if args.probe:
        print()
        print("  probe only - nothing written.  Re-run without --probe to build.")
        return 0

    # ------------------------------------------------------------------ #
    # phase 2: ONE space check for the whole run, before the first write
    # ------------------------------------------------------------------ #
    planned_rows = sum(p.get("planned", 0) for p in prepared)
    need = estimate_gb("all", planned_rows, args.reps)
    args.out.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.out).free / 1e9
    headroom = 1.0 if args.reps == "mel" else 2.0
    print()
    print(f"  planned rows {planned_rows:,}  ->  about {need:.2f} GB of arrays "
          f"for --reps {args.reps};  {free:.1f} GB free")
    if not args.skip_check and need > free - headroom:
        print(f"  ABORT BEFORE WRITING: needs {need:.2f} GB but only {free:.1f} GB is "
              f"free ({headroom:.0f} GB kept as headroom).")
        print("  options, in order of least loss:")
        print("    --reps mel          halves the cache (one representation, not two)")
        print("    free space on /kaggle/working, or delete a previous run's output")
        print("    --limit N           build a subset for a smoke test")
        print("  Nothing was written, so a re-run starts clean.")
        return 1

    # ------------------------------------------------------------------ #
    # phase 3: write, then assemble
    # ------------------------------------------------------------------ #
    # the artifact carries the script that produced it
    me = Path(__file__).resolve()
    if me != (args.out / "build_features.py").resolve():
        shutil.copy2(me, args.out / "build_features.py")
    else:
        print("  build_features.py: already at the output root (run from the output"
              " folder), nothing to copy")

    summaries: dict[str, dict] = {}
    for prep in prepared:
        summaries[prep["job"]] = commit_job(prep, args.out / prep["job"],
                                            args.reps, args.device)
        if hasattr(prep["src"], "close"):
            prep["src"].close()

    assemble(args.out, ws if (ws and ws.exists()) else None, summaries)

    print()
    print("  next: download this folder and upload it as a Kaggle Dataset, or")
    print("        create the dataset straight from this notebook version's output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
