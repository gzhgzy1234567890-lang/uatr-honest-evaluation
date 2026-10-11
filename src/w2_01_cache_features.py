#!/usr/bin/env python3
"""W2-01  Cache the acoustic representations once, so every experiment reuses them.

Why this exists
---------------
DeepShip is ~20 GB of 32 kHz stereo wav.  Every training run that re-reads and
re-resamples it burns 20-30 minutes out of a 30 GPU-hour weekly budget.  Pay that
cost exactly once here, freeze the result as a Kaggle Dataset, and let every
downstream run mount it read-only.

What is produced
----------------
Two float16 arrays plus an index, all row-aligned:

    logmel.npy   (N, 128, 313)  log-mel spectrogram, dB relative to clip peak
    spec.npy     (N, 128, 313)  linear-frequency spectrogram, same convention
    index.csv    row -> clip_id, rec_key, clip_idx, class_dir, date

Rows are written in one canonical order (recordings sorted by class, date,
variant, index; clips in time order), so a row index is a stable clip id.

Two representations on purpose: if the strict-protocol collapse showed up in mel
features only, that would be a feature-engineering artefact; if it shows up in
both, it is a property of the data.  The paper needs the second statement, so the
ablation has to be built in from the start.

Design choices worth defending
------------------------------
* STFT is computed PER CLIP, not over the whole recording.  This is what "we
  segment into non-overlapping 5 s clips" actually means in every paper that
  reports it, and it guarantees exactly 313 frames per clip.  Computing the STFT
  first and slicing frames afterwards would misalign, because
  80_000 / 256 = 312.5 frames per clip is not an integer.
* dB relative to the clip peak, not per-clip z-scored.  It removes recorder gain
  (an instrument artefact) while keeping the within-clip dynamic range that
  carries the ship signature.  Absolute level - which the attribution study needs
  as a proxy for source range - is preserved separately in the metadata.
* float16 halves the footprint to ~2.7 GB per representation with no measurable
  effect on a CNN trained on dB inputs.
* open_memmap keeps memory flat regardless of clip count.

Output size: about 5.4 GB for both.  Fits the 20 GB working directory, and once
frozen as a Kaggle Dataset it stops counting against it.

Cost: roughly 20-30 minutes, GPU optional.  One-off.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    print("ERROR: needs soundfile (pip install soundfile)", file=sys.stderr)
    raise

try:
    from scipy.signal import decimate
except ImportError:  # pragma: no cover
    decimate = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

# --------------------------------------------------------------------------- #
# frozen representation configuration - the cache is a public artifact
# --------------------------------------------------------------------------- #
SRC_SR = 32_000
TGT_SR = 16_000
DECIM = SRC_SR // TGT_SR
SEG_S = 5.0
SEG_LEN = int(TGT_SR * SEG_S)  # 80_000
N_FFT = 1024
HOP = 256
N_BINS = N_FFT // 2 + 1  # 513
N_MELS = 128
N_FRAMES = 1 + SEG_LEN // HOP  # 313
MEL_FMIN, MEL_FMAX = 20.0, 8000.0
TOP_DB = 80.0

# how many clips to transform in one GPU batch (memory guard for long recordings)
BATCH = 64


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def build_filterbanks() -> tuple[np.ndarray, np.ndarray]:
    """Return (mel_fb, linear_fb), each (128, 513).

    Written by hand rather than pulled from librosa so the cache does not depend
    on the librosa version present in whatever Kaggle image happens to be current.
    """
    freqs = np.linspace(0.0, TGT_SR / 2, N_BINS)

    # mel: triangular filters, Slaney-style area normalisation
    mel_pts = np.linspace(_hz_to_mel(np.array([MEL_FMIN]))[0], _hz_to_mel(np.array([MEL_FMAX]))[0], N_MELS + 2)
    hz_pts = _mel_to_hz(mel_pts)
    mel_fb = np.zeros((N_MELS, N_BINS), dtype=np.float32)
    for i in range(N_MELS):
        lo, mid, hi = hz_pts[i], hz_pts[i + 1], hz_pts[i + 2]
        for j, f in enumerate(freqs):
            if lo <= f <= mid and mid > lo:
                mel_fb[i, j] = (f - lo) / (mid - lo)
            elif mid < f <= hi and hi > mid:
                mel_fb[i, j] = (hi - f) / (hi - mid)
    mel_fb *= (2.0 / (hz_pts[2 : N_MELS + 2] - hz_pts[:N_MELS]))[:, None]

    # linear: average-pool the 513 bins into 128 groups, so both representations
    # share a shape and the ablation compares like with like
    lin_fb = np.zeros((N_MELS, N_BINS), dtype=np.float32)
    edges = np.linspace(0, N_BINS, N_MELS + 1).astype(int)
    for i in range(N_MELS):
        a, b = edges[i], max(edges[i + 1], edges[i] + 1)
        lin_fb[i, a:b] = 1.0 / (b - a)
    return mel_fb, lin_fb


def load_mono_16k(path: Path) -> np.ndarray:
    """Read a DeepShip wav, downmix to mono, decimate 32 kHz -> 16 kHz.

    DeepShip wavs are stereo 32 kHz; 32 -> 16 kHz is an exact factor of two, so
    polyphase decimation is both cheaper and cleaner than a general resampler.
    """
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != SRC_SR:
        raise ValueError(f"expected {SRC_SR} Hz, got {sr}")
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if decimate is not None:
        return decimate(x, DECIM, ftype="fir", zero_phase=True).astype(np.float32)
    return x[::DECIM].astype(np.float32)  # fallback: crude but adequate


def stft_power_batch(clips: np.ndarray, device: str) -> np.ndarray:
    """(B, SEG_LEN) -> (B, 513, N_FRAMES) power.

    Both branches must agree frame-for-frame, because the cache is frozen as a
    public artifact and a CPU build has to be byte-comparable with a GPU build.
    torch.stft(center=True) pads by n_fft//2 with 'reflect' and yields
    1 + SEG_LEN // HOP = 313 frames.  scipy with boundary=None/padded=False would
    yield only 309, so the scipy branch pads identically first.  (Getting this
    wrong silently produced a 309-frame cache on CPU and a 313-frame cache on
    GPU - one more reason to freeze the representation once and hash it.)
    """
    win = np.hanning(N_FFT).astype(np.float32)
    if torch is not None and device != "cpu":
        t = torch.from_numpy(clips).to(device)
        w = torch.from_numpy(win).to(device)
        spec = torch.stft(
            t, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
            window=w, center=True, return_complex=True, pad_mode="reflect",
        )
        return spec.abs().pow(2)[:, :, :N_FRAMES].cpu().numpy()

    from scipy.signal import stft as sp_stft

    pad = N_FFT // 2
    out = np.empty((clips.shape[0], N_BINS, N_FRAMES), dtype=np.float32)
    for i, c in enumerate(clips):
        c = np.pad(c, (pad, pad), mode="reflect")  # match torch's center=True
        _, _, z = sp_stft(c, fs=TGT_SR, nperseg=N_FFT, noverlap=N_FFT - HOP,
                          window=win, boundary=None, padded=False)
        out[i] = (np.abs(z) ** 2)[:, :N_FRAMES]
    return out


def to_db(power: np.ndarray) -> np.ndarray:
    """Power -> dB relative to each clip's own peak, floored at -TOP_DB.

    Operates on (B, F, T) and normalises per clip, so a quiet recording is not
    penalised for its absolute level.  Value range is exactly [-TOP_DB, 0].
    """
    peak = power.max(axis=(1, 2), keepdims=True)
    peak = np.maximum(peak, 1e-12)
    db = 10.0 * np.log10(np.maximum(power, peak * 1e-12) / peak)
    return np.maximum(db, -TOP_DB).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Cache DeepShip representations.")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--recordings", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda" if (torch and torch.cuda.is_available()) else "cpu")
    ap.add_argument("--limit", type=int, default=0, help="first N recordings only (smoke test)")
    args = ap.parse_args()

    with args.recordings.open(encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    recs.sort(key=lambda r: (r["class_dir"], r["date"], r["variant"], int(r["index"])))
    if args.limit:
        recs = recs[: args.limit]

    planned = sum(int(r["n_clips"]) for r in recs)
    bytes_each = planned * N_MELS * N_FRAMES * 2
    print(f"[plan] {len(recs)} recordings -> up to {planned:,} clips of {SEG_S:g}s", flush=True)
    print(f"[plan] 2 arrays of ({N_MELS},{N_FRAMES}) fp16, ~{2*bytes_each/1e9:.2f} GB total", flush=True)
    print(f"[plan] device={args.device}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    mel_fb, lin_fb = build_filterbanks()

    shape = (planned, N_MELS, N_FRAMES)
    logmel = np.lib.format.open_memmap(args.out / "logmel.npy", mode="w+", dtype=np.float16, shape=shape)
    spec = np.lib.format.open_memmap(args.out / "spec.npy", mode="w+", dtype=np.float16, shape=shape)

    idx_fh = (args.out / "index.csv").open("w", newline="", encoding="utf-8")
    idx_w = csv.writer(idx_fh)
    idx_w.writerow(["row", "clip_id", "rec_key", "clip_idx", "class_dir", "date"])

    row = 0
    t0 = time.time()
    failed: list[tuple[str, str]] = []

    for i, r in enumerate(recs, 1):
        rec_key = r["rec_key"]
        try:
            x = load_mono_16k(args.root / r["wav_rel"])
        except Exception as exc:
            failed.append((rec_key, str(exc)))
            print(f"  [{i}/{len(recs)}] SKIP {rec_key}: {exc}", flush=True)
            continue

        n_clips = len(x) // SEG_LEN
        if n_clips < 1:
            failed.append((rec_key, f"only {len(x)} samples after decimation"))
            continue

        x = x[: n_clips * SEG_LEN].reshape(n_clips, SEG_LEN)

        for b0 in range(0, n_clips, BATCH):
            chunk = x[b0 : b0 + BATCH]
            power = stft_power_batch(chunk, args.device)          # (b, 513, 313)
            logmel[row : row + len(chunk)] = to_db(mel_fb @ power).astype(np.float16)
            spec[row : row + len(chunk)] = to_db(lin_fb @ power).astype(np.float16)
            for j in range(len(chunk)):
                idx_w.writerow([row + j, f"{rec_key}#{b0+j}", rec_key, b0 + j, r["class_dir"], r["date"]])
            row += len(chunk)

        if i % 25 == 0 or i == len(recs):
            el = time.time() - t0
            rate = i / el if el else 0
            eta = (len(recs) - i) / rate / 60 if rate else 0
            print(f"  [{i}/{len(recs)}] rows={row:,}  {rate:.2f} rec/s  eta {eta:.1f} min", flush=True)

    logmel.flush()
    spec.flush()
    idx_fh.close()

    if row < planned:
        for name in ("logmel", "spec"):
            arr = np.load(args.out / f"{name}.npy", mmap_mode="r")
            trimmed = np.asarray(arr[:row])
            del arr
            np.save(args.out / f"{name}.npy", trimmed.astype(np.float16))
            print(f"[trim] {name}.npy -> {trimmed.shape}")

    lines = [
        "=" * 72,
        f"cached rows : {row:,} (planned {planned:,})",
        f"shape       : ({row}, {N_MELS}, {N_FRAMES}) fp16 x2 representations",
        f"failed      : {len(failed)}",
    ]
    lines += [f"    {rk}: {why}" for rk, why in failed[:10]]
    lines += [f"elapsed     : {(time.time()-t0)/60:.1f} min", f"written     : {args.out}"]
    text = "\n".join(lines)
    print()
    print(text)
    (args.out / "w2_01_cache_report.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
