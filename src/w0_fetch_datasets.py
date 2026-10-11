#!/usr/bin/env python3
"""W0  Fetch the three datasets used by this project.

Everything here is a free, public source. Only ShipsEar's ORIGINAL 90 recordings
need an email; the processed audio is on HuggingFace. See docs/DATASETS.md for the
full verification notes.

Usage
-----
    python w0_fetch_datasets.py --all --root G:/Work_work/under_water/data
    python w0_fetch_datasets.py --metafiles
    python w0_fetch_datasets.py --shipsear
    python w0_fetch_datasets.py --deepship          # needs Kaggle credentials

What each flag fetches
----------------------
--metafiles   the 4 official DeepShip metafiles from the GitHub repo
              (2 KB-ish each; the Kaggle mirror ships only 2 of the 4)
--shipsear    ShipsEar.zip from HuggingFace: 277 MB of processed ShipsEar
              (2223 clips, 16 kHz, 5 classes) under CC-BY-4.0
--deepship    the full 20 GB DeepShip mirror from Kaggle. Needs credentials:
              kaggle.com -> Settings -> API -> Create New Token, then put the
              downloaded kaggle.json at ~/.kaggle/kaggle.json

Design notes
------------
* Downloads are resumable-ish: a file that already matches its expected SHA-256
  is skipped, so re-running is cheap.
* Everything is verified against a known digest where one is available. Silent
  corruption in a 20 GB acoustic corpus would be very expensive to discover later.
* No third-party dependencies - urllib only, so it runs on a bare Python.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

UA = {"User-Agent": "uwatr-research-fetch/1.0"}

# --------------------------------------------------------------------------- #
# sources, each with the digest we expect where we have verified one
# --------------------------------------------------------------------------- #
METAFILES = {
    "cargo-metafile.txt": (
        "https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Cargo/cargo-metafile",
        "98D541A4D7A98D8048D410D254D89484B3082CE577593E9112FB4A635C0C40AA",
    ),
    "passengership-metafile.txt": (
        "https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Passengership/passengership-metafile",
        "3B2B28EEBCFFE6AA8C7E3DC880E9A226ED86920CE2CEAA41A4B5AE90209DA1F3",
    ),
    "tanker-metafile.txt": (
        "https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Tanker/tanker-metafile",
        "505F9E7C2BD0EC48D58E2C60CF2E6822FFE20CF32AC103472B786A5E5D0C90F6",
    ),
    "tug-metafile.txt": (
        "https://raw.githubusercontent.com/irfankamboh/DeepShip/main/Tug/tug-metafile",
        "9C17D512706CA36B0F48DEF74668939F59780EE5D0F295797723A21E9FF83B0B",
    ),
}

SHIPSEAR_ZIP = {
    "url": "https://huggingface.co/datasets/peng7554/DS3500/resolve/main/ShipsEar.zip",
    "name": "ShipsEar.zip",
    "sha256": "385C0F9961FBD6D8350584EA7B5E45A76A20013575C35C3C87EFC48C75347D58",
    "size": 290_700_328,
    # digest equals the LFS oid HuggingFace reports for this file, so a match
    # proves the download is byte-identical to the published object
}

DEEPSHIP_KAGGLE = ["tangqiji/deepship-raw", "deku14all1/deepship"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest().upper()


def download(url: str, dest: Path, expected_sha: str | None = None, label: str = "") -> bool:
    """Stream to a .part file, then rename. Skips if already verified."""
    label = label or dest.name
    if dest.exists():
        if expected_sha and sha256_of(dest) == expected_sha.upper():
            print(f"[skip] {label} already present and verified")
            return True
        if not expected_sha:
            print(f"[skip] {label} already present ({dest.stat().st_size:,} B)")
            return True
        print(f"[warn] {label} present but digest mismatch -> re-downloading")

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    got = 0
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = int(resp.headers.get("Content-Length") or 0)
            with part.open("wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = 100 * got / total
                        print(f"\r  {label}: {got/1e6:8.1f}/{total/1e6:.1f} MB ({pct:5.1f}%)",
                              end="", flush=True)
        print()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"\n[FAIL] {label}: {exc}")
        if part.exists():
            part.unlink()
        return False

    if expected_sha:
        actual = sha256_of(part)
        if actual != expected_sha.upper():
            print(f"[FAIL] {label} digest mismatch\n       expected {expected_sha}\n       actual   {actual}")
            part.unlink()
            return False
        print(f"[ok]   {label}  {got:,} B  sha256 verified")
    else:
        print(f"[ok]   {label}  {got:,} B")
    part.replace(dest)
    return True


# --------------------------------------------------------------------------- #
# fetchers
# --------------------------------------------------------------------------- #
def fetch_metafiles(root: Path) -> int:
    out = root / "metadata" / "raw_metafiles"
    print(f"\n=== DeepShip metafiles -> {out} ===")
    ok = 0
    for name, (url, digest) in METAFILES.items():
        if download(url, out / name, digest, label=name):
            ok += 1
    print(f"[metafiles] {ok}/{len(METAFILES)} ok")
    print("  note: the Kaggle mirror ships only cargo- and passengership-metafile;")
    print("        tanker and tug exist only in the GitHub repo, which is why we fetch all four.")
    return 0 if ok == len(METAFILES) else 1


def fetch_shipsear(root: Path) -> int:
    out = root / "external" / "shipsear_hf"
    print(f"\n=== ShipsEar (processed, HF) -> {out} ===")
    ok = download(SHIPSEAR_ZIP["url"], out / SHIPSEAR_ZIP["name"],
                  SHIPSEAR_ZIP["sha256"], label="ShipsEar.zip")
    if not ok:
        return 1
    print("  contents: shipsear_5s_16k/{0..4}/{cls}_{grp}/{cls}_{grp}_{i}.wav")
    print("            2223 clips @ 5 s / 16 kHz, 5 classes, 12 vessel-type groups")
    print("            plus train/test lists and 5 extra seed variants")
    print("  WARNING : the provided splits leak 100% (all 12 groups on both sides).")
    print("            Do not use them. Generate your own with w1_02_make_splits.py.")
    print("  WARNING : recording-level provenance is NOT preserved in this copy.")
    print("            For the original 90 recordings email dsantos@gts.uvigo.es.")
    return 0


def find_kaggle_creds() -> Path | None:
    for p in (Path.home() / ".kaggle" / "kaggle.json",
              Path.home() / ".kaggle" / "access_token"):
        if p.exists():
            return p
    return None


def fetch_deepship(root: Path) -> int:
    out = root / "external" / "deepship"
    print(f"\n=== DeepShip (full 47 h, Kaggle) -> {out} ===")
    creds = find_kaggle_creds()
    if creds is None:
        print("[needs credentials] Kaggle requires an API token for dataset downloads.")
        print()
        print("  How to get one:")
        print("    1. sign in at https://www.kaggle.com")
        print("    2. account -> Settings -> API -> 'Create New Token'")
        print("    3. save the downloaded kaggle.json to:")
        print(f"         {Path.home() / '.kaggle' / 'kaggle.json'}")
        print()
        print("  Then run either:")
        print(f"    kaggle datasets download -d {DEEPSHIP_KAGGLE[0]} -p \"{out}\" --unzip")
        print("  or simply use the Kaggle web UI: attach the dataset to a notebook")
        print("  (it mounts read-only at /kaggle/input/<slug>, which is what W2 expects).")
        print()
        print("  Recommended: skip the local download entirely and do everything on")
        print("  Kaggle - the 20 GB never needs to touch this machine.")
        return 2

    print(f"[ok] found Kaggle credentials at {creds}")
    if shutil.which("kaggle") is None:
        print("[warn] the kaggle CLI is not installed; falling back to instructions")
        print(f"    pip install kaggle")
        print(f"    kaggle datasets download -d {DEEPSHIP_KAGGLE[0]} -p \"{out}\" --unzip")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    for slug in DEEPSHIP_KAGGLE[:1]:
        print(f"  downloading {slug} ...")
        rc = subprocess.call(["kaggle", "datasets", "download", "-d", slug,
                              "-p", str(out), "--unzip"])
        if rc == 0:
            print("[ok] DeepShip downloaded")
            print("  expected content: 609 recordings, ~47 h, stereo 32 kHz, 4 classes")
            print("  verify with: python w1_01_build_metadata.py --root <dir> ...")
            return 0
        print(f"[FAIL] kaggle CLI returned {rc}")
    return 1


def fetch_uniqueship_notes() -> int:
    print("\n=== UniqueShip (not scripted - Google Drive) ===")
    print("  portal : https://uniqueshipdata.org")
    print("  paper  : https://arxiv.org/abs/2609.13659")
    print()
    print("  This one cannot be fetched by a script because the files live on Google")
    print("  Drive, which needs an interactive login. Recommended split:")
    print()
    print("    12 Class - 5 Hours Each   60 h / 4,218 vessels / 12 classes")
    print("                              26 GB unzipped, 17.8 GB zipped")
    print("      audio        https://drive.google.com/drive/folders/1HxV8AGh8ckZS3JehAU_gY2fuCnscX-eV")
    print("      spectrograms https://drive.google.com/drive/folders/1xOXX4dUmtSET7Utb4gfscZp1dlPoT76e")
    print()
    print("    5 Class - Balanced        213 h / 3,175 vessels / 5 classes")
    print("                              92 GB unzipped, 64.8 GB zipped")
    print("      audio        https://drive.google.com/drive/folders/1G0e-2Nhi6CWKuBRStlR8GTtcGyYiT2h6")
    print("      spectrograms https://drive.google.com/drive/folders/1a6wLqfuBh3PMqSlZfy-IVjTfrJMtqo11")
    print()
    print("  Tips that matter:")
    print("    * download the SPECTROGRAMS, not the audio - they are precomputed and")
    print("      skip our whole W2 step; everything downstream works the same")
    print("    * ACCESS IS GATED, verified: the embeddedfolderview endpoint (which")
    print("      serves fully public folders without login) returns 401 on all four")
    print("      folders, and the folder pages redirect to a Google sign-in page.")
    print("      The releases blurb says 'request Google Drive permission'; the")
    print("      inquiry blurb says 'no request or approval is required'. Trust the")
    print("      test, not the blurb - plan on a Google account plus a granted share")
    print("    * if Drive refuses, use the inquiry form or https://www.kitware.com/contact/")
    print("      and ask for the AWS link (the site says larger splits are moving to AWS)")
    print("    * Google Drive is not directly reachable from mainland China")
    print("    * there is also a 25-hour quick-start subset, good for a smoke test")
    print("    * no mirror exists anywhere: Kaggle, HuggingFace, ModelScope, Zenodo and")
    print("      the AWS Open Data registry all return zero results for 'uniqueship'")
    print("    * FALLBACK OF LAST RESORT: the page footer states the data is owned by")
    print("      Ocean Networks Canada under CC-BY 4.0, and ONC's data policy grants")
    print("      free unrestricted access via data.oceannetworks.ca. You could rebuild")
    print("      an equivalent from ONC - but that is the authors' entire paper's worth")
    print("      of engineering, so treat it as a last resort, not a plan")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the UATR datasets used by this project.")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--metafiles", action="store_true")
    ap.add_argument("--shipsear", action="store_true")
    ap.add_argument("--deepship", action="store_true")
    ap.add_argument("--uniqueship-notes", action="store_true")
    args = ap.parse_args()

    if not any([args.all, args.metafiles, args.shipsear, args.deepship, args.uniqueship_notes]):
        ap.print_help()
        return 0

    rc = 0
    if args.all or args.metafiles:
        rc |= fetch_metafiles(args.root)
    if args.all or args.shipsear:
        rc |= fetch_shipsear(args.root)
    if args.all or args.deepship:
        rc |= fetch_deepship(args.root)
    if args.all or args.uniqueship_notes:
        fetch_uniqueship_notes()

    print("\n" + "=" * 72)
    print("done" + ("" if rc == 0 else f" (with warnings, rc={rc})"))
    print("next: docs/DATASETS.md for the verification notes, then RUNBOOK.md for W2")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
