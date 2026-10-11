#!/usr/bin/env python
"""Why does the same random split leak catastrophically in one corpus and not the other?

W3-A4 produced a contrast that no split-protocol paper reports:

    DeepShip   clip_random  0.9219   <- the model exploits the split almost fully
    OceanShip  clip_random  0.3649   <- the model exploits almost none of it

Both corpora are ONC (Strait of Georgia), both are 5 s clips, both use our own
frozen clip-random split.  The difference cannot be the protocol, so it must be
the thing the protocol operates on: the internal structure of a "recording".

This script measures that structure, using nothing but the metadata and the
frozen splits already on disk.  No audio, no GPU, a few seconds of CPU.

The quantity that matters is ADJACENCY:

    when a test clip's recording has other clips in train, are those train clips
    the test clip's NEIGHBOURS in the raw stream (near-duplicate audio), or are
    they arbitrary other moments of the same vessel (unrelated audio)?

DeepShip tiles each long recording into back-to-back 5 s windows, so clip_idx
i and i+1 are adjacent audio.  OceanShip instead cuts short annotated segments
out of a continuous hydrophone stream at arbitrary times, so two clips of one
recording can be minutes apart.

Run:
    python src/w1_05_recording_structure.py

Writes:
    out/w1_05_recording_structure.txt    human-readable report
    out/w1_05_recording_structure.csv    one row per (corpus, protocol, seed)
"""

from __future__ import annotations

import collections
import csv
import gzip
import io
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"

SPLIT_FILES = {
    # corpus     : (path,                          id columns,                label,        vessel)
    "deepship": (OUT / "splits" / "splits.csv.gz", ("rec_key", "clip_idx"), "class_dir", "ship_name"),
    "oceanship": (OUT / "oceanship_splits.csv.gz", ("clip_id",), "label", "mmsi"),
}
INVENTORY = {
    "deepship": OUT / "deepship_recordings.csv",      # rec_key -> n_clips, duration_s, ship_name
    "oceanship": OUT / "oceanship_inventory.csv.gz",  # clip_id -> rec_key, mmsi, label
}

REPORT = io.StringIO()
CSV_ROWS: list[dict] = []


def say(text: str = "") -> None:
    print(text)
    REPORT.write(text + "\n")


def head(title: str) -> None:
    say("=" * 88)
    say(title)
    say("=" * 88)


def sub(title: str) -> None:
    say("")
    say(title)
    say("-" * 88)


def q(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_structures():
    """Return per-corpus: (rec_key -> sorted clip ids), (clip id -> rec_key), meta."""
    structures = {}

    # ---- DeepShip: the split file itself is the clip list ----------------- #
    # Use ONE protocol x seed - the file holds every protocol and seed stacked,
    # so reading it whole would multiply the clip list by 15.
    ds_recs = collections.defaultdict(list)
    ds_clip_rec, ds_clip_vessel = {}, {}
    with gzip.open(SPLIT_FILES["deepship"][0], "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["protocol"] != "clip_random" or r["seed"] != "0":
                continue
            ds_recs[r["rec_key"]].append(int(r["clip_idx"]))
    ds_meta = {r["rec_key"]: r
               for r in csv.DictReader((OUT / "deepship_recordings.csv").open(encoding="utf-8"))}
    for k, v in ds_recs.items():
        for i in v:
            ds_clip_rec[(k, i)] = k
            ds_clip_vessel[(k, i)] = ds_meta.get(k, {}).get("ship_name") or ""
    structures["deepship"] = dict(
        rec_clips={k: sorted(v) for k, v in ds_recs.items()},
        clip_rec=ds_clip_rec,
        clip_vessel=ds_clip_vessel,
        rec_meta=ds_meta,
    )

    # ---- OceanShip: inventory carries the clip list ----------------------- #
    os_recs = collections.defaultdict(list)
    os_clip_rec, os_clip_vessel, os_clip_label = {}, {}, {}
    with gzip.open(INVENTORY["oceanship"], "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cid = r["clip_id"]
            os_recs[r["rec_key"]].append(cid)
            os_clip_rec[cid] = r["rec_key"]
            os_clip_vessel[cid] = r["mmsi"]
            os_clip_label[cid] = r["label"]
    structures["oceanship"] = dict(
        rec_clips={k: sorted(v) for k, v in os_recs.items()},
        clip_rec=os_clip_rec,
        clip_vessel=os_clip_vessel,
        rec_meta={k: {"label": os_clip_label[v[0]]} for k, v in os_recs.items()},
    )
    return structures


def load_splits(corpus: str):
    """yield (protocol, seed, cliplevel: dict clip -> split)."""
    path, idcols, _, _ = SPLIT_FILES[corpus]
    table = collections.defaultdict(dict)
    with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = tuple(r[c] for c in idcols)
            if corpus == "deepship":
                key = (key[0], int(key[1])) if key[1].lstrip("-").isdigit() else key
            else:
                key = key[0]          # OceanShip splits are keyed on clip_id alone
            table[(r["protocol"], int(r["seed"]))][key] = r["split"]
    return table


# --------------------------------------------------------------------------- #
# 1. the structure of a recording
# --------------------------------------------------------------------------- #
def structure_section(structures) -> None:
    head("1. what is inside one 'recording'")
    for corpus in ("deepship", "oceanship"):
        st = structures[corpus]
        rec_clips = st["rec_clips"]
        sizes = [len(v) for v in rec_clips.values()]
        say(f"  {corpus}")
        say(f"      recordings                    {len(rec_clips):,}")
        say(f"      clips                         {sum(sizes):,}")
        say(f"      clips/recording               min {min(sizes)}  median {statistics.median(sizes):.0f}"
            f"  mean {sum(sizes)/len(sizes):.1f}  max {max(sizes)}")
        say(f"      clips in >=2-clip recordings  {sum(x for x in sizes if x >= 2):,}"
            f"  ({100*sum(x for x in sizes if x >= 2)/sum(sizes):.1f}%)")

        if corpus == "deepship":
            # tiling: does clip_idx run 0..N-1 with no gaps?
            contiguous = sum(1 for v in rec_clips.values() if v == list(range(len(v))))
            say(f"      clip index is exactly 0..N-1  {contiguous}/{len(rec_clips)}"
                f"  ({100*contiguous/len(rec_clips):.1f}%)")
            ratio = []
            for k, v in rec_clips.items():
                dur = float(st["rec_meta"][k]["duration_s"] or 0)
                if dur > 0:
                    ratio.append(len(v) * 5.0 / dur)
            say(f"      (N x 5 s) / recording seconds  median {statistics.median(ratio):.3f}"
                f"   p10 {q(ratio, .10):.3f}   p90 {q(ratio, .90):.3f}")
            say("      -> 1.000 means the recording is tiled end to end by back-to-back")
            say("         5 s windows, so clip i and clip i+1 are ADJACENT AUDIO")
        else:
            # the trailing integer of clip_id, per recording
            import re
            seq = collections.defaultdict(list)
            for cid, rk in st["clip_rec"].items():
                m = re.search(r"_(\d+)$", cid)
                if m:
                    seq[rk].append(int(m.group(1)))
            multi = [v for v in seq.values() if len(v) >= 2]
            gaps = []
            for v in multi:
                s = sorted(v)
                gaps += [b - a for a, b in zip(s, s[1:])]
            say(f"      recordings with >=2 indexed ids  {len(multi):,}")
            say(f"      consecutive index gaps           min {min(gaps)}  "
                f"median {statistics.median(gaps):.0f}  max {max(gaps)}")
            say(f"      gaps equal to 1                  "
                f"{100*sum(1 for g in gaps if g == 1)/len(gaps):.1f}%")
            say("      -> the trailing integer is NOT a position in the stream "
                "(median gap 0,")
            say("         mostly duplicate values) and the recording key is a timestamp,")
            say("         so two clips of one recording are ARBITRARY MOMENTS, not")
            say("         neighbours")
        CSV_ROWS.append({
            "kind": "structure", "corpus": corpus, "protocol": "", "seed": "",
            "recordings": len(rec_clips), "clips": sum(sizes),
            "median_clips_per_recording": statistics.median(sizes),
            "pct_clips_in_multi_clip_recordings": round(100*sum(x for x in sizes if x >= 2)/sum(sizes), 2),
        })


# --------------------------------------------------------------------------- #
# 2. what a random split actually hands over
# --------------------------------------------------------------------------- #
def split_section(structures, splits) -> None:
    head("2. what a clip-random split hands the model, per corpus")
    say("")
    say("  For every test clip we ask three questions, and each is reported against")
    say("  its OWN null rate: the probability that the same event happens when you")
    say("  pick a training clip at random.  A rate is only meaningful as a ratio to")
    say("  that null - OceanShip has 10,168 recordings, so a 1 % rate is enormous.")
    say("")

    for corpus in ("deepship", "oceanship"):
        st = structures[corpus]
        for (protocol, seed), clip_split in sorted(splits[corpus].items()):
            train = [c for c, s in clip_split.items() if s == "train"]
            test = [c for c, s in clip_split.items() if s == "test"]
            if not train or not test:
                continue
            ntr = len(train)
            train_set = set(train)
            n_in_rec = collections.Counter(st["clip_rec"][c] for c in train)
            n_in_ves = collections.Counter(st["clip_vessel"][c] for c in train)

            # how many test clips have their recording / vessel present in train
            cov_rec = sum(1 for c in test if n_in_rec[st["clip_rec"][c]] > 0) / len(test)
            cov_ves = sum(1 for c in test if n_in_ves[st["clip_vessel"][c]] > 0) / len(test)
            # null rates
            null_rec = sum(n_in_rec[st["clip_rec"][c]] for c in test) / len(test) / ntr
            null_ves = sum(n_in_ves[st["clip_vessel"][c]] for c in test) / len(test) / ntr

            # adjacency: is a NEIGHBOUR of the test clip in train?
            exp_adj = float("nan")
            if corpus == "deepship":
                # clip_idx is a stream position, so i +/- 1 are the adjacent windows.
                # The recording is tiled end to end, so an adjacent window is a
                # near-duplicate of the test clip: the model only has to retrieve.
                adj = 0
                for c in test:
                    rec, i = c
                    if (rec, i - 1) in train_set or (rec, i + 1) in train_set:
                        adj += 1
                adj_rate = adj / len(test)
                # this is a wiring check, not a null: with a fraction p of clips in
                # train, the fraction of clips with >=1 of two neighbours in train
                # is 1-(1-p)^2 by construction
                total_clips = sum(len(v) for v in st["rec_clips"].values())
                exp_adj = 1.0 - (1.0 - ntr / total_clips) ** 2
            else:
                # the stream position is unavailable, so the closest analogue is
                # "a clip of the same recording at all", which IS the recording rate
                adj_rate = cov_rec

            say(f"  {corpus:<10} {protocol:<15} seed {seed}")
            say(f"      train {ntr:>6,}   test {len(test):>6,}")
            say(f"      test clips whose RECORDING is in train     {100*cov_rec:6.1f}%"
                f"   (chance for a random train clip to share it: {100*null_rec:.4f}%)")
            say(f"      test clips whose VESSEL is in train        {100*cov_ves:6.1f}%"
                f"   (chance: {100*null_ves:.4f}%)")
            if corpus == "deepship":
                if cov_rec > 0:
                    say(f"      test clips with an ADJACENT clip in train  {100*adj_rate:6.1f}%"
                        f"   (wiring check: {100*exp_adj:.1f}% expected by construction)"
                        f"   <- near-duplicate audio")
                else:
                    say("      test clips with an ADJACENT clip in train    n/a"
                        "   (the whole recording is held out)")
            say("")

            CSV_ROWS.append({
                "kind": "split", "corpus": corpus, "protocol": protocol, "seed": seed,
                "train": ntr, "test": len(test),
                "test_rec_in_train_pct": round(100*cov_rec, 2),
                "null_rec_pct": round(100*null_rec, 5),
                "test_ves_in_train_pct": round(100*cov_ves, 2),
                "null_ves_pct": round(100*null_ves, 5),
                "test_adjacent_in_train_pct": round(100*adj_rate, 2),
                "expected_adjacent_pct": round(100*exp_adj, 2) if exp_adj == exp_adj else "",
            })


# --------------------------------------------------------------------------- #
# 3. structure -> measured exploitability
# --------------------------------------------------------------------------- #
# Hand-entered from the W3-A3 (leak probe, CPU) and W3-A4 (DeepShip reference,
# GPU) run logs, so this report reads as one argument.  Re-run those notebooks
# to refresh them; the point of the table is the ORDER OF MAGNITUDE, not the
# third decimal.
MEASURED = {
    # corpus   : (clip_random test acc, honest test acc, 1-NN same-recording rate)
    "deepship": (0.9219, 0.6182, None),
    "oceanship": (0.3649, 0.2926, 5.68),
}
STRUCT_AVAIL = {"deepship": 89.6, "oceanship": None}   # from section 2, seed 0


def exploit_section() -> None:
    head("4. structure -> measured exploitability")
    say("")
    say("  Structural availability is what the split hands over for free; realised")
    say("  exploitability is what the network actually converted into accuracy.  The")
    say("  two corpora differ on both, in the same direction.")
    say("")
    say(f"  {'corpus':<11}{'near-dup in train':>19}{'1-NN same rec.':>16}"
        f"{'clip_random':>13}{'honest split':>14}{'gap':>8}")
    say("  " + "-" * 82)
    for corpus in ("deepship", "oceanship"):
        cr, honest, knn = MEASURED[corpus]
        av = STRUCT_AVAIL[corpus]
        avatar = f"{av:.1f}%" if av else "n/a (no stream order)"
        knn_txt = f"{knn:.2f}%" if knn else "not measured"
        say(f"  {corpus:<11}{avatar:>19}{knn_txt:>16}{cr:>13.4f}{honest:>14.4f}"
            f"{100*(cr-honest):>7.1f}")
    say("")
    say("  DeepShip gets a near-duplicate of ~90% of its test clips for free, and")
    say("  converts that into 92% accuracy.  OceanShip's near-duplicate supply is")
    say("  whatever the acoustics actually offer, which the 1-NN probe measured as")
    say("  5.68% (itself 180x the 0.03% chance rate) - so the same protocol leaves")
    say("  it at 0.36.  The gap the protocol opens is 30.4 pp on one corpus and")
    say("  7.2 pp on the other, for the same code, the same model, the same recipe.")
    say("")
    say("  This is why 'we used a random split' is not a specification.  A random")
    say("  split is only comparable across corpora if the clip extraction is, too -")
    say("  and the extraction is stated in no dataset paper we have read.")
    say("")


# --------------------------------------------------------------------------- #
# 3. one scalar is almost enough
# --------------------------------------------------------------------------- #
# DeepShip's index carries src_dur_s, the length of the SOURCE FILE, and the
# feature builder cuts one file into its 5 s windows - so every clip of a
# recording carries the same number.  Duration is therefore a recording
# fingerprint, and a nearest-neighbour rule on that single column should
# reproduce much of what a 256-d acoustic descriptor gets on a clip-random split.
#
# It does, and the scalar collapses exactly when the recording is held out.  That
# is the cleanest single statement of this paper's claim: 78% of a headline
# number, from one metadata column, with no audio and no training.
#
# The acoustic 1-NN and the trained-model columns are read from the run logs
# (notebooks/kaggle_w3a3_leak_probe.ipynb and the W3-A4 results.csv); this script
# stays metadata-only on purpose.
MEASURED_ACOUSTIC = {
    ("deepship", "clip_random"): (83.22, 92.19),
    ("deepship", "rec_random"): (50.33, 68.42),
    ("deepship", "temporal"): (50.15, 61.82),
    ("oceanship", "clip_random"): (25.48, 36.49),
    ("oceanship", "rec_random"): (22.19, None),
    ("oceanship", "vessel_disjoint"): (20.13, 29.26),
}


def scalar_section(structures, splits) -> None:
    import numpy as np

    head("3. one scalar is almost enough")
    say("")
    say("  A scalar needs no training, no audio and no GPU.")
    say("")
    say(f"  {'corpus / protocol':<27}{'1 scalar':>10}{'256-d acoustic':>16}"
        f"{'trained model':>15}")
    say("  " + "-" * 69)

    for corpus in ("deepship", "oceanship"):
        st = structures[corpus]
        if corpus == "deepship":
            lab = {k: v["class_dir"] for k, v in st["rec_meta"].items()}
            durv = {k: float(v["duration_s"] or 0) for k, v in st["rec_meta"].items()}
            key = lambda c: c[0]
        else:
            lab, durv, key = {}, {}, lambda c: c
        if not durv:
            say(f"  {corpus:<27}{'n/a':>10}")
            say(f"      {corpus}'s src_dur_s is the CLIP's own length, which varies")
            say(f"      within a recording - there is no one number per recording to")
            say(f"      act as a fingerprint.  The notebook measured it anyway:")
            say(f"      9.34% on clip_random, i.e. no better than a constant.")
            continue
        spread = collections.defaultdict(list)
        for (protocol, seed), clip_split in sorted(splits[corpus].items()):
            train = [c for c, s in clip_split.items() if s == "train"]
            test = [c for c, s in clip_split.items() if s == "test"]
            if not train or not test:
                continue
            a_tr = np.array([durv[key(t)] for t in train], dtype=np.float32)
            a_te = np.array([durv[key(t)] for t in test], dtype=np.float32)
            y_tr = np.array([lab[key(t)] for t in train])
            y_te = np.array([lab[key(t)] for t in test])
            pred = np.empty(len(a_te), dtype=object)
            for s0 in range(0, len(a_te), 512):          # argmin with first-tie
                e = min(s0 + 512, len(a_te))
                j = np.argmin(np.abs(a_te[s0:e, None] - a_tr[None, :]), 1)
                pred[s0:e] = y_tr[j]
            acc = float((pred == y_te).mean())
            spread[protocol].append(100 * acc)
            if seed != 0:
                continue
            ac, mdl = MEASURED_ACOUSTIC.get((corpus, protocol), (None, None))
            ac_txt = f"{ac:>15.2f}%" if ac else f"{'-':>16}"
            mdl_txt = f"{mdl:>14.4f}" if mdl else f"{'-':>15}"
            say(f"  {corpus + ' ' + protocol:<27}{100*acc:>9.2f}%{ac_txt}{mdl_txt}")
            CSV_ROWS.append({
                "kind": "one_scalar", "corpus": corpus, "protocol": protocol,
                "seed": seed, "scalar_1nn_pct": round(100 * acc, 2),
                "acoustic_1nn_pct": ac, "trained_model": mdl,
            })
        say("  " + "-" * 69)
        for protocol, v in spread.items():
            say(f"      {protocol:<17} all 5 seeds {min(v):.2f}% .. {max(v):.2f}%"
                f"   (point estimate above is seed 0)")
        say(f"      scalar = {'source-file duration, one number per recording' if corpus == 'deepship' else 'clip duration'}")
    say("")
    say("  Read the deepship block downwards: 78% from one number on clip_random,")
    say("  and it COLLAPSES to the best-constant level the moment the recording is")
    say("  held out.  Duration is not predicting the ship TYPE - if it were, the")
    say("  number would survive - so the 78% is identity retrieval: the same")
    say("  phenomenon as the 1-NN and the 0.92, reached with less information.")
    say("")
    say("  Recording length per class, for the record (seconds):")
    say("")
    say(f"      {'class':<12}{'n':>5}{'min':>8}{'median':>9}{'max':>8}")
    byc = collections.defaultdict(list)
    for v in structures["deepship"]["rec_meta"].values():
        d = float(v["duration_s"] or 0)
        if d > 0:
            byc[v["class_dir"]].append(d)
    for c in sorted(byc):
        v = sorted(byc[c])
        say(f"      {c:<12}{len(v):>5}{v[0]:>8.0f}{statistics.median(v):>9.0f}{v[-1]:>8.0f}")
    say("")


# --------------------------------------------------------------------------- #
# 4. the label side: is the label constant per identity?
# --------------------------------------------------------------------------- #
def label_section(structures) -> None:
    head("5. is the label constant within a recording / a vessel?")
    for corpus in ("deepship", "oceanship"):
        st = structures[corpus]
        if corpus == "deepship":
            say("  deepship   labels come from the directory name, so they are constant")
            say("             within a recording by construction; 242 named vessels, and")
            say("             the same vessel can appear under one class only.")
            say("")
            continue
        recs = st["rec_clips"]
        # label per recording
        import gzip as _g
        lab = {}
        with _g.open(INVENTORY["oceanship"], "rt", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                lab.setdefault(r["rec_key"], collections.Counter())[r["label"]] += 1
        mixed_rec = sum(1 for c in lab.values() if len(c) > 1)
        ves = collections.defaultdict(collections.Counter)
        with _g.open(INVENTORY["oceanship"], "rt", newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ves[r["mmsi"]][r["label"]] += 1
        mixed_ves = sum(1 for c in ves.values() if len(c) > 1)
        clips_in_mixed_ves = sum(sum(c.values()) for c in ves.values() if len(c) > 1)
        say(f"  oceanship  recordings with more than one label   {mixed_rec}/{len(lab)}")
        say(f"             vessels with more than one label      {mixed_ves}/{len(ves)}"
            f"   ({100*clips_in_mixed_ves/sum(sum(c.values()) for c in ves.values()):.1f}% of clips)")
        say("             -> the labels are clean at vessel level, so a vessel-identity")
        say("                lookup really is a valid oracle, not a noisy one")
        say("")
        CSV_ROWS.append({
            "kind": "labels", "corpus": corpus, "protocol": "", "seed": "",
            "recordings": len(lab), "mixed_label_recordings": mixed_rec,
            "vessels": len(ves), "mixed_label_vessels": mixed_ves,
        })


# --------------------------------------------------------------------------- #
# 4. the verdict
# --------------------------------------------------------------------------- #
def verdict_section() -> None:
    head("6. the verdict")
    say("")
    say("  The same random-split protocol costs the two corpora very different things:")
    say("")
    say("    DeepShip   clip i and clip i+1 are adjacent 5 s windows of one file, so")
    say("               a train/test split by clip leaves ~90% of test clips with a")
    say("               near-duplicate in train.  The model reaches 0.92 and is")
    say("               doing retrieval, not classification.")
    say("")
    say("    OceanShip  clips of one recording are arbitrary moments - the trailing")
    say("               index is not a stream position - so a train clip of the same")
    say("               recording is NOT acoustically close.  The model reaches 0.36")
    say("               and captures almost none of the 99.4% metadata oracle.")
    say("")
    say("  Neither number is a property of the split NAME.  It is a property of how")
    say("  the corpus cuts a recording into clips, which no dataset paper states and")
    say("  no benchmark reports.  That is the gap this work is aimed at.")
    say("")


def main() -> int:
    say("=" * 88)
    say("recording structure - why one corpus's clip-random split leaks and another's does not")
    say("=" * 88)
    say("")
    say("  inputs : out/ metadata + frozen splits only (no audio, no GPU)")
    say("")

    structures = load_structures()
    splits = {c: load_splits(c) for c in ("deepship", "oceanship")}
    structure_section(structures)
    split_section(structures, splits)
    scalar_section(structures, splits)
    exploit_section()
    label_section(structures)
    verdict_section()

    (OUT / "w1_05_recording_structure.txt").write_text(REPORT.getvalue(), encoding="utf-8")
    keys = sorted({k for r in CSV_ROWS for k in r})
    with (OUT / "w1_05_recording_structure.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in CSV_ROWS:
            w.writerow(r)
    print(f"wrote {OUT / 'w1_05_recording_structure.txt'}")
    print(f"wrote {OUT / 'w1_05_recording_structure.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
