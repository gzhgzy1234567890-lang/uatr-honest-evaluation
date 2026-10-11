#!/usr/bin/env python3
"""W3 aggregation — turns run rows into the paper's tables and figures.

Three things this file exists to get right.

1. Recording-level bootstrap confidence intervals.
   Test clips are not independent.  One recording contributes dozens of clips
   that share a channel, a background noise field and an operating condition, so
   resampling clips would treat fifty clips from one ship as fifty independent
   pieces of evidence and report an interval that is far too narrow.  We
   resample RECORDINGS with replacement instead.  This is the interval a reviewer
   should ask for, and it is precisely what a recent DeepShip benchmark listed as
   future work.

2. Never an accuracy without its protocol.
   Every row carries the protocol and the measured leak rate, because a number
   quoted without them is meaningless — the same model moves by tens of points
   between protocols.

3. Where the accuracy comes from.
   For the leaky protocols the test set is split into clips whose vessel (and
   recording) was also seen in training and clips whose was not, and the metric
   is reported on each part.  That decomposition is the mechanism behind the
   headline gap: it shows how much of a 99% number is recognition and how much
   is recall of a ship the model already met.

Usage
-----
    python w3_agg.py --data /kaggle/input/under_water_features \
                     --runs /kaggle/working/w3-runs /kaggle/input/w3a-runs \
                     --out /kaggle/working/w3-tables
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

B_BOOT = 1000
CI = 95.0

PROTOCOL_ORDER = ("clip_random", "rec_random", "vessel_disjoint", "temporal", "group_disjoint")
STRICT = {"vessel_disjoint", "temporal", "group_disjoint"}

LABEL_COL = {"oceanship": "label", "deepship": "class_dir", "shipsear": "class_dir"}
REC_COL = {"oceanship": "rec_key", "deepship": "rec_key", "shipsear": "group"}
VESSEL_COL = {"oceanship": "mmsi", "deepship": "ship_name", "shipsear": "group"}
SPLIT_FILE = {"oceanship": "oceanship_splits.csv.gz", "deepship": "deepship_splits.csv.gz"}
SPLIT_KEY = {"oceanship": ["clip_id"], "deepship": ["rec_key", "clip_idx"]}


# --------------------------------------------------------------------------- #
def open_text(path: Path):
    """Read a possibly-gzipped text file by magic bytes (Kaggle expands `.gz`)."""
    path = Path(path)
    with open(path, "rb") as fh:
        head = fh.read(2)
    if head == b"\x1f\x8b":
        return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")
    return open(path, "r", newline="", encoding="utf-8", errors="replace")


def find_all(roots, pattern: str) -> list[Path]:
    """Every file named `pattern` under each root, up to three levels deep.

    Deep enough for `/kaggle/input/<slug>/...`, `/kaggle/input/<owner>/<slug>/...`
    and `/kaggle/working/<dir>/...` alike, so a run directory can be passed in,
    or the notebook output mounted as a dataset, without the caller knowing which.
    """
    out: list[Path] = []
    for r in roots:
        r = Path(r)
        if r.is_file() and r.name == pattern:
            out.append(r)
            continue
        if not r.is_dir():
            continue
        for depth in range(4):
            out += sorted(r.glob("/".join(["*"] * depth + [pattern]) if depth else pattern))
    seen, uniq = set(), []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    return uniq


def is_scratch(p: Path, base: Path | None = None) -> bool:
    """True if the path is scratch — a component below `base` starts with '_'.

    Only the part of the path BELOW the search root counts.  Inspecting the whole
    absolute path would also reject a perfectly good run directory that happens to
    sit under a folder someone named with a leading underscore.
    """
    try:
        rel = p.relative_to(base) if base is not None else Path(p.name)
    except ValueError:
        rel = Path(p.name)
    return any(str(part).startswith("_") for part in rel.parts)


def discover_run_roots() -> list[Path]:
    """Directories holding a results.csv, found under /kaggle/input and /kaggle/working.

    Scratch directories (leading underscore, relative to the root) are excluded.
    The `--smoke` probe writes a results.csv whose run id is identical to the real
    run of the same configuration, so scanning it would replace a 24-epoch
    headline number with a 2-epoch accuracy of 0.14.
    """
    found = []
    for base in (Path("/kaggle/input"), Path("/kaggle/working")):
        if not base.is_dir():
            continue
        for p in find_all([base], "results.csv"):
            if is_scratch(p, base):
                print(f"  [skip] scratch {p}")
                continue
            d = p.parent
            if d not in found:
                found.append(d)
    return found


def macro_f1(y_true, y_pred, n_classes) -> float:
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def bootstrap_over_recordings(y_true, y_pred, group, n_classes, b=B_BOOT, seed=0):
    """Resample clusters (recordings) with replacement; return metric CIs."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(group, return_inverse=True)
    if len(uniq) < 2:
        return None
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    groups = [order[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]
    accs = np.empty(b)
    f1s = np.empty(b)
    for i in range(b):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[j] for j in pick])
        accs[i] = (y_pred[idx] == y_true[idx]).mean()
        f1s[i] = macro_f1(y_true[idx], y_pred[idx], n_classes)
    lo, hi = (100 - CI) / 2, 100 - (100 - CI) / 2
    return {"acc_lo": float(np.percentile(accs, lo)), "acc_hi": float(np.percentile(accs, hi)),
            "f1_lo": float(np.percentile(f1s, lo)), "f1_hi": float(np.percentile(f1s, hi)),
            "n_clusters": int(len(uniq))}


def fmt(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
def load_train_identities(root: Path) -> dict:
    """{(corpus, protocol, seed): (train_recordings, train_vessels)} for the leak split.

    Recomputed from the frozen split file and the index, so the decomposition is
    derived from the protocol rather than from metadata smuggled through the
    prediction files.
    """
    out = {}
    for corpus, fname in SPLIT_FILE.items():
        idx_path = root / corpus / "index.csv"
        if not idx_path.exists():
            continue
        with idx_path.open(encoding="utf-8", newline="") as fh:
            idx = list(csv.DictReader(fh))
        keys = SPLIT_KEY[corpus]
        by_key = {tuple(r[k] for k in keys): r for r in idx}

        path = root / "splits" / fname
        if not path.exists():
            alt = root / "splits" / fname[:-3]
            path = alt if alt.exists() else path
        if not path.exists():
            continue
        acc: dict[tuple, dict] = defaultdict(lambda: {"rec": set(), "ves": set()})
        with open_text(path) as fh:
            for r in csv.DictReader(fh):
                if r["split"] != "train":
                    continue
                row = by_key.get(tuple(r[k] for k in keys))
                if row is None:
                    continue
                d = acc[(r["protocol"], int(r["seed"]))]
                d["rec"].add(row.get(REC_COL[corpus], ""))
                d["ves"].add(row.get(VESSEL_COL[corpus], ""))
        for k, v in acc.items():
            out[(corpus, *k)] = (v["rec"], v["ves"])
    return out


def leak_scoped_metrics(npz, train_ids):
    """Accuracy on the parts of the test set that did / did not share an identity."""
    y, p = npz["y_true"], npz["y_pred"]
    trec, tves = train_ids
    ves = np.array([str(x) for x in npz["vessel"]])
    rec = np.array([str(x) for x in npz["rec_key"]])
    out = {"all": (float((p == y).mean()) if len(y) else 0.0, int(len(y)))}
    for name, mask in (("seen_vessel", np.isin(ves, list(tves))),
                       ("unseen_vessel", ~np.isin(ves, list(tves))),
                       ("seen_recording", np.isin(rec, list(trec))),
                       ("unseen_recording", ~np.isin(rec, list(trec)))):
        out[name] = (float((p[mask] == y[mask]).mean()) if mask.sum() else None, int(mask.sum()))
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate W3 runs into tables and figures.")
    ap.add_argument("--data", type=Path, required=True, help="features root (for index + splits)")
    ap.add_argument("--runs", type=Path, nargs="*", default=None,
                    help="directories holding results.csv / predictions / embeddings")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/w3-tables"))
    ap.add_argument("--b", type=int, default=B_BOOT)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    roots = args.runs or discover_run_roots()
    # the same scratch rule applies to an explicitly passed directory, so
    # `--runs /kaggle/working/w3-runs /kaggle/working/_probe` cannot smuggle a
    # 2-epoch probe row into the headline table
    dropped = [r for r in roots if str(Path(r).name).startswith("_")]
    for d in dropped:
        print(f"  [skip] scratch {d}")
    roots = [r for r in roots if r not in dropped and Path(r).exists()]
    if not roots:
        raise SystemExit(
            "no run directory found. Pass --runs, or mount a previous notebook's "
            "output as a dataset so it lands under /kaggle/input."
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables").mkdir(exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)

    print("=" * 100)
    print("W3 aggregation")
    print("=" * 100)
    print(f"  features : {args.data}")
    print(f"  run dirs : {[str(r) for r in roots]}")

    # ------------------------------------------------------------------ results
    rows: list[dict] = []
    pred_dirs: list[Path] = []
    for r in roots:
        for res in find_all([Path(r)], "results.csv"):
            with res.open(encoding="utf-8", newline="") as fh:
                got = list(csv.DictReader(fh))
            print(f"  {res}  -> {len(got)} rows")
            rows += got
        for pd in sorted(Path(r).glob("*/predictions")) + sorted(Path(r).glob("predictions")):
            pred_dirs.append(pd)
    if not rows:
        raise SystemExit("no results.csv rows found")

    # Same run id in two places: keep the better-trained one and say so.  This is
    # the guard for a probe or a partial run having been picked up from a mounted
    # dataset, and it is deliberately not silent.
    by_id: dict[str, dict] = {}
    clashes = []
    for r in rows:
        old = by_id.get(r["run_id"])
        if old is None:
            by_id[r["run_id"]] = r
            continue
        try:
            new_key = (int(r.get("epochs") or 0), int(r.get("n_train") or 0))
            old_key = (int(old.get("epochs") or 0), int(old.get("n_train") or 0))
        except (TypeError, ValueError):
            new_key, old_key = (0, 0), (1, 0)
        if new_key > old_key:
            by_id[r["run_id"]] = r
        clashes.append((r["run_id"], old_key, new_key))
    if clashes:
        print(f"  [warn] {len(clashes)} duplicated run id(s); keeping the longest-trained row")
        for rid, a, b in clashes[:5]:
            print(f"          {rid}  kept {(max(a, b),)}  discarded {(min(a, b),)}")
    rows = list(by_id.values())
    print(f"  {len(rows)} unique runs")

    train_ids = load_train_identities(args.data)
    print(f"  train identity sets for {len(train_ids)} (corpus, protocol, seed) combos")

    # ---------------------------------------------------------- bootstrap CIs
    ci_rows, scope_rows = [], []
    for r in rows:
        run_id = r["run_id"]
        npz_path = None
        for pd in pred_dirs:
            cand = pd / f"{run_id}.npz"
            if cand.exists():
                npz_path = cand
                break
        if npz_path is None:
            continue
        d = np.load(npz_path, allow_pickle=True)
        n_classes = int(r["n_classes"])
        ci = bootstrap_over_recordings(d["y_true"], d["y_pred"], d["rec_key"],
                                       n_classes, b=args.b, seed=int(r["seed"]))
        if ci:
            ci_rows.append({"run_id": run_id, **ci})
        ids = train_ids.get((r["corpus"], r["protocol"], int(r["seed"])))
        if ids is not None:
            m = leak_scoped_metrics(d, ids)
            scope_rows.append({"run_id": run_id, "protocol": r["protocol"],
                               "corpus": r["corpus"], "rep": r["rep"],
                               "backbone": r["backbone"], "seed": r["seed"],
                               **{f"acc_{k}": v[0] for k, v in m.items()},
                               **{f"n_{k}": v[1] for k, v in m.items()}})

    ci_by_run = {c["run_id"]: c for c in ci_rows}
    if ci_rows:
        with (args.out / "tables" / "bootstrap_ci.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ci_rows[0].keys()))
            w.writeheader()
            w.writerows(ci_rows)
    if scope_rows:
        with (args.out / "tables" / "leak_scope.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(scope_rows[0].keys()))
            w.writeheader()
            w.writerows(scope_rows)

    # ------------------------------------------------------------------ export
    flat = args.out / "tables" / "all_runs.csv"
    cols = sorted({k for r in rows for k in r})
    with flat.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    L: list[str] = []
    add = L.append
    add("=" * 100)
    add("W3 tables")
    add("=" * 100)
    add("")
    add(f"runs: {len(rows)}   bootstrap: {args.b} resamples over RECORDINGS, {CI:.0f}% CI")
    add("")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["corpus"], r["protocol"], r["rep"], r["backbone"], r["method"],
                r["domain"], r["shuffled"])].append(r)

    def agg(rs: list[dict], metric="test_acc"):
        v = np.array([float(x[metric]) for x in rs if x.get(metric) not in (None, "", "None")])
        return (v.mean(), v.std(ddof=0), len(v)) if len(v) else (float("nan"), float("nan"), 0)

    # -------------------------------------------------- T1 the headline: protocol
    add("=" * 100)
    add("T2  Headline — the same model, different split units")
    add("=" * 100)
    add("")
    add(f"  {'corpus':<10}{'rep':<7}{'backbone':<16}{'method':<8}{'protocol':<17}"
        f"{'acc mean':>9}{'sd':>8}{'n':>4}{'F1':>8}{'95% CI (recording bootstrap)':>32}{'leak%':>8}")
    for (corpus, proto, rep, backbone, method, dom, shuf), rs in sorted(groups.items()):
        if shuf in ("1", 1) or method != "erm" or dom != "none":
            continue
        a, sd, n = agg(rs)
        f1 = agg(rs, "test_macro_f1")[0]
        cis = [ci_by_run[x["run_id"]] for x in rs if x["run_id"] in ci_by_run]
        ci_s = "-"
        if cis:
            lo = np.mean([c["acc_lo"] for c in cis])
            hi = np.mean([c["acc_hi"] for c in cis])
            ci_s = f"[{lo:.4f}, {hi:.4f}]"
        leak = np.mean([float(x["leak_rec_pct"]) for x in rs]) if rs[0].get("leak_rec_pct") else None
        add(f"  {corpus:<10}{rep:<7}{backbone:<16}{method:<6}{proto:<17}"
            f"{fmt(a):>9}{fmt(sd):>8}{n:>4}{fmt(f1):>8}{ci_s:>32}"
            f"{fmt(leak, 1) if leak is not None else '-':>8}")
    add("")

    # --------------------------------------------------- T2 the inflation gap
    add("=" * 100)
    add("T3  The inflation gap — accuracy lost when the split unit is made honest")
    add("=" * 100)
    add("")
    add(f"  {'corpus':<10}{'rep':<7}{'backbone':<16}"
        f"{'clip_random':>13}{'rec_random':>13}{'strict':>11}{'P1-P2':>9}{'P1-strict':>11}")
    for corpus in sorted({r["corpus"] for r in rows}):
        for rep in sorted({r["rep"] for r in rows if r["corpus"] == corpus}):
            for backbone in sorted({r["backbone"] for r in rows
                                    if r["corpus"] == corpus and r["rep"] == rep}):
                def mean_of(proto):
                    rs = [x for x in rows if x["corpus"] == corpus and x["rep"] == rep
                          and x["backbone"] == backbone and x["protocol"] == proto
                          and x["method"] == "erm" and x["shuffled"] in ("0", 0)]
                    return agg(rs)[0] if rs else None
                a1 = mean_of("clip_random")
                a2 = mean_of("rec_random")
                strict = [mean_of(p) for p in ("vessel_disjoint", "temporal", "group_disjoint")]
                a3 = next((s for s in strict if s is not None), None)
                if a1 is None and a3 is None:
                    continue
                add(f"  {corpus:<10}{rep:<7}{backbone:<16}{fmt(a1):>13}{fmt(a2):>13}{fmt(a3):>11}"
                    f"{fmt(None if (a1 is None or a2 is None) else a1 - a2):>9}"
                    f"{fmt(None if (a1 is None or a3 is None) else a1 - a3):>11}")
    add("")

    # ------------------------------- T3 where the accuracy comes from (mechanism)
    if scope_rows:
        add("=" * 100)
        add("T4  Where the accuracy comes from — test clips whose ship was already seen")
        add("=" * 100)
        add("")
        add("  For a leaky protocol the test set is not one population: some clips come")
        add("  from a ship the model trained on, some do not. Reporting one number hides")
        add("  the fact that the two parts behave completely differently.")
        add("")
        add(f"  {'corpus':<10}{'protocol':<17}{'all':>8}{'seen vessel':>13}{'unseen vessel':>15}"
            f"{'seen rec':>10}{'unseen rec':>12}")
        ag: dict[tuple, list[dict]] = defaultdict(list)
        for x in scope_rows:
            ag[(x["corpus"], x["protocol"])].append(x)
        for (corpus, proto), rs in sorted(ag.items()):
            def m(k):
                v = [x[f"acc_{k}"] for x in rs
                     if x.get(f"acc_{k}") not in (None, "", "None")]
                return float(np.mean(v)) if v else None
            add(f"  {corpus:<10}{proto:<17}{fmt(m('all')):>8}{fmt(m('seen_vessel')):>13}"
                f"{fmt(m('unseen_vessel')):>15}{fmt(m('seen_recording')):>10}"
                f"{fmt(m('unseen_recording')):>12}")
        add("")

    # ------------------------------------------------------------- T4 per-seed
    add("=" * 100)
    add("T5  Per-seed spread (the test set size is a random variable — never quote one seed)")
    add("=" * 100)
    add("")
    add(f"  {'corpus':<10}{'protocol':<30}{'acc per seed':<46}{'n_test per seed':<28}")
    per = defaultdict(list)
    for r in rows:
        if (r["method"] == "erm" and r["domain"] == "none" and r["shuffled"] in ("0", 0)
                and r["backbone"] == "cnn_small" and r["rep"] == "logmel"):
            per[(r["corpus"], r["protocol"])].append(r)
    for (corpus, proto), rs in sorted(per.items()):
        rs = sorted(rs, key=lambda x: int(x["seed"]))
        accs = [float(x["test_acc"]) for x in rs]
        ns = [int(x["n_test"]) for x in rs]
        add(f"  {corpus:<10}{proto:<30}{str([round(a, 3) for a in accs]):<46}{str(ns):<28}")
        if ns:
            add(f"  {'':<40}mean {np.mean(accs):.4f}  sd {np.std(accs):.4f}"
                f"   n_test {min(ns)}-{max(ns)} (mean {np.mean(ns):.0f}, "
                f"spread {100*(max(ns)-min(ns))/max(np.mean(ns),1):.0f}%)")
    add("")

    # ------------------------------------------------------------ T5 scale
    add("=" * 100)
    add("T6  Parameter count vs strict-protocol accuracy")
    add("=" * 100)
    add("")
    add("  Reproduces or contradicts Soti et al. (2026): no gain from 11.17M over ~157K.")
    add("")
    add(f"  {'corpus':<10}{'backbone':<16}{'params':>12}{'clip_random':>13}{'rec_random':>13}"
        f"{'strict':>11}")
    for corpus in sorted({r["corpus"] for r in rows}):
        for backbone in sorted({r["backbone"] for r in rows if r["corpus"] == corpus}):
            rs = [x for x in rows if x["corpus"] == corpus and x["backbone"] == backbone
                  and x["method"] == "erm" and x["domain"] == "none"
                  and x["shuffled"] in ("0", 0) and x["rep"] == "logmel"]
            if not rs:
                continue
            params = max(int(x["n_params"]) for x in rs)

            def mm(proto):
                s = [x for x in rs if x["protocol"] == proto]
                return agg(s)[0] if s else None
            strict = [mm(p) for p in ("vessel_disjoint", "temporal", "group_disjoint")]
            add(f"  {corpus:<10}{backbone:<16}{params:>12,}{fmt(mm('clip_random')):>13}"
                f"{fmt(mm('rec_random')):>13}"
                f"{fmt(next((v for v in strict if v is not None), None)):>11}")
    add("")

    # -------------------------------------------------------------- T6 methods
    dg = [r for r in rows if r["method"] != "erm" or r["domain"] != "none"]
    if dg:
        add("=" * 100)
        add("T7  Domain generalisation (C4) — domain labels from metadata, target domain unseen")
        add("=" * 100)
        add("")
        add("  `vessel_hash` is the NEGATIVE CONTROL: real partitions, no acoustic meaning.")
        add("  A gain there means the regulariser, not the domain structure, did the work.")
        add("")
        add(f"  {'corpus':<10}{'protocol':<17}{'rep':<7}{'backbone':<12}{'method':<7}"
            f"{'domain':<14}{'acc mean':>9}{'sd':>8}{'n':>4}{'vs ERM':>9}")
        base = {}
        for r in rows:
            if r["method"] == "erm" and r["domain"] == "none" and r["shuffled"] in ("0", 0):
                base[(r["corpus"], r["protocol"], r["rep"], r["backbone"])] = r
        g2: dict[tuple, list[dict]] = defaultdict(list)
        for r in dg:
            g2[(r["corpus"], r["protocol"], r["rep"], r["backbone"], r["method"],
                r["domain"])].append(r)
        for k, rs in sorted(g2.items()):
            a, sd, n = agg(rs)
            b = base.get(k[:4])
            delta = (a - float(b["test_acc"])) if b else None
            add(f"  {k[0]:<10}{k[1]:<17}{k[2]:<7}{k[3]:<12}{k[4]:<7}{k[5]:<14}"
                f"{fmt(a):>9}{fmt(sd):>8}{n:>4}{fmt(delta):>9}")
        add("")

    # ------------------------------------------------------ T7 cross-corpus
    cross = [r for r in rows if r["protocol"].startswith("cross_")]
    if cross:
        add("=" * 100)
        add("T8  Cross-corpus (unseen corpus) — shared 4-class vocabulary")
        add("=" * 100)
        add("")
        add(f"  {'direction':<34}{'rep':<7}{'backbone':<16}{'acc':>8}{'macro F1':>10}"
            f"{'chance':>8}{'n_test':>9}{'vocab_approx':>14}")
        for r in sorted(cross, key=lambda x: x["run_id"]):
            add(f"  {r['protocol']:<34}{r['rep']:<7}{r['backbone']:<16}"
                f"{fmt(float(r['test_acc'])):>8}{fmt(float(r['test_macro_f1'])):>10}"
                f"{fmt(float(r['chance_acc'])):>8}{r['n_test']:>9}{r.get('vocab_approx', '0'):>14}")
        add("")

    # --------------------------------------------------------- T8 diagnostics
    diag = [r for r in rows if r["shuffled"] in ("1", 1)]
    if diag:
        add("=" * 100)
        add("T9  Label-shuffling control (required by Hummel et al. 2026)")
        add("=" * 100)
        add("")
        add("  Training labels permuted, everything else identical.  A model that scores")
        add("  above chance here is not learning the task — it is reading a leak or a")
        add("  shortcut out of the split or the padding.")
        add("")
        add(f"  {'corpus':<10}{'protocol':<17}{'rep':<7}{'backbone':<16}{'acc':>8}{'chance':>8}"
            f"{'n_test':>9}")
        for r in sorted(diag, key=lambda x: x["run_id"]):
            add(f"  {r['corpus']:<10}{r['protocol']:<17}{r['rep']:<7}{r['backbone']:<16}"
                f"{fmt(float(r['test_acc'])):>8}{fmt(float(r['chance_acc'])):>8}{r['n_test']:>9}")
        add("")

    # ------------------------------------------------------------- T1 leakage
    for corpus in CORPORA_OK:
        p = args.data / "splits" / f"{corpus}_leakage_summary.csv"
        if p.exists():
            add("=" * 100)
            add("T1  Leakage audit (frozen protocol) — the table the paper opens with")
            add("=" * 100)
            add("")
            with open_text(p) as fh:
                rdr = list(csv.DictReader(fh))
            add(f"  {'protocol':<18}{'seed':>5}{'n_test':>9}{'n_ves':>7}{'n_rec':>7}"
                f"{'share vessel':>14}{'share rec':>11}{'share clip':>12}")
            for r in rdr:
                add(f"  {r['protocol']:<18}{r['seed']:>5}{r['n_test']:>9}"
                    f"{r['n_test_vessels']:>7}{r['n_test_recordings']:>7}"
                    f"{r['pct_test_share_vessel'] + '%':>14}"
                    f"{r['pct_test_share_recording'] + '%':>11}"
                    f"{r['pct_test_share_clip'] + '%':>12}")
            add("")

    text = "\n".join(L)
    print()
    print(text)
    (args.out / "tables" / "w3_tables.txt").write_text(text, encoding="utf-8")

    # ------------------------------------------------------------------ figures
    if not args.no_figures:
        try:
            make_figures(rows, groups, ci_by_run, scope_rows, args.out / "figures")
        except Exception as exc:  # a missing font must not lose the tables
            print(f"[warn] figures failed: {type(exc).__name__}: {exc}")

    print(f"\nwritten: {args.out}")
    return 0


CORPORA_OK = ("oceanship", "deepship", "shipsear")


def make_figures(rows, groups, ci_by_run, scope_rows, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def agg(rs, metric="test_acc"):
        v = np.array([float(x[metric]) for x in rs if x.get(metric) not in (None, "", "None")])
        return (v.mean(), v.std(ddof=0)) if len(v) else (float("nan"), 0.0)

    # F1: the protocol gap per backbone
    sel = [k for k in groups if k[4] == "erm" and k[6] in ("0", 0) and k[2] == "logmel"]
    if sel:
        protos = [p for p in PROTOCOL_ORDER]
        fig, ax = plt.subplots(figsize=(9, 5))
        labels, vals, errs = [], [], []
        for k in sorted(sel):
            corpus, proto = k[0], k[1]
            if proto not in protos:
                continue
            a, sd = agg(groups[k])
            labels.append(f"{corpus}\n{k[3]}\n{proto}")
            vals.append(a)
            errs.append(sd)
        if labels:
            ax.bar(range(len(vals)), vals, yerr=errs, capsize=3)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=7, rotation=90)
            ax.set_ylabel("test accuracy")
            ax.set_ylim(0, 1)
            ax.set_title("Same model, same recipe — only the split unit changes")
            fig.tight_layout()
            fig.savefig(outdir / "F2_protocol_gap.png", dpi=160)
            plt.close(fig)

    # F2: where the accuracy comes from
    if scope_rows:
        ag = defaultdict(list)
        for x in scope_rows:
            ag[x["protocol"]].append(x)
        keys = ["all", "seen_vessel", "unseen_vessel"]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        protos = [p for p in PROTOCOL_ORDER if p in ag]
        w = 0.25
        for i, kk in enumerate(keys):
            vals = []
            for p in protos:
                v = [x[f"acc_{kk}"] for x in ag[p] if x.get(f"acc_{kk}") not in (None, "", "None")]
                vals.append(float(np.mean(v)) if v else 0.0)
            ax.bar(np.arange(len(protos)) + (i - 1) * w, vals, width=w, label=kk)
        ax.set_xticks(range(len(protos)))
        ax.set_xticklabels(protos)
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Accuracy on test clips whose ship WAS vs WAS NOT seen in training")
        fig.tight_layout()
        fig.savefig(outdir / "F3_leak_scope.png", dpi=160)
        plt.close(fig)

    # F3: scale curve
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted = False
    for corpus in sorted({r["corpus"] for r in rows}):
        for proto in PROTOCOL_ORDER:
            pts = {}
            for r in rows:
                if (r["corpus"] == corpus and r["protocol"] == proto and r["rep"] == "logmel"
                        and r["method"] == "erm" and r["shuffled"] in ("0", 0)):
                    pts.setdefault(int(r["n_params"]), []).append(float(r["test_acc"]))
            if len(pts) < 2:
                continue
            xs = sorted(pts)
            ys = [float(np.mean(pts[x])) for x in xs]
            ax.semilogx(xs, ys, marker="o", label=f"{corpus} / {proto}")
            plotted = True
    if plotted:
        ax.set_xlabel("trainable parameters")
        ax.set_ylabel("test accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.set_title("Does a bigger model help once the split is honest?")
        fig.tight_layout()
        fig.savefig(outdir / "F4_scale.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
