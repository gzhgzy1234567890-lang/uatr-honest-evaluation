# =========================================================================== #
# W3-F mechanism probe: WHY does the model collapse to three classes under
# the honest (vessel-disjoint) protocol?
#
# Competitive hypotheses, decided by numbers that already exist on disk:
#   H1  ACOUSTIC CONFUSION  - tail-class information is absent from the
#       features: top-2 hit rate ~ chance, confusion spread thin.
#       -> no loss reweighting can fix it; the paper becomes a limits paper.
#   H2  PRIOR SUPPRESSION   - tail-class information IS present (decent top-2)
#       but argmax never lands there because head classes dominate.
#       -> a rebalancing fix is justified; offline logit adjustment on the
#          saved probabilities predicts its gain BEFORE any GPU is spent.
#   H3  IDENTITY FLOOR      - classes with <5 training vessels cannot be
#       learned at all (already confirmed: Port Tender 3 vessels -> 0.007,
#       Dredging 2 vessels -> 0.000).
#
# Input : the Kaggle dataset you made from the W3-A output (w3-runs folder),
#         i.e. predictions/*.npz with y_true / y_pred / classes / proba.
# Run   : CPU only, a couple of minutes.  Optionally also attach the
#         uwatr-features dataset so the exact training prior can be read from
#         oceanship_inventory.csv; otherwise a frozen copy of the corpus class
#         shares is used (the difference is negligible for this probe).
# =========================================================================== #
import glob
import json
import os
import sys
import time

import numpy as np

T0 = time.time()


def tick(msg):
    print(f"  [{time.time() - T0:6.1f}s] {msg}", flush=True)


# corpus-level clip shares of the 12 split-covered classes (from the frozen
# inventory; used only if oceanship_inventory.csv is not among the mounts)
FALLBACK_PRIOR = {
    "Tug": 11852, "Towing": 11557, "Cargo": 9662, "Pleasure Craft": 6734,
    "Passenger": 5527, "Fishing": 4966, "Sailing": 1456, "Port Tender": 1253,
    "Search and Rescue vessel": 520, "Tanker": 152, "Dredging": 53,
    "Anti-pollution equipment": 22,
}
SHORT = {
    "Anti-pollution equipment": "AntiP", "Cargo": "Cargo", "Dredging": "Dredg",
    "Fishing": "Fish", "Passenger": "Passng", "Pleasure Craft": "PleasC",
    "Port Tender": "PortT", "Sailing": "Sail",
    "Search and Rescue vessel": "SAR", "Tanker": "Tanker", "Towing": "Towing",
    "Tug": "Tug",
}
TAU_SWEEP = (0.25, 0.5, 1.0)
TOPK = (2, 3)

# ----------------------------------------------------------------- locate -- #
root = None
args = sys.argv[1:]
if "--root" in args:
    root = args[args.index("--root") + 1]
if root is None:
    for cand in sorted(glob.glob("/kaggle/input/*")):
        hits = glob.glob(os.path.join(cand, "**", "predictions"), recursive=True)
        real = [h for h in hits
                if glob.glob(os.path.join(h, "oceanship__logmel__cnn_small__*.npz"))]
        if real:
            root = real[0]
            break
if root is None:
    raise SystemExit("no predictions/ folder with oceanship npz found under "
                     "/kaggle/input - is the W3-A output dataset attached?")
print("=" * 88)
print(f"W3-F mechanism probe   predictions root: {root}")
print("=" * 88)


def load_group(protocol):
    out = []
    for s in range(5):
        hits = glob.glob(os.path.join(
            root, f"oceanship__logmel__cnn_small__{protocol}__s{s}*.npz"))
        if not hits:
            continue
        d = np.load(hits[0], allow_pickle=False)
        out.append(dict(seed=s, classes=[str(c) for c in d["classes"]],
                        y_true=d["y_true"].astype(np.int64),
                        y_pred=d["y_pred"].astype(np.int64),
                        proba=d["proba"].astype(np.float64)))
    return out


GROUPS = {}
for proto in ("vessel_disjoint", "clip_random"):
    g = load_group(proto)
    GROUPS[proto] = g
    tick(f"{proto:<16} {len(g)} runs loaded"
         + ("" if g else "   (none found - skipped)"))
if not GROUPS["vessel_disjoint"]:
    raise SystemExit("no vessel_disjoint runs found - this probe needs the "
                     "rec2 main-table predictions")

# ------------------------------------------------------------------ prior -- #
classes = GROUPS["vessel_disjoint"][0]["classes"]
prior = None
for cand in glob.glob("/kaggle/input/*/splits/oceanship_inventory.csv") + \
        glob.glob("/kaggle/input/*/*/splits/oceanship_inventory.csv"):
    try:
        import csv
        with open(cand, encoding="utf-8", newline="") as fh:
            from collections import Counter
            cnt = Counter(r["label"] for r in csv.DictReader(fh))
        if all(c in cnt for c in classes):
            prior = np.array([cnt[c] for c in classes], dtype=np.float64)
            tick(f"training prior read from {os.path.basename(cand)}")
            break
    except Exception:
        continue
if prior is None:
    prior = np.array([FALLBACK_PRIOR[c] for c in classes], dtype=np.float64)
    tick("training prior from the frozen corpus shares (inventory not mounted)")
prior = prior / prior.sum()
print(f"  classes ({len(classes)}): {classes}")
print(f"  prior    : "
      + "  ".join(f"{SHORT[c]} {100*p:.1f}%" for c, p in zip(classes, prior)))


def metrics(y_true, y_pred, k):
    acc = float((y_pred == y_true).mean())
    f1s = []
    for i in range(k):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return acc, float(np.mean(f1s))


def topk_hit(proba, y_true, kk):
    order = np.argsort(-proba, axis=1)[:, :max(kk)]
    return {k: float((order[:, :k] == y_true[:, None]).any(1).mean()) for k in kk}


# =============================================== 1. pooled confusion matrix = #
VD = GROUPS["vessel_disjoint"]
y_true = np.concatenate([g["y_true"] for g in VD])
y_pred = np.concatenate([g["y_pred"] for g in VD])
proba = np.concatenate([g["proba"] for g in VD])
k = len(classes)
print()
print("=" * 88)
print(f"1. pooled confusion matrix, vessel_disjoint, {len(VD)} seeds "
      f"({len(y_true):,} test clips) - rows = truth, cols = prediction, %")
print("=" * 88)
cm = np.zeros((k, k), dtype=np.int64)
np.add.at(cm, (y_true, y_pred), 1)
cmn = 100 * cm / np.maximum(cm.sum(1, keepdims=True), 1)
hdr = "  truth\\pred      " + "".join(f"{SHORT[c]:>8}" for c in classes)
print(hdr)
for i, c in enumerate(classes):
    row = "".join(f"{cmn[i, j]:>8.1f}" for j in range(k))
    print(f"  {SHORT[c]:<15} {row}   n={cm[i].sum():>6,}")
tick("confusion matrix done")

# ====================================== 2. per-class: recall / top-k / mass = #
print()
print("=" * 88)
print("2. per-class diagnostics, vessel_disjoint (pooled)")
print("   top2/top3 = truth among the 2/3 highest-probability classes")
print("=" * 88)
tk = topk_hit(proba, y_true, TOPK)
mass_true = proba[np.arange(len(y_true)), y_true].mean()
print(f"  {'class':<28}{'recall':>8}{'top2':>8}{'top3':>8}{'mean p(true)':>14}"
      f"{'n_test':>9}")
print("  " + "-" * 75)
percls = {}
for i, c in enumerate(classes):
    m = y_true == i
    if not m.any():
        continue
    rec = float((y_pred[m] == i).mean())
    t2 = float((np.argsort(-proba[m], axis=1)[:, :2] == i).any(1).mean())
    t3 = float((np.argsort(-proba[m], axis=1)[:, :3] == i).any(1).mean())
    mt = float(proba[m][:, i].mean())
    percls[c] = dict(recall=rec, top2=t2, top3=t3, mass=mt, n=int(m.sum()))
    print(f"  {c:<28}{rec:>8.3f}{t2:>8.3f}{t3:>8.3f}{mt:>14.3f}{int(m.sum()):>9,}")

# confusion concentration: for each tail class, how much of its WRONG mass
# lands on its single most common wrong prediction?
print()
print("  where the tail classes actually go (top-2 wrong targets, % of the class):")
tail = [c for c, v in percls.items() if v["recall"] < 0.10]
tail_info = {}
for c in tail:
    i = classes.index(c)
    m = y_true == i
    wrong = y_pred[m & (y_pred != i)]
    if len(wrong) == 0:
        continue
    vals, cnts = np.unique(wrong, return_counts=True)
    order = np.argsort(-cnts)[:2]
    tot = cnts.sum()
    conc = cnts[order[0]] / tot
    targets = ", ".join(f"{SHORT[classes[j]]} {100*cnts[j]/tot:.0f}%"
                        for j in order)
    tail_info[c] = dict(top_targets=targets, concentration=float(conc))
    print(f"    {c:<28} -> {targets}   (top-1 target takes {100*conc:.0f}% of the wrong mass)")

# ================================= 3. H1 vs H2 verdict on the tail classes = #
print()
print("=" * 88)
print("3. VERDICT on the collapse mechanism (tail = recall < 0.10, >=5 train vessels)")
print("=" * 88)
usable_tail = [c for c in tail if percls[c]["n"] >= 30 and c not in
               ("Port Tender", "Dredging", "Anti-pollution equipment")]
t2s = [percls[c]["top2"] for c in usable_tail]
r1s = [percls[c]["recall"] for c in usable_tail]
print(f"  tail classes tested ({len(usable_tail)}): "
      + ", ".join(f"{SHORT[c]}({percls[c]['top2']:.2f})" for c in usable_tail))
if t2s:
    mt2 = float(np.mean(t2s))
    print(f"  mean top-2 hit rate over tail classes: {mt2:.3f}   "
          f"(chance for a 12-class problem ~ 0.08-0.17 depending on prior)")
    if mt2 >= 0.30:
        verdict = "H2 PRIOR SUPPRESSION"
        note = ("the information IS in the features; argmax is biased toward "
                "the head. A rebalancing fix is justified.")
    elif mt2 <= 0.12:
        verdict = "H1 ACOUSTIC CONFUSION"
        note = ("the information is NOT in the features at this operating "
                "point; no loss reweighting will recover it. The fixable part "
                "is limited; the paper should report limits + the confusable "
                "group structure.")
    else:
        verdict = "MIXED"
        note = ("part of the tail is suppressed, part is genuinely confusable. "
                "Report both: the fixable part (offline-adjustment gain below) "
                "and the irreducible floor.")
    print(f"  -> {verdict}")
    print(f"     {note}")

# ========================= 4. offline logit adjustment on saved probabilities #
print()
print("=" * 88)
print("4. OFFLINE logit adjustment on the saved probabilities (no training)")
print("   p_adj(k) ~ p(k) / prior_k^tau;  tau=0 is the unmodified model")
print("=" * 88)
acc0, f10 = metrics(y_true, y_pred, k)
print(f"  {'tau':>5}{'accuracy':>10}{'macro-F1':>10}{'tail recall sum':>17}")
print("  " + "-" * 45)
adj_out = []
for tau in (0.0,) + TAU_SWEEP:
    pa = proba / np.power(prior[None, :], tau)
    yp = pa.argmax(1)
    acc, f1 = metrics(y_true, yp, k)
    tail_rec = sum(float((yp[y_true == classes.index(c)] == classes.index(c)).mean())
                   for c in usable_tail if percls[c]["n"] > 0)
    adj_out.append(dict(tau=tau, acc=acc, macro_f1=f1, tail_recall_sum=tail_rec))
    print(f"  {tau:>5.2f}{acc:>10.4f}{f1:>10.4f}{tail_rec:>17.3f}")
best = max(adj_out, key=lambda r: r["macro_f1"])
print(f"  best macro-F1 at tau={best['tau']:.2f}: "
      f"{best['macro_f1']:.4f} (baseline {f10:.4f}, gain "
      f"{100*(best['macro_f1']-f10):+.1f} pp)  accuracy {best['acc']:.4f} "
      f"({100*(best['acc']-acc0):+.1f} pp)")
print("  -> this is the CEILING of any prior-rebalancing fix, obtained for")
print("     free; a training-time loss can only approach it, not exceed it")

# ============================== 5. protocol contrast: is the info protocol- #
# dependent? tail classes under the LEAKY clip_random split                   #
print()
print("=" * 88)
print("5. same tail classes under the leaky clip_random split (protocol contrast)")
print("=" * 88)
CR = GROUPS["clip_random"]
if CR:
    yt = np.concatenate([g["y_true"] for g in CR])
    yp_ = np.concatenate([g["y_pred"] for g in CR])
    pp = np.concatenate([g["proba"] for g in CR])
    print(f"  {'class':<28}{'recall honest':>14}{'recall leaky':>14}"
          f"{'top2 honest':>13}{'top2 leaky':>12}")
    print("  " + "-" * 82)
    tkc = topk_hit(pp, yt, TOPK)
    for c in tail:
        i = classes.index(c)
        m = yt == i
        if not m.any():
            continue
        rec_l = float((yp_[m] == i).mean())
        t2_l = float((np.argsort(-pp[m], axis=1)[:, :2] == i).any(1).mean())
        print(f"  {c:<28}{percls[c]['recall']:>14.3f}{rec_l:>14.3f}"
              f"{percls[c]['top2']:>13.3f}{t2_l:>12.3f}")
    print("  if leaky >> honest on BOTH recall and top-2, the protocol itself")
    print("  destroys the tail information (identity shortcut replaced it);")
    print("  if top-2 is low in BOTH, the classes are acoustically confusable")
    print("  regardless of protocol - that is a property of the corpus.")
else:
    print("  (clip_random runs not found - skipped)")

# ------------------------------------------------------------------- save -- #
out = dict(
    kind="w3f-mechanism-probe",
    predictions_root=root,
    classes=classes,
    prior=[float(p) for p in prior],
    per_class=percls,
    tail_targets=tail_info,
    tail_verdict=dict(classes=usable_tail,
                      mean_top2=float(np.mean(t2s)) if t2s else None,
                      verdict=verdict if t2s else None),
    logit_adjustment=adj_out,
)
for cand in ("/kaggle/working", os.path.dirname(root), os.getcwd()):
    if cand and os.path.isdir(cand):
        out_path = os.path.join(cand, "w3f_mechanism.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=float)
        print()
        tick(f"summary written to {out_path}")
        break
