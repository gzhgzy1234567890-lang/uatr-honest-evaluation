"""W4-F — replace fig_decomp with the two-panel per-draw scatter (bigger info gain):

  Panel A  test accuracy vs Towing share (the dominant composition driver) over the
           25 vessel-disjoint draws, with the composition OLS fit — shows the
           metadata->accuracy mechanism directly.
  Panel B  LODO-predicted vs observed accuracy (identity line) — shows the same
           mechanism holds OUT of sample (r = 0.80, RMSE 3.1 pp).

Inputs (download both from the W3-ALL session's Output tab into paper/):
  w3e2_variance.json  -> per-draw class compositions + accuracies
  w3e3_lodo.json      -> per-draw LODO predictions
Both are small (<100 KB).  The class->name mapping for OceanShip's 12-class
vocabulary is fixed and verified here.

Usage:  python src/w4f_make_fig_scatter.py [--paper-dir PAPER]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42, "axes.linewidth": 0.8})

CELL = "oceanship/logmel/cnn_small/vessel_disjoint/rec2"
DRIVER_CLASS = "Towing"          # index 10 in the 12-class vocabulary
# OceanShip's 12-class vocabulary, sorted (verified against the runner's own
# label print: 'Anti-pollution equipment', 'Cargo', 'Dredging', 'Fishing',
# 'Passenger', 'Pleasure Craft', 'Port Tender', 'Sailing', 'Search and Rescue
# vessel', 'Tanker', 'Towing', 'Tug')
VOCAB = ["Anti-pollution equipment", "Cargo", "Dredging", "Fishing", "Passenger",
         "Pleasure Craft", "Port Tender", "Sailing", "Search and Rescue vessel",
         "Tanker", "Towing", "Tug"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=os.path.join(os.path.dirname(__file__), "..", "paper"))
    args, _ = ap.parse_known_args()
    figdir = os.path.join(args.paper_dir, "figures")

    var = json.loads((pathlib.Path(args.paper_dir) / "w3e2_variance.json").read_text(encoding="utf-8"))
    cell = var["summary"][CELL]
    classes = cell["classes"]
    shares = np.array(cell["comp_shares"])          # (n, K)
    accs = np.array(cell["accs"])
    k = VOCAB.index(DRIVER_CLASS)
    K = len(classes)
    assert classes == list(range(K)), f"unexpected class indices: {classes}"
    tow = shares[:, k]
    n = len(accs)

    # LODO predictions computed here from the released compositions (same math as
    # w3e3_lodo.py); cross-checked against the Kaggle LODO run (r 0.803 / RMSE 3.10)
    Xd = np.column_stack([np.ones(n), shares[:, 1:]])   # one share dropped (sum=1)
    y = accs
    pred = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta, _, _, _ = np.linalg.lstsq(Xd[mask], y[mask], rcond=None)
        pred[i] = Xd[i] @ beta
    err = y - pred
    rmse = float(np.sqrt((err ** 2).mean()))
    r_lodo = float(np.corrcoef(pred, y)[0, 1])
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_lodo = 1.0 - float((err ** 2).sum()) / ss_tot
    print(f"[check] LODO r={r_lodo:.3f}  RMSE={100 * rmse:.2f} pp  LODO R^2={r2_lodo:.3f}  "
          f"(Kaggle reference: 0.803 / 3.10 pp / 0.594)")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

    # Panel A: accuracy vs the dominant composition driver
    ax = axes[0]
    ax.scatter(tow, accs, s=16, color="#4c72b0", edgecolor="white", linewidth=0.4, zorder=3)
    b = np.polyfit(tow, accs, 1)
    xs = np.linspace(tow.min(), tow.max(), 50)
    ax.plot(xs, np.polyval(b, xs), color="#c44e52", lw=1.4, zorder=2)
    r = float(np.corrcoef(tow, accs)[0, 1])
    ax.set_xlabel(f"{DRIVER_CLASS} share of the test set", fontsize=8)
    ax.set_ylabel("test accuracy", fontsize=8)
    ax.set_title(f"(a) accuracy tracks composition ($r={r:+.2f}$, $n={n}$)", fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.6)

    # Panel B: LODO predicted vs observed
    ax = axes[1]
    lim = (min(y.min(), pred.min()) - 0.02, max(y.max(), pred.max()) + 0.02)
    ax.plot(lim, lim, color="#999", lw=0.9, ls="--", zorder=1)
    ax.scatter(pred, y, s=16, color="#55a868", edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("LODO-predicted accuracy", fontsize=8)
    ax.set_ylabel("observed accuracy", fontsize=8)
    ax.set_title(f"(b) out-of-sample ($r={r_lodo:.2f}$, "
                 f"RMSE ${100 * rmse:.1f}$ pp, LODO $R^2={r2_lodo:.2f}$)", fontsize=8.5)
    ax.grid(alpha=0.25, lw=0.6)

    fig.suptitle("Per-draw composition predicts per-draw accuracy (25 vessel-disjoint draws, "
                 f"SD {st.stdev(accs) * 100:.1f} pp)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(figdir, "fig_decomp.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}  (n={n}, driver r={r:+.2f}, LODO r={r_lodo:.2f})")


if __name__ == "__main__":
    main()
