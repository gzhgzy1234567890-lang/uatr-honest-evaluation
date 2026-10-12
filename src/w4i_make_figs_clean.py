"""W4-I — clean, self-contained regeneration of fig_pred / fig_confusion / fig_dose / fig_tail.

Written after a line-number-extraction bug produced a mis-banded fig_tail: this file is
self-contained (no log parsing, no fragile block extraction) so every figure is reproducible
from one auditable script.  All numbers are the released values behind the paper's tables.

Usage:  python src/w4i_make_figs_clean.py [--paper-dir PAPER]
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})
C_HONEST = "#55a868"
C_CLIP = "#c44e52"
C_B1 = "#3d5a80"
C_ACC = "#e07a5f"


def fig_pred(figdir):
    """Predicted vs observed draw SD, five schemes (legend carries scheme + random SD)."""
    # (predicted SD, observed SD, label, random-draw SD)  — released Table VI / Table IV values
    pts = [
        (1.12, 1.24, "clip-mass, cnn-small", 4.7),
        (0.70, 0.94, "clip-mass, logreg", 2.8),
        (0.92, 1.36, "clip-mass, cnn-wide", 3.1),
        (4.85, 4.85, "count-stratified", 4.7),
        (2.28, 2.60, "StratifiedGroupKFold", 4.7),
    ]
    mk = {"clip-mass, cnn-small": "o", "clip-mass, logreg": "s", "clip-mass, cnn-wide": "D",
          "count-stratified": "^", "StratifiedGroupKFold": "v"}
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    lo, hi = 0.0, 6.0
    ax.fill_between([lo, hi], [0.8 * lo, 0.8 * hi], [1.25 * lo + 0.12, 1.25 * hi + 0.12],
                    color=C_HONEST, alpha=0.08, lw=0)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=2)
    for p, o, lab, rnd in pts:
        col = C_CLIP if lab in ("count-stratified", "StratifiedGroupKFold") else C_HONEST
        ax.scatter(p, o, s=64, color=col, marker=mk[lab], zorder=4,
                   edgecolor="white", linewidth=0.7,
                   label=f"{lab}  (random SD {rnd:.1f} pp)")
    ax.text(3.35, 4.45, "$\\pm$25% band", fontsize=8.5, color=C_HONEST, rotation=27)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted draw SD — from metadata only, before training (pp)")
    ax.set_ylabel("observed draw SD (pp)")
    ax.legend(fontsize=7.4, frameon=False, loc="lower right", handlelength=1.0,
              borderaxespad=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_pred.pdf"))
    plt.close(fig)


def fig_confusion(figdir):
    """Row-normalized confusion matrix (pooled 5 vessel-disjoint draws), full 12x12 grid."""
    order = ["Anti-pollution equipment", "Cargo", "Dredging", "Fishing", "Passenger",
             "Pleasure Craft", "Port Tender", "Sailing", "Search and Rescue vessel",
             "Tanker", "Towing", "Tug"]
    short = ["AntiP", "Cargo", "Dredg", "Fish", "Passngr", "PleasC", "PortT",
             "Sail", "SAR", "Tanker", "Towing", "Tug"]
    rows = {
        "Anti-pollution equipment": {"Cargo": 18.2, "Towing": 72.7, "Tug": 9.1},
        "Cargo": {"Cargo": 58.4, "Pleasure Craft": 1.2, "Tanker": 0.1, "Towing": 5.0, "Tug": 35.3},
        "Dredging": {"Cargo": 9.2, "Fishing": 4.1, "Pleasure Craft": 1.0, "Towing": 71.4, "Tug": 14.3},
        "Fishing": {"Cargo": 6.0, "Fishing": 2.2, "Pleasure Craft": 0.6, "Port Tender": 0.3,
                    "Sailing": 0.1, "Towing": 72.4, "Tug": 18.4},
        "Passenger": {"Cargo": 8.3, "Fishing": 1.8, "Pleasure Craft": 0.4, "Port Tender": 0.3,
                      "Sailing": 0.1, "Towing": 74.5, "Tug": 14.7},
        "Pleasure Craft": {"Cargo": 38.6, "Fishing": 0.6, "Pleasure Craft": 1.9, "Tanker": 0.1,
                           "Towing": 26.1, "Tug": 32.7},
        "Port Tender": {"Cargo": 6.8, "Fishing": 2.4, "Pleasure Craft": 0.1, "Port Tender": 0.7,
                        "Towing": 69.8, "Tug": 20.2},
        "Sailing": {"Cargo": 8.4, "Fishing": 1.0, "Passenger": 0.1, "Pleasure Craft": 0.2,
                    "Towing": 80.0, "Tug": 10.2},
        "Search and Rescue vessel": {"Cargo": 11.4, "Fishing": 2.7, "Pleasure Craft": 0.4,
                                     "Towing": 73.7, "Tug": 11.8},
        "Tanker": {"Cargo": 23.7, "Towing": 15.8, "Tug": 60.4},
        "Towing": {"Cargo": 9.5, "Fishing": 2.0, "Pleasure Craft": 0.8, "Port Tender": 0.3,
                   "Towing": 72.2, "Tug": 15.0},
        "Tug": {"Cargo": 37.8, "Fishing": 0.5, "Pleasure Craft": 1.6, "Port Tender": 0.1,
                "Tanker": 0.1, "Towing": 23.1, "Tug": 36.7},
    }
    M = np.zeros((12, 12))
    for t, targets in rows.items():
        for t2, v in targets.items():
            M[order.index(t), order.index(t2)] = v
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=85)  # full grid, zeros = lightest cell
    ax.set_xticks(range(12))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(12))
    ax.set_yticklabels(short, fontsize=8)
    for i in range(12):
        for j in range(12):
            if M[i, j] >= 0.05:
                col = "white" if M[i, j] > 45 else "#1c3d5a"
                w = "bold" if i == j else "normal"
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        fontsize=7.2, color=col, fontweight=w)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
    ax.grid(which="minor", color="#9bb8d3", lw=0.5)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("predicted class (pooled vessel-disjoint, 5 draws)", fontsize=9)
    ax.set_ylabel("true class", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_confusion.pdf"), dpi=450)
    plt.close(fig)


def fig_dose(figdir):
    """Dose-response: recall vs training vessels / clips; only key classes labelled."""
    dose = [
        ("Towing", 0.722, 56, 8306), ("Cargo", 0.584, 155, 6837), ("Tug", 0.367, 80, 8382),
        ("Fishing", 0.022, 65, 3510), ("Pleasure Craft", 0.019, 143, 5287),
        ("Passenger", 0.000, 42, 3346), ("Sailing", 0.000, 56, 1616),
        ("Search and Rescue vessel", 0.000, 20, 549), ("Tanker", 0.000, 25, 1312),
        ("Port Tender", 0.007, 3, 743), ("Dredging", 0.000, 2, 26),
        ("Anti-pollution equipment", 0.000, 5, 139),
    ]
    KEY = {"Towing", "Cargo", "Tug", "Passenger", "Port Tender", "Dredging"}
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.5), sharey=True)
    for ax, idx, xlabel, rho in ((axes[0], 2, "training vessels per class", 0.68),
                                 (axes[1], 3, "training clips per class", 0.83)):
        ax.scatter([d[idx] for d in dose], [d[1] for d in dose], s=30,
                   color=C_HONEST, alpha=0.85, edgecolor="white", lw=0.6)
        for d in dose:
            if d[0] in KEY:
                ax.annotate(d[0], (d[idx], d[1]), xytext=(4, 3),
                            textcoords="offset points", fontsize=7.5, color="#333")
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=9)
        ax.text(0.98, 0.02, f"Spearman $\\rho$={rho:+.2f}", transform=ax.transAxes,
                ha="right", fontsize=8, color="#555")
        ax.set_ylim(-0.08, 0.85)
        ax.grid(alpha=0.25, lw=0.6)
    axes[0].set_ylabel("per-class recall (vessel disjoint)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_dose.pdf"))
    plt.close(fig)


def fig_tail(figdir):
    """Per-class recall with tier bands covering exactly rows a..b-1 of CLASSES."""
    CLASSES = ["Towing", "Tug", "Cargo", "Fishing", "Pleasure Craft", "Passenger",
               "Sailing", "Search and Rescue vessel", "Tanker", "Port Tender",
               "Dredging", "Anti-pollution equipment"]
    rec_cnn = {"Towing": 0.722, "Tug": 0.367, "Cargo": 0.584, "Fishing": 0.022,
               "Pleasure Craft": 0.019, "Passenger": 0.000, "Sailing": 0.000,
               "Search and Rescue vessel": 0.000, "Tanker": 0.000,
               "Port Tender": 0.007, "Dredging": 0.000, "Anti-pollution equipment": 0.000}
    rec_ft = {"Towing": 0.641, "Tug": 0.433, "Cargo": 0.322, "Fishing": 0.118,
              "Pleasure Craft": 0.099, "Passenger": 0.037, "Sailing": 0.007,
              "Search and Rescue vessel": 0.000, "Tanker": 0.000,
              "Port Tender": 0.044, "Dredging": 0.000, "Anti-pollution equipment": 0.000}
    # (label, first class index, last class index + 1, colour) — bands cover rows a..b-1
    tiers = [("learnable", 0, 3, "#0f717320"),
             ("partially rescuable", 3, 5, "#e9c46a30"),
             ("intervention-resistant", 5, 12, "#d1495b18")]
    fig, ax = plt.subplots(figsize=(6.6, 4.1))
    y = np.arange(len(CLASSES))[::-1]
    for name, a, b, col in tiers:
        ax.axhspan(y[b - 1] - 0.5, y[a] + 0.5, color=col, zorder=0)
        ax.text(1.13, (y[b - 1] + y[a]) / 2, name, fontsize=8.5, color="#555",
                style="italic", va="center", ha="left")
    ax.barh(y + 0.19, [rec_cnn[c] for c in CLASSES], height=0.36,
            label="cnn-small (from scratch)", color=C_B1)
    ax.barh(y - 0.19, [rec_ft[c] for c in CLASSES], height=0.36,
            label="fine-tuned AudioSet AST", color=C_ACC)
    ax.set_yticks(y)
    ax.set_yticklabels(CLASSES, fontsize=8.5)
    ax.set_xlim(0, 1.55)
    ax.set_ylim(-0.55, len(CLASSES) - 0.45)
    ax.set_xlabel("per-class recall, vessel-disjoint evaluation")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_tail.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=os.path.join(os.path.dirname(__file__), "..", "paper"))
    args, _ = ap.parse_known_args()
    figdir = os.path.join(args.paper_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig_pred(figdir)
    fig_confusion(figdir)
    fig_dose(figdir)
    fig_tail(figdir)
    print("wrote fig_pred / fig_confusion / fig_dose / fig_tail ->", figdir)


if __name__ == "__main__":
    main()
