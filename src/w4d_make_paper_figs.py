"""W4-D — allocator-effect figure (fig_alloc.pdf) from released per-draw statistics.

fig_alloc.pdf   allocator effect across model families (Table VI visualized):
                draw SD under random vs clip-mass allocation, with the
                per-backbone variance reductions annotated.

NOTE: this script previously also wrote fig_decomp.pdf (a stacked-SD bar chart).
That figure was WRONG (it added standard deviations linearly) and has been
replaced by the two-panel scatter from w4f_make_fig_scatter.py.  Do NOT
re-add a fig_decomp writer here — w4f is the single authoritative generator
for fig_decomp.pdf, and running this script must never clobber it.

Values are transcribed verbatim from the released per-draw result files
(W3-K remedy runs) and match Table VI to all reported digits.

Usage:  python src/w4d_make_paper_figs.py [--paper-dir PAPER]
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42, "axes.linewidth": 0.8})

C_COMP = "#4c72b0"     # composition component
C_RESID = "#c44e52"    # residual (init/optimization) component
C_RAND = "#8172b2"     # random allocator
C_CLIP = "#55a868"     # clip-mass allocator


def fig_decomp(figdir):
    protocols = ["clip\nrandom", "recording\nrandom", "vessel\ndisjoint"]
    comp = np.array([0.00, 0.92, 4.69])          # composition SD (pp)
    resid = np.array([0.41, 0.72, 1.65])         # residual SD (pp)
    share = [0, 54, 89]
    n = ["n=5", "n=5", "n=25"]

    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    y = np.arange(len(protocols))[::-1]
    ax.barh(y, comp, height=0.52, color=C_COMP, label="composition (metadata)")
    ax.barh(y, resid, height=0.52, left=comp, color=C_RESID, label="residual (init/optimization)")
    for yi, c, r, s, nn in zip(y, comp, resid, share, n):
        ax.text(c + r + 0.10, yi, f"SD {c + r:.2f} pp\n{s}% composition", va="center", fontsize=7.5)
        ax.text(0.06, yi, f"{nn}", va="center", ha="left", fontsize=6.8,
                color="white", fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(protocols, fontsize=8)
    ax.set_xlabel("draw-to-draw SD of test accuracy (pp)", fontsize=8)
    ax.set_xlim(0, 6.6)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.set_title("Where the draw variance comes from", fontsize=9)
    fig.tight_layout()
    out = os.path.join(figdir, "fig_decomp.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def fig_alloc(figdir):
    backbones = ["logreg", "cnn_small", "cnn_wide"]
    rand = {"logreg": (2.76, 5), "cnn_small": (4.69, 5), "cnn_wide": (3.07, 5)}
    clip = {"logreg": (0.94, 6), "cnn_small": (1.24, 6), "cnn_wide": (1.36, 6)}
    ratio = {"logreg": 8.7, "cnn_small": 14.4, "cnn_wide": 5.1}

    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    x = np.arange(len(backbones))
    w = 0.34
    for off, vals, color, label in ((-w / 2, rand, C_RAND, "random draws"),
                                    (w / 2, clip, C_CLIP, "clip-mass draws")):
        sd = [vals[b][0] for b in backbones]
        ax.bar(x + off, sd, width=w, color=color, alpha=0.85, label=label)
        for xi, v in zip(x + off, sd):
            ax.text(xi, v + 0.08, f"{v:.2f}", ha="center", fontsize=7.5)
    for xi, b in zip(x, backbones):
        top = max(rand[b][0], clip[b][0])
        ax.annotate(f"{ratio[b]:.1f}$\\times$", (xi, top + 0.45),
                    ha="center", fontsize=9, fontweight="bold", color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(["logreg", "cnn-small", "cnn-wide"], fontsize=9)
    ax.set_ylabel("vessel-disjoint draw SD (pp)", fontsize=9)
    ax.set_ylim(0, 8.6)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.set_title("The allocator decides the noise floor\n(OceanShip, three model families)",
                 fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(figdir, "fig_alloc.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=os.path.join(os.path.dirname(__file__), "..", "paper"))
    args, _ = ap.parse_known_args()
    figdir = os.path.join(args.paper_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig_alloc(figdir)


if __name__ == "__main__":
    main()
