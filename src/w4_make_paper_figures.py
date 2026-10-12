"""
w4_make_paper_figures.py - paper figures that need no GPU and no audio.

Fig 1 (fig_schematic.pdf): what each slicing policy lets cross the train/test
        boundary (concept schematic; numbers from out/w1_05_recording_structure.txt)
Fig 2 (fig_probe.pdf):    DeepShip one-scalar probe ladder (78.18 / 83.22 / 92.19
        under clip_random; collapse under recording/temporal splits), also from
        out/w1_05_recording_structure.txt section 3.

Usage:  python src/w4_make_paper_figures.py [--paper-dir PAPER]
Output: PAPER/figures/fig_schematic.pdf, PAPER/figures/fig_probe.pdf
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.linewidth": 0.8,
})

# vessel colours (colourblind-safe-ish, print-safe)
C_A = "#4C72B0"   # vessel A (blue)
C_B = "#DD8452"   # vessel B (orange)
C_C = "#55A868"   # vessel C (green)
C_D = "#8172B2"   # vessel D (purple)
C_EDGE = "#2b2b2b"
C_BOUND = "#C44E52"   # split boundary (red)


def _squares(ax, y, specs, sq=0.055, gap=0.012):
    """Draw a row of clip squares. specs: list of (color, alpha, label?) tuples."""
    x = 0.06
    for color, alpha, hi in specs:
        r = Rectangle((x, y), sq, sq, facecolor=color, alpha=alpha,
                      edgecolor=C_EDGE, linewidth=0.5, zorder=3)
        ax.add_patch(r)
        if hi:
            ax.add_patch(Rectangle((x - 0.008, y - 0.008), sq + 0.016, sq + 0.016,
                                   facecolor="none", edgecolor=C_BOUND,
                                   linewidth=1.4, zorder=4))
        x += sq + gap
    return x


def schematic(path):
    fig, ax = plt.subplots(figsize=(3.5, 2.55), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bx = 0.56                      # boundary x
    rows = {"clip": 0.74, "rec": 0.44, "vessel": 0.14}

    # boundary line through all rows
    ax.plot([bx, bx], [0.06, 0.97], color=C_BOUND, linestyle=(0, (4, 3)),
            linewidth=1.1, zorder=2)
    ax.text(bx + 0.008, 0.965, "train | test boundary", color=C_BOUND,
            fontsize=6.6, va="top")

    # ---- row 1: clip random -------------------------------------------------
    y = rows["clip"]
    specs = [(C_A, 1.0, 0)] * 4 + [(C_B, 1.0, 0)] + [(C_A, 1.0, 1), (C_A, 1.0, 1)] \
        + [(C_B, 1.0, 0)] * 3 + [(C_A, 1.0, 0), (C_B, 1.0, 0)]
    _squares(ax, y, specs)
    ax.text(0.055, y + 0.13, "clip random", fontsize=7.5, fontweight="bold")
    ax.text(0.055, y - 0.075,
            "adjacent 5 s windows on both sides; vessel 100% in train",
            fontsize=6.0, color="#444444")

    # ---- row 2: recording random -------------------------------------------
    y = rows["rec"]
    train = [(C_A, 1.0, 0)] * 3 + [(C_B, 1.0, 0)] * 2            # rec 1 of A, rec of B
    test = [(C_A, 0.45, 0)] * 3 + [(C_B, 0.45, 0)] * 2           # other recordings, same vessels
    x = 0.06
    for c, a, h in train:
        ax.add_patch(Rectangle((x, y), 0.055, 0.055, facecolor=c, alpha=a,
                               edgecolor=C_EDGE, linewidth=0.5, zorder=3))
        x += 0.055 + 0.012
    x = bx + 0.025
    for c, a, h in test:
        ax.add_patch(Rectangle((x, y), 0.055, 0.055, facecolor=c, alpha=a,
                               edgecolor=C_EDGE, linewidth=0.5, zorder=3))
        x += 0.055 + 0.012
    ax.text(0.055, y + 0.13, "recording random", fontsize=7.5, fontweight="bold")
    ax.text(0.055, y - 0.075,
            "recordings intact; the same vessel still crosses (98%)",
            fontsize=6.0, color="#444444")

    # ---- row 3: vessel / temporal -------------------------------------------
    y = rows["vessel"]
    specs = ([(C_A, 1.0, 0)] * 3 + [(C_B, 1.0, 0)] * 3
             + [(C_C, 1.0, 0)] * 3 + [(C_D, 1.0, 0)] * 2)
    # train zone gets A,B; test zone gets C,D
    train = [(C_A, 1.0, 0)] * 3 + [(C_B, 1.0, 0)] * 3
    test = [(C_C, 1.0, 0)] * 3 + [(C_D, 1.0, 0)] * 2
    x = 0.06
    for c, a, h in train:
        ax.add_patch(Rectangle((x, y), 0.055, 0.055, facecolor=c, alpha=a,
                               edgecolor=C_EDGE, linewidth=0.5, zorder=3))
        x += 0.055 + 0.012
    x = bx + 0.02
    for c, a, h in test:
        ax.add_patch(Rectangle((x, y), 0.055, 0.055, facecolor=c, alpha=a,
                               edgecolor=C_EDGE, linewidth=0.5, zorder=3))
        x += 0.055 + 0.012
    ax.text(0.055, y + 0.13, "vessel-disjoint / temporal", fontsize=7.5, fontweight="bold")
    ax.text(0.055, y - 0.075,
            "vessel-disjoint: no known vessel ID crosses; temporal: recordings partitioned\n"
            "by time --- identity isolation must be verified separately (see Table I)",
            fontsize=6.0, color="#444444", linespacing=1.25)

    fig.tight_layout(pad=0.15)
    fig.savefig(path)
    plt.close(fig)


def probe(path):
    # numbers verbatim from out/w1_05_recording_structure.txt section 3
    protocols = ["clip random", "recording random", "temporal"]
    scalar = [78.18, 34.61, 21.82]
    knn = [83.22, 50.33, 50.15]
    model = [92.19, 68.42, 61.82]

    fig, ax = plt.subplots(figsize=(4.2, 2.5), dpi=300)
    w = 0.26
    xs = range(len(protocols))
    b1 = ax.bar([x - w for x in xs], scalar, w, color="#DD8452",
                edgecolor=C_EDGE, linewidth=0.4, label="1 scalar (src. duration)")
    b2 = ax.bar(list(xs), knn, w, color="#55A868",
                edgecolor=C_EDGE, linewidth=0.4, label="256-d acoustic 1-NN")
    b3 = ax.bar([x + w for x in xs], model, w, color="#4C72B0",
                edgecolor=C_EDGE, linewidth=0.4, label="trained CNN")

    for bars in (b1, b2, b3):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.5,
                    f"{r.get_height():.1f}", ha="center", fontsize=6.5)

    ax.axhline(25, color="#888888", linestyle=(0, (3, 3)), linewidth=0.8,
               label="chance (4-class)")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(["Clip\nrandom", "Recording\nrandom", "Temporal"])
    ax.set_ylabel("DeepShip accuracy (%)")
    ax.set_ylim(0, 108)
    ax.set_xlim(-0.55, 2.55)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", handlelength=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=os.path.join(os.path.dirname(__file__), "..", "paper"))
    args, _ = ap.parse_known_args()
    figdir = os.path.join(args.paper_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    schematic(os.path.join(figdir, "fig_schematic.pdf"))
    probe(os.path.join(figdir, "fig_probe.pdf"))
    print("wrote", os.path.join(figdir, "fig_schematic.pdf"))
    print("wrote", os.path.join(figdir, "fig_probe.pdf"))


if __name__ == "__main__":
    main()
