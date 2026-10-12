"""W4-B — regenerate fig_ladder.pdf (the protocol ladder, dot-per-draw version).

Values are transcribed verbatim from the released per-draw result files:
  - OceanShip rec2 (5 draws/protocol):      Kaggle w3-data  (log 4)
  - DeepShip ds1   (3 draws/protocol):      Kaggle w3a4 output
  - DeepShip vessel rec2 (10 draws):        Kaggle w3m-data (log 12)
They match Table II of the paper to all reported digits.

Usage:  python src/w4b_make_fig_ladder.py [--paper-dir PAPER]
"""
import argparse
import os
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42, "axes.linewidth": 0.8})

# ---- per-draw test accuracies (verbatim from the released result files) ---- #
OCEAN = {
    "clip\nrandom": [0.3649, 0.3614, 0.3678, 0.3704, 0.3714],
    "recording\nrandom": [0.3420, 0.3641, 0.3364, 0.3633, 0.3484],
    "vessel\ndisjoint": [0.2926, 0.2770, 0.3519, 0.3792, 0.2780],
}
DEEP = {
    "clip\nrandom": [0.9179, 0.9288, 0.9191],
    "recording\nrandom": [0.6598, 0.7167, 0.6761],
    "temporal": [0.6228, 0.6244, 0.6075],
    "vessel\ndisjoint": [0.6511, 0.6279, 0.6330, 0.6125, 0.5463,
                         0.5484, 0.6873, 0.5986, 0.6164, 0.7196],
}
# OceanShip vessel-disjoint: seeds 0-4 (frozen) + seeds 5-24 (W3-N, 2026-10-10)
OCEAN["vessel\ndisjoint"] += [0.3078, 0.4160, 0.2929, 0.3100, 0.3462, 0.2786, 0.3738,
                              0.3445, 0.2461, 0.2459, 0.3084, 0.3831, 0.3888, 0.2422,
                              0.3543, 0.3672, 0.3235, 0.3035, 0.2523, 0.3306]
BAR_COLORS = ["#c44e52", "#8172b2", "#4c72b0", "#55a868"]


def panel(ax, title, data, ymax):
    means = [st.mean(v) for v in data.values()]
    sds = [100 * st.stdev(v) for v in data.values()]
    xs = range(len(data))
    ax.bar(xs, means, yerr=[s / 200 for s in sds], capsize=3, width=0.62,
           color=BAR_COLORS[: len(data)], alpha=0.35, error_kw={"lw": 0.9})
    for i, v in enumerate(data.values()):
        jitter = [((k * 0.6180339887) % 1 - 0.5) * 0.30 for k in range(len(v))]
        ax.scatter([i + j for j in jitter], v, s=14, zorder=3,
                   color=BAR_COLORS[i], edgecolor="white", linewidth=0.4)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(data.keys(), fontsize=7.5)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.3)
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(i, min(m + s / 200 + ymax * 0.045, ymax * 0.97), f"{m:.3f}", ha="center",
                fontsize=7.5, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=os.path.join(os.path.dirname(__file__), "..", "paper"))
    args, _ = ap.parse_known_args()
    figdir = os.path.join(args.paper_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
    # per-panel scales: a shared 0-1 axis visually flattens OceanShip's 14pp
    # ladder (all its bars sit below 0.4); DeepShip needs the full range (0.92)
    panel(axes[0], "OceanShip (12 classes)", OCEAN, ymax=0.50)
    panel(axes[1], "DeepShip (4 classes)", DEEP, ymax=1.02)
    axes[0].set_ylabel("test accuracy", fontsize=9)
    fig.suptitle("Same model, same recipe: the clip-vs-honest gap is 5.1 pp (paired) on "
                 "OceanShip and 30.4 pp (temporal) / 30.8 pp (vessel) on DeepShip",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(figdir, "fig_ladder.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
