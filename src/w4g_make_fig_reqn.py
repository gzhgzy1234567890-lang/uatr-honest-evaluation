"""W4-G — regenerate fig_reqn.pdf with the released 25-draw random-pool SD (4.97 pp;
the pre-registered 5-draw SD was 4.69).  Isolated from _make_figs2.py so that the
other figures keep their own provenance.

Usage:  python src/w4g_make_fig_reqn.py [--paper-dir PAPER]
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.linewidth": 0.8})
C = {"clip": "#c44e52", "honest": "#55a868", "grey": "#7f7f7f"}
CURVES = (("random draws (SD 4.97 pp)", 4.97, C["clip"], "-"),
          ("SGKF (SD 2.60 pp)", 2.60, C["grey"], "--"),
          ("clip-mass (SD 1.24 pp)", 1.24, C["honest"], "-"))

from scipy import stats as _sps

def _t_ppf(q, df):
    return float(_sps.t.ppf(q, df))

def _nct_sf(x, df, ncp):
    return float(_sps.nct.sf(x, df, ncp))

def _nct_cdf(x, df, ncp):
    return float(_sps.nct.cdf(x, df, ncp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper-dir", default=os.path.join(os.path.dirname(__file__), "..", "paper"))
    args, _ = ap.parse_known_args()
    figdir = os.path.join(args.paper_dir, "figures")
    os.makedirs(figdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    deltas = np.linspace(1.0, 5.0, 100)

    def reqn_ind_nct(sd, delta, alpha=0.05, power=0.8):
        """Two-sample t test per group size n: smallest n reaching target power."""
        for n in range(3, 500):
            df = 2 * n - 2
            ncp = delta / (sd * np.sqrt(2.0 / n))
            tc = _t_ppf(1 - alpha / 2, df)
            pw = _nct_sf(tc, df, ncp) + _nct_cdf(-tc, df, ncp)
            if pw >= power:
                return n
        return np.nan

    curve = {name: [reqn_ind_nct(sd, d) for d in deltas]
             for name, sd, _, _ in CURVES}
    for name, sd, col, ls in CURVES:
        ax.plot(deltas, curve[name], color=col, ls=ls, lw=1.6, label=name)
    ax.axhline(30, color="#bbb", lw=0.8, ls=":")
    ax.text(4.9, 34, "one week of GPU budget", fontsize=7, color="#888", ha="right")
    ax.axvline(2.0, color="#bbb", lw=0.8, ls=":")
    ax.annotate("98", (2.0, 98), xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                color=C["clip"])
    ax.annotate("28", (2.0, 28.5), xytext=(4, 0), textcoords="offset points", fontsize=7.5,
                color="#777")
    ax.annotate("8", (2.0, 8), xytext=(4, -2), textcoords="offset points", fontsize=7.5,
                color=C["honest"], fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlim(5, 1)
    ax.set_xlabel("improvement to be resolved $\\delta$ (pp)")
    ax.set_ylabel("independent draws per system ($\\alpha$=.05, power .8, non-central $t$)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    out = os.path.join(figdir, "fig_reqn.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
