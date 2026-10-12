"""W4-H — exact permutation test for the remedy variance reduction (Table VI).

Replaces the Monte-Carlo permutation p-values reported in the first submission with
EXACT two-sided p-values: all C(n1+n2, n1) scheme-label assignments are enumerated,
the statistic is the log variance ratio  log(var_clipmass / var_random)  of the
per-draw accuracies, and p = 2 * min(P(T >= t_obs), P(T <= t_obs)) over the exact
null distribution (adding the point mass so p is never zero).

Inputs are the released per-draw accuracies (run_manifest.csv); numbers below are
copied from the released aggregation JSONs so this file is self-contained.

Usage:  python src/w4h_exact_perm.py
Output: prints the exact tests; writes paper/w4h_exact_perm.json
"""
from __future__ import annotations

import itertools
import json
import math
import pathlib
from statistics import variance

# per-draw test accuracies, OceanShip vessel-disjoint, rec2 recipe (released draws)
RANDOM = {
    "cnn_small": [0.2926, 0.2770, 0.3519, 0.3792, 0.2780],
    "cnn_wide": [0.29281, 0.27541, 0.33847, 0.33512, 0.27833],
    "logreg": [0.27089, 0.26638, 0.30777, 0.30916, 0.24589],
}
CLIPMASS = {
    "cnn_small": [0.3296, 0.3177, 0.3337, 0.3318, 0.3022, 0.3140],
    "cnn_wide": [0.30340, 0.31577, 0.32557, 0.31254, 0.29077, 0.32629],
    "logreg": [0.28453, 0.28743, 0.29737, 0.29061, 0.26920, 0.28444],
}


def exact_perm(x, y):
    """Exact two-sided permutation p for log variance ratio, pooled labels."""
    pooled = list(x) + list(y)
    n1, n = len(x), len(pooled)
    t_obs = math.log(variance(y) / variance(x))
    stat = []
    for idx in itertools.combinations(range(n), n1):
        a = [pooled[i] for i in idx]
        b = [pooled[i] for i in range(n) if i not in idx]
        if variance(a) > 0 and variance(b) > 0:
            stat.append(math.log(variance(b) / variance(a)))
    stat.sort()
    hi = sum(1 for s in stat if s >= t_obs - 1e-12)
    lo = sum(1 for s in stat if s <= t_obs + 1e-12)
    p_one = min(hi, lo) / len(stat)
    p_two = min(1.0, 2 * p_one)
    return t_obs, p_two, len(stat)


def main():
    out = {}
    for mdl in RANDOM:
        t, p, m = exact_perm(RANDOM[mdl], CLIPMASS[mdl])
        vr = math.exp(t)
        out[mdl] = dict(log_var_ratio=t, variance_ratio=vr, exact_p_two_sided=p,
                        n_assignments=m)
        print(f"{mdl}: variance ratio = {vr:.2f}   exact two-sided p = {p:.4f}   "
              f"({m} assignments)")
    dest = pathlib.Path(__file__).resolve().parents[1] / "paper" / "w4h_exact_perm.json"
    dest.write_text(json.dumps(dict(kind="w4h-exact-permutation-remedy",
                                    statistic="log variance ratio",
                                    two_sided=True, cells=out), indent=1),
                    encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
