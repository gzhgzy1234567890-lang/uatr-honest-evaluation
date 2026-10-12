# Honest Evaluation of Underwater Acoustic Target Recognition (UATR)

Companion repository for the manuscript *"Honest Evaluation of Underwater Acoustic Target Recognition: Decomposing, Predicting, and Reducing Benchmark Variance"* (IEEE Journal of Oceanic Engineering, submission).

## What this paper shows

1. **Identity-disjoint evaluation has a noise floor nobody measured.** On OceanShip, the same
   model and recipe vary by **±5.0 pp** across 25 independent vessel-disjoint draws — 12× the
   clip-level noise (±0.4 pp) — while the nominal clip-vs-honest gap is 5.1 pp on OceanShip but
   30.8 pp on DeepShip.
2. **Most of that noise is metadata.** 89% of the vessel-disjoint variance is explained by the
   class composition of the test set (permutation p = 0.0006; 59% predictable out-of-sample under
   leave-one-draw-out). The composition predictor is calibrated once from per-class recalls and
   needs no retraining to score a candidate allocation.
3. **A class-wise clip-mass allocation collapses the variance** to an allocator-limited floor
   (4.69 → 1.24 pp; variance ratio 14.4, exact permutation p = 0.004), reducing the draws needed to resolve a
   2 pp improvement from ~100 to 8 per system (3–7 under fully paired comparisons) — all figures
   are non-central-t planning estimates.
4. **The remedy is correctly a no-op on DeepShip** (low composition variance), and the tail-class
   collapse under honest evaluation resists six interventions — a three-tier structure the
   metadata predicts.

## Repository layout

```
src/            all training, splitting, aggregation and figure scripts (python)
notebooks/      Kaggle notebooks (.ipynb) + single-file flat versions (.py)
frozen_splits/  SHA-256-seeded frozen split files (OceanShip + DeepShip)
paper/          manuscript (LaTeX + PDF), run_manifest.csv, aggregation JSONs, figures
```

## Key artifacts

| File | Content |
|---|---|
| `paper/run_manifest.csv` | one row per released draw: run_id, corpus, model, protocol, allocation scheme, split seed, accuracy, macro-F1 |
| `paper/w3e2_variance.json` | per-draw test-class composition + accuracy for every aggregate in the paper |
| `frozen_splits/oceanship_splits.csv.gz` | frozen OceanShip splits (all protocols/seeds used in the paper) |
| `frozen_splits/splits.csv.gz` | frozen DeepShip splits |

Every table in the paper regenerates from the per-draw result files via
`src/w3e_bootstrap_ci.py` (bootstrap CIs + paired tests) and
`src/w3e2_variance_decomposition.py` (variance decomposition with mixed-label-space guards).

## Reproducing the experiments

Training ran on Kaggle free-tier GPUs (T4/P100). Attach the feature datasets
(OceanShip / DeepShip feature arrays), open `notebooks/kaggle_w3all.ipynb` (single-cell flat
versions: `notebooks/kaggle_w3all_flat.py`, `notebooks/kaggle_w3p_flat.py`), and Run All —
resume logic skips completed draws. Every run embeds its recipe in the run id, performs a
recording- and vessel-overlap leak check, and refuses to write a leaky split.

## Data

- OceanShip: ONC (Ocean Networks Canada) hydrophone archive + AIS metadata, 53,770 clips /
  892 MMSI. DeepShip: 47 h, 242 ships / 609 recordings. See `docs/` and the manuscript
  Sec. III for preprocessing details.

## License

Code: MIT. Datasets follow the terms of their original publishers (ONC / DeepShip).
