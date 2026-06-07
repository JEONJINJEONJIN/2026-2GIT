# Task 5 Hyperparameter Sweep Summary

## Setup

- Selection mode: `loglik`
- Action scoring: description scoring
- Steering condition: `as_only_sep` (`contrast_explicit`)
- Scenarios: first 5 Team A scenarios
- Repeats: 10 per scenario/persona
- Grid: alpha `{0.5, 1.0, 2.0, 4.0, 8.0}` x layer groups `{early, middle, late}`
- Total sweep rows: 15 combinations x 2 personas = 30 metric rows

## Outputs

- Alpha sweep: `results/v2/sweeps/alpha_sweep.csv`
- Layer sweep: `results/v2/sweeps/layer_sweep.csv`
- Full alpha-layer grid: `results/v2/sweeps/alpha_layer_landscape.csv`
- Heatmaps:
  - `results/v2/sweep_figures/alpha_layer_landscape_aggressive.png`
  - `results/v2/sweep_figures/alpha_layer_landscape_cooperative.png`

## Sanity Checks

- Minimum parse_ok_rate across all sweep cells: `1.0`
- Minimum unknown_rate across all sweep cells: `0.0`
- No parse degradation observed at alpha `8.0`.

## Main Result

Aggressive alignment is saturated in most cells, so it is not useful for choosing
alpha/layer settings. Cooperative alignment is the deciding axis.

Best cooperative cell:

| alpha | layer_group | align_ITT | vs chance |
|---:|---|---:|---:|
| 4.0 | middle | 0.800 | +0.400 |

Reference alpha sweep on `middle`:

| alpha | aggressive ITT | cooperative ITT |
|---:|---:|---:|
| 0.5 | 1.000 | 0.200 |
| 1.0 | 1.000 | 0.200 |
| 2.0 | 1.000 | 0.600 |
| 4.0 | 1.000 | 0.800 |
| 8.0 | 0.800 | 0.600 |

Reference layer sweep at alpha `2.0`:

| layer_group | aggressive ITT | cooperative ITT |
|---|---:|---:|
| early | 1.000 | 0.000 |
| middle | 1.000 | 0.600 |
| late | 1.000 | 0.000 |

## Recommendation

Use `middle=[18,21,24]` and `alpha=4.0` for the next expanded run. This is the
only tested setting that improves cooperative alignment to `0.800` while keeping
aggressive alignment at `1.000` and preserving `parse_ok_rate=1.0`.
