# Slider Clamp Sweep Summary

Input CSV: `results/v4_bigfive/slider_clamp_sweep/slider_clamp_sweep_20260603.csv`

## Setup

- Profiles:
  - `social_explorer`: E=1.0, O=1.0, N=0.0, A=0.5, C=0.5
  - `reserved_conservative`: E=0.0, O=0.0, C=1.0, A=0.5, N=0.5
- Clamp values: `none`, `3.0`, `2.5`, `2.0`, `1.5`
- Scenarios: 8 fixed test scenarios
- Generations: 80 total
- Max new tokens: 220
- Alpha: 2.0
- Layers: 18, 21, 24

## Results

| Profile | clamp_x | mean norm | parse rate | max rep3 | max rep4 | mean len | max len |
|---|---:|---:|---:|---:|---:|---:|---:|
| reserved_conservative | 1.5 | 1.500 | 1.000 | 0.000 | 0.000 | 47.625 | 58 |
| reserved_conservative | 2.0 | 2.000 | 1.000 | 0.000 | 0.000 | 48.250 | 58 |
| reserved_conservative | 2.5 | 2.186 | 1.000 | 0.000 | 0.000 | 49.000 | 58 |
| reserved_conservative | 3.0 | 2.186 | 1.000 | 0.000 | 0.000 | 49.000 | 58 |
| reserved_conservative | none | 2.186 | 1.000 | 0.000 | 0.000 | 49.000 | 58 |
| social_explorer | 1.5 | 1.500 | 1.000 | 0.000 | 0.000 | 44.875 | 53 |
| social_explorer | 2.0 | 2.000 | 1.000 | 0.000 | 0.000 | 48.500 | 56 |
| social_explorer | 2.5 | 2.150 | 1.000 | 0.000 | 0.000 | 48.625 | 56 |
| social_explorer | 3.0 | 2.150 | 1.000 | 0.000 | 0.000 | 48.625 | 56 |
| social_explorer | none | 2.150 | 1.000 | 0.000 | 0.000 | 48.625 | 56 |

## Interpretation

No degeneration was observed in either tested profile. Both profiles retained a 100% parse rate, and max 3-gram / 4-gram repetition rates stayed at 0.0 for every clamp setting.

The negative-heavy `reserved_conservative` profile did **not** break at a lower clamp than the positive-heavy `social_explorer` profile in this sweep. Under these two profiles, the expected low-direction asymmetry was not observed as a format or repetition failure.

The largest finite clamp tested with both profiles coherent is `clamp_x = 3.0`. The UI backend default is therefore set to `DEFAULT_CLAMP_X = 3.0`. This should still be described as a tested demo clamp, not as proof that arbitrary 5-slider profiles are safe at every possible combination.
