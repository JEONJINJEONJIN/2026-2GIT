# Co-Movement Analysis

This report separates BFI-side movement from TRAIT-side movement. The
question is whether the two surfaces move toward the target together,
rather than whether their gap compresses into a single combined score.

## Bucket Counts

| Condition | both_positive | bfi_only | trait_only | neither |
|---|---:|---:|---:|---:|
| one_line_prompt | 10 | 0 | 0 | 0 |
| elaborate_prompt | 10 | 0 | 0 | 0 |
| as_pas_only | 0 | 0 | 10 | 0 |
| elaborate_prompt_as_pas | 10 | 0 | 0 | 0 |

## one_line_prompt

`one_line_prompt` moves both surfaces in many cells, but it also contains target directions where only the action surface moves. The largest BFI/TRAIT movement gap is 0.529 at neuroticism-high.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| neuroticism | high | +0.562 | [+0.562, +0.562] | +0.034 | [+0.029, +0.038] | 0.529 | both_positive |
| agreeableness | low | +0.472 | [+0.472, +0.472] | +0.058 | [+0.049, +0.067] | 0.414 | both_positive |
| conscientiousness | high | +0.361 | [+0.361, +0.361] | +0.024 | [+0.020, +0.028] | 0.337 | both_positive |
| extraversion | high | +0.281 | [+0.281, +0.281] | +0.036 | [+0.031, +0.042] | 0.245 | both_positive |
| extraversion | low | +0.188 | [+0.188, +0.188] | +0.039 | [+0.033, +0.044] | 0.149 | both_positive |
| openness | low | +0.200 | [+0.200, +0.200] | +0.066 | [+0.058, +0.075] | 0.134 | both_positive |
| agreeableness | high | +0.167 | [+0.167, +0.167] | +0.047 | [+0.041, +0.054] | 0.119 | both_positive |
| openness | high | +0.150 | [+0.150, +0.150] | +0.034 | [+0.029, +0.039] | 0.116 | both_positive |
| conscientiousness | low | +0.167 | [+0.167, +0.167] | +0.052 | [+0.045, +0.058] | 0.115 | both_positive |
| neuroticism | low | +0.094 | [+0.094, +0.094] | +0.013 | [+0.008, +0.017] | 0.081 | both_positive |

## elaborate_prompt

`elaborate_prompt` moves the speech-side BFI surface more reliably than the one-line prompt, while TRAIT also moves in all cells. The two movements are still not equal in magnitude, so the surfaces should be reported separately.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| neuroticism | high | +0.344 | [+0.344, +0.344] | +0.043 | [+0.038, +0.049] | 0.300 | both_positive |
| agreeableness | high | +0.194 | [+0.194, +0.194] | +0.053 | [+0.045, +0.061] | 0.142 | both_positive |
| openness | low | +0.175 | [+0.175, +0.175] | +0.059 | [+0.052, +0.066] | 0.116 | both_positive |
| extraversion | low | +0.125 | [+0.125, +0.125] | +0.040 | [+0.033, +0.046] | 0.085 | both_positive |
| extraversion | high | +0.125 | [+0.125, +0.125] | +0.046 | [+0.040, +0.052] | 0.079 | both_positive |
| agreeableness | low | +0.111 | [+0.111, +0.111] | +0.056 | [+0.047, +0.065] | 0.055 | both_positive |
| conscientiousness | high | +0.083 | [+0.083, +0.083] | +0.032 | [+0.027, +0.036] | 0.052 | both_positive |
| neuroticism | low | +0.062 | [+0.062, +0.062] | +0.012 | [+0.008, +0.016] | 0.050 | both_positive |
| conscientiousness | low | +0.028 | [+0.028, +0.028] | +0.059 | [+0.052, +0.067] | 0.032 | both_positive |
| openness | high | +0.025 | [+0.025, +0.025] | +0.034 | [+0.028, +0.040] | 0.009 | both_positive |

## as_pas_only

`as_pas_only` shows the clean action-surface effect: all cells are `trait_only`, with mean TRAIT movement +0.037 and mean BFI movement +0.000. This means AS/PAS moves action-side likelihoods without moving the BFI self-report surface.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| openness | low | +0.000 | [+0.000, +0.000] | +0.058 | [+0.051, +0.066] | 0.058 | trait_only |
| extraversion | high | +0.000 | [+0.000, +0.000] | +0.041 | [+0.037, +0.045] | 0.041 | trait_only |
| neuroticism | high | +0.000 | [+0.000, +0.000] | +0.040 | [+0.037, +0.044] | 0.040 | trait_only |
| agreeableness | low | +0.000 | [+0.000, +0.000] | +0.038 | [+0.033, +0.044] | 0.038 | trait_only |
| openness | high | +0.000 | [+0.000, +0.000] | +0.038 | [+0.033, +0.042] | 0.038 | trait_only |
| agreeableness | high | +0.000 | [+0.000, +0.000] | +0.035 | [+0.030, +0.039] | 0.035 | trait_only |
| conscientiousness | high | +0.000 | [+0.000, +0.000] | +0.034 | [+0.031, +0.038] | 0.034 | trait_only |
| conscientiousness | low | +0.000 | [+0.000, +0.000] | +0.031 | [+0.027, +0.035] | 0.031 | trait_only |
| neuroticism | low | +0.000 | [+0.000, +0.000] | +0.029 | [+0.025, +0.033] | 0.029 | trait_only |
| extraversion | low | +0.000 | [+0.000, +0.000] | +0.025 | [+0.022, +0.029] | 0.025 | trait_only |

## elaborate_prompt_as_pas

`elaborate_prompt_as_pas` produces co-movement in most cells, but the movement sizes are uneven. The clearest decoupling is neuroticism-high, where BFI moves much more than TRAIT. Overall bucket counts are {'both_positive': 10, 'bfi_only': 0, 'trait_only': 0, 'neither': 0}, with mean BFI movement +0.127 and mean TRAIT movement +0.060.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| neuroticism | high | +0.344 | [+0.344, +0.344] | +0.058 | [+0.053, +0.065] | 0.285 | both_positive |
| agreeableness | high | +0.194 | [+0.194, +0.194] | +0.065 | [+0.058, +0.073] | 0.129 | both_positive |
| openness | low | +0.175 | [+0.175, +0.175] | +0.085 | [+0.076, +0.095] | 0.090 | both_positive |
| extraversion | low | +0.125 | [+0.125, +0.125] | +0.041 | [+0.034, +0.048] | 0.084 | both_positive |
| extraversion | high | +0.125 | [+0.125, +0.125] | +0.069 | [+0.060, +0.078] | 0.056 | both_positive |
| agreeableness | low | +0.111 | [+0.111, +0.111] | +0.066 | [+0.057, +0.075] | 0.045 | both_positive |
| conscientiousness | low | +0.028 | [+0.028, +0.028] | +0.071 | [+0.061, +0.079] | 0.043 | both_positive |
| neuroticism | low | +0.062 | [+0.062, +0.062] | +0.030 | [+0.026, +0.035] | 0.032 | both_positive |
| openness | high | +0.025 | [+0.025, +0.025] | +0.056 | [+0.049, +0.064] | 0.031 | both_positive |
| conscientiousness | high | +0.083 | [+0.083, +0.083] | +0.058 | [+0.052, +0.064] | 0.025 | both_positive |
