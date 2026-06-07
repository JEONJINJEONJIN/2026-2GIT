# Co-Movement Analysis

This report separates BFI-side movement from TRAIT-side movement. The
question is whether the two surfaces move toward the target together,
rather than whether their gap compresses into a single combined score.

## Bucket Counts

| Condition | both_positive | bfi_only | trait_only | neither |
|---|---:|---:|---:|---:|
| one_line_prompt | 7 | 0 | 3 | 0 |
| elaborate_prompt | 4 | 0 | 6 | 0 |
| as_pas_only | 2 | 0 | 8 | 0 |
| elaborate_prompt_as_pas | 3 | 0 | 7 | 0 |

## one_line_prompt

`one_line_prompt` moves both surfaces in many cells, but it also contains target directions where only the action surface moves. The largest BFI/TRAIT movement gap is 0.529 at neuroticism-high.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| neuroticism | high | +0.562 | [+0.500, +0.656] | +0.034 | [+0.029, +0.038] | 0.529 | both_positive |
| agreeableness | low | +0.472 | [+0.361, +0.584] | +0.058 | [+0.049, +0.067] | 0.414 | both_positive |
| conscientiousness | high | +0.361 | [+0.250, +0.472] | +0.024 | [+0.020, +0.028] | 0.337 | both_positive |
| extraversion | high | +0.281 | [+0.125, +0.406] | +0.036 | [+0.031, +0.042] | 0.245 | both_positive |
| extraversion | low | +0.188 | [+0.094, +0.250] | +0.039 | [+0.033, +0.044] | 0.149 | both_positive |
| openness | low | +0.200 | [+0.125, +0.250] | +0.066 | [+0.058, +0.075] | 0.134 | both_positive |
| agreeableness | high | +0.167 | [+0.000, +0.333] | +0.047 | [+0.041, +0.054] | 0.119 | trait_only |
| openness | high | +0.150 | [+0.075, +0.225] | +0.034 | [+0.029, +0.039] | 0.116 | both_positive |
| conscientiousness | low | +0.167 | [-0.028, +0.333] | +0.052 | [+0.045, +0.058] | 0.115 | trait_only |
| neuroticism | low | +0.094 | [+0.000, +0.219] | +0.013 | [+0.008, +0.017] | 0.081 | trait_only |

## elaborate_prompt

`elaborate_prompt` moves the speech-side BFI surface more reliably than the one-line prompt, while TRAIT also moves in all cells. The two movements are still not equal in magnitude, so the surfaces should be reported separately.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| neuroticism | high | +0.344 | [+0.219, +0.469] | +0.043 | [+0.038, +0.049] | 0.300 | both_positive |
| agreeableness | high | +0.194 | [+0.083, +0.333] | +0.053 | [+0.045, +0.061] | 0.142 | both_positive |
| openness | low | +0.175 | [+0.100, +0.250] | +0.059 | [+0.052, +0.066] | 0.116 | both_positive |
| extraversion | low | +0.125 | [+0.031, +0.219] | +0.040 | [+0.033, +0.046] | 0.085 | both_positive |
| extraversion | high | +0.125 | [-0.094, +0.281] | +0.046 | [+0.040, +0.052] | 0.079 | trait_only |
| agreeableness | low | +0.111 | [-0.028, +0.250] | +0.056 | [+0.047, +0.065] | 0.055 | trait_only |
| conscientiousness | high | +0.083 | [-0.111, +0.222] | +0.032 | [+0.027, +0.036] | 0.052 | trait_only |
| neuroticism | low | +0.062 | [+0.000, +0.188] | +0.012 | [+0.008, +0.016] | 0.050 | trait_only |
| conscientiousness | low | +0.028 | [-0.056, +0.111] | +0.059 | [+0.052, +0.067] | 0.032 | trait_only |
| openness | high | +0.025 | [+0.000, +0.075] | +0.034 | [+0.028, +0.040] | 0.009 | trait_only |

## as_pas_only

`as_pas_only` isolates the no-prompt AS/PAS condition. Its bucket counts are {'both_positive': 2, 'bfi_only': 0, 'trait_only': 8, 'neither': 0}, with mean TRAIT movement +0.037 and mean BFI movement +0.067. Interpret this row as the direct test of whether AS also affects the BFI self-report surface.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| openness | low | +0.175 | [+0.100, +0.250] | +0.058 | [+0.051, +0.066] | 0.117 | both_positive |
| neuroticism | high | +0.156 | [+0.062, +0.219] | +0.040 | [+0.037, +0.044] | 0.116 | both_positive |
| agreeableness | high | +0.083 | [+0.000, +0.167] | +0.035 | [+0.030, +0.039] | 0.049 | trait_only |
| extraversion | low | +0.062 | [+0.000, +0.125] | +0.025 | [+0.022, +0.029] | 0.037 | trait_only |
| neuroticism | low | +0.000 | [-0.094, +0.094] | +0.029 | [+0.025, +0.033] | 0.029 | trait_only |
| conscientiousness | high | +0.056 | [+0.000, +0.111] | +0.034 | [+0.031, +0.038] | 0.021 | trait_only |
| agreeableness | low | +0.056 | [+0.000, +0.139] | +0.038 | [+0.033, +0.044] | 0.017 | trait_only |
| openness | high | +0.025 | [+0.000, +0.075] | +0.038 | [+0.033, +0.042] | 0.013 | trait_only |
| extraversion | high | +0.031 | [+0.000, +0.094] | +0.041 | [+0.037, +0.045] | 0.009 | trait_only |
| conscientiousness | low | +0.028 | [+0.000, +0.083] | +0.031 | [+0.027, +0.035] | 0.003 | trait_only |

## elaborate_prompt_as_pas

`elaborate_prompt_as_pas` produces co-movement in most cells, but the movement sizes are uneven. The clearest decoupling is neuroticism-high, where BFI moves much more than TRAIT. Overall bucket counts are {'both_positive': 3, 'bfi_only': 0, 'trait_only': 7, 'neither': 0}, with mean BFI movement +0.118 and mean TRAIT movement +0.060.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| extraversion | high | +0.312 | [+0.188, +0.406] | +0.069 | [+0.060, +0.078] | 0.244 | both_positive |
| neuroticism | high | +0.250 | [+0.000, +0.438] | +0.058 | [+0.053, +0.065] | 0.192 | trait_only |
| extraversion | low | +0.125 | [+0.031, +0.219] | +0.041 | [+0.034, +0.048] | 0.084 | both_positive |
| conscientiousness | high | +0.139 | [-0.056, +0.306] | +0.058 | [+0.052, +0.064] | 0.081 | trait_only |
| openness | high | +0.000 | [-0.075, +0.075] | +0.056 | [+0.049, +0.064] | 0.056 | trait_only |
| agreeableness | high | +0.111 | [-0.028, +0.250] | +0.065 | [+0.058, +0.073] | 0.046 | trait_only |
| agreeableness | low | +0.028 | [-0.250, +0.306] | +0.066 | [+0.057, +0.075] | 0.038 | trait_only |
| openness | low | +0.100 | [+0.025, +0.175] | +0.085 | [+0.076, +0.095] | 0.015 | both_positive |
| conscientiousness | low | +0.083 | [-0.028, +0.194] | +0.071 | [+0.061, +0.079] | 0.013 | trait_only |
| neuroticism | low | +0.031 | [+0.000, +0.094] | +0.030 | [+0.026, +0.035] | 0.001 | trait_only |
