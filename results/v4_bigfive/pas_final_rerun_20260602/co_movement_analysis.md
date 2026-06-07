# Co-Movement Analysis

This report separates BFI-side movement from TRAIT-side movement. The
question is whether the two surfaces move toward the target together,
rather than whether their gap compresses into a single combined score.

## Bucket Counts

| Condition | both_positive | bfi_only | trait_only | neither |
|---|---:|---:|---:|---:|
| prompt_loglik | 4 | 0 | 6 | 0 |
| as_loglik | 3 | 0 | 7 | 0 |
| pas_final | 0 | 0 | 10 | 0 |
| as_pas_final | 2 | 0 | 8 | 0 |
| prompt_as_pas_final | 4 | 0 | 6 | 0 |

## prompt_loglik

`prompt_loglik` has bucket counts {'both_positive': 4, 'bfi_only': 0, 'trait_only': 6, 'neither': 0}, mean BFI movement +0.127, and mean TRAIT movement +0.043.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| neuroticism | high | +0.344 | [+0.219, +0.469] | +0.043 | [+0.038, +0.048] | 0.300 | both_positive |
| agreeableness | high | +0.194 | [+0.056, +0.306] | +0.053 | [+0.045, +0.060] | 0.142 | both_positive |
| openness | low | +0.175 | [+0.100, +0.225] | +0.059 | [+0.052, +0.066] | 0.116 | both_positive |
| extraversion | low | +0.125 | [+0.031, +0.219] | +0.040 | [+0.034, +0.046] | 0.085 | both_positive |
| extraversion | high | +0.125 | [-0.094, +0.282] | +0.046 | [+0.040, +0.051] | 0.079 | trait_only |
| agreeableness | low | +0.111 | [-0.028, +0.222] | +0.056 | [+0.048, +0.065] | 0.055 | trait_only |
| conscientiousness | high | +0.083 | [-0.083, +0.250] | +0.032 | [+0.027, +0.036] | 0.052 | trait_only |
| neuroticism | low | +0.062 | [+0.000, +0.188] | +0.012 | [+0.008, +0.016] | 0.050 | trait_only |
| conscientiousness | low | +0.028 | [-0.083, +0.111] | +0.059 | [+0.052, +0.066] | 0.032 | trait_only |
| openness | high | +0.025 | [+0.000, +0.075] | +0.034 | [+0.028, +0.040] | 0.009 | trait_only |

## as_loglik

`as_loglik` has bucket counts {'both_positive': 3, 'bfi_only': 0, 'trait_only': 7, 'neither': 0}, mean BFI movement +0.067, and mean TRAIT movement +0.037.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| openness | low | +0.175 | [+0.100, +0.250] | +0.058 | [+0.051, +0.066] | 0.117 | both_positive |
| neuroticism | high | +0.156 | [+0.062, +0.219] | +0.040 | [+0.037, +0.044] | 0.116 | both_positive |
| agreeableness | high | +0.083 | [+0.027, +0.167] | +0.035 | [+0.030, +0.040] | 0.049 | both_positive |
| extraversion | low | +0.062 | [+0.000, +0.125] | +0.025 | [+0.022, +0.029] | 0.037 | trait_only |
| neuroticism | low | +0.000 | [-0.094, +0.094] | +0.029 | [+0.025, +0.033] | 0.029 | trait_only |
| conscientiousness | high | +0.056 | [+0.000, +0.139] | +0.034 | [+0.031, +0.038] | 0.021 | trait_only |
| agreeableness | low | +0.056 | [+0.000, +0.139] | +0.038 | [+0.033, +0.045] | 0.017 | trait_only |
| openness | high | +0.025 | [+0.000, +0.075] | +0.038 | [+0.033, +0.042] | 0.013 | trait_only |
| extraversion | high | +0.031 | [+0.000, +0.094] | +0.041 | [+0.037, +0.044] | 0.009 | trait_only |
| conscientiousness | low | +0.028 | [+0.000, +0.083] | +0.031 | [+0.027, +0.035] | 0.003 | trait_only |

## pas_final

`pas_final` has bucket counts {'both_positive': 0, 'bfi_only': 0, 'trait_only': 10, 'neither': 0}, mean BFI movement +0.000, and mean TRAIT movement +0.492.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| neuroticism | high | +0.000 | [+0.000, +0.000] | +0.522 | [+0.516, +0.529] | 0.522 | trait_only |
| conscientiousness | low | +0.000 | [+0.000, +0.000] | +0.516 | [+0.507, +0.524] | 0.516 | trait_only |
| extraversion | high | +0.000 | [+0.000, +0.000] | +0.510 | [+0.493, +0.522] | 0.510 | trait_only |
| agreeableness | high | +0.000 | [+0.000, +0.000] | +0.506 | [+0.497, +0.516] | 0.506 | trait_only |
| openness | low | +0.000 | [+0.000, +0.000] | +0.490 | [+0.469, +0.508] | 0.490 | trait_only |
| openness | high | +0.000 | [+0.000, +0.000] | +0.484 | [+0.461, +0.501] | 0.484 | trait_only |
| conscientiousness | high | +0.000 | [+0.000, +0.000] | +0.478 | [+0.460, +0.490] | 0.478 | trait_only |
| neuroticism | low | +0.000 | [+0.000, +0.000] | +0.478 | [+0.471, +0.484] | 0.478 | trait_only |
| agreeableness | low | +0.000 | [+0.000, +0.000] | +0.474 | [+0.448, +0.495] | 0.474 | trait_only |
| extraversion | low | +0.000 | [+0.000, +0.000] | +0.463 | [+0.435, +0.484] | 0.463 | trait_only |

## as_pas_final

`as_pas_final` has bucket counts {'both_positive': 2, 'bfi_only': 0, 'trait_only': 8, 'neither': 0}, mean BFI movement +0.067, and mean TRAIT movement +0.492.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| conscientiousness | low | +0.028 | [+0.000, +0.083] | +0.516 | [+0.508, +0.524] | 0.488 | trait_only |
| extraversion | high | +0.031 | [+0.000, +0.094] | +0.510 | [+0.494, +0.522] | 0.479 | trait_only |
| neuroticism | low | +0.000 | [-0.094, +0.094] | +0.478 | [+0.472, +0.484] | 0.478 | trait_only |
| openness | high | +0.025 | [+0.000, +0.075] | +0.484 | [+0.461, +0.501] | 0.459 | trait_only |
| agreeableness | high | +0.083 | [+0.000, +0.167] | +0.506 | [+0.495, +0.516] | 0.423 | trait_only |
| conscientiousness | high | +0.056 | [+0.000, +0.139] | +0.478 | [+0.458, +0.490] | 0.422 | trait_only |
| agreeableness | low | +0.056 | [+0.000, +0.112] | +0.474 | [+0.450, +0.495] | 0.418 | trait_only |
| extraversion | low | +0.062 | [+0.000, +0.156] | +0.463 | [+0.436, +0.484] | 0.400 | trait_only |
| neuroticism | high | +0.156 | [+0.062, +0.219] | +0.522 | [+0.515, +0.529] | 0.366 | both_positive |
| openness | low | +0.175 | [+0.100, +0.226] | +0.490 | [+0.468, +0.507] | 0.315 | both_positive |

## prompt_as_pas_final

`prompt_as_pas_final` has bucket counts {'both_positive': 4, 'bfi_only': 0, 'trait_only': 6, 'neither': 0}, mean BFI movement +0.118, and mean TRAIT movement +0.492.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| openness | high | +0.000 | [-0.075, +0.075] | +0.484 | [+0.460, +0.501] | 0.484 | trait_only |
| neuroticism | low | +0.031 | [+0.000, +0.094] | +0.478 | [+0.472, +0.484] | 0.446 | trait_only |
| agreeableness | low | +0.028 | [-0.250, +0.306] | +0.474 | [+0.449, +0.495] | 0.446 | trait_only |
| conscientiousness | low | +0.083 | [-0.028, +0.194] | +0.516 | [+0.508, +0.524] | 0.432 | trait_only |
| agreeableness | high | +0.111 | [-0.028, +0.250] | +0.506 | [+0.496, +0.516] | 0.395 | trait_only |
| openness | low | +0.100 | [+0.025, +0.175] | +0.490 | [+0.467, +0.507] | 0.390 | both_positive |
| conscientiousness | high | +0.139 | [-0.056, +0.306] | +0.478 | [+0.459, +0.490] | 0.339 | trait_only |
| extraversion | low | +0.125 | [+0.031, +0.219] | +0.463 | [+0.435, +0.482] | 0.338 | both_positive |
| neuroticism | high | +0.250 | [+0.031, +0.438] | +0.522 | [+0.516, +0.529] | 0.272 | both_positive |
| extraversion | high | +0.312 | [+0.188, +0.406] | +0.510 | [+0.495, +0.522] | 0.198 | both_positive |
