# V4 Big Five Speech-Action Consistency Report

## Research Question

Under what conditions does AS+PAS affect speech-action persona consistency for NPCs assigned a Big Five persona?

## Design Guardrails

- This report measures NPC persona consistency, not LLM morality or real personality.
- High and low trait directions are treated as value-neutral NPC character settings.
- Consistency is interpreted with target attainment and side-effect metrics.
- Test-split results must not be used to retune alpha, layers, prompts, or vector construction.

## Metric Definitions

- BFI_score and TRAIT_score are normalized to [0, 1] on the same high-direction Big Five axis.
- speech-action consistency = 1 - abs(BFI_score - TRAIT_score). This is an unsigned gap-closeness score, not a signed movement-toward-target score.
- target attainment = target-direction score - 0.5. Low targets use 1 - high_direction_score before subtracting 0.5.
- In the current final run, the primary TRAIT_score is computed from conditional log-likelihood under each condition. PAS selections are reported as agreement diagnostics and are not used to overwrite the primary TRAIT_score.

## Run Metadata

- run_id: `v4_bigfive_pas_final__seed_42`
- model_name: `Qwen/Qwen2.5-3B-Instruct`
- git_commit: `50098b5f7b94a96a25eb056f4176b7f518ac1ac4`
- data_split: `test`
- seed: `42`
- alpha: `4.0`
- layers: `[18, 21, 24]`
- vector_hash: `303b7d09d44d0e9b3b904c634a5c61570e895d315f80e8e9ddd9b815627a1880`
- config_sha256: `6f15c17fe0a535796ffd06611fffa9510fb2cf8f81a51449e83346e3d14d95d6`
- model_config_sha256: `4a0ae79778b98a396e2d2da9515301538f1930441df6d04726c9ebe196beb547`

## Primary Results

| condition | trait | target | n | consistency | BFI target | TRAIT target |
|---|---|---:|---:|---:|---:|---:|
| as_loglik | agreeableness | high | 150 | 0.887 | 0.139 | 0.029 |
| as_loglik | agreeableness | low | 150 | 0.942 | 0.000 | 0.045 |
| as_loglik | conscientiousness | high | 150 | 0.908 | 0.139 | 0.050 |
| as_loglik | conscientiousness | low | 150 | 0.926 | -0.056 | 0.015 |
| as_loglik | extraversion | high | 150 | 0.867 | 0.156 | 0.023 |
| as_loglik | extraversion | low | 150 | 0.895 | -0.062 | 0.042 |
| as_loglik | neuroticism | high | 150 | 0.922 | 0.094 | 0.018 |
| as_loglik | neuroticism | low | 150 | 0.969 | 0.062 | 0.052 |
| as_loglik | openness | high | 150 | 0.839 | 0.200 | 0.040 |
| as_loglik | openness | low | 150 | 0.939 | 0.000 | 0.055 |
| as_pas_final | agreeableness | high | 150 | 0.639 | 0.139 | 0.500 |
| as_pas_final | agreeableness | low | 150 | 0.500 | 0.000 | 0.480 |
| as_pas_final | conscientiousness | high | 150 | 0.637 | 0.139 | 0.493 |
| as_pas_final | conscientiousness | low | 150 | 0.444 | -0.056 | 0.500 |
| as_pas_final | extraversion | high | 150 | 0.654 | 0.156 | 0.493 |
| as_pas_final | extraversion | low | 150 | 0.440 | -0.062 | 0.480 |
| as_pas_final | neuroticism | high | 150 | 0.594 | 0.094 | 0.500 |
| as_pas_final | neuroticism | low | 150 | 0.562 | 0.062 | 0.500 |
| as_pas_final | openness | high | 150 | 0.695 | 0.200 | 0.487 |
| as_pas_final | openness | low | 150 | 0.500 | 0.000 | 0.487 |
| baseline_loglik | agreeableness | high | 150 | 0.927 | 0.056 | -0.006 |
| baseline_loglik | agreeableness | low | 150 | 0.927 | -0.056 | 0.006 |
| baseline_loglik | conscientiousness | high | 150 | 0.925 | 0.083 | 0.016 |
| baseline_loglik | conscientiousness | low | 150 | 0.925 | -0.083 | -0.016 |
| baseline_loglik | extraversion | high | 150 | 0.858 | 0.125 | -0.017 |
| baseline_loglik | extraversion | low | 150 | 0.858 | -0.125 | 0.017 |
| baseline_loglik | neuroticism | high | 150 | 0.952 | -0.062 | -0.022 |
| baseline_loglik | neuroticism | low | 150 | 0.952 | 0.062 | 0.022 |
| baseline_loglik | openness | high | 150 | 0.827 | 0.175 | 0.003 |
| baseline_loglik | openness | low | 150 | 0.827 | -0.175 | -0.003 |
| pas_final | agreeableness | high | 150 | 0.556 | 0.056 | 0.500 |
| pas_final | agreeableness | low | 150 | 0.447 | -0.056 | 0.480 |
| pas_final | conscientiousness | high | 150 | 0.582 | 0.083 | 0.493 |
| pas_final | conscientiousness | low | 150 | 0.417 | -0.083 | 0.500 |
| pas_final | extraversion | high | 150 | 0.623 | 0.125 | 0.493 |
| pas_final | extraversion | low | 150 | 0.380 | -0.125 | 0.480 |
| pas_final | neuroticism | high | 150 | 0.438 | -0.062 | 0.500 |
| pas_final | neuroticism | low | 150 | 0.562 | 0.062 | 0.500 |
| pas_final | openness | high | 150 | 0.670 | 0.175 | 0.487 |
| pas_final | openness | low | 150 | 0.330 | -0.175 | 0.487 |
| prompt_as_pas_final | agreeableness | high | 150 | 0.667 | 0.167 | 0.500 |
| prompt_as_pas_final | agreeableness | low | 150 | 0.473 | -0.028 | 0.480 |
| prompt_as_pas_final | conscientiousness | high | 150 | 0.719 | 0.222 | 0.493 |
| prompt_as_pas_final | conscientiousness | low | 150 | 0.500 | 0.000 | 0.500 |
| prompt_as_pas_final | extraversion | high | 150 | 0.932 | 0.438 | 0.493 |
| prompt_as_pas_final | extraversion | low | 150 | 0.500 | 0.000 | 0.480 |
| prompt_as_pas_final | neuroticism | high | 150 | 0.688 | 0.188 | 0.500 |
| prompt_as_pas_final | neuroticism | low | 150 | 0.594 | 0.094 | 0.500 |
| prompt_as_pas_final | openness | high | 150 | 0.670 | 0.175 | 0.487 |
| prompt_as_pas_final | openness | low | 150 | 0.427 | -0.075 | 0.487 |
| prompt_loglik | agreeableness | high | 150 | 0.797 | 0.250 | 0.047 |
| prompt_loglik | agreeableness | low | 150 | 0.963 | 0.056 | 0.063 |
| prompt_loglik | conscientiousness | high | 150 | 0.880 | 0.167 | 0.047 |
| prompt_loglik | conscientiousness | low | 150 | 0.900 | -0.056 | 0.044 |
| prompt_loglik | extraversion | high | 150 | 0.779 | 0.250 | 0.029 |
| prompt_loglik | extraversion | low | 150 | 0.940 | 0.000 | 0.057 |
| prompt_loglik | neuroticism | high | 150 | 0.740 | 0.281 | 0.021 |
| prompt_loglik | neuroticism | low | 150 | 0.910 | 0.125 | 0.035 |
| prompt_loglik | openness | high | 150 | 0.835 | 0.200 | 0.037 |
| prompt_loglik | openness | low | 150 | 0.941 | 0.000 | 0.056 |

## Baseline-Referenced Effects

| condition | trait | target | delta consistency | consistency 95% CI | delta BFI target | delta TRAIT target | TRAIT target 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| as_loglik | agreeableness | high | -0.040 | [-0.046, -0.034] | +0.083 | +0.035 | [+0.030, +0.039] |
| as_loglik | agreeableness | low | +0.016 | [+0.011, +0.020] | +0.056 | +0.038 | [+0.033, +0.044] |
| as_loglik | conscientiousness | high | -0.017 | [-0.021, -0.013] | +0.056 | +0.034 | [+0.031, +0.038] |
| as_loglik | conscientiousness | low | +0.002 | [-0.002, +0.005] | +0.028 | +0.031 | [+0.027, +0.035] |
| as_loglik | extraversion | high | +0.009 | [+0.005, +0.013] | +0.031 | +0.041 | [+0.037, +0.045] |
| as_loglik | extraversion | low | +0.037 | [+0.034, +0.041] | +0.062 | +0.025 | [+0.022, +0.028] |
| as_loglik | neuroticism | high | -0.030 | [-0.040, -0.021] | +0.156 | +0.040 | [+0.037, +0.044] |
| as_loglik | neuroticism | low | +0.017 | [+0.012, +0.022] | +0.000 | +0.029 | [+0.025, +0.033] |
| as_loglik | openness | high | +0.012 | [+0.008, +0.016] | +0.025 | +0.038 | [+0.033, +0.042] |
| as_loglik | openness | low | +0.112 | [+0.105, +0.118] | +0.175 | +0.058 | [+0.051, +0.065] |
| as_pas_final | agreeableness | high | -0.288 | [-0.295, -0.280] | +0.083 | +0.506 | [+0.496, +0.516] |
| as_pas_final | agreeableness | low | -0.427 | [-0.433, -0.419] | +0.056 | +0.474 | [+0.448, +0.494] |
| as_pas_final | conscientiousness | high | -0.288 | [-0.294, -0.281] | +0.056 | +0.478 | [+0.463, +0.490] |
| as_pas_final | conscientiousness | low | -0.480 | [-0.487, -0.474] | +0.028 | +0.516 | [+0.509, +0.524] |
| as_pas_final | extraversion | high | -0.204 | [-0.211, -0.196] | +0.031 | +0.510 | [+0.495, +0.522] |
| as_pas_final | extraversion | low | -0.418 | [-0.425, -0.411] | +0.062 | +0.463 | [+0.436, +0.484] |
| as_pas_final | neuroticism | high | -0.358 | [-0.363, -0.353] | +0.156 | +0.522 | [+0.516, +0.529] |
| as_pas_final | neuroticism | low | -0.389 | [-0.394, -0.384] | +0.000 | +0.478 | [+0.471, +0.484] |
| as_pas_final | openness | high | -0.133 | [-0.144, -0.122] | +0.025 | +0.484 | [+0.463, +0.501] |
| as_pas_final | openness | low | -0.327 | [-0.335, -0.319] | +0.175 | +0.490 | [+0.466, +0.506] |
| pas_final | agreeableness | high | -0.371 | [-0.378, -0.364] | +0.000 | +0.506 | [+0.496, +0.515] |
| pas_final | agreeableness | low | -0.480 | [-0.488, -0.472] | +0.000 | +0.474 | [+0.449, +0.495] |
| pas_final | conscientiousness | high | -0.343 | [-0.349, -0.336] | +0.000 | +0.478 | [+0.461, +0.490] |
| pas_final | conscientiousness | low | -0.508 | [-0.515, -0.502] | +0.000 | +0.516 | [+0.507, +0.524] |
| pas_final | extraversion | high | -0.235 | [-0.242, -0.227] | +0.000 | +0.510 | [+0.492, +0.522] |
| pas_final | extraversion | low | -0.478 | [-0.486, -0.469] | +0.000 | +0.463 | [+0.437, +0.484] |
| pas_final | neuroticism | high | -0.514 | [-0.519, -0.509] | +0.000 | +0.522 | [+0.516, +0.529] |
| pas_final | neuroticism | low | -0.389 | [-0.395, -0.385] | +0.000 | +0.478 | [+0.470, +0.484] |
| pas_final | openness | high | -0.157 | [-0.168, -0.146] | +0.000 | +0.484 | [+0.465, +0.502] |
| pas_final | openness | low | -0.498 | [-0.507, -0.486] | +0.000 | +0.490 | [+0.465, +0.507] |
| prompt_as_pas_final | agreeableness | high | -0.260 | [-0.268, -0.253] | +0.111 | +0.506 | [+0.497, +0.515] |
| prompt_as_pas_final | agreeableness | low | -0.453 | [-0.460, -0.446] | +0.028 | +0.474 | [+0.449, +0.495] |
| prompt_as_pas_final | conscientiousness | high | -0.206 | [-0.216, -0.198] | +0.139 | +0.478 | [+0.459, +0.491] |
| prompt_as_pas_final | conscientiousness | low | -0.425 | [-0.432, -0.418] | +0.083 | +0.516 | [+0.507, +0.524] |
| prompt_as_pas_final | extraversion | high | +0.074 | [+0.059, +0.085] | +0.312 | +0.510 | [+0.494, +0.522] |
| prompt_as_pas_final | extraversion | low | -0.358 | [-0.365, -0.351] | +0.125 | +0.463 | [+0.435, +0.484] |
| prompt_as_pas_final | neuroticism | high | -0.264 | [-0.269, -0.259] | +0.250 | +0.522 | [+0.516, +0.529] |
| prompt_as_pas_final | neuroticism | low | -0.358 | [-0.363, -0.353] | +0.031 | +0.478 | [+0.471, +0.484] |
| prompt_as_pas_final | openness | high | -0.157 | [-0.167, -0.147] | +0.000 | +0.484 | [+0.461, +0.500] |
| prompt_as_pas_final | openness | low | -0.400 | [-0.408, -0.392] | +0.100 | +0.490 | [+0.468, +0.507] |
| prompt_loglik | agreeableness | high | -0.130 | [-0.138, -0.122] | +0.194 | +0.053 | [+0.045, +0.060] |
| prompt_loglik | agreeableness | low | +0.037 | [+0.029, +0.044] | +0.111 | +0.056 | [+0.048, +0.065] |
| prompt_loglik | conscientiousness | high | -0.045 | [-0.050, -0.040] | +0.083 | +0.032 | [+0.027, +0.035] |
| prompt_loglik | conscientiousness | low | -0.025 | [-0.033, -0.019] | +0.028 | +0.059 | [+0.051, +0.068] |
| prompt_loglik | extraversion | high | -0.079 | [-0.086, -0.073] | +0.125 | +0.046 | [+0.039, +0.052] |
| prompt_loglik | extraversion | low | +0.082 | [+0.076, +0.087] | +0.125 | +0.040 | [+0.034, +0.046] |
| prompt_loglik | neuroticism | high | -0.212 | [-0.222, -0.204] | +0.344 | +0.043 | [+0.038, +0.048] |
| prompt_loglik | neuroticism | low | -0.042 | [-0.047, -0.037] | +0.062 | +0.012 | [+0.008, +0.016] |
| prompt_loglik | openness | high | +0.008 | [+0.003, +0.014] | +0.025 | +0.034 | [+0.029, +0.040] |
| prompt_loglik | openness | low | +0.113 | [+0.107, +0.121] | +0.175 | +0.059 | [+0.052, +0.067] |

## Paired Bootstrap Intervals

| condition | trait | target | metric | n | mean delta | CI lower | CI upper |
|---|---|---:|---|---:|---:|---:|---:|
| as_loglik | agreeableness | high | consistency | 150 | -0.040 | -0.046 | -0.034 |
| as_loglik | agreeableness | low | consistency | 150 | 0.016 | 0.011 | 0.020 |
| as_loglik | conscientiousness | high | consistency | 150 | -0.017 | -0.021 | -0.013 |
| as_loglik | conscientiousness | low | consistency | 150 | 0.002 | -0.002 | 0.005 |
| as_loglik | extraversion | high | consistency | 150 | 0.009 | 0.005 | 0.013 |
| as_loglik | extraversion | low | consistency | 150 | 0.037 | 0.034 | 0.041 |
| as_loglik | neuroticism | high | consistency | 150 | -0.030 | -0.040 | -0.021 |
| as_loglik | neuroticism | low | consistency | 150 | 0.017 | 0.012 | 0.022 |
| as_loglik | openness | high | consistency | 150 | 0.012 | 0.008 | 0.016 |
| as_loglik | openness | low | consistency | 150 | 0.112 | 0.105 | 0.118 |
| as_pas_final | agreeableness | high | consistency | 150 | -0.288 | -0.295 | -0.280 |
| as_pas_final | agreeableness | low | consistency | 150 | -0.427 | -0.433 | -0.419 |
| as_pas_final | conscientiousness | high | consistency | 150 | -0.288 | -0.294 | -0.281 |
| as_pas_final | conscientiousness | low | consistency | 150 | -0.480 | -0.487 | -0.474 |
| as_pas_final | extraversion | high | consistency | 150 | -0.204 | -0.211 | -0.196 |
| as_pas_final | extraversion | low | consistency | 150 | -0.418 | -0.425 | -0.411 |
| as_pas_final | neuroticism | high | consistency | 150 | -0.358 | -0.363 | -0.353 |
| as_pas_final | neuroticism | low | consistency | 150 | -0.389 | -0.394 | -0.384 |
| as_pas_final | openness | high | consistency | 150 | -0.133 | -0.144 | -0.122 |
| as_pas_final | openness | low | consistency | 150 | -0.327 | -0.335 | -0.319 |
| pas_final | agreeableness | high | consistency | 150 | -0.371 | -0.378 | -0.364 |
| pas_final | agreeableness | low | consistency | 150 | -0.480 | -0.488 | -0.472 |
| pas_final | conscientiousness | high | consistency | 150 | -0.343 | -0.349 | -0.336 |
| pas_final | conscientiousness | low | consistency | 150 | -0.508 | -0.515 | -0.502 |
| pas_final | extraversion | high | consistency | 150 | -0.235 | -0.242 | -0.227 |
| pas_final | extraversion | low | consistency | 150 | -0.478 | -0.486 | -0.469 |
| pas_final | neuroticism | high | consistency | 150 | -0.514 | -0.519 | -0.509 |
| pas_final | neuroticism | low | consistency | 150 | -0.389 | -0.395 | -0.385 |
| pas_final | openness | high | consistency | 150 | -0.157 | -0.168 | -0.146 |
| pas_final | openness | low | consistency | 150 | -0.498 | -0.507 | -0.486 |
| prompt_as_pas_final | agreeableness | high | consistency | 150 | -0.260 | -0.268 | -0.253 |
| prompt_as_pas_final | agreeableness | low | consistency | 150 | -0.453 | -0.460 | -0.446 |
| prompt_as_pas_final | conscientiousness | high | consistency | 150 | -0.206 | -0.216 | -0.198 |
| prompt_as_pas_final | conscientiousness | low | consistency | 150 | -0.425 | -0.432 | -0.418 |
| prompt_as_pas_final | extraversion | high | consistency | 150 | 0.074 | 0.059 | 0.085 |
| prompt_as_pas_final | extraversion | low | consistency | 150 | -0.358 | -0.365 | -0.351 |
| prompt_as_pas_final | neuroticism | high | consistency | 150 | -0.264 | -0.269 | -0.259 |
| prompt_as_pas_final | neuroticism | low | consistency | 150 | -0.358 | -0.363 | -0.353 |
| prompt_as_pas_final | openness | high | consistency | 150 | -0.157 | -0.167 | -0.147 |
| prompt_as_pas_final | openness | low | consistency | 150 | -0.400 | -0.408 | -0.392 |
| prompt_loglik | agreeableness | high | consistency | 150 | -0.130 | -0.138 | -0.122 |
| prompt_loglik | agreeableness | low | consistency | 150 | 0.037 | 0.029 | 0.044 |
| prompt_loglik | conscientiousness | high | consistency | 150 | -0.045 | -0.050 | -0.040 |
| prompt_loglik | conscientiousness | low | consistency | 150 | -0.025 | -0.033 | -0.019 |
| prompt_loglik | extraversion | high | consistency | 150 | -0.079 | -0.086 | -0.073 |
| prompt_loglik | extraversion | low | consistency | 150 | 0.082 | 0.076 | 0.087 |
| prompt_loglik | neuroticism | high | consistency | 150 | -0.212 | -0.222 | -0.204 |
| prompt_loglik | neuroticism | low | consistency | 150 | -0.042 | -0.047 | -0.037 |
| prompt_loglik | openness | high | consistency | 150 | 0.008 | 0.003 | 0.014 |
| prompt_loglik | openness | low | consistency | 150 | 0.113 | 0.107 | 0.121 |
| as_loglik | agreeableness | high | trait_target_attainment | 150 | 0.035 | 0.030 | 0.039 |
| as_loglik | agreeableness | low | trait_target_attainment | 150 | 0.038 | 0.033 | 0.044 |
| as_loglik | conscientiousness | high | trait_target_attainment | 150 | 0.034 | 0.031 | 0.038 |
| as_loglik | conscientiousness | low | trait_target_attainment | 150 | 0.031 | 0.027 | 0.035 |
| as_loglik | extraversion | high | trait_target_attainment | 150 | 0.041 | 0.037 | 0.045 |
| as_loglik | extraversion | low | trait_target_attainment | 150 | 0.025 | 0.022 | 0.028 |
| as_loglik | neuroticism | high | trait_target_attainment | 150 | 0.040 | 0.037 | 0.044 |
| as_loglik | neuroticism | low | trait_target_attainment | 150 | 0.029 | 0.025 | 0.033 |
| as_loglik | openness | high | trait_target_attainment | 150 | 0.038 | 0.033 | 0.042 |
| as_loglik | openness | low | trait_target_attainment | 150 | 0.058 | 0.051 | 0.065 |
| as_pas_final | agreeableness | high | trait_target_attainment | 150 | 0.506 | 0.496 | 0.516 |
| as_pas_final | agreeableness | low | trait_target_attainment | 150 | 0.474 | 0.448 | 0.494 |
| as_pas_final | conscientiousness | high | trait_target_attainment | 150 | 0.478 | 0.463 | 0.490 |
| as_pas_final | conscientiousness | low | trait_target_attainment | 150 | 0.516 | 0.509 | 0.524 |
| as_pas_final | extraversion | high | trait_target_attainment | 150 | 0.510 | 0.495 | 0.522 |
| as_pas_final | extraversion | low | trait_target_attainment | 150 | 0.463 | 0.436 | 0.484 |
| as_pas_final | neuroticism | high | trait_target_attainment | 150 | 0.522 | 0.516 | 0.529 |
| as_pas_final | neuroticism | low | trait_target_attainment | 150 | 0.478 | 0.471 | 0.484 |
| as_pas_final | openness | high | trait_target_attainment | 150 | 0.484 | 0.463 | 0.501 |
| as_pas_final | openness | low | trait_target_attainment | 150 | 0.490 | 0.466 | 0.506 |
| pas_final | agreeableness | high | trait_target_attainment | 150 | 0.506 | 0.496 | 0.515 |
| pas_final | agreeableness | low | trait_target_attainment | 150 | 0.474 | 0.449 | 0.495 |
| pas_final | conscientiousness | high | trait_target_attainment | 150 | 0.478 | 0.461 | 0.490 |
| pas_final | conscientiousness | low | trait_target_attainment | 150 | 0.516 | 0.507 | 0.524 |
| pas_final | extraversion | high | trait_target_attainment | 150 | 0.510 | 0.492 | 0.522 |
| pas_final | extraversion | low | trait_target_attainment | 150 | 0.463 | 0.437 | 0.484 |
| pas_final | neuroticism | high | trait_target_attainment | 150 | 0.522 | 0.516 | 0.529 |
| pas_final | neuroticism | low | trait_target_attainment | 150 | 0.478 | 0.470 | 0.484 |
| pas_final | openness | high | trait_target_attainment | 150 | 0.484 | 0.465 | 0.502 |
| pas_final | openness | low | trait_target_attainment | 150 | 0.490 | 0.465 | 0.507 |
| prompt_as_pas_final | agreeableness | high | trait_target_attainment | 150 | 0.506 | 0.497 | 0.515 |
| prompt_as_pas_final | agreeableness | low | trait_target_attainment | 150 | 0.474 | 0.449 | 0.495 |
| prompt_as_pas_final | conscientiousness | high | trait_target_attainment | 150 | 0.478 | 0.459 | 0.491 |
| prompt_as_pas_final | conscientiousness | low | trait_target_attainment | 150 | 0.516 | 0.507 | 0.524 |
| prompt_as_pas_final | extraversion | high | trait_target_attainment | 150 | 0.510 | 0.494 | 0.522 |
| prompt_as_pas_final | extraversion | low | trait_target_attainment | 150 | 0.463 | 0.435 | 0.484 |
| prompt_as_pas_final | neuroticism | high | trait_target_attainment | 150 | 0.522 | 0.516 | 0.529 |
| prompt_as_pas_final | neuroticism | low | trait_target_attainment | 150 | 0.478 | 0.471 | 0.484 |
| prompt_as_pas_final | openness | high | trait_target_attainment | 150 | 0.484 | 0.461 | 0.500 |
| prompt_as_pas_final | openness | low | trait_target_attainment | 150 | 0.490 | 0.468 | 0.507 |
| prompt_loglik | agreeableness | high | trait_target_attainment | 150 | 0.053 | 0.045 | 0.060 |
| prompt_loglik | agreeableness | low | trait_target_attainment | 150 | 0.056 | 0.048 | 0.065 |
| prompt_loglik | conscientiousness | high | trait_target_attainment | 150 | 0.032 | 0.027 | 0.035 |
| prompt_loglik | conscientiousness | low | trait_target_attainment | 150 | 0.059 | 0.051 | 0.068 |
| prompt_loglik | extraversion | high | trait_target_attainment | 150 | 0.046 | 0.039 | 0.052 |
| prompt_loglik | extraversion | low | trait_target_attainment | 150 | 0.040 | 0.034 | 0.046 |
| prompt_loglik | neuroticism | high | trait_target_attainment | 150 | 0.043 | 0.038 | 0.048 |
| prompt_loglik | neuroticism | low | trait_target_attainment | 150 | 0.012 | 0.008 | 0.016 |
| prompt_loglik | openness | high | trait_target_attainment | 150 | 0.034 | 0.029 | 0.040 |
| prompt_loglik | openness | low | trait_target_attainment | 150 | 0.059 | 0.052 | 0.067 |

## PAS/Loglik Agreement

| condition | trait | target | n | agreement rate |
|---|---|---:|---:|---:|
| as_pas_final | agreeableness | high | 150 | 0.307 |
| as_pas_final | agreeableness | low | 150 | 0.473 |
| as_pas_final | conscientiousness | high | 150 | 0.473 |
| as_pas_final | conscientiousness | low | 150 | 0.347 |
| as_pas_final | extraversion | high | 150 | 0.367 |
| as_pas_final | extraversion | low | 150 | 0.447 |
| as_pas_final | neuroticism | high | 150 | 0.307 |
| as_pas_final | neuroticism | low | 150 | 0.447 |
| as_pas_final | openness | high | 150 | 0.367 |
| as_pas_final | openness | low | 150 | 0.460 |
| pas_final | agreeableness | high | 150 | 0.187 |
| pas_final | agreeableness | low | 150 | 0.300 |
| pas_final | conscientiousness | high | 150 | 0.360 |
| pas_final | conscientiousness | low | 150 | 0.153 |
| pas_final | extraversion | high | 150 | 0.200 |
| pas_final | extraversion | low | 150 | 0.333 |
| pas_final | neuroticism | high | 150 | 0.127 |
| pas_final | neuroticism | low | 150 | 0.320 |
| pas_final | openness | high | 150 | 0.187 |
| pas_final | openness | low | 150 | 0.220 |
| prompt_as_pas_final | agreeableness | high | 150 | 0.387 |
| prompt_as_pas_final | agreeableness | low | 150 | 0.567 |
| prompt_as_pas_final | conscientiousness | high | 150 | 0.493 |
| prompt_as_pas_final | conscientiousness | low | 150 | 0.487 |
| prompt_as_pas_final | extraversion | high | 150 | 0.473 |
| prompt_as_pas_final | extraversion | low | 150 | 0.493 |
| prompt_as_pas_final | neuroticism | high | 150 | 0.420 |
| prompt_as_pas_final | neuroticism | low | 150 | 0.487 |
| prompt_as_pas_final | openness | high | 150 | 0.433 |
| prompt_as_pas_final | openness | low | 150 | 0.520 |

## Interpretation Template

- Positive result: AS+PAS improved speech-action consistency under specific trait/condition settings without degrading format validity.
- Mixed result: AS+PAS effects depended on trait direction, prompt strength, or vector-control diagnostics.
- Negative result: AS+PAS did not reliably reduce the BFI-TRAIT gap; this still constrains when activation steering is useful for NPC persona consistency.

## Claims To Avoid

- Do not claim this measures LLM morality.
- Do not claim this measures the model's real personality.
- Do not claim AS+PAS always improves consistency unless all relevant conditions support it.
- Do not rank high trait directions as better than low trait directions.
