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

- run_id: `v4_bigfive_final__seed_42`
- model_name: `Qwen/Qwen2.5-3B-Instruct`
- git_commit: `50098b5f7b94a96a25eb056f4176b7f518ac1ac4`
- data_split: `test`
- seed: `42`
- alpha: `4.0`
- layers: `[18, 21, 24]`
- vector_hash: `303b7d09d44d0e9b3b904c634a5c61570e895d315f80e8e9ddd9b815627a1880`
- config_sha256: `6867add2a7f4b40f67a56c8653a7e4ce4cffafeb4bb1ef6de5019c1b9dd42746`
- model_config_sha256: `4a0ae79778b98a396e2d2da9515301538f1930441df6d04726c9ebe196beb547`

## Primary Results

| condition | trait | target | n | consistency | BFI target | TRAIT target |
|---|---|---:|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | 150 | 0.949 | 0.056 | 0.029 |
| as_pas_only | agreeableness | low | 150 | 0.897 | -0.056 | 0.045 |
| as_pas_only | conscientiousness | high | 150 | 0.948 | 0.083 | 0.050 |
| as_pas_only | conscientiousness | low | 150 | 0.901 | -0.083 | 0.015 |
| as_pas_only | extraversion | high | 150 | 0.898 | 0.125 | 0.023 |
| as_pas_only | extraversion | low | 150 | 0.833 | -0.125 | 0.042 |
| as_pas_only | neuroticism | high | 150 | 0.919 | -0.062 | 0.018 |
| as_pas_only | neuroticism | low | 150 | 0.969 | 0.062 | 0.052 |
| as_pas_only | openness | high | 150 | 0.864 | 0.175 | 0.040 |
| as_pas_only | openness | low | 150 | 0.770 | -0.175 | 0.055 |
| baseline | agreeableness | high | 150 | 0.927 | 0.056 | -0.006 |
| baseline | agreeableness | low | 150 | 0.927 | -0.056 | 0.006 |
| baseline | conscientiousness | high | 150 | 0.925 | 0.083 | 0.016 |
| baseline | conscientiousness | low | 150 | 0.925 | -0.083 | -0.016 |
| baseline | extraversion | high | 150 | 0.858 | 0.125 | -0.017 |
| baseline | extraversion | low | 150 | 0.858 | -0.125 | 0.017 |
| baseline | neuroticism | high | 150 | 0.952 | -0.062 | -0.022 |
| baseline | neuroticism | low | 150 | 0.952 | 0.062 | 0.022 |
| baseline | openness | high | 150 | 0.827 | 0.175 | 0.003 |
| baseline | openness | low | 150 | 0.827 | -0.175 | -0.003 |
| elaborate_prompt | agreeableness | high | 150 | 0.797 | 0.250 | 0.047 |
| elaborate_prompt | agreeableness | low | 150 | 0.963 | 0.056 | 0.063 |
| elaborate_prompt | conscientiousness | high | 150 | 0.880 | 0.167 | 0.047 |
| elaborate_prompt | conscientiousness | low | 150 | 0.900 | -0.056 | 0.044 |
| elaborate_prompt | extraversion | high | 150 | 0.779 | 0.250 | 0.029 |
| elaborate_prompt | extraversion | low | 150 | 0.940 | 0.000 | 0.057 |
| elaborate_prompt | neuroticism | high | 150 | 0.740 | 0.281 | 0.021 |
| elaborate_prompt | neuroticism | low | 150 | 0.910 | 0.125 | 0.035 |
| elaborate_prompt | openness | high | 150 | 0.835 | 0.200 | 0.037 |
| elaborate_prompt | openness | low | 150 | 0.941 | 0.000 | 0.056 |
| elaborate_prompt_as_pas | agreeableness | high | 150 | 0.809 | 0.250 | 0.059 |
| elaborate_prompt_as_pas | agreeableness | low | 150 | 0.960 | 0.056 | 0.072 |
| elaborate_prompt_as_pas | conscientiousness | high | 150 | 0.904 | 0.167 | 0.074 |
| elaborate_prompt_as_pas | conscientiousness | low | 150 | 0.889 | -0.056 | 0.055 |
| elaborate_prompt_as_pas | extraversion | high | 150 | 0.802 | 0.250 | 0.052 |
| elaborate_prompt_as_pas | extraversion | low | 150 | 0.938 | 0.000 | 0.058 |
| elaborate_prompt_as_pas | neuroticism | high | 150 | 0.755 | 0.281 | 0.036 |
| elaborate_prompt_as_pas | neuroticism | low | 150 | 0.928 | 0.125 | 0.053 |
| elaborate_prompt_as_pas | openness | high | 150 | 0.857 | 0.200 | 0.059 |
| elaborate_prompt_as_pas | openness | low | 150 | 0.917 | 0.000 | 0.082 |
| one_line_prompt | agreeableness | high | 150 | 0.819 | 0.222 | 0.041 |
| one_line_prompt | agreeableness | low | 150 | 0.647 | 0.417 | 0.064 |
| one_line_prompt | conscientiousness | high | 150 | 0.595 | 0.444 | 0.039 |
| one_line_prompt | conscientiousness | low | 150 | 0.946 | 0.083 | 0.036 |
| one_line_prompt | extraversion | high | 150 | 0.613 | 0.406 | 0.019 |
| one_line_prompt | extraversion | low | 150 | 0.962 | 0.062 | 0.056 |
| one_line_prompt | neuroticism | high | 150 | 0.511 | 0.500 | 0.011 |
| one_line_prompt | neuroticism | low | 150 | 0.879 | 0.156 | 0.035 |
| one_line_prompt | openness | high | 150 | 0.712 | 0.325 | 0.037 |
| one_line_prompt | openness | low | 150 | 0.954 | 0.025 | 0.063 |

## Baseline-Referenced Effects

| condition | trait | target | delta consistency | consistency 95% CI | delta BFI target | delta TRAIT target | TRAIT target 95% CI |
|---|---|---:|---:|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | +0.022 | [+0.017, +0.028] | +0.000 | +0.035 | [+0.030, +0.039] |
| as_pas_only | agreeableness | low | -0.030 | [-0.035, -0.024] | +0.000 | +0.038 | [+0.033, +0.044] |
| as_pas_only | conscientiousness | high | +0.023 | [+0.019, +0.028] | +0.000 | +0.034 | [+0.031, +0.038] |
| as_pas_only | conscientiousness | low | -0.024 | [-0.028, -0.020] | +0.000 | +0.031 | [+0.027, +0.035] |
| as_pas_only | extraversion | high | +0.041 | [+0.037, +0.045] | +0.000 | +0.041 | [+0.037, +0.045] |
| as_pas_only | extraversion | low | -0.025 | [-0.028, -0.022] | +0.000 | +0.025 | [+0.022, +0.028] |
| as_pas_only | neuroticism | high | -0.032 | [-0.037, -0.028] | +0.000 | +0.040 | [+0.037, +0.044] |
| as_pas_only | neuroticism | low | +0.017 | [+0.012, +0.022] | +0.000 | +0.029 | [+0.025, +0.033] |
| as_pas_only | openness | high | +0.037 | [+0.032, +0.041] | +0.000 | +0.038 | [+0.033, +0.042] |
| as_pas_only | openness | low | -0.057 | [-0.064, -0.051] | +0.000 | +0.058 | [+0.051, +0.065] |
| elaborate_prompt | agreeableness | high | -0.130 | [-0.139, -0.121] | +0.194 | +0.053 | [+0.046, +0.060] |
| elaborate_prompt | agreeableness | low | +0.037 | [+0.030, +0.044] | +0.111 | +0.056 | [+0.048, +0.065] |
| elaborate_prompt | conscientiousness | high | -0.045 | [-0.050, -0.040] | +0.083 | +0.032 | [+0.028, +0.036] |
| elaborate_prompt | conscientiousness | low | -0.025 | [-0.031, -0.019] | +0.028 | +0.059 | [+0.052, +0.066] |
| elaborate_prompt | extraversion | high | -0.079 | [-0.085, -0.073] | +0.125 | +0.046 | [+0.040, +0.052] |
| elaborate_prompt | extraversion | low | +0.082 | [+0.077, +0.088] | +0.125 | +0.040 | [+0.033, +0.046] |
| elaborate_prompt | neuroticism | high | -0.212 | [-0.223, -0.202] | +0.344 | +0.043 | [+0.038, +0.049] |
| elaborate_prompt | neuroticism | low | -0.042 | [-0.048, -0.038] | +0.062 | +0.012 | [+0.008, +0.016] |
| elaborate_prompt | openness | high | +0.008 | [+0.002, +0.014] | +0.025 | +0.034 | [+0.028, +0.040] |
| elaborate_prompt | openness | low | +0.113 | [+0.107, +0.121] | +0.175 | +0.059 | [+0.052, +0.066] |
| elaborate_prompt_as_pas | agreeableness | high | -0.117 | [-0.126, -0.108] | +0.194 | +0.065 | [+0.058, +0.073] |
| elaborate_prompt_as_pas | agreeableness | low | +0.033 | [+0.026, +0.040] | +0.111 | +0.066 | [+0.057, +0.076] |
| elaborate_prompt_as_pas | conscientiousness | high | -0.020 | [-0.026, -0.015] | +0.083 | +0.058 | [+0.053, +0.064] |
| elaborate_prompt_as_pas | conscientiousness | low | -0.036 | [-0.045, -0.028] | +0.028 | +0.071 | [+0.062, +0.081] |
| elaborate_prompt_as_pas | extraversion | high | -0.056 | [-0.065, -0.049] | +0.125 | +0.069 | [+0.060, +0.076] |
| elaborate_prompt_as_pas | extraversion | low | +0.080 | [+0.074, +0.086] | +0.125 | +0.041 | [+0.034, +0.047] |
| elaborate_prompt_as_pas | neuroticism | high | -0.197 | [-0.207, -0.188] | +0.344 | +0.058 | [+0.052, +0.065] |
| elaborate_prompt_as_pas | neuroticism | low | -0.024 | [-0.030, -0.019] | +0.062 | +0.030 | [+0.025, +0.035] |
| elaborate_prompt_as_pas | openness | high | +0.030 | [+0.023, +0.037] | +0.025 | +0.056 | [+0.049, +0.063] |
| elaborate_prompt_as_pas | openness | low | +0.090 | [+0.080, +0.099] | +0.175 | +0.085 | [+0.075, +0.095] |
| one_line_prompt | agreeableness | high | -0.108 | [-0.116, -0.099] | +0.167 | +0.047 | [+0.041, +0.053] |
| one_line_prompt | agreeableness | low | -0.279 | [-0.291, -0.266] | +0.472 | +0.058 | [+0.051, +0.066] |
| one_line_prompt | conscientiousness | high | -0.330 | [-0.335, -0.324] | +0.361 | +0.024 | [+0.020, +0.028] |
| one_line_prompt | conscientiousness | low | +0.021 | [+0.011, +0.032] | +0.167 | +0.052 | [+0.045, +0.058] |
| one_line_prompt | extraversion | high | -0.245 | [-0.250, -0.240] | +0.281 | +0.036 | [+0.031, +0.041] |
| one_line_prompt | extraversion | low | +0.104 | [+0.096, +0.111] | +0.188 | +0.039 | [+0.033, +0.045] |
| one_line_prompt | neuroticism | high | -0.441 | [-0.452, -0.431] | +0.562 | +0.034 | [+0.030, +0.038] |
| one_line_prompt | neuroticism | low | -0.073 | [-0.078, -0.068] | +0.094 | +0.013 | [+0.008, +0.017] |
| one_line_prompt | openness | high | -0.115 | [-0.120, -0.110] | +0.150 | +0.034 | [+0.029, +0.039] |
| one_line_prompt | openness | low | +0.127 | [+0.119, +0.135] | +0.200 | +0.066 | [+0.058, +0.075] |

## Paired Bootstrap Intervals

| condition | trait | target | metric | n | mean delta | CI lower | CI upper |
|---|---|---:|---|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | consistency | 150 | 0.022 | 0.017 | 0.028 |
| as_pas_only | agreeableness | low | consistency | 150 | -0.030 | -0.035 | -0.024 |
| as_pas_only | conscientiousness | high | consistency | 150 | 0.023 | 0.019 | 0.028 |
| as_pas_only | conscientiousness | low | consistency | 150 | -0.024 | -0.028 | -0.020 |
| as_pas_only | extraversion | high | consistency | 150 | 0.041 | 0.037 | 0.045 |
| as_pas_only | extraversion | low | consistency | 150 | -0.025 | -0.028 | -0.022 |
| as_pas_only | neuroticism | high | consistency | 150 | -0.032 | -0.037 | -0.028 |
| as_pas_only | neuroticism | low | consistency | 150 | 0.017 | 0.012 | 0.022 |
| as_pas_only | openness | high | consistency | 150 | 0.037 | 0.032 | 0.041 |
| as_pas_only | openness | low | consistency | 150 | -0.057 | -0.064 | -0.051 |
| elaborate_prompt | agreeableness | high | consistency | 150 | -0.130 | -0.139 | -0.121 |
| elaborate_prompt | agreeableness | low | consistency | 150 | 0.037 | 0.030 | 0.044 |
| elaborate_prompt | conscientiousness | high | consistency | 150 | -0.045 | -0.050 | -0.040 |
| elaborate_prompt | conscientiousness | low | consistency | 150 | -0.025 | -0.031 | -0.019 |
| elaborate_prompt | extraversion | high | consistency | 150 | -0.079 | -0.085 | -0.073 |
| elaborate_prompt | extraversion | low | consistency | 150 | 0.082 | 0.077 | 0.088 |
| elaborate_prompt | neuroticism | high | consistency | 150 | -0.212 | -0.223 | -0.202 |
| elaborate_prompt | neuroticism | low | consistency | 150 | -0.042 | -0.048 | -0.038 |
| elaborate_prompt | openness | high | consistency | 150 | 0.008 | 0.002 | 0.014 |
| elaborate_prompt | openness | low | consistency | 150 | 0.113 | 0.107 | 0.121 |
| elaborate_prompt_as_pas | agreeableness | high | consistency | 150 | -0.117 | -0.126 | -0.108 |
| elaborate_prompt_as_pas | agreeableness | low | consistency | 150 | 0.033 | 0.026 | 0.040 |
| elaborate_prompt_as_pas | conscientiousness | high | consistency | 150 | -0.020 | -0.026 | -0.015 |
| elaborate_prompt_as_pas | conscientiousness | low | consistency | 150 | -0.036 | -0.045 | -0.028 |
| elaborate_prompt_as_pas | extraversion | high | consistency | 150 | -0.056 | -0.065 | -0.049 |
| elaborate_prompt_as_pas | extraversion | low | consistency | 150 | 0.080 | 0.074 | 0.086 |
| elaborate_prompt_as_pas | neuroticism | high | consistency | 150 | -0.197 | -0.207 | -0.188 |
| elaborate_prompt_as_pas | neuroticism | low | consistency | 150 | -0.024 | -0.030 | -0.019 |
| elaborate_prompt_as_pas | openness | high | consistency | 150 | 0.030 | 0.023 | 0.037 |
| elaborate_prompt_as_pas | openness | low | consistency | 150 | 0.090 | 0.080 | 0.099 |
| one_line_prompt | agreeableness | high | consistency | 150 | -0.108 | -0.116 | -0.099 |
| one_line_prompt | agreeableness | low | consistency | 150 | -0.279 | -0.291 | -0.266 |
| one_line_prompt | conscientiousness | high | consistency | 150 | -0.330 | -0.335 | -0.324 |
| one_line_prompt | conscientiousness | low | consistency | 150 | 0.021 | 0.011 | 0.032 |
| one_line_prompt | extraversion | high | consistency | 150 | -0.245 | -0.250 | -0.240 |
| one_line_prompt | extraversion | low | consistency | 150 | 0.104 | 0.096 | 0.111 |
| one_line_prompt | neuroticism | high | consistency | 150 | -0.441 | -0.452 | -0.431 |
| one_line_prompt | neuroticism | low | consistency | 150 | -0.073 | -0.078 | -0.068 |
| one_line_prompt | openness | high | consistency | 150 | -0.115 | -0.120 | -0.110 |
| one_line_prompt | openness | low | consistency | 150 | 0.127 | 0.119 | 0.135 |
| as_pas_only | agreeableness | high | trait_target_attainment | 150 | 0.035 | 0.030 | 0.039 |
| as_pas_only | agreeableness | low | trait_target_attainment | 150 | 0.038 | 0.033 | 0.044 |
| as_pas_only | conscientiousness | high | trait_target_attainment | 150 | 0.034 | 0.031 | 0.038 |
| as_pas_only | conscientiousness | low | trait_target_attainment | 150 | 0.031 | 0.027 | 0.035 |
| as_pas_only | extraversion | high | trait_target_attainment | 150 | 0.041 | 0.037 | 0.045 |
| as_pas_only | extraversion | low | trait_target_attainment | 150 | 0.025 | 0.022 | 0.028 |
| as_pas_only | neuroticism | high | trait_target_attainment | 150 | 0.040 | 0.037 | 0.044 |
| as_pas_only | neuroticism | low | trait_target_attainment | 150 | 0.029 | 0.025 | 0.033 |
| as_pas_only | openness | high | trait_target_attainment | 150 | 0.038 | 0.033 | 0.042 |
| as_pas_only | openness | low | trait_target_attainment | 150 | 0.058 | 0.051 | 0.065 |
| elaborate_prompt | agreeableness | high | trait_target_attainment | 150 | 0.053 | 0.046 | 0.060 |
| elaborate_prompt | agreeableness | low | trait_target_attainment | 150 | 0.056 | 0.048 | 0.065 |
| elaborate_prompt | conscientiousness | high | trait_target_attainment | 150 | 0.032 | 0.028 | 0.036 |
| elaborate_prompt | conscientiousness | low | trait_target_attainment | 150 | 0.059 | 0.052 | 0.066 |
| elaborate_prompt | extraversion | high | trait_target_attainment | 150 | 0.046 | 0.040 | 0.052 |
| elaborate_prompt | extraversion | low | trait_target_attainment | 150 | 0.040 | 0.033 | 0.046 |
| elaborate_prompt | neuroticism | high | trait_target_attainment | 150 | 0.043 | 0.038 | 0.049 |
| elaborate_prompt | neuroticism | low | trait_target_attainment | 150 | 0.012 | 0.008 | 0.016 |
| elaborate_prompt | openness | high | trait_target_attainment | 150 | 0.034 | 0.028 | 0.040 |
| elaborate_prompt | openness | low | trait_target_attainment | 150 | 0.059 | 0.052 | 0.066 |
| elaborate_prompt_as_pas | agreeableness | high | trait_target_attainment | 150 | 0.065 | 0.058 | 0.073 |
| elaborate_prompt_as_pas | agreeableness | low | trait_target_attainment | 150 | 0.066 | 0.057 | 0.076 |
| elaborate_prompt_as_pas | conscientiousness | high | trait_target_attainment | 150 | 0.058 | 0.053 | 0.064 |
| elaborate_prompt_as_pas | conscientiousness | low | trait_target_attainment | 150 | 0.071 | 0.062 | 0.081 |
| elaborate_prompt_as_pas | extraversion | high | trait_target_attainment | 150 | 0.069 | 0.060 | 0.076 |
| elaborate_prompt_as_pas | extraversion | low | trait_target_attainment | 150 | 0.041 | 0.034 | 0.047 |
| elaborate_prompt_as_pas | neuroticism | high | trait_target_attainment | 150 | 0.058 | 0.052 | 0.065 |
| elaborate_prompt_as_pas | neuroticism | low | trait_target_attainment | 150 | 0.030 | 0.025 | 0.035 |
| elaborate_prompt_as_pas | openness | high | trait_target_attainment | 150 | 0.056 | 0.049 | 0.063 |
| elaborate_prompt_as_pas | openness | low | trait_target_attainment | 150 | 0.085 | 0.075 | 0.095 |
| one_line_prompt | agreeableness | high | trait_target_attainment | 150 | 0.047 | 0.041 | 0.053 |
| one_line_prompt | agreeableness | low | trait_target_attainment | 150 | 0.058 | 0.051 | 0.066 |
| one_line_prompt | conscientiousness | high | trait_target_attainment | 150 | 0.024 | 0.020 | 0.028 |
| one_line_prompt | conscientiousness | low | trait_target_attainment | 150 | 0.052 | 0.045 | 0.058 |
| one_line_prompt | extraversion | high | trait_target_attainment | 150 | 0.036 | 0.031 | 0.041 |
| one_line_prompt | extraversion | low | trait_target_attainment | 150 | 0.039 | 0.033 | 0.045 |
| one_line_prompt | neuroticism | high | trait_target_attainment | 150 | 0.034 | 0.030 | 0.038 |
| one_line_prompt | neuroticism | low | trait_target_attainment | 150 | 0.013 | 0.008 | 0.017 |
| one_line_prompt | openness | high | trait_target_attainment | 150 | 0.034 | 0.029 | 0.039 |
| one_line_prompt | openness | low | trait_target_attainment | 150 | 0.066 | 0.058 | 0.075 |

## PAS/Loglik Agreement

| condition | trait | target | n | agreement rate |
|---|---|---:|---:|---:|
| as_pas_only | agreeableness | high | 150 | 0.307 |
| as_pas_only | agreeableness | low | 150 | 0.473 |
| as_pas_only | conscientiousness | high | 150 | 0.473 |
| as_pas_only | conscientiousness | low | 150 | 0.347 |
| as_pas_only | extraversion | high | 150 | 0.367 |
| as_pas_only | extraversion | low | 150 | 0.447 |
| as_pas_only | neuroticism | high | 150 | 0.307 |
| as_pas_only | neuroticism | low | 150 | 0.447 |
| as_pas_only | openness | high | 150 | 0.367 |
| as_pas_only | openness | low | 150 | 0.460 |
| elaborate_prompt_as_pas | agreeableness | high | 150 | 0.387 |
| elaborate_prompt_as_pas | agreeableness | low | 150 | 0.567 |
| elaborate_prompt_as_pas | conscientiousness | high | 150 | 0.493 |
| elaborate_prompt_as_pas | conscientiousness | low | 150 | 0.487 |
| elaborate_prompt_as_pas | extraversion | high | 150 | 0.473 |
| elaborate_prompt_as_pas | extraversion | low | 150 | 0.493 |
| elaborate_prompt_as_pas | neuroticism | high | 150 | 0.420 |
| elaborate_prompt_as_pas | neuroticism | low | 150 | 0.487 |
| elaborate_prompt_as_pas | openness | high | 150 | 0.433 |
| elaborate_prompt_as_pas | openness | low | 150 | 0.520 |

## Interpretation Template

- Positive result: AS+PAS improved speech-action consistency under specific trait/condition settings without degrading format validity.
- Mixed result: AS+PAS effects depended on trait direction, prompt strength, or vector-control diagnostics.
- Negative result: AS+PAS did not reliably reduce the BFI-TRAIT gap; this still constrains when activation steering is useful for NPC persona consistency.

## Claims To Avoid

- Do not claim this measures LLM morality.
- Do not claim this measures the model's real personality.
- Do not claim AS+PAS always improves consistency unless all relevant conditions support it.
- Do not rank high trait directions as better than low trait directions.
