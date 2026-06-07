# V4 Big Five Speech-Action Consistency Report

## Research Question

AS+PAS가 NPC에게 부여된 Big Five 페르소나에서 말(BFI 자기보고)과 행동(TRAIT 선택) 사이의 일관성에 어떤 조건에서 영향을 주는가?

## Design Guardrails

- This report measures NPC persona consistency, not LLM morality or real personality.
- High and low trait directions are treated as value-neutral NPC character settings.
- Consistency is interpreted with target attainment and side-effect metrics.
- Test-split results must not be used to retune alpha, layers, prompts, or vector construction.

## Run Metadata

- run_id: `v4_bigfive_pilot__seed_42`
- model_name: `Qwen/Qwen2.5-3B-Instruct`
- git_commit: `50098b5f7b94a96a25eb056f4176b7f518ac1ac4`
- data_split: `dev`
- seed: `42`
- alpha: `4.0`
- layers: `[18, 21, 24]`
- vector_hash: `93a513812b6c888bbcb29f48d5b3ad589abe8fd6c697a270021dbb48e59a19dc`
- config_sha256: `a1f6ab62db72a4bb676bbf44ea5268d8a157b240f2932218714d85f986181385`
- model_config_sha256: `4a0ae79778b98a396e2d2da9515301538f1930441df6d04726c9ebe196beb547`

## Primary Results

| condition | trait | target | n | consistency | BFI target | TRAIT target |
|---|---|---:|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | 50 | 0.952 | 0.056 | 0.032 |
| as_pas_only | agreeableness | low | 50 | 0.893 | -0.056 | 0.051 |
| baseline | agreeableness | high | 50 | 0.933 | 0.056 | -0.005 |
| baseline | agreeableness | low | 50 | 0.933 | -0.056 | 0.005 |
| elaborate_prompt | agreeableness | high | 50 | 0.808 | 0.250 | 0.058 |
| elaborate_prompt | agreeableness | low | 50 | 0.957 | 0.056 | 0.057 |
| elaborate_prompt_as_pas | agreeableness | high | 50 | 0.819 | 0.250 | 0.069 |
| elaborate_prompt_as_pas | agreeableness | low | 50 | 0.953 | 0.056 | 0.071 |
| one_line_prompt | agreeableness | high | 50 | 0.825 | 0.222 | 0.048 |
| one_line_prompt | agreeableness | low | 50 | 0.645 | 0.417 | 0.062 |

## Baseline-Referenced Effects

| condition | trait | target | delta consistency | delta BFI target | delta TRAIT target |
|---|---|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | 0.018 | 0.000 | 0.037 |
| as_pas_only | agreeableness | low | -0.040 | 0.000 | 0.046 |
| elaborate_prompt | agreeableness | high | -0.125 | 0.194 | 0.063 |
| elaborate_prompt | agreeableness | low | 0.024 | 0.111 | 0.052 |
| elaborate_prompt_as_pas | agreeableness | high | -0.114 | 0.194 | 0.074 |
| elaborate_prompt_as_pas | agreeableness | low | 0.020 | 0.111 | 0.066 |
| one_line_prompt | agreeableness | high | -0.108 | 0.167 | 0.052 |
| one_line_prompt | agreeableness | low | -0.288 | 0.472 | 0.057 |

## Paired Bootstrap Intervals

| condition | trait | target | metric | n | mean delta | CI lower | CI upper |
|---|---|---:|---|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | consistency | 50 | 0.018 | 0.010 | 0.027 |
| as_pas_only | agreeableness | low | consistency | 50 | -0.040 | -0.051 | -0.031 |
| elaborate_prompt | agreeableness | high | consistency | 50 | -0.125 | -0.140 | -0.110 |
| elaborate_prompt | agreeableness | low | consistency | 50 | 0.024 | 0.009 | 0.038 |
| elaborate_prompt_as_pas | agreeableness | high | consistency | 50 | -0.114 | -0.127 | -0.099 |
| elaborate_prompt_as_pas | agreeableness | low | consistency | 50 | 0.020 | 0.006 | 0.033 |
| one_line_prompt | agreeableness | high | consistency | 50 | -0.108 | -0.120 | -0.094 |
| one_line_prompt | agreeableness | low | consistency | 50 | -0.288 | -0.313 | -0.266 |

## PAS/Loglik Agreement

| condition | trait | target | n | agreement rate |
|---|---|---:|---:|---:|
| as_pas_only | agreeableness | high | 50 | 0.320 |
| as_pas_only | agreeableness | low | 50 | 0.460 |
| elaborate_prompt_as_pas | agreeableness | high | 50 | 0.460 |
| elaborate_prompt_as_pas | agreeableness | low | 50 | 0.660 |

## Side-Effect Checks

| condition | trait | target | n | format validity | parse success | mean words |
|---|---|---:|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | 50 | 1.000 | 1.000 | 22.2 |
| as_pas_only | agreeableness | low | 50 | 1.000 | 1.000 | 14.9 |
| baseline | agreeableness | high | 50 | 1.000 | 1.000 | 18.8 |
| baseline | agreeableness | low | 50 | 1.000 | 1.000 | 18.8 |
| elaborate_prompt | agreeableness | high | 50 | 1.000 | 1.000 | 21.5 |
| elaborate_prompt | agreeableness | low | 50 | 1.000 | 1.000 | 16.5 |
| elaborate_prompt_as_pas | agreeableness | high | 50 | 1.000 | 1.000 | 25.5 |
| elaborate_prompt_as_pas | agreeableness | low | 50 | 1.000 | 1.000 | 13.7 |
| one_line_prompt | agreeableness | high | 50 | 1.000 | 1.000 | 20.2 |
| one_line_prompt | agreeableness | low | 50 | 0.980 | 0.980 | 16.1 |

## Interpretation Template

- Positive result: AS+PAS improved speech-action consistency under specific trait/condition settings without degrading format validity.
- Mixed result: AS+PAS effects depended on trait direction, prompt strength, or vector-control diagnostics.
- Negative result: AS+PAS did not reliably reduce the BFI-TRAIT gap; this still constrains when activation steering is useful for NPC persona consistency.

## Claims To Avoid

- Do not claim this measures LLM morality.
- Do not claim this measures the model's real personality.
- Do not claim AS+PAS always improves consistency unless all relevant conditions support it.
- Do not rank high trait directions as better than low trait directions.
