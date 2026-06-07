# V4 Big Five Speech-Action Consistency Report

## Research Question

AS+PAS가 NPC에게 부여된 Big Five 페르소나에서 말(BFI 자기보고)과 행동(TRAIT 선택) 사이의 일관성에 어떤 조건에서 영향을 주는가?

## Design Guardrails

- This report measures NPC persona consistency, not LLM morality or real personality.
- High and low trait directions are treated as value-neutral NPC character settings.
- Consistency is interpreted with target attainment and side-effect metrics.
- Test-split results must not be used to retune alpha, layers, prompts, or vector construction.

## Run Metadata

- run_id: `v4_bigfive_format_smoke__seed_42`
- model_name: `Qwen/Qwen2.5-3B-Instruct`
- git_commit: `50098b5f7b94a96a25eb056f4176b7f518ac1ac4`
- data_split: `dev`
- seed: `42`
- alpha: `4.0`
- layers: `[18, 21, 24]`
- vector_hash: `93a513812b6c888bbcb29f48d5b3ad589abe8fd6c697a270021dbb48e59a19dc`
- config_sha256: `3df8fa06b7b36ea3e85b0870debf4bf99712d2726037aadc9d40a33e52477124`
- model_config_sha256: `4a0ae79778b98a396e2d2da9515301538f1930441df6d04726c9ebe196beb547`

## Primary Results

| condition | trait | target | n | consistency | BFI target | TRAIT target |
|---|---|---:|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | 10 | 0.939 | 0.056 | 0.040 |
| as_pas_only | agreeableness | low | 10 | 0.881 | -0.056 | 0.064 |
| baseline | agreeableness | high | 10 | 0.934 | 0.056 | 0.004 |
| baseline | agreeableness | low | 10 | 0.934 | -0.056 | -0.004 |
| elaborate_prompt | agreeableness | high | 10 | 0.844 | 0.250 | 0.094 |
| elaborate_prompt | agreeableness | low | 10 | 0.953 | 0.056 | 0.073 |
| elaborate_prompt_as_pas | agreeableness | high | 10 | 0.841 | 0.250 | 0.091 |
| elaborate_prompt_as_pas | agreeableness | low | 10 | 0.943 | 0.056 | 0.096 |
| one_line_prompt | agreeableness | high | 10 | 0.853 | 0.222 | 0.075 |
| one_line_prompt | agreeableness | low | 10 | 0.670 | 0.417 | 0.087 |

## Baseline-Referenced Effects

| condition | trait | target | delta consistency | delta BFI target | delta TRAIT target |
|---|---|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | 0.005 | 0.000 | 0.036 |
| as_pas_only | agreeableness | low | -0.053 | 0.000 | 0.068 |
| elaborate_prompt | agreeableness | high | -0.090 | 0.194 | 0.089 |
| elaborate_prompt | agreeableness | low | 0.020 | 0.111 | 0.078 |
| elaborate_prompt_as_pas | agreeableness | high | -0.093 | 0.194 | 0.086 |
| elaborate_prompt_as_pas | agreeableness | low | 0.009 | 0.111 | 0.101 |
| one_line_prompt | agreeableness | high | -0.081 | 0.167 | 0.071 |
| one_line_prompt | agreeableness | low | -0.264 | 0.472 | 0.091 |

## Paired Bootstrap Intervals

| condition | trait | target | metric | n | mean delta | CI lower | CI upper |
|---|---|---:|---|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | consistency | 10 | 0.005 | -0.019 | 0.024 |
| as_pas_only | agreeableness | low | consistency | 10 | -0.053 | -0.073 | -0.031 |
| elaborate_prompt | agreeableness | high | consistency | 10 | -0.090 | -0.123 | -0.057 |
| elaborate_prompt | agreeableness | low | consistency | 10 | 0.020 | -0.009 | 0.053 |
| elaborate_prompt_as_pas | agreeableness | high | consistency | 10 | -0.093 | -0.123 | -0.069 |
| elaborate_prompt_as_pas | agreeableness | low | consistency | 10 | 0.009 | -0.012 | 0.039 |
| one_line_prompt | agreeableness | high | consistency | 10 | -0.081 | -0.114 | -0.043 |
| one_line_prompt | agreeableness | low | consistency | 10 | -0.264 | -0.309 | -0.223 |

## PAS/Loglik Agreement

| condition | trait | target | n | agreement rate |
|---|---|---:|---:|---:|
| as_pas_only | agreeableness | high | 10 | 0.300 |
| as_pas_only | agreeableness | low | 10 | 0.500 |
| elaborate_prompt_as_pas | agreeableness | high | 10 | 0.400 |
| elaborate_prompt_as_pas | agreeableness | low | 10 | 0.600 |

## Side-Effect Checks

| condition | trait | target | n | format validity | parse success | mean words |
|---|---|---:|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | 10 | 1.000 | 1.000 | 23.9 |
| as_pas_only | agreeableness | low | 10 | 1.000 | 1.000 | 16.9 |
| baseline | agreeableness | high | 10 | 1.000 | 1.000 | 17.9 |
| baseline | agreeableness | low | 10 | 1.000 | 1.000 | 17.9 |
| elaborate_prompt | agreeableness | high | 10 | 1.000 | 1.000 | 21.7 |
| elaborate_prompt | agreeableness | low | 10 | 1.000 | 1.000 | 17.4 |
| elaborate_prompt_as_pas | agreeableness | high | 10 | 1.000 | 1.000 | 25.7 |
| elaborate_prompt_as_pas | agreeableness | low | 10 | 1.000 | 1.000 | 13.8 |
| one_line_prompt | agreeableness | high | 10 | 1.000 | 1.000 | 20.2 |
| one_line_prompt | agreeableness | low | 10 | 1.000 | 1.000 | 18.1 |

## Interpretation Template

- Positive result: AS+PAS improved speech-action consistency under specific trait/condition settings without degrading format validity.
- Mixed result: AS+PAS effects depended on trait direction, prompt strength, or vector-control diagnostics.
- Negative result: AS+PAS did not reliably reduce the BFI-TRAIT gap; this still constrains when activation steering is useful for NPC persona consistency.

## Claims To Avoid

- Do not claim this measures LLM morality.
- Do not claim this measures the model's real personality.
- Do not claim AS+PAS always improves consistency unless all relevant conditions support it.
- Do not rank high trait directions as better than low trait directions.
