# V4 Big Five Speech-Action Consistency Report

## Research Question

AS+PAS가 NPC에게 부여된 Big Five 페르소나에서 말(BFI 자기보고)과 행동(TRAIT 선택) 사이의 일관성에 어떤 조건에서 영향을 주는가?

## Design Guardrails

- This report measures NPC persona consistency, not LLM morality or real personality.
- High and low trait directions are treated as value-neutral NPC character settings.
- Consistency is interpreted with target attainment and side-effect metrics.
- Test-split results must not be used to retune alpha, layers, prompts, or vector construction.

## Run Metadata

- run_id: `v4_bigfive_pilot_controls__seed_42`
- model_name: `Qwen/Qwen2.5-3B-Instruct`
- git_commit: `50098b5f7b94a96a25eb056f4176b7f518ac1ac4`
- data_split: `dev`
- seed: `42`
- alpha: `4.0`
- layers: `[18, 21, 24]`
- vector_hash: `d6871022ca377a9ae4c9f1d0a6ba0a8917762ad91ca36e9104250281a00b010f`
- config_sha256: `b45df506b2fc607d009b148f2d33b36d1fb133a911253eee1a9f7637187a0caa`
- model_config_sha256: `4a0ae79778b98a396e2d2da9515301538f1930441df6d04726c9ebe196beb547`

## Primary Results

| condition | trait | target | n | consistency | BFI target | TRAIT target |
|---|---|---:|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | 50 | 0.952 | 0.056 | 0.032 |
| as_pas_only | agreeableness | low | 50 | 0.893 | -0.056 | 0.051 |
| random_vector | agreeableness | high | 50 | 0.938 | 0.056 | -0.003 |
| random_vector | agreeableness | low | 50 | 0.939 | -0.056 | -0.001 |
| unrelated_trait_vector | agreeableness | high | 50 | 0.933 | 0.056 | -0.005 |
| unrelated_trait_vector | agreeableness | low | 50 | 0.935 | -0.056 | 0.005 |
| zero_hook | agreeableness | high | 50 | 0.933 | 0.056 | -0.005 |
| zero_hook | agreeableness | low | 50 | 0.933 | -0.056 | 0.005 |

## Baseline-Referenced Effects

| condition | trait | target | delta consistency | delta BFI target | delta TRAIT target |
|---|---|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | 0.018 | 0.000 | 0.037 |
| as_pas_only | agreeableness | low | -0.040 | 0.000 | 0.046 |
| random_vector | agreeableness | high | 0.004 | 0.000 | 0.002 |
| random_vector | agreeableness | low | 0.006 | 0.000 | -0.006 |
| unrelated_trait_vector | agreeableness | high | -0.000 | 0.000 | 0.000 |
| unrelated_trait_vector | agreeableness | low | 0.002 | 0.000 | -0.000 |

## Paired Bootstrap Intervals

| condition | trait | target | metric | n | mean delta | CI lower | CI upper |
|---|---|---:|---|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | consistency | 50 | 0.018 | 0.010 | 0.027 |
| as_pas_only | agreeableness | low | consistency | 50 | -0.040 | -0.051 | -0.031 |
| random_vector | agreeableness | high | consistency | 50 | 0.004 | 0.001 | 0.008 |
| random_vector | agreeableness | low | consistency | 50 | 0.006 | 0.002 | 0.009 |
| unrelated_trait_vector | agreeableness | high | consistency | 50 | -0.000 | -0.006 | 0.006 |
| unrelated_trait_vector | agreeableness | low | consistency | 50 | 0.002 | -0.003 | 0.008 |

## PAS/Loglik Agreement

| condition | trait | target | n | agreement rate |
|---|---|---:|---:|---:|
| as_pas_only | agreeableness | high | 50 | 0.320 |
| as_pas_only | agreeableness | low | 50 | 0.460 |

## Interpretation Template

- Positive result: AS+PAS improved speech-action consistency under specific trait/condition settings without degrading format validity.
- Mixed result: AS+PAS effects depended on trait direction, prompt strength, or vector-control diagnostics.
- Negative result: AS+PAS did not reliably reduce the BFI-TRAIT gap; this still constrains when activation steering is useful for NPC persona consistency.

## Claims To Avoid

- Do not claim this measures LLM morality.
- Do not claim this measures the model's real personality.
- Do not claim AS+PAS always improves consistency unless all relevant conditions support it.
- Do not rank high trait directions as better than low trait directions.
