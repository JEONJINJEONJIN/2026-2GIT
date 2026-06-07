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
- alpha: `2.0`
- layers: `[18, 21, 24]`
- vector_hash: `93a513812b6c888bbcb29f48d5b3ad589abe8fd6c697a270021dbb48e59a19dc`
- config_sha256: `a1f6ab62db72a4bb676bbf44ea5268d8a157b240f2932218714d85f986181385`
- model_config_sha256: `4a0ae79778b98a396e2d2da9515301538f1930441df6d04726c9ebe196beb547`

## Primary Results

| condition | trait | target | n | consistency | BFI target | TRAIT target |
|---|---|---:|---:|---:|---:|---:|
| as_pas_only | agreeableness | high | 50 | 0.946 | 0.056 | 0.018 |
| as_pas_only | agreeableness | low | 50 | 0.911 | -0.056 | 0.032 |

## Baseline-Referenced Effects

| condition | trait | target | delta consistency | delta BFI target | delta TRAIT target |
|---|---|---:|---:|---:|---:|

## Paired Bootstrap Intervals

| condition | trait | target | metric | n | mean delta | CI lower | CI upper |
|---|---|---:|---|---:|---:|---:|---:|

## PAS/Loglik Agreement

| condition | trait | target | n | agreement rate |
|---|---|---:|---:|---:|
| as_pas_only | agreeableness | high | 50 | 0.280 |
| as_pas_only | agreeableness | low | 50 | 0.380 |

## Interpretation Template

- Positive result: AS+PAS improved speech-action consistency under specific trait/condition settings without degrading format validity.
- Mixed result: AS+PAS effects depended on trait direction, prompt strength, or vector-control diagnostics.
- Negative result: AS+PAS did not reliably reduce the BFI-TRAIT gap; this still constrains when activation steering is useful for NPC persona consistency.

## Claims To Avoid

- Do not claim this measures LLM morality.
- Do not claim this measures the model's real personality.
- Do not claim AS+PAS always improves consistency unless all relevant conditions support it.
- Do not rank high trait directions as better than low trait directions.
