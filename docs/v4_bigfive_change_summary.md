# V4 Big Five Change Summary

Last updated: 2026-05-25, Asia/Seoul

## Purpose

This document summarizes the main changes made during the v4 Big Five NPC
speech-action consistency work session.

The goal of the v4 line is not to measure LLM morality or the model's real
personality. The goal is to measure whether an NPC keeps the assigned persona
consistent between:

- speech: BFI self-report scores
- action: TRAIT scenario choices

The fixed research question is:

> Under what conditions does AS+PAS affect speech-action persona consistency
> for NPCs assigned a Big Five persona?

## High-Level Research Changes

### 1. Research Framing

Previous framing:

- AS+PAS reduces the gap between speech and action.
- The framing could be misread as morality or real-personality measurement.

Current framing:

- AS+PAS is studied as an intervention that may affect NPC persona
  speech-action consistency under specific conditions.
- The framing is exploratory, not universal.
- Negative or mixed results still count as useful evidence about when AS+PAS
  helps or fails.

Important claim boundary:

- Do not say this measures LLM morality.
- Do not say this measures the model's real personality.
- Do not say AS+PAS always improves consistency.
- Do not treat high trait directions as better than low trait directions.

### 2. Persona Axes

Previous axis:

- aggressive vs cooperative

Current axes:

- Agreeableness
- Conscientiousness
- Extraversion
- Neuroticism
- Openness

Reason:

- Big Five supports multidimensional NPC personas.
- High and low directions are both valid character settings.
- The traits are value-neutral enough for NPC design.
- BFI-44 and TRAIT provide defensible measurement tools.

### 3. Speech Measurement

Speech is now measured with BFI-44-style self-report scoring.

Added data/module support:

- `data/bfi/bfi44_scoring.csv`
- `data/bfi/bfi44_item_text.local.csv`
- `src/data/bfi.py`

Important note:

- `bfi44_item_text.local.csv` is local-only.
- Do not redistribute BFI item wording unless rights are explicitly cleared.

### 4. Action Measurement

Action is now measured with the TRAIT Big Five dataset instead of custom
hand-labeled scenarios.

Added normalized data:

- `data/trait_bigfive/scenarios.jsonl`
- `data/trait_bigfive/splits.json`
- `data/trait_bigfive/contrastive_pairs/`

Added code:

- `src/data/trait_convert.py`
- `src/data/trait_dataset.py`
- `scripts/11_convert_trait_bigfive.py`
- `scripts/09_build_trait_contrastive_pairs.py`

Reason:

- TRAIT gives situation plus action-choice tasks.
- Action options already have trait-direction labels.
- This improves construct validity compared with custom action labels.

## Metric Changes

### Primary Metric

The primary metric is:

```text
consistency = 1 - abs(BFI_score - TRAIT_score)
```

Interpretation:

- Higher means speech and action are closer on the same trait axis.
- This is an unsigned gap-closeness score.
- It is not a signed movement-toward-target score.

### Secondary Metrics

Target-direction movement is tracked separately:

```text
target_attainment = target_direction_score - 0.5
```

For low targets:

```text
target_direction_score = 1 - high_direction_score
```

Reason:

- Consistency alone can be misleading.
- A model can become more target-aligned in behavior while becoming less
  speech-action consistent.
- Therefore consistency and target attainment must be reported together.

## Condition Matrix

The v4 final experiment uses five conditions:

| Condition | Purpose |
|---|---|
| `baseline` | No persona prompt or intervention. |
| `one_line_prompt` | Minimal realistic persona prompting. |
| `elaborate_prompt` | Strong prompt-only condition. |
| `as_pas_only` | AS+PAS without persona prompt. |
| `elaborate_prompt_as_pas` | Strong prompt plus AS+PAS. |

Important caveat:

- The current final run does not include pure AS-only or pure PAS-only
  ablations.
- Therefore the current results cannot answer whether AS and PAS are
  individually necessary.
- Separate ablation conditions are needed for that claim.

## Current PAS Scoring Caveat

The current final run stores PAS outputs, but primary TRAIT scoring is still
computed from conditional log-likelihood.

Current behavior:

- `TRAIT_score` comes from log-likelihood action probabilities.
- PAS-selected actions are stored in `pas_loglik_agreement.csv`.
- PAS/loglik agreement is reported as a diagnostic.
- PAS does not overwrite the primary `TRAIT_score`.

Implication:

- Current AS+PAS language should be used carefully.
- For paper claims, PAS should be described as a diagnostic selection signal
  unless a PAS-as-policy scoring path is implemented and rerun.

## System Structure Added

The v4 work was separated from previous v2 artifacts.

Added namespaces:

```text
docs/v4_bigfive_*.md
configs/experiments/v4_bigfive_*.yaml
data/bfi/
data/trait_bigfive/
results/v4_bigfive/
src/data/
src/evaluation/bigfive_*.py
src/experiments/v4_bigfive_runner.py
src/controls/
src/demo/
```

Reason:

- Preserve existing v2 aggressive/cooperative work.
- Avoid mixing old metrics and new Big Five metrics.
- Make final experiment provenance easier to defend.

## Major Files Added Or Updated

### Documents

```text
docs/v4_bigfive_trait_design.md
docs/v4_bigfive_runbook.md
docs/v4_bigfive_phase2_summary.md
docs/v4_bigfive_phase4_final_analysis.md
docs/v4_bigfive_llm_handoff.md
docs/v4_bigfive_change_summary.md
```

### Configs

```text
configs/experiments/v4_bigfive_pilot.yaml
configs/experiments/v4_bigfive_pilot_controls.yaml
configs/experiments/v4_bigfive_format_smoke.yaml
configs/experiments/v4_bigfive_final.yaml
```

Final config highlights:

```yaml
split: test
alpha: 4.0
layers: [18, 21, 24]
tune_on_this_split: false
```

### Core Code

```text
src/data/bfi.py
src/data/trait_convert.py
src/data/trait_dataset.py
src/evaluation/bigfive_metrics.py
src/evaluation/bigfive_analysis.py
src/evaluation/side_effect_metrics.py
src/evaluation/paraphrase_robustness.py
src/experiments/manifest.py
src/experiments/v4_bigfive_runner.py
src/controls/vector_controls.py
```

### Existing Code Updated

```text
src/generation/prompt_builder.py
src/models/loader.py
src/projection/selector.py
requirements.txt
```

Key fixes:

- Added stricter response-format prompting.
- Added Qwen2 `model.layers` fallback support.
- Fixed CUDA/CPU mismatch in cosine similarity.
- Added `pyarrow` and `streamlit` dependencies.

### Scripts

```text
scripts/09_build_trait_contrastive_pairs.py
scripts/10_prepare_bigfive_pilot.py
scripts/11_convert_trait_bigfive.py
scripts/12_run_bigfive_pilot.py
scripts/13_check_bigfive_readiness.py
scripts/14_analyze_bigfive_results.py
scripts/15_score_side_effects.py
scripts/16_analyze_paraphrase_robustness.py
scripts/17_validate_bigfive_phase2.py
scripts/18_validate_bigfive_phase3.py
scripts/19_run_bigfive_demo.py
scripts/20_validate_bigfive_demo.py
scripts/21_make_v4_figures.py
```

## Analysis Changes

The final analysis now includes confidence intervals and all baseline-referenced
condition comparisons.

Important output files:

```text
results/v4_bigfive/final_metrics/condition_differences.csv
results/v4_bigfive/final_metrics/condition_effects_with_ci.csv
results/v4_bigfive/final_metrics/paired_bootstrap_ci.csv
results/v4_bigfive/final_metrics/analysis_report.md
results/v4_bigfive/final_metrics/final_report.md
```

`condition_effects_with_ci.csv` is the main paper-table source.

It contains:

```text
4 non-baseline conditions x 5 traits x 2 target directions = 40 rows
```

Each row includes:

- delta consistency
- consistency 95% bootstrap CI
- delta BFI target attainment
- delta TRAIT target attainment
- TRAIT target 95% bootstrap CI
- whether each CI excludes zero

Bootstrap settings:

```text
n_bootstrap = 500
seed = 42
```

## Final Test Result Summary

For `as_pas_only` versus baseline:

| Trait | Target | Delta consistency | 95% CI | Delta TRAIT target | 95% CI |
|---|---:|---:|---:|---:|---:|
| agreeableness | high | +0.022 | [+0.017, +0.028] | +0.035 | [+0.030, +0.039] |
| agreeableness | low | -0.030 | [-0.035, -0.024] | +0.038 | [+0.033, +0.044] |
| conscientiousness | high | +0.023 | [+0.019, +0.028] | +0.034 | [+0.031, +0.038] |
| conscientiousness | low | -0.024 | [-0.028, -0.020] | +0.031 | [+0.027, +0.035] |
| extraversion | high | +0.041 | [+0.037, +0.045] | +0.041 | [+0.037, +0.045] |
| extraversion | low | -0.025 | [-0.028, -0.022] | +0.025 | [+0.022, +0.028] |
| neuroticism | high | -0.032 | [-0.037, -0.028] | +0.040 | [+0.037, +0.044] |
| neuroticism | low | +0.017 | [+0.012, +0.022] | +0.029 | [+0.025, +0.033] |
| openness | high | +0.037 | [+0.032, +0.041] | +0.038 | [+0.033, +0.042] |
| openness | low | -0.057 | [-0.064, -0.051] | +0.058 | [+0.051, +0.065] |

Interpretation:

- `as_pas_only` improves TRAIT target attainment for every trait and direction.
- Consistency effects are conditional.
- Some target directions improve behavior while increasing the BFI-TRAIT gap.
- This supports the exploratory framing.

## Demo Changes

A Streamlit demo was added:

```text
src/demo/bigfive_app.py
src/demo/bigfive_demo_data.py
src/demo/bigfive_demo_validation.py
```

Demo features:

- Big Five sliders
- active trait/target direction selection
- scenario card selection
- prompt-only vs AS+PAS response comparison
- condition-level metric table
- consistency and target-attainment chart
- activation movement visualization
- final figure viewer

Import fix:

- `bigfive_app.py` now inserts the project root into `sys.path` so Streamlit can
  import `src.*` when launched from the app path.

Demo URL used during the session:

```text
http://127.0.0.1:8501
```

## Validation Performed

Recent validation command:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_bigfive_analysis.py tests\test_v4_bigfive_phase3_validation.py -q
```

Recent result:

```text
12 passed
```

Other validation added during the session:

- Phase 2 readiness validation
- Phase 3 readiness validation
- demo validation
- final analysis regeneration

## Remaining Gaps

### 1. AS-only / PAS-only ablation

The current final run cannot answer:

```text
Is AS alone necessary?
Is PAS alone necessary?
Does AS+PAS outperform each component separately?
```

Needed new conditions:

```text
as_only
pas_only
elaborate_prompt_as_only
elaborate_prompt_pas_only
```

This also requires deciding whether PAS should be treated as:

- diagnostic ranking only, or
- actual policy selection that determines `TRAIT_score`

### 2. Final split side-effect generation

Side-effect metrics exist in the design, but final split generated-response
examples may still be needed for a paper or demo.

### 3. Full five-trait control diagnostics

Phase 2 control diagnostics exist, but full final test controls may be useful
if the paper needs stronger claims against random or unrelated-vector effects.

### 4. Paper-ready tables and figures

The data is now available, but final manuscript tables should be selected from:

```text
results/v4_bigfive/final_metrics/condition_effects_with_ci.csv
results/v4_bigfive/final_figures/
```

## Recommended Next Step

The most important next technical step is to implement and run the ablation
matrix if the paper needs to answer whether AS and PAS are individually
necessary.

If the paper does not need component-level attribution, the current result is
already usable with the narrower claim:

> AS+PAS conditions shift TRAIT behavior toward target Big Five directions, but
> speech-action consistency effects depend on trait, direction, and prompt
> condition.
