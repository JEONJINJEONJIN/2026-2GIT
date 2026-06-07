# V4 Big Five LLM Handoff

Last updated: 2026-05-25, Asia/Seoul

## TL;DR

This repository now contains a working v4 Big Five NPC speech-action consistency
experiment.

Current status:

- Phase 1 design/data setup: complete.
- Phase 2 Agreeableness pilot/control/sweep/side-effect validation: complete.
- Phase 3 full Big Five test run: complete.
- Phase 4 final analysis and figures: complete.
- Streamlit demo: installed, running, and connected to final figures/results.

Main result:

> AS+PAS reliably moves TRAIT behavior toward the target Big Five direction, but
> its effect on speech-action consistency depends on trait, direction, and prompt
> condition.

Metric definition:

```text
consistency = 1 - abs(BFI_score - TRAIT_score)
```

This is an unsigned gap-closeness score on the high-direction Big Five axis. It
is not a signed movement-toward-target score. Directional interpretation must
use `bfi_target_attainment` and `trait_target_attainment` beside consistency.

Current scoring caveat:

- The primary `TRAIT_score` is computed from conditional log-likelihood under
  each condition.
- PAS selections are saved as PAS/loglik agreement diagnostics.
- PAS does not overwrite the primary `TRAIT_score` in the current final run.
- Therefore, do not claim separate AS-only or PAS-only effects from the current
  five-condition run. That requires an additional ablation.

Do not claim:

- AS+PAS always reduces the speech-action gap.
- The model has a real personality.
- High trait directions are better than low trait directions.
- This is an LLM morality evaluation.

Correct framing:

> This is an NPC persona consistency study, not a morality or real-personality
> measurement study.

## Current Demo

Streamlit is installed and the demo server was started at:

```text
http://127.0.0.1:8501
```

Server session id from the current work session:

```text
37436
```

Demo file:

```text
src/demo/bigfive_app.py
```

The demo now includes:

- Big Five sliders.
- Active trait and high/low target direction derived from sliders.
- Scenario selector.
- Condition-level behavior table.
- Consistency/target-attainment bar chart.
- Prompt-only vs prompt+AS/PAS NPC response comparison.
- Activation movement scatter.
- Final result figure viewer.

Validation command:

```powershell
.venv\Scripts\python.exe scripts\20_validate_bigfive_demo.py
```

Expected current result:

```text
OK demo_required_paths
OK demo_fallback_data
OK streamlit_dependency
OK pilot_generated_responses: 500 response rows available
OK final_figures: 5 PNG figures available
```

If the app cannot import `src`, check that `src/demo/bigfive_app.py` still adds
the project root to `sys.path`.

## Key Files Added Or Modified

Design and summary docs:

```text
docs/v4_bigfive_trait_design.md
docs/v4_bigfive_runbook.md
docs/v4_bigfive_phase2_summary.md
docs/v4_bigfive_phase4_final_analysis.md
docs/v4_bigfive_llm_handoff.md
```

Configs:

```text
configs/experiments/v4_bigfive_pilot.yaml
configs/experiments/v4_bigfive_pilot_controls.yaml
configs/experiments/v4_bigfive_format_smoke.yaml
configs/experiments/v4_bigfive_final.yaml
```

Important final config setting:

```yaml
steering:
  alpha: 4.0
  layers: [18, 21, 24]
  tune_on_this_split: false
```

Data:

```text
data/bfi/bfi44_scoring.csv
data/bfi/bfi44_item_text.local.csv
data/trait_bigfive/scenarios.jsonl
data/trait_bigfive/splits.json
data/trait_bigfive/contrastive_pairs/
```

Note:

- `data/bfi/bfi44_item_text.local.csv` is local-only and ignored by git.
- BFI wording was extracted locally from a UW-Madison ARC-linked BFI document.
- Keep BFI item text local unless redistribution rights are explicitly cleared.

Core v4 modules:

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
src/demo/bigfive_app.py
src/demo/bigfive_demo_data.py
src/demo/bigfive_demo_validation.py
```

Scripts:

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

Other important fixes:

- `requirements.txt` now includes `pyarrow` and `streamlit`.
- `src/models/loader.py` supports `model.layers` for `Qwen2Model` fallback.
- `src/projection/selector.py` handles CUDA/CPU device mismatch in cosine
  similarity.
- `src/generation/prompt_builder.py` now repeats strict response format in the
  user message, improving generated-response format validity.

## Data State

TRAIT raw parquet files are local:

```text
data/raw/trait/data/Agreeableness-00000-of-00001.parquet
data/raw/trait/data/Conscientiousness-00000-of-00001.parquet
data/raw/trait/data/Extraversion-00000-of-00001.parquet
data/raw/trait/data/Neuroticism-00000-of-00001.parquet
data/raw/trait/data/Openness-00000-of-00001.parquet
```

Converted normalized TRAIT data:

```text
data/trait_bigfive/scenarios.jsonl
data/trait_bigfive/splits.json
```

Split coverage:

```text
train: 3500
dev: 750
test: 750
```

Each Big Five trait has 1000 scenarios total and 150 test scenarios.

Contrastive pairs:

```text
data/trait_bigfive/contrastive_pairs/agreeableness.jsonl
data/trait_bigfive/contrastive_pairs/conscientiousness.jsonl
data/trait_bigfive/contrastive_pairs/extraversion.jsonl
data/trait_bigfive/contrastive_pairs/neuroticism.jsonl
data/trait_bigfive/contrastive_pairs/openness.jsonl
data/trait_bigfive/contrastive_pairs/all_traits.jsonl
```

Each trait has 2800 contrastive pairs.

## Vector State

Final full vector bundle:

```text
results/v4_bigfive/vectors/bigfive_trait_vectors_full_fp16.pt
```

It contains:

```text
agreeableness: [18, 21, 24]
conscientiousness: [18, 21, 24]
extraversion: [18, 21, 24]
neuroticism: [18, 21, 24]
openness: [18, 21, 24]
```

Phase 3 readiness was run against this bundle and all checks passed.

Validation command:

```powershell
.venv\Scripts\python.exe scripts\18_validate_bigfive_phase3.py `
  --config configs\experiments\v4_bigfive_final.yaml `
  --vectors_path results\v4_bigfive\vectors\bigfive_trait_vectors_full_fp16.pt
```

Expected status:

```text
OK phase3_final_config
OK phase3_alpha_layers
OK trait_scenarios
OK contrastive_pairs
OK vectors
```

## Run Outputs

Phase 2 main pilot, format-fixed:

```text
results/v4_bigfive/pilot_metrics_format_fixed/
```

Phase 2 control diagnostics:

```text
results/v4_bigfive/pilot_control_metrics/
```

Alpha sweep:

```text
results/v4_bigfive/alpha_sweep/alpha_sweep_summary.csv
```

Layer sweep:

```text
results/v4_bigfive/layer_sweep/layer_sweep_summary.csv
```

Phase 3 full test metrics:

```text
results/v4_bigfive/final_metrics/
```

Important final analysis files:

```text
results/v4_bigfive/final_metrics/condition_differences.csv
results/v4_bigfive/final_metrics/condition_effects_with_ci.csv
results/v4_bigfive/final_metrics/paired_bootstrap_ci.csv
results/v4_bigfive/final_metrics/final_report.md
```

`condition_effects_with_ci.csv` is the main paper-table source. It contains all
four non-baseline conditions compared against baseline:

```text
one_line_prompt
elaborate_prompt
as_pas_only
elaborate_prompt_as_pas
```

Final figures:

```text
results/v4_bigfive/final_figures/
```

Generated figure stems:

```text
01_as_pas_consistency_delta
02_as_pas_trait_target_delta
03_condition_consistency_heatmap
04_bfi_trait_scatter
05_pas_loglik_agreement
```

Each figure is available as PNG and SVG.

## Final Test Result Summary

The final run used the held-out test split with:

```text
5 traits x 2 directions x 5 conditions x 150 scenarios
```

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

- AS+PAS-only improves TRAIT target attainment for all traits and directions.
- Consistency improvement is conditional.
- High targets improve consistency for Agreeableness, Conscientiousness,
  Extraversion, and Openness.
- Low targets often improve target behavior but lower consistency because BFI
  self-report remains comparatively high.
- Neuroticism shows the reversed pattern: high worsens consistency, low improves
  consistency.

## Important Commands

Run full tests:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Run Phase 3 validation:

```powershell
.venv\Scripts\python.exe scripts\18_validate_bigfive_phase3.py `
  --config configs\experiments\v4_bigfive_final.yaml `
  --vectors_path results\v4_bigfive\vectors\bigfive_trait_vectors_full_fp16.pt
```

Analyze final metrics:

```powershell
.venv\Scripts\python.exe scripts\14_analyze_bigfive_results.py `
  --metrics_dir results\v4_bigfive\final_metrics `
  --reference_condition baseline `
  --bootstrap_samples 500
```

Generate final figures:

```powershell
.venv\Scripts\python.exe scripts\21_make_v4_figures.py `
  --metrics_dir results\v4_bigfive\final_metrics `
  --output_dir results\v4_bigfive\final_figures
```

Validate demo:

```powershell
.venv\Scripts\python.exe scripts\20_validate_bigfive_demo.py
```

Run demo:

```powershell
.venv\Scripts\streamlit.exe run src\demo\bigfive_app.py `
  --server.address 127.0.0.1 `
  --server.port 8501 `
  --server.headless true
```

## Long-Running Process Policy

The user explicitly prefers not to waste tokens polling long-running commands.

For long runs:

- Start the command.
- Report the session id and output path.
- Do not repeatedly poll.
- Only check status when the user asks.

Examples of long runs:

- Vector extraction.
- Full final metrics run.
- Full response generation with `--generate_responses`.

## Known Gaps / Next Work

Recommended next tasks:

1. Add explicit AS-only and PAS-only ablations if the paper needs to answer
   whether AS and PAS are individually necessary.
2. Polish the Streamlit demo layout if the user wants a more presentation-grade
   UI.
3. Add a small slide deck or markdown presentation outline using the final
   figures.
4. Optionally run final split side-effect response generation if final test
   response examples are needed.
5. Optionally run full test control-vector diagnostics if the paper needs
   test-split control tables beyond Phase 2 dev controls.
6. Consider adding progress logging to `scripts/12_run_bigfive_pilot.py` so long
   runs are easier to monitor without frequent external polling.

Avoid doing unless explicitly requested:

- Retuning alpha/layers after the test run.
- Changing prompt wording in a way that invalidates final test comparability.
- Replacing the BFI local overlay with committed item text.
- Deleting or reverting existing V2/V3 artifacts.

## Current Git/Workspace Note

The workspace is dirty and contains many pre-existing changes and generated
artifacts. Do not revert unrelated files. Treat existing changes as user-owned
unless clearly created in the current task.
