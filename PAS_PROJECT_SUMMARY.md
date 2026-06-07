# PAS Project Summary

Last updated: 2026-05-25

## Scope Note

This document summarizes the repository as implemented under `C:/GIT`. It focuses on implemented code, configuration, experiment documents, canonical data, and generated analysis artifacts. Binary papers, `.docx` files, cached outputs, and large generated result files were treated as reference artifacts rather than source modules.

Several requested midterm details describe the original AS/PAS design, while the current repository has already moved through v2 and v4 extensions. Where the current files differ from the requested framing, this document names the exact mismatch instead of hiding it.

## 1. Project Overview

The project investigates persona consistency in LLM-based NPCs. The core problem is that an NPC may be prompted to have a persona, but its selected action may not match that persona. The original persona axis is `aggressive` versus `cooperative`; later work extends the same idea to Big Five NPC personas.

The main technical intervention is Activation Steering (`AS`). The system extracts a persona direction vector, `v_persona`, from contrastive pairs and injects that direction into selected transformer layers during inference. A related Persona-Projected Action Selection (`PAS`) component embeds candidate actions and ranks them by cosine similarity against the persona vector.

The original research question is:

> Can Activation Steering shift an LLM-controlled NPC's action choices toward a target persona without relying only on explicit persona prompting?

The current v2 research framing is narrower and more defensible:

> Activation Steering changes forced-choice action-description preferences compared with neutral and prompt-only baselines, but prompt conditioning remains stronger in the current run.

The current v4 Big Five framing is:

> Under what conditions does AS+PAS affect speech-action persona consistency for NPCs assigned a Big Five persona?

Important claim boundaries:

- This project does not measure LLM morality.
- This project does not claim to measure the model's real personality.
- The current v2 pipeline measures closed-set action-description preference, not fully open-ended NPC behavior.
- The current v4 final run stores PAS as a diagnostic signal; PAS does not overwrite the primary `TRAIT_score`.

## 2. Architecture

### Backbone

The model backbone is HuggingFace Transformers, not Ollama.

Configured model:

```yaml
model:
  name: Qwen/Qwen2.5-3B-Instruct
  device: cuda
  dtype: float16
  quantization: null
```

Primary loader:

- `src/models/loader.py`
- Uses `AutoTokenizer.from_pretrained`
- Uses `AutoModelForCausalLM.from_pretrained`
- Falls back to `AutoModelForImageTextToText` and `AutoModel` for compatible nonstandard architectures
- Resolves transformer layers through `get_model_layers`

Supported layer paths:

- `model.model.layers`
- `model.layers`
- `model.model.language_model.layers`
- `model.language_model.model.layers`

Current Qwen2.5 config:

- `num_layers: 36`
- `hidden_dim: 2048`
- main steering layers: `[18, 21, 24]`
- hook target: `post_block_residual`

### Data Flow

```text
Team A CSVs
  -> scripts/00_convert_team_a_csvs.py
  -> data/scenarios/*.jsonl
  -> data/actions/*.jsonl
  -> data/contrastive_pairs/*.jsonl

contrastive pairs
  -> scripts/02_extract_vectors.py
  -> src/steering/extractor.py
  -> src/steering/vector.py
  -> results/v2/vectors/steering_vectors_separate_fp16.pt

scenarios + model + vectors
  -> scripts/03_run_experiment.py
  -> AS hooks through src/steering/injector.py
  -> loglik or generation selection
  -> results/v2/final_raw/*.jsonl

raw results
  -> scripts/04_evaluate.py
  -> src/evaluation/as_metrics.py
  -> analysis/cluster_bootstrap.py
  -> analysis/mixed_effects.py
  -> results/v2/final_analysis_v2/*.csv

analysis outputs
  -> scripts/07_final_report.py
  -> scripts/08_make_slide_assets.py
  -> report tables and slide assets
```

### `v_persona` Extraction

The AS vector extraction follows a CAA-style contrastive activation addition design.

1. Build contrastive pairs with matched format:
   - positive persona text
   - negative persona text
2. Run each side through the same HuggingFace model.
3. Capture the last input token hidden state at selected transformer layers.
4. Compute the persona vector:

```text
v_persona[layer] = mean(positive_activations[layer]) - mean(negative_activations[layer])
```

5. Normalize the vector when `normalize_vectors: true`.

Implemented by:

- `src/steering/extractor.py`
- `src/steering/vector.py`
- `scripts/02_extract_vectors.py`

Current vector modes include:

- `contrast_negated`: use one contrast direction and negate it for the opposite persona.
- `contrast_explicit`: store explicit aggressive/cooperative directions from the same contrast axis.
- `separate`: neutral-anchor vectors.
- `orthogonalized`: neutral-anchor vectors with a common component removed.
- `trait`: Big Five trait high/low contrast vectors.

Important v2 vector finding:

- Neutral-anchor aggressive and cooperative vectors had mean cosine `0.717225`.
- This suggests both captured a shared non-neutral component instead of clean opposite persona directions.
- The final v2 interpretation treats the explicit aggressive/cooperative contrast axis as the valid persona axis.

### Hidden State Injection

The general hook formula is:

```text
h' = h + alpha * beta * v_p
```

In the codebase:

- `src/models/hooks.py` supports `alpha * beta * steering_vector`, where `beta_fn` can provide per-position scaling.
- `src/steering/injector.py` implements the current uniform AS path:

```text
h' = h + alpha * v
```

That means `beta = 1.0` in the current `SteeringInjector` path.

The injection target is the first tensor returned by each selected transformer block, corresponding to the post-block residual hidden state. Hooks are registered before scoring/generation and removed immediately after the run.

### Prompting and Generation

`src/generation/prompt_builder.py` creates prompts for:

- neutral NPC behavior
- persona-prompted NPC behavior
- strict `[Speech]` plus `<Action>` generation
- action-choice log-likelihood scoring
- Big Five one-line and elaborate persona prompts

The legacy generation contract is:

```text
[Speech] short dialogue
<Action>action_id</Action>
```

The v2 primary measurement no longer depends on free-form parsing. It scores action descriptions directly by conditional log-likelihood.

### PAS: Persona-Projected Action Selection

PAS embeds each candidate action description using the model's hidden state at a selected layer, then ranks actions by cosine similarity with `v_persona`.

Implemented by:

- `src/projection/embedder.py`
- `src/projection/selector.py`
- `src/experiments/v4_bigfive_runner.py`

Scoring:

```text
score(action_i) = cosine_similarity(v_persona, action_embedding_i)
```

Selection:

```text
selected_action = argmax_i score(action_i)
```

The `ProjectedActionResolver` can keep a generated action if its cosine score exceeds a threshold, otherwise it falls back to the top PAS-ranked action. In the current v4 final analysis, PAS is reported through `pas_loglik_agreement.csv` and `pas_agreement_summary.csv`; it is not the primary scorer for `TRAIT_score`.

## 3. Experiment Design

### Original Midterm AS Design

The original AS-only design in `AS_IMPLEMENTATION_DESIGN.md` and the docstring of `scripts/03_run_experiment.py` uses three conditions:

| Condition | Meaning |
|---|---|
| `neutral_baseline` | neutral prompt, no steering |
| `prompt_baseline` | explicit persona prompt, no steering |
| `as_only` | neutral prompt plus Activation Steering |

The requested midterm structure is:

```text
10 scenarios x 5 actions
```

The repository's original milestone text also mentions:

```text
10 scenarios x 2 personas x 3 conditions x 30 repeats
```

Each scenario has five actions:

- 2 aggressive actions
- 1 neutral action
- 2 cooperative actions

Therefore, the chance action-level alignment rate for `aggressive` or `cooperative` is `0.40`.

### Current v2 Experiment Design

The current `configs/experiment.yaml` has already expanded beyond the original three-condition design:

| Condition | Meaning |
|---|---|
| `neutral_baseline` | no persona prompt, no steering |
| `prompt_baseline` | persona prompt only |
| `as_only_neg` | AS only, contrast vector with negated opposite direction |
| `as_only_sep` | corrected separate-vector ablation; effectively the same contrast direction in the final interpretation |
| `prompt_as` | persona prompt plus AS |

Current config:

```yaml
personas: ["aggressive", "cooperative"]
num_scenarios: 30
num_repeats: 1
total_runs_per_condition: 60
```

Current final v2 run:

- model: `Qwen/Qwen2.5-3B-Instruct`
- selection mode: `loglik`
- scenarios: 30 Team A scenarios
- actions per scenario: 5
- alpha: `4.0`
- layers: `[18, 21, 24]`
- repeats: 1, because log-likelihood scoring is deterministic

### Current v4 Big Five Extension

The repository also contains a later Big Five line under `configs/experiments/v4_bigfive_*.yaml`, `data/trait_bigfive/`, `src/data/`, `src/experiments/v4_bigfive_runner.py`, and `results/v4_bigfive/`.

The v4 final conditions are:

- `baseline`
- `one_line_prompt`
- `elaborate_prompt`
- `as_pas_only`
- `elaborate_prompt_as_pas`

This extension measures BFI self-report versus TRAIT action-choice consistency. It is useful context, but it is separate from the original midterm aggressive/cooperative AS design.

### Metrics

Original/midterm metrics:

- `persona_alignment`: whether selected action's `persona_alignment` matches the target persona.
- axis score `(-2 to +2)`: requested in the prompt, but [UNCLEAR] as a current implemented metric name. The repository primarily implements alignment categories and rates, not a persistent `axis_score` column.
- action selection distribution: counts/proportions of selected actions per condition/persona/scenario.

Implemented v2 metrics:

- `parse_ok_rate`
- `unknown_rate`
- `persona_alignment_rate`
- `neutral_action_rate`
- `misaligned_action_rate`
- `action_entropy`
- `unique_actions`
- ITT and per-protocol alignment rates
- Wilson confidence intervals
- odds ratios versus `neutral_baseline`
- scenario-cluster bootstrap confidence intervals
- mixed-effects logistic regression:

```text
aligned ~ condition * persona + (1 | scenario_id)
```

Implemented v4 metrics:

- `speech_action_consistency = 1 - abs(BFI_score - TRAIT_score)`
- `target_attainment = target_direction_score - 0.5`
- `pas_loglik_agreement`
- side-effect metrics such as response length and format validity
- paraphrase robustness summaries

### Current Results Snapshot

Final v2 alignment rates:

| Condition | aggressive | cooperative |
|---|---:|---:|
| `neutral_baseline` | 0.433 | 0.367 |
| `prompt_baseline` | 0.900 | 0.767 |
| `as_only_neg` | 0.600 | 0.700 |
| `as_only_sep` | 0.600 | 0.700 |
| `prompt_as` | 0.800 | 0.733 |

Interpretation:

- v2 fixed the parser confound: `parse_ok_rate = 1.000`, `unknown_rate = 0.000`.
- AS-only improves over neutral.
- Prompt-only is stronger than AS-only.
- `prompt_as` does not improve over `prompt_baseline` in the current final v2 run.
- Current v2 results should be presented as forced-choice action-description preference shifts, not as definitive open-ended behavior change.

## 4. Module Inventory

### Core Source Modules

| File | Role |
|---|---|
| `src/__init__.py` | Marks `src` as the main Python package. |
| `src/models/__init__.py` | Marks the model-loading package. |
| `src/models/loader.py` | Loads HuggingFace model/tokenizer and resolves transformer layer paths. |
| `src/models/hooks.py` | Provides reusable forward hook management and the general `alpha * beta * steering_vector` hook. |
| `src/steering/__init__.py` | Marks the steering package. |
| `src/steering/extractor.py` | Extracts last-input-token hidden states from contrastive pairs. |
| `src/steering/vector.py` | Computes, validates, saves, loads, and compares CAA steering vectors. |
| `src/steering/injector.py` | Registers uniform post-block AS hooks implementing `h' = h + alpha * v`. |
| `src/generation/__init__.py` | Marks the generation package. |
| `src/generation/prompt_builder.py` | Builds neutral, persona, action-choice, contrastive, and Big Five prompts. |
| `src/generation/generator.py` | Wraps HuggingFace text generation with optional steering hooks. |
| `src/generation/parser.py` | Parses `[Speech]` and `<Action>` tags from generated NPC output. |
| `src/projection/__init__.py` | Marks the PAS/projection package. |
| `src/projection/embedder.py` | Embeds action descriptions as hidden states from a selected model layer. |
| `src/projection/selector.py` | Implements cosine-similarity PAS scoring and projected action resolution. |
| `src/evaluation/__init__.py` | Marks the evaluation package. |
| `src/evaluation/as_metrics.py` | Computes legacy aggressive/cooperative AS metrics and statistics. |
| `src/evaluation/loglik_selector.py` | Scores candidate actions by conditional log-likelihood and selects the best action. |
| `src/evaluation/agreement.py` | Provides speech/action agreement utilities for action definitions. |
| `src/evaluation/distribution.py` | Computes action distribution and persona-alignment distribution summaries. |
| `src/evaluation/layer_analysis.py` | Summarizes layer/beta strategy sweeps. |
| `src/evaluation/quantization_compare.py` | Compares fp16 and quantized steering vectors. |
| `src/evaluation/bigfive_metrics.py` | Defines Big Five consistency, target-attainment, probability-mass, and PAS agreement metrics. |
| `src/evaluation/bigfive_analysis.py` | Aggregates v4 CSVs, computes deltas/CIs, and builds Markdown reports. |
| `src/evaluation/side_effect_metrics.py` | Measures response length, format validity, and parser-related side effects. |
| `src/evaluation/paraphrase_robustness.py` | Summarizes robustness across paraphrased action descriptions. |
| `src/controls/__init__.py` | Marks the vector-control package. |
| `src/controls/vector_controls.py` | Builds zero, random, unrelated-trait, signed, filtered, and trait contrast vector controls. |
| `src/data/__init__.py` | Marks the data package. |
| `src/data/bfi.py` | Loads and scores BFI-44 metadata and optional local item text. |
| `src/data/trait_convert.py` | Converts raw TRAIT Big Five rows into the normalized repository schema. |
| `src/data/trait_dataset.py` | Loads normalized TRAIT scenarios and builds trait contrastive pairs. |
| `src/experiments/__init__.py` | Marks the experiment package. |
| `src/experiments/manifest.py` | Stores run provenance such as config hashes, vector hashes, model, and commit. |
| `src/experiments/v4_bigfive_plan.py` | Builds and validates v4 run plans from config/data. |
| `src/experiments/v4_bigfive_readiness.py` | Checks BFI, TRAIT, contrastive-pair, vector, and config readiness. |
| `src/experiments/v4_bigfive_phase2_validation.py` | Validates Phase 2 pilot/control readiness. |
| `src/experiments/v4_bigfive_phase3_validation.py` | Validates Phase 3 final-run readiness and tuning constraints. |
| `src/experiments/v4_bigfive_runner.py` | Runs v4 BFI scoring, TRAIT loglik scoring, AS generation, and PAS diagnostics. |
| `src/demo/__init__.py` | Marks the demo package. |
| `src/demo/bigfive_app.py` | Streamlit UI for the Big Five demo and final figures. |
| `src/demo/bigfive_demo_data.py` | Loads or synthesizes demo scenarios/responses and activation movement points. |
| `src/demo/bigfive_demo_validation.py` | Validates demo paths, fallback data, Streamlit dependency, and figure availability. |
| `src/demo/live_inference.py` | Loads live model/vector state and compares prompt-only against AS/PAS generation. |
| `src/demo/live_inference_tab.py` | Streamlit tab for live prompt-only versus AS/PAS inference. |
| `src/utils/__init__.py` | Marks utility package. |
| `src/utils/config.py` | Loads YAML config files and resolves project root paths. |
| `src/utils/seed.py` | Sets Python, NumPy, and Torch random seeds. |

### Analysis Modules

| File | Role |
|---|---|
| `analysis/__init__.py` | Marks analysis helpers as a package. |
| `analysis/cluster_bootstrap.py` | Computes scenario-cluster bootstrap intervals for alignment rates. |
| `analysis/mixed_effects.py` | Fits mixed-effects logistic regression for alignment outcomes. |

### Scripts

| File | Role |
|---|---|
| `scripts/00_convert_team_a_csvs.py` | Converts Team A CSVs into canonical scenarios, action definitions, and contrastive pairs. |
| `scripts/01_generate_pairs.py` | Generates legacy contrastive pair files. |
| `scripts/02_extract_vectors.py` | Extracts AS vectors in single, separate, or Big Five trait modes. |
| `scripts/03_run_experiment.py` | Runs v2 AS experiments in `generate` or `loglik` selection mode. |
| `scripts/04_evaluate.py` | Evaluates v2 raw JSONL outputs into CSV metrics and figures. |
| `scripts/05_visualize.py` | Produces visualization outputs for legacy/v2 metrics and vector comparisons. |
| `scripts/06_run_sweeps.py` | Runs alpha/layer sweeps for v2 steering settings. |
| `scripts/07_final_report.py` | Builds the v2 final report from metrics and diagnostics. |
| `scripts/08_make_slide_assets.py` | Creates slide-oriented figures/assets for the v2 presentation. |
| `scripts/09_build_trait_contrastive_pairs.py` | Builds Big Five trait contrastive pairs from normalized TRAIT data. |
| `scripts/10_prepare_bigfive_pilot.py` | Prepares v4 Big Five pilot inputs and planning artifacts. |
| `scripts/11_convert_trait_bigfive.py` | Converts raw TRAIT parquet/JSONL data into normalized Big Five scenarios. |
| `scripts/12_run_bigfive_pilot.py` | Runs the v4 Big Five pilot/final scoring pipeline. |
| `scripts/13_check_bigfive_readiness.py` | CLI wrapper for Big Five readiness checks. |
| `scripts/14_analyze_bigfive_results.py` | Analyzes Big Five result CSVs and writes reports. |
| `scripts/15_score_side_effects.py` | Scores side-effect metrics for generated responses. |
| `scripts/16_analyze_paraphrase_robustness.py` | Analyzes paraphrase robustness outputs. |
| `scripts/17_validate_bigfive_phase2.py` | CLI wrapper for Phase 2 validation. |
| `scripts/18_validate_bigfive_phase3.py` | CLI wrapper for Phase 3 validation. |
| `scripts/19_run_bigfive_demo.py` | Launches the Big Five Streamlit demo. |
| `scripts/20_validate_bigfive_demo.py` | Validates demo readiness from the command line. |
| `scripts/21_make_v4_figures.py` | Generates final v4 figures as PNG/SVG. |

### Config Files

| File | Role |
|---|---|
| `configs/model.yaml` | HuggingFace model, dtype, device, quantization, and architecture metadata. |
| `configs/steering.yaml` | AS vector paths, layer groups, alpha values, hook target, and contrastive-pair paths. |
| `configs/experiment.yaml` | v2 condition matrix, personas, scenario count, repeats, generation settings, and output paths. |
| `configs/experiments/v3_controls_dev.yaml` | v3 control-development experiment config. |
| `configs/experiments/v3_final_test.yaml` | v3 final-test experiment config. |
| `configs/experiments/v3_measurement_validation.yaml` | v3 measurement-validation config. |
| `configs/experiments/v4_bigfive_controls.yaml` | v4 vector-control diagnostics config. |
| `configs/experiments/v4_bigfive_pilot.yaml` | v4 Big Five pilot config. |
| `configs/experiments/v4_bigfive_pilot_controls.yaml` | v4 pilot control-vector config. |
| `configs/experiments/v4_bigfive_format_smoke.yaml` | v4 format smoke-test config. |
| `configs/experiments/v4_bigfive_final.yaml` | v4 final held-out test config. |

### Tests

| File | Role |
|---|---|
| `tests/__init__.py` | Marks the test package. |
| `tests/test_analysis_v2.py` | Tests v2 analysis helpers and output contracts. |
| `tests/test_as_metrics.py` | Tests aggressive/cooperative AS metrics. |
| `tests/test_bfi.py` | Tests BFI loading, scoring, reverse coding, and validation. |
| `tests/test_bigfive_analysis.py` | Tests v4 Big Five analysis report and delta calculations. |
| `tests/test_bigfive_demo_data.py` | Tests demo data loading and fallback behavior. |
| `tests/test_bigfive_demo_validation.py` | Tests demo readiness checks. |
| `tests/test_bigfive_metrics.py` | Tests Big Five metric formulas. |
| `tests/test_bigfive_pilot_script.py` | Tests the Big Five pilot runner script interface. |
| `tests/test_bigfive_prompt_builder.py` | Tests Big Five prompt styles and validation. |
| `tests/test_extractor.py` | Tests activation extraction behavior. |
| `tests/test_injector.py` | Tests AS injection hook behavior. |
| `tests/test_live_inference.py` | Tests live inference helper logic. |
| `tests/test_loglik_selector.py` | Tests conditional log-likelihood action selection. |
| `tests/test_manifest.py` | Tests run manifest creation and hashing. |
| `tests/test_model_loader.py` | Tests model layer resolution paths. |
| `tests/test_paraphrase_robustness.py` | Tests paraphrase robustness summaries. |
| `tests/test_parser.py` | Tests `[Speech]` and `<Action>` parsing. |
| `tests/test_selector.py` | Tests PAS cosine selector and resolver behavior. |
| `tests/test_side_effect_metrics.py` | Tests response format/length side-effect metrics. |
| `tests/test_side_effect_script.py` | Tests side-effect scoring script behavior. |
| `tests/test_steering_vector.py` | Tests CAA vector computation, normalization, validation, and persistence. |
| `tests/test_trait_convert.py` | Tests raw TRAIT conversion. |
| `tests/test_trait_dataset.py` | Tests normalized TRAIT scenario parsing and flattening. |
| `tests/test_trait_pair_script.py` | Tests trait contrastive-pair generation script. |
| `tests/test_v4_bigfive_phase2_validation.py` | Tests Phase 2 validation logic. |
| `tests/test_v4_bigfive_phase3_validation.py` | Tests Phase 3 validation logic. |
| `tests/test_v4_bigfive_plan.py` | Tests v4 run-plan construction and validation. |
| `tests/test_v4_bigfive_readiness.py` | Tests v4 readiness checker. |
| `tests/test_v4_bigfive_runner.py` | Tests v4 runner scoring and vector resolution helpers. |
| `tests/test_vector_controls.py` | Tests vector-control helper behavior. |
| `tests/test_vector_modes.py` | Tests v2 vector-mode resolution behavior. |

### Documentation Files

| File | Role |
|---|---|
| `AS_IMPLEMENTATION_DESIGN.md` | Original AS-first architecture and three-condition experiment design note; text encoding is partially corrupted in the file. |
| `docs/llm_project_brief.md` | v2 design-review brief and high-level architecture summary. |
| `docs/progress_summary_for_llm.md` | Progress summary for another LLM or collaborator. |
| `docs/slide_visualization_redesign_brief.md` | Visual redesign brief for presentation assets. |
| `docs/team_b_llm_prompt.md` | Prompt for Team B analysis/reporting assistance. |
| `docs/team_b_progress_handoff.md` | Team B evaluation/reporting handoff. |
| `docs/v3_experiment_design.md` | v3 design plan for stronger controls and measurement validation. |
| `docs/v4_bigfive_trait_design.md` | Full v4 Big Five research and system design. |
| `docs/v4_bigfive_runbook.md` | Operational runbook for v4 Big Five experiments. |
| `docs/v4_bigfive_phase2_summary.md` | Phase 2 pilot/control summary. |
| `docs/v4_bigfive_phase4_final_analysis.md` | Phase 4 final v4 analysis interpretation. |
| `docs/v4_bigfive_llm_handoff.md` | v4 handoff for another LLM/collaborator. |
| `docs/v4_bigfive_change_summary.md` | Summary of v4 changes and current caveats. |
| `docs/v4_bigfive_live_inference_plan.md` | Plan for live AS/PAS inference in the demo; owner listed as Jin. |
| `docs/v4_bigfive_module_map.md` | Module dependency map for v4; text has encoding corruption in responsibility prose. |

### Canonical Data and Artifact Directories

| Path | Role |
|---|---|
| `data/scenarios/scenarios.jsonl` | Default aggressive/cooperative scenario set with five labeled actions per scenario. |
| `data/scenarios/team_a_scenarios.jsonl` | Converted Team A 30-scenario set used in v2 final runs. |
| `data/actions/action_definitions.jsonl` | Default flattened action definitions. |
| `data/actions/team_a_action_definitions.jsonl` | Team A action definitions used in v2 final runs. |
| `data/contrastive_pairs/*.jsonl` | Aggressive/cooperative and neutral-anchor contrastive pair sets. |
| `data/trait_bigfive/` | Normalized v4 Big Five TRAIT scenarios, splits, and contrastive pairs. |
| `data/bfi/` | BFI scoring metadata and local-only item text template. |
| `results/v1_legacy/` | Legacy parser-based outputs and metrics. |
| `results/v2/` | v2 vectors, raw outputs, final metrics, sweeps, figures, and slide assets. |
| `results/v4_bigfive/` | v4 Big Five vectors, metrics, controls, sweeps, figures, and reports. |

## 5. Team Role Breakdown

The repository gives partial team ownership information.

| Role / Member | Ownership |
|---|---|
| Team A | Source scenario/action CSVs and the 30 Team A scenario/action dataset. Exact individual ownership is [UNCLEAR]. |
| Team B | Evaluation, analysis, result interpretation, report writing, presentation material preparation. This is explicit in `docs/team_b_progress_handoff.md`. |
| Jin | Owner of the v4 live inference plan in `docs/v4_bigfive_live_inference_plan.md`. |
| Vector extraction owner | [UNCLEAR]. Implemented primarily in `scripts/02_extract_vectors.py`, `src/steering/extractor.py`, and `src/steering/vector.py`. |
| Dataset conversion owner | [UNCLEAR]. Implemented primarily in `scripts/00_convert_team_a_csvs.py`, `scripts/11_convert_trait_bigfive.py`, and `src/data/*`. |
| Evaluation framework owner | [UNCLEAR]. Team B receives and interprets outputs, while implementation lives in `src/evaluation/*`, `analysis/*`, and `scripts/04_evaluate.py`. |
| Demo owner | [UNCLEAR] beyond Jin owning the live inference plan. Implemented in `src/demo/*` and `scripts/19_run_bigfive_demo.py`. |
| Midterm presentation owner | [UNCLEAR]. Existing handoff says Team B should prepare clean tables, result interpretation, failure cases, report text, and slide-ready figures. |

## 6. Current Status and Next Steps

### Done

- HuggingFace model loading is implemented.
- AS vector extraction from contrastive pairs is implemented.
- Post-block residual steering injection is implemented.
- PAS cosine action ranking is implemented.
- Legacy free-form generation and parsing path is implemented.
- v2 closed-set log-likelihood action selection is implemented.
- Team A CSV conversion into canonical JSONL is implemented.
- v2 final run exists for 30 Team A scenarios.
- v2 evaluation outputs exist, including quality metrics, alignment metrics, action distribution, cluster bootstrap, mixed effects, and final report.
- v2 slide assets exist.
- v4 Big Five extension exists with BFI/TRAIT data pipeline, AS/PAS conditions, control diagnostics, demo, final figures, and final analysis.
- Tests exist across steering, PAS, generation parsing, loglik scoring, v2/v4 analysis, data conversion, manifests, vector modes, and demos.

### Pending

- Decide whether the midterm presentation should focus on the original three-condition AS design or the current five-condition v2 result.
- If the presentation must use exactly `neutral_baseline / prompt_baseline / as_only`, map current `as_only_neg` or `as_only_sep` to the midterm `as_only` label and disclose the exact mapping.
- Reintroduce or define the requested axis score `(-2 to +2)` if it is required for slides; current code does not expose a clear persistent `axis_score` metric.
- Add random-vector, zero-hook, wrong-vector, and shuffled-label controls to strengthen persona-specificity claims in the aggressive/cooperative setting.
- Add held-out splits for v2 if stronger generalization claims are needed.
- Add paraphrase-controlled action descriptions for v2 to test wording sensitivity.
- Decide whether PAS should remain diagnostic or become the actual policy selection mechanism in future runs.

### Immediate Next Actions for Midterm Presentation

1. Use the v2 result as the main technical progress story:
   - parser failure was fixed by log-likelihood action scoring;
   - AS-only improves over neutral;
   - prompt-only remains stronger;
   - prompt+AS does not beat prompt-only in the current setting.
2. Show the pipeline:
   - contrastive pairs -> `v_persona` -> hidden-state injection -> action scoring -> `persona_alignment`.
3. Use one table with alignment rates and a chance baseline line at `0.40`.
4. Include the vector-design finding:
   - neutral-anchor vectors had cosine `0.717225`, so the final vector axis uses the explicit aggressive/cooperative contrast.
5. State limitations clearly:
   - closed-set action-description preference is not open-ended behavior;
   - no full negative-control matrix yet;
   - prompt ceiling likely explains why `prompt_as` does not improve over `prompt_baseline`.

## 7. Known Issues and Open Questions

### Known Issues

- `AS_IMPLEMENTATION_DESIGN.md` contains partially corrupted Korean text encoding, although enough technical content remains readable.
- Some scenario/action JSONL content also appears text-encoding corrupted in the terminal output.
- The original three-condition design and the current v2 config do not match exactly.
- The requested `10 scenarios x 5 actions` design does not match the current v2 `num_scenarios: 30` config.
- The requested axis score `(-2 to +2)` is [UNCLEAR] as an implemented persistent metric.
- `as_only_neg` and `as_only_sep` should not be interpreted as independent mechanisms after the vector-design correction.
- Legacy `speech_action_agreement` used an English keyword heuristic and should not be treated as a valid primary metric for Korean/free-form outputs.
- v2 log-likelihood scoring removes parse failure but changes the measurement target to closed-set action-description preference.
- v4 final results report PAS/loglik agreement diagnostically; PAS does not currently overwrite primary `TRAIT_score`.

### Open Questions

- Should the midterm use the simpler original `as_only` condition name, or the current explicit `as_only_neg` / `as_only_sep` naming?
- Should PAS be promoted from diagnostic ranking to final action policy selection?
- Should future AS injection use a real `beta_fn` schedule, or keep uniform `beta = 1.0`?
- How should `axis score (-2 to +2)` be defined from five actions: by label order, manually assigned action score, or persona-direction probability mass?
- Are the five action descriptions balanced for length, emotional intensity, and persona cue strength?
- Are Team A labels externally validated, or only internally assigned?
- Which member owns the final midterm deck and final paper text? [UNCLEAR]
- Should the next experiment prioritize v2 controls or the v4 Big Five speech-action consistency line?

## Recommended Midterm Claim

Use:

> We implemented a HuggingFace-based Activation Steering pipeline for LLM NPC persona control. The system extracts `v_persona` from contrastive pairs, injects it into selected transformer layers, and evaluates whether selected actions match the target `persona_alignment`. In the current v2 closed-set evaluation, AS-only improves over the neutral baseline, while prompt-only remains stronger; therefore the result supports AS as a measurable steering signal but not yet as a complete replacement for prompting.

Avoid:

> AS always makes NPCs persona-consistent.

Avoid:

> PAS is the primary final decision policy in the current v4 final run.

Avoid:

> The experiment proves real model personality change.
