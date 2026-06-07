# LLM Project Brief: Activation Steering Experiment Design Review

## Copy-Paste Prompt for Another LLM

You are being asked to improve the experimental design and code architecture of an activation steering evaluation pipeline. Do not merely summarize the current results. Treat this as a design-review and redesign task.

The current project measures whether steering internal activations of `Qwen/Qwen2.5-3B-Instruct` changes NPC action selection toward a target persona, specifically `aggressive` or `cooperative`. The user suspects that the overall experiment structure may still be improved. Your job is to identify better experimental designs, cleaner controls, stronger measurements, and code architecture changes that would make the results more trustworthy.

Please analyze the experimental design, methodology, statistics, vector construction, action-selection measurement, and code architecture. Focus on:

1. What the current design is actually measuring.
2. Whether the experimental structure matches the research goal.
3. Which parts of the design could be confounded or underpowered.
4. How to redesign the pipeline to make the causal claims stronger.
5. How to reorganize the code so future experiments are easier to run, compare, and audit.
6. What alternative experimental structures should be considered.

Key context:

- The original pipeline used free-form generation and parsed the generated action. This had high parse failure and unknown rates, so alignment percentages were not reliable.
- The v2 pipeline now uses description-based conditional log-likelihood scoring over exactly five candidate actions per scenario. This removes parse failure from the main evaluation.
- The experiment compares conditions:
  - `neutral_baseline`: no persona prompt, no steering.
  - `prompt_baseline`: persona prompt only.
  - `as_only_neg`: activation steering only, using a contrast vector and its negative direction.
  - `as_only_sep`: activation steering only, now explicitly using the same contrast direction as a corrected separate-vector ablation.
  - `prompt_as`: persona prompt plus activation steering.
- Personas are `aggressive` and `cooperative`.
- Final run uses 30 Team A scenarios, deterministic loglik scoring, 1 repeat per scenario/persona/condition, alpha `4.0`, and middle layers `[18, 21, 24]`.
- Statistical analysis uses scenario-cluster bootstrap and mixed-effects logistic regression:
  - `aligned ~ condition * persona + (1 | scenario_id)`
- Final result summary:
  - parse_ok is 100 percent and unknown is 0 percent in the v2 final run.
  - Steering-only improves over neutral, especially for cooperative alignment.
  - Prompt-only is stronger than activation steering.
  - Prompt plus activation steering does not improve over prompt-only.
  - Neutral-anchor separate vector extraction produced aggressive/cooperative vectors with cosine `0.717`, suggesting both captured a shared non-neutral direction. Orthogonalized vectors are almost identical to the contrast direction, so the final analysis treats the contrast vector as the valid persona axis.

Please review this with the following questions in mind:

1. Is description log-likelihood scoring an appropriate replacement for free-form action generation in this setting?
2. Does the experiment still measure action selection behavior, or only model preference among descriptions?
3. Are the condition comparisons fair, especially `prompt_baseline` vs `prompt_as`?
4. Is the vector design conclusion valid: neutral-anchor separate vectors mostly capture non-neutral-vs-neutral rather than aggressive-vs-cooperative?
5. Is the mixed-effects/bootstrap analysis sufficient for 30 scenario clusters?
6. What additional controls would make the claims stronger?
7. What should be stated cautiously in a final report?
8. If you were redesigning this experiment from scratch, what would the improved design look like?
9. What should be changed in the code architecture to support that improved design?

Expected output:

- A concise diagnosis of the current design.
- A proposed improved experimental design.
- A revised condition matrix.
- A revised metric/statistics plan.
- A revised code/data architecture plan.
- A prioritized implementation roadmap.

## Why This Document Exists

This document is meant to help another LLM reason about how to improve the whole experiment, not just interpret the current output. The current v2 pipeline is already a substantial improvement over the legacy parser-based setup, but it may still have design weaknesses:

- It may measure forced-choice action-description preference rather than open-ended NPC behavior.
- The action descriptions themselves may introduce wording, length, or style bias.
- Alpha/layer choices were selected from a limited sweep.
- Prompt-only is strong, so prompt+AS may suffer from saturation or signal conflict.
- Vector extraction may capture dimensions other than persona.
- Scenario count is only 30 clusters.
- The code supports the current workflow but could be more modular for future ablations.

The reviewing LLM should use this document to propose a better experimental design and a cleaner architecture for future iterations.

## Project Goal

The goal is to evaluate whether activation steering can shift an LLM-controlled NPC's chosen action toward a target persona. Each scenario has five candidate actions. Actions are labeled as aggressive, cooperative, or neutral. For each persona, alignment means selecting an action whose label matches the target persona.

The central research question is:

> Can activation steering alone, or activation steering combined with persona prompting, reliably increase persona-aligned NPC action choices compared with neutral and prompt-only baselines?

The v2 pipeline also answers a measurement question:

> Can a log-likelihood action selector remove parsing artifacts from free-form generation and produce more trustworthy alignment metrics?

The next design question is:

> What experiment structure would most convincingly test whether activation steering changes persona-relevant decision behavior, rather than merely changing scores over a fixed set of action descriptions?

This is the main reason for giving this brief to another LLM.

## Desired Design Improvements To Consider

The reviewing LLM should explicitly consider improvements along these axes:

1. Measurement target:
   - Separate closed-set action preference from open-ended generation behavior.
   - Decide whether the primary endpoint should be forced-choice, generation, ranking, or multi-turn behavior.

2. Condition design:
   - Ensure fair comparison between prompt-only, AS-only, prompt+AS, random-vector, wrong-vector, and no-op hook controls.
   - Consider whether prompt+AS should use weaker prompts to avoid saturation.

3. Scenario design:
   - Increase scenario clusters or introduce held-out scenario splits.
   - Balance scenario difficulty and baseline action preference.
   - Test whether some scenarios are already biased toward one persona.

4. Action design:
   - Control action description length, specificity, and affective intensity.
   - Add paraphrase sets per action to estimate wording sensitivity.
   - Consider ranking all actions rather than only argmax selection.

5. Vector design:
   - Audit contrastive pair quality.
   - Compare contrast, neutral-anchor, orthogonalized, random, shuffled-label, and unrelated-trait vectors.
   - Evaluate vector effects on both target behavior and off-target language/style.

6. Statistical design:
   - Use scenario-cluster bootstrap and mixed-effects models.
   - Add held-out validation for alpha/layer selection.
   - Report uncertainty for condition differences, not only condition rates.

7. Code architecture:
   - Separate data preparation, vector extraction, scoring, experiment orchestration, metrics, and reporting.
   - Make experiment manifests explicit and immutable.
   - Store every run with config snapshots, git commit, vector hash, model name, and random seeds.
   - Make ablations easy to define through config rather than script edits.

## Why The Pipeline Was Reworked

The legacy pipeline generated free-form model text and parsed the selected action. This caused several methodological problems:

- `parse_ok` was only around 42-54 percent in some runs.
- `unknown` rates varied by condition/persona, sometimes around 27-52 percent.
- Unknowns made denominators non-comparable across conditions.
- Two-proportion z-tests treated all rows as independent, even though rows were clustered by scenario.
- Cooperative steering was assumed to be `-v_aggressive` without validating whether this direction was meaningful.
- Alignment rates were compared without explicitly showing the 0.40 chance baseline for aggressive/cooperative action labels.
- The old speech metric used English keyword classification on Korean output and was invalid.

The v2 system was designed to remove or reduce those problems.

## High-Level Architecture

The pipeline has seven conceptual stages:

1. Convert Team A data into canonical scenarios, action definitions, and contrastive pairs.
2. Extract activation steering vectors from contrastive text pairs.
3. Run experiments under multiple conditions.
4. Select actions by log-likelihood over action descriptions.
5. Evaluate action alignment with quality metrics, ITT/PP metrics, bootstrap intervals, and mixed-effects modeling.
6. Sweep steering hyperparameters.
7. Produce final reports and figures.

Data flow:

```text
Team A CSVs
  -> scripts/00_convert_team_a_csvs.py
  -> data/scenarios/*.jsonl
  -> data/actions/*.jsonl
  -> data/contrastive_pairs/*.jsonl

contrastive pairs
  -> scripts/02_extract_vectors.py
  -> results/v2/vectors/steering_vectors_separate_fp16.pt

scenarios + actions + vectors
  -> scripts/03_run_experiment.py
  -> results/v2/final_raw/*.jsonl

raw outputs
  -> scripts/04_evaluate.py
  -> results/v2/final_analysis_v2/*.csv
  -> results/v2/final_figures_v2/*.png

analysis outputs
  -> scripts/07_final_report.py
  -> results/v2/final_analysis_v2/final_report.md
```

## Main Code Structure

### Configuration

- `configs/model.yaml`
  - Model loading configuration.
- `configs/steering.yaml`
  - Model name, hook target, alpha values, layer groups, vector paths, and extraction paths.
  - Current default alpha is `4.0`.
  - Current main layer group is `middle = [18, 21, 24]`.
- `configs/experiment.yaml`
  - Final condition list.
  - Personas.
  - Number of scenarios/repeats.
  - Output directories.

### Data Preparation

- `scripts/00_convert_team_a_csvs.py`
  - Converts source scenario/action CSV files into canonical JSONL data.
  - Creates action definitions.
  - Creates contrastive pairs.
  - The v2 default creates aggressive/cooperative contrastive pairs from action combinations, giving 120 contrastive pairs from 30 scenarios.
- `data/scenarios/team_a_scenarios.jsonl`
  - Final 30-scenario set.
- `data/actions/team_a_action_definitions.jsonl`
  - Candidate actions and labels.
- `data/contrastive_pairs/*.jsonl`
  - Pair sets used for vector extraction.

### Model Loading and Hooks

- `src/models/loader.py`
  - Loads tokenizer/model.
- `src/models/hooks.py`
  - Hook utilities.
- `src/steering/injector.py`
  - Applies steering vectors during forward passes.
- `src/steering/extractor.py`
  - Extracts hidden-state differences from contrastive examples.
- `src/steering/vector.py`
  - Steering vector data structures/utilities.

### Prompting and Generation

- `src/generation/prompt_builder.py`
  - Builds scenario prompts.
  - Supports persona prompt conditions and description-based action scoring format.
- `src/generation/generator.py`
  - Free-form generation path, retained for comparison.
- `src/generation/parser.py`
  - Legacy parser for generated text.

### Log-Likelihood Action Selection

- `src/evaluation/loglik_selector.py`
  - Core v2 action selector.
  - Computes `log p(action_text | prompt)` for each candidate action.
  - Supports length normalization.
  - Returns selected index, raw log-likelihoods, normalized log-likelihoods, and softmax probabilities.

This is the most important measurement change. It converts action choice from unreliable free-form parsing into a closed-set scoring task.

### Experiment Runner

- `scripts/03_run_experiment.py`
  - Main experiment entry point.
  - Supports `--selection_mode generate` and `--selection_mode loglik`.
  - Default is loglik.
  - Supports condition-specific vector modes:
    - `contrast_negated`
    - `contrast_explicit`
    - `separate`
    - `orthogonalized`
  - Writes one JSONL output per condition.

### Metrics and Statistics

- `src/evaluation/as_metrics.py`
  - Computes quality metrics and alignment metrics.
  - Separates:
    - Quality table: `parse_ok_rate`, `unknown_rate`.
    - Alignment table: ITT and per-protocol alignment, chance baseline, entropy, unique actions, odds ratios.
  - Removes invalid speech agreement metric from default output.
- `analysis/cluster_bootstrap.py`
  - Resamples scenarios, not rows.
  - Produces cluster-bootstrap 95 percent CIs.
- `analysis/mixed_effects.py`
  - Fits mixed-effects logistic regression:
    - `aligned ~ condition * persona + (1 | scenario_id)`
  - Uses `statsmodels` `BinomialBayesMixedGLM`.
- `scripts/04_evaluate.py`
  - Runs the full evaluation pipeline and writes CSVs/figures.

### Sweeps and Final Reports

- `scripts/06_run_sweeps.py`
  - Runs alpha/layer sweeps.
  - Produces sweep CSVs and heatmaps.
- `scripts/07_final_report.py`
  - Builds final v2 report.
  - Adds paired scenario-cluster bootstrap differences.
- `results/v2/final_analysis_v2/final_report.md`
  - Main final result summary.
- `results/v2/final_analysis_v2/task6_final_run_report.md`
  - Task 6 run-specific report.

### Tests

- `tests/test_loglik_selector.py`
  - Tests log-likelihood selector behavior.
- `tests/test_as_metrics.py`
  - Tests v2 metrics.
- `tests/test_analysis_v2.py`
  - Tests bootstrap/mixed-effects utilities.
- `tests/test_vector_modes.py`
  - Tests vector mode handling and Team A conversion additions.

Current test status after Task 7:

```text
77 passed
```

## Experimental Conditions

Final conditions in `configs/experiment.yaml`:

| condition | persona prompt | activation steering | vector mode |
|---|---:|---:|---|
| `neutral_baseline` | no | no | none |
| `prompt_baseline` | yes | no | none |
| `as_only_neg` | no | yes | `contrast_negated` |
| `as_only_sep` | no | yes | `contrast_explicit` |
| `prompt_as` | yes | yes | `contrast_explicit` |

Important nuance:

`as_only_sep` is named as a separate-vector ablation, but after Task 4 diagnostics it no longer uses the neutral-anchor vector. It uses the explicit contrast vector. This is intentional. The neutral-anchor approach was found to produce aggressive/cooperative vectors pointing in a similar direction, so the final `as_only_sep` condition demonstrates that the corrected best separate design collapses back to the contrast direction.

## Vector Design

Three vector ideas were tested:

1. Contrast vector:
   - `v_contrast = aggressive - cooperative`
   - aggressive uses `+v_contrast`
   - cooperative uses `-v_contrast`

2. Neutral-anchor separate vectors:
   - `v_aggressive = aggressive - neutral`
   - `v_cooperative = cooperative - neutral`

3. Orthogonalized separate vectors:
   - Compute common component:
     - `common = (v_aggressive + v_cooperative) / 2`
   - Remove common component:
     - `v_aggressive_orth = v_aggressive - common`
     - `v_cooperative_orth = v_cooperative - common`

Observed mean cosines:

| comparison | mean cosine |
|---|---:|
| neutral-anchor aggressive vs cooperative | 0.717 |
| contrast explicit aggressive vs cooperative | -1.000 |
| orthogonalized aggressive vs cooperative | -1.000 |
| contrast explicit aggressive vs orthogonalized aggressive | 0.954 |
| contrast explicit cooperative vs orthogonalized cooperative | 0.954 |

Interpretation:

The neutral-anchor separate vectors mostly capture a shared non-neutral-vs-neutral direction rather than an aggressive-vs-cooperative axis. Orthogonalization recovers a direction very close to the original contrast vector. Therefore, the contrast vector is the best-supported persona steering direction in the current data.

## Final Run Results

Final output directory:

- `results/v2/final_raw/`
- `results/v2/final_analysis_v2/`
- `results/v2/final_figures_v2/`

Quality:

| metric | value |
|---|---:|
| minimum parse_ok_rate | 1.000 |
| maximum unknown_rate | 0.000 |

Because loglik selection is closed-set, ITT and per-protocol rates are identical.

Alignment rates:

| condition | aggressive | cooperative |
|---|---:|---:|
| `neutral_baseline` | 0.433 | 0.367 |
| `prompt_baseline` | 0.900 | 0.767 |
| `as_only_neg` | 0.600 | 0.700 |
| `as_only_sep` | 0.600 | 0.700 |
| `prompt_as` | 0.800 | 0.733 |

Chance baseline is 0.400 for both aggressive and cooperative personas because each scenario has two aligned actions out of five for each target persona.

Paired scenario-cluster differences:

| comparison | persona | diff | 95 percent CI |
|---|---|---:|---|
| `as_only_sep - neutral_baseline` | aggressive | 0.167 | [0.000, 0.333] |
| `as_only_sep - neutral_baseline` | cooperative | 0.333 | [0.133, 0.533] |
| `prompt_as - neutral_baseline` | aggressive | 0.367 | [0.167, 0.567] |
| `prompt_as - neutral_baseline` | cooperative | 0.367 | [0.133, 0.567] |
| `prompt_baseline - neutral_baseline` | aggressive | 0.467 | [0.300, 0.633] |
| `prompt_baseline - neutral_baseline` | cooperative | 0.400 | [0.200, 0.600] |
| `prompt_as - prompt_baseline` | aggressive | -0.100 | [-0.233, 0.000] |
| `prompt_as - prompt_baseline` | cooperative | -0.033 | [-0.100, 0.000] |

Mixed-effects condition effects versus neutral:

| term | odds ratio | 95 percent CI | p |
|---|---:|---|---:|
| `as_only_neg` | 1.978 | [1.106, 3.540] | 0.0216 |
| `as_only_sep` | 1.978 | [1.106, 3.540] | 0.0216 |
| `prompt_as` | 5.818 | [3.063, 11.050] | 7.42e-08 |
| `prompt_baseline` | 12.477 | [6.093, 25.552] | 5.15e-12 |

Scenario random-effect variance:

- SD: 1.024
- variance: 1.048

This supports scenario-cluster-aware analysis.

## Current Interpretation

The v2 pipeline supports these claims:

1. The previous parser/unknown confound is fixed by log-likelihood action selection.
2. Activation steering alone has a measurable positive effect over neutral baseline.
3. Prompt conditioning is stronger than activation steering under the current settings.
4. Prompt plus activation steering does not outperform prompt-only in the final run.
5. The neutral-anchor separate-vector design is not supported by the cosine diagnostics; it mostly captures shared non-neutrality.
6. Scenario identity matters enough that row-independent tests are inappropriate.

Claims that should be made cautiously:

- Do not claim activation steering is generally weaker than prompting. The current result only covers this model, prompt format, action-description scoring method, alpha/layer choice, and scenario set.
- Do not claim prompt+AS is harmful. The paired differences are non-positive versus prompt-only, but the upper CIs reach 0.000 in this run, so the safest statement is "no observed improvement over prompt-only."
- Do not claim the cooperative direction is inherently the negative aggressive direction in all models. In this experiment, orthogonalized and contrast directions align strongly, but this is data/model dependent.
- Do not claim loglik scoring is the same behavioral object as free-form NPC generation. It is a cleaner forced-choice measurement of action preference.

## Important Methodological Caveats

1. Loglik scoring measures closed-set action preference, not open-ended generation behavior.
2. Candidate action descriptions may have wording or length effects despite length normalization.
3. The scenario set is larger than before but still only 30 clusters.
4. Alpha/layer selection was based on a smaller sweep and should ideally be validated on a held-out scenario set.
5. Prompt and steering may interact nonlinearly; prompt+AS not beating prompt-only may be due to saturation or conflicting signals.
6. Vector extraction quality depends on contrastive pair quality. If pairs differ in style, specificity, or intensity, vectors may encode those dimensions.
7. The action labels assume two aggressive, two cooperative, and one neutral action per scenario. The validity of alignment depends on those labels.

## Suggested Next Analyses

1. Held-out validation:
   - Select alpha/layer on one scenario split.
   - Report final effects on a held-out split.

2. Pair quality audit:
   - Manually inspect contrastive pairs.
   - Check whether aggressive/cooperative pairs differ in length, syntax, emotional intensity, or specificity.

3. Description wording control:
   - Run the loglik selector with paraphrased action descriptions.
   - Test whether the same actions remain preferred.

4. Calibration analysis:
   - Use softmax probabilities from loglik outputs.
   - Analyze margins between the selected action and runner-up action.

5. Free-form comparison:
   - Keep a small generate-mode run for ecological comparison.
   - Report it separately as a noisier behavioral generation endpoint.

6. Cross-model replication:
   - Repeat on another instruct model if compute allows.

7. Vector ablations:
   - Compare contrast, orthogonalized, neutral-anchor, and random-vector controls under the same alpha/layer settings.

## How To Reproduce The Current V2 Run

Prepare data:

```powershell
.\.venv\Scripts\python.exe scripts\00_convert_team_a_csvs.py --update-defaults
```

Extract vectors:

```powershell
.\.venv\Scripts\python.exe scripts\02_extract_vectors.py --mode separate --quantization fp16 --layer_group all --output_dir results\v2\vectors
```

Run final experiment:

```powershell
.\.venv\Scripts\python.exe scripts\03_run_experiment.py --selection_mode loglik --conditions neutral_baseline,prompt_baseline,as_only_neg,as_only_sep,prompt_as --num_scenarios 30 --num_repeats 1 --scenarios_path data\scenarios\team_a_scenarios.jsonl --output_dir results\v2\final_raw --alpha 4.0 --layer_group middle
```

Evaluate:

```powershell
.\.venv\Scripts\python.exe scripts\04_evaluate.py --input_dir results\v2\final_raw --output_dir results\v2\final_analysis_v2 --figures_dir results\v2\final_figures_v2 --scenarios_path data\scenarios\team_a_scenarios.jsonl --bootstrap_iters 1000
```

Build final report:

```powershell
.\.venv\Scripts\python.exe scripts\07_final_report.py --output_dir results\v2\final_analysis_v2 --bootstrap_iters 1000
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Files Another LLM Should Read First

Read these in order:

1. `docs/llm_project_brief.md`
2. `results/v2/final_analysis_v2/final_report.md`
3. `results/v2/vector_diagnostics/task4_vector_design_report.md`
4. `configs/experiment.yaml`
5. `configs/steering.yaml`
6. `src/evaluation/loglik_selector.py`
7. `scripts/03_run_experiment.py`
8. `scripts/04_evaluate.py`
9. `src/evaluation/as_metrics.py`
10. `analysis/cluster_bootstrap.py`
11. `analysis/mixed_effects.py`

## Advice Request For The Reviewing LLM

Please give feedback in this structure:

1. Main methodological strengths.
2. Main methodological weaknesses.
3. Whether the final conclusions are supported.
4. Which claims should be weakened or reframed.
5. Additional controls or analyses to run before publication.
6. Code architecture risks or reproducibility problems.
7. Suggested wording for the final report's limitations section.
