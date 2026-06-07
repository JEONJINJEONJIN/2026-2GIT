# Progress Summary for LLM Handoff

## Purpose of This Document

This document summarizes the current state of the activation steering experiment project so another LLM can quickly understand what has been done, what the current results mean, and what remains to improve.

The project evaluates whether activation steering can shift `Qwen/Qwen2.5-3B-Instruct` toward persona-aligned NPC action choices for two personas:

- `aggressive`
- `cooperative`

Each scenario has five candidate actions. For each target persona, two actions are aligned, two are aligned with the opposite persona, and one is neutral. Chance alignment for aggressive/cooperative is therefore `2/5 = 0.40`.

## Current Branch and Workspace

- Working branch: `refactor/eval-v2`
- Workspace root: `C:/GIT`
- Main result directory: `C:/GIT/results/v2/`
- Legacy results moved to: `C:/GIT/results/v1_legacy/`

## High-Level Status

The V2 evaluation pipeline has been implemented and run end-to-end.

Completed:

1. Replaced unreliable free-form parsing with log-likelihood action selection.
2. Rebuilt metrics around quality, ITT/PP alignment, chance baselines, bootstrap CIs, and mixed-effects modeling.
3. Diagnosed the vector design problem.
4. Ran alpha/layer sweeps.
5. Expanded the experiment to 30 scenarios and 120 action-derived contrastive pairs.
6. Ran the final V2 experiment.
7. Generated final reports and slide draft assets.
8. Started V3 design planning based on methodological weaknesses.

Current test status:

```text
77 passed
```

## Why V2 Was Needed

The legacy pipeline used free-form generation and then parsed the generated text into an action. This created major measurement problems:

- Low parse success.
- High and uneven unknown rates.
- Non-comparable denominators across conditions.
- Invalid speech/action agreement metric.
- Row-independent tests despite scenario clustering.
- Unvalidated assumption that cooperative steering could be represented as the negative aggressive direction.

V2 was built to remove the parsing confound and provide more defensible statistics.

## Main V2 Pipeline

The current V2 flow is:

```text
Team A CSV data
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

## Key Code Additions and Changes

### Log-Likelihood Action Selection

Added:

- `src/evaluation/loglik_selector.py`

Purpose:

- Computes conditional log-likelihood for each candidate action description:
  - `log p(action_text | prompt)`
- Selects the action with the highest normalized log-likelihood.
- Returns selected index, log-likelihoods, normalized log-likelihoods, and softmax probabilities.

This is the core V2 measurement change.

### Experiment Runner

Modified:

- `scripts/03_run_experiment.py`

Key changes:

- Added `--selection_mode {generate, loglik}`.
- Default mode is now `loglik`.
- Supports multiple vector modes:
  - `contrast_negated`
  - `contrast_explicit`
  - `separate`
  - `orthogonalized`
- Writes comparable result schema for generate/loglik modes.

### Metrics

Added/modified:

- `src/evaluation/as_metrics.py`
- `scripts/04_evaluate.py`

Key changes:

- Separates quality metrics from alignment metrics.
- Reports:
  - `parse_ok_rate`
  - `unknown_rate`
  - ITT alignment
  - per-protocol alignment
  - chance baseline
  - alignment vs chance
  - odds ratios vs neutral
  - entropy
  - unique actions
- Removed invalid speech metric from default output.

### Statistical Analysis

Added:

- `analysis/cluster_bootstrap.py`
- `analysis/mixed_effects.py`

Methods:

- Scenario-cluster bootstrap, resampling by `scenario_id`.
- Mixed-effects logistic regression:

```text
aligned ~ condition * persona + (1 | scenario_id)
```

Implementation currently uses `statsmodels` `BinomialBayesMixedGLM`.

### Vector Extraction

Modified:

- `scripts/02_extract_vectors.py`

Supported vector sets:

- contrast vector
- neutral-anchor aggressive vector
- neutral-anchor cooperative vector
- contrast explicit persona directions
- orthogonalized vectors

Vector output:

- `results/v2/vectors/steering_vectors_separate_fp16.pt`
- `results/v2/vectors/vector_set_cosine_similarities.csv`

### Sweep Runner

Added:

- `scripts/06_run_sweeps.py`

Used for alpha/layer sweep.

### Final Report Script

Added:

- `scripts/07_final_report.py`

Produces:

- `results/v2/final_analysis_v2/final_report.md`
- `results/v2/final_analysis_v2/paired_cluster_differences.csv`

### Slide Asset Script

Added:

- `scripts/08_make_slide_assets.py`

Produces draft slide figures:

- `results/v2/slide_assets/01_alignment_bar_chart.png`
- `results/v2/slide_assets/02_vector_design_diagram.png`
- `results/v2/slide_assets/03_pipeline_v1_v2_flow.png`
- `results/v2/slide_assets/04_mixed_effects_forest_plot.png`

These are draft assets and need redesign for better readability.

## Final V2 Conditions

From `configs/experiment.yaml`:

| condition | persona prompt | activation steering | vector mode |
|---|---:|---:|---|
| `neutral_baseline` | no | no | none |
| `prompt_baseline` | yes | no | none |
| `as_only_neg` | no | yes | `contrast_negated` |
| `as_only_sep` | no | yes | `contrast_explicit` |
| `prompt_as` | yes | yes | `contrast_explicit` |

Important:

- `as_only_neg` and `as_only_sep` are effectively the same corrected contrast-vector condition after the Task 4 vector-design correction.
- They are retained as an ablation/history artifact, but should not be interpreted as two independent steering mechanisms.

## Final V2 Run Setup

Final run:

- model: `Qwen/Qwen2.5-3B-Instruct`
- scoring: description-based log-likelihood
- scenarios: 30 Team A scenarios
- repeats: 1 deterministic loglik decision per scenario/persona/condition
- personas: aggressive and cooperative
- conditions: five listed above
- steering alpha: `4.0`
- steering layers: `middle = [18, 21, 24]`
- vector source: 120 action-derived aggressive/cooperative contrastive pairs

Raw outputs:

- `results/v2/final_raw/`

Final analysis:

- `results/v2/final_analysis_v2/`

## Final V2 Results

### Quality

Loglik selection removed the parser problem:

| metric | value |
|---|---:|
| parse_ok_rate | 1.000 |
| unknown_rate | 0.000 |

Since parse is always valid, ITT and per-protocol alignment are identical in the final V2 run.

### Alignment Rates

| condition | aggressive | cooperative |
|---|---:|---:|
| `neutral_baseline` | 0.433 | 0.367 |
| `prompt_baseline` | 0.900 | 0.767 |
| `as_only_neg` | 0.600 | 0.700 |
| `as_only_sep` | 0.600 | 0.700 |
| `prompt_as` | 0.800 | 0.733 |

Interpretation:

- AS-only improves over neutral.
- Prompt-only is strongest.
- Prompt+AS does not improve over prompt-only in this run.

### Paired Scenario-Cluster Differences

From `paired_cluster_differences.csv`:

| comparison | persona | diff | 95% CI |
|---|---|---:|---|
| `as_only_sep - neutral_baseline` | aggressive | 0.167 | [0.000, 0.333] |
| `as_only_sep - neutral_baseline` | cooperative | 0.333 | [0.133, 0.533] |
| `prompt_as - neutral_baseline` | aggressive | 0.367 | [0.167, 0.567] |
| `prompt_as - neutral_baseline` | cooperative | 0.367 | [0.133, 0.567] |
| `prompt_baseline - neutral_baseline` | aggressive | 0.467 | [0.300, 0.633] |
| `prompt_baseline - neutral_baseline` | cooperative | 0.400 | [0.200, 0.600] |
| `prompt_as - prompt_baseline` | aggressive | -0.100 | [-0.233, 0.000] |
| `prompt_as - prompt_baseline` | cooperative | -0.033 | [-0.100, 0.000] |

Important caution:

- The prompt baseline, especially aggressive prompt-only at `0.900`, has a ceiling effect.
- The design has little headroom to detect an additional prompt+AS gain.
- The safest wording is "no observed improvement over prompt-only," not "prompt+AS is harmful."

### Mixed-Effects Results

Model:

```text
aligned ~ condition * persona + (1 | scenario_id)
```

Main condition effects versus neutral:

| condition | OR | 95% CI | p |
|---|---:|---|---:|
| `as_only_neg` | 1.978 | [1.106, 3.540] | 0.0216 |
| `as_only_sep` | 1.978 | [1.106, 3.540] | 0.0216 |
| `prompt_as` | 5.818 | [3.063, 11.050] | 7.42e-08 |
| `prompt_baseline` | 12.477 | [6.093, 25.552] | 5.15e-12 |

Scenario random-effect:

- SD: 1.024
- variance: 1.048

This supports the decision to use scenario-cluster-aware analysis.

## Vector Design Finding

Task 4 found an important vector design issue.

Neutral-anchor vectors:

- `v_aggressive = aggressive - neutral`
- `v_cooperative = cooperative - neutral`

These were expected to represent distinct persona directions. Instead, they pointed in a similar direction:

| comparison | mean cosine |
|---|---:|
| neutral-anchor aggressive vs cooperative | 0.717 |
| contrast explicit aggressive vs cooperative | -1.000 |
| orthogonalized aggressive vs cooperative | -1.000 |
| contrast aggressive vs orthogonalized aggressive | 0.954 |
| contrast cooperative vs orthogonalized cooperative | 0.954 |

Interpretation:

- Neutral-anchor extraction likely captured a shared non-neutral-vs-neutral component.
- Orthogonalization recovered a direction very close to the aggressive-vs-cooperative contrast vector.
- The final V2 steering direction should therefore be interpreted as a contrast-vector persona axis.

## Main Methodological Weaknesses Still Remaining

The following weaknesses were identified after V2:

1. Construct validity:
   - The primary endpoint is `log p(action_description | prompt)`.
   - This measures forced-choice action-description preference, not necessarily open-ended behavior.
   - It may reflect preference for persona-aligned wording rather than behavior.

2. Prompt ceiling effect:
   - Prompt-only aggressive alignment is already `0.900`.
   - There is little room to detect improvement from prompt+AS.

3. Missing negative controls:
   - No random vector.
   - No zero-vector hook.
   - No wrong-persona vector.
   - Therefore AS-only improvement may not yet prove persona-specific steering.

4. In-sample vector/evaluation overlap:
   - Vector extraction and evaluation currently use the same 30 scenario universe.
   - Alpha/layer selection was also based on non-held-out data.

5. Action label validation:
   - The action labels are not yet externally validated.
   - Inter-rater reliability is not reported.

6. Contrastive pair audit:
   - Pair quality has not been fully audited for length, style, intensity, or specificity confounds.

7. Duplicate AS conditions:
   - `as_only_neg` and `as_only_sep` are identical after correction and should be explained clearly.

8. Mixed-effects model limitations:
   - 30 scenario clusters is modest.
   - `BinomialBayesMixedGLM` estimates should be interpreted alongside cluster bootstrap.

## Current Interpretation

Defensible V2 claims:

1. The parser/unknown confound is fixed for the closed-set evaluation.
2. Activation steering alone increases persona alignment relative to neutral in this forced-choice setup.
3. Prompt conditioning is stronger than activation steering under the current setup.
4. Prompt+AS does not show improvement over prompt-only in this run.
5. Neutral-anchor separate-vector extraction is not supported; the contrast direction is the better-supported persona axis.

Claims to avoid:

- Do not claim AS generally changes real NPC behavior based only on loglik description scoring.
- Do not claim prompt+AS is harmful.
- Do not claim AS is generally weaker than prompting across models/tasks.
- Do not treat `as_only_neg` and `as_only_sep` as independent evidence.

## V3 Planning Status

V3 design has been started but not fully implemented.

Created:

- `docs/v3_experiment_design.md`
- `configs/experiments/v3_measurement_validation.yaml`
- `configs/experiments/v3_controls_dev.yaml`
- `configs/experiments/v3_final_test.yaml`

V3 goals:

1. Validate whether loglik action-description scoring is a good measurement proxy.
2. Add negative controls:
   - zero hook
   - random normalized vector
   - wrong-persona vector
   - shuffled-label vector
3. Split train/dev/test:
   - train for vector extraction
   - dev for alpha/layer selection
   - test for final claims
4. Add paraphrase-aware action scoring.
5. Use `aligned_probability_mass` as a primary metric instead of only top-1 alignment.
6. Keep free-form generation as a secondary endpoint.
7. Add PAS cosine-projection action selection as a separate scoring mode.

PAS clarification:

- PAS here means selecting the candidate action whose hidden-state action embedding has the highest cosine similarity with the steering/persona vector.
- Prototype code exists:
  - `src/projection/selector.py`
  - `src/projection/embedder.py`
  - `tests/test_selector.py`
- PAS has not yet been integrated into `scripts/03_run_experiment.py`.
- No final V2 numbers currently use PAS.
- Future V3 should compare:
  - `loglik`: text likelihood preference
  - `pas`: hidden-space action/vector similarity
  - `generate`: open-ended behavior proxy

## Slide and Visualization Status

Draft slide assets were generated, but their readability is poor and they should be redesigned.

Generated draft assets:

- `results/v2/slide_assets/01_alignment_bar_chart.png`
- `results/v2/slide_assets/02_vector_design_diagram.png`
- `results/v2/slide_assets/03_pipeline_v1_v2_flow.png`
- `results/v2/slide_assets/04_mixed_effects_forest_plot.png`

Brief for another LLM/designer:

- `docs/slide_visualization_redesign_brief.md`

That brief includes all source data, core numbers, current asset paths, and redesign instructions.

## Important Files for Review

Read these first:

1. `docs/progress_summary_for_llm.md`
2. `docs/llm_project_brief.md`
3. `docs/v3_experiment_design.md`
4. `docs/slide_visualization_redesign_brief.md`
5. `results/v2/final_analysis_v2/final_report.md`
6. `results/v2/final_analysis_v2/task6_final_run_report.md`
7. `results/v2/vector_diagnostics/task4_vector_design_report.md`
8. `configs/experiment.yaml`
9. `configs/steering.yaml`
10. `configs/experiments/v3_measurement_validation.yaml`
11. `configs/experiments/v3_controls_dev.yaml`
12. `configs/experiments/v3_final_test.yaml`

Important code files:

1. `src/evaluation/loglik_selector.py`
2. `src/evaluation/as_metrics.py`
3. `analysis/cluster_bootstrap.py`
4. `analysis/mixed_effects.py`
5. `scripts/00_convert_team_a_csvs.py`
6. `scripts/02_extract_vectors.py`
7. `scripts/03_run_experiment.py`
8. `scripts/04_evaluate.py`
9. `scripts/06_run_sweeps.py`
10. `scripts/07_final_report.py`
11. `scripts/08_make_slide_assets.py`

## Suggested Next Steps

Recommended implementation order:

1. Clean up presentation assets using `docs/slide_visualization_redesign_brief.md`.
2. Start V3 implementation with data split utilities.
3. Add paraphrase support to action definitions.
4. Integrate PAS cosine-projection selector into the main experiment runner.
5. Extend loglik selector to score multiple paraphrases per action.
6. Add aligned probability mass, rank, margin, and PAS/loglik agreement metrics.
7. Add zero/random/wrong-vector controls.
8. Add run manifest saving with config snapshots and vector hashes.
9. Run V3 measurement validation comparing loglik, PAS, and generate.
10. Run V3 controls on dev.
11. Freeze alpha/layer setting.
12. Run held-out V3 final test.

## Best One-Sentence Summary

V2 successfully fixed the parsing problem and produced a cleaner forced-choice activation steering result, but the main remaining work is V3: proving that the effect is persona-specific, held-out, and not merely a preference for persona-flavored action descriptions.
