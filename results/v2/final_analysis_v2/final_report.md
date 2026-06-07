# Activation Steering Evaluation V2 Final Report

## Executive Summary

- Description log-likelihood scoring removed the parser confound: every final cell has parse_ok = 1.000 and unknown = 0.000.
- Steering-only improves over neutral, but prompt conditioning remains the strongest intervention in this run.
- Prompt plus activation steering is positive versus neutral, but it does not improve over prompt-only.
- Neutral-anchor separate vectors point in a similar direction, supporting the interpretation that this extraction mostly captures a shared non-neutral component.
- Scenario-level variance is substantial, so clustered bootstrap and mixed-effects estimates should be used instead of row-independent tests.

## Experimental Setup

- Model: Qwen2.5-3B-Instruct.
- Scoring: conditional log-likelihood over action descriptions.
- Scenarios: 30 Team A scenarios.
- Repeats: 1 deterministic loglik decision per scenario/persona/condition.
- Personas: aggressive and cooperative.
- Conditions: neutral_baseline, prompt_baseline, as_only_neg, as_only_sep, prompt_as.
- Steering: alpha 4.0, middle layers [18, 21, 24].
- Vector estimation: 120 action-derived aggressive/cooperative contrastive pairs.

## Quality Metrics

- Minimum parse_ok_rate: 1.000.
- Maximum unknown_rate: 0.000.
- ITT and per-protocol alignment are therefore identical in the final run.

## Alignment Rates

| condition | aggressive | cooperative |
|---|---:|---:|
| neutral_baseline | 0.433 | 0.367 |
| prompt_baseline | 0.900 | 0.767 |
| as_only_neg | 0.600 | 0.700 |
| as_only_sep | 0.600 | 0.700 |
| prompt_as | 0.800 | 0.733 |

Chance baseline is 0.400 for both aggressive and cooperative personas.

## Cluster Bootstrap Intervals

| condition | aggressive 95% CI | cooperative 95% CI |
|---|---|---|
| neutral_baseline | [0.267, 0.600] | [0.200, 0.533] |
| prompt_baseline | [0.800, 1.000] | [0.600, 0.900] |
| as_only_neg | [0.433, 0.767] | [0.533, 0.867] |
| as_only_sep | [0.433, 0.767] | [0.533, 0.867] |
| prompt_as | [0.633, 0.933] | [0.567, 0.867] |

## Paired Condition Differences

| comparison | persona | diff | 95% CI |
|---|---|---:|---|
| as_only_neg - neutral_baseline | aggressive | 0.167 | [0.000, 0.333] |
| as_only_neg - neutral_baseline | cooperative | 0.333 | [0.133, 0.533] |
| as_only_sep - neutral_baseline | aggressive | 0.167 | [0.000, 0.333] |
| as_only_sep - neutral_baseline | cooperative | 0.333 | [0.133, 0.533] |
| prompt_as - neutral_baseline | aggressive | 0.367 | [0.167, 0.567] |
| prompt_as - neutral_baseline | cooperative | 0.367 | [0.133, 0.567] |
| prompt_baseline - neutral_baseline | aggressive | 0.467 | [0.300, 0.633] |
| prompt_baseline - neutral_baseline | cooperative | 0.400 | [0.200, 0.600] |
| prompt_as - prompt_baseline | aggressive | -0.100 | [-0.233, 0.000] |
| prompt_as - prompt_baseline | cooperative | -0.033 | [-0.100, 0.000] |
| prompt_as - as_only_sep | aggressive | 0.200 | [0.067, 0.333] |
| prompt_as - as_only_sep | cooperative | 0.033 | [-0.067, 0.133] |

## Mixed-Effects Logistic Regression

Model: `aligned ~ condition * persona + (1 | scenario_id)`.

| term | odds ratio | 95% CI | p |
|---|---:|---|---:|
| as_only_neg vs neutral | 1.978 | [1.106, 3.540] | 0.0216 |
| as_only_sep vs neutral | 1.978 | [1.106, 3.540] | 0.0216 |
| prompt_as vs neutral | 5.818 | [3.063, 11.050] | 7.42e-08 |
| prompt_baseline vs neutral | 12.477 | [6.093, 25.552] | 5.15e-12 |

Scenario random-effect SD is 1.024 and variance is 1.048.

## Vector Diagnosis

| comparison | mean cosine |
|---|---:|
| sep_neutral_anchor.aggressive vs sep_neutral_anchor.cooperative | 0.717 |
| neg.aggressive vs neg.cooperative | -1.000 |
| sep_orthogonalized.aggressive vs sep_orthogonalized.cooperative | -1.000 |
| neg.aggressive vs sep_orthogonalized.aggressive | 0.954 |
| neg.cooperative vs sep_orthogonalized.cooperative | 0.954 |

Interpretation: neutral-anchor separate extraction did not recover opposed persona vectors. The orthogonalized direction is very close to the explicit contrast direction, and the final `as_only_sep` condition is best read as an explicit contrast-vector ablation rather than a successful neutral-anchor separation.

## Final Interpretation

The v2 pipeline fixes the major measurement problem from the legacy generation setup. With parse failures removed, the remaining result is more interpretable: activation steering has a measurable effect over neutral, especially for cooperative alignment, but prompt conditioning is still stronger. Adding steering to the prompt does not improve over prompt-only under the selected alpha/layer setting. The vector diagnostics also show that the original neutral-anchor separate-vector idea is not well supported by this data, because it captures a shared non-neutral direction more than an aggressive-versus-cooperative axis.

## Artifacts

- Raw final outputs: `results/v2/final_raw/`.
- Main metrics: `results/v2/final_analysis_v2/alignment_metrics.csv`.
- Quality metrics: `results/v2/final_analysis_v2/quality_metrics.csv`.
- Paired differences: `results/v2/final_analysis_v2/paired_cluster_differences.csv`.
- Mixed effects: `results/v2/final_analysis_v2/mixed_effects_fixed_effects.csv`.
- Vector diagnostics: `results/v2/vector_diagnostics/vector_design_key_cosines.csv`.
- Figures: `results/v2/final_figures_v2/`.
