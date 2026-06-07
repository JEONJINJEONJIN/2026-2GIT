# Task 6 Final Run Report

## Setup

- Selection mode: log-likelihood description scoring.
- Model: Qwen2.5-3B-Instruct.
- Scenarios: 30 Team A scenarios.
- Repeats: 1 per scenario/persona/condition. Log-likelihood scoring is deterministic, so repeated sampling is not needed for the final loglik run.
- Conditions: `neutral_baseline`, `prompt_baseline`, `as_only_neg`, `as_only_sep`, `prompt_as`.
- Personas: `aggressive`, `cooperative`.
- Steering: alpha `4.0`, layer group `middle = [18, 21, 24]`.
- Vector source: re-extracted from 120 action-derived aggressive/cooperative contrastive pairs.

## Vector Design Result

The neutral-anchor separate vectors did not isolate opposite persona directions:

| comparison | mean cosine |
|---|---:|
| neutral-anchor aggressive vs cooperative | 0.717225 |
| contrast explicit aggressive vs cooperative | -1.000000 |
| orthogonalized aggressive vs cooperative | -0.999999 |
| contrast explicit aggressive vs orthogonalized aggressive | 0.953981 |
| contrast explicit cooperative vs orthogonalized cooperative | 0.953991 |

Interpretation: under the linear representation hypothesis, cooperative was initially tested as the negative aggressive direction. Separate extraction with neutral anchors instead produced two vectors pointing mostly in the same direction, suggesting a strong shared non-neutral-vs-neutral component. Orthogonalization recovers a direction very close to the contrast vector, so the final run keeps `as_only_neg` and `as_only_sep` as an explicit ablation: both use the contrast direction, and should therefore match.

## Quality Metrics

All final loglik outputs are parse-valid:

| condition | persona | n | parse ok | unknown |
|---|---|---:|---:|---:|
| all conditions | both personas | 30 each | 1.000 | 0.000 |

This satisfies the Task 1 success criterion: parse failure is 0 percent.

## Alignment Metrics

Chance alignment baseline is 0.40 for both aggressive and cooperative personas.

| condition | aggressive ITT/PP | cooperative ITT/PP |
|---|---:|---:|
| neutral_baseline | 0.433 | 0.367 |
| prompt_baseline | 0.900 | 0.767 |
| as_only_neg | 0.600 | 0.700 |
| as_only_sep | 0.600 | 0.700 |
| prompt_as | 0.800 | 0.733 |

Because parse_ok is 100 percent, ITT and PP are identical in this run.

## Cluster Bootstrap 95 Percent CI

| condition | aggressive CI | cooperative CI |
|---|---|---|
| neutral_baseline | [0.267, 0.600] | [0.200, 0.533] |
| prompt_baseline | [0.800, 1.000] | [0.600, 0.900] |
| as_only_neg | [0.433, 0.767] | [0.533, 0.867] |
| as_only_sep | [0.433, 0.767] | [0.533, 0.867] |
| prompt_as | [0.633, 0.933] | [0.567, 0.867] |

## Mixed-Effects Logistic Regression

Model: `aligned ~ condition * persona + (1 | scenario_id)`, fit with `BinomialBayesMixedGLM`.

Random scenario effect:

- scenario SD: 1.0237
- scenario variance: 1.0480

Condition main effects versus `neutral_baseline`:

| term | odds ratio | 95 percent CI | p |
|---|---:|---|---:|
| as_only_neg | 1.978 | [1.106, 3.540] | 0.0216 |
| as_only_sep | 1.978 | [1.106, 3.540] | 0.0216 |
| prompt_as | 5.818 | [3.063, 11.050] | 7.42e-08 |
| prompt_baseline | 12.477 | [6.093, 25.552] | 5.15e-12 |

Interaction terms:

| term | odds ratio | 95 percent CI | p |
|---|---:|---|---:|
| as_only_neg x cooperative | 2.538 | [1.094, 5.886] | 0.0300 |
| as_only_sep x cooperative | 2.538 | [1.094, 5.886] | 0.0300 |
| prompt_as x cooperative | 1.109 | [0.463, 2.655] | 0.8171 |
| prompt_baseline x cooperative | 0.669 | [0.268, 1.668] | 0.3883 |

## Takeaways

1. Loglik description scoring fixed the parser confound: parse_ok is 100 percent and unknown is 0 percent for every cell.
2. `as_only_neg` and `as_only_sep` match exactly in alignment, as intended after the vector-design correction.
3. Steering-only improves over neutral, but prompt conditioning remains stronger in this run.
4. `prompt_as` is positive, but does not beat `prompt_baseline` here; prompt-only is the strongest condition by alignment rate.
5. Scenario variance is non-trivial, supporting the decision to use clustered/bootstrap and mixed-effects analysis instead of treating all rows as independent.

## Output Files

- Raw results: `results/v2/final_raw/`
- Main metrics: `results/v2/final_analysis_v2/alignment_metrics.csv`
- Quality metrics: `results/v2/final_analysis_v2/quality_metrics.csv`
- Cluster bootstrap: `results/v2/final_analysis_v2/cluster_bootstrap_alignment.csv`
- Mixed effects: `results/v2/final_analysis_v2/mixed_effects_fixed_effects.csv`
- Vector diagnostics: `results/v2/vector_diagnostics/vector_design_key_cosines.csv`
- Figures: `results/v2/final_figures_v2/`
