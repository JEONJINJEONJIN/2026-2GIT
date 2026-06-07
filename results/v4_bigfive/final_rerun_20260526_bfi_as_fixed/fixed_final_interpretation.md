# Fixed v4 Big Five Final Rerun Interpretation

Run directory: `results/v4_bigfive/final_rerun_20260526_bfi_as_fixed`

This document summarizes the fixed rerun after correcting the BFI-side AS measurement path and the BFI bootstrap unit. No model inference was rerun for this document; it uses the completed fixed rerun CSVs and post-processing outputs in this directory.

## 1. What Was Fixed

Two methodological issues were corrected before this rerun:

1. BFI self-report scoring now applies AS hooks for AS/PAS conditions.
   - `as_pas_only`: `steering_applied=True` for 440 BFI rows.
   - `elaborate_prompt_as_pas`: `steering_applied=True` for 440 BFI rows.
   - Non-AS conditions remain `steering_applied=False`.

2. BFI movement CIs are now computed from BFI item-level samples rather than from one pre-averaged BFI scalar.
   - This removes the degenerate CI pattern where CI bounds were identical to the point estimate.
   - BFI intervals are item-level bootstrap intervals.
   - TRAIT intervals remain scenario-level bootstrap intervals.

The previous claim that `as_pas_only` leaves the BFI surface exactly unchanged should not be used. After the fix, `as_pas_only` produces small BFI-side movement in some cells, but its dominant and reliable effect remains on the TRAIT/action surface.

## 2. Run Configuration

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Conditions | `baseline`, `one_line_prompt`, `elaborate_prompt`, `as_pas_only`, `elaborate_prompt_as_pas` |
| Traits | `agreeableness`, `conscientiousness`, `extraversion`, `neuroticism`, `openness` |
| Directions | `high`, `low` |
| AS alpha | `4.0` |
| AS layers | `[18, 21, 24]` |
| Vector file | `results/v4_bigfive/vectors/bigfive_trait_vectors_full_fp16.pt` |
| BFI items | 44 |
| TRAIT scenario rows | 750 |
| Per condition/trait/direction rows | 150 |

Generated source result files:

- `bfi_scores.csv`
- `trait_scores.csv`
- `consistency_metrics.csv`
- `pas_loglik_agreement.csv`
- `run_manifest.json`

Generated analysis files:

- `consistency_summary.csv`
- `condition_differences.csv`
- `condition_effects_with_ci.csv`
- `paired_scenario_differences.csv`
- `paired_bootstrap_ci.csv`
- `pas_agreement_summary.csv`
- `analysis_report.md`
- `final_report.md`
- `separated_movement_summary.csv`
- `co_movement_analysis.md`
- `analysis_report_v2.md`

## 3. Primary Result: Separated BFI and TRAIT Movement

The primary metric is now separated movement:

- `delta_bfi_target`: target-direction movement on the BFI self-report surface.
- `delta_trait_target`: target-direction movement on the TRAIT/action log-likelihood surface.

The overall pattern is that every non-baseline condition reliably moves the TRAIT/action surface, while BFI/self-report movement is larger under prompt conditions and weaker under `as_pas_only`.

| Condition | Mean delta_bfi_target | Mean delta_trait_target | Co-movement count |
|---|---:|---:|---|
| `one_line_prompt` | +0.264 | +0.040 | 7 both-positive, 3 TRAIT-only |
| `elaborate_prompt` | +0.127 | +0.043 | 4 both-positive, 6 TRAIT-only |
| `as_pas_only` | +0.067 | +0.037 | 2 both-positive, 8 TRAIT-only |
| `elaborate_prompt_as_pas` | +0.118 | +0.060 | 3 both-positive, 7 TRAIT-only |

Across the 40 non-baseline trait/direction cells:

- 16 cells are `both_positive`.
- 24 cells are `trait_only`.
- 0 cells are `bfi_only`.
- 0 cells are `neither`.

This supports the reframed finding: AS/PAS reliably moves the action surface; prompt conditions move the speech/self-report surface more strongly; the two surfaces move independently rather than collapsing into one defensible consistency number.

## 4. AS/PAS-Only Condition

`as_pas_only` is the cleanest condition for asking what AS/PAS does without an explicit persona prompt.

| Trait | Direction | delta_bfi_target | BFI 95% CI | delta_trait_target | TRAIT 95% CI | Flag |
|---|---|---:|---|---:|---|---|
| agreeableness | high | +0.083 | [0.000, 0.167] | +0.035 | [0.030, 0.039] | trait_only |
| agreeableness | low | +0.056 | [0.000, 0.139] | +0.038 | [0.033, 0.044] | trait_only |
| conscientiousness | high | +0.056 | [0.000, 0.111] | +0.034 | [0.031, 0.038] | trait_only |
| conscientiousness | low | +0.028 | [0.000, 0.083] | +0.031 | [0.027, 0.035] | trait_only |
| extraversion | high | +0.031 | [0.000, 0.094] | +0.041 | [0.037, 0.045] | trait_only |
| extraversion | low | +0.063 | [0.000, 0.125] | +0.025 | [0.022, 0.029] | trait_only |
| neuroticism | high | +0.156 | [0.063, 0.219] | +0.040 | [0.037, 0.044] | both_positive |
| neuroticism | low | +0.000 | [-0.094, 0.094] | +0.029 | [0.025, 0.033] | trait_only |
| openness | high | +0.025 | [0.000, 0.075] | +0.038 | [0.033, 0.042] | trait_only |
| openness | low | +0.175 | [0.100, 0.250] | +0.058 | [0.051, 0.066] | both_positive |

Interpretation:

`as_pas_only` moves the TRAIT/action surface in all 10 trait/direction cells with CIs excluding 0. BFI movement is smaller and only clearly positive in 2 cells: neuroticism-high and openness-low. This means the fixed result does not support the old sentence "AS/PAS has no BFI effect." The safer claim is: AS/PAS alone primarily and reliably affects the TRAIT/action surface, with limited BFI-side spillover in a minority of cells.

## 5. Prompt Plus AS/PAS Condition

`elaborate_prompt_as_pas` combines explicit persona prompting with AS/PAS.

| Trait | Direction | delta_bfi_target | BFI 95% CI | delta_trait_target | TRAIT 95% CI | Flag |
|---|---|---:|---|---:|---|---|
| agreeableness | high | +0.111 | [-0.028, 0.250] | +0.065 | [0.058, 0.073] | trait_only |
| agreeableness | low | +0.028 | [-0.250, 0.306] | +0.066 | [0.057, 0.075] | trait_only |
| conscientiousness | high | +0.139 | [-0.056, 0.306] | +0.058 | [0.052, 0.064] | trait_only |
| conscientiousness | low | +0.083 | [-0.028, 0.194] | +0.071 | [0.061, 0.079] | trait_only |
| extraversion | high | +0.313 | [0.188, 0.406] | +0.069 | [0.060, 0.078] | both_positive |
| extraversion | low | +0.125 | [0.031, 0.219] | +0.041 | [0.034, 0.048] | both_positive |
| neuroticism | high | +0.250 | [0.000, 0.438] | +0.058 | [0.053, 0.065] | trait_only |
| neuroticism | low | +0.031 | [0.000, 0.094] | +0.030 | [0.026, 0.035] | trait_only |
| openness | high | +0.000 | [-0.075, 0.075] | +0.056 | [0.049, 0.064] | trait_only |
| openness | low | +0.100 | [0.025, 0.175] | +0.085 | [0.076, 0.095] | both_positive |

Interpretation:

This condition has the largest mean TRAIT movement among the four non-baseline conditions: +0.060. However, BFI movement is not uniformly reliable. Only 3 of 10 cells are both-positive. The neuroticism-high case remains a clear speech-action decoupling case: BFI moves by +0.250 while TRAIT moves by only +0.058, and the BFI CI touches 0 in this fixed run. The earlier pre-fix value for this case was larger (`delta_bfi=+0.344`, `delta_trait=+0.058`), but the fixed run still shows the same qualitative issue: speech/self-report movement can be much larger than action-side movement.

## 6. Largest Surface Decoupling Cases

Rows below are sorted by `abs(delta_bfi_target - delta_trait_target)`.

| Condition | Trait | Direction | delta_bfi_target | delta_trait_target | Absolute gap | Flag |
|---|---|---|---:|---:|---:|---|
| `one_line_prompt` | neuroticism | high | +0.563 | +0.034 | 0.529 | both_positive |
| `one_line_prompt` | agreeableness | low | +0.472 | +0.058 | 0.414 | both_positive |
| `one_line_prompt` | conscientiousness | high | +0.361 | +0.024 | 0.337 | both_positive |
| `elaborate_prompt` | neuroticism | high | +0.344 | +0.043 | 0.300 | both_positive |
| `one_line_prompt` | extraversion | high | +0.281 | +0.036 | 0.245 | both_positive |
| `elaborate_prompt_as_pas` | extraversion | high | +0.313 | +0.069 | 0.244 | both_positive |
| `elaborate_prompt_as_pas` | neuroticism | high | +0.250 | +0.058 | 0.192 | trait_only |
| `one_line_prompt` | extraversion | low | +0.188 | +0.039 | 0.149 | both_positive |
| `elaborate_prompt` | agreeableness | high | +0.194 | +0.053 | 0.142 | both_positive |
| `one_line_prompt` | openness | low | +0.200 | +0.066 | 0.134 | both_positive |
| `one_line_prompt` | agreeableness | high | +0.167 | +0.047 | 0.119 | trait_only |
| `as_pas_only` | openness | low | +0.175 | +0.058 | 0.117 | both_positive |

The largest decoupling cases are mostly prompt-driven. This is the central result: prompts can strongly move the self-report surface without comparably moving action-side scoring.

## 7. Gap Metric as Auxiliary Signal

The old metric:

```text
speech_action_consistency = 1 - abs(BFI_score - TRAIT_score)
```

should be treated only as a gap behavior descriptor. It is not a primary consistency metric because BFI and TRAIT are measured through different modalities and scales.

Mean gap metric by condition:

| Condition | Mean gap metric |
|---|---:|
| `baseline` | 0.898 |
| `as_pas_only` | 0.910 |
| `elaborate_prompt` | 0.868 |
| `elaborate_prompt_as_pas` | 0.866 |
| `one_line_prompt` | 0.764 |

Interpretation:

The gap metric becomes worse under prompt-heavy conditions because BFI moves more than TRAIT. This should not be described as "AS/PAS reduces speech-action mismatch." It is better described as evidence that BFI and TRAIT surfaces do not move together, especially under explicit prompting.

The `as_pas_only` average gap is slightly higher than baseline, but that should not be overclaimed. The separated movement table is more informative: AS/PAS-only produces reliable TRAIT movement in all cells and limited BFI movement in only 2 cells.

## 8. PAS Diagnostic Result

PAS is implemented and logged, but in this v4 final run it is diagnostic rather than the final action selector for `TRAIT_score`.

Mean `pas_loglik_agreement_rate`:

| Condition | Mean PAS/loglik agreement |
|---|---:|
| `as_pas_only` | 0.399 |
| `elaborate_prompt_as_pas` | 0.476 |

Interpretation:

PAS and log-likelihood scoring agree less than half the time on average. This matters architecturally: if the project goal is to fix "speech says one thing, action selection does another," PAS should probably become the final action selector in the next experiment instead of remaining only a diagnostic comparison.

## 9. Final Claim for Presentation

Recommended claim:

> The fixed v4 rerun shows that AS/PAS reliably moves the TRAIT/action surface. Prompting moves the BFI/self-report surface more strongly, but BFI and TRAIT do not move together. The practical problem is not solved by a single gap metric; it should be analyzed as two separate surfaces plus an explicit co-movement question.

Avoid this claim:

> AS/PAS reduces speech-action mismatch.

That statement is too strong for the current evidence and depends on a gap metric with scale-mismatch problems.

Better framing:

1. `one_line_prompt` is strongest on BFI movement but creates the largest surface decoupling.
2. `as_pas_only` is reliable on TRAIT/action movement and has limited BFI-side spillover after the hook fix.
3. `elaborate_prompt_as_pas` gives the largest mean TRAIT movement, but still does not make BFI and TRAIT co-move uniformly.
4. The next experiment should test PAS as the actual final action selector, because the current v4 run logs PAS agreement but does not let PAS decide the primary `TRAIT_score`.

## 10. Immediate Next Steps

1. For the midterm, present separated movement as the main result: BFI-side movement and TRAIT-side movement.
2. Treat the gap metric only as an auxiliary visualization of surface decoupling.
3. Explicitly state that the earlier BFI CI issue was fixed by item-level bootstrap and BFI-side AS hook application.
4. Run the next experiment with PAS as the final selector, not only as `pas_loglik_agreement`.
5. Compare action selection distribution under log-likelihood selection versus PAS selection to directly test whether PAS solves the action-side persona mismatch.
