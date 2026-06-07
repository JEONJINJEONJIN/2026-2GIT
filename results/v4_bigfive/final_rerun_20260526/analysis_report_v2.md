# V4 Final Rerun Analysis v2: Separated BFI/TRAIT Movement

## 1. What Changed In This Reframing

The earlier report used `speech_action_consistency = 1 - abs(BFI_score - TRAIT_score)` as the headline metric. This v2 analysis demotes that gap metric to an auxiliary signal because BFI and TRAIT are measured through different modalities and scales: BFI is a self-report surface, while TRAIT is an action-side log-likelihood surface.

The primary result is now reported as two separate movements:

- `delta_bfi_target`: movement of BFI-side target attainment relative to baseline.
- `delta_trait_target`: movement of TRAIT-side target attainment relative to baseline.

The central question is whether these two surfaces move together. The answer is explicit in `co_movement_flag`.

CI note: BFI movement intervals in this report are computed from raw `bfi_scores.csv` item rows with item-level bootstrap. TRAIT movement intervals are computed from `consistency_metrics.csv` with scenario-level bootstrap.

## 2. BFI-Side Movement Results (Primary)

| Condition | Positive BFI cells | Mean delta BFI target | Largest absolute BFI movement |
|---|---:|---:|---:|
| `one_line_prompt` | 7 / 10 | +0.264 | 0.562 |
| `elaborate_prompt` | 4 / 10 | +0.127 | 0.344 |
| `as_pas_only` | 0 / 10 | +0.000 | 0.000 |
| `elaborate_prompt_as_pas` | 4 / 10 | +0.127 | 0.344 |

BFI-side movement should be read separately from TRAIT movement. Prompted conditions usually move the self-report surface, and no-prompt AS/PAS conditions test whether activation steering also changes that surface. This is why BFI must not be hidden inside a combined gap score.

## 3. TRAIT-Side Movement Results (Primary)

| Condition | Positive TRAIT cells | Mean delta TRAIT target | Largest absolute TRAIT movement |
|---|---:|---:|---:|
| `one_line_prompt` | 10 / 10 | +0.040 | 0.066 |
| `elaborate_prompt` | 10 / 10 | +0.043 | 0.059 |
| `as_pas_only` | 10 / 10 | +0.037 | 0.058 |
| `elaborate_prompt_as_pas` | 10 / 10 | +0.060 | 0.085 |

TRAIT-side movement is reliable across all non-baseline conditions in this rerun. AS/PAS reliably moves the action surface, and `elaborate_prompt_as_pas` has the strongest mean TRAIT target movement.

## 4. Co-Movement Analysis

| Condition | both_positive | bfi_only | trait_only | neither |
|---|---:|---:|---:|---:|
| `one_line_prompt` | 7 | 0 | 3 | 0 |
| `elaborate_prompt` | 4 | 0 | 6 | 0 |
| `as_pas_only` | 0 | 0 | 10 | 0 |
| `elaborate_prompt_as_pas` | 4 | 0 | 6 | 0 |

The real finding is decoupling. `as_pas_only` is pure action-surface movement: TRAIT moves, BFI does not. Prompt+AS/PAS moves both surfaces in most cells, but not by the same amount. The elaborate_prompt_as_pas neuroticism-high case is the clearest example: `delta_bfi_target = +0.344`, while `delta_trait_target = +0.058`. Speech-side movement is much larger than action-side movement.

This supports the reframed claim: AS/PAS reliably moves the action surface; prompt+AS/PAS moves the speech surface more than the action surface; the two surfaces move independently.

## 5. Gap Metric As Auxiliary Signal

The gap metric is retained only as descriptive gap behavior:

```text
gap_behavior = 1 - abs(BFI_score - TRAIT_score)
```

Because BFI and TRAIT are different measurement surfaces, this value should not be treated as the primary success criterion. It is useful for diagnosing whether separated movement created a smaller or larger gap, but it does not by itself explain which surface moved.

`as_pas_only` has the asymmetric pattern reported earlier: gap behavior improves on A-high/C-high/E-high/N-low/O-high and decreases on the opposite directions. This should be interpreted as evidence that the model's baseline self-report is shifted by alignment tuning and baseline tendencies, not as evidence that AS/PAS fails on those traits.

| Condition | Trait | Target | Delta gap behavior | Gap 95% CI | Delta BFI target | Delta TRAIT target |
|---|---|---|---:|---|---:|---:|
| `as_pas_only` | agreeableness | high | +0.022 | [+0.017, +0.028] | +0.000 | +0.035 |
| `as_pas_only` | agreeableness | low | -0.030 | [-0.035, -0.024] | +0.000 | +0.038 |
| `as_pas_only` | conscientiousness | high | +0.023 | [+0.019, +0.028] | +0.000 | +0.034 |
| `as_pas_only` | conscientiousness | low | -0.024 | [-0.028, -0.020] | +0.000 | +0.031 |
| `as_pas_only` | extraversion | high | +0.041 | [+0.037, +0.045] | +0.000 | +0.041 |
| `as_pas_only` | extraversion | low | -0.025 | [-0.028, -0.022] | +0.000 | +0.025 |
| `as_pas_only` | neuroticism | high | -0.032 | [-0.037, -0.028] | +0.000 | +0.040 |
| `as_pas_only` | neuroticism | low | +0.017 | [+0.012, +0.022] | +0.000 | +0.029 |
| `as_pas_only` | openness | high | +0.037 | [+0.032, +0.041] | +0.000 | +0.038 |
| `as_pas_only` | openness | low | -0.057 | [-0.064, -0.051] | +0.000 | +0.058 |
| `elaborate_prompt_as_pas` | agreeableness | high | -0.117 | [-0.126, -0.108] | +0.194 | +0.065 |
| `elaborate_prompt_as_pas` | agreeableness | low | +0.033 | [+0.026, +0.040] | +0.111 | +0.066 |
| `elaborate_prompt_as_pas` | conscientiousness | high | -0.020 | [-0.026, -0.015] | +0.083 | +0.058 |
| `elaborate_prompt_as_pas` | conscientiousness | low | -0.036 | [-0.045, -0.028] | +0.028 | +0.071 |
| `elaborate_prompt_as_pas` | extraversion | high | -0.056 | [-0.065, -0.049] | +0.125 | +0.069 |
| `elaborate_prompt_as_pas` | extraversion | low | +0.080 | [+0.074, +0.086] | +0.125 | +0.041 |
| `elaborate_prompt_as_pas` | neuroticism | high | -0.197 | [-0.207, -0.188] | +0.344 | +0.058 |
| `elaborate_prompt_as_pas` | neuroticism | low | -0.024 | [-0.030, -0.019] | +0.062 | +0.030 |
| `elaborate_prompt_as_pas` | openness | high | +0.030 | [+0.023, +0.037] | +0.025 | +0.056 |
| `elaborate_prompt_as_pas` | openness | low | +0.090 | [+0.080, +0.099] | +0.175 | +0.085 |

## 6. Implications For Next Experiment

The next experiment should test PAS as an actual final action selector or reranker, not only as a diagnostic agreement signal.

Recommended policy conditions:

| Condition | AS | Prompt | PAS final selector |
|---|---:|---:|---:|
| `baseline` | no | no | no |
| `prompt_baseline` | no | yes | no |
| `as_only` | yes | no | no |
| `pas_only` | no | no | yes |
| `as_pas` | yes | no | yes |
| `prompt_as` | yes | yes | no |
| `prompt_pas` | no | yes | yes |
| `prompt_as_pas` | yes | yes | yes |

For PAS selector conditions, use one of:

```text
final_action = pas_selected_action
```

or:

```text
final_action = PAS rerank among loglik-top-k actions
```

Primary reporting should remain separated: BFI movement, TRAIT movement, and co-movement. The gap metric should stay auxiliary.

PAS/loglik agreement remains useful as a diagnostic: moderate agreement means PAS is not merely copying log-likelihood selection and may produce different final actions if promoted to a policy.
