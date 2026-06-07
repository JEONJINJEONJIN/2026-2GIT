# V4 Final Rerun 20260526 Result Analysis

Run output:

```text
results/v4_bigfive/final_rerun_20260526/
```

## 1. Run Summary

This is a full v4 Big Five final rerun using the existing final configuration.
No experimental design changes were made.

| Field | Value |
|---|---|
| Config | `configs/experiments/v4_bigfive_final.yaml` |
| Model | `Qwen/Qwen2.5-3B-Instruct` |
| Split | `test` |
| Traits | `agreeableness`, `conscientiousness`, `neuroticism`, `openness`, `extraversion` |
| Target directions | `high`, `low` |
| Conditions | `baseline`, `one_line_prompt`, `elaborate_prompt`, `as_pas_only`, `elaborate_prompt_as_pas` |
| Alpha | `4.0` |
| Layers | `[18, 21, 24]` |
| Vector mode | `trait_contrast` for AS/PAS conditions |
| Vector file | `results/v4_bigfive/vectors/bigfive_trait_vectors_full_fp16.pt` |
| Seed | `42` |
| Git commit | `50098b5f7b94a96a25eb056f4176b7f518ac1ac4` |

Primary raw outputs:

| File | Rows | Purpose |
|---|---:|---|
| `bfi_scores.csv` | 2,200 | BFI-side self-report scoring rows |
| `trait_scores.csv` | 7,500 | TRAIT action-side log-likelihood scoring rows |
| `consistency_metrics.csv` | 7,500 | Combined BFI/TRAIT rows with gap behavior |
| `pas_loglik_agreement.csv` | 3,000 | PAS-selected action versus loglik-selected action diagnostics |
| `run_manifest.json` | 1 | Run provenance |

Post-processing outputs generated for this rerun:

| File | Purpose |
|---|---|
| `consistency_summary.csv` | Existing-style summary with gap behavior |
| `condition_effects_with_ci.csv` | Existing-style baseline-referenced deltas and bootstrap CIs |
| `pas_agreement_summary.csv` | PAS/loglik agreement by condition, trait, and direction |
| `separated_movement_summary.csv` | Primary separated BFI/TRAIT movement summary |
| `co_movement_analysis.md` | Co-movement bucket analysis |
| `analysis_report_v2.md` | Reframed separated-surface analysis |
| `final_report.md` | Existing-style auto-generated final report |

## 2. Methodological Framing

The most defensible interpretation is to report BFI-side movement and TRAIT-side movement separately.

The earlier combined metric was:

```text
speech_action_consistency = 1 - abs(BFI_score - TRAIT_score)
```

That gap metric is useful as an auxiliary signal, but it should not be the primary success criterion. BFI and TRAIT are different surfaces:

- `BFI_score`: self-report / speech-side surface
- `TRAIT_score`: action-side log-likelihood surface

The main question is therefore:

```text
Does BFI move toward target?
Does TRAIT move toward target?
Do the two surfaces move together?
```

Important caveat:

```text
PAS does not choose the final action in this run.
TRAIT_score is still based on log-likelihood action scoring.
PAS is diagnostic through PAS/loglik agreement.
```

## 3. Co-Movement Summary

Bucket definition:

| Flag | Meaning |
|---|---|
| `both_positive` | BFI and TRAIT deltas are both positive and both CIs exclude zero |
| `bfi_only` | only BFI delta is positive with CI excluding zero |
| `trait_only` | only TRAIT delta is positive with CI excluding zero |
| `neither` | neither surface has positive movement with CI excluding zero |

Overall condition summary:

| Condition | both_positive | bfi_only | trait_only | neither | Mean delta BFI | Mean delta TRAIT |
|---|---:|---:|---:|---:|---:|---:|
| `one_line_prompt` | 10 | 0 | 0 | 0 | +0.264 | +0.040 |
| `elaborate_prompt` | 10 | 0 | 0 | 0 | +0.127 | +0.043 |
| `as_pas_only` | 0 | 0 | 10 | 0 | +0.000 | +0.037 |
| `elaborate_prompt_as_pas` | 10 | 0 | 0 | 0 | +0.127 | +0.060 |

Interpretation:

- Prompted conditions move the BFI/self-report surface.
- AS/PAS moves the TRAIT/action surface.
- `as_pas_only` is pure action-surface movement: TRAIT moves, BFI does not.
- `elaborate_prompt_as_pas` moves both surfaces, but the BFI surface often moves more than the TRAIT surface.
- The two surfaces move independently enough that collapsing them into one gap number hides the actual result.

## 4. Primary Result: `as_pas_only`

`as_pas_only` is the cleanest test of AS/PAS without explicit persona prompt text.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Flag |
|---|---|---:|---|---:|---|---|
| agreeableness | high | +0.000 | [+0.000, +0.000] | +0.035 | [+0.030, +0.039] | trait_only |
| agreeableness | low | +0.000 | [+0.000, +0.000] | +0.038 | [+0.033, +0.044] | trait_only |
| conscientiousness | high | +0.000 | [+0.000, +0.000] | +0.034 | [+0.031, +0.038] | trait_only |
| conscientiousness | low | +0.000 | [+0.000, +0.000] | +0.031 | [+0.027, +0.035] | trait_only |
| extraversion | high | +0.000 | [+0.000, +0.000] | +0.041 | [+0.037, +0.045] | trait_only |
| extraversion | low | +0.000 | [+0.000, +0.000] | +0.025 | [+0.022, +0.029] | trait_only |
| neuroticism | high | +0.000 | [+0.000, +0.000] | +0.040 | [+0.037, +0.044] | trait_only |
| neuroticism | low | +0.000 | [+0.000, +0.000] | +0.029 | [+0.025, +0.033] | trait_only |
| openness | high | +0.000 | [+0.000, +0.000] | +0.038 | [+0.033, +0.042] | trait_only |
| openness | low | +0.000 | [+0.000, +0.000] | +0.058 | [+0.051, +0.066] | trait_only |

Interpretation:

`as_pas_only` reliably moves the action surface. All 10 trait/direction cells show positive TRAIT target movement with confidence intervals above zero. BFI does not move because this condition has no persona prompt, so every cell is `trait_only`.

This is strong evidence that AS/PAS changes the action-side likelihood surface. It is not evidence that AS/PAS alone changes the self-report/speech surface.

## 5. Primary Result: `elaborate_prompt_as_pas`

This condition combines strong persona prompting with AS/PAS.

| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs movement gap | Flag |
|---|---|---:|---|---:|---|---:|---|
| agreeableness | high | +0.194 | [+0.194, +0.194] | +0.065 | [+0.058, +0.073] | 0.129 | both_positive |
| agreeableness | low | +0.111 | [+0.111, +0.111] | +0.066 | [+0.057, +0.075] | 0.045 | both_positive |
| conscientiousness | high | +0.083 | [+0.083, +0.083] | +0.058 | [+0.052, +0.064] | 0.025 | both_positive |
| conscientiousness | low | +0.028 | [+0.028, +0.028] | +0.071 | [+0.061, +0.079] | 0.043 | both_positive |
| extraversion | high | +0.125 | [+0.125, +0.125] | +0.069 | [+0.060, +0.078] | 0.056 | both_positive |
| extraversion | low | +0.125 | [+0.125, +0.125] | +0.041 | [+0.034, +0.048] | 0.084 | both_positive |
| neuroticism | high | +0.344 | [+0.344, +0.344] | +0.058 | [+0.053, +0.065] | 0.285 | both_positive |
| neuroticism | low | +0.062 | [+0.062, +0.062] | +0.030 | [+0.026, +0.035] | 0.032 | both_positive |
| openness | high | +0.025 | [+0.025, +0.025] | +0.056 | [+0.049, +0.064] | 0.031 | both_positive |
| openness | low | +0.175 | [+0.175, +0.175] | +0.085 | [+0.076, +0.095] | 0.090 | both_positive |

Interpretation:

`elaborate_prompt_as_pas` moves both BFI and TRAIT in all 10 cells, but not at the same magnitude. The clearest decoupling case is:

```text
neuroticism high:
delta_bfi_target   = +0.344
delta_trait_target = +0.058
```

This is the strongest example of speech/action surface decoupling in the rerun. The prompt strongly moves the self-report surface, while the action surface moves more modestly.

## 6. Gap Behavior As Auxiliary Signal

The gap metric remains useful for describing whether the two surfaces got closer or farther apart, but it should not be treated as the primary success metric.

For `as_pas_only`, gap behavior is asymmetric:

| Trait | Target | Delta gap behavior | Gap 95% CI | Delta TRAIT target |
|---|---|---:|---|---:|
| agreeableness | high | +0.022 | [+0.017, +0.028] | +0.035 |
| agreeableness | low | -0.030 | [-0.035, -0.024] | +0.038 |
| conscientiousness | high | +0.023 | [+0.019, +0.028] | +0.034 |
| conscientiousness | low | -0.024 | [-0.028, -0.020] | +0.031 |
| extraversion | high | +0.041 | [+0.037, +0.045] | +0.041 |
| extraversion | low | -0.025 | [-0.028, -0.022] | +0.025 |
| neuroticism | high | -0.032 | [-0.037, -0.028] | +0.040 |
| neuroticism | low | +0.017 | [+0.012, +0.022] | +0.029 |
| openness | high | +0.037 | [+0.032, +0.041] | +0.038 |
| openness | low | -0.057 | [-0.064, -0.051] | +0.058 |

Interpretation:

The gap improves on:

```text
A-high, C-high, E-high, N-low, O-high
```

The gap decreases on the opposite directions:

```text
A-low, C-low, E-low, N-high, O-low
```

This should not be framed as AS/PAS failing on those traits. It is better interpreted as evidence that the model's baseline self-report surface is already shifted by model alignment and baseline tendencies. AS/PAS moves the action surface, but the self-report surface does not necessarily move with it.

## 7. PAS / Loglik Agreement

PAS is diagnostic in this run.

| Condition | Mean PAS/loglik agreement | Min | Max |
|---|---:|---:|---:|
| `as_pas_only` | 0.399 | 0.307 | 0.473 |
| `elaborate_prompt_as_pas` | 0.476 | 0.387 | 0.567 |

Interpretation:

PAS and loglik are not the same decision signal. Agreement is moderate, not high. If PAS is promoted from diagnostic to final selector or reranker, the final selected actions may differ materially from the current loglik-based `TRAIT_score`.

## 8. Main Result

The full rerun supports this claim:

> AS/PAS reliably moves the TRAIT/action surface toward target Big Five directions. Prompted conditions move the BFI/self-report surface, often more strongly than the action surface. The two surfaces move independently, so they should be reported separately.

It does not support this claim:

> AS/PAS reduces speech-action mismatch.

More precise:

> AS/PAS is an action-surface control mechanism. Prompting is a self-report/speech-surface control mechanism. A speech-action consistency system needs a policy that explicitly coordinates the two.

## 9. Implication For The Next Experiment

The next experiment should test PAS as an actual action policy, not only as a diagnostic.

Recommended condition matrix:

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

For PAS selector conditions:

```text
final_action = pas_selected_action
```

or:

```text
final_action = PAS rerank among loglik-top-k actions
```

Primary reporting should stay separated:

- BFI movement
- TRAIT movement
- co-movement flag
- PAS/loglik agreement
- gap behavior only as auxiliary

## 10. Bottom Line

The result is not "AS/PAS solved speech-action mismatch." The result is sharper:

> AS/PAS robustly moves the action side. Prompting robustly moves the self-report side. The two do not naturally lock together. The next system step is to make PAS an actual final action selector or reranker and then test whether that coordination reduces mismatch without hiding the two surfaces inside a single gap metric.
