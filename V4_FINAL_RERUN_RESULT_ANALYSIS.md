# V4 Final Rerun Result Analysis

Run output: `results/v4_bigfive/final_rerun_20260525/`

## 1. Run Summary

This is a full rerun of the existing v4 Big Five final configuration.

Config:

```text
configs/experiments/v4_bigfive_final.yaml
```

Model:

```text
Qwen/Qwen2.5-3B-Instruct
```

Run settings:

| Field | Value |
|---|---|
| split | `test` |
| traits | `agreeableness`, `conscientiousness`, `neuroticism`, `openness`, `extraversion` |
| target directions | `high`, `low` |
| conditions | `baseline`, `one_line_prompt`, `elaborate_prompt`, `as_pas_only`, `elaborate_prompt_as_pas` |
| alpha | `4.0` |
| layers | `[18, 21, 24]` |
| vector mode | `trait_contrast` for AS/PAS conditions |
| seed | `42` |
| vector file | `results/v4_bigfive/vectors/bigfive_trait_vectors_full_fp16.pt` |

Generated files:

| File | Rows | Purpose |
|---|---:|---|
| `bfi_scores.csv` | 2,200 | BFI-style speech/self-report scoring rows |
| `trait_scores.csv` | 7,500 | TRAIT action-side log-likelihood scoring rows |
| `consistency_metrics.csv` | 7,500 | Combined BFI/TRAIT speech-action consistency rows |
| `pas_loglik_agreement.csv` | 3,000 | PAS-selected action versus loglik-selected action diagnostics |
| `run_manifest.json` | 1 | Run provenance |

Analysis files generated after the run:

| File | Purpose |
|---|---|
| `consistency_summary.csv` | Mean consistency and target-attainment by condition/trait/direction |
| `condition_differences.csv` | Baseline-referenced deltas |
| `condition_effects_with_ci.csv` | Baseline-referenced deltas with bootstrap confidence intervals |
| `paired_scenario_differences.csv` | Scenario-paired condition differences |
| `paired_bootstrap_ci.csv` | Bootstrap intervals |
| `pas_agreement_summary.csv` | PAS/loglik agreement by condition/trait/direction |
| `analysis_report.md` | Auto-generated detailed analysis |
| `final_report.md` | Auto-generated final report |

## 2. What This Experiment Actually Tests

This run tests whether AS/PAS conditions affect Big Five NPC speech-action consistency.

The experiment treats:

```text
speech side = BFI_score
action side = TRAIT_score
```

Primary metric:

```text
speech_action_consistency = 1 - abs(BFI_score - TRAIT_score)
```

Secondary target-direction metric:

```text
target_attainment = target_direction_score - 0.5
```

Important caveat:

```text
TRAIT_score is still computed from log-likelihood action scoring.
PAS is stored as a diagnostic agreement signal.
PAS does not choose the final action in this run.
```

So this result can answer:

> Does AS/PAS-conditioned inference move action-side TRAIT scores, and how does that affect BFI-TRAIT consistency?

It cannot answer:

> If PAS directly selects the final action, does speech-action mismatch decrease?

That requires a separate `final_action = pas_selected_action` or PAS-rerank experiment.

## 3. High-Level Result

Aggregate baseline-referenced effects:

| Condition | Positive consistency cells | Positive TRAIT target cells | Mean delta consistency | Mean delta TRAIT target |
|---|---:|---:|---:|---:|
| `one_line_prompt` | 3 / 10 | 10 / 10 | -0.134 | +0.040 |
| `elaborate_prompt` | 4 / 10 | 10 / 10 | -0.029 | +0.043 |
| `as_pas_only` | 5 / 10 | 10 / 10 | -0.003 | +0.037 |
| `elaborate_prompt_as_pas` | 4 / 10 | 10 / 10 | -0.022 | +0.060 |

Main interpretation:

1. Every non-baseline condition improves TRAIT target attainment in all 10 trait/direction cells.
2. AS/PAS reliably moves the action-side score toward the requested target direction.
3. Speech-action consistency does not uniformly improve.
4. Consistency improves only when the action-side movement reduces the BFI-TRAIT gap.
5. Prompting moves BFI self-report strongly, which can either help or hurt consistency depending on whether TRAIT behavior moves with it.

The strongest defensible claim is:

> AS/PAS moves action-side TRAIT behavior toward target Big Five directions, but its effect on speech-action consistency is conditional by trait, direction, and prompt condition.

Avoid claiming:

> AS/PAS always reduces speech-action mismatch.

## 4. `as_pas_only` Result

`as_pas_only` is the cleanest condition for seeing AS/PAS without persona prompt text.

| Trait | Target | Delta consistency | Consistency 95% CI | Delta BFI target | Delta TRAIT target | TRAIT target 95% CI |
|---|---|---:|---|---:|---:|---|
| agreeableness | high | +0.022 | [+0.017, +0.028] | +0.000 | +0.035 | [+0.030, +0.039] |
| agreeableness | low | -0.030 | [-0.035, -0.024] | +0.000 | +0.038 | [+0.033, +0.044] |
| conscientiousness | high | +0.023 | [+0.019, +0.028] | +0.000 | +0.034 | [+0.031, +0.038] |
| conscientiousness | low | -0.024 | [-0.028, -0.020] | +0.000 | +0.031 | [+0.027, +0.035] |
| extraversion | high | +0.041 | [+0.037, +0.045] | +0.000 | +0.041 | [+0.037, +0.045] |
| extraversion | low | -0.025 | [-0.028, -0.022] | +0.000 | +0.025 | [+0.022, +0.028] |
| neuroticism | high | -0.032 | [-0.037, -0.028] | +0.000 | +0.040 | [+0.037, +0.044] |
| neuroticism | low | +0.017 | [+0.012, +0.022] | +0.000 | +0.029 | [+0.025, +0.033] |
| openness | high | +0.037 | [+0.032, +0.041] | +0.000 | +0.038 | [+0.033, +0.042] |
| openness | low | -0.057 | [-0.064, -0.051] | +0.000 | +0.058 | [+0.051, +0.065] |

Interpretation:

- TRAIT target attainment improves in all 10 cells.
- Consistency improves in 5 of 10 cells.
- Consistency decreases in the other 5 cells even though behavior moves toward the target.
- This happens because consistency is a gap metric, not a target-success metric.

The key pattern:

| Pattern | Traits/directions |
|---|---|
| Consistency improves | agreeableness high, conscientiousness high, extraversion high, neuroticism low, openness high |
| Consistency decreases | agreeableness low, conscientiousness low, extraversion low, neuroticism high, openness low |

This suggests the model's baseline BFI self-report is not centered symmetrically. When AS/PAS pushes behavior toward a direction that the BFI side does not share, the action target score improves but speech-action consistency can decrease.

## 5. `elaborate_prompt_as_pas` Result

This condition combines strong persona prompting with AS/PAS.

| Trait | Target | Delta consistency | Consistency 95% CI | Delta BFI target | Delta TRAIT target | TRAIT target 95% CI |
|---|---|---:|---|---:|---:|---|
| agreeableness | high | -0.117 | [-0.126, -0.108] | +0.194 | +0.065 | [+0.058, +0.073] |
| agreeableness | low | +0.033 | [+0.026, +0.040] | +0.111 | +0.066 | [+0.057, +0.076] |
| conscientiousness | high | -0.020 | [-0.026, -0.015] | +0.083 | +0.058 | [+0.053, +0.064] |
| conscientiousness | low | -0.036 | [-0.045, -0.028] | +0.028 | +0.071 | [+0.062, +0.081] |
| extraversion | high | -0.056 | [-0.065, -0.049] | +0.125 | +0.069 | [+0.060, +0.076] |
| extraversion | low | +0.080 | [+0.074, +0.086] | +0.125 | +0.041 | [+0.034, +0.047] |
| neuroticism | high | -0.197 | [-0.207, -0.188] | +0.344 | +0.058 | [+0.052, +0.065] |
| neuroticism | low | -0.024 | [-0.030, -0.019] | +0.062 | +0.030 | [+0.025, +0.035] |
| openness | high | +0.030 | [+0.023, +0.037] | +0.025 | +0.056 | [+0.049, +0.063] |
| openness | low | +0.090 | [+0.080, +0.099] | +0.175 | +0.085 | [+0.075, +0.095] |

Interpretation:

- The combined condition has stronger mean TRAIT target movement than `as_pas_only`:

```text
as_pas_only mean delta TRAIT target = +0.037
elaborate_prompt_as_pas mean delta TRAIT target = +0.060
```

- However, mean consistency is still negative:

```text
elaborate_prompt_as_pas mean delta consistency = -0.022
```

Reason:

- Elaborate prompting moves BFI self-report strongly.
- AS/PAS moves TRAIT behavior, but often by a smaller amount.
- If BFI moves more than TRAIT, the gap can increase even when both move in the target direction.

So prompt + AS/PAS is better for target-direction behavior movement, but not automatically better for speech-action consistency.

## 6. PAS/Loglik Agreement

PAS/loglik agreement measures whether PAS and log-likelihood chose the same concrete action.

| Condition | Mean agreement | Min | Max |
|---|---:|---:|---:|
| `as_pas_only` | 0.399 | 0.307 | 0.473 |
| `elaborate_prompt_as_pas` | 0.476 | 0.387 | 0.567 |

Detailed agreement:

| Condition | Trait | Target | n | Agreement |
|---|---|---|---:|---:|
| `as_pas_only` | agreeableness | high | 150 | 0.307 |
| `as_pas_only` | agreeableness | low | 150 | 0.473 |
| `as_pas_only` | conscientiousness | high | 150 | 0.473 |
| `as_pas_only` | conscientiousness | low | 150 | 0.347 |
| `as_pas_only` | extraversion | high | 150 | 0.367 |
| `as_pas_only` | extraversion | low | 150 | 0.447 |
| `as_pas_only` | neuroticism | high | 150 | 0.307 |
| `as_pas_only` | neuroticism | low | 150 | 0.447 |
| `as_pas_only` | openness | high | 150 | 0.367 |
| `as_pas_only` | openness | low | 150 | 0.460 |
| `elaborate_prompt_as_pas` | agreeableness | high | 150 | 0.387 |
| `elaborate_prompt_as_pas` | agreeableness | low | 150 | 0.567 |
| `elaborate_prompt_as_pas` | conscientiousness | high | 150 | 0.493 |
| `elaborate_prompt_as_pas` | conscientiousness | low | 150 | 0.487 |
| `elaborate_prompt_as_pas` | extraversion | high | 150 | 0.473 |
| `elaborate_prompt_as_pas` | extraversion | low | 150 | 0.493 |
| `elaborate_prompt_as_pas` | neuroticism | high | 150 | 0.420 |
| `elaborate_prompt_as_pas` | neuroticism | low | 150 | 0.487 |
| `elaborate_prompt_as_pas` | openness | high | 150 | 0.433 |
| `elaborate_prompt_as_pas` | openness | low | 150 | 0.520 |

Interpretation:

- PAS and loglik are related but not equivalent.
- Agreement is moderate, not high.
- PAS has a distinct persona-projection signal.
- If PAS were used as the final action selector, the selected actions could differ substantially from the current primary `TRAIT_score` path.

This is why the next experiment should not merely report PAS/loglik agreement. It should run actual PAS-final-selection conditions.

## 7. Main Answer to the Research Question

Research question:

> Under what conditions does AS+PAS affect speech-action persona consistency for NPCs assigned a Big Five persona?

Answer from this rerun:

> AS/PAS reliably affects the action side by moving TRAIT target attainment upward across all traits and directions. However, speech-action consistency is conditional. It improves when action-side movement reduces the BFI-TRAIT gap and decreases when action-side movement moves behavior away from the BFI self-report score.

In plain terms:

- AS/PAS can push what the NPC does.
- Prompting can push what the NPC says or self-reports.
- These two surfaces do not always move together.
- Therefore AS/PAS is not yet a full solution to speech-action mismatch.
- It is a strong component for action-side control.

## 8. What This Means for the Current System

The current system is aligned with the problem direction, but it is not yet the final solution.

What it currently does well:

- Extracts trait/persona vectors.
- Applies activation steering.
- Measures BFI-side speech/self-report.
- Measures TRAIT-side action preference.
- Stores PAS as a distinct cosine-projection diagnostic.
- Shows that action-side target movement is robust.

What it does not yet do:

- It does not let PAS directly choose the final action.
- It does not run `pas_only` versus `as_only` versus `as_pas` as separate final-action policies.
- It does not include a mismatch repair loop such as:

```text
generate speech
score candidate actions
choose or rerank action by PAS
verify speech-action consistency
regenerate or rerank if mismatch remains
```

## 9. Recommended Next Experiment

To directly test the user's intended system goal, add a PAS-final-selector experiment.

Required condition matrix:

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
final_action = PAS rerank among plausible/loglik-top-k actions
```

Recommended primary metrics:

- `speech_action_consistency`
- `trait_target_attainment`
- `bfi_target_attainment`
- PAS/loglik agreement
- action diversity
- scenario-level validity or human/manual sanity checks

The cleanest next step is to implement a runner option like:

```text
--action_policy loglik|pas|pas_topk_rerank
```

Then rerun at least:

```text
baseline
as_only
pas_only
as_pas
```

This would directly answer whether PAS as an actual action policy reduces speech-action mismatch.

## 10. Final Takeaway

Use this claim:

> The v4 final rerun confirms that AS/PAS conditions consistently move action-side TRAIT scores toward target Big Five directions. However, speech-action consistency improves only in some trait/direction settings because the speech-side BFI score and action-side TRAIT score do not always move together.

Do not use this claim:

> AS/PAS solved speech-action mismatch.

More accurate:

> The current system provides an action-control mechanism and a consistency measurement framework. To become a direct speech-action mismatch solution, PAS must be evaluated as an actual final action selector or reranker, not only as a diagnostic agreement signal.
