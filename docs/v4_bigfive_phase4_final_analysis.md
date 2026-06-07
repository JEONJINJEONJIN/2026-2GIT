# V4 Big Five Phase 4 Final Analysis

## Status

Phase 3 full Big Five test run is complete and analyzed.

Final run output:

```text
results/v4_bigfive/final_metrics/
```

Key files:

```text
results/v4_bigfive/final_metrics/consistency_summary.csv
results/v4_bigfive/final_metrics/condition_differences.csv
results/v4_bigfive/final_metrics/condition_effects_with_ci.csv
results/v4_bigfive/final_metrics/paired_bootstrap_ci.csv
results/v4_bigfive/final_metrics/pas_agreement_summary.csv
results/v4_bigfive/final_metrics/final_report.md
results/v4_bigfive/final_metrics/run_manifest.json
```

The final run used:

```yaml
model: Qwen/Qwen2.5-3B-Instruct
split: test
traits:
  - agreeableness
  - conscientiousness
  - extraversion
  - neuroticism
  - openness
target_directions: [high, low]
conditions:
  - baseline
  - one_line_prompt
  - elaborate_prompt
  - as_pas_only
  - elaborate_prompt_as_pas
alpha: 4.0
layers: [18, 21, 24]
seed: 42
```

The alpha/layers were fixed from Phase 2 dev results before the test run.

## Research Question

> Under what conditions does AS+PAS affect speech-action persona consistency
> for NPCs assigned a Big Five persona?

This remains the correct framing. The final results do not support a universal
"AS+PAS reduces the speech-action gap" claim. They do support a narrower and
more defensible claim:

> AS+PAS reliably moves TRAIT behavior in the target persona direction, but its
> effect on speech-action consistency depends on trait, direction, and prompt
> condition.

## Metric Definition And Scoring Caveat

The primary consistency metric is:

```text
consistency = 1 - abs(BFI_score - TRAIT_score)
```

Both scores are normalized to `[0, 1]` on the same high-direction Big Five axis.
This metric is unsigned: it measures how close the speech score and behavior
score are, not whether either score moved toward the requested target.

Target-direction movement must be interpreted through:

```text
target_attainment = target_direction_score - 0.5
```

For low targets, `target_direction_score = 1 - high_direction_score`.

Current final-run caveat:

- The primary `TRAIT_score` is computed from conditional log-likelihood.
- PAS outputs are reported through `pas_loglik_agreement.csv`.
- PAS does not overwrite the primary `TRAIT_score` in this run.
- The current five-condition matrix compares prompting and AS+PAS conditions,
  but it does not contain pure AS-only or pure PAS-only ablations.

## Primary Result: AS+PAS-Only

Baseline-referenced `as_pas_only` effects:

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

Bootstrap intervals use 500 paired scenario resamples. In this final run, all
listed `as_pas_only` consistency and TRAIT-target intervals exclude zero.

Interpretation:

- AS+PAS-only increases TRAIT target attainment for every trait and direction.
- Consistency improves when the TRAIT shift moves behavior closer to the BFI
  score.
- Consistency decreases when the TRAIT shift moves behavior away from the BFI
  score, even if it moves behavior toward the requested target direction.
- This is not a failure of the metric. It is exactly why target attainment must
  be reported beside consistency.

## Prompting and AS+PAS Combination

`elaborate_prompt_as_pas` often increases TRAIT target attainment more than
prompting alone, but it does not always improve consistency.

The full baseline-referenced condition comparison is stored in:

```text
results/v4_bigfive/final_metrics/condition_effects_with_ci.csv
```

This table includes every non-baseline condition:

```text
one_line_prompt
elaborate_prompt
as_pas_only
elaborate_prompt_as_pas
```

Selected baseline-referenced effects:

| Trait | Target | Condition | Delta consistency | Delta BFI target | Delta TRAIT target |
|---|---:|---|---:|---:|---:|
| Agreeableness | high | elaborate_prompt_as_pas | -0.117 | +0.194 | +0.065 |
| Agreeableness | low | elaborate_prompt_as_pas | +0.033 | +0.111 | +0.066 |
| Conscientiousness | high | elaborate_prompt_as_pas | -0.020 | +0.083 | +0.058 |
| Extraversion | low | elaborate_prompt_as_pas | +0.080 | +0.125 | +0.041 |
| Neuroticism | high | elaborate_prompt_as_pas | -0.197 | +0.344 | +0.058 |
| Openness | low | elaborate_prompt_as_pas | +0.090 | +0.175 | +0.085 |

Interpretation:

- Prompting strongly changes BFI self-report.
- AS+PAS changes TRAIT behavior.
- When prompting changes speech more than AS+PAS changes behavior, the
  speech-action gap can grow.
- When both move in compatible amounts, consistency improves.
- Therefore AS+PAS is complementary to prompting for behavior movement, but not
  guaranteed to be complementary for speech-action consistency.

## Trait-Direction Pattern

The final result reveals a consistent asymmetry:

- For Agreeableness, Conscientiousness, Extraversion, and Openness:
  - high target: AS+PAS-only improves consistency.
  - low target: AS+PAS-only lowers consistency while improving TRAIT target
    attainment.
- For Neuroticism:
  - high target: AS+PAS-only lowers consistency while improving TRAIT target
    attainment.
  - low target: AS+PAS-only improves consistency.

Likely explanation:

- Baseline BFI scores are not centered at 0.5 for every trait.
- If the model's self-report already leans high or low for a trait, steering
  behavior toward the opposite target can improve target attainment while
  increasing the speech-action gap.
- This reinforces the need to treat consistency and target attainment as
  distinct metrics.

## PAS and Log-Likelihood Agreement

PAS/loglik agreement rates for AS+PAS conditions are moderate rather than high.

Examples:

| Condition | Trait | Target | Agreement |
|---|---|---:|---:|
| as_pas_only | Agreeableness | high | 0.307 |
| as_pas_only | Agreeableness | low | 0.473 |
| as_pas_only | Conscientiousness | high | 0.473 |
| as_pas_only | Extraversion | low | 0.447 |
| as_pas_only | Openness | low | 0.460 |
| elaborate_prompt_as_pas | Agreeableness | low | 0.567 |
| elaborate_prompt_as_pas | Openness | low | 0.520 |

Interpretation:

- PAS and log-likelihood are related but not redundant decision signals.
- The moderate agreement is useful: PAS is not merely copying log-likelihood
  choice behavior.
- For the paper, report PAS/loglik agreement as a diagnostic, not as the main
  success metric.

## Main Takeaways

1. AS+PAS reliably affects behavior.

   Across all five Big Five traits and both target directions, AS+PAS-only
   improves TRAIT target attainment relative to baseline.

2. Speech-action consistency is conditional.

   AS+PAS improves consistency for some trait/direction pairs and worsens it
   for others. This supports the exploratory framing and rules out a universal
   improvement claim.

3. Prompting and AS+PAS act on different surfaces.

   Prompting moves BFI self-report strongly. AS+PAS moves TRAIT behavior.
   Combining them can help behavior movement, but it can also increase the gap
   if speech shifts more than behavior.

4. Direction matters.

   High/low directions are not symmetric because baseline BFI and baseline
   TRAIT scores are not centered the same way for every trait.

5. The revised design was necessary.

   If the study had only measured behavior movement, the conclusion would look
   uniformly positive. If it had only measured consistency, behavior improvement
   would be hidden. The paired BFI/TRAIT design exposes the actual tradeoff.

## Recommended Paper Framing

Use:

> We investigate when activation steering plus persona-action selection affects
> speech-action consistency for Big Five NPC personas.

Use:

> AS+PAS consistently shifts action choices toward target trait directions, but
> its effect on speech-action consistency depends on trait direction and prompt
> condition.

Avoid:

> AS+PAS makes NPCs more consistent.

Avoid:

> AS+PAS reduces the speech-action gap.

Avoid:

> The model has a real Big Five personality.

Avoid:

> High trait directions are better than low trait directions.

## Suggested Result Narrative

The final analysis suggests that AS+PAS is best understood as a behavioral
intervention rather than a direct consistency optimizer. It can move decisions
in trait-consistent directions, but whether this improves NPC immersion depends
on whether the model's self-reported persona moves with it. In several high
trait conditions, behavior movement and BFI self-report are aligned, increasing
speech-action consistency. In several low trait conditions, behavior moves
toward the target but away from the model's self-report, lowering consistency.

This finding is useful because it clarifies what AS+PAS can and cannot solve.
It can correct action choice tendencies, but it does not automatically align
the model's spoken self-description with those choices. For NPC systems, this
means activation-level and selection-level interventions should be evaluated
with both target attainment and speech-action consistency, not either metric
alone.

## Remaining Work

1. Run side-effect response generation on the final split if final-response
   examples are needed for the paper or demo.
2. Add control-vector diagnostics for the full five-trait test setting if the
   paper needs test-split control tables beyond the Phase 2 dev diagnostics.
3. Prepare the Streamlit demo using the final result examples.

## Figure Inventory

Final figures are stored in:

```text
results/v4_bigfive/final_figures/
```

Each figure is exported as both PNG and SVG.

| Figure | File stem | Purpose |
|---|---|---|
| AS+PAS consistency delta | `01_as_pas_consistency_delta` | Shows that AS+PAS-only improves consistency for some trait/direction pairs and worsens it for others. |
| AS+PAS TRAIT target delta | `02_as_pas_trait_target_delta` | Shows that AS+PAS-only moves behavior toward the target direction across all traits/directions. |
| Condition consistency heatmap | `03_condition_consistency_heatmap` | Compares prompt-only, AS+PAS-only, and combined conditions against baseline. |
| BFI vs TRAIT scatter | `04_bfi_trait_scatter` | Visualizes the speech-action gap directly on the shared `[0, 1]` trait scale. |
| PAS/loglik agreement | `05_pas_loglik_agreement` | Reports whether PAS and log-likelihood select the same behavior. |

Recommended paper usage:

- Use Figure 1 and Figure 2 together. The contrast is the core result:
  behavior movement is broad, but consistency improvement is conditional.
- Use the heatmap to explain why prompt-only or combined conditions can worsen
  consistency despite improving target attainment.
- Use the BFI/TRAIT scatter when introducing the measurement design.
- Use PAS/loglik agreement as a diagnostic figure, not as a headline result.
