# V4 Big Five Phase 2 Summary

## Purpose

Phase 2 tested whether the v4 Big Five design is ready to move from an
Agreeableness-only pilot to the full five-trait experiment.

The guiding question remains:

> Under what conditions does AS+PAS affect speech-action persona consistency
> for NPCs assigned a Big Five persona?

This phase does not claim that AS+PAS universally improves consistency. The
goal is to check whether the effect is measurable, whether it survives control
diagnostics, whether alpha/layer choices can be fixed on dev data, and whether
side effects are acceptable before Phase 3.

## Completed Runs

| Run | Scope | Output |
|---|---|---|
| Main pilot | Agreeableness, high/low, 50 dev scenarios, 5 conditions | `results/v4_bigfive/pilot_metrics_format_fixed/` |
| Control diagnostics | zero hook, random vector, unrelated trait vector, AS+PAS | `results/v4_bigfive/pilot_control_metrics/` |
| Alpha sweep | AS+PAS only, middle layers, alpha 0.5/1.0/2.0/4.0 | `results/v4_bigfive/alpha_sweep/alpha_sweep_summary.csv` |
| Layer sweep | AS+PAS only, alpha 4.0, early/middle/late layers | `results/v4_bigfive/layer_sweep/layer_sweep_summary.csv` |
| Format-fix validation | Full pilot rerun after stricter output prompt | `results/v4_bigfive/pilot_metrics_format_fixed/side_effect_summary.csv` |

All runs used the dev split. Test data has not been used for tuning.

## Main Pilot Result

Baseline-referenced effects from the format-fixed pilot:

| Condition | Target | Delta consistency | Delta BFI target | Delta TRAIT target |
|---|---:|---:|---:|---:|
| `as_pas_only` | high | +0.018 | +0.000 | +0.037 |
| `as_pas_only` | low | -0.040 | +0.000 | +0.046 |
| `elaborate_prompt` | high | -0.125 | +0.194 | +0.063 |
| `elaborate_prompt` | low | +0.024 | +0.111 | +0.052 |
| `elaborate_prompt_as_pas` | high | -0.114 | +0.194 | +0.074 |
| `elaborate_prompt_as_pas` | low | +0.020 | +0.111 | +0.066 |
| `one_line_prompt` | high | -0.108 | +0.167 | +0.052 |
| `one_line_prompt` | low | -0.288 | +0.472 | +0.057 |

Interpretation:

- AS+PAS changes TRAIT behavior in the target direction for both high and low
  Agreeableness.
- Consistency does not move uniformly. High Agreeableness improves under
  AS+PAS-only, while low Agreeableness shows better target behavior but lower
  speech-action consistency.
- Prompting strongly moves BFI self-report. This can increase the gap when
  action choices do not move by the same amount.
- The result supports the exploratory framing: AS+PAS affects the speech-action
  gap under some conditions, not universally.

## Control Diagnostics

Zero-hook-referenced control results:

| Condition | Target | Delta consistency | Delta TRAIT target |
|---|---:|---:|---:|
| `as_pas_only` | high | +0.018 | +0.037 |
| `as_pas_only` | low | -0.040 | +0.046 |
| `random_vector` | high | +0.004 | +0.002 |
| `random_vector` | low | +0.006 | -0.006 |
| `unrelated_trait_vector` | high | -0.000 | +0.000 |
| `unrelated_trait_vector` | low | +0.002 | -0.000 |

Bootstrap intervals for consistency:

| Condition | Target | Mean delta | 95% CI |
|---|---:|---:|---:|
| `as_pas_only` | high | +0.018 | [+0.010, +0.027] |
| `as_pas_only` | low | -0.040 | [-0.051, -0.031] |
| `random_vector` | high | +0.004 | [+0.001, +0.008] |
| `random_vector` | low | +0.006 | [+0.002, +0.009] |
| `unrelated_trait_vector` | high | -0.000 | [-0.006, +0.006] |
| `unrelated_trait_vector` | low | +0.002 | [-0.003, +0.008] |

Interpretation:

- The unrelated trait vector is effectively null.
- The random vector creates a small perturbation effect, but it is much smaller
  than the AS+PAS high-direction effect and does not reproduce the target
  behavior pattern.
- The diagnostic supports treating the AS+PAS effect as trait-specific enough
  to justify Phase 3, while still reporting random-vector sensitivity.

## Alpha Sweep

AS+PAS-only, middle layers `[18, 21, 24]`:

| Alpha | Target | Consistency | TRAIT target | PAS/loglik agreement |
|---:|---:|---:|---:|---:|
| 0.5 | high | 0.938 | 0.002 | 0.28 |
| 0.5 | low | 0.928 | 0.012 | 0.28 |
| 1.0 | high | 0.942 | 0.008 | 0.28 |
| 1.0 | low | 0.922 | 0.019 | 0.34 |
| 2.0 | high | 0.946 | 0.018 | 0.28 |
| 2.0 | low | 0.911 | 0.032 | 0.38 |
| 4.0 | high | 0.952 | 0.032 | 0.32 |
| 4.0 | low | 0.893 | 0.051 | 0.46 |

Interpretation:

- Higher alpha produces stronger target-direction behavioral movement.
- For high Agreeableness, alpha 4.0 improves both behavior and consistency.
- For low Agreeableness, alpha 4.0 improves behavior the most but lowers
  consistency because BFI self-report remains comparatively high.
- Since Phase 3 is meant to test whether AS+PAS can move behavior and how that
  affects the speech-action gap, alpha 4.0 is the most informative dev-selected
  value.

## Layer Sweep

AS+PAS-only, alpha 4.0:

| Layer group | Layers | Target | Consistency | TRAIT target | PAS/loglik agreement |
|---|---|---:|---:|---:|---:|
| early | `[6, 9]` | high | 0.938 | 0.000 | 0.18 |
| early | `[6, 9]` | low | 0.931 | 0.010 | 0.30 |
| middle | `[18, 21, 24]` | high | 0.952 | 0.032 | 0.32 |
| middle | `[18, 21, 24]` | low | 0.893 | 0.051 | 0.46 |
| late | `[30, 33, 35]` | high | 0.936 | -0.002 | 0.20 |
| late | `[30, 33, 35]` | low | 0.930 | 0.008 | 0.30 |

Interpretation:

- Middle layers produce the clearest target-direction behavioral movement.
- Early and late layers are more stable in consistency but mostly fail to move
  behavior.
- For Phase 3, middle layers are the right dev-selected setting because the
  experiment needs a detectable intervention, not merely a stable no-op.

## Side Effects

The response-format prompt was tightened after the first pilot because the
model often omitted the `<Action>...</Action>` tag.

After the fix:

| Condition | Target | Parse success | Format validity |
|---|---:|---:|---:|
| `baseline` | high | 1.00 | 1.00 |
| `baseline` | low | 1.00 | 1.00 |
| `one_line_prompt` | high | 1.00 | 1.00 |
| `one_line_prompt` | low | 0.98 | 0.98 |
| `elaborate_prompt` | high | 1.00 | 1.00 |
| `elaborate_prompt` | low | 1.00 | 1.00 |
| `as_pas_only` | high | 1.00 | 1.00 |
| `as_pas_only` | low | 1.00 | 1.00 |
| `elaborate_prompt_as_pas` | high | 1.00 | 1.00 |
| `elaborate_prompt_as_pas` | low | 1.00 | 1.00 |

The format fix did not change BFI scores, TRAIT scores, consistency, or target
attainment. It only stabilized generated response formatting.

## Phase 2 Decision

Phase 2 passes with caveats.

Reasons to proceed:

- TRAIT data, BFI scoring, manifest capture, vector extraction, and analysis
  scripts are all operational.
- AS+PAS produces measurable target-direction behavior movement on the dev
  pilot.
- Control diagnostics do not reproduce the main effect with an unrelated trait
  vector.
- Alpha/layer settings can be fixed from dev data before touching test.
- Response format validity is now acceptable for NPC-system side-effect checks.

Caveats to report:

- AS+PAS does not universally improve speech-action consistency.
- Prompting can increase BFI self-report more than TRAIT behavior, increasing
  the speech-action gap.
- Low Agreeableness is the clearest mixed-result case: behavior moves toward
  the target direction, but consistency decreases under strong AS+PAS-only.
- Random vector controls produce small non-zero shifts, so Phase 3 should keep
  random-vector diagnostics in the report.

## Phase 3 Fixed Settings

Use the following dev-selected settings for the full Big Five run:

```yaml
steering:
  alpha: 4.0
  layers: [18, 21, 24]
  tune_on_this_split: false
```

Do not retune alpha, layers, prompts, vectors, or scoring after test results are
generated.

## Next Steps

1. Update `configs/experiments/v4_bigfive_final.yaml` with the fixed Phase 3
   settings above.
2. Run `scripts/18_validate_bigfive_phase3.py`.
3. Extract vectors for the remaining Big Five traits at layers `[18, 21, 24]`.
4. Run the full five-trait experiment on the configured final split.
5. Report Phase 3 results with the Phase 2 caveats intact.
