# Team B LLM Prompt

Copy and paste the prompt below into another LLM. It is written for Team Member B, who is responsible for evaluation, analysis, reporting, manual inspection, and presentation material preparation.

---

You are assisting Team Member B, who is responsible for evaluation, analysis, result interpretation, manual inspection, and report/presentation writing for an activation steering experiment.

You do not need GPU access. Assume the raw experiment outputs and analysis CSV files are already generated. Your task is to review the current results, identify what can be reported, what should be treated cautiously, and what tables/figures should be prepared for Report #2 and presentation slides.

## Project Context

The experiment evaluates whether activation steering can shift `Qwen/Qwen2.5-3B-Instruct` toward persona-aligned NPC action choices.

Target personas:

- aggressive
- cooperative

Each scenario has five candidate actions:

- 2 aggressive
- 2 cooperative
- 1 neutral

Therefore chance alignment for aggressive/cooperative is `2/5 = 0.40`.

## Important Pipeline Change

The legacy pipeline used free-form generation and then parsed the model output. This produced high parse failure and unknown rates, so the old alignment rates were methodologically weak.

The current V2 pipeline primarily uses closed-set description log-likelihood scoring:

```text
prompt + candidate action description
  -> compute log p(action_description | prompt)
  -> select argmax action
```

This means:

- parse_ok is now 100%.
- unknown rate is now 0%.
- The primary result is cleaner than free-form generation.
- But it measures forced-choice action-description preference, not open-ended NPC behavior.

Please keep that distinction explicit in the report.

## Team B Main Responsibilities

Please help Team B with:

1. Reviewing outputs from `scripts/04_evaluate.py`.
2. Interpreting persona alignment, entropy, and action distribution.
3. Interpreting distribution shift if relevant.
4. Treating `speech_action_agreement` carefully:
   - In V2, the old speech metric was removed from default reporting because the English keyword classifier was invalid for Korean/free-form text.
   - Do not report it as a valid metric unless a new Korean-compatible classifier or manual annotation exists.
5. Creating clean result tables.
6. Creating graph recommendations for presentation.
7. Checking manual annotation samples.
8. Summarizing failure cases:
   - parse fail
   - unknown
   - neutral collapse
   - action repetition
   - scenario-specific instability
   - description wording sensitivity
9. Writing the methods/results/limitations section for Report #2.

## Current Result Files

Use these files:

Raw final outputs:

- `C:/GIT/results/v2/final_raw/neutral_baseline.jsonl`
- `C:/GIT/results/v2/final_raw/prompt_baseline.jsonl`
- `C:/GIT/results/v2/final_raw/as_only_neg.jsonl`
- `C:/GIT/results/v2/final_raw/as_only_sep.jsonl`
- `C:/GIT/results/v2/final_raw/prompt_as.jsonl`

Analysis outputs:

- `C:/GIT/results/v2/final_analysis_v2/quality_metrics.csv`
- `C:/GIT/results/v2/final_analysis_v2/alignment_metrics.csv`
- `C:/GIT/results/v2/final_analysis_v2/cluster_bootstrap_alignment.csv`
- `C:/GIT/results/v2/final_analysis_v2/paired_cluster_differences.csv`
- `C:/GIT/results/v2/final_analysis_v2/mixed_effects_fixed_effects.csv`
- `C:/GIT/results/v2/final_analysis_v2/mixed_effects_random_effects.csv`
- `C:/GIT/results/v2/final_analysis_v2/action_distribution.csv`
- `C:/GIT/results/v2/final_analysis_v2/scenario_entropy.csv`
- `C:/GIT/results/v2/final_analysis_v2/annotated_results.csv`

Reports:

- `C:/GIT/results/v2/final_analysis_v2/final_report.md`
- `C:/GIT/results/v2/final_analysis_v2/task6_final_run_report.md`
- `C:/GIT/docs/progress_summary_for_llm.md`

Slide draft assets:

- `C:/GIT/results/v2/slide_assets/`
- `C:/GIT/docs/slide_visualization_redesign_brief.md`

## Final V2 Conditions

| condition | meaning |
|---|---|
| `neutral_baseline` | no persona prompt, no activation steering |
| `prompt_baseline` | persona prompt only |
| `as_only_neg` | activation steering only, contrast vector direction |
| `as_only_sep` | same corrected contrast direction, retained as vector-design ablation |
| `prompt_as` | persona prompt plus activation steering |

Important:

`as_only_neg` and `as_only_sep` are effectively identical after the vector correction. Do not interpret them as independent evidence. In presentation/report tables, either collapse them into `AS only (contrast)` or explicitly note that they are equivalent.

## Core Numbers

Quality:

| metric | value |
|---|---:|
| parse_ok_rate | 1.000 |
| unknown_rate | 0.000 |

Alignment:

| condition | aggressive | cooperative |
|---|---:|---:|
| neutral_baseline | 0.433 | 0.367 |
| prompt_baseline | 0.900 | 0.767 |
| as_only_neg | 0.600 | 0.700 |
| as_only_sep | 0.600 | 0.700 |
| prompt_as | 0.800 | 0.733 |

Paired cluster-bootstrap differences:

| comparison | persona | diff | 95% CI |
|---|---|---:|---|
| as_only_sep - neutral_baseline | aggressive | 0.167 | [0.000, 0.333] |
| as_only_sep - neutral_baseline | cooperative | 0.333 | [0.133, 0.533] |
| prompt_as - neutral_baseline | aggressive | 0.367 | [0.167, 0.567] |
| prompt_as - neutral_baseline | cooperative | 0.367 | [0.133, 0.567] |
| prompt_baseline - neutral_baseline | aggressive | 0.467 | [0.300, 0.633] |
| prompt_baseline - neutral_baseline | cooperative | 0.400 | [0.200, 0.600] |
| prompt_as - prompt_baseline | aggressive | -0.100 | [-0.233, 0.000] |
| prompt_as - prompt_baseline | cooperative | -0.033 | [-0.100, 0.000] |

Mixed-effects condition effects versus neutral:

| condition | OR | 95% CI | p |
|---|---:|---|---:|
| as_only_neg | 1.978 | [1.106, 3.540] | 0.0216 |
| as_only_sep | 1.978 | [1.106, 3.540] | 0.0216 |
| prompt_as | 5.818 | [3.063, 11.050] | 7.42e-08 |
| prompt_baseline | 12.477 | [6.093, 25.552] | 5.15e-12 |

Scenario random-effect variance:

- SD: 1.024
- variance: 1.048

## Key Interpretation

The defensible V2 interpretation is:

1. V2 fixed the parse/unknown confound for the closed-set evaluation.
2. Activation steering alone improves alignment above neutral.
3. Prompt-only is stronger than activation steering in this run.
4. Prompt+AS does not improve over prompt-only, but this should be stated cautiously because prompt-only has a ceiling effect.
5. The current primary metric is closed-set action-description preference, not fully open-ended behavior.

Avoid overclaiming:

- Do not say AS generally changes real NPC behavior.
- Do not say prompt+AS is harmful.
- Do not say AS is generally weaker than prompting across all settings.
- Do not treat `as_only_neg` and `as_only_sep` as independent effects.

## Requested Output From You

Please produce:

1. A clean condition-level result table for the report.
2. A short interpretation of whether AS beats neutral.
3. A short interpretation of whether AS beats prompt baseline.
4. A note on `prompt_as` versus `prompt_baseline` and the ceiling effect.
5. A paragraph explaining why V2 loglik scoring fixed parse failure but still has construct-validity limits.
6. A list of failure cases to manually inspect from `annotated_results.csv` or raw JSONL.
7. Suggested presentation figures and what each should show.
8. A Report #2-ready methods paragraph.
9. A Report #2-ready results paragraph.
10. A Report #2-ready limitations paragraph.

Please write clearly and conservatively. The audience should understand that V2 is a major measurement improvement, but not yet definitive proof of persona-specific behavioral steering.
