# Slide Visualization Redesign Brief

## Copy-Paste Prompt for Another LLM

You are being asked to redesign a set of slide-ready visualizations for an activation steering experiment. The current figures exist, but their readability is not good enough for presentation. Please do not simply reuse the current layout. Use the data and intent below to create cleaner, more legible, slide-quality visuals.

The goal is to produce presentation figures that communicate the experiment clearly to an audience that may not know the codebase. Prioritize readability, visual hierarchy, compact labels, and defensible interpretation.

Please create improved versions of these four slide visuals:

1. Condition-level alignment rate bar chart.
2. Vector design comparison diagram.
3. V1 vs V2 evaluation pipeline flow.
4. Mixed-effects OR + CI forest plot.

Important: the current generated assets are only reference drafts. Redesign them.

## Current Asset Locations

Current generated figures are here:

- `C:/GIT/results/v2/slide_assets/01_alignment_bar_chart.png`
- `C:/GIT/results/v2/slide_assets/02_vector_design_diagram.png`
- `C:/GIT/results/v2/slide_assets/03_pipeline_v1_v2_flow.png`
- `C:/GIT/results/v2/slide_assets/04_mixed_effects_forest_plot.png`

SVG versions also exist:

- `C:/GIT/results/v2/slide_assets/01_alignment_bar_chart.svg`
- `C:/GIT/results/v2/slide_assets/02_vector_design_diagram.svg`
- `C:/GIT/results/v2/slide_assets/03_pipeline_v1_v2_flow.svg`
- `C:/GIT/results/v2/slide_assets/04_mixed_effects_forest_plot.svg`

Current generation script:

- `C:/GIT/scripts/08_make_slide_assets.py`

Reference README:

- `C:/GIT/results/v2/slide_assets/README.md`

These drafts are readable as raw content references, but the layout and typography need improvement.

## Source Data Files

Use these as the authoritative data sources:

- Alignment metrics:
  - `C:/GIT/results/v2/final_analysis_v2/alignment_metrics.csv`
- Quality metrics:
  - `C:/GIT/results/v2/final_analysis_v2/quality_metrics.csv`
- Mixed-effects fixed effects:
  - `C:/GIT/results/v2/final_analysis_v2/mixed_effects_fixed_effects.csv`
- Cluster bootstrap:
  - `C:/GIT/results/v2/final_analysis_v2/cluster_bootstrap_alignment.csv`
- Paired differences:
  - `C:/GIT/results/v2/final_analysis_v2/paired_cluster_differences.csv`
- Vector cosine diagnostics:
  - `C:/GIT/results/v2/vector_diagnostics/vector_design_key_cosines.csv`
- Final report:
  - `C:/GIT/results/v2/final_analysis_v2/final_report.md`

## Experiment Context

This project evaluates activation steering for NPC action selection.

Model:

- `Qwen/Qwen2.5-3B-Instruct`

Task:

- Each scenario has 5 candidate actions.
- Actions are labeled as aggressive, cooperative, or neutral.
- For each target persona, alignment means choosing an action with the matching label.

Final V2 conditions:

| condition | meaning |
|---|---|
| `neutral_baseline` | no persona prompt, no steering |
| `prompt_baseline` | persona prompt only |
| `as_only_neg` | activation steering only, contrast vector direction |
| `as_only_sep` | same corrected contrast direction, kept as vector-design ablation |
| `prompt_as` | persona prompt plus activation steering |

Important interpretation:

- `as_only_neg` and `as_only_sep` are not independent effects.
- After the Task 4 vector correction, both use the contrast direction and produce identical results.
- If both are shown in the bar chart, add a clear note.
- In the forest plot, it is better to collapse them into one `AS only (contrast)` row.

## Core Numeric Results

### Quality

V2 log-likelihood scoring removes parser failure:

| metric | value |
|---|---:|
| parse_ok_rate | 1.000 |
| unknown_rate | 0.000 |

Legacy V1 free-form generation had parse_ok only around 42-54 percent.

### Alignment Rates

Chance baseline:

- aggressive: 2/5 = 0.40
- cooperative: 2/5 = 0.40

Final alignment rates:

| condition | aggressive | cooperative |
|---|---:|---:|
| neutral_baseline | 0.433 | 0.367 |
| prompt_baseline | 0.900 | 0.767 |
| as_only_neg | 0.600 | 0.700 |
| as_only_sep | 0.600 | 0.700 |
| prompt_as | 0.800 | 0.733 |

Wilson 95 percent CIs from `alignment_metrics.csv`:

| condition | persona | estimate | CI |
|---|---|---:|---|
| neutral_baseline | aggressive | 0.433 | [0.274, 0.608] |
| neutral_baseline | cooperative | 0.367 | [0.219, 0.545] |
| prompt_baseline | aggressive | 0.900 | [0.744, 0.965] |
| prompt_baseline | cooperative | 0.767 | [0.591, 0.882] |
| as_only_neg | aggressive | 0.600 | [0.423, 0.754] |
| as_only_neg | cooperative | 0.700 | [0.521, 0.833] |
| as_only_sep | aggressive | 0.600 | [0.423, 0.754] |
| as_only_sep | cooperative | 0.700 | [0.521, 0.833] |
| prompt_as | aggressive | 0.800 | [0.627, 0.905] |
| prompt_as | cooperative | 0.733 | [0.556, 0.858] |

### Mixed-Effects ORs

Model:

```text
aligned ~ condition * persona + (1 | scenario_id)
```

Use these condition effects versus neutral:

| condition | OR | 95 percent CI | p |
|---|---:|---|---:|
| AS only contrast | 1.978 | [1.106, 3.540] | 0.0216 |
| prompt_as | 5.818 | [3.063, 11.050] | 7.42e-08 |
| prompt_baseline | 12.477 | [6.093, 25.552] | 5.15e-12 |

Note:

- `as_only_neg` and `as_only_sep` have the same OR. Use one row for the forest plot unless the slide explicitly discusses their equivalence.

### Vector Cosines

Use these in the vector design diagram:

| comparison | mean cosine |
|---|---:|
| neutral-anchor aggressive vs cooperative | 0.717 |
| contrast aggressive vs cooperative | -1.000 |
| orthogonalized aggressive vs cooperative | -1.000 |
| contrast aggressive vs orthogonalized aggressive | 0.954 |
| contrast cooperative vs orthogonalized cooperative | 0.954 |

Interpretation:

- Neutral-anchor separate extraction did not recover opposite persona directions.
- Both neutral-anchor vectors point in a similar broad direction.
- This suggests they capture a shared non-neutral component.
- Orthogonalization recovers a direction very close to the contrast vector.

## Figure 1: Alignment Bar Chart

Purpose:

- Show the main experimental result at a glance.
- Audience should immediately see:
  - prompt-only is strongest,
  - AS-only improves over neutral,
  - prompt+AS does not exceed prompt-only,
  - chance baseline is 0.40.

Required design:

- x-axis: conditions.
- y-axis: alignment rate from 0 to 1.
- Two bars per condition:
  - aggressive
  - cooperative
- Add horizontal dashed line at 0.40 labeled `chance`.
- Include 95 percent CIs, but do not let error bars dominate.
- Use short condition labels:
  - `Neutral`
  - `Prompt`
  - `AS only`
  - `AS only (same vector)`
  - `Prompt + AS`

Possible improvement:

- Since `as_only_neg` and `as_only_sep` are identical, either:
  1. show both but add a bracket/note saying "same corrected contrast vector", or
  2. collapse them into one `AS only` bar for presentation clarity.

Recommendation:

- For a main-results slide, collapse to four visual groups:
  - Neutral
  - Prompt
  - AS only
  - Prompt + AS
- Mention in speaker notes that `as_only_neg/as_only_sep` were identical after correction.

Problems with current draft:

- Too much horizontal whitespace.
- Title is too large.
- The y-axis label is oversized.
- Condition labels are bulky.
- Duplicate AS conditions may confuse the audience.

## Figure 2: Vector Design Diagram

Purpose:

- Explain the core vector-design discovery visually.
- This is likely a key "Jin part" slide.

Required message:

1. Neutral-anchor vectors:
   - `v_aggressive = aggressive - neutral`
   - `v_cooperative = cooperative - neutral`
   - cosine = 0.717
   - they point in a similar broad direction.

2. Corrected contrast axis:
   - aggressive uses `+v_contrast`
   - cooperative uses `-v_contrast`
   - cosine = -1.000

3. Orthogonalized vector:
   - aligns with contrast axis, cosine = 0.954.

Preferred visual design:

- Left panel: two arrows from neutral pointing roughly same direction.
- Right panel: one horizontal axis with opposite arrows.
- Add one concise takeaway:
  - `Neutral-anchor extraction captured non-neutrality, not persona opposition.`

Problems with current draft:

- Too much empty space.
- Text is too large and uneven.
- The "origin" and vector labels are awkwardly placed.
- It should feel like a clean explanatory diagram, not a raw matplotlib plot.

## Figure 3: V1 vs V2 Pipeline Flow

Purpose:

- Explain why the evaluation pipeline changed.
- This is background/motivation, not the main result.

Required message:

- V1:
  - prompt -> free-form generation -> parser -> action label
  - parse_ok only around 42-54 percent
  - many unknowns

- V2:
  - prompt + 5 candidate actions -> score `log p(action | prompt)` -> argmax/probabilities
  - parse_ok = 100 percent
  - unknown = 0 percent

Important caveat:

- V2 is cleaner, but it measures forced-choice action-description preference.
- Do not imply it is identical to open-ended behavior.

Preferred visual design:

- Two horizontal rows.
- Use red/gray for V1 failure path.
- Use blue/green for V2 fixed path.
- Put the key numbers at the far right:
  - `parse_ok 42-54%`
  - `parse_ok 100%`
- Keep labels short.

Problems with current draft:

- Layout is too wide.
- The boxes are oversized.
- There is too much vertical whitespace.
- It should be denser and more slide-friendly.

## Figure 4: OR + CI Forest Plot

Purpose:

- Show mixed-effects results compactly.
- Audience should see that prompt effects are larger than AS-only effects.

Required rows:

| row | OR | CI |
|---|---:|---|
| AS only contrast | 1.98 | [1.11, 3.54] |
| Prompt + AS | 5.82 | [3.06, 11.05] |
| Prompt only | 12.48 | [6.09, 25.55] |

Required design:

- x-axis on log scale.
- vertical reference line at OR = 1.
- points with horizontal CIs.
- sort from largest OR at top to smallest at bottom, or use narrative order:
  - AS only
  - Prompt + AS
  - Prompt only
- Use concise title:
  - `Condition Effects vs Neutral`

Problems with current draft:

- Too much empty vertical space.
- Title is too large.
- Footnote is too long.
- It should fit cleanly on one slide beside one or two bullet takeaways.

## Preferred Output Format

Please output:

1. Improved figure files:
   - PNG, 16:9 slide-friendly.
   - SVG or PDF if possible.

2. A short README:
   - what each figure shows,
   - which CSV it uses,
   - one recommended slide title,
   - one recommended speaker note.

3. If making code:
   - write a reproducible script,
   - avoid manual edits only,
   - keep paths configurable.

Suggested output directory:

- `C:/GIT/results/v2/slide_assets_redesigned/`

Suggested script path:

- `C:/GIT/scripts/09_redesign_slide_assets.py`

## Presentation Narrative

Recommended slide order:

1. Pipeline flow:
   - "Why we changed the measurement."
2. Alignment bar chart:
   - "What changed across conditions."
3. Forest plot:
   - "How strong the effects are versus neutral."
4. Vector design diagram:
   - "Why the separate-vector attempt collapsed back to contrast."

Possible one-sentence story:

> V2 fixed the parser problem and showed that activation steering moves alignment above neutral, but prompt conditioning remains stronger; vector diagnostics suggest the contrast direction is the defensible persona axis.

## Tone and Design Style

Make the visuals:

- clean,
- compact,
- readable from a projector,
- not overly statistical,
- not cluttered with every caveat,
- but honest about the key caveats.

Avoid:

- huge titles,
- crowded footnotes,
- raw CSV-like labels,
- excessive whitespace,
- duplicated AS rows without explanation,
- long paragraphs inside figures.

Use:

- simple titles,
- short labels,
- consistent colors,
- clear baselines/reference lines,
- speaker-note style caveats outside the plot when possible.
