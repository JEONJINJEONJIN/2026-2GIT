# Task 4 Vector Design Diagnostic

## 핵심 발견

Linear representation hypothesis 하에 `cooperative = -aggressive`를 가정했으나,
neutral을 anchor로 한 separate extraction 결과 두 벡터가 mean cosine `0.717225`로
거의 같은 방향을 가리켰다. 이는 모델 내부 표현에서 `aggressive vs cooperative`보다
`non-neutral vs neutral` 축이 더 강하게 형성되어 있음을 시사한다.

## 적용한 수정

- `as_only_neg`: 기존 contrast vector의 양/음 방향 사용.
- `as_only_sep`: separate extraction의 최종 선택지를 contrast-explicit로 고정.
  - `v_aggressive = v_contrast`
  - `v_cooperative = -v_contrast`
- `as_only_sep_neutral_anchor`: 실패한 neutral-anchor separate를 진단용 condition으로 보존.
- `as_only_orth`: neutral-anchor vectors에서 common component를 제거한 orthogonalized condition 추가.

## Key Cosines

| comparison | mean cosine |
|---|---:|
| neg aggressive vs cooperative | -1.000000 |
| sep neutral-anchor aggressive vs cooperative | 0.717225 |
| sep orthogonalized aggressive vs cooperative | -0.999999 |

Full matrix:
`results/v2/vector_diagnostics/vector_design_cosine_matrix.csv`

Pair inspection sample:
`results/v2/vector_diagnostics/neutral_anchor_pair_samples.csv`

Current pilot comparison:
`results/v2/pilot_vector_design_analysis_v2/alignment_metrics.csv`

## Pilot Sanity Check

All conditions reached `parse_ok_rate = 1.0` and `unknown_rate = 0.0`.

| condition | aggressive ITT | cooperative ITT |
|---|---:|---:|
| neutral_baseline | 1.000 | 0.000 |
| prompt_baseline | 1.000 | 0.667 |
| as_only_neg | 1.000 | 0.667 |
| as_only_sep | 1.000 | 0.667 |
| as_only_sep_neutral_anchor | 1.000 | 0.000 |
| as_only_orth | 1.000 | 0.333 |

`as_only_sep` now matches `as_only_neg`, confirming that the explicit contrast
directions are mathematically and behaviorally equivalent in this pilot.
The neutral-anchor diagnostic condition fails on cooperative alignment, which
supports removing it from the main sweep and keeping it as an ablation.
