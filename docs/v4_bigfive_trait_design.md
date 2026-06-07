# Big Five 기반 NPC 말-행동 일관성 실험 설계

## 1. Research Motivation

이 연구는 LLM의 도덕성이나 실제 성격을 측정하는 연구가 아니다. 목표는 게임 NPC에게 부여한 페르소나가 **말과 행동에서 일관되게 유지되는지**를 측정하는 것이다.

핵심 문제는 다음과 같다.

> NPC에게 특정 페르소나를 부여해도, 모델의 자기보고식 말과 실제 행동 선택이 어긋나면 플레이어가 느끼는 몰입이 깨진다.

따라서 최종 연구 질문은 다음으로 고정한다.

> AS+PAS가 NPC에게 부여된 Big Five 페르소나에서 말(BFI 자기보고)과 행동(TRAIT 선택) 사이의 일관성에 어떤 조건에서 영향을 주는가?

연구 framing은 검증형이 아니라 탐색형으로 둔다. 즉, “AS+PAS가 항상 말-행동 간격을 줄인다”가 아니라 “어떤 조건에서 말-행동 일관성에 영향을 주는가”를 조사한다.

이 framing은 결과가 긍정적, 혼합적, 또는 부정적이어도 연구 기여를 유지한다.

## 2. Always-On Design Principles

이 섹션은 모든 실험 설계와 시스템 설계 전에 확인해야 하는 기준이다. 새 기능, 새 지표, 새 조건, 새 데모 요소를 추가할 때는 아래 원칙 중 어떤 원칙을 만족하는지 명확히 설명해야 한다.

| 원칙 | 이유 |
|---|---|
| 측정 대상은 도덕성이 아니라 페르소나 일관성이다 | 연구 framing이 흐려지면 “LLM이 착한가/나쁜가”로 오해된다. 본 연구는 NPC 몰입 유지 문제다. |
| 말과 행동은 분리해서 측정한다 | LLM은 자기보고에서는 페르소나를 잘 말하지만 실제 선택 행동은 다를 수 있다. 이 간극이 연구 대상이다. |
| 자체 정의보다 검증된 심리 도구와 공개 데이터셋을 우선한다 | aggressive/cooperative 자체 라벨링은 방어력이 약하다. Big Five, BFI-44, TRAIT를 쓰면 construct validity가 올라간다. |
| 모든 trait 방향은 가치중립적으로 다룬다 | high/low 모두 NPC 캐릭터로 유효해야 한다. 특정 방향을 “좋은 성격”으로 해석하면 연구가 도덕성 평가로 미끄러진다. |
| 결과가 부정적이어도 남는 질문으로 설계한다 | “AS+PAS가 무조건 줄인다”가 아니라 “어떤 조건에서 영향을 주는가”로 framing해야 혼합 결과도 contribution이 된다. |
| 효과보다 먼저 confound를 제거한다 | random vector, unrelated trait vector, paraphrase test 없이는 AS+PAS 효과인지 단순 perturbation인지 구분할 수 없다. |
| 본 실험 전 pilot으로 효과 크기를 확인한다 | 5축 전체를 바로 돌리면 비용이 크고 실패 원인 분석이 어렵다. Agreeableness 1축으로 먼저 검증한다. |
| 성능 개선과 부작용을 같이 본다 | consistency가 올라가도 coherence, perplexity, format validity가 망가지면 NPC 시스템으로는 실패다. |
| train/dev/test 역할을 섞지 않는다 | vector extraction, alpha/layer 선택, 최종 보고가 같은 데이터에서 이뤄지면 과적합 주장이 생긴다. |
| 모든 run은 재현 가능해야 한다 | manifest, config snapshot, vector hash, model name, seed를 저장하지 않으면 결과를 검증하거나 발표 후 방어하기 어렵다. |

### Design Decision Checklist

모든 주요 설계 결정은 아래 질문에 답할 수 있어야 한다.

1. 이 결정은 말-행동 일관성을 더 정확히 측정하는가?
2. 이 결정은 도덕성 평가가 아니라 NPC 페르소나 일관성 평가라는 framing을 유지하는가?
3. high/low trait 방향을 가치중립적으로 다루는가?
4. AS+PAS 효과와 단순 perturbation, prompt 효과, action wording 효과를 구분할 수 있는가?
5. pilot, dev, test의 역할을 분리하고 있는가?
6. 나중에 같은 run을 재현할 수 있는 metadata를 저장하는가?
7. consistency가 올라갈 때 coherence나 format validity 같은 부작용도 함께 확인하는가?

## 3. Experimental Design Rationale

### 3.1 Big Five로 바꾸는 이유

기존 `aggressive/cooperative` 축은 직관적이지만 자체 정의에 가깝다. 이 축은 NPC 행동 연구에는 쓸 수 있지만, 논문에서 construct validity를 방어하기 어렵다.

Big Five는 다음 이유로 더 적합하다.

- NPC 성격을 다축적으로 표현할 수 있다.
- 각 축의 high/low가 모두 캐릭터 속성으로 유효하다.
- 특정 방향을 도덕적으로 우월하게 해석하지 않아도 된다.
- BFI-44와 TRAIT처럼 연결 가능한 측정 도구가 있다.
- 논문에서 “왜 이 성격 축인가?”에 대한 방어가 가능하다.

사용할 trait은 다음 5개다.

- Agreeableness
- Conscientiousness
- Neuroticism
- Openness
- Extraversion

각 trait은 `high`와 `low` 방향 모두 실험한다. 이때 high 방향이 좋은 성격이고 low 방향이 나쁜 성격이라는 해석은 금지한다. 두 방향 모두 NPC 페르소나의 유효한 스타일로 다룬다.

### 3.2 BFI-44를 말 측정으로 쓰는 이유

말 영역은 LLM이 자기 자신을 어떤 페르소나로 설명하는지를 본다. BFI-44는 Big Five 자기보고식 측정 도구이므로 “말로 드러나는 성격 자기인식”을 측정하는 데 적합하다.

설계 이유는 다음과 같다.

- trait별 정량 점수를 얻을 수 있다.
- reverse scoring과 평균 계산이 명확하다.
- 기존 심리측정 문헌에 근거할 수 있다.
- 자체 질문지보다 방어력이 높다.

출력은 다음 형태로 정규화한다.

```text
BFI_score(condition, persona_profile, trait) in [0, 1]
```

BFI-44는 말 측정용이다. 행동 선택의 ground truth로 사용하지 않는다.

### 3.3 TRAIT를 행동 측정으로 쓰는 이유

행동 영역은 “상황이 주어졌을 때 어떤 행동을 선택하는가”를 본다. 기존 자체 시나리오는 라벨링 신뢰도와 scenario coverage 문제가 있으므로 TRAIT의 Big Five 문항을 사용한다.

설계 이유는 다음과 같다.

- 상황 + 행동 선택지 구조가 NPC decision task와 잘 맞는다.
- 각 행동 선택지가 trait 방향으로 라벨링되어 있다.
- ATOMIC-10X 기반이라 상황 다양성이 있다.
- human validation 근거를 제시할 수 있다.
- 자체 action labeling보다 논문 방어력이 높다.

행동 점수는 high 방향 행동의 probability mass로 계산한다.

```text
TRAIT_score = P(high-direction actions)
```

low target persona는 해석 시 다음처럼 target 방향 점수로 변환한다.

```text
TRAIT_target_score = 1 - TRAIT_score
```

TRAIT는 행동 측정용이다. LLM의 “진짜 성격”을 측정한다고 해석하지 않는다.

### 3.4 Consistency 지표를 쓰는 이유

기존 지표는 “행동이 페르소나 방향으로 이동했는가”를 봤다. 새 지표는 “말과 행동이 서로 맞는가”를 본다.

핵심 지표는 다음이다.

```text
consistency = 1 - abs(BFI_score - TRAIT_score)
```

설계 이유는 다음과 같다.

- 연구 동기인 말-행동 불일치를 직접 측정한다.
- BFI와 TRAIT를 같은 `[0, 1]` trait scale에서 비교할 수 있다.
- 값이 클수록 NPC가 말과 행동에서 같은 페르소나를 유지한다고 해석할 수 있다.

단, consistency만 높다고 target persona를 잘 따른다는 뜻은 아니다. 둘 다 baseline 쪽으로 낮게 맞아도 consistency는 높을 수 있다. 따라서 `target_attainment`를 보조 지표로 반드시 함께 보고한다.

## 4. Condition Matrix Rationale

본 실험 조건은 5개로 고정한다.

| 조건 | 설계 이유 |
|---|---|
| `baseline` | 모델의 기본 말-행동 일관성 기준선이다. |
| `one_line_prompt` | 가장 현실적인 최소 persona prompting 효과를 본다. |
| `elaborate_prompt` | 강한 prompt만으로 도달 가능한 상한선을 본다. |
| `as_pas_only` | 프롬프트 없이 activation-level intervention만으로 변화가 생기는지 본다. |
| `elaborate_prompt_as_pas` | AS+PAS가 prompt와 보완적인지, 아니면 중복/충돌하는지 본다. |

이 조건 설계의 핵심은 다음 두 질문을 분리하는 것이다.

1. AS+PAS가 prompt를 대체할 수 있는가?
2. AS+PAS가 prompt를 보완할 수 있는가?

`one_line_prompt`와 `elaborate_prompt`를 분리하는 이유는 prompt ceiling을 확인하기 위해서다. 강한 prompt가 이미 거의 상한에 도달하면 `elaborate_prompt_as_pas`에서 추가 개선을 보기 어렵다. 이 경우 AS+PAS가 효과가 없다고 단정하기보다 ceiling 또는 signal conflict 가능성을 함께 해석해야 한다.

## 5. System Design Rationale

### 5.1 `v4_bigfive`로 분리하는 이유

기존 V2는 aggressive/cooperative 실험 결과와 코드가 이미 존재한다. 새 실험은 측정 도구, 데이터셋, 지표, 조건이 모두 달라지므로 기존 구조를 덮어쓰면 추적이 어렵다.

따라서 새 실험은 다음 네임스페이스로 분리한다.

```text
docs/v4_bigfive_trait_design.md
configs/experiments/v4_bigfive_*.yaml
results/v4_bigfive/
data/bfi/
data/trait_bigfive/
```

설계 이유는 다음과 같다.

- V2 결과를 보존한다.
- 새 실험의 책임 경계를 명확히 한다.
- 나중에 발표/논문에서 V2와 V4를 구분하기 쉽다.
- aggressive/cooperative 전용 metric과 Big Five consistency metric이 섞이는 것을 방지한다.

### 5.2 Manifest 기반 run 저장 이유

모든 run은 같은 결과라도 설정이 다르면 다른 실험이다. 따라서 run마다 manifest를 저장한다.

필수 저장 항목은 다음이다.

- model name
- git commit
- condition
- trait
- target direction
- prompt style
- AS/PAS 사용 여부
- alpha/layer
- vector hash
- data split
- seed
- timestamp

설계 이유는 다음과 같다.

- 결과 재현성을 확보한다.
- dev/test leakage를 확인할 수 있다.
- vector나 config가 바뀐 뒤에도 과거 결과를 검증할 수 있다.
- 논문/발표 후 “정확히 어떤 조건에서 얻은 결과인가?”에 답할 수 있다.

### 5.3 데이터 모듈을 분리하는 이유

BFI와 TRAIT는 서로 다른 역할을 갖는다.

- BFI: 말/self-report 측정
- TRAIT: 행동/decision 측정

두 데이터 소스를 하나의 ad hoc loader로 처리하면 scoring, split, 라벨 방향이 섞일 위험이 있다. 따라서 별도 모듈로 분리한다.

```text
src/data/bfi.py
src/data/trait_dataset.py
```

이 분리는 실험의 construct validity를 지키기 위한 시스템 설계다.

## 6. Data Sources

### 6.1 BFI-44

BFI-44는 Big Five 자기보고형 측정 도구로 사용한다. 문항별 trait, reverse scoring 여부, Likert score를 저장한다.

예상 데이터 구조:

```text
data/bfi/bfi44_items.csv
```

필수 columns:

```text
item_id, trait, text, reverse_scored
```

주의 사항:

- BFI-44 문항 재배포 가능 여부는 문서에 넣기 전 확인한다.
- 재배포가 불확실하면 item text를 repo에 저장하지 않고 item metadata와 external reference만 둔다.

### 6.2 TRAIT Big Five

TRAIT에서 Big Five 관련 문항만 추출해 행동 측정 데이터로 사용한다.

예상 데이터 구조:

```text
data/trait_bigfive/scenarios.jsonl
data/trait_bigfive/actions.jsonl
data/trait_bigfive/splits.json
```

정규화된 scenario schema:

```json
{
  "scenario_id": "trait_bfi_000001",
  "trait": "agreeableness",
  "prompt": "Situation text",
  "actions": [
    {"id": "trait_bfi_000001_a", "text": "Action option", "trait_direction": "high"},
    {"id": "trait_bfi_000001_b", "text": "Action option", "trait_direction": "high"},
    {"id": "trait_bfi_000001_c", "text": "Action option", "trait_direction": "low"},
    {"id": "trait_bfi_000001_d", "text": "Action option", "trait_direction": "low"}
  ],
  "split": "train"
}
```

TRAIT 접근 권한이 필요한 경우가 있으므로 loader는 데이터가 없을 때 조용히 fallback하지 않는다. 명확한 error message를 출력해야 한다.

## 7. Metrics

### 7.1 Primary Metric

주 지표:

```text
consistency = 1 - abs(BFI_score - TRAIT_score)
```

출력 파일:

```text
results/v4_bigfive/consistency_metrics.csv
```

필수 columns:

```text
run_id, condition, persona_profile_id, trait, target_direction, bfi_score, trait_score, consistency
```

### 7.2 Secondary Metrics

보조 지표:

- `bfi_target_attainment`: BFI 점수가 목표 방향으로 이동했는가
- `trait_target_attainment`: 행동 선택이 목표 방향으로 이동했는가
- `pas_loglik_agreement`: PAS와 loglik 선택이 같은 행동을 고르는가
- `paraphrase_stability`: scenario/action paraphrase 후 score가 유지되는가
- `coherence_score`: 응답 자연스러움
- `format_validity`: 요구 형식을 지키는가
- `perplexity`: language quality proxy

보조 지표를 두는 이유는 primary consistency의 해석 한계를 보완하기 위해서다.

## 8. AS+PAS Design

AS vector는 trait별 high-vs-low contrast로 추출한다.

```text
v_trait = mean(hidden(high_trait_response) - hidden(low_trait_response))
```

사용 규칙:

```text
high target: +v_trait
low target:  -v_trait
multi-trait profile: weighted sum of trait vectors
```

PAS는 행동 후보 embedding과 persona vector의 cosine similarity를 계산해 action ranking을 만든다.

AS+PAS 조건에서는 다음을 모두 저장한다.

- loglik selected action
- PAS selected action
- cosine scores
- loglik probabilities
- PAS/loglik agreement
- final selected action

Current implementation note:

- Primary `TRAIT_score` is computed from conditional log-likelihood.
- PAS selection is stored as a diagnostic decision signal.
- PAS does not overwrite `TRAIT_score` unless a separate PAS-as-policy scoring
  path is explicitly implemented.
- Therefore AS-only and PAS-only effects require additional ablation conditions.

논문에서는 AS 단독이 아니라 **AS+PAS intervention**으로 표현한다. AS와 PAS를 분리해서 주장하려면 별도 ablation 조건이 필요하다.

## 9. Reliability Checks

### 9.1 무관 벡터 대조

추가 diagnostic 조건:

- `zero_hook`
- `random_vector`
- `unrelated_trait_vector`

이유:

- AS+PAS 효과가 persona-specific direction 때문인지 확인한다.
- 단순 activation perturbation으로도 같은 효과가 나는지 배제한다.
- unrelated trait vector가 같은 변화를 만들면 trait-specific claim은 약해진다.

이 조건들은 main 5-condition matrix에 섞지 않는다. 별도 reliability table로 보고한다.

구현상 diagnostic vector mode는 다음과 같이 처리한다.

| vector mode | 처리 |
|---|---|
| `trait_contrast` | target trait의 high/low contrast vector를 사용한다. low target은 부호를 반전한다. |
| `zero` | 같은 layer shape의 zero vector를 사용한다. injector normalization은 끈다. |
| `random` | 같은 layer shape의 deterministic random unit vector를 사용한다. |
| `unrelated_trait` | target trait이 아닌 다른 trait vector를 stable order로 선택한다. |

`zero`, `random`, `unrelated_trait`도 layer shape와 trait vector inventory를 확인해야 하므로 vector bundle을 요구한다. 이는 GPU 실행 전에 readiness checker에서 확인한다.

### 9.2 패러프레이즈 견고성

같은 TRAIT 문항을 다른 표현으로 바꿔도 결과가 유지되는지 확인한다.

이유:

- action text wording에 과적합된 결과인지 확인한다.
- loglik/PAS가 실제 행동 의미를 보는지, 표현 선호를 보는지 분리한다.
- forced-choice task의 construct validity를 점검한다.

측정:

- action choice agreement
- trait score variance
- consistency variance

구현 기준:

- paraphrase group은 `paraphrase_group_id`로 묶는다.
- 원문 행은 가능하면 `paraphrase_variant=original`로 표시한다.
- robustness summary는 condition/trait/target_direction별 action agreement와 trait score range를 별도 산출물로 저장한다.

### 9.3 부작용 측정

측정:

- perplexity
- coherence score
- response length
- format validity
- parse success
- valid action selection

이유:

- consistency 개선이 NPC 품질 저하를 대가로 얻어진 것인지 확인한다.
- AS+PAS가 행동 선택만 바꾸고 대화 품질을 망가뜨리면 실제 NPC 시스템으로는 실패다.

구현 기준:

- `response length`, `format validity`, `parse success`, `valid action selection`은 모델 평가기 없이 deterministic metric으로 먼저 계산한다.
- `perplexity`와 `coherence score`는 별도 평가기가 준비되면 같은 `side_effect_metrics.csv`에 추가한다.
- side-effect 결과는 primary consistency 결과를 덮어쓰지 않고 별도 산출물로 저장한다.

## 10. Implementation Plan

### 10.1 새 문서

추가:

```text
docs/v4_bigfive_trait_design.md
```

문서 구조:

1. Research motivation
2. Always-on design principles
3. Experimental design rationale
4. System design rationale
5. Data sources
6. Metrics
7. Conditions
8. Controls
9. Pilot plan
10. Full experiment plan
11. Demo plan
12. Paper framing

### 10.2 새 설정 파일

추가 예정:

```text
configs/experiments/v4_bigfive_pilot.yaml
configs/experiments/v4_bigfive_final.yaml
configs/experiments/v4_bigfive_controls.yaml
```

기존 `configs/experiment.yaml`은 V2 호환을 위해 유지한다.

### 10.3 새 데이터 구조

추가 예정:

```text
data/bfi/bfi44_items.csv
data/trait_bigfive/scenarios.jsonl
data/trait_bigfive/actions.jsonl
data/trait_bigfive/splits.json
```

### 10.4 새 코드 모듈

추가 예정:

```text
src/data/bfi.py
src/data/trait_dataset.py
src/evaluation/bigfive_metrics.py
src/evaluation/side_effect_metrics.py
src/evaluation/paraphrase_robustness.py
src/experiments/manifest.py
src/controls/vector_controls.py
src/demo/bigfive_app.py
```

확장 예정:

```text
src/generation/prompt_builder.py
scripts/02_extract_vectors.py
scripts/03_run_experiment.py
scripts/04_evaluate.py
```

### 10.5 Phase 1 구현 현황

Phase 1에서 고정한 구현 단위는 다음과 같다.

```text
docs/v4_bigfive_trait_design.md
configs/experiments/v4_bigfive_pilot.yaml
configs/experiments/v4_bigfive_controls.yaml
configs/experiments/v4_bigfive_final.yaml
data/bfi/bfi44_scoring.csv
data/bfi/README.md
data/trait_bigfive/README.md
src/data/bfi.py
src/data/trait_convert.py
src/data/trait_dataset.py
src/evaluation/bigfive_metrics.py
src/evaluation/side_effect_metrics.py
src/evaluation/paraphrase_robustness.py
src/experiments/manifest.py
src/experiments/v4_bigfive_plan.py
src/experiments/v4_bigfive_runner.py
scripts/09_build_trait_contrastive_pairs.py
scripts/10_prepare_bigfive_pilot.py
scripts/11_convert_trait_bigfive.py
scripts/12_run_bigfive_pilot.py
scripts/13_check_bigfive_readiness.py
scripts/14_analyze_bigfive_results.py
scripts/15_score_side_effects.py
scripts/16_analyze_paraphrase_robustness.py
scripts/17_validate_bigfive_phase2.py
scripts/18_validate_bigfive_phase3.py
scripts/19_run_bigfive_demo.py
scripts/20_validate_bigfive_demo.py
```

Phase 1의 목적은 실제 실험 결과를 만드는 것이 아니라, 데이터 변환, 지표 계산, prompt construction, manifest 저장, pilot 실행 준비가 같은 schema 위에서 움직이도록 고정하는 것이다.

실제 pilot 전 필요한 외부 입력은 다음 두 가지다.

1. 승인된 TRAIT raw export.
2. 재배포 권한이 확인된 BFI-44 item text 또는 `data/bfi/bfi44_item_text.local.csv` 로컬 전용 item text 파일.

Pilot 실행 전에는 readiness checker를 먼저 실행한다.

```powershell
.venv\Scripts\python.exe scripts\13_check_bigfive_readiness.py --allow_errors
```

이 checker는 다음 항목을 GPU 실행 전에 확인한다.

- v4 config parsing
- BFI item count, trait coverage, item text 존재 여부
- BFI local text overlay 존재 여부
- TRAIT normalized scenario file 존재 여부와 split/trait coverage
- trait contrastive pair file 존재 여부
- AS/PAS 조건에서 vector bundle 필요 여부

실제 scoring run이 끝나면 metric 파일과 같은 디렉터리에 `run_manifest.json`을 저장한다. 이 manifest에는 model name, git commit, selected conditions, trait, target direction, split, seed, alpha/layer, vector hash, config hash, model config hash, input data hash를 포함한다. 이 파일이 없으면 결과 CSV만으로는 어떤 설정에서 나온 결과인지 방어할 수 없으므로 최종 분석에 포함하지 않는다.

`ERROR`가 남아 있으면 실제 pilot scoring을 실행하지 않는다.

Pilot scoring 후에는 분석 스크립트를 실행한다.

```powershell
.venv\Scripts\python.exe scripts\14_analyze_bigfive_results.py --metrics_dir results/v4_bigfive/pilot_metrics
```

이 스크립트는 다음 산출물을 만든다.

- `consistency_summary.csv`
- `condition_differences.csv`
- `paired_scenario_differences.csv`
- `paired_bootstrap_ci.csv`
- `pas_agreement_summary.csv` if PAS rows exist
- `analysis_report.md`
- `final_report.md`

## 11. Execution Plan

### Phase 1: 설계 문서와 데이터 고정

1. `docs/v4_bigfive_trait_design.md` 작성
2. BFI-44 scoring table 추가
3. TRAIT Big Five 변환 스크립트 작성
4. train/dev/test split 생성
5. manifest schema 정의

### Phase 2: Pilot

범위:

- Agreeableness 1축
- TRAIT 50문항
- 5조건
- random/unrelated vector diagnostic 포함

Pilot 판단 기준:

- AS+PAS 조건에서 baseline 대비 consistency 또는 target attainment가 방향성 있게 개선되는가
- 무관 벡터에서 같은 효과가 재현되지 않는가
- coherence와 format validity가 크게 무너지지 않는가

Pilot이 실패하면 5축 전체 실험으로 넘어가지 않는다. 실패 원인을 먼저 분리한다.

### Phase 3: Full Big Five

범위:

- 5 traits
- high/low target 모두
- dev에서 alpha/layer 선택
- test에서 최종 결과 고정

test 이후에는 hyperparameter를 바꾸지 않는다.

### Phase 4: Analysis

출력:

```text
results/v4_bigfive/bfi_scores.csv
results/v4_bigfive/trait_scores.csv
results/v4_bigfive/consistency_metrics.csv
results/v4_bigfive/paired_bootstrap_ci.csv
results/v4_bigfive/control_vector_results.csv
results/v4_bigfive/side_effect_metrics.csv
results/v4_bigfive/side_effect_summary.csv
results/v4_bigfive/final_report.md
```

분석 모델:

```text
consistency ~ condition * trait + (1 | scenario_id)
```

보조 분석:

- trait별 paired bootstrap CI
- PAS/loglik agreement
- target attainment
- paraphrase robustness
- side-effect metrics

### Phase 5: Demo

Streamlit 기반으로 구현한다.

화면:

- Big Five 5축 slider
- scenario card
- Prompt-only NPC 응답
- AS+PAS NPC 응답
- 대사 + 행동 비교
- 2D PCA activation movement

설계 이유:

- 수업 데모는 논문 지표를 직관적으로 보여줘야 한다.
- 단순히 “AS가 움직인다”가 아니라 “NPC 말과 행동이 더 맞거나 어긋난다”를 보여줘야 한다.
- live model 실행은 비용과 지연이 크므로 기본은 precomputed 결과를 사용한다.

## 12. Paper Framing

최종 framing:

> AS+PAS가 어떤 조건에서 NPC의 말-행동 페르소나 일관성에 영향을 주는지 조사한다.

피해야 할 표현:

- LLM 도덕성을 측정한다
- LLM의 진짜 성격을 측정한다
- AS+PAS가 항상 말-행동 간격을 줄인다
- high trait이 low trait보다 좋다

사용할 표현:

- persona consistency
- speech-action consistency
- NPC immersion
- trait-conditioned behavior
- exploratory condition analysis

결과 해석 원칙:

- 긍정 결과: AS+PAS가 특정 trait/조건에서 consistency를 개선한다.
- 혼합 결과: AS+PAS 효과는 trait과 prompt strength에 의존한다.
- 부정 결과: AS+PAS가 BFI 자기보고와 TRAIT 행동 선택의 간극을 안정적으로 줄이지 못했다는 것도 contribution으로 남긴다.

## 13. Assumptions and Open Checks

- TRAIT는 접근 권한이 필요할 수 있으므로 로컬 다운로드 또는 Hugging Face 접근 설정이 필요하다.
- BFI-44 문항 재배포 가능 여부는 문서에 넣기 전 확인한다.
- TRAIT의 human validation 수치는 원문 기준으로 정확히 구분한다.
- 기존 V2 코드는 삭제하지 않고 `v4_bigfive`로 분리한다.
- AS와 PAS를 각각 독립 효과로 주장하려면 별도 ablation 조건을 추가해야 한다.

## 14. References To Verify

- TRAIT paper: https://arxiv.org/abs/2406.14703
- TRAIT dataset: https://huggingface.co/datasets/mirlab/TRAIT
- TRAIT GitHub: https://github.com/pull-ups/TRAIT
- BFI reference page: https://sjdm.org/dmidi/Big_Five_Inventory.html

## 15. Execution Runbook

실제 pilot 실행 순서와 명령은 `docs/v4_bigfive_runbook.md`에 고정한다. 실험 실행자는 해당 runbook을 기준으로 readiness, vector extraction, pilot scoring, analysis, side-effect checks를 수행한다.
