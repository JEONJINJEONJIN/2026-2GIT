# Activation Steering V3 Experiment Design

## 목적

V3의 목적은 V2 결과를 단순히 확장하는 것이 아니라, activation steering이 실제로 persona-relevant decision behavior를 바꾸는지 더 설득력 있게 검증하는 것이다.

V2는 free-form generation parsing 문제를 해결했다. 그러나 V2의 핵심 endpoint는 `log p(action_description | prompt)`이기 때문에, 이것이 실제 행동 선택 변화인지, 아니면 persona와 가까운 어휘/문체가 포함된 description 선호 변화인지 분리되지 않는다.

따라서 V3의 핵심 질문은 다음이다.

> Activation steering이 단순한 description wording preference가 아니라, held-out scenario에서 persona-specific action preference를 일관되게 바꾸는가?

## V3의 설계 원칙

1. 측정 타당도 먼저 검증한다.
   - forced-choice loglik은 유지하되, description wording confound를 직접 측정한다.
   - free-form generation은 별도 secondary endpoint로 분리한다.

2. persona-specific effect를 negative control로 확인한다.
   - random vector, zero-vector hook, wrong-persona vector를 추가한다.
   - AS 효과가 "아무 벡터나 넣으면 생기는 효과"인지 구분한다.

3. vector extraction과 evaluation을 분리한다.
   - train/dev/test scenario split을 만든다.
   - vector extraction과 alpha/layer tuning은 train/dev에서만 한다.
   - 최종 주장은 held-out test에서만 한다.

4. prompt ceiling을 낮춘 별도 조건을 둔다.
   - prompt_baseline이 0.9에 가까우면 prompt+AS 추가 효과를 보기 어렵다.
   - strong prompt와 weak prompt를 분리한다.

5. condition을 해석 가능한 단위로 줄인다.
   - `as_only_neg`와 `as_only_sep`처럼 수학적으로 동일한 조건은 최종 primary table에서 중복 행으로 두지 않는다.
   - 동일 조건은 diagnostic/ablation table로만 보고한다.

## V3 단계 구성

V3는 한 번에 거대한 full run으로 가지 않고, 세 개의 gate를 통과하도록 설계한다.

### V3a: Measurement Validation

목적:

- loglik description scoring이 행동 선택의 적절한 proxy인지 점검한다.
- action description wording confound를 측정한다.
- PAS cosine-projection selection이 loglik scoring과 다른 정보를 주는지 비교한다.

핵심 추가:

1. paraphrase set
   - 각 action에 대해 3-5개 paraphrase description을 만든다.
   - 같은 action label이지만 다른 표현을 가진 description에서 선택이 안정적인지 본다.

2. label-masked descriptions
   - aggressive/cooperative 단어를 직접적으로 풍기는 표현을 줄인 neutral wording 버전을 만든다.
   - 예: "압박한다", "협조한다" 같은 명시적 persona cue를 줄이고 행동 결과 중심으로 표현한다.

3. ranking endpoint
   - argmax뿐 아니라 aligned action들의 평균 rank, loglik margin, probability mass를 보고한다.

Primary measurement check:

| metric | 목적 |
|---|---|
| action-level stability across paraphrases | 같은 행동이 표현만 바뀌어도 선택되는지 |
| label-level alignment | persona label 기준 alignment |
| description-cue sensitivity | persona 어휘가 강한 description에 과도하게 끌리는지 |
| top-1 margin | 선택이 확실한지, 근소한 차이인지 |
| aligned probability mass | argmax 하나가 아니라 label 전체 선호가 움직이는지 |
| PAS/loglik agreement | hidden-space cosine 선택과 likelihood 선택이 같은 행동을 고르는지 |

Decision gate:

- paraphrase 간 action 선택이 너무 불안정하면, V3b/V3c의 primary endpoint를 argmax alignment가 아니라 probability mass/rank metric으로 바꾼다.
- PAS와 loglik가 크게 불일치하면, 두 endpoint를 분리해서 보고한다. PAS는 activation-space action alignment, loglik는 text likelihood preference로 해석한다.

### PAS Selection Mode

PAS here means projection-based action selection:

```text
action description -> hidden-state embedding
steering/persona vector -> target direction
select action with max cosine(hidden(action), steering_vector)
```

Current status:

- Prototype modules exist:
  - `src/projection/selector.py`
  - `src/projection/embedder.py`
  - `tests/test_selector.py`
- PAS is not yet integrated into `scripts/03_run_experiment.py`.
- No final V2 PAS results have been produced.

V3 should add PAS as a formal scoring mode:

| scoring mode | primary object measured |
|---|---|
| `loglik` | text likelihood preference over action descriptions |
| `pas` | hidden-space similarity between action embedding and steering vector |
| `generate` | open-ended generated action, parsed afterward |

PAS-specific outputs:

- selected action id
- cosine scores for all actions
- action ranking by cosine
- PAS/loglik agreement
- PAS alignment rate
- PAS aligned probability-like mass if cosine scores are softmaxed

PAS caveat:

- PAS depends heavily on how action embeddings are computed.
- The action embedding layer should match or be systematically compared against steering layers.
- PAS still uses action descriptions, so paraphrase sensitivity must be tested.

### V3b: Causal Control Experiment

목적:

- AS 효과가 persona-specific vector 때문인지 검증한다.

Primary condition matrix:

| condition | persona prompt | hook | vector |
|---|---:|---:|---|
| `neutral_baseline` | no | no | none |
| `zero_hook` | no | yes | zero vector |
| `random_vector` | no | yes | random normalized vector |
| `wrong_persona_vector` | no | yes | opposite/wrong target direction |
| `as_only` | no | yes | target contrast vector |
| `weak_prompt_baseline` | weak | no | none |
| `weak_prompt_as` | weak | yes | target contrast vector |
| `strong_prompt_baseline` | strong | no | none |
| `strong_prompt_as` | strong | yes | target contrast vector |

해석:

- `as_only > zero_hook`: 단순 hook 삽입 효과를 넘는지.
- `as_only > random_vector`: 임의 표현 perturbation을 넘는지.
- `as_only > wrong_persona_vector`: persona-specific direction인지.
- `weak_prompt_as > weak_prompt_baseline`: ceiling이 낮은 prompt에서 AS가 추가 정보를 주는지.
- `strong_prompt_as ~= strong_prompt_baseline`: ceiling/saturation 여부를 확인하는 보조 결과.

Important:

- `as_only_neg`와 `as_only_sep`는 V3 primary condition에서 제거한다.
- contrast direction과 orthogonalized direction 비교는 별도 vector diagnostic ablation으로 둔다.

### V3c: Held-Out Generalization

목적:

- in-sample 효과가 아니라 held-out scenario에서도 재현되는지 확인한다.

Split:

| split | 용도 | 비율 |
|---|---|---:|
| train | vector extraction pair 생성 | 50% |
| dev | alpha/layer/vector mode 선택 | 25% |
| test | final report only | 25% |

권장:

- 현재 30 scenario만으로는 split하면 test cluster가 너무 작다.
- 가능하면 60-100 scenario로 확장한다.
- 최소 조건:
  - train 30
  - dev 15
  - test 15
- 권장 조건:
  - train 50
  - dev 25
  - test 25

Rule:

- test split에서는 alpha/layer/vector design을 바꾸지 않는다.
- test 결과를 보고 condition을 추가하거나 제거하지 않는다.
- final claim은 test split 기준으로 작성한다.

## V3 데이터 설계

### Scenario schema

각 scenario는 다음 정보를 가져야 한다.

```json
{
  "scenario_id": "sc_001",
  "split": "train|dev|test",
  "context": "...",
  "available_actions": ["..."],
  "domain": "conflict|resource|conversation|risk|teamwork",
  "difficulty": "low|medium|high",
  "baseline_bias": null
}
```

### Action schema

각 action은 label과 paraphrase를 분리한다.

```json
{
  "action_id": "sc_001_act_001",
  "scenario_id": "sc_001",
  "label": "aggressive|cooperative|neutral",
  "canonical_description": "...",
  "paraphrases": [
    {"paraphrase_id": "p1", "text": "...", "style": "neutral"},
    {"paraphrase_id": "p2", "text": "...", "style": "direct"},
    {"paraphrase_id": "p3", "text": "...", "style": "minimal"}
  ],
  "labeler_votes": {
    "aggressive": 0,
    "cooperative": 0,
    "neutral": 0
  }
}
```

### Label validation

V3에서 action label은 최소한 다음 중 하나를 만족해야 한다.

1. human double annotation + disagreement review.
2. LLM-as-judge 3-5회 vote + human spot check.
3. 기존 라벨 유지 시, "unvalidated labels"로 limitation에 명시.

Primary report에는 label confidence를 포함한다.

## V3 Vector Design

Primary vector:

- `v_contrast = aggressive - cooperative`
- aggressive target: `+v_contrast`
- cooperative target: `-v_contrast`

Diagnostic vectors:

| vector | 목적 |
|---|---|
| `zero` | hook만 켜졌을 때 영향 |
| `random_normalized` | 임의 perturbation 영향 |
| `wrong_persona` | direction specificity |
| `orthogonalized` | contrast와 거의 같은지 재확인 |
| `neutral_anchor` | non-neutral component 재검증 |
| `shuffled_label` | pair label 무작위화 control |

Vector extraction rules:

- train split만 사용한다.
- dev/test scenario action text를 vector extraction에 쓰지 않는다.
- pair 품질 audit을 통과한 pair만 사용한다.
- vector file에는 다음 metadata를 저장한다.
  - source split
  - pair count
  - pair hash
  - model name
  - layer list
  - extraction method
  - normalization setting

## V3 Metrics

Primary metrics:

| metric | 정의 |
|---|---|
| aligned probability mass | aligned action descriptions의 softmax probability 합 |
| top-1 alignment | argmax action이 target persona label인지 |
| aligned mean rank | aligned actions의 평균 rank |
| top-1 margin | 1위와 2위 normalized loglik 차이 |
| paraphrase stability | paraphrase variant 간 same-action consistency |
| PAS top-1 alignment | cosine-projection selector가 target persona action을 고르는 비율 |
| PAS/loglik agreement | 두 selector가 같은 action을 고르는 비율 |

Secondary metrics:

| metric | 정의 |
|---|---|
| free-form parse_ok | generation endpoint quality |
| free-form alignment | parsed generation alignment |
| off-target style shift | 말투/길이/강도 변화 |
| entropy | action distribution uncertainty |

Why probability mass should be primary:

- top-1은 30 scenario에서 0.033 단위로만 움직인다.
- probability mass는 더 연속적인 정보가 있어 power가 높다.
- action description wording에 대한 민감도를 rank/mass에서 더 잘 볼 수 있다.

## V3 Statistics Plan

Primary model:

```text
aligned_mass ~ condition * persona + (1 | scenario_id)
```

If aligned_mass is continuous:

- beta regression or mixed-effects linear model on logit-transformed mass.
- cluster bootstrap as robust companion.

For top-1 alignment:

```text
top1_aligned ~ condition * persona + (1 | scenario_id)
```

Required reporting:

- condition-level estimates with 95% CI.
- paired condition differences with scenario-cluster bootstrap CI.
- persona interaction effects.
- scenario random-effect variance.
- split-specific results: dev and test separated.

Do not use:

- row-independent two-proportion z-test.
- chi-square independence test over all rows as if independent.
- Bonferroni correction over legacy pairwise tests as the main inference.

## V3 Code Architecture

V3 should move from script-driven experiments to manifest-driven experiments.

### Proposed directories

```text
configs/
  experiments/
    v3_measurement_validation.yaml
    v3_controls_dev.yaml
    v3_final_test.yaml

src/
  data/
    schemas.py
    splits.py
    validation.py
  scoring/
    base.py
    loglik.py
    generation.py
    ranking.py
  experiments/
    manifest.py
    runner.py
    conditions.py
  controls/
    vector_factory.py
    random_vectors.py
    zero_vectors.py
  reporting/
    final_report.py
    tables.py

results/
  v3/
    runs/
      <run_id>/
        manifest.yaml
        config_snapshot/
        raw/
        metrics/
        figures/
        report.md
```

### Run manifest

Every run should save:

```yaml
run_id: "v3_dev_001"
model_name: "Qwen/Qwen2.5-3B-Instruct"
git_commit: "<commit>"
created_at: "<timestamp>"
scenario_split: "dev"
conditions:
  - neutral_baseline
  - zero_hook
  - random_vector
  - wrong_persona_vector
  - as_only
scoring_modes:
  - loglik
  - ranking
vector_file: "results/v3/vectors/train_contrast_fp16.pt"
vector_hash: "<sha256>"
alpha: 4.0
layers: [18, 21, 24]
seed: 42
```

This prevents accidental in-sample tuning and makes later review easier.

## V3 Implementation Roadmap

### Step 1: Design freeze

- Create `docs/v3_experiment_design.md`.
- Decide primary endpoint:
  - recommended: aligned probability mass.
- Decide minimum scenario count.
- Decide whether to generate paraphrases manually, via LLM, or both.

### Step 2: Data split and schema

- Add `split` field to scenarios.
- Add paraphrase support to action definitions.
- Add label validation fields.
- Write tests for split integrity:
  - no test scenario in vector extraction pairs.
  - no dev/test text in train vector file metadata.

### Step 3: Measurement validation

- Implement paraphrase-aware loglik scoring.
- Integrate PAS as `--selection_mode pas`.
- Compute action stability, label stability, rank, margin, probability mass.
- Compare loglik-selected actions against PAS-selected actions.
- Decide whether top-1 alignment remains acceptable.

### Step 4: Negative controls

- Implement zero vector.
- Implement random normalized vector with fixed seed.
- Implement wrong-persona vector condition.
- Implement shuffled-label vector extraction.

### Step 5: Dev sweep

- Use train vectors.
- Sweep alpha/layer on dev only.
- Choose one final setting before test.

### Step 6: Final test run

- Run fixed condition matrix on test.
- Do not tune after test.
- Report primary result on test only.

### Step 7: Secondary generation endpoint

- Run smaller free-form generation comparison.
- Report parse_ok and alignment separately from forced-choice endpoint.
- Use this only as ecological support, not as the primary clean estimate.

## V3 Success Criteria

V3 is successful if it can answer:

1. Does AS beat zero/random/wrong-vector controls?
2. Does AS generalize to held-out scenarios?
3. Does AS affect aligned probability mass, not just top-1 argmax?
4. Are effects stable across paraphrases of the same actions?
5. Does weak_prompt_as improve over weak_prompt_baseline without ceiling saturation?
6. Are labels and contrastive pairs sufficiently audited to support the interpretation?

Minimum defensible claim after V3:

> Under a held-out forced-choice action preference setup, persona contrast activation steering increases persona-aligned action probability mass beyond zero/random/wrong-vector controls.

Stronger claim only if secondary generation agrees:

> The same steering direction also shifts open-ended NPC action generation toward the target persona, although generation remains noisier than forced-choice scoring.

## Immediate Next Coding Tasks

1. Add V3 config files under `configs/experiments/`.
2. Add scenario split utilities.
3. Extend action schema to support paraphrases.
4. Integrate PAS cosine-projection selection into the main experiment runner.
5. Extend loglik selector to score multiple paraphrases per action.
6. Add probability-mass, rank, margin, and PAS/loglik agreement metrics.
7. Add zero/random/wrong-vector controls.
8. Add run manifest saving.
9. Add split leakage tests.
