# AS Implementation Design v2

결정사항이 반영된 코딩용 설계 문서. 미결 사항 없음.

## 1. Goal

AS-first 아키텍처로 재시작한다.

AS(Activation Steering)는 contrastive example에서 persona direction vector를 추출하고, generation 중 model hidden state에 더해 NPC 출력을 target persona 방향으로 유도하는 방식이다.

PAS는 AS 검증 후 다음 단계에서 붙인다. 현재 구현에서는 model이 생성한 `<Action>...</Action>`이 final action이다.

## 2. Core Research Question

Activation steering 단독으로 LLM NPC의 generated speech와 generated action을 target persona axis로 shift시킬 수 있는가?

Persona axis:

- `aggressive`
- `cooperative`

Conditions:

- `neutral_baseline`: persona 없는 중립 prompt, no steering
- `prompt_baseline`: persona 명시 prompt, no steering
- `as_only`: 중립 prompt + persona vector 주입

`prompt_baseline` 없으면 AS 효과와 prompt 효과가 분리되지 않는다. 따라서 3-condition 구조를 필수로 둔다.

## 3. Minimal AS Pipeline

```text
contrastive pair templates
  -> positive/negative prompt 생성 (형식 동일, persona 단어만 다름)
  -> forward pass (generation 없음)
  -> 선택 layer의 last input token hidden state 캡처
  -> steering vector 계산: mean(positive) - mean(negative)
  -> unit vector 정규화 후 저장

scenario + target persona
  -> 중립 NPC prompt 빌드
  -> target persona vector를 선택 layer post-block residual에 주입
  -> [Speech] / <Action> 파싱
  -> raw result 저장
  -> speech tone, action persona label, distribution shift 평가
```

## 4. Implementation Decisions

Model:

- `Qwen/Qwen2.5-3B-Instruct`
- fp16
- hook path: `model.model.layers[i]`
- Qwen has 36 layers, so code indices are `0..35`

Layer groups:

```text
shallow: 6, 9, 12
middle:  18, 21, 24
deep:    30, 33, 35
```

The original design note used `36` as the last deep layer, but the implementation uses zero-based Python indices, so the last valid layer is `35`.

Token aggregation:

- last input token only
- no span average

Injection:

- hook target: post-block residual stream
- uniform injection only
- unit vector normalization

Core equation:

```text
h' = h + alpha * v_persona
```

Initial alpha values:

- `0.5`
- `1.0`
- `2.0`
- `4.0`

Default alpha:

- `2.0`

## 5. Prompt And Output Contract

The model must output exactly:

```text
[Speech] short in-character dialogue
<Action>action_id</Action>
```

Prompt action format:

```text
Available actions:
- act_001: Draw your sword and charge at the enemy
- act_002: Offer to trade supplies with the traveler
- act_003: Stand your ground and wait for their move
```

The model chooses action IDs, not action labels. Evaluation maps action ID to persona alignment.

## 6. Result Schema

Each JSONL row:

```json
{
  "condition": "as_only",
  "persona": "aggressive",
  "scenario_id": "scn_fantasy_001",
  "repeat": 0,
  "seed": 42,
  "model_name": "Qwen/Qwen2.5-3B-Instruct",
  "model_dtype": "float16",
  "alpha": 2.0,
  "layers": [18, 21, 24],
  "hook_target": "post_block_residual",
  "token_aggregation": "last_input_token",
  "vector_normalized": true,
  "gen_config": {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 128,
    "do_sample": true
  },
  "generated_text": "...",
  "speech": "...",
  "parsed_action": "act_001",
  "final_action": "act_001",
  "parse_ok": true,
  "timestamp": 1234567890.0,
  "git_commit": "abc123"
}
```

`final_action = parsed_action` is fixed in AS-only experiments.

## 7. Evaluation Design

Speech tone classification:

- `aggressive`
- `cooperative`
- `neutral`

The preferred full-run judge is `Gemma 4 E4B-it`. The current implementation also provides a heuristic classifier so unit tests and smoke checks do not require loading a judge model.

Action persona alignment:

```text
aggressive persona + aggressive action -> aligned
cooperative persona + cooperative action -> aligned
neutral action -> neutral
opposite persona action -> misaligned
missing/invalid action -> unknown
```

Metrics:

- parse_ok rate
- generated action persona alignment
- speech-action agreement
- action distribution
- action entropy
- condition-wise distribution shift

Statistical tests:

- Action distribution: chi-square + Bonferroni + Cramer's V
- Action entropy: Mann-Whitney U + Bonferroni + Cliff's delta
- Persona alignment rate: two-proportion z-test + Bonferroni

## 8. Implementation Order

1. `configs/steering.yaml`: model, layers, alpha range, hook target
2. `configs/experiment.yaml`: 3 conditions, persona list, scenario list, repeats
3. contrastive pair templates and generated pair file
4. `src/steering/extractor.py`: last input token hidden state capture
5. `src/steering/vector.py`: CAA + unit normalization + metadata persistence
6. `src/generation/prompt_builder.py`: action ID prompt contract
7. `src/generation/parser.py`: speech/action ID parsing
8. `src/steering/injector.py`: uniform post-block residual hook
9. `scripts/03_run_experiment.py`: 3-condition runner
10. `scripts/04_evaluate.py`: AS metrics and CSV/chart outputs
11. AS smoke tests and full run
12. after AS stabilizes: delete `src/projection/` and PAS tests

## 9. First Milestone

```text
10 scenarios x 2 personas x 3 conditions x 30 repeats

AS-only가 neutral_baseline 및 prompt_baseline 대비
측정 가능한 action distribution shift를 만드는지 확인.
PAS 및 action reranking 없음.
```

Success criteria:

- projection code 없이 실험 실행
- parse_ok >= 90%
- AS가 neutral_baseline 대비 action distribution 변화
- evaluation reports persona alignment, speech-action agreement, and entropy
- prompt_baseline vs as_only 분리 가능
