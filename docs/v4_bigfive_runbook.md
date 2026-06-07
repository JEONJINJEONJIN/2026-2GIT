# V4 Big Five Pilot Runbook

이 문서는 Big Five 기반 NPC 말-행동 일관성 실험을 실제로 실행할 때 따르는 운영 순서다. 목적은 실행자가 바뀌어도 같은 입력, 같은 split, 같은 vector, 같은 manifest 기준으로 pilot을 재현할 수 있게 하는 것이다.

## 1. Required Inputs

실제 pilot 실행 전 다음 파일이 있어야 한다.

```text
data/bfi/bfi44_scoring.csv
data/bfi/bfi44_item_text.local.csv
data/trait_bigfive/scenarios.jsonl
data/trait_bigfive/splits.json
data/trait_bigfive/contrastive_pairs/agreeableness.jsonl
results/v4_bigfive/vectors/bigfive_trait_vectors_fp16.pt
```

BFI scoring metadata는 repository에 둔다. 실제 문항 wording은 `data/bfi/bfi44_item_text.local.csv`에 로컬 전용으로 둔다.

```powershell
Copy-Item data\bfi\bfi44_item_text.local.example.csv data\bfi\bfi44_item_text.local.csv
```

그 다음 `text` column을 승인된 출처의 BFI-44 wording으로 채운다. `.local.csv` 파일은 git-ignored 상태라 커밋하지 않는다.

TRAIT 파일은 원본 export를 직접 넣는 것이 아니라 `scripts/11_convert_trait_bigfive.py`로 변환한 normalized schema를 사용한다. pilot은 기본적으로 `agreeableness`와 `dev` split을 사용한다.

## 2. Convert TRAIT Big Five Data

원본 TRAIT export가 `data/raw/trait_export.jsonl`에 있다고 가정한다.

```powershell
.venv\Scripts\python.exe scripts\11_convert_trait_bigfive.py `
  --input data/raw/trait_export.jsonl `
  --output data/trait_bigfive/scenarios.jsonl `
  --splits_output data/trait_bigfive/splits.json
```

확인 기준:

- `scenarios.jsonl`이 생성된다.
- `splits.json`이 생성된다.
- 출력 로그의 `By trait`에 Big Five 5축이 모두 포함된다.
- pilot만 먼저 돌릴 때도 `agreeableness`의 `dev` split row가 있어야 한다.

## 3. Build Contrastive Pairs

AS vector extraction은 train split의 high-vs-low action pair를 사용한다.

```powershell
.venv\Scripts\python.exe scripts\09_build_trait_contrastive_pairs.py `
  --scenarios_path data/trait_bigfive/scenarios.jsonl `
  --output_dir data/trait_bigfive/contrastive_pairs `
  --splits train `
  --traits agreeableness
```

full Big Five vector를 만들 때는 `--traits`를 생략하거나 5축을 모두 지정한다.

```powershell
.venv\Scripts\python.exe scripts\09_build_trait_contrastive_pairs.py `
  --scenarios_path data/trait_bigfive/scenarios.jsonl `
  --output_dir data/trait_bigfive/contrastive_pairs `
  --splits train
```

확인 기준:

- pilot: `data/trait_bigfive/contrastive_pairs/agreeableness.jsonl`
- full/control: 5축 각각의 `.jsonl` 파일

`unrelated_trait_vector` control을 실행하려면 최소 2개 trait vector가 필요하다. 따라서 control config를 돌릴 때는 agreeableness만 추출한 vector bundle로는 부족하다.

## 4. Preflight Readiness

Phase 2 전체 구현/입력 검증:

```powershell
.venv\Scripts\python.exe scripts\17_validate_bigfive_phase2.py
```

외부 데이터가 아직 없고 코드/설정 준비 상태만 확인하려면:

```powershell
.venv\Scripts\python.exe scripts\17_validate_bigfive_phase2.py --allow_missing_external
```

pilot readiness:

```powershell
.venv\Scripts\python.exe scripts\13_check_bigfive_readiness.py `
  --config configs/experiments/v4_bigfive_pilot.yaml
```

vector bundle까지 있는 경우:

```powershell
.venv\Scripts\python.exe scripts\13_check_bigfive_readiness.py `
  --config configs/experiments/v4_bigfive_pilot.yaml `
  --vectors_path results/v4_bigfive/vectors/bigfive_trait_vectors_fp16.pt
```

실행 기준:

- `ERROR`가 있으면 pilot scoring을 실행하지 않는다.
- `WARN vectors`는 prompt-only 조건만 먼저 확인할 때는 허용할 수 있다.
- AS/PAS 조건을 포함하면 `--vectors_path`가 필요하다.

현재 외부 데이터가 없을 때 예상되는 실패는 다음이다.

```text
ERROR bfi_item_text_overlay
ERROR trait_scenarios
WARN contrastive_pairs
WARN vectors
```

이 실패는 코드 문제가 아니라 입력 데이터 미준비 상태를 의미한다.

## 5. Prepare Run Plan

LLM scoring 전에 scenario-level run plan과 manifest bundle을 생성한다.

```powershell
.venv\Scripts\python.exe scripts\10_prepare_bigfive_pilot.py `
  --config configs/experiments/v4_bigfive_pilot.yaml `
  --model_config configs/model.yaml `
  --seed 42
```

산출물:

```text
results/v4_bigfive/runs/run_plan.csv
results/v4_bigfive/runs/plan_summary.json
results/v4_bigfive/runs/manifests/*/manifest.json
```

이 단계는 모델을 로드하지 않는다. 데이터/split/schema 검증과 실행 단위 고정이 목적이다.

## 6. Extract Big Five Trait Vectors

pilot용 agreeableness vector:

```powershell
.venv\Scripts\python.exe scripts\02_extract_vectors.py `
  --mode trait `
  --trait_pairs_dir data/trait_bigfive/contrastive_pairs `
  --traits agreeableness `
  --output_dir results/v4_bigfive/vectors `
  --output_file results/v4_bigfive/vectors/bigfive_trait_vectors_fp16.pt `
  --layer_group middle `
  --quantization fp16
```

control run까지 고려한 5축 vector:

```powershell
.venv\Scripts\python.exe scripts\02_extract_vectors.py `
  --mode trait `
  --trait_pairs_dir data/trait_bigfive/contrastive_pairs `
  --output_dir results/v4_bigfive/vectors `
  --output_file results/v4_bigfive/vectors/bigfive_trait_vectors_fp16.pt `
  --layer_group middle `
  --quantization fp16
```

확인 기준:

- `bigfive_trait_vectors_fp16.pt` 생성
- `trait_vector_cosine_similarities.csv` 생성
- readiness checker에서 `vectors: ok`

## 7. Run Pilot Scoring

prompt-only 조건만 먼저 확인:

```powershell
.venv\Scripts\python.exe scripts\12_run_bigfive_pilot.py `
  --config configs/experiments/v4_bigfive_pilot.yaml `
  --model_config configs/model.yaml `
  --conditions baseline,one_line_prompt,elaborate_prompt `
  --seed 42
```

AS/PAS 포함 pilot:

```powershell
.venv\Scripts\python.exe scripts\12_run_bigfive_pilot.py `
  --config configs/experiments/v4_bigfive_pilot.yaml `
  --model_config configs/model.yaml `
  --vectors_path results/v4_bigfive/vectors/bigfive_trait_vectors_fp16.pt `
  --alpha 1.0 `
  --layers 18,21,24 `
  --seed 42
```

free-form NPC 응답과 side-effect metric까지 같이 산출하려면 다음 옵션을 추가한다.

```powershell
  --generate_responses `
  --response_max_new_tokens 96
```

산출물:

```text
results/v4_bigfive/pilot_metrics/bfi_scores.csv
results/v4_bigfive/pilot_metrics/trait_scores.csv
results/v4_bigfive/pilot_metrics/consistency_metrics.csv
results/v4_bigfive/pilot_metrics/pas_loglik_agreement.csv
results/v4_bigfive/pilot_metrics/generated_responses.csv
results/v4_bigfive/pilot_metrics/side_effect_metrics.csv
results/v4_bigfive/pilot_metrics/run_manifest.json
```

`run_manifest.json`이 없으면 해당 metric bundle은 최종 분석에 포함하지 않는다.

## 8. Analyze Pilot Results

```powershell
.venv\Scripts\python.exe scripts\14_analyze_bigfive_results.py `
  --metrics_dir results/v4_bigfive/pilot_metrics `
  --reference_condition baseline `
  --bootstrap_samples 1000 `
  --bootstrap_seed 42
```

산출물:

```text
results/v4_bigfive/pilot_metrics/consistency_summary.csv
results/v4_bigfive/pilot_metrics/condition_differences.csv
results/v4_bigfive/pilot_metrics/paired_scenario_differences.csv
results/v4_bigfive/pilot_metrics/paired_bootstrap_ci.csv
results/v4_bigfive/pilot_metrics/pas_agreement_summary.csv
results/v4_bigfive/pilot_metrics/side_effect_summary.csv
results/v4_bigfive/pilot_metrics/analysis_report.md
results/v4_bigfive/pilot_metrics/final_report.md
```

판단 기준:

- AS/PAS 조건이 baseline 대비 consistency 또는 target attainment에서 방향성 있게 개선되는가.
- random/unrelated vector에서 같은 효과가 재현되지 않는가.
- PAS/loglik agreement가 지나치게 낮지 않은가.
- side-effect metrics에서 format validity가 유지되는가.

## 9. Side-Effect Metrics

free-form NPC response CSV가 있을 때 실행한다. CSV에는 최소 `raw` column이 있어야 하며, 가능한 경우 `valid_actions`를 `a1|a2|a3|a4` 형식으로 넣는다.

```powershell
.venv\Scripts\python.exe scripts\15_score_side_effects.py `
  --input results/v4_bigfive/pilot_metrics/generated_responses.csv `
  --output results/v4_bigfive/pilot_metrics/side_effect_metrics.csv
```

현재 deterministic metric:

- response length
- speech/action tag presence
- parse success
- format validity

coherence score와 perplexity는 별도 evaluator가 준비되면 같은 CSV schema에 추가한다.

## 10. Paraphrase Robustness

paraphrase 실험 결과 CSV에는 다음 column이 필요하다.

```text
condition,trait,target_direction,scenario_id,paraphrase_group_id,paraphrase_variant,loglik_selected_action,trait_score
```

실행:

```powershell
.venv\Scripts\python.exe scripts\16_analyze_paraphrase_robustness.py `
  --input results/v4_bigfive/trait_scores_with_paraphrases.csv `
  --output results/v4_bigfive/paraphrase_robustness.csv
```

해석:

- `mean_action_agreement`가 높을수록 같은 의미의 다른 표현에서 선택 행동이 안정적이다.
- `mean_trait_score_range`가 낮을수록 표현 변화에 따른 trait score 흔들림이 작다.

## 11. Control Vector Run

full control config는 다음으로 실행한다.

```powershell
.venv\Scripts\python.exe scripts\13_check_bigfive_readiness.py `
  --config configs/experiments/v4_bigfive_controls.yaml `
  --vectors_path results/v4_bigfive/vectors/bigfive_trait_vectors_fp16.pt
```

readiness가 통과하면:

```powershell
.venv\Scripts\python.exe scripts\12_run_bigfive_pilot.py `
  --config configs/experiments/v4_bigfive_controls.yaml `
  --model_config configs/model.yaml `
  --vectors_path results/v4_bigfive/vectors/bigfive_trait_vectors_fp16.pt `
  --alpha 1.0 `
  --layers 18,21,24 `
  --seed 42
```

control run의 핵심 판단:

- `trait_contrast`에서만 개선되고 `random`/`unrelated_trait`에서는 같은 개선이 나오지 않아야 persona-specific effect 주장이 가능하다.
- `zero`는 hook/injection path 자체의 부작용을 확인하는 기준선이다.

## 12. Stop Conditions

다음 중 하나라도 발생하면 full Big Five로 넘어가지 않는다.

- BFI text가 비어 있다.
- TRAIT `dev` split에 pilot trait scenario가 없다.
- AS/PAS 조건인데 vector bundle이 없다.
- `unrelated_trait` control인데 vector bundle에 trait vector가 1개뿐이다.
- `run_manifest.json`이 생성되지 않았다.
- format validity가 크게 떨어진다.
- random/unrelated vector가 target vector와 같은 수준의 개선을 보인다.

## 13. Full Big Five Promotion

pilot을 통과한 뒤에만 `configs/experiments/v4_bigfive_final.yaml`로 확장한다.

Phase 3 final readiness:

```powershell
.venv\Scripts\python.exe scripts\18_validate_bigfive_phase3.py `
  --vectors_path results/v4_bigfive/vectors/bigfive_trait_vectors_fp16.pt
```

외부 데이터와 pilot/dev tuning 값이 아직 없고 config scaffold만 확인하려면:

```powershell
.venv\Scripts\python.exe scripts\18_validate_bigfive_phase3.py `
  --allow_missing_external `
  --allow_missing_tuning
```

full run 원칙:

- dev split에서 alpha/layer를 선택한다.
- test split에서 최종 결과를 산출한다.
- test 이후에는 hyperparameter를 바꾸지 않는다.
- final report는 manifest hash와 config hash를 포함한다.

## 14. Class Demo

Demo readiness:

```powershell
.venv\Scripts\python.exe scripts\20_validate_bigfive_demo.py
```

Streamlit demo scaffold:

```powershell
.venv\Scripts\python.exe scripts\19_run_bigfive_demo.py
```

Demo behavior:

- Big Five 5축 slider를 보여준다.
- slider에서 가장 높은 trait에 맞는 scenario card를 고른다.
- prompt-only NPC와 prompt + AS/PAS NPC 응답을 나란히 보여준다.
- pilot 결과가 없으면 built-in sample responses를 사용한다.
- `results/v4_bigfive/pilot_metrics/generated_responses.csv`가 있으면 해당 응답을 사용한다.
- activation movement는 현재 PCA placeholder이며, 실제 activation dump가 준비되면 교체한다.
## 15. Live Inference Demo

The Streamlit app now includes a `Live Inference` tab. Unlike the other tabs,
this tab runs the model at request time and does not read generated outputs from
CSV.

Launch:

```powershell
.venv\Scripts\streamlit.exe run src\demo\bigfive_app.py `
  --server.address 127.0.0.1 `
  --server.port 8501 `
  --server.headless true
```

Heavy smoke test:

```powershell
.venv\Scripts\python.exe -m src.demo.live_inference `
  --trait extraversion `
  --direction high `
  --scenario-index 0
```

Expected behavior:

- The app opens with the original precomputed viewer tabs unchanged.
- The `Live Inference` tab shows trait, direction, alpha, layer, scenario, and
  token controls.
- Clicking `Generate live responses` loads the model once through
  `st.cache_resource`.
- The left column generates an elaborate prompt-only response.
- The right column generates an elaborate prompt plus AS response with hooks on
  the selected layers.
- The right column shows one PAS cosine number for the parsed action, or the
  top PAS action if the generated action cannot be parsed.

Default live settings:

```text
model config: configs/model.yaml
vector bundle: results/v4_bigfive/vectors/bigfive_trait_vectors_full_fp16.pt
generation: greedy
alpha: 4.0
layers: [18, 21, 24]
max_new_tokens: 96
scenario filter: selected trait only
```

First load can take about a minute depending on GPU and Hugging Face cache
state. Subsequent Generate clicks reuse the cached model.

If VRAM is exhausted:

- Stop the Streamlit process to release VRAM.
- Set `configs/model.yaml` to use 4-bit quantization:

```yaml
quantization: "4bit"
load_in_4bit: true
```

- Relaunch the app.
- If 4-bit still fails, use a smaller compatible Qwen instruct model for the
  live demo only and clearly say that the quantitative results still come from
  the fixed final experiment configuration.

Do not persist live responses into `results/v4_bigfive/`. The live tab is a
demo-only interaction surface, not part of the final test metrics.
