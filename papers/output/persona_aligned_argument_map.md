# Persona-Aligned LLM NPC Behavior: 전제-이유-근거 문헌 정리

작성일: 2026-05-03

정리 범위: 현재 폴더 안의 PDF 26개만 사용했다. 폴더에 없는 외부 논문은 제외했다.

## 0. 배정 원칙

이 문서는 사용자 제공 글의 구분을 따른다.

| 구분 | 의미 | 이 문서에서의 역할 |
| --- | --- | --- |
| 전제 | 이유와 주장을 이어주는 원칙 | 왜 이 문제가 게임 NPC 연구에서 중요한지 정당화 |
| 이유 | 글쓴이가 주장에 대해 생각해낸 설명 | 왜 AS+PAS가 필요한지 해석 |
| 근거 | 바깥세상에서 확인 가능한 사실 | 논문에서 직접 확인되는 실험 결과, 수치, 방법 |

추가 제약도 반영했다.

- 한 번 배정한 논문은 다른 전제, 이유, 근거에서 다시 사용하지 않는다.
- 하나의 전제, 이유, 근거 항목에 들어가는 논문은 최대 3개로 제한한다.
- 논문을 적게 쓰는 대신, 각 논문이 해당 항목에 어떻게 기여하는지 더 구체적으로 설명한다.

## 1. 중심 주장

> LLM NPC의 persona alignment 문제는 성격처럼 말하게 만드는 문제가 아니라, 성격에 맞는 행동을 선택하게 만드는 문제다. 따라서 표현 단계에서 내부 표현을 성격 방향으로 기울이는 AS와, 행동 선택 단계에서 같은 성격 축으로 후보 행동을 다시 정렬하는 PAS를 결합해야 한다.

논증의 흐름은 다음과 같다.

```text
전제: 좋은 게임 NPC는 말, 감정, 가치, 계획, 행동이 같은 persona 원리에 맞게 정렬되어야 한다.
이유: LLM은 언어적 선언과 실제 선택 사이에 괴리를 보이며, 기존 방법은 최종 행동 선택 단계의 정렬을 충분히 다루지 못한다.
근거: 폴더 내 논문들이 보여주는 성격 표현, 행동 불일치, persona 불안정성, activation steering, planning 한계를 바탕으로 한다.
주장: AS+PAS 이중 개입으로 표현 단계와 선택 단계를 동시에 정렬해야 한다.
```

## 2. 전제 정리

### 전제 1. 게임 NPC의 설득력은 말투가 아니라 반복되는 행동 패턴에서 나온다

**전제 문장.** 플레이어는 NPC를 자기소개 한 문장이 아니라 반복되는 선택, 감정 반응, 기억된 관계, 계획된 행동으로 판단한다. 따라서 persona-aligned NPC는 "그럴듯하게 말하는 NPC"가 아니라 "그 성격대로 반응하고 행동하는 NPC"여야 한다.

| 문헌 | 전제에 기여하는 내용 |
| --- | --- |
| Generative Agents | 이 논문은 LLM 에이전트를 단발 대사 생성기가 아니라 memory stream, reflection, planning을 가진 시뮬레이션 객체로 설계한다. 에이전트는 경험을 기억으로 저장하고, 중요한 기억을 반성으로 압축하며, 이를 바탕으로 다음 행동을 계획한다. 즉 NPC의 설득력은 "무엇을 말했는가"보다 "이전 경험을 바탕으로 어떤 행동 패턴을 이어가는가"에서 나온다는 전제를 뒷받침한다. |
| Player-Driven Emergence in LLM-Driven Game Narrative | 이 논문은 GPT-4 기반 NPC와 플레이어의 상호작용 로그를 narrative node graph로 변환해 분석한다. 플레이어는 원래 설계되지 않은 물건, 장소, 해결 전략, 서사 노드를 만들어냈고, 이는 LLM NPC의 반응이 게임 서사 구조 자체에 영향을 줄 수 있음을 보여준다. 따라서 NPC 행동 정합성은 단순 품질 문제가 아니라 narrative coherence의 조건이다. |
| AN APPRAISAL-BASED CHAIN-OF-EMOTION ARCHITECTURE | 이 논문은 감정을 바로 생성하지 않고, 먼저 상황을 appraisal한 뒤 감정 반응을 도출하는 Chain-of-Emotion 구조를 제안한다. 게임 에이전트의 감정이 설득력을 가지려면 사건이 NPC의 목표, 기대, 관계에 어떤 의미를 갖는지 평가해야 한다는 점을 보인다. 이는 행동 역시 내부 상태와 연결된 반응이어야 한다는 전제와 맞닿아 있다. |

### 전제 2. 말, 가치 선언, 선호 선언과 실제 행동 선택은 구분되어야 한다

**전제 문장.** LLM이 어떤 원칙, 가치, 선호를 말로 선언하는 것과 실제 상황에서 그에 맞는 행동을 고르는 것은 같은 능력이 아니다. 따라서 persona alignment는 발화 정렬만으로 평가될 수 없다.

| 문헌 | 전제에 기여하는 내용 |
| --- | --- |
| 9253_Large_Language_Models_Oft | 이 논문은 Words and Deeds Consistency Test를 통해 word question과 deed question을 엄격히 대응시켜 평가한다. 결과는 여러 모델과 opinion, non-ethical value, ethical value, theory/application 영역에서 말과 행동의 불일치가 널리 나타남을 보여준다. 특히 word alignment나 deed alignment 중 하나만 수행했을 때 다른 쪽이 예측 가능하게 개선되지 않는다는 점은, 말과 행동을 별도 축으로 보아야 한다는 전제를 강하게 만든다. |
| Mind the Value-Action Gap | 이 논문은 ValueActionLens로 LLM의 stated value와 value-informed action의 정렬을 평가한다. 12개 문화권, 11개 사회 주제, 14.8k 행동 사례를 사용해 모델이 어떤 가치를 동의한다고 말해도 그 가치에 맞는 행동을 고르지 않을 수 있음을 보인다. NPC 관점에서는 "나는 충성스럽다", "나는 냉혹하다" 같은 선언만으로 실제 선택을 예측할 수 없다는 의미다. |
| Alignment Revisited | 이 논문은 stated preference와 revealed preference를 분리해 측정한다. 일반 원칙 질문에서는 특정 선호를 말하더라도, 맥락이 들어간 forced binary choice에서는 다른 선택을 할 수 있고, 작은 prompt format 변화도 선택을 바꿀 수 있다. 이는 persona prompt로 선언된 원칙이 구체 상황의 행동 선택으로 자동 전이되지 않는다는 전제를 뒷받침한다. |

### 전제 3. 장기 persona drift와 단일 결정 mismatch는 다른 문제다

**전제 문장.** persona drift는 긴 대화에서 정체성이 흐트러지는 문제이고, 단일 결정 mismatch는 한 선택 시점에서 발화와 행동이 어긋나는 문제다. 두 문제는 연결되어 있지만 같은 처방으로 해결되지 않는다.

| 문헌 | 전제에 기여하는 내용 |
| --- | --- |
| Can LLM Agents Maintain a Persona in Discourse | 이 논문은 OCEAN 성격 특성을 부여한 두 에이전트의 dyadic discourse를 생성한 뒤, judge agent들이 원래 성격을 추론하게 한다. 결과는 LLM이 성격 기반 대화를 어느 정도 생성할 수 있지만, trait adherence가 모델 조합과 담화 설정에 따라 크게 달라진다는 것이다. 이는 장기 담화 속 persona 유지가 별도의 안정성 문제임을 보여준다. |
| Persistant Instability in LLM's Peronality Measurements | PERSIST는 25개 open-source 모델과 2M+ 응답을 대상으로 질문 순서, reasoning mode, persona prompt, conversation history가 성격 측정에 미치는 영향을 분석한다. 질문 재배열만으로도 큰 측정 변동이 생기고, reasoning이나 history가 오히려 variability를 키울 수 있다는 결과는 persona 안정성이 단순 prompt 강화로 해결되지 않음을 보여준다. |
| Examining Identity Drift in Conversationsof LLM Agents | 이 논문은 9개 LLM의 multi-turn conversation에서 identity drift를 분석한다. 큰 모델이 더 큰 drift를 보일 수 있고, persona 부여가 항상 identity 유지에 도움이 되지는 않는다는 결과를 제시한다. 따라서 장기 drift는 memory/state 관리 문제로 따로 다루고, 본 프로젝트의 단일 행동 선택 mismatch와 구분해야 한다. |

### 전제 4. 고수준 persona 개념은 내부 표현 방향으로 다룰 수 있다

**전제 문장.** 공격성, 협조성, 탐색성, 정직성 같은 고수준 행동/성격 개념은 모델 내부 표현 공간에서 방향성 있는 벡터로 근사될 수 있다. 이 전제가 있어야 AS가 가능해진다.

| 문헌 | 전제에 기여하는 내용 |
| --- | --- |
| The Linear Representation Hypothesis and the Geometry of LLMs | 이 논문은 고수준 개념이 표현 공간에서 선형 방향으로 나타난다는 가설을 counterfactual 관점에서 정식화한다. linear probing과 steering이 어떻게 연결되는지 설명하고, 어떤 inner product를 쓰는지가 해석과 제어에 중요함을 보인다. 이는 persona vector를 단순한 비유가 아니라 이론적으로 다룰 수 있는 대상으로 만들어준다. |
| Steering Llama 2 via Contrastive Activation Addition | 이 논문은 긍정/부정 예시 쌍의 activation 차이를 평균해 steering vector를 만들고, 추론 중 residual stream에 더해 행동 경향을 조절한다. sycophancy나 hallucination 같은 고수준 행동을 조절할 수 있다는 점은 "성격 방향 벡터"를 구성해 AS에 사용할 수 있다는 실험적 근거가 된다. |
| CONTROLLING LARGE LANGUAGE MODEL AGENTS WITH ENTROPIC ACTIVATION STEERING | 이 논문은 activation steering을 순차 의사결정 에이전트의 exploration/action entropy 조절에 적용한다. 즉 steering이 단순 문체 변경에 머무르지 않고, 에이전트가 어떤 행동 경향을 보이는지에도 영향을 줄 수 있음을 보인다. 이는 NPC 행동 선택 이전의 내부 상태를 성격 방향으로 기울이는 AS의 가능성을 뒷받침한다. |

### 전제 5. 추론은 계획이나 행동 선택과 동일하지 않다

**전제 문장.** LLM이 단계별 이유를 잘 말한다고 해서 장기적으로 일관된 계획을 세우거나 persona에 맞는 행동을 고른다는 보장은 없다. 행동 선택에는 별도의 평가와 선택 구조가 필요하다.

| 문헌 | 전제에 기여하는 내용 |
| --- | --- |
| Why Reasoning Fails to Plan | 이 논문은 step-wise reasoning이 long-horizon planning에서 greedy policy처럼 작동해 초기의 근시안적 선택을 증폭시킬 수 있다고 분석한다. locally plausible한 다음 행동을 고르는 능력과 미래 결과를 고려해 계획을 세우는 능력은 다르다는 것이다. 이는 "그럴듯한 이유 생성"과 "persona에 맞는 행동 선택"을 분리해야 한다는 전제를 제공한다. |
| PlanGenLLMs A Modern Survey of LLM Planning Capabilities | 이 survey는 LLM planning을 completeness, executability, optimality, representation, generalization, efficiency라는 기준으로 정리한다. 계획 능력은 단순 언어 생성 능력이나 CoT 성능으로 환원되지 않으며, 실행 가능성과 최적성 같은 별도 기준을 요구한다. 게임 NPC의 행동도 이런 planning/action selection 관점에서 봐야 한다. |

## 3. 이유 정리

이유는 논문에서 바로 나온 문장이 아니라, 위 전제와 문헌들을 바탕으로 이 프로젝트가 도출하는 해석이다.

### 이유 1. 성격 표현 능력은 출발점일 뿐, 행동 선택 정렬을 보장하지 않는다

**이유 문장.** LLM이 성격 검사나 글쓰기에서 특정 persona를 잘 표현할 수 있다는 사실은 중요하지만, 그것만으로 NPC가 실제 상황에서 같은 성격에 맞는 행동을 고를 것이라고 볼 수는 없다.

| 문헌 | 이 이유와의 연결 |
| --- | --- |
| PersonaLLM | 이 논문은 GPT-3.5와 GPT-4에 Big Five persona를 부여했을 때 BFI 응답과 story writing에서 지정 성격과 일관된 언어 패턴이 나타남을 보인다. 따라서 본 프로젝트는 "LLM이 성격을 전혀 표현하지 못한다"는 문제가 아니라, 표현된 성격이 행동 선택까지 이어지는지가 문제라고 설정할 수 있다. |
| Evaluating the Efficacy of LLMs to Emulate Realistic Human Personalities | 이 논문은 IPIP-50과 50,000명 이상의 인간 설문 데이터를 기준으로 LLM의 personality profile alignment를 평가한다. frontier model은 일부 경우 매우 높은 정렬을 보이지만, local model은 0% 정렬까지 떨어진다. 이는 persona 표현 능력이 모델별로 크게 흔들리므로, 프로젝트가 행동 선택 단계에서 추가 정렬 장치를 두어야 한다는 이유가 된다. |

### 이유 2. 자기보고 성격은 행동 성격의 충분조건이 아니다

**이유 문장.** NPC가 스스로를 어떤 성격이라고 말하거나, 설문에서 그 성격처럼 응답하더라도, 실제 선택 상황에서 같은 경향을 보인다고 가정하면 안 된다.

| 문헌 | 이 이유와의 연결 |
| --- | --- |
| Is Self-knowledge and Action Consistent or Not | 이 논문은 self-knowledge와 action을 구분하고, LLM이 주장하는 성격과 실제 행동 경향이 얼마나 일치하는지 평가한다. 결론은 LLM이 인간 같은 성향을 흉내낼 수는 있지만, 자기보고와 행동이 안정적으로 맞물리지는 않는다는 것이다. 따라서 본 프로젝트는 NPC의 persona card나 자기소개를 행동 선택의 충분한 근거로 삼지 않는다. |
| The Personality Illusion | 이 논문은 instruction tuning이나 persona injection이 self-report trait expression을 안정화할 수 있지만, 그 self-report가 실제 behavior를 안정적으로 예측하지 못한다고 주장한다. 이는 LLM personality가 "행동 기반 성격"이라기보다 "언어적 일관성의 환상"일 수 있음을 보여주며, 행동 선택 단계의 별도 정렬이 필요하다는 이유를 만든다. |

### 이유 3. 내부 목표와 상호작용 맥락은 행동 선택에서 쉽게 흐려진다

**이유 문장.** LLM이 암묵 목표나 persona 맥락을 내부적으로 유지한다고 가정하기 어렵고, 다른 에이전트와 상호작용하면 언어적 개성도 섞일 수 있다. 따라서 행동 선택 시점에는 현재 persona와 목표를 명시적으로 다시 반영해야 한다.

| 문헌 | 이 이유와의 연결 |
| --- | --- |
| Probing the Lack of Stable Internal Beliefs in LLMs | 이 논문은 20문답식 추리 게임에서 LLM이 명시적으로 target을 계속 제공받지 않으면 implicit goal을 유지하지 못하고 drift한다는 점을 보인다. NPC가 비밀 목표, 적대감, 충성심 같은 상태를 암묵적으로 유지한다고 기대하면 행동 선택이 흔들릴 수 있다. PAS는 행동 후보를 고르는 순간에 이런 상태를 다시 반영하는 장치로 해석될 수 있다. |
| LLM Agents in Interaction | 이 논문은 persona-conditioned agents가 상호작용 후 상대의 언어 스타일에 맞춰지는 linguistic alignment를 보인다고 분석한다. 장기 상호작용에서 개성이 희석될 수 있다는 뜻이다. 따라서 대화 맥락이 아무리 풍부해도, 최종 행동 선택 단계에서 persona 방향을 다시 확인하는 절차가 필요하다는 이유가 된다. |

### 이유 4. 성격은 프로필 문장이 아니라 상황별 반응 함수로 다루어야 한다

**이유 문장.** "공격적인 인물", "협조적인 인물", "악역" 같은 persona는 배경 설명이 아니라 같은 상황에서 다른 행동을 고르게 만드는 반응 규칙이어야 한다.

| 문헌 | 이 이유와의 연결 |
| --- | --- |
| P-React Synthesizing Topic-Adaptive Reactions of Personality Traits | 이 논문은 Big Five 성격 특성을 mixture of LoRA experts와 Personality Specialization Loss로 모델링한다. 핵심은 성격을 이름표가 아니라 topic-adaptive reaction으로 다룬다는 점이다. 이 관점은 PAS에서 행동 후보를 성격 축에 따라 평가하는 설계와 잘 맞는다. |
| Too Good to be Bad | 이 논문은 safety-aligned LLM이 morally ambiguous하거나 villainous한 캐릭터를 충실히 연기하는 데 어려움을 보인다고 분석한다. 특히 deceitful, manipulative 같은 특성은 안전 원칙과 충돌해 피상적 공격성으로 대체될 수 있다. 따라서 persona-aligned action selection은 성격 정합성과 안전/서사 균형을 동시에 고려해야 한다. |

## 4. 근거 정리

근거는 논문에서 직접 확인 가능한 사실이다. 이 절에는 전제와 이유에서 이미 사용한 논문을 다시 넣지 않았다.

### 근거 1. 학습 기반 role-playing은 강력하지만 training-free 보완층의 필요성을 남긴다

| 문헌 | 확인 가능한 사실 | 이 프로젝트에서의 쓰임 |
| --- | --- | --- |
| Character-LLM A Trainable Agent for Role-Playing | 이 논문은 profile, experience, emotional state를 "experience upload" 형태로 구성하고, 특정 인물처럼 행동하도록 모델을 학습시키는 Character-LLM을 제안한다. 평가에서는 인터뷰와 LLM judge를 통해 trained agent가 캐릭터 지식과 경험을 더 잘 유지하는지 확인한다. | 이 결과는 캐릭터 특성을 데이터와 학습으로 강화할 수 있음을 보여준다. 동시에 캐릭터마다 경험 데이터를 구성하고 학습해야 하므로, 본 프로젝트의 AS+PAS는 이런 학습 기반 모델 위에도 얹을 수 있는 runtime 보완층으로 포지셔닝할 수 있다. |
| CoSER Coordinatin LLM_Based Persona Simulation of Established Roles | CoSER는 771권의 책에서 17,966명 캐릭터를 수집하고, dialogue, plot summary, character experience, inner thoughts 등을 포함한 role-playing 데이터와 GCA 평가 방식을 제안한다. CoSER 8B/70B 모델을 통해 established character role-playing 성능을 높인다. | 이 논문은 "캐릭터답게 행동하려면 풍부한 상황·경험 데이터가 중요하다"는 강한 근거다. 다만 대규모 데이터와 학습이 필요하므로, 본 프로젝트는 그런 방식과 경쟁하기보다 최종 행동 선택을 보정하는 lightweight layer로 제안될 수 있다. |

### 근거 2. 다중 캐릭터 시스템은 역할별 모듈화가 필요하지만, 최종 선택은 여전히 별도 문제다

| 문헌 | 확인 가능한 사실 | 이 프로젝트에서의 쓰임 |
| --- | --- | --- |
| Neeko Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent | Neeko는 캐릭터별 LoRA block을 독립적으로 학습하고, gating network로 현재 역할에 맞는 LoRA를 동적으로 선택한다. seen roles뿐 아니라 unseen/novel characters를 다루기 위해 fusion과 expansion 전략도 제안한다. | 이 논문은 NPC가 많아질수록 하나의 prompt가 아니라 역할별 모듈화가 필요하다는 사실을 보여준다. 하지만 어떤 모듈이 활성화되더라도, 게임 엔진 안에서는 최종 행동 후보 중 하나를 골라야 한다. AS+PAS는 이런 adapter 기반 시스템 뒤에서 action selection을 persona 축에 맞추는 후단 절차가 될 수 있다. |

### 근거 3. 내부 표현에 의도가 있어도 실제 행동으로 이어지지 않을 수 있다

| 문헌 | 확인 가능한 사실 | 이 프로젝트에서의 쓰임 |
| --- | --- | --- |
| ASA Training-Free Representation Engineering for Tool-Calling Agents | ASA는 tool necessity가 mid-layer activation에서는 거의 해독 가능하지만, 모델이 실제 tool mode로 들어가지 않는 Lazy Agent failure mode를 지적한다. training-free steering과 router-conditioned mixture를 사용해 strict tool-use F1을 0.18에서 0.50으로 개선하고 false positive rate도 0.15에서 0.05로 낮춘다. | 이 결과는 본 프로젝트의 PAS 필요성을 가장 직접적으로 뒷받침한다. 내부 표현에 어떤 의도나 필요성이 존재해도 그것이 실제 행동으로 자동 변환되지는 않는다. 마찬가지로 AS가 persona 방향을 내부에 만들더라도, 최종 행동 선택 단계에서 다시 정렬하지 않으면 speech-action mismatch가 남을 수 있다. |

## 5. 논증에 바로 넣을 수 있는 문단

### 문제 제기

LLM NPC의 persona alignment는 성격을 말로 표현하는 능력만으로 해결되지 않는다. 성격, 가치, 선호를 언어로 선언하는 능력과 실제 상황에서 그에 맞는 행동을 고르는 능력은 분리될 수 있다. 따라서 본 연구는 NPC의 persona alignment를 발화 스타일 제어가 아니라 행동 선택 정렬 문제로 재정의한다.

### 연구 공백

기존 접근은 서로 다른 층위를 다룬다. 일부 연구는 memory, reflection, planning을 통해 장기 행동을 구성하고, 일부 연구는 성격 표현과 자기보고를 평가하며, 또 다른 연구는 activation steering으로 내부 표현을 조절한다. 그러나 게임 NPC의 최종 행동은 대개 후보 행동 중 하나를 고르는 이산적 선택으로 구현된다. 이 선택 단계 자체를 persona axis에 맞춰 다시 정렬하는 절차가 부족하다는 점이 본 연구의 공백이다.

### AS+PAS 정당화

AS는 내부 표현을 성격 방향으로 기울이는 단계다. 하지만 내부 표현이 바뀌었다고 해서 그 신호가 최종 행동으로 그대로 보존된다고 가정할 수 없다. PAS는 AS 이후의 표현을 행동 후보 embedding에 투영하여, 후보 행동 중 persona 축과 가장 잘 맞는 선택을 고르는 단계다. 이 구조는 표현 단계와 선택 단계를 같은 성격 방향으로 두 번 정렬한다는 점에서 기존 방법의 빈틈을 보완한다.

## 6. 최종 논증 지도

```text
[전제 1]
게임 NPC의 설득력은 반복되는 행동 패턴에서 나온다.
  사용 논문 수: 3

[전제 2]
말, 가치 선언, 선호 선언과 실제 행동 선택은 구분되어야 한다.
  사용 논문 수: 3

[전제 3]
장기 persona drift와 단일 결정 mismatch는 다른 문제다.
  사용 논문 수: 3

[전제 4]
고수준 persona 개념은 내부 표현 방향으로 다룰 수 있다.
  사용 논문 수: 3

[전제 5]
추론은 계획이나 행동 선택과 동일하지 않다.
  사용 논문 수: 2

[이유 1]
성격 표현 능력은 출발점일 뿐, 행동 선택 정렬을 보장하지 않는다.
  사용 논문 수: 2

[이유 2]
자기보고 성격은 행동 성격의 충분조건이 아니다.
  사용 논문 수: 2

[이유 3]
내부 목표와 상호작용 맥락은 행동 선택에서 쉽게 흐려진다.
  사용 논문 수: 2

[이유 4]
성격은 프로필 문장이 아니라 상황별 반응 함수로 다루어야 한다.
  사용 논문 수: 2

[근거 1]
학습 기반 role-playing은 강력하지만 training-free 보완층의 필요성을 남긴다.
  사용 논문 수: 2

[근거 2]
다중 캐릭터 시스템은 역할별 모듈화가 필요하지만, 최종 선택은 여전히 별도 문제다.
  사용 논문 수: 1

[근거 3]
내부 표현에 의도가 있어도 실제 행동으로 이어지지 않을 수 있다.
  사용 논문 수: 1

[주장]
AS+PAS는 persona-aligned LLM NPC behavior를 위해 필요한 이중 개입 구조다.
```

## 7. 배정 검증

- 사용한 PDF 수: 26개
- 각 항목당 사용 논문 수: 3개 이하
- 전제, 이유, 근거 사이 논문 중복: 없음
- 외부 논문 사용: 없음
