# LLM Persona/Game Agent Literature Review

작성일: 2026-05-02

이 문서는 현재 폴더의 PDF 19개를 읽고 정리한 문헌 요약이다. 핵심 관점은 "LLM 기반 게임 NPC/페르소나 에이전트를 어떻게 설계하고, 성격과 감정의 일관성을 어떻게 평가하며, 모델 내부 표현을 어떻게 제어할 수 있는가"이다.

## 1. 전체 요약

이 문헌 묶음은 크게 네 질문으로 연결된다.

1. LLM 에이전트가 게임 세계에서 믿을 만한 NPC처럼 행동할 수 있는가?
2. 프롬프트나 학습으로 부여한 성격, 역할, 감정은 장기 대화에서 유지되는가?
3. LLM의 "성격"은 실제 행동 성향인가, 아니면 언어적으로 그럴듯한 자기보고인가?
4. activation steering, LoRA, representation engineering 같은 내부 제어 기법으로 행동을 더 안정적으로 만들 수 있는가?

가장 중요한 결론은 다음과 같다. LLM은 짧은 상호작용에서는 매우 그럴듯한 성격, 감정, 역할 연기를 생성한다. 그러나 긴 대화, 암묵 목표 유지, 실제 행동 과제, 도덕적으로 복잡한 캐릭터 연기에서는 일관성이 쉽게 무너진다. 따라서 게임 NPC를 만들 때는 단순히 "너는 이런 성격이다"라는 프롬프트보다, 기억, 반성, 계획, 감정 평가, 명시적 상태 추적, 역할별 어댑터, 평가 루프를 결합한 구조가 필요하다.

프로젝트 관점에서 가장 실용적인 방향은 다음과 같다.

- NPC 아키텍처: Generative Agents의 memory/reflection/planning 구조를 기본 뼈대로 삼는다.
- 감정 표현: Chain-of-Emotion처럼 상황을 먼저 appraisal하고 감정을 생성한다.
- 페르소나 안정성: identity drift와 implicit consistency 문제를 전제로 두고, 대화 히스토리만 믿지 말고 별도의 belief/persona state를 둔다.
- 역할 특화: Character-LLM, CoSER, Neeko, P-React처럼 캐릭터 지식과 성격 반응을 데이터 또는 LoRA 단위로 분리한다.
- 평가: 자기보고식 성격 검사만 쓰지 말고, 실제 행동 로그, 언어 패턴, 인간 평가, 자동 평가를 함께 본다.
- 제어: activation steering 계열은 추론 시점에 행동 경향을 조절할 가능성이 있지만, 안전성과 오용 위험도 같이 고려해야 한다.

## 2. 주제별 분류

### A. 게임 NPC와 상호작용형 에이전트 아키텍처

- Generative Agents
- AN APPRAISAL-BASED CHAIN-OF-EMOTION ARCHITECTURE
- Player-Driven Emergence in LLM-Driven Game Narrative
- LLM Agents in Interaction

핵심 아이디어: LLM NPC는 단발 대사 생성기가 아니라 기억, 반성, 계획, 감정 상태, 상호작용 기록을 가진 시뮬레이션 객체로 설계해야 한다. 특히 플레이어가 예측하지 못한 행동을 할 수 있으므로, 고정 스크립트보다 동적 narrative graph와 로그 기반 분석이 중요하다.

### B. 역할 연기와 캐릭터 시뮬레이션

- Character-LLM
- CoSER
- Neeko
- Too Good to be Bad

핵심 아이디어: 캐릭터 연기는 프롬프트만으로는 한계가 있고, 캐릭터 경험/대사/내면 생각/상황 정보를 학습하거나 검색하는 방식이 효과적이다. 그러나 안전 정렬된 모델은 악역, 기만적 캐릭터, 조작적 캐릭터처럼 비친사회적 역할을 충실히 수행하지 못하는 경향이 있다.

### C. 성격, 페르소나, 정체성 일관성 평가

- PersonaLLM
- Evaluating the Efficacy of LLMs to Emulate Realistic Human Personalities
- Is Self-knowledge and Action Consistent or Not
- The Personality Illusion
- Examining Identity Drift in Conversations of LLM Agents
- Probing the Lack of Stable Internal Beliefs in LLMs

핵심 아이디어: LLM은 Big Five 같은 성격 프로필을 언어적으로 잘 흉내낼 수 있지만, 그 성격이 실제 행동을 안정적으로 예측하지는 않는다. 자기보고, 글쓰기 스타일, 대화 행동, 암묵 목표 유지 능력 사이에 간극이 있다. 따라서 "성격이 있다"보다 "성격처럼 보이는 언어 패턴을 특정 조건에서 생성한다"가 더 정확하다.

### D. 내부 표현 제어와 activation steering

- Steering Llama 2 via Contrastive Activation Addition
- CONTROLLING LARGE LANGUAGE MODEL AGENTS WITH ENTROPIC ACTIVATION STEERING
- ASA Training-Free Representation Engineering for Tool-Calling Agents
- The Linear Representation Hypothesis and the Geometry of LLMs

핵심 아이디어: LLM 내부에는 행동, 개념, 불확실성, 도구 사용 의도 등이 어느 정도 선형 방향으로 표현될 수 있다. activation steering은 프롬프트나 파인튜닝 없이 추론 시점에 행동 경향을 조절할 수 있는 방법을 제공한다. 게임 NPC에서는 탐색성, 감정 강도, 공격성, 협조성, 도구 사용 여부 같은 행동 변수를 조절하는 데 응용 가능하다.

## 3. 논문별 핵심 아이디어

### 1. Generative Agents

LLM 기반 에이전트를 작은 마을 시뮬레이션에 배치하여, 기억하고 반성하고 계획하는 인간 유사 행동을 생성한 연구다. 핵심 구조는 memory stream, reflection, planning이다. 에이전트는 경험을 자연어 기억으로 저장하고, 중요한 기억을 반성으로 요약하며, 그 결과를 다시 행동 계획에 사용한다.

프로젝트 시사점: 게임 NPC의 기본 아키텍처로 가장 직접적이다. 단순 대화보다 "오늘 무엇을 했는가", "누구를 만났는가", "무엇을 기억하는가"를 상태로 관리해야 사회적 행동이 나온다.

### 2. AN APPRAISAL-BASED CHAIN-OF-EMOTION ARCHITECTURE

게임 에이전트의 감정 생성을 심리학의 appraisal 이론에 맞춘 Chain-of-Emotion 구조로 설계한다. 모델이 바로 감정을 말하게 하지 않고, 상황을 평가한 뒤 감정을 도출하게 하면 감정 지능 과제와 사용자 경험 평가에서 더 좋은 결과가 나온다.

프로젝트 시사점: NPC 감정은 "기쁨/분노" 라벨을 직접 넣기보다, 사건이 목표/관계/기대에 어떤 의미인지 먼저 평가하는 중간 단계를 둬야 설득력 있다.

### 3. Player-Driven Emergence in LLM-Driven Game Narrative

GPT-4 기반 NPC가 있는 텍스트 어드벤처에서 플레이어들이 원래 설계되지 않은 서사 노드를 만들어내는 현상을 분석한다. 플레이 로그를 narrative node graph로 변환하여, 플레이어가 새로운 물건, 장소, NPC, 해결 전략을 제안하며 emergent narrative를 만든다는 점을 보인다.

프로젝트 시사점: LLM NPC의 장점은 "모든 답을 미리 스크립트로 쓰지 않아도 된다"는 데 있다. 대신 emergent 행동을 기록하고, 디자이너가 채택 가능한 노드와 막다른 노드를 구분하는 도구가 필요하다.

### 4. LLM Agents in Interaction

성격 프로필을 부여한 GPT-3.5 에이전트들이 협동 글쓰기 과제를 수행할 때, 성격 일관성과 언어적 alignment가 어떻게 변하는지 본다. 성격은 어느 정도 드러나지만, 상호작용 후 파트너의 언어 스타일에 맞춰지며 두 에이전트의 언어가 비슷해진다.

프로젝트 시사점: NPC끼리 오래 대화하게 두면 각자의 개성이 희석될 수 있다. 개성 유지와 사회적 적응 사이의 균형 장치가 필요하다.

### 5. Character-LLM

특정 인물의 프로필, 경험, 감정 상태를 "experience upload" 형태로 구성하고, 이를 통해 Beethoven, Cleopatra 같은 특정 인물 역할을 수행하는 trainable agent를 만든다. 프롬프트 기반 캐릭터보다 경험 기억과 캐릭터 지식을 더 잘 유지하는 것이 목표다.

프로젝트 시사점: 중요한 NPC는 프롬프트 한 장보다 사건 경험 데이터가 필요하다. "캐릭터가 겪은 장면"을 데이터 단위로 관리하면 장기적 캐릭터성이 강해진다.

### 6. CoSER

771권의 책에서 17,966명 캐릭터 데이터를 구성하고, 대사, 플롯 요약, 캐릭터 경험, 내면 생각 등을 활용해 established character role-playing을 훈련/평가한다. given-circumstance acting이라는 방식으로 여러 캐릭터가 같은 장면 속에서 순차적으로 역할을 수행한다.

프로젝트 시사점: 역할 연기 평가에는 단순 Q&A보다 "주어진 장면 속에서 이 캐릭터라면 어떻게 말하고 행동하는가"가 더 적합하다.

### 7. Neeko

Multi-Character Role-Playing을 위해 캐릭터별 LoRA 블록과 gating network를 사용한다. 역할마다 별도 LoRA를 두고, 현재 캐릭터에 맞는 어댑터를 동적으로 선택한다. 새로운 캐릭터에는 fusion/expansion 전략을 사용한다.

프로젝트 시사점: NPC가 많을 때 하나의 거대한 프롬프트로 처리하기보다, 역할별 adapter나 profile module을 분리하는 설계가 효율적이다.

### 8. P-React

Big Five 성격 특성을 더 심리학적으로 모델링하기 위해 OCEAN-Chat 데이터셋과 personality-specialized LoRA experts를 사용한다. Personality Specialization Loss를 통해 각 expert가 특정 성격 특성 반응에 특화되도록 만든다.

프로젝트 시사점: 성격을 캐릭터 배경 설명이 아니라 행동 반응 함수로 다뤄야 한다. 예를 들어 같은 사건에도 높은 agreeableness와 낮은 agreeableness가 다르게 반응하도록 설계할 수 있다.

### 9. PersonaLLM

GPT-3.5/GPT-4에 Big Five 성격 프로필을 부여하고, BFI 검사와 이야기 쓰기 과제로 성격 표현을 평가한다. 자기보고식 BFI 점수는 지정한 성격과 잘 맞고, 글쓰기에서도 일부 성격별 언어 패턴이 나타난다. 다만 AI가 쓴 글임을 알리면 인간 평가자의 성격 판별 정확도가 낮아진다.

프로젝트 시사점: 성격 프롬프트는 표면적 언어 스타일을 꽤 잘 바꾼다. 하지만 이것만으로 실제 의사결정 일관성을 보장한다고 보면 위험하다.

### 10. Evaluating the Efficacy of LLMs to Emulate Realistic Human Personalities

IPIP-50과 50,000개 이상의 인간 성격 설문 데이터를 기준으로 여러 frontier/local LLM이 인간 성격 프로필과 얼마나 잘 정렬되는지 평가한다. 일부 로컬 모델은 특정 프로필에서 0% 정렬을 보였고, GPT-4 같은 frontier 모델은 일부 프로필에서 100% 정렬을 보였다.

프로젝트 시사점: 모델 선택이 성격 NPC 품질에 큰 영향을 준다. 작은 로컬 모델을 쓸 경우 성격 프롬프트가 잘 먹히는지 별도 검증해야 한다.

### 11. Is Self-knowledge and Action Consistent or Not

LLM이 성격 설문에서 스스로 주장하는 성격과 실제 시나리오 행동이 일치하는지 평가한다. 결론은 상당한 불일치가 있다는 것이다. LLM은 인간 같은 성향을 흉내낼 수 있지만, 자기보고와 행동 경향이 안정적으로 맞물리지는 않는다.

프로젝트 시사점: NPC 성격을 설문으로 설정했다면, 실제 행동 로그로 다시 검증해야 한다. "나는 책임감이 높다"는 응답과 실제 책임감 있는 행동은 별개다.

### 12. The Personality Illusion

LLM의 성격 표현을 훈련 단계, 자기보고, 행동 과제, persona injection 관점에서 체계적으로 분석한다. instruction tuning/RLHF는 자기보고 성격 표현을 안정화하지만, 자기보고 성격이 실제 행동을 잘 예측하지 못한다. persona injection도 자기보고는 바꾸지만 행동에는 제한적이거나 불일치한 효과를 낸다.

프로젝트 시사점: LLM personality는 "행동 기반 성격"이라기보다 "언어적 일관성의 환상"일 수 있다. NPC 평가에서 겉으로 그럴듯한 말투만 보면 안 된다.

### 13. Examining Identity Drift in Conversations of LLM Agents

9개 LLM의 장기 대화에서 identity drift를 분석한다. 큰 모델일수록 identity drift가 더 크고, 모델 계열 차이보다 파라미터 크기의 영향이 컸으며, persona를 부여해도 identity 유지에 꼭 도움이 되지는 않았다.

프로젝트 시사점: 장기 대화 NPC는 시간이 갈수록 말투와 정체성이 변할 수 있다. 대화 히스토리 압축, persona reminder, state anchoring이 필요하다.

### 14. Probing the Lack of Stable Internal Beliefs in LLMs

20문답식 추리 게임에서 LLM이 몰래 정한 목표를 대화 중 계속 유지할 수 있는지 본다. 목표를 명시적으로 컨텍스트에 계속 제공하지 않으면 내부 목표가 drift한다. 외부적으로 그럴듯한 답을 하더라도 내부 belief가 안정적이라는 보장은 없다.

프로젝트 시사점: NPC의 비밀, 목표, 계획은 모델이 "암묵적으로 기억하겠지"라고 두면 안 된다. 별도 상태 저장소에 명시적으로 넣고 매 턴 제공해야 한다.

### 15. Too Good to be Bad

안전 정렬된 LLM이 악역, 반사회적 캐릭터, 도덕적으로 낮은 캐릭터를 얼마나 충실히 연기하는지 평가한다. 캐릭터의 도덕성이 낮아질수록 role-playing fidelity가 단조롭게 하락한다. 특히 deceitful, manipulative 같은 안전 원칙과 충돌하는 특성에서 약하다.

프로젝트 시사점: 게임에는 악역도 필요하지만, 모델은 악역을 무해한 거친 말투 정도로 평탄화할 수 있다. 안전을 유지하면서도 서사적 악역성을 표현하는 별도 가이드라인이 필요하다.

### 16. Steering Llama 2 via Contrastive Activation Addition

Contrastive Activation Addition(CAA)은 긍정/부정 예시 쌍의 activation 차이를 평균해 steering vector를 만들고, 추론 중 residual stream에 더해 모델 행동을 조절한다. sycophancy, hallucination 등 고수준 행동을 프롬프트나 파인튜닝 위에서 추가로 제어할 수 있음을 보인다.

프로젝트 시사점: NPC의 말투, 정직성, 협조성, 감정 강도 같은 고수준 행동 변수를 런타임에 조절하는 가능성을 제공한다.

### 17. CONTROLLING LARGE LANGUAGE MODEL AGENTS WITH ENTROPIC ACTIVATION STEERING

Entropic Activation Steering(EAST)은 LLM 에이전트의 탐색성을 activation 수준에서 조절한다. bandit 같은 순차 의사결정 환경에서 모델이 너무 빨리 한 행동에 고착되는 문제를 보이고, action entropy와 uncertainty 표현을 조절해 더 탐색적인 행동을 유도한다.

프로젝트 시사점: NPC가 매번 익숙한 선택만 반복하거나 너무 빨리 한 전략에 고착될 때, 탐색성/불확실성 제어가 유용할 수 있다.

### 18. ASA Training-Free Representation Engineering for Tool-Calling Agents

Activation Steering Adapter(ASA)는 도구 호출이 필요한 의도가 activation에는 드러나지만 실제 행동으로 이어지지 않는 representation-behavior gap을 다룬다. 학습 없이 추론 시점 steering과 router-conditioned mixture를 사용해 strict tool-use F1을 개선한다.

프로젝트 시사점: NPC가 게임 API, 검색, 기억 조회, 행동 실행 도구를 호출해야 할 때 "필요성을 알지만 호출하지 않는" lazy agent 문제가 생길 수 있다. 도구 호출 결정을 별도 제어 대상으로 봐야 한다.

### 19. The Linear Representation Hypothesis and the Geometry of LLMs

고수준 개념이 모델 표현 공간에서 선형 방향으로 표현된다는 가설을 counterfactual 관점에서 정식화한다. linear probing과 steering이 어떻게 연결되는지 설명하고, 어떤 inner product를 쓰느냐가 해석과 제어에 중요함을 보인다.

프로젝트 시사점: activation steering을 사용할 때 "개념 방향"은 단순 벡터 계산 문제가 아니라, 어떤 표현 공간과 기하를 가정하는지에 따라 달라진다. 이론적 기반 문헌으로 유용하다.

## 4. 프로젝트 적용 제안

### 권장 NPC 구조

```text
Player/Event
  -> Perception
  -> Appraisal: 이 사건이 NPC의 목표, 관계, 기대에 어떤 의미인가?
  -> Emotion State Update
  -> Memory Retrieval
  -> Belief/Goal State Check
  -> Plan
  -> Dialogue/Action Generation
  -> Log + Reflection
```

이 구조는 Generative Agents의 memory/reflection/planning, Chain-of-Emotion의 appraisal, Probing Stable Internal Beliefs의 explicit state tracking을 합친 형태다.

### 평가 체크리스트

- 성격 자기보고: BFI, IPIP 등
- 행동 일관성: 같은 성격이 실제 선택에서도 반복되는가?
- 장기 대화 안정성: 10턴, 30턴, 100턴 후 말투와 목표가 유지되는가?
- 암묵 목표 유지: 비밀 목표나 계획을 컨텍스트 없이 유지할 수 있는가?
- 상호작용 후 개성 유지: NPC끼리 대화한 뒤 서로 말투가 과도하게 섞이지 않는가?
- 감정 설득력: 감정 라벨보다 appraisal 근거가 자연스러운가?
- 플레이어 창발성: 플레이어가 만든 새로운 narrative node를 기록하고 분류하는가?
- 안전/서사 균형: 악역이 안전하게 표현되면서도 지나치게 무해화되지 않는가?

### 구현 우선순위

1. NPC별 persona card, memory stream, current goals, hidden beliefs를 별도 상태로 관리한다.
2. 매 턴 LLM 입력에 전체 히스토리가 아니라 검색된 기억과 현재 상태 요약을 제공한다.
3. 대사 생성 전에 appraisal/emotion planning 단계를 둔다.
4. 행동 로그를 자동으로 평가하여 identity drift, goal drift, personality drift를 측정한다.
5. 주요 NPC는 캐릭터별 예시 대사/경험 데이터를 구축한다.
6. 여러 NPC가 필요하면 역할별 adapter 또는 retrieval profile을 분리한다.
7. 나중에 필요하면 activation steering 계열을 탐색성, 도구 호출, 말투 제어에 실험적으로 붙인다.

## 5. 한 줄 결론

LLM은 게임 NPC의 대사 생성기를 넘어 "상호작용하는 캐릭터 시뮬레이터"가 될 수 있지만, 믿을 만한 캐릭터성을 얻으려면 프롬프트보다 상태 관리, 기억 구조, 감정 평가, 행동 기반 검증이 훨씬 중요하다.
