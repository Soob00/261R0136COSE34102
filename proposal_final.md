# Final Project Proposal
## Team 12
### What Context Errors Hurt Small LMs Most? A Controlled Study on RumourEval Stance Detection

---

## 1. 연구 배경 및 문제 정의

RumourEval 2019 Task A는 루머 스레드의 reply-level stance classification을 다루는 공식 shared task이다. 각 reply는 `support`, `deny`, `query`, `comment` 중 하나로 분류되며, 특히 `support`와 `deny`는 데이터 불균형 때문에 상대적으로 어렵다. 이 태스크에서는 단순히 reply 텍스트만 보는 것이 아니라, source tweet, parent reply, sibling reply 등 thread context를 어떻게 활용하느냐가 성능을 크게 좌우한다.

그러나 context는 항상 도움이 되는 것이 아니다. 같은 thread 안에 있는 문맥이라도, 어떤 문맥은 stance 판단에 직접 유용하지만 어떤 문맥은 topic만 맞고 stance 신호는 약하며, 어떤 문맥은 오히려 잘못된 방향으로 모델을 유도할 수 있다. 최근 연구들은 큰 모델도 긴 social context를 잘 처리하지 못할 수 있고, 오염되거나 혼동적인 evidence는 성능뿐 아니라 calibration까지 악화시킬 수 있음을 보여주었다. 따라서 본 연구는 단순히 "context를 넣으면 좋아지는가"를 묻지 않는다. 대신 **small LM이 어떤 종류의 context 오류에 특히 취약한가**를 정량적으로 분석한다.

이 프로젝트의 핵심 아이디어는 `context quality`를 일반론으로 다루지 않고, context를 오류 유형별로 나누어 통제된 조건에서 비교하는 것이다. 이를 통해 모델이 실패하는 패턴을 더 명확히 보이고, 소형 모델의 한계를 분석하는 실증적 결과를 얻고자 한다.

본 연구의 목표는 다음 한 문장으로 요약된다.

> RumourEval stance detection에서 small LM은 어떤 context 오류 유형에 가장 크게 흔들리는가?

---

## 2. 연구 질문

본 연구는 다음 질문에 답하는 것을 목표로 한다.

1. small LM은 RumourEval stance detection에서 useful context와 noisy context를 얼마나 다르게 활용하는가?
2. topic-relevant but stance-irrelevant context와 conflicting context는 어떤 클래스(`support`, `deny`, `query`, `comment`)에 가장 큰 손실을 주는가?
3. 모델 규모가 커질수록 context 오류에 대한 민감도는 얼마나 줄어드는가?
4. optional하게, thinking mode는 noisy context에서 작은 모델의 안정성을 개선하는가, 아니면 더 불안정하게 만드는가?

---

## 3. 가설

본 연구는 다음과 같은 단순하지만 검증 가능한 가설을 둔다.

- **H1.** useful context는 small LM의 stance 성능을 개선한다.
- **H2.** topic-relevant but stance-irrelevant context와 conflicting context는 성능을 유의하게 저하시킨다.
- **H3.** 이 저하는 `comment`보다 `support`와 `deny`에서 더 크게 나타난다.
- **H4.** 더 큰 small LM은 전체 성능은 높겠지만, context 오류 민감도는 완전히 사라지지 않는다.

---

## 4. 데이터 및 범위

### 4.1 메인 데이터셋

- **RumourEval 2019 Task A**
- task: reply-level stance classification
- labels: `support`, `deny`, `query`, `comment`
- primary metric: **macro-F1**

### 4.2 연구 범위

본 프로젝트는 stance classification에만 집중한다. veracity prediction이나 rumor resolution까지 확장하면 태스크 정의가 복잡해지고, 2개월 범위에서 해석이 어려워진다. 따라서 본 연구는 **stance only**로 범위를 고정한다.

### 4.3 평가용 context condition

각 reply에 대해 다음과 같은 context condition을 비교한다.

- **No context**: reply만 입력
- **Useful context**: source tweet + stance 판단에 도움이 되는 parent or thread context
- **Topic-relevant but stance-irrelevant context**: 주제는 맞지만 stance 신호는 약한 context
- **Conflicting / misleading context**: reply의 stance를 잘못 유도하는 context
- **Optional lexical distractor**: stance cue처럼 보이지만 실제 label과 직접 연결되지 않는 context

이 taxonomy의 목적은 단순한 context 유무 비교가 아니라, 모델이 **어떤 오류 유형에서 무너지는지** 분리해서 보는 데 있다.

---

## 5. 제안 방법

### 5.1 기본 접근

본 연구는 복잡한 retrieval system을 새로 구축하지 않고, **통제된 context stress test**로 문제를 다룬다. 즉, context를 더 많이 가져오는 시스템을 만드는 대신, 동일한 reply에 서로 다른 context condition을 붙여 모델의 반응을 비교한다.

### 5.2 모델

작은 규모에서 현실적으로 실험 가능한 모델을 사용한다.

- **Qwen2.5-0.5B-Instruct**
- **Qwen2.5-1.5B-Instruct**

가능하다면 보조 sanity baseline으로 reply-only encoder classifier를 추가해, prompt-based small LM과 전통적 fine-tuning baseline을 비교한다. 다만 핵심 비교축은 small LM 내부의 scale 차이와 context condition 차이다.

### 5.3 입력 형식

각 실험에서는 동일한 reply를 기준으로 context condition만 바꾼다.

예시 입력 구성:

- `reply only`
- `reply + source tweet`
- `reply + source tweet + parent reply`
- `reply + noisy / conflicting context`

모델 출력은 4-class stance label 하나로 고정한다.

### 5.4 분석 축

메인 분석은 다음의 두 축이다.

1. **Model size**: 0.5B vs 1.5B
2. **Context condition**: useful / irrelevant / conflicting / no-context

이 구조를 통해 성능 저하가 모델 크기 때문인지, context 오류 때문인지, 혹은 둘의 상호작용 때문인지 분리해서 볼 수 있다.

---

## 6. Baselines 및 Ablation

### 6.1 Baselines

- **Reply-only baseline**: context 없이 target reply만 입력
- **Reply + source baseline**: 가장 기본적인 context 추가
- **Reply + useful context baseline**: 제안한 strongest context condition

### 6.2 Ablation

- **Context source ablation**
  - reply only
  - reply + source tweet
  - reply + parent reply
  - reply + selected useful context

- **Context error ablation**
  - useful context
  - topic-relevant but stance-irrelevant context
  - conflicting context
  - optional lexical distractor

- **Scale ablation**
  - 0.5B vs 1.5B

- **Optional thinking ablation**
  - thinking vs non-thinking
  - noisy context에서만 제한적으로 비교

---

## 7. 평가 지표

RumourEval의 공식 평가는 macro-F1이지만, class imbalance가 크기 때문에 macro-F1만으로는 충분하지 않다. 따라서 다음 지표를 함께 보고한다.

- **Macro-F1**
- **Per-class F1** for `support`, `deny`, `query`, `comment`
- **Minority-class average F1**: `support` + `deny`
- **Context sensitivity gap**: useful context 대비 conflicting context에서의 성능 감소량
- **Confusion matrix**: 어떤 클래스가 어떤 오류 조건에서 가장 쉽게 흔들리는지 확인

가능하다면 bootstrap confidence interval을 사용해 조건 간 차이가 우연이 아닌지 점검한다.

---

## 8. 기대 결과

본 연구가 보여주고자 하는 핵심 결과는 다음과 같다.

1. small LM은 RumourEval stance detection에서 context를 무조건적으로 활용하는 것이 아니라, context 오류 유형에 따라 민감도가 크게 달라진다.
2. 특히 conflicting context는 `support`와 `deny` 같은 informative minority class에서 더 큰 성능 저하를 일으킬 가능성이 높다.
3. 모델 규모가 커지면 전반적인 성능은 좋아지더라도, noisy context에 대한 취약성이 완전히 사라지지는 않을 수 있다.
4. 결과적으로 "context matters"가 아니라 **"어떤 context error가 small LM을 가장 망가뜨리는가"**를 정량적으로 제시할 수 있다.

---

## 9. 리스크 및 대응

### 9.1 context taxonomy 모호성

- **리스크**: useful / irrelevant / conflicting 경계가 모호할 수 있다.
- **대응**: 50~100개 샘플을 먼저 수작업으로 검수하고 annotation rule을 고정한다.

### 9.2 class imbalance

- **리스크**: support와 deny의 표본 수가 적어 결과가 흔들릴 수 있다.
- **대응**: macro-F1 외에 class-wise F1과 minority-class average F1을 함께 보고, seed를 여러 번 반복한다.

### 9.3 thinking mode의 영향이 작을 가능성

- **리스크**: thinking mode가 의미 있는 차이를 만들지 못할 수 있다.
- **대응**: main contribution에서 제외하고 optional ablation으로만 둔다.

### 9.4 context 생성 비용

- **리스크**: 전체 데이터셋에 대해 context를 정밀하게 구성하는 비용이 크다.
- **대응**: 전체를 건드리지 말고, dev/test subset 중심으로 controlled stress test를 수행한다.

---

## 10. 2개월 마일스톤

- **1-2주**: RumourEval Task A 전처리, baseline 파이프라인 구축, context taxonomy 초안 작성
- **3-4주**: useful / irrelevant / conflicting context condition 구성, 0.5B baseline 실험
- **5-6주**: 1.5B 실험 및 scale ablation, class-wise analysis
- **7주**: optional thinking ablation, error analysis, confidence interval 추정
- **8주**: 결과 정리, 시각화, 최종 보고서 및 발표 자료 작성

---

## 11. 기대 기여

1. **RumourEval stance detection에서 small LM의 context error sensitivity를 정량화**
   - 단순한 context 유무가 아니라, 오류 유형별 민감도를 보여준다.

2. **controlled context stress test 제안**
   - useful / irrelevant / conflicting context를 구분하여 실패 모드를 분석하는 실험 설계를 제시한다.

3. **small-model scale 분석**
   - 0.5B와 1.5B 비교를 통해 작은 모델에서 규모가 robustness에 어떤 영향을 주는지 분석한다.

4. **분석 중심의 실용적 프로젝트**
   - 새로운 대형 모델을 만들지 않고도, 기존 shared task에서 의미 있는 분석 결과를 도출한다.

---

## 12. 참고 방향

이 프로젝트는 RumourEval 2019 Task A와 후속 stance detection, social context robustness, noisy evidence sensitivity 관련 연구를 바탕으로 한다. 특히 다음 관점을 중심으로 선행연구를 정리할 예정이다.

- RumourEval stance classification과 class imbalance 문제
- social context가 stance detection에 미치는 영향
- noisy / misleading context가 model performance와 calibration에 주는 영향
- small LM scale과 robustness의 관계

---

## 13. 한 줄 요약

**RumourEval 2019 Task A에서 small LM이 어떤 context 오류 유형에 가장 취약한지를 controlled stress test로 분석하는 2개월짜리 실증 연구이다.**
