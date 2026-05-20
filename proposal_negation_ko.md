# 최종 프로젝트 제안서
## Team 12
### 한국어 부정문과 이중부정에 대한 언어모델의 취약성 분석

---

## 1. 연구 배경 및 문제 정의

부정문은 자연어 의미를 뒤집거나 제한하는 핵심 언어 현상이다. 그러나 최근 연구들은 사전학습 언어모델과 NLI 모델이 부정이 포함된 문장에서 여전히 불안정한 추론 성능을 보인다는 점을 반복적으로 보고하고 있다. 특히 negation cue가 포함되어 있어도 실제 라벨 결정에 얼마나 중요한지, 모델이 이를 진짜로 이해하는지, 혹은 표면 패턴만 학습하는지는 여전히 중요한 연구 문제다.

본 프로젝트는 한국어 환경에서 다음 질문을 다룬다.

- 한국어 NLI 모델은 부정문과 이중부정에서 실제로 얼마나 취약한가?
- 어떤 부정 표현 유형에서 오류가 특히 많이 발생하는가?
- 기존 공개 데이터셋과 일반 fine-tuning만으로는 왜 이 현상을 충분히 해결하지 못하는가?

본 연구는 새로운 대형 모델을 만드는 것이 아니라, 공개 한국어 NLI 데이터와 기존 사전학습 모델을 사용해 부정 현상에 대한 취약성을 현상 중심(phenomenon-oriented)으로 분석하는 것을 목표로 한다.

---

## 2. 핵심 아이디어

일반적인 NLI 성능만 보면 모델이 잘 작동하는 것처럼 보일 수 있다. 하지만 전체 평균 점수는 특정 언어 현상에서의 실패를 가릴 수 있다. 따라서 본 프로젝트는 전체 NLI 성능을 다시 보고하는 대신, 공개 한국어 NLI 데이터에서 부정 관련 샘플을 식별하고 세부 유형별로 분해해 성능을 측정한다.

핵심 접근은 다음과 같다.

1. 공개 한국어 NLI 데이터셋에서 부정문 및 이중부정 샘플을 자동 추출한다.
2. 규칙 기반으로 부정 유형 taxonomy를 만든다.
3. 여러 한국어 NLI 모델을 학습하거나 불러와 전체 성능과 부정 subset 성능을 비교한다.
4. 오류를 contradiction, entailment, neutral 혼동 패턴과 부정 cue 유형별로 분석한다.
5. 선택적으로 negation-focused augmentation 또는 재가중 학습이 실제로 도움되는지 소규모로 검증한다.

즉, 이 프로젝트의 핵심은 “한국어 모델이 부정문을 잘 못한다”를 막연히 주장하는 것이 아니라, 어떤 유형에서 얼마나 무너지는지 체계적으로 보여주는 것이다.

---

## 3. 데이터셋 및 실험 자원

### 3.1 주 데이터셋

다음 공개 데이터셋을 우선 사용한다.

- KorNLI
  - 출처: Ham et al. (2020), *KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding*
  - 링크: https://aclanthology.org/2020.findings-emnlp.39/
- KLUE-NLI
  - 공개 데이터셋 카드: https://huggingface.co/datasets/klue/viewer/nli

KorNLI는 규모가 크고, KLUE-NLI는 비교적 정제된 한국어 NLI 벤치마크이므로 두 데이터셋을 함께 사용하면 대규모 분석과 깔끔한 보고를 동시에 할 수 있다.

### 3.2 데이터 구축 원칙

본 프로젝트는 새로운 대규모 데이터셋 구축을 목표로 하지 않는다. 대신 기존 데이터셋에서 부정 현상을 자동 탐지하여 subset을 구성한다.

자동 탐지 대상 예시:

- `안`
- `못`
- `아니다`
- `없다`
- `-지 않다`
- `-지 못하다`
- 이중부정 패턴

초기에는 규칙 기반 필터로 후보를 추출하고, 이후 소규모 샘플만 수작업 검수한다. 이 방식은 데이터셋 구축 비용을 최소화하면서도 분석의 신뢰도를 확보할 수 있다.

---

## 4. 관련 연구 및 Anchor Papers

### 4.1 Anchor Papers

1. Ham et al. (2020), *KorNLI and KorSTS: New Benchmark Datasets for Korean Natural Language Understanding*  
   - 링크: https://aclanthology.org/2020.findings-emnlp.39/  
   - 역할: 한국어 NLI 실험의 기본 데이터 자원

2. She et al. (2023), *ScoNe: Benchmarking Negation Reasoning in Language Models With Fine-Tuning and In-Context Learning*  
   - 링크: https://aclanthology.org/2023.acl-short.154/  
   - 역할: negation reasoning을 별도 benchmark와 contrast set으로 평가해야 한다는 핵심 근거

3. Hossain et al. (2022), *An Analysis of Negation in Natural Language Understanding Corpora*  
   - 링크: https://aclanthology.org/2022.acl-short.81/  
   - 역할: 일반 NLU 코퍼스에서 negation이 과소대표되거나 중요하지 않은 경우가 많다는 문제 제기

### 4.2 Previous Work

4. Hosseini et al. (2021), *Understanding by Understanding Not: Modeling Negation in Language Models*  
   - 링크: https://aclanthology.org/2021.naacl-main.102/  
   - 부정 이해를 위한 별도 학습 objective가 필요할 수 있음을 보임

5. Gubelmann and Handschuh (2022), *Context Matters: A Pragmatic Study of PLMs’ Negation Understanding*  
   - 링크: https://aclanthology.org/2022.acl-long.315/  
   - negation 이해를 semantic 관점뿐 아니라 pragmatic context에서도 봐야 한다고 주장

6. Singh et al. (2023), *NLMs: Augmenting Negation in Language Models*  
   - 링크: https://aclanthology.org/2023.findings-emnlp.873/  
   - negation-aware augmentation이 실제 개선을 만들 수 있음을 제시

7. Rezaei and Blanco (2024), *Paraphrasing in Affirmative Terms Improves Negation Understanding*  
   - 링크: https://aclanthology.org/2024.acl-short.55/  
   - negation을 affirmative paraphrase와 함께 해석하는 접근이 도움될 수 있음을 보임

8. Weller et al. (2024), *A Benchmark for Negation Understanding in Information Retrieval*  
   - 링크: https://aclanthology.org/2024.eacl-long.139/  
   - negation 취약성이 NLI뿐 아니라 retrieval 같은 실용적 설정에서도 문제임을 보여줌

### 4.3 본 연구의 위치

기존 연구는 주로 영어 중심이며, 한국어에서 negation phenomenon를 NLI 관점에서 체계적으로 분석한 연구는 상대적으로 부족하다. 따라서 본 프로젝트는 “새로운 negation benchmark를 만드는 것”보다, 공개 한국어 NLI 데이터에서 실제 현상 수준의 취약성을 드러내고 그 유형별 패턴을 정리하는 데 초점을 둔다.

---

## 5. 연구 질문 및 가설

### RQ1
한국어 NLI 모델은 전체 성능 대비 부정문 subset에서 유의미한 성능 저하를 보이는가?

### RQ2
부정 cue 유형(`안`, `못`, `없다`, `아니다`, `-지 않다`, 이중부정`)에 따라 오류 패턴이 다르게 나타나는가?

### RQ3
contradiction 판단이 entailment 또는 neutral보다 부정 현상에 더 민감하게 무너지는가?

### RQ4
간단한 negation-focused augmentation 또는 loss reweighting이 취약성 완화에 실질적인 도움을 주는가?

예상 가설은 다음과 같다.

- H1: 전체 NLI 성능 대비 negation subset 성능은 유의미하게 낮다.
- H2: `안`, `못`보다 scope가 복잡한 `아니다`, `없다`, 이중부정에서 오류가 더 많다.
- H3: contradiction class가 가장 큰 성능 저하를 보인다.
- H4: 단순한 augmentation은 일부 회복을 만들 수 있지만, cue 유형 전반을 고르게 해결하지는 못한다.

---

## 6. 제안 파이프라인

### 6.1 Step 1: 부정 샘플 자동 추출

KorNLI와 KLUE-NLI에서 premise와 hypothesis에 대해 부정 cue 규칙을 적용해 negation candidate를 식별한다.

예시 규칙:

- exact token match: `안`, `못`, `없`, `아니`
- ending pattern: `-지 않`, `-지 못`
- double negation candidate: 두 개 이상의 부정 cue 공존

### 6.2 Step 2: Negation Taxonomy 부여

자동 추출된 샘플을 다음과 같은 유형으로 나눈다.

- 단순 부정
- 능력/불가능 부정
- 존재 부정
- copular negation
- 이중부정
- negation scope가 긴 문장

이 단계는 규칙 기반 tagging + 소규모 수작업 검수로 진행한다.

### 6.3 Step 3: Baseline 모델 학습 및 평가

후보 모델:

- KLUE-RoBERTa
- KoELECTRA
- KR-BERT 또는 유사 한국어 encoder

우선 standard fine-tuning을 수행하고 다음을 비교한다.

- 전체 dev/test 성능
- negation subset 성능
- cue별 성능
- label별 confusion matrix

### 6.4 Step 4: Optional Mitigation

시간이 허락하면 다음 중 하나만 소규모로 수행한다.

- negation subset oversampling
- negation-focused data augmentation
- loss reweighting

핵심은 “복잡한 새 모델”이 아니라, 현상 중심 분석 이후 간단한 완화 전략이 실제로 도움이 되는지 확인하는 것이다.

---

## 7. 평가 방법

### 7.1 전체 성능

- Accuracy
- Macro-F1

### 7.2 Negation 중심 평가

- Negation subset Accuracy
- Negation subset Macro-F1
- Contradiction F1 on negation subset
- Cue-type별 성능

### 7.3 비교 지표

- `Negation gap = overall accuracy - negation subset accuracy`
- `Contradiction drop = overall contradiction F1 - negation contradiction F1`

### 7.4 정성 분석

다음 유형의 오류를 수동 분석한다.

- 부정 cue를 무시한 경우
- 부정 scope를 잘못 해석한 경우
- 이중부정을 단순 부정으로 처리한 경우
- 세계지식/상식과 부정을 함께 처리하지 못한 경우

---

## 8. 예상 기여

- 공개 한국어 NLI 데이터에서 부정 현상 취약성을 체계적으로 분해해 제시
- 부정 cue 유형별 성능 저하와 오류 패턴 분석
- 한국어 NLI 모델 평가에서 overall score만으로는 가려지는 현상을 드러냄
- 간단한 완화 전략이 어느 정도까지 실제 도움이 되는지 실증적으로 점검

---

## 9. 문제될 수 있는 점과 대응 전략

### 9.1 데이터셋 자체의 한계

KorNLI와 KLUE-NLI는 negation을 위해 설계된 데이터셋이 아니다. 따라서 부정 샘플 수가 충분하지 않거나, 부정이 라벨 결정에 결정적이지 않은 경우가 있을 수 있다.

대응:

- negation candidate 수와 실제 유효 샘플 비율을 먼저 보고
- 필요하면 두 데이터셋을 함께 사용한다.
- 전체 주장 범위를 “한국어 NLI에서의 negation phenomenon 분석”으로 제한한다.

### 9.2 규칙 기반 추출의 잡음

표면 cue만으로는 실제 부정 scope를 정확히 반영하지 못할 수 있다.

대응:

- 자동 추출 후 소규모 샘플 검수
- cue type별 precision을 점검
- 과한 언어학적 주장은 피하고 empirical analysis로 한정

### 9.3 단순 분석으로 보일 위험

단순히 subset 성능만 보고 끝내면 기여가 약해 보일 수 있다.

대응:

- cue taxonomy
- label confusion
- negation gap
- optional mitigation

을 함께 포함해 분석 폭을 넓힌다.

### 9.4 한국어 언어 현상의 복잡성

한국어 부정은 어미, 보조용언, 서술어 구조와 얽혀 있어 단순 token match가 충분하지 않을 수 있다.

대응:

- 초기에는 high-precision cue 위주로 시작
- 범위를 넓히기보다 reliable subset을 우선 분석

---

## 10. 8주 일정

### 1주차
- 관련 논문 정독
- KorNLI, KLUE-NLI 구조 파악
- negation cue 목록 초안 작성

### 2주차
- negation candidate 자동 추출 스크립트 작성
- 소규모 샘플 검수
- taxonomy 확정

### 3주차
- baseline 모델 1개 fine-tuning
- 전체 성능 및 negation subset 성능 1차 측정

### 4주차
- baseline 모델 2~3개 비교
- cue type별 성능 분석

### 5주차
- confusion matrix 및 오류 분석
- negation gap 정리

### 6주차
- optional mitigation 1개 구현
- 전후 비교 실험

### 7주차
- 결과 시각화
- 표/그림/에러 케이스 정리

### 8주차
- 최종 보고서 작성
- 발표자료 정리
- 재현성 체크

---

## 11. 역할 분담 예시

### 역할 1: 데이터 및 현상 분석 담당

- KorNLI, KLUE-NLI 로딩
- negation cue 규칙 설계
- subset 추출 및 샘플 검수
- taxonomy 정리

### 역할 2: 모델 학습 및 실험 담당

- baseline 모델 fine-tuning
- 하이퍼파라미터 정리
- 전체/subset 성능 측정
- optional mitigation 구현

### 역할 3: 분석 및 보고서 담당

- confusion matrix
- cue별 성능 표 작성
- 정성 오류 분석
- related work, 보고서, 발표자료 정리

2인 팀이라면 역할 1과 3을 한 명이 맡고, 역할 2를 다른 한 명이 맡는 구조도 가능하다.

---

## 12. 한 줄 요약

본 프로젝트는 한국어 NLI에서 부정문과 이중부정이 평균 성능 뒤에 가려진 취약점인지 확인하고, 그 실패 유형과 완화 가능성을 현상 중심으로 분석하는 연구이다.
