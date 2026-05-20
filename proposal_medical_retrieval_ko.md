# 최종 프로젝트 제안서
## Team 12
### 노이즈가 있는 의료 질의에 대한 Retrieval-Oriented Canonicalization 연구

---

## 1. 문제 정의

온라인 건강 정보나 의료 관련 주장은 종종 비격식적이고, 축약이 많고, 표현이 불완전한 형태로 작성된다. 이런 입력은 검증용 evidence가 잘 정리되어 있어도 retrieval 단계에서 쉽게 실패할 수 있다. 즉, evidence corpus의 품질 문제라기보다 입력 질의가 evidence의 표현 방식과 잘 맞지 않아 검색이 깨지는 경우가 많다.

본 프로젝트는 의료 misinformation 전체를 open-domain에서 해결하려는 것이 아니다. 대신 다음과 같은 좁고 검증 가능한 병목에 집중한다.

`noisy medical input -> retrieval-friendly canonical query -> closed evidence retrieval -> final verdict`

핵심 연구 질문은 다음과 같다.

노이즈가 있는 의료 입력 때문에 발생하는 retrieval 성능 저하를, 보수적이고 entity-aware한 canonicalization이 얼마나 회복할 수 있는가?

---

## 2. 과제 설정

본 연구는 HealthFC English를 기반으로 하는 closed-evidence 의료 claim verification 환경을 다룬다.

- 입력: 사용자 스타일의 noisy medical text
- 중간 산출물: retrieval에 적합한 canonical verification query
- retrieval 대상: HealthFC evidence sentence로 구성한 고정 evidence pool
- 최종 라벨: Supported / Refuted / NEI

여기서 중요한 점은 이 과제를 open-ended claim rewriting으로 보지 않는다는 것이다. 우리는 문장을 멋지게 다시 쓰는 것이 아니라, 의미를 최대한 보존하면서 evidence retrieval에 유리한 형태로 질의를 canonicalize하는 것을 목표로 한다.

예시는 다음과 같다.

```text
Noisy input:
"vit d helps depression right?? #supplements"

Cleanup:
"vit d helps depression right?"

Canonical query:
"Does vitamin D help with depression?"

Retrieved evidence:
closed HealthFC evidence corpus에서 top-k sentence retrieval

Final verdict:
Supported / Refuted / NEI
```

---

## 3. 데이터 구성

주 데이터셋은 HealthFC의 영어 subset을 사용한다. 우선적으로 활용할 필드는 다음과 같다.

- `en_claim`
- `en_top_sentences`
- `label`
- `en_explanation`

Evidence corpus는 모든 `en_top_sentences`를 모아 중복 제거한 뒤 구성한다.

질의(query) 측 데이터는 세 종류로 구분한다.

- `Clean query`: 원래의 `en_claim`
- `Synthetic noisy query`: clean claim에 자동으로 noise를 주입한 버전
- `Human-authored noisy query`: held-out subset을 사람이 직접 SNS/커뮤니티 스타일로 다시 쓴 버전

Synthetic noisy query는 주로 개발과 보조 실험에 사용하고, 최종 평가는 human-authored noisy query에서 수행한다.

학기 프로젝트 범위를 고려해, human-authored noisy set은 소규모로 구성한다. 현실적인 목표는 약 150-300개 claim 수준이다.

---

## 4. 제안 파이프라인

### 4.1 Surface Cleanup

이 단계는 의미를 새로 만들지 않고, 표면적 노이즈만 제거하는 가벼운 전처리 단계이다.

예상 작업은 다음과 같다.

- URL, hashtag, emoji 제거
- 반복 punctuation 정리
- whitespace와 casing 정리
- obvious shorthand 정리

예시:

```text
"vit d helps depression right?? #mood #supplements"
-> "vit d helps depression right?"
```

이 단계는 단순 전처리 같지만, `cleanup-only` baseline이 되므로 매우 중요하다.

### 4.2 Entity-Aware Canonicalization

이 단계가 프로젝트의 핵심이다.

목표는 noisy medical text를 retrieval-friendly한 canonical query로 변환하는 것이다. 중요한 점은 자유로운 생성이 아니라, 의미를 유지한 채 보수적으로 canonicalization하는 것이다.

이 모듈은 네 개의 하위 단계로 구성한다.

1. `Medical mention detection`
   - 질병, 약물, 영양제, 증상, 처치 관련 mention 탐지
   - 구현 후보: scispaCy 등 lightweight biomedical NER 도구

2. `Abbreviation and synonym normalization`
   - lay term, 줄임말, 흔한 약칭을 canonical form으로 정규화
   - 예:
     - `vit d -> vitamin D`
     - `high bp -> high blood pressure`
   - 애매한 경우는 공격적으로 바꾸지 않고 원문에 가깝게 유지한다.

3. `Claim pattern canonicalization`
   - 자주 등장하는 관계 패턴을 소수의 템플릿으로 정리
   - 예:
     - `X helps Y`
     - `X causes Y`
     - `X has side effects`
     - `X is better than Y`

4. `Canonical query rendering`
   - 정규화된 entity와 relation을 이용해 표준 query 형식으로 렌더링
   - 예:
     - `vit d helps depression`
     - `Does vitamin D help with depression?`

전체 모듈은 명시적으로 conservative하게 설계한다. entity나 relation이 불확실하면 minimal edit에 가깝게 유지한다.

### 4.3 Retrieval

Retrieval은 오직 closed HealthFC evidence corpus에서만 수행한다.

필수 retriever:

- BM25

선택 가능한 stronger baseline:

- MedCPT 또는 biomedical dense retriever

핵심 질문은 dense retrieval이 BM25보다 일반적으로 강한가가 아니다. canonicalization이 clean-noisy gap을 얼마나 줄이는가가 핵심이다.

### 4.4 Verification

Verification 모듈은 의도적으로 단순하게 유지한다. 이 프로젝트에서 verifier는 주인공이 아니라 retrieval 품질의 downstream consumer이기 때문이다.

- 입력: `[query] + [top-k evidence sentences]`
- 출력: `Supported / Refuted / NEI`
- 모델 후보:
  - BERT-style classifier
  - BioBERT-style classifier

---

## 5. V1과 V2의 의미

여기서 `v1`과 `v2`는 서로 다른 연구가 아니라 구현 범위를 구분하기 위한 단계다.

### V1: 필수 완성 버전

이 버전은 이번 학기 프로젝트로서 반드시 완성해야 하는 최소 완성형이다.

- HealthFC English only
- `en_top_sentences` 기반 closed evidence corpus 구축
- synthetic noisy query 생성
- 소규모 human-authored noisy eval set 구축
- cleanup-only baseline
- conservative entity-aware canonicalization
  - rule-based cleanup
  - abbreviation expansion
  - 제한적인 synonym normalization
  - template-based query rendering
- BM25 retrieval
- simple verifier
- retrieval 및 end-to-end verification 평가

즉, V1만 안정적으로 끝내도 프로젝트는 충분히 성립한다.

### V2: 선택적 확장 버전

이 버전은 V1이 충분히 안정화된 뒤 시간이 남을 경우 추가하는 확장형이다.

- MedCPT 같은 biomedical dense retriever 추가
- SapBERT 기반 candidate reranking
- single-query 대신 multi-query expansion
- generic rewrite baseline 추가
- noise type 및 entity type별 상세 error analysis

V2가 완성되지 않아도, V1이 잘 완성되어 있으면 프로젝트 전체는 성공할 수 있다.

---

## 6. Baseline 및 Ablation

최소한 다음 네 조건은 반드시 비교해야 한다.

- `Oracle clean query`
- `Raw noisy query`
- `Cleanup-only`
- `Proposed canonicalization`

시간이 허락하면 다음도 추가할 수 있다.

- `Generic rewrite baseline`
- `Dense retrieval with / without canonicalization`

Ablation은 다음처럼 설계할 수 있다.

- abbreviation expansion 제거
- medical synonym normalization 제거
- template-based canonical rendering 제거
- BM25 only vs BM25 + dense retrieval

---

## 7. 평가 계획

평가는 세 수준에서 수행한다.

### 7.1 Retrieval 평가

- Recall@k
- MRR

핵심 질문:

Clean query 대비 noisy query에서 떨어진 retrieval 성능을 얼마나 회복할 수 있는가?

### 7.2 Verification 평가

- Supported / Refuted / NEI에 대한 Macro-F1

### 7.3 Joint 평가

- 최종 label이 맞고, 동시에 top-k 안에 최소 하나의 gold evidence가 포함되는지 측정

### 7.4 Error Analysis

실패 사례는 다음과 같은 noise type별로 나누어 분석한다.

- abbreviation
- misspelling
- lay term vs clinical term mismatch
- underspecified entity
- conversational phrasing

이 프로젝트의 가치는 단순히 숫자 하나를 올리는 데 있지 않고, retrieval bottleneck을 어떤 유형에서 얼마나 회복하는지 보여주는 데 있다.

---

## 8. 리스크와 범위 통제

- claim rewriting이 open-ended generation으로 커질 수 있다.
  - 이를 막기 위해 free-form generation이 아니라 conservative canonicalization으로 한정한다.

- human-authored noisy data 구축 비용이 커질 수 있다.
  - held-out subset 규모를 작고 현실적으로 제한한다.

- biomedical normalization이 잘못된 의학적 가정을 넣을 수 있다.
  - confidence가 높은 abbreviation/synonym만 정규화하고, 불확실한 경우 최소 수정으로 유지한다.

- verifier가 retrieval 효과를 가릴 수 있다.
  - verifier는 단순하게 두고, retrieval을 주 평가 대상으로 둔다.

---

## 9. 기대 기여

- noisy medical query robustness를 closed-corpus fact verification 맥락에서 분석하는 현실적인 학기 프로젝트
- retrieval-oriented medical canonicalization 파이프라인 제안
- clean, noisy, cleanup-only, canonicalized query를 분리해 비교하는 평가 프로토콜 제시

본 프로젝트의 핵심 기여는 거대한 생성 모델이 아니다. query-side canonicalization이 medical fact verification에서 retrieval failure를 실제로 회복하는지에 대한 실증 분석이다.

---

## 10. 8주 마일스톤

- 1-2주
  - HealthFC English 분석
  - closed evidence corpus 구축
  - BM25로 clean vs noisy retrieval gap 확인

- 3-4주
  - cleanup-only baseline 구현
  - V1 canonicalization 모듈 구현

- 5-6주
  - human-authored noisy eval subset 구축
  - retrieval 실험 및 ablation 수행

- 7-8주
  - simple verifier 추가
  - error analysis, 보고서, 발표자료 정리

---

## 11. 참고 논문

1. Vladika et al. (2024), HealthFC: Verifying Health Claims with Evidence-Based Medical Fact-Checking  
2. Möller et al. (2025), Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning  
3. Guo et al. (2025), A Systematic Survey of Claim Verification: Corpora, Systems, and Case Studies  
4. Sundriyal et al. (2023), From Chaos to Clarity: Claim Normalization to Empower Fact-Checking  
5. CheckThat! Lab CLEF 2025 Task 2 on Claim Normalization  
6. Jin et al. (2023), MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval  
7. Liu et al. (2021), SapBERT: Self-Alignment Pretraining for Biomedical Entity Representations  
8. Neumann et al. (2019), ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing  
9. Chakraborty et al. (2023), Evaluating the Robustness of Biomedical Concept Normalization  
10. Wang et al. (2025), GuRE: Generative Query REwriter for Legal Passage Retrieval
