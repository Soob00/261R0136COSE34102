# Counterfactual Benchmark 제작 가이드

## 목적

이 문서는 본 프로젝트에서 사용하는 한국어 혐오/공격 표현 탐지용 counterfactual 평가셋의 제작 및 정제 규칙을 정리한 공통 가이드다.

이 문서는 다음 작업에 공통으로 사용한다.

- [counterfactual_eval_v3.json](/C:/nlp_project/data/counterfactual_eval_v3.json) 확장
- [stereotype_flip_v2.json](/C:/nlp_project/data/stereotype_flip_v2.json) 같은 기존 셋 수정
- LLM을 이용한 새 후보 문장 생성
- 사람 검수 및 최종 벤치마크 확정

## 현재 사용 중인 데이터

- [kold_v1.json](/C:/nlp_project/data/kold_v1.json)
  - 메인 학습/검증 데이터셋
- [hard_negative_testset.json](/C:/nlp_project/data/hard_negative_testset.json)
  - 여러 hard-case를 포함한 더 큰 후보 풀
- [counterfactual_eval_v3.json](/C:/nlp_project/data/counterfactual_eval_v3.json)
  - 최근 평가에 사용한 정제된 벤치마크

## v3가 실제로 만들어진 방식

`v3`는 이 워크스페이스에서 LLM이 처음부터 끝까지 자동 생성한 셋이 아니다.

실제로는 아래 절차를 거쳤다.

1. [hard_negative_testset.json](/C:/nlp_project/data/hard_negative_testset.json)의 더 큰 후보 풀에서 시작
2. semantic stability와 label preservation이 더 낫다고 판단되는 pair를 수동 선별
3. 아래 3개 카테고리로 재구성
   - `adversarial_clean`
   - `stereotype_flip_strict`
   - `subtle_stereotype`
4. 최종 선별본을 [counterfactual_eval_v3.json](/C:/nlp_project/data/counterfactual_eval_v3.json)으로 저장

즉, 현재 벤치마크 제작 방식은 다음과 같다.

- 후보 생성 또는 후보 풀 구축
- 정제
- 카테고리 재분류
- 최종 검수

LLM 자동 생성 결과를 그대로 채택하는 방식이 아니다.

## 핵심 설계 원칙

향후 어떤 버전의 benchmark를 만들더라도 아래 원칙을 공통으로 적용한다.

1. 최소 수정 원칙
- 가능하면 identity term 또는 target phrase 외에는 거의 바꾸지 않는다.
- 문법 보정을 제외하고, 추가 논리나 결과, 정당화 표현을 새로 넣지 않는다.

2. label preservation 우선
- 치환 후에도 gold label이 유지되어야 한다.
- 치환 때문에 gold label이 바뀌면 counterfactual consistency pair로는 부적절하다.

3. 의미 안정성 유지
- original과 counterfactual의 화행적 힘(pragmatic force)이 유사해야 한다.
- 치환으로 인해 새로운 현실적 우려, 자격 문제, 관리 부담 등이 생기면 pair를 폐기한다.

4. 카테고리 분리
- 모든 hard case를 하나의 셋으로 섞지 않는다.
- 최소한 아래는 분리한다.
  - 비공격적 clean identity context
  - 명시적 stereotype/exclusion 공격
  - 완곡한 concern framing / subtle stereotype

5. consistency만으로 판단하지 않기
- consistency와 accuracy를 함께 본다.
- 모델은 일관되게 틀릴 수도 있다.

6. 최종 human review 필수
- 가능하면 최소 2인이 독립적으로 검수한다.

## 포함 기준

아래 조건을 모두 만족하면 benchmark 후보로 포함할 수 있다.

1. original 문장이 자연스러운 한국어다.
2. counterfactual 문장도 자연스러운 한국어다.
3. 핵심 framing이 original과 counterfactual 사이에서 유지된다.
4. 두 문장 모두에 대해 `expected_label`이 타당하다.
5. pair가 특정 평가 카테고리에 분명히 속한다.
6. 단순 단어 바꾸기가 아니라 의미 있는 failure mode를 테스트한다.

## 제외 기준

아래 중 하나라도 해당하면 pair를 버리거나 수정한다.

1. 치환으로 사회적 의미가 지나치게 바뀐다.
2. 치환 후 일반적이고 합리적인 situational concern이 새로 생긴다.
3. original과 counterfactual이 서로 다른 gold label을 요구한다.
4. identity term 외 단어가 너무 많이 바뀐다.
5. 문장이 부자연스럽거나 템플릿 티가 너무 난다.
6. 이미 있는 pair와 현상이 중복되는데 추가 가치가 없다.

## 카테고리 정의

### `adversarial_clean`

예상 라벨: `0`

목적:
- identity term이 들어갔다는 이유만으로 false positive가 나는지 확인

특징:
- 중립, 지지, 권리 보장, 차별 반대, 보도 맥락
- 타깃 집단에 대한 공격이 없어야 함

좋은 예:
- 권리 보장, 보호, 지원, 차별 실태 보도

나쁜 예:
- 우려, 부담, 위험, 불편함, 배제 논리가 암시되는 문장

### `stereotype_flip_strict`

예상 라벨: `1`

목적:
- 명시적인 stereotype/exclusion 문맥에서 집단 치환 시 예측이 흔들리는지 확인

특징:
- 광범위한 일반화
- 채용/참여/역할 배제
- 능력 부정
- 위협 집단화
- 집단 단위의 부적합/열등/위험 프레이밍

좋은 예:
- “X는 원래 위험하다”
- “X는 그 역할에 앉히면 안 된다”
- “X는 정상적인 자격이 없다”

나쁜 예:
- 치환 후 단지 현실적 상황 우려가 되는 경우

### `subtle_stereotype`

현재 v3 기준 예상 라벨: `1`

목적:
- 노골적 비하 없이 concern framing, burden framing, social reaction framing 형태로 나타나는 간접적 편향을 측정

특징:
- 명시적 비하어 없음
- “걱정된다”, “복잡해진다”, “적응이 필요하다”, “사람들 반응이 신경 쓰인다” 같은 완곡한 표현
- microaggression이나 polite exclusion과 겹칠 수 있음

주의:
- 이 카테고리가 가장 label validity가 흔들리기 쉽다.
- KOLD의 `OFF` 정의에 맞는지 반드시 사람 검수를 거쳐야 한다.

## JSON 포맷

각 pair는 아래 구조를 따른다.

```json
{
  "id": 1,
  "category": "adversarial_clean",
  "original": "...",
  "counterfactual": "...",
  "expected_label": 0,
  "identity_orig": "조선족",
  "identity_cf": "한국인",
  "trap": "간단한 설명",
  "source_id": 1
}
```

필드 의미:

- `id`: 현재 benchmark 내부 id
- `category`: 평가 카테고리
- `expected_label`: 두 문장에 공통으로 기대하는 label
- `trap`: 의도한 shortcut/failure mode 설명
- `source_id`: 더 큰 후보 풀에서 온 경우 원본 id

## 사람 검수 루브릭

각 pair에 대해 아래를 확인한다.

1. 유창성
- 두 문장이 모두 자연스러운가?

2. 최소 수정
- 수정이 identity term 중심으로 제한되었는가?

3. 의미 보존
- 치환 후에도 핵심 framing이 유지되는가?

4. 라벨 타당성
- `expected_label`이 두 문장 모두에 적용 가능한가?

5. 카테고리 적합성
- 현재 카테고리 정의에 맞는가?

권장 판정:

- `accept`
- `revise`
- `reject`

## 향후 버전 제작 권장 절차

1. 더 큰 후보 풀을 생성하거나 수집한다.
2. obvious failure를 자동 필터링한다.
3. label preservation과 category fit을 수동 점검한다.
4. source id와 수정 이력을 남긴다.
5. baseline과 비교 모델로 평가한다.
6. 오해를 유발하는 pair나 category는 제거 또는 재정의한다.

## 논문 보고 시 명시할 항목

최종 보고서에는 아래를 명확히 적는 것을 권장한다.

- 후보 pair를 어떻게 만들었는지
- 몇 개를 제거했는지
- 카테고리별 개수
- 사람 검수를 했는지
- consistency와 accuracy를 모두 측정했는지

이번 프로젝트에서 특히 중요한 점은, benchmark 설계에 따라 결론이 크게 달라진다는 사실 자체가 핵심 결과이기 때문이다.
