# Final Project Proposal (Revised)
## Team 12
### 한국어 혐오 표현 탐지의 고정관념 숏컷 의존성 완화: Counterfactual Consistency 학습과 Stereotype Flip 평가를 중심으로

---

## 1. 연구 배경 및 문제 정의

한국어 혐오 표현 탐지 모델은 KOLD와 같은 대규모 데이터셋에서 높은 분류 성능을 달성하지만, 이러한 성능이 실제 문맥 이해에 기반한 것인지, 혹은 특정 집단 키워드와 혐오 레이블 간의 통계적 공동 출현을 학습한 결과인지는 별개의 문제다. 예를 들어, 특정 문장에서 `여성`, `조선족`, `장애인`과 같은 identity term이 포함되었을 때 모델이 문맥 전체가 아니라 해당 키워드 자체에 과도하게 반응한다면, 이는 의미 이해가 아닌 `shortcut reliance`의 징후로 해석할 수 있다.

본 연구는 특히 명시적 비하어 없이 구조적 차별이나 고정관념만 드러나는 문장에서 이러한 shortcut reliance가 더 두드러질 수 있다는 점에 주목한다. 실제로 동일한 문장 구조에서 identity term만 교체했을 때 예측이 불안정하게 뒤집히는 현상은, 모델이 혐오의 의미적 맥락보다는 표면 키워드에 의존하고 있음을 보여주는 직접적 증거가 될 수 있다.

기존 연구들은 identity term이 포함된 비독성 문장에서의 false positive, 집단별 오류율 불균형, lexical shortcut 문제를 지적해 왔으나, 한국어 혐오 탐지 환경에서 이러한 shortcut을 `counterfactual consistency` 관점에서 체계적으로 측정하고 완화하는 연구는 아직 많지 않다. 따라서 본 연구는 한국어 혐오 탐지 모델의 고정관념 기반 shortcut reliance를 정량적으로 드러내고, 이를 완화하는 실용적 학습 방법과 평가 프레임을 제안하고자 한다.

본 연구의 주요 기여는 다음과 같다.

1. 한국어 혐오 탐지에서 `identity shortcut failure`를 counterfactual consistency 관점에서 정량화한다.
2. `Stereotype Flip v3` 평가셋을 구축하여 shortcut failure를 정밀하게 측정하는 한국어 benchmark를 제시한다.
3. 단순한 `consistency regularization`만으로 해당 문제를 완화할 수 있음을 실험적으로 보인다.

---

## 2. 연구 질문

본 연구는 다음 질문에 답하는 것을 목표로 한다.

1. 한국어 혐오 탐지 모델은 identity term 자체를 혐오 판단의 shortcut으로 학습하는가?
2. 동일한 문맥에서 identity term을 마스킹하거나 counterfactual로 치환했을 때, 모델의 예측 일관성은 얼마나 유지되는가?
3. Counterfactual consistency를 직접적으로 학습 목표에 포함시키면, 기존 분류 성능을 크게 해치지 않으면서 shortcut reliance를 완화할 수 있는가?

쉽게 말하면, 본 연구는 "모델이 문장을 이해해서 혐오를 탐지하는가, 아니면 특정 집단 키워드에 반응해서 혐오를 예측하는가?"를 묻는 프로젝트이다.

---

## 3. 제안 방법

본 연구는 `counterfactual consistency`를 직접 학습 목표에 반영하는 간결한 접근을 채택한다. 핵심 아이디어는 모델이 identity term의 표면 형태가 바뀌거나 마스킹되더라도, 문장의 혐오 맥락이 유지되는 경우 비슷한 예측을 하도록 학습시키는 것이다.

### 3.0 Anchor Paper와 아이디어의 출발점

본 연구의 가장 직접적인 anchor paper는 **Lu et al. (2024), _Take Its Essence, Discard Its Dross! Debiasing for Toxic Language Detection via Counterfactual Causal Effect_**이다. 이 논문은 toxic language detection에서 lexical bias의 "유용한 영향"과 "오해를 부르는 영향"을 구분하려 하며, counterfactual 관점에서 debiasing을 수행한다. 즉, 본 연구의 문제의식인 "모델이 문맥이 아니라 특정 편향 토큰에 과도하게 반응한다"는 출발점은 이 논문과 직접 연결된다.

그러나 본 연구는 해당 논문의 causal framework 자체를 재현하려 하지 않는다. 대신 한국어 KOLD 환경에서 shortcut failure를 더 직접적으로 측정할 수 있는 counterfactual benchmark와 consistency regularization 중심의 실험 설정을 채택한다.

- 복잡한 causal decomposition 대신, `counterfactual consistency`를 직접 측정하고 학습 목표에 반영한다.
- 영어 toxic language 환경 대신, `KOLD`를 기반으로 한 한국어 혐오 탐지 환경을 다룬다.
- 기존 benchmark에만 의존하지 않고, `Stereotype Flip v3`라는 한국어 특화 counterfactual 평가셋을 구축한다.
- 정교한 인과 프레임워크의 성능 재현보다, 재현 가능한 간단한 regularization과 평가 설계를 우선한다.

따라서 본 연구의 아이디어는 한 논문에서 완전히 가져온 것이 아니라, 세 가지 흐름이 만나는 지점에서 나온다.

1. **영어권 toxic language debiasing 연구**  
   lexical shortcut과 spurious artifact 문제를 다룬 선행연구들에서 문제의식을 가져온다.

2. **counterfactual fairness 관점**  
   identity term을 바꿨을 때 예측이 얼마나 안정적인지를 보는 평가 관점을 계승한다.

3. **한국어 KOLD 환경에서의 실제 관찰**  
   현재 파일럿 실험에서 identity term 치환에 따라 예측이 흔들리는 현상을 직접 확인했고, 이 경험이 본 연구 질문을 구체화했다.

즉, 본 연구는 `Lu et al. (2024)의 counterfactual debiasing 문제의식`, `Ramponi and Tonelli (2022)의 spurious artifact 분석 프레임`, 그리고 `KOLD 기반 한국어 환경에서의 실험 관찰`이 결합되어 나온 아이디어다.

### 3.1 Dual-Input Counterfactual Pipeline

각 훈련 문장에 대해 두 가지 변형 입력을 구성한다.

- **Original Input ($x$)**: 원본 문장
- **Masked or Neutralized Input ($x_{mask}$)**: 문장 내 identity term을 `[MASK]` 또는 중립 토큰으로 치환한 문장

필요할 경우 일부 샘플에 대해서는 identity term을 다른 집단 키워드로 바꾼 `counterfactual input ($x_{cf}$)`도 보조적으로 활용할 수 있다. 본 연구의 핵심 방법은 `consistency regularization`이며, `masking/neutral replacement`는 이를 보조하는 baseline 또는 augmentation으로 위치시킨다.

### 3.2 Loss Function

모델은 KLUE-RoBERTa 기반 단일 인코더를 사용하며, 총 손실은 다음 두 요소로 구성된다.

1. **Classification Loss ($L_{hate}$)**  
   원본 문장에 대한 혐오 여부 정답지와의 교차 엔트로피 손실

2. **Consistency Loss ($L_{cons}$)**  
   원본 입력과 변형 입력의 예측 분포 차이를 줄이기 위한 손실  
   예: MSE, KL-divergence, 혹은 binary cross-entropy 기반 거리

전체 손실은 다음과 같이 정의한다.

$$
L_{total} = L_{hate} + \lambda L_{cons}
$$

필요할 경우 identity term masking 자체를 데이터 증강으로 사용하는 `masking regularization`을 추가할 수 있다. 그러나 제안 방법의 핵심은 구조적 복잡성을 늘리는 것이 아니라, 모델이 `identity term이 아니라 맥락을 보도록` 직접 압력을 주는 데 있다.

### 3.3 설계 의도

이 설계의 직관은 단순하다. 예를 들어 `"여성이 운전하면 위험하다"`라는 문장에서 `여성`을 마스킹하더라도, 문장의 차별적 맥락은 상당 부분 유지된다. 만약 모델이 원본과 마스킹 문장에 대해 완전히 다른 예측을 한다면, 이는 혐오 맥락보다 특정 키워드에 의존하고 있다는 뜻이다. 반대로 두 입력에 대해 일관된 판단을 하도록 학습시키면, 모델이 문맥적 신호를 더 활용하도록 유도할 수 있다.

---

## 4. 데이터 및 실험 설계

### 4.1 훈련 데이터

- **KOLD (Korean Offensive Language Dataset)**  
  훈련 및 검증에는 KOLD 공식 split을 사용한다. 이는 기존 한국어 혐오 탐지 연구와 비교 가능한 baseline을 확보하기 위함이다.

### 4.2 핵심 평가셋: Stereotype Flip v3

본 연구의 핵심 기여 중 하나는 `Stereotype Flip v3` 평가셋 구축이다. 기존 파일럿 실험에서 사용한 소규모 평가셋은 아이디어 검증에는 유용했지만, 통계적 신뢰도를 갖춘 분석을 수행하기에는 규모가 제한적이었다. 따라서 최종 실험에서는 **최소 150쌍(300문장) 규모**의 counterfactual 평가셋을 구축한다.

평가셋은 다음 두 범주를 중심으로 구성한다.

- **Adversarial Clean**  
  identity term이 포함되어 있지만 혐오 표현이 아닌 문장. 모델의 false positive 취약성을 측정한다.

- **Stereotype Flip Strict**  
  identity term이 바뀌어도 구조적 차별 또는 배제 맥락이 유지되는 문장. 모델의 true positive 유지력과 counterfactual consistency를 측정한다.

`Subtle Stereotype` 유형은 탐색적 분석 대상으로 유지하되, 본 프로젝트의 핵심 기여는 Adversarial Clean과 Stereotype Flip Strict에 집중한다. 이렇게 범위를 제한함으로써 통계적 신뢰도를 높이고, 실패 모드를 명확히 정의할 수 있다.

### 4.3 평가셋 품질 관리

Stereotype Flip 평가셋의 품질은 본 연구의 성패를 좌우하므로, 다음 기준을 적용한다.

1. identity term 외 어휘 변화 최소화
2. 원문과 치환문 사이의 구조적 의미 동등성 유지
3. 명시적 비하어 의존 최소화
4. 최소 2인 이상 독립 검수 후 불일치 사례 제외

이 과정을 통해 모델의 예측 변화가 실제로 identity term 의존성 때문인지 더 설득력 있게 해석할 수 있도록 한다.

---

## 5. Baselines 및 Ablation

### 5.1 Baselines

- **Single-head KLUE-RoBERTa baseline**
- **Masking augmentation only**: 보조 baseline
- **Consistency regularization only**: 핵심 방법
- **Full model: Masking + Consistency regularization**: 결합 효과 점검용 확장 설정

원래 검토했던 dual-head + orthogonality 모델은 비교용 참고 baseline으로만 제한적으로 언급할 수 있으나, 본 제안서의 중심 방법은 아니다.

### 5.2 Hyperparameter Sensitivity

- $\lambda \in \{0.05, 0.1, 0.2\}$
- masking strength 또는 replacement strategy 비교

학기 프로젝트 범위를 고려해 지나치게 많은 조합을 탐색하기보다, 핵심 설정을 중심으로 제한된 sensitivity 실험을 수행한다.

---

## 6. 평가 지표 및 통계적 신뢰성

### 6.1 주요 평가 지표

핵심 지표는 아래 두 가지다.

- **Strict Pair Accuracy**: stereotype_flip_strict에서 original과 counterfactual을 모두 올바르게 맞히는 비율
- **Strict TPR (counterfactual)**: stereotype_flip_strict에서 치환된 문장에 대한 positive recall

보조 지표는 다음과 같다.

- **Macro-F1**: KOLD 공식 테스트셋에서의 일반 분류 성능
- **False Positive Rate on Adversarial Clean**: identity term이 포함된 정상 문장에서의 오탐 비율
- **Counterfactual Consistency Score (CCS)**: original/counterfactual 또는 original/masked pair에서 예측이 일치하는 비율
- **Flip Rate**: identity term 변경 시 예측이 뒤집히는 비율
- **Group-wise FPR/FNR disparity**: 집단별 오류 편차 분석

### 6.2 핵심 가설

- **H1**: Baseline 모델은 Stereotype Flip Strict에서 낮은 consistency를 보여 shortcut reliance를 드러낸다.
- **H2**: Consistency regularization은 KOLD Macro-F1을 크게 해치지 않으면서 Strict consistency를 개선한다.
- **H3**: Masking을 결합한 full model은 consistency 개선을 추가로 강화한다.

### 6.3 통계적 엄밀성

소규모 평가셋 기반 주장에 대한 우려를 줄이기 위해, 모든 주요 실험은 최소 3개의 seed로 반복한다. 보고 시에는 평균과 표준편차를 함께 제시하며, 주요 비교에는 적절한 통계 검정을 적용한다. 기본적으로 paired setting에 적합한 검정을 우선 고려하고, 필요 시 bootstrap confidence interval을 함께 제시하여 결과의 안정성을 보완한다.

본 연구의 목표는 "완전한 편향 제거"를 주장하는 것이 아니라, `strict counterfactual robustness와 group-wise robustness의 유의미한 개선`을 실증적으로 보이는 데 있다.

---

## 7. 분석 계획

본 연구는 단순한 평균 점수 비교를 넘어, 모델이 어떤 유형의 문장에서 여전히 실패하는지를 분석한다.

- **Error Analysis**  
  Stereotype Flip Strict와 Adversarial Clean에서 남아 있는 실패 사례를 유형별로 분류한다.

- **Subtle Stereotype 탐색 분석**  
  본 프로젝트의 핵심 평가지표는 아니지만, subtle stereotype 사례에서의 실패를 질적으로 분석하여 후속 연구 방향을 제시한다.

- **Attention 또는 Saliency 보조 분석**  
  가능하다면 학습 전후 모델이 identity term과 주변 맥락 단어에 얼마나 주목하는지 보조적으로 관찰한다. 다만 이는 핵심 기여가 아니라 해석을 돕는 보조 증거로만 사용한다.

---

## 8. 리스크 관리

본 연구는 평가셋 품질이 가장 큰 리스크임을 전제로 설계한다.

- **평가셋 확장 리스크**  
  파일럿 규모의 34쌍은 초기 신호 확인에는 충분했으나, 최종 결론을 내리기에는 작다. 따라서 최소 150쌍 구축을 필수 마일스톤으로 둔다.

- **가설 미달 시 대응**  
  baseline consistency가 예상보다 높게 나올 경우, 구조적 의미 동등성이 더 높은 pair를 추가 구축하고, identity term 외 표면 변화가 적은 stricter set으로 재정의한다.

- **주장 범위 통제**  
  본 연구는 subtle stereotype 전체를 해결한다고 주장하지 않으며, 명확히 정의된 shortcut failure mode에 대한 개선을 목표로 한다.

---

## 9. 기대 기여

1. **한국어 혐오 탐지에서 identity shortcut reliance를 정량화**  
   단순 정확도가 아닌 counterfactual consistency 기반으로 shortcut 문제를 드러낸다.

2. **Stereotype Flip v3 평가셋 구축**  
   최소 150쌍 규모의 한국어 counterfactual benchmark를 구축하여 재현 가능한 평가 자산을 제공한다. 본 프로젝트에서 논문 가치의 상당 부분은 이 평가 자산의 설계와 정제에서 나온다.

3. **실용적 완화 기법 제안**  
   복잡한 구조 변경 없이 consistency regularization 중심의 간결한 방법으로 shortcut reliance를 완화하는 현실적 접근을 제시한다.

4. **한국어 혐오 탐지의 한계와 후속 과제 제시**  
   subtle stereotype과 집단별 편차 문제를 함께 분석함으로써 후속 연구의 방향을 제안한다.

---

## 10. 8주 마일스톤

- **1-2주**: KOLD 분석, baseline 재현, Stereotype Flip v3 초안 확장
- **3-4주**: masking 및 consistency regularization 구현, 초기 ablation 수행
- **5-6주**: 평가셋 150쌍 검수 완료, 3-seed 반복 실험 및 통계 검정
- **7주**: error analysis, group-wise fairness 분석, 시각화
- **8주**: 최종 보고서 및 발표자료 정리

---

## 11. Anchor References

1. Dixon et al. (2018), *Measuring and Mitigating Unintended Bias in Text Classification*  
   identity term이 포함된 비독성 문장에서의 false positive 문제를 정량화한 대표 연구.

2. Sap et al. (2019), *The Risk of Racial Bias in Hate Speech Detection*  
   집단별 오류율 불균형과 데이터 artifact 문제를 지적한 연구.

3. Ravfogel et al. (2020), *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection*  
   보호 속성 제거 관점의 대표 연구로, 본 연구는 이보다 단순하고 실용적인 regularization 접근을 택한다.

4. Zhou et al. (2021), *Challenges in Automated Debiasing for Toxic Language Detection*  
   lexical bias와 debiasing의 한계를 다룬 연구.

5. Ramponi and Tonelli (2022), *Features or Spurious Artifacts? Data-centric Baselines for Fair and Robust Hate Speech Detection*  
   spurious artifact와 shortcut reliance 문제를 데이터 중심으로 분석한 연구.

6. Lu et al. (2024), *Take Its Essence, Discard Its Dross! Debiasing for Toxic Language Detection via Counterfactual Causal Effect*  
   counterfactual 관점의 toxic language debiasing 연구로, 본 연구의 직접적인 anchor paper이다. 다만 본 연구는 해당 논문의 복잡한 causal framework를 그대로 따르지 않고, 한국어 KOLD 환경에 맞춘 간단한 consistency regularization과 평가셋 구축으로 방향을 단순화한다.

7. KOLD 관련 원 논문 및 한국어 혐오 탐지 벤치마크 연구  
   한국어 환경에서의 baseline 비교를 위한 핵심 참고문헌.

---

## 12. 결론

본 연구는 한국어 혐오 표현 탐지 모델이 identity term 자체를 shortcut으로 사용하는 문제를, counterfactual consistency 관점에서 분석하고 완화하는 것을 목표로 한다. 기존 proposal이 representation disentanglement 구조에 무게를 두었다면, 본 개정안은 `Stereotype Flip 평가셋`, `consistency regularization`, `핵심 지표 중심 평가`, `통계적 신뢰성 확보`에 초점을 맞춘다. 이를 통해 학기 프로젝트 범위 안에서 구현 가능하면서도, 한국어 혐오 탐지의 shortcut bias를 설득력 있게 드러내고 완화하는 연구를 제안한다.
