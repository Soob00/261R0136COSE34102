# Final Project Proposal
## Team 12
### 직교 제약 기반 Dual-Head 학습을 통한 한국어 혐오 표현의 고정관념 숏컷 의존성 완화

---

## 1. 연구 배경 및 문제 정의

기존 한국어 혐오 탐지 모델은 KOLD와 같은 대규모 혐오 표현 데이터셋에서 높은 분류 성능(Macro-F1 0.85 이상)을 달성한다. 그러나 이 성능이 문장의 의미를 진정으로 이해한 결과인지, 아니면 데이터셋 내에서 특정 집단 키워드(identity term)와 혐오 레이블 간의 통계적 공동 출현 패턴을 외운 결과인지는 별개의 문제다.

본 연구의 핵심 가설은 다음과 같다: **KLUE-RoBERTa 기반 단일 헤드 분류기는 KOLD 학습 과정에서 '여성', '조선족', '장애인' 등 특정 집단 키워드 자체를 혐오 판단의 주요 단서(shortcut)로 학습하며, 이로 인해 키워드가 중립적·긍정적 맥락에 배치되거나, 동일한 구조적 차별 문장에서 집단 키워드만 교체될 경우 예측이 불안정하게 뒤집히는 구조적 취약점을 가진다.**

이 취약점은 단순 정확도 저하가 아니라 공정성(fairness) 측면의 실패로, 기존 연구에서 반복적으로 지적되어 온 문제다. 대표적으로 Dixon et al. (2018)은 identity term이 포함된 비독성 문장에서 false positive가 높아지는 unintended bias를, Sap et al. (2019)과 Xia et al. (2020)은 특정 집단 언급에서 오류율이 불균형하게 나타나는 group-wise error disparity를 보고했다. Zhou et al. (2021)과 Ramponi and Tonelli (2022)는 표면 단서와 데이터 artifact에 과의존하는 spurious lexical shortcut reliance를, Lu et al. (2024)는 lexical bias의 유용한 신호와 오해를 부르는 신호를 구분하지 못하는 문제를 다루었다.

본 연구는 이 중에서도 **고정관념 기반 숏컷 의존(Stereotypical Shortcut Reliance)** — 특정 집단 키워드가 구조적 차별 문맥에 등장할 때 모델이 문장 의미가 아닌 키워드 자체에 반응하는 현상 — 을 핵심 타겟으로 삼는다. 연구 범위를 이 하나의 failure mode에 집중함으로써 실증적으로 검증 가능한 명확한 실험 설계를 달성한다.

---

## 2. 제안 방법

### 2.1 Dual-Head Architecture
KLUE-RoBERTa를 공유 인코더로 사용하고, 그 위에 두 개의 소형 projection head를 둔다.

- **Classification Head**: 공유 인코더 출력 $h$를 $d$차원 은닉 표현 $z_{cls}$로 투영한 뒤, 혐오 라벨을 예측하는 주 분류 헤드
- **Semantic Head**: 동일한 $h$를 $z_{sem}$으로 투영하며, MLM 보조 목표를 통해 문맥 정보를 보존하도록 학습하는 보조 헤드

### 2.2 Orthogonality Constraint
두 헤드가 같은 방향의 표현(특히 identity-term shortcut 방향)을 재사용하지 않도록, 코사인 유사도 기반 직교 제약 손실을 적용한다. 이 제약은 최종 로짓이 아닌 $z_{cls}$와 $z_{sem}$ 사이에 적용한다.

$$
L_{ortho} = \left( \frac{z_{sem} \cdot z_{cls}}{\|z_{sem}\|\|z_{cls}\|} \right)^2
$$

전체 손실 함수:

$$
L_{total} = L_{hate\_cls} + \alpha L_{mlm} + \lambda L_{ortho}
$$

### 2.3 설계 의도

직교 제약의 역할은 $z_{cls}$가 단기적으로 유용한 shortcut(identity term 방향)을 재사용할 때, $z_{sem}$이 같은 방향을 학습하지 못하도록 압력을 가하는 것이다. 이는 classification head가 문장 의미 정보가 아닌 키워드 방향에만 의존하는 것을 간접적으로 억제한다.

$L_{mlm}$은 semantic head가 의미 있는 문맥 정보를 보존하도록 강제함으로써, "bias 방향을 제거하다가 유용한 문맥 정보까지 손상"되는 현상을 방지하는 보조 장치다 (Ravfogel et al., 2020의 문제의식과 연결).

완전한 disentanglement를 주장하는 것이 아니라, **identity-term shortcut 의존을 완화하여 고정관념 기반 문장에서 예측 일관성(counterfactual consistency)을 높이는 것**이 본 연구의 실용적 목표다.

---

## 3. 데이터 및 실험 설계

### 3.1 데이터셋

- **훈련/검증**: KOLD (Korean Offensive Language Dataset, 약 40k examples). 공식 split 유지.
- **핵심 평가셋 (신규 구축)**: Stereotype Flip Evaluation Set — 명시적 비하어 없이 구조적 차별/배제 문장으로만 구성된 60쌍. 각 쌍은 동일한 문장 구조에서 identity term만 치환하며, 두 문장의 label은 동일하게 1(공격적)로 유지됨.

**Stereotype Flip 평가셋 설계 원칙:**
- 명시적 비하어(쓸모없다, 죽어야 해 등) 완전 배제 — 비하어가 있으면 모델이 identity term 없이도 혐오를 탐지하여 shortcut 노출 불가
- 합리화를 가장한 배제(Polite Exclusion), 은근한 능력 폄하(Micro-invalidation), 잠재적 위험군 취급(Presumed Threat) 3가지 gray area 포함
- 타겟 집단: 여성, 조선족/외국인, 장애인, 노인, 무슬림, 탈북민, 성소수자 (KOLD 내 혐오 레이블 편중이 높은 집단 우선)

**설계 근거**: 모델이 orig와 cf에서 다른 예측을 낸다면, 그것은 문장 의미가 아닌 identity term 토큰에 반응하는 것이므로 shortcut learning의 직접적 증거가 된다.

### 3.2 Baseline 및 Ablation

- Single-head KLUE-RoBERTa baseline (주 baseline)
- Single-head + counterfactual data augmentation (CDA)
- Dual-head only (lambda=0)
- Dual-head + orthogonality only (alpha=0)
- Dual-head + MLM only (lambda=0)
- Full model (L_mlm + L_ortho)
- Hyperparameter sensitivity: lambda in {0, 0.01, 0.05}, alpha in {0.1, 0.3}
- Head dimension: d in {128, 256}

### 3.3 평가 지표 및 가설

**핵심 가설:**
- **H1 (shortcut 노출)**: KOLD로 훈련된 Baseline은 Stereotype Flip 평가셋에서 counterfactual consistency < 0.70을 기록한다 — identity-term shortcut 학습의 증거.
- **H2 (Dual-head 회복)**: Dual-head + orthogonality 모델은 KOLD Macro-F1을 Baseline 수준으로 유지하면서, Stereotype Flip consistency를 0.80 이상으로 회복한다.
- **H3 (MLM 기여)**: Semantic head의 MLM 보조 목표는 H2의 consistency 회복을 추가로 강화한다.

**평가 지표:**
- Performance: Macro-F1, Class-wise F1 (KOLD 공식 테스트셋)
- Robustness: Stereotype Flip Counterfactual Consistency Score
- Fairness (보조): 타겟 집단별 FPR/FNR 편차

**평가 우선순위**: 핵심 기여는 Stereotype Flip consistency 검증(H1+H2)이며, H3와 fairness 지표는 보조 분석으로 보고한다.

### 3.4 한계 및 오류 분석 (Limitations)

Stereotype Flip에 집중함으로써 의도적으로 다루지 않는 failure mode들을 명시한다:

- **Implicit Hate (우회적 혐오)**: 명시적 identity term 없이 은유/우회 표현으로 혐오를 전달하는 문장. 예비 실험에서 Baseline과 Dual-head 모두 accuracy 0.10~0.20 수준으로 탐지 실패. 이는 ortho constraint의 한계가 아니라, semantic head가 암묵적 의미를 포착할 충분한 용량을 갖추지 못한 것으로 해석된다. 향후 더 강력한 MLM 학습이나 외부 semantic supervision이 필요한 영역으로 제시.
- **Adversarial Clean (중립 맥락 FP)**: 중립/긍정 맥락에서 identity term이 등장할 때 false positive 발생 여부. 예비 실험에서 Baseline consistency 1.00으로 현재 설계된 함정 수준에서는 취약점 미탐지. 더 정교한 함정 설계가 필요한 미래 연구 방향으로 제시.

이러한 한계의 투명한 보고는 연구의 범위를 명확히 하고, 재현 가능한 결과의 신뢰성을 높이는 데 기여한다.

---

## 4. 리스크 관리

- **Stereotype Flip 품질 관리**: 쌍 구성 시 (1) 명시적 비하어 없음, (2) 두 문장의 구조적 의미 동등성, (3) identity term 외 어휘 변화 최소화 3가지 기준으로 필터링. 최소 2인 독립 검수 후 불일치 쌍 제외.
- **H1 미달 시 대응**: Baseline consistency가 0.70 이상으로 나올 경우 — 함정 쌍의 문장 구조를 더 단순화(비하어 완전 제거 + identity term 외 변화 최소화)하거나, 더 많은 학습(epochs 증가)으로 shortcut 심화 후 재평가.
- **신뢰성**: 최소 3 seed 평균과 표준편차 보고.
- **주장 범위 통제**: 완전한 disentanglement 증명이 아닌, Stereotype Flip consistency 개선의 실증적 검증을 기여로 규정.

---

## 5. 기대 기여

1. **Stereotype Flip 평가셋 구축**: 명시적 비하어 없이 구조적 차별 문장으로만 구성된 한국어 혐오 탐지 전용 counterfactual 평가셋 최초 제시. 이 평가셋 자체가 재현 가능한 benchmark로 활용 가능.
2. **Identity-term shortcut의 실증적 노출**: KOLD 학습 모델에서 고정관념 기반 shortcut 학습이 실제로 발생함을 Stereotype Flip consistency 수치로 정량화.
3. **경량 Dual-head 프레임워크**: 추가 데이터나 복잡한 adversarial 학습 없이, orthogonality constraint만으로 shortcut 의존을 완화하는 재현 가능한 접근 제시.
4. **한계의 투명한 보고**: Implicit Hate 탐지 실패를 Limitations에 명시하여 연구의 신뢰성 제고 및 후속 연구 방향 제시.

---

## 6. 8주 마일스톤

- **1-2주**: KOLD 분석, split 확정, Baseline 학습 및 Stereotype Flip 평가셋 초안 구축
- **3-4주**: Dual-head 구현, L_ortho 적용, H1/H2 예비 검증
- **5-6주**: Ablation 실험 (lambda/alpha/d sensitivity), Stereotype Flip 평가셋 품질 검수 완료, H3 검증
- **7주**: Fairness 보조 분석, Limitations 섹션 작성, 결과 시각화
- **8주**: 최종 보고서/발표자료 정리, 재현성 체크리스트 확정

---

## 7. Anchor References (Structured)

### 7.1 Representation Disentanglement and Debiasing

1. Elazar and Goldberg (2018), Adversarial Removal of Demographic Attributes from Text Data
   요지: 표현에서 특정 속성 정보를 제거하기 위한 adversarial 접근을 제시했다.
   차이: 본 연구는 adversarial game 대신 dual-head + 직교 제약으로 경량 학습을 지향한다.

2. Ravfogel et al. (2020), Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection
   요지: 반복적 선형 투영으로 보호 속성 정보를 약화시키는 방법을 제시했다.
   차이: 본 연구는 후처리 투영보다 end-to-end 학습 중 decoupling을 직접 유도한다.

### 7.2 Hate Speech Fairness and Robustness

3. Dixon et al. (2018), Measuring and Mitigating Unintended Bias in Text Classification
   연결: identity-term false positive bias, unintended bias
   차이: 본 연구는 한국어 혐오 탐지에서 identity term 치환 기반 consistency를 정량화한다.

4. Sap et al. (2019), The Risk of Racial Bias in Hate Speech Detection
   연결: group-wise error disparity, dataset artifact
   차이: 한국어 KOLD에서 특정 집단 언급과 오류 편차의 관계를 점검한다.

5. Xia et al. (2020), Demoting Racial Bias in Hate Speech Detection
   연결: group-wise error disparity, false positive disparity
   차이: 한국어 혐오 탐지에 맞춰 subgroup별 FPR/TPR/FNR 편차를 분석한다.

6. Zhou et al. (2021), Challenges in Automated Debiasing for Toxic Language Detection
   연결: lexical bias, spurious lexical shortcut reliance
   차이: 한국어 혐오 탐지에서 lexical shortcut과 group-wise 편차를 함께 측정한다.

7. Ramponi and Tonelli (2022), Features or Spurious Artifacts? Data-centric Baselines for Fair and Robust Hate Speech Detection
   연결: spurious lexical shortcut reliance, dataset artifact
   차이: 한국어 환경에서 키워드 변화와 counterfactual consistency 중심으로 점검한다.

8. Lu et al. (2024), Take Its Essence, Discard Its Dross! Debiasing for Toxic Language Detection via Counterfactual Causal Effect (ACL Anthology: 2024.lrec-main.1353)
   연결: misleading lexical bias, biased token over-reliance
   차이: 인과 추정 모듈 대신 dual-head + orthogonality + MLM 보조목표로 학기 프로젝트 범위의 재현 가능성을 높인다.

9. Casula and Tonelli (2025), On the Impact of Hate Speech Synthetic Data on Model Fairness (ACL Anthology: 2025.clicit-1.21)
   차이: 데이터 생성보다 representation decoupling 학습으로 편향 민감도를 줄이는 방법을 제시한다.

10. Bauer et al. (2025), Towards Fairness Assessment of Dutch Hate Speech Detection (ACL Anthology: 2025.woah-1.28)
    차이: counterfactual fairness 평가 관점을 한국어 KOLD에 이식하고, FPR/TPR/FNR 편차를 보고한다.

### 7.3 Korean Hate Speech Benchmarks

11. Moon et al. (2020), BEEP! Korean Corpus of Online News Comments for Toxic Speech Detection
    차이: 데이터셋 구축보다 표현 디커플링 학습과 강건성 검증에 초점.

12. Lim et al. (2022), K-MHaS: A Korean Multi-label Hate Speech Detection Dataset
    차이: 단순 분류 성능을 넘어 bias-semantic decoupling과 공정성 편차 분석을 함께 다룬다.
