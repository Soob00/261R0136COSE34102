# 프로젝트 제안서
## 한국어 언어 추론을 위한 Chain-of-Thought Distillation 전략 비교 연구
### NLP 기말 프로젝트 — 3인팀

---

## 1. 연구 동기

대형 언어 모델(LLM)은 Chain-of-Thought(CoT) 프롬프팅을 통해 뛰어난 추론 능력을 보여주고 있습니다. 그러나 이러한 연구는 **대부분 영어를 대상으로** 이루어졌으며, 한국어 소형 모델의 추론 능력 향상에 관한 연구는 매우 부족한 상황입니다.

**핵심 연구 질문:**
> *"대형 모델의 Chain-of-Thought 추론 능력을 소형 한국어 모델에 효과적으로 전달할 수 있는가? 그리고 한국어의 언어적 특성을 고려했을 때 어떤 Distillation 전략이 가장 효과적인가?"*

---

## 2. 핵심 논문 (Anchor Paper)

**"Teaching Small Language Models Reasoning through Counterfactual Distillation"**
— Feng et al., EMNLP 2024

### 요약
- **문제:** 기존 CoT Distillation은 학습 패턴을 표면적으로 외우게 되어, 분포가 다른 데이터에서 성능이 급락함
- **핵심 아이디어:** *반사실적(Counterfactual)* 학습 데이터 생성 — 정답 레이블을 반전시킨 뒤 Teacher LLM에게 새로운 CoT를 생성하게 함. 이를 통해 Student 모델이 "왜 이 답이 맞는가"를 학습하게 함
- **결과:** 여러 추론 벤치마크에서 Standard CoT-SFT 대비 성능 향상

### 우리가 가져올 것
Counterfactual Distillation 프레임워크를 **영어 수학/논리 문제가 아닌 한국어 NLI 추론 태스크**에 적용

---

## 3. 연구 방법 및 혁신점

### 파이프라인

```
[Teacher LLM: GPT-4o]
  ↓  한국어 NLI 문제 입력
  ↓  단계별 CoT 풀이 생성 (최대 256 토큰 — 프롬프트로 제한)
  ↓
[CoT 학습 데이터 — 길이 제어됨]
  ↓  3가지 Distillation 전략 각각 적용
  ↓
[Student: EXAONE-3-1.2B]
  ↓  QLoRA fine-tuning (시퀀스 길이 1024 제한, OOM 방지)
  ↓
[평가: 인도메인 + OOD]
```

### 혁신: 한국어 언어학적 가설

단순히 "영어 방법론을 한국어에 적용"하는 데 그치지 않고, **검증 가능한 언어학적 가설**을 설정합니다:

| 한국어 특성 | 가설 | 예측 결과 |
|---|---|---|
| **SOV 어순** | 한국어는 결론이 문장 끝에 위치 → 결론을 먼저 제시하는 영어식 CoT는 한국어에 부자연스러움 | **Self-Guided > Standard CoT-SFT** (한국어 NLI에서) |
| **교착어적 특성** | 논리적 연결어가 형태소에 내포됨 (-므로, -기 때문에) → 레이블 반전 시 형태소 일관성이 깨질 수 있음 | **Counterfactual Distillation**이 한국어에서는 영어보다 덜 자연스러운 Rationale 생성 가능 |
| **경어 체계** | 존댓말/반말에 따라 추론 스타일이 달라질 수 있음 | Rationale 품질이 어체에 따라 달라지며, Self-Guided 필터링이 이를 완화할 수 있음 |

이 가설들은 **실험적으로 검증 가능** — 각 전략의 결과가 가설을 확인하거나 반증하며, 이것이 논문의 분석 섹션이 됨

### 3가지 Distillation 전략 (팀원 1인 1전략)

| # | 전략 | 설명 | 근거 논문 |
|---|---|---|---|
| 1 | **Standard CoT-SFT** | Teacher CoT 출력을 그대로 학습에 사용 | Wei et al., NeurIPS 2022 |
| 2 | **Counterfactual Distillation** | 레이블 반전 CoT 쌍 생성 → OOD 일반화 향상 | Feng et al., EMNLP 2024 |
| 3 | **Self-Guided Rationale Selection** | CoT Rationale을 난이도/품질 기준으로 필터링 후 학습 | EMNLP 2025 Findings |

---

## 4. 데이터셋

모든 데이터셋 **공개 제공** — 직접 Annotation 불필요

| 데이터셋 | 태스크 | 규모 | 역할 |
|---|---|---|---|
| **KLUE-NLI** | 자연어 추론 (함의/모순/중립) | 30K | 메인 학습 + 인도메인 평가 |
| **XNLI (한국어)** | 다국어 NLI — 동일 태스크, Wikipedia 기반 분포 이동 | 5K (테스트) | OOD 평가 (같은 추론 형태, 다른 분포) |
| **LogicKor** | 한국어 논리 추론 벤치마크 (다분야) | — | 광범위한 추론 일반화 테스트 |

**KorQuAD 제외 이유:**
KorQuAD 1.0은 *Extractive QA* — 지문에서 답 구간을 찾는 독해 태스크. 다단계 추론을 평가하지 않아 본 연구 목적에 부적합.

**XNLI를 OOD로 선정한 이유:**
동일한 추론 태스크(NLI: 함의/모순/중립)이지만 텍스트 분포가 다름(위키피디아 기반 다국어 텍스트). 표면 패턴이 아닌 **추론 능력 자체의 전이 여부**를 검증할 수 있는 가장 원칙적인 설정.

**CoT 길이 제어:**
GPT-4o 프롬프트: *"한국어로, 단계별로, 최대 256 토큰 이내로 추론 과정을 설명하세요."*
→ 지나치게 긴 시퀀스로 인한 Student 학습 시 OOM 방지

---

## 5. 모델

| 역할 | 모델 | 선택 이유 |
|---|---|---|
| **Teacher** | GPT-4o (API) | 한국어 CoT 생성 품질 최고 |
| **Student** | EXAONE-3-1.2B (LG AI Research) | 소형, 한국어 특화, 오픈소스, Colab 실행 가능 |
| **Baseline** | EXAONE-3-1.2B zero-shot + few-shot | Fine-tuning 없이 비교 기준선 |

**컴퓨팅 계획:**
- QLoRA (4-bit 양자화 + LoRA 어댑터) — VRAM 사용량 대폭 절감
- 최대 시퀀스 길이: 1,024 토큰
- Gradient Checkpointing 활성화
- Google Colab Pro + 추가 GPU 구매

---

## 6. 평가

| 지표 | 방법 | 측정 내용 |
|---|---|---|
| **정확도(Accuracy)** | 정답 매칭 (함의/모순/중립) | 최종 답변 정확도 |
| **OOD 일반화** | KLUE-NLI 학습 → XNLI 한국어 테스트 | 다른 분포로 추론 능력 전이 여부 |
| **추론 충실도(Faithfulness)** | ① **ROSCOE** (비지도 자동 단계별 점수) ② **LLM-as-a-Judge** (GPT-4o가 CoT가 정답을 논리적으로 지지하는지 평가) | CoT 추론 과정이 실제로 정답을 이끌어내는가 |
| **Ablation: CoT vs No-CoT** | 답변만 SFT vs 전체 CoT-SFT 비교 | 추론 체인이 실제로 도움이 되는가 |

**Faithfulness 평가 파이프라인:**
1. ROSCOE (ICLR 2023): 비지도 방식, 참조 생성 불필요 → 전체 테스트셋 적용
2. GPT-4o-as-Judge: 샘플 200개에 대해 CoT가 레이블을 논리적으로 지지하는지 평가 → API 비용 통제

---

## 7. 마일스톤

### 중간 점검 (~4-5주차)
- [ ] GPT-4o CoT 생성 파이프라인 완성 (256토큰 제한, 한국어 프롬프팅 검증)
- [ ] Baseline 측정 완료: EXAONE-3-1.2B zero-shot on KLUE-NLI + XNLI 한국어
- [ ] 전략 1 (Standard CoT-SFT) 구현 및 baseline 대비 성능 향상 확인
- [ ] QLoRA 설정 안정화 확인 (seq len 1024에서 OOM 없음)

### 최종 (8주차)
- [ ] 3가지 전략 모두 구현 및 비교 완료
- [ ] XNLI 한국어 OOD 평가 완료
- [ ] LogicKor 일반화 테스트 완료
- [ ] ROSCOE + LLM-as-Judge 충실도 평가 완료
- [ ] 한국어 언어학 가설 검증/반증 결과 정리
- [ ] Ablation 스터디 완료
- [ ] 논문 작성 완료

---

## 8. 팀원 역할 분담

| 팀원 | 담당 |
|---|---|
| 팀원 1 | GPT-4o CoT 생성 파이프라인 + Standard CoT-SFT (전략 1) + ROSCOE 평가 |
| 팀원 2 | Counterfactual Distillation (전략 2) + KLUE-NLI 인도메인 실험 |
| 팀원 3 | Self-Guided Rationale Selection (전략 3) + XNLI OOD + LLM-as-Judge 충실도 평가 |

---

## 9. 관련 연구

| 논문 | 학회 | 연도 | 관련성 |
|---|---|---|---|
| Chain-of-Thought Prompting Elicits Reasoning in LLMs — Wei et al. | NeurIPS | 2022 | CoT 기초 방법론 |
| Teaching Small LMs Reasoning via Counterfactual Distillation — Feng et al. | EMNLP | 2024 | **핵심 앵커 논문** |
| Towards Efficient CoT Distillation: Self-Guided Rationale Selector | EMNLP Findings | 2025 | 전략 3 기반 |
| ROSCOE: A Suite of Metrics for Step-by-Step Reasoning | ICLR | 2023 | 충실도 자동 평가 |
| Measuring Faithfulness in Chain-of-Thought Reasoning — Anthropic | arXiv | 2023 | 충실도 평가 방법론 |
| CODI: Compressing CoT into Continuous Space | EMNLP | 2025 | 관련 Distillation 연구 |
| CLIcK: Cultural & Linguistic Intelligence in Korean — Kim et al. | LREC-COLING | 2024 | 한국어 벤치마크 참고 |
| Making Qwen3 Think in Korean with Reinforcement Learning | arXiv | 2025 | 한국어 추론 — 근접 관련 연구 |
| KLUE: Korean Language Understanding Evaluation | NeurIPS | 2021 | KLUE-NLI 출처 |

---

## 10. 이 프로젝트를 선택한 이유

- **시의성:** 추론 Distillation은 2026년 NLP의 가장 핵심 주제 (DeepSeek-R1, o3, Claude 3.7)
- **참신성:** 한국어 특화 CoT Distillation은 거의 미연구 상태; 검증 가능한 언어학적 가설 추가로 단순 "영어 방법론 한국어 적용"을 넘어섬
- **실현 가능성:** 공개 데이터셋 + QLoRA로 Colab Pro에서 실행 가능, 컴퓨팅 리스크 대응 완료
- **명확한 기여:** 3가지 Distillation 전략 × 한국어 언어학 가설 × 인도메인 + OOD 평가
- **엄밀한 평가:** ROSCOE (완전 자동) + LLM-as-Judge (샘플링) 충실도 평가; XNLI 원칙적 OOD 평가
