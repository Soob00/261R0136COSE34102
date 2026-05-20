# 프로젝트 제안서
## Quantization이 소형 언어모델의 통사적·의미적 이해 능력에 미치는 단계적 손실과 자원 효율성 분석
### NLP 기말 프로젝트 (Colab Pro 기준 현실화 버전)

---

## 1. 연구 배경 및 문제 정의

최근 대규모 언어모델은 다양한 자연어처리 과제에서 높은 성능을 달성하고 있지만, 이러한 성능은 막대한 메모리 사용량과 추론 비용을 전제로 한다. 따라서 모바일, 임베디드, 또는 제한된 GPU 환경에서 실제로 모델을 활용하기 위해서는 모델 경량화가 필수적이며, 그중에서도 `quantization`은 가장 널리 사용되는 효율화 기법 중 하나이다. Quantization은 모델의 가중치와 연산을 낮은 정밀도로 표현함으로써 메모리 사용량을 줄이고 추론 속도를 개선할 수 있다는 장점이 있다. 특히 본 프로젝트는 Colab Pro 수준의 제한된 실험 환경에서도 재현 가능한 범위에서, low-bit quantization이 언어 이해 능력에 미치는 영향을 분석하는 것을 목표로 한다.

그러나 기존 연구의 다수는 quantization의 효과를 전체 정확도, perplexity, latency, memory reduction과 같은 거시적 지표로만 평가한다. 이러한 접근은 "모델이 전체적으로 얼마나 나빠졌는가"는 보여주지만, "어떤 언어적 능력이 먼저 손상되는가"는 설명하지 못한다. 예를 들어, 양자화 이후에도 단순 문법성 판단은 비교적 유지될 수 있지만, 장거리 대명사 해소나 복잡한 통사 구조 이해는 더 빠르게 붕괴할 가능성이 있다. 즉, quantization의 영향은 단순한 평균 성능 저하가 아니라, 언어 능력별 `selective degradation`으로 나타날 수 있다.

본 연구의 핵심 문제의식은 여기에 있다. 우리는 quantization이 언어모델의 다양한 언어적 능력을 균등하게 손상시키는지, 혹은 특정 언어 현상에 더 치명적인 영향을 주는지를 분석하고자 한다. 특히 통사 구조 이해와 대명사 지시 해석처럼 성격이 다른 언어 능력을 분리해서 살펴봄으로써, "모델을 가볍게 만들 때 무엇을 먼저 잃게 되는가"를 NLP 관점에서 해석하고자 한다. 다만 학기 프로젝트의 현실성을 고려해, 본 연구는 새로운 양자화 알고리즘을 제안하기보다는 기존 양자화 기법을 활용하여 `언어 능력별 손실 패턴`을 분석하는 데 초점을 둔다.

---

## 2. 핵심 연구 질문

본 연구는 다음 질문에 답하는 것을 목표로 한다.

1. Quantization은 언어모델의 언어 능력을 균등하게 손상시키는가?
2. 통사적 능력과 의미적 능력은 양자화에 대해 서로 다른 민감도를 가지는가?
3. 언어 능력 손실이 시작되는 임계점은 메모리 절감과 추론 속도 향상이라는 시스템 이득과 어떻게 trade-off를 이루는가?
4. 실제 on-device 환경을 고려했을 때, 언어 능력 보존과 자원 효율성 사이의 최적 operating point는 어디인가?

쉽게 말하면, 이 연구는 "양자화를 하면 모델이 무엇부터 먼저 까먹는가?"를 정량적으로 묻는 프로젝트이다.

---

## 3. 관련 연구 및 최신 논문 정리

### 3.1 Quantization 핵심 기반 논문

- **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**  
  대규모 Transformer에 8-bit 양자화를 실제로 적용 가능한 수준으로 정리한 대표적 출발점이다. 대규모 모델을 효율적으로 서빙하는 연구 흐름의 기본 참고점이 된다.

- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**  
  post-training quantization의 가장 널리 쓰이는 baseline 중 하나로, 학기 프로젝트에서 재현 및 비교 기준으로 적합하다.

- **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**  
  activation outlier 문제를 다루며 W8A8 계열 양자화의 안정성을 높인 논문으로, 양자화 방법 간 비교 실험의 중요한 기준점이 된다.

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**  
  실제 배포 환경에서 강한 성능을 보이는 4-bit 계열 방법론으로, low-bit 설정의 현실적인 baseline으로 적합하다.

### 3.2 Quantization 영향 분석 관련 최신 논문

- **How Does Quantization Affect Multilingual LLMs?** (EMNLP Findings 2024)  
  양자화의 부정적 영향이 언어별로 다르며, 자동 평가 지표가 실제 손실을 과소평가할 수 있음을 보여준다. 이는 quantization의 영향이 단순 평균 점수로 환원되지 않는다는 점을 뒷받침한다.

- **Does quantization affect models' performance on long-context tasks?** (EMNLP 2025)  
  8-bit 양자화는 상대적으로 안정적이지만, 4-bit에서는 long-context 태스크에서 큰 손실이 발생할 수 있음을 보였다. 이는 quantization damage가 task-dependent하다는 점을 보여주는 최신 증거다.

- **Why Do Some Inputs Break Low-Bit LLM Quantization?** (EMNLP 2025)  
  일부 입력이 저비트 양자화에서 유독 취약해지는 이유를 분석하며, 입력 특성과 계층별 민감도가 손실 양상에 영향을 줄 수 있음을 제안한다.

### 3.3 언어 능력 평가를 위한 핵심 benchmark

- **Holmes: A Benchmark to Assess the Linguistic Competence of Language Models** (TACL 2024)  
  syntax, morphology, semantics, discourse 등 다양한 언어 현상을 세분화해 평가하는 benchmark이다. 본 연구의 문제의식과 가장 가깝지만, quantization 맥락에서 체계적으로 활용된 사례는 아직 제한적이다.

- **BLiMP: The Benchmark of Linguistic Minimal Pairs for English**  
  최소대립쌍을 기반으로 주어-동사 일치, filler-gap dependency, binding 등 구체적 통사 현상을 직접 측정할 수 있다.

- **GAP: Gendered Ambiguous Pronouns**  
  대명사 지시 해석 능력을 평가하는 대표적 benchmark로, 의미적·담화적 이해를 살펴보기에 적합하다.

위 선행연구들을 종합하면, quantization 연구는 충분히 축적되어 있고, linguistic benchmark도 잘 마련되어 있다. 그러나 이 둘을 결합해 "양자화가 어떤 언어 능력을 어떤 순서로 무너뜨리는가"를 분석하는 연구는 상대적으로 부족하다. 본 연구는 바로 이 공백을 메우고자 한다.

---

## 4. 연구의 차별점과 가설

본 연구의 차별점은 quantization을 단순한 압축 기법으로 보지 않고, `언어 능력에 대한 스트레스 테스트`로 활용한다는 데 있다. 기존 연구가 전체 정확도와 시스템 성능의 trade-off를 주로 분석했다면, 우리는 언어적 능력을 세부 현상 단위로 분해해 손실 곡선을 그린다. 또한 이를 메모리 사용량 및 추론 속도와 함께 분석함으로써, "무엇을 잃고 무엇을 얻는가"를 더 해석 가능하게 제시한다.

본 연구는 다음과 같은 가설을 설정한다.

1. INT8 양자화는 대부분의 언어 능력을 비교적 안정적으로 보존하지만, INT4 이하에서는 의미적·담화적 능력이 더 빠르게 손상된다.
2. 통사 능력도 모든 하위 현상이 동일하게 반응하지 않으며, 장거리 의존성이나 binding과 같은 복잡한 현상이 먼저 악화된다.
3. 전체 평균 정확도는 quantization 손실을 과소평가하며, phenomenon-level 분석을 통해 더 큰 손실이 드러난다.
4. 가장 큰 자원 절감이 나타나는 지점과 가장 심각한 언어 능력 붕괴가 시작되는 지점은 일치하지 않을 수 있다.

---

## 5. 데이터셋 구성 및 수집 방법

본 연구는 통사적 능력과 의미적 능력을 각각 측정할 수 있는 데이터셋을 중심으로 실험을 구성한다.

### 5.1 BLiMP

BLiMP는 문법적으로 올바른 문장과 잘못된 문장을 최소대립쌍 형태로 제공하는 benchmark로, 통사 현상을 세부적으로 측정할 수 있다. 본 연구에서는 BLiMP의 전체 점수뿐 아니라 subject-verb agreement, filler-gap dependency, binding, NPI licensing 등 현상별 accuracy를 별도로 기록하여 어떤 통사 현상이 먼저 손상되는지 분석할 예정이다.

### 5.2 GAP

GAP는 ambiguous pronoun resolution 데이터셋으로, 문장 속 대명사가 어떤 인물을 가리키는지 예측하도록 구성되어 있다. 이는 단순 문법성이 아니라 의미적·담화적 이해를 요구하므로, 양자화가 semantic/anaphora 능력에 미치는 영향을 측정하는 데 적합하다.

### 5.3 선택적 보조 데이터셋: WinoGrande 또는 Holmes subset

시간과 자원이 허용된다면 WinoGrande를 추가해 pronoun resolution 및 commonsense reasoning 환경에서의 robustness를 검증할 수 있다. 또한 Holmes의 일부 subset을 활용하면 syntax 외의 semantics/discourse 현상으로 분석 범위를 확장할 수 있다. 그러나 본 프로젝트의 `minimum viable project (MVP)`는 BLiMP와 GAP 두 축만으로도 완결되도록 설계한다.

### 5.4 데이터 수집 절차

데이터셋은 공식 배포처 또는 Hugging Face `datasets`를 통해 수집한다. 수집 후에는 다음과 같은 전처리를 수행한다.

- 공식 split 유지
- 입력 형식 통일
- phenomenon 또는 difficulty metadata 유지
- GAP에서 pronoun-candidate distance, ambiguity 수준 등 추가 메타정보 정리

이를 통해 단순 평균 점수 외에 구조적 분석이 가능한 평가셋을 구성한다.

---

## 6. 모델 및 양자화 설정

메인 실험 모델은 Colab Pro 환경에서 안정적으로 반복 평가가 가능한 크기의 공개 모델로 제한한다. 우선 후보는 `Qwen2.5-3B` 또는 `Llama-3.2-3B`이며, 이 중 하나를 대표 모델로 선택한다. 프로젝트의 핵심은 여러 모델을 넓게 비교하는 것이 아니라, 하나의 모델을 대상으로 양자화 수준에 따른 언어 능력 변화를 정밀하게 분석하는 데 있다.

적용할 양자화 설정은 다음과 같다.

- BF16 또는 FP16 baseline
- INT8
- INT4-NF4

핵심 비교는 `baseline / INT8 / INT4-NF4`의 세 조건으로 구성한다. 이 설정은 Colab Pro 환경에서도 비교적 안정적으로 실행 가능하며, bit-width에 따른 손실 곡선을 관찰하기에 충분하다. 시간이 허락할 경우에만 GPTQ 또는 AWQ 중 하나를 추가해 method-level 보조 실험을 수행한다.

---

## 7. 실험 파이프라인

본 연구의 실험 파이프라인은 다음과 같다.

1. Baseline 모델을 준비하고 FP16 또는 BF16 상태에서 baseline 언어 성능과 시스템 성능을 측정한다.
2. 동일 모델에 INT8 및 INT4-NF4 양자화를 적용한다.
3. 각 모델을 BLiMP와 GAP에 평가하여 언어 능력 변화를 측정한다.
4. 각 실험에서 peak memory usage와 latency를 함께 기록하고, 가능하면 throughput도 측정한다.
5. bit-width별, phenomenon별 성능 저하 곡선을 시각화한다.
6. 언어 능력 보존율과 자원 절감 효과를 함께 분석하여 trade-off plot을 도출한다.

평가 방식은 가능한 한 prompt engineering에 의존하지 않고 `likelihood-based scoring`을 사용한다. BLiMP에서는 grammatical sentence가 ungrammatical sentence보다 더 높은 확률을 받는지 비교하고, GAP에서는 대명사가 가리키는 정답 후보가 더 높은 점수를 받는지를 기준으로 평가한다. 이렇게 하면 instruction tuning이나 prompt formatting의 영향을 줄이고, 양자화의 순수한 영향을 보다 직접적으로 측정할 수 있다.

### 7.1 Minimum Viable Project (MVP)

본 프로젝트의 MVP는 다음 네 요소가 모두 완료되면 성립한다.

1. `3B급 공개 모델 1개`에 대해 baseline, INT8, INT4-NF4 세 버전 준비
2. `BLiMP`로 통사 현상별 성능 측정
3. `GAP`로 대명사 지시 해석 성능 측정
4. 각 설정에서 `메모리 사용량`과 `latency`를 측정하여 trade-off plot 제시

이 MVP만 달성해도 "양자화가 통사와 의미 능력을 동일하게 손상시키지 않는다"는 핵심 연구 질문에는 답할 수 있다. 이후 여유가 있으면 WinoGrande, method ablation, 추가 profiling을 확장 실험으로 붙이는 구조다.

---

## 8. Evaluation 계획

### 8.1 언어 성능 평가

- BLiMP overall accuracy
- BLiMP phenomenon-level accuracy
- GAP accuracy 또는 F1
- 선택적으로 WinoGrande accuracy
- Relative degradation rate: `(FP16 score - quantized score) / FP16 score`

이 지표들을 통해 quantization이 특정 언어 능력에 더 큰 영향을 미치는지 분석한다.

### 8.2 시스템 성능 평가

- Peak GPU memory usage
- Inference latency
- Throughput (선택)

이 지표를 통해 quantization이 실제 배포 환경에서 제공하는 효율 이득을 수치화한다.

### 8.3 통합 분석

최종적으로는 아래와 같은 통합 지표를 사용해 결과를 해석한다.

- `Linguistic Retention Score`: 양자화 후 점수 / FP16 점수
- `Efficiency Gain Score`: 메모리 절감률, latency improvement
- Trade-off plot: 언어 능력 보존율과 자원 효율성 사이의 균형점 시각화

이 분석을 통해 "정확도 손실 5% 이내에서 가장 큰 메모리 절감을 주는 설정"과 같은 실질적 가이드라인을 제시할 수 있다. 학기 프로젝트의 범위에서는 복잡한 시스템 공동설계보다, 재현 가능한 memory-latency-performance trade-off를 제시하는 것을 우선 목표로 한다.

---

## 9. Ablation Study

본 연구에서는 다음과 같은 ablation을 수행한다.

### 9.1 Bit-width Ablation

FP16, INT8, INT4를 비교해 어느 수준에서 언어 능력 손실이 급격히 커지는지 분석한다.

### 9.2 Method Ablation

동일한 4-bit 조건에서도 GPTQ, AWQ, NF4가 서로 다른 손실 양상을 보이는지 비교한다. 다만 이는 확장 실험으로 두고, MVP 단계에서는 NF4를 기준 설정으로 사용한다.

### 9.3 Phenomenon Ablation

BLiMP 내부 세부 현상별로 결과를 분해해, 어떤 통사 현상이 가장 먼저 무너지는지 분석한다.

### 9.4 Distance/Complexity Ablation

GAP에서 pronoun과 antecedent 사이의 거리, 문장 길이, ambiguity 수준 등을 기준으로 성능 변화를 측정해 장거리 의존성이 low-bit 설정에서 더 취약한지 검증한다.

이러한 ablation은 평균 점수만으로는 드러나지 않는 구조적 약점을 밝히는 데 핵심적이다.

---

## 10. 기대 결과 및 기여

본 연구의 기대 기여는 다음과 같다.

1. Quantization의 영향을 전체 정확도 수준이 아니라 `언어 능력별 단계적 손실`로 분석한다.
2. 통사적 능력과 의미적 능력이 양자화에 대해 서로 다른 민감도를 가진다는 점을 실험적으로 보여준다.
3. 언어 능력 보존율과 자원 효율성 사이의 trade-off를 정량화하여, 실제 on-device NLP 환경을 위한 가이드라인을 제시한다.
4. Efficient NLP 연구에 언어학적 해석 가능성을 추가함으로써, 단순 시스템 최적화가 아닌 분석 중심의 NLP 프로젝트로 확장한다.

이 프로젝트의 가장 큰 장점은 "모델을 얼마나 줄일 수 있는가"에 그치지 않고, "줄였을 때 어떤 언어 지능이 먼저 사라지는가"를 보여준다는 점이다. 따라서 본 연구는 수업의 NLP 정체성과 최근 efficient NLP 흐름을 동시에 만족시키는 주제로 볼 수 있다.

또한 본 설계는 Colab Pro 수준의 자원에서도 수행 가능하도록 범위를 조정했다는 점에서 현실성이 높다. 즉, 지나치게 큰 모델이나 복잡한 양자화 알고리즘 비교에 의존하지 않고도, NLP적으로 의미 있는 분석 결과를 도출할 수 있다.

---

## 11. 예상 일정

- **~ 4/10**: 관련 논문 정리, 3B급 모델 선정, baseline 구축
- **~ 4/25**: INT8 및 INT4-NF4 적용, BLiMP/GAP 1차 평가 완료
- **~ 5/10**: 메모리·latency 측정, phenomenon-level 분석 및 distance ablation 수행
- **~ 5/25**: 그래프 시각화, optional 확장 실험 수행, 최종 보고서 작성
- **~ 5/29**: 최종 발표 및 제출

---

## 12. 참고할 핵심 논문

1. Dettmers et al. **LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale**. arXiv 2022.  
   https://arxiv.org/abs/2208.07339

2. Frantar et al. **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**. arXiv 2022.  
   https://arxiv.org/abs/2210.17323

3. Xiao et al. **SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models**. ICML 2023.  
   https://proceedings.mlr.press/v202/xiao23c.html

4. Lin et al. **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**. MLSys 2024.  
   https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html

5. Marchisio et al. **How Does Quantization Affect Multilingual LLMs?** Findings of EMNLP 2024.  
   https://aclanthology.org/2024.findings-emnlp.935/

6. Mekala et al. **Does quantization affect models' performance on long-context tasks?** EMNLP 2025.  
   https://aclanthology.org/2025.emnlp-main.479/

7. Wang et al. **Why Do Some Inputs Break Low-Bit LLM Quantization?** EMNLP 2025.  
   https://aclanthology.org/2025.emnlp-main.168/

8. Waldis et al. **Holmes: A Benchmark to Assess the Linguistic Competence of Language Models**. TACL 2024.  
   https://aclanthology.org/2024.tacl-1.88/

9. Warstadt et al. **BLiMP: The Benchmark of Linguistic Minimal Pairs for English**. TACL 2020.  
   https://aclanthology.org/2020.tacl-1.25/

10. Webster et al. **Mind the GAP: A Balanced Corpus of Gendered Ambiguous Pronouns**. TACL 2018.  
    https://arxiv.org/abs/1810.05201
