# Validity-Gated Counterfactual Consistency Regularization for Fair Korean Hate Speech Detection

## 1. Introduction

Large pre-trained language models fine-tuned for hate speech detection tend to rely on the mere presence of identity terms (e.g., 여성, 조선족, 무슬림) as a shortcut rather than understanding the actual offensive content.
A model that exploits such shortcuts will produce inconsistent predictions when the identity term is replaced by a socially equivalent counterpart, revealing a target-group bias.

Counterfactual consistency regularization (CCR) addresses this by penalizing the model when its predictions change under identity-term substitution.
However, a critical and largely unexamined question is: *which substitutions are valid?*
Swapping 트랜스젠더 with 이성애자 conflates two different social categories; swapping 노인 with 청년 within the age category preserves social structure.
Using invalid substitutions in CCR may introduce contradictory training signal and harm both accuracy and fairness.

We ask: **Does restricting CCR to same-category identity swaps (validity gating) improve the fairness–accuracy trade-off in Korean hate speech detection?**

## 2. Related Work

**Counterfactual fairness in hate speech detection.**
Lu et al. (WOAH 2021) show that applying logit pairing only to linguistically valid counterfactual pairs reduces target-group bias without degrading accuracy.
We adapt this insight to the CCR framework (KL-divergence consistency loss) and study it for the first time in Korean.

**Hate speech detection frameworks.**
Balkir et al. (2022) propose a unified framework combining accuracy, robustness, and fairness; we borrow their evaluation perspective.

**Korean hate speech corpora.**
KOLD (Jeong et al., EMNLP 2022) and K-MHaS (Lee et al., 2022) provide Korean hate speech baselines.
We use **K-HATERS** (Heo et al., EMNLP 2023), which uniquely provides per-example target-group labels, enabling per-group FPR evaluation—an evaluation methodology absent from prior Korean hate speech training studies.

**Gap addressed.**
Prior Korean hate speech work focuses on dataset construction and baseline comparison.
No prior work examines the effect of *counterfactual pair validity* on consistency regularization in Korean, nor evaluates target-group FPR on K-HATERS.

## 3. Method

### 3.1 Base Classifier

We fine-tune **KLUE-RoBERTa-base** with a linear classification head on binary hate labels.

```
L_cls = CrossEntropy(f(x), y)
```

### 3.2 Consistency Regularization

For each training sentence $x$ containing an identity term $t$, we generate a counterfactual $x'$ by substituting $t$ with a counterpart $t'$.
The consistency loss penalizes divergence between the model's output distributions on the original and counterfactual:

```
L_cons = sym-KL( p(x) || p(x') )
       = [ KL(p(x) || p(x')) + KL(p(x') || p(x)) ] / 2
```

Total loss:

```
L = L_cls + λ · gate(x, x') · L_cons
```

### 3.3 Validity Gate

We define a lexicon of 26 directed swap pairs grouped into 6 social categories:

| Category | Example swap |
|---|---|
| gender | 여성 ↔ 남성, 페미 ↔ 한남 |
| ethnicity | 조선족 ↔ 한국인, 외국인 ↔ 내국인 |
| age | 노인 ↔ 청년 |
| religion | 무슬림 ↔ 기독교인 |
| sexuality | 동성애자 ↔ 이성애자 |
| disability | 장애인 ↔ 비장애인 |

The gate function:

```
gate(x, x') = 1   if swap(t → t') where category(t) == category(t')
            = 0   otherwise
```

By construction, all pairs in our lexicon are same-category; the "Naive Swap" baseline uses these pairs *without* this semantic constraint, allowing future work to extend to cross-category pairs as an ablation.

### 3.4 Ablation Conditions

| Condition | $L_{cls}$ | Counterfactual | Gate | $\lambda$ |
|---|:---:|---|:---:|---:|
| **Baseline** | ✓ | — | — | 0 |
| **Masking CCR** | ✓ | $t$ → `[MASK]` | — | 0.1 |
| **Naive Swap CCR** | ✓ | $t$ → $t'$ (same-cat) | ✗ | 0.1 |
| **Validity-Gated CCR** | ✓ | $t$ → $t'$ (same-cat) | ✓ | 0.1 |

*Note:* Masking CCR is the prior-work approach; Validity-Gated CCR is our proposed method.
The comparison Masking vs. Naive Swap vs. Validity-Gated isolates the contribution of (a) using actual identity swap over masking, and (b) validity gating over unfiltered swaps.

## 4. Data

### 4.1 Primary Dataset: K-HATERS

| Split | Size |
|---|---|
| Train | 172,000 |
| Validation | 10,000 |
| Test | 10,000 |

**Label binarization:**
- `normal` → 0
- `offensive`, `L1_hate`, `L2_hate` → 1

**Target-group labels** (available per example): political, individual, gender, region, age, job, others.
These enable per-group FPR evaluation.

### 4.2 Counterfactual Pair Statistics

~9.5% of training sentences contain at least one identity term from our lexicon (~16,300 pairs).
Category distribution: gender (~45%), ethnicity (~45%), age (~6%), religion/sexuality/disability (~4% combined).

Pairs are used **only for the CCR loss**, not added to the training set.

## 5. Evaluation

### 5.1 Clean Performance

- Macro-F1 on K-HATERS test set

### 5.2 Counterfactual Robustness

Evaluated on all test sentences with a swappable identity term:

- **Flip Rate**: $\Pr[\hat{y}(x) \neq \hat{y}(x')]$ — lower is better
- **Mean Logit Gap**: $\mathbb{E}[|p_{\text{hate}}(x) - p_{\text{hate}}(x')|]$ — lower is better

### 5.3 Fairness

Using K-HATERS `target_label` annotations:

- **Per-group FPR**: false positive rate for label=0 examples, per target group
- **FPR Gap**: $\max_g \text{FPR}_g - \min_g \text{FPR}_g$ — lower indicates more equalized FPR across groups

### 5.4 Statistical Test

Paired t-test (3 seeds) on Flip Rate: Baseline vs. Validity-Gated CCR.

## 6. Experiments

All conditions: KLUE-RoBERTa-base, 5 epochs, batch size 16, AdamW (lr=2e-5, wd=0.01), 3 seeds (42, 123, 456).
λ sensitivity analysis (0.05, 0.1, 0.2) on Validity-Gated CCR (3 epochs).

## 7. Expected Results

We hypothesize:

1. Both Masking CCR and Swap CCR reduce Flip Rate over Baseline, confirming that consistency regularization mitigates identity shortcuts.
2. Validity-Gated CCR achieves a better fairness–accuracy trade-off than Naive Swap CCR, demonstrating the value of same-category constraints.
3. The FPR gap across target groups narrows under Validity-Gated CCR, while Macro-F1 is maintained.

If Validity-Gated CCR reduces Flip Rate and FPR Gap without degrading F1, this supports the claim that *counterfactual pair quality matters* for fairness training in Korean.

## 8. Conclusion

We propose validity-gated counterfactual consistency regularization for Korean hate speech detection.
By restricting consistency penalties to same-category identity swaps and evaluating with per-group FPR on K-HATERS, we provide the first systematic study of counterfactual pair validity in Korean hate speech fairness training.
Our results are expected to show that validity gating reduces target-group bias with minimal accuracy loss, whereas naive masking or unfiltered swaps yield suboptimal trade-offs.

## References

- Lu et al. (2021). *Improving Counterfactual Generation for Fair Hate Speech Detection.* WOAH @ ACL 2021.
- Balkir et al. (2022). *An Effective, Robust and Fairness-aware Hate Speech Detection Framework.* arXiv:2409.17191.
- Jeong et al. (2022). *KOLD: Korean Offensive Language Dataset.* EMNLP 2022.
- Heo et al. (2023). *K-HATERS: A Hate Speech Dataset with Target-Specific Ratings.* EMNLP Findings 2023.
- Lee et al. (2022). *K-MHaS: A Multi-label Hate Speech Detection Dataset in Korean.* arXiv:2208.10684.
- Park et al. (2021). *KLUE: Korean Language Understanding Evaluation.* NeurIPS 2021 Datasets Track.
