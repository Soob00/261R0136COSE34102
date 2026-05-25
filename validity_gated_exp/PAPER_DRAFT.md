# Paper Draft: Validity-Gated CCR for Korean Hate Detection

Working title:

> Validity-Gated Counterfactual Consistency Regularization for Korean
> Hate/Offensive Language Detection

## Abstract

Hate and offensive language classifiers can overreact to identity terms,
predicting toxicity from group mentions rather than from harmful context. This
problem is especially delicate in Korean, where identity terms interact with
particles, slang, quoted speech, and culturally specific contexts. We study
counterfactual consistency regularization for Korean hate/offensive language
detection using KLUE-RoBERTa on K-HATERS. We compare standard fine-tuning,
ungated identity-swap consistency, strict validity-gated consistency, and a
coverage-matched strict variant that compensates for the lower number of valid
counterfactual pairs. Across three seeds, the coverage-matched strict method
preserves Macro-F1 while achieving the best Pair Accuracy and Strict Pair
Accuracy among confirmed methods. The results suggest that counterfactual
regularization is more reliable when the validity of identity swaps is modeled
explicitly, rather than treating every lexical swap as a valid invariance.

## 1. Introduction

Korean hate/offensive language detection systems are often trained as standard
text classifiers. This can produce good aggregate accuracy while hiding a
fairness and robustness problem: the model may treat identity mentions as
shortcuts. For example, replacing one group term with another can flip the model
prediction even when the offensive intent of the sentence should remain the
same. Conversely, neutral identity mentions can be incorrectly treated as hate.

Counterfactual data augmentation and consistency regularization are natural
tools for this problem. However, naive identity swapping is risky. In Korean,
simple lexical replacement can break morphology, change pragmatic meaning, or
turn a valid sentence into an unnatural one. If such invalid pairs are used as
training signals, the model may learn artificial invariances rather than
meaningful robustness.

This project asks:

> Can validity-gated counterfactual consistency improve Korean hate/offensive
> language detection robustness without sacrificing standard classification
> performance?

Our contribution is not merely a new evaluation set. We intervene in the
training objective by adding a consistency loss over counterfactual pairs, and
we study whether strict validity filtering plus coverage-aware weighting gives
a better signal than ungated swaps.

## 2. Related Work

The report should connect three lines of work:

1. Korean hate/offensive language datasets and detection models.
   - Use K-HATERS as the primary dataset/task reference.
   - Optionally mention other Korean hate speech resources such as K-MHaS as
     broader task context.

2. Identity-term bias and unintended bias in toxicity classifiers.
   - Prior work shows that toxicity models can associate identity mentions with
     toxic labels even in neutral contexts.
   - This motivates evaluating both base task performance and identity
     counterfactual robustness.

3. Counterfactual fairness and counterfactual data augmentation.
   - Prior counterfactual fairness work motivates comparing original examples
     with identity-swapped variants.
   - Data augmentation alone is not enough for Korean because not every swap is
     semantically valid.

Positioning sentence:

> Unlike prior work that applies identity swaps broadly, this project focuses on
> the validity of Korean counterfactual pairs and studies how validity filtering
> changes the strength and quality of counterfactual consistency
> regularization.

## 3. Method

### 3.1 Base Model

We fine-tune `klue/roberta-base` for binary hate/offensive classification on
K-HATERS. The baseline uses standard cross-entropy loss.

### 3.2 Counterfactual Pair Construction

We detect identity terms using a curated Korean identity lexicon covering
categories such as gender, ethnicity, religion, age, sexuality, and disability.
For each eligible example, one identity term is replaced with a controlled
counterpart. The construction is one-directional because reverse mappings can be
ambiguous, especially when majority-group terms are counterparts for multiple
minority-group terms.

### 3.3 Validity Gate

The strict validity gate filters generated counterfactual pairs before they are
used for consistency regularization. The gate is designed to reject pairs that
are likely to be morphologically awkward, semantically changed, or dependent on
identity-specific context. This is a heuristic, not an oracle; therefore the
report includes qualitative error analysis.

### 3.4 Training Objective

For counterfactual methods, the model is trained with:

```text
L = L_cls + lambda * L_consistency
```

where `L_cls` is cross-entropy on the original labeled example and
`L_consistency` is a KL consistency loss between the prediction distribution for
the original input and the counterfactual input.

Methods:

- `Baseline`: no counterfactual consistency.
- `Naive Swap`: applies consistency to every generated identity swap with
  `lambda=0.1`.
- `Strict-Gated`: applies consistency only to strict-valid pairs with
  `lambda=0.1`.
- `Strict-Matched`: uses the same strict gate but scales lambda to compensate
  for lower valid-pair coverage. In the confirmed run, `lambda=0.1297`.
- `Strict_lam=0.15/0.25`: fixed-lambda sensitivity ablations.

## 4. Evaluation

Primary metrics:

- Macro-F1: standard hate/offensive detection performance.
- Pair Accuracy: percentage of original/counterfactual pairs where both
  predictions are correct.
- Strict Pair Accuracy: same as Pair Accuracy, restricted to strict-valid
  evaluation pairs.

Secondary metrics:

- Flip Rate: percentage of pairs where the predicted label changes.
- Prob Gap: average absolute probability difference across a pair.
- FPR Gap: subgroup false-positive gap. Use only as a secondary diagnostic due
  to small subgroup support in the confirmed run.

## 5. Confirmed Results

| Method | Lambda | Macro-F1 | Pair Acc | Strict Pair Acc | Flip Rate | Strict Flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.0000 | 0.7882+/-0.0026 | 0.8029+/-0.0102 | 0.8076+/-0.0128 | 0.0549+/-0.0112 | 0.0543+/-0.0123 |
| Naive Swap | 0.1000 | 0.7916+/-0.0017 | 0.8168+/-0.0037 | 0.8171+/-0.0023 | 0.0212+/-0.0041 | 0.0229+/-0.0040 |
| Strict-Gated | 0.1000 | 0.7907+/-0.0035 | 0.8220+/-0.0047 | 0.8248+/-0.0088 | 0.0234+/-0.0090 | 0.0219+/-0.0097 |
| Strict-Matched | 0.1297 | 0.7906+/-0.0051 | 0.8264+/-0.0038 | 0.8295+/-0.0135 | 0.0205+/-0.0046 | 0.0200+/-0.0103 |

The strongest confirmed method is `Strict-Matched`. It improves Strict Pair
Accuracy over Baseline by about `+0.0219` and over Naive Swap by about
`+0.0124`, while keeping Macro-F1 close to the best confirmed method.

## 6. Analysis

The results show three useful patterns.

First, all counterfactual methods preserve Macro-F1. This matters because a
fairness or robustness method is not useful if it substantially damages the
base hate/offensive classification task.

Second, Naive Swap strongly reduces flip rate and probability gaps. This is
expected because it receives broad counterfactual supervision from more pairs.
However, ungated swaps may include invalid counterfactuals, so lower flip rate
alone is not enough to claim a better method.

Third, Strict-Matched achieves the best pair-level correctness. This supports
the core hypothesis: when strict validity filtering reduces the number of
counterfactual pairs, coverage-aware lambda scaling can recover enough training
signal while preserving the semantic advantage of the gate.

Qualitative analysis should emphasize that remaining errors come from multiple
sources: true identity-term sensitivity, borderline decision-boundary examples,
annotation ambiguity, Korean morphology artifacts, and swaps that pass the
heuristic gate but still alter context.

## 7. Limitations

- The strict validity gate is heuristic and cannot guarantee semantic validity.
- The identity lexicon is manually curated and does not cover all groups or
  intersectional identities.
- Counterfactual swaps are one-directional, which avoids ambiguous reverse
  mappings but limits coverage.
- FPR Gap should remain secondary because subgroup support can be very small.
- The current experiments use one encoder family, `klue/roberta-base`.

## 8. Conclusion

The current evidence supports a positive but careful claim: validity-gated
counterfactual consistency can improve Korean hate/offensive detection
robustness when the lower coverage of valid counterfactual pairs is handled
explicitly. The strongest confirmed method, Strict-Matched, provides the best
pair-level correctness while preserving standard Macro-F1. The remaining
ablation work should test whether this improvement is robust to lambda changes
and whether the effect comes from validity filtering, regularization strength,
or both.
