# Project Proposal

## Title

Validity-Gated Counterfactual Consistency Regularization for Korean
Hate/Offensive Language Detection

## Problem

Korean hate/offensive language classifiers can rely too heavily on identity
terms. A model may change its prediction when an identity term is swapped even
when the harmful intent should remain unchanged, or it may falsely flag neutral
identity mentions as hate. This project studies whether counterfactual
consistency regularization can reduce this identity sensitivity without hurting
the base classification task.

## Key Idea

Naive identity swapping is not always valid in Korean. Lexical replacement can
break morphology, create unnatural slang forms, or change the social meaning of
the sentence. We therefore compare ungated counterfactual consistency with a
validity-gated version that only regularizes on stricter, more defensible
counterfactual pairs.

## Research Question

Does validity-gated counterfactual consistency regularization improve
correctness-aware counterfactual robustness compared with both standard
fine-tuning and ungated identity-swap consistency?

## Dataset

- Training/evaluation task: K-HATERS binary hate/offensive detection.
- Model: `klue/roberta-base`.
- Counterfactual pairs: generated from identity mentions using a curated Korean
  identity swap map.
- Identity categories: gender, ethnicity, religion, age, sexuality, disability.

Generated result JSON/log files are not committed because they can be large and
environment-specific. Confirmed metrics are recorded in
`CONFIRMED_RESULTS_2026_05_24.md`.

## Methods

1. `Baseline`
   - Standard KLUE-RoBERTa fine-tuning.
   - Loss: cross-entropy only.

2. `Naive Swap`
   - Adds KL consistency loss for all generated identity swaps.
   - Lambda: `0.1`.

3. `Strict-Gated`
   - Adds KL consistency loss only for strict-valid counterfactual pairs.
   - Lambda: `0.1`.

4. `Strict-Matched`
   - Same strict gate, but lambda is scaled to compensate for lower valid-pair
     coverage.
   - Confirmed run lambda: `0.1297`.

5. `Strict_lam=0.15/0.25`
   - Fixed-lambda ablations for strict gating.
   - Purpose: test whether the strict result depends on regularization strength.

## Metrics

Primary:

- Macro-F1: preserves standard hate/offensive detection performance.
- Pair Accuracy: original and counterfactual predictions are both correct.
- Strict Pair Accuracy: Pair Accuracy on strict-valid evaluation pairs.

Secondary:

- Flip Rate: prediction changes across a pair.
- Prob Gap: probability difference across a pair.
- FPR Gap: subgroup false-positive diagnostic, not a main claim because current
  group support is small.

## Confirmed Result Summary

The strongest confirmed method is `Strict-Matched`.

| Method | Lambda | Macro-F1 | Pair Acc | Strict Pair Acc |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.0000 | 0.7882+/-0.0026 | 0.8029+/-0.0102 | 0.8076+/-0.0128 |
| Naive Swap | 0.1000 | 0.7916+/-0.0017 | 0.8168+/-0.0037 | 0.8171+/-0.0023 |
| Strict-Gated | 0.1000 | 0.7907+/-0.0035 | 0.8220+/-0.0047 | 0.8248+/-0.0088 |
| Strict-Matched | 0.1297 | 0.7906+/-0.0051 | 0.8264+/-0.0038 | 0.8295+/-0.0135 |

Interpretation:

> Strict-Matched preserves Macro-F1 and achieves the best pair-level
> correctness, suggesting that strict validity filtering is useful when its
> lower counterfactual coverage is compensated by coverage-aware regularization.

## Reference Papers To Discuss

- K-HATERS: use as the Korean hate/offensive task and dataset reference.
- Dixon et al., "Measuring and Mitigating Unintended Bias in Text
  Classification": use for identity-term bias motivation.
- Garg et al., "Counterfactual Fairness in Text Classification through
  Robustness": use for counterfactual consistency/fairness framing.
- Wadhwa et al., "Fairness for Text Classification Tasks with Identity
  Information Data Augmentation Methods": use for identity-based augmentation
  comparison.

Verify exact citation formatting before final submission.

## Expected Contribution

This project contributes a Korean-specific study of the validity-coverage
tradeoff in counterfactual consistency regularization. The novelty is not only
that identity counterfactuals are evaluated, but that their semantic validity is
modeled as part of the training signal.

## Feasibility

The core experiments have already produced usable three-seed results. Remaining
work is feasible within the report deadline:

1. finish `Strict_lam=0.15/0.25`;
2. rerun `compare_results.py` on all JSON files;
3. write error analysis from saved examples;
4. add a pipeline diagram and final result table to the report.
