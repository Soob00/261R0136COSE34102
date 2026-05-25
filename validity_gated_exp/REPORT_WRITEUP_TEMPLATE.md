# Report Writeup Template

Use this after `compare_results.py` prints the final `Claim assessment`.

## Title

Validity-Gated Counterfactual Consistency Regularization for Korean Hate Speech Detection

## Abstract Template

Replace the bracketed fields after the final run.

> We study whether identity-term counterfactuals can improve Korean hate speech detection without teaching models invalid invariances. We fine-tune KLUE-RoBERTa on K-HATERS and compare a standard classifier, ungated identity-swap consistency regularization, and validity-gated variants that filter generated counterfactuals using Korean-specific lexical, morphological, and semantic constraints. Across three seeds, `[MAIN_METHOD]` achieves `[F1]` Macro-F1 and `[STRICT_PAIR_ACC]` Strict Pair Accuracy, compared with `[NAIVE_STRICT_PAIR_ACC]` for ungated swaps. Our results support `[CLAIM_ASSESSMENT_LEVEL]`: `[ONE_SENTENCE_CLAIM]`. Error examples show `[QUALITATIVE_FINDING]`.

## Method Section Skeleton

1. Task and dataset
   - Binary hate/offensive detection on K-HATERS.
   - Report train/validation/test sizes and positive label rate.

2. Counterfactual construction
   - Detect one swappable identity term using the curated Korean identity lexicon.
   - Generate one-directional controlled swaps.
   - Keep one-directional swaps as a limitation because reverse mappings can be ambiguous.

3. Validity gates
   - Naive Swap: every generated swap is used for CCR.
   - Strict-Gated: only strict-valid pairs are used.
   - Strict-Matched: strict gate with coverage-matched lambda.
   - Strict_lam variants: fixed lambda sensitivity for strict gate.

4. Training objective
   - Cross-entropy for hate detection.
   - KL consistency between original prediction and counterfactual prediction for valid pairs.
   - Mention that original logits from the classification forward pass are reused as KL anchor.

5. Metrics
   - Macro-F1 for base task preservation.
   - Strict Pair Accuracy as primary counterfactual correctness metric.
   - Pair Accuracy, Flip Rate, Prob Gap as secondary metrics.
   - FPR Gap only as a secondary fairness diagnostic when group support is small.

## Results Table Placeholder

Use the `Markdown table` section from `compare_results.py`.

| Method | Macro-F1 | Pair Acc | Strict Pair Acc | Flip Rate | Strict Flip | Prob Gap | Strict Prob Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | | | | | | | |
| Naive Swap | | | | | | | |
| Strict-Gated | | | | | | | |
| Strict-Matched | | | | | | | |
| Strict_lam=0.15 | | | | | | | |
| Strict_lam=0.25 | | | | | | | |

## Claim Branches

### If `strong_gated`

Claim:

> Validity-gated CCR improves counterfactual robustness while preserving hate detection performance.

Use when:

- Best gated row matches or beats Naive Swap on Strict Pair Acc.
- Macro-F1 is within 0.01 of the reference.
- Error examples do not show obvious gate artifacts.

Discussion angle:

- Strict filtering removes invalid counterfactuals without losing useful consistency signal.
- Use construction analysis and examples to argue Korean-specific gating is meaningful.

### If `soft_consistency_tradeoff`

Claim:

> Ungated swaps produce stronger hard-label invariance, while validity-gated CCR offers a safer probability-stability trade-off.

Use when:

- Naive Swap has higher Strict Pair Acc.
- Best gated row preserves Macro-F1.
- Best gated row improves Strict Prob Gap over Naive.

Discussion angle:

- Naive receives more counterfactual signal, but some signal may be invalid.
- Gated method is conservative and useful when probability stability matters.

### If `validity_coverage_tradeoff`

Claim:

> Validity filtering exposes a coverage-validity trade-off in Korean counterfactual regularization.

Use when:

- Best gated row preserves Macro-F1.
- Naive Swap beats gated rows on hard pair metrics.
- Strict-Matched or TrainCF% indicates lower valid-CF coverage.

Discussion angle:

- The method is valuable as an analysis of when counterfactuals should be trusted.
- Do not claim gated superiority.
- Emphasize invalid-pair filtering, coverage statistics, and qualitative examples.

### If `diagnostic_only`

Claim:

> The current strict gate is too conservative or misaligned for a positive method claim.

Use when:

- Best gated row fails to preserve Macro-F1.
- Pair metrics do not compensate for base task degradation.

Discussion angle:

- Present this as a negative/diagnostic result.
- Focus on why identity-counterfactual regularization is hard in Korean.
- Use error examples and rejection breakdown to propose future gate improvements.

## Error Analysis Checklist

Pull examples from:

```bash
python validity_gated_exp/compare_results.py \
  validity_gated_exp/results_core_followup.json \
  --show_examples \
  --example_bucket both_wrong \
  --example_bucket strict_flip \
  --example_bucket false_positive_original \
  --max_examples 2
```

Write one paragraph each for:

- `both_wrong`: consistency without correctness.
- `strict_flip`: identity-sensitive instability even after strict validity filtering.
- `false_positive_original`: neutral identity mentions predicted as hate.

## Limitations

- The identity lexicon and swap map are manually curated and one-directional.
- Strict validity is a heuristic, not a semantic oracle.
- K-HATERS identity subgroup support may be too small for strong FPR Gap claims.
- Results use KLUE-RoBERTa-base; larger Korean LMs may behave differently.
- Counterfactuals are generated from available identity mentions and do not cover all protected groups or intersectional identities.

## Final Sanity Checklist

- `Report readiness audit` has no `FAIL`.
- All main rows come from the same commit, model, max length, and clean git state.
- Main table uses three seeds.
- The paper claim matches `Claim assessment`.
- Qualitative examples support the chosen claim rather than contradicting it.
