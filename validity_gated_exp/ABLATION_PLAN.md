# Ablation Plan and Interpretation Guide

This file explains why the remaining ablations matter and how to interpret them
without overclaiming.

## Current Confirmed Method Ladder

The confirmed ladder is:

1. `Baseline`: no counterfactual consistency.
2. `Naive Swap`: broad ungated counterfactual consistency, `lambda=0.1`.
3. `Strict-Gated`: strict-valid pairs only, `lambda=0.1`.
4. `Strict-Matched`: strict-valid pairs only, coverage-matched
   `lambda=0.1297`.

The current main candidate is `Strict-Matched` because it has the best Pair
Accuracy and Strict Pair Accuracy while preserving Macro-F1.

## Why Lambda Ablation Is Needed

Strict gating filters out invalid pairs, so it sees fewer counterfactual
training examples than Naive Swap. If `Strict-Gated` underperforms, the reason
could be:

- the gate removed noisy/invalid pairs and improved supervision quality;
- the gate removed too many useful pairs and weakened the consistency signal;
- the chosen lambda was too small or too large for the lower-coverage setting.

Lambda ablation separates these explanations.

## Pending Fixed-Lambda Runs

Team members should run these independently with separate result paths:

```bash
python validity_gated_exp/run_exp.py \
  --exp Strict_lam=0.15 \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 1 \
  --result_path validity_gated_exp/results_strict_lam_015.json \
  2>&1 | tee train_strict_lam_015.log
```

```bash
python validity_gated_exp/run_exp.py \
  --exp Strict_lam=0.25 \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 1 \
  --result_path validity_gated_exp/results_strict_lam_025.json \
  2>&1 | tee train_strict_lam_025.log
```

Never write parallel runs to the same `result_path`.

## How To Compare

After the JSON files are available:

```bash
python validity_gated_exp/compare_results.py \
  validity_gated_exp/results_core_followup.json \
  validity_gated_exp/results_strict_lam_015.json \
  validity_gated_exp/results_strict_lam_025.json \
  --show_examples \
  --example_bucket strict_flip \
  --example_bucket both_wrong \
  --max_examples 3
```

## Interpretation Matrix

### Case 1: Strict-Matched remains best

Use this as the cleanest result:

> Coverage-matched strict gating gives the best balance between semantic
> validity and consistency strength.

The fixed lambda runs become robustness checks showing that the matched value is
not an arbitrary hand-picked setting.

### Case 2: Strict_lam=0.15 beats Strict-Matched

This is still good. It means the matched lambda was slightly conservative, and
the strict method benefits from a little more consistency pressure. The report
can use `Strict_lam=0.15` as the best strict row, but should state clearly that
lambda sensitivity was tested.

Preferred wording:

> Strict gating benefits from moderately stronger regularization than the
> ungated baseline because the gate supplies fewer but cleaner counterfactual
> pairs.

### Case 3: Strict_lam=0.25 improves Pair Acc but hurts Macro-F1

This shows a tradeoff. Do not use `0.25` as the headline method unless the F1
drop is small and justified.

Preferred wording:

> Strong consistency pressure can improve pair robustness but begins to compete
> with the original classification objective.

### Case 4: Strict_lam=0.25 is worse on both F1 and Pair Acc

This supports using `Strict-Matched` or `0.15` as a safer setting.

Preferred wording:

> Excessive regularization over even valid pairs can over-constrain the model.

### Case 5: Naive remains best on soft metrics only

This is not a failure. Naive often has lower probability gaps because it sees
more counterfactual pairs. The paper should distinguish:

- soft consistency: Flip Rate and Prob Gap;
- correctness-aware consistency: Pair Accuracy and Strict Pair Accuracy.

If Naive has lower Prob Gap but Strict has higher Strict Pair Accuracy, the
claim should be:

> Ungated swaps produce stronger soft invariance, while validity-gated CCR
> improves correctness-aware pair robustness.

## Additional Ablations If Time Allows

Only run these after the main report text is stable:

1. `Strict_lam=0.05`
   - Tests whether weaker strict regularization still improves over Baseline.
2. `Strict_lam=0.20`
   - Fills the gap between `0.15` and `0.25`.
3. Category-level analysis
   - Break down strict flips by `gender`, `ethnicity`, `sexuality`, `religion`,
     `age`, and `disability`.
4. Gate rejection analysis
   - Report how many generated pairs are rejected and why.

Do not expand the method search unless the current confirmed result collapses.
The report deadline makes error analysis and clear writing more valuable than a
large hyperparameter sweep.
