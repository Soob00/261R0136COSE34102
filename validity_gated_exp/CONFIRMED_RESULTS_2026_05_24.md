# Confirmed Results Snapshot: 2026-05-24

This file records the currently confirmed report-grade results shared from the
Jupyter run. Generated JSON/log artifacts are intentionally git-ignored, so this
document is the version-controlled evidence summary for team coordination.

## Run Metadata

- Result file on Jupyter: `validity_gated_exp/results_core_followup.json`
- Save stage when first checked: `after Strict-Gated`
- Confirmed later: `Strict-Matched` completed
- Code commit reported by result metadata: `abebc67`
- Gate version: `2026-05-24-strict-context-v2`
- Model: `klue/roberta-base`
- Seeds: `42`, `123`, `456`
- Epochs: `3`
- Batch size: `64`
- Core lambda: `0.1`

Important caveat: this snapshot is based on the user's Jupyter output. Before
final submission, rerun `compare_results.py` on the final JSON and check that
`_meta.is_final=true` if the full command completed.

## Confirmed Main Results

| Method | Lambda | Macro-F1 | Pair Acc | Strict Pair Acc | Flip Rate | Strict Flip | Prob Gap | Strict Prob Gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.0000 | 0.7882+/-0.0026 | 0.8029+/-0.0102 | 0.8076+/-0.0128 | 0.0549+/-0.0112 | 0.0543+/-0.0123 | 0.0440+/-0.0049 | 0.0447+/-0.0034 |
| Naive Swap | 0.1000 | 0.7916+/-0.0017 | 0.8168+/-0.0037 | 0.8171+/-0.0023 | 0.0212+/-0.0041 | 0.0229+/-0.0040 | 0.0168+/-0.0006 | 0.0165+/-0.0009 |
| Strict-Gated | 0.1000 | 0.7907+/-0.0035 | 0.8220+/-0.0047 | 0.8248+/-0.0088 | 0.0234+/-0.0090 | 0.0219+/-0.0097 | 0.0198+/-0.0012 | 0.0190+/-0.0008 |
| Strict-Matched | 0.1297 | 0.7906+/-0.0051 | 0.8264+/-0.0038 | 0.8295+/-0.0135 | 0.0205+/-0.0046 | 0.0200+/-0.0103 | pending | pending |

`Strict-Matched` seed-level values:

- Macro-F1: `[0.790311173895808, 0.7958660779404328, 0.7857313799387539]`
- Pair Accuracy: `[0.8307692307692308, 0.8241758241758241, 0.8241758241758241]`
- Strict Pair Accuracy: `[0.8342857142857143, 0.8142857142857143, 0.84]`
- Flip Rate: `[0.024175824175824177, 0.02197802197802198, 0.015384615384615385]`
- Strict Flip Rate: `[0.02857142857142857, 0.022857142857142857, 0.008571428571428572]`

## Deltas Against Baseline

| Method | Delta Macro-F1 | Delta Pair Acc | Delta Strict Pair Acc | Delta Flip Rate | Delta Strict Flip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive Swap | +0.0034 | +0.0139 | +0.0095 | -0.0337 | -0.0314 |
| Strict-Gated | +0.0025 | +0.0191 | +0.0172 | -0.0315 | -0.0324 |
| Strict-Matched | +0.0024 | +0.0235 | +0.0219 | -0.0344 | -0.0343 |

## Interpretation

The strongest current result is `Strict-Matched`. It preserves standard
classification quality while giving the best pair-level correctness:

- Macro-F1 remains close to Naive Swap and above Baseline.
- Pair Accuracy is the best among confirmed methods.
- Strict Pair Accuracy is also the best among confirmed methods.
- Flip rates are reduced substantially relative to Baseline and are comparable
  to Naive Swap.

This supports the paper claim that validity-gated counterfactual consistency is
most effective when the lower coverage of strict valid pairs is compensated by a
coverage-matched regularization weight.

## What Not To Overclaim

- Do not claim that lower flip rate alone proves fairness. A model can be
  consistently wrong on both the original and counterfactual example.
- Do not use FPR Gap as a main claim. The available result metadata reported
  very low minimum group support (`minN=3`), so FPR Gap should remain a
  secondary diagnostic.
- Do not describe `Strict-Matched` lambda as tuned on the test set. It is a
  coverage-matched lambda derived from the relative amount of valid
  counterfactual supervision.

## Current Main Claim

Use this wording unless later lambda sweeps contradict it:

> Coverage-matched validity-gated counterfactual consistency regularization
> improves counterfactual pair correctness for Korean hate/offensive language
> detection while preserving Macro-F1. Compared with ungated identity swaps, the
> method uses a more semantically controlled counterfactual signal and achieves
> higher Strict Pair Accuracy.

## Pending Results

The following are ablations, not required for the core positive claim:

- `Strict_lam=0.15`
- `Strict_lam=0.25`

When they finish, update this file or add a new dated snapshot before writing
the final result table.
