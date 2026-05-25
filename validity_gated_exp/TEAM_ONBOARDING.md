# Team Onboarding: Validity-Gated CCR

This project studies identity robustness in Korean hate/offensive language
detection. The working method is **Validity-Gated Counterfactual Consistency
Regularization**.

## Research Question

Can a Korean hate/offensive language model become less sensitive to identity
term swaps while preserving normal detection performance?

The important distinction is not just whether counterfactual consistency helps.
The project asks whether **semantically valid** counterfactual consistency is a
better and more defensible training signal than naive identity swapping.

## Why We Moved Away From Synthetic Evaluation

The earlier `eval_final.json` synthetic setup is not the main evidence path.
It can be useful for debugging, but it is weak as final paper evidence because:

- the test construction itself can dominate the result;
- synthetic pairs make external validity harder to defend;
- a strong result can look like overfitting to the generated template rather
  than improving Korean hate detection robustness.

The current direction uses real Korean hate/offensive data and generated
counterfactual pairs as training/evaluation probes, with explicit validity
checks and error analysis.

## Core Methods

- `Baseline`: KLUE-RoBERTa fine-tuning without counterfactual consistency.
- `Naive Swap`: consistency regularization on every generated identity swap.
- `Strict-Gated`: consistency regularization only on strict-valid swaps.
- `Strict-Matched`: Strict-Gated with lambda scaled to compensate for lower
  strict-valid coverage.
- `Strict_lam=0.15` and `Strict_lam=0.25`: lambda sensitivity for strict gating.

`Naive Swap` is expected to be a strong baseline. It can improve invariance
because it uses more counterfactual pairs, but some pairs may be semantically
invalid. The paper should not claim that Naive is conceptually clean just
because it has strong Pair Accuracy.

## Report-Grade Command

```bash
python validity_gated_exp/run_exp.py \
  --exp Baseline "Naive Swap" Strict-Gated Strict-Matched Strict_lam=0.15 Strict_lam=0.25 \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_core_followup.json \
  2>&1 | tee train_core_followup.log
```

If the run is interrupted and the code version includes resume support, rerun
the same command with:

```bash
--resume_completed
```

Resume only skips saved rows when the method config and seed evidence match.
Otherwise it retrains.

## Environment Recovery

If the notebook already has CUDA torch installed but crashes on packages such as
`datasets`, `sklearn`, or `kiwipiepy`, install the non-torch runtime deps:

```bash
python -m pip install datasets transformers kiwipiepy scipy tqdm numpy scikit-learn
```

On code versions that include `requirements-runtime.txt`, this equivalent
command is preferred:

```bash
python -m pip install -r validity_gated_exp/requirements-runtime.txt
```

Then verify:

```bash
python -c "import torch, transformers, datasets, kiwipiepy, sklearn, scipy, tqdm, numpy; print('deps ok'); print(torch.cuda.is_available())"
```

## How To Read Results

After the run:

```bash
python validity_gated_exp/compare_results.py \
  validity_gated_exp/results_core_followup.json \
  --show_examples \
  --example_bucket both_wrong \
  --example_bucket strict_flip \
  --example_bucket false_positive_original \
  --max_examples 2
```

Read sections in this order:

1. `Report readiness audit`: no `FAIL` before using numbers in the report.
2. `Macro-F1`: counterfactual methods should not substantially harm detection.
3. `Pair Accuracy` and `Strict Pair Acc`: primary robustness metrics.
4. `TrainCF%`, `ConsBatch%`, `ValidCF/B`: explain strict-gate coverage.
5. `Claim assessment`: use this to set the paper claim strength.
6. qualitative examples: check whether low flip rate hides consistently wrong
   predictions.

## Claim Decision Tree

### Strong gated result

Use this if a strict-family method, preferably `Strict-Matched`, is close to or
better than `Naive Swap` on Pair Accuracy / Strict Pair Accuracy while preserving
Macro-F1.

Possible claim:

> Validity-gated counterfactual consistency improves identity robustness while
> preserving Korean hate/offensive detection performance; coverage-matched
> regularization mitigates the sparsity cost of stricter validity filtering.

### Soft consistency tradeoff

Use this if `Naive Swap` has the best Pair Accuracy but strict-family methods
still improve over `Baseline` and keep Macro-F1 stable.

Possible claim:

> Naive identity-swap consistency gives strong hard invariance, but
> validity-gated consistency is a more semantically controlled alternative. The
> central tradeoff is counterfactual coverage versus semantic validity.

### Validity coverage tradeoff

Use this if strict methods are weaker mainly because they see fewer valid
counterfactuals, and `Strict-Matched` or higher lambda narrows the gap.

Possible claim:

> Strict validity gates reduce noisy counterfactual supervision, but their
> benefit depends on enough valid pair coverage or coverage-aware weighting.

### Diagnostic-only result

Use this if strict-family methods fail to improve over `Baseline` or damage F1.

Possible claim:

> In this Korean hate/offensive setting, strict semantic gating exposes a
> coverage bottleneck: validity filtering alone is not sufficient unless paired
> with better counterfactual generation or stronger coverage-aware objectives.

## Current Local Code Status

The local repository includes improvements that may not yet be on GitHub:

- runtime environment checker;
- torch-free runtime requirements;
- fail-fast dependency diagnostics in `run_exp.py`;
- `Strict-Matched` and arbitrary `Strict_lam=<value>` support;
- report readiness and claim classification in `compare_results.py`;
- qualitative error examples;
- compatible partial-run resume with `--resume_completed`.

Before relying on Jupyter results, confirm which commit was used:

```bash
git rev-parse --short HEAD
git status --short --branch
```

If the Jupyter clone is behind the local version, the numeric results may still
be usable if the method code is equivalent, but newer convenience features such
as `env_check.py`, `requirements-runtime.txt`, and `--resume_completed` may be
missing.

## Current Stop Point

Do not change the method again while the report-grade run is in progress. The
next decision should come from the actual `results_core_followup.json` and
`compare_results.py` output.
