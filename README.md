# nlp-korean

Project repo for **Validity-Gated Counterfactual Consistency Regularization** in Korean hate/offensive language detection.

Use this repo only for the current project work. Do not continue from the older `Soob00/hi` clone.

## What Is Here

- `validity_gated_exp/run_exp.py`: main experiment runner.
- `validity_gated_exp/dataset.py`: K-HATERS loading, identity swaps, validity gates.
- `validity_gated_exp/check_data.py`: data and counterfactual pair sanity check.
- `validity_gated_exp/analyze_cf_pairs.py`: construction statistics for the report.
- `validity_gated_exp/RUNNING.md`: step-by-step runbook.
- `validity_gated_exp/PROJECT_DIRECTION.md`: critical project direction notes.
- `validity_gated_exp/CONFIRMED_RESULTS_2026_05_24.md`: current confirmed result snapshot.
- `validity_gated_exp/PROJECT_PROPOSAL.md`: concise proposal for the report topic.
- `validity_gated_exp/PAPER_DRAFT.md`: report/proposal draft for the current topic.
- `validity_gated_exp/ABLATION_PLAN.md`: remaining lambda ablations and interpretation guide.
- `results/`: shared report-grade JSON outputs, summaries, and final tables.

Generated files such as checkpoints, logs, temporary JSON results, CSV summaries, and generated counterfactual pairs under `validity_gated_exp/` are intentionally ignored by git. Copy final shareable result JSON files into `results/raw/`.

## Fresh Server Setup

On the Jupyter/Ubuntu server:

```bash
cd ~
git clone https://github.com/angellashin/nlp-korean.git
cd nlp-korean
```

Before installing PyTorch, check disk space:

```bash
df -h
```

If the root disk is almost full, clear broken old installs/caches first:

```bash
rm -rf ~/.cache/pip
rm -rf ~/.cache/huggingface
```

Create a clean venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Install PyTorch first, then the rest:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r validity_gated_exp/requirements.txt
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
python validity_gated_exp/run_exp.py --help
```

## Naive Swap First Run

Smoke test:

```bash
python validity_gated_exp/run_exp.py \
  --exp "Naive Swap" \
  --seeds 42 \
  --subset 512 \
  --epochs 1 \
  --batch_size 8 \
  --num_workers 0 \
  --result_path validity_gated_exp/results_naive_smoke.json \
  2>&1 | tee train_naive_smoke.log
```

Full Naive Swap run:

```bash
python validity_gated_exp/run_exp.py \
  --exp "Naive Swap" \
  --seeds 42 123 456 \
  --epochs 3 \
  --batch_size 64 \
  --num_workers 2 \
  --result_path validity_gated_exp/results_naive.json \
  2>&1 | tee train_naive.log
```

If CUDA runs out of memory, retry with `--batch_size 32`, then `16`.

After any run, compare JSON result files:

```bash
python validity_gated_exp/compare_results.py \
  validity_gated_exp/results_naive.json \
  validity_gated_exp/results_core_followup.json
```
