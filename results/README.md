# Results Folder

Use this folder to share report-grade experiment outputs across teammates.

Generated result files under `validity_gated_exp/` are ignored by git, so copy
the final JSON and summary artifacts here when they are ready to share.

## Folder Layout

```text
results/
  README.md
  raw/          # JSON result files from run_exp.py
  summaries/    # compare_results.py text outputs and qualitative examples
  tables/       # cleaned markdown/CSV tables for the report
  logs/         # optional notes about training logs; *.log files are ignored
```

## Required Raw JSON Files

Put the final JSON files here with these exact names:

```text
results/raw/results_core_followup.json
results/raw/results_masking_cons_reg.json
results/raw/results_strict_lam_005.json
results/raw/results_strict_lam_015.json
results/raw/results_strict_lam_025.json
```

Expected contents:

| File | Expected rows |
| --- | --- |
| `results_core_followup.json` | `Baseline`, `Naive Swap`, `Strict-Gated`, `Strict-Matched` |
| `results_masking_cons_reg.json` | `Masking Cons Reg` |
| `results_strict_lam_005.json` | `Strict_lam=0.05` |
| `results_strict_lam_015.json` | `Strict_lam=0.15` |
| `results_strict_lam_025.json` | `Strict_lam=0.25` |

If one run file also contains extra rows, keep it, but do not overwrite another
teammate's JSON with a different result path.

## Required Summary Files

After all JSON files are collected, generate:

```text
results/summaries/compare_all_methods.txt
results/tables/final_results_table.md
```

Use:

```bash
python validity_gated_exp/compare_results.py \
  results/raw/results_core_followup.json \
  results/raw/results_masking_cons_reg.json \
  results/raw/results_strict_lam_005.json \
  results/raw/results_strict_lam_015.json \
  results/raw/results_strict_lam_025.json \
  --show_examples \
  --example_bucket strict_flip \
  --example_bucket both_wrong \
  --example_bucket false_positive_original \
  --max_examples 3 \
  2>&1 | tee results/summaries/compare_all_methods.txt
```

## Quick Integrity Check

Run this before writing the report table:

```bash
python - <<'PY'
import json
from pathlib import Path

paths = [
    "results/raw/results_core_followup.json",
    "results/raw/results_masking_cons_reg.json",
    "results/raw/results_strict_lam_005.json",
    "results/raw/results_strict_lam_015.json",
    "results/raw/results_strict_lam_025.json",
]

for p in paths:
    path = Path(p)
    print(f"\n=== {p} ===")
    if not path.exists():
        print("MISSING")
        continue
    d = json.load(open(path, encoding="utf-8"))
    m = d.get("_meta", {})
    print("is_final:", m.get("is_final"))
    print("save_stage:", m.get("save_stage"))
    print("completed:", m.get("completed_experiments"))
    for name, result in d.items():
        if name == "_meta":
            continue
        cfg = result.get("config", {})
        print(f"- {name}: mode={cfg.get('mode')} lambda={cfg.get('lambda')}")
PY
```

## Naming Rules

- Use lowercase lambda filenames: `005`, `015`, `025`.
- Keep raw files as `.json`.
- Keep human-readable summaries as `.txt` or `.md`.
- Do not commit model checkpoints or Hugging Face caches.
- Training `.log` files are ignored by git; if a log excerpt is needed, copy
  the relevant lines into `results/logs/<run_name>_notes.md`.

## Recommended Report Interpretation

Use `Macro-F1` to verify base-task performance is preserved. Use `Strict Pair
Acc` as the main counterfactual robustness metric. Use `Flip Rate` and `Prob
Gap` as supporting metrics. Keep `FPR Gap` secondary because current group
support is small.
