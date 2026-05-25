"""
Dependency-light preflight for run_exp.py commands.

This script intentionally avoids torch/transformers imports, so it can validate
the planned experiment set locally before launching an expensive Jupyter/GPU run.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Any

from experiment_utils import CORE_REPORT_EXPERIMENTS, resolve_requested_experiments


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 3e-5
DEFAULT_LAMBDA = 0.1
DEFAULT_MAX_LEN = 128
DEFAULT_NUM_WORKERS = 4


def git_value(args: list[str], cwd: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return proc.stdout.strip()


def git_commit(cwd: Path) -> str | None:
    return git_value(["rev-parse", "--short", "HEAD"], cwd)


def git_dirty(cwd: Path) -> bool | None:
    status = git_value(["status", "--porcelain"], cwd)
    if status is None:
        return None
    return bool(status)


def result_path(base_dir: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return base_dir / "results_final.json"


def format_lam(value: float) -> str:
    return f"{value:g}"


def strict_family_present(planned: list[str]) -> bool:
    return any(
        name == "Strict-Gated"
        or name == "Strict-Matched"
        or name.startswith("Strict_lam=")
        for name in planned
    )


def build_preflight_report(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    notes: list[str] = []

    try:
        base_experiments, strict_lambdas = resolve_requested_experiments(args.exp)
    except ValueError as exc:
        return {
            "failures": [str(exc)],
            "warnings": [],
            "notes": [],
            "planned": [],
            "result_path": result_path(Path(args.base_dir), args.result_path),
            "commit": git_commit(repo_root),
            "dirty": git_dirty(repo_root),
        }

    planned = list(base_experiments) + [f"Strict_lam={format_lam(lam)}" for lam in strict_lambdas]
    seeds = args.seeds or DEFAULT_SEEDS
    out_path = result_path(Path(args.base_dir), args.result_path)
    commit = git_commit(repo_root)
    dirty = git_dirty(repo_root)

    if args.expected_commit and commit != args.expected_commit:
        failures.append(f"Current commit {commit} does not match expected {args.expected_commit}.")
    if args.require_clean and dirty:
        failures.append("Git worktree is dirty; commit or stash changes before a report run.")
    if args.fresh_result_path and out_path.exists():
        failures.append(f"Result path already exists: {out_path}")
    elif out_path.exists():
        warnings.append(f"Result path already exists and may merge/rename rows: {out_path}")

    missing_core = [name for name in CORE_REPORT_EXPERIMENTS if name not in planned]
    if missing_core and args.require_core:
        failures.append(f"Missing core report experiments: {', '.join(missing_core)}.")
    elif missing_core:
        warnings.append(f"Missing core report experiments: {', '.join(missing_core)}.")

    if "Naive Swap" in planned and not strict_family_present(planned):
        warnings.append("Naive Swap is planned without any strict-family comparison.")
    if "Strict-Gated" in planned and "Strict-Matched" not in planned and not strict_lambdas:
        warnings.append("Strict-Gated is planned without Strict-Matched or Strict_lam follow-up.")
    if len(seeds) < 3:
        warnings.append(f"Only {len(seeds)} seed(s) requested; final report table should use at least 3.")
    if args.epochs < 3:
        warnings.append(f"Only {args.epochs} epoch(s) requested; use 3 for report runs unless debugging.")

    if args.batch_size <= 0:
        failures.append("batch_size must be positive.")
    if args.epochs <= 0:
        failures.append("epochs must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        failures.append("lr must be positive.")
    if not math.isfinite(args.lambda_value) or args.lambda_value <= 0:
        failures.append("lambda must be positive for consistency-regularized runs.")
    if args.max_len <= 0:
        failures.append("max_len must be positive.")
    if args.num_workers < 0:
        failures.append("num_workers cannot be negative.")

    total_model_fits = len(planned) * len(seeds)
    total_train_epochs = total_model_fits * args.epochs
    if "Strict-Matched" in planned:
        notes.append("Strict-Matched lambda is computed after CF construction from strict-valid coverage.")
    notes.append(f"Planned model fits: {total_model_fits} ({len(planned)} experiments x {len(seeds)} seeds).")
    notes.append(f"Planned train epochs: {total_train_epochs}.")
    if args.subset:
        notes.append(f"Subset run requested: train subset={args.subset}; do not use as final report evidence.")

    return {
        "failures": failures,
        "warnings": warnings,
        "notes": notes,
        "planned": planned,
        "result_path": out_path,
        "commit": commit,
        "dirty": dirty,
        "seeds": seeds,
    }


def print_report(report: dict[str, Any]) -> None:
    print("Preflight run plan")
    print("------------------")
    print(f"Git commit: {report.get('commit')} dirty={report.get('dirty')}")
    print(f"Result path: {report.get('result_path')}")
    print("Experiments:")
    for name in report.get("planned", []):
        print(f"- {name}")
    if report.get("seeds"):
        print(f"Seeds: {report['seeds']}")

    for note in report.get("notes", []):
        print(f"NOTE: {note}")
    for warning in report.get("warnings", []):
        print(f"WARN: {warning}")
    for failure in report.get("failures", []):
        print(f"FAIL: {failure}")

    if report.get("failures"):
        print("PREFLIGHT FAIL")
    else:
        print("PREFLIGHT PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--lambda_", "--lambda", dest="lambda_value", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--base_dir", default=str(SCRIPT_DIR))
    parser.add_argument("--result_path", default=None)
    parser.add_argument("--expected_commit", default=None)
    parser.add_argument("--require_clean", action="store_true")
    parser.add_argument("--require_core", action="store_true")
    parser.add_argument("--fresh_result_path", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = SCRIPT_DIR.parent
    report = build_preflight_report(args, repo_root)
    print_report(report)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
