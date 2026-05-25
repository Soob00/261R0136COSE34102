"""
Compare experiment result JSON files produced by run_exp.py.

This script is intentionally dependency-light: it uses only the Python standard
library so it can run locally even when the training environment is not set up.

Usage:
    python validity_gated_exp/compare_results.py validity_gated_exp/results_core.json
    python validity_gated_exp/compare_results.py results_naive.json results_strict_lam02.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from experiment_utils import ERROR_EXAMPLE_BUCKETS, unique_result_name


PRIMARY_METRICS = [
    ("f1", "F1", "higher"),
    ("pair_accuracy", "PairAcc", "higher"),
    ("strict_pair_accuracy", "S-PairAcc", "higher"),
    ("flip_rate", "Flip", "lower"),
    ("strict_flip_rate", "S-Flip", "lower"),
    ("prob_gap", "ProbGap", "lower"),
    ("strict_prob_gap", "S-ProbGap", "lower"),
    ("fpr_gap", "FPRGap", "lower"),
    ("fpr_min_group_n", "FPR minN", "higher"),
    ("train_valid_cf_ratio", "TrainCF%", "higher"),
    ("cons_batch_ratio", "ConsBatch%", "higher"),
    ("avg_valid_cf_per_batch", "ValidCF/B", "higher"),
]

CORE_METHODS = ("Baseline", "Naive Swap", "Strict-Gated")
REPORT_REQUIRED_METRICS = (
    "f1",
    "pair_accuracy",
    "strict_pair_accuracy",
    "flip_rate",
    "strict_flip_rate",
    "prob_gap",
    "strict_prob_gap",
)
REPORT_CONFIG_KEYS = ("git_commit", "gate_version", "model", "max_len", "epochs", "batch_size", "lr")
F1_TOLERANCE = 0.01


def load_results(paths: list[Path]) -> dict[str, dict[str, Any]]:
    results, _ = load_results_with_metadata(paths)
    return results


def duplicate_result_name(name: str, metrics: dict[str, Any], path: Path, existing: set[str]) -> str:
    config = metrics.get("config")
    lam = config.get("lambda") if isinstance(config, dict) else None
    return unique_result_name(name, existing, lambda_value=lam, source=path.stem)


def load_results_with_metadata(paths: list[Path]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    merged: dict[str, dict[str, Any]] = {}
    metadata: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("_meta")
        if isinstance(meta, dict):
            metadata.append({"path": str(path), **meta})
        else:
            metadata.append({"path": str(path), "missing_meta": True})
        for name, metrics in data.items():
            if isinstance(metrics, dict) and "f1" in metrics:
                result_name = name
                duplicate_of = None
                if result_name in merged:
                    duplicate_of = name
                    result_name = duplicate_result_name(name, metrics, path, set(merged))
                merged[result_name] = {
                    **metrics,
                    "_source_path": str(path),
                    "_original_name": duplicate_of or name,
                    "_renamed_duplicate": duplicate_of is not None,
                }
    return merged, metadata


def fmt(values: Any, scale: float = 1.0) -> str:
    if not isinstance(values, list) or not values:
        return "N/A"
    vals = [v * scale for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not vals:
        return "N/A"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{mean(vals):.4f}±{pstdev(vals):.4f}"


def mean_or_none(values: Any) -> float | None:
    if not isinstance(values, list) or not values:
        return None
    vals = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return mean(vals) if vals else None


def fmt_num(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def delta_str(base: float | None, cur: float | None, direction: str) -> str:
    if base is None or cur is None:
        return "N/A"
    delta = cur - base
    good = delta > 0 if direction == "higher" else delta < 0
    marker = "+" if good else "-"
    return f"{delta:+.4f} {marker}"


def is_strict_family(name: str) -> bool:
    return (
        name == "Strict-Gated"
        or name == "Strict-Matched"
        or name.startswith("Strict_lam=")
        or name.startswith("Strict-Gated [")
        or name.startswith("Strict-Matched [")
    )


def best_variant_by(results: dict[str, dict[str, Any]], names: list[str], metric: str) -> tuple[str, float] | None:
    scored = []
    for name in names:
        value = mean_or_none(results[name].get(metric))
        if value is not None:
            scored.append((name, value))
    if not scored:
        return None
    return max(scored, key=lambda x: x[1])


def assess_claim(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return a structured paper-claim recommendation from the current results."""
    missing = [name for name in CORE_METHODS if name not in results]
    if missing:
        return {
            "level": "incomplete",
            "main_method": None,
            "headline": "Run the missing core methods before writing a result claim.",
            "rationale": [f"Missing core methods: {', '.join(missing)}."],
            "report_action": "Do not use this result set as the final main table.",
        }

    strict_names = [name for name in results if is_strict_family(name)]
    best_strict = best_variant_by(results, strict_names, "strict_pair_accuracy")
    if not best_strict:
        return {
            "level": "incomplete",
            "main_method": None,
            "headline": "No strict-family result has Strict PairAcc.",
            "rationale": ["Run Strict-Gated, Strict-Matched, or Strict_lam=* with current metrics."],
            "report_action": "Do not claim validity-gated behavior yet.",
        }

    best_name, best_sp = best_strict
    baseline_f1 = mean_or_none(results["Baseline"].get("f1"))
    naive_f1 = mean_or_none(results["Naive Swap"].get("f1"))
    best_f1 = mean_or_none(results[best_name].get("f1"))
    naive_sp = mean_or_none(results["Naive Swap"].get("strict_pair_accuracy"))
    naive_gap = mean_or_none(results["Naive Swap"].get("strict_prob_gap"))
    best_gap = mean_or_none(results[best_name].get("strict_prob_gap"))
    baseline_ref = baseline_f1 if baseline_f1 is not None else naive_f1

    if best_f1 is None or naive_sp is None or baseline_ref is None:
        return {
            "level": "incomplete",
            "main_method": best_name,
            "headline": "The best gated row is present but key F1/PairAcc metrics are missing.",
            "rationale": [
                f"Best gated row: {best_name}.",
                "Need Macro-F1, Naive Strict PairAcc, and gated Strict PairAcc.",
            ],
            "report_action": "Rerun or repair the result file before choosing a claim.",
        }

    f1_preserved = best_f1 >= baseline_ref - F1_TOLERANCE
    beats_naive_pair = best_sp >= naive_sp
    improves_prob_gap = best_gap is not None and naive_gap is not None and best_gap <= naive_gap
    has_examples = bool(results[best_name].get("fairness_error_examples"))

    rationale = [
        f"Best gated row: {best_name}.",
        f"Macro-F1: {best_f1:.4f} vs reference {baseline_ref:.4f} (tolerance {F1_TOLERANCE:.2f}).",
        f"Strict PairAcc: gated {best_sp:.4f} vs Naive {naive_sp:.4f}.",
    ]
    if best_gap is not None and naive_gap is not None:
        rationale.append(f"Strict ProbGap: gated {best_gap:.4f} vs Naive {naive_gap:.4f}.")
    if not has_examples:
        rationale.append("Qualitative error examples are missing for the best gated row.")

    if beats_naive_pair and f1_preserved:
        level = "strong_gated"
        headline = "Validity-gated CCR is the main result."
        report_action = (
            "Use the best gated row as the primary method: it preserves Macro-F1 "
            "and matches or beats Naive on Strict PairAcc."
        )
    elif f1_preserved and improves_prob_gap:
        level = "soft_consistency_tradeoff"
        headline = "Use a tradeoff claim: Naive wins hard PairAcc, gated improves soft consistency."
        report_action = (
            "Report Naive as the strongest hard-invariance baseline and the best gated row "
            "as a validity-filtered method with probability-stability benefits."
        )
    elif f1_preserved:
        level = "validity_coverage_tradeoff"
        headline = "Use a validity-coverage tradeoff claim."
        report_action = (
            "Do not claim gated superiority. Emphasize that filtering invalid CFs is meaningful, "
            "but reduced CF coverage can make Naive Swap stronger on hard pair metrics."
        )
    else:
        level = "diagnostic_only"
        headline = "Do not make the gated method the main positive result yet."
        report_action = (
            "Treat this as a diagnostic study unless another gated variant preserves F1. "
            "Inspect lambda, coverage, and error examples before more training."
        )

    return {
        "level": level,
        "main_method": best_name,
        "headline": headline,
        "rationale": rationale,
        "report_action": report_action,
    }


def paired_delta(values_a: Any, values_b: Any) -> list[float]:
    if not isinstance(values_a, list) or not isinstance(values_b, list):
        return []
    if len(values_a) != len(values_b) or not values_a:
        return []
    deltas = []
    for a, b in zip(values_a, values_b):
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return []
        if math.isnan(a) or math.isnan(b):
            return []
        deltas.append(float(a) - float(b))
    return deltas


def metric_seed_count(values: Any) -> int:
    if not isinstance(values, list):
        return 0
    valid = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return len(valid)


def iter_error_examples(metrics: dict[str, Any], bucket: str):
    """Yield saved qualitative examples for one bucket across seeds."""
    saved = metrics.get("fairness_error_examples")
    if not isinstance(saved, list):
        return
    for seed_entry in saved:
        if not isinstance(seed_entry, dict):
            continue
        seed = seed_entry.get("seed")
        examples_by_bucket = seed_entry.get("examples")
        if not isinstance(examples_by_bucket, dict):
            continue
        examples = examples_by_bucket.get(bucket, [])
        if not isinstance(examples, list):
            continue
        for example in examples:
            if isinstance(example, dict):
                yield seed, example


def bucket_example_count(metrics: dict[str, Any], bucket: str) -> int:
    return sum(1 for _ in iter_error_examples(metrics, bucket))


def shorten(text: Any, max_chars: int = 90) -> str:
    if text is None:
        return ""
    text = str(text).replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def audit_report_readiness(
    results: dict[str, dict[str, Any]],
    metadata: list[dict[str, Any]],
    min_seeds: int = 3,
) -> tuple[list[str], list[str], list[str]]:
    """Return FAIL/WARN/PASS-style checks for paper table readiness."""
    failures: list[str] = []
    warnings: list[str] = []
    passes: list[str] = []

    if any(meta.get("missing_meta") for meta in metadata):
        failures.append("At least one result file is missing _meta; do not mix old results into final tables.")
    else:
        passes.append("All loaded result files have _meta.")

    for key in ("git_commit", "gate_version", "model", "max_len"):
        vals = {m.get(key) for m in metadata if not m.get("missing_meta")}
        vals.discard(None)
        if len(vals) > 1:
            failures.append(f"Loaded result files mix different {key} values: {sorted(vals)}.")
    dirty_files = [m.get("path", "<unknown>") for m in metadata if m.get("git_dirty")]
    if dirty_files:
        failures.append(f"Some result files were produced from dirty git state: {dirty_files}.")
    checkpoint_files = [
        m.get("path", "<unknown>")
        for m in metadata
        if not m.get("missing_meta") and m.get("is_final") is False
    ]
    if checkpoint_files:
        warnings.append(f"Some result files are incremental checkpoints, not final saves: {checkpoint_files}.")

    missing_core = [name for name in CORE_METHODS if name not in results]
    if missing_core:
        failures.append(f"Missing core methods for a report table: {', '.join(missing_core)}.")
    else:
        passes.append("Core methods are present: Baseline, Naive Swap, Strict-Gated.")

    core_seed_counts: dict[str, int] = {}
    for name in CORE_METHODS:
        metrics = results.get(name)
        if not metrics:
            continue
        f1_count = metric_seed_count(metrics.get("f1"))
        core_seed_counts[name] = f1_count
        if f1_count < min_seeds:
            failures.append(f"{name} has only {f1_count} valid F1 seed(s); expected at least {min_seeds}.")

        missing_metrics = [key for key in REPORT_REQUIRED_METRICS if metric_seed_count(metrics.get(key)) == 0]
        if missing_metrics:
            failures.append(f"{name} is missing report-critical metrics: {', '.join(missing_metrics)}.")

        count_mismatches = [
            key for key in REPORT_REQUIRED_METRICS
            if metric_seed_count(metrics.get(key)) not in (0, f1_count)
        ]
        if count_mismatches:
            failures.append(f"{name} has metric seed-count mismatches: {', '.join(count_mismatches)}.")

        config = metrics.get("config")
        if not isinstance(config, dict):
            failures.append(f"{name} is missing per-experiment config.")
        else:
            missing_config = [key for key in REPORT_CONFIG_KEYS if config.get(key) is None]
            if missing_config:
                failures.append(f"{name} config is missing keys: {', '.join(missing_config)}.")
            if config.get("git_dirty"):
                failures.append(f"{name} was run from dirty git state.")

        fpr_min = mean_or_none(metrics.get("fpr_min_group_n"))
        if fpr_min is not None and fpr_min < 20:
            warnings.append(f"{name} has low FPR support (FPR minN={fpr_min:.1f}); keep FPR Gap secondary.")

        if not metrics.get("fairness_error_examples"):
            warnings.append(f"{name} is missing fairness_error_examples; qualitative error analysis will be weaker.")

    if core_seed_counts and len(set(core_seed_counts.values())) > 1:
        failures.append(f"Core methods have different seed counts: {core_seed_counts}.")
    elif (
        not missing_core
        and len(core_seed_counts) == len(CORE_METHODS)
        and all(v >= min_seeds for v in core_seed_counts.values())
    ):
        passes.append(f"Core methods have at least {min_seeds} seeds.")

    naive = results.get("Naive Swap")
    strict_names = [name for name in results if is_strict_family(name)]
    best_strict = best_variant_by(results, strict_names, "strict_pair_accuracy")
    if naive and best_strict:
        best_name, best_sp = best_strict
        naive_sp = mean_or_none(naive.get("strict_pair_accuracy"))
        if naive_sp is not None and best_sp < naive_sp:
            if "Strict-Matched" not in results:
                warnings.append("Naive beats the best gated row but Strict-Matched is missing.")
            if not has_strict_lambda_followup(results):
                warnings.append("Naive beats the best gated row but no Strict_lam follow-up is present.")
        deltas = paired_delta(results[best_name].get("strict_pair_accuracy"), naive.get("strict_pair_accuracy"))
        if not deltas:
            warnings.append("Best gated vs Naive paired seed comparison is unavailable.")
        else:
            passes.append(f"Best gated vs Naive has matched seed diagnostics ({len(deltas)} seed(s)).")

    return failures, warnings, passes


def has_strict_lambda_followup(results: dict[str, dict[str, Any]]) -> bool:
    return any(name.startswith("Strict_lam=") for name in results)


def recommended_next_steps(results: dict[str, dict[str, Any]]) -> list[str]:
    steps: list[str] = []
    missing_core = [name for name in ("Baseline", "Naive Swap", "Strict-Gated") if name not in results]
    if missing_core:
        steps.append(f"Run the missing core methods under the current commit: {', '.join(missing_core)}.")

    naive = results.get("Naive Swap")
    strict_names = [name for name in results if is_strict_family(name)]
    best_strict = best_variant_by(results, strict_names, "strict_pair_accuracy")
    if naive and best_strict:
        best_name, best_sp = best_strict
        naive_sp = mean_or_none(naive.get("strict_pair_accuracy"))
        best_f1 = mean_or_none(results[best_name].get("f1"))
        naive_f1 = mean_or_none(naive.get("f1"))
        f1_close = best_f1 is not None and naive_f1 is not None and abs(best_f1 - naive_f1) <= 0.01

        if naive_sp is not None and best_sp < naive_sp:
            if "Strict-Matched" not in results:
                steps.append("Run Strict-Matched to test whether the strict gate is simply under-regularized by lower CF coverage.")
            if not has_strict_lambda_followup(results):
                steps.append("Run targeted Strict_lam follow-ups, e.g. Strict_lam=0.15 and Strict_lam=0.25, before changing the method.")
            if f1_close:
                steps.append("If gated variants still trail Naive, keep the method as a validity-coverage tradeoff rather than forcing a stronger claim.")
        elif naive_sp is not None and best_sp >= naive_sp and f1_close:
            steps.append("Freeze the method search and move to error analysis/report writing; the gated result is strong enough.")

    if not steps:
        steps.append("No automatic follow-up is triggered; inspect metadata warnings and qualitative errors before launching more runs.")
    return steps


def print_table(results: dict[str, dict[str, Any]]) -> None:
    name_w = max(12, *(len(k) for k in results))
    headers = ["Experiment"] + [label for _, label, _ in PRIMARY_METRICS]
    widths = [name_w] + [13] * len(PRIMARY_METRICS)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for name, metrics in results.items():
        row = [name.ljust(name_w)]
        for key, _, _ in PRIMARY_METRICS:
            scale = 100.0 if key in ("train_valid_cf_ratio", "cons_batch_ratio") else 1.0
            row.append(fmt(metrics.get(key), scale=scale).rjust(13))
        print("  ".join(row))


def print_metadata_warnings(metadata: list[dict[str, Any]]) -> None:
    print("Result metadata")
    print("---------------")
    if not metadata:
        print("No metadata found.")
        return

    for meta in metadata:
        if meta.get("missing_meta"):
            print(f"- {meta['path']}: missing _meta (likely old result; avoid mixing in final tables)")
        else:
            print(
                f"- {meta['path']}: commit={meta.get('git_commit')} "
                f"gate={meta.get('gate_version')} model={meta.get('model')} "
                f"lambda={meta.get('lambda')} seeds={meta.get('seeds')}"
            )

    for key in ("git_commit", "gate_version", "model", "max_len"):
        vals = {m.get(key) for m in metadata if not m.get("missing_meta")}
        vals.discard(None)
        if len(vals) > 1:
            print(f"WARNING: result files mix different {key} values: {sorted(vals)}")
    dirty = [m["path"] for m in metadata if m.get("git_dirty")]
    if dirty:
        print(f"WARNING: result files were produced from dirty git state: {dirty}")


def print_experiment_config_warnings(results: dict[str, dict[str, Any]]) -> None:
    print("\nExperiment configs")
    print("------------------")
    configs: list[tuple[str, dict[str, Any]]] = []
    for name, metrics in results.items():
        config = metrics.get("config")
        if not isinstance(config, dict):
            print(f"- {name}: missing per-experiment config (likely old result)")
            continue
        configs.append((name, config))
        print(
            f"- {name}: mode={config.get('mode')} lambda={config.get('lambda')} "
            f"strategy={config.get('lambda_strategy', 'unknown')} "
            f"commit={config.get('git_commit')} dirty={config.get('git_dirty')} "
            f"gate={config.get('gate_version')} model={config.get('model')}"
        )

    for key in ("git_commit", "gate_version", "model", "max_len", "epochs", "batch_size", "lr"):
        vals = {c.get(key) for _, c in configs}
        vals.discard(None)
        if len(vals) > 1:
            print(f"WARNING: experiments mix different {key} values: {sorted(vals)}")
    dirty_methods = [name for name, config in configs if config.get("git_dirty")]
    if dirty_methods:
        print(f"WARNING: experiments were run from dirty git state: {dirty_methods}")
    renamed = [
        (name, metrics.get("_original_name"), metrics.get("_source_path"))
        for name, metrics in results.items()
        if metrics.get("_renamed_duplicate")
    ]
    for name, original, source in renamed:
        print(f"WARNING: duplicate experiment name '{original}' from {source} was renamed to '{name}'")


def print_baseline_deltas(results: dict[str, dict[str, Any]]) -> None:
    if "Baseline" not in results:
        return
    base = results["Baseline"]
    print("\nDelta vs Baseline")
    print("-----------------")
    for name, metrics in results.items():
        if name == "Baseline":
            continue
        print(f"\n{name}")
        for key, label, direction in PRIMARY_METRICS:
            if key in ("train_valid_cf_ratio", "cons_batch_ratio", "avg_valid_cf_per_batch", "fpr_min_group_n"):
                continue
            b = mean_or_none(base.get(key))
            c = mean_or_none(metrics.get(key))
            print(f"  {label:<12} {delta_str(b, c, direction)}")


def print_interpretation_notes(results: dict[str, dict[str, Any]]) -> None:
    print("\nInterpretation guardrails")
    print("-------------------------")
    print("- Do not rank methods by flip rate alone; low flip can hide consistently wrong pairs.")
    print("- Prefer Macro-F1 + Strict PairAcc as the main claim when available.")
    print("- TrainCF% explains regularization strength: a stricter gate may lose because it sees fewer CF pairs.")
    low_fpr_support = [
        (name, n) for name, metrics in results.items()
        if (n := mean_or_none(metrics.get("fpr_min_group_n"))) is not None and n < 20
    ]
    if low_fpr_support:
        details = ", ".join(f"{name}=minN {n:.1f}" for name, n in low_fpr_support)
        print(f"- FPR Gap has small normal-group support ({details}); keep FPR Gap as a secondary metric.")

    naive = results.get("Naive Swap")
    strict = results.get("Strict-Gated")
    matched = results.get("Strict-Matched")
    if naive and strict:
        naive_sp = mean_or_none(naive.get("strict_pair_accuracy"))
        strict_sp = mean_or_none(strict.get("strict_pair_accuracy"))
        naive_f1 = mean_or_none(naive.get("f1"))
        strict_f1 = mean_or_none(strict.get("f1"))
        naive_gap = mean_or_none(naive.get("strict_prob_gap"))
        strict_gap = mean_or_none(strict.get("strict_prob_gap"))
        naive_cf = mean_or_none(naive.get("train_valid_cf_ratio"))
        strict_cf = mean_or_none(strict.get("train_valid_cf_ratio"))
        naive_cb = mean_or_none(naive.get("cons_batch_ratio"))
        strict_cb = mean_or_none(strict.get("cons_batch_ratio"))
        naive_vb = mean_or_none(naive.get("avg_valid_cf_per_batch"))
        strict_vb = mean_or_none(strict.get("avg_valid_cf_per_batch"))
        if naive_sp is not None and strict_sp is not None:
            if strict_sp >= naive_sp:
                print("- Strict-Gated beats or matches Naive on Strict PairAcc: this supports the validity-gated claim.")
            else:
                print("- Naive beats Strict on Strict PairAcc: frame the result as an invariance-validity tradeoff.")
            print("\nPaper-claim suggestion")
            print("----------------------")
            f1_close = (
                naive_f1 is not None and strict_f1 is not None
                and abs(strict_f1 - naive_f1) <= 0.01
            )
            gap_better = (
                naive_gap is not None and strict_gap is not None
                and strict_gap <= naive_gap
            )
            if strict_sp >= naive_sp and f1_close:
                print("Use the strong claim: validity-gated CCR improves or matches Naive while preserving F1.")
            elif strict_sp < naive_sp and f1_close and gap_better:
                print("Use the tradeoff claim: Naive gives stronger hard-label invariance, Strict gives comparable F1 and softer probability stability.")
            elif strict_sp < naive_sp:
                print("Use the diagnostic claim: current strict gate is conservative; analyze TrainCF% and invalid-pair examples.")
            else:
                print("Use a cautious claim: identity-swap CCR helps, but gate benefits depend on metric choice.")
            if naive_cf is not None and strict_cf is not None:
                print(f"TrainCF coverage: Naive={100*naive_cf:.2f}% vs Strict={100*strict_cf:.2f}%.")
            if naive_cb is not None and strict_cb is not None:
                print(f"Regularized batches: Naive={100*naive_cb:.2f}% vs Strict={100*strict_cb:.2f}%.")
            if naive_vb is not None and strict_vb is not None:
                print(f"Valid CF per batch: Naive={naive_vb:.2f} vs Strict={strict_vb:.2f}.")
        else:
            print("- Strict/Naive PairAcc is missing for at least one method; rerun both with the same current code.")

    if strict and matched:
        strict_sp = mean_or_none(strict.get("strict_pair_accuracy"))
        matched_sp = mean_or_none(matched.get("strict_pair_accuracy"))
        strict_gap = mean_or_none(strict.get("strict_prob_gap"))
        matched_gap = mean_or_none(matched.get("strict_prob_gap"))
        strict_lam = mean_or_none(strict.get("lambda"))
        matched_lam = mean_or_none(matched.get("lambda"))
        if strict_sp is not None and matched_sp is not None:
            print("\nCoverage-matched diagnostic")
            print("---------------------------")
            print(f"Strict lambda={strict_lam} vs Strict-Matched lambda={matched_lam}.")
            if matched_sp > strict_sp:
                print("- Strict-Matched improves Strict PairAcc: Strict-Gated was likely under-regularized by lower CF coverage.")
            elif matched_gap is not None and strict_gap is not None and matched_gap < strict_gap:
                print("- Strict-Matched improves probability stability but not hard pair accuracy; report this as a soft-consistency gain.")
            else:
                print("- Strict-Matched does not improve Strict: the gate may be filtering useful signal, not merely reducing coverage.")

    strict_names = [name for name in results if is_strict_family(name)]
    best_strict = best_variant_by(results, strict_names, "strict_pair_accuracy")
    if best_strict:
        best_name, best_sp = best_strict
        best_f1 = mean_or_none(results[best_name].get("f1"))
        best_gap = mean_or_none(results[best_name].get("strict_prob_gap"))
        naive_sp = mean_or_none(naive.get("strict_pair_accuracy")) if naive else None
        naive_f1 = mean_or_none(naive.get("f1")) if naive else None
        naive_gap = mean_or_none(naive.get("strict_prob_gap")) if naive else None
        print("\nBest strict-family variant")
        print("--------------------------")
        print(
            f"{best_name}: Strict PairAcc={best_sp:.4f}, "
            f"F1={fmt_num(best_f1)}, Strict ProbGap={fmt_num(best_gap)}"
        )
        if naive_sp is not None:
            f1_close = best_f1 is not None and naive_f1 is not None and abs(best_f1 - naive_f1) <= 0.01
            if best_sp >= naive_sp and f1_close:
                print("- Use this as the main gated result: it matches/beats Naive on Strict PairAcc while preserving F1.")
            elif best_sp < naive_sp and f1_close:
                print("- Use this as the strongest gated result, but frame Naive vs gated as a validity-coverage tradeoff.")
            else:
                print("- Use cautiously: compare F1 and pair metrics before making this the main result.")
            if best_gap is not None and naive_gap is not None and best_gap < naive_gap:
                print("- It also improves Strict ProbGap over Naive, useful as a soft-consistency argument.")
        if "Strict-Matched" not in results and "Naive Swap" in results and best_name != "Strict-Matched":
            print("- If Naive still beats this variant, run Strict-Matched to separate low coverage from gate quality.")


def print_naive_vs_best_gated_diagnostic(results: dict[str, dict[str, Any]]) -> None:
    naive = results.get("Naive Swap")
    strict_names = [name for name in results if is_strict_family(name)]
    best_strict = best_variant_by(results, strict_names, "strict_pair_accuracy")
    if not naive or not best_strict:
        return

    best_name, _ = best_strict
    best_metrics = results[best_name]
    comparisons = [
        ("Strict PairAcc", "strict_pair_accuracy", "higher"),
        ("Macro-F1", "f1", "higher"),
        ("Strict ProbGap", "strict_prob_gap", "lower"),
    ]
    print("\nNaive vs best gated paired diagnostic")
    print("-------------------------------------")
    print(f"Best gated row: {best_name}")
    for label, key, direction in comparisons:
        deltas = paired_delta(best_metrics.get(key), naive.get(key))
        if not deltas:
            print(f"- {label}: paired seed comparison unavailable.")
            continue
        mean_delta = mean(deltas)
        wins = sum(d > 0 for d in deltas) if direction == "higher" else sum(d < 0 for d in deltas)
        sign = "+" if mean_delta >= 0 else ""
        good_word = "higher" if direction == "higher" else "lower"
        print(
            f"- {label}: gated-naive mean delta={sign}{mean_delta:.4f}; "
            f"gated is {good_word} on {wins}/{len(deltas)} matched seeds."
        )


def print_claim_assessment(results: dict[str, dict[str, Any]]) -> None:
    assessment = assess_claim(results)
    print("\nClaim assessment")
    print("----------------")
    print(f"Level: {assessment['level']}")
    if assessment.get("main_method"):
        print(f"Main method: {assessment['main_method']}")
    print(f"Headline: {assessment['headline']}")
    for item in assessment.get("rationale", []):
        print(f"- {item}")
    print(f"Report action: {assessment['report_action']}")


def print_next_step_recommendations(results: dict[str, dict[str, Any]]) -> None:
    print("\nRecommended next steps")
    print("----------------------")
    for step in recommended_next_steps(results):
        print(f"- {step}")


def print_report_readiness_audit(results: dict[str, dict[str, Any]], metadata: list[dict[str, Any]]) -> None:
    failures, warnings, passes = audit_report_readiness(results, metadata)
    print("\nReport readiness audit")
    print("----------------------")
    if not failures and not warnings:
        print("PASS: Results are ready for a main report table, subject to qualitative error analysis.")
    for item in failures:
        print(f"FAIL: {item}")
    for item in warnings:
        print(f"WARN: {item}")
    for item in passes:
        print(f"PASS: {item}")


def print_error_example_summary(results: dict[str, dict[str, Any]]) -> None:
    print("\nSaved qualitative examples")
    print("--------------------------")
    print("Counts are capped saved examples, not total error counts.")
    buckets = [
        "flip",
        "strict_flip",
        "both_wrong",
        "strict_both_wrong",
        "false_positive_original",
        "false_positive_cf",
    ]
    name_w = max(12, *(len(name) for name in results))
    header = ["Experiment"] + buckets
    widths = [name_w] + [max(8, len(bucket)) for bucket in buckets]
    print("  ".join(item.ljust(width) for item, width in zip(header, widths)))
    print("  ".join("-" * width for width in widths))
    for name, metrics in results.items():
        row = [name.ljust(name_w)]
        for bucket, width in zip(buckets, widths[1:]):
            row.append(str(bucket_example_count(metrics, bucket)).rjust(width))
        print("  ".join(row))


def print_error_examples(
    results: dict[str, dict[str, Any]],
    buckets: list[str],
    max_examples: int,
) -> None:
    print("\nQualitative error examples")
    print("--------------------------")
    for bucket in buckets:
        print(f"\n[{bucket}]")
        any_printed = False
        for name, metrics in results.items():
            for i, (seed, ex) in enumerate(iter_error_examples(metrics, bucket)):
                if i >= max_examples:
                    break
                any_printed = True
                print(
                    f"- {name} seed={seed} label={ex.get('label')} "
                    f"pred={ex.get('pred')} cf_pred={ex.get('cf_pred')} "
                    f"prob={ex.get('prob')} cf_prob={ex.get('cf_prob')} "
                    f"gap={ex.get('prob_gap')} cat={ex.get('category')} "
                    f"terms={ex.get('orig_term')}->{ex.get('swap_term')} "
                    f"strict={ex.get('strict_valid')}"
                )
                print(f"  orig: {shorten(ex.get('text'))}")
                print(f"  cf  : {shorten(ex.get('cf_text'))}")
        if not any_printed:
            print("- No saved examples.")


def print_markdown_table(results: dict[str, dict[str, Any]]) -> None:
    print("\nMarkdown table")
    print("--------------")
    print("| Method | Macro-F1 | Pair Acc | Strict Pair Acc | Flip Rate | Strict Flip | Prob Gap | Strict Prob Gap | FPR Gap | FPR minN | Train CF% | Cons Batch% |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, metrics in results.items():
        print(
            f"| {name} | {fmt(metrics.get('f1'))} | {fmt(metrics.get('pair_accuracy'))} | "
            f"{fmt(metrics.get('strict_pair_accuracy'))} | {fmt(metrics.get('flip_rate'))} | "
            f"{fmt(metrics.get('strict_flip_rate'))} | {fmt(metrics.get('prob_gap'))} | "
            f"{fmt(metrics.get('strict_prob_gap'))} | {fmt(metrics.get('fpr_gap'))} | "
            f"{fmt(metrics.get('fpr_min_group_n'))} | {fmt(metrics.get('train_valid_cf_ratio'), scale=100.0)} | "
            f"{fmt(metrics.get('cons_batch_ratio'), scale=100.0)} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json", nargs="+", type=Path, help="result JSON path(s)")
    parser.add_argument(
        "--show_examples",
        action="store_true",
        help="print saved qualitative error examples after the summary",
    )
    parser.add_argument(
        "--example_bucket",
        action="append",
        choices=ERROR_EXAMPLE_BUCKETS,
        help="bucket to print with --show_examples; repeatable",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=2,
        help="max examples per experiment and bucket when --show_examples is used",
    )
    args = parser.parse_args()
    results, metadata = load_results_with_metadata(args.json)
    if not results:
        raise SystemExit("No valid experiment results found.")
    print_metadata_warnings(metadata)
    print_experiment_config_warnings(results)
    print()
    print_table(results)
    print_baseline_deltas(results)
    print_interpretation_notes(results)
    print_naive_vs_best_gated_diagnostic(results)
    print_claim_assessment(results)
    print_next_step_recommendations(results)
    print_report_readiness_audit(results, metadata)
    print_error_example_summary(results)
    if args.show_examples:
        buckets = args.example_bucket or ["both_wrong", "strict_flip", "false_positive_original"]
        print_error_examples(results, buckets, args.max_examples)
    print_markdown_table(results)


if __name__ == "__main__":
    main()
