"""
Analyze generated counterfactual pairs without GPU dependencies.

Report-useful outputs:
  1. Overall pair count / pass rate
  2. Category-wise validity statistics
  3. Strict-only rejection breakdown
  4. Reason-by-category rejection matrix
  5. Pass / reject examples

Usage:
    python validity_gated_exp/analyze_cf_pairs.py
    python validity_gated_exp/analyze_cf_pairs.py --jsonl validity_gated_exp/data/cf_pairs_train.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REASON_LABELS = {
    "grammar": "strict_valid_grammar",
    "semantics": "strict_valid_semantics",
    "asym_pair": "strict_label_preserving",
    "comparison": "strict_no_comparison",
    "harmful_obj": "strict_no_harmful_obj",
    "age_context": "strict_no_age_contradiction",
}
REASON_ORDER = ["semantics", "asym_pair", "comparison", "harmful_obj", "age_context", "grammar", "unknown"]
REASON_LABELS_READABLE = {
    "semantics": "semantic blacklist",
    "asym_pair": "asymmetric pair (label not preserved)",
    "comparison": "comparison expression",
    "harmful_obj": "harmful object/event context",
    "age_context": "explicit age context contradiction",
    "grammar": "grammar check failed",
    "unknown": "unknown",
}
DEFAULT_CATEGORIES = ["gender", "ethnicity", "religion", "age", "sexuality", "disability"]


def pct(num: int, denom: int) -> float:
    return (num / denom * 100.0) if denom else 0.0


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def get_reject_reason(pair: dict[str, Any]) -> str:
    """Return the first failed strict-gate field by priority order."""
    for reason, field in REASON_LABELS.items():
        if not pair.get(field, True):
            return reason
    return "unknown"


def load_pairs(jsonl_path: Path) -> list[dict[str, Any]]:
    pairs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def analyze_pairs(pairs: list[dict[str, Any]], train_total: int) -> dict[str, Any]:
    n_swap = len(pairs)
    n_base = sum(1 for p in pairs if p.get("base_use_for_ccr"))
    n_strict = sum(1 for p in pairs if p.get("strict_use_for_ccr"))
    extra_rejected = [p for p in pairs if p.get("base_use_for_ccr") and not p.get("strict_use_for_ccr")]

    cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"swap": 0, "base": 0, "strict": 0})
    reason_cnt: Counter[str] = Counter()
    reason_by_cat: dict[str, Counter[str]] = defaultdict(Counter)
    gate_versions = Counter(p.get("gate_version", "missing") for p in pairs)

    for p in pairs:
        cat = p.get("category", "unknown")
        cat_stats[cat]["swap"] += 1
        cat_stats[cat]["base"] += int(bool(p.get("base_use_for_ccr")))
        cat_stats[cat]["strict"] += int(bool(p.get("strict_use_for_ccr")))
        if p.get("base_use_for_ccr") and not p.get("strict_use_for_ccr"):
            reason = get_reject_reason(p)
            reason_cnt[reason] += 1
            reason_by_cat[cat][reason] += 1

    return {
        "train_total": train_total,
        "n_swap": n_swap,
        "n_base": n_base,
        "n_strict": n_strict,
        "gate_versions": gate_versions,
        "extra_rejected": extra_rejected,
        "reason_cnt": reason_cnt,
        "cat_stats": cat_stats,
        "reason_by_cat": reason_by_cat,
    }


def ordered_categories(cat_stats: dict[str, dict[str, int]]) -> list[str]:
    extras = sorted(c for c in cat_stats if c not in DEFAULT_CATEGORIES)
    return DEFAULT_CATEGORIES + extras


def build_report_lines(
    pairs: list[dict[str, Any]],
    train_total: int,
    examples_per_reason: int,
    max_chars: int,
) -> list[str]:
    stats = analyze_pairs(pairs, train_total)
    n_swap = stats["n_swap"]
    n_base = stats["n_base"]
    n_strict = stats["n_strict"]
    extra_rejected = stats["extra_rejected"]
    reason_cnt = stats["reason_cnt"]
    cat_stats = stats["cat_stats"]
    reason_by_cat = stats["reason_by_cat"]
    gate_versions = stats["gate_versions"]

    lines: list[str] = []

    def pr(s: str = "") -> None:
        lines.append(s)

    pr("=" * 65)
    pr("  [1] CF Construction Statistics")
    pr("=" * 65)
    col = 32
    pr(f'  {"Item":<{col}} Value')
    pr(f'  {"-" * col} --------')
    pr(f'  {"Train samples":<{col}} {train_total:,}')
    pr(f'  {"Swappable samples":<{col}} {n_swap:,}  ({pct(n_swap, train_total):.1f}% of train)')
    pr(f'  {"Base-valid pairs":<{col}} {n_base:,}  ({pct(n_base, n_swap):.1f}% of swappable)')
    pr(f'  {"Strict-valid pairs":<{col}} {n_strict:,}  ({pct(n_strict, n_swap):.1f}% of swappable)')
    pr(f'  {"Gate versions":<{col}} {dict(gate_versions)}')
    if "missing" in gate_versions:
        pr()
        pr("  WARNING: gate_version is missing. This JSONL may have been generated by an older gate;")
        pr("           rerun check_data.py or run_exp.py before using construction stats in the report.")

    pr()
    pr("  Strict gate additionally filters:")
    pr(
        f'  {"Rejected by strict (vs base)":<{col}} {n_base - n_strict:,}  '
        f'({pct(n_base - n_strict, n_base):.1f}% of base-valid)'
    )

    pr()
    pr("  Strict-only rejection breakdown (base-valid -> strict-rejected):")
    if extra_rejected:
        for reason, cnt in reason_cnt.most_common():
            pr(f"    {reason:<20}: {cnt:,}  ({pct(cnt, len(extra_rejected)):.1f}%)")
    else:
        pr("    none                : 0  (0.0%)")

    pr()
    pr("=" * 65)
    pr("  [2] Category-wise Validity Statistics")
    pr("=" * 65)
    pr(f'  {"Category":<12} {"Swappable":>10} {"Base-valid":>11} {"Strict-valid":>13} {"Base%":>7} {"Strict%":>8}')
    pr(f'  {"-" * 12} {"-" * 10} {"-" * 11} {"-" * 13} {"-" * 7} {"-" * 8}')
    for cat in ordered_categories(cat_stats):
        s = cat_stats.get(cat, {"swap": 0, "base": 0, "strict": 0})
        if s["swap"] == 0:
            pr(f"  {cat:<12} {'N/A':>10}")
            continue
        pr(
            f'  {cat:<12} {s["swap"]:>10,} {s["base"]:>11,} {s["strict"]:>13,} '
            f'{pct(s["base"], s["swap"]):>6.1f}% {pct(s["strict"], s["swap"]):>7.1f}%'
        )

    pr()
    pr("=" * 65)
    pr("  [3] Strict-only Rejection Matrix")
    pr("=" * 65)
    reasons_seen = [r for r in REASON_ORDER if r in reason_cnt]
    if not reasons_seen:
        reasons_seen = ["none"]
    header = f'  {"Category":<12}' + "".join(f" {reason[:12]:>12}" for reason in reasons_seen) + f" {'Total':>8}"
    pr(header)
    pr("  " + "-" * (len(header) - 2))
    for cat in ordered_categories(cat_stats):
        if cat_stats.get(cat, {}).get("swap", 0) == 0:
            continue
        total = sum(reason_by_cat.get(cat, Counter()).values())
        row = f"  {cat:<12}"
        for reason in reasons_seen:
            row += f" {reason_by_cat.get(cat, Counter()).get(reason, 0):>12,}"
        row += f" {total:>8,}"
        pr(row)

    pr()
    pr("=" * 65)
    pr("  [4] Qualitative Pass / Reject Examples")
    pr("=" * 65)
    pr()
    pr("  [PASS examples — strict-valid=True]")
    pr()
    shown_cats: set[str] = set()
    for p in pairs:
        if not p.get("strict_use_for_ccr"):
            continue
        cat = p.get("category", "unknown")
        if cat in shown_cats:
            continue
        shown_cats.add(cat)
        pr(f"  Category : {cat}")
        pr(f'  Original : {shorten(p.get("original", ""), max_chars)}')
        pr(f'  CF       : {shorten(p.get("cf", ""), max_chars)}')
        pr(f'  Label    : {p.get("label")}')
        pr()
        if len(shown_cats) >= 4:
            break

    pr("  [REJECT examples — strict-valid=False]")
    pr()
    shown_by_reason: Counter[str] = Counter()
    for p in extra_rejected:
        reason = get_reject_reason(p)
        if shown_by_reason[reason] >= examples_per_reason:
            continue
        shown_by_reason[reason] += 1
        pr(f'  Category : {p.get("category", "unknown")}')
        pr(f'  Original : {shorten(p.get("original", ""), max_chars)}')
        pr(f'  CF       : {shorten(p.get("cf", ""), max_chars)}')
        pr(f"  Reason   : {REASON_LABELS_READABLE.get(reason, reason)}")
        pr(f'  (base_valid={p.get("base_use_for_ccr")}, strict_valid={p.get("strict_use_for_ccr")})')
        pr()
        if all(shown_by_reason[r] >= examples_per_reason for r in reason_cnt):
            break

    pr("=" * 65)
    pr("  완료.")
    pr("=" * 65)
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default="validity_gated_exp/data/cf_pairs_train.jsonl",
        help="cf_pairs_train.jsonl path",
    )
    parser.add_argument("--train_total", type=int, default=172157, help="train sample count")
    parser.add_argument("--out", default=None, help="write report text to this path")
    parser.add_argument("--examples_per_reason", type=int, default=1)
    parser.add_argument("--max_chars", type=int, default=120)
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    pairs = load_pairs(jsonl_path)
    lines = build_report_lines(
        pairs,
        train_total=args.train_total,
        examples_per_reason=args.examples_per_reason,
        max_chars=args.max_chars,
    )
    print("\n".join(lines))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n결과 저장 -> {out_path}")


if __name__ == "__main__":
    main()
