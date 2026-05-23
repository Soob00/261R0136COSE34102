import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "validity_gated_exp" / "analyze_cf_pairs.py"

spec = importlib.util.spec_from_file_location("analyze_cf_pairs", MODULE_PATH)
analyze_cf_pairs = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(analyze_cf_pairs)


class AnalyzeCfPairsTest(unittest.TestCase):
    def sample_pairs(self):
        return [
            {
                "original": "여성은 위험하다",
                "cf": "남성은 위험하다",
                "category": "gender",
                "label": 1,
                "gate_version": "v1",
                "base_use_for_ccr": True,
                "strict_use_for_ccr": True,
                "strict_valid_grammar": True,
                "strict_valid_semantics": True,
                "strict_label_preserving": True,
                "strict_no_comparison": True,
                "strict_no_harmful_obj": True,
                "strict_no_age_contradiction": True,
            },
            {
                "original": "무슬림 테러 관련 글",
                "cf": "기독교인 테러 관련 글",
                "category": "religion",
                "label": 1,
                "gate_version": "v1",
                "base_use_for_ccr": True,
                "strict_use_for_ccr": False,
                "strict_valid_grammar": True,
                "strict_valid_semantics": False,
                "strict_label_preserving": False,
                "strict_no_comparison": True,
                "strict_no_harmful_obj": True,
                "strict_no_age_contradiction": True,
            },
            {
                "original": "노인보다 청년이 낫다",
                "cf": "청년보다 청년이 낫다",
                "category": "age",
                "label": 0,
                "gate_version": "v1",
                "base_use_for_ccr": True,
                "strict_use_for_ccr": False,
                "strict_valid_grammar": True,
                "strict_valid_semantics": True,
                "strict_label_preserving": True,
                "strict_no_comparison": False,
                "strict_no_harmful_obj": True,
                "strict_no_age_contradiction": True,
            },
        ]

    def test_get_reject_reason_uses_priority_order(self):
        pair = self.sample_pairs()[1]
        self.assertEqual(analyze_cf_pairs.get_reject_reason(pair), "semantics")

    def test_analyze_pairs_counts_reasons_by_category(self):
        stats = analyze_cf_pairs.analyze_pairs(self.sample_pairs(), train_total=10)
        self.assertEqual(stats["n_swap"], 3)
        self.assertEqual(stats["n_base"], 3)
        self.assertEqual(stats["n_strict"], 1)
        self.assertEqual(stats["reason_cnt"]["semantics"], 1)
        self.assertEqual(stats["reason_cnt"]["comparison"], 1)
        self.assertEqual(stats["reason_by_cat"]["religion"]["semantics"], 1)
        self.assertEqual(stats["reason_by_cat"]["age"]["comparison"], 1)

    def test_build_report_lines_includes_rejection_matrix(self):
        lines = analyze_cf_pairs.build_report_lines(
            self.sample_pairs(),
            train_total=10,
            examples_per_reason=1,
            max_chars=80,
        )
        text = "\n".join(lines)
        self.assertIn("[3] Strict-only Rejection Matrix", text)
        self.assertIn("religion", text)
        self.assertIn("semantics", text)
        self.assertIn("comparison expression", text)
        self.assertIn("Rejected by strict", text)


if __name__ == "__main__":
    unittest.main()
