import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "validity_gated_exp" / "preflight_run.py"
sys.path.insert(0, str(MODULE_PATH.parent))

spec = importlib.util.spec_from_file_location("preflight_run", MODULE_PATH)
preflight_run = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(preflight_run)


def args(**overrides):
    defaults = {
        "exp": ["Baseline", "Naive Swap", "Strict-Gated", "Strict-Matched", "Strict_lam=0.15"],
        "seeds": [42, 123, 456],
        "epochs": 3,
        "batch_size": 64,
        "lr": 3e-5,
        "lambda_value": 0.1,
        "subset": 0,
        "max_len": 128,
        "num_workers": 2,
        "base_dir": str(REPO_ROOT / "validity_gated_exp"),
        "result_path": None,
        "expected_commit": None,
        "require_clean": False,
        "require_core": True,
        "fresh_result_path": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class PreflightRunTest(unittest.TestCase):
    def test_preflight_accepts_core_followup_plan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = preflight_run.build_preflight_report(
                args(base_dir=tmpdir),
                REPO_ROOT,
            )

        self.assertEqual(report["failures"], [])
        self.assertIn("Strict_lam=0.15", report["planned"])
        self.assertIn("Strict-Matched", report["planned"])
        self.assertIn("Planned model fits: 15", "\n".join(report["notes"]))

    def test_preflight_warns_when_core_methods_are_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = preflight_run.build_preflight_report(
                args(exp=["Naive Swap"], base_dir=tmpdir, require_core=False),
                REPO_ROOT,
            )

        self.assertEqual(report["failures"], [])
        joined = "\n".join(report["warnings"])
        self.assertIn("Missing core report experiments", joined)
        self.assertIn("without any strict-family comparison", joined)

    def test_preflight_can_require_fresh_result_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = Path(tmpdir) / "results.json"
            result.write_text("{}", encoding="utf-8")
            report = preflight_run.build_preflight_report(
                args(result_path=str(result), fresh_result_path=True),
                REPO_ROOT,
            )

        self.assertIn("Result path already exists", "\n".join(report["failures"]))

    def test_preflight_fails_on_bad_experiment_tag(self):
        report = preflight_run.build_preflight_report(
            args(exp=["Strict"], require_core=False),
            REPO_ROOT,
        )

        self.assertIn("Unknown --exp tag", "\n".join(report["failures"]))

    def test_preflight_rejects_nonfinite_numeric_values(self):
        report = preflight_run.build_preflight_report(
            args(lr=float("inf"), lambda_value=float("nan")),
            REPO_ROOT,
        )

        joined = "\n".join(report["failures"])
        self.assertIn("lr must be positive", joined)
        self.assertIn("lambda must be positive", joined)

    def test_print_report_marks_pass_and_fail(self):
        passing = {
            "failures": [],
            "warnings": [],
            "notes": [],
            "planned": ["Baseline"],
            "result_path": Path("results.json"),
            "commit": "abc123",
            "dirty": False,
            "seeds": [42, 123, 456],
        }
        failing = {**passing, "failures": ["bad"]}

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            preflight_run.print_report(passing)
            preflight_run.print_report(failing)

        text = out.getvalue()
        self.assertIn("PREFLIGHT PASS", text)
        self.assertIn("PREFLIGHT FAIL", text)


if __name__ == "__main__":
    unittest.main()
