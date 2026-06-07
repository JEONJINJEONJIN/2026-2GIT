import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bigfive_analysis import (
    build_final_report,
    condition_effects_with_ci,
    condition_differences,
    paired_bootstrap_ci,
    paired_scenario_differences,
    summarize_consistency,
    summarize_pas_agreement,
    summarize_side_effects,
)


class TestBigFiveAnalysis(unittest.TestCase):
    def test_summarize_and_differences(self):
        rows = self._consistency_rows()
        summary = summarize_consistency(rows)
        differences = condition_differences(summary)
        paired = paired_scenario_differences(rows)

        baseline = [
            row for row in summary
            if row["condition"] == "baseline"
        ][0]
        self.assertAlmostEqual(baseline["mean_consistency"], 0.75)
        self.assertEqual(len(differences), 1)
        self.assertAlmostEqual(differences[0]["delta_consistency"], 0.1)
        self.assertEqual(paired[0]["n_paired_scenarios"], 2)
        self.assertAlmostEqual(paired[0]["mean_paired_delta_consistency"], 0.1)

    def test_paired_bootstrap_ci(self):
        rows = self._consistency_rows()
        ci_rows = paired_bootstrap_ci(rows, n_bootstrap=50, seed=1)

        self.assertEqual(len(ci_rows), 1)
        self.assertEqual(ci_rows[0]["condition"], "as_pas_only")
        self.assertEqual(ci_rows[0]["n_paired_scenarios"], 2)
        self.assertAlmostEqual(ci_rows[0]["mean_delta"], 0.1)
        self.assertLessEqual(ci_rows[0]["ci_lower"], ci_rows[0]["mean_delta"])
        self.assertGreaterEqual(ci_rows[0]["ci_upper"], ci_rows[0]["mean_delta"])

    def test_condition_effects_with_ci(self):
        rows = self._consistency_rows()
        summary = summarize_consistency(rows)
        differences = condition_differences(summary)
        bootstrap = paired_bootstrap_ci(rows, n_bootstrap=50, seed=1)
        bootstrap += paired_bootstrap_ci(
            rows,
            metric_column="trait_target_attainment",
            n_bootstrap=50,
            seed=1,
        )

        effects = condition_effects_with_ci(differences, bootstrap)

        self.assertEqual(len(effects), 1)
        self.assertIn("consistency_ci_lower", effects[0])
        self.assertIn("trait_target_ci_upper", effects[0])
        self.assertLessEqual(
            effects[0]["consistency_ci_lower"],
            effects[0]["consistency_ci_upper"],
        )
        self.assertAlmostEqual(effects[0]["delta_consistency"], 0.1)

    def test_summarize_pas_agreement(self):
        rows = [
            {
                "condition": "as_pas_only",
                "trait": "agreeableness",
                "target_direction": "high",
                "pas_loglik_agreement": "true",
            },
            {
                "condition": "as_pas_only",
                "trait": "agreeableness",
                "target_direction": "high",
                "pas_loglik_agreement": "false",
            },
        ]
        summary = summarize_pas_agreement(rows)
        self.assertAlmostEqual(summary[0]["pas_loglik_agreement_rate"], 0.5)

    def test_summarize_side_effects(self):
        rows = [
            {
                "condition": "baseline",
                "trait": "agreeableness",
                "target_direction": "high",
                "parse_success": "true",
                "format_validity": "true",
                "has_speech_tag": "true",
                "has_action_tag": "true",
                "response_length_chars": "40",
                "response_length_words": "5",
            },
            {
                "condition": "baseline",
                "trait": "agreeableness",
                "target_direction": "high",
                "parse_success": "false",
                "format_validity": "false",
                "has_speech_tag": "true",
                "has_action_tag": "false",
                "response_length_chars": "20",
                "response_length_words": "3",
            },
        ]
        summary = summarize_side_effects(rows)
        self.assertEqual(summary[0]["n"], 2)
        self.assertAlmostEqual(summary[0]["parse_success_rate"], 0.5)
        self.assertAlmostEqual(summary[0]["format_validity_rate"], 0.5)
        self.assertAlmostEqual(summary[0]["mean_response_length_words"], 4.0)

    def test_build_final_report_includes_guardrails_and_metadata(self):
        report = build_final_report(
            manifest={
                "run_id": "run1",
                "model_name": "test-model",
                "git_commit": "abc123",
                "data_split": "dev",
                "seed": 42,
                "alpha": 1.0,
                "layers": [18, 21, 24],
                "vector_hash": "hash",
                "extra": {
                    "config_sha256": "cfg",
                    "model_config_sha256": "modelcfg",
                },
            },
            summary_rows=summarize_consistency(self._consistency_rows()),
            difference_rows=condition_differences(
                summarize_consistency(self._consistency_rows())
            ),
            bootstrap_rows=paired_bootstrap_ci(
                self._consistency_rows(),
                n_bootstrap=10,
            ),
        )

        self.assertIn("V4 Big Five Speech-Action Consistency Report", report)
        self.assertIn("run_id: `run1`", report)
        self.assertIn("1 - abs(BFI_score - TRAIT_score)", report)
        self.assertIn("Do not claim this measures LLM morality", report)

    def test_analysis_script_outputs_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics_dir = Path(tmp) / "metrics"
            metrics_dir.mkdir()
            self._write_csv(metrics_dir / "consistency_metrics.csv", self._consistency_rows())
            self._write_csv(
                metrics_dir / "pas_loglik_agreement.csv",
                [
                    {
                        "condition": "as_pas_only",
                        "trait": "agreeableness",
                        "target_direction": "high",
                        "pas_loglik_agreement": "true",
                    }
                ],
            )
            self._write_csv(
                metrics_dir / "side_effect_metrics.csv",
                [
                    {
                        "condition": "baseline",
                        "trait": "agreeableness",
                        "target_direction": "high",
                        "parse_success": "true",
                        "format_validity": "true",
                        "has_speech_tag": "true",
                        "has_action_tag": "true",
                        "response_length_chars": "40",
                        "response_length_words": "5",
                    }
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/14_analyze_bigfive_results.py",
                    "--metrics_dir",
                    str(metrics_dir),
                ],
                cwd=Path(__file__).resolve().parent.parent,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertTrue((metrics_dir / "consistency_summary.csv").exists())
            self.assertTrue((metrics_dir / "condition_differences.csv").exists())
            self.assertTrue((metrics_dir / "condition_effects_with_ci.csv").exists())
            self.assertTrue((metrics_dir / "paired_scenario_differences.csv").exists())
            self.assertTrue((metrics_dir / "paired_bootstrap_ci.csv").exists())
            self.assertTrue((metrics_dir / "pas_agreement_summary.csv").exists())
            self.assertTrue((metrics_dir / "side_effect_summary.csv").exists())
            self.assertTrue((metrics_dir / "analysis_report.md").exists())
            self.assertTrue((metrics_dir / "final_report.md").exists())
            self.assertIn("Wrote report", result.stdout)
            self.assertIn("Wrote final report", result.stdout)

    @staticmethod
    def _consistency_rows():
        return [
            {
                "condition": "baseline",
                "scenario_id": "s1",
                "trait": "agreeableness",
                "target_direction": "high",
                "bfi_score": "0.7",
                "trait_score": "0.6",
                "consistency": "0.9",
                "bfi_target_attainment": "0.2",
                "trait_target_attainment": "0.1",
            },
            {
                "condition": "baseline",
                "scenario_id": "s2",
                "trait": "agreeableness",
                "target_direction": "high",
                "bfi_score": "0.4",
                "trait_score": "0.1",
                "consistency": "0.6",
                "bfi_target_attainment": "-0.1",
                "trait_target_attainment": "-0.4",
            },
            {
                "condition": "as_pas_only",
                "scenario_id": "s1",
                "trait": "agreeableness",
                "target_direction": "high",
                "bfi_score": "0.8",
                "trait_score": "0.7",
                "consistency": "0.9",
                "bfi_target_attainment": "0.3",
                "trait_target_attainment": "0.2",
            },
            {
                "condition": "as_pas_only",
                "scenario_id": "s2",
                "trait": "agreeableness",
                "target_direction": "high",
                "bfi_score": "0.5",
                "trait_score": "0.3",
                "consistency": "0.8",
                "bfi_target_attainment": "0.0",
                "trait_target_attainment": "-0.2",
            },
        ]

    @staticmethod
    def _write_csv(path, rows):
        rows = list(rows)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
