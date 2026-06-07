"""Tests for v2 clustered analysis helpers."""

import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.cluster_bootstrap import cluster_bootstrap_alignment
from analysis.mixed_effects import prepare_alignment_frame


class TestClusterBootstrap(unittest.TestCase):
    def test_cluster_bootstrap_outputs_itt_and_pp(self):
        df = pd.DataFrame([
            {
                "condition": "neutral_baseline",
                "persona": "aggressive",
                "scenario_id": "s1",
                "parse_ok": True,
                "action_alignment_category": "aligned",
            },
            {
                "condition": "neutral_baseline",
                "persona": "aggressive",
                "scenario_id": "s2",
                "parse_ok": False,
                "action_alignment_category": "unknown",
            },
            {
                "condition": "as_only",
                "persona": "aggressive",
                "scenario_id": "s1",
                "parse_ok": True,
                "action_alignment_category": "misaligned",
            },
            {
                "condition": "as_only",
                "persona": "aggressive",
                "scenario_id": "s2",
                "parse_ok": True,
                "action_alignment_category": "aligned",
            },
        ])

        result = cluster_bootstrap_alignment(df, n_bootstrap=20, seed=7)
        self.assertEqual(set(result["estimand"]), {"itt", "pp"})
        self.assertIn("ci_low", result.columns)
        self.assertIn("ci_high", result.columns)


class TestMixedEffectsPrep(unittest.TestCase):
    def test_prepare_alignment_frame_adds_binary_outcome(self):
        df = pd.DataFrame([
            {
                "condition": "neutral_baseline",
                "persona": "cooperative",
                "scenario_id": "s1",
                "action_alignment_category": "aligned",
            },
            {
                "condition": "as_only",
                "persona": "cooperative",
                "scenario_id": "s1",
                "action_alignment_category": "misaligned",
            },
        ])

        model_df = prepare_alignment_frame(df)
        self.assertEqual(model_df["aligned"].tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
