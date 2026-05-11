"""Tests for AS-only evaluation metrics."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.as_metrics import (
    action_alignment_category,
    action_entropy,
    annotate_entries,
    build_action_alignment_map,
    classify_speech_heuristic,
    speech_action_agreement,
    summarize_group,
)


class TestASMetrics(unittest.TestCase):
    def setUp(self):
        self.scenarios = [
            {
                "actions": [
                    {"id": "act_001", "persona_alignment": "aggressive"},
                    {"id": "act_002", "persona_alignment": "cooperative"},
                    {"id": "act_003", "persona_alignment": "neutral"},
                ]
            }
        ]
        self.mapping = build_action_alignment_map(self.scenarios)

    def test_action_alignment_map_uses_action_ids(self):
        self.assertEqual(self.mapping["act_001"], "aggressive")
        self.assertEqual(self.mapping["act_002"], "cooperative")

    def test_alignment_category(self):
        self.assertEqual(
            action_alignment_category("aggressive", "act_001", self.mapping),
            "aligned",
        )
        self.assertEqual(
            action_alignment_category("cooperative", "act_001", self.mapping),
            "misaligned",
        )
        self.assertEqual(
            action_alignment_category("aggressive", "act_003", self.mapping),
            "neutral",
        )

    def test_speech_action_agreement(self):
        self.assertTrue(speech_action_agreement("aggressive", "aggressive"))
        self.assertFalse(speech_action_agreement("cooperative", "aggressive"))

    def test_heuristic_speech_classifier(self):
        self.assertEqual(
            classify_speech_heuristic("Stand down or I will crush you."),
            "aggressive",
        )
        self.assertEqual(
            classify_speech_heuristic("Let us negotiate a fair solution."),
            "cooperative",
        )

    def test_entropy(self):
        self.assertAlmostEqual(action_entropy(["a", "a", "a"]), 0.0)
        self.assertGreater(action_entropy(["a", "b"]), 0.0)

    def test_annotate_and_summarize(self):
        entries = [
            {
                "persona": "aggressive",
                "speech": "I will crush this blockade.",
                "final_action": "act_001",
                "parse_ok": True,
            },
            {
                "persona": "aggressive",
                "speech": "Let us negotiate.",
                "final_action": "act_002",
                "parse_ok": True,
            },
        ]
        annotated = annotate_entries(entries, self.mapping)
        self.assertEqual(annotated[0]["action_alignment_category"], "aligned")
        self.assertEqual(annotated[1]["action_alignment_category"], "misaligned")
        summary = summarize_group(annotated)
        self.assertEqual(summary["n"], 2)
        self.assertEqual(summary["parse_ok_rate"], 1.0)
        self.assertEqual(summary["persona_alignment_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
