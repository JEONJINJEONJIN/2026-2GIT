import unittest

from src.evaluation.side_effect_metrics import (
    build_side_effect_row,
    format_validity,
    has_action_tag,
    has_speech_tag,
    response_length_chars,
    response_length_words,
    summarize_side_effect_rows,
)


class TestSideEffectMetrics(unittest.TestCase):
    def test_length_metrics(self):
        text = "[Speech] I can help.\n<Action>a1</Action>"
        self.assertEqual(response_length_chars(text), len(text))
        self.assertEqual(response_length_words(text), 5)

    def test_tag_detection_is_case_insensitive(self):
        self.assertTrue(has_speech_tag("[speech] hi"))
        self.assertTrue(has_action_tag("<action>a1</action>"))
        self.assertFalse(has_action_tag("Action: a1"))

    def test_build_side_effect_row_valid_response(self):
        row = build_side_effect_row(
            "[Speech] I can help.\n<Action>a1</Action>",
            valid_actions=["a1", "a2"],
            condition="baseline",
            scenario_id="s1",
            trait="agreeableness",
            target_direction="high",
            coherence_score=0.8,
            perplexity=12.5,
        )
        self.assertEqual(row["condition"], "baseline")
        self.assertEqual(row["parsed_action"], "a1")
        self.assertTrue(row["parse_success"])
        self.assertTrue(row["format_validity"])
        self.assertEqual(row["coherence_score"], 0.8)
        self.assertEqual(row["perplexity"], 12.5)

    def test_invalid_action_fails_format_validity(self):
        row = build_side_effect_row(
            "[Speech] I can help.\n<Action>unknown</Action>",
            valid_actions=["a1"],
        )
        self.assertFalse(row["parse_success"])
        self.assertFalse(row["format_validity"])

    def test_missing_speech_can_be_allowed(self):
        row = build_side_effect_row(
            "<Action>a1</Action>",
            valid_actions=["a1"],
            require_speech=False,
        )
        self.assertTrue(row["parse_success"])
        self.assertTrue(row["format_validity"])
        self.assertFalse(format_validity(row, require_speech=True))

    def test_summary(self):
        rows = [
            build_side_effect_row("[Speech] Hi\n<Action>a1</Action>", ["a1"]),
            build_side_effect_row("[Speech] Hi\n<Action>bad</Action>", ["a1"]),
        ]
        summary = summarize_side_effect_rows(rows)
        self.assertEqual(summary["n"], 2.0)
        self.assertEqual(summary["parse_success_rate"], 0.5)
        self.assertEqual(summary["format_validity_rate"], 0.5)

    def test_empty_summary(self):
        summary = summarize_side_effect_rows([])
        self.assertEqual(summary["n"], 0.0)
        self.assertEqual(summary["format_validity_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
