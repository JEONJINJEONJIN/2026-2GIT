import unittest

from src.evaluation.paraphrase_robustness import summarize_paraphrase_robustness


class TestParaphraseRobustness(unittest.TestCase):
    def test_summarizes_action_and_score_stability(self):
        rows = [
            {
                "condition": "baseline",
                "trait": "agreeableness",
                "target_direction": "high",
                "paraphrase_group_id": "g1",
                "paraphrase_variant": "original",
                "loglik_selected_action": "a1",
                "trait_score": "0.70",
            },
            {
                "condition": "baseline",
                "trait": "agreeableness",
                "target_direction": "high",
                "paraphrase_group_id": "g1",
                "paraphrase_variant": "p1",
                "loglik_selected_action": "a1",
                "trait_score": "0.60",
            },
            {
                "condition": "baseline",
                "trait": "agreeableness",
                "target_direction": "high",
                "paraphrase_group_id": "g1",
                "paraphrase_variant": "p2",
                "loglik_selected_action": "a2",
                "trait_score": "0.50",
            },
        ]

        summary = summarize_paraphrase_robustness(rows)

        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n_paraphrase_groups"], 1)
        self.assertAlmostEqual(summary[0]["mean_action_agreement"], 2 / 3)
        self.assertAlmostEqual(summary[0]["mean_trait_score_range"], 0.2)

    def test_ignores_singletons_and_missing_group_ids(self):
        rows = [
            {
                "condition": "baseline",
                "trait": "agreeableness",
                "target_direction": "high",
                "paraphrase_group_id": "g1",
                "loglik_selected_action": "a1",
                "trait_score": "0.70",
            },
            {
                "condition": "baseline",
                "trait": "agreeableness",
                "target_direction": "high",
                "paraphrase_group_id": "",
                "loglik_selected_action": "a1",
                "trait_score": "0.70",
            },
        ]

        self.assertEqual(summarize_paraphrase_robustness(rows), [])


if __name__ == "__main__":
    unittest.main()
