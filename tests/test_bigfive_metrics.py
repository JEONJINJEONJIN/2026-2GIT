import unittest

from src.evaluation.bigfive_metrics import (
    mean_consistency_by_trait,
    pas_loglik_agreement,
    speech_action_consistency,
    target_attainment,
    target_direction_score,
    trait_high_probability_mass,
)


class TestBigFiveMetrics(unittest.TestCase):
    def test_speech_action_consistency(self):
        self.assertAlmostEqual(speech_action_consistency(0.8, 0.7), 0.9)
        self.assertAlmostEqual(speech_action_consistency(0.0, 1.0), 0.0)

    def test_speech_action_consistency_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            speech_action_consistency(1.2, 0.5)

    def test_target_direction_score(self):
        self.assertAlmostEqual(target_direction_score(0.8, "high"), 0.8)
        self.assertAlmostEqual(target_direction_score(0.8, "low"), 0.2)

    def test_target_attainment_is_signed_from_neutral(self):
        self.assertAlmostEqual(target_attainment(0.8, "high"), 0.3)
        self.assertAlmostEqual(target_attainment(0.2, "low"), 0.3)
        self.assertAlmostEqual(target_attainment(0.8, "low"), -0.3)

    def test_trait_high_probability_mass_normalizes_weights(self):
        probs = {"a": 2.0, "b": 1.0, "c": 1.0}
        directions = {"a": "high", "b": "low", "c": "high"}
        self.assertAlmostEqual(trait_high_probability_mass(probs, directions), 0.75)

    def test_trait_high_probability_mass_requires_directions(self):
        with self.assertRaises(ValueError):
            trait_high_probability_mass({"a": 1.0}, {})

    def test_pas_loglik_agreement(self):
        self.assertTrue(pas_loglik_agreement("a1", "a1"))
        self.assertFalse(pas_loglik_agreement("a1", "a2"))
        self.assertFalse(pas_loglik_agreement(None, "a2"))

    def test_mean_consistency_by_trait(self):
        rows = [
            {"trait": "agreeableness", "consistency": 0.8},
            {"trait": "agreeableness", "consistency": 1.0},
            {"trait": "openness", "consistency": 0.5},
        ]
        result = mean_consistency_by_trait(rows)
        self.assertAlmostEqual(result["agreeableness"], 0.9)
        self.assertAlmostEqual(result["openness"], 0.5)


if __name__ == "__main__":
    unittest.main()

