import unittest

from src.demo.bigfive_demo_data import (
    CONFLICT_SCENARIO_PRESET_IDS,
    DEMO_PERSONA_PRESETS,
    activation_points,
    choose_demo_scenario,
    load_demo_scenarios,
    prioritize_conflict_scenarios,
    response_pair_for_scenario,
    sample_responses,
    sample_scenarios,
)


class TestBigFiveDemoData(unittest.TestCase):
    def test_choose_demo_scenario_uses_largest_trait_slider(self):
        scenario = choose_demo_scenario(
            sample_scenarios(),
            {
                "agreeableness": 0.2,
                "conscientiousness": 0.4,
                "neuroticism": 0.1,
                "openness": 0.9,
                "extraversion": 0.3,
            },
        )

        self.assertEqual(scenario.trait, "openness")

    def test_response_pair_for_scenario(self):
        scenario = sample_scenarios()[0]
        prompt_response, as_pas_response = response_pair_for_scenario(
            sample_responses(),
            scenario,
        )

        self.assertEqual(prompt_response.condition, "elaborate_prompt")
        self.assertEqual(as_pas_response.condition, "elaborate_prompt_as_pas")
        self.assertEqual(prompt_response.action, "a1")

    def test_activation_points(self):
        points = activation_points(
            {
                "agreeableness": 0.8,
                "conscientiousness": 0.2,
                "neuroticism": 0.3,
                "openness": 0.9,
                "extraversion": 0.7,
            }
        )

        self.assertEqual([point["label"] for point in points], ["baseline", "prompt", "AS+PAS"])
        self.assertGreater(points[-1]["x"], points[0]["x"])

    def test_demo_persona_presets_are_distinctive_and_complete(self):
        self.assertGreaterEqual(len(DEMO_PERSONA_PRESETS), 5)
        seen = set()
        for preset in DEMO_PERSONA_PRESETS:
            self.assertIn("label", preset)
            values = preset["values"]
            self.assertEqual(
                set(values),
                {
                    "agreeableness",
                    "conscientiousness",
                    "neuroticism",
                    "openness",
                    "extraversion",
                },
            )
            self.assertGreaterEqual(max(abs(value - 0.5) for value in values.values()), 0.3)
            seen.add(preset["id"])
        self.assertEqual(len(seen), len(DEMO_PERSONA_PRESETS))

    def test_conflict_scenario_presets_exist_and_are_prioritized(self):
        scenarios = load_demo_scenarios("data/trait_bigfive/scenarios.jsonl")
        scenario_ids = {scenario.scenario_id for scenario in scenarios}

        self.assertGreaterEqual(len(CONFLICT_SCENARIO_PRESET_IDS), 5)
        self.assertTrue(set(CONFLICT_SCENARIO_PRESET_IDS).issubset(scenario_ids))
        prioritized = prioritize_conflict_scenarios(scenarios)
        self.assertEqual(
            [scenario.scenario_id for scenario in prioritized[: len(CONFLICT_SCENARIO_PRESET_IDS)]],
            list(CONFLICT_SCENARIO_PRESET_IDS),
        )


if __name__ == "__main__":
    unittest.main()
