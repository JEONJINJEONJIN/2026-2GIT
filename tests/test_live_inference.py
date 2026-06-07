import unittest

import src.demo.live_inference as live_inference
from src.data.trait_dataset import TraitAction, TraitScenario
from src.demo.live_inference import (
    LiveInferenceEngine,
    filter_scenarios_by_trait,
    parse_action_id,
    select_pas_score,
)
from src.demo.live_inference_tab import coerce_trait_scenario
from src.demo.bigfive_demo_data import sample_scenarios


class TestLiveInferenceHelpers(unittest.TestCase):
    def test_parse_action_id(self):
        self.assertEqual(
            parse_action_id("[Speech] hi\n<Action>act_1</Action>"),
            "act_1",
        )
        self.assertEqual(parse_action_id("no structured action"), "")

    def test_filter_scenarios_by_trait(self):
        scenarios = [
            self._scenario("s1", "agreeableness"),
            self._scenario("s2", "openness"),
        ]

        filtered = filter_scenarios_by_trait(scenarios, "openness")

        self.assertEqual([scenario.scenario_id for scenario in filtered], ["s2"])

    def test_select_pas_score_uses_generated_action_when_available(self):
        score, top_action, used_fallback = select_pas_score(
            {"a1": 0.1, "a2": 0.4},
            [("a2", 0.4), ("a1", 0.1)],
            "a1",
        )

        self.assertAlmostEqual(score, 0.1)
        self.assertEqual(top_action, "a2")
        self.assertFalse(used_fallback)

    def test_select_pas_score_falls_back_to_top_action(self):
        score, top_action, used_fallback = select_pas_score(
            {"a1": 0.1, "a2": 0.4},
            [("a2", 0.4), ("a1", 0.1)],
            "",
        )

        self.assertAlmostEqual(score, 0.4)
        self.assertEqual(top_action, "a2")
        self.assertTrue(used_fallback)

    def test_coerce_demo_scenario_to_trait_scenario(self):
        scenario = coerce_trait_scenario(sample_scenarios()[0])

        self.assertIsInstance(scenario, TraitScenario)
        self.assertEqual(scenario.split, "test")
        self.assertEqual(len(scenario.actions), 2)

    def test_steering_hooks_clear_on_exception(self):
        created = []

        class FakeInjector:
            def __init__(self, model, vectors, normalize_vectors=True):
                self.clear_count = 0
                created.append(self)

            def inject(self, layers, alpha):
                self.layers = layers
                self.alpha = alpha

            def clear(self):
                self.clear_count += 1

        original = live_inference.SteeringInjector
        live_inference.SteeringInjector = FakeInjector
        try:
            engine = LiveInferenceEngine.__new__(LiveInferenceEngine)
            engine.model = object()
            engine._active_hook_contexts = 0

            with self.assertRaises(RuntimeError):
                with engine._steering_hooks({0: object()}, [0], 1.0):
                    raise RuntimeError("boom")

            self.assertEqual(created[0].clear_count, 1)
            self.assertEqual(engine._active_hook_contexts, 0)
        finally:
            live_inference.SteeringInjector = original

    @staticmethod
    def _scenario(scenario_id: str, trait: str) -> TraitScenario:
        return TraitScenario(
            scenario_id=scenario_id,
            trait=trait,
            prompt="A situation.",
            split="test",
            actions=(
                TraitAction("a1", "Do the high thing.", "high"),
                TraitAction("a2", "Do the low thing.", "low"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
