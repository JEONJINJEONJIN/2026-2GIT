import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch

from src.data.bfi import BFIItem
from src.data.trait_dataset import TraitAction, TraitScenario
from src.experiments.v4_bigfive_runner import (
    RuntimeCondition,
    apply_pas_final_selection,
    build_bfi_messages,
    combine_consistency_row,
    generate_trait_scenario_response,
    parse_likert_response,
    resolve_condition_persona_vectors,
    resolve_trait_persona_vectors,
    score_bfi_with_generator,
    score_trait_scenario_by_loglik,
    selected_layer_for_pas,
    write_csv_rows,
)


class TestV4BigFiveRunner(unittest.TestCase):
    def test_build_bfi_messages_requires_item_text(self):
        item = BFIItem("bfi_01", "agreeableness", text="")
        with self.assertRaises(ValueError):
            build_bfi_messages(
                item,
                trait="agreeableness",
                target_direction="high",
                prompt_style="one_line",
            )

    def test_build_bfi_messages_includes_persona_text(self):
        item = BFIItem("bfi_01", "agreeableness", text="I am helpful.")
        messages = build_bfi_messages(
            item,
            trait="agreeableness",
            target_direction="high",
            prompt_style="one_line",
        )
        self.assertIn("highly agreeable NPC", messages[0]["content"])
        self.assertIn("I am helpful.", messages[1]["content"])

    def test_parse_likert_response(self):
        self.assertEqual(parse_likert_response("4"), 4)
        self.assertEqual(parse_likert_response("Answer: 2"), 2)
        with self.assertRaises(ValueError):
            parse_likert_response("agree")

    def test_score_bfi_with_generator_applies_steering(self):
        class FakeGenerator:
            model = object()

            def generate(self, messages, **kwargs):
                return "4"

        items = [BFIItem("bfi_01", "agreeableness", text="I am helpful.")]
        vectors = {1: torch.tensor([1.0, 0.0])}
        with patch("src.experiments.v4_bigfive_runner.SteeringInjector") as injector:
            scores, rows = score_bfi_with_generator(
                FakeGenerator(),
                items,
                trait="agreeableness",
                target_direction="high",
                prompt_style="none",
                persona_vectors=vectors,
                layer_indices=[1],
                alpha=4.0,
                use_steering=True,
            )

        injector.assert_called_once()
        injector.return_value.inject.assert_called_once_with([1], alpha=4.0)
        injector.return_value.clear.assert_called_once()
        self.assertAlmostEqual(scores["agreeableness"], 0.75)
        self.assertTrue(rows[0]["steering_applied"])

    def test_resolve_trait_persona_vectors_applies_low_sign(self):
        bundle = {
            "vectors": {
                "trait_contrast": {
                    "agreeableness": {
                        1: torch.tensor([1.0, 2.0]),
                    }
                }
            }
        }
        high = resolve_trait_persona_vectors(bundle, "agreeableness", "high")
        low = resolve_trait_persona_vectors(bundle, "agreeableness", "low")
        self.assertTrue(torch.equal(high[1], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(low[1], torch.tensor([-1.0, -2.0])))

    def test_resolve_condition_persona_vectors_supports_zero_mode(self):
        bundle = {
            "vectors": {
                "trait_contrast": {
                    "agreeableness": {
                        1: torch.tensor([1.0, 2.0]),
                    }
                }
            }
        }
        zero = resolve_condition_persona_vectors(
            bundle,
            trait="agreeableness",
            target_direction="high",
            vector_mode="zero",
        )
        self.assertTrue(torch.equal(zero[1], torch.tensor([0.0, 0.0])))

    def test_combine_consistency_row(self):
        row = combine_consistency_row(
            {"agreeableness": 0.8},
            {
                "condition": "baseline",
                "scenario_id": "s1",
                "trait": "agreeableness",
                "target_direction": "high",
                "trait_score": 0.7,
                "trait_target_score": 0.7,
                "trait_target_attainment": 0.2,
            },
        )
        self.assertAlmostEqual(row["consistency"], 0.9)
        self.assertAlmostEqual(row["bfi_target_attainment"], 0.3)

    def test_score_trait_scenario_by_loglik_uses_high_mass(self):
        scenario = self._scenario()
        condition = RuntimeCondition(
            name="baseline",
            prompt_style="none",
            steering=False,
            pas=False,
        )
        fake_selection = {
            "selected_idx": 0,
            "logliks": [0.0, -1.0, -2.0, -3.0],
            "normalized_logliks": [0.0, -1.0, -2.0, -3.0],
            "softmax_probs": [0.4, 0.1, 0.3, 0.2],
        }
        with patch(
            "src.experiments.v4_bigfive_runner.select_action_by_loglik",
            return_value=fake_selection,
        ):
            row = score_trait_scenario_by_loglik(
                model=object(),
                tokenizer=object(),
                scenario=scenario,
                condition=condition,
                target_direction="high",
            )
        self.assertEqual(row["loglik_selected_action"], "s1_a")
        self.assertAlmostEqual(row["trait_score"], 0.5)
        self.assertAlmostEqual(row["trait_target_score"], 0.5)
        self.assertEqual(row["final_selected_action"], "s1_a")
        self.assertEqual(row["selection_policy"], "loglik")

    def test_apply_pas_final_selection_overwrites_trait_score(self):
        loglik_row = {
            "condition": "as_pas_final",
            "scenario_id": "s1",
            "trait": "agreeableness",
            "target_direction": "low",
            "selection_policy": "loglik",
            "loglik_selected_action": "s1_a",
            "loglik_selected_direction": "high",
            "final_selected_action": "s1_a",
            "final_selected_direction": "high",
            "trait_score": 1.0,
            "trait_target_score": 0.0,
            "trait_target_attainment": -0.5,
        }
        pas_row = {
            "pas_selected_action": "s1_c",
            "pas_selected_direction": "low",
        }

        row = apply_pas_final_selection(loglik_row, pas_row)

        self.assertEqual(row["selection_policy"], "pas_final")
        self.assertEqual(row["final_selected_action"], "s1_c")
        self.assertEqual(row["final_selected_direction"], "low")
        self.assertEqual(row["pas_selected_action"], "s1_c")
        self.assertAlmostEqual(row["trait_score"], 0.0)
        self.assertAlmostEqual(row["trait_target_score"], 1.0)
        self.assertAlmostEqual(row["trait_target_attainment"], 0.5)

    def test_generate_trait_scenario_response(self):
        class FakeGenerator:
            def generate(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return "[Speech] I can help.\n<Action>s1_a</Action>"

        generator = FakeGenerator()
        scenario = self._scenario()
        condition = RuntimeCondition(
            name="one_line_prompt",
            prompt_style="one_line",
            steering=False,
            pas=False,
        )

        row = generate_trait_scenario_response(
            generator,
            model=object(),
            scenario=scenario,
            condition=condition,
            target_direction="high",
            max_new_tokens=32,
        )

        self.assertEqual(row["condition"], "one_line_prompt")
        self.assertEqual(row["valid_actions"], "s1_a|s1_b|s1_c|s1_d")
        self.assertEqual(json.loads(row["action_directions_json"])["s1_a"], "high")
        self.assertIn("highly agreeable NPC", generator.messages[0]["content"])
        self.assertEqual(generator.kwargs["max_new_tokens"], 32)
        self.assertIn("<Action>s1_a</Action>", row["raw"])

    def test_selected_layer_for_pas_uses_middle_sorted_layer(self):
        self.assertEqual(selected_layer_for_pas([24, 18, 21]), 21)
        with self.assertRaises(ValueError):
            selected_layer_for_pas([])

    def test_write_csv_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            count = write_csv_rows([{"a": 1, "b": 2}, {"a": 3, "c": 4}], path)
            text = path.read_text(encoding="utf-8")
        self.assertEqual(count, 2)
        self.assertIn("a,b,c", text.splitlines()[0])

    @staticmethod
    def _scenario():
        return TraitScenario(
            scenario_id="s1",
            trait="agreeableness",
            prompt="A teammate asks for help.",
            split="dev",
            actions=(
                TraitAction("s1_a", "Offer help.", "high"),
                TraitAction("s1_b", "Listen first.", "high"),
                TraitAction("s1_c", "Refuse.", "low"),
                TraitAction("s1_d", "Ignore.", "low"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
