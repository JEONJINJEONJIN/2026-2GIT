import unittest

from src.generation.prompt_builder import (
    PromptBuilder,
    normalize_bigfive_trait,
    normalize_prompt_style,
    normalize_trait_direction,
)


class TestBigFivePromptBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = PromptBuilder()
        self.scenario = {
            "genre": "fantasy_rpg",
            "context": "A villager asks the NPC to mediate a dispute.",
            "actions": [
                {"id": "a1", "description": "Listen to both sides."},
                {"id": "a2", "description": "Dismiss the request."},
            ],
        }

    def test_none_prompt_style_uses_neutral_template(self):
        messages = self.builder.build_chat_messages(
            self.scenario,
            prompt_style="none",
            trait="agreeableness",
            target_direction="high",
        )
        self.assertNotIn("highly agreeable", messages[0]["content"])
        self.assertIn("You are an NPC", messages[0]["content"])

    def test_one_line_bigfive_prompt(self):
        messages = self.builder.build_chat_messages(
            self.scenario,
            prompt_style="one_line",
            trait="agreeableness",
            target_direction="high",
        )
        self.assertIn("highly agreeable NPC", messages[0]["content"])
        self.assertIn("Available actions", messages[0]["content"])
        self.assertIn("Return exactly two lines", messages[1]["content"])
        self.assertIn("Valid action IDs: a1, a2", messages[1]["content"])
        self.assertIn("<Action>one_action_id</Action>", messages[1]["content"])

    def test_elaborate_bigfive_action_choice_prompt(self):
        messages = self.builder.build_action_choice_messages(
            self.scenario,
            prompt_style="elaborate",
            trait="openness",
            target_direction="low",
        )
        self.assertIn("low-openness persona", messages[0]["content"])
        self.assertIn("Return only the chosen action text", messages[0]["content"])
        self.assertIn("Listen to both sides.", messages[0]["content"])

    def test_legacy_persona_prompt_still_works(self):
        messages = self.builder.build_chat_messages(
            self.scenario,
            persona="aggressive",
            include_persona=True,
        )
        self.assertIn("aggressive NPC", messages[0]["content"])

    def test_invalid_prompt_style_rejected(self):
        with self.assertRaises(ValueError):
            normalize_prompt_style("verbose")

    def test_invalid_trait_rejected(self):
        with self.assertRaises(ValueError):
            normalize_bigfive_trait("humor")

    def test_invalid_direction_rejected(self):
        with self.assertRaises(ValueError):
            normalize_trait_direction("middle")


if __name__ == "__main__":
    unittest.main()
