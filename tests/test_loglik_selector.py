"""Tests for conditional log-likelihood action selection."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.loglik_selector import select_action_by_loglik
from src.generation.prompt_builder import PromptBuilder


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"P": 0, " ": 1, "A": 2, "B": 3}

    def __call__(self, text, return_tensors="pt"):
        ids = [self.vocab[ch] for ch in text]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join(message["content"] for message in messages)
        return text + " "


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        logits = torch.zeros(batch, seq_len, 4, device=input_ids.device)
        logits[..., 2] = 3.0
        logits[..., 3] = 1.0
        return type("Output", (), {"logits": logits})


class TestLoglikSelector(unittest.TestCase):
    def test_selects_highest_likelihood_action(self):
        result = select_action_by_loglik(
            FakeModel(),
            FakeTokenizer(),
            "P ",
            ["A", "B"],
        )

        self.assertEqual(result["selected_idx"], 0)
        self.assertGreater(result["logliks"][0], result["logliks"][1])
        self.assertAlmostEqual(sum(result["softmax_probs"]), 1.0, places=5)

    def test_supports_chat_message_prompt(self):
        result = select_action_by_loglik(
            FakeModel(),
            FakeTokenizer(),
            [{"role": "user", "content": "P"}],
            ["A", "B"],
        )

        self.assertEqual(result["selected_idx"], 0)


class TestDescriptionScoringPrompt(unittest.TestCase):
    def test_action_choice_prompt_uses_descriptions(self):
        builder = PromptBuilder()
        messages = builder.build_action_choice_messages(
            {
                "genre": "fantasy",
                "context": "A guard blocks the gate.",
                "actions": [
                    {
                        "id": "act_001",
                        "description": "Threaten the guard to move aside.",
                    },
                    {
                        "id": "act_002",
                        "description": "Ask the guard for a peaceful compromise.",
                    },
                ],
            }
        )

        system_content = messages[0]["content"]
        self.assertIn("Threaten the guard", system_content)
        self.assertIn("peaceful compromise", system_content)
        self.assertNotIn("act_001", system_content)
        self.assertNotIn("action ID", system_content)


if __name__ == "__main__":
    unittest.main()
