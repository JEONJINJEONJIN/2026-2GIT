import json
import tempfile
import unittest
from pathlib import Path

from src.data.trait_dataset import (
    build_trait_contrastive_pairs,
    flatten_trait_actions,
    load_trait_scenarios,
    write_jsonl,
)


class TestTraitDataset(unittest.TestCase):
    def test_load_trait_scenarios_filters_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.jsonl"
            rows = [
                self._row("s1", "train"),
                self._row("s2", "dev"),
            ]
            path.write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            scenarios = load_trait_scenarios(path, allowed_splits=["dev"])
        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0].scenario_id, "s2")

    def test_load_trait_scenarios_requires_file(self):
        with self.assertRaises(FileNotFoundError):
            load_trait_scenarios("missing_trait_file.jsonl")

    def test_load_trait_scenarios_validates_four_actions(self):
        row = self._row("s1", "train")
        row["actions"] = row["actions"][:3]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_trait_scenarios(path)

    def test_flatten_trait_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.jsonl"
            path.write_text(json.dumps(self._row("s1", "train")) + "\n", encoding="utf-8")
            scenarios = load_trait_scenarios(path)
        actions = flatten_trait_actions(scenarios)
        self.assertEqual(len(actions), 4)
        self.assertEqual(actions[0]["scenario_id"], "s1")

    def test_build_trait_contrastive_pairs_uses_high_as_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenarios.jsonl"
            path.write_text(json.dumps(self._row("s1", "train")) + "\n", encoding="utf-8")
            scenarios = load_trait_scenarios(path)
        pairs = build_trait_contrastive_pairs(scenarios)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(
            {pair["positive_direction"] for pair in pairs},
            {"high"},
        )
        self.assertEqual(
            {pair["negative_direction"] for pair in pairs},
            {"low"},
        )
        self.assertIn("high-agreeableness NPC", pairs[0]["positive"])
        self.assertIn("low-agreeableness NPC", pairs[0]["negative"])

    def test_write_jsonl_returns_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            count = write_jsonl([{"a": 1}, {"b": 2}], path)
            rows = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(count, 2)
        self.assertEqual(len(rows), 2)

    @staticmethod
    def _row(scenario_id, split):
        return {
            "scenario_id": scenario_id,
            "trait": "agreeableness",
            "prompt": "A teammate asks for help during a difficult task.",
            "split": split,
            "actions": [
                {"id": f"{scenario_id}_a", "text": "Offer help.", "trait_direction": "high"},
                {"id": f"{scenario_id}_b", "text": "Listen first.", "trait_direction": "high"},
                {"id": f"{scenario_id}_c", "text": "Refuse quickly.", "trait_direction": "low"},
                {"id": f"{scenario_id}_d", "text": "Ignore them.", "trait_direction": "low"},
            ],
        }


if __name__ == "__main__":
    unittest.main()
