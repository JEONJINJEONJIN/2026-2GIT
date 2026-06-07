import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "15_score_side_effects.py"
SPEC = importlib.util.spec_from_file_location("score_side_effects", SCRIPT_PATH)
score_side_effects = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(score_side_effects)


class TestSideEffectScript(unittest.TestCase):
    def test_score_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "responses.csv"
            output_path = tmp_path / "side_effects.csv"
            with open(input_path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=[
                        "condition",
                        "scenario_id",
                        "trait",
                        "target_direction",
                        "raw",
                        "valid_actions",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "condition": "baseline",
                        "scenario_id": "s1",
                        "trait": "agreeableness",
                        "target_direction": "high",
                        "raw": "[Speech] Hi\n<Action>a1</Action>",
                        "valid_actions": "a1|a2",
                    }
                )

            count, summary = score_side_effects.score_csv(input_path, output_path)
            output_text = output_path.read_text(encoding="utf-8")

        self.assertEqual(count, 1)
        self.assertEqual(summary["format_validity_rate"], 1.0)
        self.assertIn("format_validity", output_text)


if __name__ == "__main__":
    unittest.main()
