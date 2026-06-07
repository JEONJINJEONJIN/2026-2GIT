import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestTraitPairScript(unittest.TestCase):
    def test_script_writes_per_trait_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scenarios_path = tmp_path / "scenarios.jsonl"
            output_dir = tmp_path / "pairs"
            scenarios_path.write_text(
                json.dumps(self._row("s1", "train")) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/09_build_trait_contrastive_pairs.py",
                    "--scenarios_path",
                    str(scenarios_path),
                    "--output_dir",
                    str(output_dir),
                    "--traits",
                    "agreeableness",
                    "--splits",
                    "train",
                ],
                cwd=Path(__file__).resolve().parent.parent,
                check=True,
                capture_output=True,
                text=True,
            )

            trait_path = output_dir / "agreeableness.jsonl"
            all_path = output_dir / "all_traits.jsonl"
            self.assertTrue(trait_path.exists())
            self.assertTrue(all_path.exists())
            self.assertIn("Wrote 4 agreeableness pairs", result.stdout)
            self.assertEqual(len(trait_path.read_text(encoding="utf-8").splitlines()), 4)

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
