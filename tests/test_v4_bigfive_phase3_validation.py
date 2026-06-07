import tempfile
import unittest
from pathlib import Path

import yaml

from src.experiments.v4_bigfive_phase3_validation import (
    check_final_config,
    check_final_tuning,
    downgrade_tuning_blockers,
)
from src.experiments.v4_bigfive_readiness import ReadinessCheck


class TestV4BigFivePhase3Validation(unittest.TestCase):
    def test_check_final_config_accepts_expected_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final.yaml"
            self._write_config(path, alpha=1.0, layers=[18, 21, 24])

            checks = check_final_config(path)

        self.assertEqual(checks[0].status, "ok")

    def test_check_final_config_rejects_pilot_trait_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final.yaml"
            self._write_config(path, traits=["agreeableness"])

            checks = check_final_config(path)

        self.assertEqual(checks[0].status, "error")
        self.assertIn("all Big Five", checks[0].details)

    def test_check_final_tuning_requires_alpha_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final.yaml"
            self._write_config(path, alpha=None, layers=None)

            checks = check_final_tuning(path)

        self.assertEqual(checks[0].status, "error")
        self.assertIn("alpha", checks[0].details)

    def test_check_final_tuning_accepts_fixed_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final.yaml"
            self._write_config(path, alpha=1.0, layers=[18, 21, 24])

            checks = check_final_tuning(path)

        self.assertEqual(checks[0].status, "ok")

    def test_downgrade_tuning_blockers(self):
        checks = [
            ReadinessCheck("phase3_alpha_layers", "error", "missing"),
            ReadinessCheck("phase3_final_config", "error", "bad"),
        ]

        downgraded = downgrade_tuning_blockers(checks)

        self.assertEqual(downgraded[0].status, "warn")
        self.assertEqual(downgraded[1].status, "error")

    def _write_config(
        self,
        path: Path,
        traits: list[str] | None = None,
        split: str = "test",
        alpha=None,
        layers=None,
    ) -> None:
        traits = traits or [
            "agreeableness",
            "conscientiousness",
            "neuroticism",
            "openness",
            "extraversion",
        ]
        path.write_text(
            yaml.safe_dump(
                {
                    "experiment": {
                        "name": "v4_bigfive_final",
                        "version": "v4_bigfive",
                        "split": split,
                        "traits": traits,
                        "target_directions": ["high", "low"],
                        "conditions": [
                            {
                                "name": "baseline",
                                "prompt_style": "none",
                                "steering": False,
                                "pas": False,
                            },
                            {
                                "name": "one_line_prompt",
                                "prompt_style": "one_line",
                                "steering": False,
                                "pas": False,
                            },
                            {
                                "name": "elaborate_prompt",
                                "prompt_style": "elaborate",
                                "steering": False,
                                "pas": False,
                            },
                            {
                                "name": "as_pas_only",
                                "prompt_style": "none",
                                "steering": True,
                                "pas": True,
                                "vector_mode": "trait_contrast",
                            },
                            {
                                "name": "elaborate_prompt_as_pas",
                                "prompt_style": "elaborate",
                                "steering": True,
                                "pas": True,
                                "vector_mode": "trait_contrast",
                            },
                        ],
                        "steering": {
                            "alpha": alpha,
                            "layers": layers,
                            "tune_on_this_split": False,
                        },
                    }
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
