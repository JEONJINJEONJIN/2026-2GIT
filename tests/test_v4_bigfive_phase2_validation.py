import tempfile
import unittest
from pathlib import Path

import yaml

from src.experiments.v4_bigfive_phase2_validation import (
    check_pilot_config,
    downgrade_known_external_blockers,
)
from src.experiments.v4_bigfive_readiness import ReadinessCheck


class TestV4BigFivePhase2Validation(unittest.TestCase):
    def test_check_pilot_config_accepts_expected_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pilot.yaml"
            self._write_config(path)

            checks = check_pilot_config(path)

        self.assertEqual(checks[0].status, "ok")

    def test_check_pilot_config_rejects_missing_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pilot.yaml"
            self._write_config(path, include_as_pas=False)

            checks = check_pilot_config(path)

        self.assertEqual(checks[0].status, "error")
        self.assertIn("missing", checks[0].details)

    def test_downgrade_known_external_blockers(self):
        checks = [
            ReadinessCheck("trait_scenarios", "error", "missing"),
            ReadinessCheck("config", "error", "bad"),
        ]

        downgraded = downgrade_known_external_blockers(checks)

        self.assertEqual(downgraded[0].status, "warn")
        self.assertEqual(downgraded[1].status, "error")

    def _write_config(self, path: Path, include_as_pas: bool = True) -> None:
        conditions = [
            {"name": "baseline", "prompt_style": "none", "steering": False, "pas": False},
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
        ]
        if include_as_pas:
            conditions.extend(
                [
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
                ]
            )
        path.write_text(
            yaml.safe_dump(
                {
                    "experiment": {
                        "name": "v4_bigfive_pilot",
                        "version": "v4_bigfive",
                        "split": "dev",
                        "traits": ["agreeableness"],
                        "target_directions": ["high", "low"],
                        "conditions": conditions,
                    }
                }
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
