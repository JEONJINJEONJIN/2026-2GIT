import json
import tempfile
import unittest
from pathlib import Path

import yaml
import torch

from src.data.trait_dataset import write_jsonl
from src.experiments.v4_bigfive_readiness import (
    readiness_has_errors,
    run_readiness_checks,
)


class TestV4BigFiveReadiness(unittest.TestCase):
    def test_readiness_reports_missing_bfi_text_as_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._write_fixture(root, bfi_text="")
            checks = run_readiness_checks(config, root)

        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["config"].status, "ok")
        self.assertEqual(by_name["bfi_item_text"].status, "error")
        self.assertTrue(readiness_has_errors(checks))

    def test_readiness_errors_when_configured_text_overlay_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._write_fixture(
                root,
                bfi_text="",
                bfi_item_text_path="data/bfi/missing.local.csv",
            )
            checks = run_readiness_checks(config, root)

        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["bfi_item_text_overlay"].status, "error")
        self.assertTrue(readiness_has_errors(checks))

    def test_readiness_uses_text_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overlay_path = root / "data" / "bfi" / "bfi44_item_text.local.csv"
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.write_text(
                "\n".join(
                    [
                        "item_id,text",
                        "bfi_01,I am helpful.",
                        "bfi_02,I am organized.",
                        "bfi_03,I worry often.",
                        "bfi_04,I like new ideas.",
                        "bfi_05,I talk often.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = self._write_fixture(
                root,
                bfi_text="",
                bfi_item_text_path="data/bfi/bfi44_item_text.local.csv",
            )
            checks = run_readiness_checks(config, root)

        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["bfi_item_text"].status, "ok")
        self.assertEqual(by_name["bfi_item_text_overlay"].status, "ok")

    def test_readiness_ok_without_vectors_when_no_as_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._write_fixture(root, bfi_text="I am helpful.", as_condition=False)
            checks = run_readiness_checks(config, root)

        statuses = {check.name: check.status for check in checks}
        self.assertEqual(statuses["bfi_item_text"], "ok")
        self.assertEqual(statuses["trait_split_coverage"], "ok")
        self.assertEqual(statuses["vectors"], "ok")
        self.assertFalse(readiness_has_errors(checks))

    def test_readiness_warns_when_vectors_required_but_missing_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._write_fixture(root, bfi_text="I am helpful.", as_condition=True)
            checks = run_readiness_checks(config, root)

        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["vectors"].status, "warn")
        self.assertFalse(readiness_has_errors(checks))

    def test_readiness_errors_when_unrelated_control_has_one_trait_vector(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = self._write_fixture(
                root,
                bfi_text="I am helpful.",
                as_condition=True,
                vector_mode="unrelated_trait",
            )
            vectors_path = root / "vectors.pt"
            torch.save(
                {
                    "vectors": {
                        "trait_contrast": {
                            "agreeableness": {
                                18: torch.tensor([1.0, 0.0]),
                            }
                        }
                    }
                },
                vectors_path,
            )
            checks = run_readiness_checks(
                config,
                root,
                vectors_path="vectors.pt",
            )

        by_name = {check.name: check for check in checks}
        self.assertEqual(by_name["vectors"].status, "error")
        self.assertIn("at least two trait vectors", by_name["vectors"].details)

    def _write_fixture(
        self,
        root: Path,
        bfi_text: str,
        as_condition: bool = False,
        vector_mode: str = "trait_contrast",
        bfi_item_text_path: str | None = None,
    ) -> Path:
        bfi_path = root / "data" / "bfi" / "bfi44_scoring.csv"
        bfi_path.parent.mkdir(parents=True, exist_ok=True)
        bfi_path.write_text(
            "\n".join(
                [
                    "item_id,trait,reverse_scored,text",
                    f"bfi_01,agreeableness,false,{bfi_text}",
                    "bfi_02,conscientiousness,false,I am organized.",
                    "bfi_03,neuroticism,false,I worry often.",
                    "bfi_04,openness,false,I like new ideas.",
                    "bfi_05,extraversion,false,I talk often.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        scenarios_path = root / "data" / "trait_bigfive" / "scenarios.jsonl"
        rows = [
            {
                "scenario_id": "s1",
                "trait": "agreeableness",
                "prompt": "A teammate asks for help.",
                "split": "dev",
                "actions": [
                    {"id": "s1_a", "text": "Help.", "trait_direction": "high"},
                    {"id": "s1_b", "text": "Listen.", "trait_direction": "high"},
                    {"id": "s1_c", "text": "Refuse.", "trait_direction": "low"},
                    {"id": "s1_d", "text": "Ignore.", "trait_direction": "low"},
                ],
            }
        ]
        write_jsonl(rows, scenarios_path)

        config = {
            "experiment": {
                "name": "readiness_test",
                "version": "v4_bigfive",
                "split": "dev",
                "traits": ["agreeableness"],
                "target_directions": ["high"],
                "conditions": [
                    {
                        "name": "baseline",
                        "prompt_style": "none",
                        "steering": False,
                        "pas": False,
                    }
                ],
                "scoring": {
                    "bfi_items_path": "data/bfi/bfi44_scoring.csv",
                    "trait_scenarios_path": "data/trait_bigfive/scenarios.jsonl",
                },
            }
        }
        if bfi_item_text_path is not None:
            config["experiment"]["scoring"]["bfi_item_text_path"] = bfi_item_text_path
        if as_condition:
            config["experiment"]["conditions"].append(
                {
                    "name": "as_pas_only",
                    "prompt_style": "none",
                    "steering": True,
                    "pas": True,
                    "vector_mode": vector_mode,
                }
            )
        config_path = root / "config.yaml"
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return config_path


if __name__ == "__main__":
    unittest.main()
