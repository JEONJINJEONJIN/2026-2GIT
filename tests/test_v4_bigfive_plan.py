import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.data.trait_dataset import load_trait_scenarios
from src.experiments.v4_bigfive_plan import (
    build_v4_run_plan,
    load_v4_experiment,
    parse_v4_conditions,
    validate_v4_inputs,
    write_manifest_bundle,
    write_plan_summary,
    write_run_plan,
)


class TestV4BigFivePlan(unittest.TestCase):
    def test_build_v4_run_plan_crosses_conditions_traits_targets_scenarios(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_config(root)
            scenarios_path = self._write_scenarios(root, count=3)
            experiment = load_v4_experiment(config_path)
            scenarios = load_trait_scenarios(scenarios_path, allowed_splits=["dev"])

            rows = build_v4_run_plan(experiment, scenarios, seed=7)

        self.assertEqual(len(rows), 2 * 1 * 2 * 3)
        self.assertEqual({row.seed for row in rows}, {7})
        self.assertEqual({row.trait for row in rows}, {"agreeableness"})
        self.assertEqual({row.target_direction for row in rows}, {"high", "low"})

    def test_parse_conditions_requires_vector_mode_for_steering(self):
        experiment = {
            "conditions": [
                {
                    "name": "as_pas_only",
                    "prompt_style": "none",
                    "steering": True,
                    "pas": True,
                }
            ]
        }
        with self.assertRaises(ValueError):
            parse_v4_conditions(experiment)

    def test_parse_conditions_rejects_unknown_vector_mode(self):
        experiment = {
            "conditions": [
                {
                    "name": "bad_control",
                    "prompt_style": "none",
                    "steering": True,
                    "pas": False,
                    "vector_mode": "not_a_mode",
                }
            ]
        }
        with self.assertRaises(ValueError):
            parse_v4_conditions(experiment)

    def test_parse_conditions_supports_pas_final_policy(self):
        experiment = {
            "conditions": [
                {
                    "name": "as_pas_final",
                    "prompt_style": "none",
                    "steering": True,
                    "pas": True,
                    "vector_mode": "trait_contrast",
                    "selection_policy": "pas_final",
                }
            ]
        }

        conditions = parse_v4_conditions(experiment)

        self.assertEqual(conditions[0].selection_policy, "pas_final")

    def test_parse_conditions_rejects_pas_final_without_pas(self):
        experiment = {
            "conditions": [
                {
                    "name": "bad_pas_final",
                    "prompt_style": "none",
                    "steering": False,
                    "pas": False,
                    "selection_policy": "pas_final",
                }
            ]
        }
        with self.assertRaises(ValueError):
            parse_v4_conditions(experiment)

    def test_validate_v4_inputs_reports_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_config(root)
            self._write_bfi(root)
            self._write_scenarios(root, count=1)
            experiment = load_v4_experiment(config_path)

            metadata = validate_v4_inputs(experiment, root)

        self.assertEqual(metadata["bfi_item_count"], 5)
        self.assertEqual(metadata["trait_scenario_count"], 1)
        self.assertEqual(len(metadata["bfi_file_sha256"]), 64)
        self.assertEqual(len(metadata["trait_file_sha256"]), 64)

    def test_validate_v4_inputs_uses_bfi_text_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_config(root, bfi_text_path="data/bfi/text.local.csv")
            self._write_bfi(root)
            text_path = root / "data" / "bfi" / "text.local.csv"
            text_path.write_text(
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
            self._write_scenarios(root, count=1)
            experiment = load_v4_experiment(config_path)

            metadata = validate_v4_inputs(experiment, root)

        self.assertEqual(metadata["bfi_item_text_path"], str(text_path))
        self.assertEqual(len(metadata["bfi_item_text_sha256"]), 64)

    def test_write_plan_summary_and_manifests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = self._write_config(root)
            scenarios_path = self._write_scenarios(root, count=1)
            experiment = load_v4_experiment(config_path)
            scenarios = load_trait_scenarios(scenarios_path, allowed_splits=["dev"])
            rows = build_v4_run_plan(experiment, scenarios)
            plan_path = root / "run_plan.csv"
            summary_path = root / "summary.json"
            manifest_dir = root / "manifests"

            plan_count = write_run_plan(rows, plan_path)
            write_plan_summary(rows, summary_path, input_metadata={"x": 1})
            manifest_count = write_manifest_bundle(
                rows,
                manifest_dir,
                model_name="test-model",
                project_root=root,
                input_metadata={"x": 1},
            )

            with open(plan_path, "r", encoding="utf-8", newline="") as fh:
                csv_rows = list(csv.DictReader(fh))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifests = list(manifest_dir.glob("*/manifest.json"))

        self.assertEqual(plan_count, len(rows))
        self.assertEqual(len(csv_rows), len(rows))
        self.assertEqual(summary["row_count"], len(rows))
        self.assertEqual(manifest_count, len(rows))
        self.assertEqual(len(manifests), len(rows))

    def _write_config(self, root: Path, bfi_text_path: str | None = None) -> Path:
        self._write_bfi(root)
        self._write_scenarios(root, count=1)
        config = {
            "experiment": {
                "name": "v4_bigfive_test",
                "version": "v4_bigfive",
                "split": "dev",
                "traits": ["agreeableness"],
                "target_directions": ["high", "low"],
                "conditions": [
                    {
                        "name": "baseline",
                        "prompt_style": "none",
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
                ],
                "scoring": {
                    "bfi_items_path": "data/bfi/bfi44_scoring.csv",
                    "trait_scenarios_path": "data/trait_bigfive/scenarios.jsonl",
                },
            }
        }
        if bfi_text_path is not None:
            config["experiment"]["scoring"]["bfi_item_text_path"] = bfi_text_path
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config), encoding="utf-8")
        return path

    def _write_bfi(self, root: Path) -> Path:
        path = root / "data" / "bfi" / "bfi44_scoring.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "\n".join(
                [
                    "item_id,trait,reverse_scored,text",
                    "bfi_01,agreeableness,false,",
                    "bfi_02,conscientiousness,false,",
                    "bfi_03,neuroticism,false,",
                    "bfi_04,openness,false,",
                    "bfi_05,extraversion,false,",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return path

    def _write_scenarios(self, root: Path, count: int) -> Path:
        path = root / "data" / "trait_bigfive" / "scenarios.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [self._scenario(f"s{i:03d}") for i in range(count)]
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _scenario(scenario_id):
        return {
            "scenario_id": scenario_id,
            "trait": "agreeableness",
            "prompt": "A teammate asks for help during a difficult task.",
            "split": "dev",
            "actions": [
                {"id": f"{scenario_id}_a", "text": "Offer help.", "trait_direction": "high"},
                {"id": f"{scenario_id}_b", "text": "Listen first.", "trait_direction": "high"},
                {"id": f"{scenario_id}_c", "text": "Refuse quickly.", "trait_direction": "low"},
                {"id": f"{scenario_id}_d", "text": "Ignore them.", "trait_direction": "low"},
            ],
        }


if __name__ == "__main__":
    unittest.main()
