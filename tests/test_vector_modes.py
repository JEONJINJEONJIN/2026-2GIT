"""Tests for contrastive pair conversion and vector-mode resolution."""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def load_script_module(filename: str, module_name: str):
    path = PROJECT_ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestVectorModes(unittest.TestCase):
    def test_resolve_contrast_negated_vectors(self):
        run_experiment = load_script_module(
            "03_run_experiment.py",
            "run_experiment",
        )
        vectors = {"contrast": {0: torch.tensor([1.0, 0.0])}}
        aggressive = run_experiment.resolve_persona_vectors(
            vectors,
            "aggressive",
            "contrast_negated",
        )
        cooperative = run_experiment.resolve_persona_vectors(
            vectors,
            "cooperative",
            "contrast_negated",
        )

        self.assertTrue(torch.allclose(aggressive[0], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(cooperative[0], torch.tensor([-1.0, -0.0])))

    def test_resolve_separate_vectors(self):
        run_experiment = load_script_module(
            "03_run_experiment.py",
            "run_experiment",
        )
        vectors = {
            "aggressive": {0: torch.tensor([1.0, 0.0])},
            "cooperative": {0: torch.tensor([0.0, 1.0])},
        }
        result = run_experiment.resolve_persona_vectors(
            vectors,
            "cooperative",
            "separate",
        )

        self.assertTrue(torch.allclose(result[0], torch.tensor([0.0, 1.0])))

    def test_resolve_contrast_explicit_vectors(self):
        run_experiment = load_script_module(
            "03_run_experiment.py",
            "run_experiment",
        )
        vectors = {
            "contrast_explicit": {
                "aggressive": {0: torch.tensor([1.0, 0.0])},
                "cooperative": {0: torch.tensor([-1.0, 0.0])},
            }
        }
        result = run_experiment.resolve_persona_vectors(
            vectors,
            "cooperative",
            "contrast_explicit",
        )

        self.assertTrue(torch.allclose(result[0], torch.tensor([-1.0, 0.0])))

    def test_resolve_orthogonalized_vectors(self):
        run_experiment = load_script_module(
            "03_run_experiment.py",
            "run_experiment",
        )
        vectors = {
            "orthogonalized": {
                "aggressive": {0: torch.tensor([1.0, 0.0])},
                "cooperative": {0: torch.tensor([-1.0, 0.0])},
            }
        }
        result = run_experiment.resolve_persona_vectors(
            vectors,
            "cooperative",
            "orthogonalized",
        )

        self.assertTrue(torch.allclose(result[0], torch.tensor([-1.0, 0.0])))

    def test_collect_available_layers_nested_bundle(self):
        run_experiment = load_script_module(
            "03_run_experiment.py",
            "run_experiment",
        )
        vectors = {
            "contrast": {0: torch.tensor([1.0])},
            "orthogonalized": {
                "aggressive": {1: torch.tensor([1.0])},
                "cooperative": {2: torch.tensor([1.0])},
            },
        }

        self.assertEqual(run_experiment.collect_available_layers(vectors), {0, 1, 2})

    def test_build_neutral_anchor_pairs(self):
        converter = load_script_module(
            "00_convert_team_a_csvs.py",
            "convert_team_a_csvs",
        )
        scenarios = [
            {
                "id": "sc_001",
                "context": "A gate is blocked.",
                "actions": [
                    {
                        "id": "a1",
                        "description": "Threaten the guard.",
                        "persona_alignment": "aggressive",
                    },
                    {
                        "id": "n1",
                        "description": "Wait for instructions.",
                        "persona_alignment": "neutral",
                    },
                ],
            }
        ]

        pairs = converter.build_neutral_anchor_pairs(scenarios, "aggressive")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["positive_action_id"], "a1")
        self.assertEqual(pairs[0]["negative_action_id"], "n1")
        self.assertIn("Threaten the guard", pairs[0]["positive"])
        self.assertIn("Wait for instructions", pairs[0]["negative"])

    def test_build_action_contrastive_pairs_uses_all_action_combinations(self):
        converter = load_script_module(
            "00_convert_team_a_csvs.py",
            "convert_team_a_csvs_grid",
        )
        scenarios = [
            {
                "id": "sc_001",
                "context": "A gate is blocked.",
                "actions": [
                    {
                        "id": "a1",
                        "description": "Threaten the guard.",
                        "persona_alignment": "aggressive",
                    },
                    {
                        "id": "a2",
                        "description": "Force the gate open.",
                        "persona_alignment": "aggressive",
                    },
                    {
                        "id": "c1",
                        "description": "Negotiate with the guard.",
                        "persona_alignment": "cooperative",
                    },
                    {
                        "id": "c2",
                        "description": "Offer to help repair the gate.",
                        "persona_alignment": "cooperative",
                    },
                ],
            }
        ]

        pairs = converter.build_action_contrastive_pairs(scenarios)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(
            {pair["positive_action_id"] for pair in pairs},
            {"a1", "a2"},
        )
        self.assertEqual(
            {pair["negative_action_id"] for pair in pairs},
            {"c1", "c2"},
        )


if __name__ == "__main__":
    unittest.main()
