"""Tests for steering vector computation and persistence."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.steering.vector import SteeringVectorComputer


class TestSteeringVectorComputer(unittest.TestCase):
    def setUp(self):
        self.computer = SteeringVectorComputer()

    def test_compute_mean_difference_without_normalization(self):
        positive = [torch.tensor([2.0, 0.0]), torch.tensor([4.0, 0.0])]
        negative = [torch.tensor([0.0, 0.0]), torch.tensor([0.0, 0.0])]
        vector = self.computer.compute_from_activations(
            positive,
            negative,
            normalize=False,
        )
        self.assertTrue(torch.allclose(vector, torch.tensor([3.0, 0.0])))

    def test_compute_unit_vector(self):
        positive = [torch.tensor([3.0, 4.0])]
        negative = [torch.tensor([0.0, 0.0])]
        vector = self.computer.compute_from_activations(
            positive,
            negative,
            normalize=True,
        )
        self.assertAlmostEqual(vector.norm().item(), 1.0, places=6)

    def test_save_load_bundle(self):
        vectors = {1: torch.tensor([1.0, 0.0])}
        metadata = {"normalized": True, "layers": [1]}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vectors.pt"
            self.computer.save_vectors(vectors, path, metadata=metadata)
            bundle = self.computer.load_bundle(path)
        self.assertEqual(bundle["metadata"]["layers"], [1])
        self.assertTrue(torch.allclose(bundle["vectors"][1], vectors[1]))


if __name__ == "__main__":
    unittest.main()
