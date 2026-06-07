"""Tests for PersonaActionSelector.

Tests cosine_similarity computation, select_action returning the
highest-scoring action, and get_ranking returning a sorted list.
Uses simple known vectors for deterministic verification.
"""

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.projection.selector import (
    PersonaActionSelector,
    ProjectedActionResolver,
    cosine_similarity,
)


class TestCosineSimilarity(unittest.TestCase):
    """Test the cosine_similarity utility function."""

    def test_identical_vectors(self):
        """Cosine similarity of identical vectors should be 1.0."""
        v = torch.tensor([1.0, 2.0, 3.0])
        sim = cosine_similarity(v, v)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_opposite_vectors(self):
        """Cosine similarity of opposite vectors should be -1.0."""
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([-1.0, 0.0, 0.0])
        sim = cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, -1.0, places=5)

    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal vectors should be 0.0."""
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        sim = cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 0.0, places=5)

    def test_known_value(self):
        """Test with a known computation."""
        v1 = torch.tensor([3.0, 4.0])
        v2 = torch.tensor([4.0, 3.0])
        # dot = 12+12 = 24, norms = 5*5 = 25, sim = 24/25 = 0.96
        expected = 24.0 / 25.0
        sim = cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, expected, places=5)

    def test_zero_vector(self):
        """Cosine similarity with a zero vector should return 0.0."""
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.zeros(3)
        sim = cosine_similarity(v1, v2)
        self.assertEqual(sim, 0.0)

    def test_scaled_vectors(self):
        """Cosine similarity should be invariant to scaling."""
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.tensor([2.0, 4.0, 6.0])  # 2 * v1
        sim = cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_mixed_device_vectors_when_cuda_available(self):
        """Cosine similarity should handle CPU action vectors with CUDA persona vectors."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        v1 = torch.tensor([1.0, 0.0, 0.0], device="cuda")
        v2 = torch.tensor([1.0, 0.0, 0.0])
        sim = cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, 1.0, places=5)

    def test_negative_scaled_vectors(self):
        """Negative scaling should give -1.0 similarity."""
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = -v1
        sim = cosine_similarity(v1, v2)
        self.assertAlmostEqual(sim, -1.0, places=5)


class TestPersonaActionSelector(unittest.TestCase):
    """Test PersonaActionSelector with known vectors."""

    def setUp(self):
        # Persona vector pointing in the "aggressive" direction
        self.persona_vector = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # Action embeddings with known alignments
        self.action_embeddings = {
            "attack": torch.tensor([0.9, 0.1, 0.0, 0.0]),   # Very aligned
            "negotiate": torch.tensor([-0.8, 0.2, 0.0, 0.0]),  # Opposite
            "intimidate": torch.tensor([0.7, 0.3, 0.0, 0.0]),  # Somewhat aligned
            "detour": torch.tensor([0.0, 0.0, 1.0, 0.0]),      # Orthogonal
        }

        self.selector = PersonaActionSelector(
            self.persona_vector, self.action_embeddings
        )

    def test_select_action_returns_highest_scoring(self):
        """select_action should return the action with highest cosine similarity."""
        result = self.selector.select_action()
        self.assertEqual(result, "attack")

    def test_compute_scores_returns_all_actions(self):
        """compute_scores should return a score for every action."""
        scores = self.selector.compute_scores()
        self.assertEqual(set(scores.keys()), set(self.action_embeddings.keys()))

    def test_compute_scores_values_correct(self):
        """Scores should match expected cosine similarity values."""
        scores = self.selector.compute_scores()

        # attack: [0.9, 0.1, 0, 0] vs [1, 0, 0, 0]
        # cos = 0.9 / sqrt(0.82) = 0.9 / 0.9055 ~ 0.9939
        self.assertGreater(scores["attack"], 0.99)

        # negotiate: [-0.8, 0.2, 0, 0] vs [1, 0, 0, 0]
        # cos = -0.8 / sqrt(0.68) ~ -0.970
        self.assertLess(scores["negotiate"], -0.9)

        # detour: [0, 0, 1, 0] vs [1, 0, 0, 0]
        # cos = 0 / 1 = 0
        self.assertAlmostEqual(scores["detour"], 0.0, places=5)

    def test_get_ranking_returns_sorted_list(self):
        """get_ranking should return (action_id, score) tuples sorted descending."""
        ranking = self.selector.get_ranking()

        self.assertEqual(len(ranking), len(self.action_embeddings))

        # Verify sorted descending
        scores = [score for _, score in ranking]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(scores[i], scores[i + 1])

        # First should be "attack" (most aligned)
        self.assertEqual(ranking[0][0], "attack")

        # Last should be "negotiate" (most opposite)
        self.assertEqual(ranking[-1][0], "negotiate")

    def test_ranking_with_tie(self):
        """get_ranking should handle tied scores gracefully."""
        persona = torch.tensor([0.0, 0.0, 1.0])
        embeddings = {
            "a": torch.tensor([0.0, 0.0, 1.0]),
            "b": torch.tensor([0.0, 0.0, 1.0]),
        }
        selector = PersonaActionSelector(persona, embeddings)
        ranking = selector.get_ranking()
        self.assertEqual(len(ranking), 2)
        # Both should have score 1.0
        self.assertAlmostEqual(ranking[0][1], 1.0, places=5)
        self.assertAlmostEqual(ranking[1][1], 1.0, places=5)


class TestProjectedActionResolver(unittest.TestCase):
    """Test ProjectedActionResolver with known vectors."""

    def setUp(self):
        self.action_embeddings = {
            "attack": torch.tensor([1.0, 0.0, 0.0]),
            "negotiate": torch.tensor([-1.0, 0.0, 0.0]),
            "detour": torch.tensor([0.0, 1.0, 0.0]),
        }
        self.resolver = ProjectedActionResolver(self.action_embeddings)

    def test_resolve_keeps_aligned_action(self):
        """Should keep the generated action when it is well-aligned."""
        persona = torch.tensor([1.0, 0.0, 0.0])
        result = self.resolver.resolve(persona, "attack", threshold=0.0)
        self.assertEqual(result, "attack")

    def test_resolve_overrides_misaligned_action(self):
        """Should override when the generated action is below threshold."""
        persona = torch.tensor([1.0, 0.0, 0.0])
        # negotiate has similarity -1.0, below threshold 0.5
        result = self.resolver.resolve(persona, "negotiate", threshold=0.5)
        self.assertEqual(result, "attack")

    def test_resolve_unknown_action_falls_back(self):
        """Should fall back to projection when the generated action is unknown."""
        persona = torch.tensor([1.0, 0.0, 0.0])
        result = self.resolver.resolve(persona, "unknown_action", threshold=0.0)
        self.assertEqual(result, "attack")

    def test_resolve_batch(self):
        """resolve_batch should handle multiple actions."""
        persona = torch.tensor([1.0, 0.0, 0.0])
        actions = ["attack", "negotiate", "unknown"]
        results = self.resolver.resolve_batch(persona, actions, threshold=0.5)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], "attack")    # Aligned, kept
        self.assertEqual(results[1], "attack")    # Misaligned, overridden
        self.assertEqual(results[2], "attack")    # Unknown, fallback


if __name__ == "__main__":
    unittest.main()
