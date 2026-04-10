"""Steering vector computation via Contrastive Activation Addition (CAA).

Computes steering vectors as the mean-difference between positive and negative
activation sets, and provides persistence utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def compute_cosine_similarity(vec1: Tensor, vec2: Tensor) -> float:
    """Return the cosine similarity between two 1-D tensors.

    Args:
        vec1: First vector.
        vec2: Second vector.

    Returns:
        Scalar cosine similarity in ``[-1, 1]``.
    """
    return torch.nn.functional.cosine_similarity(
        vec1.unsqueeze(0), vec2.unsqueeze(0)
    ).item()


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------

class SteeringVectorComputer:
    """Computes, saves, and loads CAA steering vectors."""

    def compute_from_activations(
        self,
        positive_acts: List[Tensor],
        negative_acts: List[Tensor],
    ) -> Tensor:
        """Compute a single steering vector as mean(positive) - mean(negative).

        Args:
            positive_acts: List of 1-D activation tensors from the positive
                side of contrastive pairs.
            negative_acts: Corresponding negative activation tensors.

        Returns:
            1-D tensor of shape ``(hidden_dim,)``.
        """
        pos_mean = torch.stack(positive_acts).mean(dim=0)
        neg_mean = torch.stack(negative_acts).mean(dim=0)
        return pos_mean - neg_mean

    def compute_from_pairs(
        self,
        extractor,
        pairs: List[dict],
        layer_indices: Optional[Sequence[int]] = None,
    ) -> Dict[int, Tensor]:
        """Extract activations from contrastive pairs and compute vectors.

        This is a convenience method that chains
        :pymethod:`ActivationExtractor.extract_from_pairs` with
        :pymethod:`compute_from_activations`.

        Args:
            extractor: An :class:`~steering.extractor.ActivationExtractor`.
            pairs: List of dicts with ``'positive'`` / ``'negative'`` keys.
            layer_indices: Layers to target (``None`` = all).

        Returns:
            Dict mapping each layer index to its steering vector tensor.
        """
        all_acts = extractor.extract_from_pairs(pairs, layer_indices)

        vectors: Dict[int, Tensor] = {}
        for layer_idx in all_acts["positive"]:
            vectors[layer_idx] = self.compute_from_activations(
                all_acts["positive"][layer_idx],
                all_acts["negative"][layer_idx],
            )
        return vectors

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_vectors(
        self,
        vectors: Dict[int, Tensor],
        path: Union[str, Path],
    ) -> None:
        """Save steering vectors to a ``.pt`` file.

        Args:
            vectors: Dict mapping layer index to steering vector tensor.
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(vectors, path)

    def load_vectors(self, path: Union[str, Path]) -> Dict[int, Tensor]:
        """Load steering vectors from a ``.pt`` file.

        Args:
            path: Path to a file previously saved with :meth:`save_vectors`.

        Returns:
            Dict mapping layer index to steering vector tensor.
        """
        return torch.load(path, map_location="cpu")
