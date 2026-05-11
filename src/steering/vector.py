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
    """Computes, validates, saves, and loads CAA steering vectors."""

    def compute_from_activations(
        self,
        positive_acts: List[Tensor],
        negative_acts: List[Tensor],
        normalize: bool = True,
    ) -> Tensor:
        """Compute a single steering vector as mean(positive) - mean(negative).

        Args:
            positive_acts: List of 1-D activation tensors from the positive
                side of contrastive pairs.
            negative_acts: Corresponding negative activation tensors.
            normalize: Whether to return a unit vector.

        Returns:
            1-D tensor of shape ``(hidden_dim,)``.
        """
        if len(positive_acts) == 0 or len(negative_acts) == 0:
            raise ValueError("Activation lists must not be empty.")
        if len(positive_acts) != len(negative_acts):
            raise ValueError("Positive and negative activation counts must match.")

        pos_mean = torch.stack(positive_acts).mean(dim=0)
        neg_mean = torch.stack(negative_acts).mean(dim=0)
        vector = pos_mean - neg_mean
        if normalize:
            return self.normalize_vector(vector)
        return vector

    def compute_from_pairs(
        self,
        extractor,
        pairs: List[dict],
        layer_indices: Optional[Sequence[int]] = None,
        normalize: bool = True,
        return_metadata: bool = False,
        metadata: Optional[dict] = None,
    ):
        """Extract activations from contrastive pairs and compute vectors.

        This is a convenience method that chains
        :pymethod:`ActivationExtractor.extract_from_pairs` with
        :pymethod:`compute_from_activations`.

        Args:
            extractor: An :class:`~steering.extractor.ActivationExtractor`.
            pairs: List of dicts with ``'positive'`` / ``'negative'`` keys.
            layer_indices: Layers to target (``None`` = all).
            normalize: Whether to save unit steering vectors.
            return_metadata: When true, return ``(vectors, metadata)``.
            metadata: Optional metadata to merge into the generated metadata.

        Returns:
            Dict mapping each layer index to a vector, or ``(vectors, metadata)``.
        """
        all_acts = extractor.extract_from_pairs(pairs, layer_indices)

        vectors: Dict[int, Tensor] = {}
        original_norms: Dict[str, float] = {}
        for layer_idx in all_acts["positive"]:
            raw_vector = self.compute_from_activations(
                all_acts["positive"][layer_idx],
                all_acts["negative"][layer_idx],
                normalize=False,
            )
            original_norms[str(layer_idx)] = float(raw_vector.norm().item())
            vectors[layer_idx] = (
                self.normalize_vector(raw_vector) if normalize else raw_vector
            )

        self.validate_vectors(vectors, requested_layers=layer_indices)

        generated_metadata = {
            "pair_count": len(pairs),
            "layers": sorted(vectors.keys()),
            "token_aggregation": "last_input_token",
            "normalized": bool(normalize),
            "original_norms": original_norms,
            "layer_cosine_similarities": self.compute_neighbor_similarities(vectors),
        }
        if metadata:
            generated_metadata.update(metadata)

        if return_metadata:
            return vectors, generated_metadata
        return vectors

    @staticmethod
    def normalize_vector(vector: Tensor, eps: float = 1e-12) -> Tensor:
        """Return a unit vector, preserving dtype and raising on zero norm."""
        norm = vector.norm()
        if not torch.isfinite(norm) or norm.item() <= eps:
            raise ValueError("Cannot normalize a zero, NaN, or Inf steering vector.")
        return vector / norm

    @staticmethod
    def validate_vectors(
        vectors: Dict[int, Tensor],
        requested_layers: Optional[Sequence[int]] = None,
    ) -> None:
        """Validate vector presence and numerical health."""
        if not vectors:
            raise ValueError("No steering vectors were computed.")

        if requested_layers is not None:
            missing = sorted(set(requested_layers) - set(vectors.keys()))
            if missing:
                raise ValueError(f"Missing steering vectors for layers: {missing}")

        for layer_idx, vector in vectors.items():
            if vector.dim() != 1:
                raise ValueError(f"Layer {layer_idx} vector must be 1-D.")
            if not torch.isfinite(vector).all():
                raise ValueError(f"Layer {layer_idx} vector contains NaN or Inf.")
            if vector.norm().item() == 0.0:
                raise ValueError(f"Layer {layer_idx} vector has zero norm.")

    @staticmethod
    def compute_neighbor_similarities(vectors: Dict[int, Tensor]) -> Dict[str, float]:
        """Compute cosine similarity between adjacent saved layers."""
        similarities: Dict[str, float] = {}
        layers = sorted(vectors.keys())
        for left, right in zip(layers, layers[1:]):
            similarities[f"{left}-{right}"] = compute_cosine_similarity(
                vectors[left].float(),
                vectors[right].float(),
            )
        return similarities

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_vectors(
        self,
        vectors: Dict[int, Tensor],
        path: Union[str, Path],
        metadata: Optional[dict] = None,
    ) -> None:
        """Save steering vectors to a ``.pt`` file.

        Args:
            vectors: Dict mapping layer index to steering vector tensor.
            path: Destination file path.
            metadata: Optional JSON-like metadata dict.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"vectors": vectors, "metadata": metadata or {}}
        torch.save(payload, path)

    def load_vectors(self, path: Union[str, Path]) -> Dict[int, Tensor]:
        """Load steering vectors from a ``.pt`` file.

        Args:
            path: Path to a file previously saved with :meth:`save_vectors`.

        Returns:
            Dict mapping layer index to steering vector tensor.
        """
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "vectors" in payload:
            return payload["vectors"]
        return payload

    def load_bundle(self, path: Union[str, Path]) -> dict:
        """Load vectors and metadata, supporting old vector-only files."""
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and "vectors" in payload:
            return payload
        return {"vectors": payload, "metadata": {}}
