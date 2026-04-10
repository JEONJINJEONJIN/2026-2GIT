"""Comparison utilities for fp16 vs 4-bit quantized steering vectors."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


class QuantizationComparator:
    """Compare full-precision and quantized steering vectors layer by layer."""

    def __init__(
        self,
        fp16_vectors: Dict[int, torch.Tensor],
        quantized_vectors: Dict[int, torch.Tensor],
    ):
        """
        Args:
            fp16_vectors: dict mapping layer index to the fp16 steering
                vector tensor for that layer.
            quantized_vectors: dict mapping layer index to the 4-bit (or
                other quantized) steering vector tensor for that layer.
        """
        self._fp16 = fp16_vectors
        self._quantized = quantized_vectors

    def compute_layer_similarities(self) -> Dict[int, float]:
        """Compute cosine similarity between fp16 and quantized vectors per layer.

        Returns:
            dict mapping layer_idx to cosine similarity (float in [-1, 1]).
        """
        similarities: Dict[int, float] = {}
        common_layers = sorted(set(self._fp16.keys()) & set(self._quantized.keys()))

        for layer_idx in common_layers:
            v_fp16 = self._fp16[layer_idx].float().flatten()
            v_quant = self._quantized[layer_idx].float().flatten()

            cos_sim = torch.nn.functional.cosine_similarity(
                v_fp16.unsqueeze(0), v_quant.unsqueeze(0)
            )
            similarities[layer_idx] = float(cos_sim.item())

        return similarities

    def is_quantization_safe(self, threshold: float = 0.95) -> Dict:
        """Determine whether quantization preserved vector directions.

        Args:
            threshold: minimum acceptable cosine similarity.

        Returns:
            dict with 'safe' (bool), 'min_similarity' (float), and
            'unsafe_layers' (list of layer indices below threshold).
        """
        similarities = self.compute_layer_similarities()

        if not similarities:
            return {"safe": False, "min_similarity": 0.0, "unsafe_layers": []}

        unsafe_layers: List[int] = [
            idx for idx, sim in similarities.items() if sim < threshold
        ]
        min_sim = min(similarities.values())

        return {
            "safe": len(unsafe_layers) == 0,
            "min_similarity": min_sim,
            "unsafe_layers": sorted(unsafe_layers),
        }

    def plot_similarity(
        self,
        save_path: Optional[str] = None,
        threshold: float = 0.95,
    ) -> None:
        """Bar chart of cosine similarity per layer with a threshold line.

        Args:
            save_path: if provided, save figure to this path.
            threshold: draw a red horizontal line at this value.
        """
        similarities = self.compute_layer_similarities()
        if not similarities:
            return

        layers = sorted(similarities.keys())
        sims = [similarities[l] for l in layers]

        fig, ax = plt.subplots(figsize=(max(6, len(layers) * 0.5), 5))

        colors = ["#e74c3c" if s < threshold else "#2ecc71" for s in sims]
        ax.bar([str(l) for l in layers], sims, color=colors, edgecolor="white")
        ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold ({threshold})")

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("FP16 vs Quantized Steering Vector Similarity")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
