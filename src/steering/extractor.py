"""Activation extraction from contrastive pairs for CAA steering.

Extracts hidden-state activations at specified transformer layers using
forward hooks. Designed for Qwen2.5-3B-Instruct (36 layers, hidden_dim=2048).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor


class ActivationExtractor:
    """Extracts last-token hidden states from specified transformer layers."""

    def __init__(self, model, tokenizer):
        """
        Args:
            model: A HuggingFace causal-LM (e.g. Qwen2.5-3B-Instruct).
            tokenizer: The corresponding tokenizer.
        """
        self.model = model
        self.tokenizer = tokenizer
        self._hooks: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_activations(
        self,
        text: str,
        layer_indices: Optional[Sequence[int]] = None,
    ) -> Dict[int, Tensor]:
        """Extract last-token hidden states from the given text.

        Args:
            text: Input string to feed through the model.
            layer_indices: Which layers to capture.  ``None`` means all 36
                layers (indices 0-35).

        Returns:
            Dict mapping each requested layer index to a 1-D tensor of shape
            ``(hidden_dim,)``.
        """
        if layer_indices is None:
            layer_indices = list(range(len(self.model.model.layers)))

        activations: Dict[int, Tensor] = {}

        # Register forward hooks -------------------------------------------
        def _make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                # output is a tuple; the first element is the hidden state
                # tensor of shape (batch, seq_len, hidden_dim).
                hidden = output[0]
                # Grab last token position, squeeze batch dim.
                activations[layer_idx] = hidden[0, -1, :].detach().cpu()
            return hook_fn

        try:
            for idx in layer_indices:
                layer_module = self.model.model.layers[idx]
                handle = layer_module.register_forward_hook(_make_hook(idx))
                self._hooks.append(handle)

            inputs = self.tokenizer(text, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                self.model(**inputs)
        finally:
            self._cleanup_hooks()

        return activations

    def extract_from_pairs(
        self,
        pairs: List[dict],
        layer_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, Dict[int, List[Tensor]]]:
        """Extract activations for a list of contrastive pairs.

        Args:
            pairs: List of dicts, each with ``'positive'`` and ``'negative'``
                string fields.
            layer_indices: Which layers to capture (``None`` = all).

        Returns:
            ``{'positive': {layer_idx: [tensor, ...]},
              'negative': {layer_idx: [tensor, ...]}}``
        """
        result: Dict[str, Dict[int, List[Tensor]]] = {
            "positive": {},
            "negative": {},
        }

        for pair in pairs:
            for key in ("positive", "negative"):
                acts = self.extract_activations(pair[key], layer_indices)
                for layer_idx, tensor in acts.items():
                    result[key].setdefault(layer_idx, []).append(tensor)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
