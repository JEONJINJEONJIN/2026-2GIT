"""Activation extraction from contrastive pairs for AS steering."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor
from src.models.loader import get_model_layers


class ActivationExtractor:
    """Extracts last-input-token hidden states from transformer layers."""

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
        """Extract last-input-token hidden states from the given text.

        Args:
            text: Input string to feed through the model.
            layer_indices: Which layers to capture. ``None`` means all layers.

        Returns:
            Dict mapping each requested layer index to a 1-D tensor of shape
            ``(hidden_dim,)``.
        """
        layers = get_model_layers(self.model)
        if layer_indices is None:
            layer_indices = list(range(len(layers)))

        self._validate_layers(layer_indices, len(layers))

        activations: Dict[int, Tensor] = {}

        def _make_hook(layer_idx: int):
            def hook_fn(module, input, output):
                hidden = output[0] if isinstance(output, tuple) else output
                if hidden.dim() == 3:
                    activations[layer_idx] = hidden[0, -1, :].detach().cpu()
                elif hidden.dim() == 2:
                    activations[layer_idx] = hidden[-1, :].detach().cpu()
                else:
                    raise ValueError(
                        f"Unsupported hidden state rank {hidden.dim()} "
                        f"at layer {layer_idx}."
                    )
            return hook_fn

        try:
            for idx in layer_indices:
                layer_module = layers[idx]
                handle = layer_module.register_forward_hook(_make_hook(idx))
                self._hooks.append(handle)

            inputs = self.tokenizer(text, return_tensors="pt")
            device = self._get_model_device()
            inputs = {k: v.to(device) for k, v in inputs.items()}

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

    def _get_model_device(self) -> torch.device:
        """Return a real model device, falling back to CPU for test doubles."""
        device = getattr(self.model, "device", None)
        if isinstance(device, torch.device):
            return device
        try:
            return next(self.model.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            return torch.device("cpu")

    @staticmethod
    def _validate_layers(layer_indices: Sequence[int], num_layers: int) -> None:
        invalid = [idx for idx in layer_indices if idx < 0 or idx >= num_layers]
        if invalid:
            raise ValueError(
                f"Invalid layer index/indices {invalid}; model has "
                f"{num_layers} layers indexed 0..{num_layers - 1}."
            )

    def _cleanup_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
