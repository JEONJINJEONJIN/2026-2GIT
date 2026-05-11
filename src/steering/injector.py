"""Uniform post-block activation steering injection."""

from __future__ import annotations

from typing import Dict, Sequence, Union

from torch import Tensor

from src.models.loader import get_model_layers
from src.steering.vector import SteeringVectorComputer


class SteeringInjector:
    """Inject unit steering vectors into transformer block outputs.

    The hook target is the post-block residual stream: the first tensor returned
    by ``model.model.layers[layer_idx]``. Only uniform injection is implemented
    for the AS-first rebuild.
    """

    def __init__(
        self,
        model,
        steering_vectors: Dict[int, Tensor],
        normalize_vectors: bool = True,
    ):
        self.model = model
        self.steering_vectors = steering_vectors
        self.normalize_vectors = normalize_vectors
        self._hooks: list = []

    def inject(
        self,
        layer_indices: Sequence[int],
        alpha: Union[float, Dict[int, float]],
    ) -> None:
        """Register uniform AS hooks on the requested layers."""
        self.clear()
        layers = get_model_layers(self.model)
        self._validate_layers(layer_indices, len(layers))

        for layer_idx in layer_indices:
            if layer_idx not in self.steering_vectors:
                raise ValueError(f"No steering vector available for layer {layer_idx}.")

            vector = self.steering_vectors[layer_idx].detach().clone()
            if self.normalize_vectors:
                vector = SteeringVectorComputer.normalize_vector(vector)

            layer_alpha = alpha[layer_idx] if isinstance(alpha, dict) else alpha
            handle = layers[layer_idx].register_forward_hook(
                self._make_hook(vector, float(layer_alpha))
            )
            self._hooks.append(handle)

    def clear(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
        return False

    @staticmethod
    def _make_hook(steering_vector: Tensor, alpha: float):
        """Create a hook implementing ``h' = h + alpha * v``."""

        def hook_fn(module, input, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            vector = steering_vector.to(device=hidden.device, dtype=hidden.dtype)
            modified = hidden + alpha * vector
            if is_tuple:
                return (modified,) + output[1:]
            return modified

        return hook_fn

    @staticmethod
    def _validate_layers(layer_indices: Sequence[int], num_layers: int) -> None:
        invalid = [idx for idx in layer_indices if idx < 0 or idx >= num_layers]
        if invalid:
            raise ValueError(
                f"Invalid layer index/indices {invalid}; model has "
                f"{num_layers} layers indexed 0..{num_layers - 1}."
            )
