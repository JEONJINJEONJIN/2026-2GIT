"""Forward hook management for activation steering.

Provides :class:`HookManager`, a context-manager-aware registry that
attaches PyTorch forward hooks to transformer decoder layers and
cleanly removes them when done.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle as RemovableHook
from src.models.loader import get_model_layers

logger = logging.getLogger(__name__)


class HookManager:
    """Register, track, and remove forward hooks on transformer layers.

    Usage::

        with HookManager() as hm:
            hm.register_steering_hook(model, layer_idx=15,
                                      steering_vector=sv, alpha=2.0)
            outputs = model.generate(...)
        # hooks are automatically removed here

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        self._handles: List[RemovableHook] = []

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_all()

    # ------------------------------------------------------------------
    # Generic hook registration
    # ------------------------------------------------------------------

    def register_hook(
        self,
        model: nn.Module,
        layer_idx: int,
        hook_fn: Callable,
    ) -> RemovableHook:
        """Attach a forward hook to ``model.model.layers[layer_idx]``.

        Parameters
        ----------
        model:
            A HuggingFace causal LM whose inner transformer is accessed
            via ``model.model.layers``.
        layer_idx:
            Index of the decoder layer to hook.
        hook_fn:
            A callable with signature
            ``(module, input, output) -> output | None``.

        Returns
        -------
        RemovableHook
            The handle returned by :meth:`nn.Module.register_forward_hook`.
        """
        layer = get_model_layers(model)[layer_idx]
        handle = layer.register_forward_hook(hook_fn)
        self._handles.append(handle)
        logger.debug("Hook registered on layer %d (%d total)", layer_idx, len(self._handles))
        return handle

    # ------------------------------------------------------------------
    # Steering hook
    # ------------------------------------------------------------------

    def register_steering_hook(
        self,
        model: nn.Module,
        layer_idx: int,
        steering_vector: Tensor,
        alpha: float,
        beta_fn: Optional[Callable[[int], Tensor]] = None,
    ) -> RemovableHook:
        """Register a hook that injects a steering vector into hidden states.

        The hook modifies the first element of the layer output
        (``output[0]``, the hidden states tensor of shape
        ``(batch, seq_len, hidden_dim)``) by adding::

            alpha * beta * steering_vector

        where *beta* is either 1.0 (uniform) or a per-position tensor
        returned by *beta_fn*.

        Parameters
        ----------
        model:
            HuggingFace causal LM.
        layer_idx:
            Decoder layer index to hook.
        steering_vector:
            1-D tensor of shape ``(hidden_dim,)`` representing the
            steering direction.
        alpha:
            Global scalar multiplier for the steering vector.
        beta_fn:
            Optional callable ``(seq_len: int) -> Tensor`` that returns
            a tensor broadcastable to ``(1, seq_len, 1)`` containing
            per-position scaling factors.  When ``None``, uniform
            scaling (beta=1) is used.

        Returns
        -------
        RemovableHook
            Handle for the registered hook.
        """
        # Ensure the vector lives on the correct device / dtype lazily
        # (we resolve at hook call time so the model can be moved after
        # registration).
        sv = steering_vector.detach().clone()

        def _steering_hook(
            module: nn.Module,
            input: Tuple[Tensor, ...],
            output: Any,
        ) -> Any:
            hidden_states = output[0]  # (batch, seq, hidden) or (seq, hidden)

            vec = sv.to(device=hidden_states.device, dtype=hidden_states.dtype)

            # seq_len is dim 1 for 3D, dim 0 for 2D
            seq_len = hidden_states.size(1) if hidden_states.dim() == 3 else hidden_states.size(0)

            if beta_fn is not None:
                beta = beta_fn(seq_len)
                if isinstance(beta, Tensor):
                    beta = beta.to(device=hidden_states.device, dtype=hidden_states.dtype)
                    # Ensure shape is broadcastable: (1, seq_len, 1)
                    if beta.dim() == 1:
                        beta = beta.unsqueeze(0).unsqueeze(-1)
                    elif beta.dim() == 2:
                        beta = beta.unsqueeze(-1)
                delta = alpha * beta * vec
            else:
                delta = alpha * vec

            modified = hidden_states + delta

            # Reconstruct the output tuple with modified hidden states.
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        handle = self.register_hook(model, layer_idx, _steering_hook)
        logger.info(
            "Steering hook registered on layer %d (alpha=%.2f, beta_fn=%s)",
            layer_idx,
            alpha,
            "custom" if beta_fn else "uniform",
        )
        return handle

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_all(self) -> None:
        """Remove all registered hooks."""
        count = len(self._handles)
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if count:
            logger.info("Removed %d hook(s)", count)
