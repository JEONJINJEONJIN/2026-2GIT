"""Layer/position-aware steering injection for CAA.

Registers forward hooks that add scaled steering vectors to transformer layer
outputs during generation. Supports uniform, action-boosted, and action-only
beta strategies for differentiating speech vs. action regions.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch import Tensor


# ------------------------------------------------------------------
# Action tag detection
# ------------------------------------------------------------------

class ActionTagDetector:
    """Tracks whether the current generation position is inside
    ``<Action>...</Action>`` tags.

    Call :meth:`update` with each batch of newly generated token IDs to keep
    the detector in sync with generation progress.
    """

    def __init__(self):
        self._in_action: bool = False
        self._buffer: str = ""

    def reset(self) -> None:
        """Reset internal state for a new generation run."""
        self._in_action = False
        self._buffer = ""

    def update(self, token_ids: Union[List[int], Tensor], tokenizer) -> None:
        """Decode new tokens and update the action-region state.

        Args:
            token_ids: Newly generated token IDs (1-D sequence).
            tokenizer: The tokenizer used for decoding.
        """
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.tolist()

        new_text = tokenizer.decode(token_ids, skip_special_tokens=True)
        self._buffer += new_text

        # Scan for tag transitions in the accumulated buffer.
        while True:
            if not self._in_action:
                open_pos = self._buffer.find("<Action>")
                if open_pos != -1:
                    self._in_action = True
                    # Keep buffer from the tag onward so we can find the close.
                    self._buffer = self._buffer[open_pos + len("<Action>"):]
                else:
                    # Keep only a tail long enough to contain a partial tag.
                    self._buffer = self._buffer[-16:] if len(self._buffer) > 16 else self._buffer
                    break
            else:
                close_pos = self._buffer.find("</Action>")
                if close_pos != -1:
                    self._in_action = False
                    self._buffer = self._buffer[close_pos + len("</Action>"):]
                else:
                    self._buffer = self._buffer[-16:] if len(self._buffer) > 16 else self._buffer
                    break

    def is_in_action_region(self) -> bool:
        """Return ``True`` if the most recent token falls inside an action region."""
        return self._in_action


# ------------------------------------------------------------------
# Steering injector
# ------------------------------------------------------------------

class SteeringInjector:
    """Injects steering vectors into transformer layers via forward hooks.

    Supports context-manager usage::

        with SteeringInjector(model, vectors) as inj:
            inj.inject([10, 14, 18], alpha=1.5)
            output = model.generate(...)
    """

    def __init__(
        self,
        model,
        steering_vectors: Dict[int, Tensor],
        hook_manager=None,
    ):
        """
        Args:
            model: The HuggingFace causal-LM to steer.
            steering_vectors: Dict mapping layer index to a 1-D steering
                vector of shape ``(hidden_dim,)``.
            hook_manager: Optional external hook manager. If ``None``, hooks
                are tracked internally.
        """
        self.model = model
        self.steering_vectors = steering_vectors
        self.hook_manager = hook_manager
        self._hooks: list = []
        self.action_detector = ActionTagDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(
        self,
        layer_indices: Sequence[int],
        alpha: Union[float, Dict[int, float]],
        beta_strategy: str = "uniform",
        beta_config: Optional[dict] = None,
    ) -> None:
        """Register steering hooks on the specified layers.

        Args:
            layer_indices: Which layers to hook.
            alpha: Overall steering strength. Either a single float applied to
                every layer, or a dict ``{layer_idx: float}`` for per-layer
                control.
            beta_strategy: One of ``"uniform"``, ``"action_boosted"``, or
                ``"action_only"``.

                * **uniform** -- ``beta = 1.0`` everywhere.
                * **action_boosted** -- ``beta = speech_mult`` outside action
                  tags, ``beta = action_mult`` inside them.
                * **action_only** -- ``beta = 0.0`` outside action tags,
                  ``beta = action_mult`` inside them.

            beta_config: Dict with optional keys ``speech_mult`` (default 1.0)
                and ``action_mult`` (default 2.0) used by the non-uniform beta
                strategies.
        """
        # Clear any previously registered hooks.
        self.clear()
        self.action_detector.reset()

        beta_config = beta_config or {}
        speech_mult: float = beta_config.get("speech_mult", 1.0)
        action_mult: float = beta_config.get("action_mult", 2.0)

        beta_fn = self._build_beta_fn(beta_strategy, speech_mult, action_mult)

        for idx in layer_indices:
            if idx not in self.steering_vectors:
                continue

            layer_alpha = alpha[idx] if isinstance(alpha, dict) else alpha
            sv = self.steering_vectors[idx]

            handle = self.model.model.layers[idx].register_forward_hook(
                self._make_injection_hook(sv, layer_alpha, beta_fn)
            )
            self._hooks.append(handle)

    def clear(self) -> None:
        """Remove all injection hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_beta_fn(
        self,
        strategy: str,
        speech_mult: float,
        action_mult: float,
    ) -> Callable[[], float]:
        """Return a callable that yields the current beta multiplier."""
        if strategy == "uniform":
            def beta_fn() -> float:
                return 1.0
        elif strategy == "action_boosted":
            def beta_fn() -> float:
                return action_mult if self.action_detector.is_in_action_region() else speech_mult
        elif strategy == "action_only":
            def beta_fn() -> float:
                return action_mult if self.action_detector.is_in_action_region() else 0.0
        else:
            raise ValueError(
                f"Unknown beta_strategy '{strategy}'. "
                "Choose from 'uniform', 'action_boosted', 'action_only'."
            )
        return beta_fn

    @staticmethod
    def _make_injection_hook(
        steering_vector: Tensor,
        alpha: float,
        beta_fn: Callable[[], float],
    ):
        """Create a forward hook that additively injects the steering vector."""
        def hook_fn(module, input, output):
            hidden = output[0]  # (batch, seq_len, hidden_dim)
            device = hidden.device
            sv = steering_vector.to(device)
            beta = beta_fn()
            # Add scaled steering vector to every position in the sequence.
            hidden = hidden + alpha * beta * sv
            # Reconstruct the output tuple with the modified hidden state.
            return (hidden,) + output[1:]
        return hook_fn
