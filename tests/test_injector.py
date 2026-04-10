"""Tests for SteeringInjector (HookManager) and ActionTagDetector.

Tests that the HookManager correctly modifies hidden states when
a steering hook is registered, and verifies ActionTagDetector-style
state tracking for position-aware beta scheduling.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.hooks import HookManager


class FakeTransformerLayer(nn.Module):
    """A simple fake transformer layer for testing hooks."""

    def __init__(self, hidden_dim: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.eye_(self.linear.weight)  # Identity transform

    def forward(self, x):
        return (self.linear(x),)


def create_test_model(num_layers: int = 3, hidden_dim: int = 64):
    """Create a minimal model with the structure model.model.layers."""
    layers = nn.ModuleList([
        FakeTransformerLayer(hidden_dim) for _ in range(num_layers)
    ])

    inner = MagicMock()
    inner.layers = layers

    model = MagicMock()
    model.model = inner

    return model, layers


class ActionTagDetector:
    """Tracks whether generation is in speech or action region.

    This is a simple state machine that detects <Action> and </Action>
    tags in generated tokens to determine position-aware beta values.
    """

    def __init__(self):
        self.in_action = False
        self.buffer = ""

    def update(self, token: str) -> bool:
        """Update state with a new token.

        Args:
            token: The newly generated token string.

        Returns:
            True if currently inside an <Action>...</Action> region.
        """
        self.buffer += token

        if "<Action>" in self.buffer and "</Action>" not in self.buffer:
            self.in_action = True
        elif "</Action>" in self.buffer:
            self.in_action = False
            # Reset buffer after closing tag
            idx = self.buffer.index("</Action>") + len("</Action>")
            self.buffer = self.buffer[idx:]

        return self.in_action

    def reset(self):
        """Reset the detector state."""
        self.in_action = False
        self.buffer = ""

    @property
    def is_in_action(self) -> bool:
        """Whether the detector is currently inside an action tag."""
        return self.in_action


class TestActionTagDetector(unittest.TestCase):
    """Test ActionTagDetector state tracking."""

    def test_initial_state(self):
        """Detector should start outside action region."""
        detector = ActionTagDetector()
        self.assertFalse(detector.is_in_action)

    def test_detects_action_start(self):
        """Should detect entering an <Action> region."""
        detector = ActionTagDetector()

        detector.update("[Speech] I will fight you! ")
        self.assertFalse(detector.is_in_action)

        detector.update("<Action>")
        self.assertTrue(detector.is_in_action)

    def test_detects_action_end(self):
        """Should detect leaving an </Action> region."""
        detector = ActionTagDetector()

        detector.update("<Action>")
        self.assertTrue(detector.is_in_action)

        detector.update("attack")
        self.assertTrue(detector.is_in_action)

        detector.update("</Action>")
        self.assertFalse(detector.is_in_action)

    def test_incremental_token_detection(self):
        """Should handle tags split across multiple tokens."""
        detector = ActionTagDetector()

        detector.update("<")
        self.assertFalse(detector.is_in_action)

        detector.update("Action")
        self.assertFalse(detector.is_in_action)

        detector.update(">")
        self.assertTrue(detector.is_in_action)

    def test_reset(self):
        """Reset should clear all state."""
        detector = ActionTagDetector()
        detector.update("<Action>attack")
        self.assertTrue(detector.is_in_action)

        detector.reset()
        self.assertFalse(detector.is_in_action)
        self.assertEqual(detector.buffer, "")

    def test_full_generation_sequence(self):
        """Test a full generation sequence with speech and action."""
        detector = ActionTagDetector()

        tokens = [
            "[Speech]", " I", " challenge", " you", "! ",
            "<Action>", "attack", "</Action>",
        ]
        expected_in_action = [
            False, False, False, False, False,
            True, True, False,
        ]

        for token, expected in zip(tokens, expected_in_action):
            detector.update(token)
            self.assertEqual(
                detector.is_in_action, expected,
                f"After token '{token}': expected in_action={expected}",
            )


class TestHookManagerInjection(unittest.TestCase):
    """Test that HookManager correctly modifies hidden states."""

    def setUp(self):
        self.hidden_dim = 64
        self.model, self.layers = create_test_model(
            num_layers=3, hidden_dim=self.hidden_dim
        )

    def test_steering_hook_modifies_output(self):
        """Steering hook should add alpha * steering_vector to hidden states."""
        steering_vector = torch.ones(self.hidden_dim) * 0.5
        alpha = 2.0
        layer_idx = 1

        # Get baseline output
        x = torch.randn(1, 10, self.hidden_dim)
        baseline_output = self.layers[layer_idx](x)[0]

        # Register steering hook and get modified output
        with HookManager() as hm:
            hm.register_steering_hook(
                self.model, layer_idx, steering_vector, alpha
            )
            modified_output = self.layers[layer_idx](x)[0]

        # Expected modification: baseline + alpha * steering_vector
        expected = baseline_output + alpha * steering_vector
        self.assertTrue(
            torch.allclose(modified_output, expected, atol=1e-5),
            "Steering hook should add alpha * steering_vector to output",
        )

    def test_hooks_removed_after_context_manager(self):
        """Hooks should be automatically removed when context manager exits."""
        steering_vector = torch.randn(self.hidden_dim)

        with HookManager() as hm:
            hm.register_steering_hook(self.model, 0, steering_vector, 1.0)
            self.assertEqual(len(hm._handles), 1)

        # After exit, handles should be cleared
        self.assertEqual(len(hm._handles), 0)

    def test_multiple_hooks(self):
        """Multiple hooks on different layers should all be applied."""
        sv1 = torch.ones(self.hidden_dim) * 0.1
        sv2 = torch.ones(self.hidden_dim) * 0.2

        with HookManager() as hm:
            hm.register_steering_hook(self.model, 0, sv1, 1.0)
            hm.register_steering_hook(self.model, 2, sv2, 1.0)
            self.assertEqual(len(hm._handles), 2)

    def test_steering_with_beta_fn(self):
        """Steering hook with a beta_fn should apply position-wise scaling."""
        steering_vector = torch.ones(self.hidden_dim)
        alpha = 1.0
        layer_idx = 0

        # Beta function that scales action positions by 2x
        def beta_fn(seq_len: int) -> torch.Tensor:
            beta = torch.ones(seq_len)
            # Boost last 3 positions (simulating action region)
            beta[-3:] = 2.0
            return beta

        x = torch.randn(1, 10, self.hidden_dim)
        baseline_output = self.layers[layer_idx](x)[0]

        with HookManager() as hm:
            hm.register_steering_hook(
                self.model, layer_idx, steering_vector, alpha,
                beta_fn=beta_fn,
            )
            modified_output = self.layers[layer_idx](x)[0]

        # Check that non-action positions got beta=1.0 scaling
        delta_early = modified_output[0, 0, :] - baseline_output[0, 0, :]
        self.assertTrue(
            torch.allclose(delta_early, steering_vector, atol=1e-5),
            "Early positions should get beta=1.0",
        )

        # Check that action positions got beta=2.0 scaling
        delta_late = modified_output[0, -1, :] - baseline_output[0, -1, :]
        self.assertTrue(
            torch.allclose(delta_late, 2.0 * steering_vector, atol=1e-5),
            "Action positions should get beta=2.0",
        )

    def test_remove_all_explicit(self):
        """Explicit remove_all should clear all hooks."""
        hm = HookManager()
        sv = torch.randn(self.hidden_dim)
        hm.register_steering_hook(self.model, 0, sv, 1.0)
        hm.register_steering_hook(self.model, 1, sv, 1.0)
        self.assertEqual(len(hm._handles), 2)

        hm.remove_all()
        self.assertEqual(len(hm._handles), 0)

    def test_no_modification_without_hook(self):
        """Without any hook, the layer output should be unchanged."""
        x = torch.randn(1, 5, self.hidden_dim)

        with HookManager():
            output = self.layers[0](x)[0]

        # With identity weights and no hook, output should match input
        self.assertTrue(
            torch.allclose(output, x, atol=1e-5),
            "Without hooks, output should match identity transform",
        )


if __name__ == "__main__":
    unittest.main()
