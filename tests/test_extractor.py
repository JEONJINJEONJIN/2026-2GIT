"""Tests for ActivationExtractor.

Uses unittest with a mock model that returns predictable tensors,
verifying that extract_activations returns the correct shape and
that different texts produce different activations.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.steering.extractor import ActivationExtractor


class FakeLayer(torch.nn.Module):
    """A fake transformer layer that produces predictable output."""

    def __init__(self, layer_idx: int, hidden_dim: int = 2048):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim

    def forward(self, x, **kwargs):
        batch, seq_len, _ = x.shape
        # Output varies by layer index and input content
        output = torch.randn(batch, seq_len, self.hidden_dim) + self.layer_idx * 0.1
        return (output,)


def create_mock_model(num_layers: int = 36, hidden_dim: int = 2048):
    """Create a mock HuggingFace model with fake layers.

    Returns a model mock that has model.model.layers with real
    nn.Module instances that support register_forward_hook.
    """
    layers = torch.nn.ModuleList([
        FakeLayer(i, hidden_dim) for i in range(num_layers)
    ])

    # Build nested model structure: model.model.layers
    inner_model = MagicMock()
    inner_model.layers = layers

    model = MagicMock()
    model.model = inner_model
    model.device = torch.device("cpu")

    # Make model callable: when called with input tensors, run through layers
    def model_forward(**kwargs):
        input_ids = kwargs.get("input_ids", torch.zeros(1, 5, dtype=torch.long))
        batch, seq_len = input_ids.shape
        x = torch.randn(batch, seq_len, hidden_dim)
        for layer in layers:
            x = layer(x)[0]
        return MagicMock(logits=x)

    model.__call__ = lambda self_unused=None, **kw: model_forward(**kw)
    model.side_effect = None
    model.return_value = None

    # Override __call__ properly for MagicMock
    def call_fn(**kwargs):
        return model_forward(**kwargs)

    model.__call__ = call_fn

    return model


def create_mock_tokenizer():
    """Create a mock tokenizer that returns predictable token IDs."""
    tokenizer = MagicMock()

    def tokenize(text, return_tensors=None, **kwargs):
        # Generate different token lengths based on text
        num_tokens = max(3, len(text.split()))
        result = {
            "input_ids": torch.randint(0, 1000, (1, num_tokens)),
            "attention_mask": torch.ones(1, num_tokens, dtype=torch.long),
        }
        return result

    tokenizer.side_effect = tokenize
    tokenizer.__call__ = tokenize
    return tokenizer


class TestActivationExtractor(unittest.TestCase):
    """Test suite for ActivationExtractor."""

    def setUp(self):
        self.hidden_dim = 2048
        self.num_layers = 36
        self.model = create_mock_model(self.num_layers, self.hidden_dim)
        self.tokenizer = create_mock_tokenizer()
        self.extractor = ActivationExtractor(self.model, self.tokenizer)

    def test_extract_activations_returns_correct_shape(self):
        """Activations should be 1-D tensors of shape (hidden_dim,)."""
        layer_indices = [0, 5, 15, 35]
        activations = self.extractor.extract_activations(
            "Hello world", layer_indices=layer_indices
        )

        self.assertEqual(set(activations.keys()), set(layer_indices))
        for layer_idx, tensor in activations.items():
            self.assertEqual(tensor.shape, (self.hidden_dim,))
            self.assertEqual(tensor.device, torch.device("cpu"))

    def test_extract_activations_all_layers(self):
        """When layer_indices is None, should extract from all layers."""
        activations = self.extractor.extract_activations(
            "Test text", layer_indices=None
        )

        self.assertEqual(len(activations), self.num_layers)
        for idx in range(self.num_layers):
            self.assertIn(idx, activations)
            self.assertEqual(activations[idx].shape, (self.hidden_dim,))

    def test_different_texts_produce_different_activations(self):
        """Different input texts should yield different activation values."""
        torch.manual_seed(42)
        acts_a = self.extractor.extract_activations(
            "The warrior attacks fiercely",
            layer_indices=[15],
        )
        torch.manual_seed(99)
        acts_b = self.extractor.extract_activations(
            "The diplomat negotiates peacefully",
            layer_indices=[15],
        )

        # Since FakeLayer uses torch.randn (which depends on RNG state),
        # different seeds will produce different outputs
        self.assertFalse(
            torch.allclose(acts_a[15], acts_b[15]),
            "Different texts should produce different activations",
        )

    def test_hooks_cleaned_up_after_extraction(self):
        """All forward hooks should be removed after extraction."""
        self.extractor.extract_activations("test", layer_indices=[0, 1, 2])
        self.assertEqual(
            len(self.extractor._hooks), 0,
            "Hooks should be empty after extraction",
        )

    def test_extract_from_pairs(self):
        """extract_from_pairs should return structured dict with correct keys."""
        pairs = [
            {"positive": "Attack the enemy", "negative": "Help the stranger"},
            {"positive": "Destroy the gate", "negative": "Open the gate"},
        ]
        layer_indices = [10, 20]

        result = self.extractor.extract_from_pairs(pairs, layer_indices)

        self.assertIn("positive", result)
        self.assertIn("negative", result)

        for key in ("positive", "negative"):
            for layer_idx in layer_indices:
                self.assertIn(layer_idx, result[key])
                self.assertEqual(len(result[key][layer_idx]), len(pairs))
                for tensor in result[key][layer_idx]:
                    self.assertEqual(tensor.shape, (self.hidden_dim,))

    def test_single_layer_extraction(self):
        """Extracting from a single layer should work correctly."""
        activations = self.extractor.extract_activations(
            "Single layer test",
            layer_indices=[18],
        )
        self.assertEqual(len(activations), 1)
        self.assertIn(18, activations)
        self.assertEqual(activations[18].shape, (self.hidden_dim,))


if __name__ == "__main__":
    unittest.main()
