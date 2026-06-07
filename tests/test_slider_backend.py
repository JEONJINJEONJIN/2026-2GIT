import unittest

import torch

from src.demo.slider_backend import (
    GenerationCache,
    SliderDemoBackend,
    normalize_persona,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeGenerator:
    model = object()
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def generate(self, prompt_or_messages, **kwargs):
        self.calls.append((prompt_or_messages, kwargs))
        return "generated text"


class EmptyThenTextGenerator:
    model = object()
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []

    def generate(self, prompt_or_messages, **kwargs):
        self.calls.append((prompt_or_messages, kwargs))
        if len(self.calls) == 1:
            return ""
        return "retry text"


class TestSliderBackend(unittest.TestCase):
    def test_cache_key_rounds_persona_to_two_decimals(self):
        first = GenerationCache.key({"O": 0.734}, "scenario", "baseline")
        second = GenerationCache.key({"O": 0.73}, "scenario", "baseline")

        self.assertEqual(first, second)

    def test_normalize_persona_accepts_short_keys(self):
        persona = normalize_persona({"O": 0.73, "E": 0.2})

        self.assertAlmostEqual(persona["openness"], 0.73)
        self.assertAlmostEqual(persona["extraversion"], 0.2)
        self.assertAlmostEqual(persona["agreeableness"], 0.5)

    def test_generate_returns_contract_dict_and_cache_hit(self):
        backend = SliderDemoBackend(
            generator=FakeGenerator(),
            model=object(),
            tokenizer=FakeTokenizer(),
            vector_bundle={
                "vectors": {
                    "trait_contrast": {
                        "openness": {1: torch.tensor([1.0, 0.0])},
                    }
                }
            },
            scenarios=[],
            layers=[1],
            prepopulate=False,
        )

        first = backend.generate({"O": 0.73}, "raw scenario", "baseline")
        second = backend.generate({"O": 0.731}, "raw scenario", "baseline")

        self.assertEqual(first, {
            "text": "generated text",
            "condition": "baseline",
            "cache_hit": False,
        })
        self.assertEqual(second, {
            "text": "generated text",
            "condition": "baseline",
            "cache_hit": True,
        })

    def test_generate_nonempty_retries_empty_output(self):
        generator = EmptyThenTextGenerator()
        backend = SliderDemoBackend(
            generator=generator,
            model=object(),
            tokenizer=FakeTokenizer(),
            vector_bundle={
                "vectors": {
                    "trait_contrast": {
                        "openness": {1: torch.tensor([1.0, 0.0])},
                    }
                }
            },
            scenarios=[],
            layers=[1],
            prepopulate=False,
        )

        text = backend.generate_nonempty("prompt")

        self.assertEqual(text, "retry text")
        self.assertEqual(len(generator.calls), 2)


if __name__ == "__main__":
    unittest.main()
