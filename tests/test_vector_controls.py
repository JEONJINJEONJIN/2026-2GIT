import unittest

import torch

from src.controls.vector_controls import (
    choose_unrelated_trait,
    closed_form_slider_norm,
    normalized_random_like,
    resolve_condition_vectors,
    resolve_slider_composite_vectors,
    resolve_slider_persona_vectors,
    should_normalize_injector_vectors,
    slider_strength,
    stable_seed,
)


class TestVectorControls(unittest.TestCase):
    def setUp(self):
        self.bundle = {
            "vectors": {
                "trait_contrast": {
                    "agreeableness": {
                        1: torch.tensor([1.0, 0.0]),
                        2: torch.tensor([0.0, 1.0]),
                    },
                    "openness": {
                        1: torch.tensor([0.5, 0.5]),
                        2: torch.tensor([0.25, 0.75]),
                    },
                }
            }
        }

    def test_trait_contrast_high_and_low(self):
        high = resolve_condition_vectors(
            self.bundle,
            trait="agreeableness",
            target_direction="high",
            vector_mode="trait_contrast",
        )
        low = resolve_condition_vectors(
            self.bundle,
            trait="agreeableness",
            target_direction="low",
            vector_mode="trait_contrast",
        )
        self.assertTrue(torch.equal(high[1], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(low[1], torch.tensor([-1.0, -0.0])))

    def test_zero_mode_preserves_shape(self):
        vectors = resolve_condition_vectors(
            self.bundle,
            trait="agreeableness",
            target_direction="high",
            vector_mode="zero",
        )
        self.assertTrue(torch.equal(vectors[1], torch.tensor([0.0, 0.0])))
        self.assertFalse(should_normalize_injector_vectors("zero"))

    def test_random_mode_is_deterministic_and_unit_norm(self):
        first = resolve_condition_vectors(
            self.bundle,
            trait="agreeableness",
            target_direction="high",
            vector_mode="random",
            seed=123,
        )
        second = resolve_condition_vectors(
            self.bundle,
            trait="agreeableness",
            target_direction="high",
            vector_mode="random",
            seed=123,
        )
        self.assertTrue(torch.allclose(first[1], second[1]))
        self.assertAlmostEqual(first[1].float().norm().item(), 1.0)

    def test_unrelated_trait_uses_non_target_trait(self):
        vectors = resolve_condition_vectors(
            self.bundle,
            trait="agreeableness",
            target_direction="high",
            vector_mode="unrelated_trait",
        )
        self.assertTrue(torch.equal(vectors[1], torch.tensor([0.5, 0.5])))

    def test_layer_filter_validates_missing_layers(self):
        with self.assertRaises(ValueError):
            resolve_condition_vectors(
                self.bundle,
                trait="agreeableness",
                target_direction="high",
                vector_mode="trait_contrast",
                layer_indices=[3],
            )

    def test_choose_unrelated_trait_requires_candidate(self):
        with self.assertRaises(ValueError):
            choose_unrelated_trait({"agreeableness": {1: torch.tensor([1.0])}}, "agreeableness")

    def test_stable_seed(self):
        self.assertEqual(stable_seed(1, "a"), stable_seed(1, "a"))
        self.assertNotEqual(stable_seed(1, "a"), stable_seed(1, "b"))

    def test_normalized_random_like(self):
        vector = normalized_random_like(torch.zeros(4), seed=5)
        self.assertEqual(tuple(vector.shape), (4,))
        self.assertAlmostEqual(vector.float().norm().item(), 1.0)

    def test_slider_composite_neutral_is_zero(self):
        vectors = resolve_slider_composite_vectors(
            self.bundle,
            {"agreeableness": 0.5, "openness": 0.5},
            layer_indices=[1],
        )

        self.assertTrue(torch.equal(vectors[1], torch.tensor([0.0, 0.0])))

    def test_slider_composite_preserves_signed_strength(self):
        high = resolve_slider_composite_vectors(
            self.bundle,
            {"agreeableness": 1.0},
            layer_indices=[1],
        )
        low = resolve_slider_composite_vectors(
            self.bundle,
            {"agreeableness": 0.0},
            layer_indices=[1],
        )

        self.assertTrue(torch.allclose(high[1], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.allclose(low[1], torch.tensor([-1.0, 0.0])))

    def test_slider_composite_clamps_multi_trait_norm(self):
        vectors = resolve_slider_composite_vectors(
            self.bundle,
            {"agreeableness": 1.0, "openness": 1.0},
            layer_indices=[1],
            clamp_norm=1.0,
        )

        self.assertLessEqual(vectors[1].float().norm().item(), 1.000001)

    def test_slider_composite_rejects_out_of_range_value(self):
        with self.assertRaises(ValueError):
            resolve_slider_composite_vectors(
                self.bundle,
                {"agreeableness": 1.2},
                layer_indices=[1],
            )

    def test_slider_strength_mapping(self):
        self.assertAlmostEqual(slider_strength(0.5), 0.0)
        self.assertAlmostEqual(slider_strength(1.0), 1.0)
        self.assertAlmostEqual(slider_strength(0.0), -1.0)

    def test_slider_persona_vectors_keep_alpha_separate(self):
        vectors = resolve_slider_persona_vectors(
            self.bundle,
            {"agreeableness": 1.0},
            layer_indices=[1],
            alpha=2.0,
            clamp_x=None,
        )

        self.assertTrue(torch.allclose(vectors[1], torch.tensor([2.0, 0.0])))

    def test_slider_persona_vectors_clamp_before_alpha(self):
        bundle = {
            "vectors": {
                "trait_contrast": {
                    "a": {1: torch.tensor([1.0, 0.0])},
                    "b": {1: torch.tensor([0.0, 1.0])},
                }
            }
        }

        vectors = resolve_slider_persona_vectors(
            bundle,
            {"a": 1.0, "b": 1.0},
            layer_indices=[1],
            alpha=2.0,
            clamp_x=1.0,
        )

        self.assertAlmostEqual(vectors[1].float().norm().item(), 2.0, places=6)

    def test_closed_form_slider_norm_matches_geometry(self):
        bundle = {
            "vectors": {
                "trait_contrast": {
                    "a": {1: torch.tensor([1.0, 0.0])},
                    "b": {1: torch.tensor([0.0, 1.0])},
                }
            }
        }

        norm = closed_form_slider_norm(bundle, {"a": 1.0, "b": 1.0}, layer_idx=1)

        self.assertAlmostEqual(norm, 2 ** 0.5)


if __name__ == "__main__":
    unittest.main()
