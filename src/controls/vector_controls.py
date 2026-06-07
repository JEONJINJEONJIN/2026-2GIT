"""Vector control factories for Big Five AS/PAS diagnostics."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import torch


SUPPORTED_VECTOR_MODES = {
    "trait_contrast",
    "zero",
    "random",
    "unrelated_trait",
}


def resolve_condition_vectors(
    vector_bundle: Mapping[str, object],
    trait: str,
    target_direction: str,
    vector_mode: str | None,
    layer_indices: Sequence[int] | None = None,
    seed: int = 42,
) -> dict[int, torch.Tensor]:
    """Resolve condition-specific vectors from a v4 trait vector bundle."""

    mode = vector_mode or "trait_contrast"
    if mode not in SUPPORTED_VECTOR_MODES:
        raise ValueError(f"Unsupported vector_mode: {mode!r}")

    trait_vectors = get_trait_contrast_vectors(vector_bundle)
    if trait not in trait_vectors:
        raise ValueError(f"Vector bundle missing trait vector: {trait}")
    base = signed_trait_vectors(trait_vectors[trait], target_direction)
    base = filter_layers(base, layer_indices)

    if mode == "trait_contrast":
        return base
    if mode == "zero":
        return {layer: torch.zeros_like(vector) for layer, vector in base.items()}
    if mode == "random":
        return {
            layer: normalized_random_like(vector, stable_seed(seed, trait, target_direction, layer))
            for layer, vector in base.items()
        }
    if mode == "unrelated_trait":
        unrelated_trait = choose_unrelated_trait(trait_vectors, trait)
        return filter_layers(
            signed_trait_vectors(trait_vectors[unrelated_trait], target_direction),
            layer_indices,
        )
    raise AssertionError(f"Unhandled vector mode: {mode}")


def get_trait_contrast_vectors(vector_bundle: Mapping[str, object]) -> Mapping[str, Mapping[int, torch.Tensor]]:
    """Extract ``vectors['trait_contrast']`` from a bundle-like object."""

    vectors = vector_bundle.get("vectors", vector_bundle)
    if "trait_contrast" not in vectors:
        raise ValueError("Vector bundle missing vectors['trait_contrast']")
    return vectors["trait_contrast"]


def signed_trait_vectors(
    vectors: Mapping[int, torch.Tensor],
    target_direction: str,
) -> dict[int, torch.Tensor]:
    """Return +vectors for high targets and -vectors for low targets."""

    direction = target_direction.strip().lower()
    if direction not in {"high", "low"}:
        raise ValueError("target_direction must be high or low")
    sign = 1.0 if direction == "high" else -1.0
    return {int(layer): sign * vector for layer, vector in vectors.items()}


def filter_layers(
    vectors: Mapping[int, torch.Tensor],
    layer_indices: Sequence[int] | None,
) -> dict[int, torch.Tensor]:
    """Return only requested layers, validating presence."""

    if layer_indices is None:
        return {int(layer): vector for layer, vector in vectors.items()}
    missing = sorted(set(layer_indices) - {int(layer) for layer in vectors})
    if missing:
        raise ValueError(f"Vector bundle missing requested layer(s): {missing}")
    return {int(layer): vectors[int(layer)] for layer in layer_indices}


def normalized_random_like(reference: torch.Tensor, seed: int) -> torch.Tensor:
    """Create a deterministic unit random vector with the reference shape."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    vector = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
    norm = vector.float().norm()
    if norm.item() == 0.0:
        raise ValueError("Generated zero random vector")
    return (vector.float() / norm).to(dtype=reference.dtype)


def choose_unrelated_trait(
    trait_vectors: Mapping[str, Mapping[int, torch.Tensor]],
    trait: str,
) -> str:
    """Choose a stable non-target trait for unrelated-trait controls."""

    candidates = sorted(set(trait_vectors) - {trait})
    if not candidates:
        raise ValueError("Need at least one unrelated trait vector")
    return candidates[0]


def stable_seed(seed: int, *parts: object) -> int:
    """Build a deterministic 32-bit seed from a base seed and labels."""

    text = "::".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def should_normalize_injector_vectors(vector_mode: str | None) -> bool:
    """Return whether SteeringInjector should normalize vectors for this mode."""

    return vector_mode != "zero"


def resolve_slider_composite_vectors(
    vector_bundle: Mapping[str, object],
    trait_values: Mapping[str, float],
    layer_indices: Sequence[int] | None = None,
    clamp_norm: float = 1.0,
) -> dict[int, torch.Tensor]:
    """Resolve demo-time slider vectors as a clamped multi-trait composition.

    ``trait_values`` uses UI slider values in ``[0, 1]``. Each value is mapped to
    a signed strength in ``[-1, 1]`` with ``strength = 2 * (value - 0.5)``.

    The returned vectors already encode slider strength and norm clamping. Pass
    them to ``SteeringInjector(..., normalize_vectors=False)`` so that the UI
    strength is not erased by per-layer unit normalization.
    """

    if clamp_norm <= 0.0:
        raise ValueError("clamp_norm must be positive")

    trait_vectors = get_trait_contrast_vectors(vector_bundle)
    unknown = sorted(set(trait_values) - set(trait_vectors))
    if unknown:
        raise ValueError(f"Vector bundle missing trait vector(s): {unknown}")

    if layer_indices is None:
        all_layers = sorted({
            int(layer)
            for vectors in trait_vectors.values()
            for layer in vectors
        })
    else:
        all_layers = [int(layer) for layer in layer_indices]

    composite: dict[int, torch.Tensor] = {}
    for layer in all_layers:
        layer_sum = None
        reference = None
        for trait, value in trait_values.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Slider value for {trait} must be in [0, 1]")
            vectors = trait_vectors[trait]
            if layer not in {int(item) for item in vectors}:
                raise ValueError(f"Vector bundle missing layer {layer} for {trait}")
            vector = vectors[layer] if layer in vectors else vectors[str(layer)]
            reference = vector
            strength = 2.0 * (float(value) - 0.5)
            if strength == 0.0:
                continue
            unit = unit_vector(vector)
            term = unit.float() * strength
            layer_sum = term if layer_sum is None else layer_sum + term

        if reference is None:
            raise ValueError("trait_values cannot be empty")
        if layer_sum is None:
            layer_sum = torch.zeros_like(reference, dtype=torch.float32)
        norm = layer_sum.norm()
        if norm.item() > clamp_norm:
            layer_sum = layer_sum * (float(clamp_norm) / norm)
        composite[layer] = layer_sum.to(dtype=reference.dtype)

    return composite


def resolve_slider_persona_vectors(
    vector_bundle: Mapping[str, object],
    trait_values: Mapping[str, float],
    layer_indices: Sequence[int] | None,
    alpha: float = 2.0,
    clamp_x: float | None = None,
) -> dict[int, torch.Tensor]:
    """Resolve slider-controlled persona vectors with separated alpha.

    ``trait_values`` maps Big Five trait names to slider values in ``[0, 1]``.
    For each layer, each slider value becomes ``strength = 2 * (value - 0.5)``.
    Those signed strengths are applied to unit-normalized trait vectors and
    summed in unit space. ``clamp_x`` also applies in unit space before the AS
    multiplier, so ``clamp_x=2.0`` means at most twice a single unit trait axis.

    The returned vectors include the ``alpha`` multiplier. Callers should pass
    them to ``SteeringInjector(..., normalize_vectors=False)`` and use
    ``inject(..., alpha=1.0)`` to avoid erasing slider magnitude.
    """

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if clamp_x is not None and clamp_x <= 0.0:
        raise ValueError("clamp_x must be positive when provided")

    unit_vectors = resolve_slider_unit_vectors(
        vector_bundle,
        trait_values,
        layer_indices=layer_indices,
        clamp_x=clamp_x,
    )
    return {
        layer: (vector.float() * float(alpha)).to(dtype=vector.dtype, device=vector.device)
        for layer, vector in unit_vectors.items()
    }


def resolve_slider_unit_vectors(
    vector_bundle: Mapping[str, object],
    trait_values: Mapping[str, float],
    layer_indices: Sequence[int] | None,
    clamp_x: float | None = None,
) -> dict[int, torch.Tensor]:
    """Return slider composite vectors before alpha multiplication."""

    if clamp_x is not None and clamp_x <= 0.0:
        raise ValueError("clamp_x must be positive when provided")

    trait_vectors = get_trait_contrast_vectors(vector_bundle)
    unknown = sorted(set(trait_values) - set(trait_vectors))
    if unknown:
        raise ValueError(f"Vector bundle missing trait vector(s): {unknown}")
    if not trait_values:
        raise ValueError("trait_values cannot be empty")

    if layer_indices is None:
        all_layers = sorted({
            int(layer)
            for vectors in trait_vectors.values()
            for layer in vectors
        })
    else:
        all_layers = [int(layer) for layer in layer_indices]

    composite: dict[int, torch.Tensor] = {}
    for layer in all_layers:
        layer_sum = None
        reference = None
        for trait, value in trait_values.items():
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Slider value for {trait} must be in [0, 1]")
            vector = layer_vector(trait_vectors[trait], layer, trait)
            reference = vector
            strength = slider_strength(value)
            if strength == 0.0:
                continue
            term = unit_vector(vector) * strength
            layer_sum = term if layer_sum is None else layer_sum + term

        if reference is None:
            raise ValueError("trait_values cannot be empty")
        if layer_sum is None:
            layer_sum = torch.zeros_like(reference, dtype=torch.float32)
        norm = layer_sum.float().norm()
        if clamp_x is not None and norm.item() > clamp_x:
            layer_sum = layer_sum * (float(clamp_x) / norm)
        composite[layer] = layer_sum.to(dtype=reference.dtype, device=reference.device)
    return composite


def slider_strength(value: float) -> float:
    """Map a UI slider value in [0, 1] to a signed strength in [-1, 1]."""

    if value < 0.0 or value > 1.0:
        raise ValueError("Slider value must be in [0, 1]")
    return float(2.0 * (float(value) - 0.5))


def layer_vector(
    vectors: Mapping[int, torch.Tensor],
    layer: int,
    trait: str,
) -> torch.Tensor:
    """Return one layer vector while tolerating int-like keys."""

    if layer in vectors:
        return vectors[layer]
    str_layer = str(layer)
    if str_layer in vectors:
        return vectors[str_layer]
    raise ValueError(f"Vector bundle missing layer {layer} for {trait}")


def slider_unit_norms(
    vector_bundle: Mapping[str, object],
    trait_values: Mapping[str, float],
    layer_indices: Sequence[int],
    clamp_x: float | None = None,
) -> dict[int, float]:
    """Return per-layer unit-space norms for a slider persona."""

    vectors = resolve_slider_unit_vectors(
        vector_bundle,
        trait_values,
        layer_indices=layer_indices,
        clamp_x=clamp_x,
    )
    return {layer: float(vector.float().norm()) for layer, vector in vectors.items()}


def closed_form_slider_norm(
    vector_bundle: Mapping[str, object],
    trait_values: Mapping[str, float],
    layer_idx: int,
) -> float:
    """Compute the unclamped unit-space composite norm by pairwise cosines."""

    trait_vectors = get_trait_contrast_vectors(vector_bundle)
    strengths = {
        trait: slider_strength(value)
        for trait, value in trait_values.items()
    }
    squared = sum(strength * strength for strength in strengths.values())
    traits = list(strengths)
    for index, trait_a in enumerate(traits):
        vector_a = unit_vector(layer_vector(trait_vectors[trait_a], layer_idx, trait_a))
        for trait_b in traits[index + 1:]:
            vector_b = unit_vector(layer_vector(trait_vectors[trait_b], layer_idx, trait_b))
            cosine = float(torch.dot(vector_a, vector_b))
            squared += 2.0 * strengths[trait_a] * strengths[trait_b] * cosine
    return float(max(squared, 0.0) ** 0.5)


def unit_vector(vector: torch.Tensor) -> torch.Tensor:
    """Return a float unit vector with validation."""

    norm = vector.float().norm()
    if norm.item() == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return vector.float() / norm
