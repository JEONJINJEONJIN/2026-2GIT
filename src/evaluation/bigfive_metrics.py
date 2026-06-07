"""Metrics for Big Five speech-action consistency experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


VALID_TARGET_DIRECTIONS = {"high", "low"}


def validate_unit_interval(value: float, name: str) -> None:
    """Raise if *value* is outside ``[0, 1]``."""

    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def speech_action_consistency(bfi_score: float, trait_score: float) -> float:
    """Return ``1 - abs(BFI_score - TRAIT_score)``.

    Both scores must already be normalized to ``[0, 1]`` on the same trait axis.
    """

    validate_unit_interval(bfi_score, "bfi_score")
    validate_unit_interval(trait_score, "trait_score")
    return float(1.0 - abs(bfi_score - trait_score))


def target_direction_score(high_direction_score: float, target_direction: str) -> float:
    """Convert a high-direction score into target-direction score.

    For a high target, the target score is the high-direction score. For a low
    target, it is ``1 - high_direction_score``.
    """

    validate_unit_interval(high_direction_score, "high_direction_score")
    direction = normalize_target_direction(target_direction)
    if direction == "high":
        return float(high_direction_score)
    return float(1.0 - high_direction_score)


def target_attainment(
    high_direction_score: float,
    target_direction: str,
    neutral_point: float = 0.5,
) -> float:
    """Return signed movement toward the target direction from neutral."""

    validate_unit_interval(neutral_point, "neutral_point")
    target_score = target_direction_score(high_direction_score, target_direction)
    return float(target_score - neutral_point)


def trait_high_probability_mass(
    action_probabilities: Mapping[str, float],
    action_directions: Mapping[str, str],
) -> float:
    """Sum probability mass assigned to high-direction actions.

    Probabilities are normalized if they sum to a positive value other than 1.
    This supports both softmax probabilities and unnormalized positive weights.
    """

    if not action_probabilities:
        raise ValueError("action_probabilities cannot be empty")
    missing = set(action_probabilities) - set(action_directions)
    if missing:
        raise ValueError(f"Missing action direction(s): {sorted(missing)}")

    total = 0.0
    high_mass = 0.0
    for action_id, probability in action_probabilities.items():
        if probability < 0.0:
            raise ValueError(f"Action probability cannot be negative: {action_id}")
        direction = normalize_target_direction(action_directions[action_id])
        total += float(probability)
        if direction == "high":
            high_mass += float(probability)

    if total <= 0.0:
        raise ValueError("Action probabilities must sum to a positive value")
    return float(high_mass / total)


def pas_loglik_agreement(pas_action_id: str | None, loglik_action_id: str | None) -> bool:
    """Return whether PAS and loglik selected the same concrete action."""

    if not pas_action_id or not loglik_action_id:
        return False
    return pas_action_id == loglik_action_id


def mean_consistency_by_trait(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """Aggregate rows containing ``trait`` and ``consistency`` fields."""

    grouped: dict[str, list[float]] = {}
    for row in rows:
        trait = str(row.get("trait") or "").strip().lower()
        if not trait:
            raise ValueError("Missing trait in consistency row")
        consistency = float(row["consistency"])
        validate_unit_interval(consistency, "consistency")
        grouped.setdefault(trait, []).append(consistency)

    return {
        trait: float(sum(values) / len(values))
        for trait, values in grouped.items()
    }


def normalize_target_direction(target_direction: str) -> str:
    """Normalize and validate a target direction label."""

    direction = target_direction.strip().lower()
    if direction not in VALID_TARGET_DIRECTIONS:
        raise ValueError(f"target_direction must be high or low, got {target_direction!r}")
    return direction

