"""BFI-44 scoring helpers for Big Five self-report measurements.

This module intentionally stores scoring metadata separately from item text.
Before adding full BFI-44 wording to the repository, verify redistribution
permissions for the item text.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


BIG_FIVE_TRAITS = {
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
    "extraversion",
}


@dataclass(frozen=True)
class BFIItem:
    """Scoring metadata for one BFI-44 item."""

    item_id: str
    trait: str
    reverse_scored: bool = False
    text: str = ""


def load_bfi_items(
    path: str | Path,
    text_path: str | Path | None = None,
) -> list[BFIItem]:
    """Load BFI item metadata from CSV.

    Required columns are ``item_id``, ``trait``, and ``reverse_scored``.
    ``text`` is optional because the repository may omit item wording until
    redistribution permission is confirmed. If ``text_path`` is provided, it
    must be a local CSV with ``item_id`` and ``text`` columns; text from that
    file overrides the metadata CSV text.
    """

    rows: list[BFIItem] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"item_id", "trait", "reverse_scored"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"BFI item CSV missing required columns: {sorted(missing)}")

        for index, row in enumerate(reader, start=2):
            trait = (row.get("trait") or "").strip().lower()
            if trait not in BIG_FIVE_TRAITS:
                raise ValueError(f"Unsupported BFI trait on row {index}: {trait!r}")
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                raise ValueError(f"Missing BFI item_id on row {index}")
            rows.append(
                BFIItem(
                    item_id=item_id,
                    trait=trait,
                    reverse_scored=parse_bool(row.get("reverse_scored", "")),
                    text=(row.get("text") or "").strip(),
                )
            )
    if text_path is None:
        return rows
    return apply_bfi_text_overlay(rows, load_bfi_text_overlay(text_path))


def load_bfi_text_overlay(path: str | Path) -> dict[str, str]:
    """Load local-only BFI item wording from ``item_id,text`` CSV."""

    overlay: dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"item_id", "text"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"BFI text overlay CSV missing required columns: {sorted(missing)}"
            )
        for index, row in enumerate(reader, start=2):
            item_id = (row.get("item_id") or "").strip()
            if not item_id:
                raise ValueError(f"Missing BFI item_id in text overlay row {index}")
            if item_id in overlay:
                raise ValueError(f"Duplicate BFI item_id in text overlay: {item_id}")
            overlay[item_id] = (row.get("text") or "").strip()
    return overlay


def apply_bfi_text_overlay(
    items: Iterable[BFIItem],
    text_by_item_id: Mapping[str, str],
) -> list[BFIItem]:
    """Return BFI items with local text overlaid by item id."""

    items = list(items)
    known_ids = {item.item_id for item in items}
    unknown_ids = set(text_by_item_id) - known_ids
    if unknown_ids:
        raise ValueError(f"BFI text overlay has unknown item_id(s): {sorted(unknown_ids)}")
    return [
        BFIItem(
            item_id=item.item_id,
            trait=item.trait,
            reverse_scored=item.reverse_scored,
            text=text_by_item_id.get(item.item_id, item.text),
        )
        for item in items
    ]


def parse_bool(value: str | bool) -> bool:
    """Parse common CSV boolean values."""

    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def reverse_likert_score(score: float, min_score: float = 1.0, max_score: float = 5.0) -> float:
    """Reverse-score a Likert value on an inclusive scale."""

    validate_likert_score(score, min_score=min_score, max_score=max_score)
    return min_score + max_score - score


def normalize_likert_score(
    score: float,
    min_score: float = 1.0,
    max_score: float = 5.0,
) -> float:
    """Normalize a Likert value to ``[0, 1]``."""

    validate_likert_score(score, min_score=min_score, max_score=max_score)
    if max_score <= min_score:
        raise ValueError("max_score must be greater than min_score")
    return float((score - min_score) / (max_score - min_score))


def validate_likert_score(
    score: float,
    min_score: float = 1.0,
    max_score: float = 5.0,
) -> None:
    """Raise if *score* is outside the expected Likert range."""

    if score < min_score or score > max_score:
        raise ValueError(
            f"Likert score {score!r} outside [{min_score!r}, {max_score!r}]"
        )


def score_bfi_responses(
    items: Iterable[BFIItem],
    responses: Mapping[str, float],
    min_score: float = 1.0,
    max_score: float = 5.0,
) -> dict[str, float]:
    """Compute normalized trait means from BFI item responses.

    Missing responses are an error. Silent omission would change trait means and
    make condition comparisons non-comparable.
    """

    by_trait: dict[str, list[float]] = {trait: [] for trait in BIG_FIVE_TRAITS}
    for item in items:
        if item.item_id not in responses:
            raise ValueError(f"Missing response for BFI item: {item.item_id}")
        score = float(responses[item.item_id])
        if item.reverse_scored:
            score = reverse_likert_score(score, min_score=min_score, max_score=max_score)
        by_trait[item.trait].append(
            normalize_likert_score(score, min_score=min_score, max_score=max_score)
        )

    return {
        trait: float(sum(values) / len(values))
        for trait, values in by_trait.items()
        if values
    }
