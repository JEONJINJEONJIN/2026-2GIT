"""Side-effect metrics for generated NPC responses.

These metrics are intentionally model-independent. They catch formatting and
surface-quality regressions before evaluator-based coherence or perplexity
scores are available.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any

from src.generation.parser import ActionParser

SPEECH_TAG_PATTERN = re.compile(r"\[Speech\]", re.IGNORECASE)
ACTION_TAG_PATTERN = re.compile(r"<Action>\s*.*?\s*</Action>", re.IGNORECASE | re.DOTALL)


def response_length_chars(text: str) -> int:
    """Return character count for a generated response."""

    return len(text)


def response_length_words(text: str) -> int:
    """Return whitespace-token count for a generated response."""

    return len(text.split())


def has_speech_tag(text: str) -> bool:
    """Return whether the response contains a [Speech] tag."""

    return SPEECH_TAG_PATTERN.search(text) is not None


def has_action_tag(text: str) -> bool:
    """Return whether the response contains an <Action>...</Action> tag."""

    return ACTION_TAG_PATTERN.search(text) is not None


def format_validity(parsed: Mapping[str, Any], require_speech: bool = True) -> bool:
    """Return whether parsed output satisfies the NPC response contract.

    A valid response must include a parseable valid action. If speech is
    required, the extracted speech field must also be non-empty.
    """

    action_ok = bool(parsed.get("parse_success"))
    speech_ok = bool(str(parsed.get("speech", "")).strip())
    return action_ok and (speech_ok or not require_speech)


def build_side_effect_row(
    text: str,
    valid_actions: Iterable[str] | None = None,
    *,
    condition: str | None = None,
    scenario_id: str | None = None,
    trait: str | None = None,
    target_direction: str | None = None,
    coherence_score: float | None = None,
    perplexity: float | None = None,
    require_speech: bool = True,
) -> dict[str, Any]:
    """Build one side-effect metric row for a generated NPC response."""

    parsed = ActionParser().parse(text, valid_actions=valid_actions)
    valid = format_validity(parsed, require_speech=require_speech)
    return {
        "condition": condition,
        "scenario_id": scenario_id,
        "trait": trait,
        "target_direction": target_direction,
        "response_length_chars": response_length_chars(text),
        "response_length_words": response_length_words(text),
        "has_speech_tag": has_speech_tag(text),
        "has_action_tag": has_action_tag(text),
        "parse_success": bool(parsed["parse_success"]),
        "format_validity": valid,
        "parsed_action": parsed["action"],
        "speech": parsed["speech"],
        "coherence_score": coherence_score,
        "perplexity": perplexity,
        "raw": text,
    }


def summarize_side_effect_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Summarize side-effect rows with aggregate rates and length means."""

    rows = list(rows)
    if not rows:
        return {
            "n": 0.0,
            "parse_success_rate": 0.0,
            "format_validity_rate": 0.0,
            "mean_response_length_chars": 0.0,
            "mean_response_length_words": 0.0,
        }

    n = float(len(rows))
    return {
        "n": n,
        "parse_success_rate": sum(bool(row.get("parse_success")) for row in rows) / n,
        "format_validity_rate": sum(bool(row.get("format_validity")) for row in rows) / n,
        "mean_response_length_chars": sum(
            float(row.get("response_length_chars", 0.0)) for row in rows
        )
        / n,
        "mean_response_length_words": sum(
            float(row.get("response_length_words", 0.0)) for row in rows
        )
        / n,
    }
