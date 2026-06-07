"""TRAIT Big Five internal-schema loader.

The loader expects already-converted local JSONL files. It does not silently
download or fallback because TRAIT may require explicit dataset access.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.data.bfi import BIG_FIVE_TRAITS


VALID_DIRECTIONS = {"high", "low"}
VALID_SPLITS = {"train", "dev", "test"}


@dataclass(frozen=True)
class TraitAction:
    """One action option in a TRAIT Big Five scenario."""

    id: str
    text: str
    trait_direction: str


@dataclass(frozen=True)
class TraitScenario:
    """Normalized TRAIT Big Five scenario."""

    scenario_id: str
    trait: str
    prompt: str
    actions: tuple[TraitAction, ...]
    split: str


def load_jsonl(path: str | Path) -> list[dict]:
    """Load a JSONL file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"TRAIT Big Five file not found: {path}. "
            "Create it from the TRAIT dataset before running v4 experiments."
        )
    rows = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def parse_trait_scenario(row: dict) -> TraitScenario:
    """Validate and parse one normalized TRAIT Big Five row."""

    scenario_id = str(row.get("scenario_id") or "").strip()
    trait = str(row.get("trait") or "").strip().lower()
    prompt = str(row.get("prompt") or "").strip()
    split = str(row.get("split") or "").strip().lower()
    raw_actions = row.get("actions")

    if not scenario_id:
        raise ValueError("TRAIT scenario missing scenario_id")
    if trait not in BIG_FIVE_TRAITS:
        raise ValueError(f"Unsupported TRAIT Big Five trait: {trait!r}")
    if not prompt:
        raise ValueError(f"TRAIT scenario {scenario_id} missing prompt")
    if split not in VALID_SPLITS:
        raise ValueError(f"TRAIT scenario {scenario_id} has invalid split: {split!r}")
    if not isinstance(raw_actions, list) or len(raw_actions) != 4:
        raise ValueError(f"TRAIT scenario {scenario_id} must contain exactly 4 actions")

    actions = tuple(parse_trait_action(action, scenario_id) for action in raw_actions)
    directions = {action.trait_direction for action in actions}
    if directions != VALID_DIRECTIONS:
        raise ValueError(
            f"TRAIT scenario {scenario_id} must include high and low actions"
        )
    return TraitScenario(
        scenario_id=scenario_id,
        trait=trait,
        prompt=prompt,
        actions=actions,
        split=split,
    )


def parse_trait_action(row: dict, scenario_id: str) -> TraitAction:
    """Validate and parse one action option."""

    action_id = str(row.get("id") or "").strip()
    text = str(row.get("text") or "").strip()
    direction = str(row.get("trait_direction") or "").strip().lower()
    if not action_id:
        raise ValueError(f"TRAIT scenario {scenario_id} has action without id")
    if not text:
        raise ValueError(f"TRAIT scenario {scenario_id} action {action_id} missing text")
    if direction not in VALID_DIRECTIONS:
        raise ValueError(
            f"TRAIT scenario {scenario_id} action {action_id} has invalid direction: "
            f"{direction!r}"
        )
    return TraitAction(id=action_id, text=text, trait_direction=direction)


def load_trait_scenarios(
    path: str | Path,
    allowed_splits: Iterable[str] | None = None,
) -> list[TraitScenario]:
    """Load normalized TRAIT Big Five scenarios, optionally filtering by split."""

    scenarios = [parse_trait_scenario(row) for row in load_jsonl(path)]
    if allowed_splits is None:
        return scenarios
    allowed = {split.lower() for split in allowed_splits}
    invalid = allowed - VALID_SPLITS
    if invalid:
        raise ValueError(f"Invalid allowed split(s): {sorted(invalid)}")
    return [scenario for scenario in scenarios if scenario.split in allowed]


def flatten_trait_actions(scenarios: Iterable[TraitScenario]) -> list[dict]:
    """Return action rows suitable for CSV/JSONL reporting."""

    rows = []
    for scenario in scenarios:
        for action in scenario.actions:
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "trait": scenario.trait,
                    "split": scenario.split,
                    "action_id": action.id,
                    "text": action.text,
                    "trait_direction": action.trait_direction,
                }
            )
    return rows


def build_trait_contrastive_pairs(
    scenarios: Iterable[TraitScenario],
    trait: str | None = None,
) -> list[dict]:
    """Build high-vs-low contrastive pairs for trait steering extraction.

    Positive text always represents the high trait direction and negative text
    represents the low direction. Low-target steering is handled later by
    negating the extracted vector.
    """

    trait_filter = trait.lower() if trait else None
    if trait_filter is not None and trait_filter not in BIG_FIVE_TRAITS:
        raise ValueError(f"Unsupported TRAIT Big Five trait: {trait!r}")

    pairs: list[dict] = []
    for scenario in scenarios:
        if trait_filter is not None and scenario.trait != trait_filter:
            continue
        high_actions = [
            action for action in scenario.actions
            if action.trait_direction == "high"
        ]
        low_actions = [
            action for action in scenario.actions
            if action.trait_direction == "low"
        ]
        if not high_actions or not low_actions:
            raise ValueError(
                f"TRAIT scenario {scenario.scenario_id} must contain high and low actions"
            )
        for high_index, high_action in enumerate(high_actions, start=1):
            for low_index, low_action in enumerate(low_actions, start=1):
                pair_id = (
                    f"{scenario.scenario_id}_{scenario.trait}_"
                    f"high_low_{high_index:02d}_{low_index:02d}"
                )
                pairs.append(
                    {
                        "id": pair_id,
                        "positive": format_trait_contrastive_text(
                            scenario=scenario,
                            action=high_action,
                            direction="high",
                        ),
                        "negative": format_trait_contrastive_text(
                            scenario=scenario,
                            action=low_action,
                            direction="low",
                        ),
                        "trait": scenario.trait,
                        "positive_direction": "high",
                        "negative_direction": "low",
                        "situation": scenario.prompt,
                        "positive_response": high_action.text,
                        "negative_response": low_action.text,
                        "positive_action_id": high_action.id,
                        "negative_action_id": low_action.id,
                        "source_template": scenario.scenario_id,
                        "split": scenario.split,
                        "strategy": "trait_bigfive_high_vs_low_action_grid",
                    }
                )
    return pairs


def format_trait_contrastive_text(
    scenario: TraitScenario,
    action: TraitAction,
    direction: str,
) -> str:
    """Format one side of a TRAIT contrastive pair for activation extraction."""

    direction = direction.lower()
    if direction not in VALID_DIRECTIONS:
        raise ValueError(f"Invalid trait direction: {direction!r}")
    trait_label = scenario.trait.replace("_", " ")
    return (
        "[INST] "
        f"You are a {direction}-{trait_label} NPC in a fictional game. "
        "Read the situation and produce the provided in-character action response.\n"
        f"Situation: {scenario.prompt} "
        "[/INST]\n"
        f"{action.text}"
    )


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    """Write rows to JSONL and return the number of rows written."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count
