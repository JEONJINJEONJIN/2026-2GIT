"""Convert Team A CSV datasets into project JSONL inputs.

The project runtime expects:
- contrastive pairs as JSONL rows with positive/negative prompt strings.
- scenarios as JSONL rows with context plus action choices.

This script converts the Korean CSV files prepared by Team A into those
formats without overwriting the existing baseline files by default.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONTRASTIVE_CSV = PROJECT_ROOT / "contrastive_pairs.csv"
DEFAULT_SCENARIOS_CSV = PROJECT_ROOT / "scenarios_actions.csv"

DEFAULT_CONTRASTIVE_OUTPUT = (
    PROJECT_ROOT / "data" / "contrastive_pairs" / "team_a_aggressive_cooperative.jsonl"
)
DEFAULT_AGGRESSIVE_NEUTRAL_OUTPUT = (
    PROJECT_ROOT / "data" / "contrastive_pairs" / "team_a_aggressive_neutral.jsonl"
)
DEFAULT_COOPERATIVE_NEUTRAL_OUTPUT = (
    PROJECT_ROOT / "data" / "contrastive_pairs" / "team_a_cooperative_neutral.jsonl"
)
DEFAULT_SCENARIOS_OUTPUT = PROJECT_ROOT / "data" / "scenarios" / "team_a_scenarios.jsonl"
DEFAULT_ACTIONS_OUTPUT = PROJECT_ROOT / "data" / "actions" / "team_a_action_definitions.jsonl"

PIPELINE_CONTRASTIVE_OUTPUT = (
    PROJECT_ROOT / "data" / "contrastive_pairs" / "aggressive_cooperative.jsonl"
)
PIPELINE_AGGRESSIVE_NEUTRAL_OUTPUT = (
    PROJECT_ROOT / "data" / "contrastive_pairs" / "aggressive_neutral.jsonl"
)
PIPELINE_COOPERATIVE_NEUTRAL_OUTPUT = (
    PROJECT_ROOT / "data" / "contrastive_pairs" / "cooperative_neutral.jsonl"
)
PIPELINE_SCENARIOS_OUTPUT = PROJECT_ROOT / "data" / "scenarios" / "scenarios.jsonl"
PIPELINE_ACTIONS_OUTPUT = PROJECT_ROOT / "data" / "actions" / "action_definitions.jsonl"


GENRE_MAP = {
    "판타지": "fantasy_rpg",
    "SF": "sci_fi",
    "현대": "modern",
    "서부": "western",
    "해적": "pirate",
    "궁중": "court_intrigue",
    "전쟁": "war",
    "상단": "merchant",
    "범죄": "crime",
    "무협": "martial_arts",
}


CONTRASTIVE_PROMPT_TEMPLATE = (
    "[INST] You are {article} {persona} NPC in a fictional game. "
    "Read the situation and produce the provided in-character response.\n"
    "Situation: {situation} [/INST]\n"
    "{response}"
)


def require_columns(rows: list[dict[str, str]], required: Iterable[str], source: Path) -> None:
    missing = [column for column in required if not rows or column not in rows[0]]
    if missing:
        raise ValueError(f"{source} is missing required column(s): {missing}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path} has no data rows.")
    return rows


def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_contrastive_rows(csv_rows: list[dict[str, str]]) -> list[dict]:
    require_columns(
        csv_rows,
        ["pair_id", "situation", "aggressive", "cooperative"],
        DEFAULT_CONTRASTIVE_CSV,
    )

    seen_ids: set[str] = set()
    output = []
    for row in csv_rows:
        pair_id = row["pair_id"].strip()
        if pair_id in seen_ids:
            raise ValueError(f"Duplicate contrastive pair_id: {pair_id}")
        seen_ids.add(pair_id)

        situation = row["situation"].strip()
        aggressive = row["aggressive"].strip()
        cooperative = row["cooperative"].strip()
        if not situation or not aggressive or not cooperative:
            raise ValueError(f"Empty field in contrastive pair: {pair_id}")

        output.append(
            {
                "id": pair_id,
                "positive": CONTRASTIVE_PROMPT_TEMPLATE.format(
                    article="an",
                    persona="aggressive",
                    situation=situation,
                    response=aggressive,
                ),
                "negative": CONTRASTIVE_PROMPT_TEMPLATE.format(
                    article="a",
                    persona="cooperative",
                    situation=situation,
                    response=cooperative,
                ),
                "situation": situation,
                "positive_response": aggressive,
                "negative_response": cooperative,
                "source_template": pair_id,
                "strategy": "team_a_csv",
            }
        )
    return output


def build_action_contrastive_pairs(scenarios: list[dict]) -> list[dict]:
    """Build all aggressive-vs-cooperative action pairs from scenario choices."""
    rows = []
    for scenario in scenarios:
        aggressive_actions = [
            action for action in scenario.get("actions", [])
            if action.get("persona_alignment") == "aggressive"
        ]
        cooperative_actions = [
            action for action in scenario.get("actions", [])
            if action.get("persona_alignment") == "cooperative"
        ]
        if not aggressive_actions or not cooperative_actions:
            raise ValueError(
                f"Need aggressive and cooperative actions in scenario: {scenario.get('id')}"
            )
        for aggressive_index, aggressive in enumerate(aggressive_actions, start=1):
            for cooperative_index, cooperative in enumerate(cooperative_actions, start=1):
                pair_id = (
                    f"{scenario['id']}_aggressive_cooperative_"
                    f"{aggressive_index:02d}_{cooperative_index:02d}"
                )
                rows.append({
                    "id": pair_id,
                    "positive": CONTRASTIVE_PROMPT_TEMPLATE.format(
                        article="an",
                        persona="aggressive",
                        situation=scenario["context"],
                        response=aggressive["description"],
                    ),
                    "negative": CONTRASTIVE_PROMPT_TEMPLATE.format(
                        article="a",
                        persona="cooperative",
                        situation=scenario["context"],
                        response=cooperative["description"],
                    ),
                    "situation": scenario["context"],
                    "positive_response": aggressive["description"],
                    "negative_response": cooperative["description"],
                    "positive_action_id": aggressive["id"],
                    "negative_action_id": cooperative["id"],
                    "source_template": scenario["id"],
                    "strategy": "team_a_aggressive_vs_cooperative_action_grid",
                })
    return rows


def build_scenario_rows(csv_rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    required = [
        "scenario_id",
        "genre",
        "situation",
        "action_agg_1",
        "action_agg_2",
        "action_neu",
        "action_coop_1",
        "action_coop_2",
    ]
    require_columns(csv_rows, required, DEFAULT_SCENARIOS_CSV)

    seen_ids: set[str] = set()
    scenarios = []
    actions_flat = []
    action_specs = [
        ("action_agg_1", "aggressive_1", "aggressive"),
        ("action_agg_2", "aggressive_2", "aggressive"),
        ("action_neu", "neutral", "neutral"),
        ("action_coop_1", "cooperative_1", "cooperative"),
        ("action_coop_2", "cooperative_2", "cooperative"),
    ]

    for row in csv_rows:
        scenario_id = row["scenario_id"].strip()
        if scenario_id in seen_ids:
            raise ValueError(f"Duplicate scenario_id: {scenario_id}")
        seen_ids.add(scenario_id)

        situation = row["situation"].strip()
        genre_raw = row["genre"].strip()
        if not situation or not genre_raw:
            raise ValueError(f"Empty scenario field in: {scenario_id}")

        actions = []
        for index, (column, label, alignment) in enumerate(action_specs, start=1):
            description = row[column].strip()
            if not description:
                raise ValueError(f"Empty {column} in scenario: {scenario_id}")
            action = {
                "id": f"{scenario_id}_act_{index:03d}",
                "label": label,
                "description": description,
                "persona_alignment": alignment,
            }
            actions.append(action)
            actions_flat.append(action)

        scenarios.append(
            {
                "id": scenario_id,
                "genre": GENRE_MAP.get(genre_raw, genre_raw),
                "source_genre": genre_raw,
                "context": situation,
                "actions": actions,
            }
        )

    return scenarios, actions_flat


def build_neutral_anchor_pairs(scenarios: list[dict], target_persona: str) -> list[dict]:
    """Build target-vs-neutral contrastive pairs from scenario action choices."""
    if target_persona not in {"aggressive", "cooperative"}:
        raise ValueError(f"Unsupported target persona: {target_persona}")

    rows = []
    for scenario in scenarios:
        neutral_actions = [
            action for action in scenario.get("actions", [])
            if action.get("persona_alignment") == "neutral"
        ]
        target_actions = [
            action for action in scenario.get("actions", [])
            if action.get("persona_alignment") == target_persona
        ]
        if not neutral_actions:
            raise ValueError(f"No neutral action in scenario: {scenario.get('id')}")
        neutral = neutral_actions[0]
        for index, action in enumerate(target_actions, start=1):
            pair_id = f"{scenario['id']}_{target_persona}_neutral_{index:02d}"
            article = "an" if target_persona == "aggressive" else "a"
            rows.append({
                "id": pair_id,
                "positive": CONTRASTIVE_PROMPT_TEMPLATE.format(
                    article=article,
                    persona=target_persona,
                    situation=scenario["context"],
                    response=action["description"],
                ),
                "negative": CONTRASTIVE_PROMPT_TEMPLATE.format(
                    article="a",
                    persona="neutral",
                    situation=scenario["context"],
                    response=neutral["description"],
                ),
                "situation": scenario["context"],
                "positive_response": action["description"],
                "negative_response": neutral["description"],
                "positive_action_id": action["id"],
                "negative_action_id": neutral["id"],
                "source_template": scenario["id"],
                "strategy": f"team_a_{target_persona}_vs_neutral_action",
            })
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Team A CSV files to project JSONL inputs."
    )
    parser.add_argument("--contrastive_csv", type=Path, default=DEFAULT_CONTRASTIVE_CSV)
    parser.add_argument("--scenarios_csv", type=Path, default=DEFAULT_SCENARIOS_CSV)
    parser.add_argument(
        "--contrastive_source",
        choices=["actions", "csv"],
        default="actions",
        help=(
            "Use all scenario action combinations (actions) or the legacy "
            "contrastive CSV rows (csv)."
        ),
    )
    parser.add_argument("--contrastive_output", type=Path, default=DEFAULT_CONTRASTIVE_OUTPUT)
    parser.add_argument(
        "--aggressive_neutral_output",
        type=Path,
        default=DEFAULT_AGGRESSIVE_NEUTRAL_OUTPUT,
    )
    parser.add_argument(
        "--cooperative_neutral_output",
        type=Path,
        default=DEFAULT_COOPERATIVE_NEUTRAL_OUTPUT,
    )
    parser.add_argument("--scenarios_output", type=Path, default=DEFAULT_SCENARIOS_OUTPUT)
    parser.add_argument("--actions_output", type=Path, default=DEFAULT_ACTIONS_OUTPUT)
    parser.add_argument(
        "--update-defaults",
        action="store_true",
        help="Also overwrite the pipeline default JSONL files used by scripts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scenario_rows, action_rows = build_scenario_rows(load_csv(args.scenarios_csv))
    if args.contrastive_source == "csv":
        contrastive_rows = build_contrastive_rows(load_csv(args.contrastive_csv))
    else:
        contrastive_rows = build_action_contrastive_pairs(scenario_rows)
    aggressive_neutral_rows = build_neutral_anchor_pairs(scenario_rows, "aggressive")
    cooperative_neutral_rows = build_neutral_anchor_pairs(scenario_rows, "cooperative")

    contrastive_count = write_jsonl(contrastive_rows, args.contrastive_output)
    aggressive_neutral_count = write_jsonl(
        aggressive_neutral_rows,
        args.aggressive_neutral_output,
    )
    cooperative_neutral_count = write_jsonl(
        cooperative_neutral_rows,
        args.cooperative_neutral_output,
    )
    scenario_count = write_jsonl(scenario_rows, args.scenarios_output)
    action_count = write_jsonl(action_rows, args.actions_output)

    print(f"Wrote {contrastive_count} contrastive pairs: {args.contrastive_output}")
    print(
        f"Wrote {aggressive_neutral_count} aggressive-neutral pairs: "
        f"{args.aggressive_neutral_output}"
    )
    print(
        f"Wrote {cooperative_neutral_count} cooperative-neutral pairs: "
        f"{args.cooperative_neutral_output}"
    )
    print(f"Wrote {scenario_count} scenarios: {args.scenarios_output}")
    print(f"Wrote {action_count} action definitions: {args.actions_output}")

    if args.update_defaults:
        write_jsonl(contrastive_rows, PIPELINE_CONTRASTIVE_OUTPUT)
        write_jsonl(aggressive_neutral_rows, PIPELINE_AGGRESSIVE_NEUTRAL_OUTPUT)
        write_jsonl(cooperative_neutral_rows, PIPELINE_COOPERATIVE_NEUTRAL_OUTPUT)
        write_jsonl(scenario_rows, PIPELINE_SCENARIOS_OUTPUT)
        write_jsonl(action_rows, PIPELINE_ACTIONS_OUTPUT)
        print("Updated pipeline default JSONL files.")


if __name__ == "__main__":
    main()
