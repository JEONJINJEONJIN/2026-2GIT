"""Data helpers for the v4 Big Five Streamlit demo."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

BIG_FIVE_TRAITS = (
    "agreeableness",
    "conscientiousness",
    "neuroticism",
    "openness",
    "extraversion",
)

DEMO_PERSONA_PRESETS = (
    {
        "id": "warm_diplomat",
        "label": "Warm Diplomat",
        "values": {
            "agreeableness": 0.90,
            "conscientiousness": 0.65,
            "neuroticism": 0.25,
            "openness": 0.60,
            "extraversion": 0.70,
        },
    },
    {
        "id": "cold_operator",
        "label": "Cold Operator",
        "values": {
            "agreeableness": 0.10,
            "conscientiousness": 0.80,
            "neuroticism": 0.25,
            "openness": 0.35,
            "extraversion": 0.55,
        },
    },
    {
        "id": "reckless_visionary",
        "label": "Reckless Visionary",
        "values": {
            "agreeableness": 0.45,
            "conscientiousness": 0.15,
            "neuroticism": 0.65,
            "openness": 0.95,
            "extraversion": 0.85,
        },
    },
    {
        "id": "anxious_perfectionist",
        "label": "Anxious Perfectionist",
        "values": {
            "agreeableness": 0.55,
            "conscientiousness": 0.95,
            "neuroticism": 0.90,
            "openness": 0.35,
            "extraversion": 0.25,
        },
    },
    {
        "id": "guarded_rebel",
        "label": "Guarded Rebel",
        "values": {
            "agreeableness": 0.20,
            "conscientiousness": 0.25,
            "neuroticism": 0.70,
            "openness": 0.80,
            "extraversion": 0.20,
        },
    },
)

CONFLICT_SCENARIO_PRESET_IDS = (
    "trait_agreeableness_00853",
    "trait_neuroticism_00879",
    "trait_neuroticism_00917",
    "trait_neuroticism_00919",
    "trait_neuroticism_00974",
    "trait_neuroticism_00982",
    "trait_conscientiousness_01000",
    "trait_conscientiousness_00915",
    "trait_extraversion_00952",
    "trait_openness_00999",
    "trait_openness_00890",
    "trait_agreeableness_00925",
    "trait_agreeableness_00926",
)


@dataclass(frozen=True)
class DemoScenario:
    """One scenario card shown in the class demo."""

    scenario_id: str
    trait: str
    target_direction: str
    prompt: str
    actions: tuple[dict, ...]


@dataclass(frozen=True)
class DemoResponse:
    """One generated response used by the side-by-side demo."""

    scenario_id: str
    condition: str
    trait: str
    target_direction: str
    speech: str
    action: str
    raw: str


def load_demo_scenarios(path: str | Path | None) -> list[DemoScenario]:
    """Load normalized TRAIT scenarios, falling back to built-in examples."""

    if path is None or not Path(path).exists():
        return sample_scenarios()
    scenarios = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            scenarios.append(
                DemoScenario(
                    scenario_id=str(row["scenario_id"]),
                    trait=str(row["trait"]),
                    target_direction="high",
                    prompt=str(row["prompt"]),
                    actions=tuple(dict(action) for action in row.get("actions", [])),
                )
            )
    return scenarios or sample_scenarios()


def load_demo_responses(path: str | Path | None) -> list[DemoResponse]:
    """Load generated response rows, falling back to deterministic examples."""

    if path is None or not Path(path).exists():
        return sample_responses()
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    responses = []
    for row in rows:
        raw = row.get("raw", "")
        responses.append(
            DemoResponse(
                scenario_id=str(row.get("scenario_id", "")),
                condition=str(row.get("condition", "")),
                trait=str(row.get("trait", "")),
                target_direction=str(row.get("target_direction", "")),
                speech=extract_speech_from_raw(raw),
                action=extract_action_from_raw(raw),
                raw=raw,
            )
        )
    return responses or sample_responses()


def choose_demo_scenario(
    scenarios: Iterable[DemoScenario],
    trait_scores: Mapping[str, float],
) -> DemoScenario:
    """Choose a scenario whose trait is most emphasized by current sliders."""

    scenarios = list(scenarios)
    if not scenarios:
        raise ValueError("At least one scenario is required")
    available_traits = {scenario.trait for scenario in scenarios}
    target_trait = max(
        [trait for trait in BIG_FIVE_TRAITS if trait in available_traits],
        key=lambda trait: abs(float(trait_scores.get(trait, 0.5)) - 0.5),
    )
    for scenario in scenarios:
        if scenario.trait == target_trait:
            return scenario
    return scenarios[0]


def prioritize_conflict_scenarios(
    scenarios: Iterable[DemoScenario],
) -> list[DemoScenario]:
    """Return scenarios with conflict presets first, preserving the rest."""

    scenarios = list(scenarios)
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    prioritized = [
        by_id[scenario_id]
        for scenario_id in CONFLICT_SCENARIO_PRESET_IDS
        if scenario_id in by_id
    ]
    seen = {scenario.scenario_id for scenario in prioritized}
    prioritized.extend(
        scenario for scenario in scenarios
        if scenario.scenario_id not in seen
    )
    return prioritized


def response_pair_for_scenario(
    responses: Iterable[DemoResponse],
    scenario: DemoScenario,
    *,
    prompt_condition: str = "elaborate_prompt",
    as_pas_condition: str = "elaborate_prompt_as_pas",
) -> tuple[DemoResponse, DemoResponse]:
    """Return prompt-only and AS+PAS responses for a scenario."""

    responses = list(responses)
    prompt_response = _find_response(responses, scenario, prompt_condition)
    as_pas_response = _find_response(responses, scenario, as_pas_condition)
    return prompt_response, as_pas_response


def activation_points(trait_scores: Mapping[str, float]) -> list[dict]:
    """Return simple 2D placeholder points for activation movement."""

    x = float(trait_scores.get("openness", 0.5)) - float(
        trait_scores.get("conscientiousness", 0.5)
    )
    y = float(trait_scores.get("extraversion", 0.5)) - float(
        trait_scores.get("neuroticism", 0.5)
    )
    agreeableness_shift = float(trait_scores.get("agreeableness", 0.5)) - 0.5
    return [
        {"label": "baseline", "x": 0.0, "y": 0.0},
        {"label": "prompt", "x": x * 0.5, "y": y * 0.5},
        {"label": "AS+PAS", "x": x + agreeableness_shift * 0.3, "y": y},
    ]


def sample_scenarios() -> list[DemoScenario]:
    """Return built-in demo scenarios for no-data mode."""

    return [
        DemoScenario(
            scenario_id="demo_agreeableness_001",
            trait="agreeableness",
            target_direction="high",
            prompt="A teammate asks the NPC for help after making a costly mistake.",
            actions=(
                {"id": "a1", "text": "Help them recover and reassure them.", "trait_direction": "high"},
                {"id": "a2", "text": "Refuse and focus on personal goals.", "trait_direction": "low"},
            ),
        ),
        DemoScenario(
            scenario_id="demo_openness_001",
            trait="openness",
            target_direction="high",
            prompt="The party discovers an unfamiliar device in a quiet ruin.",
            actions=(
                {"id": "a1", "text": "Experiment carefully with the device.", "trait_direction": "high"},
                {"id": "a2", "text": "Ignore it and follow the known route.", "trait_direction": "low"},
            ),
        ),
    ]


def sample_responses() -> list[DemoResponse]:
    """Return built-in paired responses for no-data mode."""

    return [
        DemoResponse(
            "demo_agreeableness_001",
            "elaborate_prompt",
            "agreeableness",
            "high",
            "We can fix this if we stay calm.",
            "a1",
            "[Speech] We can fix this if we stay calm.\n<Action>a1</Action>",
        ),
        DemoResponse(
            "demo_agreeableness_001",
            "elaborate_prompt_as_pas",
            "agreeableness",
            "high",
            "I will help you repair the damage and keep the team together.",
            "a1",
            "[Speech] I will help you repair the damage and keep the team together.\n<Action>a1</Action>",
        ),
        DemoResponse(
            "demo_openness_001",
            "elaborate_prompt",
            "openness",
            "high",
            "This might teach us something useful.",
            "a1",
            "[Speech] This might teach us something useful.\n<Action>a1</Action>",
        ),
        DemoResponse(
            "demo_openness_001",
            "elaborate_prompt_as_pas",
            "openness",
            "high",
            "Let us test it carefully and learn what it can reveal.",
            "a1",
            "[Speech] Let us test it carefully and learn what it can reveal.\n<Action>a1</Action>",
        ),
    ]


def extract_speech_from_raw(raw: str) -> str:
    """Extract speech text from a generated response."""

    if "[Speech]" not in raw:
        return raw.split("<Action>")[0].strip()
    speech = raw.split("[Speech]", 1)[1].split("<Action>", 1)[0]
    return speech.strip()


def extract_action_from_raw(raw: str) -> str:
    """Extract action id from a generated response."""

    if "<Action>" not in raw or "</Action>" not in raw:
        return ""
    return raw.split("<Action>", 1)[1].split("</Action>", 1)[0].strip()


def _find_response(
    responses: list[DemoResponse],
    scenario: DemoScenario,
    condition: str,
) -> DemoResponse:
    for response in responses:
        if (
            response.scenario_id == scenario.scenario_id
            and response.condition == condition
            and response.target_direction == scenario.target_direction
        ):
            return response
    for response in responses:
        if response.scenario_id == scenario.scenario_id and response.condition == condition:
            return response
    target_action = _preferred_action_for_condition(scenario, condition)
    action_text = target_action.get("text") or target_action.get("description") or target_action.get("id", "")
    speech = (
        "I will choose the action that best matches this persona setting."
        if action_text
        else "No precomputed response is available."
    )
    return DemoResponse(
        scenario.scenario_id,
        condition,
        scenario.trait,
        scenario.target_direction,
        speech,
        str(target_action.get("id", "")),
        f"[Speech] {speech}\n<Action>{target_action.get('id', '')}</Action>",
    )


def _preferred_action_for_condition(
    scenario: DemoScenario,
    condition: str,
) -> dict:
    """Return a deterministic fallback action for scenario-only demo mode."""

    actions = list(scenario.actions)
    if not actions:
        return {"id": "", "text": ""}
    target_direction = scenario.target_direction
    if "as_pas" in condition:
        for action in actions:
            if str(action.get("trait_direction", "")).lower() == target_direction:
                return action
    return actions[0]
