"""Runtime scoring helpers for v4 Big Five experiments."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch

from src.controls.vector_controls import (
    resolve_condition_vectors,
    should_normalize_injector_vectors,
)
from src.data.bfi import BFIItem, score_bfi_responses
from src.data.trait_dataset import TraitAction, TraitScenario
from src.evaluation.bigfive_metrics import (
    pas_loglik_agreement,
    speech_action_consistency,
    target_attainment,
    target_direction_score,
    trait_high_probability_mass,
)
from src.evaluation.loglik_selector import select_action_by_loglik
from src.generation.generator import TextGenerator
from src.generation.prompt_builder import PromptBuilder
from src.projection.embedder import ActionEmbedder
from src.projection.selector import PersonaActionSelector
from src.steering.injector import SteeringInjector
from src.steering.vector import SteeringVectorComputer


LIKERT_PATTERN = re.compile(r"\b([1-5])\b")


@dataclass(frozen=True)
class RuntimeCondition:
    """Resolved condition settings for a model-scoring run."""

    name: str
    prompt_style: str
    steering: bool
    pas: bool
    vector_mode: str | None = None
    selection_policy: str = "loglik"


def build_bfi_messages(
    item: BFIItem,
    trait: str,
    target_direction: str,
    prompt_style: str,
    prompt_builder: PromptBuilder | None = None,
) -> list[dict]:
    """Build messages for one BFI self-report item."""

    if not item.text.strip():
        raise ValueError(
            f"BFI item {item.item_id} has no text. Add item wording before BFI scoring."
        )
    prompt_builder = prompt_builder or PromptBuilder()
    persona_text = prompt_builder._get_bigfive_description(
        trait=trait,
        target_direction=target_direction,
        prompt_style=prompt_style,
    )
    if persona_text:
        system = (
            "You are answering a Big Five self-report item as the described NPC.\n"
            f"{persona_text}\n"
            "Return only one integer from 1 to 5."
        )
    else:
        system = (
            "You are answering a Big Five self-report item as an NPC.\n"
            "Return only one integer from 1 to 5."
        )
    user = (
        "Rate how well this statement describes you.\n"
        "1 = strongly disagree, 2 = disagree, 3 = neutral, "
        "4 = agree, 5 = strongly agree.\n"
        f"Statement: {item.text}\n"
        "Answer:"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_likert_response(text: str) -> int:
    """Parse a 1-5 Likert score from model text."""

    match = LIKERT_PATTERN.search(text.strip())
    if not match:
        raise ValueError(f"Could not parse Likert score from: {text!r}")
    return int(match.group(1))


def score_bfi_with_generator(
    generator: TextGenerator,
    items: Iterable[BFIItem],
    trait: str,
    target_direction: str,
    prompt_style: str,
    prompt_builder: PromptBuilder | None = None,
    persona_vectors: dict[int, torch.Tensor] | None = None,
    layer_indices: list[int] | None = None,
    alpha: float | None = None,
    normalize_vectors: bool = True,
    use_steering: bool = False,
) -> tuple[dict[str, float], list[dict]]:
    """Generate BFI responses and return trait scores plus raw item rows."""

    prompt_builder = prompt_builder or PromptBuilder()
    raw_rows = []
    responses = {}
    injector = None
    if use_steering:
        if persona_vectors is None or not layer_indices or alpha is None:
            raise ValueError(
                "BFI steering requires persona_vectors, layer_indices, and alpha"
            )
        injector = SteeringInjector(
            generator.model,
            persona_vectors,
            normalize_vectors=normalize_vectors,
        )
        injector.inject(layer_indices, alpha=alpha)

    try:
        for item in items:
            messages = build_bfi_messages(
                item,
                trait=trait,
                target_direction=target_direction,
                prompt_style=prompt_style,
                prompt_builder=prompt_builder,
            )
            output = generator.generate(messages, max_new_tokens=8, do_sample=False)
            score = parse_likert_response(output)
            responses[item.item_id] = score
            raw_rows.append(
                {
                    "item_id": item.item_id,
                    "item_trait": item.trait,
                    "reverse_scored": item.reverse_scored,
                    "raw_output": output,
                    "likert_score": score,
                    "steering_applied": bool(use_steering),
                }
            )
    finally:
        if injector is not None:
            injector.clear()
    return score_bfi_responses(items, responses), raw_rows


def action_texts_and_directions(
    scenario: TraitScenario,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Return action ids, action texts, and action direction map."""

    action_ids = [action.id for action in scenario.actions]
    action_texts = [action.text for action in scenario.actions]
    directions = {action.id: action.trait_direction for action in scenario.actions}
    return action_ids, action_texts, directions


def score_trait_scenario_by_loglik(
    model,
    tokenizer,
    scenario: TraitScenario,
    condition: RuntimeCondition,
    target_direction: str,
    prompt_builder: PromptBuilder | None = None,
    persona_vectors: dict[int, torch.Tensor] | None = None,
    layer_indices: list[int] | None = None,
    alpha: float | None = None,
    normalize_vectors: bool = True,
) -> dict:
    """Score one TRAIT scenario with conditional log-likelihood."""

    prompt_builder = prompt_builder or PromptBuilder()
    action_ids, action_texts, directions = action_texts_and_directions(scenario)
    scenario_dict = trait_scenario_to_prompt_dict(scenario)
    messages = prompt_builder.build_action_choice_messages(
        scenario_dict,
        prompt_style=condition.prompt_style,
        trait=scenario.trait,
        target_direction=target_direction,
    )

    injector = None
    if condition.steering:
        if persona_vectors is None or not layer_indices or alpha is None:
            raise ValueError(
                f"Condition {condition.name} requires persona_vectors, layer_indices, and alpha"
            )
        injector = SteeringInjector(
            model,
            persona_vectors,
            normalize_vectors=normalize_vectors,
        )
        injector.inject(layer_indices, alpha=alpha)

    try:
        selection = select_action_by_loglik(
            model,
            tokenizer,
            messages,
            action_texts,
            length_normalize=True,
        )
    finally:
        if injector is not None:
            injector.clear()

    selected_idx = selection["selected_idx"]
    selected_action_id = action_ids[selected_idx]
    probs = {
        action_id: float(prob)
        for action_id, prob in zip(action_ids, selection["softmax_probs"])
    }
    trait_score = trait_high_probability_mass(probs, directions)
    return {
        "scenario_id": scenario.scenario_id,
        "trait": scenario.trait,
        "target_direction": target_direction,
        "condition": condition.name,
        "prompt_style": condition.prompt_style,
        "selection_policy": condition.selection_policy,
        "loglik_selected_action": selected_action_id,
        "loglik_selected_direction": directions[selected_action_id],
        "final_selected_action": selected_action_id,
        "final_selected_direction": directions[selected_action_id],
        "action_ids": action_ids,
        "action_texts": action_texts,
        "action_directions": [directions[action_id] for action_id in action_ids],
        "action_logliks": selection["logliks"],
        "action_normalized_logliks": selection["normalized_logliks"],
        "action_softmax_probs": selection["softmax_probs"],
        "trait_score": trait_score,
        "trait_target_score": target_direction_score(trait_score, target_direction),
        "trait_target_attainment": target_attainment(trait_score, target_direction),
    }


def apply_pas_final_selection(
    loglik_row: Mapping[str, object],
    pas_row: Mapping[str, object],
) -> dict:
    """Return a TRAIT row whose final action and score come from PAS."""

    selected_action = str(pas_row["pas_selected_action"])
    selected_direction = str(pas_row["pas_selected_direction"])
    target_direction = str(loglik_row["target_direction"])
    trait_score = 1.0 if selected_direction == "high" else 0.0
    row = dict(loglik_row)
    row.update(
        {
            "selection_policy": "pas_final",
            "final_selected_action": selected_action,
            "final_selected_direction": selected_direction,
            "pas_selected_action": selected_action,
            "pas_selected_direction": selected_direction,
            "trait_score": trait_score,
            "trait_target_score": target_direction_score(
                trait_score,
                target_direction,
            ),
            "trait_target_attainment": target_attainment(
                trait_score,
                target_direction,
            ),
        }
    )
    return row


def score_trait_scenario_by_pas(
    model,
    tokenizer,
    scenario: TraitScenario,
    persona_vector: torch.Tensor,
    layer_idx: int,
) -> dict:
    """Score one TRAIT scenario by PAS cosine projection."""

    embedder = ActionEmbedder(model, tokenizer, layer_idx=layer_idx)
    action_defs = [
        {"id": action.id, "description": action.text}
        for action in scenario.actions
    ]
    embeddings = embedder.embed_actions(action_defs)
    selector = PersonaActionSelector(persona_vector, embeddings)
    scores = selector.compute_scores()
    ranking = selector.get_ranking()
    selected_action = ranking[0][0]
    directions = {action.id: action.trait_direction for action in scenario.actions}
    return {
        "pas_selected_action": selected_action,
        "pas_selected_direction": directions[selected_action],
        "pas_cosine_scores": scores,
        "pas_ranking": ranking,
    }


def generate_trait_scenario_response(
    generator: TextGenerator,
    model,
    scenario: TraitScenario,
    condition: RuntimeCondition,
    target_direction: str,
    prompt_builder: PromptBuilder | None = None,
    persona_vectors: dict[int, torch.Tensor] | None = None,
    layer_indices: list[int] | None = None,
    alpha: float | None = None,
    normalize_vectors: bool = True,
    max_new_tokens: int = 96,
) -> dict:
    """Generate one free-form NPC response for side-effect checks."""

    prompt_builder = prompt_builder or PromptBuilder()
    scenario_dict = trait_scenario_to_prompt_dict(scenario)
    messages = prompt_builder.build_chat_messages(
        scenario_dict,
        prompt_style=condition.prompt_style,
        trait=scenario.trait,
        target_direction=target_direction,
    )
    action_ids, _, directions = action_texts_and_directions(scenario)

    injector = None
    if condition.steering:
        if persona_vectors is None or not layer_indices or alpha is None:
            raise ValueError(
                f"Condition {condition.name} requires persona_vectors, layer_indices, and alpha"
            )
        injector = SteeringInjector(
            model,
            persona_vectors,
            normalize_vectors=normalize_vectors,
        )
        injector.inject(layer_indices, alpha=alpha)

    try:
        raw = generator.generate(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    finally:
        if injector is not None:
            injector.clear()

    return {
        "scenario_id": scenario.scenario_id,
        "trait": scenario.trait,
        "target_direction": target_direction,
        "condition": condition.name,
        "prompt_style": condition.prompt_style,
        "raw": raw,
        "valid_actions": "|".join(action_ids),
        "action_directions_json": json.dumps({
            action_id: directions[action_id] for action_id in action_ids
        }),
    }


def resolve_trait_persona_vectors(
    vector_bundle: Mapping[str, object],
    trait: str,
    target_direction: str,
) -> dict[int, torch.Tensor]:
    """Resolve signed trait vectors from a v4 trait vector bundle."""

    vectors = vector_bundle.get("vectors", vector_bundle)
    if "trait_contrast" not in vectors:
        raise ValueError("Vector bundle missing vectors['trait_contrast']")
    trait_vectors = vectors["trait_contrast"]
    if trait not in trait_vectors:
        raise ValueError(f"Vector bundle missing trait vector: {trait}")
    sign = 1.0 if target_direction == "high" else -1.0
    return {
        int(layer): sign * vector
        for layer, vector in trait_vectors[trait].items()
    }


def resolve_condition_persona_vectors(
    vector_bundle: Mapping[str, object],
    trait: str,
    target_direction: str,
    vector_mode: str | None,
    layer_indices: list[int] | None = None,
    seed: int = 42,
) -> dict[int, torch.Tensor]:
    """Resolve vectors for primary and diagnostic v4 vector modes."""

    return resolve_condition_vectors(
        vector_bundle,
        trait=trait,
        target_direction=target_direction,
        vector_mode=vector_mode,
        layer_indices=layer_indices,
        seed=seed,
    )


def load_vector_bundle(path: str | Path) -> dict:
    """Load a steering vector bundle."""

    return SteeringVectorComputer().load_bundle(path)


def combine_consistency_row(
    bfi_scores: Mapping[str, float],
    trait_row: Mapping[str, object],
) -> dict:
    """Combine BFI and TRAIT scoring outputs into one consistency row."""

    trait = str(trait_row["trait"])
    if trait not in bfi_scores:
        raise ValueError(f"BFI scores missing trait: {trait}")
    bfi_score = float(bfi_scores[trait])
    trait_score = float(trait_row["trait_score"])
    target_direction = str(trait_row["target_direction"])
    return {
        "condition": trait_row["condition"],
        "scenario_id": trait_row["scenario_id"],
        "trait": trait,
        "target_direction": target_direction,
        "bfi_score": bfi_score,
        "trait_score": trait_score,
        "consistency": speech_action_consistency(bfi_score, trait_score),
        "bfi_target_score": target_direction_score(bfi_score, target_direction),
        "bfi_target_attainment": target_attainment(bfi_score, target_direction),
        "trait_target_score": trait_row["trait_target_score"],
        "trait_target_attainment": trait_row["trait_target_attainment"],
    }


def trait_scenario_to_prompt_dict(scenario: TraitScenario) -> dict:
    """Convert normalized TRAIT scenario dataclass to PromptBuilder input."""

    return {
        "id": scenario.scenario_id,
        "genre": "general",
        "context": scenario.prompt,
        "actions": [
            {
                "id": action.id,
                "description": action.text,
                "trait_direction": action.trait_direction,
            }
            for action in scenario.actions
        ],
    }


def write_csv_rows(rows: Iterable[Mapping[str, object]], path: str | Path) -> int:
    """Write dict rows to CSV, preserving all fieldnames discovered."""

    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return 0
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return len(rows)


def selected_layer_for_pas(layer_indices: list[int]) -> int:
    """Pick a stable PAS embedding layer from steering layers."""

    if not layer_indices:
        raise ValueError("At least one layer is required for PAS")
    return sorted(layer_indices)[len(layer_indices) // 2]
