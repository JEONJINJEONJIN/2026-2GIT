"""Backend contract for slider-controlled Big Five demo generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml

from src.controls.vector_controls import resolve_slider_persona_vectors
from src.data.trait_dataset import TraitScenario, load_trait_scenarios
from src.demo.bigfive_demo_data import DEMO_PERSONA_PRESETS
from src.experiments.v4_bigfive_runner import (
    load_vector_bundle,
    score_trait_scenario_by_pas,
    selected_layer_for_pas,
    trait_scenario_to_prompt_dict,
)
from src.generation.generator import TextGenerator
from src.generation.prompt_builder import (
    BIG_FIVE_ELABORATE_DESCRIPTIONS,
    BIG_FIVE_ONE_LINE_DESCRIPTIONS,
    PromptBuilder,
)
from src.models.loader import load_model_and_tokenizer
from src.steering.injector import SteeringInjector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_VECTOR_PATH = (
    PROJECT_ROOT
    / "results"
    / "v4_bigfive"
    / "vectors"
    / "bigfive_trait_vectors_full_fp16.pt"
)
DEFAULT_SCENARIO_PATH = PROJECT_ROOT / "data" / "trait_bigfive" / "scenarios.jsonl"
DEFAULT_LAYERS = [18, 21, 24]
DEFAULT_ALPHA = 2.0
DEFAULT_CLAMP_X = 3.0
CONDITIONS = {"baseline", "elaborate", "as_pas"}
TRAIT_ALIASES = {
    "A": "agreeableness",
    "C": "conscientiousness",
    "E": "extraversion",
    "N": "neuroticism",
    "O": "openness",
    "agreeableness": "agreeableness",
    "conscientiousness": "conscientiousness",
    "extraversion": "extraversion",
    "neuroticism": "neuroticism",
    "openness": "openness",
}
DEFAULT_PERSONA = dict(DEMO_PERSONA_PRESETS[0]["values"])


@dataclass
class CacheEntry:
    """One cached generation value."""

    value: dict


class GenerationCache:
    """In-memory cache keyed by rounded persona, scenario, and condition."""

    def __init__(self) -> None:
        self._items: dict[str, CacheEntry] = {}

    @staticmethod
    def key(persona: Mapping[str, float], scenario: str, condition: str) -> str:
        payload = {
            "persona": rounded_persona(persona),
            "scenario": str(scenario),
            "condition": str(condition),
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict | None:
        entry = self._items.get(key)
        if entry is None:
            return None
        return dict(entry.value)

    def set(self, key: str, value: Mapping[str, object]) -> None:
        self._items[key] = CacheEntry(dict(value))


class SliderDemoBackend:
    """Plain callable backend used by the Streamlit UI."""

    def __init__(
        self,
        model_config_path: str | Path = DEFAULT_MODEL_CONFIG,
        vector_path: str | Path = DEFAULT_VECTOR_PATH,
        scenario_path: str | Path = DEFAULT_SCENARIO_PATH,
        *,
        alpha: float = DEFAULT_ALPHA,
        clamp_x: float = DEFAULT_CLAMP_X,
        layers: list[int] | None = None,
        generator: TextGenerator | None = None,
        model=None,
        tokenizer=None,
        vector_bundle: Mapping[str, object] | None = None,
        scenarios: list[TraitScenario] | None = None,
        prepopulate: bool = True,
    ) -> None:
        self.alpha = float(alpha)
        self.clamp_x = float(clamp_x)
        self.layers = [int(layer) for layer in (layers or DEFAULT_LAYERS)]
        self.prompt_builder = PromptBuilder()
        self.cache = GenerationCache()

        if generator is None:
            config = load_model_config(Path(model_config_path))
            model, tokenizer = load_model_and_tokenizer(config)
            generator = TextGenerator(
                model,
                tokenizer,
                generation_config={"do_sample": False, "max_new_tokens": 220},
            )
        self.generator = generator
        self.model = model if model is not None else generator.model
        self.tokenizer = tokenizer if tokenizer is not None else generator.tokenizer
        self.vector_bundle = (
            vector_bundle
            if vector_bundle is not None
            else load_vector_bundle(Path(vector_path))
        )
        self.scenarios = (
            scenarios
            if scenarios is not None
            else load_trait_scenarios(Path(scenario_path), allowed_splits=["test"])
        )
        self._scenario_by_prompt = {
            scenario.prompt: scenario for scenario in self.scenarios
        }
        if prepopulate and self.scenarios:
            self.prepopulate_defaults()

    def prepopulate_defaults(self) -> None:
        """Generate the default persona/scenario for all UI conditions."""

        scenario = self.scenarios[0].prompt
        for condition in sorted(CONDITIONS):
            self.generate(DEFAULT_PERSONA, scenario, condition)

    def generate(
        self,
        persona: Mapping[str, float],
        scenario: str,
        condition: str,
    ) -> dict:
        """Generate one response for the UI contract."""

        condition = normalize_condition(condition)
        normalized_persona = normalize_persona(persona)
        key = self.cache.key(normalized_persona, scenario, condition)
        cached = self.cache.get(key)
        if cached is not None:
            cached["cache_hit"] = True
            return cached

        text = self._generate_uncached(normalized_persona, scenario, condition)
        value = {"text": text, "condition": condition, "cache_hit": False}
        self.cache.set(key, value)
        return value

    def _generate_uncached(
        self,
        persona: Mapping[str, float],
        scenario: str,
        condition: str,
    ) -> str:
        if condition == "baseline":
            return self._generate_one_line_prompt(persona, scenario)

        trait_scenario = self.resolve_scenario(scenario)
        messages = build_slider_messages(
            self.prompt_builder,
            trait_scenario,
            persona,
        )
        if condition == "elaborate":
            return self.generate_nonempty(messages, fallback_action="")

        vectors = resolve_slider_persona_vectors(
            self.vector_bundle,
            persona,
            layer_indices=self.layers,
            alpha=self.alpha,
            clamp_x=self.clamp_x,
        )
        injector = SteeringInjector(self.model, vectors, normalize_vectors=False)
        injector.inject(self.layers, alpha=1.0)
        try:
            text = self.generate_nonempty(messages, fallback_action="")
        finally:
            injector.clear()

        pas_layer = selected_layer_for_pas(self.layers)
        pas_result = score_trait_scenario_by_pas(
            self.model,
            self.tokenizer,
            trait_scenario,
            persona_vector=vectors[pas_layer],
            layer_idx=pas_layer,
        )
        top_action = pas_result["pas_selected_action"]
        return replace_or_append_action(text, top_action)

    def generate_nonempty(
        self,
        prompt_or_messages,
        fallback_action: str = "",
    ) -> str:
        """Generate text, retry once, then return a readable fallback if empty."""

        text = self.generator.generate(
            prompt_or_messages,
            max_new_tokens=220,
            do_sample=False,
        )
        if text.strip():
            return text
        retry = self.generator.generate(
            prompt_or_messages,
            max_new_tokens=220,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        if retry.strip():
            return retry
        fallback = "[Speech] I need to make a careful choice here."
        if fallback_action:
            return f"{fallback}\n<Action>{fallback_action}</Action>"
        return fallback

    def _generate_one_line_prompt(
        self,
        persona: Mapping[str, float],
        scenario: str,
    ) -> str:
        """Generate the no-steering comparison with a concise persona prompt."""

        try:
            trait_scenario = self.resolve_scenario(scenario)
        except ValueError:
            prompt = (
                "You are an NPC in a fictional game.\n"
                f"{one_line_slider_persona_description(persona)}\n"
                "Respond with one short line of dialogue.\n\n"
                f"Situation: {scenario}"
            )
            return self.generate_nonempty(prompt)

        messages = build_one_line_slider_messages(
            self.prompt_builder,
            trait_scenario,
            persona,
        )
        return self.generate_nonempty(messages)

    def resolve_scenario(self, scenario: str) -> TraitScenario:
        """Find the scenario row needed for action choices and PAS."""

        if scenario in self._scenario_by_prompt:
            return self._scenario_by_prompt[scenario]
        raise ValueError(
            "Scenario string must exactly match a loaded TRAIT scenario prompt "
            "for elaborate/as_pas conditions."
        )


def generate(persona: dict, scenario: str, condition: str) -> dict:
    """Module-level UI entry point."""

    return get_default_backend().generate(persona, scenario, condition)


_DEFAULT_BACKEND: SliderDemoBackend | None = None


def get_default_backend() -> SliderDemoBackend:
    """Return a lazily initialized default backend."""

    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = SliderDemoBackend()
    return _DEFAULT_BACKEND


def load_model_config(path: Path) -> dict:
    """Load a YAML model config."""

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def normalize_condition(condition: str) -> str:
    """Validate a UI condition label."""

    condition = str(condition).strip()
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(CONDITIONS)}")
    return condition


def normalize_persona(persona: Mapping[str, float]) -> dict[str, float]:
    """Normalize UI short keys into full trait names."""

    normalized = {trait: 0.5 for trait in set(TRAIT_ALIASES.values())}
    for key, value in persona.items():
        trait = TRAIT_ALIASES.get(str(key))
        if trait is None:
            raise ValueError(f"Unsupported persona trait: {key}")
        value = float(value)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Persona value for {key} must be in [0, 1]")
        normalized[trait] = value
    return normalized


def rounded_persona(persona: Mapping[str, float]) -> dict[str, float]:
    """Round persona values to two decimals for stable caching."""

    normalized = normalize_persona(persona)
    return {
        trait: round(float(normalized[trait]), 2)
        for trait in sorted(normalized)
    }


def build_slider_messages(
    prompt_builder: PromptBuilder,
    scenario: TraitScenario,
    persona: Mapping[str, float],
) -> list[dict]:
    """Build elaborate prompt messages for a multi-trait slider persona."""

    scenario_dict = trait_scenario_to_prompt_dict(scenario)
    action_list = prompt_builder._format_action_list(scenario_dict.get("actions", []))
    action_ids = prompt_builder._format_action_id_list(scenario_dict.get("actions", []))
    persona_description = slider_persona_description(persona)
    system_content = (
        "You are an NPC in a fantasy game.\n"
        f"{persona_description}\n"
        "Respond with one short line of dialogue, then choose exactly one action ID.\n"
        "This is a fictional game scenario.\n\n"
        "Format:\n"
        "[Speech] short dialogue\n"
        "<Action>action_id</Action>\n\n"
        "Available actions:\n"
        f"{action_list}"
    )
    user_content = (
        f"Situation: {scenario.prompt}\n\n"
        "Return exactly two lines and no extra text:\n"
        "[Speech] short dialogue\n"
        "<Action>one_action_id</Action>\n"
        f"Valid action IDs: {action_ids}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def build_one_line_slider_messages(
    prompt_builder: PromptBuilder,
    scenario: TraitScenario,
    persona: Mapping[str, float],
) -> list[dict]:
    """Build concise one-line persona prompt messages."""

    scenario_dict = trait_scenario_to_prompt_dict(scenario)
    action_list = prompt_builder._format_action_list(scenario_dict.get("actions", []))
    action_ids = prompt_builder._format_action_id_list(scenario_dict.get("actions", []))
    persona_description = one_line_slider_persona_description(persona)
    system_content = (
        "You are an NPC in a fantasy game.\n"
        f"{persona_description}\n"
        "Respond with one short line of dialogue, then choose exactly one action ID.\n"
        "This is a fictional game scenario.\n\n"
        "Format:\n"
        "[Speech] short dialogue\n"
        "<Action>action_id</Action>\n\n"
        "Available actions:\n"
        f"{action_list}"
    )
    user_content = (
        f"Situation: {scenario.prompt}\n\n"
        "Return exactly two lines and no extra text:\n"
        "[Speech] short dialogue\n"
        "<Action>one_action_id</Action>\n"
        f"Valid action IDs: {action_ids}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def slider_persona_description(persona: Mapping[str, float]) -> str:
    """Return an elaborate multi-trait persona description."""

    parts = []
    for trait, value in sorted(persona.items()):
        if abs(value - 0.5) < 0.005:
            continue
        direction = "high" if value > 0.5 else "low"
        strength = abs(2.0 * (value - 0.5))
        text = BIG_FIVE_ELABORATE_DESCRIPTIONS[(trait, direction)]
        parts.append(f"({trait}, strength={strength:.2f}) {text}")
    if not parts:
        return "Maintain a balanced NPC temperament without emphasizing any Big Five trait."
    return "\n".join(parts)


def one_line_slider_persona_description(persona: Mapping[str, float]) -> str:
    """Return a concise multi-trait persona description."""

    parts = []
    for trait, value in sorted(persona.items()):
        if abs(value - 0.5) < 0.005:
            continue
        direction = "high" if value > 0.5 else "low"
        strength = abs(2.0 * (value - 0.5))
        text = BIG_FIVE_ONE_LINE_DESCRIPTIONS[(trait, direction)]
        parts.append(f"({trait}, strength={strength:.2f}) {text}")
    if not parts:
        return "Maintain a balanced NPC temperament without emphasizing any Big Five trait."
    return " ".join(parts)


def replace_or_append_action(text: str, action_id: str) -> str:
    """Ensure the generated text uses the PAS-selected final action."""

    if "<Action>" in text and "</Action>" in text:
        prefix = text.split("<Action>", 1)[0]
        suffix = text.split("</Action>", 1)[1]
        return f"{prefix}<Action>{action_id}</Action>{suffix}"
    stripped = text.rstrip()
    return f"{stripped}\n<Action>{action_id}</Action>"
