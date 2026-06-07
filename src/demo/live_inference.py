"""Live model inference for the v4 Big Five Streamlit demo."""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml

from src.data.trait_dataset import TraitScenario, load_trait_scenarios
from src.experiments.v4_bigfive_runner import (
    load_vector_bundle,
    resolve_trait_persona_vectors,
    score_trait_scenario_by_pas,
    selected_layer_for_pas,
    trait_scenario_to_prompt_dict,
)
from src.generation.generator import TextGenerator
from src.generation.prompt_builder import PromptBuilder
from src.models.loader import get_model_layers, load_model_and_tokenizer
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


@dataclass(frozen=True)
class GenerationResult:
    """One live generation output and its lightweight diagnostics."""

    text: str
    mode: str
    elapsed_sec: float
    n_tokens: int
    parsed_action_id: str
    layers_hooked: list[int]
    pas_score: float | None
    pas_top_action_id: str | None
    pas_used_fallback: bool = False


class LiveInferenceEngine:
    """Load one model instance and run live prompt-only / AS generations."""

    def __init__(
        self,
        model_config_path: str | Path = DEFAULT_MODEL_CONFIG,
        vector_path: str | Path = DEFAULT_VECTOR_PATH,
        model_name: str | None = None,
    ) -> None:
        self.model_config_path = Path(model_config_path)
        self.vector_path = Path(vector_path)
        self.model_config = self._load_model_config(self.model_config_path, model_name)
        self.model_name = str(self.model_config.get("model", {}).get("name", model_name))
        self.model, self.tokenizer = load_model_and_tokenizer(self.model_config)
        self.generator = TextGenerator(
            self.model,
            self.tokenizer,
            generation_config={
                "do_sample": False,
                "max_new_tokens": 96,
            },
        )
        if not self.vector_path.exists():
            raise FileNotFoundError(f"Missing v4 vector bundle: {self.vector_path}")
        self.vector_bundle = load_vector_bundle(self.vector_path)
        self.prompt_builder = PromptBuilder()
        self._active_hook_contexts = 0

    @staticmethod
    def _load_model_config(path: Path, model_name: str | None) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing model config: {path}")
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if model_name:
            config.setdefault("model", {})["name"] = model_name
        return config

    @lru_cache(maxsize=None)
    def get_persona_vectors(
        self,
        trait: str,
        direction: str,
    ) -> dict[int, object]:
        """Return signed layer-indexed steering vectors for one persona."""

        return resolve_trait_persona_vectors(self.vector_bundle, trait, direction)

    def available_layers(self) -> list[int]:
        """Return valid layer indices for the loaded model."""

        return list(range(len(get_model_layers(self.model))))

    def generate(
        self,
        scenario: TraitScenario,
        trait: str,
        direction: str,
        mode: Literal["prompt_only", "as_pas"],
        alpha: float = 4.0,
        layers: list[int] | tuple[int, ...] = (18, 21, 24),
        max_new_tokens: int = 96,
    ) -> GenerationResult:
        """Generate a single live response under the requested mode."""

        layers = [int(layer) for layer in layers]
        if mode not in {"prompt_only", "as_pas"}:
            raise ValueError(f"Unsupported live inference mode: {mode}")
        if mode == "prompt_only" and self._active_hook_contexts:
            raise RuntimeError("Refusing prompt_only generation while AS hooks are active")

        messages = self._build_messages(scenario, trait, direction)
        start = time.perf_counter()
        if mode == "as_pas":
            vectors = self.get_persona_vectors(trait, direction)
            with self._steering_hooks(vectors, layers, alpha):
                text = self.generator.generate(
                    messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            hooked_layers = layers
        else:
            text = self.generator.generate(
                messages,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            hooked_layers = []
        elapsed = time.perf_counter() - start

        parsed_action_id = parse_action_id(text)
        pas_score = None
        pas_top_action_id = None
        pas_used_fallback = False
        if mode == "as_pas":
            pas_score, pas_top_action_id, pas_used_fallback = self.pas_score_for_output(
                scenario=scenario,
                trait=trait,
                direction=direction,
                parsed_action_id=parsed_action_id,
                layers=layers,
            )

        return GenerationResult(
            text=text,
            mode=mode,
            elapsed_sec=elapsed,
            n_tokens=len(self.tokenizer.encode(text, add_special_tokens=False)),
            parsed_action_id=parsed_action_id,
            layers_hooked=hooked_layers,
            pas_score=pas_score,
            pas_top_action_id=pas_top_action_id,
            pas_used_fallback=pas_used_fallback,
        )

    def _build_messages(
        self,
        scenario: TraitScenario,
        trait: str,
        direction: str,
    ) -> list[dict]:
        return self.prompt_builder.build_chat_messages(
            trait_scenario_to_prompt_dict(scenario),
            prompt_style="elaborate",
            trait=trait,
            target_direction=direction,
        )

    @contextmanager
    def _steering_hooks(self, vectors: dict[int, object], layers: list[int], alpha: float):
        injector = SteeringInjector(self.model, vectors, normalize_vectors=True)
        self._active_hook_contexts += 1
        try:
            injector.inject(layers, alpha=alpha)
            yield
        finally:
            injector.clear()
            self._active_hook_contexts -= 1

    def pas_score_for_output(
        self,
        scenario: TraitScenario,
        trait: str,
        direction: str,
        parsed_action_id: str,
        layers: list[int],
    ) -> tuple[float | None, str | None, bool]:
        """Return PAS cosine for the generated action or fallback top action."""

        pas_layer = selected_layer_for_pas(layers)
        vectors = self.get_persona_vectors(trait, direction)
        pas_result = score_trait_scenario_by_pas(
            self.model,
            self.tokenizer,
            scenario,
            persona_vector=vectors[pas_layer],
            layer_idx=pas_layer,
        )
        return select_pas_score(
            pas_result.get("pas_cosine_scores", {}),
            pas_result.get("pas_ranking", []),
            parsed_action_id,
        )


def parse_action_id(text: str) -> str:
    """Extract an ``<Action>...</Action>`` id from generated text."""

    if "<Action>" not in text or "</Action>" not in text:
        return ""
    return text.split("<Action>", 1)[1].split("</Action>", 1)[0].strip()


def filter_scenarios_by_trait(
    scenarios: list[TraitScenario],
    trait: str,
) -> list[TraitScenario]:
    """Return scenarios for the selected Big Five trait."""

    return [scenario for scenario in scenarios if scenario.trait == trait]


def select_pas_score(
    scores: dict,
    ranking: list,
    parsed_action_id: str,
) -> tuple[float | None, str | None, bool]:
    """Select PAS score for a generated action or fallback to top-ranked action."""

    top_action = str(ranking[0][0]) if ranking else None
    if parsed_action_id and parsed_action_id in scores:
        return float(scores[parsed_action_id]), top_action, False
    if top_action and top_action in scores:
        return float(scores[top_action]), top_action, True
    return None, top_action, True


def smoke_main() -> None:
    parser = argparse.ArgumentParser(description="Run one v4 live inference smoke test.")
    parser.add_argument("--trait", default="extraversion")
    parser.add_argument("--direction", default="high", choices=["high", "low"])
    parser.add_argument("--scenario-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    scenarios = filter_scenarios_by_trait(
        load_trait_scenarios(DEFAULT_SCENARIO_PATH, allowed_splits=["test"]),
        args.trait,
    )
    if not scenarios:
        raise SystemExit(f"No scenarios found for trait: {args.trait}")
    scenario = scenarios[args.scenario_index % len(scenarios)]
    engine = LiveInferenceEngine()
    prompt = engine.generate(
        scenario,
        trait=args.trait,
        direction=args.direction,
        mode="prompt_only",
        max_new_tokens=args.max_new_tokens,
    )
    as_pas = engine.generate(
        scenario,
        trait=args.trait,
        direction=args.direction,
        mode="as_pas",
        max_new_tokens=args.max_new_tokens,
    )
    print("PROMPT_ONLY")
    print(f"elapsed={prompt.elapsed_sec:.2f}s tokens={prompt.n_tokens}")
    print(prompt.text)
    print()
    print("AS_PAS")
    print(
        f"elapsed={as_pas.elapsed_sec:.2f}s tokens={as_pas.n_tokens} "
        f"pas_score={as_pas.pas_score}"
    )
    print(as_pas.text)


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    smoke_main()
