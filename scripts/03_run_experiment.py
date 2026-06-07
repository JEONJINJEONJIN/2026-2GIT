"""AS-only experiment runner.

Runs three conditions:
  - neutral_baseline: neutral prompt, no steering
  - prompt_baseline: persona prompt, no steering
  - as_only: neutral prompt plus activation steering
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.evaluation.loglik_selector import select_action_by_loglik
from src.generation.generator import TextGenerator
from src.generation.parser import ActionParser
from src.generation.prompt_builder import PromptBuilder
from src.models.loader import load_model_and_tokenizer
from src.steering.injector import SteeringInjector
from src.steering.vector import SteeringVectorComputer
from src.utils.config import load_config
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def signed_vectors(vectors: dict[int, object], persona: str) -> dict:
    sign = 1.0 if persona == "aggressive" else -1.0
    return {layer: sign * vector for layer, vector in vectors.items()}


def resolve_persona_vectors(vectors: dict, persona: str, vector_mode: str) -> dict:
    """Resolve condition/persona-specific vectors from flat or nested bundles."""
    if vector_mode == "separate":
        if persona not in vectors:
            raise ValueError(
                f"Separate vector bundle has no vector set for persona: {persona}"
            )
        return vectors[persona]
    if vector_mode in {"contrast_explicit", "orthogonalized"}:
        if vector_mode not in vectors or persona not in vectors[vector_mode]:
            raise ValueError(
                f"Vector bundle has no {vector_mode} set for persona: {persona}"
            )
        return vectors[vector_mode][persona]

    if "contrast" in vectors and isinstance(vectors["contrast"], dict):
        return signed_vectors(vectors["contrast"], persona)
    return signed_vectors(vectors, persona)


def collect_available_layers(vectors: dict) -> set[int]:
    """Collect all layer indices from a flat or nested vector bundle."""
    layers = set()
    for key, value in vectors.items():
        if isinstance(key, int):
            layers.add(key)
        elif isinstance(value, dict):
            layers.update(collect_available_layers(value))
    return layers


def main():
    parser = argparse.ArgumentParser(description="Run AS-only NPC experiments.")
    parser.add_argument("--config_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--conditions", type=str, default=None)
    parser.add_argument("--vectors_path", type=str, default=None)
    parser.add_argument("--scenarios_path", type=str, default=None)
    parser.add_argument("--layer_group", type=str, default="middle")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--selection_mode",
        choices=["generate", "loglik"],
        default="loglik",
        help="Use free-form generation or log-likelihood action selection.",
    )
    parser.add_argument(
        "--num_scenarios",
        type=int,
        default=None,
        help="Override experiment.yaml num_scenarios for smoke runs.",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=None,
        help="Override experiment.yaml num_repeats for smoke runs.",
    )
    args = parser.parse_args()

    config_dir = Path(args.config_dir) if args.config_dir else PROJECT_ROOT / "configs"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "v2" / "raw"
    scenarios_path = Path(args.scenarios_path) if args.scenarios_path else (
        PROJECT_ROOT / "data" / "scenarios" / "scenarios.jsonl"
    )

    model_config = load_config(config_dir / "model.yaml")
    steering_config = load_config(config_dir / "steering.yaml")
    experiment_config = load_config(config_dir / "experiment.yaml")

    model_cfg = model_config.get("model", model_config)
    steering_cfg = steering_config.get("steering", steering_config)
    exp_cfg = experiment_config.get("experiment", experiment_config)

    seed = exp_cfg.get("seed", 42)
    set_seed(seed)

    all_conditions = {c["name"]: c for c in exp_cfg.get("conditions", [])}
    if args.conditions:
        condition_names = [c.strip() for c in args.conditions.split(",") if c.strip()]
    else:
        condition_names = list(all_conditions.keys())

    unknown = [name for name in condition_names if name not in all_conditions]
    if unknown:
        raise ValueError(f"Unknown condition(s): {unknown}")

    layer_groups = steering_cfg.get("layer_groups", {})
    layer_indices = layer_groups.get(args.layer_group)
    if layer_indices is None:
        raise ValueError(
            f"Unknown layer group '{args.layer_group}'. "
            f"Available groups: {sorted(layer_groups.keys())}"
        )

    alpha = args.alpha if args.alpha is not None else steering_cfg.get("default_alpha", 2.0)
    normalize_vectors = bool(steering_cfg.get("normalize_vectors", True))
    token_aggregation = steering_cfg.get("token_aggregation", "last_input_token")
    hook_target = steering_cfg.get("hook_target", "post_block_residual")

    needs_steering = any(all_conditions[name].get("steering", False) for name in condition_names)
    vectors = None
    vector_metadata = {}
    if needs_steering:
        vectors_path = Path(args.vectors_path) if args.vectors_path else (
            PROJECT_ROOT / steering_cfg.get("vector_path", "results/vectors/steering_vectors_fp16.pt")
        )
        logger.info("Loading steering vectors from: %s", vectors_path)
        bundle = SteeringVectorComputer().load_bundle(vectors_path)
        vectors = bundle["vectors"]
        vector_metadata = bundle.get("metadata", {})
        available_layers = collect_available_layers(vectors)
        missing = sorted(set(layer_indices) - available_layers)
        if missing:
            raise ValueError(f"Steering vector file is missing layers: {missing}")

    logger.info("Loading model: %s", model_cfg.get("name"))
    model, tokenizer = load_model_and_tokenizer(model_config)

    scenarios = load_jsonl(scenarios_path)
    num_scenarios = (
        args.num_scenarios
        if args.num_scenarios is not None
        else exp_cfg.get("num_scenarios", len(scenarios))
    )
    scenarios = scenarios[:num_scenarios]
    personas = exp_cfg.get("personas", ["aggressive", "cooperative"])
    num_repeats = (
        args.num_repeats
        if args.num_repeats is not None
        else exp_cfg.get("num_repeats", 30)
    )
    gen_config = exp_cfg.get("generation", {})
    selection_mode = args.selection_mode

    prompt_builder = PromptBuilder()
    parser_obj = ActionParser()
    generator = TextGenerator(model, tokenizer, generation_config=gen_config)
    git_commit = get_git_commit()

    output_dir.mkdir(parents=True, exist_ok=True)

    for condition_name in condition_names:
        condition = all_conditions[condition_name]
        use_steering = bool(condition.get("steering", False))
        include_persona = bool(condition.get("persona_prompt", False))
        vector_mode = condition.get("vector_mode", "contrast_negated")
        output_file = output_dir / f"{condition_name}.jsonl"

        total_runs = len(personas) * len(scenarios) * num_repeats
        logger.info(
            "Running %s (steering=%s, persona_prompt=%s, selection=%s, n=%d)",
            condition_name,
            use_steering,
            include_persona,
            selection_mode,
            total_runs,
        )

        with open(output_file, "w", encoding="utf-8") as f:
            with tqdm(total=total_runs, desc=condition_name) as pbar:
                for persona in personas:
                    persona_vectors = (
                        resolve_persona_vectors(vectors, persona, vector_mode)
                        if use_steering and vectors
                        else None
                    )

                    for scenario in scenarios:
                        scenario_actions = scenario.get("actions", [])
                        valid_actions = [a["id"] for a in scenario_actions]
                        action_texts = [
                            prompt_builder.action_scoring_text(action)
                            for action in scenario_actions
                        ]
                        generate_messages = prompt_builder.build_chat_messages(
                            scenario,
                            persona=persona,
                            include_persona=include_persona,
                        )
                        loglik_messages = prompt_builder.build_action_choice_messages(
                            scenario,
                            persona=persona,
                            include_persona=include_persona,
                        )

                        for repeat_idx in range(num_repeats):
                            if selection_mode == "loglik" and not action_texts:
                                raise ValueError(
                                    f"Scenario {scenario.get('id')} has no valid actions."
                                )

                            injector = None
                            if persona_vectors is not None:
                                injector = SteeringInjector(
                                    model,
                                    persona_vectors,
                                    normalize_vectors=normalize_vectors,
                                )
                                injector.inject(layer_indices, alpha=alpha)

                            try:
                                if selection_mode == "loglik":
                                    selection = select_action_by_loglik(
                                        model,
                                        tokenizer,
                                        loglik_messages,
                                        action_texts,
                                        length_normalize=True,
                                    )
                                    selected_idx = selection["selected_idx"]
                                    action = valid_actions[selected_idx]
                                    selected_text = action_texts[selected_idx]
                                    generated_text = (
                                        f"{selected_text}\n<Action>{action}</Action>"
                                    )
                                    parsed = {
                                        "speech": selected_text,
                                        "action": action,
                                        "parse_success": True,
                                        "tag_found": True,
                                    }
                                else:
                                    generated_text = generator.generate(generate_messages)
                                    parsed = parser_obj.parse(
                                        generated_text,
                                        valid_actions=valid_actions,
                                    )
                                    action = parsed["action"]
                                    selection = {
                                        "logliks": None,
                                        "normalized_logliks": None,
                                        "softmax_probs": None,
                                    }
                            finally:
                                if injector is not None:
                                    injector.clear()

                            result = {
                                "condition": condition_name,
                                "persona": persona,
                                "scenario_id": scenario.get("id", ""),
                                "repeat": repeat_idx,
                                "seed": seed,
                                "model_name": model_cfg.get("name"),
                                "model_dtype": model_cfg.get("dtype"),
                                "alpha": alpha if use_steering else None,
                                "layers": layer_indices if use_steering else [],
                                "hook_target": hook_target if use_steering else None,
                                "token_aggregation": token_aggregation,
                                "vector_normalized": normalize_vectors if use_steering else None,
                                "vector_metadata": vector_metadata if use_steering else {},
                                "vector_mode": vector_mode if use_steering else None,
                                "gen_config": gen_config,
                                "selection_mode": selection_mode,
                                "action_candidates": valid_actions,
                                "action_candidate_texts": action_texts,
                                "action_logliks": selection["logliks"],
                                "action_normalized_logliks": selection[
                                    "normalized_logliks"
                                ],
                                "action_softmax_probs": selection["softmax_probs"],
                                "generated_text": generated_text,
                                "speech": parsed["speech"],
                                "parsed_action": action,
                                "final_action": action,
                                "parse_ok": bool(parsed["parse_success"]),
                                "tag_found": bool(parsed["tag_found"]),
                                "timestamp": time.time(),
                                "git_commit": git_commit,
                            }
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            pbar.update(1)

        logger.info("Saved results to: %s", output_file)

    logger.info("AS-only experiment complete.")


if __name__ == "__main__":
    main()
