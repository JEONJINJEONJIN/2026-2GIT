"""Full experiment runner.

For each condition (baseline, steering_only, steering_projection, projection_only),
for each persona x scenario x repeat:
  - Build prompt
  - Optionally inject steering vector
  - Generate response
  - Parse action from response
  - Optionally apply projection-based action selection
  - Save result as a JSONL line
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tqdm import tqdm

from src.models.hooks import HookManager
from src.models.loader import load_model_and_tokenizer
from src.generation.generator import TextGenerator
from src.generation.prompt_builder import PromptBuilder
from src.projection.embedder import ActionEmbedder
from src.projection.selector import PersonaActionSelector, ProjectedActionResolver
from src.steering.vector import SteeringVectorComputer
from src.utils.config import load_config
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_action(text: str) -> str | None:
    """Extract action from <Action>...</Action> tags in generated text."""
    pattern = r"<Action>(.*?)</Action>"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def parse_speech(text: str) -> str:
    """Extract speech content from [Speech] tag."""
    pattern = r"\[Speech\]\s*(.*?)(?=<Action>|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def load_scenarios(scenarios_path: Path) -> list[dict]:
    """Load scenarios from a JSONL file."""
    scenarios = []
    with open(scenarios_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description="Run the full RSA experiment."
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        default=None,
        help="Directory containing config files (default: configs/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for raw results (default: results/raw/).",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated list of conditions to run. "
             "Default: baseline,steering_only,steering_projection,projection_only",
    )
    parser.add_argument(
        "--vectors_path",
        type=str,
        default=None,
        help="Path to steering vectors .pt file.",
    )
    parser.add_argument(
        "--scenarios_path",
        type=str,
        default=None,
        help="Path to scenarios JSONL file.",
    )
    args = parser.parse_args()

    # Resolve paths
    config_dir = Path(args.config_dir) if args.config_dir else (
        PROJECT_ROOT / "configs"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "raw"
    )
    scenarios_path = Path(args.scenarios_path) if args.scenarios_path else (
        PROJECT_ROOT / "data" / "scenarios" / "scenarios.jsonl"
    )

    # Load configs
    model_config = load_config(config_dir / "model.yaml")
    steering_config = load_config(config_dir / "steering.yaml")
    experiment_config = load_config(config_dir / "experiment.yaml")

    exp_cfg = experiment_config.get("experiment", experiment_config)
    seed = exp_cfg.get("seed", 42)
    set_seed(seed)

    # Determine conditions to run
    all_conditions = {
        c["name"]: c for c in exp_cfg.get("conditions", [])
    }
    if args.conditions:
        condition_names = [c.strip() for c in args.conditions.split(",")]
    else:
        condition_names = list(all_conditions.keys())

    logger.info("Conditions to run: %s", condition_names)

    # Load model
    logger.info("Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_config)

    # Load steering vectors if any condition uses steering
    vectors = None
    needs_steering = any(
        all_conditions.get(c, {}).get("steering", False) for c in condition_names
    )
    if needs_steering:
        vectors_path = Path(args.vectors_path) if args.vectors_path else (
            PROJECT_ROOT / "results" / "vectors" / "steering_vectors_fp16.pt"
        )
        logger.info("Loading steering vectors from: %s", vectors_path)
        computer = SteeringVectorComputer()
        vectors = computer.load_vectors(vectors_path)

    # Load scenarios
    logger.info("Loading scenarios from: %s", scenarios_path)
    scenarios = load_scenarios(scenarios_path)

    num_scenarios = exp_cfg.get("num_scenarios", len(scenarios))
    scenarios = scenarios[:num_scenarios]
    logger.info("Using %d scenarios.", len(scenarios))

    personas = exp_cfg.get("personas", ["aggressive", "cooperative"])
    num_repeats = exp_cfg.get("num_repeats", 5)
    gen_config = exp_cfg.get("generation", {})

    # Setup components
    prompt_builder = PromptBuilder()
    generator = TextGenerator(model, tokenizer, generation_config=gen_config)

    # Precompute action embeddings for projection conditions
    needs_projection = any(
        all_conditions.get(c, {}).get("projection", False) for c in condition_names
    )
    action_embeddings = None
    if needs_projection:
        logger.info("Computing action embeddings for projection...")
        steer_cfg = steering_config.get("steering", steering_config)
        layer_groups = steer_cfg.get("layer_groups", {})
        middle_layers = layer_groups.get("middle", [15])
        embed_layer = middle_layers[len(middle_layers) // 2] if middle_layers else 15

        embedder = ActionEmbedder(model, tokenizer, layer_idx=embed_layer)
        all_action_defs = []
        for scenario in scenarios:
            for action in scenario.get("actions", []):
                all_action_defs.append(action)
        action_embeddings = embedder.embed_actions(all_action_defs)
        logger.info("Computed embeddings for %d actions.", len(action_embeddings))

    # Prepare output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute alpha from steering config
    steer_cfg = steering_config.get("steering", steering_config)
    alpha_presets = steer_cfg.get("alpha_presets", {})
    alpha = alpha_presets.get("pilot", [2.0])[1] if "pilot" in alpha_presets else 2.0

    # Layer selection for steering
    layer_groups = steer_cfg.get("layer_groups", {})
    middle_layers = layer_groups.get("middle", [15])
    steering_layer = middle_layers[len(middle_layers) // 2] if middle_layers else 15

    # Run experiment
    for condition_name in condition_names:
        condition = all_conditions.get(condition_name)
        if condition is None:
            logger.warning("Unknown condition '%s', skipping.", condition_name)
            continue

        use_steering = condition.get("steering", False)
        use_projection = condition.get("projection", False)

        output_file = output_dir / f"{condition_name}.jsonl"
        logger.info(
            "Running condition: %s (steering=%s, projection=%s)",
            condition_name, use_steering, use_projection,
        )

        total_runs = len(personas) * len(scenarios) * num_repeats
        results = []

        with tqdm(total=total_runs, desc=condition_name) as pbar:
            for persona in personas:
                for scenario in scenarios:
                    for repeat_idx in range(num_repeats):
                        # Build prompt
                        messages = prompt_builder.build_chat_messages(
                            scenario, persona
                        )

                        # Optionally inject steering
                        hook_manager = None
                        if use_steering and vectors is not None:
                            hook_manager = HookManager()
                            if steering_layer in vectors:
                                sv = vectors[steering_layer]
                                hook_manager.register_steering_hook(
                                    model, steering_layer, sv, alpha
                                )

                        try:
                            # Generate
                            generated_text = generator.generate(messages)
                        finally:
                            if hook_manager is not None:
                                hook_manager.remove_all()

                        # Parse action
                        action = parse_action(generated_text)
                        speech = parse_speech(generated_text)

                        # Optionally apply projection
                        final_action = action
                        projection_override = False
                        if use_projection and action_embeddings is not None and vectors is not None:
                            if steering_layer in vectors:
                                persona_vector = vectors[steering_layer]
                                resolver = ProjectedActionResolver(action_embeddings)
                                final_action = resolver.resolve(
                                    persona_vector, action or "", threshold=0.0
                                )
                                projection_override = (final_action != action)

                        result = {
                            "condition": condition_name,
                            "persona": persona,
                            "scenario_id": scenario.get("id", ""),
                            "repeat": repeat_idx,
                            "generated_text": generated_text,
                            "parsed_action": action,
                            "final_action": final_action,
                            "projection_override": projection_override,
                            "speech": speech,
                            "timestamp": time.time(),
                        }
                        results.append(result)
                        pbar.update(1)

        # Save results
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.info("Saved %d results to: %s", len(results), output_file)

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
