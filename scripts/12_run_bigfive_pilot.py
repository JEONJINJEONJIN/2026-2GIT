"""Run v4 Big Five pilot scoring.

This runner requires normalized TRAIT scenarios and BFI item text. It loads the
model only after validating those inputs, so missing data fails before GPU work.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bfi import load_bfi_items
from src.data.trait_dataset import load_trait_scenarios
from src.experiments.manifest import (
    RunManifest,
    file_sha256,
    get_git_commit,
    save_manifest,
)
from src.experiments.v4_bigfive_plan import (
    load_v4_experiment,
    parse_v4_conditions,
    resolve_optional_project_path,
    select_v4_scenarios,
    validate_v4_inputs,
)
from src.experiments.v4_bigfive_runner import (
    RuntimeCondition,
    apply_pas_final_selection,
    combine_consistency_row,
    generate_trait_scenario_response,
    load_vector_bundle,
    resolve_condition_persona_vectors,
    score_bfi_with_generator,
    score_trait_scenario_by_loglik,
    score_trait_scenario_by_pas,
    selected_layer_for_pas,
    should_normalize_injector_vectors,
    write_csv_rows,
)
from src.evaluation.side_effect_metrics import build_side_effect_row
from src.generation.generator import TextGenerator
from src.generation.prompt_builder import PromptBuilder
from src.models.loader import load_model_and_tokenizer
from src.utils.config import load_config
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v4 Big Five pilot scoring.")
    parser.add_argument("--config", default="configs/experiments/v4_bigfive_pilot.yaml")
    parser.add_argument("--model_config", default="configs/model.yaml")
    parser.add_argument("--vectors_path", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layers", default="18,21,24")
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--metrics_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generate_responses",
        action="store_true",
        help="Also generate free-form NPC responses for side-effect metrics.",
    )
    parser.add_argument("--response_max_new_tokens", type=int, default=96)
    args = parser.parse_args()

    set_seed(args.seed)
    experiment = load_v4_experiment(PROJECT_ROOT / args.config)
    input_metadata = validate_v4_inputs(experiment, PROJECT_ROOT)
    model_config = load_config(PROJECT_ROOT / args.model_config)
    model_name = model_config.get("model", model_config).get("name", "unknown-model")

    scoring = experiment.get("scoring", {})
    bfi_items = load_bfi_items(
        PROJECT_ROOT / scoring["bfi_items_path"],
        text_path=resolve_optional_project_path(
            PROJECT_ROOT,
            scoring.get("bfi_item_text_path"),
        ),
    )
    if any(not item.text.strip() for item in bfi_items):
        raise ValueError(
            "BFI item text is empty. Fill data/bfi/bfi44_scoring.csv text column "
            "or provide a BFI item file with text before running model scoring."
        )
    scenarios = load_trait_scenarios(
        PROJECT_ROOT / scoring["trait_scenarios_path"],
        allowed_splits=[experiment["split"]],
    )

    conditions = parse_v4_conditions(experiment)
    if args.conditions:
        allowed = {name.strip() for name in args.conditions.split(",") if name.strip()}
        conditions = [condition for condition in conditions if condition.name in allowed]
    if not conditions:
        raise ValueError("No conditions selected")

    needs_vectors = any(condition.steering or condition.pas for condition in conditions)
    vector_bundle = None
    if needs_vectors:
        if not args.vectors_path:
            raise ValueError("Selected AS/PAS conditions require --vectors_path")
        vector_bundle = load_vector_bundle(PROJECT_ROOT / args.vectors_path)

    layer_indices = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    pas_layer = selected_layer_for_pas(layer_indices)

    print(f"Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_config)
    generator = TextGenerator(model, tokenizer, generation_config={"do_sample": False})
    prompt_builder = PromptBuilder()

    bfi_rows = []
    trait_rows = []
    consistency_rows = []
    pas_rows = []
    generated_response_rows = []
    side_effect_rows = []

    traits = experiment["traits"]
    target_directions = experiment["target_directions"]
    max_scenarios = experiment.get("max_trait_scenarios")

    for condition_config in conditions:
        condition = RuntimeCondition(
            name=condition_config.name,
            prompt_style=condition_config.prompt_style,
            steering=condition_config.steering,
            pas=condition_config.pas,
            vector_mode=condition_config.vector_mode,
            selection_policy=condition_config.selection_policy,
        )
        for trait in traits:
            selected_scenarios = select_v4_scenarios(
                scenarios,
                trait=trait,
                split=experiment["split"],
                max_scenarios=max_scenarios,
            )
            for target_direction in target_directions:
                persona_vectors = None
                pas_vector = None
                if condition.steering or condition.pas:
                    persona_vectors = resolve_condition_persona_vectors(
                        vector_bundle,
                        trait=trait,
                        target_direction=target_direction,
                        vector_mode=condition.vector_mode,
                        layer_indices=layer_indices,
                        seed=args.seed,
                    )
                    pas_vector = persona_vectors[pas_layer]

                bfi_scores, raw_bfi_rows = score_bfi_with_generator(
                    generator,
                    bfi_items,
                    trait=trait,
                    target_direction=target_direction,
                    prompt_style=condition.prompt_style,
                    prompt_builder=prompt_builder,
                    persona_vectors=persona_vectors,
                    layer_indices=layer_indices,
                    alpha=args.alpha,
                    normalize_vectors=should_normalize_injector_vectors(
                        condition.vector_mode
                    ),
                    use_steering=condition.steering,
                )
                for row in raw_bfi_rows:
                    bfi_rows.append(
                        {
                            "condition": condition.name,
                            "trait": trait,
                            "target_direction": target_direction,
                            **row,
                            "bfi_scores_json": json.dumps(bfi_scores),
                        }
                    )

                for scenario in selected_scenarios:
                    loglik_trait_row = score_trait_scenario_by_loglik(
                        model,
                        tokenizer,
                        scenario=scenario,
                        condition=condition,
                        target_direction=target_direction,
                        prompt_builder=prompt_builder,
                        persona_vectors=persona_vectors,
                        layer_indices=layer_indices,
                        alpha=args.alpha,
                        normalize_vectors=should_normalize_injector_vectors(
                            condition.vector_mode
                        ),
                    )
                    trait_row = loglik_trait_row
                    pas_row = None

                    if condition.pas:
                        pas_row = score_trait_scenario_by_pas(
                            model,
                            tokenizer,
                            scenario=scenario,
                            persona_vector=pas_vector,
                            layer_idx=pas_layer,
                        )
                        pas_rows.append(
                            {
                                "condition": condition.name,
                                "scenario_id": scenario.scenario_id,
                                "trait": trait,
                                "target_direction": target_direction,
                                "pas_layer": pas_layer,
                                "pas_loglik_agreement": pas_row[
                                    "pas_selected_action"
                                ]
                                == loglik_trait_row["loglik_selected_action"],
                                "selection_policy": condition.selection_policy,
                                **pas_row,
                            }
                        )

                    if condition.selection_policy == "pas_final":
                        if pas_row is None:
                            raise ValueError(
                                f"Condition {condition.name} uses pas_final without PAS"
                            )
                        trait_row = apply_pas_final_selection(
                            loglik_trait_row,
                            pas_row,
                        )

                    trait_rows.append(trait_row)
                    consistency_rows.append(
                        combine_consistency_row(bfi_scores, trait_row)
                    )

                    if args.generate_responses:
                        response_row = generate_trait_scenario_response(
                            generator,
                            model,
                            scenario=scenario,
                            condition=condition,
                            target_direction=target_direction,
                            prompt_builder=prompt_builder,
                            persona_vectors=persona_vectors,
                            layer_indices=layer_indices,
                            alpha=args.alpha,
                            normalize_vectors=should_normalize_injector_vectors(
                                condition.vector_mode
                            ),
                            max_new_tokens=args.response_max_new_tokens,
                        )
                        generated_response_rows.append(response_row)
                        side_effect_rows.append(
                            build_side_effect_row(
                                response_row["raw"],
                                valid_actions=response_row["valid_actions"].split("|"),
                                condition=condition.name,
                                scenario_id=scenario.scenario_id,
                                trait=trait,
                                target_direction=target_direction,
                            )
                        )

    output = experiment.get("output", {})
    metrics_dir = (
        PROJECT_ROOT / args.metrics_dir
        if args.metrics_dir
        else PROJECT_ROOT / output.get("metrics_dir", "results/v4_bigfive/pilot_metrics")
    )
    write_csv_rows(bfi_rows, metrics_dir / "bfi_scores.csv")
    write_csv_rows(trait_rows, metrics_dir / "trait_scores.csv")
    write_csv_rows(consistency_rows, metrics_dir / "consistency_metrics.csv")
    write_csv_rows(pas_rows, metrics_dir / "pas_loglik_agreement.csv")
    if args.generate_responses:
        write_csv_rows(
            generated_response_rows,
            metrics_dir / "generated_responses.csv",
        )
        write_csv_rows(side_effect_rows, metrics_dir / "side_effect_metrics.csv")
    save_execution_manifest(
        metrics_dir / "run_manifest.json",
        experiment=experiment,
        model_name=model_name,
        conditions=conditions,
        traits=traits,
        target_directions=target_directions,
        split=experiment["split"],
        seed=args.seed,
        alpha=args.alpha,
        layer_indices=layer_indices,
        config_path=PROJECT_ROOT / args.config,
        model_config_path=PROJECT_ROOT / args.model_config,
        vectors_path=PROJECT_ROOT / args.vectors_path if args.vectors_path else None,
        input_metadata=input_metadata,
        generate_responses=args.generate_responses,
        response_max_new_tokens=args.response_max_new_tokens,
    )

    print(f"Saved BFI rows: {metrics_dir / 'bfi_scores.csv'}")
    print(f"Saved TRAIT rows: {metrics_dir / 'trait_scores.csv'}")
    print(f"Saved consistency rows: {metrics_dir / 'consistency_metrics.csv'}")
    print(f"Saved run manifest: {metrics_dir / 'run_manifest.json'}")
    if args.generate_responses:
        print(f"Saved generated responses: {metrics_dir / 'generated_responses.csv'}")
        print(f"Saved side-effect rows: {metrics_dir / 'side_effect_metrics.csv'}")
    if pas_rows:
        print(f"Saved PAS rows: {metrics_dir / 'pas_loglik_agreement.csv'}")


def save_execution_manifest(
    path: Path,
    *,
    experiment: dict,
    model_name: str,
    conditions: list,
    traits: list[str],
    target_directions: list[str],
    split: str,
    seed: int,
    alpha: float,
    layer_indices: list[int],
    config_path: Path,
    model_config_path: Path,
    vectors_path: Path | None,
    input_metadata: dict,
    generate_responses: bool = False,
    response_max_new_tokens: int = 96,
) -> None:
    """Save one manifest for the scoring invocation and metric bundle."""

    selected_conditions = [condition.name for condition in conditions]
    vector_hash = file_sha256(vectors_path) if vectors_path else None
    manifest = RunManifest(
        run_id=f"{experiment.get('name', 'v4_bigfive')}__seed_{seed}",
        model_name=model_name,
        condition=",".join(selected_conditions),
        trait=",".join(traits),
        target_direction=",".join(target_directions),
        prompt_style="mixed",
        uses_as=any(condition.steering for condition in conditions),
        uses_pas=any(condition.pas for condition in conditions),
        data_split=split,
        seed=seed,
        alpha=alpha,
        layers=layer_indices,
        vector_hash=vector_hash,
        git_commit=get_git_commit(PROJECT_ROOT),
        extra={
            "experiment_name": experiment.get("name"),
            "selected_conditions": selected_conditions,
            "vector_modes": {
                condition.name: condition.vector_mode for condition in conditions
            },
            "selection_policies": {
                condition.name: condition.selection_policy for condition in conditions
            },
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "model_config_path": str(model_config_path),
            "model_config_sha256": file_sha256(model_config_path),
            "vectors_path": str(vectors_path) if vectors_path else None,
            "input_metadata": input_metadata,
            "generate_responses": generate_responses,
            "response_max_new_tokens": response_max_new_tokens,
        },
    )
    save_manifest(manifest, path)


if __name__ == "__main__":
    main()
