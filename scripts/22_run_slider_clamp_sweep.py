"""Run slider-composite clamp sweep for the Big Five demo backend."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controls.vector_controls import (
    closed_form_slider_norm,
    resolve_slider_persona_vectors,
    slider_unit_norms,
)
from src.data.trait_dataset import load_trait_scenarios
from src.demo.slider_backend import (
    DEFAULT_ALPHA,
    DEFAULT_LAYERS,
    build_slider_messages,
    replace_or_append_action,
)
from src.experiments.v4_bigfive_runner import (
    load_vector_bundle,
    score_trait_scenario_by_pas,
    selected_layer_for_pas,
)
from src.generation.parser import ActionParser
from src.generation.prompt_builder import PromptBuilder
from src.models.loader import load_model_and_tokenizer
from src.steering.injector import SteeringInjector
from src.utils.config import load_config


PROFILES = {
    "social_explorer": {
        "extraversion": 1.0,
        "openness": 1.0,
        "neuroticism": 0.0,
        "agreeableness": 0.5,
        "conscientiousness": 0.5,
    },
    "reserved_conservative": {
        "extraversion": 0.0,
        "openness": 0.0,
        "conscientiousness": 1.0,
        "agreeableness": 0.5,
        "neuroticism": 0.5,
    },
}
CLAMPS = [None, 3.0, 2.5, 2.0, 1.5]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_config", default="configs/model.yaml")
    parser.add_argument(
        "--vectors_path",
        default="results/v4_bigfive/vectors/bigfive_trait_vectors_full_fp16.pt",
    )
    parser.add_argument(
        "--scenario_path",
        default="data/trait_bigfive/scenarios.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/v4_bigfive/slider_clamp_sweep/slider_clamp_sweep.csv",
    )
    parser.add_argument("--n_scenarios", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--layers", default="18,21,24")
    args = parser.parse_args()

    layers = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    if not layers:
        raise ValueError("At least one layer is required")
    if args.n_scenarios < 8:
        raise ValueError("--n_scenarios must be >= 8")

    scenarios = sorted(
        load_trait_scenarios(PROJECT_ROOT / args.scenario_path, allowed_splits=["test"]),
        key=lambda scenario: scenario.scenario_id,
    )[: args.n_scenarios]
    vector_bundle = load_vector_bundle(PROJECT_ROOT / args.vectors_path)
    model_config = load_config(PROJECT_ROOT / args.model_config)
    model, tokenizer = load_model_and_tokenizer(model_config)
    prompt_builder = PromptBuilder()
    action_parser = ActionParser()
    pas_layer = selected_layer_for_pas(layers)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    from src.generation.generator import TextGenerator

    generator = TextGenerator(
        model,
        tokenizer,
        generation_config={"do_sample": False, "max_new_tokens": args.max_new_tokens},
    )

    for profile_name, persona in PROFILES.items():
        for clamp_x in CLAMPS:
            unit_norms = slider_unit_norms(
                vector_bundle,
                persona,
                layer_indices=layers,
                clamp_x=clamp_x,
            )
            closed_norm = closed_form_slider_norm(
                vector_bundle,
                persona,
                layer_idx=pas_layer,
            )
            expected = min(closed_norm, clamp_x) if clamp_x is not None else closed_norm
            measured = unit_norms[pas_layer]
            if abs(measured - expected) > 1e-2:
                raise AssertionError(
                    f"norm mismatch for {profile_name} clamp={clamp_x}: "
                    f"measured={measured:.4f}, expected={expected:.4f}"
                )

            vectors = resolve_slider_persona_vectors(
                vector_bundle,
                persona,
                layer_indices=layers,
                alpha=args.alpha,
                clamp_x=clamp_x,
            )
            for scenario in scenarios:
                messages = build_slider_messages(prompt_builder, scenario, persona)
                injector = SteeringInjector(model, vectors, normalize_vectors=False)
                injector.inject(layers, alpha=1.0)
                try:
                    raw = generator.generate(
                        messages,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                finally:
                    injector.clear()

                valid_actions = [action.id for action in scenario.actions]
                parsed = action_parser.parse(raw, valid_actions=valid_actions)
                pas_result = score_trait_scenario_by_pas(
                    model,
                    tokenizer,
                    scenario,
                    persona_vector=vectors[pas_layer],
                    layer_idx=pas_layer,
                )
                top_action = pas_result["pas_selected_action"]
                pas_score = float(pas_result["pas_cosine_scores"][top_action])
                final_text = replace_or_append_action(raw, top_action)

                rows.append(
                    {
                        "profile": profile_name,
                        "clamp_x": "none" if clamp_x is None else clamp_x,
                        "scenario_id": scenario.scenario_id,
                        "norm": measured,
                        "closed_form_norm": closed_norm,
                        "rep_rate_3": ngram_repetition_rate(raw, 3),
                        "rep_rate_4": ngram_repetition_rate(raw, 4),
                        "len": len(tokenizer.encode(raw, add_special_tokens=False)),
                        "parse_ok": bool(parsed["parse_success"]),
                        "format_failure": not bool(parsed["parse_success"]),
                        "pas_score": pas_score,
                        "pas_selected_action": top_action,
                        "raw_text": raw,
                        "final_text": final_text,
                    }
                )

    write_rows(output_path, rows)
    print(f"Wrote {len(rows)} rows: {output_path}")


def ngram_repetition_rate(text: str, n: int) -> float:
    """Return repeated n-gram fraction among all n-grams."""

    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    repeated = len(ngrams) - len(set(ngrams))
    return float(repeated / len(ngrams))


def write_rows(path: Path, rows: list[dict]) -> None:
    """Write CSV rows."""

    fieldnames = [
        "profile",
        "clamp_x",
        "scenario_id",
        "norm",
        "closed_form_norm",
        "rep_rate_3",
        "rep_rate_4",
        "len",
        "parse_ok",
        "format_failure",
        "pas_score",
        "pas_selected_action",
        "raw_text",
        "final_text",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
