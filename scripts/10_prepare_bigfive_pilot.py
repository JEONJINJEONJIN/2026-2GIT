"""Prepare a v4 Big Five pilot run plan and manifests.

This script validates local BFI/TRAIT inputs and writes a scenario-level run
plan. It does not load the LLM or score model outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.trait_dataset import load_trait_scenarios
from src.experiments.v4_bigfive_plan import (
    build_v4_run_plan,
    load_v4_experiment,
    validate_v4_inputs,
    write_manifest_bundle,
    write_plan_summary,
    write_run_plan,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare v4 Big Five pilot plan.")
    parser.add_argument(
        "--config",
        default="configs/experiments/v4_bigfive_pilot.yaml",
        help="v4 Big Five experiment config.",
    )
    parser.add_argument(
        "--model_config",
        default="configs/model.yaml",
        help="Model config used for manifest metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed to record in the run plan.",
    )
    args = parser.parse_args()

    experiment = load_v4_experiment(PROJECT_ROOT / args.config)
    model_config = load_config(PROJECT_ROOT / args.model_config)
    model_name = model_config.get("model", model_config).get("name", "unknown-model")
    input_metadata = validate_v4_inputs(experiment, PROJECT_ROOT)

    scoring = experiment.get("scoring", {})
    scenarios = load_trait_scenarios(
        PROJECT_ROOT / scoring.get("trait_scenarios_path"),
        allowed_splits=[experiment["split"]],
    )
    rows = build_v4_run_plan(experiment, scenarios, seed=args.seed)

    output = experiment.get("output", {})
    run_root = PROJECT_ROOT / output.get("run_root", "results/v4_bigfive/runs")
    plan_path = run_root / "run_plan.csv"
    summary_path = run_root / "plan_summary.json"
    manifest_dir = run_root / "manifests"

    row_count = write_run_plan(rows, plan_path)
    write_plan_summary(rows, summary_path, input_metadata=input_metadata)
    manifest_count = write_manifest_bundle(
        rows,
        manifest_dir,
        model_name=model_name,
        project_root=PROJECT_ROOT,
        input_metadata=input_metadata,
    )

    print(f"Wrote {row_count} plan rows: {plan_path}")
    print(f"Wrote {manifest_count} manifests: {manifest_dir}")
    print(f"Wrote plan summary: {summary_path}")


if __name__ == "__main__":
    main()
