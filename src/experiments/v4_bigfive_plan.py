"""Planning utilities for v4 Big Five pilot and final runs."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.controls.vector_controls import SUPPORTED_VECTOR_MODES
from src.data.bfi import BIG_FIVE_TRAITS, load_bfi_items
from src.data.trait_dataset import TraitScenario, load_trait_scenarios
from src.experiments.manifest import RunManifest, file_sha256, get_git_commit, save_manifest
from src.generation.prompt_builder import (
    normalize_bigfive_trait,
    normalize_prompt_style,
    normalize_trait_direction,
)
from src.utils.config import load_config


@dataclass(frozen=True)
class V4Condition:
    """One condition in the v4 Big Five experiment matrix."""

    name: str
    prompt_style: str
    steering: bool
    pas: bool
    vector_mode: str | None = None
    selection_policy: str = "loglik"


@dataclass(frozen=True)
class V4RunPlanRow:
    """One scenario-level unit in a v4 experiment plan."""

    run_id: str
    experiment_name: str
    condition: str
    trait: str
    target_direction: str
    prompt_style: str
    uses_as: bool
    uses_pas: bool
    vector_mode: str | None
    selection_policy: str
    scenario_id: str
    data_split: str
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_v4_experiment(path: str | Path) -> dict[str, Any]:
    """Load and return the inner ``experiment`` config."""

    config = load_config(path)
    experiment = config.get("experiment", config)
    if not isinstance(experiment, dict):
        raise ValueError(f"Invalid v4 experiment config: {path}")
    if experiment.get("version") != "v4_bigfive":
        raise ValueError("v4 experiment config must set version: v4_bigfive")
    return experiment


def parse_v4_conditions(experiment: Mapping[str, Any]) -> list[V4Condition]:
    """Parse and validate condition definitions."""

    raw_conditions = experiment.get("conditions", [])
    if not raw_conditions:
        raise ValueError("v4 experiment config must define at least one condition")
    conditions = []
    seen_names = set()
    for row in raw_conditions:
        name = str(row.get("name") or "").strip()
        if not name:
            raise ValueError("v4 condition missing name")
        if name in seen_names:
            raise ValueError(f"Duplicate v4 condition name: {name}")
        seen_names.add(name)
        prompt_style = normalize_prompt_style(row.get("prompt_style"))
        steering = bool(row.get("steering", False))
        pas = bool(row.get("pas", False))
        vector_mode = row.get("vector_mode")
        selection_policy = str(row.get("selection_policy", "loglik")).strip().lower()
        if selection_policy not in {"loglik", "pas_final"}:
            raise ValueError(
                f"Unsupported selection_policy for condition {name}: "
                f"{selection_policy!r}"
            )
        if selection_policy == "pas_final" and not pas:
            raise ValueError(f"Condition {name} uses pas_final but pas is false")
        if steering and not vector_mode:
            raise ValueError(f"Steering condition {name} must define vector_mode")
        if vector_mode and str(vector_mode) not in SUPPORTED_VECTOR_MODES:
            raise ValueError(
                f"Unsupported vector_mode for condition {name}: {vector_mode!r}"
            )
        conditions.append(
            V4Condition(
                name=name,
                prompt_style=prompt_style,
                steering=steering,
                pas=pas,
                vector_mode=str(vector_mode) if vector_mode else None,
                selection_policy=selection_policy,
            )
        )
    return conditions


def select_v4_scenarios(
    scenarios: Iterable[TraitScenario],
    trait: str,
    split: str,
    max_scenarios: int | None = None,
) -> list[TraitScenario]:
    """Filter scenarios by trait/split and apply a stable limit."""

    trait = normalize_bigfive_trait(trait)
    selected = [
        scenario for scenario in scenarios
        if scenario.trait == trait and scenario.split == split
    ]
    selected = sorted(selected, key=lambda scenario: scenario.scenario_id)
    if max_scenarios is not None:
        selected = selected[:max_scenarios]
    if not selected:
        raise ValueError(f"No TRAIT scenarios for trait={trait!r}, split={split!r}")
    return selected


def build_v4_run_plan(
    experiment: Mapping[str, Any],
    scenarios: Iterable[TraitScenario],
    seed: int = 42,
) -> list[V4RunPlanRow]:
    """Build condition x trait x target-direction x scenario plan rows."""

    experiment_name = str(experiment.get("name") or "v4_bigfive")
    split = str(experiment.get("split") or "").strip().lower()
    if not split:
        raise ValueError("v4 experiment config must define split")
    traits = [normalize_bigfive_trait(trait) for trait in experiment.get("traits", [])]
    if not traits:
        raise ValueError("v4 experiment config must define traits")
    target_directions = [
        normalize_trait_direction(direction)
        for direction in experiment.get("target_directions", [])
    ]
    if not target_directions:
        raise ValueError("v4 experiment config must define target_directions")
    max_scenarios = experiment.get("max_trait_scenarios")
    max_scenarios = int(max_scenarios) if max_scenarios is not None else None
    conditions = parse_v4_conditions(experiment)

    rows: list[V4RunPlanRow] = []
    for condition in conditions:
        for trait in traits:
            trait_scenarios = select_v4_scenarios(
                scenarios,
                trait=trait,
                split=split,
                max_scenarios=max_scenarios,
            )
            for target_direction in target_directions:
                for scenario in trait_scenarios:
                    run_id = stable_run_id(
                        experiment_name,
                        condition.name,
                        trait,
                        target_direction,
                        scenario.scenario_id,
                    )
                    rows.append(
                        V4RunPlanRow(
                            run_id=run_id,
                            experiment_name=experiment_name,
                            condition=condition.name,
                            trait=trait,
                            target_direction=target_direction,
                            prompt_style=condition.prompt_style,
                            uses_as=condition.steering,
                            uses_pas=condition.pas,
                            vector_mode=condition.vector_mode,
                            selection_policy=condition.selection_policy,
                            scenario_id=scenario.scenario_id,
                            data_split=scenario.split,
                            seed=seed,
                        )
                    )
    return rows


def validate_v4_inputs(
    experiment: Mapping[str, Any],
    project_root: str | Path,
) -> dict[str, Any]:
    """Validate BFI metadata and TRAIT scenario availability for a v4 config."""

    project_root = Path(project_root)
    scoring = experiment.get("scoring", {})
    bfi_path = project_root / scoring.get("bfi_items_path", "data/bfi/bfi44_scoring.csv")
    bfi_text_path = resolve_optional_project_path(
        project_root,
        scoring.get("bfi_item_text_path"),
    )
    trait_path = project_root / scoring.get(
        "trait_scenarios_path",
        "data/trait_bigfive/scenarios.jsonl",
    )

    bfi_items = load_bfi_items(bfi_path, text_path=bfi_text_path)
    scenarios = load_trait_scenarios(trait_path, allowed_splits=[experiment["split"]])
    traits = {scenario.trait for scenario in scenarios}
    missing_traits = set(experiment.get("traits", [])) - traits
    if missing_traits:
        raise ValueError(
            "TRAIT scenarios missing configured trait(s): "
            f"{sorted(missing_traits)}"
        )

    return {
        "bfi_items_path": str(bfi_path),
        "bfi_item_count": len(bfi_items),
        "bfi_file_sha256": file_sha256(bfi_path),
        "bfi_item_text_path": str(bfi_text_path) if bfi_text_path else None,
        "bfi_item_text_sha256": file_sha256(bfi_text_path) if bfi_text_path else None,
        "trait_scenarios_path": str(trait_path),
        "trait_scenario_count": len(scenarios),
        "trait_file_sha256": file_sha256(trait_path),
    }


def resolve_optional_project_path(
    project_root: str | Path,
    path: str | Path | None,
) -> Path | None:
    """Resolve optional project-relative paths."""

    if path is None or str(path).strip() == "":
        return None
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def write_run_plan(rows: Iterable[V4RunPlanRow], path: str | Path) -> int:
    """Write a v4 run plan to CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fieldnames = list(V4RunPlanRow.__dataclass_fields__.keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())
    return len(rows)


def write_manifest_bundle(
    rows: Iterable[V4RunPlanRow],
    output_dir: str | Path,
    model_name: str,
    project_root: str | Path,
    input_metadata: Mapping[str, Any] | None = None,
) -> int:
    """Write one manifest JSON per run-plan row."""

    output_dir = Path(output_dir)
    project_root = Path(project_root)
    input_metadata = dict(input_metadata or {})
    git_commit = get_git_commit(project_root)
    count = 0
    for row in rows:
        manifest = RunManifest(
            run_id=row.run_id,
            model_name=model_name,
            condition=row.condition,
            trait=row.trait,
            target_direction=row.target_direction,
            prompt_style=row.prompt_style,
            uses_as=row.uses_as,
            uses_pas=row.uses_pas,
            data_split=row.data_split,
            seed=row.seed,
            git_commit=git_commit,
            extra={
                "experiment_name": row.experiment_name,
                "scenario_id": row.scenario_id,
                "vector_mode": row.vector_mode,
                "selection_policy": row.selection_policy,
                "input_metadata": input_metadata,
            },
        )
        save_manifest(manifest, output_dir / row.run_id / "manifest.json")
        count += 1
    return count


def write_plan_summary(
    rows: Iterable[V4RunPlanRow],
    path: str | Path,
    input_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a compact JSON summary for a generated v4 plan."""

    rows = list(rows)
    summary = {
        "row_count": len(rows),
        "conditions": sorted({row.condition for row in rows}),
        "traits": sorted({row.trait for row in rows}),
        "target_directions": sorted({row.target_direction for row in rows}),
        "scenario_count": len({row.scenario_id for row in rows}),
        "input_metadata": dict(input_metadata or {}),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_run_id(*parts: object) -> str:
    """Build a filesystem-safe deterministic run id from plan components."""

    raw = "__".join(str(part) for part in parts)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")
    return cleaned
