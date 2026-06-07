"""Preflight checks for v4 Big Five pilot/final runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.data.bfi import BIG_FIVE_TRAITS, load_bfi_items
from src.data.trait_dataset import load_jsonl, load_trait_scenarios
from src.experiments.v4_bigfive_plan import load_v4_experiment, parse_v4_conditions
from src.steering.vector import SteeringVectorComputer


@dataclass(frozen=True)
class ReadinessCheck:
    """One preflight check result."""

    name: str
    status: str
    details: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def run_readiness_checks(
    config_path: str | Path,
    project_root: str | Path,
    vectors_path: str | Path | None = None,
) -> list[ReadinessCheck]:
    """Run all v4 preflight checks for a config."""

    project_root = Path(project_root)
    checks: list[ReadinessCheck] = []
    try:
        experiment = load_v4_experiment(config_path)
        conditions = parse_v4_conditions(experiment)
    except Exception as exc:
        return [ReadinessCheck("config", "error", str(exc))]

    checks.append(
        ReadinessCheck(
            "config",
            "ok",
            f"{experiment.get('name')} with {len(conditions)} conditions",
        )
    )
    checks.extend(check_bfi(experiment, project_root))
    checks.extend(check_trait_scenarios(experiment, project_root))
    checks.extend(check_contrastive_pairs(experiment, project_root))
    checks.extend(check_vectors(experiment, project_root, vectors_path))
    return checks


def check_bfi(
    experiment: Mapping[str, Any],
    project_root: Path,
) -> list[ReadinessCheck]:
    """Check BFI scoring metadata and item text readiness."""

    scoring = experiment.get("scoring", {})
    bfi_path = project_root / scoring.get("bfi_items_path", "data/bfi/bfi44_scoring.csv")
    bfi_text_path = scoring.get("bfi_item_text_path")
    bfi_text_path = project_root / bfi_text_path if bfi_text_path else None
    if not bfi_path.exists():
        return [ReadinessCheck("bfi_items", "error", f"missing file: {bfi_path}")]
    if bfi_text_path is not None and not bfi_text_path.exists():
        return [
            ReadinessCheck(
                "bfi_item_text_overlay",
                "error",
                f"missing file: {bfi_text_path}",
            )
        ]
    try:
        items = load_bfi_items(bfi_path, text_path=bfi_text_path)
    except Exception as exc:
        return [ReadinessCheck("bfi_items", "error", str(exc))]

    traits = {item.trait for item in items}
    missing_traits = BIG_FIVE_TRAITS - traits
    empty_text = [item.item_id for item in items if not item.text.strip()]
    checks = [
        ReadinessCheck(
            "bfi_items",
            "ok" if len(items) == 44 else "warn",
            f"{len(items)} items, traits={sorted(traits)}",
        )
    ]
    if missing_traits:
        checks.append(
            ReadinessCheck(
                "bfi_trait_coverage",
                "error",
                f"missing traits: {sorted(missing_traits)}",
            )
        )
    else:
        checks.append(ReadinessCheck("bfi_trait_coverage", "ok", "all Big Five traits present"))
    if empty_text:
        checks.append(
            ReadinessCheck(
                "bfi_item_text",
                "error",
                f"{len(empty_text)} item texts empty; model scoring will fail fast",
            )
        )
    else:
        checks.append(ReadinessCheck("bfi_item_text", "ok", "all item texts populated"))
    if bfi_text_path is not None:
        checks.append(
            ReadinessCheck(
                "bfi_item_text_overlay",
                "ok",
                f"loaded local text overlay: {bfi_text_path}",
            )
        )
    return checks


def check_trait_scenarios(
    experiment: Mapping[str, Any],
    project_root: Path,
) -> list[ReadinessCheck]:
    """Check normalized TRAIT scenarios and configured split coverage."""

    scoring = experiment.get("scoring", {})
    scenarios_path = project_root / scoring.get(
        "trait_scenarios_path",
        "data/trait_bigfive/scenarios.jsonl",
    )
    split = str(experiment.get("split", "")).lower()
    configured_traits = {str(trait).lower() for trait in experiment.get("traits", [])}
    if not scenarios_path.exists():
        return [ReadinessCheck("trait_scenarios", "error", f"missing file: {scenarios_path}")]
    try:
        scenarios = load_trait_scenarios(scenarios_path, allowed_splits=[split])
    except Exception as exc:
        return [ReadinessCheck("trait_scenarios", "error", str(exc))]

    by_trait: dict[str, int] = {}
    for scenario in scenarios:
        by_trait[scenario.trait] = by_trait.get(scenario.trait, 0) + 1
    missing_traits = configured_traits - set(by_trait)
    checks = [
        ReadinessCheck(
            "trait_scenarios",
            "ok" if scenarios else "error",
            f"{len(scenarios)} scenarios in split={split}",
        )
    ]
    if missing_traits:
        checks.append(
            ReadinessCheck(
                "trait_split_coverage",
                "error",
                f"missing configured traits in split {split}: {sorted(missing_traits)}",
            )
        )
    else:
        checks.append(
            ReadinessCheck(
                "trait_split_coverage",
                "ok",
                json.dumps(by_trait, sort_keys=True),
            )
        )
    return checks


def check_contrastive_pairs(
    experiment: Mapping[str, Any],
    project_root: Path,
) -> list[ReadinessCheck]:
    """Check per-trait contrastive pair files for vector extraction."""

    pair_dir = project_root / "data" / "trait_bigfive" / "contrastive_pairs"
    configured_traits = [str(trait).lower() for trait in experiment.get("traits", [])]
    if not pair_dir.exists():
        return [ReadinessCheck("contrastive_pairs", "warn", f"missing directory: {pair_dir}")]

    checks = []
    missing = []
    counts = {}
    for trait in configured_traits:
        path = pair_dir / f"{trait}.jsonl"
        if not path.exists():
            missing.append(trait)
            continue
        try:
            counts[trait] = len(load_jsonl(path))
        except Exception as exc:
            checks.append(ReadinessCheck(f"contrastive_pairs.{trait}", "error", str(exc)))
    if missing:
        checks.append(
            ReadinessCheck(
                "contrastive_pairs",
                "warn",
                f"missing pair files for traits: {sorted(missing)}",
            )
        )
    elif counts:
        checks.append(
            ReadinessCheck(
                "contrastive_pairs",
                "ok",
                json.dumps(counts, sort_keys=True),
            )
        )
    return checks


def check_vectors(
    experiment: Mapping[str, Any],
    project_root: Path,
    vectors_path: str | Path | None,
) -> list[ReadinessCheck]:
    """Check optional trait vector bundle readiness."""

    conditions = parse_v4_conditions(experiment)
    needs_vectors = any(condition.steering or condition.pas for condition in conditions)
    if not needs_vectors:
        return [ReadinessCheck("vectors", "ok", "no selected condition requires vectors")]
    if vectors_path is None:
        return [ReadinessCheck("vectors", "warn", "AS/PAS conditions require --vectors_path")]
    path = project_root / vectors_path
    if not path.exists():
        return [ReadinessCheck("vectors", "error", f"missing vector file: {path}")]
    try:
        bundle = SteeringVectorComputer().load_bundle(path)
        vectors = bundle.get("vectors", bundle)
        trait_vectors = vectors["trait_contrast"]
    except Exception as exc:
        return [ReadinessCheck("vectors", "error", f"invalid vector bundle: {exc}")]

    configured_traits = {str(trait).lower() for trait in experiment.get("traits", [])}
    missing_traits = configured_traits - set(trait_vectors)
    if missing_traits:
        return [
            ReadinessCheck(
                "vectors",
                "error",
                f"missing trait vectors: {sorted(missing_traits)}",
            )
        ]
    vector_modes = {
        condition.vector_mode
        for condition in conditions
        if condition.vector_mode is not None
    }
    if "unrelated_trait" in vector_modes and len(trait_vectors) < 2:
        return [
            ReadinessCheck(
                "vectors",
                "error",
                "unrelated_trait control requires at least two trait vectors",
            )
        ]
    layer_counts = {
        trait: len(layers)
        for trait, layers in trait_vectors.items()
        if trait in configured_traits
    }
    return [ReadinessCheck("vectors", "ok", json.dumps(layer_counts, sort_keys=True))]


def readiness_has_errors(checks: Iterable[ReadinessCheck]) -> bool:
    """Return true if any check status is ``error``."""

    return any(check.status == "error" for check in checks)
