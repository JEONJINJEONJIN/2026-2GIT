"""Validation checks for v4 Big Five Phase 3 full-run readiness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.data.bfi import BIG_FIVE_TRAITS
from src.experiments.v4_bigfive_phase2_validation import (
    PHASE2_EXPECTED_CONDITIONS,
    check_required_paths,
    downgrade_known_external_blockers,
)
from src.experiments.v4_bigfive_plan import load_v4_experiment, parse_v4_conditions
from src.experiments.v4_bigfive_readiness import ReadinessCheck, run_readiness_checks


PHASE3_REQUIRED_PATHS = (
    "configs/experiments/v4_bigfive_final.yaml",
    "configs/experiments/v4_bigfive_controls.yaml",
    "scripts/12_run_bigfive_pilot.py",
    "scripts/14_analyze_bigfive_results.py",
    "src/evaluation/bigfive_analysis.py",
    "src/experiments/v4_bigfive_runner.py",
)

PHASE3_TUNING_BLOCKERS = {
    "phase3_alpha_layers",
}


def run_phase3_validation(
    project_root: str | Path,
    config_path: str | Path,
    vectors_path: str | Path | None = None,
    allow_missing_external: bool = False,
    allow_missing_tuning: bool = False,
) -> list[ReadinessCheck]:
    """Run Phase 3 implementation, config, tuning, and input validation."""

    project_root = Path(project_root)
    checks: list[ReadinessCheck] = []
    checks.extend(check_required_paths(project_root))
    checks.extend(check_phase3_required_paths(project_root))
    checks.extend(check_final_config(config_path))
    checks.extend(check_final_tuning(config_path))
    if allow_missing_tuning:
        checks = downgrade_tuning_blockers(checks)
    readiness = run_readiness_checks(config_path, project_root, vectors_path=vectors_path)
    if allow_missing_external:
        readiness = downgrade_known_external_blockers(readiness)
    checks.extend(readiness)
    return checks


def check_phase3_required_paths(project_root: Path) -> list[ReadinessCheck]:
    """Check Phase 3-specific files."""

    missing = [
        path for path in PHASE3_REQUIRED_PATHS
        if not (project_root / path).exists()
    ]
    if missing:
        return [
            ReadinessCheck(
                "phase3_required_paths",
                "error",
                f"missing required files: {missing}",
            )
        ]
    return [
        ReadinessCheck(
            "phase3_required_paths",
            "ok",
            f"{len(PHASE3_REQUIRED_PATHS)} required files present",
        )
    ]


def check_final_config(config_path: str | Path) -> list[ReadinessCheck]:
    """Check final full Big Five config shape."""

    try:
        experiment = load_v4_experiment(config_path)
        conditions = parse_v4_conditions(experiment)
    except Exception as exc:
        return [ReadinessCheck("phase3_final_config", "error", str(exc))]

    condition_names = {condition.name for condition in conditions}
    missing_conditions = PHASE2_EXPECTED_CONDITIONS - condition_names
    extra_conditions = condition_names - PHASE2_EXPECTED_CONDITIONS
    if missing_conditions or extra_conditions:
        return [
            ReadinessCheck(
                "phase3_final_config",
                "error",
                "unexpected condition set: "
                f"missing={sorted(missing_conditions)}, extra={sorted(extra_conditions)}",
            )
        ]
    traits = {str(trait).lower() for trait in experiment.get("traits", [])}
    if traits != BIG_FIVE_TRAITS:
        return [
            ReadinessCheck(
                "phase3_final_config",
                "error",
                f"final config must include all Big Five traits, got {sorted(traits)}",
            )
        ]
    if str(experiment.get("split", "")).lower() != "test":
        return [
            ReadinessCheck(
                "phase3_final_config",
                "error",
                f"final config must use split=test, got {experiment.get('split')}",
            )
        ]
    target_directions = {str(item).lower() for item in experiment.get("target_directions", [])}
    if target_directions != {"high", "low"}:
        return [
            ReadinessCheck(
                "phase3_final_config",
                "error",
                f"final config must use high/low targets, got {sorted(target_directions)}",
            )
        ]
    return [
        ReadinessCheck(
            "phase3_final_config",
            "ok",
            "5 traits, high/low targets, 5 conditions, test split",
        )
    ]


def check_final_tuning(config_path: str | Path) -> list[ReadinessCheck]:
    """Check that final run alpha/layers have been fixed from pilot/dev."""

    experiment = load_v4_experiment(config_path)
    steering = experiment.get("steering", {})
    alpha = steering.get("alpha")
    layers = steering.get("layers")
    tune_on_this_split = bool(steering.get("tune_on_this_split", False))
    if tune_on_this_split:
        return [
            ReadinessCheck(
                "phase3_tuning_split",
                "error",
                "final config must not tune on test split",
            )
        ]
    if alpha is None or not layers:
        return [
            ReadinessCheck(
                "phase3_alpha_layers",
                "error",
                "fill steering.alpha and steering.layers from Phase 2/dev before final run",
            )
        ]
    if not _valid_numeric_alpha(alpha) or not _valid_layer_list(layers):
        return [
            ReadinessCheck(
                "phase3_alpha_layers",
                "error",
                f"invalid alpha/layers: alpha={alpha!r}, layers={layers!r}",
            )
        ]
    return [
        ReadinessCheck(
            "phase3_alpha_layers",
            "ok",
            f"alpha={alpha}, layers={layers}",
        )
    ]


def downgrade_tuning_blockers(
    checks: list[ReadinessCheck],
) -> list[ReadinessCheck]:
    """Downgrade known pilot-dependent tuning blockers to warnings."""

    downgraded = []
    for check in checks:
        if check.status == "error" and check.name in PHASE3_TUNING_BLOCKERS:
            downgraded.append(
                ReadinessCheck(
                    check.name,
                    "warn",
                    f"pilot/dev tuning not fixed yet: {check.details}",
                )
            )
        else:
            downgraded.append(check)
    return downgraded


def validation_has_errors(checks: list[ReadinessCheck]) -> bool:
    """Return whether validation has hard errors."""

    return any(check.status == "error" for check in checks)


def _valid_numeric_alpha(alpha: Any) -> bool:
    return isinstance(alpha, int | float) and float(alpha) > 0.0


def _valid_layer_list(layers: Any) -> bool:
    return (
        isinstance(layers, list)
        and bool(layers)
        and all(isinstance(layer, int) for layer in layers)
    )
