"""Validation checks for v4 Big Five Phase 2 pilot readiness."""

from __future__ import annotations

from pathlib import Path

from src.experiments.v4_bigfive_plan import load_v4_experiment, parse_v4_conditions
from src.experiments.v4_bigfive_readiness import ReadinessCheck, run_readiness_checks


PHASE2_REQUIRED_PATHS = (
    "docs/v4_bigfive_trait_design.md",
    "docs/v4_bigfive_runbook.md",
    "configs/experiments/v4_bigfive_pilot.yaml",
    "data/bfi/bfi44_scoring.csv",
    "data/bfi/bfi44_item_text.local.example.csv",
    "scripts/09_build_trait_contrastive_pairs.py",
    "scripts/10_prepare_bigfive_pilot.py",
    "scripts/11_convert_trait_bigfive.py",
    "scripts/12_run_bigfive_pilot.py",
    "scripts/13_check_bigfive_readiness.py",
    "scripts/14_analyze_bigfive_results.py",
    "scripts/15_score_side_effects.py",
    "scripts/16_analyze_paraphrase_robustness.py",
    "src/data/bfi.py",
    "src/data/trait_convert.py",
    "src/data/trait_dataset.py",
    "src/evaluation/bigfive_metrics.py",
    "src/evaluation/bigfive_analysis.py",
    "src/evaluation/side_effect_metrics.py",
    "src/evaluation/paraphrase_robustness.py",
    "src/experiments/manifest.py",
    "src/experiments/v4_bigfive_plan.py",
    "src/experiments/v4_bigfive_runner.py",
    "src/experiments/v4_bigfive_readiness.py",
    "src/controls/vector_controls.py",
)

PHASE2_EXPECTED_CONDITIONS = {
    "baseline",
    "one_line_prompt",
    "elaborate_prompt",
    "as_pas_only",
    "elaborate_prompt_as_pas",
}

KNOWN_EXTERNAL_BLOCKERS = {
    "bfi_item_text_overlay",
    "bfi_item_text",
    "trait_scenarios",
    "contrastive_pairs",
    "vectors",
}


def run_phase2_validation(
    project_root: str | Path,
    config_path: str | Path,
    vectors_path: str | Path | None = None,
    allow_missing_external: bool = False,
) -> list[ReadinessCheck]:
    """Run Phase 2 implementation and external-input validation."""

    project_root = Path(project_root)
    checks: list[ReadinessCheck] = []
    checks.extend(check_required_paths(project_root))
    checks.extend(check_pilot_config(config_path))
    readiness = run_readiness_checks(config_path, project_root, vectors_path=vectors_path)
    checks.extend(
        downgrade_known_external_blockers(readiness)
        if allow_missing_external
        else readiness
    )
    return checks


def check_required_paths(project_root: Path) -> list[ReadinessCheck]:
    """Check that Phase 2 implementation artifacts are present."""

    missing = [
        path for path in PHASE2_REQUIRED_PATHS
        if not (project_root / path).exists()
    ]
    if missing:
        return [
            ReadinessCheck(
                "phase2_required_paths",
                "error",
                f"missing required files: {missing}",
            )
        ]
    return [
        ReadinessCheck(
            "phase2_required_paths",
            "ok",
            f"{len(PHASE2_REQUIRED_PATHS)} required files present",
        )
    ]


def check_pilot_config(config_path: str | Path) -> list[ReadinessCheck]:
    """Check Phase 2 pilot config shape."""

    try:
        experiment = load_v4_experiment(config_path)
        conditions = parse_v4_conditions(experiment)
    except Exception as exc:
        return [ReadinessCheck("phase2_pilot_config", "error", str(exc))]

    condition_names = {condition.name for condition in conditions}
    missing_conditions = PHASE2_EXPECTED_CONDITIONS - condition_names
    extra_conditions = condition_names - PHASE2_EXPECTED_CONDITIONS
    if missing_conditions or extra_conditions:
        return [
            ReadinessCheck(
                "phase2_pilot_config",
                "error",
                "unexpected condition set: "
                f"missing={sorted(missing_conditions)}, extra={sorted(extra_conditions)}",
            )
        ]
    if experiment.get("traits") != ["agreeableness"]:
        return [
            ReadinessCheck(
                "phase2_pilot_config",
                "error",
                f"pilot must use agreeableness only, got {experiment.get('traits')}",
            )
        ]
    return [
        ReadinessCheck(
            "phase2_pilot_config",
            "ok",
            "5 pilot conditions, agreeableness-only pilot",
        )
    ]


def downgrade_known_external_blockers(
    checks: list[ReadinessCheck],
) -> list[ReadinessCheck]:
    """Downgrade known missing external-data blockers to warnings."""

    downgraded = []
    for check in checks:
        if check.status == "error" and check.name in KNOWN_EXTERNAL_BLOCKERS:
            downgraded.append(
                ReadinessCheck(
                    check.name,
                    "warn",
                    f"external input not present yet: {check.details}",
                )
            )
        else:
            downgraded.append(check)
    return downgraded


def validation_has_errors(checks: list[ReadinessCheck]) -> bool:
    """Return whether validation has hard errors."""

    return any(check.status == "error" for check in checks)
