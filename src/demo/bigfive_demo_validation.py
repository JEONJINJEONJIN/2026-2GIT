"""Validation checks for the v4 Big Five class demo."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path

from src.demo.bigfive_demo_data import (
    activation_points,
    choose_demo_scenario,
    load_demo_responses,
    load_demo_scenarios,
    response_pair_for_scenario,
)


@dataclass(frozen=True)
class DemoCheck:
    """One demo validation check result."""

    name: str
    status: str
    details: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEMO_REQUIRED_PATHS = (
    "src/demo/bigfive_app.py",
    "src/demo/bigfive_demo_data.py",
    "src/demo/live_inference.py",
    "src/demo/live_inference_tab.py",
    "scripts/19_run_bigfive_demo.py",
)


def run_demo_validation(project_root: str | Path) -> list[DemoCheck]:
    """Validate demo files, fallback data, and optional dependencies."""

    project_root = Path(project_root)
    checks: list[DemoCheck] = []
    checks.extend(check_demo_paths(project_root))
    checks.extend(check_demo_fallback(project_root))
    checks.extend(check_streamlit_dependency())
    checks.extend(check_pilot_response_data(project_root))
    checks.extend(check_final_figures(project_root))
    return checks


def check_demo_paths(project_root: Path) -> list[DemoCheck]:
    """Check demo implementation paths."""

    missing = [
        path for path in DEMO_REQUIRED_PATHS
        if not (project_root / path).exists()
    ]
    if missing:
        return [
            DemoCheck(
                "demo_required_paths",
                "error",
                f"missing required files: {missing}",
            )
        ]
    return [
        DemoCheck(
            "demo_required_paths",
            "ok",
            f"{len(DEMO_REQUIRED_PATHS)} required files present",
        )
    ]


def check_demo_fallback(project_root: Path) -> list[DemoCheck]:
    """Check that demo can run with built-in fallback data."""

    scenarios = load_demo_scenarios(project_root / "missing_scenarios.jsonl")
    responses = load_demo_responses(project_root / "missing_responses.csv")
    scenario = choose_demo_scenario(
        scenarios,
        {
            "agreeableness": 0.8,
            "conscientiousness": 0.5,
            "neuroticism": 0.5,
            "openness": 0.4,
            "extraversion": 0.5,
        },
    )
    prompt_response, as_pas_response = response_pair_for_scenario(responses, scenario)
    points = activation_points({"agreeableness": 0.8, "openness": 0.6})
    if not prompt_response.speech or not as_pas_response.speech or len(points) != 3:
        return [
            DemoCheck(
                "demo_fallback_data",
                "error",
                "sample scenario, response pair, or activation points are invalid",
            )
        ]
    return [
        DemoCheck(
            "demo_fallback_data",
            "ok",
            f"scenario={scenario.scenario_id}, prompt={prompt_response.condition}, as_pas={as_pas_response.condition}",
        )
    ]


def check_streamlit_dependency() -> list[DemoCheck]:
    """Check whether Streamlit is installed."""

    if importlib.util.find_spec("streamlit") is None:
        return [
            DemoCheck(
                "streamlit_dependency",
                "warn",
                "streamlit is not installed; install requirements before launching UI",
            )
        ]
    return [DemoCheck("streamlit_dependency", "ok", "streamlit installed")]


def check_pilot_response_data(project_root: Path) -> list[DemoCheck]:
    """Check whether pilot-generated demo response data is available."""

    response_path = (
        project_root
        / "results"
        / "v4_bigfive"
        / "pilot_metrics_format_fixed"
        / "generated_responses.csv"
    )
    if not response_path.exists():
        response_path = project_root / "results" / "v4_bigfive" / "pilot_metrics" / "generated_responses.csv"
    if not response_path.exists():
        return [
            DemoCheck(
                "pilot_generated_responses",
                "warn",
                f"missing file: {response_path}; demo will use built-in samples",
            )
        ]
    responses = load_demo_responses(response_path)
    return [
        DemoCheck(
            "pilot_generated_responses",
            "ok",
            f"{len(responses)} response rows available",
        )
    ]


def check_final_figures(project_root: Path) -> list[DemoCheck]:
    """Check whether final analysis figures are available for display."""

    figure_dir = project_root / "results" / "v4_bigfive" / "final_figures"
    figures = sorted(figure_dir.glob("*.png")) if figure_dir.exists() else []
    if len(figures) < 5:
        return [
            DemoCheck(
                "final_figures",
                "warn",
                f"expected at least 5 final figures, found {len(figures)}",
            )
        ]
    return [
        DemoCheck(
            "final_figures",
            "ok",
            f"{len(figures)} PNG figures available",
        )
    ]


def demo_validation_has_errors(checks: list[DemoCheck]) -> bool:
    """Return whether demo validation has hard errors."""

    return any(check.status == "error" for check in checks)
