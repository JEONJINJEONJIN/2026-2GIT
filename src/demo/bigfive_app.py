"""Streamlit demo for v4 Big Five NPC speech-action consistency."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.demo.bigfive_demo_data import (
    BIG_FIVE_TRAITS,
    DEMO_PERSONA_PRESETS,
    activation_points,
    choose_demo_scenario,
    load_demo_responses,
    load_demo_scenarios,
    prioritize_conflict_scenarios,
    response_pair_for_scenario,
)
from src.demo.live_inference import LiveInferenceEngine
from src.demo.live_inference_tab import render_live_inference_tab

DEMO_RESPONSE_CANDIDATES = (
    "results/v4_bigfive/pilot_metrics_format_fixed/generated_responses.csv",
    "results/v4_bigfive/pilot_metrics/generated_responses.csv",
)
FINAL_METRICS_DIR = "results/v4_bigfive/final_metrics"
FINAL_FIGURE_DIR = "results/v4_bigfive/final_figures"


def main() -> None:
    try:
        import pandas as pd
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Streamlit demo requires optional dependencies. "
            "Install streamlit and pandas, then rerun this app."
        ) from exc

    st.set_page_config(page_title="Big Five NPC Consistency", layout="wide")
    st.title("Big Five NPC Consistency")

    @st.cache_resource
    def get_live_engine():
        return LiveInferenceEngine()

    scenarios = prioritize_conflict_scenarios(
        load_demo_scenarios(PROJECT_ROOT / "data/trait_bigfive/scenarios.jsonl")
    )
    response_path = next(
        (
            PROJECT_ROOT / candidate
            for candidate in DEMO_RESPONSE_CANDIDATES
            if (PROJECT_ROOT / candidate).exists()
        ),
        None,
    )
    responses = load_demo_responses(response_path)
    final_metrics_dir = PROJECT_ROOT / FINAL_METRICS_DIR
    consistency_summary = _read_csv(final_metrics_dir / "consistency_summary.csv", pd)
    condition_differences = _read_csv(final_metrics_dir / "condition_differences.csv", pd)
    trait_scores_df = _read_csv(final_metrics_dir / "trait_scores.csv", pd)

    if not trait_scores_df.empty:
        final_scenario_ids = set(trait_scores_df["scenario_id"].astype(str))
        final_scenarios = [
            scenario for scenario in scenarios
            if scenario.scenario_id in final_scenario_ids
        ]
        if final_scenarios:
            scenarios = final_scenarios

    with st.sidebar:
        st.header("Persona")
        persona_preset = st.selectbox(
            "Preset",
            options=list(DEMO_PERSONA_PRESETS),
            format_func=lambda preset: preset["label"],
        )
        preset_values = persona_preset["values"]
        trait_scores = {
            trait: st.slider(
                trait.replace("_", " ").title(),
                min_value=0.0,
                max_value=1.0,
                value=float(preset_values.get(trait, 0.5)),
                step=0.05,
            )
            for trait in BIG_FIVE_TRAITS
        }
        selected_trait = max(
            BIG_FIVE_TRAITS,
            key=lambda trait: abs(float(trait_scores.get(trait, 0.5)) - 0.5),
        )
        target_direction = "high" if trait_scores[selected_trait] >= 0.5 else "low"
        st.divider()
        st.metric("Active trait", selected_trait.title())
        st.metric("Target direction", target_direction)

    scenario = choose_demo_scenario(scenarios, trait_scores)
    conflict_scenarios = prioritize_conflict_scenarios(scenarios)
    trait_scenarios = [
        item for item in conflict_scenarios
        if item.trait == selected_trait
    ] or [scenario]
    preset_ids = {item.scenario_id for item in conflict_scenarios[:5]}
    scenario_candidates = []
    for item in conflict_scenarios[:5] + trait_scenarios:
        if item.scenario_id not in {candidate.scenario_id for candidate in scenario_candidates}:
            scenario_candidates.append(item)
    scenario_labels = {
        (
            f"[Conflict preset] {item.scenario_id}: {item.prompt[:72]}"
            if item.scenario_id in preset_ids
            else f"{item.scenario_id}: {item.prompt[:72]}"
        ): item
        for item in scenario_candidates[:55]
    }
    selected_label = st.selectbox(
        "Scenario",
        options=list(scenario_labels),
        index=0,
    )
    scenario = replace(
        scenario_labels[selected_label],
        target_direction=target_direction,
    )
    prompt_response, as_pas_response = response_pair_for_scenario(responses, scenario)

    summary_tab, npc_tab, chart_tab, figure_tab, live_tab = st.tabs(
        [
            "Interactive Scores",
            "NPC Comparison",
            "Activation View",
            "Figures",
            "Live Inference",
        ]
    )

    with summary_tab:
        st.subheader("Scenario Card")
        st.caption(f"{scenario.trait.title()} / {scenario.target_direction}")
        st.write(scenario.prompt)

        action_df = pd.DataFrame(scenario.actions)
        if not action_df.empty:
            st.dataframe(action_df, use_container_width=True, hide_index=True)

        scenario_rows = _scenario_condition_rows(
            trait_scores_df,
            scenario.scenario_id,
            scenario.target_direction,
        )
        if not scenario_rows.empty:
            st.subheader("Condition-Level Behavior Selection")
            st.dataframe(
                scenario_rows[
                    [
                        "condition",
                        "loglik_selected_direction",
                        "trait_score",
                        "trait_target_attainment",
                        "loglik_selected_action",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        delta_rows = _condition_delta_rows(
            condition_differences,
            selected_trait,
            target_direction,
        )
        if not delta_rows.empty:
            st.subheader("Final Test Delta vs Baseline")
            st.bar_chart(
                delta_rows.set_index("condition")[
                    ["delta_consistency", "delta_trait_target_attainment"]
                ]
            )

        score_rows = _summary_rows(
            consistency_summary,
            selected_trait,
            target_direction,
        )
        if not score_rows.empty:
            st.subheader("Speech vs Behavior Scores")
            st.dataframe(
                score_rows[
                    [
                        "condition",
                        "mean_bfi_score",
                        "mean_trait_score",
                        "mean_consistency",
                        "mean_trait_target_attainment",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with npc_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("Prompt-only NPC")
            st.write(prompt_response.speech)
            st.code(prompt_response.action or "no action", language="text")
        with right:
            st.subheader("Prompt + AS/PAS NPC")
            st.write(as_pas_response.speech)
            st.code(as_pas_response.action or "no action", language="text")

    with chart_tab:
        st.subheader("Activation Movement")
        points = pd.DataFrame(activation_points(trait_scores))
        st.scatter_chart(points, x="x", y="y", color="label")

    with figure_tab:
        figure_dir = PROJECT_ROOT / FINAL_FIGURE_DIR
        if figure_dir.exists():
            st.subheader("Final Result Figures")
            figure_files = sorted(figure_dir.glob("*.png"))
            figure_choice = st.selectbox(
                "Figure",
                options=[figure.name for figure in figure_files],
            )
            selected_figure = figure_dir / figure_choice
            st.image(str(selected_figure), caption=selected_figure.stem, use_container_width=True)

    with live_tab:
        render_live_inference_tab(get_live_engine, scenarios)


def _read_csv(path: Path, pd):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _scenario_condition_rows(df, scenario_id: str, target_direction: str):
    if df.empty:
        return df
    rows = df[
        (df["scenario_id"].astype(str) == scenario_id)
        & (df["target_direction"].astype(str) == target_direction)
    ].copy()
    if rows.empty:
        return rows
    order = {
        "baseline": 0,
        "one_line_prompt": 1,
        "elaborate_prompt": 2,
        "as_pas_only": 3,
        "elaborate_prompt_as_pas": 4,
    }
    rows["_order"] = rows["condition"].map(order).fillna(99)
    return rows.sort_values("_order").drop(columns=["_order"])


def _condition_delta_rows(df, trait: str, target_direction: str):
    if df.empty:
        return df
    rows = df[
        (df["trait"].astype(str) == trait)
        & (df["target_direction"].astype(str) == target_direction)
    ].copy()
    if rows.empty:
        return rows
    return rows.sort_values("condition")


def _summary_rows(df, trait: str, target_direction: str):
    if df.empty:
        return df
    rows = df[
        (df["trait"].astype(str) == trait)
        & (df["target_direction"].astype(str) == target_direction)
    ].copy()
    if rows.empty:
        return rows
    return rows.sort_values("condition")


if __name__ == "__main__":
    main()
