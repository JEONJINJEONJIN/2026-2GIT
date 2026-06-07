"""Streamlit tab for live v4 Big Five inference."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from src.data.trait_dataset import TraitAction, TraitScenario
from src.demo.bigfive_demo_data import BIG_FIVE_TRAITS, DemoScenario
from src.demo.live_inference import GenerationResult, filter_scenarios_by_trait


DEFAULT_LAYERS = [18, 21, 24]


def render_live_inference_tab(
    engine_factory: Callable[[], object],
    scenarios: Iterable[DemoScenario | TraitScenario],
) -> None:
    """Render the live inference Streamlit tab."""

    import streamlit as st

    live_scenarios = [coerce_trait_scenario(scenario) for scenario in scenarios]
    st.subheader("Live Inference")
    st.caption(
        "This tab runs the model at request time. The other tabs show precomputed "
        "experiment outputs."
    )

    trait = st.radio(
        "Trait",
        options=list(BIG_FIVE_TRAITS),
        index=list(BIG_FIVE_TRAITS).index("extraversion"),
        format_func=lambda value: value.replace("_", " ").title(),
        horizontal=True,
    )
    direction = st.radio("Direction", options=["high", "low"], horizontal=True)
    alpha = st.slider("Alpha", min_value=0.0, max_value=8.0, value=4.0, step=0.5)
    layers = st.multiselect(
        "Layers",
        options=list(range(36)),
        default=DEFAULT_LAYERS,
        help="Default layers match the final v4 experiment.",
    )
    max_new_tokens = st.slider(
        "Max new tokens",
        min_value=32,
        max_value=200,
        value=96,
        step=16,
    )

    trait_scenarios = filter_scenarios_by_trait(live_scenarios, trait)
    if not trait_scenarios:
        st.warning(f"No scenarios available for trait: {trait}")
        return
    scenario_lookup = {
        f"{scenario.scenario_id}: {scenario.prompt[:90]}": scenario
        for scenario in trait_scenarios[:100]
    }
    selected_label = st.selectbox("Scenario", options=list(scenario_lookup))
    scenario = scenario_lookup[selected_label]

    st.markdown("**Scenario**")
    st.write(scenario.prompt)
    st.dataframe(
        [
            {
                "id": action.id,
                "text": action.text,
                "trait_direction": action.trait_direction,
            }
            for action in scenario.actions
        ],
        use_container_width=True,
        hide_index=True,
    )

    if not layers:
        st.warning("Select at least one layer before generating.")
        return

    if st.button("Generate live responses", type="primary"):
        try:
            with st.spinner("Loading model and generating live responses..."):
                engine = engine_factory()
                prompt_result = engine.generate(
                    scenario=scenario,
                    trait=trait,
                    direction=direction,
                    mode="prompt_only",
                    alpha=alpha,
                    layers=layers,
                    max_new_tokens=max_new_tokens,
                )
                as_pas_result = engine.generate(
                    scenario=scenario,
                    trait=trait,
                    direction=direction,
                    mode="as_pas",
                    alpha=alpha,
                    layers=layers,
                    max_new_tokens=max_new_tokens,
                )
        except Exception as exc:
            st.error(f"Live inference failed: {exc}")
            return

        left, right = st.columns(2)
        with left:
            _render_result("Prompt only", prompt_result)
        with right:
            _render_result("Elaborate prompt + AS", as_pas_result)

        with st.expander("Debug info"):
            st.write(
                {
                    "model_name": getattr(engine, "model_name", "unknown"),
                    "vector_path": str(getattr(engine, "vector_path", "")),
                    "trait": trait,
                    "direction": direction,
                    "alpha": alpha,
                    "layers": layers,
                    "scenario_id": scenario.scenario_id,
                    "prompt_only_action": prompt_result.parsed_action_id,
                    "as_action": as_pas_result.parsed_action_id,
                    "as_pas_top_action": as_pas_result.pas_top_action_id,
                    "pas_score_fallback": as_pas_result.pas_used_fallback,
                    "max_new_tokens": max_new_tokens,
                    "generation": "greedy",
                }
            )


def _render_result(title: str, result: GenerationResult) -> None:
    import streamlit as st

    st.markdown(f"### {title}")
    st.code(result.text.strip() or "(empty output)", language="text")
    st.caption(
        f"tokens={result.n_tokens} | elapsed={result.elapsed_sec:.2f}s | "
        f"action={result.parsed_action_id or 'unparsed'}"
    )
    if result.pas_score is not None:
        st.metric("PAS cosine", f"{result.pas_score:.3f}")


def coerce_trait_scenario(scenario: DemoScenario | TraitScenario) -> TraitScenario:
    """Convert demo scenarios to the experiment scenario dataclass."""

    if isinstance(scenario, TraitScenario):
        return scenario
    return TraitScenario(
        scenario_id=scenario.scenario_id,
        trait=scenario.trait,
        prompt=scenario.prompt,
        split="test",
        actions=tuple(
            TraitAction(
                id=str(action.get("id", "")),
                text=str(action.get("text") or action.get("description") or ""),
                trait_direction=str(action.get("trait_direction", "")).lower(),
            )
            for action in scenario.actions
        ),
    )
