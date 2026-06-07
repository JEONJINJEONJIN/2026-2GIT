"""Throwaway Streamlit harness for manually testing the slider demo backend.

Run from the repository root:

    streamlit run dev_harness.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.controls.vector_controls import slider_unit_norms
from src.data.trait_dataset import load_trait_scenarios
from src.demo.bigfive_demo_data import CONFLICT_SCENARIO_PRESET_IDS
from src.experiments.v4_bigfive_runner import load_vector_bundle
from src.demo.slider_backend import (
    CONDITIONS,
    DEFAULT_LAYERS,
    DEFAULT_PERSONA,
    DEFAULT_SCENARIO_PATH,
    DEFAULT_VECTOR_PATH,
    generate,
)


OCEAN_TRAITS = (
    ("O", "openness", "Openness"),
    ("C", "conscientiousness", "Conscientiousness"),
    ("E", "extraversion", "Extraversion"),
    ("A", "agreeableness", "Agreeableness"),
    ("N", "neuroticism", "Neuroticism"),
)

CONDITION_ORDER = ("baseline", "elaborate", "as_pas")

PRESETS = {
    "balanced": {
        "openness": 0.5,
        "conscientiousness": 0.5,
        "extraversion": 0.5,
        "agreeableness": 0.5,
        "neuroticism": 0.5,
    },
    "social_explorer": {
        "openness": 1.0,
        "conscientiousness": 0.5,
        "extraversion": 1.0,
        "agreeableness": 0.5,
        "neuroticism": 0.0,
    },
    "anxious_creative": {
        "openness": 1.0,
        "conscientiousness": 0.5,
        "extraversion": 0.0,
        "agreeableness": 0.5,
        "neuroticism": 1.0,
    },
    "reserved_conservative": {
        "openness": 0.0,
        "conscientiousness": 1.0,
        "extraversion": 0.0,
        "agreeableness": 0.5,
        "neuroticism": 0.5,
    },
    "all_extreme": {
        "openness": 1.0,
        "conscientiousness": 1.0,
        "extraversion": 1.0,
        "agreeableness": 1.0,
        "neuroticism": 1.0,
    },
}


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Slider Backend Dev Harness", layout="wide")
    st.title("Slider Backend Dev Harness")
    st.caption("Internal manual test harness. Conditions are intentionally labeled.")

    @st.cache_resource
    def cached_vector_bundle():
        return get_vector_bundle()

    @st.cache_data
    def cached_default_scenario() -> str:
        return get_default_scenario()

    @st.cache_data
    def cached_scenario_presets() -> list[dict]:
        return get_scenario_presets()

    initialize_slider_state(st)
    initialize_scenario_state(st, cached_default_scenario())
    vector_bundle = cached_vector_bundle()
    scenario_presets = cached_scenario_presets()

    with st.sidebar:
        st.header("Persona")
        st.caption("Defaults match backend DEFAULT_PERSONA so prepopulated cache can hit.")

        preset_cols = st.columns(2)
        for index, (name, values) in enumerate(PRESETS.items()):
            with preset_cols[index % 2]:
                if st.button(name, use_container_width=True):
                    set_slider_values(st, values)

        persona = render_sliders(st)
        norm_value, per_layer_norms = composite_norm(vector_bundle, persona)
        render_norm_readout(st, norm_value, per_layer_norms)

        st.divider()
        st.write("Backend contract")
        st.code(
            "generate(persona: dict, scenario: str, condition: str) -> dict\n"
            f"conditions = {sorted(CONDITIONS)}\n"
            "returns keys = text, condition, cache_hit",
            language="python",
        )

    st.subheader("Scenario Presets")
    selected_scenario = st.selectbox(
        "Choose a conflict scenario",
        options=scenario_presets,
        format_func=lambda item: item["label"],
    )
    if st.button("Load selected scenario"):
        st.session_state["scenario_text"] = selected_scenario["prompt"]

    scenario = st.text_area(
        "Scenario",
        height=150,
        key="scenario_text",
        help=(
            "Default matches backend prepopulated cache. For elaborate/as_pas, "
            "the scenario must exactly match a loaded TRAIT scenario prompt."
        ),
    )
    with st.expander("Current scenario metadata", expanded=False):
        current = scenario_metadata_for_prompt(scenario, scenario_presets)
        if current:
            st.write(current)
        else:
            st.write("Custom or non-preset scenario text.")

    if st.button("Run all three", type="primary"):
        results = run_all_conditions(persona, scenario)
        render_results(results)


def initialize_slider_state(st) -> None:
    """Initialize OCEAN sliders from the real backend default persona."""

    for short_key, trait, _label in OCEAN_TRAITS:
        state_key = slider_state_key(short_key)
        if state_key not in st.session_state:
            st.session_state[state_key] = float(DEFAULT_PERSONA.get(trait, 0.5))


def initialize_scenario_state(st, default_scenario: str) -> None:
    """Initialize scenario text area from the backend default scenario."""

    if "scenario_text" not in st.session_state:
        st.session_state["scenario_text"] = default_scenario


def set_slider_values(st, values: dict[str, float]) -> None:
    """Apply a preset to Streamlit session state."""

    for short_key, trait, _label in OCEAN_TRAITS:
        st.session_state[slider_state_key(short_key)] = float(values.get(trait, 0.5))


def render_sliders(st) -> dict[str, float]:
    """Render sidebar sliders and return a full-trait persona dict."""

    persona: dict[str, float] = {}
    for short_key, trait, label in OCEAN_TRAITS:
        persona[trait] = float(
            st.slider(
                f"{short_key} - {label}",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key=slider_state_key(short_key),
            )
        )
    return persona


def slider_state_key(short_key: str) -> str:
    return f"persona_{short_key}"


def run_all_conditions(persona: dict[str, float], scenario: str) -> list[dict]:
    """Call backend generate sequentially and time each condition."""

    rows = []
    for condition in CONDITION_ORDER:
        start = time.perf_counter()
        try:
            result = generate(persona, scenario, condition)
            error = None
        except Exception as exc:  # noqa: BLE001 - this harness must surface backend errors.
            result = {
                "text": "",
                "condition": condition,
                "cache_hit": False,
            }
            error = str(exc)
        latency = time.perf_counter() - start
        text = str(result.get("text", ""))
        display_text = clean_display_text(text)
        rows.append(
            {
                "condition": condition,
                "text": text,
                "display_text": display_text,
                "cache_hit": bool(result.get("cache_hit", False)),
                "latency": latency,
                "word_count": len(display_text.split()),
                "error": error,
            }
        )
    return rows


def render_results(rows: list[dict]) -> None:
    """Render three side-by-side condition outputs plus latency diagnostics."""

    import streamlit as st

    columns = st.columns(3)
    for column, row in zip(columns, rows):
        with column:
            st.subheader(row["condition"])
            if row["error"]:
                st.error(row["error"])
            else:
                st.text_area(
                    "Output",
                    value=row["display_text"],
                    height=260,
                    key=f"output_{row['condition']}_{time.time_ns()}",
                )
            st.caption(
                f"latency={row['latency']:.2f}s | "
                f"cache_hit={row['cache_hit']} | "
                f"words={row['word_count']}"
            )

    total_latency = sum(float(row["latency"]) for row in rows)
    st.metric("Sequential total latency", f"{total_latency:.2f}s")

    by_condition = {row["condition"]: row for row in rows}
    baseline_latency = float(by_condition["baseline"]["latency"])
    elaborate_latency = float(by_condition["elaborate"]["latency"])
    as_pas_latency = float(by_condition["as_pas"]["latency"])
    slower_non_pas = max(baseline_latency, elaborate_latency)
    if slower_non_pas > 0 and as_pas_latency > 1.5 * slower_non_pas:
        st.warning(
            "as_pas latency is >1.5x the slower non-PAS condition "
            "(PAS best-of-N cost likely visible)."
        )
    else:
        st.info("as_pas latency is within 1.5x of the slower non-PAS condition.")


def clean_display_text(text: str) -> str:
    """Remove internal response tags/action IDs for the manual harness display."""

    raw = str(text).strip()
    text = re.sub(r"<Action>.*?</Action>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\[Speech\]", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^\s*short\s+dialogue\s*,?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip().strip('"').strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    if text:
        return text
    if raw:
        return raw
    return "(empty output)"


def composite_norm(vector_bundle, persona: dict[str, float]) -> tuple[float, dict[int, float]]:
    """Return the live pre-alpha composite norm computed by vector_controls."""

    per_layer = slider_unit_norms(
        vector_bundle,
        persona,
        layer_indices=DEFAULT_LAYERS,
        clamp_x=None,
    )
    return max(per_layer.values()), per_layer


def render_norm_readout(st, norm_value: float, per_layer_norms: dict[int, float]) -> None:
    """Display clamp sweep norm zones."""

    st.subheader("Composite Norm")
    st.metric("pre-alpha max layer norm", f"{norm_value:.3f}")
    st.caption(
        "Per layer: "
        + ", ".join(f"L{layer}={value:.3f}" for layer, value in per_layer_norms.items())
    )
    if norm_value < 2.2:
        st.success("<2.2 tested-safe")
    elif norm_value <= 2.6:
        st.warning("2.2-2.6 above sweep")
    else:
        st.error(">2.6 untested worst case")


def get_default_scenario() -> str:
    """Return the exact default scenario used by SliderDemoBackend.prepopulate_defaults."""

    return get_test_scenarios()[0].prompt


def get_scenario_presets() -> list[dict]:
    """Return backend-default scenario plus conflict presets from the dataset."""

    scenarios = get_test_scenarios()
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    presets = [
        {
            "id": "backend_default",
            "label": "[Backend default] " + summarize_prompt(scenarios[0].prompt),
            "scenario_id": scenarios[0].scenario_id,
            "trait": scenarios[0].trait,
            "prompt": scenarios[0].prompt,
        }
    ]
    for scenario_id in CONFLICT_SCENARIO_PRESET_IDS:
        scenario = by_id.get(scenario_id)
        if scenario is None:
            continue
        presets.append(
            {
                "id": scenario_id,
                "label": f"[Conflict] {scenario.trait} | {summarize_prompt(scenario.prompt)}",
                "scenario_id": scenario.scenario_id,
                "trait": scenario.trait,
                "prompt": scenario.prompt,
            }
        )
    return presets


def scenario_metadata_for_prompt(prompt: str, presets: list[dict]) -> dict | None:
    """Return metadata for the currently selected preset prompt."""

    for item in presets:
        if item["prompt"] == prompt:
            return {
                "scenario_id": item["scenario_id"],
                "trait": item["trait"],
                "preset": item["id"],
            }
    return None


def summarize_prompt(prompt: str, max_chars: int = 92) -> str:
    """Shorten a scenario prompt for selectbox labels."""

    prompt = " ".join(str(prompt).split())
    if len(prompt) <= max_chars:
        return prompt
    return prompt[: max_chars - 3].rstrip() + "..."


def get_test_scenarios():
    """Load backend default test scenarios."""

    return load_trait_scenarios(DEFAULT_SCENARIO_PATH, allowed_splits=["test"])


def get_vector_bundle():
    """Load the real vector bundle for live norm readout."""

    return load_vector_bundle(DEFAULT_VECTOR_PATH)


if __name__ == "__main__":
    main()
