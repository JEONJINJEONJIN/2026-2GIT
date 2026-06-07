# v4 Big Five — Live Inference Tab Implementation Plan

**Target executor:** Codex (autonomous coding agent)
**Owner:** Jin
**Last updated:** 2026-05-25, Asia/Seoul
**Scope:** Add a real-time inference tab to the existing Streamlit demo so that the v4 Big Five system satisfies the course requirement of *interactive system development*, not just *precomputed result viewer*.

---

## 1. Goal

Extend `src/demo/bigfive_app.py` with a new tab — **"Live Inference"** — where the user can:

1. Pick a Big Five trait and target direction.
2. Pick a scenario from `data/trait_bigfive/scenarios.jsonl`.
3. Adjust `alpha` and `layers` controls for AS+PAS.
4. Press a button and watch the model generate two responses side-by-side:
   - **Left:** prompt-only (no AS/PAS hooks).
   - **Right:** elaborate_prompt + AS+PAS hooks active.

This converts the demo from a CSV explorer into a working interactive system that actually invokes the LLM at request time, while keeping the precomputed analysis tabs untouched.

---

## 2. Non-Goals

Do **not** do any of the following in this task:

- Do not modify the existing precomputed viewer tabs (`condition comparison`, `figure viewer`, etc.).
- Do not change anything under `results/v4_bigfive/`.
- Do not re-run the full v4 experiment.
- Do not introduce new conditions or metrics.
- Do not change the BFI scoring path or TRAIT log-likelihood scoring path.
- Do not implement AS-only / PAS-only ablations (that is a separate workstream).
- Do not write the response into any persistent results CSV.

The output of this task is a **demo-only feature**. It must not bleed into the experimental codepath.

---

## 3. Why This Matters (Context for Codex)

The course assignment is "develop one interactive system." The current `bigfive_app.py` is honest about being a viewer over precomputed CSV/PNG files. A live inference tab makes the system genuinely interactive at the model level. The precomputed results remain the quantitative evidence; the live tab becomes the demonstration surface during the presentation.

The presenter needs to be able to say:
> "Watch this — I move the slider, I press Generate, and the model produces a new response with AS+PAS active. Compare it to the prompt-only version on the left."

That sentence must be true at presentation time. Every design choice in this plan serves that sentence.

---

## 4. Existing Assets to Reuse

Codex must reuse the following modules. Do not reimplement them.

| File | Purpose | Reuse for |
|---|---|---|
| `src/models/loader.py` | Loads Qwen model + tokenizer | One-time model load in the live tab |
| `src/controls/vector_controls.py` | AS hooks (vector injection at chosen layers) | Toggling AS on/off during generation |
| `src/projection/selector.py` | PAS cosine-similarity action selection | PAS scoring of generated text against `v_persona` |
| `src/generation/prompt_builder.py` | Builds elaborate / one-line / baseline prompts | Build the prompt strings for both columns |
| `src/data/trait_dataset.py` | TRAIT scenario loading | Loading `scenarios.jsonl` |
| `data/trait_bigfive/contrastive_pairs/` | High/low contrastive pairs per trait | Extracting / caching `v_persona` |
| `src/experiments/v4_bigfive_runner.py` | Reference for hook setup, generation kwargs | Read this to match the experimental codepath exactly |

**Critical:** the live tab must use the *same* `v_persona`, hook layers, and generation config as the experimental runner. Otherwise the demo shows behavior that does not match the reported results.

---

## 5. Architecture

### 5.1 New module: `src/demo/live_inference.py`

A single class `LiveInferenceEngine` encapsulates model state and generation.

```python
class LiveInferenceEngine:
    def __init__(self, model_name: str, device: str = "cuda"):
        """Load model + tokenizer once. Heavy."""

    @lru_cache(maxsize=None)
    def get_v_persona(self, trait: str, direction: str) -> torch.Tensor:
        """Extract or load cached v_persona for (trait, direction).
        Must match the extraction method used in the v4 final run."""

    def generate(
        self,
        scenario_text: str,
        trait: str,
        direction: str,
        mode: Literal["prompt_only", "as_pas"],
        alpha: float = 4.0,
        layers: list[int] = (18, 21, 24),
        max_new_tokens: int = 200,
        temperature: float = 0.7,
    ) -> GenerationResult:
        """Generate one response under the requested mode.
        Returns text + diagnostics (PAS score, hooked layers, time elapsed)."""
```

`GenerationResult` is a dataclass:

```python
@dataclass
class GenerationResult:
    text: str
    mode: str
    elapsed_sec: float
    pas_score: float | None       # cosine similarity vs v_persona, PAS-style
    n_tokens: int
    layers_hooked: list[int]
```

### 5.2 New module: `src/demo/live_inference_tab.py`

Streamlit rendering only. No model logic. Imports `LiveInferenceEngine` and renders the UI.

```python
def render_live_inference_tab(engine: LiveInferenceEngine) -> None:
    """Render the Live Inference tab. Pure UI; engine is injected."""
```

### 5.3 Integration into `bigfive_app.py`

Add the live tab alongside existing tabs. The model engine is constructed once via `st.cache_resource` and passed to the tab renderer.

```python
@st.cache_resource
def get_engine() -> LiveInferenceEngine:
    return LiveInferenceEngine(model_name=CFG.model_name)

# Inside main():
tabs = st.tabs([..., "Live Inference"])
with tabs[-1]:
    render_live_inference_tab(get_engine())
```

`st.cache_resource` ensures the model loads exactly once per Streamlit process.

---

## 6. UI Layout (Live Inference Tab)

Top: a short markdown paragraph explaining that this tab runs the model live, unlike other tabs.

Controls (single column, top):

| Control | Type | Options / Range | Default |
|---|---|---|---|
| Trait | radio | Agreeableness, Conscientiousness, Extraversion, Neuroticism, Openness | Extraversion |
| Direction | radio | high, low | high |
| Alpha | slider | 0.0 – 8.0, step 0.5 | 4.0 |
| Layers | multiselect | 0..N-1 (depends on model) | [18, 21, 24] |
| Scenario | selectbox | titles from `scenarios.jsonl` filtered by selected trait | first match |
| Generate | button | — | — |

Below the button, two side-by-side columns:

- **Left column ("Prompt only")**: shows the elaborate prompt used, then the generated response, then a small diagnostics line (tokens, elapsed sec).
- **Right column ("Elaborate prompt + AS+PAS")**: shows the same prompt, the generated response, diagnostics line, and PAS score.

Below both columns, an expandable "Debug info" section with:
- selected v_persona shape and norm
- layers hooked
- generation kwargs

---

## 7. Data Flow

1. **App start**
   - Model + tokenizer loaded via `st.cache_resource`.
   - `v_persona` for all 10 (trait, direction) combinations precomputed and cached in memory. This costs one extra startup hit but eliminates latency on every Generate.
2. **User clicks Generate**
   - Build elaborate prompt via `prompt_builder` for the selected (trait, direction).
   - Call `engine.generate(..., mode="prompt_only")` → result A.
   - Call `engine.generate(..., mode="as_pas")` → result B.
   - Render both side-by-side.
3. **Subsequent clicks** reuse the cached engine and v_persona; only the forward passes are new work.

---

## 8. Hook Management Rules

This is the highest-risk area. Read carefully.

- The AS hooks **must be installed before** the `as_pas` generation call and **removed immediately after**, even on exception. Use a context manager.
- The `prompt_only` generation call **must not** have hooks installed. Verify by running it first.
- Hooks must target the same layers used in the final experiment (`[18, 21, 24]` by default; configurable via UI).
- Vector injection magnitude: `h' = h + alpha * v_persona` for each hooked layer. Match the formula in `src/controls/vector_controls.py` exactly.

Implement a context manager in `live_inference.py`:

```python
@contextmanager
def as_hooks(model, layers, v_persona, alpha):
    handles = install_hooks(model, layers, v_persona, alpha)
    try:
        yield
    finally:
        for h in handles:
            h.remove()
```

---

## 9. Caching Strategy

| Object | Cache method | Reason |
|---|---|---|
| `LiveInferenceEngine` instance | `st.cache_resource` | Model is 9B+, must load once per process |
| `v_persona` per (trait, direction) | in-class `lru_cache` | Cheap to recompute but no reason to |
| Generated outputs | **no caching** | Each generate must actually run the model |

Do not cache generated outputs. The whole point of the tab is that generation is live.

---

## 10. Implementation Phases

Implement in order. Do not skip ahead.

### Phase 1 — Engine skeleton
- Create `src/demo/live_inference.py` with the class skeleton.
- Implement `__init__` (load model + tokenizer via existing `loader.py`).
- Implement `get_v_persona` using the existing contrastive pair extraction logic.
- Add a `__main__` block that runs a smoke test: load model, extract one v_persona, generate one prompt-only response, generate one AS+PAS response, print both.
- **Acceptance:** running `python -m src.demo.live_inference` prints two responses and does not crash.

### Phase 2 — Tab UI
- Create `src/demo/live_inference_tab.py` with `render_live_inference_tab(engine)`.
- Implement all controls listed in section 6.
- Wire the Generate button to call `engine.generate` twice.
- Display results in two columns.
- **Acceptance:** launching the Streamlit app and opening the new tab shows the controls and generates real text on button press.

### Phase 3 — Diagnostics and polish
- Add elapsed-time display under each response.
- Add PAS score under the AS+PAS column.
- Add the "Debug info" expander.
- Show a `st.spinner("Generating...")` during the call.
- **Acceptance:** every UI element in section 6 is present and functional.

### Phase 4 — Hook safety pass
- Add an explicit test: run AS+PAS, then run prompt-only, then run AS+PAS again. Confirm that the prompt-only call between is not affected by leftover hooks.
- Add an assertion in `engine.generate` that no AS hooks are present when `mode="prompt_only"`.
- **Acceptance:** alternating generations produce visibly distinct outputs in the expected direction.

### Phase 5 — Runbook update
- Append a section to `docs/v4_bigfive_runbook.md` titled "Live Inference Demo" with launch instructions, expected first-load time, and a fallback note if VRAM is exhausted.
- **Acceptance:** a fresh reader can launch the demo using only the runbook.

---

## 11. Acceptance Criteria (Whole Task)

A reviewer must be able to verify all of the following.

1. Launching `streamlit run src/demo/bigfive_app.py` opens the app.
2. The new tab "Live Inference" exists alongside the original tabs.
3. The original tabs are functionally unchanged.
4. Initial model load completes within a reasonable time (target: under 60 seconds on RTX 4080 Super).
5. After load, each Generate click returns both responses in under 20 seconds total (target: under 10 seconds each).
6. AS+PAS and prompt-only outputs are produced by independent forward passes; they are not preloaded.
7. The AS+PAS output noticeably differs from prompt-only for at least 8 of the 10 (trait, direction) combinations during manual testing.
8. No exception leaves AS hooks installed (verified by Phase 4 test).
9. `docs/v4_bigfive_runbook.md` contains a "Live Inference Demo" section.

---

## 12. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| VRAM exhaustion on RTX 4080 Super (16GB) with Qwen 9B + hooks | High | Use the same quantization / dtype as the final experiment; document a fallback to Qwen 4B in the runbook |
| Streamlit reruns triggering re-load | High | Strict `st.cache_resource` usage; never put model load inside a regular function |
| AS hook leak across calls | High | Context manager + Phase 4 verification test |
| v_persona mismatch with experimental runner | Medium | Extract via the same script used for the final run; add a checksum comparison in the smoke test |
| Generation too slow for live demo | Medium | Cap `max_new_tokens` at 200 by default; expose as a control if needed |
| User selects a scenario unrelated to selected trait | Low | Filter scenario dropdown by selected trait |

---

## 13. Open Decisions (Flag to Jin, Do Not Decide Alone)

Codex must surface these as questions before finalizing, not silently guess:

1. **PAS scoring in the live tab**: should the PAS score be displayed as a cosine similarity number only, or also as a small bar chart showing both candidates? Default for now: number only.
2. **Generation determinism**: use temperature 0.7 with sampling (more demo-friendly variety) or greedy decoding (more reproducible)? Default for now: temperature 0.7.
3. **Scenario filtering**: should the scenario dropdown be filtered to scenarios where the selected trait is the labeled axis, or show all scenarios? Default for now: filter by trait.

---

## 14. Out-of-Scope Reminders (Do Not Drift)

Codex tends to over-extend. Stop at the boundary.

- Do **not** add an "evaluation" button that scores the live response against BFI. That is a different feature.
- Do **not** persist live responses to disk.
- Do **not** add multi-turn dialogue. Single-turn only.
- Do **not** add a "compare two AS+PAS configs" view. Only prompt-only vs AS+PAS.
- Do **not** refactor `vector_controls.py` or `selector.py` even if the code looks improvable.

---

## 15. Deliverables Checklist

- [ ] `src/demo/live_inference.py` — engine class
- [ ] `src/demo/live_inference_tab.py` — Streamlit UI
- [ ] `src/demo/bigfive_app.py` — updated to include the new tab
- [ ] `docs/v4_bigfive_runbook.md` — appended "Live Inference Demo" section
- [ ] Smoke test passes: `python -m src.demo.live_inference`
- [ ] Manual UI test passes: model loads, generation works, both columns populate, alternating runs are clean
- [ ] All acceptance criteria in section 11 verified

End of plan.
