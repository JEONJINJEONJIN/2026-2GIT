"""Generate slide-ready figures from V2 activation steering results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "results" / "v2" / "final_analysis_v2"
VECTOR_DIR = ROOT / "results" / "v2" / "vector_diagnostics"
OUT_DIR = ROOT / "results" / "v2" / "slide_assets"

COLORS = {
    "aggressive": "#C44536",
    "cooperative": "#2A9D8F",
    "neutral": "#595959",
    "prompt": "#4C78A8",
    "as": "#F58518",
    "bg": "#FFFFFF",
    "grid": "#E8E8E8",
    "text": "#222222",
}

CONDITION_ORDER = [
    "neutral_baseline",
    "prompt_baseline",
    "as_only_neg",
    "as_only_sep",
    "prompt_as",
]

CONDITION_LABELS = {
    "neutral_baseline": "Neutral",
    "prompt_baseline": "Prompt",
    "as_only_neg": "AS only\nneg",
    "as_only_sep": "AS only\nsep",
    "prompt_as": "Prompt\n+ AS",
}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_alignment_bar() -> None:
    df = pd.read_csv(ANALYSIS_DIR / "alignment_metrics.csv")
    fig, ax = plt.subplots(figsize=(11.5, 6.4))

    x = np.arange(len(CONDITION_ORDER))
    width = 0.34
    personas = ["aggressive", "cooperative"]
    offsets = [-width / 2, width / 2]

    for persona, offset in zip(personas, offsets):
        rows = (
            df[df["persona"] == persona]
            .set_index("condition")
            .loc[CONDITION_ORDER]
            .reset_index()
        )
        values = rows["align_itt"].to_numpy()
        lower = values - rows["align_itt_ci_low"].to_numpy()
        upper = rows["align_itt_ci_high"].to_numpy() - values
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[persona],
            label=persona.capitalize(),
            edgecolor="white",
            linewidth=1.1,
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#333333",
            elinewidth=1,
            capsize=3,
            capthick=1,
            alpha=0.75,
        )
        for xi, yi in zip(x + offset, values):
            ax.text(xi, yi + 0.035, f"{yi:.2f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0.40, color="#333333", linestyle=(0, (4, 3)), linewidth=1.4)
    ax.text(
        len(CONDITION_ORDER) - 0.2,
        0.415,
        "chance = 0.40",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#333333",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Alignment rate")
    ax.set_title("Persona Alignment by Condition")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=1)
    ax.legend(loc="upper left", frameon=False, ncols=2)
    ax.text(
        0.01,
        -0.18,
        "Error bars: Wilson 95% CI. Loglik scoring gives parse_ok = 100%, unknown = 0%.",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, "01_alignment_bar_chart")


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 2.1) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x - width / 2, y - 0.35),
        width,
        0.7,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=color,
        facecolor="#FFFFFF",
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center", fontsize=11, color=COLORS["text"])


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#555555") -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color=color,
        )
    )


def plot_pipeline_flow() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(0.3, 6.55, "Why V2 Changed the Measurement Pipeline", fontsize=19, weight="bold", color=COLORS["text"])

    ax.text(0.8, 5.75, "V1: free-form generation", fontsize=14, weight="bold", color=COLORS["text"])
    draw_box(ax, (1.7, 4.9), "Scenario\nprompt", "#888888")
    draw_box(ax, (4.1, 4.9), "Free-form\nmodel text", "#888888")
    draw_box(ax, (6.5, 4.9), "Parser", "#888888")
    draw_box(ax, (9.2, 4.9), "Action label\nwith unknowns", COLORS["aggressive"], width=2.5)
    arrow(ax, (2.75, 4.9), (3.05, 4.9))
    arrow(ax, (5.15, 4.9), (5.45, 4.9))
    arrow(ax, (7.55, 4.9), (7.95, 4.9))
    ax.text(10.9, 4.9, "parse_ok only\n~42-54%", ha="center", va="center", fontsize=12, color=COLORS["aggressive"])

    ax.text(0.8, 3.15, "V2: closed-set log-likelihood scoring", fontsize=14, weight="bold", color=COLORS["text"])
    draw_box(ax, (1.5, 2.3), "Scenario\nprompt", COLORS["prompt"], width=1.8)
    draw_box(ax, (3.7, 2.3), "5 candidate\nactions", COLORS["prompt"], width=1.8)
    draw_box(ax, (6.1, 2.3), "Score\nlog p(action|prompt)", COLORS["prompt"], width=2.4)
    draw_box(ax, (9.2, 2.3), "Argmax +\nprobabilities", COLORS["cooperative"], width=2.2)
    arrow(ax, (2.4, 2.3), (2.8, 2.3), COLORS["prompt"])
    arrow(ax, (4.6, 2.3), (4.9, 2.3), COLORS["prompt"])
    arrow(ax, (7.3, 2.3), (8.1, 2.3), COLORS["prompt"])
    ax.text(10.9, 2.3, "parse_ok\n100%", ha="center", va="center", fontsize=12, color=COLORS["cooperative"])

    ax.text(
        0.8,
        0.9,
        "Takeaway: V2 removes parser failure, but should be described as forced-choice action-description preference.",
        fontsize=12,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, "03_pipeline_v1_v2_flow")


def plot_vector_design_diagram() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.suptitle("Vector Design Finding", fontsize=19, weight="bold", y=0.98)

    ax = axes[0]
    ax.set_title("Neutral-anchor extraction", fontsize=14, weight="bold")
    ax.set_xlim(-0.2, 1.8)
    ax.set_ylim(-0.6, 1.35)
    ax.axis("off")
    ax.scatter([0], [0], s=65, color="#444444")
    ax.text(0, -0.15, "neutral", ha="center", va="top", fontsize=11)
    arrow(ax, (0, 0), (1.15, 0.78), COLORS["aggressive"])
    arrow(ax, (0, 0), (1.35, 0.42), COLORS["cooperative"])
    ax.text(1.18, 0.86, "v_aggressive", color=COLORS["aggressive"], fontsize=12, weight="bold")
    ax.text(1.36, 0.48, "v_cooperative", color=COLORS["cooperative"], fontsize=12, weight="bold")
    ax.text(
        0.75,
        -0.38,
        "cosine = 0.717\nsame broad direction\n(non-neutral component)",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F7F7F7", ec="#DDDDDD"),
    )

    ax = axes[1]
    ax.set_title("Corrected persona axis", fontsize=14, weight="bold")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.85, 1.2)
    ax.axis("off")
    ax.scatter([0], [0], s=65, color="#444444")
    ax.text(0, -0.15, "origin", ha="center", va="top", fontsize=11)
    arrow(ax, (0, 0), (1.05, 0), COLORS["aggressive"])
    arrow(ax, (0, 0), (-1.05, 0), COLORS["cooperative"])
    ax.text(1.1, 0.08, "+ contrast\naggressive", color=COLORS["aggressive"], fontsize=12, weight="bold", ha="left")
    ax.text(-1.1, 0.08, "- contrast\ncooperative", color=COLORS["cooperative"], fontsize=12, weight="bold", ha="right")
    ax.plot([-1.15, 1.15], [0.28, 0.28], color="#999999", linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0, 0.4, "opposite directions: cosine = -1.000", ha="center", fontsize=12)
    ax.text(
        0,
        -0.58,
        "Orthogonalized vector aligns with contrast:\ncosine = 0.954",
        ha="center",
        va="center",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F7F7F7", ec="#DDDDDD"),
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, "02_vector_design_diagram")


def _term_label(term: str) -> str | None:
    if "[T.as_only_neg]" in term and ":" not in term:
        return "AS only\n(contrast)"
    if "[T.prompt_as]" in term and ":" not in term:
        return "Prompt + AS"
    if "[T.prompt_baseline]" in term and ":" not in term:
        return "Prompt only"
    return None


def plot_forest_plot() -> None:
    df = pd.read_csv(ANALYSIS_DIR / "mixed_effects_fixed_effects.csv")
    rows = []
    for _, row in df.iterrows():
        label = _term_label(str(row["term"]))
        if label:
            rows.append({
                "label": label,
                "or": row["odds_ratio"],
                "low": row["or_ci_low"],
                "high": row["or_ci_high"],
                "p": row["p_value"],
            })
    plot_df = pd.DataFrame(rows)
    order = ["Prompt only", "Prompt + AS", "AS only\n(contrast)"]
    plot_df["_order"] = plot_df["label"].map({label: i for i, label in enumerate(order)})
    plot_df = plot_df.sort_values("_order", ascending=False)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y = np.arange(len(plot_df))
    ax.errorbar(
        plot_df["or"],
        y,
        xerr=np.vstack([plot_df["or"] - plot_df["low"], plot_df["high"] - plot_df["or"]]),
        fmt="o",
        color=COLORS["prompt"],
        ecolor="#555555",
        elinewidth=2,
        capsize=4,
        markersize=8,
    )
    ax.axvline(1.0, color="#333333", linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xlim(0.75, 35)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["label"])
    ax.set_xlabel("Odds ratio vs neutral baseline (log scale)")
    ax.set_title("Mixed-Effects Condition Effects")
    ax.grid(axis="x", color=COLORS["grid"], linewidth=1)

    for yi, (_, row) in zip(y, plot_df.iterrows()):
        ax.text(
            row["high"] * 1.08,
            yi,
            f"OR {row['or']:.2f} [{row['low']:.2f}, {row['high']:.2f}]",
            va="center",
            fontsize=10,
            color="#333333",
        )

    ax.text(
        0.01,
        -0.18,
        "Model: aligned ~ condition * persona + (1 | scenario_id). AS neg/sep are identical contrast-vector conditions.",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
    )
    fig.tight_layout()
    save_figure(fig, "04_mixed_effects_forest_plot")


def write_index() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = """# Slide Assets

Generated figures for the activation steering presentation.

## Files

1. `01_alignment_bar_chart.png`
   - Main result: alignment rate by condition and persona.
   - Includes 0.40 chance baseline and Wilson 95% CI.

2. `02_vector_design_diagram.png`
   - Explains the vector-design finding.
   - Neutral-anchor vectors point in a similar direction, while contrast/orthogonalized vectors recover the persona axis.

3. `03_pipeline_v1_v2_flow.png`
   - Shows why V2 moved from free-form parsing to closed-set loglik scoring.

4. `04_mixed_effects_forest_plot.png`
   - Shows mixed-effects odds ratios vs neutral baseline.
   - Collapses duplicate AS neg/sep interpretation into a single AS-only contrast row.

SVG versions are also saved for editing in slide tools.
"""
    (OUT_DIR / "README.md").write_text(index, encoding="utf-8")


def main() -> None:
    setup_style()
    plot_alignment_bar()
    plot_vector_design_diagram()
    plot_pipeline_flow()
    plot_forest_plot()
    write_index()
    print(f"Saved slide assets to: {OUT_DIR}")


if __name__ == "__main__":
    main()
