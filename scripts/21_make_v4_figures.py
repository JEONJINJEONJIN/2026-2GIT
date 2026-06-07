"""Generate publication/slide figures for v4 Big Five final metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TRAIT_ORDER = [
    "agreeableness",
    "conscientiousness",
    "extraversion",
    "neuroticism",
    "openness",
]

TRAIT_LABELS = {
    "agreeableness": "Agree.",
    "conscientiousness": "Consc.",
    "extraversion": "Extra.",
    "neuroticism": "Neuro.",
    "openness": "Open.",
}

CONDITION_ORDER = [
    "baseline",
    "one_line_prompt",
    "elaborate_prompt",
    "as_pas_only",
    "elaborate_prompt_as_pas",
]

CONDITION_LABELS = {
    "baseline": "Baseline",
    "one_line_prompt": "One-line\nprompt",
    "elaborate_prompt": "Elaborate\nprompt",
    "as_pas_only": "AS+PAS\nonly",
    "elaborate_prompt_as_pas": "Elaborate\n+ AS+PAS",
}

COLORS = {
    "high": "#2A9D8F",
    "low": "#C44536",
    "grid": "#E7E3DD",
    "text": "#202124",
    "zero": "#454545",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.edgecolor": "#333333",
            "text.color": COLORS["text"],
            "axes.labelcolor": COLORS["text"],
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _ordered_trait_positions() -> tuple[np.ndarray, list[str]]:
    x = np.arange(len(TRAIT_ORDER))
    labels = [TRAIT_LABELS[trait] for trait in TRAIT_ORDER]
    return x, labels


def plot_as_pas_delta_bars(differences: pd.DataFrame, output_dir: Path) -> None:
    subset = differences[differences["condition"] == "as_pas_only"].copy()
    x, labels = _ordered_trait_positions()
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for target, offset in [("high", -width / 2), ("low", width / 2)]:
        rows = (
            subset[subset["target_direction"] == target]
            .set_index("trait")
            .loc[TRAIT_ORDER]
        )
        values = rows["delta_consistency"].to_numpy()
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[target],
            edgecolor="white",
            linewidth=1,
            label=target.capitalize(),
        )

    ax.axhline(0, color=COLORS["zero"], linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta speech-action consistency")
    ax.set_title("AS+PAS-only Effect on Consistency")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=1)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    save_figure(fig, output_dir, "01_as_pas_consistency_delta")


def plot_as_pas_trait_target_bars(differences: pd.DataFrame, output_dir: Path) -> None:
    subset = differences[differences["condition"] == "as_pas_only"].copy()
    x, labels = _ordered_trait_positions()
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    for target, offset in [("high", -width / 2), ("low", width / 2)]:
        rows = (
            subset[subset["target_direction"] == target]
            .set_index("trait")
            .loc[TRAIT_ORDER]
        )
        values = rows["delta_trait_target_attainment"].to_numpy()
        ax.bar(
            x + offset,
            values,
            width=width,
            color=COLORS[target],
            edgecolor="white",
            linewidth=1,
            label=target.capitalize(),
        )

    ax.axhline(0, color=COLORS["zero"], linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta TRAIT target attainment")
    ax.set_title("AS+PAS-only Effect on Behavior Target Attainment")
    ax.grid(axis="y", color=COLORS["grid"], linewidth=1)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    save_figure(fig, output_dir, "02_as_pas_trait_target_delta")


def plot_condition_heatmap(differences: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    row_labels = []
    for trait in TRAIT_ORDER:
        for target in ("high", "low"):
            row = []
            for condition in CONDITION_ORDER[1:]:
                match = differences[
                    (differences["trait"] == trait)
                    & (differences["target_direction"] == target)
                    & (differences["condition"] == condition)
                ]
                row.append(float(match.iloc[0]["delta_consistency"]))
            rows.append(row)
            row_labels.append(f"{TRAIT_LABELS[trait]} {target}")

    matrix = np.array(rows)
    limit = max(abs(matrix.min()), abs(matrix.max()))
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(CONDITION_ORDER) - 1))
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER[1:]])
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title("Consistency Delta by Condition")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.02f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111" if abs(matrix[i, j]) < limit * 0.55 else "white",
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Delta vs baseline")
    fig.tight_layout()
    save_figure(fig, output_dir, "03_condition_consistency_heatmap")


def plot_bfi_trait_scatter(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    markers = {"high": "o", "low": "s"}
    color_by_condition = {
        "baseline": "#4C78A8",
        "one_line_prompt": "#F58518",
        "elaborate_prompt": "#54A24B",
        "as_pas_only": "#B279A2",
        "elaborate_prompt_as_pas": "#E45756",
    }

    for condition in CONDITION_ORDER:
        for target in ("high", "low"):
            rows = summary[
                (summary["condition"] == condition)
                & (summary["target_direction"] == target)
            ]
            ax.scatter(
                rows["mean_bfi_score"],
                rows["mean_trait_score"],
                s=64,
                marker=markers[target],
                color=color_by_condition[condition],
                alpha=0.82,
                edgecolor="white",
                linewidth=0.7,
                label=f"{CONDITION_LABELS[condition].replace(chr(10), ' ')} {target}",
            )

    ax.plot([0, 1], [0, 1], color="#333333", linestyle=(0, (4, 3)), linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("BFI score (speech)")
    ax.set_ylabel("TRAIT score (behavior)")
    ax.set_title("Speech vs Behavior Trait Scores")
    ax.grid(color=COLORS["grid"], linewidth=1)
    ax.legend(frameon=False, fontsize=7, ncols=2, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    save_figure(fig, output_dir, "04_bfi_trait_scatter")


def plot_pas_agreement(pas_summary: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    labels = []
    for trait in TRAIT_ORDER:
        for target in ("high", "low"):
            values = []
            for condition in ("as_pas_only", "elaborate_prompt_as_pas"):
                match = pas_summary[
                    (pas_summary["trait"] == trait)
                    & (pas_summary["target_direction"] == target)
                    & (pas_summary["condition"] == condition)
                ]
                values.append(float(match.iloc[0]["pas_loglik_agreement_rate"]))
            rows.append(values)
            labels.append(f"{TRAIT_LABELS[trait]} {target}")

    matrix = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.8, 6.4))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["AS+PAS only", "Elaborate\n+ AS+PAS"])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("PAS / Loglik Agreement")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Agreement rate")
    fig.tight_layout()
    save_figure(fig, output_dir, "05_pas_loglik_agreement")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate v4 Big Five final figures.")
    parser.add_argument(
        "--metrics_dir",
        default="results/v4_bigfive/final_metrics",
        help="Directory containing analyzed v4 final metric CSVs.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/v4_bigfive/final_figures",
        help="Directory for PNG/SVG figure output.",
    )
    args = parser.parse_args()

    setup_style()
    metrics_dir = PROJECT_ROOT / args.metrics_dir
    output_dir = PROJECT_ROOT / args.output_dir

    differences = pd.read_csv(metrics_dir / "condition_differences.csv")
    summary = pd.read_csv(metrics_dir / "consistency_summary.csv")
    pas_summary = pd.read_csv(metrics_dir / "pas_agreement_summary.csv")

    plot_as_pas_delta_bars(differences, output_dir)
    plot_as_pas_trait_target_bars(differences, output_dir)
    plot_condition_heatmap(differences, output_dir)
    plot_bfi_trait_scatter(summary, output_dir)
    plot_pas_agreement(pas_summary, output_dir)

    print(f"Wrote figures to: {output_dir}")


if __name__ == "__main__":
    main()
