"""Visualization script.

Loads metrics from results/metrics/ and generates:
  - Layer heatmap of steering vector norms
  - Condition comparison bar chart (persona alignment)
  - Alpha sweep plot
  - Quantization similarity plot
Saves figures to results/figures/.
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def plot_layer_heatmap(vectors_dir: Path, output_dir: Path) -> None:
    """Generate a heatmap of steering vector norms across layers.

    Looks for .pt files in vectors_dir containing layer->tensor dicts.
    """
    vector_files = sorted(vectors_dir.glob("steering_vectors_*.pt"))
    if not vector_files:
        logger.warning("No steering vector files found in %s", vectors_dir)
        return

    fig, ax = plt.subplots(figsize=(14, 4))

    for vec_file in vector_files:
        label = vec_file.stem.replace("steering_vectors_", "")
        vectors = torch.load(vec_file, map_location="cpu")
        layers = sorted(vectors.keys())
        norms = [vectors[l].norm().item() for l in layers]
        ax.plot(layers, norms, marker="o", markersize=3, label=label)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Steering Vector L2 Norm")
    ax.set_title("Steering Vector Norms by Layer")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Also create a proper heatmap if multiple quantization modes
    if len(vector_files) > 1:
        all_norms = []
        labels = []
        for vec_file in vector_files:
            label = vec_file.stem.replace("steering_vectors_", "")
            labels.append(label)
            vectors = torch.load(vec_file, map_location="cpu")
            layers = sorted(vectors.keys())
            norms = [vectors[l].norm().item() for l in layers]
            all_norms.append(norms)

        fig_hm, ax_hm = plt.subplots(figsize=(16, 3))
        im = ax_hm.imshow(all_norms, aspect="auto", cmap="viridis")
        ax_hm.set_xticks(range(len(layers)))
        ax_hm.set_xticklabels(layers, fontsize=6)
        ax_hm.set_yticks(range(len(labels)))
        ax_hm.set_yticklabels(labels)
        ax_hm.set_xlabel("Layer Index")
        ax_hm.set_title("Steering Vector Norms Heatmap")
        plt.colorbar(im, ax=ax_hm, label="L2 Norm")
        fig_hm.tight_layout()
        fig_hm.savefig(output_dir / "layer_heatmap.png", dpi=150)
        plt.close(fig_hm)
        logger.info("Saved layer_heatmap.png")

    fig.tight_layout()
    fig.savefig(output_dir / "layer_norms.png", dpi=150)
    plt.close(fig)
    logger.info("Saved layer_norms.png")


def plot_condition_comparison(metrics_dir: Path, output_dir: Path) -> None:
    """Generate a bar chart comparing persona alignment across conditions."""
    alignment_path = metrics_dir / "persona_alignment.csv"
    if not alignment_path.exists():
        logger.warning("persona_alignment.csv not found in %s", metrics_dir)
        return

    df = pd.read_csv(alignment_path)

    conditions = df["condition"].unique()
    personas = df["persona"].unique()

    x = np.arange(len(conditions))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, persona in enumerate(personas):
        persona_data = df[df["persona"] == persona]
        rates = []
        for cond in conditions:
            row = persona_data[persona_data["condition"] == cond]
            rate = row["alignment_rate"].values[0] if len(row) > 0 else 0.0
            rates.append(rate)
        offset = (i - len(personas) / 2 + 0.5) * width
        bars = ax.bar(x + offset, rates, width, label=persona)
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rate:.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xlabel("Condition")
    ax.set_ylabel("Persona Alignment Rate")
    ax.set_title("Persona Alignment by Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=15, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "condition_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved condition_comparison.png")


def plot_alpha_sweep(metrics_dir: Path, output_dir: Path) -> None:
    """Generate an alpha sweep plot if alpha sweep data exists.

    Looks for files named alpha_sweep_*.csv or a single alpha_sweep.csv.
    Falls back to creating a placeholder if no data is available.
    """
    sweep_files = sorted(metrics_dir.glob("alpha_sweep*.csv"))
    if not sweep_files:
        logger.warning(
            "No alpha sweep data found in %s. "
            "Creating placeholder plot.", metrics_dir
        )
        # Create placeholder with expected format
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel("Alpha (Steering Strength)")
        ax.set_ylabel("Persona Alignment Rate")
        ax.set_title("Alpha Sweep (No Data Available)")
        ax.text(
            0.5, 0.5, "Run alpha sweep experiment\nto populate this plot",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="gray",
        )
        fig.tight_layout()
        fig.savefig(output_dir / "alpha_sweep.png", dpi=150)
        plt.close(fig)
        logger.info("Saved alpha_sweep.png (placeholder)")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for sweep_file in sweep_files:
        df = pd.read_csv(sweep_file)
        if "alpha" in df.columns and "alignment_rate" in df.columns:
            for persona in df["persona"].unique() if "persona" in df.columns else ["all"]:
                if "persona" in df.columns:
                    subset = df[df["persona"] == persona]
                else:
                    subset = df
                ax.plot(
                    subset["alpha"], subset["alignment_rate"],
                    marker="o", label=f"{sweep_file.stem} - {persona}",
                )

    ax.set_xlabel("Alpha (Steering Strength)")
    ax.set_ylabel("Persona Alignment Rate")
    ax.set_title("Effect of Alpha on Persona Alignment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "alpha_sweep.png", dpi=150)
    plt.close(fig)
    logger.info("Saved alpha_sweep.png")


def plot_quantization_similarity(vectors_dir: Path, output_dir: Path) -> None:
    """Plot cosine similarity between fp16 and 4-bit steering vectors.

    Compares vectors from steering_vectors_fp16.pt and
    steering_vectors_4bit.pt, if both exist.
    """
    fp16_path = vectors_dir / "steering_vectors_fp16.pt"
    q4_path = vectors_dir / "steering_vectors_4bit.pt"

    if not fp16_path.exists() or not q4_path.exists():
        logger.warning(
            "Need both fp16 and 4bit vector files for quantization similarity plot. "
            "Creating placeholder."
        )
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Quantization Similarity: fp16 vs 4-bit (No Data)")
        ax.text(
            0.5, 0.5,
            "Extract vectors with both fp16 and 4bit\nto populate this plot",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="gray",
        )
        fig.tight_layout()
        fig.savefig(output_dir / "quantization_similarity.png", dpi=150)
        plt.close(fig)
        logger.info("Saved quantization_similarity.png (placeholder)")
        return

    vectors_fp16 = torch.load(fp16_path, map_location="cpu")
    vectors_4bit = torch.load(q4_path, map_location="cpu")

    common_layers = sorted(set(vectors_fp16.keys()) & set(vectors_4bit.keys()))
    similarities = []
    for layer in common_layers:
        v1 = vectors_fp16[layer].float()
        v2 = vectors_4bit[layer].float()
        cos_sim = torch.nn.functional.cosine_similarity(
            v1.unsqueeze(0), v2.unsqueeze(0)
        ).item()
        similarities.append(cos_sim)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(common_layers)), similarities, color="steelblue", alpha=0.8)
    ax.set_xticks(range(len(common_layers)))
    ax.set_xticklabels(common_layers, fontsize=7)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Steering Vector Similarity: fp16 vs 4-bit")
    ax.axhline(y=0.9, color="red", linestyle="--", alpha=0.5, label="0.9 threshold")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "quantization_similarity.png", dpi=150)
    plt.close(fig)
    logger.info("Saved quantization_similarity.png")

    # Log summary
    mean_sim = np.mean(similarities)
    min_sim = np.min(similarities)
    logger.info(
        "Quantization similarity: mean=%.4f, min=%.4f", mean_sim, min_sim
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualization plots from metrics."
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default=None,
        help="Directory with metric CSV files (default: results/metrics/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save figures (default: results/figures/).",
    )
    parser.add_argument(
        "--vectors_dir",
        type=str,
        default=None,
        help="Directory with steering vector .pt files (default: results/vectors/).",
    )
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir) if args.metrics_dir else (
        PROJECT_ROOT / "results" / "metrics"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "figures"
    )
    vectors_dir = Path(args.vectors_dir) if args.vectors_dir else (
        PROJECT_ROOT / "results" / "vectors"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualizations...")

    # Layer heatmap
    plot_layer_heatmap(vectors_dir, output_dir)

    # Condition comparison bar chart
    plot_condition_comparison(metrics_dir, output_dir)

    # Alpha sweep plot
    plot_alpha_sweep(metrics_dir, output_dir)

    # Quantization similarity plot
    plot_quantization_similarity(vectors_dir, output_dir)

    logger.info("All figures saved to: %s", output_dir)


if __name__ == "__main__":
    main()
