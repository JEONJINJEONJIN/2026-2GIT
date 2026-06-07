"""Run alpha/layer sweeps for v2 activation-steering evaluation."""

from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def alpha_slug(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def run_command(command: list[str], cwd: Path) -> None:
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def load_sweep_metrics(
    analysis_dir: Path,
    alpha: float,
    layer_group: str,
    condition: str,
) -> list[dict]:
    alignment_path = analysis_dir / "alignment_metrics.csv"
    quality_path = analysis_dir / "quality_metrics.csv"
    alignment = pd.read_csv(alignment_path)
    quality = pd.read_csv(quality_path)
    merged = alignment.merge(
        quality[["condition", "persona", "parse_ok_rate", "unknown_rate"]],
        on=["condition", "persona"],
        how="left",
    )
    merged = merged[merged["condition"] == condition]

    rows = []
    for row in merged.to_dict("records"):
        rows.append({
            "alpha": alpha,
            "layer_group": layer_group,
            "condition": row["condition"],
            "persona": row["persona"],
            "n_total": row["n_total"],
            "n_valid": row["n_valid"],
            "parse_ok_rate": row["parse_ok_rate"],
            "unknown_rate": row["unknown_rate"],
            "align_itt": row["align_itt"],
            "align_itt_ci_low": row["align_itt_ci_low"],
            "align_itt_ci_high": row["align_itt_ci_high"],
            "align_itt_vs_chance": row["align_itt_vs_chance"],
            "align_pp": row["align_pp"],
            "align_pp_ci_low": row["align_pp_ci_low"],
            "align_pp_ci_high": row["align_pp_ci_high"],
            "align_pp_vs_chance": row["align_pp_vs_chance"],
            "chance_alignment_rate": row["chance_alignment_rate"],
            "entropy": row["entropy"],
            "unique_actions": row["unique_actions"],
        })
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_landscape(rows: list[dict], figures_dir: Path) -> None:
    if not rows:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for persona, group in df.groupby("persona"):
        pivot = group.pivot_table(
            index="layer_group",
            columns="alpha",
            values="align_itt",
            aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        image = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(value) for value in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("alpha")
        ax.set_ylabel("layer group")
        ax.set_title(f"Alpha-layer landscape ({persona})")
        for y in range(pivot.shape[0]):
            for x in range(pivot.shape[1]):
                value = pivot.iloc[y, x]
                if pd.notna(value):
                    ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white")
        fig.colorbar(image, ax=ax, label="ITT alignment")
        fig.tight_layout()
        fig.savefig(figures_dir / f"alpha_layer_landscape_{persona}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 alpha/layer sweeps.")
    parser.add_argument("--config_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default=None)
    parser.add_argument("--condition", type=str, default="as_only_sep")
    parser.add_argument("--scenarios_path", type=str, default=None)
    parser.add_argument("--alphas", type=str, default=None)
    parser.add_argument("--layer_groups", type=str, default="early,middle,late")
    parser.add_argument("--reference_alpha", type=float, default=2.0)
    parser.add_argument("--reference_layer_group", type=str, default="middle")
    parser.add_argument("--num_scenarios", type=int, default=5)
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--bootstrap_iters", type=int, default=200)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    config_dir = Path(args.config_dir) if args.config_dir else PROJECT_ROOT / "configs"
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "v2" / "sweeps"
    )
    figures_dir = Path(args.figures_dir) if args.figures_dir else (
        PROJECT_ROOT / "results" / "v2" / "sweep_figures"
    )
    scenarios_path = Path(args.scenarios_path) if args.scenarios_path else (
        PROJECT_ROOT / "data" / "scenarios" / "team_a_scenarios.jsonl"
    )

    steering_config = load_config(config_dir / "steering.yaml")
    steering_cfg = steering_config.get("steering", steering_config)
    alphas = (
        parse_float_list(args.alphas)
        if args.alphas
        else [float(value) for value in steering_cfg.get("alpha_values", [])]
    )
    layer_groups = parse_str_list(args.layer_groups)

    all_rows = []
    for layer_group in layer_groups:
        for alpha in alphas:
            slug = f"alpha_{alpha_slug(alpha)}__layer_{layer_group}"
            raw_dir = output_dir / "raw" / slug
            analysis_dir = output_dir / "analysis" / slug
            run_file = raw_dir / f"{args.condition}.jsonl"
            analysis_file = analysis_dir / "alignment_metrics.csv"

            if not (args.skip_existing and run_file.exists()):
                run_command([
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "03_run_experiment.py"),
                    "--selection_mode",
                    "loglik",
                    "--conditions",
                    args.condition,
                    "--num_scenarios",
                    str(args.num_scenarios),
                    "--num_repeats",
                    str(args.num_repeats),
                    "--scenarios_path",
                    str(scenarios_path),
                    "--output_dir",
                    str(raw_dir),
                    "--layer_group",
                    layer_group,
                    "--alpha",
                    str(alpha),
                ], PROJECT_ROOT)

            if not (args.skip_existing and analysis_file.exists()):
                run_command([
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "04_evaluate.py"),
                    "--input_dir",
                    str(raw_dir),
                    "--output_dir",
                    str(analysis_dir),
                    "--figures_dir",
                    str(output_dir / "figures" / slug),
                    "--scenarios_path",
                    str(scenarios_path),
                    "--bootstrap_iters",
                    str(args.bootstrap_iters),
                ], PROJECT_ROOT)

            all_rows.extend(
                load_sweep_metrics(analysis_dir, alpha, layer_group, args.condition)
            )

    landscape_path = output_dir / "alpha_layer_landscape.csv"
    write_rows(landscape_path, all_rows)

    alpha_rows = [
        row for row in all_rows
        if row["layer_group"] == args.reference_layer_group
    ]
    write_rows(output_dir / "alpha_sweep.csv", alpha_rows)

    layer_rows = [
        row for row in all_rows
        if float(row["alpha"]) == float(args.reference_alpha)
    ]
    write_rows(output_dir / "layer_sweep.csv", layer_rows)

    plot_landscape(all_rows, figures_dir)
    logger.info("Saved sweep rows to: %s", output_dir)
    logger.info("Saved sweep figures to: %s", figures_dir)


if __name__ == "__main__":
    main()
