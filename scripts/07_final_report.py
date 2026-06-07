"""Create final v2 analysis report artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_COMPARISONS = (
    ("as_only_neg", "neutral_baseline"),
    ("as_only_sep", "neutral_baseline"),
    ("prompt_as", "neutral_baseline"),
    ("prompt_baseline", "neutral_baseline"),
    ("prompt_as", "prompt_baseline"),
    ("prompt_as", "as_only_sep"),
)


def _fmt_rate(value: float) -> str:
    return f"{value:.3f}"


def _fmt_ci(low: float, high: float) -> str:
    return f"[{low:.3f}, {high:.3f}]"


def compute_paired_cluster_differences(
    annotated_df: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Return paired condition differences with scenario-cluster bootstrap CIs."""
    df = annotated_df.copy()
    df["aligned"] = (df["action_alignment_category"] == "aligned").astype(float)
    rng = np.random.default_rng(seed)
    scenarios = sorted(df["scenario_id"].dropna().unique())
    rows = []

    for left, right in DEFAULT_COMPARISONS:
        for persona in sorted(df["persona"].dropna().unique()):
            persona_df = df[df["persona"] == persona]
            left_df = persona_df[persona_df["condition"] == left]
            right_df = persona_df[persona_df["condition"] == right]
            estimate = float(left_df["aligned"].mean() - right_df["aligned"].mean())

            values = []
            for _ in range(n_bootstrap):
                sampled = rng.choice(scenarios, size=len(scenarios), replace=True)
                boot = pd.concat(
                    [persona_df[persona_df["scenario_id"] == scenario] for scenario in sampled],
                    ignore_index=True,
                )
                values.append(
                    boot[boot["condition"] == left]["aligned"].mean()
                    - boot[boot["condition"] == right]["aligned"].mean()
                )
            ci_low, ci_high = np.quantile(values, [0.025, 0.975])
            rows.append({
                "comparison": f"{left} - {right}",
                "persona": persona,
                "diff": estimate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_bootstrap": n_bootstrap,
                "n_clusters": len(scenarios),
            })
    return pd.DataFrame(rows)


def build_report(
    output_dir: Path,
    paired: pd.DataFrame,
) -> str:
    quality = pd.read_csv(output_dir / "quality_metrics.csv")
    alignment = pd.read_csv(output_dir / "alignment_metrics.csv")
    bootstrap = pd.read_csv(output_dir / "cluster_bootstrap_alignment.csv")
    mixed = pd.read_csv(output_dir / "mixed_effects_fixed_effects.csv")
    random_effects = pd.read_csv(output_dir / "mixed_effects_random_effects.csv")
    key_cosines = pd.read_csv(output_dir.parent / "vector_diagnostics" / "vector_design_key_cosines.csv")

    condition_order = [
        "neutral_baseline",
        "prompt_baseline",
        "as_only_neg",
        "as_only_sep",
        "prompt_as",
    ]

    lines = [
        "# Activation Steering Evaluation V2 Final Report",
        "",
        "## Executive Summary",
        "",
        "- Description log-likelihood scoring removed the parser confound: every final cell has parse_ok = 1.000 and unknown = 0.000.",
        "- Steering-only improves over neutral, but prompt conditioning remains the strongest intervention in this run.",
        "- Prompt plus activation steering is positive versus neutral, but it does not improve over prompt-only.",
        "- Neutral-anchor separate vectors point in a similar direction, supporting the interpretation that this extraction mostly captures a shared non-neutral component.",
        "- Scenario-level variance is substantial, so clustered bootstrap and mixed-effects estimates should be used instead of row-independent tests.",
        "",
        "## Experimental Setup",
        "",
        "- Model: Qwen2.5-3B-Instruct.",
        "- Scoring: conditional log-likelihood over action descriptions.",
        "- Scenarios: 30 Team A scenarios.",
        "- Repeats: 1 deterministic loglik decision per scenario/persona/condition.",
        "- Personas: aggressive and cooperative.",
        "- Conditions: neutral_baseline, prompt_baseline, as_only_neg, as_only_sep, prompt_as.",
        "- Steering: alpha 4.0, middle layers [18, 21, 24].",
        "- Vector estimation: 120 action-derived aggressive/cooperative contrastive pairs.",
        "",
        "## Quality Metrics",
        "",
    ]

    parse_ok_min = quality["parse_ok_rate"].min()
    unknown_max = quality["unknown_rate"].max()
    lines.extend([
        f"- Minimum parse_ok_rate: {_fmt_rate(parse_ok_min)}.",
        f"- Maximum unknown_rate: {_fmt_rate(unknown_max)}.",
        "- ITT and per-protocol alignment are therefore identical in the final run.",
        "",
        "## Alignment Rates",
        "",
        "| condition | aggressive | cooperative |",
        "|---|---:|---:|",
    ])

    for condition in condition_order:
        row_a = alignment[(alignment["condition"] == condition) & (alignment["persona"] == "aggressive")].iloc[0]
        row_c = alignment[(alignment["condition"] == condition) & (alignment["persona"] == "cooperative")].iloc[0]
        lines.append(
            f"| {condition} | {_fmt_rate(row_a['align_itt'])} | {_fmt_rate(row_c['align_itt'])} |"
        )

    lines.extend([
        "",
        "Chance baseline is 0.400 for both aggressive and cooperative personas.",
        "",
        "## Cluster Bootstrap Intervals",
        "",
        "| condition | aggressive 95% CI | cooperative 95% CI |",
        "|---|---|---|",
    ])

    for condition in condition_order:
        row_a = bootstrap[
            (bootstrap["condition"] == condition)
            & (bootstrap["persona"] == "aggressive")
            & (bootstrap["estimand"] == "itt")
        ].iloc[0]
        row_c = bootstrap[
            (bootstrap["condition"] == condition)
            & (bootstrap["persona"] == "cooperative")
            & (bootstrap["estimand"] == "itt")
        ].iloc[0]
        lines.append(
            f"| {condition} | {_fmt_ci(row_a['ci_low'], row_a['ci_high'])} | {_fmt_ci(row_c['ci_low'], row_c['ci_high'])} |"
        )

    lines.extend([
        "",
        "## Paired Condition Differences",
        "",
        "| comparison | persona | diff | 95% CI |",
        "|---|---|---:|---|",
    ])
    for _, row in paired.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['persona']} | {_fmt_rate(row['diff'])} | {_fmt_ci(row['ci_low'], row['ci_high'])} |"
        )

    lines.extend([
        "",
        "## Mixed-Effects Logistic Regression",
        "",
        "Model: `aligned ~ condition * persona + (1 | scenario_id)`.",
        "",
        "| term | odds ratio | 95% CI | p |",
        "|---|---:|---|---:|",
    ])
    keep_terms = [
        "C(condition, Treatment(reference='neutral_baseline'))[T.as_only_neg]",
        "C(condition, Treatment(reference='neutral_baseline'))[T.as_only_sep]",
        "C(condition, Treatment(reference='neutral_baseline'))[T.prompt_as]",
        "C(condition, Treatment(reference='neutral_baseline'))[T.prompt_baseline]",
    ]
    labels = {
        keep_terms[0]: "as_only_neg vs neutral",
        keep_terms[1]: "as_only_sep vs neutral",
        keep_terms[2]: "prompt_as vs neutral",
        keep_terms[3]: "prompt_baseline vs neutral",
    }
    for term in keep_terms:
        row = mixed[mixed["term"] == term].iloc[0]
        lines.append(
            f"| {labels[term]} | {_fmt_rate(row['odds_ratio'])} | {_fmt_ci(row['or_ci_low'], row['or_ci_high'])} | {row['p_value']:.3g} |"
        )

    scenario_var = float(random_effects.iloc[0]["estimate_variance"])
    scenario_sd = float(random_effects.iloc[0]["estimate_sd"])
    lines.extend([
        "",
        f"Scenario random-effect SD is {_fmt_rate(scenario_sd)} and variance is {_fmt_rate(scenario_var)}.",
        "",
        "## Vector Diagnosis",
        "",
        "| comparison | mean cosine |",
        "|---|---:|",
    ])
    for _, row in key_cosines.iterrows():
        lines.append(f"| {row['vector_a']} vs {row['vector_b']} | {_fmt_rate(row['cosine_mean'])} |")

    lines.extend([
        "",
        "Interpretation: neutral-anchor separate extraction did not recover opposed persona vectors. The orthogonalized direction is very close to the explicit contrast direction, and the final `as_only_sep` condition is best read as an explicit contrast-vector ablation rather than a successful neutral-anchor separation.",
        "",
        "## Final Interpretation",
        "",
        "The v2 pipeline fixes the major measurement problem from the legacy generation setup. With parse failures removed, the remaining result is more interpretable: activation steering has a measurable effect over neutral, especially for cooperative alignment, but prompt conditioning is still stronger. Adding steering to the prompt does not improve over prompt-only under the selected alpha/layer setting. The vector diagnostics also show that the original neutral-anchor separate-vector idea is not well supported by this data, because it captures a shared non-neutral direction more than an aggressive-versus-cooperative axis.",
        "",
        "## Artifacts",
        "",
        "- Raw final outputs: `results/v2/final_raw/`.",
        "- Main metrics: `results/v2/final_analysis_v2/alignment_metrics.csv`.",
        "- Quality metrics: `results/v2/final_analysis_v2/quality_metrics.csv`.",
        "- Paired differences: `results/v2/final_analysis_v2/paired_cluster_differences.csv`.",
        "- Mixed effects: `results/v2/final_analysis_v2/mixed_effects_fixed_effects.csv`.",
        "- Vector diagnostics: `results/v2/vector_diagnostics/vector_design_key_cosines.csv`.",
        "- Figures: `results/v2/final_figures_v2/`.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/v2/final_analysis_v2")
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    annotated = pd.read_csv(output_dir / "annotated_results.csv")
    paired = compute_paired_cluster_differences(
        annotated,
        n_bootstrap=args.bootstrap_iters,
        seed=args.seed,
    )
    paired_path = output_dir / "paired_cluster_differences.csv"
    paired.to_csv(paired_path, index=False)

    report = build_report(output_dir, paired)
    report_path = output_dir / "final_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved paired differences to: {paired_path}")
    print(f"Saved final report to: {report_path}")


if __name__ == "__main__":
    main()
