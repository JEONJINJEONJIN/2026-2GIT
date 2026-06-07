"""Analyze v4 Big Five pilot/final metric CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bigfive_analysis import (
    build_final_report,
    build_markdown_report,
    condition_effects_with_ci,
    condition_differences,
    load_csv_rows,
    load_json_file,
    paired_bootstrap_ci,
    paired_scenario_differences,
    summarize_consistency,
    summarize_pas_agreement,
    summarize_side_effects,
    write_csv_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze v4 Big Five result CSVs.")
    parser.add_argument(
        "--metrics_dir",
        default="results/v4_bigfive/pilot_metrics",
        help="Directory containing consistency_metrics.csv.",
    )
    parser.add_argument(
        "--reference_condition",
        default="baseline",
        help="Condition used as difference reference.",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=1000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap_metrics",
        nargs="+",
        default=["consistency", "trait_target_attainment"],
        help="Scenario-level metric columns to bootstrap against the reference.",
    )
    args = parser.parse_args()

    metrics_dir = PROJECT_ROOT / args.metrics_dir
    consistency_path = metrics_dir / "consistency_metrics.csv"
    if not consistency_path.exists():
        raise FileNotFoundError(f"Missing consistency metrics: {consistency_path}")

    consistency_rows = load_csv_rows(consistency_path)
    summary = summarize_consistency(consistency_rows)
    differences = condition_differences(
        summary,
        reference_condition=args.reference_condition,
    )
    paired = paired_scenario_differences(
        consistency_rows,
        reference_condition=args.reference_condition,
    )
    bootstrap = []
    for metric in args.bootstrap_metrics:
        bootstrap.extend(
            paired_bootstrap_ci(
                consistency_rows,
                reference_condition=args.reference_condition,
                metric_column=metric,
                n_bootstrap=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
        )
    effects_with_ci = condition_effects_with_ci(differences, bootstrap)

    pas_summary = []
    pas_path = metrics_dir / "pas_loglik_agreement.csv"
    if pas_path.exists() and pas_path.stat().st_size > 0:
        pas_summary = summarize_pas_agreement(load_csv_rows(pas_path))

    side_effect_summary = []
    side_effect_path = metrics_dir / "side_effect_metrics.csv"
    if side_effect_path.exists() and side_effect_path.stat().st_size > 0:
        side_effect_summary = summarize_side_effects(load_csv_rows(side_effect_path))

    write_csv_rows(summary, metrics_dir / "consistency_summary.csv")
    write_csv_rows(differences, metrics_dir / "condition_differences.csv")
    write_csv_rows(effects_with_ci, metrics_dir / "condition_effects_with_ci.csv")
    write_csv_rows(paired, metrics_dir / "paired_scenario_differences.csv")
    write_csv_rows(bootstrap, metrics_dir / "paired_bootstrap_ci.csv")
    if pas_summary:
        write_csv_rows(pas_summary, metrics_dir / "pas_agreement_summary.csv")
    if side_effect_summary:
        write_csv_rows(side_effect_summary, metrics_dir / "side_effect_summary.csv")

    report = build_markdown_report(
        summary,
        differences,
        paired,
        bootstrap_rows=bootstrap,
        condition_effect_rows=effects_with_ci,
        pas_rows=pas_summary,
        side_effect_rows=side_effect_summary,
    )
    report_path = metrics_dir / "analysis_report.md"
    report_path.write_text(report, encoding="utf-8")

    manifest = None
    manifest_path = metrics_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = load_json_file(manifest_path)
    final_report = build_final_report(
        manifest=manifest,
        summary_rows=summary,
        difference_rows=differences,
        bootstrap_rows=bootstrap,
        condition_effect_rows=effects_with_ci,
        pas_rows=pas_summary,
        side_effect_rows=side_effect_summary,
    )
    final_report_path = metrics_dir / "final_report.md"
    final_report_path.write_text(final_report, encoding="utf-8")

    print(f"Wrote consistency summary: {metrics_dir / 'consistency_summary.csv'}")
    print(f"Wrote condition differences: {metrics_dir / 'condition_differences.csv'}")
    print(f"Wrote condition effects with CI: {metrics_dir / 'condition_effects_with_ci.csv'}")
    print(f"Wrote paired differences: {metrics_dir / 'paired_scenario_differences.csv'}")
    print(f"Wrote bootstrap CI: {metrics_dir / 'paired_bootstrap_ci.csv'}")
    if pas_summary:
        print(f"Wrote PAS summary: {metrics_dir / 'pas_agreement_summary.csv'}")
    if side_effect_summary:
        print(f"Wrote side-effect summary: {metrics_dir / 'side_effect_summary.csv'}")
    print(f"Wrote report: {report_path}")
    print(f"Wrote final report: {final_report_path}")


if __name__ == "__main__":
    main()
