"""Analyze paraphrase robustness for v4 Big Five metric rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.bigfive_analysis import load_csv_rows, write_csv_rows  # noqa: E402
from src.evaluation.paraphrase_robustness import (  # noqa: E402
    summarize_paraphrase_robustness,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze stability across paraphrased TRAIT scenarios."
    )
    parser.add_argument(
        "--input",
        default="results/v4_bigfive/trait_scores_with_paraphrases.csv",
        help="CSV with trait score rows and paraphrase_group_id",
    )
    parser.add_argument(
        "--output",
        default="results/v4_bigfive/paraphrase_robustness.csv",
        help="Output summary CSV",
    )
    parser.add_argument("--group_column", default="paraphrase_group_id")
    parser.add_argument("--variant_column", default="paraphrase_variant")
    parser.add_argument("--action_column", default="loglik_selected_action")
    parser.add_argument("--score_column", default="trait_score")
    args = parser.parse_args()

    rows = load_csv_rows(PROJECT_ROOT / args.input)
    summary = summarize_paraphrase_robustness(
        rows,
        group_column=args.group_column,
        variant_column=args.variant_column,
        action_column=args.action_column,
        score_column=args.score_column,
    )
    count = write_csv_rows(summary, PROJECT_ROOT / args.output)
    print(f"Saved paraphrase robustness rows: {count}")


if __name__ == "__main__":
    main()
