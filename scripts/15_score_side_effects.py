"""Score side-effect metrics for generated NPC response CSV files."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.side_effect_metrics import (  # noqa: E402
    build_side_effect_row,
    summarize_side_effect_rows,
)
from src.experiments.v4_bigfive_runner import write_csv_rows  # noqa: E402


def _split_actions(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split("|") if item.strip()]


def score_csv(
    input_path: str | Path,
    output_path: str | Path,
    *,
    text_column: str = "raw",
    actions_column: str = "valid_actions",
    require_speech: bool = True,
) -> tuple[int, dict[str, float]]:
    """Score side-effect metrics from a CSV containing generated responses."""

    input_path = Path(input_path)
    with open(input_path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_path}")
        if text_column not in reader.fieldnames:
            raise ValueError(
                f"Input CSV missing text column {text_column!r}. "
                f"Available columns: {reader.fieldnames}"
            )

        rows = []
        for row in reader:
            side_effect_row = build_side_effect_row(
                row.get(text_column, ""),
                valid_actions=_split_actions(row.get(actions_column)),
                condition=row.get("condition"),
                scenario_id=row.get("scenario_id"),
                trait=row.get("trait"),
                target_direction=row.get("target_direction"),
                coherence_score=_optional_float(row.get("coherence_score")),
                perplexity=_optional_float(row.get("perplexity")),
                require_speech=require_speech,
            )
            rows.append(side_effect_row)

    count = write_csv_rows(rows, output_path)
    return count, summarize_side_effect_rows(rows)


def _optional_float(value: str | None) -> float | None:
    if value is None or not str(value).strip():
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score side-effect metrics for generated NPC responses."
    )
    parser.add_argument("--input", required=True, help="CSV with generated responses")
    parser.add_argument(
        "--output",
        default="results/v4_bigfive/side_effect_metrics.csv",
        help="Output CSV path",
    )
    parser.add_argument("--text_column", default="raw")
    parser.add_argument("--actions_column", default="valid_actions")
    parser.add_argument(
        "--allow_missing_speech",
        action="store_true",
        help="Do not require non-empty speech for format_validity",
    )
    args = parser.parse_args()

    count, summary = score_csv(
        PROJECT_ROOT / args.input,
        PROJECT_ROOT / args.output,
        text_column=args.text_column,
        actions_column=args.actions_column,
        require_speech=not args.allow_missing_speech,
    )
    print(f"Saved side-effect rows: {count}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
