"""Convert exported TRAIT rows to the v4 Big Five scenario schema.

The expected raw fields are the Hugging Face TRAIT card fields:
``personality``, ``question``, ``response_high1``, ``response_high2``,
``response_low1``, and ``response_low2``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bfi import BIG_FIVE_TRAITS
from src.data.trait_convert import (
    convert_raw_trait_rows,
    load_raw_trait_rows,
    write_split_index,
    write_normalized_scenarios,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw TRAIT export to normalized Big Five scenarios."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Raw TRAIT JSONL, JSON, or CSV export.",
    )
    parser.add_argument(
        "--output",
        default="data/trait_bigfive/scenarios.jsonl",
        help="Normalized output JSONL path.",
    )
    parser.add_argument(
        "--traits",
        default=",".join(sorted(BIG_FIVE_TRAITS)),
        help="Comma-separated Big Five traits to include.",
    )
    parser.add_argument(
        "--split_ratios",
        default="0.7,0.15,0.15",
        help="Comma-separated train,dev,test ratios. Default: 0.7,0.15,0.15.",
    )
    parser.add_argument(
        "--splits_output",
        default="data/trait_bigfive/splits.json",
        help="Split index JSON path.",
    )
    args = parser.parse_args()

    traits = [trait.strip().lower() for trait in args.traits.split(",") if trait.strip()]
    split_ratios = tuple(
        float(item.strip())
        for item in args.split_ratios.split(",")
        if item.strip()
    )
    if len(split_ratios) != 3:
        raise ValueError("--split_ratios must contain exactly three values")

    raw_rows = load_raw_trait_rows(args.input)
    scenarios = convert_raw_trait_rows(
        raw_rows,
        split_ratios=split_ratios,
        include_traits=traits,
    )
    count = write_normalized_scenarios(scenarios, args.output)
    write_split_index(scenarios, args.splits_output)
    by_trait = {}
    by_split = {}
    for scenario in scenarios:
        by_trait[scenario["trait"]] = by_trait.get(scenario["trait"], 0) + 1
        by_split[scenario["split"]] = by_split.get(scenario["split"], 0) + 1

    print(f"Read {len(raw_rows)} raw TRAIT rows")
    print(f"Wrote {count} normalized Big Five scenarios: {args.output}")
    print(f"Wrote split index: {args.splits_output}")
    print(f"By trait: {by_trait}")
    print(f"By split: {by_split}")


if __name__ == "__main__":
    main()
