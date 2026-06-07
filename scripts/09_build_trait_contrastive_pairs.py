"""Build Big Five TRAIT contrastive pair JSONL files.

Input must already be converted to the normalized schema loaded by
``src.data.trait_dataset``. This script does not download TRAIT data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bfi import BIG_FIVE_TRAITS
from src.data.trait_dataset import (
    build_trait_contrastive_pairs,
    load_trait_scenarios,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build high-vs-low Big Five TRAIT contrastive pairs."
    )
    parser.add_argument(
        "--scenarios_path",
        default="data/trait_bigfive/scenarios.jsonl",
        help="Normalized TRAIT Big Five scenarios JSONL.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/trait_bigfive/contrastive_pairs",
        help="Directory for per-trait pair JSONL files.",
    )
    parser.add_argument(
        "--splits",
        default="train",
        help="Comma-separated scenario splits to include. Default: train.",
    )
    parser.add_argument(
        "--traits",
        default=",".join(sorted(BIG_FIVE_TRAITS)),
        help="Comma-separated traits to export. Default: all Big Five traits.",
    )
    args = parser.parse_args()

    allowed_splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    traits = [trait.strip().lower() for trait in args.traits.split(",") if trait.strip()]
    invalid_traits = set(traits) - BIG_FIVE_TRAITS
    if invalid_traits:
        raise ValueError(f"Unsupported trait(s): {sorted(invalid_traits)}")

    scenarios = load_trait_scenarios(args.scenarios_path, allowed_splits=allowed_splits)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for trait in traits:
        pairs = build_trait_contrastive_pairs(scenarios, trait=trait)
        count = write_jsonl(pairs, output_dir / f"{trait}.jsonl")
        print(f"Wrote {count} {trait} pairs")
        all_pairs.extend(pairs)

    total = write_jsonl(all_pairs, output_dir / "all_traits.jsonl")
    print(f"Wrote {total} total pairs")


if __name__ == "__main__":
    main()
