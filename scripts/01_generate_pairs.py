"""Contrastive pair generation script.

Loads pair templates from data/contrastive_pairs/templates/pair_templates.jsonl,
generates variations (currently copies templates as a starting point),
and saves generated pairs to the output path.
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_templates(templates_path: Path) -> list[dict]:
    """Load pair templates from a JSONL file."""
    templates = []
    with open(templates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                templates.append(json.loads(line))
    return templates


def generate_variations(template: dict, num_variations: int) -> list[dict]:
    """Generate contrastive pair variations from a single template.

    For now, this creates variations by filling the situation into the
    positive/negative prompts. Future versions will use LLM-based
    synonym replacement and context shifts.

    Args:
        template: A template dict with 'positive', 'negative', 'situation' keys.
        num_variations: Number of variations to generate per template.

    Returns:
        List of contrastive pair dicts with 'positive' and 'negative' keys.
    """
    pairs = []
    situation = template.get("situation", "")
    positive_template = template.get("positive", "")
    negative_template = template.get("negative", "")

    for i in range(num_variations):
        pair = {
            "id": f"{template.get('id', 'cp_unknown')}_var{i:03d}",
            "positive": positive_template.replace("{situation}", situation),
            "negative": negative_template.replace("{situation}", situation),
            "situation": situation,
            "source_template": template.get("id", "unknown"),
        }
        pairs.append(pair)

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Generate contrastive pairs from templates."
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=100,
        help="Total number of pairs to generate (default: 100).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSONL path. Default: data/contrastive_pairs/aggressive_cooperative.jsonl",
    )
    parser.add_argument(
        "--templates_path",
        type=str,
        default=None,
        help="Path to templates JSONL. Default: data/contrastive_pairs/templates/pair_templates.jsonl",
    )
    args = parser.parse_args()

    # Resolve paths
    templates_path = Path(args.templates_path) if args.templates_path else (
        PROJECT_ROOT / "data" / "contrastive_pairs" / "templates" / "pair_templates.jsonl"
    )
    output_path = Path(args.output_path) if args.output_path else (
        PROJECT_ROOT / "data" / "contrastive_pairs" / "aggressive_cooperative.jsonl"
    )

    print(f"Loading templates from: {templates_path}")
    templates = load_templates(templates_path)
    print(f"Loaded {len(templates)} templates.")

    if len(templates) == 0:
        print("No templates found. Exiting.")
        return

    # Distribute num_pairs across templates
    variations_per_template = max(1, args.num_pairs // len(templates))
    remainder = args.num_pairs % len(templates)

    all_pairs = []
    for i, template in enumerate(templates):
        n = variations_per_template + (1 if i < remainder else 0)
        pairs = generate_variations(template, n)
        all_pairs.extend(pairs)

    # Trim to exact count
    all_pairs = all_pairs[: args.num_pairs]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Generated {len(all_pairs)} contrastive pairs.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
