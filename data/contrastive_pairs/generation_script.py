"""LLM-based contrastive pair generation helper.

Provides template-based generation with simple variation strategies
(synonym replacement, context shifts) for expanding the set of
contrastive pairs used in steering vector extraction.
"""

import json
import random
import re
from pathlib import Path
from typing import Any


# Synonym pools for simple variation
AGGRESSIVE_SYNONYMS = [
    ("aggressive", "hostile"),
    ("aggressive", "combative"),
    ("aggressive", "belligerent"),
    ("aggressive", "confrontational"),
    ("aggressive", "fierce"),
    ("warrior", "fighter"),
    ("warrior", "warlord"),
    ("warrior", "enforcer"),
    ("warrior", "raider"),
    ("force", "violence"),
    ("force", "brute strength"),
    ("force", "raw power"),
    ("dominates", "overpowers"),
    ("dominates", "crushes"),
    ("dominates", "subjugates"),
    ("intimidation", "fear"),
    ("intimidation", "threats"),
    ("intimidation", "menace"),
]

COOPERATIVE_SYNONYMS = [
    ("cooperative", "diplomatic"),
    ("cooperative", "peaceable"),
    ("cooperative", "collaborative"),
    ("cooperative", "harmonious"),
    ("cooperative", "amicable"),
    ("mediator", "peacemaker"),
    ("mediator", "negotiator"),
    ("mediator", "diplomat"),
    ("mediator", "conciliator"),
    ("respect", "understanding"),
    ("respect", "empathy"),
    ("respect", "compassion"),
    ("resolves", "settles"),
    ("resolves", "addresses"),
    ("resolves", "reconciles"),
    ("dialogue", "conversation"),
    ("dialogue", "discussion"),
    ("dialogue", "communication"),
]

SITUATION_MODIFIERS = [
    "During a heavy rainstorm, ",
    "Late at night, ",
    "At the break of dawn, ",
    "In the middle of a crowded market, ",
    "While you are exhausted from travel, ",
    "After receiving troubling news, ",
    "With your allies watching closely, ",
    "Under the cover of darkness, ",
    "In the presence of neutral observers, ",
    "While resources are desperately scarce, ",
    "During a festival celebration, ",
    "As tensions reach a breaking point, ",
    "With winter approaching rapidly, ",
    "After a long period of peace, ",
    "While recovering from a recent battle, ",
]


def apply_synonym_replacement(
    text: str,
    synonym_pool: list[tuple[str, str]],
    num_replacements: int = 2,
) -> str:
    """Apply random synonym replacements to a text.

    Args:
        text: Input text to modify.
        synonym_pool: List of (original, replacement) tuples.
        num_replacements: Maximum number of replacements to apply.

    Returns:
        Modified text with synonym replacements.
    """
    available = [
        (orig, repl) for orig, repl in synonym_pool
        if orig.lower() in text.lower()
    ]
    if not available:
        return text

    replacements = random.sample(available, min(num_replacements, len(available)))
    result = text
    for original, replacement in replacements:
        # Case-insensitive replacement (first occurrence only)
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        result = pattern.sub(replacement, result, count=1)

    return result


def apply_context_shift(situation: str) -> str:
    """Add a contextual modifier to a situation description.

    Args:
        situation: Original situation text.

    Returns:
        Situation with a random context modifier prepended.
    """
    modifier = random.choice(SITUATION_MODIFIERS)
    # Lowercase the first character of the situation for natural reading
    if situation and situation[0].isupper():
        situation = situation[0].lower() + situation[1:]
    return modifier + situation


def generate_pairs_from_template(
    template: dict[str, Any],
    num_variations: int = 10,
) -> list[dict]:
    """Generate contrastive pair variations from a single template.

    Applies a mix of strategies:
    1. Direct copy with situation filled in (base pair)
    2. Synonym replacement on positive/negative prompts
    3. Context shifts on the situation

    Args:
        template: A template dict with 'id', 'positive', 'negative', 'situation'.
        num_variations: Number of variations to generate.

    Returns:
        List of contrastive pair dicts, each with 'id', 'positive',
        'negative', 'situation', 'source_template', and 'strategy' keys.
    """
    pairs = []
    template_id = template.get("id", "unknown")
    positive_tmpl = template.get("positive", "")
    negative_tmpl = template.get("negative", "")
    base_situation = template.get("situation", "")

    for i in range(num_variations):
        # Choose a strategy based on index distribution
        if i == 0:
            # First variation is always the direct template fill
            strategy = "direct"
            situation = base_situation
            positive = positive_tmpl.replace("{situation}", situation)
            negative = negative_tmpl.replace("{situation}", situation)

        elif i % 3 == 1:
            # Synonym replacement on prompts
            strategy = "synonym_replacement"
            situation = base_situation
            positive = positive_tmpl.replace("{situation}", situation)
            negative = negative_tmpl.replace("{situation}", situation)
            positive = apply_synonym_replacement(
                positive, AGGRESSIVE_SYNONYMS, num_replacements=2
            )
            negative = apply_synonym_replacement(
                negative, COOPERATIVE_SYNONYMS, num_replacements=2
            )

        elif i % 3 == 2:
            # Context shift on situation
            strategy = "context_shift"
            situation = apply_context_shift(base_situation)
            positive = positive_tmpl.replace("{situation}", situation)
            negative = negative_tmpl.replace("{situation}", situation)

        else:
            # Combined: both synonym replacement and context shift
            strategy = "combined"
            situation = apply_context_shift(base_situation)
            positive = positive_tmpl.replace("{situation}", situation)
            negative = negative_tmpl.replace("{situation}", situation)
            positive = apply_synonym_replacement(
                positive, AGGRESSIVE_SYNONYMS, num_replacements=1
            )
            negative = apply_synonym_replacement(
                negative, COOPERATIVE_SYNONYMS, num_replacements=1
            )

        pair = {
            "id": f"{template_id}_var{i:03d}",
            "positive": positive,
            "negative": negative,
            "situation": situation,
            "source_template": template_id,
            "strategy": strategy,
        }
        pairs.append(pair)

    return pairs


def load_templates(templates_path: str | Path) -> list[dict]:
    """Load templates from a JSONL file.

    Args:
        templates_path: Path to the templates JSONL file.

    Returns:
        List of template dicts.
    """
    templates = []
    with open(templates_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                templates.append(json.loads(line))
    return templates


def save_pairs(pairs: list[dict], output_path: str | Path) -> None:
    """Save generated pairs to a JSONL file.

    Args:
        pairs: List of pair dicts.
        output_path: Path for the output JSONL file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Generate expanded contrastive pairs from templates."
    )
    parser.add_argument(
        "--templates_path",
        type=str,
        default=None,
        help="Path to templates JSONL.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--num_variations",
        type=int,
        default=10,
        help="Number of variations per template (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    templates_path = args.templates_path or str(
        PROJECT_ROOT / "data" / "contrastive_pairs" / "templates" / "pair_templates.jsonl"
    )
    output_path = args.output_path or str(
        PROJECT_ROOT / "data" / "contrastive_pairs" / "aggressive_cooperative_expanded.jsonl"
    )

    print(f"Loading templates from: {templates_path}")
    templates = load_templates(templates_path)
    print(f"Loaded {len(templates)} templates.")

    all_pairs = []
    for template in templates:
        pairs = generate_pairs_from_template(template, args.num_variations)
        all_pairs.extend(pairs)

    save_pairs(all_pairs, output_path)
    print(f"Generated {len(all_pairs)} contrastive pairs.")
    print(f"Saved to: {output_path}")

    # Print strategy distribution
    from collections import Counter
    strategy_counts = Counter(p["strategy"] for p in all_pairs)
    print("\nStrategy distribution:")
    for strategy, count in strategy_counts.most_common():
        print(f"  {strategy}: {count}")
