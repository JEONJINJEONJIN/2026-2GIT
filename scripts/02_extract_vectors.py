"""Steering vector extraction script.

Loads a model (fp16 or 4-bit), loads contrastive pairs, extracts
activations using ActivationExtractor, computes steering vectors
via SteeringVectorComputer, and saves results.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.loader import load_model_and_tokenizer
from src.steering.extractor import ActivationExtractor
from src.steering.vector import SteeringVectorComputer
from src.utils.config import load_config
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_pairs(pairs_path: Path) -> list[dict]:
    """Load contrastive pairs from a JSONL file."""
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Extract steering vectors from contrastive pairs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config YAML (default: configs/model.yaml).",
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        default=None,
        help="Path to contrastive pairs JSONL. Default: data/contrastive_pairs/aggressive_cooperative.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save vectors. Default: results/vectors/",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["fp16", "4bit"],
        default="fp16",
        help="Model quantization mode (default: fp16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    # Resolve paths
    config_path = Path(args.config) if args.config else (
        PROJECT_ROOT / "configs" / "model.yaml"
    )
    pairs_path = Path(args.pairs_path) if args.pairs_path else (
        PROJECT_ROOT / "data" / "contrastive_pairs" / "aggressive_cooperative.jsonl"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "vectors"
    )

    set_seed(args.seed)

    # Load config and override quantization if specified
    logger.info("Loading config from: %s", config_path)
    config = load_config(config_path)

    if args.quantization == "4bit":
        config.setdefault("model", config)
        model_cfg = config.get("model", config)
        model_cfg["quantization"] = "4bit"
        model_cfg["load_in_4bit"] = True
    else:
        model_cfg = config.get("model", config)
        model_cfg["quantization"] = None
        model_cfg["load_in_4bit"] = False

    # Load model
    logger.info("Loading model (%s)...", args.quantization)
    model, tokenizer = load_model_and_tokenizer(config)

    # Load pairs
    logger.info("Loading contrastive pairs from: %s", pairs_path)
    pairs = load_pairs(pairs_path)
    logger.info("Loaded %d contrastive pairs.", len(pairs))

    # Extract and compute vectors
    extractor = ActivationExtractor(model, tokenizer)
    computer = SteeringVectorComputer()

    logger.info("Computing steering vectors for all layers...")
    vectors = computer.compute_from_pairs(extractor, pairs)
    logger.info("Computed vectors for %d layers.", len(vectors))

    # Save vectors
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"steering_vectors_{args.quantization}.pt"
    computer.save_vectors(vectors, output_file)
    logger.info("Saved steering vectors to: %s", output_file)

    # Log vector norms for diagnostics
    for layer_idx in sorted(vectors.keys()):
        norm = vectors[layer_idx].norm().item()
        logger.info("  Layer %2d: norm = %.4f", layer_idx, norm)

    logger.info("Vector extraction complete.")


if __name__ == "__main__":
    main()
