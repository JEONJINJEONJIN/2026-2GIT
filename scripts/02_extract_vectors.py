"""AS steering vector extraction script.

Loads a model (fp16 or 4-bit), loads contrastive pairs, extracts
activations using ActivationExtractor, computes steering vectors
via SteeringVectorComputer, and saves results.
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bfi import BIG_FIVE_TRAITS
from src.models.loader import load_model_and_tokenizer
from src.steering.extractor import ActivationExtractor
from src.steering.vector import SteeringVectorComputer, compute_cosine_similarity
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


def compute_vector_set(
    computer: SteeringVectorComputer,
    extractor: ActivationExtractor,
    pairs: list[dict],
    layer_indices: list[int],
    normalize_vectors: bool,
    metadata: dict,
) -> tuple[dict, dict]:
    return computer.compute_from_pairs(
        extractor,
        pairs,
        layer_indices=layer_indices,
        normalize=normalize_vectors,
        return_metadata=True,
        metadata=metadata,
    )


def normalize_vector_set(computer: SteeringVectorComputer, vectors: dict) -> dict:
    return {
        layer: computer.normalize_vector(vector)
        for layer, vector in vectors.items()
    }


def negate_vector_set(vectors: dict) -> dict:
    return {layer: -vector for layer, vector in vectors.items()}


def orthogonalized_vector_sets(
    computer: SteeringVectorComputer,
    aggressive_vectors: dict,
    cooperative_vectors: dict,
) -> tuple[dict, dict]:
    """Remove the common non-neutral component from two persona vectors."""
    shared_layers = sorted(set(aggressive_vectors) & set(cooperative_vectors))
    aggressive_orth = {}
    cooperative_orth = {}
    for layer_idx in shared_layers:
        common = (aggressive_vectors[layer_idx] + cooperative_vectors[layer_idx]) / 2
        aggressive_orth[layer_idx] = aggressive_vectors[layer_idx] - common
        cooperative_orth[layer_idx] = cooperative_vectors[layer_idx] - common
    return (
        normalize_vector_set(computer, aggressive_orth),
        normalize_vector_set(computer, cooperative_orth),
    )


def flatten_vector_sets(vector_sets: dict) -> dict[str, dict]:
    """Flatten nested vector groups to named layer-vector maps for diagnostics."""
    flattened = {}
    for name, vectors in vector_sets.items():
        if not isinstance(vectors, dict):
            continue
        if vectors and all(isinstance(layer, int) for layer in vectors):
            flattened[name] = vectors
        else:
            for child_name, child_vectors in vectors.items():
                flattened[f"{name}.{child_name}"] = child_vectors
    return flattened


def vector_set_cosines(vector_sets: dict[str, dict]) -> list[dict]:
    rows = []
    flattened = flatten_vector_sets(vector_sets)
    names = sorted(flattened)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            left_vectors = flattened[left_name]
            right_vectors = flattened[right_name]
            shared_layers = sorted(set(left_vectors) & set(right_vectors))
            values = []
            for layer_idx in shared_layers:
                cosine = compute_cosine_similarity(
                    left_vectors[layer_idx].float(),
                    right_vectors[layer_idx].float(),
                )
                values.append(cosine)
                rows.append({
                    "left": left_name,
                    "right": right_name,
                    "layer": layer_idx,
                    "cosine_similarity": cosine,
                })
            if values:
                rows.append({
                    "left": left_name,
                    "right": right_name,
                    "layer": "mean",
                    "cosine_similarity": sum(values) / len(values),
                })
    return rows


def write_cosine_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["left", "right", "layer", "cosine_similarity"],
        )
        writer.writeheader()
        writer.writerows(rows)


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
        "--steering_config",
        type=str,
        default=None,
        help="Path to steering config YAML (default: configs/steering.yaml).",
    )
    parser.add_argument(
        "--pairs_path",
        type=str,
        default=None,
        help="Path to contrastive pairs JSONL. Default: data/contrastive_pairs/aggressive_cooperative.jsonl",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "separate", "trait"],
        default="single",
        help=(
            "Extract one contrast vector, legacy separate vector sets, or "
            "Big Five trait vectors."
        ),
    )
    parser.add_argument("--contrast_pairs_path", type=str, default=None)
    parser.add_argument("--aggressive_pairs_path", type=str, default=None)
    parser.add_argument("--cooperative_pairs_path", type=str, default=None)
    parser.add_argument(
        "--trait_pairs_dir",
        type=str,
        default=None,
        help=(
            "Directory containing per-trait JSONL pair files for --mode trait. "
            "Default: data/trait_bigfive/contrastive_pairs."
        ),
    )
    parser.add_argument(
        "--traits",
        type=str,
        default=None,
        help="Comma-separated Big Five traits for --mode trait. Default: all traits.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save vectors. Default: results/v2/vectors/",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional explicit .pt output path.",
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
    parser.add_argument(
        "--layer_group",
        type=str,
        default="all",
        help="Layer group from configs/steering.yaml (default: all).",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated explicit layer indices; overrides --layer_group.",
    )
    args = parser.parse_args()

    # Resolve paths
    config_path = Path(args.config) if args.config else (
        PROJECT_ROOT / "configs" / "model.yaml"
    )
    steering_config_path = Path(args.steering_config) if args.steering_config else (
        PROJECT_ROOT / "configs" / "steering.yaml"
    )
    pairs_path = Path(args.pairs_path) if args.pairs_path else (
        PROJECT_ROOT / "data" / "contrastive_pairs" / "aggressive_cooperative.jsonl"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "v2" / "vectors"
    )

    set_seed(args.seed)

    # Load config and override quantization if specified
    logger.info("Loading model config from: %s", config_path)
    config = load_config(config_path)
    logger.info("Loading steering config from: %s", steering_config_path)
    steering_config = load_config(steering_config_path)
    steering_cfg = steering_config.get("steering", steering_config)

    if args.quantization == "4bit":
        config.setdefault("model", config)
        model_cfg = config.get("model", config)
        model_cfg["quantization"] = "4bit"
        model_cfg["load_in_4bit"] = True
    else:
        model_cfg = config.get("model", config)
        model_cfg["quantization"] = None
        model_cfg["load_in_4bit"] = False

    if args.layers:
        layer_indices = [int(item.strip()) for item in args.layers.split(",") if item.strip()]
    else:
        layer_groups = steering_cfg.get("layer_groups", {})
        layer_indices = layer_groups.get(args.layer_group)
        if layer_indices is None:
            raise ValueError(
                f"Unknown layer group '{args.layer_group}'. "
                f"Available groups: {sorted(layer_groups.keys())}"
            )
    normalize_vectors = bool(steering_cfg.get("normalize_vectors", True))

    # Load model
    logger.info("Loading model (%s)...", args.quantization)
    model, tokenizer = load_model_and_tokenizer(config)

    # Extract and compute vectors
    extractor = ActivationExtractor(model, tokenizer)
    computer = SteeringVectorComputer()

    metadata = {
        "model_name": model_cfg.get("name"),
        "model_dtype": model_cfg.get("dtype"),
        "quantization": args.quantization,
        "hook_target": steering_cfg.get("hook_target", "post_block_residual"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "single":
        logger.info("Loading contrastive pairs from: %s", pairs_path)
        pairs = load_pairs(pairs_path)
        logger.info("Loaded %d contrastive pairs.", len(pairs))

        logger.info("Computing steering vectors for layers: %s", layer_indices)
        vectors, vector_metadata = compute_vector_set(
            computer,
            extractor,
            pairs,
            layer_indices=layer_indices,
            normalize_vectors=normalize_vectors,
            metadata=metadata,
        )
        logger.info("Computed vectors for %d layers.", len(vectors))

        output_file = (
            Path(args.output_file)
            if args.output_file
            else output_dir / f"steering_vectors_{args.quantization}.pt"
        )
        computer.save_vectors(vectors, output_file, metadata=vector_metadata)
        logger.info("Saved steering vectors to: %s", output_file)

        for layer_idx in sorted(vectors.keys()):
            norm = vectors[layer_idx].norm().item()
            logger.info("  Layer %2d: norm = %.4f", layer_idx, norm)
        logger.info("Metadata: %s", vector_metadata)
    elif args.mode == "separate":
        contrast_path = Path(args.contrast_pairs_path) if args.contrast_pairs_path else pairs_path
        aggressive_path = Path(args.aggressive_pairs_path) if args.aggressive_pairs_path else (
            PROJECT_ROOT / "data" / "contrastive_pairs" / "aggressive_neutral.jsonl"
        )
        cooperative_path = Path(args.cooperative_pairs_path) if args.cooperative_pairs_path else (
            PROJECT_ROOT / "data" / "contrastive_pairs" / "cooperative_neutral.jsonl"
        )
        pair_specs = {
            "contrast": contrast_path,
            "neutral_anchor.aggressive": aggressive_path,
            "neutral_anchor.cooperative": cooperative_path,
        }
        computed_sets = {}
        vector_metadata = {}
        for set_name, set_path in pair_specs.items():
            logger.info("Loading %s pairs from: %s", set_name, set_path)
            pairs = load_pairs(set_path)
            logger.info("Loaded %d %s pairs.", len(pairs), set_name)
            set_vectors, set_metadata = compute_vector_set(
                computer,
                extractor,
                pairs,
                layer_indices=layer_indices,
                normalize_vectors=normalize_vectors,
                metadata={**metadata, "vector_set": set_name},
            )
            computed_sets[set_name] = set_vectors
            vector_metadata[set_name] = set_metadata

        contrast_vectors = computed_sets["contrast"]
        neutral_anchor_aggressive = computed_sets["neutral_anchor.aggressive"]
        neutral_anchor_cooperative = computed_sets["neutral_anchor.cooperative"]
        contrast_explicit = {
            "aggressive": contrast_vectors,
            "cooperative": negate_vector_set(contrast_vectors),
        }
        orth_aggressive, orth_cooperative = orthogonalized_vector_sets(
            computer,
            neutral_anchor_aggressive,
            neutral_anchor_cooperative,
        )
        vector_sets = {
            "contrast": contrast_vectors,
            "aggressive": neutral_anchor_aggressive,
            "cooperative": neutral_anchor_cooperative,
            "contrast_explicit": contrast_explicit,
            "orthogonalized": {
                "aggressive": orth_aggressive,
                "cooperative": orth_cooperative,
            },
        }

        cosine_rows = vector_set_cosines(vector_sets)
        cosine_csv = output_dir / "vector_set_cosine_similarities.csv"
        write_cosine_csv(cosine_rows, cosine_csv)

        output_file = (
            Path(args.output_file)
            if args.output_file
            else output_dir / f"steering_vectors_separate_{args.quantization}.pt"
        )
        payload = {
            "vectors": vector_sets,
            "metadata": {
                "mode": "separate",
                "vector_sets": vector_metadata,
                "cosine_similarity_csv": str(cosine_csv),
                "between_vector_cosine_similarities": cosine_rows,
            },
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        import torch

        torch.save(payload, output_file)
        logger.info("Saved separate steering vectors to: %s", output_file)
        logger.info("Saved vector-set cosine similarities to: %s", cosine_csv)
    else:
        trait_pairs_dir = Path(args.trait_pairs_dir) if args.trait_pairs_dir else (
            PROJECT_ROOT / "data" / "trait_bigfive" / "contrastive_pairs"
        )
        traits = (
            [item.strip().lower() for item in args.traits.split(",") if item.strip()]
            if args.traits
            else sorted(BIG_FIVE_TRAITS)
        )
        invalid_traits = set(traits) - BIG_FIVE_TRAITS
        if invalid_traits:
            raise ValueError(f"Unsupported Big Five trait(s): {sorted(invalid_traits)}")

        trait_vectors = {}
        vector_metadata = {}
        for trait in traits:
            trait_path = trait_pairs_dir / f"{trait}.jsonl"
            logger.info("Loading %s trait pairs from: %s", trait, trait_path)
            pairs = load_pairs(trait_path)
            logger.info("Loaded %d %s pairs.", len(pairs), trait)
            vectors, trait_metadata = compute_vector_set(
                computer,
                extractor,
                pairs,
                layer_indices=layer_indices,
                normalize_vectors=normalize_vectors,
                metadata={
                    **metadata,
                    "mode": "trait",
                    "trait": trait,
                    "positive_direction": "high",
                    "negative_direction": "low",
                    "source_pairs_path": str(trait_path),
                },
            )
            trait_vectors[trait] = vectors
            vector_metadata[trait] = trait_metadata

        vector_sets = {"trait_contrast": trait_vectors}
        cosine_rows = vector_set_cosines(vector_sets)
        cosine_csv = output_dir / "trait_vector_cosine_similarities.csv"
        write_cosine_csv(cosine_rows, cosine_csv)

        output_file = (
            Path(args.output_file)
            if args.output_file
            else output_dir / f"bigfive_trait_vectors_{args.quantization}.pt"
        )
        payload = {
            "vectors": vector_sets,
            "metadata": {
                "mode": "trait",
                "traits": traits,
                "trait_vector_metadata": vector_metadata,
                "cosine_similarity_csv": str(cosine_csv),
                "between_vector_cosine_similarities": cosine_rows,
                "direction_rule": {
                    "high_target": "+trait_contrast[trait]",
                    "low_target": "-trait_contrast[trait]",
                },
            },
        }
        output_file.parent.mkdir(parents=True, exist_ok=True)
        import torch

        torch.save(payload, output_file)
        logger.info("Saved Big Five trait steering vectors to: %s", output_file)
        logger.info("Saved trait-vector cosine similarities to: %s", cosine_csv)

    logger.info("Vector extraction complete.")


if __name__ == "__main__":
    main()
