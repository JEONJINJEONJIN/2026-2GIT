"""Convert raw TRAIT rows to the v4 normalized Big Five schema."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from src.data.bfi import BIG_FIVE_TRAITS


RAW_TRAIT_REQUIRED_FIELDS = {
    "personality",
    "question",
    "response_high1",
    "response_high2",
    "response_low1",
    "response_low2",
}

TRAIT_NAME_ALIASES = {
    "agreeableness": "agreeableness",
    "conscientiousness": "conscientiousness",
    "neuroticism": "neuroticism",
    "openness": "openness",
    "openness to experience": "openness",
    "extraversion": "extraversion",
}

BALANCED_ACTION_ORDERS = (
    ("high1", "high2", "low1", "low2"),
    ("low1", "high1", "low2", "high2"),
    ("high2", "low1", "high1", "low2"),
    ("low2", "low1", "high2", "high1"),
)


def load_raw_trait_rows(path: str | Path) -> list[dict]:
    """Load raw TRAIT rows from JSONL, JSON, CSV, Parquet, or a directory."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw TRAIT input file not found: {path}")
    if path.is_dir():
        rows = []
        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix.lower() in {".jsonl", ".json", ".csv", ".parquet"}:
                rows.extend(load_raw_trait_rows(child))
        if not rows:
            raise ValueError(f"Directory contains no supported TRAIT files: {path}")
        return rows
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl_rows(path)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [dict(row) for row in data]
        raise ValueError(f"JSON TRAIT input must be a list of rows: {path}")
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Reading TRAIT parquet files requires pandas with pyarrow or fastparquet"
            ) from exc
        return [dict(row) for row in pd.read_parquet(path).to_dict("records")]
    raise ValueError(f"Unsupported TRAIT input extension: {path.suffix}")


def load_jsonl_rows(path: Path) -> list[dict]:
    """Load JSONL rows."""

    rows = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line_number, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
    return rows


def normalize_trait_name(value: str) -> str | None:
    """Normalize raw TRAIT personality names to internal Big Five labels."""

    normalized = value.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return TRAIT_NAME_ALIASES.get(normalized)


def validate_raw_trait_row(row: Mapping[str, object], row_index: int) -> None:
    """Validate raw TRAIT fields required by the dataset card."""

    missing = [field for field in RAW_TRAIT_REQUIRED_FIELDS if field not in row]
    if missing:
        raise ValueError(f"Raw TRAIT row {row_index} missing fields: {sorted(missing)}")
    for field in RAW_TRAIT_REQUIRED_FIELDS:
        if not str(row.get(field) or "").strip():
            raise ValueError(f"Raw TRAIT row {row_index} has empty field: {field}")


def convert_raw_trait_rows(
    rows: Iterable[Mapping[str, object]],
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    include_traits: Iterable[str] | None = None,
) -> list[dict]:
    """Convert raw TRAIT rows to normalized scenario rows.

    Splits are assigned per trait in stable input order. This keeps each trait's
    train/dev/test counts balanced enough for pilot and full-run planning.
    """

    allowed_traits = (
        {trait.lower() for trait in include_traits}
        if include_traits is not None
        else set(BIG_FIVE_TRAITS)
    )
    invalid = allowed_traits - BIG_FIVE_TRAITS
    if invalid:
        raise ValueError(f"Unsupported Big Five trait(s): {sorted(invalid)}")

    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row_index, row in enumerate(rows, start=1):
        validate_raw_trait_row(row, row_index)
        trait = normalize_trait_name(str(row["personality"]))
        if trait is None or trait not in allowed_traits:
            continue
        grouped[trait].append(row)

    scenarios = []
    for trait in sorted(grouped):
        trait_rows = grouped[trait]
        total = len(trait_rows)
        for index, row in enumerate(trait_rows):
            scenario_id = f"trait_{trait}_{index + 1:05d}"
            scenarios.append(
                convert_raw_trait_row(
                    row,
                    trait=trait,
                    scenario_id=scenario_id,
                    split=assign_split(index, total, split_ratios),
                    order_index=index,
                )
            )
    return scenarios


def convert_raw_trait_row(
    row: Mapping[str, object],
    trait: str,
    scenario_id: str,
    split: str,
    order_index: int,
) -> dict:
    """Convert one raw TRAIT row to one normalized scenario row."""

    action_specs = {
        "high1": (str(row["response_high1"]).strip(), "high"),
        "high2": (str(row["response_high2"]).strip(), "high"),
        "low1": (str(row["response_low1"]).strip(), "low"),
        "low2": (str(row["response_low2"]).strip(), "low"),
    }
    order = BALANCED_ACTION_ORDERS[order_index % len(BALANCED_ACTION_ORDERS)]
    actions = []
    for action_index, key in enumerate(order, start=1):
        text, direction = action_specs[key]
        actions.append(
            {
                "id": f"{scenario_id}_act_{action_index:03d}",
                "text": text,
                "trait_direction": direction,
                "source_option": key,
            }
        )
    return {
        "scenario_id": scenario_id,
        "trait": trait,
        "prompt": str(row["question"]).strip(),
        "actions": actions,
        "split": split,
        "source_personality": str(row["personality"]).strip(),
    }


def assign_split(
    index: int,
    total: int,
    split_ratios: tuple[float, float, float],
) -> str:
    """Assign train/dev/test by stable per-trait index."""

    if total <= 0:
        raise ValueError("total must be positive")
    train_ratio, dev_ratio, test_ratio = split_ratios
    ratio_sum = train_ratio + dev_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")
    train_count = int(total * train_ratio)
    dev_count = int(total * dev_ratio)
    if total >= 3:
        train_count = max(1, train_count)
        dev_count = max(1, dev_count)
    if index < train_count:
        return "train"
    if index < train_count + dev_count:
        return "dev"
    return "test"


def write_normalized_scenarios(rows: Iterable[dict], path: str | Path) -> int:
    """Write normalized scenario rows to JSONL."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_split_index(rows: Iterable[Mapping[str, object]]) -> dict[str, dict[str, list[str]]]:
    """Build ``split -> trait -> scenario ids`` index for auditability."""

    index: dict[str, dict[str, list[str]]] = {
        "train": defaultdict(list),
        "dev": defaultdict(list),
        "test": defaultdict(list),
    }
    for row in rows:
        split = str(row.get("split") or "").strip().lower()
        trait = str(row.get("trait") or "").strip().lower()
        scenario_id = str(row.get("scenario_id") or "").strip()
        if split not in index:
            raise ValueError(f"Unknown split in normalized TRAIT row: {split!r}")
        if not trait or not scenario_id:
            raise ValueError("Normalized TRAIT row missing trait or scenario_id")
        index[split][trait].append(scenario_id)

    return {
        split: {trait: ids for trait, ids in sorted(traits.items())}
        for split, traits in index.items()
    }


def write_split_index(rows: Iterable[Mapping[str, object]], path: str | Path) -> None:
    """Write split index JSON for converted TRAIT scenarios."""

    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_split_index(rows), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
