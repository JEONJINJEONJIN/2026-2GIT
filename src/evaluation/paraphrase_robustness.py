"""Paraphrase robustness metrics for v4 Big Five scenario variants."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from src.evaluation.bigfive_analysis import mean

GROUP_KEYS = ("condition", "trait", "target_direction")


def summarize_paraphrase_robustness(
    rows: Iterable[Mapping[str, Any]],
    *,
    group_column: str = "paraphrase_group_id",
    variant_column: str = "paraphrase_variant",
    action_column: str = "loglik_selected_action",
    score_column: str = "trait_score",
) -> list[dict[str, Any]]:
    """Summarize action and score stability across paraphrase variants.

    Each paraphrase group should contain two or more rows with the same
    condition/trait/target_direction and different surface forms of the same
    underlying scenario.
    """

    grouped_rows = _group_rows(rows, group_column)
    stability_by_condition: dict[tuple[str, str, str], list[dict[str, float]]] = (
        defaultdict(list)
    )

    for key, group in sorted(grouped_rows.items()):
        if len(group) < 2:
            continue
        reference = _reference_row(group, variant_column)
        actions = [str(row.get(action_column, "")) for row in group]
        reference_action = str(reference.get(action_column, ""))
        scores = [float(row[score_column]) for row in group]
        same_action_count = sum(action == reference_action for action in actions)
        stability_by_condition[key[:3]].append(
            {
                "action_agreement": same_action_count / len(group),
                "trait_score_range": max(scores) - min(scores),
                "trait_score_std_proxy": mean(abs(score - mean(scores)) for score in scores),
            }
        )

    summary = []
    for (condition, trait, target_direction), values in sorted(
        stability_by_condition.items()
    ):
        summary.append(
            {
                "condition": condition,
                "trait": trait,
                "target_direction": target_direction,
                "n_paraphrase_groups": len(values),
                "mean_action_agreement": mean(
                    item["action_agreement"] for item in values
                ),
                "mean_trait_score_range": mean(
                    item["trait_score_range"] for item in values
                ),
                "mean_trait_score_std_proxy": mean(
                    item["trait_score_std_proxy"] for item in values
                ),
            }
        )
    return summary


def _group_rows(
    rows: Iterable[Mapping[str, Any]],
    group_column: str,
) -> dict[tuple[str, str, str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = str(row.get(group_column, "")).strip()
        if not group_id:
            continue
        key = (
            str(row["condition"]),
            str(row["trait"]),
            str(row["target_direction"]),
            group_id,
        )
        grouped[key].append(row)
    return grouped


def _reference_row(
    rows: list[Mapping[str, Any]],
    variant_column: str,
) -> Mapping[str, Any]:
    for row in rows:
        if str(row.get(variant_column, "")).strip().lower() in {
            "original",
            "source",
            "base",
            "0",
        }:
            return row
    return rows[0]
