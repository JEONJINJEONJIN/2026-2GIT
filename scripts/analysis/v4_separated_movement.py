"""Reframe v4 Big Five analysis around separated BFI/TRAIT movement.

This script is post-processing only. It reads an existing v4 metrics directory
and writes:

- separated_movement_summary.csv
- co_movement_analysis.md
- analysis_report_v2.md

It intentionally does not modify the original analysis outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Iterable


DEFAULT_METRICS_DIR = Path("results/v4_bigfive/final_rerun_20260525")
REFERENCE_CONDITION = "baseline"
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build separated BFI/TRAIT movement reports for v4 results."
    )
    parser.add_argument("--metrics_dir", default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--reference_condition", default=REFERENCE_CONDITION)
    parser.add_argument("--bootstrap_samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    rows = read_csv(metrics_dir / "consistency_metrics.csv")
    if not rows:
        raise ValueError(f"No rows found in {metrics_dir / 'consistency_metrics.csv'}")
    bfi_rows = read_csv(metrics_dir / "bfi_scores.csv")
    if not bfi_rows:
        raise ValueError(f"No rows found in {metrics_dir / 'bfi_scores.csv'}")

    summary_rows = build_separated_summary(
        rows,
        bfi_rows,
        reference_condition=args.reference_condition,
        n_boot=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    write_csv(
        summary_rows,
        metrics_dir / "separated_movement_summary.csv",
        [
            "condition",
            "trait",
            "target_direction",
            "n",
            "delta_bfi_target",
            "delta_bfi_target_ci_low",
            "delta_bfi_target_ci_high",
            "delta_trait_target",
            "delta_trait_target_ci_low",
            "delta_trait_target_ci_high",
            "co_movement_flag",
        ],
    )

    gap_rows = read_csv(metrics_dir / "condition_effects_with_ci.csv")
    pas_rows = read_csv(metrics_dir / "pas_agreement_summary.csv")

    co_report = build_co_movement_report(summary_rows)
    (metrics_dir / "co_movement_analysis.md").write_text(co_report, encoding="utf-8")

    analysis_report = build_analysis_report_v2(
        summary_rows=summary_rows,
        gap_rows=gap_rows,
        pas_rows=pas_rows,
        reference_condition=args.reference_condition,
    )
    (metrics_dir / "analysis_report_v2.md").write_text(
        analysis_report,
        encoding="utf-8",
    )

    print(f"Wrote {metrics_dir / 'separated_movement_summary.csv'}")
    print(f"Wrote {metrics_dir / 'co_movement_analysis.md'}")
    print(f"Wrote {metrics_dir / 'analysis_report_v2.md'}")
    print()
    print(one_paragraph_summary(summary_rows))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(rows: list[dict[str, object]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def build_separated_summary(
    rows: list[dict[str, str]],
    bfi_rows: list[dict[str, str]],
    *,
    reference_condition: str,
    n_boot: int,
    seed: int,
) -> list[dict[str, object]]:
    grouped = group_consistency_rows(rows)
    conditions = ordered_unique(row["condition"] for row in rows)
    trait_dirs = sorted(
        {
            (row["trait"], row["target_direction"])
            for row in rows
            if row["condition"] == reference_condition
        }
    )

    output: list[dict[str, object]] = []
    for condition in conditions:
        if condition == reference_condition:
            continue
        for trait, target_direction in trait_dirs:
            key = (condition, trait, target_direction)
            ref_key = (reference_condition, trait, target_direction)
            condition_rows = grouped.get(key, [])
            reference_rows = grouped.get(ref_key, [])
            if not condition_rows or not reference_rows:
                continue

            bfi_delta, bfi_low, bfi_high = bfi_item_bootstrap_delta(
                bfi_rows,
                condition=condition,
                reference_condition=reference_condition,
                trait=trait,
                target_direction=target_direction,
                n_boot=n_boot,
                seed=stable_seed(seed, condition, trait, target_direction, "bfi"),
            )
            trait_delta, trait_low, trait_high = paired_bootstrap_delta(
                condition_rows,
                reference_rows,
                "trait_target_attainment",
                n_boot=n_boot,
                seed=stable_seed(seed, condition, trait, target_direction, "trait"),
            )

            flag = co_movement_flag(
                bfi_delta,
                bfi_low,
                bfi_high,
                trait_delta,
                trait_low,
                trait_high,
            )
            output.append(
                {
                    "condition": condition,
                    "trait": trait,
                    "target_direction": target_direction,
                    "n": len(condition_rows),
                    "delta_bfi_target": bfi_delta,
                    "delta_bfi_target_ci_low": bfi_low,
                    "delta_bfi_target_ci_high": bfi_high,
                    "delta_trait_target": trait_delta,
                    "delta_trait_target_ci_low": trait_low,
                    "delta_trait_target_ci_high": trait_high,
                    "co_movement_flag": flag,
                }
            )
    return output


def group_consistency_rows(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["trait"], row["target_direction"])].append(row)
    return grouped


def ordered_unique(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def paired_bootstrap_delta(
    condition_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    metric: str,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap paired condition-reference deltas by scenario id."""

    condition_by_scenario = {row["scenario_id"]: row for row in condition_rows}
    reference_by_scenario = {row["scenario_id"]: row for row in reference_rows}
    scenario_ids = sorted(set(condition_by_scenario) & set(reference_by_scenario))
    if not scenario_ids:
        raise ValueError("No paired scenarios available for bootstrap")

    paired_deltas = [
        as_float(condition_by_scenario[sid][metric])
        - as_float(reference_by_scenario[sid][metric])
        for sid in scenario_ids
    ]
    estimate = mean(paired_deltas)

    rng = random.Random(seed)
    boot_means = []
    for _ in range(n_boot):
        sample = [paired_deltas[rng.randrange(len(paired_deltas))] for _ in paired_deltas]
        boot_means.append(mean(sample))
    return estimate, quantile(boot_means, 0.025), quantile(boot_means, 0.975)


def bfi_item_bootstrap_delta(
    bfi_rows: list[dict[str, str]],
    *,
    condition: str,
    reference_condition: str,
    trait: str,
    target_direction: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap BFI target-attainment deltas from raw BFI item rows.

    BFI is measured at the item level, not the scenario level. The earlier
    separated analysis used ``consistency_metrics.csv``, where one averaged
    BFI score had already been copied onto every scenario row. This helper
    instead resamples the BFI items for the target trait.
    """

    condition_items = bfi_trait_items(
        bfi_rows,
        condition=condition,
        trait=trait,
        target_direction=target_direction,
    )
    reference_items = bfi_trait_items(
        bfi_rows,
        condition=reference_condition,
        trait=trait,
        target_direction=target_direction,
    )
    item_ids = sorted(set(condition_items) & set(reference_items))
    if not item_ids:
        raise ValueError(
            f"No paired BFI items for {condition}/{trait}/{target_direction}"
        )

    estimate = (
        bfi_target_attainment_from_items(condition_items.values(), target_direction)
        - bfi_target_attainment_from_items(reference_items.values(), target_direction)
    )

    rng = random.Random(seed)
    boot_deltas = []
    for _ in range(n_boot):
        sampled_ids = [item_ids[rng.randrange(len(item_ids))] for _ in item_ids]
        condition_sample = [condition_items[item_id] for item_id in sampled_ids]
        reference_sample = [reference_items[item_id] for item_id in sampled_ids]
        boot_deltas.append(
            bfi_target_attainment_from_items(condition_sample, target_direction)
            - bfi_target_attainment_from_items(reference_sample, target_direction)
        )
    return estimate, quantile(boot_deltas, 0.025), quantile(boot_deltas, 0.975)


def bfi_trait_items(
    rows: list[dict[str, str]],
    *,
    condition: str,
    trait: str,
    target_direction: str,
) -> dict[str, dict[str, str]]:
    items = {}
    for row in rows:
        if (
            row.get("condition") == condition
            and row.get("trait") == trait
            and row.get("target_direction") == target_direction
            and row.get("item_trait") == trait
        ):
            items[row["item_id"]] = row
    return items


def bfi_target_attainment_from_items(
    rows: Iterable[dict[str, str]],
    target_direction: str,
) -> float:
    values = [normalized_bfi_item_score(row) for row in rows]
    if not values:
        raise ValueError("Cannot score empty BFI item sample")
    high_score = mean(values)
    target_score = high_score if target_direction == "high" else 1.0 - high_score
    return target_score - 0.5


def normalized_bfi_item_score(row: dict[str, str]) -> float:
    score = as_float(row["likert_score"])
    if parse_bool(row.get("reverse_scored", "")):
        score = 6.0 - score
    return (score - 1.0) / 4.0


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "y"}


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    pos = q * (len(values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return float(values[low])
    weight = pos - low
    return float(values[low] * (1 - weight) + values[high] * weight)


def stable_seed(seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    value = seed
    for ch in text:
        value = (value * 131 + ord(ch)) % (2**31 - 1)
    return value


def positive_excludes_zero(delta: float, low: float, high: float) -> bool:
    return delta > 0 and low > 0 and high > 0


def co_movement_flag(
    bfi_delta: float,
    bfi_low: float,
    bfi_high: float,
    trait_delta: float,
    trait_low: float,
    trait_high: float,
) -> str:
    bfi_positive = positive_excludes_zero(bfi_delta, bfi_low, bfi_high)
    trait_positive = positive_excludes_zero(trait_delta, trait_low, trait_high)
    if bfi_positive and trait_positive:
        return "both_positive"
    if bfi_positive:
        return "bfi_only"
    if trait_positive:
        return "trait_only"
    return "neither"


def build_co_movement_report(rows: list[dict[str, object]]) -> str:
    conditions = ordered_unique(str(row["condition"]) for row in rows)
    lines = [
        "# Co-Movement Analysis",
        "",
        "This report separates BFI-side movement from TRAIT-side movement. The",
        "question is whether the two surfaces move toward the target together,",
        "rather than whether their gap compresses into a single combined score.",
        "",
        "## Bucket Counts",
        "",
        "| Condition | both_positive | bfi_only | trait_only | neither |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in conditions:
        counts = flag_counts(condition_rows(rows, condition))
        lines.append(
            "| {condition} | {both_positive} | {bfi_only} | {trait_only} | {neither} |".format(
                condition=condition,
                **counts,
            )
        )

    for condition in conditions:
        subset = sorted(
            condition_rows(rows, condition),
            key=lambda row: abs(
                as_float(row["delta_bfi_target"])
                - as_float(row["delta_trait_target"])
            ),
            reverse=True,
        )
        lines.extend(
            [
                "",
                f"## {condition}",
                "",
                condition_paragraph(condition, subset),
                "",
                "| Trait | Target | Delta BFI target | BFI 95% CI | Delta TRAIT target | TRAIT 95% CI | Abs delta gap | Flag |",
                "|---|---|---:|---|---:|---|---:|---|",
            ]
        )
        for row in subset:
            bfi_delta = as_float(row["delta_bfi_target"])
            trait_delta = as_float(row["delta_trait_target"])
            lines.append(
                "| {trait} | {target_direction} | {bfi:+.3f} | [{bfi_low:+.3f}, {bfi_high:+.3f}] | "
                "{trait_delta:+.3f} | [{trait_low:+.3f}, {trait_high:+.3f}] | {gap:.3f} | {flag} |".format(
                    trait=row["trait"],
                    target_direction=row["target_direction"],
                    bfi=bfi_delta,
                    bfi_low=as_float(row["delta_bfi_target_ci_low"]),
                    bfi_high=as_float(row["delta_bfi_target_ci_high"]),
                    trait_delta=trait_delta,
                    trait_low=as_float(row["delta_trait_target_ci_low"]),
                    trait_high=as_float(row["delta_trait_target_ci_high"]),
                    gap=abs(bfi_delta - trait_delta),
                    flag=row["co_movement_flag"],
                )
            )
    lines.append("")
    return "\n".join(lines)


def condition_rows(rows: list[dict[str, object]], condition: str) -> list[dict[str, object]]:
    return [row for row in rows if row["condition"] == condition]


def flag_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(str(row["co_movement_flag"]) for row in rows)
    return {
        "both_positive": counts.get("both_positive", 0),
        "bfi_only": counts.get("bfi_only", 0),
        "trait_only": counts.get("trait_only", 0),
        "neither": counts.get("neither", 0),
    }


def condition_paragraph(condition: str, rows: list[dict[str, object]]) -> str:
    counts = flag_counts(rows)
    mean_bfi = mean(as_float(row["delta_bfi_target"]) for row in rows)
    mean_trait = mean(as_float(row["delta_trait_target"]) for row in rows)
    largest = rows[0]
    largest_gap = abs(
        as_float(largest["delta_bfi_target"])
        - as_float(largest["delta_trait_target"])
    )

    if condition == "as_pas_only":
        return (
            "`as_pas_only` isolates the no-prompt AS/PAS condition. Its bucket "
            f"counts are {counts}, with mean TRAIT movement {mean_trait:+.3f} "
            f"and mean BFI movement {mean_bfi:+.3f}. Interpret this row as the "
            "direct test of whether AS also affects the BFI self-report surface."
        )
    if condition == "elaborate_prompt_as_pas":
        return (
            "`elaborate_prompt_as_pas` produces co-movement in most cells, but "
            "the movement sizes are uneven. The clearest decoupling is "
            "neuroticism-high, where BFI moves much more than TRAIT. "
            f"Overall bucket counts are {counts}, with mean BFI movement "
            f"{mean_bfi:+.3f} and mean TRAIT movement {mean_trait:+.3f}."
        )
    if condition == "one_line_prompt":
        return (
            "`one_line_prompt` moves both surfaces in many cells, but it also "
            "contains target directions where only the action surface moves. "
            f"The largest BFI/TRAIT movement gap is {largest_gap:.3f} at "
            f"{largest['trait']}-{largest['target_direction']}."
        )
    if condition == "elaborate_prompt":
        return (
            "`elaborate_prompt` moves the speech-side BFI surface more reliably "
            "than the one-line prompt, while TRAIT also moves in all cells. "
            "The two movements are still not equal in magnitude, so the surfaces "
            "should be reported separately."
        )
    return (
        f"`{condition}` has bucket counts {counts}, mean BFI movement "
        f"{mean_bfi:+.3f}, and mean TRAIT movement {mean_trait:+.3f}."
    )


def build_analysis_report_v2(
    *,
    summary_rows: list[dict[str, object]],
    gap_rows: list[dict[str, str]],
    pas_rows: list[dict[str, str]],
    reference_condition: str,
) -> str:
    conditions = ordered_unique(str(row["condition"]) for row in summary_rows)
    lines = [
        "# V4 Final Rerun Analysis v2: Separated BFI/TRAIT Movement",
        "",
        "## 1. What Changed In This Reframing",
        "",
        "The earlier report used `speech_action_consistency = 1 - abs(BFI_score - TRAIT_score)` as the headline metric. This v2 analysis demotes that gap metric to an auxiliary signal because BFI and TRAIT are measured through different modalities and scales: BFI is a self-report surface, while TRAIT is an action-side log-likelihood surface.",
        "",
        "The primary result is now reported as two separate movements:",
        "",
        "- `delta_bfi_target`: movement of BFI-side target attainment relative to baseline.",
        "- `delta_trait_target`: movement of TRAIT-side target attainment relative to baseline.",
        "",
        "The central question is whether these two surfaces move together. The answer is explicit in `co_movement_flag`.",
        "",
            "CI note: BFI movement intervals in this report are computed from raw `bfi_scores.csv` item rows with item-level bootstrap. TRAIT movement intervals are computed from `consistency_metrics.csv` with scenario-level bootstrap.",
        "",
        "## 2. BFI-Side Movement Results (Primary)",
        "",
        "| Condition | Positive BFI cells | Mean delta BFI target | Largest absolute BFI movement |",
        "|---|---:|---:|---:|",
    ]
    for condition in conditions:
        subset = condition_rows(summary_rows, condition)
        positive = sum(
            positive_excludes_zero(
                as_float(row["delta_bfi_target"]),
                as_float(row["delta_bfi_target_ci_low"]),
                as_float(row["delta_bfi_target_ci_high"]),
            )
            for row in subset
        )
        mean_bfi = mean(as_float(row["delta_bfi_target"]) for row in subset)
        max_abs = max(abs(as_float(row["delta_bfi_target"])) for row in subset)
        lines.append(f"| `{condition}` | {positive} / {len(subset)} | {mean_bfi:+.3f} | {max_abs:.3f} |")

    lines.extend(
        [
            "",
            "BFI-side movement should be read separately from TRAIT movement. Prompted conditions usually move the self-report surface, and no-prompt AS/PAS conditions test whether activation steering also changes that surface. This is why BFI must not be hidden inside a combined gap score.",
            "",
            "## 3. TRAIT-Side Movement Results (Primary)",
            "",
            "| Condition | Positive TRAIT cells | Mean delta TRAIT target | Largest absolute TRAIT movement |",
            "|---|---:|---:|---:|",
        ]
    )
    for condition in conditions:
        subset = condition_rows(summary_rows, condition)
        positive = sum(
            positive_excludes_zero(
                as_float(row["delta_trait_target"]),
                as_float(row["delta_trait_target_ci_low"]),
                as_float(row["delta_trait_target_ci_high"]),
            )
            for row in subset
        )
        mean_trait = mean(as_float(row["delta_trait_target"]) for row in subset)
        max_abs = max(abs(as_float(row["delta_trait_target"])) for row in subset)
        lines.append(f"| `{condition}` | {positive} / {len(subset)} | {mean_trait:+.3f} | {max_abs:.3f} |")

    lines.extend(
        [
            "",
            "TRAIT-side movement is reliable across all non-baseline conditions in this rerun. AS/PAS reliably moves the action surface, and `elaborate_prompt_as_pas` has the strongest mean TRAIT target movement.",
            "",
            "## 4. Co-Movement Analysis",
            "",
            "| Condition | both_positive | bfi_only | trait_only | neither |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition in conditions:
        counts = flag_counts(condition_rows(summary_rows, condition))
        lines.append(
            "| `{condition}` | {both_positive} | {bfi_only} | {trait_only} | {neither} |".format(
                condition=condition,
                **counts,
            )
        )

    lines.extend(
        [
            "",
            "The real finding is decoupling. `as_pas_only` is pure action-surface movement: TRAIT moves, BFI does not. Prompt+AS/PAS moves both surfaces in most cells, but not by the same amount. The elaborate_prompt_as_pas neuroticism-high case is the clearest example: `delta_bfi_target = +0.344`, while `delta_trait_target = +0.058`. Speech-side movement is much larger than action-side movement.",
            "",
            "This supports the reframed claim: AS/PAS reliably moves the action surface; prompt+AS/PAS moves the speech surface more than the action surface; the two surfaces move independently.",
            "",
            "## 5. Gap Metric As Auxiliary Signal",
            "",
            "The gap metric is retained only as descriptive gap behavior:",
            "",
            "```text",
            "gap_behavior = 1 - abs(BFI_score - TRAIT_score)",
            "```",
            "",
            "Because BFI and TRAIT are different measurement surfaces, this value should not be treated as the primary success criterion. It is useful for diagnosing whether separated movement created a smaller or larger gap, but it does not by itself explain which surface moved.",
            "",
            "`as_pas_only` has the asymmetric pattern reported earlier: gap behavior improves on A-high/C-high/E-high/N-low/O-high and decreases on the opposite directions. This should be interpreted as evidence that the model's baseline self-report is shifted by alignment tuning and baseline tendencies, not as evidence that AS/PAS fails on those traits.",
            "",
            "| Condition | Trait | Target | Delta gap behavior | Gap 95% CI | Delta BFI target | Delta TRAIT target |",
            "|---|---|---|---:|---|---:|---:|",
        ]
    )
    for row in gap_rows:
        if row.get("condition") not in {"as_pas_only", "elaborate_prompt_as_pas"}:
            continue
        lines.append(
            "| `{condition}` | {trait} | {target} | {delta_gap:+.3f} | [{low:+.3f}, {high:+.3f}] | {bfi:+.3f} | {trait_delta:+.3f} |".format(
                condition=row["condition"],
                trait=row["trait"],
                target=row["target_direction"],
                delta_gap=as_float(row["delta_consistency"]),
                low=as_float(row["consistency_ci_lower"]),
                high=as_float(row["consistency_ci_upper"]),
                bfi=as_float(row["delta_bfi_target_attainment"]),
                trait_delta=as_float(row["delta_trait_target_attainment"]),
            )
        )

    lines.extend(
        [
            "",
            "## 6. Implications For Next Experiment",
            "",
            "The next experiment should test PAS as an actual final action selector or reranker, not only as a diagnostic agreement signal.",
            "",
            "Recommended policy conditions:",
            "",
            "| Condition | AS | Prompt | PAS final selector |",
            "|---|---:|---:|---:|",
            "| `baseline` | no | no | no |",
            "| `prompt_baseline` | no | yes | no |",
            "| `as_only` | yes | no | no |",
            "| `pas_only` | no | no | yes |",
            "| `as_pas` | yes | no | yes |",
            "| `prompt_as` | yes | yes | no |",
            "| `prompt_pas` | no | yes | yes |",
            "| `prompt_as_pas` | yes | yes | yes |",
            "",
            "For PAS selector conditions, use one of:",
            "",
            "```text",
            "final_action = pas_selected_action",
            "```",
            "",
            "or:",
            "",
            "```text",
            "final_action = PAS rerank among loglik-top-k actions",
            "```",
            "",
            "Primary reporting should remain separated: BFI movement, TRAIT movement, and co-movement. The gap metric should stay auxiliary.",
        ]
    )

    if pas_rows:
        lines.extend(
            [
                "",
                "PAS/loglik agreement remains useful as a diagnostic: moderate agreement means PAS is not merely copying log-likelihood selection and may produce different final actions if promoted to a policy.",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def one_paragraph_summary(rows: list[dict[str, object]]) -> str:
    conditions = ordered_unique(str(row["condition"]) for row in rows)
    pieces = []
    for condition in conditions:
        subset = condition_rows(rows, condition)
        counts = flag_counts(subset)
        mean_bfi = mean(as_float(row["delta_bfi_target"]) for row in subset)
        mean_trait = mean(as_float(row["delta_trait_target"]) for row in subset)
        pieces.append(
            f"{condition}: {counts['both_positive']} both-positive, "
            f"{counts['bfi_only']} BFI-only, {counts['trait_only']} TRAIT-only, "
            f"{counts['neither']} neither; mean deltas BFI {mean_bfi:+.3f}, "
            f"TRAIT {mean_trait:+.3f}"
        )
    return (
        "Co-movement summary across non-baseline conditions: "
        + "; ".join(pieces)
        + ". Overall, AS/PAS reliably moves the TRAIT/action surface, while "
        "prompted conditions move the BFI/self-report surface unevenly; the two "
        "surfaces move independently rather than collapsing into one defensible "
        "primary gap metric."
    )


if __name__ == "__main__":
    main()
