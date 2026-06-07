"""Analysis helpers for v4 Big Five pilot/final metric CSVs."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping


GROUP_KEYS = ("condition", "trait", "target_direction")


def load_csv_rows(path: str | Path) -> list[dict]:
    """Load CSV rows as dictionaries."""

    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def load_json_file(path: str | Path) -> dict:
    """Load a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean(values: Iterable[float]) -> float:
    """Return arithmetic mean, raising on empty input."""

    values = list(values)
    if not values:
        raise ValueError("Cannot compute mean of empty values")
    return float(sum(values) / len(values))


def summarize_consistency(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    """Summarize consistency rows by condition, trait, and target direction."""

    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in GROUP_KEYS)
        groups[key].append(row)

    summary = []
    for (condition, trait, target_direction), group in sorted(groups.items()):
        summary.append(
            {
                "condition": condition,
                "trait": trait,
                "target_direction": target_direction,
                "n": len(group),
                "mean_consistency": mean(float(row["consistency"]) for row in group),
                "mean_bfi_score": mean(float(row["bfi_score"]) for row in group),
                "mean_trait_score": mean(float(row["trait_score"]) for row in group),
                "mean_bfi_target_attainment": mean(
                    float(row["bfi_target_attainment"]) for row in group
                ),
                "mean_trait_target_attainment": mean(
                    float(row["trait_target_attainment"]) for row in group
                ),
            }
        )
    return summary


def condition_differences(
    summary_rows: Iterable[Mapping[str, object]],
    reference_condition: str = "baseline",
) -> list[dict]:
    """Compute condition-level differences versus reference summary rows."""

    summary = list(summary_rows)
    reference = {
        (str(row["trait"]), str(row["target_direction"])): row
        for row in summary
        if row["condition"] == reference_condition
    }
    rows = []
    for row in summary:
        condition = str(row["condition"])
        if condition == reference_condition:
            continue
        key = (str(row["trait"]), str(row["target_direction"]))
        ref = reference.get(key)
        if ref is None:
            continue
        rows.append(
            {
                "condition": condition,
                "reference_condition": reference_condition,
                "trait": key[0],
                "target_direction": key[1],
                "delta_consistency": float(row["mean_consistency"])
                - float(ref["mean_consistency"]),
                "delta_bfi_target_attainment": float(row["mean_bfi_target_attainment"])
                - float(ref["mean_bfi_target_attainment"]),
                "delta_trait_target_attainment": float(row["mean_trait_target_attainment"])
                - float(ref["mean_trait_target_attainment"]),
            }
        )
    return rows


def condition_effects_with_ci(
    difference_rows: Iterable[Mapping[str, object]],
    bootstrap_rows: Iterable[Mapping[str, object]],
) -> list[dict]:
    """Attach paired bootstrap intervals to baseline-referenced effect rows."""

    ci_by_key = {
        (
            str(row["condition"]),
            str(row["reference_condition"]),
            str(row["trait"]),
            str(row["target_direction"]),
            str(row["metric"]),
        ): row
        for row in bootstrap_rows
    }
    rows = []
    for row in difference_rows:
        condition = str(row["condition"])
        reference = str(row["reference_condition"])
        trait = str(row["trait"])
        target = str(row["target_direction"])
        merged = dict(row)
        attach_ci(
            merged,
            ci_by_key.get((condition, reference, trait, target, "consistency")),
            prefix="consistency",
        )
        attach_ci(
            merged,
            ci_by_key.get(
                (condition, reference, trait, target, "trait_target_attainment")
            ),
            prefix="trait_target",
        )
        rows.append(merged)
    return rows


def attach_ci(row: dict, ci_row: Mapping[str, object] | None, prefix: str) -> None:
    """Attach lower/upper interval fields to one effect row."""

    if ci_row is None:
        return
    lower = float(ci_row["ci_lower"])
    upper = float(ci_row["ci_upper"])
    row[f"{prefix}_ci_lower"] = lower
    row[f"{prefix}_ci_upper"] = upper
    row[f"{prefix}_ci_excludes_zero"] = lower > 0.0 or upper < 0.0


def paired_scenario_differences(
    rows: Iterable[Mapping[str, object]],
    reference_condition: str = "baseline",
) -> list[dict]:
    """Compute paired scenario-level consistency differences versus reference."""

    by_key = {}
    for row in rows:
        key = (
            str(row["condition"]),
            str(row["trait"]),
            str(row["target_direction"]),
            str(row["scenario_id"]),
        )
        by_key[key] = row

    diffs_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (condition, trait, target_direction, scenario_id), row in by_key.items():
        if condition == reference_condition:
            continue
        ref = by_key.get((reference_condition, trait, target_direction, scenario_id))
        if ref is None:
            continue
        diffs_by_group[(condition, trait, target_direction)].append(
            float(row["consistency"]) - float(ref["consistency"])
        )

    return [
        {
            "condition": condition,
            "reference_condition": reference_condition,
            "trait": trait,
            "target_direction": target_direction,
            "n_paired_scenarios": len(values),
            "mean_paired_delta_consistency": mean(values),
        }
        for (condition, trait, target_direction), values in sorted(diffs_by_group.items())
    ]


def paired_bootstrap_ci(
    rows: Iterable[Mapping[str, object]],
    reference_condition: str = "baseline",
    metric_column: str = "consistency",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> list[dict]:
    """Bootstrap paired scenario deltas against a reference condition."""

    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be between 0 and 1")

    deltas_by_group = paired_metric_deltas(
        rows,
        reference_condition=reference_condition,
        metric_column=metric_column,
    )
    rng = random.Random(seed)
    alpha = (1.0 - ci) / 2.0
    lower_q = alpha
    upper_q = 1.0 - alpha
    summary = []
    for (condition, trait, target_direction), deltas in sorted(deltas_by_group.items()):
        boot_means = []
        for _ in range(n_bootstrap):
            sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
            boot_means.append(mean(sample))
        boot_means.sort()
        summary.append(
            {
                "condition": condition,
                "reference_condition": reference_condition,
                "trait": trait,
                "target_direction": target_direction,
                "metric": metric_column,
                "n_paired_scenarios": len(deltas),
                "mean_delta": mean(deltas),
                "ci": ci,
                "ci_lower": quantile_sorted(boot_means, lower_q),
                "ci_upper": quantile_sorted(boot_means, upper_q),
                "n_bootstrap": n_bootstrap,
                "seed": seed,
            }
        )
    return summary


def paired_metric_deltas(
    rows: Iterable[Mapping[str, object]],
    reference_condition: str,
    metric_column: str,
) -> dict[tuple[str, str, str], list[float]]:
    """Return paired scenario deltas grouped by condition, trait, and target."""

    by_key = {}
    for row in rows:
        key = (
            str(row["condition"]),
            str(row["trait"]),
            str(row["target_direction"]),
            str(row["scenario_id"]),
        )
        by_key[key] = row

    deltas_by_group: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for (condition, trait, target_direction, scenario_id), row in by_key.items():
        if condition == reference_condition:
            continue
        ref = by_key.get((reference_condition, trait, target_direction, scenario_id))
        if ref is None:
            continue
        deltas_by_group[(condition, trait, target_direction)].append(
            float(row[metric_column]) - float(ref[metric_column])
        )
    return deltas_by_group


def quantile_sorted(values: list[float], q: float) -> float:
    """Return a linear-interpolated quantile from sorted values."""

    if not values:
        raise ValueError("Cannot compute quantile of empty values")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be between 0 and 1")
    if len(values) == 1:
        return float(values[0])
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def summarize_pas_agreement(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    """Summarize PAS/loglik agreement rows by condition, trait, and direction."""

    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in GROUP_KEYS)
        groups[key].append(row)
    summary = []
    for (condition, trait, target_direction), group in sorted(groups.items()):
        agreements = [
            parse_bool(row.get("pas_loglik_agreement", False))
            for row in group
        ]
        summary.append(
            {
                "condition": condition,
                "trait": trait,
                "target_direction": target_direction,
                "n": len(group),
                "pas_loglik_agreement_rate": sum(agreements) / len(agreements)
                if agreements
                else 0.0,
            }
        )
    return summary


def summarize_side_effects(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    """Summarize side-effect metric rows by condition, trait, and direction."""

    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in GROUP_KEYS)
        groups[key].append(row)

    summary = []
    for (condition, trait, target_direction), group in sorted(groups.items()):
        summary.append(
            {
                "condition": condition,
                "trait": trait,
                "target_direction": target_direction,
                "n": len(group),
                "parse_success_rate": mean(
                    1.0 if parse_bool(row.get("parse_success", False)) else 0.0
                    for row in group
                ),
                "format_validity_rate": mean(
                    1.0 if parse_bool(row.get("format_validity", False)) else 0.0
                    for row in group
                ),
                "speech_tag_rate": mean(
                    1.0 if parse_bool(row.get("has_speech_tag", False)) else 0.0
                    for row in group
                ),
                "action_tag_rate": mean(
                    1.0 if parse_bool(row.get("has_action_tag", False)) else 0.0
                    for row in group
                ),
                "mean_response_length_chars": mean(
                    float(row.get("response_length_chars", 0.0)) for row in group
                ),
                "mean_response_length_words": mean(
                    float(row.get("response_length_words", 0.0)) for row in group
                ),
            }
        )
    return summary


def parse_bool(value: object) -> bool:
    """Parse bool-like CSV values."""

    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def write_csv_rows(rows: Iterable[Mapping[str, object]], path: str | Path) -> int:
    """Write dictionaries to CSV."""

    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return 0
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)
    return len(rows)


def format_signed(value: object) -> str:
    """Format numeric values with an explicit sign."""

    return f"{float(value):+.3f}"


def format_ci(row: Mapping[str, object], prefix: str) -> str:
    """Format a CI attached by ``condition_effects_with_ci``."""

    lower_key = f"{prefix}_ci_lower"
    upper_key = f"{prefix}_ci_upper"
    if lower_key not in row or upper_key not in row:
        return "n/a"
    return f"[{float(row[lower_key]):+.3f}, {float(row[upper_key]):+.3f}]"


def build_markdown_report(
    summary_rows: Iterable[Mapping[str, object]],
    difference_rows: Iterable[Mapping[str, object]],
    paired_rows: Iterable[Mapping[str, object]],
    bootstrap_rows: Iterable[Mapping[str, object]] | None = None,
    condition_effect_rows: Iterable[Mapping[str, object]] | None = None,
    pas_rows: Iterable[Mapping[str, object]] | None = None,
    side_effect_rows: Iterable[Mapping[str, object]] | None = None,
) -> str:
    """Build a compact Markdown report from v4 analysis rows."""

    summary_rows = list(summary_rows)
    difference_rows = list(difference_rows)
    paired_rows = list(paired_rows)
    bootstrap_rows = list(bootstrap_rows or [])
    condition_effect_rows = list(
        condition_effect_rows
        if condition_effect_rows is not None
        else condition_effects_with_ci(difference_rows, bootstrap_rows)
    )
    pas_rows = list(pas_rows or [])
    side_effect_rows = list(side_effect_rows or [])
    lines = [
        "# V4 Big Five Analysis",
        "",
        "## Consistency Summary",
        "",
        "| condition | trait | target | n | consistency | BFI target | TRAIT target |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {condition} | {trait} | {target_direction} | {n} | {mean_consistency:.3f} | "
            "{mean_bfi_target_attainment:.3f} | {mean_trait_target_attainment:.3f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Differences vs Baseline",
            "",
            "| condition | trait | target | delta consistency | consistency 95% CI | delta BFI target | delta TRAIT target | TRAIT target 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in condition_effect_rows:
        lines.append(
            "| {condition} | {trait} | {target_direction} | {delta_consistency} | "
            "{consistency_ci} | {delta_bfi_target_attainment} | "
            "{delta_trait_target_attainment} | {trait_target_ci} |".format(
                condition=row["condition"],
                trait=row["trait"],
                target_direction=row["target_direction"],
                delta_consistency=format_signed(row["delta_consistency"]),
                consistency_ci=format_ci(row, "consistency"),
                delta_bfi_target_attainment=format_signed(
                    row["delta_bfi_target_attainment"]
                ),
                delta_trait_target_attainment=format_signed(
                    row["delta_trait_target_attainment"]
                ),
                trait_target_ci=format_ci(row, "trait_target"),
            )
        )

    lines.extend(
        [
            "",
            "## Paired Scenario Differences",
            "",
            "| condition | trait | target | paired n | mean paired delta consistency |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            "| {condition} | {trait} | {target_direction} | {n_paired_scenarios} | "
            "{mean_paired_delta_consistency:.3f} |".format(**row)
        )

    if bootstrap_rows:
        lines.extend(
            [
                "",
                "## Paired Bootstrap CI",
                "",
                "| condition | trait | target | metric | n | mean delta | CI lower | CI upper |",
                "|---|---|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in bootstrap_rows:
            lines.append(
                "| {condition} | {trait} | {target_direction} | {metric} | "
                "{n_paired_scenarios} | {mean_delta:.3f} | {ci_lower:.3f} | "
                "{ci_upper:.3f} |".format(**row)
            )

    if pas_rows:
        lines.extend(
            [
                "",
                "## PAS/Loglik Agreement",
                "",
                "| condition | trait | target | n | agreement rate |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in pas_rows:
            lines.append(
                "| {condition} | {trait} | {target_direction} | {n} | "
                "{pas_loglik_agreement_rate:.3f} |".format(**row)
            )

    if side_effect_rows:
        lines.extend(
            [
                "",
                "## Side-Effect Summary",
                "",
                "| condition | trait | target | n | format validity | parse success | words |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in side_effect_rows:
            lines.append(
                "| {condition} | {trait} | {target_direction} | {n} | "
                "{format_validity_rate:.3f} | {parse_success_rate:.3f} | "
                "{mean_response_length_words:.1f} |".format(**row)
            )

    lines.append("")
    return "\n".join(lines)


def build_final_report(
    *,
    manifest: Mapping[str, object] | None,
    summary_rows: Iterable[Mapping[str, object]],
    difference_rows: Iterable[Mapping[str, object]],
    bootstrap_rows: Iterable[Mapping[str, object]],
    condition_effect_rows: Iterable[Mapping[str, object]] | None = None,
    pas_rows: Iterable[Mapping[str, object]] | None = None,
    side_effect_rows: Iterable[Mapping[str, object]] | None = None,
) -> str:
    """Build a publication-oriented final report skeleton."""

    summary_rows = list(summary_rows)
    difference_rows = list(difference_rows)
    bootstrap_rows = list(bootstrap_rows)
    condition_effect_rows = list(
        condition_effect_rows
        if condition_effect_rows is not None
        else condition_effects_with_ci(difference_rows, bootstrap_rows)
    )
    pas_rows = list(pas_rows or [])
    side_effect_rows = list(side_effect_rows or [])
    manifest = dict(manifest or {})
    extra = dict(manifest.get("extra", {}) or {})

    lines = [
        "# V4 Big Five Speech-Action Consistency Report",
        "",
        "## Research Question",
        "",
        (
            "AS+PAS가 NPC에게 부여된 Big Five 페르소나에서 말(BFI 자기보고)과 "
            "행동(TRAIT 선택) 사이의 일관성에 어떤 조건에서 영향을 주는가?"
        ),
        "",
        "## Design Guardrails",
        "",
        "- This report measures NPC persona consistency, not LLM morality or real personality.",
        "- High and low trait directions are treated as value-neutral NPC character settings.",
        "- Consistency is interpreted with target attainment and side-effect metrics.",
        "- Test-split results must not be used to retune alpha, layers, prompts, or vector construction.",
        "",
        "## Metric Definitions",
        "",
        "- BFI_score and TRAIT_score are normalized to [0, 1] on the same high-direction Big Five axis.",
        "- speech-action consistency = 1 - abs(BFI_score - TRAIT_score). This is an unsigned gap-closeness score, not a signed movement-toward-target score.",
        "- target attainment = target-direction score - 0.5. Low targets use 1 - high_direction_score before subtracting 0.5.",
        "- In the current final run, the primary TRAIT_score is computed from conditional log-likelihood under each condition. PAS selections are reported as agreement diagnostics and are not used to overwrite the primary TRAIT_score.",
        "",
        "## Run Metadata",
        "",
        f"- run_id: `{manifest.get('run_id', 'unknown')}`",
        f"- model_name: `{manifest.get('model_name', 'unknown')}`",
        f"- git_commit: `{manifest.get('git_commit', 'unknown')}`",
        f"- data_split: `{manifest.get('data_split', 'unknown')}`",
        f"- seed: `{manifest.get('seed', 'unknown')}`",
        f"- alpha: `{manifest.get('alpha', 'unknown')}`",
        f"- layers: `{manifest.get('layers', 'unknown')}`",
        f"- vector_hash: `{manifest.get('vector_hash', 'none')}`",
        f"- config_sha256: `{extra.get('config_sha256', 'unknown')}`",
        f"- model_config_sha256: `{extra.get('model_config_sha256', 'unknown')}`",
        "",
        "## Primary Results",
        "",
        "| condition | trait | target | n | consistency | BFI target | TRAIT target |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    lines[4] = (
        "Under what conditions does AS+PAS affect speech-action persona "
        "consistency for NPCs assigned a Big Five persona?"
    )
    for row in summary_rows:
        lines.append(
            "| {condition} | {trait} | {target_direction} | {n} | {mean_consistency:.3f} | "
            "{mean_bfi_target_attainment:.3f} | {mean_trait_target_attainment:.3f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Baseline-Referenced Effects",
            "",
            "| condition | trait | target | delta consistency | consistency 95% CI | delta BFI target | delta TRAIT target | TRAIT target 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in condition_effect_rows:
        lines.append(
            "| {condition} | {trait} | {target_direction} | {delta_consistency} | "
            "{consistency_ci} | {delta_bfi_target_attainment} | "
            "{delta_trait_target_attainment} | {trait_target_ci} |".format(
                condition=row["condition"],
                trait=row["trait"],
                target_direction=row["target_direction"],
                delta_consistency=format_signed(row["delta_consistency"]),
                consistency_ci=format_ci(row, "consistency"),
                delta_bfi_target_attainment=format_signed(
                    row["delta_bfi_target_attainment"]
                ),
                delta_trait_target_attainment=format_signed(
                    row["delta_trait_target_attainment"]
                ),
                trait_target_ci=format_ci(row, "trait_target"),
            )
        )

    lines.extend(
        [
            "",
            "## Paired Bootstrap Intervals",
            "",
            "| condition | trait | target | metric | n | mean delta | CI lower | CI upper |",
            "|---|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in bootstrap_rows:
        lines.append(
            "| {condition} | {trait} | {target_direction} | {metric} | "
            "{n_paired_scenarios} | {mean_delta:.3f} | {ci_lower:.3f} | "
            "{ci_upper:.3f} |".format(**row)
        )

    if pas_rows:
        lines.extend(
            [
                "",
                "## PAS/Loglik Agreement",
                "",
                "| condition | trait | target | n | agreement rate |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in pas_rows:
            lines.append(
                "| {condition} | {trait} | {target_direction} | {n} | "
                "{pas_loglik_agreement_rate:.3f} |".format(**row)
            )

    if side_effect_rows:
        lines.extend(
            [
                "",
                "## Side-Effect Checks",
                "",
                "| condition | trait | target | n | format validity | parse success | mean words |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in side_effect_rows:
            lines.append(
                "| {condition} | {trait} | {target_direction} | {n} | "
                "{format_validity_rate:.3f} | {parse_success_rate:.3f} | "
                "{mean_response_length_words:.1f} |".format(**row)
            )

    lines.extend(
        [
            "",
            "## Interpretation Template",
            "",
            "- Positive result: AS+PAS improved speech-action consistency under specific trait/condition settings without degrading format validity.",
            "- Mixed result: AS+PAS effects depended on trait direction, prompt strength, or vector-control diagnostics.",
            "- Negative result: AS+PAS did not reliably reduce the BFI-TRAIT gap; this still constrains when activation steering is useful for NPC persona consistency.",
            "",
            "## Claims To Avoid",
            "",
            "- Do not claim this measures LLM morality.",
            "- Do not claim this measures the model's real personality.",
            "- Do not claim AS+PAS always improves consistency unless all relevant conditions support it.",
            "- Do not rank high trait directions as better than low trait directions.",
            "",
        ]
    )
    return "\n".join(lines)
