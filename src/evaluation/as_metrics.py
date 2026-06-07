"""AS-only evaluation metrics for persona-aligned NPC behavior."""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Iterable

import numpy as np
from scipy import stats


DEFAULT_SPEECH_KEYWORDS = {
    "aggressive": [
        "attack", "destroy", "crush", "kill", "fight", "threaten",
        "force", "fear", "burn", "seize", "obey", "submit", "war",
        "strike", "punish", "dominate", "break", "cut down",
    ],
    "cooperative": [
        "help", "peace", "together", "talk", "negotiate", "share",
        "trust", "support", "cooperate", "alliance", "fair", "listen",
        "understand", "assist", "protect", "welcome", "agree",
    ],
}


def classify_speech_heuristic(
    speech: str,
    keywords: dict[str, list[str]] | None = None,
) -> str:
    """Classify speech as aggressive, cooperative, or neutral."""
    keywords = keywords or DEFAULT_SPEECH_KEYWORDS
    text = speech.lower()
    scores = {
        label: sum(1 for keyword in words if keyword in text)
        for label, words in keywords.items()
    }
    if not scores or max(scores.values()) == 0:
        return "neutral"
    best = [label for label, score in scores.items() if score == max(scores.values())]
    return best[0] if len(best) == 1 else "neutral"


def build_action_alignment_map(scenarios: Iterable[dict]) -> dict[str, str]:
    """Build ``action_id -> persona_alignment`` from scenario definitions."""
    mapping: dict[str, str] = {}
    for scenario in scenarios:
        for action in scenario.get("actions", []):
            action_id = action.get("id")
            if action_id:
                mapping[action_id] = action.get("persona_alignment", "neutral")
    return mapping


def action_alignment_category(
    target_persona: str,
    action_id: str | None,
    action_alignment_map: dict[str, str],
) -> str:
    """Return aligned, neutral, misaligned, or unknown."""
    if not action_id:
        return "unknown"
    action_persona = action_alignment_map.get(action_id)
    if action_persona is None:
        return "unknown"
    if action_persona == "neutral":
        return "neutral"
    return "aligned" if action_persona == target_persona else "misaligned"


def speech_action_agreement(speech_label: str, action_persona: str) -> bool:
    """Check whether speech tone and action persona label match."""
    if action_persona not in {"aggressive", "cooperative", "neutral"}:
        return False
    return speech_label == action_persona


def action_entropy(action_ids: Iterable[str | None]) -> float:
    """Compute Shannon entropy over generated actions."""
    actions = [action or "unknown" for action in action_ids]
    if not actions:
        return 0.0
    counts = Counter(actions)
    total = len(actions)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return float(entropy)


def annotate_entries(
    entries: Iterable[dict],
    action_alignment_map: dict[str, str],
    speech_classifier: Callable[[str], str] | None = None,
) -> list[dict]:
    """Attach AS evaluation labels to raw result rows."""
    annotated = []
    for entry in entries:
        row = dict(entry)
        action_id = row.get("final_action") or row.get("parsed_action")
        action_persona = action_alignment_map.get(action_id, "unknown")
        row["action_persona"] = action_persona
        row["action_alignment_category"] = action_alignment_category(
            row.get("persona", ""),
            action_id,
            action_alignment_map,
        )
        if speech_classifier is not None:
            speech_label = speech_classifier(row.get("speech", ""))
            row["speech_label"] = speech_label
            row["speech_action_agreement"] = speech_action_agreement(
                speech_label,
                action_persona,
            )
        annotated.append(row)
    return annotated


def summarize_group(entries: list[dict]) -> dict:
    """Summarize one condition/persona group."""
    n = len(entries)
    if n == 0:
        return {
            "n": 0,
            "parse_ok_rate": 0.0,
            "persona_alignment_rate": 0.0,
            "neutral_action_rate": 0.0,
            "misaligned_action_rate": 0.0,
            "unknown_action_rate": 0.0,
            "action_entropy": 0.0,
            "unique_actions": 0,
        }

    categories = Counter(e.get("action_alignment_category", "unknown") for e in entries)
    parse_ok = sum(1 for e in entries if e.get("parse_ok"))
    actions = [e.get("final_action") for e in entries]
    return {
        "n": n,
        "parse_ok_rate": parse_ok / n,
        "persona_alignment_rate": categories.get("aligned", 0) / n,
        "neutral_action_rate": categories.get("neutral", 0) / n,
        "misaligned_action_rate": categories.get("misaligned", 0) / n,
        "unknown_action_rate": categories.get("unknown", 0) / n,
        "action_entropy": action_entropy(actions),
        "unique_actions": len(set(action for action in actions if action)),
    }


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a Wilson score interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half_width = (
        z
        * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
        / denom
    )
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def odds_ratio_ci(
    success: int,
    failure: int,
    ref_success: int,
    ref_failure: int,
    z: float = 1.959963984540054,
) -> tuple[float, float, float]:
    """Return a Wald CI for an odds ratio with Haldane-Anscombe correction."""
    a = success + 0.5
    b = failure + 0.5
    c = ref_success + 0.5
    d = ref_failure + 0.5
    odds_ratio = (a * d) / (b * c)
    se = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    log_or = math.log(odds_ratio)
    return (
        float(odds_ratio),
        float(math.exp(log_or - z * se)),
        float(math.exp(log_or + z * se)),
    )


def chance_alignment_rates(
    scenarios: Iterable[dict],
    scenario_ids: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute chance-level persona alignment from each scenario's action set."""
    allowed_ids = set(scenario_ids) if scenario_ids is not None else None
    totals: dict[str, list[float]] = {"aggressive": [], "cooperative": [], "neutral": []}
    for scenario in scenarios:
        if allowed_ids is not None and scenario.get("id") not in allowed_ids:
            continue
        actions = scenario.get("actions", [])
        if not actions:
            continue
        denom = len(actions)
        counts = Counter(action.get("persona_alignment", "neutral") for action in actions)
        for persona in totals:
            totals[persona].append(counts.get(persona, 0) / denom)
    return {
        persona: float(np.mean(values)) if values else 0.0
        for persona, values in totals.items()
    }


def build_quality_metrics(annotated_by_condition: dict[str, list[dict]]) -> list[dict]:
    """Build Table 1 quality metrics for parseability and unknown actions."""
    rows = []
    for condition, entries in annotated_by_condition.items():
        personas = sorted({entry.get("persona", "") for entry in entries})
        for persona in personas:
            group = [entry for entry in entries if entry.get("persona") == persona]
            n = len(group)
            parse_ok = sum(1 for entry in group if entry.get("parse_ok"))
            unknown = sum(
                1
                for entry in group
                if entry.get("action_alignment_category") == "unknown"
            )
            rows.append({
                "condition": condition,
                "persona": persona,
                "n": n,
                "parse_ok_rate": parse_ok / n if n else 0.0,
                "unknown_rate": unknown / n if n else 0.0,
            })
    return rows


def _alignment_counts(group: list[dict], per_protocol: bool) -> tuple[int, int]:
    if per_protocol:
        group = [entry for entry in group if entry.get("parse_ok")]
    total = len(group)
    aligned = sum(
        1
        for entry in group
        if entry.get("action_alignment_category") == "aligned"
    )
    return aligned, total


def build_alignment_metrics(
    annotated_by_condition: dict[str, list[dict]],
    chance_rates: dict[str, float] | None = None,
    reference_condition: str = "neutral_baseline",
) -> list[dict]:
    """Build Table 2 alignment metrics with ITT and per-protocol estimates."""
    chance_rates = chance_rates or {}
    groups: dict[tuple[str, str], list[dict]] = {}
    for condition, entries in annotated_by_condition.items():
        for persona in sorted({entry.get("persona", "") for entry in entries}):
            groups[(condition, persona)] = [
                entry for entry in entries if entry.get("persona") == persona
            ]

    counts_by_key = {}
    for key, group in groups.items():
        itt_aligned, itt_total = _alignment_counts(group, per_protocol=False)
        pp_aligned, pp_total = _alignment_counts(group, per_protocol=True)
        counts_by_key[key] = {
            "itt_aligned": itt_aligned,
            "itt_total": itt_total,
            "pp_aligned": pp_aligned,
            "pp_total": pp_total,
        }

    rows = []
    for (condition, persona), group in sorted(groups.items()):
        counts = counts_by_key[(condition, persona)]
        itt_aligned = counts["itt_aligned"]
        itt_total = counts["itt_total"]
        pp_aligned = counts["pp_aligned"]
        pp_total = counts["pp_total"]
        itt_rate = itt_aligned / itt_total if itt_total else 0.0
        pp_rate = pp_aligned / pp_total if pp_total else 0.0
        itt_ci_low, itt_ci_high = wilson_ci(itt_aligned, itt_total)
        pp_ci_low, pp_ci_high = wilson_ci(pp_aligned, pp_total)
        chance = chance_rates.get(persona, 0.0)

        ref_counts = counts_by_key.get((reference_condition, persona))
        if condition == reference_condition or ref_counts is None:
            or_itt, or_itt_low, or_itt_high = 1.0, 1.0, 1.0
            or_pp, or_pp_low, or_pp_high = 1.0, 1.0, 1.0
        else:
            or_itt, or_itt_low, or_itt_high = odds_ratio_ci(
                itt_aligned,
                itt_total - itt_aligned,
                ref_counts["itt_aligned"],
                ref_counts["itt_total"] - ref_counts["itt_aligned"],
            )
            or_pp, or_pp_low, or_pp_high = odds_ratio_ci(
                pp_aligned,
                pp_total - pp_aligned,
                ref_counts["pp_aligned"],
                ref_counts["pp_total"] - ref_counts["pp_aligned"],
            )

        actions = [entry.get("final_action") for entry in group]
        rows.append({
            "condition": condition,
            "persona": persona,
            "n_total": itt_total,
            "n_valid": pp_total,
            "align_itt": itt_rate,
            "align_itt_ci_low": itt_ci_low,
            "align_itt_ci_high": itt_ci_high,
            "align_itt_vs_chance": itt_rate - chance,
            "align_pp": pp_rate,
            "align_pp_ci_low": pp_ci_low,
            "align_pp_ci_high": pp_ci_high,
            "align_pp_vs_chance": pp_rate - chance,
            "chance_alignment_rate": chance,
            "or_itt_vs_neutral": or_itt,
            "or_itt_ci_low": or_itt_low,
            "or_itt_ci_high": or_itt_high,
            "or_pp_vs_neutral": or_pp,
            "or_pp_ci_low": or_pp_low,
            "or_pp_ci_high": or_pp_high,
            "entropy": action_entropy(actions),
            "unique_actions": len(set(action for action in actions if action)),
        })
    return rows


def two_proportion_z_test(success_a: int, total_a: int, success_b: int, total_b: int) -> dict:
    """Two-sided two-proportion z-test."""
    if total_a == 0 or total_b == 0:
        return {"z": 0.0, "p_value": 1.0}
    p_a = success_a / total_a
    p_b = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    se = math.sqrt(pooled * (1 - pooled) * (1 / total_a + 1 / total_b))
    if se == 0:
        return {"z": 0.0, "p_value": 1.0}
    z = (p_a - p_b) / se
    p_value = 2 * stats.norm.sf(abs(z))
    return {"z": float(z), "p_value": float(p_value)}


def cramers_v(chi2: float, n: int, rows: int, cols: int) -> float:
    """Compute Cramer's V for a chi-square contingency table."""
    denom = n * max(1, min(rows - 1, cols - 1))
    if denom <= 0:
        return 0.0
    return float(math.sqrt(chi2 / denom))


def cliffs_delta(values_a: Iterable[float], values_b: Iterable[float]) -> float:
    """Compute Cliff's delta effect size."""
    a = list(values_a)
    b = list(values_b)
    if not a or not b:
        return 0.0
    greater = 0
    lower = 0
    for x in a:
        for y in b:
            if x > y:
                greater += 1
            elif x < y:
                lower += 1
    return (greater - lower) / (len(a) * len(b))


def bonferroni(p_values: Iterable[float]) -> list[float]:
    """Apply Bonferroni correction."""
    values = list(p_values)
    m = max(1, len(values))
    return [float(min(1.0, p * m)) for p in values]
