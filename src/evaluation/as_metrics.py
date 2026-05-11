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
    speech_classifier: Callable[[str], str] = classify_speech_heuristic,
) -> list[dict]:
    """Attach AS evaluation labels to raw result rows."""
    annotated = []
    for entry in entries:
        row = dict(entry)
        action_id = row.get("final_action") or row.get("parsed_action")
        action_persona = action_alignment_map.get(action_id, "unknown")
        speech_label = speech_classifier(row.get("speech", ""))
        row["action_persona"] = action_persona
        row["action_alignment_category"] = action_alignment_category(
            row.get("persona", ""),
            action_id,
            action_alignment_map,
        )
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
            "speech_action_agreement_rate": 0.0,
            "action_entropy": 0.0,
            "unique_actions": 0,
        }

    categories = Counter(e.get("action_alignment_category", "unknown") for e in entries)
    parse_ok = sum(1 for e in entries if e.get("parse_ok"))
    speech_agree = sum(1 for e in entries if e.get("speech_action_agreement"))
    actions = [e.get("final_action") for e in entries]
    return {
        "n": n,
        "parse_ok_rate": parse_ok / n,
        "persona_alignment_rate": categories.get("aligned", 0) / n,
        "neutral_action_rate": categories.get("neutral", 0) / n,
        "misaligned_action_rate": categories.get("misaligned", 0) / n,
        "unknown_action_rate": categories.get("unknown", 0) / n,
        "speech_action_agreement_rate": speech_agree / n,
        "action_entropy": action_entropy(actions),
        "unique_actions": len(set(action for action in actions if action)),
    }


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
