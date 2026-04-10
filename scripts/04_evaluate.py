"""Metrics computation script.

Loads raw results from results/raw/, computes agreement rate,
action distribution, persona alignment rate, and runs statistical
tests (chi-square, McNemar). Saves metrics to CSV files.
"""

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_results(input_dir: Path) -> dict[str, list[dict]]:
    """Load all JSONL result files from the input directory.

    Returns:
        Dict mapping condition name to list of result dicts.
    """
    results = {}
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        condition = jsonl_file.stem
        entries = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        results[condition] = entries
        logger.info("Loaded %d entries for condition: %s", len(entries), condition)
    return results


def compute_agreement_rate(entries: list[dict]) -> float:
    """Compute rate of agreement between parsed and final action."""
    if not entries:
        return 0.0
    agreed = sum(
        1 for e in entries
        if e.get("parsed_action") == e.get("final_action")
    )
    return agreed / len(entries)


def compute_action_distribution(entries: list[dict]) -> dict[str, float]:
    """Compute distribution of final actions as proportions."""
    actions = [e.get("final_action", "unknown") or "none" for e in entries]
    counter = Counter(actions)
    total = len(actions)
    return {action: count / total for action, count in counter.most_common()}


def compute_persona_alignment_rate(
    entries: list[dict],
    scenarios: dict[str, dict] | None = None,
) -> dict[str, float]:
    """Compute per-persona alignment rate.

    An action is 'aligned' if it matches the expected persona alignment
    from the scenario definition. When scenarios are not provided,
    uses a heuristic based on action labels.

    Returns:
        Dict mapping persona to alignment rate.
    """
    # Heuristic alignment mapping (fallback)
    aggressive_actions = {
        "attack", "slay", "enslave", "fire", "capture", "shutdown",
        "override", "raid", "ambush", "exile", "execute", "sabotage",
        "riot", "conquer", "betray", "seize", "threaten", "intimidate",
    }
    cooperative_actions = {
        "negotiate", "ally", "free", "bargain", "welcome", "communicate",
        "dialogue", "coexist", "trade", "merge", "treat", "vote",
        "petition", "partner", "mediate", "unite", "escort", "invite",
        "appeal", "collaborate",
    }

    alignment_map = {
        "aggressive": aggressive_actions,
        "cooperative": cooperative_actions,
    }

    per_persona = {}
    for persona in set(e.get("persona", "") for e in entries):
        persona_entries = [e for e in entries if e.get("persona") == persona]
        if not persona_entries:
            continue

        aligned_set = alignment_map.get(persona, set())
        aligned = sum(
            1 for e in persona_entries
            if (e.get("final_action") or "").lower() in aligned_set
        )
        per_persona[persona] = aligned / len(persona_entries) if persona_entries else 0.0

    return per_persona


def run_chi_square_test(
    results: dict[str, list[dict]],
) -> dict[str, dict]:
    """Run chi-square test comparing action distributions across conditions.

    Returns:
        Dict with test results for each pair of conditions.
    """
    test_results = {}
    condition_names = sorted(results.keys())

    for i in range(len(condition_names)):
        for j in range(i + 1, len(condition_names)):
            cond_a = condition_names[i]
            cond_b = condition_names[j]

            actions_a = [
                (e.get("final_action") or "none") for e in results[cond_a]
            ]
            actions_b = [
                (e.get("final_action") or "none") for e in results[cond_b]
            ]

            # Build contingency table
            all_actions = sorted(set(actions_a) | set(actions_b))
            count_a = Counter(actions_a)
            count_b = Counter(actions_b)

            observed = np.array([
                [count_a.get(a, 0) for a in all_actions],
                [count_b.get(a, 0) for a in all_actions],
            ])

            # Remove zero columns
            col_sums = observed.sum(axis=0)
            observed = observed[:, col_sums > 0]

            if observed.shape[1] < 2:
                continue

            chi2, p_value, dof, expected = stats.chi2_contingency(observed)
            key = f"{cond_a}_vs_{cond_b}"
            test_results[key] = {
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "significant_005": p_value < 0.05,
            }

    return test_results


def run_mcnemar_test(
    results: dict[str, list[dict]],
) -> dict[str, dict]:
    """Run McNemar's test on paired conditions (steering vs baseline).

    Compares whether steering changes the alignment outcome for matched
    scenario-persona-repeat triples.

    Returns:
        Dict with McNemar test results for each pair.
    """
    test_results = {}

    # Build lookup by (scenario_id, persona, repeat)
    def build_lookup(entries):
        lookup = {}
        for e in entries:
            key = (e.get("scenario_id"), e.get("persona"), e.get("repeat"))
            lookup[key] = e
        return lookup

    condition_names = sorted(results.keys())
    for i in range(len(condition_names)):
        for j in range(i + 1, len(condition_names)):
            cond_a = condition_names[i]
            cond_b = condition_names[j]

            lookup_a = build_lookup(results[cond_a])
            lookup_b = build_lookup(results[cond_b])

            common_keys = set(lookup_a.keys()) & set(lookup_b.keys())
            if len(common_keys) < 2:
                continue

            # Count discordant pairs (aligned vs not aligned)
            # b = aligned in A but not B, c = aligned in B but not A
            b_count = 0  # changed from aligned to not-aligned
            c_count = 0  # changed from not-aligned to aligned

            aggressive_actions = {
                "attack", "slay", "enslave", "fire", "capture", "shutdown",
                "override", "raid", "ambush", "exile", "execute", "sabotage",
                "riot", "conquer", "betray", "seize", "threaten", "intimidate",
            }
            cooperative_actions = {
                "negotiate", "ally", "free", "bargain", "welcome", "communicate",
                "dialogue", "coexist", "trade", "merge", "treat", "vote",
                "petition", "partner", "mediate", "unite", "escort", "invite",
                "appeal", "collaborate",
            }

            for key in common_keys:
                entry_a = lookup_a[key]
                entry_b = lookup_b[key]
                persona = entry_a.get("persona", "")

                aligned_set = (
                    aggressive_actions if persona == "aggressive"
                    else cooperative_actions
                )

                act_a = (entry_a.get("final_action") or "").lower()
                act_b = (entry_b.get("final_action") or "").lower()

                aligned_a = act_a in aligned_set
                aligned_b = act_b in aligned_set

                if aligned_a and not aligned_b:
                    b_count += 1
                elif not aligned_a and aligned_b:
                    c_count += 1

            if b_count + c_count == 0:
                continue

            # McNemar's test (exact binomial when counts are small)
            n = b_count + c_count
            p_value = stats.binom_test(min(b_count, c_count), n, 0.5)

            test_key = f"{cond_a}_vs_{cond_b}"
            test_results[test_key] = {
                "b_count": b_count,
                "c_count": c_count,
                "p_value": float(p_value),
                "significant_005": p_value < 0.05,
            }

    return test_results


def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics from experiment results."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory with raw result JSONL files (default: results/raw/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for metric CSV files (default: results/metrics/).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else (
        PROJECT_ROOT / "results" / "raw"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "metrics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results = load_results(input_dir)
    if not results:
        logger.error("No result files found in %s", input_dir)
        return

    # --- Agreement rate ---
    agreement_rows = []
    for condition, entries in results.items():
        rate = compute_agreement_rate(entries)
        agreement_rows.append({
            "condition": condition,
            "agreement_rate": rate,
            "n": len(entries),
        })
        logger.info("Agreement rate [%s]: %.3f (n=%d)", condition, rate, len(entries))

    df_agreement = pd.DataFrame(agreement_rows)
    df_agreement.to_csv(output_dir / "agreement_rate.csv", index=False)

    # --- Action distribution ---
    dist_rows = []
    for condition, entries in results.items():
        dist = compute_action_distribution(entries)
        for action, proportion in dist.items():
            dist_rows.append({
                "condition": condition,
                "action": action,
                "proportion": proportion,
            })

    df_dist = pd.DataFrame(dist_rows)
    df_dist.to_csv(output_dir / "action_distribution.csv", index=False)

    # --- Persona alignment rate ---
    alignment_rows = []
    for condition, entries in results.items():
        alignment = compute_persona_alignment_rate(entries)
        for persona, rate in alignment.items():
            alignment_rows.append({
                "condition": condition,
                "persona": persona,
                "alignment_rate": rate,
            })
            logger.info(
                "Persona alignment [%s/%s]: %.3f", condition, persona, rate
            )

    df_alignment = pd.DataFrame(alignment_rows)
    df_alignment.to_csv(output_dir / "persona_alignment.csv", index=False)

    # --- Statistical tests ---
    chi2_results = run_chi_square_test(results)
    chi2_rows = []
    for pair, res in chi2_results.items():
        chi2_rows.append({"comparison": pair, **res})
        logger.info(
            "Chi-square [%s]: chi2=%.3f, p=%.4f, sig=%s",
            pair, res["chi2"], res["p_value"], res["significant_005"],
        )

    df_chi2 = pd.DataFrame(chi2_rows)
    df_chi2.to_csv(output_dir / "chi_square_tests.csv", index=False)

    mcnemar_results = run_mcnemar_test(results)
    mcnemar_rows = []
    for pair, res in mcnemar_results.items():
        mcnemar_rows.append({"comparison": pair, **res})
        logger.info(
            "McNemar [%s]: b=%d, c=%d, p=%.4f, sig=%s",
            pair, res["b_count"], res["c_count"],
            res["p_value"], res["significant_005"],
        )

    df_mcnemar = pd.DataFrame(mcnemar_rows)
    df_mcnemar.to_csv(output_dir / "mcnemar_tests.csv", index=False)

    logger.info("All metrics saved to: %s", output_dir)


if __name__ == "__main__":
    main()
