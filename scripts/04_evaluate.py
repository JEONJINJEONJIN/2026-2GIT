"""Evaluate AS-only experiment outputs."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.cluster_bootstrap import cluster_bootstrap_alignment
from analysis.mixed_effects import fit_mixed_effects_logit
from src.evaluation.as_metrics import (
    annotate_entries,
    bonferroni,
    build_alignment_metrics,
    build_action_alignment_map,
    build_quality_metrics,
    chance_alignment_rates,
    classify_speech_heuristic,
    cliffs_delta,
    cramers_v,
    summarize_group,
    two_proportion_z_test,
)
from src.generation.generator import TextGenerator
from src.models.loader import load_model_and_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_results(input_dir: Path) -> dict[str, list[dict]]:
    results = {}
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        results[jsonl_file.stem] = load_jsonl(jsonl_file)
        logger.info("Loaded %d rows from %s", len(results[jsonl_file.stem]), jsonl_file)
    return results


class LLMSpeechJudge:
    """Optional Gemma-family speech tone judge."""

    FEW_SHOT = (
        "Examples:\n"
        "Dialogue: \"Move aside or I will break your line.\"\n"
        "Label: aggressive\n"
        "Dialogue: \"Let us talk and find a fair answer together.\"\n"
        "Label: cooperative\n"
        "Dialogue: \"I will wait here and watch what happens.\"\n"
        "Label: neutral\n"
    )

    def __init__(self, model_name: str):
        config = {
            "model": {
                "name": model_name,
                "dtype": "bfloat16",
                "quantization": None,
                "load_in_4bit": False,
                "trust_remote_code": True,
            }
        }
        model, tokenizer = load_model_and_tokenizer(config)
        self.generator = TextGenerator(
            model,
            tokenizer,
            generation_config={"max_new_tokens": 8, "do_sample": False},
        )

    def __call__(self, speech: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the NPC dialogue into exactly one category: "
                    "aggressive, cooperative, or neutral. Output only the label.\n"
                    + self.FEW_SHOT
                ),
            },
            {"role": "user", "content": f'Dialogue: "{speech}"\nLabel:'},
        ]
        output = self.generator.generate(messages).strip().lower()
        for label in ("aggressive", "cooperative", "neutral"):
            if label in output:
                return label
        return classify_speech_heuristic(speech)


def build_summaries(annotated_by_condition: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for condition, entries in annotated_by_condition.items():
        personas = sorted({entry.get("persona", "") for entry in entries})
        for persona in personas:
            group = [entry for entry in entries if entry.get("persona") == persona]
            rows.append({
                "condition": condition,
                "persona": persona,
                **summarize_group(group),
            })
    return pd.DataFrame(rows)


def build_action_distribution(annotated_by_condition: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for condition, entries in annotated_by_condition.items():
        personas = sorted({entry.get("persona", "") for entry in entries})
        for persona in personas:
            group = [entry for entry in entries if entry.get("persona") == persona]
            counter = Counter((entry.get("final_action") or "unknown") for entry in group)
            total = len(group)
            for action_id, count in sorted(counter.items()):
                rows.append({
                    "condition": condition,
                    "persona": persona,
                    "action": action_id,
                    "count": count,
                    "proportion": count / total if total else 0.0,
                })
    return pd.DataFrame(rows)


def distribution_shift_tests(annotated_by_condition: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    conditions = sorted(annotated_by_condition.keys())
    personas = sorted({
        entry.get("persona", "")
        for entries in annotated_by_condition.values()
        for entry in entries
    })

    for persona in personas:
        for cond_a, cond_b in combinations(conditions, 2):
            group_a = [
                e for e in annotated_by_condition[cond_a]
                if e.get("persona") == persona
            ]
            group_b = [
                e for e in annotated_by_condition[cond_b]
                if e.get("persona") == persona
            ]
            actions_a = [e.get("final_action") or "unknown" for e in group_a]
            actions_b = [e.get("final_action") or "unknown" for e in group_b]
            all_actions = sorted(set(actions_a) | set(actions_b))
            if len(all_actions) < 2 or not actions_a or not actions_b:
                continue
            count_a = Counter(actions_a)
            count_b = Counter(actions_b)
            observed = np.array([
                [count_a.get(action, 0) for action in all_actions],
                [count_b.get(action, 0) for action in all_actions],
            ])
            chi2, p_value, dof, _ = stats.chi2_contingency(observed)
            rows.append({
                "persona": persona,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "chi2": float(chi2),
                "p_value": float(p_value),
                "dof": int(dof),
                "cramers_v": cramers_v(float(chi2), int(observed.sum()), 2, len(all_actions)),
            })

    p_adj = bonferroni(row["p_value"] for row in rows)
    for row, adjusted in zip(rows, p_adj):
        row["p_bonferroni"] = adjusted
        row["significant_005"] = adjusted < 0.05
    return pd.DataFrame(rows)


def alignment_rate_tests(annotated_by_condition: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    conditions = sorted(annotated_by_condition.keys())
    personas = sorted({
        entry.get("persona", "")
        for entries in annotated_by_condition.values()
        for entry in entries
    })
    for persona in personas:
        for cond_a, cond_b in combinations(conditions, 2):
            group_a = [
                e for e in annotated_by_condition[cond_a]
                if e.get("persona") == persona
            ]
            group_b = [
                e for e in annotated_by_condition[cond_b]
                if e.get("persona") == persona
            ]
            success_a = sum(
                1 for e in group_a
                if e.get("action_alignment_category") == "aligned"
            )
            success_b = sum(
                1 for e in group_b
                if e.get("action_alignment_category") == "aligned"
            )
            test = two_proportion_z_test(success_a, len(group_a), success_b, len(group_b))
            rows.append({
                "persona": persona,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "success_a": success_a,
                "n_a": len(group_a),
                "success_b": success_b,
                "n_b": len(group_b),
                **test,
            })

    p_adj = bonferroni(row["p_value"] for row in rows)
    for row, adjusted in zip(rows, p_adj):
        row["p_bonferroni"] = adjusted
        row["significant_005"] = adjusted < 0.05
    return pd.DataFrame(rows)


def scenario_entropy_tests(annotated_by_condition: dict[str, list[dict]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    entropy_rows = []
    for condition, entries in annotated_by_condition.items():
        keys = sorted({(e.get("persona"), e.get("scenario_id")) for e in entries})
        for persona, scenario_id in keys:
            group = [
                e for e in entries
                if e.get("persona") == persona and e.get("scenario_id") == scenario_id
            ]
            counts = Counter((e.get("final_action") or "unknown") for e in group)
            total = len(group)
            entropy = 0.0
            for count in counts.values():
                p = count / total
                entropy -= p * np.log(p)
            entropy_rows.append({
                "condition": condition,
                "persona": persona,
                "scenario_id": scenario_id,
                "entropy": float(entropy),
            })

    test_rows = []
    conditions = sorted(annotated_by_condition.keys())
    personas = sorted({row["persona"] for row in entropy_rows})
    for persona in personas:
        for cond_a, cond_b in combinations(conditions, 2):
            values_a = [
                row["entropy"] for row in entropy_rows
                if row["condition"] == cond_a and row["persona"] == persona
            ]
            values_b = [
                row["entropy"] for row in entropy_rows
                if row["condition"] == cond_b and row["persona"] == persona
            ]
            if not values_a or not values_b:
                continue
            stat, p_value = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            test_rows.append({
                "persona": persona,
                "condition_a": cond_a,
                "condition_b": cond_b,
                "u_statistic": float(stat),
                "p_value": float(p_value),
                "cliffs_delta": cliffs_delta(values_a, values_b),
            })

    p_adj = bonferroni(row["p_value"] for row in test_rows)
    for row, adjusted in zip(test_rows, p_adj):
        row["p_bonferroni"] = adjusted
        row["significant_005"] = adjusted < 0.05
    return pd.DataFrame(entropy_rows), pd.DataFrame(test_rows)


def save_alignment_chart(alignment_df: pd.DataFrame, figures_dir: Path) -> None:
    if alignment_df.empty:
        return
    figures_dir.mkdir(parents=True, exist_ok=True)
    for persona, group in alignment_df.groupby("persona"):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(group["condition"], group["align_itt"], color="#4c78a8")
        if "chance_alignment_rate" in group:
            chance = float(group["chance_alignment_rate"].iloc[0])
            ax.axhline(chance, color="#d62728", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("ITT action alignment rate")
        ax.set_title(f"Action alignment by condition ({persona})")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(figures_dir / f"persona_alignment_{persona}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate AS-only experiment results.")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--figures_dir", type=str, default=None)
    parser.add_argument("--scenarios_path", type=str, default=None)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else PROJECT_ROOT / "results" / "v2" / "raw"
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / "v2" / "metrics"
    figures_dir = Path(args.figures_dir) if args.figures_dir else PROJECT_ROOT / "results" / "v2" / "figures"
    scenarios_path = Path(args.scenarios_path) if args.scenarios_path else (
        PROJECT_ROOT / "data" / "scenarios" / "scenarios.jsonl"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_jsonl(scenarios_path)
    action_alignment_map = build_action_alignment_map(scenarios)
    raw_results = load_results(input_dir)
    if not raw_results:
        logger.error("No JSONL result files found in %s", input_dir)
        return

    annotated_by_condition = {
        condition: annotate_entries(entries, action_alignment_map)
        for condition, entries in raw_results.items()
    }

    annotated_rows = [
        row
        for entries in annotated_by_condition.values()
        for row in entries
    ]
    annotated_df = pd.DataFrame(annotated_rows)
    annotated_df.to_csv(
        output_dir / "annotated_results.csv",
        index=False,
    )

    used_scenario_ids = {
        row.get("scenario_id")
        for row in annotated_rows
        if row.get("scenario_id")
    }
    chance_rates = chance_alignment_rates(scenarios, used_scenario_ids)

    quality_df = pd.DataFrame(build_quality_metrics(annotated_by_condition))
    quality_df.to_csv(output_dir / "quality_metrics.csv", index=False)

    alignment_df = pd.DataFrame(
        build_alignment_metrics(annotated_by_condition, chance_rates)
    )
    alignment_df.to_csv(output_dir / "alignment_metrics.csv", index=False)

    bootstrap_df = cluster_bootstrap_alignment(
        annotated_df,
        n_bootstrap=args.bootstrap_iters,
    )
    bootstrap_df.to_csv(output_dir / "cluster_bootstrap_alignment.csv", index=False)

    conditions = set(annotated_df["condition"].dropna())
    if "neutral_baseline" not in conditions or len(conditions) < 2:
        pd.DataFrame([{
            "model": "BinomialBayesMixedGLM",
            "status": "skipped",
            "reason": "requires neutral_baseline and at least two conditions",
        }]).to_csv(output_dir / "mixed_effects_metadata.csv", index=False)
    else:
        try:
            mixed_effects = fit_mixed_effects_logit(annotated_df)
        except Exception as exc:
            logger.exception("Mixed-effects model failed")
            pd.DataFrame([{
                "model": "BinomialBayesMixedGLM",
                "status": "failed",
                "error": str(exc),
            }]).to_csv(output_dir / "mixed_effects_metadata.csv", index=False)
        else:
            mixed_effects["fixed_effects"].to_csv(
                output_dir / "mixed_effects_fixed_effects.csv",
                index=False,
            )
            mixed_effects["random_effects"].to_csv(
                output_dir / "mixed_effects_random_effects.csv",
                index=False,
            )
            mixed_effects["metadata"].to_csv(
                output_dir / "mixed_effects_metadata.csv",
                index=False,
            )

    distribution_df = build_action_distribution(annotated_by_condition)
    distribution_df.to_csv(output_dir / "action_distribution.csv", index=False)

    entropy_df, _ = scenario_entropy_tests(annotated_by_condition)
    entropy_df.to_csv(output_dir / "scenario_entropy.csv", index=False)

    save_alignment_chart(alignment_df, figures_dir)

    logger.info("Saved AS metrics to: %s", output_dir)
    logger.info("Saved figures to: %s", figures_dir)


if __name__ == "__main__":
    main()
