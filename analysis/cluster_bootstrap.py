"""Scenario-cluster bootstrap intervals for alignment metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _alignment_rate(group: pd.DataFrame, per_protocol: bool) -> float:
    if per_protocol:
        group = group[group["parse_ok"].astype(bool)]
    if len(group) == 0:
        return np.nan
    return float((group["action_alignment_category"] == "aligned").mean())


def cluster_bootstrap_alignment(
    annotated_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    cluster_col: str = "scenario_id",
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap alignment rates by resampling scenarios with replacement."""
    required = {
        "condition",
        "persona",
        cluster_col,
        "parse_ok",
        "action_alignment_category",
    }
    missing = required - set(annotated_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = annotated_df.copy()
    clusters = sorted(df[cluster_col].dropna().unique())
    if not clusters:
        raise ValueError(f"No clusters found in column {cluster_col!r}")

    rng = np.random.default_rng(seed)
    keys = sorted(df[["condition", "persona"]].drop_duplicates().itertuples(index=False, name=None))
    original = {
        (condition, persona, "itt"): _alignment_rate(
            df[(df["condition"] == condition) & (df["persona"] == persona)],
            per_protocol=False,
        )
        for condition, persona in keys
    }
    original.update({
        (condition, persona, "pp"): _alignment_rate(
            df[(df["condition"] == condition) & (df["persona"] == persona)],
            per_protocol=True,
        )
        for condition, persona in keys
    })

    boot_values: dict[tuple[str, str, str], list[float]] = {
        key: [] for key in original
    }
    grouped = {cluster: rows for cluster, rows in df.groupby(cluster_col, sort=False)}

    for _ in range(n_bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample_df = pd.concat([grouped[cluster] for cluster in sampled], ignore_index=True)
        for condition, persona in keys:
            group = sample_df[
                (sample_df["condition"] == condition)
                & (sample_df["persona"] == persona)
            ]
            boot_values[(condition, persona, "itt")].append(
                _alignment_rate(group, per_protocol=False)
            )
            boot_values[(condition, persona, "pp")].append(
                _alignment_rate(group, per_protocol=True)
            )

    rows = []
    for condition, persona in keys:
        for estimand in ("itt", "pp"):
            values = np.array(boot_values[(condition, persona, estimand)], dtype=float)
            values = values[~np.isnan(values)]
            estimate = original[(condition, persona, estimand)]
            if len(values):
                ci_low, ci_high = np.quantile(values, [0.025, 0.975])
            else:
                ci_low, ci_high = np.nan, np.nan
            rows.append({
                "condition": condition,
                "persona": persona,
                "estimand": estimand,
                "estimate": estimate,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "n_bootstrap": n_bootstrap,
                "n_clusters": len(clusters),
                "cluster_col": cluster_col,
            })
    return pd.DataFrame(rows)
