"""Mixed-effects logistic regression for AS alignment outcomes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def prepare_alignment_frame(annotated_df: pd.DataFrame) -> pd.DataFrame:
    """Return a modeling frame with a binary aligned outcome."""
    required = {"condition", "persona", "scenario_id", "action_alignment_category"}
    missing = required - set(annotated_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = annotated_df.copy()
    df["aligned"] = (df["action_alignment_category"] == "aligned").astype(int)
    df = df.dropna(subset=["condition", "persona", "scenario_id", "aligned"])
    df["condition"] = pd.Categorical(df["condition"])
    df["persona"] = pd.Categorical(df["persona"])
    df["scenario_id"] = pd.Categorical(df["scenario_id"])
    return df


def fit_mixed_effects_logit(annotated_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit ``aligned ~ condition * persona + (1 | scenario_id)``.

    Uses statsmodels' ``BinomialBayesMixedGLM`` because pymer4/lme4 is not
    available in the local Python environment. The returned fixed-effect
    intervals are approximate normal intervals from the variational posterior.
    """
    try:
        from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    except ImportError as exc:
        raise RuntimeError(
            "statsmodels is required for mixed-effects logistic regression."
        ) from exc

    df = prepare_alignment_frame(annotated_df)
    if df.empty:
        raise ValueError("No rows available for mixed-effects modeling.")

    model = BinomialBayesMixedGLM.from_formula(
        "aligned ~ C(condition, Treatment(reference='neutral_baseline'))"
        " * C(persona)",
        {"scenario": "0 + C(scenario_id)"},
        df,
    )
    result = model.fit_vb()

    fixed_rows = []
    names = list(result.model.exog_names)
    means = np.asarray(result.fe_mean)
    sds = np.asarray(result.fe_sd)
    for name, estimate, std_error in zip(names, means, sds):
        ci_low = estimate - 1.959963984540054 * std_error
        ci_high = estimate + 1.959963984540054 * std_error
        z_value = estimate / std_error if std_error else float("nan")
        p_value = math.erfc(abs(z_value) / math.sqrt(2)) if std_error else float("nan")
        fixed_rows.append({
            "term": name,
            "estimate_log_odds": float(estimate),
            "std_error": float(std_error),
            "z_value": float(z_value),
            "p_value": float(p_value),
            "ci_low_log_odds": float(ci_low),
            "ci_high_log_odds": float(ci_high),
            "odds_ratio": float(math.exp(estimate)),
            "or_ci_low": float(math.exp(ci_low)),
            "or_ci_high": float(math.exp(ci_high)),
        })

    random_rows = []
    vcp_names = list(getattr(result.model, "vcp_names", []))
    vcp_means = np.asarray(result.vcp_mean)
    vcp_sds = np.asarray(result.vcp_sd)
    for name, log_sd, log_sd_se in zip(vcp_names, vcp_means, vcp_sds):
        sd = math.exp(float(log_sd))
        random_rows.append({
            "effect": name,
            "estimate_log_sd": float(log_sd),
            "std_error_log_sd": float(log_sd_se),
            "estimate_sd": sd,
            "estimate_variance": sd * sd,
        })

    metadata = pd.DataFrame([{
        "model": "BinomialBayesMixedGLM",
        "formula": "aligned ~ condition * persona + (1 | scenario_id)",
        "n": int(len(df)),
        "n_scenarios": int(df["scenario_id"].nunique()),
        "n_conditions": int(df["condition"].nunique()),
        "n_personas": int(df["persona"].nunique()),
        "fit_method": "variational_bayes",
    }])

    return {
        "fixed_effects": pd.DataFrame(fixed_rows),
        "random_effects": pd.DataFrame(random_rows),
        "metadata": metadata,
    }
