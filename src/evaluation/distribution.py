"""Action distribution analysis for LLM NPC experiments."""

from collections import Counter
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


class DistributionAnalyzer:
    """Analyzes the distribution of chosen actions across experiment runs."""

    def __init__(self, action_definitions: List[Dict]):
        """
        Args:
            action_definitions: list of dicts with 'id' and
                'persona_alignment' fields.
        """
        self._alignment_map: Dict[str, str] = {}
        self._action_ids: List[str] = []
        for action_def in action_definitions:
            aid = action_def["id"]
            self._action_ids.append(aid)
            self._alignment_map[aid] = action_def["persona_alignment"]

    def compute_distribution(self, action_ids: List[str]) -> Dict:
        """Compute count and proportion for each action.

        Args:
            action_ids: list of chosen action identifiers.

        Returns:
            dict with 'counts' ({action_id: int}) and
            'proportions' ({action_id: float}).
        """
        total = len(action_ids)
        counts = dict(Counter(action_ids))

        proportions = {}
        for aid, count in counts.items():
            proportions[aid] = count / total if total > 0 else 0.0

        return {"counts": counts, "proportions": proportions}

    def compute_persona_alignment_rate(
        self, action_ids: List[str], target_persona: str
    ) -> float:
        """Fraction of actions that align with the target persona.

        Args:
            action_ids: list of chosen action identifiers.
            target_persona: the persona to check alignment against
                (e.g., "aggressive", "cooperative").

        Returns:
            float in [0, 1] representing alignment rate.
        """
        if not action_ids:
            return 0.0
        aligned = sum(
            1
            for aid in action_ids
            if self._alignment_map.get(aid, "neutral") == target_persona
        )
        return aligned / len(action_ids)

    def chi_square_test(
        self,
        action_ids: List[str],
        expected_distribution: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Chi-square goodness-of-fit test on the action distribution.

        Args:
            action_ids: list of chosen action identifiers.
            expected_distribution: optional dict mapping action_id to
                expected proportion.  If None, a uniform distribution over
                all known actions is assumed.

        Returns:
            dict with 'chi2' (float), 'p_value' (float), and
            'significant' (bool, p < 0.05).
        """
        counts = Counter(action_ids)
        total = len(action_ids)

        # Build ordered arrays so observed/expected line up
        if expected_distribution is not None:
            all_ids = sorted(set(list(counts.keys()) + list(expected_distribution.keys())))
        else:
            all_ids = sorted(set(list(counts.keys()) + self._action_ids))

        observed = np.array([counts.get(aid, 0) for aid in all_ids], dtype=float)

        if expected_distribution is not None:
            expected_props = np.array(
                [expected_distribution.get(aid, 0.0) for aid in all_ids], dtype=float
            )
            # Normalise in case proportions don't sum to 1
            if expected_props.sum() > 0:
                expected_props = expected_props / expected_props.sum()
            expected = expected_props * total
        else:
            # Uniform expectation
            expected = np.full(len(all_ids), total / len(all_ids))

        # Drop categories where expected is 0 to avoid division errors
        mask = expected > 0
        observed = observed[mask]
        expected = expected[mask]

        if len(observed) < 2:
            return {"chi2": 0.0, "p_value": 1.0, "significant": False}

        chi2, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

        return {
            "chi2": float(chi2),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
        }

    def compare_conditions(
        self,
        condition_a_actions: List[str],
        condition_b_actions: List[str],
    ) -> Dict:
        """Compare action distributions between two experimental conditions.

        Uses a chi-square test of independence on the contingency table
        built from the two conditions' action counts.  Cramer's V is
        reported as effect size.

        Args:
            condition_a_actions: action ids from condition A.
            condition_b_actions: action ids from condition B.

        Returns:
            dict with 'chi2', 'p_value', 'significant', and 'effect_size'.
        """
        counts_a = Counter(condition_a_actions)
        counts_b = Counter(condition_b_actions)

        all_ids = sorted(set(list(counts_a.keys()) + list(counts_b.keys())))

        if len(all_ids) < 2:
            return {
                "chi2": 0.0,
                "p_value": 1.0,
                "significant": False,
                "effect_size": 0.0,
            }

        row_a = [counts_a.get(aid, 0) for aid in all_ids]
        row_b = [counts_b.get(aid, 0) for aid in all_ids]

        contingency = np.array([row_a, row_b])

        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

        # Cramer's V
        n = contingency.sum()
        k = min(contingency.shape)
        cramers_v = float(np.sqrt(chi2 / (n * (k - 1)))) if n * (k - 1) > 0 else 0.0

        return {
            "chi2": float(chi2),
            "p_value": float(p_value),
            "significant": bool(p_value < 0.05),
            "effect_size": cramers_v,
        }
