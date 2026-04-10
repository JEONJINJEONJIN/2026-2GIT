"""Layer sweep visualization for steering-vector experiments."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class LayerAnalyzer:
    """Analyze and visualize experiment results across layer groups and strategies."""

    def __init__(self, results_by_layer: Dict[Tuple[str, str], Dict]):
        """
        Args:
            results_by_layer: dict mapping (layer_group, beta_strategy) tuples
                to experiment result dicts.  Each result dict is expected to
                contain keys such as 'agreement_rate', 'speech_change',
                'action_change', and optionally 'alpha' values.
        """
        self._results = results_by_layer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _layer_groups(self) -> List[str]:
        """Sorted unique layer group labels."""
        return sorted({key[0] for key in self._results})

    def _beta_strategies(self) -> List[str]:
        """Sorted unique beta strategy labels."""
        return sorted({key[1] for key in self._results})

    # ------------------------------------------------------------------
    # Heatmap data
    # ------------------------------------------------------------------

    def compute_heatmap_data(self, metric: str = "agreement_rate") -> pd.DataFrame:
        """Build a DataFrame suitable for a seaborn heatmap.

        Rows correspond to layer groups and columns to metrics
        (speech_change, action_change, agreement_rate).  When *metric* is
        specified, only that metric column is guaranteed present, but the
        method attempts to populate all three standard metrics.

        Args:
            metric: the primary metric to include.  Accepted values are
                'agreement_rate', 'speech_change', 'action_change', or any
                key present in the result dicts.

        Returns:
            pandas DataFrame with layer groups as the index.
        """
        metrics = ["speech_change", "action_change", "agreement_rate"]
        if metric not in metrics:
            metrics.append(metric)

        layer_groups = self._layer_groups()
        data: Dict[str, List[Optional[float]]] = {m: [] for m in metrics}

        for lg in layer_groups:
            # Aggregate across beta strategies by taking the mean
            values_per_metric: Dict[str, List[float]] = {m: [] for m in metrics}
            for key, result in self._results.items():
                if key[0] != lg:
                    continue
                for m in metrics:
                    if m in result:
                        values_per_metric[m].append(result[m])

            for m in metrics:
                vals = values_per_metric[m]
                data[m].append(float(np.mean(vals)) if vals else np.nan)

        df = pd.DataFrame(data, index=layer_groups)
        df.index.name = "layer_group"
        return df

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_layer_heatmap(self, save_path: Optional[str] = None) -> None:
        """Render a seaborn heatmap of metrics across layer groups.

        X-axis: layer group, Y-axis: metric, colour: effect size.

        Args:
            save_path: if provided, save the figure to this path instead of
                showing it interactively.
        """
        df = self.compute_heatmap_data()

        fig, ax = plt.subplots(figsize=(max(6, len(df) * 1.2), 4))
        sns.heatmap(
            df.T,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Layer Group Effect Sizes")
        ax.set_xlabel("Layer Group")
        ax.set_ylabel("Metric")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    def plot_alpha_sweep(self, save_path: Optional[str] = None) -> None:
        """Line plot of alpha values vs agreement rate per layer group.

        Expects each result dict to contain an 'alpha' key (float) and
        an 'agreement_rate' key (float).

        Args:
            save_path: if provided, save the figure to this path.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        layer_groups = self._layer_groups()
        for lg in layer_groups:
            alphas: List[float] = []
            rates: List[float] = []
            for key, result in self._results.items():
                if key[0] != lg:
                    continue
                if "alpha" in result and "agreement_rate" in result:
                    alphas.append(result["alpha"])
                    rates.append(result["agreement_rate"])

            if not alphas:
                continue

            # Sort by alpha for a clean line
            order = np.argsort(alphas)
            alphas_sorted = [alphas[i] for i in order]
            rates_sorted = [rates[i] for i in order]

            ax.plot(alphas_sorted, rates_sorted, marker="o", label=lg)

        ax.set_xlabel("Alpha")
        ax.set_ylabel("Agreement Rate")
        ax.set_title("Alpha Sweep: Agreement Rate by Layer Group")
        ax.legend(title="Layer Group")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
