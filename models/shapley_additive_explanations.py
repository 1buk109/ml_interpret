from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.special import factorial


@dataclass
class ShapleyAdditiveExplanations:
    """
    Shapley Additive Explanations

    Parameters
    ----------
    estimator: object
        A fitted estimator.
    X: np.ndarray
        A feature matrix.
    var_names: list
        A list of feature names.
    """
    estimator: Any
    X: np.ndarray
    var_names: list[str]

    def __post_init__(self) -> None:
        self.baseline = self.estimator.predict(self.X).mean()
        _, self.n_features = self.X.shape
        self.subsets =[
            subset
            for idx_feature in range(self.n_features + 1)
            for subset in combinations(range(self.n_features), idx_feature)
        ]
    
    def _get_expected_value(self, subset: tuple[int, ...]) -> np.ndarray:
        """
        
        """
        _X = self.X.copy()
        if subset is not None:
            _s = list(subset)
            _X[:, _s] = self.X[self.i, _s]
        return self.estimator.predict(_X).mean()
    
    def _calc_weighted_marginal_contribution(
        self,
        j: int,
        s_union_j: tuple[int, ...],
    ) -> float:
        """
        """
        subset = tuple(set(s_union_j) - {j})
        n_subset = len(subset)
        weight = factorial(n_subset) * factorial(self.n_features - n_subset - 1)
        marginal_contribution = (
            self.expected_values[s_union_j] - self.expected_values[subset]
        )
        return weight * marginal_contribution
    
    def sharpley_additive_explanations(self, id_to_compute: int) -> None:
        """
        """
        self.i = id_to_compute
        self.expected_values = {
            subset: self._get_expected_value(subset)
            for subset in self.subsets
        }
        shap_values = np.zeros(self.n_features)
        for j in range(self.n_features):
            shap_values[j] = np.sum([
                self._calc_weighted_marginal_contribution(j, s_union_j)
                for s_union_j in self.subsets
                if j in s_union_j
            ]) / factorial(self.n_features)
        self.df_shap = pd.DataFrame(
            data={
                "var_names": self.var_names,
                "feature_value": self.X[id_to_compute],
                "shap_values": shap_values,
            }
        )
    
    def plot_shap(self) -> None:
        """
        """
        fig = go.Figure(
            data=[
                go.Bar(
                    x=self.df_shap["shap_values"],
                    y=self.df_shap["var_names"],
                    orientation="h",
                    marker_color="salmon",
                )
            ],
            layout=go.Layout(
                title_text="Shapley Additive Explanations",
                title_x=0.5,
                xaxis_title="SHAP value",
                yaxis_title="Feature",
                yaxis_autorange="reversed",
            )
        )
        fig.show()
