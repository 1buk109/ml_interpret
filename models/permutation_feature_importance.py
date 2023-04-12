from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error


@dataclass
class PermutationFeatureImportance():
    """
    Permutation feature importance.

    Parameters
    ----------
    model : sklearn.base.BaseEstimator
        The model to evaluate.
    """

    estimator: Any
    X: np.ndarray
    y: np.ndarray
    var_names: list[str]

    def __post_init__(self) -> None:
        self._baseline = mean_squared_error(
            self.y, self.estimator.predict(self.X), squared=False
        )
    
    def _permute(self, X: np.ndarray, col: int) -> np.ndarray:
        X_permuted = X.copy()
        X_permuted[:, col] = np.random.permutation(X_permuted[:, col])
        return X_permuted
    
    def calc_permuted_metrics(self, col: int) -> float:
        X_permuted = self._permute(self.X, col)
        y_pred = self.estimator.predict(X_permuted)
        return mean_squared_error(self.y, y_pred, squared=False)

    def _parse_pfi(self, metrics_permuted: np.ndarray) -> pd.DataFrame:
        df_pfi = pd.DataFrame(
            data = {
                'var_name': self.var_names,
                'baseline': self._baseline,
                'permutation': metrics_permuted,
                'difference': metrics_permuted - self._baseline,
                'ratio': metrics_permuted / self._baseline
            }
        )
        return df_pfi.sort_values('difference')
    
    def calc_permutation_feature_importance(self, n_shuffle: int = 10) -> None:
        _, n_dim = self.X.shape
        metrics_permuted = [
            np.mean(
                [self.calc_permuted_metrics(n) for _ in range(n_shuffle)]
            )
            for n in range(n_dim)
        ]
        self.df_pfi = self._parse_pfi(metrics_permuted)
    
    def plot_permutation_feature_importance(self) -> None:
        """
        plot permutation feature importance.

        Parameters
        ----------
        n_shuffle : int
            Number of times to shuffle each feature.

        """
        fig = go.Figure(
            data=[
                go.Bar(
                    x=self.df_pfi['difference'],
                    y=self.df_pfi['var_name'],
                    text=self.df_pfi['difference'],
                    customdata=self.df_pfi,
                    hovertemplate=
                        'baseline: %{customdata[1]:.2f}<br>'
                        'permutation: %{customdata[2]:.2f}<br>'
                        'difference: %{customdata[3]:.2f}<br>'
                        'ratio: %{customdata[4]:.2f}<br>',
                    textposition='auto',
                    texttemplate='%{text:.2f}',
                    marker_color='lightsalmon',
                    orientation='h'
                )
            ]
        )
        fig.update_layout(
            title='Permutation Feature Importance',
            xaxis_title='Difference in RMSE',
            yaxis_title='Feature',
        )
        fig.show()
