from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

from models.permutation_feature_importance import PermutationFeatureImportance


class GroupedPermutationFeatureImportance(PermutationFeatureImportance):
    """
    Grouped permutation feature importance: override PermutationFeatureImportance class.
    """

    def _column_to_index(self, cols: list[str]) -> list[int]:
        return [self.var_names.index(col) for col in cols]

    def calc_permuted_metrics(self, cols: list[str]) -> float:
        """
        Calculate the RMSE of the grouped permuted data.

        Parameters
        ----------
        cols : list[int]
            The list of columns to permute (coverient each other).
        """
        X_permuted = self.X.copy()
        cols = self._column_to_index(cols)
        X_permuted[:, cols] = np.random.permutation(X_permuted[:, cols])
        y_pred = self.estimator.predict(X_permuted)
        return mean_squared_error(self.y, y_pred, squared=False)

    def _parse_grouped_pfi(
        self,
        metrics_permuted: np.ndarray,
        var_groups: list[list: str]
        ) -> pd.DataFrame:
        df_grouped_pfi = pd.DataFrame(
            data = {
                'var_name': [','.join(var_group) for var_group in var_groups],
                'baseline': self._baseline,
                'permutation': metrics_permuted,
                'difference': metrics_permuted - self._baseline,
                'ratio': metrics_permuted / self._baseline
            }
        )
        return df_grouped_pfi.sort_values('permutation')
    
    def calc_permutation_feature_importance(
        self,
        n_shuffle: int = 10,
        var_groups: list[list: str] | None = None
        ) -> None:
        """
        Calculate the grouprd permutation feature importance.

        Parameters
        ----------
        n_shuffle : int
            Number of times to shuffle each feature.
        var_groups : list[list: str] | None
            The list of lists of variables to permute together.
        """
        if var_groups is None:
            var_groups = [[var] for var in self.var_names]
        metrics_permuted = [
            np.mean(
                [self.calc_permuted_metrics(var_group) for _ in range(n_shuffle)]
            )
            for var_group in var_groups
        ]
        self.df_pfi = self._parse_grouped_pfi(metrics_permuted, var_groups)
