from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.express as px


@dataclass
class PartialDependence:
    """
    Partial dependence plots for a single feature.
    
    Parameters
    ----------
    estimator : sklearn estimator
        A fitted estimator.
    X: np.ndarray
        The training data.
    var_names: list
        The names of the features.
    """

    estimator: Any
    X: np.array
    var_names: list[str]

    def _counterfactual_prediction(
        self,
        var_index: int,
        value_to_replace: float,
    ) -> np.ndarray:
        """
        Create a counterfactual dataset to replace the value in var_index to value_to_replace.
        Then, predict the counterfactual dataset.

        Parameters
        ----------
        var_index : int
            The index of the column to replace.
        value_to_replace : float
            The value to replace.
        """
        X_counterfactual = self.X.copy()
        X_counterfactual[:, var_index] = value_to_replace
        return self.estimator.predict(X_counterfactual)
    
    def partial_dependence(
        self,
        var_name: str,
        n_grid: Optional[int] = 50,
    ) -> None:
        """
        Calculate the partial dependence for a single feature.

        Parameters
        ----------
        var_name : str
            The name of the feature.
        n_grid : int, optional
            The number of grid points to use, by default 50.
        """
        self.target_var_name = var_name
        var_index = self.var_names.index(var_name)
        var_range = np.linspace(
            self.X[:, var_index].min(),
            self.X[:, var_index].max(),
            num=n_grid,
        )
        avg_predictions = [
            self._counterfactual_prediction(var_index, value).mean()
            for value in var_range
        ]
        self.df_partial_dependence = pd.DataFrame(
            data={
                var_name: var_range,
                'avg_predictions': avg_predictions,
            }
        )
    
    def plot_partial_dependence(self) -> None:
        """
        Plot the partial dependence for a single feature.
        """
        fig = px.line(
            self.df_partial_dependence,
            x=self.target_var_name,
            y='avg_predictions',
            title=f'Partial dependence plot for {self.target_var_name}',
        )
        fig.show()
