from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from models.partial_dependence import PartialDependence


class IndividualConditionalExpectation(PartialDependence):
    def individual_conditional_expectation(
        self,
        var_name: str,
        ids_to_compute: list[int],
        n_grid: Optional[int] = 50,
    ):
        """
        calculate the individual conditional expectation for a single feature.

        Parameters
        ----------
        var_name : str
            The name of the feature to consider ICE.
        ids_to_compute : list[int]
            The index of the rows to compute.
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
        individual_prediction = np.array([
            self._counterfactual_prediction(var_index, value)[ids_to_compute]
            for value in var_range
        ])
        self.df_ice = (
            pd.DataFrame(
                data=individual_prediction,
                columns=ids_to_compute)
            .assign(**{var_name: var_range})
            .melt(id_vars=var_name, var_name='id', value_name='prediction')
        )
        self.df_instance = pd.DataFrame(
            data=self.X[ids_to_compute],
            columns=self.var_names) \
        .assign(
            instace=ids_to_compute,
            prediction=self.estimator.predict(self.X[ids_to_compute])
        )

    def plot_individual_conditional_expectation(self) -> None:
        """
        Plot the individual conditional expectation for a single feature.
        """
        fig = go.Figure()
        for id in self.df_ice['id'].unique():
            _df_ice = self.df_ice.query('id == @id')
            fig.add_trace(
                go.Scatter(
                    x=_df_ice['x1'],
                    y=_df_ice['prediction'],
                    mode='lines',
                    name=id,
                    line=dict(
                        color='blue',
                        width=1,
                    ),
                ),
            )
        fig.add_trace(
            go.Scatter(
                x=self.df_instance['x1'],
                y=self.df_instance['prediction'],
                mode='markers',
                name='Instance',
                marker=dict(
                    color='red',
                    size=10,
                ),
            ),
        )
        fig.update_layout(
            title='Individual conditional expectation for x1',
            xaxis_title='x1',
            yaxis_title='prediction',
        )
        fig.show()
            