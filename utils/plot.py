import plotly.graph_objects as go


def plot_coefficients(lr):
    n_coefs = len(lr.coef_)
    fig = go.Figure(
        data=[
            go.Bar(
                x=lr.coef_,
                y=[f'x{i}' for i in range(n_coefs)],
                orientation='h',
                name='Regression coefficients',
            ),
        ]
    )
    fig.update_layout(  
        title='Regression coefficients',
        xaxis_title='Coefficient',
        yaxis_title='Feature',
        template='plotly_white',
    )
    fig.show()
