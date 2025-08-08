import plotly.graph_objs as go
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np

class VisualizationDashboard:
    def __init__(self, data):
        """
        data: pandas DataFrame with columns for scenarios, losses, and other risk metrics
        """
        self.data = data
        self.app = dash.Dash(__name__)
        self._build_layout()
        self._register_callbacks()

    def _build_layout(self):
        self.app.layout = html.Div([
            html.H1("RiskSimPro Interactive Dashboard"),

            html.Div([
                html.Label("Select Scenario:"),
                dcc.Dropdown(
                    id='scenario-dropdown',
                    options=[{'label': str(s), 'value': s} for s in sorted(self.data['scenario'].unique())],
                    value=sorted(self.data['scenario'].unique())[0]
                )
            ], style={'width': '30%', 'display': 'inline-block'}),

            dcc.Graph(id='loss-distribution'),

            html.Div([
                html.Label("Select Risk Metric:"),
                dcc.RadioItems(
                    id='metric-selector',
                    options=[
                        {'label': 'Loss', 'value': 'loss'},
                        {'label': 'Profit & Loss', 'value': 'pnl'},
                        {'label': 'Value at Risk (VaR)', 'value': 'var'},
                        {'label': 'Conditional VaR (CVaR)', 'value': 'cvar'}
                    ],
                    value='loss',
                    labelStyle={'display': 'inline-block', 'margin-right': '10px'}
                )
            ], style={'margin-top': '20px'}),

            html.Div(id='stats-output', style={'padding': '20px', 'fontSize': '16px'})
        ])

    def _register_callbacks(self):
        @self.app.callback(
            Output('loss-distribution', 'figure'),
            Output('stats-output', 'children'),
            Input('scenario-dropdown', 'value'),
            Input('metric-selector', 'value'))
        def update_graph(selected_scenario, selected_metric):
            filtered = self.data[self.data['scenario'] == selected_scenario]

            if selected_metric == 'loss':
                y = filtered['loss']
                title = f'Loss Distribution for Scenario {selected_scenario}'
            elif selected_metric == 'pnl':
                y = filtered['pnl']
                title = f'Profit & Loss Distribution for Scenario {selected_scenario}'
            elif selected_metric == 'var':
                y = filtered['loss']
                title = f'VaR (95%) for Scenario {selected_scenario}'
            elif selected_metric == 'cvar':
                y = filtered['loss']
                title = f'CVaR (95%) for Scenario {selected_scenario}'
            else:
                y = filtered['loss']
                title = f'Loss Distribution for Scenario {selected_scenario}'

            if selected_metric in ['loss', 'pnl']:
                fig = px.histogram(y, nbins=50, title=title, labels={'value': selected_metric.capitalize()})
            else:
                # Calculate VaR or CVaR 95%
                var_95 = np.percentile(y, 5)
                if selected_metric == 'var':
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=-var_95,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"VaR 95% ({selected_scenario})"},
                        gauge={
                            'axis': {'range': [min(y), max(y)]},
                            'bar': {'color': "red"},
                        }))
                else:
                    cvar_95 = y[y <= var_95].mean()
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=-cvar_95,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"CVaR 95% ({selected_scenario})"},
                        gauge={
                            'axis': {'range': [min(y), max(y)]},
                            'bar': {'color': "red"},
                        }))

            stats = f"Mean: {np.mean(y):.4f} | Std Dev: {np.std(y):.4f} | Median: {np.median(y):.4f}"            
            return fig, stats

    def run(self, debug=False):
        self.app.run_server(debug=debug)
