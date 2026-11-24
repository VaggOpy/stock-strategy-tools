"""Sensitivity maps: Sharpe of every parameter combination in the train and the test window."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tradetool.charts.price import NEUTRAL, SERIES

# diverging: red for losing combinations, grey at 0, blue for winning ones
SCALE = [[0, "#e34948"], [0.5, "#e8e6e1"], [1, "#2a78d6"]]
MARKERS = {"fixed": "circle", "grid": "x", "optuna": "diamond", "plateau": "square", "pooled": "star"}


def sensitivity_chart(axes: dict, s_in: np.ndarray, s_out: np.ndarray, picks: dict[str, list[int]],
                      height=480) -> go.Figure:
    names = list(axes)
    shape = [len(v) for v in axes.values()]
    titles = ["Εντός δείγματος (εκπαίδευση)", "Εκτός δείγματος (έλεγχος)"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=titles, horizontal_spacing=0.08, shared_yaxes=True)
    picks = {m: c for m, c in picks.items() if m != "topk"}

    if len(names) == 1:
        x = axes[names[0]]
        for col, s in ((1, s_in), (2, s_out)):
            fig.add_trace(go.Scatter(x=x, y=s, mode="lines+markers", name="Sharpe", showlegend=False,
                                     line=dict(color=SERIES[0], width=2)), row=1, col=col)
            fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1), row=1, col=col)
            for j, (m, cols) in enumerate(picks.items()):
                fig.add_trace(go.Scatter(x=[x[cols[0]]], y=[s[cols[0]]], mode="markers", name=m,
                                         showlegend=col == 1, legendgroup=m,
                                         marker=dict(symbol=MARKERS.get(m, "circle"), size=12,
                                                     color=SERIES[(j + 1) % len(SERIES)],
                                                     line=dict(color="white", width=1))), row=1, col=col)
        fig.update_xaxes(title_text=names[0])
        fig.update_yaxes(title_text="Sharpe", row=1, col=1)
    else:
        lim = float(np.nanmax(np.abs(np.concatenate([s_in, s_out])))) or 1.0
        x, y = axes[names[1]], axes[names[0]]
        for col, s in ((1, s_in), (2, s_out)):
            fig.add_trace(go.Heatmap(z=s.reshape(shape), x=x, y=y, colorscale=SCALE, zmin=-lim, zmax=lim,
                                     showscale=col == 2, colorbar=dict(title="Sharpe"),
                                     hovertemplate=f"{names[0]}=%{{y}}<br>{names[1]}=%{{x}}<br>Sharpe %{{z:.2f}}"
                                                   "<extra></extra>"), row=1, col=col)
            for m, cols in picks.items():
                i, j = np.unravel_index(cols[0], shape)
                fig.add_trace(go.Scatter(x=[x[j]], y=[y[i]], mode="markers", name=m, showlegend=col == 1,
                                         legendgroup=m, marker=dict(symbol=MARKERS.get(m, "circle"), size=13,
                                                                    color="black", line=dict(color="white",
                                                                                             width=1.5))),
                              row=1, col=col)
        fig.update_xaxes(title_text=names[1])
        fig.update_yaxes(title_text=names[0], row=1, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", yanchor="top", y=-0.15, x=0))
    return fig
