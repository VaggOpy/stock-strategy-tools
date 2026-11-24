"""Charts for backtest results: equity curves with drawdown, and trade markers on the price chart."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tradetool.charts.price import DOWN, NEUTRAL, SERIES, UP
from tradetool.indicators.registry import INDICATORS


def equity_chart(curves: dict[str, pd.Series], log=False, height=560) -> go.Figure:
    """curves: name -> equity starting at 1. The first one is the strategy. Lower panel: drawdown."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.7, 0.3],
                        subplot_titles=["", "Drawdown"])
    fig.update_yaxes(title_text="Αξία (αρχή = 1)", row=1, col=1)
    styles = [dict(color=SERIES[0], width=2), dict(color=SERIES[1], width=1.5),
              dict(color=NEUTRAL, width=1.5, dash="dot")]
    for (name, eq), line in zip(curves.items(), styles):
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name, line=line, legendgroup=name), row=1, col=1)
        dd = (eq / eq.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=name, line=line, legendgroup=name, showlegend=False),
                      row=2, col=1)
    fig.update_yaxes(ticksuffix="%", row=2, col=1)
    if log:
        fig.update_yaxes(type="log", row=1, col=1)
    fig.update_layout(height=height, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig


def rule_indicators(rule: dict) -> list[tuple[str, dict]]:
    """The indicators a resolved rule uses, as (key, params) for price_chart."""
    out = []
    for c in rule["entry"] + rule.get("exit", []):
        for op in (c["left"], c["right"]):
            if "ind" in op:
                ind = INDICATORS[op["ind"]]
                item = (ind.key, {p.name: op[p.name] for p in ind.params if p.name in op})
                if item not in out:
                    out.append(item)
    return out


def add_trades(fig: go.Figure, trades: pd.DataFrame, start=None):
    t = trades if start is None else trades[trades["entry_date"] >= pd.Timestamp(start)]
    fig.add_trace(go.Scatter(x=t["entry_date"], y=t["entry_price"], mode="markers", name="Αγορά",
                             marker=dict(symbol="triangle-up", size=11, color=UP, line=dict(color="white", width=1))),
                  row=1, col=1)
    done = t.dropna(subset=["exit_date"])
    fig.add_trace(go.Scatter(x=done["exit_date"], y=done["exit_price"], mode="markers", name="Πώληση",
                             text=done["reason"], hovertemplate="%{y:.2f} (%{text})",
                             marker=dict(symbol="triangle-down", size=11, color=DOWN,
                                         line=dict(color="white", width=1))),
                  row=1, col=1)
    return fig
