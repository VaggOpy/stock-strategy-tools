"""Price charts with plotly: several chart types, volume and indicator panels."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tradetool.indicators.registry import INDICATORS

CHART_TYPES = {
    "candles": "Κεριά",
    "ohlc": "OHLC bars",
    "line": "Γραμμή",
    "area": "Area",
    "heikin_ashi": "Heikin-Ashi",
    "relative": "Σχετική απόδοση vs SPY",
}

UP, DOWN = "#1baf7a", "#e34948"
SERIES = ["#2a78d6", "#eb6834", "#6250d6", "#eda100", "#e87ba4", "#008300"]
NEUTRAL = "#8a8985"


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    o, c = df["open"].to_numpy(), ha_close.to_numpy()
    ha_open = np.empty(len(df))
    ha_open[0] = (o[0] + df["close"].iloc[0]) / 2
    for i in range(1, len(df)):
        ha_open[i] = (ha_open[i - 1] + c[i - 1]) / 2
    ha = pd.DataFrame({"open": ha_open, "close": c}, index=df.index)
    ha["high"] = pd.concat([df["high"], ha["open"], ha["close"]], axis=1).max(axis=1)
    ha["low"] = pd.concat([df["low"], ha["open"], ha["close"]], axis=1).min(axis=1)
    return ha


def _price_traces(df, kind, benchmark, ticker):
    if kind in ("candles", "heikin_ashi"):
        d = heikin_ashi(df) if kind == "heikin_ashi" else df
        return [go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
                               name=ticker, increasing_line_color=UP, decreasing_line_color=DOWN,
                               increasing_fillcolor=UP, decreasing_fillcolor=DOWN, line_width=1)]
    if kind == "ohlc":
        return [go.Ohlc(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                        name=ticker, increasing_line_color=UP, decreasing_line_color=DOWN)]
    if kind == "line":
        return [go.Scatter(x=df.index, y=df["close"], name=ticker, line=dict(color=SERIES[0], width=2))]
    if kind == "area":
        return [go.Scatter(x=df.index, y=df["close"], name=ticker, fill="tozeroy",
                           line=dict(color=SERIES[0], width=2), fillcolor="rgba(42,120,214,0.15)")]
    if kind == "relative":
        if benchmark is None:
            raise ValueError("relative chart needs a benchmark series")
        b = benchmark.reindex(df.index).ffill()
        rel = lambda s: (s / s.iloc[0] - 1) * 100
        return [go.Scatter(x=df.index, y=rel(df["close"]), name=ticker, line=dict(color=SERIES[0], width=2)),
                go.Scatter(x=df.index, y=rel(b), name="SPY", line=dict(color=NEUTRAL, width=2, dash="dot"))]
    raise ValueError(f"unknown chart type {kind}")


def price_chart(df: pd.DataFrame, kind="candles", indicators=(), ticker="", benchmark=None,
                start=None, log=False, volume=True, height=750) -> go.Figure:
    """indicators: list of (key, params dict). Indicators and Heikin-Ashi are computed on the whole
    df and only then cut to `start`, so long averages are already warmed up in the visible window."""
    start = pd.Timestamp(start) if start is not None else df.index.min()
    cut = lambda x: x[x.index >= start]
    overlays = [(k, p) for k, p in indicators if INDICATORS[k].overlay]
    panels = [(k, p) for k, p in indicators if not INDICATORS[k].overlay]
    if kind == "relative":
        overlays = []

    rows = 1 + int(volume) + len(panels)
    heights = [0.6] + [0.12] * int(volume) + [0.18] * len(panels)
    titles = [""] + ["Volume"] * int(volume) + [INDICATORS[k].name_with(**p) for k, p in panels]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=heights, subplot_titles=titles)

    if kind == "heikin_ashi":
        traces = _price_traces(cut(heikin_ashi(df)), "candles", None, ticker)
    else:
        traces = _price_traces(cut(df), kind, benchmark, ticker)
    for t in traces:
        fig.add_trace(t, row=1, col=1)

    color_i = 1
    for key, params in overlays:
        ind = INDICATORS[key]
        out = cut(ind.compute(df, **params))
        color = SERIES[color_i % len(SERIES)]
        color_i += 1
        for j, col in enumerate(out.columns):
            name = ind.name_with(**params) + ("" if len(out.columns) == 1 else f" {col}")
            dash = "solid" if j == 0 or col == "mid" else "dot"
            fig.add_trace(go.Scatter(x=out.index, y=out[col], name=name, legendgroup=key + str(params),
                                     line=dict(color=color, width=1.5, dash=dash)), row=1, col=1)

    row = 2
    if volume:
        d = cut(df)
        up = d["close"] >= d["open"]
        fig.add_trace(go.Bar(x=d.index, y=d["volume"], name="Volume", showlegend=False,
                             marker_color=np.where(up, UP, DOWN), opacity=0.6), row=row, col=1)
        row += 1

    for key, params in panels:
        ind = INDICATORS[key]
        out = cut(ind.compute(df, **params))
        for j, col in enumerate(out.columns):
            if col == "hist":
                fig.add_trace(go.Bar(x=out.index, y=out[col], name="hist", showlegend=False,
                                     marker_color=np.where(out[col] >= 0, UP, DOWN)), row=row, col=1)
            else:
                fig.add_trace(go.Scatter(x=out.index, y=out[col], name=f"{key} {col}" if len(out.columns) > 1 else key,
                                         showlegend=False, line=dict(color=SERIES[j], width=1.5)), row=row, col=1)
        for lvl in ind.levels:
            fig.add_hline(y=lvl, line=dict(color=NEUTRAL, width=1, dash="dash"), row=row, col=1)
        row += 1

    # hide weekends and market holidays so candles sit next to each other
    shown = cut(df).index
    missing = pd.bdate_range(shown.min(), shown.max()).difference(shown)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(values=missing)])
    fig.update_xaxes(rangeslider_visible=False)
    if log and kind != "relative":
        fig.update_yaxes(type="log", row=1, col=1)
    if kind == "relative":
        fig.update_yaxes(ticksuffix="%", row=1, col=1)
    fig.update_layout(height=height, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0))
    return fig
