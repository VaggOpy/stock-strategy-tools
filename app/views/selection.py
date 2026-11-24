import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tradetool.backtest.metrics import buy_and_hold, metrics
from tradetool.backtest.portfolio import simulate
from tradetool.charts.backtest import equity_chart
from tradetool.charts.price import NEUTRAL, SERIES
from tradetool.data import prices
from tradetool.data.universe import load_universe
from tradetool.selection import model
from tradetool.selection.groups import sic_groups

LABELS = {"momentum": "Momentum", "fundamentals": "Θεμελιώδη", "group_sic": "Κλάδος (SIC)",
          "group_cluster": "Ομάδα (clustering)"}


@st.cache_data
def universe():
    return load_universe()


@st.cache_resource(show_spinner="Υπολογισμός παραγόντων για όλους τους μήνες (μία φορά, περίπου 2 λεπτά)...")
def factors(sp500_then: bool):
    u = universe()
    stocks = u.query("type == 'stock'")["ticker"].tolist()
    ranks, close = model.cached_factors(stocks, u, sp500_then=sp500_then)
    open_ = prices.panel(stocks, "open").reindex(index=close.index, columns=close.columns)
    return ranks, close, open_


u = universe()

with st.sidebar:
    then = st.radio("Μετοχές", ["Μόνο όσες ήταν τότε στον S&P 500", "Οι σημερινές, σε όλη την περίοδο"],
                    help="Με τις σημερινές μετοχές, ένα backtest του 2020 μπορεί να αγοράσει μια μετοχή που μπήκε "
                         "στον δείκτη το 2024 επειδή είχε ανέβει. Αυτό ευνοεί κάθε στρατηγική που αγοράζει ό,τι ανεβαίνει.")
ranks, close, open_ = factors(then.startswith("Μόνο"))
last = ranks["momentum"].index[-1]

with st.sidebar:
    st.caption("Βάρη παραγόντων")
    weights = {k: st.slider(v, 0.0, 1.0, 0.0 if k == "group_cluster" else 1.0, 0.25, key=f"w_{k}")
               for k, v in LABELS.items()}
    n = st.slider("Μετοχές στο καλάθι", 5, 50, 20, 5)
    cost = st.number_input("Κόστος ανά συναλλαγή (bps)", 0.0, 50.0, 5.0, 1.0)

st.title("Επιλογή μετοχών")
if sum(weights.values()) == 0:
    st.warning("Όλα τα βάρη είναι 0.")
    st.stop()
score = model.score(ranks, weights)

st.subheader(f"Καλάθι στις {last:%d/%m/%Y}")
st.caption("Κάθε παράγοντας είναι εκατοστημόριο ανάμεσα στις μετοχές (1 = η καλύτερη), με όσα ήταν γνωστά εκείνη "
           "τη μέρα. Το σκορ είναι ο σταθμισμένος μέσος τους.")
s = score.loc[last].dropna().sort_values(ascending=False).head(n)
info = u.set_index("ticker").reindex(s.index)
table = pd.DataFrame({"Εταιρεία": info["name"], "Κλάδος SEC": info["sic_desc"], "Σκορ": s}
                     | {LABELS[k]: ranks[k].loc[last, s.index] for k in LABELS})
st.dataframe(table.style.format({c: "{:.2f}" for c in ["Σκορ", *LABELS.values()]}, na_rep="-"), width="stretch")

st.subheader("Δύναμη κλάδων (group strength)")
groups = sic_groups(u)
mom = ranks["momentum"].loc[last]
g = pd.DataFrame({"group": groups, "momentum": mom}).dropna()
names = u.set_index("ticker")["sic_desc"]
strength = (g.groupby("group").agg(strength=("momentum", "mean"), members=("momentum", "size"))
            .query("members >= 3").sort_values("strength"))
strength["label"] = [f"{k.replace('sic', 'SIC ').replace('div', 'Τομέας ')}: "
                     f"{names.reindex(g.index[g['group'] == k]).mode().iat[0][:40]}" for k in strength.index]
show = pd.concat([strength.head(8), strength.tail(8)])
fig = go.Figure(go.Bar(x=show["strength"], y=show["label"], orientation="h",
                       marker_color=[SERIES[0] if v >= 0.5 else NEUTRAL for v in show["strength"]],
                       customdata=show["members"],
                       hovertemplate="%{y}<br>μέσο momentum %{x:.2f}<br>%{customdata} μετοχές<extra></extra>"))
fig.add_vline(x=0.5, line=dict(color=NEUTRAL, dash="dot"))
fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="μέσο momentum των μελών (0-1)")
st.plotly_chart(fig, width="stretch")
st.caption("Οι 8 πιο δυνατοί και οι 8 πιο αδύναμοι κλάδοι (ομάδες SIC δύο ψηφίων με τουλάχιστον 3 μετοχές). "
           "Δίπλα στον κωδικό, ο πιο συχνός κλάδος των μελών.")

st.subheader("Backtest του καλαθιού")
# month ends with enough scored stocks; the last date is today's basket, not traded yet
scored = ranks["momentum"].notna().sum(axis=1)
months = scored.index[:-1][scored.iloc[:-1] >= 200]
res = simulate(model.baskets(score.loc[months], n), open_, close, cost)
idx = res.equity.index
eligible = ranks["momentum"].loc[months].notna()
ew = simulate({d: pd.Series(1 / eligible.loc[d].sum(), index=eligible.columns[eligible.loc[d]])
               for d in eligible.index}, open_, close, cost).equity
spy = buy_and_hold(prices.load("SPY"), idx)
curves = {f"Καλάθι {n} μετοχών": res.equity, "Όλες οι μετοχές, ίσα βάρη": ew.reindex(idx), "SPY": spy}
st.plotly_chart(equity_chart(curves, log=True), width="stretch")
m = pd.DataFrame({k: metrics(v.dropna()) for k, v in curves.items()}).T[["cagr", "volatility", "sharpe", "max_drawdown"]]
m.columns = ["CAGR", "Μεταβλητότητα", "Sharpe", "Max drawdown"]
st.dataframe(m.style.format({"CAGR": "{:.1%}", "Μεταβλητότητα": "{:.1%}", "Sharpe": "{:.2f}",
                             "Max drawdown": "{:.1%}"}), width="stretch")
st.caption(f"Αλλαγή καλαθιού κάθε μήνα στο άνοιγμα της πρώτης μέρας, με το σκορ του προηγούμενου κλεισίματος. "
           f"Μέσος τζίρος ανά αλλαγή {res.turnover.mean():.0%}. Οι μετοχές είναι τα σημερινά μέλη των δεικτών, οπότε "
           "και το καλάθι με όλες τις μετοχές είναι ευνοημένο (survivorship bias).")
