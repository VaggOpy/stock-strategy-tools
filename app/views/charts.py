import pandas as pd
import streamlit as st

from tradetool.charts.price import CHART_TYPES, price_chart
from tradetool.data import fundamentals, prices
from tradetool.data.universe import load_universe
from tradetool.indicators.registry import INDICATORS

PERIODS = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "Όλο": None}


@st.cache_data
def universe():
    return load_universe()


@st.cache_data
def load(ticker):
    return prices.load(ticker)


@st.cache_data
def load_fund(ticker):
    try:
        return fundamentals.load(ticker)
    except Exception:
        return None


u = universe()

with st.sidebar:
    kind_filter = st.radio("Τι", ["Μετοχές", "ETFs"], horizontal=True)
    sub = u[u["type"] == ("stock" if kind_filter == "Μετοχές" else "etf")]
    labels = dict(zip(sub["ticker"], sub["ticker"] + "  " + sub["name"].fillna("")))
    default = "AAPL" if kind_filter == "Μετοχές" else "SPY"
    ticker = st.selectbox("Σύμβολο", list(labels), index=list(labels).index(default), format_func=labels.get)

    kind = st.selectbox("Τύπος γραφήματος", list(CHART_TYPES), format_func=CHART_TYPES.get)
    period = st.radio("Περίοδος", list(PERIODS), index=2, horizontal=True)
    c1, c2 = st.columns(2)
    log = c1.checkbox("Log κλίμακα")
    volume = c2.checkbox("Όγκος", value=True)
    show_filings = st.checkbox("Ημερομηνίες κατάθεσης στη SEC", disabled=kind_filter != "Μετοχές")

    st.divider()
    chosen = st.multiselect("Δείκτες", list(INDICATORS), default=["EMA", "RSI"],
                            format_func=lambda k: f"{k}  {INDICATORS[k].label}")
    indicators = []
    for key in chosen:
        ind = INDICATORS[key]
        with st.expander(ind.label, expanded=False):
            if [p.name for p in ind.params] == ["length"]:
                # one length parameter: allow several lines at once, e.g. EMA 21, 50, 200
                default_len = "21, 50, 200" if key == "EMA" else f"{ind.params[0].default:g}"
                text = st.text_input("Μήκη (με κόμμα)", default_len, key=f"{key}_len")
                for v in text.replace(" ", "").split(","):
                    if v.isdigit() and ind.params[0].low <= int(v) <= ind.params[0].high:
                        indicators.append((key, {"length": int(v)}))
            else:
                params = {}
                for p in ind.params:
                    if float(p.step).is_integer():
                        params[p.name] = st.number_input(p.name, int(p.low), int(p.high), int(p.default),
                                                         key=f"{key}_{p.name}")
                    else:
                        params[p.name] = st.number_input(p.name, float(p.low), float(p.high), float(p.default),
                                                         step=float(p.step), key=f"{key}_{p.name}")
                indicators.append((key, params))

try:
    df = load(ticker)
except KeyError:
    st.error(f"Δεν υπάρχουν δεδομένα για {ticker}.")
    st.stop()

n = PERIODS[period]
start = df.index[-n] if n and n < len(df) else df.index[0]
benchmark = load("SPY")["close"] if kind == "relative" else None

row = u[u["ticker"] == ticker].iloc[0]
last, prev = df["close"].iloc[-1], df["close"].iloc[-2]
year = df["close"].iloc[-252:]
st.subheader(f"{ticker}  {row['name']}")
if isinstance(row.get("sector"), str):
    st.caption(f"{row['sector']} · {row['industry']}")
m1, m2, m3 = st.columns(3)
m1.metric("Κλείσιμο", f"{df['close_raw'].iloc[-1]:,.2f}", f"{(last / prev - 1) * 100:+.2f}%")
m2.metric("Από υψηλό 52 εβδ.", f"{(last / year.max() - 1) * 100:.1f}%")
m3.metric("Απόδοση περιόδου", f"{(last / df.loc[start, 'close'] - 1) * 100:+.1f}%")

f = load_fund(ticker) if row["type"] == "stock" else None
filings = None
if show_filings and f is not None and len(f):
    filings = f.loc[f["concept"] == "revenue", "filed"].unique()

fig = price_chart(df, kind=kind, indicators=indicators, ticker=ticker, benchmark=benchmark,
                  start=start, log=log, volume=volume, events=filings)
st.plotly_chart(fig, use_container_width=True, theme="streamlit")

st.caption(f"Ημερήσια δεδομένα {df.index[0].date()} έως {df.index[-1].date()}. Οι τιμές στο γράφημα είναι "
           "διορθωμένες για splits και μερίσματα· το «Κλείσιμο» πάνω είναι η πραγματική τιμή.")

if f is not None and len(f):
    st.subheader("Θεμελιώδη (SEC)")
    today = df.index[-1] + pd.Timedelta(days=1)
    g1, g2, g3 = st.columns(3)
    g1.metric("Έσοδα 12μήνου", f"{fundamentals.ttm(f, 'revenue', today) / 1e9:,.1f} δισ.")
    for col, concept, label in [(g2, "revenue", "Αύξηση εσόδων"), (g3, "eps_diluted", "Αύξηση EPS")]:
        gr = fundamentals.ttm_growth(f, concept, today)
        col.metric(label + " (12μ. έναντι προηγ. 12μ.)", "-" if pd.isna(gr) else f"{gr * 100:+.1f}%")

    flows = ["revenue", "net_income", "eps_diluted", "op_cash_flow"]
    w = f[f["concept"].isin(flows)].pivot_table(index="end", columns="concept", values="value")
    filed = f[f["concept"] == "revenue"].set_index("end")["filed"]
    w = w.reindex(columns=flows).tail(8).iloc[::-1]
    table = pd.DataFrame({
        "Τρίμηνο έως": w.index.date,
        "Γνωστό από": (filed.reindex(w.index) + pd.Timedelta(days=1)).dt.date.values,
        "Έσοδα (εκ.)": (w["revenue"] / 1e6).round(0),
        "Καθαρά κέρδη (εκ.)": (w["net_income"] / 1e6).round(0),
        "EPS": w["eps_diluted"].round(2),
        "Λειτ. ταμ. ροές (εκ.)": (w["op_cash_flow"] / 1e6).round(0),
    })
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Κάθε τιμή είναι η πρώτη που κατατέθηκε στη SEC και χρησιμοποιείται από την επόμενη μέρα της "
               "κατάθεσης. Το 4ο τρίμηνο και οι ταμειακές ροές υπολογίζονται από σωρευτικές τιμές (π.χ. έτος - 9μηνο).")
elif row["type"] == "stock":
    st.info("Δεν βρέθηκαν τριμηνιαία στοιχεία στη SEC (π.χ. ξένες εταιρείες που καταθέτουν μόνο ετήσιο 20-F).")
