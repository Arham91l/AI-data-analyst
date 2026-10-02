import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent import (
    calc_max_drawdown,
    calc_rsi,
    calc_vol,
    get_llm,
    make_agent,
    rsi_label
)


st.set_page_config(
    page_title="AI Stock Analyst",
    page_icon="📈",
    layout="wide"
)


# ---------- Load data ----------
@st.cache_data
def load_data():

    d = pd.read_csv("market_agent_data.csv")

    d = d.rename(
        columns={"Daily_Return_%": "daily_return_pct"}
    )

    d["Date"] = pd.to_datetime(d["Date"])

    return (
        d.sort_values(["Ticker", "Date"])
        .reset_index(drop=True)
    )


# ---------- Hugging Face token ----------
def hf_token():
    return st.secrets["HF_TOKEN"]


# ---------- Load agent ----------
@st.cache_resource(show_spinner="Loading model...")
def load_agent(model, token):

    llm = get_llm(
        model,
        token
    )

    return make_agent(
        llm,
        df
    )


# ---------- Show tool trace ----------
def show_trace(trace):

    if trace:

        with st.expander(
            f"Tool calls ({len(trace)})"
        ):

            for name, args, out in trace:

                st.code(
                    f"{name}({args})\n-> {out}",
                    language="text"
                )


# ---------- Data ----------
df = load_data()

tickers = sorted(
    df["Ticker"].unique()
)


# ---------- Sidebar ----------
with st.sidebar:

    st.title("📈 AI Stock Analyst")

    ticker = st.selectbox(
        "Ticker",
        tickers
    )

    days = st.slider(
        "Chart window (trading days)",
        30,
        750,
        180,
        step=10
    )

    with st.expander("LLM settings"):

        model = st.text_input(
            "Hugging Face Model",
            "Qwen/Qwen3-8B"
        )

    st.caption(
        f"Data: {df['Date'].min():%d %b %Y} "
        f"to {df['Date'].max():%d %b %Y}"
    )


# ---------- Tabs ----------
tab_overview, tab_compare, tab_ask = st.tabs(
    [
        "Overview",
        "Compare",
        "Ask the analyst"
    ]
)


# =========================================================
# Overview
# =========================================================

with tab_overview:

    sub = (
        df[df["Ticker"] == ticker]
        .reset_index(drop=True)
        .copy()
    )

    close = sub["Close"]

    sub["SMA20"] = (
        close.rolling(20).mean()
    )

    sub["SMA50"] = (
        close.rolling(50).mean()
    )

    last = close.iloc[-1]
    prev = close.iloc[-2]

    rsi_val = calc_rsi(
        close,
        14
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Latest close",
        f"${last:,.2f}",
        f"{last / prev - 1:.2%}"
    )

    c2.metric(
        "30d volatility (ann.)",
        f"{calc_vol(close.tail(31)):.1%}"
    )

    c3.metric(
        "RSI (14)",
        f"{rsi_val:.1f}",
        rsi_label(rsi_val),
        delta_color="off"
    )

    if len(close) > 252:

        c4.metric(
            "1y return",
            f"{close.iloc[-1] / close.iloc[-253] - 1:.1%}"
        )

    c5.metric(
        "Max drawdown (1y)",
        f"{calc_max_drawdown(close.tail(252)):.1%}"
    )

    view = sub.tail(days)

    fig = go.Figure(
        go.Candlestick(
            x=view["Date"],
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            name=ticker
        )
    )

    fig.add_scatter(
        x=view["Date"],
        y=view["SMA20"],
        name="SMA 20"
    )

    fig.add_scatter(
        x=view["Date"],
        y=view["SMA50"],
        name="SMA 50"
    )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=480,
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=0
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.bar_chart(
        view.set_index("Date")["Volume"],
        height=150
    )


# =========================================================
# Compare
# =========================================================

with tab_compare:

    sel = st.multiselect(
        "Tickers",
        tickers,
        default=tickers[:4]
    )

    if len(sel) >= 2:

        wide = (
            df[df["Ticker"].isin(sel)]
            .pivot(
                index="Date",
                columns="Ticker",
                values="Close"
            )
            .tail(days)
        )

        norm = (
            wide / wide.iloc[0]
        ) * 100

        st.subheader(
            "Normalised performance (start = 100)"
        )

        st.plotly_chart(
            px.line(norm),
            use_container_width=True
        )

        st.subheader(
            "Correlation of daily returns"
        )

        corr = (
            wide
            .pct_change()
            .dropna()
            .corr()
            .round(2)
        )

        st.plotly_chart(
            px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1
            ),
            use_container_width=True
        )

    else:

        st.info(
            "Select at least two tickers."
        )


# =========================================================
# Ask the analyst
# =========================================================

with tab_ask:

    st.caption(
        "Try: 'Which ticker had the highest average close?' / "
        "'Is META overbought?' / "
        "'Compare volatility of AAPL and TSLA over 60 days'"
    )

    if "chat" not in st.session_state:

        st.session_state.chat = []

    # Display previous messages
    for m in st.session_state.chat:

        with st.chat_message(m["role"]):

            st.markdown(
                m["content"]
            )

            show_trace(
                m.get("trace")
            )

    # New question
    if q := st.chat_input(
        "Ask about the stocks..."
    ):

        history = [
            (m["role"], m["content"])
            for m in st.session_state.chat[-6:]
        ]

        st.session_state.chat.append(
            {
                "role": "user",
                "content": q
            }
        )

        with st.chat_message("user"):

            st.markdown(q)

        with st.chat_message("assistant"):

            try:

                run = load_agent(model, hf_token())

                with st.spinner(
                    "Analysing..."
                ):

                    answer, trace = run(
                        q,
                        history
                    )

            except Exception as e:

                answer = f"Model error: {e}"
                trace = []

            st.markdown(answer)

            show_trace(trace)

        st.session_state.chat.append(
            {
                "role": "assistant",
                "content": answer,
                "trace": trace
            }
        )


st.divider()

st.caption(
    "Educational analysis of historical data. "
    "Not investment advice."
)