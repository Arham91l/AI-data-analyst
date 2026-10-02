import os
import re
import sqlite3

import pandas as pd
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool


# ---------- indicators ----------
def close_series(df, ticker, n=None):
    s = df.loc[df["Ticker"] == ticker, "Close"]
    return s.tail(n) if n else s


def calc_vol(close):
    """Annualised volatility: std of daily returns x sqrt(252)."""
    return close.pct_change().dropna().std() * 252 ** 0.5


def calc_rsi(close, window=14):
    """RSI using simple averages of gains/losses (Cutler's RSI)."""
    d = close.diff().dropna().tail(window)
    gain = d.clip(lower=0).mean()
    loss = (-d.clip(upper=0)).mean()
    return 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)


def calc_max_drawdown(close):
    return (close / close.cummax() - 1).min()


def rsi_label(value):
    return "overbought" if value > 70 else "oversold" if value < 30 else "neutral"


# ---------- tools ----------
def build_tools(df):

    conn = sqlite3.connect(":memory:", check_same_thread=False)

    df.assign(
        Date=df["Date"].dt.strftime("%Y-%m-%d")
    ).to_sql(
        "market_data",
        conn,
        index=False
    )

    conn.execute(
        "CREATE INDEX idx_ticker_date ON market_data (Ticker, Date)"
    )

    conn.execute("PRAGMA query_only = ON")

    tickers = set(df["Ticker"].unique())

    def valid(t):
        t = t.strip().upper()
        return t if t in tickers else None

    def unknown(t):
        return (
            f"No data for '{t}'. Available tickers: "
            f"{', '.join(sorted(tickers))}"
        )

    def table_names():
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()

        return {r[0] for r in rows}

    @tool
    def list_tables() -> str:
        """List database tables."""
        return ", ".join(sorted(table_names()))

    @tool
    def get_schema(table: str) -> str:
        """Get column names, types and sample rows of a table."""

        table = table.strip()

        if table not in table_names():
            return (
                f"Unknown table '{table}'. "
                f"Available: {', '.join(sorted(table_names()))}"
            )

        cols = pd.read_sql_query(
            f"PRAGMA table_info({table})",
            conn
        )[["name", "type"]]

        sample = pd.read_sql_query(
            f"SELECT * FROM {table} LIMIT 3",
            conn
        )

        return (
            f"Columns:\n{cols.to_string(index=False)}"
            f"\n\nSample rows:\n{sample.to_string(index=False)}"
        )

    @tool
    def execute_sql(query: str) -> str:
        """Run one read-only SQLite SELECT query on market_data."""

        q = query.strip().rstrip(";").strip()

        if not q.lower().startswith("select") or ";" in q:
            return "Only a single SELECT statement is allowed."

        try:
            out = pd.read_sql_query(
                f"SELECT * FROM ({q}) LIMIT 50",
                conn
            )

        except Exception as e:
            return f"SQL error: {e}"

        return (
            out.to_string(index=False)
            if not out.empty
            else "Query returned no results."
        )

    @tool
    def volatility(ticker: str, window: int = 30) -> str:
        """Annualised volatility of one ticker."""

        t = valid(ticker)

        if not t:
            return unknown(ticker)

        v = calc_vol(
            close_series(df, t, window + 1)
        )

        return (
            f"{t} annualised volatility over last "
            f"{window} trading days: {v:.2%}"
        )

    @tool
    def rsi(ticker: str, window: int = 14) -> str:
        """Calculate RSI of one ticker."""

        t = valid(ticker)

        if not t:
            return unknown(ticker)

        v = calc_rsi(
            close_series(df, t, window + 1),
            window
        )

        return (
            f"{t} RSI({window}) = "
            f"{v:.1f} ({rsi_label(v)})"
        )

    @tool
    def macd(ticker: str) -> str:
        """Calculate MACD, signal line and histogram."""

        t = valid(ticker)

        if not t:
            return unknown(ticker)

        c = close_series(df, t)

        line = (
            c.ewm(span=12, adjust=False).mean()
            -
            c.ewm(span=26, adjust=False).mean()
        )

        signal = line.ewm(
            span=9,
            adjust=False
        ).mean()

        hist = line.iloc[-1] - signal.iloc[-1]

        side = "above" if hist > 0 else "below"

        return (
            f"{t} MACD={line.iloc[-1]:.2f}, "
            f"signal={signal.iloc[-1]:.2f}, "
            f"histogram={hist:.2f} "
            f"(MACD {side} signal)"
        )

    @tool
    def bollinger_bands(ticker: str) -> str:
        """Calculate 20-day Bollinger Bands."""

        t = valid(ticker)

        if not t:
            return unknown(ticker)

        c = close_series(df, t)

        sma = c.rolling(20).mean().iloc[-1]
        std = c.rolling(20).std().iloc[-1]

        upper = sma + 2 * std
        lower = sma - 2 * std
        last = c.iloc[-1]

        if last > upper:
            pos = "above upper band"
        elif last < lower:
            pos = "below lower band"
        else:
            pos = "inside the bands"

        return (
            f"{t} close={last:.2f}, "
            f"SMA20={sma:.2f}, "
            f"upper={upper:.2f}, "
            f"lower={lower:.2f} "
            f"({pos})"
        )

    @tool
    def max_drawdown(ticker: str, days: int = 252) -> str:
        """Calculate maximum drawdown."""

        t = valid(ticker)

        if not t:
            return unknown(ticker)

        value = calc_max_drawdown(
            close_series(df, t, days)
        )

        return (
            f"{t} max drawdown over last "
            f"{days} trading days: {value:.2%}"
        )

    return [
        list_tables,
        get_schema,
        execute_sql,
        volatility,
        rsi,
        macd,
        bollinger_bands,
        max_drawdown,
    ]


# ---------- Hugging Face LLM ----------
def get_llm(model, hf_token):
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    endpoint = HuggingFaceEndpoint(
        repo_id=model,
        task="text-generation",
        max_new_tokens=512,
        temperature=0.01,
        huggingfacehub_api_token=hf_token,
    )

    return ChatHuggingFace(llm=endpoint)


# ---------- agent ----------
def make_agent(llm, df):

    tools = build_tools(df)

    tool_map = {
        t.name: t
        for t in tools
    }

    bound = llm.bind_tools(tools)

    system = SystemMessage(
        f"""
You are a stock market data analyst.

Data covers ONLY these tickers:
{', '.join(sorted(df['Ticker'].unique()))}

Data period:
{df['Date'].min():%Y-%m-%d}
to
{df['Date'].max():%Y-%m-%d}

Rules:

- Every number must come from a tool.
- Never guess numerical values.
- For comparisons or rankings across tickers, use execute_sql.
- Use get_schema before writing SQL if necessary.
- Repeat tool labels such as overbought, oversold and neutral.
- Do not provide buy/sell advice.
"""
    )

    def run(question, history=None, max_steps=6):

        messages = [system]

        for role, text in (history or []):

            if role == "user":
                messages.append(
                    HumanMessage(text)
                )

            else:
                messages.append(
                    AIMessage(text)
                )

        messages.append(
            HumanMessage(question)
        )

        trace = []

        for _ in range(max_steps):

            ai = bound.invoke(messages)

            messages.append(ai)

            if not ai.tool_calls:

                text = re.sub(
                    r"<think>.*?</think>",
                    "",
                    str(ai.content),
                    flags=re.S
                ).strip()

                return text, trace

            for call in ai.tool_calls:

                tool_name = call["name"]

                t = tool_map.get(tool_name)

                try:

                    out = (
                        t.invoke(call["args"])
                        if t
                        else f"Unknown tool {tool_name}"
                    )

                except Exception as e:

                    out = f"Tool error: {e}"

                trace.append(
                    (
                        tool_name,
                        call["args"],
                        str(out)
                    )
                )

                messages.append(
                    ToolMessage(
                        content=str(out),
                        tool_call_id=call["id"]
                    )
                )

        return "Stopped: too many steps.", trace

    return run