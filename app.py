# app.py
import io
import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

# -------------
# Config
# -------------
st.set_page_config(page_title="Stock Market Visualizer", layout="wide")

# -------------
# Utils / Indicators
# -------------
@st.cache_data
def fetch_data(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLCV data via yfinance and return DataFrame with Date index."""
    ticker_obj = yf.Ticker(ticker)
    df = ticker_obj.history(period=period, interval=interval, actions=False)
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    return df

@st.cache_data
def fetch_multiple(tickers: List[str], period: str = "6mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        try:
            out[t] = fetch_data(t, period=period, interval=interval)
        except Exception:
            out[t] = pd.DataFrame()
    return out

def sma(series: pd.Series, window: int):
    return series.rolling(window=window).mean()

def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0):
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower

def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

# -------------
# Plotting
# -------------
def make_candlestick_with_indicators(
    df: pd.DataFrame,
    ma_windows: List[int] = None,
    show_rsi: bool = True,
    show_bollinger: bool = True,
    bollinger_window: int = 20,
    bollinger_std: float = 2.0,
    colors: Dict[str, str] = None,
    rsi_thresholds: Dict[str, int] = None,
):
    ma_windows = ma_windows or []
    colors = colors or {"up": "#00A86B", "down": "#D62728", "ma": "#1f77b4", "bb": "#FFA500"}
    rsi_thresholds = rsi_thresholds or {"overbought": 70, "oversold": 30}

    if show_rsi:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
                            vertical_spacing=0.08, row_heights=[0.75, 0.25])
    else:
        fig = make_subplots(rows=1, cols=1)

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="Price", increasing_line_color=colors["up"], decreasing_line_color=colors["down"]
        ),
        row=1, col=1
    )

    # Moving averages
    for w in ma_windows:
        ma_series = sma(df["Close"], w)
        fig.add_trace(go.Scatter(x=df.index, y=ma_series, mode="lines", name=f"MA{w}", line=dict(width=1.5)), row=1, col=1)

    # Bollinger
    if show_bollinger:
        ma_, upper, lower = bollinger_bands(df["Close"], window=bollinger_window, n_std=bollinger_std)
        fig.add_trace(go.Scatter(x=df.index, y=upper, name="BB Upper", line=dict(width=1), opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, name="BB Lower", fill='tonexty', line=dict(width=1), opacity=0.4), row=1, col=1)

    # RSI
    if show_rsi:
        rsi_s = rsi(df["Close"])
        fig.add_trace(go.Scatter(x=df.index, y=rsi_s, name="RSI", line=dict(width=1)), row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        # RSI thresholds lines
        fig.add_hline(y=rsi_thresholds["overbought"], line=dict(color="red", dash="dash"), row=2, col=1)
        fig.add_hline(y=rsi_thresholds["oversold"], line=dict(color="green", dash="dash"), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_white", height=750 if show_rsi else 600)
    return fig

# -------------
# Portfolio helpers
# -------------
def parse_portfolio_file(bytes_data: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(bytes_data))
    elif name.endswith(".xls") or name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(bytes_data))
    else:
        raise ValueError("Unsupported file type. Use CSV or Excel.")
    df.columns = [c.strip().lower() for c in df.columns]
    if "ticker" not in df.columns:
        raise ValueError("Portfolio must contain a 'ticker' column.")
    return df

def portfolio_value(port_df: pd.DataFrame, price_map: Dict[str, float]):
    df = port_df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0)
        df["price"] = df["ticker"].map(price_map)
        df["value"] = df["quantity"] * df["price"]
    elif "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
        df["price"] = df["ticker"].map(price_map)
        df["norm_weight"] = df["weight"] / df["weight"].sum()
        df["value"] = df["norm_weight"] * df["price"]
    else:
        df["price"] = df["ticker"].map(price_map)
        df["value"] = df["price"]
    total = df["value"].sum()
    return df, total

# -------------
# UI
# -------------
st.title("📈 Stock Market Visualizer")
st.markdown("Interactive Streamlit app using `yfinance` + `plotly`. Candlesticks, indicators, portfolio valuation, correlation, chart export, and config save/share.")

# Sidebar controls + saved configs
with st.sidebar:
    st.header("Controls & Config")
    ticker_input = st.text_input("Ticker (comma for multiple)", value="AAPL")
    period = st.selectbox("Period", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=2)
    interval = st.selectbox("Interval", options=["1d", "1wk", "1d", "60m"], index=0)
    ma_windows_str = st.text_input("Moving average windows (comma)", value="20,50")
    show_rsi = st.checkbox("Show RSI", value=True)
    show_bollinger = st.checkbox("Show Bollinger Bands", value=True)
    bb_window = st.number_input("Bollinger window", min_value=5, max_value=200, value=20)
    bb_std = st.number_input("Bollinger n_std", min_value=1.0, max_value=3.5, step=0.1, value=2.0)
    rsi_over = st.number_input("RSI Overbought", min_value=50, max_value=95, value=70)
    rsi_under = st.number_input("RSI Oversold", min_value=5, max_value=50, value=30)
    color_up = st.color_picker("Candle up color", "#00A86B")
    color_down = st.color_picker("Candle down color", "#D62728")
    btn_load = st.button("Load & Plot")
    st.markdown("---")
    st.subheader("Save / Load Configs")
    if "saved_configs" not in st.session_state:
        st.session_state["saved_configs"] = {}
    cfg_name = st.text_input("Preset name", value="my-config")
    if st.button("Save preset to session"):
        cfg = {
            "ticker_input": ticker_input,
            "period": period,
            "interval": interval,
            "ma_windows_str": ma_windows_str,
            "show_rsi": show_rsi,
            "show_bollinger": show_bollinger,
            "bb_window": bb_window,
            "bb_std": bb_std,
            "rsi_over": rsi_over,
            "rsi_under": rsi_under,
            "color_up": color_up,
            "color_down": color_down,
        }
        st.session_state.saved_configs[cfg_name] = cfg
        st.success(f"Saved preset: {cfg_name}")
    if st.session_state.saved_configs:
        chosen = st.selectbox("Load preset from session", options=list(st.session_state.saved_configs.keys()))
        if st.button("Load chosen preset"):
            loaded = st.session_state.saved_configs[chosen]
            # Overwrite local variables by simple instruction to user (Streamlit cannot auto-populate UI inputs)
            st.write("Loaded preset (you can copy values):")
            st.json(loaded)
    st.download_button("Download all presets (JSON)", data=json.dumps(st.session_state.get("saved_configs", {}), indent=2), file_name="presets.json")

# Main plotting area
col1, col2 = st.columns([3, 1])
with col1:
    if btn_load:
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        if len(tickers) == 0:
            st.warning("Please enter a ticker.")
        else:
            # For single ticker show chart; for multiple show comparison table + small charts
            if len(tickers) == 1:
                t = tickers[0]
                df = fetch_data(t, period=period, interval=interval)
                if df.empty:
                    st.error("No data found for ticker.")
                else:
                    ma_windows = []
                    try:
                        ma_windows = [int(x.strip()) for x in ma_windows_str.split(",") if x.strip()]
                    except Exception:
                        ma_windows = []
                    fig = make_candlestick_with_indicators(
                        df,
                        ma_windows=ma_windows,
                        show_rsi=show_rsi,
                        show_bollinger=show_bollinger,
                        bollinger_window=int(bb_window),
                        bollinger_std=float(bb_std),
                        colors={"up": color_up, "down": color_down},
                        rsi_thresholds={"overbought": int(rsi_over), "oversold": int(rsi_under)}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Snapshot key metrics
                    st.subheader(f"{t} latest OHLC")
                    last = df.iloc[-1]
                    st.metric("Close", f"{last['Close']:.2f}")
                    st.metric("Change", f"{((last['Close']-last['Open'])/last['Open']*100):.2f}%")
                    # financial info attempt
                    try:
                        info = yf.Ticker(t).info
                        info_display = {
                            "marketCap": info.get("marketCap"),
                            "trailingPE": info.get("trailingPE"),
                            "forwardPE": info.get("forwardPE"),
                            "dividendYield": info.get("dividendYield"),
                            "beta": info.get("beta"),
                        }
                        st.write("Key financial fields from yfinance.info:")
                        st.json(info_display)
                    except Exception:
                        st.info("No extra info from yfinance for this ticker.")

                    # Export
                    if st.button("Export chart PNG"):
                        bytes_img = pio.to_image(fig, format="png", width=1200, height=700, scale=2)
                        st.download_button("Download PNG", data=bytes_img, file_name=f"{t}_chart.png", mime="image/png")
                    if st.button("Export chart HTML"):
                        html = pio.to_html(fig, full_html=True)
                        st.download_button("Download HTML", data=html, file_name=f"{t}_chart.html", mime="text/html")

            else:
                # multiple tickers: show small multiples of close and a combined table
                data_map = fetch_multiple(tickers, period=period, interval=interval)
                close_df = pd.DataFrame({t: (data_map[t]["Close"] if (t in data_map and not data_map[t].empty) else pd.Series(dtype=float)) for t in tickers})
                if close_df.empty:
                    st.error("No data returned for provided tickers.")
                else:
                    st.subheader("Close price chart (multiple)")
                    fig_multi = go.Figure()
                    for t in close_df.columns:
                        fig_multi.add_trace(go.Scatter(x=close_df.index, y=close_df[t], name=t))
                    st.plotly_chart(fig_multi, use_container_width=True)

                    st.subheader("Correlation (returns)")
                    returns = close_df.pct_change().dropna()
                    corr = returns.corr()
                    st.dataframe(corr.style.format("{:.3f}"))

with col2:
    st.header("Portfolio")
    uploaded = st.file_uploader("Upload portfolio CSV/XLSX (columns: ticker, quantity or weight)", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            raw = uploaded.read()
            port_df = parse_portfolio_file(raw, uploaded.name)
            st.write("Preview:")
            st.dataframe(port_df.head())

            tickers = port_df["ticker"].astype(str).str.upper().unique().tolist()
            price_map = {}
            for t in tickers:
                d = fetch_data(t, period="5d", interval="1d")
                price_map[t] = float(d["Close"].iloc[-1]) if (not d.empty) else np.nan
            pv_df, total = portfolio_value(port_df, price_map)
            st.subheader("Valuation")
            st.dataframe(pv_df)
            st.metric("Total portfolio (approx)", f"{total:.2f}")

            if st.button("Download valuation CSV"):
                buf = io.StringIO()
                pv_df.to_csv(buf, index=False)
                st.download_button("Download CSV", data=buf.getvalue(), file_name="portfolio_valuation.csv")
        except Exception as e:
            st.error(f"Failed parsing portfolio: {e}")

st.markdown("---")
st.header("Correlation & Analysis (bulk)")
st.markdown("Enter tickers, select period and run correlation on returns.")
with st.form("corr"):
    tickers_text = st.text_input("Tickers (comma)", value="AAPL, MSFT, GOOGL, TSLA")
    corr_period = st.selectbox("Period for correlation", options=["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    corr_interval = st.selectbox("Interval", options=["1d", "1wk"], index=0)
    run = st.form_submit_button("Run")
if run:
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    data_map = fetch_multiple(tickers, period=corr_period, interval=corr_interval)
    close_df = pd.DataFrame({t: data_map[t]["Close"] for t in tickers if (t in data_map and not data_map[t].empty)})
    if close_df.empty:
        st.error("No data for tickers.")
    else:
        returns = close_df.pct_change().dropna()
        corr = returns.corr()
        st.subheader("Correlation matrix")
        st.dataframe(corr.style.format("{:.3f}"))
        fig_heat = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1))
        fig_heat.update_layout(height=600)
        st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")
st.info("To push this app to GitHub: initialize a repo, add files, commit, add remote, and push. See README.md for exact commands.")
