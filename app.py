# app.py
# -*- coding: utf-8 -*-
"""
Stock Market Visualizer - 稳健完整版
- 兼容性强：处理 yfinance 返回的多种格式（单票、批量、MultiIndex、Series）
- 处理缺失列（Open/High/Low/Close/Adj Close）并会在必要时进行安全填充，避免 KeyError
- 含指标解释、组合估值、相关性分析、导出与配置保存
注意：
- PNG 导出需要 kaleido：pip install -U kaleido
- 推荐在虚拟环境中安装以下依赖（requirements.txt）：
  streamlit, yfinance, pandas, numpy, plotly, kaleido, openpyxl
"""

import io
import json
import time
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st

# -------------------------
# 页面配置
# -------------------------
st.set_page_config(page_title="Stock Market Visualizer", layout="wide")
st.title("📈 Stock Market Visualizer — 股票市场可视化工具")
st.markdown("说明：使用 `yfinance` 获取数据，并可视化 K 线与技术指标。")

# -------------------------
# 辅助函数：处理索引与缺失列
# -------------------------
def _remove_tz_index(df: pd.DataFrame) -> pd.DataFrame:
    """将带 tz 的 DatetimeIndex 变为无时区（inplace copy）"""
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df = df.copy()
        try:
            df.index = df.index.tz_convert(None)
        except Exception:
            df.index = df.index.tz_localize(None)
    return df

def ensure_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    确保 DataFrame 包含 'Open','High','Low','Close' 列（若缺失则尽量填充）
    - 优先使用 'Adj Close' 填充 'Close'
    - 若 Open/High/Low 缺失，用 Close 和前值做合理填充（避免 KeyError）
    NOTE: 这种填充仅为避免可视化/指标计算报错；非严格 OHLC，还请知悉。
    """
    if df is None or df.empty:
        return df

    # Make a copy to avoid unexpected side-effects
    df = df.copy()

    # Ensure index has no tz
    df = _remove_tz_index(df)

    # Fill Close with Adj Close if needed
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']

    # If Close still missing -> cannot proceed meaningfully; return as-is (caller will handle)
    if 'Close' not in df.columns:
        return df

    # Fill missing Open/High/Low:
    # Strategy:
    # - If Open missing, use previous Close when available, otherwise use Close
    # - High = max(Open, Close) if missing
    # - Low = min(Open, Close) if missing
    close = df['Close']

    # Open
    if 'Open' not in df.columns:
        # previous close
        prev_close = close.shift(1)
        df['Open'] = prev_close.fillna(close)
    else:
        # if Open exists but NaNs, fill with prev close / close
        df['Open'] = df['Open'].fillna(close.shift(1).fillna(close))

    # High
    if 'High' not in df.columns:
        df['High'] = pd.concat([df['Open'], close], axis=1).max(axis=1)
    else:
        df['High'] = df['High'].fillna(pd.concat([df['Open'], close], axis=1).max(axis=1))

    # Low
    if 'Low' not in df.columns:
        df['Low'] = pd.concat([df['Open'], close], axis=1).min(axis=1)
    else:
        df['Low'] = df['Low'].fillna(pd.concat([df['Open'], close], axis=1).min(axis=1))

    return df

# -------------------------
# 数据抓取（带缓存）
# -------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_data_single(ticker: str, period: str = "6mo", interval: str = "1d", retries: int = 3) -> pd.DataFrame:
    """
    获取单只股票数据（用 yf.download），并确保常用列存在
    - 返回 DataFrame（可能含 Open/High/Low/Close/Adj Close/Volume）
    - 对网络错误进行简单重试
    """
    ticker = str(ticker).strip().upper()
    if ticker == "":
        return pd.DataFrame()

    last_exc = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False, auto_adjust=False)
            if df is None or df.empty:
                # yfinance 可能返回空 DF（例如无交易或代码错误），直接返回空
                return pd.DataFrame()
            df = _remove_tz_index(df)
            # If 'Adj Close' exists and 'Close' missing, create Close
            if 'Close' not in df.columns and 'Adj Close' in df.columns:
                df['Close'] = df['Adj Close']
            # Ensure OHLC columns for safety
            df = ensure_ohlc_columns(df)
            return df
        except Exception as e:
            last_exc = e
            # exponential-ish backoff but bounded
            time.sleep(min(1 + attempt, 5))
            continue
    # after retries
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_data_batch(tickers: List[str], period: str = "6mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    批量抓取多只股票数据。
    - 尝试一次性下载（yf.download 支持），若返回 MultiIndex 则拆分
    - 返回字典 {ticker: DataFrame}
    """
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        return {}
    result: Dict[str, pd.DataFrame] = {t: pd.DataFrame() for t in tickers}
    try:
        raw = yf.download(tickers, period=period, interval=interval, group_by="ticker", threads=True, progress=False, auto_adjust=False)
        if raw is None or raw.empty:
            # fallback: do single fetch per ticker
            for t in tickers:
                try:
                    result[t] = fetch_data_single(t, period=period, interval=interval)
                except Exception:
                    result[t] = pd.DataFrame()
            return result

        if isinstance(raw.columns, pd.MultiIndex):
            # typical case when multiple tickers requested
            # raw[ticker] gives DataFrame for that ticker
            for t in tickers:
                if t in raw.columns.levels[0]:
                    df_t = raw[t].copy()
                    df_t = _remove_tz_index(df_t)
                    # ensure Close and OHLC
                    if 'Close' not in df_t.columns and 'Adj Close' in df_t.columns:
                        df_t['Close'] = df_t['Adj Close']
                    df_t = ensure_ohlc_columns(df_t)
                    result[t] = df_t
                else:
                    result[t] = pd.DataFrame()
        else:
            # raw has single-level columns (behavior varies); fallback to per-ticker pulls
            for t in tickers:
                try:
                    result[t] = fetch_data_single(t, period=period, interval=interval)
                except Exception:
                    result[t] = pd.DataFrame()
    except Exception:
        # if batch fails entirely, fall back to per-ticker
        for t in tickers:
            try:
                result[t] = fetch_data_single(t, period=period, interval=interval)
            except Exception:
                result[t] = pd.DataFrame()
    return result

# -------------------------
# 技术指标计算（在不丢失OHLC的前提下）
# -------------------------
def calculate_technical_indicators(df: pd.DataFrame, ma_windows: List[int] = (20, 50),
                                   rsi_period: int = 14, bb_window: int = 20, bb_std: float = 2.0) -> pd.DataFrame:
    """
    在不删除原有列的前提下为 df 添加指标列（MA{w}, BB_Mid, BB_Upper, BB_Lower, RSI）
    - df 必须至少包含 Close（若缺失，函数会直接返回 df）
    - 返回同索引扩展后的 df（原 df 会被 .copy() 后修改）
    """
    if df is None or df.empty:
        return df

    # If df doesn't have Close but has Adj Close, use it
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df = df.copy()
        df['Close'] = df['Adj Close']

    if 'Close' not in df.columns:
        # 没有 Close，则无法计算指标，直接返回
        return df

    df = df.copy()
    close = df['Close'].astype(float)

    # MA
    for w in ma_windows:
        col = f"MA{w}"
        df[col] = close.rolling(window=w, min_periods=1).mean()

    # Bollinger
    df['BB_Mid'] = close.rolling(window=bb_window, min_periods=1).mean()
    std = close.rolling(window=bb_window, min_periods=1).std(ddof=0)
    df['BB_Upper'] = df['BB_Mid'] + bb_std * std
    df['BB_Lower'] = df['BB_Mid'] - bb_std * std

    # RSI (simple rolling average method)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=rsi_period, min_periods=1).mean()
    ma_down = down.rolling(window=rsi_period, min_periods=1).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(0)

    return df

# -------------------------
# 绘图：K 线 + 指标
# -------------------------
def make_candlestick_figure(df: pd.DataFrame,
                            title: Optional[str] = None,
                            ma_windows: List[int] = (20, 50),
                            show_rsi: bool = True,
                            show_bb: bool = True,
                            colors: dict = None,
                            rsi_thresholds: dict = None) -> go.Figure:
    """
    返回 Plotly Figure；函数内部会确保绘图所需列存在（用 ensure_ohlc_columns）
    """
    if df is None or df.empty:
        return go.Figure()

    # Ensure OHLC available for plotting (this will not overwrite existing non-NaN values)
    df_plot = ensure_ohlc_columns(df)

    colors = colors or {"up": "#00A86B", "down": "#D62728"}
    rsi_thresholds = rsi_thresholds or {"overbought": 70, "oversold": 30}

    # Basic candlestick trace
    candle = go.Candlestick(
        x=df_plot.index,
        open=df_plot['Open'],
        high=df_plot['High'],
        low=df_plot['Low'],
        close=df_plot['Close'],
        name='Price',
        increasing_line_color=colors['up'],
        decreasing_line_color=colors['down']
    )

    # Subplots if RSI shown
    if show_rsi:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.06)
        fig.add_trace(candle, row=1, col=1)
    else:
        fig = go.Figure()
        fig.add_trace(candle)

    # MA lines
    for w in ma_windows:
        col = f"MA{w}"
        if col in df_plot.columns:
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], mode='lines', name=col, line=dict(width=1.5)),
                          row=1 if show_rsi else 1, col=1)

    # Bollinger bands
    if show_bb and 'BB_Upper' in df_plot.columns and 'BB_Lower' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Upper'], name='BB Upper', line=dict(width=1), opacity=0.6),
                      row=1 if show_rsi else 1, col=1)
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['BB_Lower'], name='BB Lower', line=dict(width=1), opacity=0.4, fill='tonexty'),
                      row=1 if show_rsi else 1, col=1)

    # RSI subplot
    if show_rsi and 'RSI' in df_plot.columns:
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['RSI'], name='RSI', line=dict(width=1)), row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        # Thresholds
        fig.add_hline(y=rsi_thresholds.get('overbought', 70), line=dict(color='red', dash='dash'), row=2, col=1)
        fig.add_hline(y=rsi_thresholds.get('oversold', 30), line=dict(color='green', dash='dash'), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_white', height=750 if show_rsi else 600)
    if title:
        fig.update_layout(title=title)
    return fig

# -------------------------
# 投资组合解析与估值
# -------------------------
def parse_portfolio_file(bytes_data: bytes, filename: str) -> pd.DataFrame:
    """解析上传的 CSV/XLS/XLSX，返回标准化的 DataFrame（含 ticker 小写列名）"""
    name = filename.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(bytes_data))
    elif name.endswith('.xls') or name.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(bytes_data))
    else:
        raise ValueError("仅支持 CSV / XLS / XLSX 文件")

    if df is None or df.empty:
        raise ValueError("上传文件为空或无法解析")

    # Normalize columns to lower-case without spaces
    df.columns = [str(c).strip().lower() for c in df.columns]
    if 'ticker' not in df.columns:
        if 'symbol' in df.columns:
            df.rename(columns={'symbol': 'ticker'}, inplace=True)
        else:
            raise ValueError("上传文件需包含 'ticker' 或 'symbol' 列")
    return df

def portfolio_value(port_df: pd.DataFrame, price_map: Dict[str, float]) -> (pd.DataFrame, float):
    """根据 price_map（{TICKER: price}）计算组合每项市值与总市值"""
    if port_df is None or port_df.empty:
        return pd.DataFrame(), 0.0

    df = port_df.copy()
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    df['price'] = df['ticker'].map(price_map)
    # allow quantity or weight; if none present, value=price
    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
        df['value'] = df['quantity'] * df['price']
    elif 'weight' in df.columns:
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0)
        total_w = df['weight'].sum()
        if total_w == 0:
            df['value'] = 0.0
        else:
            # This is illustrative: treat price as scalar and scale by normalized weight
            df['value'] = (df['weight'] / total_w) * df['price']
    else:
        df['value'] = df['price']

    total = df['value'].sum(min_count=1)
    total = float(total) if not np.isnan(total) else 0.0
    return df, total

# -------------------------
# UI：侧边栏控件
# -------------------------
st.sidebar.header("参数设置")
ticker_input = st.sidebar.text_input("股票代码（逗号分隔，例如 AAPL,MSFT）", value="AAPL")
period = st.sidebar.selectbox("时间范围 (period)", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=2)
interval = st.sidebar.selectbox("数据间隔 (interval)", options=["1d", "1wk", "1h", "30m"], index=0)
ma_windows_str = st.sidebar.text_input("移动均线窗口（逗号），例如：20,50", value="20,50")
show_rsi = st.sidebar.checkbox("显示 RSI 子图", value=True)
show_bb = st.sidebar.checkbox("显示 Bollinger 带", value=True)
bb_window = st.sidebar.number_input("Bollinger 窗口", min_value=5, max_value=200, value=20)
bb_std = st.sidebar.number_input("Bollinger std (n)", min_value=1.0, max_value=3.5, step=0.1, value=2.0)
rsi_over = st.sidebar.number_input("RSI 过热阈值", min_value=50, max_value=95, value=70)
rsi_under = st.sidebar.number_input("RSI 超卖阈值", min_value=5, max_value=50, value=30)
color_up = st.sidebar.color_picker("上涨蜡烛颜色", "#00A86B")
color_down = st.sidebar.color_picker("下跌蜡烛颜色", "#D62728")

# Presets (session)
if 'presets' not in st.session_state:
    st.session_state['presets'] = {}

preset_name = st.sidebar.text_input("预设名称（会话级）", value="my-preset")
if st.sidebar.button("保存当前预设到会话"):
    st.session_state['presets'][preset_name] = {
        "ticker_input": ticker_input,
        "period": period,
        "interval": interval,
        "ma_windows_str": ma_windows_str,
        "show_rsi": show_rsi,
        "show_bb": show_bb,
        "bb_window": int(bb_window),
        "bb_std": float(bb_std),
        "rsi_over": int(rsi_over),
        "rsi_under": int(rsi_under),
        "color_up": color_up,
        "color_down": color_down
    }
    st.sidebar.success(f"已保存预设：{preset_name}")

if st.session_state['presets']:
    chosen = st.sidebar.selectbox("加载会话预设", options=list(st.session_state['presets'].keys()))
    if st.sidebar.button("显示选中预设（手动复制生效）"):
        st.sidebar.json(st.session_state['presets'][chosen])
st.sidebar.download_button("下载所有会话预设（JSON）", data=json.dumps(st.session_state['presets'], ensure_ascii=False, indent=2),
                           file_name="presets.json", mime="application/json")

# -------------------------
# 主区：输入解析 & 功能执行
# -------------------------
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
try:
    ma_windows = [int(x.strip()) for x in ma_windows_str.split(",") if x.strip()]
except Exception:
    ma_windows = [20, 50]

col_left, col_right = st.columns([3, 1])

with col_left:
    st.subheader("图表与分析")

    if not tickers:
        st.info("请在侧边栏输入至少一个股票代码（Ticker）")
    elif len(tickers) == 1:
        # 单票视图：K 线 + 指标 + 导出
        ticker = tickers[0]
        try:
            df = fetch_data_single(ticker, period=period, interval=interval)
        except Exception as e:
            st.error(f"获取数据异常：{e}")
            df = pd.DataFrame()

        if df is None or df.empty:
            st.warning("未能获取到数据，请检查股票代码、时间范围或间隔。")
        else:
            # 计算指标（会返回带 Close 的 df，但不丢失 OHLC）
            df = calculate_technical_indicators(df, ma_windows=ma_windows, rsi_period=14, bb_window=bb_window, bb_std=bb_std)
            # 绘图
            fig = make_candlestick_figure(df, title=f"{ticker} — {period} / {interval}", ma_windows=ma_windows,
                                          show_rsi=show_rsi, show_bb=show_bb,
                                          colors={"up": color_up, "down": color_down},
                                          rsi_thresholds={"overbought": rsi_over, "oversold": rsi_under})
            st.plotly_chart(fig, use_container_width=True)

            # 最新数值（安全获取）
            close_series = df['Close'].dropna() if 'Close' in df.columns else pd.Series(dtype=float)
            if not close_series.empty:
                last_close = float(close_series.iloc[-1])
                prev_close = float(close_series.iloc[-2]) if len(close_series) >= 2 else None
                st.metric("最新收盘价", f"{last_close:.4f}")
                if prev_close is not None:
                    pct = (last_close / prev_close - 1) * 100
                    st.metric("最近日涨跌幅(%)", f"{pct:.2f}%")
                # RSI last
                if 'RSI' in df.columns and not np.isnan(df['RSI'].dropna().iloc[-1]):
                    st.metric("RSI(14)", f"{float(df['RSI'].dropna().iloc[-1]):.2f}")
            else:
                st.info("数据不足以显示最新价格/指标。")

            # 导出按钮（PNG & HTML）
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                if st.button("导出 PNG（需要 kaleido）"):
                    try:
                        png_bytes = pio.to_image(fig, format='png', width=1200, height=700, scale=2)
                        st.download_button("下载 PNG", data=png_bytes, file_name=f"{ticker}_chart.png", mime="image/png")
                    except Exception as e:
                        st.error("PNG 导出失败（可能未安装 kaleido）。请运行：pip install -U kaleido\n错误：" + str(e))
            with exp_col2:
                if st.button("导出 HTML"):
                    try:
                        html = pio.to_html(fig, full_html=True)
                        st.download_button("下载 HTML", data=html, file_name=f"{ticker}_chart.html", mime="text/html")
                    except Exception as e:
                        st.error(f"HTML 导出失败：{e}")

            # 尝试显示 yfinance.info 的部分字段（非必须，可能为空）
            try:
                info = yf.Ticker(ticker).info
                st.markdown("**yfinance.info（部分字段）**")
                st.json({
                    "shortName": info.get("shortName"),
                    "marketCap": info.get("marketCap"),
                    "trailingPE": info.get("trailingPE"),
                    "forwardPE": info.get("forwardPE"),
                    "dividendYield": info.get("dividendYield"),
                })
            except Exception:
                st.info("无法获取 yfinance.info（可能被限流或该字段不存在）")

    else:
        # 多票视图：对比收盘价与相关性
        st.subheader("多票对比 & 相关性")
        try:
            data_map = fetch_data_batch(tickers, period=period, interval=interval)
        except Exception as e:
            st.error(f"批量抓取失败：{e}")
            data_map = {t: pd.DataFrame() for t in tickers}

        # Build close DataFrame
        close_series_map = {}
        for t in tickers:
            df_t = data_map.get(t, pd.DataFrame())
            if df_t is None or df_t.empty:
                continue
            # Prefer Adj Close if available
            if 'Adj Close' in df_t.columns and not df_t['Adj Close'].dropna().empty:
                close_series_map[t] = df_t['Adj Close'].rename(t)
            elif 'Close' in df_t.columns and not df_t['Close'].dropna().empty:
                close_series_map[t] = df_t['Close'].rename(t)
        if not close_series_map:
            st.warning("未能取得任何收盘价数据（请检查代码/时间范围/间隔）。")
        else:
            close_df = pd.concat(close_series_map.values(), axis=1)
            close_df = close_df.sort_index().fillna(method='ffill').dropna(how='all')
            st.line_chart(close_df)

            # Returns & correlation
            returns = close_df.pct_change().dropna(how='all')
            if returns.shape[1] >= 2:
                corr = returns.corr()
                st.subheader("收益率相关性矩阵")
                st.dataframe(corr.style.format("{:.3f}"))
                try:
                    import plotly.express as px
                    fig_corr = px.imshow(corr.values, x=corr.columns, y=corr.index, color_continuous_scale='RdBu', zmin=-1, zmax=1, title="相关性热力图")
                    st.plotly_chart(fig_corr, use_container_width=True)
                except Exception:
                    st.info("绘制相关性热力图需要 plotly.express。")
            else:
                st.info("至少需要两只有效股票来计算相关性。")

with col_right:
    st.subheader("工具面板")

    # 投资组合上传与估值
    st.markdown("### 投资组合（上传 CSV/XLSX，含 ticker/quantity 或 weight）")
    uploaded = st.file_uploader("上传投资组合文件", type=['csv','xls','xlsx'])
    if uploaded is not None:
        try:
            port_df = parse_portfolio_file(uploaded.read(), uploaded.name)
            st.write("上传数据预览：")
            st.dataframe(port_df.head())
            # 获取每个 ticker 的最新价格（5 日内）
            uniques = port_df['ticker'].astype(str).str.upper().unique().tolist()
            price_map = {}
            batch_prices = fetch_data_batch(uniques, period="7d", interval="1d")
            for t in uniques:
                df_t = batch_prices.get(t, pd.DataFrame())
                if df_t is not None and not df_t.empty and 'Close' in df_t.columns and not df_t['Close'].dropna().empty:
                    price_map[t] = float(df_t['Close'].dropna().iloc[-1])
                else:
                    price_map[t] = np.nan
            pv_df, total_val = portfolio_value(port_df, price_map)
            st.markdown("组合估值（近似）")
            st.dataframe(pv_df)
            st.metric("组合总价值（近似）", f"{total_val:.2f}")
            csv_buf = io.StringIO()
            pv_df.to_csv(csv_buf, index=False)
            st.download_button("下载估值 CSV", data=csv_buf.getvalue(), file_name="portfolio_valuation.csv", mime="text/csv")
        except Exception as e:
            st.error(f"解析组合失败：{e}")

    # 配置保存/加载
    st.markdown("### 配置保存 / 加载")
    cfg = {
        "ticker_input": ticker_input,
        "period": period,
        "interval": interval,
        "ma_windows_str": ma_windows_str,
        "show_rsi": show_rsi,
        "show_bb": show_bb,
        "bb_window": bb_window,
        "bb_std": float(bb_std),
        "rsi_over": int(rsi_over),
        "rsi_under": int(rsi_under),
        "color_up": color_up,
        "color_down": color_down
    }
    st.download_button("下载当前配置（JSON）", data=json.dumps(cfg, ensure_ascii=False, indent=2), file_name="smv_config.json", mime="application/json")
    cfg_file = st.file_uploader("上传配置 JSON（预览）", type=["json"])
    if cfg_file is not None:
        try:
            loaded_cfg = json.load(cfg_file)
            st.json(loaded_cfg)
            st.info("配置仅作预览，需手动复制到侧边栏控件以生效。")
        except Exception as e:
            st.error(f"配置解析失败：{e}")

    # 简要说明 & 导出帮助
    st.markdown("---")
    st.markdown("**导出说明**：PNG 导出需要安装 `kaleido`（`pip install -U kaleido`）。HTML 导出无需额外依赖。")
    st.markdown("**注意**：yfinance 在公共环境（如 Streamlit Cloud）可能遭遇限流，请适度减少请求频率或改用付费数据源。")

# -------------------------
# 技术指标解释（底部）
# -------------------------
st.markdown("---")
st.header("📚 技术指标说明")
st.markdown("""
**移动平均线 (MA)**：对收盘价取 N 日平均，用于平滑价格波动，常见 MA20（短期）、MA50（中期）、MA200（长期）。  
**RSI (Relative Strength Index)**：衡量价格涨跌速率，一般 14 日 RSI 常用，0-100，>70 通常视为超买，<30 视为超卖。  
**布林带 (Bollinger Bands)**：由中轨（通常为 20 日 MA）与上下轨（中轨 ± n * 标准差）组成，上下轨反映波动范围，带宽扩大代表波动增大。  
**蜡烛图 (Candlestick)**：每个时间单位显示开/高/低/收信息，是短期价格行为分析的基本图形。  
**相关性 (Correlation)**：通过收益率计算资产间的相关程度（-1 到 1），构建组合时常用于风险分散分析。  
**提示**：以上指标为技术分析工具，不构成投资建议，建议结合基本面与风险管理（仓位、止损）使用。
""")
