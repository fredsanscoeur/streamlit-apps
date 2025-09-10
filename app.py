# app.py
# -*- coding: utf-8 -*-
"""
Stock Market Visualizer - 完整修正版
修复 BB_Upper 多列报错
功能：
1. 股票数据抓取（单票/多票）
2. 技术指标计算（MA, RSI, Bollinger Bands）
3. K线图 + 技术指标 Plotly 绘制
4. 投资组合上传/估值
5. 图表导出 PNG/HTML
6. 界面中文注释 + 技术指标说明
"""

import io
import time
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st

# -------------------------
# 页面设置
# -------------------------
st.set_page_config(page_title="📈 股票市场可视化工具", layout="wide")
st.title("📈 股票市场可视化工具")
st.markdown("说明：使用 `yfinance` 获取股票数据，并可视化 K 线图及技术指标。")

# -------------------------
# 数据抓取函数
# -------------------------
@st.cache_data(ttl=900)
def fetch_data_single(ticker: str, period: str = "6mo", interval: str = "1d", retries: int = 3) -> pd.DataFrame:
    """获取单票股票历史数据"""
    ticker = ticker.strip().upper()
    last_exc = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False)
            if df.empty:
                continue
            df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            last_exc = e
            time.sleep(1 + attempt)
    if last_exc:
        raise last_exc
    return pd.DataFrame()

@st.cache_data(ttl=900)
def fetch_data_batch(tickers: List[str], period: str = "6mo", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """批量获取多票数据"""
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    result = {t: pd.DataFrame() for t in tickers}
    if not tickers:
        return result
    try:
        raw = yf.download(tickers, period=period, interval=interval, group_by="ticker", threads=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            for t in tickers:
                if t in raw.columns.levels[0]:
                    df_t = raw[t].copy()
                    df_t.index = df_t.index.tz_localize(None)
                    result[t] = df_t
        else:
            for t in tickers:
                result[t] = fetch_data_single(t, period=period, interval=interval)
    except Exception:
        for t in tickers:
            try:
                result[t] = fetch_data_single(t, period=period, interval=interval)
            except:
                result[t] = pd.DataFrame()
    return result

# -------------------------
# 技术指标计算函数
# -------------------------
def calculate_technical_indicators(df: pd.DataFrame, ma_windows=[20,50], rsi_period=14, bb_window=20, bb_std=2.0) -> pd.DataFrame:
    """
    计算 MA、RSI、Bollinger Bands
    修复多列报错，确保 df 是单票 DataFrame
    """
    if df.empty:
        return df

    # 如果 df 是 MultiIndex 列，先取 Close 或 Adj Close 单列
    if isinstance(df.columns, pd.MultiIndex):
        if 'Adj Close' in df.columns.get_level_values(0):
            df_single = df['Adj Close'].copy()
        else:
            df_single = df['Close'].copy()
        df = df_single.to_frame('Close')
    elif 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    elif 'Close' not in df.columns:
        return df

    df = df.sort_index()
    close = df['Close']

    # 移动均线
    for w in ma_windows:
        df[f'MA{w}'] = close.rolling(window=w, min_periods=1).mean()

    # Bollinger Bands
    df['BB_Mid'] = close.rolling(window=bb_window, min_periods=1).mean()
    std = close.rolling(window=bb_window, min_periods=1).std(ddof=0)
    df['BB_Upper'] = df['BB_Mid'] + bb_std * std
    df['BB_Lower'] = df['BB_Mid'] - bb_std * std

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0,np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(0)
    return df

# -------------------------
# K线图绘制函数
# -------------------------
def make_candlestick_figure(df: pd.DataFrame, ma_windows=[20,50], show_rsi=True, show_bb=True, colors=None, rsi_thresholds=None):
    if df.empty:
        return go.Figure()
    colors = colors or {"up":"#00A86B","down":"#D62728"}
    rsi_thresholds = rsi_thresholds or {"overbought":70,"oversold":30}

    candle = go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                             increasing_line_color=colors['up'], decreasing_line_color=colors['down'], name='价格')
    if show_rsi:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7,0.3])
        fig.add_trace(candle, row=1, col=1)
    else:
        fig = go.Figure(candle)

    # 添加均线
    for w in ma_windows:
        col = f'MA{w}'
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col), row=1 if show_rsi else 1, col=1)

    # 添加布林带
    if show_bb and 'BB_Upper' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB上轨', line=dict(width=1), opacity=0.6), row=1 if show_rsi else 1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB下轨', line=dict(width=1), opacity=0.6, fill='tonexty'), row=1 if show_rsi else 1, col=1)

    # 添加RSI
    if show_rsi and 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0,100])
        fig.add_hline(y=rsi_thresholds['overbought'], line=dict(color='red', dash='dash'), row=2, col=1)
        fig.add_hline(y=rsi_thresholds['oversold'], line=dict(color='green', dash='dash'), row=2, col=1)

    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_white', height=700 if show_rsi else 600)
    return fig

# -------------------------
# 投资组合解析与估值
# -------------------------
def parse_portfolio_file(bytes_data: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(bytes_data))
    elif name.endswith('.xls') or name.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(bytes_data))
    else:
        raise ValueError("仅支持 CSV/XLS/XLSX 文件")
    df.columns = [c.lower().strip() for c in df.columns]
    if 'ticker' not in df.columns:
        if 'symbol' in df.columns:
            df.rename(columns={'symbol':'ticker'}, inplace=True)
        else:
            raise ValueError("文件必须包含 ticker 或 symbol 列")
    return df

def portfolio_value(port_df: pd.DataFrame, price_map: Dict[str,float]) -> (pd.DataFrame,float):
    df = port_df.copy()
    df['ticker'] = df['ticker'].str.upper().str.strip()
    df['price'] = df['ticker'].map(price_map)
    if 'quantity' in df.columns:
        df['value'] = df['quantity'] * df['price']
    else:
        df['value'] = df['price']
    total = df['value'].sum(min_count=1)
    total = float(total) if not np.isnan(total) else 0.0
    return df, total

# -------------------------
# 侧边栏参数
# -------------------------
st.sidebar.header("参数设置")
ticker_input = st.sidebar.text_input("股票代码（逗号分隔）", value="AAPL")
period = st.sidebar.selectbox("时间范围", ["1mo","3mo","6mo","1y","2y","5y"], index=2)
interval = st.sidebar.selectbox("数据间隔", ["1d","1wk","1h"], index=0)
ma_input = st.sidebar.text_input("移动均线窗口 (逗号)", "20,50")
show_rsi = st.sidebar.checkbox("显示RSI", value=True)
show_bb = st.sidebar.checkbox("显示Bollinger", value=True)
bb_window = st.sidebar.number_input("Bollinger窗口", min_value=5,max_value=200,value=20)
bb_std = st.sidebar.number_input("Bollinger std", min_value=1.0,max_value=3.5,value=2.0)
rsi_over = st.sidebar.number_input("RSI超买阈值",50,95,70)
rsi_under = st.sidebar.number_input("RSI超卖阈值",5,50,30)
color_up = st.sidebar.color_picker("上涨蜡烛", "#00A86B")
color_down = st.sidebar.color_picker("下跌蜡烛", "#D62728")

# -------------------------
# 主区
# -------------------------
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
try:
    ma_windows = [int(x) for x in ma_input.split(",") if x.strip()]
except:
    ma_windows = [20,50]

col_main, col_side = st.columns([3,1])
with col_main:
    if len(tickers)==0:
        st.info("请输入股票代码")
    elif len(tickers)==1:
        t = tickers[0]
        try:
            df = fetch_data_single(t, period, interval)
            df = calculate_technical_indicators(df, ma_windows=ma_windows, bb_window=bb_window, bb_std=bb_std)
            fig = make_candlestick_figure(df, ma_windows=ma_windows, show_rsi=show_rsi, show_bb=show_bb,
                                          colors={"up":color_up,"down":color_down},
                                          rsi_thresholds={"overbought":rsi_over,"oversold":rsi_under})
            st.plotly_chart(fig, use_container_width=True)

            if not df.empty:
                last = df.iloc[-1]
                st.metric("最新收盘价", f"{last['Close']:.4f}")
                st.metric("涨跌幅(%)", f"{((last['Close']/df['Close'].iloc[-2]-1)*100):.2f}%")

        except Exception as e:
            st.error(f"获取股票数据失败: {e}")

with col_side:
    st.subheader("技术指标说明")
    st.markdown("""
    **移动均线 (MA)**：用于观察趋势变化，短期均线穿越长期均线可能为买入/卖出信号。  
    **RSI (相对强弱指数)**：0-100，通常>70为超买，<30为超卖，用于判断价格反转可能。  
    **布林带 (Bollinger Bands)**：上轨/下轨包络价格，价格接触上轨可能回落，接触下轨可能反弹。  
    """)

# -------------------------
# 投资组合上传
# -------------------------
st.sidebar.header("投资组合分析")
uploaded_file = st.sidebar.file_uploader("上传 CSV/XLS/XLSX 投资组合", type=["csv","xls","xlsx"])
if uploaded_file:
    try:
        port_df = parse_portfolio_file(uploaded_file.read(), uploaded_file.name)
        price_map = {t: fetch_data_single(t, "1d", "1d")['Close'][-1] for t in port_df['ticker']}
        port_df_val, total_val = portfolio_value(port_df, price_map)
        st.sidebar.subheader("投资组合估值")
        st.sidebar.dataframe(port_df_val)
        st.sidebar.metric("总市值", f"{total_val:.2f}")
    except Exception as e:
        st.sidebar.error(f"解析投资组合失败: {e}")

# -------------------------
# 图表导出
# -------------------------
st.sidebar.header("图表导出")
export_png = st.sidebar.button("导出 PNG")
export_html = st.sidebar.button("导出 HTML")
if 'fig' in locals() and fig:
    if export_png:
        try:
            pio.write_image(fig, "chart.png")
            st.sidebar.success("已导出 chart.png")
        except Exception as e:
            st.sidebar.error(f"PNG 导出失败: {e}")
    if export_html:
        try:
            fig.write_html("chart.html")
            st.sidebar.success("已导出 chart.html")
        except Exception as e:
            st.sidebar.error(f"HTML 导出失败: {e}")
