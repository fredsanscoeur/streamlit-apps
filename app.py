# app.py
# -*- coding: utf-8 -*-
"""
Stock Market Visualizer 股票市场可视化工具
-----------------------------------------
功能：
1. 使用 yfinance 获取实时与历史股票数据
2. 交互式K线图（蜡烛图）+ 技术指标（均线、RSI、布林带）
3. 投资组合追踪（上传CSV/Excel）
4. 财务指标分析与股票相关性分析
5. 可定制化的可视化参数（时间范围、阈值、颜色等）
6. 支持导出图表为 PNG / HTML
7. 支持保存与分享用户配置
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
import json

# -----------------------
# 工具函数
# -----------------------

def load_stock_data(ticker, start, end):
    """获取股票历史数据"""
    return yf.download(ticker, start=start, end=end)

def calculate_technical_indicators(df):
    """计算常见技术指标：均线、RSI、布林带"""
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 布林带
    df['BB_Mid'] = df['Close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['Close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['Close'].rolling(20).std()

    return df

def plot_candlestick(df, ticker, show_ma=True, show_rsi=True, show_bb=True):
    """绘制交互式K线图 + 技术指标"""
    fig = go.Figure()

    # K线图
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick'
    ))

    # 移动均线
    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50'))

    # 布林带
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], line=dict(color='gray', dash='dot'), name='BB Upper'))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], line=dict(color='gray', dash='dot'), name='BB Lower'))

    fig.update_layout(title=f"{ticker} 股票K线图", xaxis_rangeslider_visible=False)
    return fig

def plot_rsi(df):
    """绘制 RSI 图表"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line=dict(color='blue'), name='RSI'))
    fig.update_layout(title="RSI 指标", yaxis=dict(range=[0, 100]))
    return fig

def download_chart(fig, filetype="png"):
    """导出图表为 PNG 或 HTML"""
    if filetype == "png":
        buffer = BytesIO()
        fig.write_image(buffer, format="png")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        return f'<a href="data:file/png;base64,{b64}" download="chart.png">📥 下载PNG</a>'
    elif filetype == "html":
        buffer = BytesIO()
        fig.write_html(buffer)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        return f'<a href="data:text/html;base64,{b64}" download="chart.html">📥 下载HTML</a>'

# -----------------------
# Streamlit 应用界面
# -----------------------

st.set_page_config(page_title="Stock Market Visualizer", layout="wide")

st.title("📈 Stock Market Visualizer 股票市场可视化工具")

# 用户输入股票代码与时间范围
st.sidebar.header("参数设置")
ticker = st.sidebar.text_input("输入股票代码 (如 AAPL, TSLA, BABA):", "AAPL")
start_date = st.sidebar.date_input("开始日期", pd.to_datetime("2022-01-01"))
end_date = st.sidebar.date_input("结束日期", pd.to_datetime("today"))

# 下载股票数据
df = load_stock_data(ticker, start=start_date, end=end_date)

if df.empty:
    st.error("未获取到数据，请检查股票代码或时间范围。")
    st.stop()

# 计算技术指标
df = calculate_technical_indicators(df)

# 选择展示内容
show_ma = st.sidebar.checkbox("显示移动均线", True)
show_rsi = st.sidebar.checkbox("显示RSI指标", True)
show_bb = st.sidebar.checkbox("显示布林带", True)

# 绘制图表
st.subheader(f"📊 {ticker} 股票分析图表")
fig_candle = plot_candlestick(df, ticker, show_ma, show_rsi, show_bb)
st.plotly_chart(fig_candle, use_container_width=True)

if show_rsi:
    fig_rsi = plot_rsi(df)
    st.plotly_chart(fig_rsi, use_container_width=True)

# 导出功能
st.markdown("### 导出图表")
st.markdown(download_chart(fig_candle, "png"), unsafe_allow_html=True)
st.markdown(download_chart(fig_candle, "html"), unsafe_allow_html=True)

# -----------------------
# 投资组合追踪功能
# -----------------------
st.subheader("💼 投资组合追踪")
uploaded_file = st.file_uploader("上传投资组合文件 (CSV/Excel，需包含 'Ticker' 列)", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        portfolio = pd.read_csv(uploaded_file)
    else:
        portfolio = pd.read_excel(uploaded_file)

    tickers = portfolio['Ticker'].dropna().unique().tolist()
    st.write("📌 投资组合股票列表:", tickers)

    data = {}
    for t in tickers:
        data[t] = yf.download(t, start=start_date, end=end_date)['Adj Close']
    portfolio_df = pd.DataFrame(data)

    st.line_chart(portfolio_df)

    # 股票相关性
    corr = portfolio_df.corr()
    st.subheader("📊 投资组合相关性分析")
    st.write(corr)
    fig_corr = px.imshow(corr, text_auto=True, title="相关性热力图")
    st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------
# 保存与分享配置
# -----------------------
st.sidebar.subheader("配置管理")
if st.sidebar.button("保存当前配置"):
    config = {
        "ticker": ticker,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "show_ma": show_ma,
        "show_rsi": show_rsi,
        "show_bb": show_bb
    }
    with open("user_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    st.sidebar.success("配置已保存到 user_config.json")

if st.sidebar.button("加载配置"):
    try:
        with open("user_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        st.sidebar.success("配置已加载")
        st.write("加载的配置:", config)
    except FileNotFoundError:
        st.sidebar.error("未找到配置文件，请先保存一次。")
