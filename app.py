# app.py
# -*- coding: utf-8 -*-
"""
Stock Market Visualizer - Streamlit 应用
功能：
 - 使用 yfinance 获取历史/实时数据（带缓存和重试逻辑，减少限流）
 - 单票交互式 K 线（Plotly）+ 指标（均线、RSI、Bollinger）
 - 投资组合上传（CSV/XLSX）估值与相关性分析
 - 导出图表为 PNG / HTML
 - 保存/加载用户配置（JSON）
中文注释详尽，方便直接上传到 GitHub
"""

import io
import json
import time
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

# -------------------------
# 页面配置
# -------------------------
st.set_page_config(page_title="Stock Market Visualizer", layout="wide")
st.title("📈 Stock Market Visualizer — 股票市场可视化工具")
st.write("说明：本应用使用 yfinance 抓取数据，集成 K 线、均线、RSI、布林带、组合估值与相关性分析。")

# -------------------------
# 数据抓取：带缓存 + 重试 + 批量下载
# -------------------------
@st.cache_data(ttl=900, show_spinner=False)  # 缓存 15 分钟，避免频繁请求导致限流
def fetch_data_single(ticker: str, period: str = "6mo", interval: str = "1d",
                      retries: int = 3, delay: int = 3) -> pd.DataFrame:
    """
    抓取单只股票的历史数据（封装 yf.download）
    当发生 YFRateLimitError 时会自动重试（递增等待）
    返回：DataFrame（index=Datetime, columns: Open/High/Low/Close/Adj Close/Volume）
    """
    ticker = str(ticker).strip()
    if ticker == "":
        return pd.DataFrame()

    for attempt in range(retries):
        try:
            # 使用 yf.download（相比 Ticker.history 更适合批量场景）
            df = yf.download(ticker, period=period, interval=interval, progress=False, threads=False, auto_adjust=False)
            # 确保时间索引无时区
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.tz_localize(None)
            return df
        except YFRateLimitError:
            if attempt < retries - 1:
                wait = delay * (attempt + 1)
                # 在页面提醒用户
                st.warning(f"Yahoo 限流 — 稍后重试 {wait} 秒（第 {attempt+1}/{retries} 次）...")
                time.sleep(wait)
            else:
                st.error("Yahoo Finance 已限流，且重试失败。请稍后再试或降低请求频率。")
                raise
        except Exception as e:
            # 其他错误直接抛出（会在调用处展示）
            raise

@st.cache_data(ttl=900, show_spinner=False)
def fetch_data_batch(tickers: List[str], period: str = "6mo", interval: str = "1d",
                     retries: int = 3, delay: int = 3) -> Dict[str, pd.DataFrame]:
    """
    批量抓取多个 ticker 的数据（使用 yf.download 的 group_by 功能更节省请求）
    返回：{ticker: DataFrame}
    """
    # 清理输入
    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    result = {t: pd.DataFrame() for t in tickers}
    if not tickers:
        return result

    # 尝试多次
    for attempt in range(retries):
        try:
            # 当 tickers 长度为1 时，传入字符串以获取标准列；多于1时传入列表以获得 MultiIndex（后续拆分）
            tickers_param = tickers[0] if len(tickers) == 1 else tickers
            raw = yf.download(tickers_param, period=period, interval=interval, group_by="ticker",
                              threads=True, progress=False, auto_adjust=False)
            # 处理返回结果：
            if raw.empty:
                # 没有数据
                return result

            # 若为多票下载，raw.columns 可能是 MultiIndex，按 ticker 拆分
            if isinstance(raw.columns, pd.MultiIndex):
                # 每个 ticker 一个子 DataFrame
                for t in tickers:
                    if t in raw.columns.levels[0]:
                        df_t = raw[t].copy()
                        if isinstance(df_t.index, pd.DatetimeIndex):
                            df_t.index = df_t.index.tz_localize(None)
                        result[t] = df_t
            else:
                # 单票或直接返回的表格（列名为 Open, High, ...）
                # 如果用户请求多票但 yf 返回单票（极少见），尝试将其分配给第一个 ticker
                if len(tickers) == 1:
                    df_t = raw.copy()
                    if isinstance(df_t.index, pd.DatetimeIndex):
                        df_t.index = df_t.index.tz_localize(None)
                    result[tickers[0]] = df_t
                else:
                    # 如果多票但未返回 MultiIndex，则我们尝试逐 ticker 请求（降级方案）
                    for t in tickers:
                        result[t] = fetch_data_single(t, period=period, interval=interval)
            return result
        except YFRateLimitError:
            if attempt < retries - 1:
                wait = delay * (attempt + 1)
                st.warning(f"Yahoo 限流 — 批量请求将在 {wait} 秒后重试（第 {attempt+1}/{retries} 次）...")
                time.sleep(wait)
            else:
                st.error("Yahoo Finance 对批量请求限流，重试失败。")
                raise
        except Exception as e:
            # 若批量直接失败，回退到单独逐票抓取（更慢但常能成功）
            st.warning(f"批量下载失败，尝试逐个抓取（原因：{e}）")
            for t in tickers:
                try:
                    result[t] = fetch_data_single(t, period=period, interval=interval)
                except Exception:
                    result[t] = pd.DataFrame()
            return result

# -------------------------
# 技术指标计算（健壮处理索引与数据类型）
# -------------------------
def calculate_technical_indicators(df: pd.DataFrame,
                                   ma_windows: List[int] = (20, 50),
                                   rsi_period: int = 14,
                                   bb_window: int = 20,
                                   bb_std: float = 2.0) -> pd.DataFrame:
    """
    在原 DataFrame 上添加列：MA{w}、RSI、BB_Mid/Upper/Lower
    做法：
     - 使用相同的索引（不会引入错位）
     - 对 Close 做类型强制转换与 NaN 保护
     - 返回同样索引的 DataFrame（不改变原始行数）
    """
    if df is None or df.empty:
        return df

    # 尝试获取 'Close' 或 'Adj Close'
    if 'Close' in df.columns:
        close = df['Close'].astype(float).copy()
    elif 'Adj Close' in df.columns:
        close = df['Adj Close'].astype(float).copy()
    else:
        raise ValueError("DataFrame 中不包含 'Close' 或 'Adj Close' 列，无法计算指标。")

    # 保证索引按时间升序
    close = close.sort_index()

    # 创建一个指标表以确保索引对齐，再合并入原 df
    ind = pd.DataFrame(index=close.index)
    # 简单移动平均（可指定多个窗口）
    for w in ma_windows:
        # min_periods=1 使得前期也有数值（你也可以改为 min_periods=w）
        ind[f"MA{w}"] = close.rolling(window=w, min_periods=1).mean()

    # Bollinger Bands
    mid = close.rolling(window=bb_window, min_periods=1).mean()
    std = close.rolling(window=bb_window, min_periods=1).std(ddof=0)  # ddof=0 更稳健
    ind['BB_Mid'] = mid
    ind['BB_Upper'] = mid + bb_std * std
    ind['BB_Lower'] = mid - bb_std * std

    # RSI（使用简单 Rolling 平均）
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=1).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    ind['RSI'] = 100 - (100 / (1 + rs))
    ind['RSI'] = ind['RSI'].fillna(0)

    # 把指标合并回原来的 DataFrame（按索引对齐，避免赋值时索引不一致）
    # 注意：不覆盖原列，直接新增列
    for col in ind.columns:
        df[col] = ind[col]

    return df

# -------------------------
# 绘图函数：K线 + 指标（Plotly）
# -------------------------
def make_candlestick_plot(df: pd.DataFrame,
                          title: str = "Price",
                          ma_windows: List[int] = None,
                          show_rsi: bool = True,
                          show_bb: bool = True,
                          colors: Dict[str, str] = None,
                          rsi_thresholds: Dict[str, int] = None):
    """
    返回 Plotly Figure（若 show_rsi=True，则返回带子图的 figure）
    """
    if df is None or df.empty:
        raise ValueError("传入数据为空，无法绘图。")

    ma_windows = ma_windows or []
    colors = colors or {"up": "#00A86B", "down": "#D62728", "ma": "#1f77b4", "bb": "#FFA500"}
    rsi_thresholds = rsi_thresholds or {"overbought": 70, "oversold": 30}

    # 基本 K 线
    candle = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color=colors.get("up", "#00A86B"),
        decreasing_line_color=colors.get("down", "#D62728"),
    )

    if show_rsi:
        # 两行子图：价格（含 MA、BB）在上，RSI 在下
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            vertical_spacing=0.06, row_heights=[0.75, 0.25])
        fig.add_trace(candle, row=1, col=1)
    else:
        fig = go.Figure()
        fig.add_trace(candle)

    # 添加移动均线
    for w in ma_windows:
        col = f"MA{w}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=col, line=dict(width=1.5)), row=1 if show_rsi else 1, col=1)

    # 添加布林带
    if show_bb and 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(width=1), opacity=0.6), row=1 if show_rsi else 1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', line=dict(width=1), opacity=0.6, fill='tonexty'), row=1 if show_rsi else 1, col=1)

    # 添加 RSI 子图
    if show_rsi and 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(width=1)), row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
        # 添加阈值线
        fig.add_hline(y=rsi_thresholds.get('overbought', 70), line=dict(color='red', dash='dash'), row=2, col=1)
        fig.add_hline(y=rsi_thresholds.get('oversold', 30), line=dict(color='green', dash='dash'), row=2, col=1)

    fig.update_layout(title=title, xaxis_rangeslider_visible=False, template="plotly_white", height=750 if show_rsi else 600)
    return fig

# -------------------------
# 投资组合解析与估值函数
# -------------------------
def parse_portfolio_file(bytes_data: bytes, filename: str) -> pd.DataFrame:
    """
    解析上传的 CSV/XLSX 文件，返回 DataFrame（要求至少有 'ticker' 列，忽略大小写）
    """
    name = filename.lower()
    if name.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(bytes_data))
    elif name.endswith('.xls') or name.endswith('.xlsx'):
        df = pd.read_excel(io.BytesIO(bytes_data))
    else:
        raise ValueError("仅支持 CSV 或 Excel 文件。")

    # 规范化列名
    df.columns = [c.strip().lower() for c in df.columns]
    if 'ticker' not in df.columns:
        # 兼容 'Ticker' 大小写情况
        if 'symbol' in df.columns:
            df.rename(columns={'symbol': 'ticker'}, inplace=True)
        else:
            raise ValueError("上传文件必须包含 'ticker' 列（标签不区分大小写）。")
    return df

def portfolio_value(port_df: pd.DataFrame, price_map: Dict[str, float]) -> (pd.DataFrame, float):
    """
    使用 price_map 计算 portfolio 每项价值并返回总价值
    price_map: {TICKER: last_close_price}
    支持 'quantity' 或 'weight' 列
    """
    if port_df is None or port_df.empty:
        return pd.DataFrame(), 0.0

    df = port_df.copy()
    df['ticker'] = df['ticker'].astype(str).str.upper().str.strip()
    prices = df['ticker'].map(price_map)
    df['price'] = prices

    if 'quantity' in df.columns:
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
        df['value'] = df['quantity'] * df['price']
    elif 'weight' in df.columns:
        df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0)
        total_weight = df['weight'].sum()
        if total_weight == 0:
            df['value'] = 0
        else:
            df['value'] = (df['weight'] / total_weight) * df['price']  # 归一后 * price（示例估值）
    else:
        # 若无数量/权重，则直接用 price 填充
        df['value'] = df['price']

    total = df['value'].sum(min_count=1)
    return df, float(total if not np.isnan(total) else 0.0)

# -------------------------
# UI：侧边栏设置与主区域布局（中文）
# -------------------------
st.sidebar.header("数据源与参数")
# 支持多个 ticker 的输入（逗号分隔）
ticker_input = st.sidebar.text_input("输入股票代码（逗号分隔，例如：AAPL, MSFT）", value="AAPL")
period = st.sidebar.selectbox("时间范围 (period)", options=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"], index=2)
interval = st.sidebar.selectbox("数据间隔 (interval)", options=["1d", "1wk", "1h", "30m"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("指标显示设置")
ma_windows_str = st.sidebar.text_input("移动均线窗口（逗号分隔，例如 20,50）", value="20,50")
show_rsi = st.sidebar.checkbox("显示 RSI", value=True)
show_bb = st.sidebar.checkbox("显示 Bollinger 带", value=True)
bb_window = st.sidebar.number_input("Bollinger 窗口", min_value=5, max_value=200, value=20)
bb_std = st.sidebar.number_input("Bollinger n_std", min_value=1.0, max_value=3.5, step=0.1, value=2.0)
rsi_over = st.sidebar.number_input("RSI 过热阈值", min_value=50, max_value=95, value=70)
rsi_under = st.sidebar.number_input("RSI 超卖阈值", min_value=5, max_value=50, value=30)

st.sidebar.markdown("---")
st.sidebar.subheader("导出与配置")
if 'saved_configs' not in st.session_state:
    st.session_state['saved_configs'] = {}

cfg_name = st.sidebar.text_input("当前配置名称（保存后可在会话中加载）", value="my-preset")
if st.sidebar.button("保存当前配置到会话"):
    st.session_state['saved_configs'][cfg_name] = {
        "ticker_input": ticker_input,
        "period": period,
        "interval": interval,
        "ma_windows_str": ma_windows_str,
        "show_rsi": show_rsi,
        "show_bb": show_bb,
        "bb_window": int(bb_window),
        "bb_std": float(bb_std),
        "rsi_over": int(rsi_over),
        "rsi_under": int(rsi_under)
    }
    st.sidebar.success(f"已保存配置：{cfg_name}")

if st.session_state['saved_configs']:
    chosen = st.sidebar.selectbox("加载会话中的配置", options=list(st.session_state['saved_configs'].keys()))
    if st.sidebar.button("加载选中配置"):
        loaded = st.session_state['saved_configs'][chosen]
        # 注意：Streamlit 无法动态回写所有控件，这里展示已加载的配置，用户可手动复制到输入框
        st.sidebar.info("已加载配置（请手动将值复制到上方控件以生效）")
        st.sidebar.json(loaded)

st.sidebar.download_button("下载所有会话配置 (JSON)", data=json.dumps(st.session_state['saved_configs'], indent=2, ensure_ascii=False),
                          file_name="presets.json", mime="application/json")

# -------------------------
# 主流程：加载与显示数据
# -------------------------
# 解析 tickers
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
if not tickers:
    st.error("请在侧边栏输入至少一个股票代码（Ticker）。")
    st.stop()

# 转换 ma_windows
try:
    ma_windows = [int(x.strip()) for x in ma_windows_str.split(",") if x.strip()]
except Exception:
    ma_windows = [20, 50]

# 主区左：图表（单票时绘制 K 线；多票显示对比图与相关性）
col_main, col_side = st.columns([3, 1])

with col_main:
    st.subheader("图表与指标")
    # 若只输入 1 只股票，显示 K 线 + 指标
    if len(tickers) == 1:
        t = tickers[0]
        try:
            df = fetch_data_single(t, period=period, interval=interval)
        except Exception as e:
            st.error(f"数据抓取失败（{t}）：{e}")
            st.stop()

        if df is None or df.empty:
            st.warning("未能获取到该股票的数据，请检查代码或更换时间范围。")
        else:
            # 计算技术指标（健壮）
            try:
                df = calculate_technical_indicators(df, ma_windows=ma_windows, rsi_period=14, bb_window=bb_window, bb_std=bb_std)
            except Exception as e:
                st.warning(f"指标计算失败：{e}")
            # 绘制图表
            try:
                fig = make_candlestick_plot(df,
                                            title=f"{t} 价格（{period}，{interval}）",
                                            ma_windows=ma_windows,
                                            show_rsi=show_rsi,
                                            show_bb=show_bb,
                                            rsi_thresholds={"overbought": rsi_over, "oversold": rsi_under})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"绘图失败：{e}")

            # 显示最新 OHLC 信息
            if not df.empty:
                last = df.iloc[-1]
                st.metric("最新收盘价", f"{last['Close']:.4f}")
                st.write("最近几行数据预览：")
                st.dataframe(df.tail().style.format("{:.4f}"))

            # 导出按钮（PNG / HTML）
            st.markdown("#### 导出图表")
            try:
                png_bytes = pio.to_image(fig, format='png', width=1200, height=700, scale=2)
                st.download_button("下载 PNG", data=png_bytes, file_name=f"{t}_chart.png", mime="image/png")
            except Exception as e:
                st.info(f"PNG 导出需要 kaleido：{e}")

            try:
                html = pio.to_html(fig, full_html=True)
                st.download_button("下载 HTML", data=html, file_name=f"{t}_chart.html", mime="text/html")
            except Exception as e:
                st.warning(f"HTML 导出失败：{e}")

    else:
        # 多票：显示收盘价对比图与相关性
        st.subheader("多票对比与相关性")
        try:
            data_map = fetch_data_batch(tickers, period=period, interval=interval)
        except Exception as e:
            st.error(f"批量数据抓取失败：{e}")
            data_map = {t: pd.DataFrame() for t in tickers}

        # 构建收盘价表格（对齐索引）
        close_dict = {}
        for t in tickers:
            df_t = data_map.get(t, pd.DataFrame())
            if df_t is None or df_t.empty:
                continue
            # 使用 Adjusted Close if exist, otherwise Close
            if 'Adj Close' in df_t.columns:
                close_dict[t] = df_t['Adj Close'].rename(t)
            elif 'Close' in df_t.columns:
                close_dict[t] = df_t['Close'].rename(t)

        if not close_dict:
            st.warning("未获取到任何股票的收盘价数据。")
        else:
            close_df = pd.concat(close_dict.values(), axis=1)
            close_df.columns = list(close_dict.keys())
            st.line_chart(close_df.fillna(method='ffill'))

            # 计算并显示相关性（returns）
            returns = close_df.pct_change().dropna(how='all').dropna(axis=1, how='all')
            if returns.shape[1] >= 2:
                corr = returns.corr()
                st.subheader("收益率相关性矩阵")
                st.dataframe(corr.style.format("{:.3f}"))
                # 热力图
                try:
                    import plotly.express as px
                    fig_corr = px.imshow(corr.values, x=corr.columns, y=corr.index, color_continuous_scale='RdBu', zmin=-1, zmax=1, title="相关性热力图")
                    st.plotly_chart(fig_corr, use_container_width=True)
                except Exception:
                    st.write("无法绘制相关性热力图（缺少 plotly.express 或其它错误）")
            else:
                st.info("至少需要两只有效股票才能计算相关性。")

with col_side:
    st.subheader("投资组合估值（侧栏）")
    uploaded = st.file_uploader("上传投资组合 CSV/XLSX（列名需包含 ticker，或 ticker 与 quantity/weight）", type=['csv', 'xlsx', 'xls'])
    if uploaded:
        try:
            raw = uploaded.read()
            port_df = parse_portfolio_file(raw, uploaded.name)
            st.write("投资组合预览：")
            st.dataframe(port_df.head())
            # 获取组合内 tickers 的最新价（使用短期历史最后一行）
            port_tickers = port_df['ticker'].astype(str).str.upper().unique().tolist()
            price_map = {}
            # 使用批量抓取以减少请求次数
            try:
                batch = fetch_data_batch(port_tickers, period="5d", interval="1d")
                for t in port_tickers:
                    df_t = batch.get(t, pd.DataFrame())
                    if df_t is None or df_t.empty:
                        price_map[t] = np.nan
                    else:
                        # 优先使用 Adj Close
                        price_map[t] = float(df_t['Adj Close'].dropna().iloc[-1]) if 'Adj Close' in df_t.columns else float(df_t['Close'].dropna().iloc[-1])
            except Exception:
                # 降级逐票获取
                for t in port_tickers:
                    try:
                        df_t = fetch_data_single(t, period="5d", interval="1d")
                        price_map[t] = float(df_t['Adj Close'].dropna().iloc[-1]) if 'Adj Close' in df_t.columns else float(df_t['Close'].dropna().iloc[-1])
                    except Exception:
                        price_map[t] = np.nan

            pv_df, total_value = portfolio_value(port_df, price_map)
            st.subheader("组合估值结果")
            st.dataframe(pv_df)
            st.metric("组合总价值（近似）", f"{total_value:.2f}")

            # 下载估值
            csv_buf = io.StringIO()
            pv_df.to_csv(csv_buf, index=False)
            st.download_button("下载估值 CSV", data=csv_buf.getvalue(), file_name="portfolio_valuation.csv", mime="text/csv")
        except Exception as e:
            st.error(f"上传或计算失败：{e}")

# -------------------------
# 配置保存/加载（文件级）
# -------------------------
st.markdown("---")
st.header("配置保存与加载")
col_s1, col_s2 = st.columns(2)
with col_s1:
    if st.button("保存当前配置到本地 (下载 JSON)"):
        cfg = {
            "ticker_input": ticker_input,
            "period": period,
            "interval": interval,
            "ma_windows_str": ma_windows_str,
            "show_rsi": show_rsi,
            "show_bb": show_bb,
            "bb_window": bb_window,
            "bb_std": bb_std,
            "rsi_over": rsi_over,
            "rsi_under": rsi_under
        }
        st.download_button("点击下载配置文件", data=json.dumps(cfg, indent=2, ensure_ascii=False), file_name="smv_config.json", mime="application/json")
with col_s2:
    cfg_upload = st.file_uploader("上传配置 JSON（可覆盖当前控件，需手动复制值）", type=["json"])
    if cfg_upload:
        try:
            loaded_cfg = json.load(cfg_upload)
            st.success("配置文件已解析（请手动将值复制到侧边栏控件以生效）")
            st.json(loaded_cfg)
        except Exception as e:
            st.error(f"配置加载失败：{e}")

st.caption("提示：若在 Streamlit Cloud 上出现 yfinance 限流，请尝试降低请求频率或使用更小时间段；也可替换为付费数据源以获得稳定的实时数据。")

# 结束
