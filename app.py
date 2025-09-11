# app.py
"""
Stock Market Visualizer（股票市场可视化器）
依赖: streamlit, akshare, pandas, numpy, plotly, scikit-learn, statsmodels (optional), kaleido (for png export), openpyxl
"""

import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import io
from sklearn.linear_model import LinearRegression
import os

# ========== Helpers: indicators ==========
def moving_average(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(0)

def bollinger_bands(series, window=20, n_std=2):
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower

# ========== Data fetch wrappers with graceful fallback ==========
def fetch_daily_akshare(symbol: str, start: str=None, end: str=None) -> pd.DataFrame:
    """
    尝试使用 akshare 获取 A 股日线数据。
    symbol: 支持 'sh600000' 或 'sz000001' 或 '600000' (常见 user 习惯)
    start/end: 'YYYY-MM-DD' 或 None
    返回 DataFrame: index=日期, columns: open, high, low, close, volume, turnover
    """
    # normalize symbol
    s = symbol.strip()
    # akshare expects 'sh600000' or 'sz000001' for some functions; try to be flexible
    if len(s) == 6 and s.isdigit():
        # try to infer exchange: common heuristic: '6'->sh, '0'/'3'->sz
        if s.startswith('6') or s.startswith('9'):
            s = 'sh' + s
        else:
            s = 'sz' + s

    # try several akshare functions with try/except
    try:
        # ak.stock_zh_a_daily returns 'date' column and OHLCV
        df = ak.stock_zh_a_daily(symbol=s, start_date=(start.replace('-','') if start else None), end_date=(end.replace('-','') if end else None))
        # ak returns ascending by default? convert and normalize
        if '日期' in df.columns:
            df = df.rename(columns={'日期':'date'})
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        # Normalize columns
        col_map = {}
        for c in df.columns:
            lc = c.lower()
            if '开盘' in c or 'open' in lc:
                col_map[c] = 'open'
            if '最高' in c or 'high' in lc:
                col_map[c] = 'high'
            if '最低' in c or 'low' in lc:
                col_map[c] = 'low'
            if '收盘' in c or 'close' in lc:
                col_map[c] = 'close'
            if '成交量' in c or 'volume' in lc:
                col_map[c] = 'volume'
            if '成交额' in c or 'turnover' in lc:
                col_map[c] = 'turnover'
            if '振幅' in c:
                col_map[c] = c  # keep
        df = df.rename(columns=col_map)
        # ensure numeric types
        for col in ['open','high','low','close','volume','turnover']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"从 akshare 获取数据失败: {e}")
        return pd.DataFrame()

def fetch_valuation(symbol: str):
    """
    尝试获取市盈率、市净率等估值信息（如果 akshare 有相应接口）。
    返回 dict 或空 dict。
    """
    try:
        # akshare 的接口可能变化，尝试常用接口
        # 首选：ak.stock_individual_info_em or ak.stock_zh_a_spot (spot includes pe/pb)
        # Try spot first (more likely to have pe/pb)
        s = symbol.strip()
        if len(s) == 6 and s.isdigit():
            if s.startswith('6') or s.startswith('9'):
                s = 'sh' + s
            else:
                s = 'sz' + s
        # stock_zh_a_spot returns a DataFrame with 'code' and 'pe' maybe
        df_spot = ak.stock_zh_a_spot()
        if not df_spot.empty:
            row = df_spot[df_spot['代码'] == s.replace('sh','').replace('sz','')]
            if row.shape[0] == 0:
                # try code with exchange prefix removed
                row = df_spot[df_spot['代码'] == s[-6:]]
            if row.shape[0] > 0:
                row0 = row.iloc[0].to_dict()
                # try to extract pe,pb
                pe = row0.get('市盈率(TTM)') or row0.get('pe') or row0.get('市盈率')
                pb = row0.get('市净率') or row0.get('pb')
                return {'pe': pe, 'pb': pb, 'raw': row0}
    except Exception:
        pass
    # fallback empty
    return {}

def try_get_chip_distribution(symbol: str, start_date=None, end_date=None):
    """
    筹码分布（'筹码分布'）数据来源在 akshare 上面并非总有稳定接口。
    这里我们用一种近似方法：用不同价格区间的持仓/成交量分布来估算筹码分布（基于历史成交量/价格）。
    返回 DataFrame: price_bin, volume
    """
    try:
        df = fetch_daily_akshare(symbol, start_date, end_date)
        if df.empty:
            return pd.DataFrame()
        # Approximate: use close prices and volumes to aggregate across bins
        data = df[['close','volume']].dropna()
        # Consider volumes applied to price buckets
        n_bins = 30
        bins = np.linspace(data['close'].min(), data['close'].max(), n_bins+1)
        data['price_bin'] = pd.cut(data['close'], bins=bins, include_lowest=True)
        grouped = data.groupby('price_bin')['volume'].sum().reset_index()
        # center of bin
        grouped['price'] = grouped['price_bin'].apply(lambda x: x.mid)
        grouped = grouped[['price','volume']].sort_values('price', ascending=False).reset_index(drop=True)
        return grouped
    except Exception:
        return pd.DataFrame()

# ========== Plotting ==========
def plot_candlestick_with_indicators(df: pd.DataFrame, title='Candlestick', ma_windows=[5,10,20], show_rsi=False, show_boll=False, colors=None):
    """返回 plotly Figure"""
    df = df.copy().dropna(subset=['open','high','low','close'])
    fig = make_subplots(rows=(2 if show_rsi else 1), cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25] if show_rsi else [1],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
                                 name='Price'), row=1, col=1)

    # MAs
    for w in ma_windows:
        ma = moving_average(df['close'], w)
        fig.add_trace(go.Scatter(x=df.index, y=ma, name=f'MA{w}', mode='lines', line=dict(width=1.2)), row=1, col=1)

    if show_boll:
        ma, upper, lower = bollinger_bands(df['close'], window=20, n_std=2)
        fig.add_trace(go.Scatter(x=df.index, y=upper, name='Bollinger Upper', mode='lines', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, name='Bollinger Lower', mode='lines', line=dict(dash='dash')), row=1, col=1)

    # volume as bar in same subplot (secondary y)
    if 'volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume', marker=dict(opacity=0.3)), row=1, col=1)

    if show_rsi:
        r = rsi(df['close'])
        fig.add_trace(go.Scatter(x=df.index, y=r, name='RSI', mode='lines'), row=2, col=1)
        fig.update_yaxes(title_text='RSI', row=2, col=1, range=[0,100])

    fig.update_layout(title=title, xaxis_rangeslider_visible=False, template='plotly_white', height=700)
    return fig

def plot_chip_distribution(df_chip: pd.DataFrame, title='筹码分布'):
    if df_chip.empty:
        st.info("没有可用的筹码分布数据（已尝试基于历史成交估算）。")
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df_chip['price'], y=df_chip['volume'], orientation='v', name='筹码量'))
    fig.update_layout(title=title, xaxis_title='价格', yaxis_title='量', template='plotly_white')
    return fig

def plot_correlation_heatmap(returns_df: pd.DataFrame, title='相关性矩阵'):
    corr = returns_df.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', zmin=-1, zmax=1))
    fig.update_layout(title=title, height=600)
    return fig

# ========== Forecast baseline ==========
def simple_linear_forecast(series: pd.Series, forecast_days=7):
    """
    用线性回归在时间索引上对价格做简单预测（返回 forecast_df with columns date, predicted）
    这是一个简单基线预测，适合快速演示；生产可替换为更复杂模型（Prophet, ARIMA, LSTM 等）。
    """
    s = series.dropna()
    if len(s) < 10:
        return pd.DataFrame()
    X = np.arange(len(s)).reshape(-1,1)
    y = s.values.reshape(-1,1)
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.arange(len(s), len(s)+forecast_days).reshape(-1,1)
    preds = model.predict(future_X).flatten()
    last_date = s.index[-1]
    dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
    return pd.DataFrame({'date': dates, 'predicted': preds})

# ========== Streamlit UI ==========
st.set_page_config(page_title="Stock Market Visualizer", layout='wide')
st.title("📈 Stock Market Visualizer（股票市场可视化器）")

# Sidebar: inputs
st.sidebar.header("🔎 股票查询")
ticker = st.sidebar.text_input("输入股票代码（如 600519 或 sh600519 或 sz000001）", value="600519")
start_date = st.sidebar.date_input("开始日期", value=(datetime.now() - timedelta(days=365)))
end_date = st.sidebar.date_input("结束日期", value=datetime.now())
ma_string = st.sidebar.text_input("移动平均窗口（逗号分隔）", value="5,10,20")
ma_windows = [int(x.strip()) for x in ma_string.split(',') if x.strip().isdigit()][:5]
show_rsi = st.sidebar.checkbox("显示 RSI", value=True)
show_boll = st.sidebar.checkbox("显示 布林带", value=True)
show_chip = st.sidebar.checkbox("显示 筹码分布（估算）", value=True)
fetch_button = st.sidebar.button("获取并绘制")

# Portfolio upload
st.sidebar.header("📁 投资组合")
uploaded_file = st.sidebar.file_uploader("上传组合 CSV 或 Excel（columns: code, shares 或 qty）", type=['csv','xls','xlsx'])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            portfolio_df = pd.read_csv(uploaded_file)
        else:
            portfolio_df = pd.read_excel(uploaded_file)
        st.sidebar.success(f"已加载组合：{portfolio_df.shape[0]} 项")
    except Exception as e:
        st.sidebar.error(f"读取组合文件失败: {e}")
        portfolio_df = None
else:
    portfolio_df = None

# Config save / load
st.sidebar.header("⚙️ 配置")
if 'saved_configs' not in st.session_state:
    st.session_state['saved_configs'] = {}
cfg_name = st.sidebar.text_input("配置名称（保存当前设置）", value="my_config")
if st.sidebar.button("保存配置"):
    cfg = {'ticker': ticker, 'start': str(start_date), 'end': str(end_date), 'ma': ma_windows, 'rsi': show_rsi, 'boll': show_boll}
    st.session_state['saved_configs'][cfg_name] = cfg
    st.sidebar.success("已保存配置到 session。")

if st.session_state.get('saved_configs'):
    sel = st.sidebar.selectbox("加载已保存配置", options=list(st.session_state['saved_configs'].keys()))
    if st.sidebar.button("加载配置"):
        c = st.session_state['saved_configs'][sel]
        # apply (note: we cannot directly change widget values; notify user)
        st.sidebar.write("已保存的配置（请手动在输入框中确认/复制）：")
        st.sidebar.json(c)

# Main panel
col1, col2 = st.columns([3,1])

with col1:
    st.subheader("K 线图与叠加指标")
    if fetch_button:
        with st.spinner("获取数据中..."):
            df = fetch_daily_akshare(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        if df.empty:
            st.warning("未能获取到指定股票的日线数据。请检查代码或日期。")
        else:
            fig = plot_candlestick_with_indicators(df, title=f"{ticker} K 线", ma_windows=ma_windows, show_rsi=show_rsi, show_boll=show_boll)
            st.plotly_chart(fig, use_container_width=True)
            # Export options
            st.markdown("**导出图表**")
            col_e1, col_e2 = st.columns(2)
            with col_e1:
                if st.button("导出为 HTML"):
                    html_bytes = fig.to_html().encode('utf-8')
                    st.download_button(label="下载 HTML", data=html_bytes, file_name=f"{ticker}_chart.html", mime="text/html")
            with col_e2:
                if st.button("导出为 PNG"):
                    # requires kaleido
                    try:
                        img_bytes = fig.to_image(format="png", width=1400, height=800, scale=2)
                        st.download_button("下载 PNG", data=img_bytes, file_name=f"{ticker}_chart.png", mime="image/png")
                    except Exception as e:
                        st.error(f"导出 PNG 失败（缺少 kaleido?): {e}")

            # Valuation
            val = fetch_valuation(ticker)
            if val:
                st.markdown("**估值信息（尝试从 akshare 获取）**")
                st.json(val)
            else:
                st.info("无法获取到估值（PE/PB），akshare 接口可能有变化或未覆盖该股票。")

            # chip distribution
            if show_chip:
                chip_df = try_get_chip_distribution(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if not chip_df.empty:
                    chip_fig = plot_chip_distribution(chip_df, title=f"{ticker} - 筹码分布（近似）")
                    st.plotly_chart(chip_fig, use_container_width=True)
                else:
                    st.info("筹码分布数据不可用或估算失败。")

            # Forecast
            st.markdown("### 历史K线比对与简单预测")
            st.write("使用简单线性回归作为基线短期预测（演示用途）。")
            forecast_days = st.number_input("预测天数", min_value=1, max_value=90, value=7)
            if st.button("生成预测"):
                if 'close' in df.columns:
                    fc_df = simple_linear_forecast(df['close'], forecast_days=int(forecast_days))
                    if fc_df.empty:
                        st.warning("样本太少，无法生成预测。")
                    else:
                        # show prediction on chart
                        pred_fig = fig
                        pred_fig.add_trace(go.Scatter(x=fc_df['date'], y=fc_df['predicted'], name='预测', mode='lines+markers'), row=1, col=1)
                        st.plotly_chart(pred_fig, use_container_width=True)
                        st.dataframe(fc_df)
                else:
                    st.warning("数据中没有 close 列，无法预测。")

with col2:
    st.subheader("组合 & 快速分析")
    if portfolio_df is not None:
        st.write("已上传投资组合：")
        st.dataframe(portfolio_df)
        # try to fetch prices and compute current value
        pv = []
        for idx, row in portfolio_df.iterrows():
            code = str(row.get('code') or row.get('ticker') or row.get('symbol'))
            qty = float(row.get('shares') or row.get('qty') or row.get('quantity') or 0)
            if pd.isna(code):
                continue
            df_sym = fetch_daily_akshare(code, (datetime.now()-timedelta(days=10)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
            price = None
            if not df_sym.empty:
                price = df_sym['close'].iloc[-1]
            pv.append({'code': code, 'qty': qty, 'price': price, 'value': (price*qty if price else None)})
        pv_df = pd.DataFrame(pv)
        st.write("组合估值（基于最近收盘价）")
        st.dataframe(pv_df)
        if pv_df['value'].notnull().any():
            total = pv_df['value'].sum(skipna=True)
            st.metric("组合估算总价值（RMB）", f"{total:,.2f}")

    st.markdown("---")
    st.subheader("相关性分析")
    st.write("输入多个股票代码（逗号分隔），计算收益相关性。")
    tickers_multi = st.text_input("代码列表（逗号分隔）", value="600519,000001,600000")
    if st.button("计算相关性"):
        codes = [c.strip() for c in tickers_multi.split(',') if c.strip()]
        price_dict = {}
        for c in codes:
            dfc = fetch_daily_akshare(c, (datetime.now()-timedelta(days=365)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
            if not dfc.empty and 'close' in dfc.columns:
                price_dict[c] = dfc['close'].rename(c)
        if len(price_dict) < 2:
            st.warning("至少需要两个有效股票数据以计算相关性。")
        else:
            price_df = pd.concat(price_dict.values(), axis=1).dropna()
            returns = price_df.pct_change().dropna()
            corr_fig = plot_correlation_heatmap(returns, title='收益相关性热力图')
            st.plotly_chart(corr_fig, use_container_width=True)

    st.markdown("---")
    st.subheader("财务比率 & 基本面（尝试获取）")
    symbol_fin = st.text_input("查询基本面股票代码（例：600519）", value=ticker)
    if st.button("获取财务指标"):
        try:
            # try akshare financial interfaces - may vary by ak version
            try:
                fin_df = ak.stock_financial_analysis_indicator(symbol_fin)
                st.dataframe(fin_df.head(20))
            except Exception:
                st.info("未能通过 ak.stock_financial_analysis_indicator 获取，尝试其它接口...")
                try:
                    fin_df2 = ak.stock_fina_indicator(symbol_fin)
                    st.dataframe(fin_df2.head(20))
                except Exception as e:
                    st.error(f"未能获取财务数据：{e}")
        except Exception as e:
            st.error(f"获取财务数据失败：{e}")

# Footer: repo / save config export
st.sidebar.markdown("---")
st.sidebar.subheader("📦 项目与 GitHub")
st.sidebar.markdown("""
本地运行后你可以将代码推送到 GitHub。README 包含完整命令。
""")
if st.sidebar.button("下载 当前配置 (JSON)"):
    cfg = {'ticker': ticker, 'start': str(start_date), 'end': str(end_date), 'ma': ma_windows, 'rsi': show_rsi, 'boll': show_boll}
    st.download_button("下载 JSON 配置", data=json.dumps(cfg, ensure_ascii=False, indent=2).encode('utf-8'),
                       file_name=f"{ticker}_config.json", mime="application/json")

st.markdown("---")
st.caption("提示：akshare 的个别接口或字段名随版本会变化。如遇接口不可用，建议升级 akshare 到最新版或参照 akshare 文档替换函数名。")

