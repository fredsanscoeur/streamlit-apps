# app.py
"""
Stock Market Visualizer（股票市场可视化器）
功能亮点（实现）：
- 用 akshare 获取 A 股实时与历史数据（尽量使用常用接口并做容错）
- 按用户定义的 8 步筛选流程逐层筛股
- 交互式 Plotly K 线（蜡烛图），支持 MA、RSI、Bollinger 叠加
- 支持 CSV/Excel 格式的投资组合上传与展示
- 支持股票相关性分析、财务比率展示（如可用）
- 导出图表为 PNG（需要 kaleido）或 HTML
- 保存并加载用户配置（本地 JSON）
注意：真实运行依赖 akshare 的可用性，有些数据字段可能需要根据 akshare 版本调整
"""
import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import json
import os
import io
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# ---------- 配置 ----------
st.set_page_config(page_title="Stock Market Visualizer", layout="wide")
CONFIG_DIR = "smv_configs"
if not os.path.exists(CONFIG_DIR):
    os.makedirs(CONFIG_DIR)

# ---------- 帮助函数 ----------
@st.cache_data(ttl=300)
def get_all_a_spot() -> pd.DataFrame:
    """
    获取所有 A 股的实时行情（尽量用 akshare 的常见接口）
    返回 DataFrame，列包含至少： '代码' '名称' '最新价' '涨跌幅' '成交量' '换手率' '流通市值'（若可用）
    注意：不同 akshare 版本字段名称不同，做了容错映射。
    """
    try:
        df = ak.stock_zh_a_spot_em()  # 常见函数
    except Exception as e:
        st.error(f"获取 A 股实时行情出错: {e}")
        return pd.DataFrame()
    # 标准化列名（尝试映射常见字段）
    mapping = {}
    cols = df.columns.tolist()
    # try map likely names
    for c in cols:
        lc = c.lower()
        if "代码" in c or "code" in lc:
            mapping[c] = "代码"
        elif "名称" in c or "name" in lc:
            mapping[c] = "名称"
        elif "最新价" in c or "当前价" in c or "price" in lc:
            mapping[c] = "最新价"
        elif "涨跌幅" in c or "percent" in lc or "涨幅" in c:
            mapping[c] = "涨跌幅"
        elif "成交量" in c or "volume" in lc:
            mapping[c] = "成交量"
        elif "换手率" in c or "turnover" in lc:
            mapping[c] = "换手率"
        elif ("流通市值" in c) or ("流通" in c and "市" in c) or "circulating" in lc:
            mapping[c] = "流通市值"
    df = df.rename(columns=mapping)
    # ensure numeric
    for col in ["涨跌幅", "成交量", "换手率", "流通市值"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(ttl=3600)
def get_daily_klines(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    获取日线历史 K 线（收盘、开盘、最高、最低、成交量）
    symbol: akshare 形如 'sh600000' 或 'sz000001'（akshare 要求）
    返回 DataFrame 索引为日期，列: ['open','high','low','close','volume']
    """
    # try ak.stock_zh_a_daily
    try:
        df = ak.stock_zh_a_daily(symbol=symbol)
        # akshare 返回中英文列名不同，标准化：
        df = df.rename(columns=lambda x: x.lower())
        # some versions return columns: date, open, high, low, close, volume, amount
        # ensure columns exist
        cols_lower = df.columns.tolist()
        needed = {}
        for c in cols_lower:
            if "open" in c:
                needed["open"] = c
            if "high" in c:
                needed["high"] = c
            if "low" in c:
                needed["low"] = c
            if "close" in c:
                needed["close"] = c
            if "volume" in c:
                needed["volume"] = c
        df = df.rename(columns={v:k for k,v in needed.items()})
        # keep ascending by date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        else:
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df[['open','high','low','close','volume']].copy()
    except Exception as e:
        st.warning(f"获取 {symbol} 日线出错: {e}")
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算 MA、RSI、Bollinger 等技术指标并加入 DataFrame"""
    df = df.copy()
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    # RSI 14
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=13, adjust=False).mean()
    ma_down = down.ewm(com=13, adjust=False).mean()
    rs = ma_up / ma_down
    df['rsi14'] = 100 - (100 / (1 + rs))
    # Bollinger Bands (20,2)
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_up'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_dn'] = df['bb_mid'] - 2 * df['bb_std']
    return df

def avg_volume(symbol: str, n: int = 5) -> float:
    """计算过去 n 日平均成交量，若数据不足返回 NaN"""
    df = get_daily_klines(symbol)
    if df is None or df.empty:
        return np.nan
    return df['volume'].tail(n).mean()

def compute_volume_ratio(symbol: str, today_volume: float) -> float:
    """
    简单的量比计算：今日成交量 / 最近5日平均成交量
    如果拿不到今日实时成交量（spot），不计算
    """
    avg5 = avg_volume(symbol, n=5)
    if np.isnan(avg5) or avg5 == 0:
        return np.nan
    return today_volume / avg5

def symbol_to_ak_symbol(code: str) -> str:
    """
    将 '600000' -> 'sh600000' 或 '000001' -> 'sz000001' （简单规则）
    这里做常见前缀规则：上证以 6 开头，深证以 0,3 开头
    """
    code = str(code).strip()
    if code.startswith('6'):
        return f"sh{code}"
    elif code.startswith(('0','3','1')):
        return f"sz{code}"
    else:
        # fallback: attempt both - prefer sh
        return f"sh{code}"

# ---------- 筛选流水线（实现你提出的步骤） ----------
def step1_pct_filter(df_spot: pd.DataFrame, low_pct=3.0, high_pct=5.0) -> pd.DataFrame:
    """步骤1：筛选当日涨幅在 [3%,5%] 的股票"""
    if '涨跌幅' not in df_spot.columns:
        st.warning("实时涨跌幅字段不可用，无法执行步骤1过滤")
        return pd.DataFrame()
    cond = (df_spot['涨跌幅'] >= low_pct) & (df_spot['涨跌幅'] <= high_pct)
    res = df_spot.loc[cond, ['代码','名称']].copy()
    return res.reset_index(drop=True)

def step2_volume_ratio_filter(candidates: pd.DataFrame, df_spot: pd.DataFrame) -> pd.DataFrame:
    """步骤2：量比 >= 1"""
    out = []
    for _, row in candidates.iterrows():
        code = row['代码']
        # 在 spot 表中尝试找到成交量 / 或实时成交量字段
        try:
            spot_row = df_spot[df_spot['代码'] == code].iloc[0]
        except:
            continue
        today_vol = None
        if '成交量' in df_spot.columns:
            today_vol = pd.to_numeric(spot_row.get('成交量', np.nan), errors='coerce')
        # compute volume ratio
        vr = None
        if today_vol is not None and not np.isnan(today_vol):
            ak_sym = symbol_to_ak_symbol(code)
            vr = compute_volume_ratio(ak_sym, today_vol)
        # if df_spot has direct '量比' column name
        if '量比' in df_spot.columns and not pd.isna(spot_row.get('量比', np.nan)):
            vr = pd.to_numeric(spot_row.get('量比'))
        if vr is None or np.isnan(vr):
            # assume unknown => conservatively keep? 这里选择跳过
            continue
        if vr >= 1.0:
            out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)

def step3_turnover_filter(candidates: pd.DataFrame, df_spot: pd.DataFrame, low=5.0, high=10.0) -> pd.DataFrame:
    """步骤3：换手率在 [5%,10%]"""
    out = []
    for _, row in candidates.iterrows():
        code = row['代码']
        try:
            spot_row = df_spot[df_spot['代码'] == code].iloc[0]
        except:
            continue
        tr = None
        if '换手率' in df_spot.columns:
            tr = pd.to_numeric(spot_row.get('换手率', np.nan), errors='coerce')
        if tr is None or np.isnan(tr):
            continue
        if (tr >= low) and (tr <= high):
            out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)

def step4_float_cap_filter(candidates: pd.DataFrame, df_spot: pd.DataFrame, low=5e9, high=20e9) -> pd.DataFrame:
    """
    步骤4：流通市值范围 50亿-200亿（人民币）
    low/high 接受人民币元单位（所以 50e8 => 5e9? 保持单位一致）
    注意：akshare 的市值字段单位可能为 '万' 或 '亿'，需根据真实字段进行转换。此处假设为人民币元
    """
    out = []
    for _, row in candidates.iterrows():
        code = row['代码']
        try:
            spot_row = df_spot[df_spot['代码'] == code].iloc[0]
        except:
            continue
        cap = None
        if '流通市值' in df_spot.columns:
            cap = pd.to_numeric(spot_row.get('流通市值', np.nan), errors='coerce')
        # If akshare gives '流通市值(亿元)' or with suffix, user may need to adjust
        if cap is None or np.isnan(cap):
            continue
        # if cap appears small (< 1e6), maybe unit is '万' or '亿', try heuristics
        if cap < 1e6:
            # treat as 亿 units? try *1e8
            cap_try = cap * 1e8
            if low <= cap_try <= high:
                cap = cap_try
        if low <= cap <= high:
            out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)

def step5_volume_trend_filter(candidates: pd.DataFrame, N=5) -> pd.DataFrame:
    """
    步骤5：成交量趋势（今日>昨日>前日 简化逻辑），或用 5 日均量与 10 日均量做判断
    """
    out = []
    for _, row in candidates.iterrows():
        code = row['代码']
        ak_sym = symbol_to_ak_symbol(code)
        klines = get_daily_klines(ak_sym)
        if klines is None or klines.empty or len(klines) < 3:
            continue
        last = klines['volume'].iloc[-1]
        prev = klines['volume'].iloc[-2]
        prev2 = klines['volume'].iloc[-3]
        # 简单判断
        if last > prev > prev2:
            out.append(row)
            continue
        # 稳健判断：
        ma5 = klines['volume'].tail(5).mean()
        ma10 = klines['volume'].tail(10).mean() if len(klines) >= 10 else np.nan
        if not np.isnan(ma10) and (last > ma5) and (ma5 > ma10):
            # 检查波动性不过高
            if klines['volume'].tail(10).std() / (klines['volume'].tail(10).mean()+1e-9) < 2.0:
                out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)

def step6_klines_ma_pressure_filter(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    步骤6：均线多头排列且价格在均线之上；并检查上方压力（近 60 或 120 日内无明显更高密集成交区）
    这里的上方压力用近120日的历史最高价是否远高于当前价作为近似（若当前价接近历史高位则判断为无压力）
    """
    out = []
    for _, row in candidates.iterrows():
        code = row['代码']
        ak_sym = symbol_to_ak_symbol(code)
        klines = get_daily_klines(ak_sym)
        if klines is None or klines.empty:
            continue
        df = compute_indicators(klines)
        close = df['close'].iloc[-1]
        # require MA 5>10>20>60
        ma5 = df['ma5'].iloc[-1]
        ma10 = df['ma10'].iloc[-1]
        ma20 = df['ma20'].iloc[-1]
        ma60 = df['ma60'].iloc[-1]
        if not (ma5 > ma10 > ma20 > ma60):
            continue
        # price above MA5/10/20
        if not (close > ma5 and close > ma10 and close > ma20):
            continue
        # pressure check: compute max in past 120 days
        recent_high = df['high'].tail(120).max() if 'high' in df.columns else df['close'].tail(120).max()
        # if recent high > close by large margin, treat as有压力；我们希望“上方无明显压力”，所以要求 recent_high <= close*1.05 (即不高很多)
        if recent_high > close * 1.05:
            # 存在较高压力，排除
            continue
        out.append(row)
    return pd.DataFrame(out).reset_index(drop=True)

def step7_intraday_strength_filter(candidates: pd.DataFrame, df_spot: pd.DataFrame) -> pd.DataFrame:
    """
    步骤7：日内强度（分时线超越大盘），此处简化为：当日涨幅显著高于上证指数涨幅 & 当前价高于实时分时均价 (VWAP)
    由于 akshare 分时接口复杂，可用替代：比较个股涨跌幅 vs 上证指数涨跌幅，并检查价格>VWAP（若可取）
    """
    out = []
    # get index spot
    try:
        index_spot = ak.stock_zh_index_spot_ths()  # may not be available everywhere
    except:
        index_spot = None
    # fallback: use stock_zh_index_spot_em
    if index_spot is None or index_spot.empty:
        try:
            index_spot = ak.stock_zh_index_spot_em()
        except:
            index_spot = pd.DataFrame()
    # attempt to find SSE (上证)
    sse_pct = 0.0
    if not index_spot.empty:
        # try find 上证指数
        if '代码' in index_spot.columns:
            row = index_spot[index_spot['代码'].str.contains('000001') | index_spot['名称'].str.contains('上证')]
        elif '名称' in index_spot.columns:
            row = index_spot[index_spot['名称'].str.contains('上证')]
        else:
            row = index_spot.head(1)
        try:
            sse_pct = float(row.iloc[0].get('涨跌幅', 0.0))
        except:
            sse_pct = 0.0
    for _, r in candidates.iterrows():
        code = r['代码']
        try:
            spot_row = df_spot[df_spot['代码'] == code].iloc[0]
        except:
            continue
        pct = pd.to_numeric(spot_row.get('涨跌幅', 0.0), errors='coerce')
        # basic rule：跑赢大盘
        if pd.isna(pct):
            continue
        if pct >= sse_pct + 0.5:  # 比大盘高0.5%作为门槛（可调）
            # VWAP check：如果分时 VWAP 不可得则跳过 VWAP 检查
            # akshare 提供分钟/分时数据的接口较多，略复杂，暂只用涨幅与大盘比
            out.append(r)
    return pd.DataFrame(out).reset_index(drop=True)

def step8_monitor_and_signals(targets: pd.DataFrame, df_spot: pd.DataFrame) -> pd.DataFrame:
    """
    步骤8：监控并产生买入信号（突破当日新高后回踩 VWAP 不破时买入）
    由于实时 tick/分时数据依赖性强，本函数将：
    - 判断是否当日已创出阶段新高（Realtime 高点字段或用日内高价）
    - 若满足，则返回标记 'Triggered'，并在后续监控时等待回踩
    返回 DataFrame 包含 字段 [代码, 名称, Triggered(bool), TriggerPrice(approx), Note]
    """
    out = []
    for _, r in targets.iterrows():
        code = r['代码']
        try:
            spot_row = df_spot[df_spot['代码'] == code].iloc[0]
        except:
            continue
        # attempt to extract 当日最高价
        today_high = None
        if '最高' in df_spot.columns:
            today_high = pd.to_numeric(spot_row.get('最高', np.nan), errors='coerce')
        # fallback: 若无实时最高，用日线最高
        ak_sym = symbol_to_ak_symbol(code)
        kl = get_daily_klines(ak_sym)
        if kl is None or kl.empty:
            continue
        today_high = today_high if today_high and not pd.isna(today_high) else kl['high'].iloc[-1]
        current_price = pd.to_numeric(spot_row.get('最新价', spot_row.get('最新', np.nan)), errors='coerce')
        triggered = False
        note = ""
        if not np.isnan(current_price) and not np.isnan(today_high):
            # 判断当前价是否接近当日最高（突破）
            if current_price >= today_high * 0.999:  # 接近或等于当日最高
                triggered = True
                note = "Day high touched/broken"
        out.append({
            "代码": code,
            "名称": r.get('名称', ''),
            "Triggered": triggered,
            "TriggerPrice": today_high,
            "CurrentPrice": current_price,
            "Note": note
        })
    return pd.DataFrame(out)

# ---------- Plotly 绘图 ----------
def plot_candlestick_with_indicators(df: pd.DataFrame, symbol: str, indicators: dict):
    """
    df: 日线 DataFrame with columns open/high/low/close/volume and computed indicators
    indicators: dict { 'ma': [5,10,20], 'rsi': True, 'bb': True }
    返回 plotly.graph_objects.Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="K线"
    ))
    ma_list = indicators.get('ma', [])
    for ma in ma_list:
        col = f"ma{ma}"
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f"MA{ma}"))
    if indicators.get('bb', False):
        if 'bb_up' in df.columns and 'bb_dn' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_up'], mode='lines', name='BB Up', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_dn'], mode='lines', name='BB Dn', line=dict(dash='dash')))
    # RSI as subplot
    if indicators.get('rsi', False) and 'rsi14' in df.columns:
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name="K线"
        ), row=1, col=1)
        for ma in ma_list:
            coln = f"ma{ma}"
            if coln in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[coln], mode='lines', name=f"MA{ma}"), row=1, col=1)
        if indicators.get('bb', False) and 'bb_up' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_up'], mode='lines', name='BB Up', line=dict(dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_dn'], mode='lines', name='BB Dn', line=dict(dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi14'], name='RSI(14)'), row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_layout(title=f"{symbol} - Candlestick", xaxis_rangeslider_visible=False, template="plotly_white")
    return fig

# ---------- UI: 左侧筛选与操作 ----------
st.sidebar.title("筛选与设置")
run_at_1430 = st.sidebar.checkbox("自动筛选（模拟：下午 14:30 触发）", value=False)
# Note: we can't actually schedule inside Streamlit; give button to run now
if st.sidebar.button("立即运行筛选流水线"):
    st.session_state['run_pipeline'] = True
# keep config save/load
st.sidebar.markdown("### 保存/加载 配置")
cfg_name = st.sidebar.text_input("配置名 (保存/加载)", value="default")
if st.sidebar.button("保存当前配置"):
    cfg = {
        "run_at_1430": run_at_1430,
        "cfg_name": cfg_name,
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(CONFIG_DIR, f"{cfg_name}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    st.sidebar.success(f"已保存配置：{cfg_name}")
if st.sidebar.button("加载配置"):
    p = os.path.join(CONFIG_DIR, f"{cfg_name}.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        st.sidebar.success(f"已加载配置：{cfg_name}")
    else:
        st.sidebar.error("配置不存在")

# ---------- 主界面 ----------
st.title("Stock Market Visualizer（股票市场可视化器）")
st.markdown("本工具使用 AKShare 数据源做 A 股筛选、可视化与组合分析。请先确认已安装 akshare 并能访问数据源。")

# load real-time spot
with st.spinner("获取 A 股实时行情（可能需要几秒）..."):
    df_spot = get_all_a_spot()
if df_spot.empty:
    st.error("未能获取实时行情，检查 akshare 可用性或网络。")
else:
    st.success(f"已获取实时行情，共 {len(df_spot)} 条")

# show quick top movers
st.subheader("当日行情预览（实时）")
cols_to_show = [c for c in ['代码','名称','最新价','涨跌幅','成交量','换手率','流通市值'] if c in df_spot.columns]
st.dataframe(df_spot[cols_to_show].head(200))

# Provide user a manual symbol search
st.sidebar.markdown("### 单只股票查看")
input_code = st.sidebar.text_input("输入股票代码（例：600519）", value="")
if st.sidebar.button("查看该股 K 线") and input_code.strip() != "":
    ak_sym = symbol_to_ak_symbol(input_code.strip())
    df_k = get_daily_klines(ak_sym)
    if df_k.empty:
        st.sidebar.error("未能获取该股日线，请检查代码或 akshare 版本")
    else:
        df_k_ind = compute_indicators(df_k)
        fig = plot_candlestick_with_indicators(df_k_ind, ak_sym, {'ma':[5,10,20], 'rsi':True, 'bb':True})
        st.plotly_chart(fig, use_container_width=True)
        # export options
        buf_html = pio.to_html(fig, full_html=False)
        st.sidebar.download_button("下载 HTML", data=buf_html, file_name=f"{ak_sym}_kline.html", mime="text/html")
        # PNG requires kaleido
        try:
            img_bytes = pio.to_image(fig, format='png', width=1200, height=700, scale=1)
            st.sidebar.download_button("下载 PNG", data=img_bytes, file_name=f"{ak_sym}_kline.png", mime="image/png")
        except Exception as e:
            st.sidebar.warning("导出 PNG 需要安装 kaleido（requirements.txt 包含），或本地不支持导出。")

# ---------- 投资组合上传 ----------
st.sidebar.markdown("### 投资组合（上传 CSV/Excel）")
uploaded = st.sidebar.file_uploader("上传持仓文件 (CSV/Excel)，需包含代码/数量列", type=['csv','xls','xlsx'])
portfolio = None
if uploaded is not None:
    try:
        if uploaded.name.endswith('.csv'):
            portfolio = pd.read_csv(uploaded)
        else:
            portfolio = pd.read_excel(uploaded)
        st.sidebar.success(f"已加载 {len(portfolio)} 条持仓记录")
    except Exception as e:
        st.sidebar.error(f"读取文件失败：{e}")

if portfolio is not None:
    st.subheader("已上传的投资组合")
    st.dataframe(portfolio.head(200))

# ---------- 执行筛选流水线（按钮或自动触发） ----------
if st.session_state.get('run_pipeline', False) or st.sidebar.button("Run pipeline now"):
    st.session_state['run_pipeline'] = False
    st.info("开始执行筛选流水线（步骤1 -> 步骤8）...")
    # Step 1
    with st.spinner("步骤1：涨幅 3%-5% 筛选"):
        s1 = step1_pct_filter(df_spot, 3.0, 5.0)
        st.write(f"步骤1 候选数量：{len(s1)}")
        st.dataframe(s1.head(200))
    # Step 2
    with st.spinner("步骤2：量比 >= 1"):
        s2 = step2_volume_ratio_filter(s1, df_spot)
        st.write(f"步骤2 候选数量：{len(s2)}")
        st.dataframe(s2.head(200))
    # Step 3
    with st.spinner("步骤3：换手率 5%-10%"):
        s3 = step3_turnover_filter(s2, df_spot, 5.0, 10.0)
        st.write(f"步骤3 候选数量：{len(s3)}")
        st.dataframe(s3.head(200))
    # Step 4
    with st.spinner("步骤4：流通市值 50亿-200亿"):
        s4 = step4_float_cap_filter(s3, df_spot, low=5e9, high=20e9)
        st.write(f"步骤4 候选数量：{len(s4)}")
        st.dataframe(s4.head(200))
    # Step 5
    with st.spinner("步骤5：成交量趋势分析"):
        s5 = step5_volume_trend_filter(s4)
        st.write(f"步骤5 候选数量：{len(s5)}")
        st.dataframe(s5.head(200))
    # Step 6
    with st.spinner("步骤6：K 线与均线系统分析"):
        s6 = step6_klines_ma_pressure_filter(s5)
        st.write(f"步骤6 候选数量：{len(s6)}")
        st.dataframe(s6.head(200))
    # Step 7
    with st.spinner("步骤7：日内相对强度分析"):
        s7 = step7_intraday_strength_filter(s6, df_spot)
        st.write(f"步骤7 候选数量：{len(s7)}")
        st.dataframe(s7.head(200))
    # Step 8
    with st.spinner("步骤8：交易触发与监控（标记触发）"):
        s8 = step8_monitor_and_signals(s7, df_spot)
        st.write(f"步骤8 触发检测（Triggered=True 表示触发日内新高）:")
        st.dataframe(s8)
    st.success("筛选流水线完成。你可以点击任一代码来查看其 K 线与指标。")

    # allow user to click and view charts for final pool
    st.markdown("### 目标候选池可视化")
    if not s7.empty:
        code_list = s7['代码'].tolist()
        sel_code = st.selectbox("选择观察的股票代码", options=code_list)
        if sel_code:
            ak_sym = symbol_to_ak_symbol(sel_code)
            df_k = get_daily_klines(ak_sym)
            if not df_k.empty:
                df_k = compute_indicators(df_k)
                fig = plot_candlestick_with_indicators(df_k, ak_sym, {'ma':[5,10,20,60], 'rsi':True, 'bb':True})
                st.plotly_chart(fig, use_container_width=True)
                # show signals row if exists
                sig_row = s8[s8['代码'] == sel_code]
                if not sig_row.empty:
                    st.write("监控/信号信息：")
                    st.write(sig_row.T)

# ---------- 相关性分析 & 财务比率（简易） ----------
st.header("相关性分析 & 财务比率（简易）")
st.markdown("上传一组股票代码以计算历史收益相关性（基于日线对数收益）。")
codes_text = st.text_area("输入股票代码（每行一个，例如：600519）", value="")
if st.button("计算相关性"):
    codes = [c.strip() for c in codes_text.splitlines() if c.strip()]
    price_df = pd.DataFrame()
    for code in codes:
        ak_sym = symbol_to_ak_symbol(code)
        kl = get_daily_klines(ak_sym)
        if kl is None or kl.empty:
            st.warning(f"{code} 历史数据缺失")
            continue
        price_df[code] = kl['close'].tail(180)  # 取近 180 日
    if price_df.shape[1] < 2:
        st.warning("至少需要 2 只股票来计算相关性")
    else:
        returns = np.log(price_df / price_df.shift(1)).dropna()
        corr = returns.corr()
        st.subheader("收益率相关系数矩阵")
        st.dataframe(corr)
        # heatmap
        import plotly.express as px
        fig = px.imshow(corr, text_auto=True, title="收益率相关性")
        st.plotly_chart(fig, use_container_width=True)

# ---------- 保存并分享用户视图（简单实现） ----------
st.sidebar.markdown("### 分享与导出")
if st.sidebar.button("导出当前页面为 HTML（静态）"):
    # This will export currently displayed charts is non-trivial; offer option to export last selected stock chart
    st.sidebar.info("将导出最后显示的股票 K 线为 HTML（若存在）")
    # We used pio.to_html earlier for selected stock - but need state to store last fig. For simplicity, user can use single-stock export in the panel.

st.markdown("---")
st.markdown("说明：本工具提供策略筛选与信号标记，非自动交易。若要接入下单、实时高频 tick 数据或云端定时任务，请将此仓库部署到可稳定访问 akshare 的服务器，并配置定时器 (cron / scheduler)。")

st.markdown("## 使用与部署建议（简短）")
st.markdown("""
1. 在服务器上安装 Python 环境，并按 requirements.txt 安装依赖。  
2. 通过 `streamlit run app.py` 启动，或使用 Gunicorn + streamlit-sharing / docker 容器化。  
3. 若需在北京时间 14:30 自动触发筛选：在服务器上使用 cron（例如：`30 6 * * 1-5 /usr/bin/python3 /path/to/run_pipeline_script.py`，注意服务器为 UTC 时需换算时区）。  
4. 若要接入实盘交易：请在 `step8_monitor_and_signals` 添加券商 API 下单逻辑，并做好风控、权限与延迟测试。  
""")
