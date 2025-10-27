"""
Streamlit app: Futu NQ & ES 期货实时监控面板（修正版）
----------------------------------------------------
已修復 f-string 未終止造成的 SyntaxError，並整理錯誤處理與說明。

使用步驟：
1) 安裝依賴：
   pip install streamlit futu yfinance pandas
   （若不使用 Futu，可不安裝 futu；若使用 Futu，請先啟動 FutuOpenD）
2) 將此檔另存為 futu_streamlit_monitor.py
3) 執行：
   streamlit run futu_streamlit_monitor.py

備註：Streamlit 會在非標準執行環境顯示 “missing ScriptRunContext” 的警告，這通常可忽略。

"""

import streamlit as st
from datetime import datetime
import time
import pandas as pd
import io

# Try imports
try:
    from futu import OpenQuoteContext
    FUTU_AVAILABLE = True
except Exception:
    FUTU_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except Exception:
    YFINANCE_AVAILABLE = False

st.set_page_config(page_title='NQ & ES Monitor (Futu/YF)', layout='wide')

st.title('NQ & ES 期货实时监控（Streamlit） — 修正版')

# Sidebar controls
st.sidebar.header('设置')
use_futu = st.sidebar.checkbox('优先使用 FutuOpenD（若不可用回退至 yfinance）', value=FUTU_AVAILABLE)
host = st.sidebar.text_input('FutuOpenD host', value='127.0.0.1')
port = st.sidebar.number_input('FutuOpenD port', value=11111, step=1)
interval = st.sidebar.number_input('轮询间隔（秒）', value=10, min_value=1, step=1)

st.sidebar.markdown('---')
st.sidebar.subheader('期货符号（请依富途客户端确认）')
NQ_symbol = st.sidebar.text_input('NQ (示例)', value='NQmain.US')
ES_symbol = st.sidebar.text_input('ES (示例)', value='ESmain.US')

st.sidebar.markdown('---')
st.sidebar.subheader('阈值（百分比）')
th_up_pct = st.sidebar.number_input('上涨阈值（%）', value=0.3, step=0.1)
th_down_pct = st.sidebar.number_input('下跌阈值（%）', value=0.3, step=0.1)
# Note: store as positive numbers in UI; convert to decimals and negative for down
th_up = th_up_pct / 100.0
th_down = -abs(th_down_pct / 100.0)

# Storage for session
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame(columns=['timestamp','NQ_price','NQ_prev','NQ_pct','ES_price','ES_prev','ES_pct','signal','note'])
if 'running' not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns([2,1])
with col1:
    st.header('即時狀態')
    status_placeholder = st.empty()
    table_placeholder = st.empty()
with col2:
    st.header('控制')
    start = st.button('開始監控')
    stop = st.button('停止')
    clear = st.button('清除紀錄')

# Helper functions

def get_prices_futu(oqc, symbols):
    results = {}
    for name, sym in symbols.items():
        try:
            ret, data = oqc.get_market_snapshot([sym])
            if ret == 0 and len(data) > 0:
                row = data.iloc[0]
                last_price = row.get('last_price', None)
                prev_close = row.get('last_close', None) or row.get('prev_close', None)
                # Ensure numeric or None
                last_price = float(last_price) if last_price is not None else None
                prev_close = float(prev_close) if prev_close is not None else None
                results[name] = (last_price, prev_close)
            else:
                results[name] = (None, None)
        except Exception:
            results[name] = (None, None)
    return results


def get_prices_yf(tickers):
    results = {}
    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            # fast_info may vary by yfinance version; use history fallback
            info = getattr(t, 'fast_info', {}) or {}
            last = info.get('last_price') or info.get('last_close') or info.get('regular_market_price')
            prev = info.get('previous_close') or info.get('last_close')
            if last is None:
                hist = t.history(period='1d')
                if not hist.empty:
                    last = hist['Close'].iloc[-1]
                    prev = hist['Close'].iloc[-1]
            last_price = float(last) if last is not None else None
            prev_close = float(prev) if prev is not None else None
            results[name] = (last_price, prev_close)
        except Exception:
            results[name] = (None, None)
    return results


def compute_pct(price, prev):
    if price is None or prev is None or prev == 0:
        return None
    return (price - prev) / prev


def aggregate(pcts, up, down):
    bullish = sum(1 for v in pcts.values() if v is not None and v >= up)
    bearish = sum(1 for v in pcts.values() if v is not None and v <= down)
    valid = sum(1 for v in pcts.values() if v is not None)
    if valid == 0:
        return 'UNKNOWN', 'No valid data'
    if bullish == valid:
        return 'RISK ON', 'All futures above bullish threshold'
    if bearish == valid:
        return 'RISK OFF', 'All futures below bearish threshold'
    return 'NEUTRAL', f'{bullish} bullish / {bearish} bearish / {valid - bullish - bearish} neutral'

# Start/stop logic
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False
if clear:
    st.session_state.df = st.session_state.df.iloc[0:0]

# Main loop runner
if st.session_state.running:
    try:
        oqc = None
        if use_futu and FUTU_AVAILABLE:
            try:
                oqc = OpenQuoteContext(host=host, port=port)
            except Exception:
                oqc = None

        symbols = {'NQ': NQ_symbol, 'ES': ES_symbol}
        yf_symbols = {'NQ': 'NQ=F', 'ES': 'ES=F'}

        if oqc is not None:
            prices = get_prices_futu(oqc, symbols)
        else:
            if YFINANCE_AVAILABLE:
                prices = get_prices_yf(yf_symbols)
            else:
                prices = {'NQ': (None, None), 'ES': (None, None)}

        pcts = {}
        for k, v in prices.items():
            pcts[k] = compute_pct(v[0], v[1])

        signal, note = aggregate(pcts, th_up, th_down)

        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        row = {
            'timestamp': now,
            'NQ_price': prices['NQ'][0], 'NQ_prev': prices['NQ'][1], 'NQ_pct': pcts['NQ'],
            'ES_price': prices['ES'][0], 'ES_prev': prices['ES'][1], 'ES_pct': pcts['ES'],
            'signal': signal, 'note': note
        }

        # Prepend to dataframe (keep recent first)
        df_new = pd.DataFrame([row])
        st.session_state.df = pd.concat([df_new, st.session_state.df], ignore_index=True).head(5000)

        # Update UI
        status_text = f"Last update (UTC): {now}  \nSignal: {signal} — {note}"
        status_placeholder.markdown(status_text)
        table_placeholder.dataframe(st.session_state.df.rename(columns=lambda c: c.replace('_',' ')), height=420)

        # Charts
        with st.expander('價格走勢圖（近 60 筆）'):
            df_plot = st.session_state.df.head(60).iloc[::-1]
            if not df_plot.empty:
                df_plot = df_plot.reset_index(drop=True)
                st.line_chart(df_plot[['NQ_price','ES_price']])

        # CSV download
        csv_buf = io.StringIO()
        st.session_state.df.to_csv(csv_buf, index=False)
        st.download_button('下載 CSV（含紀錄）', csv_buf.getvalue(), file_name='futu_nq_es_log.csv')

        if oqc is not None:
            try:
                oqc.close()
            except Exception:
                pass

        # Sleep then rerun
        time.sleep(interval)
        st.experimental_rerun()

    except Exception as e:
        st.error(f'執行錯誤: {e}')
        st.session_state.running = False
else:
    st.info('停止中 — 按「開始監控」啟動實時監控，或修改側邊欄設定後再啟動。')
    if not FUTU_AVAILABLE and use_futu:
        st.warning('系統檢測未安裝 futu 套件或 FutuOpenD 無法使用，將回退至 yfinance（若可用）。')
    if not YFINANCE_AVAILABLE and not FUTU_AVAILABLE:
        st.error('未偵測到可用的資料來源：請安裝 futu 或 yfinance，或啟動 FutuOpenD。')

# Footer
st.markdown('---')
st.caption('工具僅供市場監控示例，不構成投資建議。')
