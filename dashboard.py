import os
import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')  # Set Matplotlib backend for Streamlit Cloud
import matplotlib.pyplot as plt
from analyse import add_technical_indicators, run_ai_model, placeholder_ai_analysis
from tensorflow.keras.models import load_model
import ccxt

# =============================
# Load Binance API credentials
# =============================
BINANCE_API_KEY = st.secrets.get("BINANCE_API_KEY", None)
BINANCE_API_SECRET = st.secrets.get("BINANCE_API_SECRET", None)

# Load AI model
model_path = "chart_lstm_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("AI model loaded successfully!")
else:
    model = None
    st.warning("AI model not found. Using placeholder analysis.")

# =============================
# Functions to fetch data
# =============================
@st.cache_data
def fetch_binance_data(symbol='BTC/USDT', timeframe='1d', limit=365):
    try:
        exchange = ccxt.binance({
            'apiKey': BINANCE_API_KEY,
            'secret': BINANCE_API_SECRET
        })
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Binance data fetch failed: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_coindcx_data(symbol='BTC/INR', timeframe='1d', limit=365):
    try:
        exchange = ccxt.coindcx()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"CoinDCX data fetch failed: {e}")
        return pd.DataFrame()

# =============================
# Streamlit Layout
# =============================
st.title("Crypto Dashboard Pro 🚀")

# Sidebar
st.sidebar.header("Select Crypto")
symbol = st.sidebar.text_input("Enter crypto symbol (e.g., BTC/USDT or BTC/INR):", "BTC/USDT")
timeframe = st.sidebar.selectbox("Select timeframe:", ['1d', '1h', '1m'])
limit = st.sidebar.number_input("Number of historical candles:", min_value=30, max_value=1000, value=365)

# Fetch Data
st.info(f"Fetching {symbol} data...")
if '/INR' in symbol.upper():
    df = fetch_coindcx_data(symbol.upper(), timeframe, limit)
else:
    df = fetch_binance_data(symbol.upper(), timeframe, limit)

# =============================
# Display Results
# =============================
if not df.empty:
    st.success(f"Data fetched successfully for {symbol}")
    df = add_technical_indicators(df)

    st.subheader("Price Data")
    st.dataframe(df.tail())

    st.subheader("Candlestick Chart")
    mc = mpf.make_marketcolors(up="green", down="red", wick="black")
    s = mpf.make_mpf_style(marketcolors=mc)
    fig, ax = mpf.plot(df.set_index('Timestamp'),
                       type='candle',
                       style=s,
                       volume=True,
                       show_nontrading=True,
                       returnfig=True)
    st.pyplot(fig)

    st.subheader("AI / Technical Analysis")
    try:
        trend = run_ai_model(model, df)
        st.write(trend)
    except Exception as e:
        st.error(f"AI analysis failed: {e}")
        st.write("Falling back to placeholder analysis")
        st.write(placeholder_ai_analysis(df))

else:
    st.error("No data available to display.")

st.write("---")
st.write("🔑 Built with Streamlit, CCXT, mplfinance, and TensorFlow")
