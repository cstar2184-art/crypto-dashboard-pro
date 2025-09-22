import os
import streamlit as st
import pandas as pd
import mplfinance as mpf
import matplotlib
matplotlib.use('Agg')  # Backend for Streamlit Cloud
import matplotlib.pyplot as plt
from analyse import add_technical_indicators, run_ai_model, placeholder_ai_analysis
from tensorflow.keras.models import load_model
import ccxt

# Load AI model
model_path = "chart_lstm_model.h5"
if os.path.exists(model_path):
    model = load_model(model_path)
    st.success("AI model loaded successfully!")
else:
    model = None
    st.warning("AI model not found. Using placeholder analysis.")

# Binance setup with API keys
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'your_api_key_here')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'your_api_secret_here')

binance = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True
})

# Fetch Binance data
@st.cache_data
def fetch_binance_data(symbol='BTC/USDT', timeframe='1d', limit=365):
    try:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['Timestamp','Open','High','Low','Close','Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Binance data fetch failed: {e}")
        return pd.DataFrame()

# Fetch CoinDCX data
@st.cache_data
def fetch_coindcx_data(symbol='BTC/INR', timeframe='1d', limit=365):
    try:
        exchange = ccxt.coindcx()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['Timestamp','Open','High','Low','Close','Volume'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        return df
    except Exception as e:
        st.error(f"CoinDCX data fetch failed: {e}")
        return pd.DataFrame()

# Streamlit layout
st.title("Crypto Dashboard Pro")
st.sidebar.header("Select Crypto")
symbol = st.sidebar.text_input("Enter crypto symbol (BTC/USDT or BTC/INR):", "BTC/USDT")
timeframe = st.sidebar.selectbox("Select timeframe:", ['1d','1h','1m'])
limit = st.sidebar.number_input("Number of historical candles:", min_value=30, max_value=1000, value=365)

# Fetch data
st.info(f"Fetching {symbol} data...")
if '/INR' in symbol.upper():
    df = fetch_coindcx_data(symbol.upper(), timeframe, limit)
else:
    df = fetch_binance_data(symbol.upper(), timeframe, limit)

if not df.empty:
    st.success(f"Data fetched successfully for {symbol}")
    df = add_technical_indicators(df)

    st.subheader("Price Data")
    st.dataframe(df.tail())

    st.subheader("Candlestick Chart")
    mc = mpf.make_marketcolors(up="green", down="red", wick="black")
    s = mpf.make_mpf_style(marketcolors=mc)
    fig, ax = mpf.plot(df.set_index('Timestamp'), type='candle', style=s, volume=True, returnfig=True)
    st.pyplot(fig)

    st.subheader("AI / Technical Analysis")
    trend = run_ai_model(model, df)
    st.write(trend)
else:
    st.error("No data available.")

st.write("---")
st.write("Built with Streamlit, CCXT, mplfinance, and TensorFlow")

# Commit message:
"""
Initial commit of dashboard.py
- Renamed main Streamlit file from app.py to dashboard.py
- Fetches data from Binance or CoinDCX using API keys and ccxt
- Displays candlestick chart with mplfinance
- Integrates AI model predictions and placeholder fallback
- Uses chart_lstm_model.h5 for consistency with training script
- Added matplotlib Agg backend and caching for performance
"""
