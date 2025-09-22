import os
import ccxt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from analyse import add_technical_indicators

# Binance API keys
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'your_api_key_here')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'your_api_secret_here')

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True
})

# Fetch BTC/USDT data
try:
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=365)
    df = pd.DataFrame(ohlcv, columns=['Timestamp','Open','High','Low','Close','Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
except Exception as e:
    print(f"Data fetch failed: {e}")
    exit(1)

# Add technical indicators
df = add_technical_indicators(df)
df = df[['Close','RSI','MACD']].dropna()

# Preprocess
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

X = []
y = []
window_size = 20
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i])
    y.append(scaled_data[i,0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 3))  # 3 features: Close, RSI, MACD

# Build LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],3)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# Save model
model.save("chart_lstm_model.h5")
print("AI model saved as chart_lstm_model.h5")

# Commit message:
"""
Initial commit of train_model.py
- Fetches BTC/USDT data using Binance API keys
- Adds Close, RSI, MACD as features
- Builds and trains LSTM model
- Saves model as chart_lstm_model.h5 for Streamlit app
"""
