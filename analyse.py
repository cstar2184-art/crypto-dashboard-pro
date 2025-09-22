import pandas as pd
import ta
import numpy as np

# Add technical indicators
def add_technical_indicators(df):
    if df.empty:
        return df
    df = df.copy()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['Bollinger_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
    df['Bollinger_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
    return df

# Placeholder AI analysis
def placeholder_ai_analysis(df):
    if df.empty:
        return "No data available"
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1]:
        return "Uptrend (Placeholder)"
    elif df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1]:
        return "Downtrend (Placeholder)"
    else:
        return "Neutral (Placeholder)"

# Run AI model with LSTM input consistency
def run_ai_model(model, df):
    if model is None:
        return placeholder_ai_analysis(df)
    required_columns = ['Close', 'RSI', 'MACD']
    if not all(col in df.columns for col in required_columns):
        return placeholder_ai_analysis(df)
    
    df_features = df[required_columns].ffill().dropna().values

    # Ensure input shape matches LSTM training: (1, window_size, 3)
    window_size = 20
    if len(df_features) < window_size:
        return placeholder_ai_analysis(df)
    
    X_input = np.array([df_features[-window_size:]])  # shape (1, window_size, 3)
    
    try:
        prediction = model.predict(X_input, verbose=0)
        trend = "Uptrend" if prediction[-1,0] > df_features[-1,0] else "Downtrend"
        return f"AI trend prediction: {trend} (Value: {prediction[-1,0]:.2f})"
    except Exception as e:
        return f"AI model prediction failed: {e}"

# Commit message:
"""
Initial commit of analyse.py
- Defines functions for technical indicators and AI analysis
- Adds RSI, MACD, Bollinger Bands using ta library
- Implements placeholder analysis for fallback
- Ensures LSTM input shape matches training data
"""
