import yfinance as yf
import pandas as pd
import numpy as np
import os

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def download_stock_data(ticker, start='2010-01-01', end='2025-06-30'):
    df = yf.download(ticker, start=start, end=end)
    df.dropna(inplace=True)

    # Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['MACD'], df['MACD_Signal'] = compute_macd(df['Close'])

    return df

def save_data(ticker, df, save_dir='data/price'):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{ticker}_price.csv')
    df.to_csv(file_path)
    print(f"[✓] Saved {ticker} data to {file_path}")

def main():
    tickers = ['TSLA', 'MSFT', 'AAPL']
    for ticker in tickers:
        df = download_stock_data(ticker)
        save_data(ticker, df)

if __name__ == "__main__":
    main()
