import pandas as pd
import numpy as np
import torch
import joblib
from backend.model.train_model import FusionMLP
from datetime import datetime

FEATURE_ORDER = [
    "High", "Low", "Open", "Volume", "SMA_20", "SMA_50",
    "RSI", "MACD", "MACD_Signal", "avg_sentiment"
]

def backtest(ticker_file, model_path="models/fusion_mlp.pt", scaler_path="models/scaler.pkl", initial_cash=10000):
    # Load dataset
    df = pd.read_csv(ticker_file, parse_dates=["Date"])
    df = df.dropna().reset_index(drop=True)

    features = df[FEATURE_ORDER]
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(features)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    # Load model
    model = FusionMLP(input_dim=len(FEATURE_ORDER))
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        preds = (model(X_tensor) > 0.5).float().squeeze().numpy()

    df["Prediction"] = preds

    # Strategy simulation
    cash = initial_cash
    equity_curve = []
    daily_returns = []

    for i in range(len(df)-1):
        if df.loc[i, "Prediction"] == 1:
            r = df.loc[i+1, "Close"] / df.loc[i, "Close"] - 1
            cash *= (1 + r)
            daily_returns.append(r)
        else:
            daily_returns.append(0)
        equity_curve.append(cash)

    df = df.iloc[:-1].copy()
    df["Equity"] = equity_curve

    # Metrics
    total_return = (cash - initial_cash) / initial_cash * 100
    win_rate = ((df["Prediction"] == 1) & (df["next_close"] > df["Close"])).mean() * 100

    # Buy & Hold baseline
    buy_hold_return = (df.iloc[-1]["Close"] / df.iloc[0]["Close"] - 1) * 100

    # Sharpe ratio (assuming 252 trading days)
    daily_r = np.array(daily_returns)
    sharpe_ratio = (np.mean(daily_r) / np.std(daily_r)) * np.sqrt(252) if np.std(daily_r) > 0 else 0

    # Max drawdown
    rolling_max = np.maximum.accumulate(df["Equity"])
    drawdown = (df["Equity"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100

    metrics = {
        "Final Equity": cash,
        "Total Return %": total_return,
        "Win Rate %": win_rate,
        "Buy & Hold Return %": buy_hold_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown %": max_drawdown
    }

    return df, metrics

if __name__ == "__main__":
    df_results, metrics = backtest("data/final/AAPL_final.csv")
    print(metrics)
    df_results.to_csv("backtest_results.csv", index=False)
