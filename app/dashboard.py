import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import yfinance as yf
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from backend.model.train_model import FusionMLP
from backend.model.backtest import backtest  # import the improved backtest

# --- Constants ---
FEATURE_ORDER = [
    "High", "Low", "Open", "Volume", "SMA_20", "SMA_50",
    "RSI", "MACD", "MACD_Signal", "avg_sentiment"
]

# --- Load model + scaler ---
scaler = joblib.load("models/scaler.pkl")
model = FusionMLP(input_dim=len(FEATURE_ORDER))
model.load_state_dict(torch.load("models/fusion_mlp.pt"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

st.set_page_config(page_title="📈 Stock Predictor with Backtesting", layout="centered")
st.title("📈 Multimodal Stock Predictor")
tabs = st.tabs(["🔮 Prediction", "📊 Backtesting"])

# ------------------------- #
#  🔮 PREDICTION TAB
# ------------------------- #
with tabs[0]:
    ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):", value="AAPL").upper()

    if st.button("Predict Movement"):
        try:
            df = yf.download(ticker, period="90d", interval="1d")
            df.dropna(inplace=True)
            df["SMA_20"] = df["Close"].rolling(20).mean()
            df["SMA_50"] = df["Close"].rolling(50).mean()

            delta = df["Close"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df["RSI"] = 100 - (100 / (1 + rs))

            df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
            df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

            latest = df.iloc[-1]

            # Fetch headlines
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            page = requests.get(url, headers=headers)
            soup = BeautifulSoup(page.content, "html.parser")
            news_table = soup.find("table", class_="fullview-news-outer")
            headlines = [row.findAll("td")[1].text.strip() for row in news_table.findAll("tr")[:5]]

            # FinBERT sentiment
            def get_sentiment_score(texts):
                scores = []
                for text in texts:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True)
                    with torch.no_grad():
                        outputs = finbert(**inputs)
                        probs = F.softmax(outputs.logits, dim=-1)
                        scores.append(probs[0][2].item())
                return round(np.mean(scores), 4)

            avg_sentiment = get_sentiment_score(headlines)

            # Prepare input
            input_data = {
                "High": float(latest["High"]),
                "Low": float(latest["Low"]),
                "Open": float(latest["Open"]),
                "Volume": float(latest["Volume"]),
                "SMA_20": float(latest["SMA_20"]),
                "SMA_50": float(latest["SMA_50"]),
                "RSI": float(latest["RSI"]),
                "MACD": float(latest["MACD"]),
                "MACD_Signal": float(latest["MACD_Signal"]),
                "avg_sentiment": float(avg_sentiment)
            }

            input_df = pd.DataFrame([input_data])[FEATURE_ORDER]
            input_scaled = scaler.transform(input_df)
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

            with torch.no_grad():
                pred = model(input_tensor)
                score = pred.squeeze().item()
                label = "📈 UP" if score > 0.5 else "📉 DOWN"

            st.success(f"Prediction for **{ticker}**: **{label}**")
            st.caption(f"Model confidence: `{score:.4f}`")
            st.caption(f"Avg. sentiment score: `{avg_sentiment:.4f}`")

            # Chart
            st.markdown("### 📊 Recent Price Trend")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df.tail(60).index, df.tail(60)["Close"], label="Close Price", color="skyblue")
            ax.plot(df.tail(60).index, df.tail(60)["SMA_20"], label="SMA 20", color="green", linestyle="--")
            ax.plot(df.tail(60).index, df.tail(60)["SMA_50"], label="SMA 50", color="red", linestyle="--")
            ax.axvline(x=df.tail(1).index[0], color="orange", linestyle="--", label="Prediction Point")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {str(e)}")

# ------------------------- #
#  📊 BACKTEST TAB
# ------------------------- #
with tabs[1]:
    st.subheader("📊 Backtest Model Strategy")

    ticker_bt = st.text_input("Enter ticker for backtest:", value="AAPL").upper()
    start_date = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("today"))

    if st.button("Run Backtest"):
        try:
            df_bt, metrics = backtest(f"data/final/{ticker_bt}_final.csv")

            # ✅ Filter by date range
            df_bt["Date"] = pd.to_datetime(df_bt["Date"])
            df_bt = df_bt[(df_bt["Date"] >= pd.to_datetime(start_date)) & (df_bt["Date"] <= pd.to_datetime(end_date))]

            if df_bt.empty:
                st.warning("⚠️ No data available for this date range.")
            else:
                # Re-run strategy on filtered data
                df_bt = df_bt.reset_index(drop=True)
                cash = 10000
                equity_curve = []

                for i in range(len(df_bt)-1):
                    if df_bt.loc[i, "Prediction"] == 1:
                        cash *= df_bt.loc[i+1, "Close"] / df_bt.loc[i, "Close"]
                    equity_curve.append(cash)

                df_bt = df_bt.iloc[:-1].copy()
                df_bt["Equity"] = equity_curve

                total_return = (cash - 10000) / 10000 * 100
                win_rate = ((df_bt["Prediction"] == 1) & (df_bt["next_close"] > df_bt["Close"])).mean() * 100

                st.success(f"💰 Final Equity: ${cash:,.2f}")
                st.info(f"📈 Total Return: {total_return:.2f}% | Win Rate: {win_rate:.2f}%")

                # Interactive chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df_bt["Date"], y=df_bt["Equity"],
                    mode="lines", name="Strategy Equity", line=dict(color="purple")
                ))
                fig.update_layout(
                    title=f"Equity Curve - {ticker_bt} ({start_date} to {end_date})",
                    xaxis_title="Date", yaxis_title="Equity ($)",
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Backtest failed: {str(e)}")
