import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import joblib
import yfinance as yf
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from backend.model.train_model import FusionMLP

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

# --- Load FinBERT ---
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
finbert = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# --- Streamlit UI ---
st.set_page_config(page_title="📈 Stock Movement Predictor", layout="centered")
st.title("📈 Multimodal Stock Predictor")
st.subheader("Enter a stock ticker to get next-day prediction using indicators + news sentiment")

ticker = st.text_input("Enter stock ticker (e.g., AAPL, TSLA):", value="AAPL").upper()

if st.button("🔍 Predict Movement"):
    try:
        # --- Fetch OHLCV Data ---
        df = yf.download(ticker, period="90d", interval="1d")
        df.dropna(inplace=True)
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
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

        # --- Fetch News from FinViz ---
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        page = requests.get(url, headers=headers)
        soup = BeautifulSoup(page.content, "html.parser")
        news_table = soup.find("table", class_="fullview-news-outer")
        headlines = [row.findAll("td")[1].text.strip() for row in news_table.findAll("tr")[:5]]

        # --- Run FinBERT on headlines ---
        def get_sentiment_score(texts):
            scores = []
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = finbert(**inputs)
                    probs = F.softmax(outputs.logits, dim=-1)
                    scores.append(probs[0][2].item())  # Positive score
            return round(np.mean(scores), 4)

        avg_sentiment = float(get_sentiment_score(headlines))

        # --- Assemble input row ---
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

        # --- Predict ---
        with torch.no_grad():
            pred = model(input_tensor)
            score = pred.squeeze().item()
            label = "📈 UP" if score > 0.5 else "📉 DOWN"

        # --- Display ---
        st.success(f"Prediction for **{ticker}**: **{label}**")
        st.caption(f"Model confidence: `{score:.4f}`")
        st.caption(f"Avg. sentiment score (FinBERT): `{avg_sentiment:.4f}`")

        # --- Price Chart ---
        st.markdown("### 📊 Recent Price Trend")
        plot_df = df.tail(60).copy()
        plot_df["Date"] = plot_df.index

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_df["Date"], plot_df["Close"], label="Close Price", color="skyblue")
        ax.axvline(x=plot_df["Date"].iloc[-1], color="orange", linestyle="--", label="Prediction Point")
        ax.set_title(f"{ticker} Closing Price - Last 60 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # --- Headlines ---
        st.markdown("### 📰 Recent Headlines Used")
        for h in headlines:
            st.markdown(f"- {h}")



        # --- Price Chart with SMA Overlay ---
        st.markdown("### 📊 Recent Price Trend with SMA")

        plot_df = df.tail(60).copy()
        plot_df["Date"] = plot_df.index

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(plot_df["Date"], plot_df["Close"], label="Close Price", color="skyblue")
        ax.plot(plot_df["Date"], plot_df["SMA_20"], label="SMA 20", color="green", linestyle="--")
        ax.plot(plot_df["Date"], plot_df["SMA_50"], label="SMA 50", color="red", linestyle="--")
        ax.axvline(x=plot_df["Date"].iloc[-1], color="orange", linestyle="--", label="Prediction Point")
        ax.set_title(f"{ticker} Closing Price - Last 60 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # --- Confidence Bar ---
        st.markdown("### 🧠 Model Confidence")

        bar_fig, bar_ax = plt.subplots(figsize=(5, 1.2))
        bar_ax.barh(["Confidence"], [score], color="green" if score > 0.5 else "red")
        bar_ax.set_xlim(0, 1)
        bar_ax.set_xlabel("Probability of UP")
        bar_ax.set_yticks([])
        st.pyplot(bar_fig)


    except Exception as e:
        st.error(f"❌ Failed to fetch or predict for {ticker}: {str(e)}")
