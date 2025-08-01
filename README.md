# 📈 Multimodal Stock Prediction using Technical Indicators & News Sentiment
A deep learning project that predicts **next-day stock movement (UP/DOWN)** by **fusing technical indicators** (SMA, RSI, MACD, Volume) with **news sentiment scores** extracted using **FinBERT**. The project integrates **quantitative** and **qualitative** signals to achieve improved forecasting accuracy compared to traditional models.

## 🚀 Features
- Collects historical stock data using **Yahoo Finance API**  
- Scrapes financial news headlines using **MarketAux / NewsCatcher API**  
- Extracts sentiment scores using **FinBERT (BERT-based sentiment model)**  
- Implements a **dual-branch multimodal neural network** to fuse technical & sentiment features  
- Benchmarks against **Random Forest**, **LSTM**, and **FinBERT-only models**  
- Provides **precision, recall, and confusion matrix** for evaluation  
- Includes **trained model (.pt)** and **backtesting script**  

## 🏗 Project Architecture
### 🔹 Fusion Model Structure


## 📊 Dataset
- **Stocks:** AAPL, TSLA, MSFT  
- **Technical Indicators:** SMA-20, SMA-50, RSI (>70 = overbought, <30 = oversold), MACD, Volume  
- **Sentiment Data:** News headlines scraped daily, scored with FinBERT (range [-1, 1])  

## ⚙️ Preprocessing
- Numerical features normalized to [0, 1]  
- Sentiment scores aggregated per day per stock  
- Missing values forward-filled for weekends/holidays  
- Data aligned by timestamp to match price and sentiment  

## 🧠 Model Details
- **Technical Branch:** 3-layer MLP (64, 32, 16 units, ReLU)  
- **Sentiment Branch:** 2-layer Dense (32, 16 units, ReLU)  
- **Fusion:** Concatenated output → Dense (64, 32) → Sigmoid  
- **Optimizer:** Adam (lr = 1e-3), **Loss:** Binary Cross-Entropy  
- **Training:** 30 epochs, early stopping, batch size 32  
- **Baselines:** Random Forest, LSTM, FinBERT-only  

## 📈 Results
| Model           | AAPL Accuracy | TSLA Accuracy | MSFT Accuracy | Avg Accuracy |
|-----------------|---------------|---------------|---------------|--------------|
| Random Forest   | 50.2%         | 51.8%         | 51.0%         | 51.0%        |
| LSTM            | 52.5%         | 53.4%         | 53.1%         | 53.0%        |
| FinBERT-only    | 54.8%         | 54.2%         | 54.7%         | 54.6%        |
| **Multimodal**  | **56.9%**     | **56.3%**     | **57.1%**     | **56.8%**    |

✅ The multimodal model **outperforms all baselines** by effectively combining technical indicators with sentiment data.

## 🔮 Future Improvements
- Expand dataset to include 2017–2023 and more stocks  
- Implement **Temporal Fusion Transformers** for multi-horizon forecasting  
- Integrate **Reddit/Twitter sentiment**  
- Add **Bollinger Bands, Stochastic Oscillator, SHAP explainability**  
- Deploy as a **real-time dashboard with backtesting**  


## 🛠 How to Run Locally
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/prashantsonibps/Multimodal-Stock-Predicter.git
cd Multimodal-Stock-Predicter
