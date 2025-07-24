import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load pretrained FinBERT model
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

labels_map = {0: "negative", 1: "neutral", 2: "positive"}

def get_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        label_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label_id].item()
    return labels_map[label_id], round(confidence, 4), round(probs[0][2].item(), 4)  # (label, confidence, positivity_score)

def process_news_file(ticker, input_dir="data/news", output_dir="data/news_sentiment"):
    os.makedirs(output_dir, exist_ok=True)
    input_file = os.path.join(input_dir, f"{ticker}_news.csv")
    print(f"→ Reading: {input_file}")
    
    try:
        df = pd.read_csv(input_file)
        print(f"→ Found {len(df)} rows for {ticker}")
        print(f"→ Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"[ERROR] Failed to load {ticker}: {e}")
        return

    if 'headline' not in df.columns:
        print(f"[!] No 'headline' column found in {ticker}_news.csv — aborting")
        return

    sentiments = df['headline'].apply(get_sentiment)
    df[['label', 'confidence', 'positivity_score']] = pd.DataFrame(sentiments.tolist(), index=df.index)

    out_file = os.path.join(output_dir, f"{ticker}_news_sentiment.csv")
    df.to_csv(out_file, index=False)
    print(f"[✓] Saved sentiment results for {ticker} ➝ {out_file}")


def main():
    tickers = ['TSLA', 'MSFT', 'AAPL']
    for ticker in tickers:
        process_news_file(ticker)

if __name__ == "__main__":
    main()
