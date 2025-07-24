import pandas as pd
import os

def load_price_data(ticker):
    df = pd.read_csv(f"data/price/{ticker}_price.csv", index_col=0, parse_dates=True)
    df.reset_index(inplace=True)
    
    # Rename first column to "Date" if needed
    if df.columns[0] != "Date":
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    return df


def load_sentiment_data(ticker):
    df = pd.read_csv(f"data/news_sentiment/{ticker}_news_sentiment.csv")
    # FinViz does not give date, so we approximate from timestamp
    df['Date'] = pd.to_datetime('today').normalize()  # today for now (MVP)
    return df

def create_next_day_label(df):
    df['next_close'] = df['Close'].shift(-1)
    df['target'] = (df['next_close'] > df['Close']).astype(int)
    return df

def build_dataset(ticker):
    print(f"[...] Processing {ticker}")
    
    price_df = load_price_data(ticker)
    news_df = load_sentiment_data(ticker)

    # MVP: Approximate all news to today and average positivity score
    avg_positivity = news_df['positivity_score'].mean()
    price_df['avg_sentiment'] = avg_positivity

    price_df = create_next_day_label(price_df)
    price_df.dropna(inplace=True)

    out_path = f"data/final/{ticker}_final.csv"
    os.makedirs("data/final", exist_ok=True)
    price_df.to_csv(out_path, index=False)
    print(f"[✓] Final dataset saved ➝ {out_path}")

def main():
    tickers = ['TSLA', 'MSFT', 'AAPL']
    for ticker in tickers:
        build_dataset(ticker)

if __name__ == "__main__":
    main()
