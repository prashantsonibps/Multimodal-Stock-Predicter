import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def fetch_finviz_news(ticker):
    url = f'https://finviz.com/quote.ashx?t={ticker}'
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')

    news_table = soup.find('table', class_='fullview-news-outer')
    rows = news_table.findAll('tr') if news_table else []

    news_data = []
    for row in rows:
        cols = row.findAll('td')
        if len(cols) == 2:
            timestamp = cols[0].text.strip()
            headline = cols[1].text.strip()
            news_data.append([timestamp, headline])

    return pd.DataFrame(news_data, columns=['timestamp', 'headline'])

def save_news(ticker, df, save_dir='data/news'):
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f'{ticker}_news.csv')
    df.to_csv(file_path, index=False)
    print(f"[✓] Saved news for {ticker} to {file_path}")

def main():
    tickers = ['TSLA', 'MSFT', 'AAPL']
    for ticker in tickers:
        df = fetch_finviz_news(ticker)
        save_news(ticker, df)

if __name__ == "__main__":
    main()
