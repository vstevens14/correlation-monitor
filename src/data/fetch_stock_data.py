import yfinance as yf
import pandas as pd

def fetch_stock_data(symbol, start_date='2020-01-01', end_date=None):
    """
    Fetch stock data for a given symbol using yfinance
    symbol: Stock ticker (e.g., 'AAPL', 'MSFT')
    start_date: Start date in 'YYYY-MM-DD' format
    end_date: End date (defaults to today)
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    # Rename columns to lowercase for consistency
    data.columns = data.columns.str.lower()
    
    return data

if __name__ == "__main__":
    # Test the function
    print("Fetching AAPL stock data from 2020...")
    df = fetch_stock_data('AAPL', start_date='2020-01-01')
    print(f"\nData range: {df.index.min()} to {df.index.max()}")
    print(f"Total days: {len(df)}")
    print("\nFirst few rows:")
    print(df.head())