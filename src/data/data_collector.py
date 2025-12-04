import os
import pandas as pd
from fetch_stock_data import fetch_stock_data
from fetch_economic_data import fetch_fred_data
from datetime import datetime

def save_stock_data(symbol, start_date='2020-01-01', filepath=None):
    """Fetch and save stock data to CSV"""
    if filepath is None:
        filepath = f"data/raw/{symbol}_stock_data.csv"
    
    data = fetch_stock_data(symbol, start_date=start_date)
    data.to_csv(filepath)
    print(f"Saved {symbol} data to {filepath} ({len(data)} days)")
    return data

def save_economic_data(series_id, name, start_date='2020-01-01', filepath=None):
    """Fetch and save economic data to CSV"""
    if filepath is None:
        filepath = f"data/raw/{name}_economic_data.csv"
    
    data = fetch_fred_data(series_id, start_date=start_date)
    data.to_csv(filepath)
    print(f"Saved {name} data to {filepath}")
    return data

if __name__ == "__main__":
    # Collect stock data
    print("Collecting stock data...")
    save_stock_data('AAPL', start_date='2020-01-01')
    save_stock_data('MSFT', start_date='2020-01-01')
    save_stock_data('GOOGL', start_date='2020-01-01')
    
    print("\nCollecting economic data...")
    # Unemployment rate
    save_economic_data('UNRATE', 'unemployment_rate', start_date='2020-01-01')
    # GDP (quarterly data)
    save_economic_data('GDP', 'gdp', start_date='2020-01-01')
    # Consumer Price Index (Inflation)
    save_economic_data('CPIAUCSL', 'cpi', start_date='2020-01-01')
    # Federal Funds Rate
    save_economic_data('FEDFUNDS', 'fed_funds_rate', start_date='2020-01-01')
    
    print("\nData collection complete!")