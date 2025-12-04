import pandas as pd
import numpy as np

def load_stock_data(filepath):
    """Load stock data from CSV"""
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    return df

def calculate_returns(df, column='close'):
    """Calculate daily returns"""
    df['daily_return'] = df[column].pct_change()
    df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
    return df

def add_moving_averages(df, windows=[20, 50], column='close'):
    """Add moving averages"""
    for window in windows:
        df[f'MA_{window}'] = df[column].rolling(window=window).mean()
    return df

if __name__ == "__main__":
    # Test the processor
    print("Loading and processing AAPL data...")
    df = load_stock_data('data/raw/AAPL_stock_data.csv')
    df = calculate_returns(df)
    df = add_moving_averages(df)
    
    print("\nProcessed data preview:")
    print(df.tail())
    
    # Save processed data
    df.to_csv('data/processed/AAPL_processed.csv')
    print("\nSaved processed data to data/processed/")