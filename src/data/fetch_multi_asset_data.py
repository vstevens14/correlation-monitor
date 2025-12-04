import yfinance as yf
import pandas as pd

def fetch_fx_data(pair, start_date='2020-01-01', end_date=None):
    """
    Fetch FX data (currency pairs)
    pair: Currency pair ticker (e.g., 'EURUSD=X', 'JPY=X')
    """
    ticker = yf.Ticker(pair)
    data = ticker.history(start=start_date, end=end_date)
    data.columns = data.columns.str.lower()
    return data

def fetch_commodity_data(symbol, start_date='2020-01-01', end_date=None):
    """
    Fetch commodity data
    symbol: Commodity ticker (e.g., 'GC=F' for gold, 'CL=F' for oil)
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    data.columns = data.columns.str.lower()
    return data

def fetch_rates_data(symbol, start_date='2020-01-01', end_date=None):
    """
    Fetch rates/treasury data
    symbol: Rates ticker (e.g., '^TNX' for 10-year, 'TLT' for bond ETF)
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    data.columns = data.columns.str.lower()
    return data

def fetch_index_data(symbol, start_date='2020-01-01', end_date=None):
    """
    Fetch index data
    symbol: Index ticker (e.g., 'SPY', '^VIX', 'DX-Y.NYB' for dollar index)
    """
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    data.columns = data.columns.str.lower()
    return data

if __name__ == "__main__":
    print("Testing multi-asset data fetchers...\n")
    
    print("1. Fetching EUR/USD...")
    eurusd = fetch_fx_data('EURUSD=X', start_date='2020-01-01')
    print(f"   Got {len(eurusd)} days of data")
    print(f"   Latest price: {eurusd['close'].iloc[-1]:.4f}\n")
    
    print("2. Fetching Gold...")
    gold = fetch_commodity_data('GC=F', start_date='2020-01-01')
    print(f"   Got {len(gold)} days of data")
    print(f"   Latest price: ${gold['close'].iloc[-1]:.2f}\n")
    
    print("3. Fetching 10-Year Treasury Yield...")
    tnx = fetch_rates_data('^TNX', start_date='2020-01-01')
    print(f"   Got {len(tnx)} days of data")
    print(f"   Latest yield: {tnx['close'].iloc[-1]:.2f}%\n")
    
    print("4. Fetching S&P 500 (SPY)...")
    spy = fetch_index_data('SPY', start_date='2020-01-01')
    print(f"   Got {len(spy)} days of data")
    print(f"   Latest price: ${spy['close'].iloc[-1]:.2f}")