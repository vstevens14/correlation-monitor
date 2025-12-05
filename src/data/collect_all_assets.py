import pandas as pd
from fetch_stock_data import fetch_stock_data
from fetch_multi_asset_data import fetch_fx_data, fetch_commodity_data, fetch_rates_data, fetch_index_data
from fetch_economic_data import fetch_fred_data
import os
import time

def save_data(data, filepath, asset_name):
    """Save data to CSV with info message"""
    data.to_csv(filepath)
    print(f"✓ Saved {asset_name}: {len(data)} days ({data.index.min().date()} to {data.index.max().date()})")

def get_sp500_tickers():
    """Fetch current S&P 500 tickers from Wikipedia"""
    import urllib.request
    
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    # Add headers to avoid 403 error
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response:
        tables = pd.read_html(response.read())
    
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()
    
    # Clean up tickers (some have dots that need conversion)
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    
    return tickers

def collect_all_assets(start_date='2020-01-01', num_stocks=50):
    """
    Collect data for all asset classes
    num_stocks: how many S&P 500 stocks to fetch (default 50, max 505)
    """
    
    print("="*60)
    print("COLLECTING CROSS-ASSET DATA")
    print("="*60)
    
    # Get S&P 500 tickers dynamically
    print("\n📊 Fetching S&P 500 list...")
    sp500_tickers = get_sp500_tickers()
    print(f"Found {len(sp500_tickers)} S&P 500 companies")
    
    # Select top N stocks (or all if num_stocks > 505)
    selected_stocks = sp500_tickers[:min(num_stocks, len(sp500_tickers))]
    
    # EQUITIES
    print(f"\n📈 EQUITIES (Fetching {len(selected_stocks)} stocks):")
    
    for i, ticker in enumerate(selected_stocks, 1):
        try:
            data = fetch_stock_data(ticker, start_date=start_date)
            save_data(data, f'data/raw/{ticker}_stock.csv', f'{ticker} ({i}/{len(selected_stocks)})')
            time.sleep(0.5)  # Reduced delay since we're fetching many
        except Exception as e:
            print(f"✗ Failed to fetch {ticker}: {e}")
    
    # Add major ETFs
    print("\n📈 MAJOR ETFs:")
    etfs = {
        'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100', 'DIA': 'Dow Jones',
        'IWM': 'Russell 2000', 'VTI': 'Total Market',
        'XLF': 'Financials', 'XLE': 'Energy', 'XLK': 'Technology',
        'XLV': 'Healthcare', 'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples', 'XLI': 'Industrials', 'XLU': 'Utilities'
    }
    
    for ticker, name in etfs.items():
        try:
            data = fetch_stock_data(ticker, start_date=start_date)
            save_data(data, f'data/raw/{ticker}_etf.csv', name)
            time.sleep(0.5)
        except Exception as e:
            print(f"✗ Failed to fetch {name}: {e}")
    
    # FX (FOREIGN EXCHANGE)
    print("\n💱 FOREIGN EXCHANGE:")
    fx_pairs = {
        'EURUSD=X': 'EUR/USD',
        'JPY=X': 'USD/JPY',
        'GBPUSD=X': 'GBP/USD',
        'AUDUSD=X': 'AUD/USD',
        'USDCAD=X': 'USD/CAD',
        'USDCHF=X': 'USD/CHF',
        'NZDUSD=X': 'NZD/USD',
        'DX-Y.NYB': 'US Dollar Index'
    }
    
    for ticker, name in fx_pairs.items():
        try:
            data = fetch_fx_data(ticker, start_date=start_date)
            save_data(data, f'data/raw/{ticker.replace("=", "").replace(".", "_")}_fx.csv', name)
            time.sleep(1)
        except Exception as e:
            print(f"✗ Failed to fetch {name}: {e}")
    
    # COMMODITIES
    print("\n🏆 COMMODITIES:")
    commodities = {
        'GC=F': 'Gold Futures',
        'SI=F': 'Silver Futures',
        'CL=F': 'Crude Oil Futures',
        'NG=F': 'Natural Gas Futures',
        'HG=F': 'Copper Futures',
        'ZC=F': 'Corn Futures',
        'ZW=F': 'Wheat Futures'
    }
    
    for ticker, name in commodities.items():
        try:
            data = fetch_commodity_data(ticker, start_date=start_date)
            save_data(data, f'data/raw/{ticker.replace("=", "")}_commodity.csv', name)
            time.sleep(1)
        except Exception as e:
            print(f"✗ Failed to fetch {name}: {e}")
    
    # RATES
    print("\n📊 RATES & TREASURIES:")
    rates = {
        '^TNX': '10-Year Treasury Yield',
        '^FVX': '5-Year Treasury Yield',
        '^IRX': '13-Week Treasury Bill',
        'TLT': '20+ Year Treasury Bond ETF',
        'IEF': '7-10 Year Treasury ETF',
        'SHY': '1-3 Year Treasury ETF'
    }
    
    for ticker, name in rates.items():
        try:
            data = fetch_rates_data(ticker, start_date=start_date)
            save_data(data, f'data/raw/{ticker.replace("^", "")}_rates.csv', name)
            time.sleep(1)
        except Exception as e:
            print(f"✗ Failed to fetch {name}: {e}")
    
    # ECONOMIC INDICATORS
    print("\n📉 ECONOMIC INDICATORS:")
    indicators = {
        'UNRATE': 'Unemployment Rate',
        'GDP': 'GDP',
        'CPIAUCSL': 'Consumer Price Index',
        'FEDFUNDS': 'Federal Funds Rate',
        'DGS10': '10-Year Treasury Rate',
        'DGS2': '2-Year Treasury Rate',
        'T10Y2Y': '10Y-2Y Treasury Spread'
    }
    
    for series_id, name in indicators.items():
        try:
            data = fetch_fred_data(series_id, start_date=start_date)
            data.to_csv(f'data/raw/{series_id}_economic.csv')
            print(f"✓ Saved {name}")
            time.sleep(1)
        except Exception as e:
            print(f"✗ Failed to fetch {name}: {e}")
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    # Fetch 100 S&P 500 stocks (adjust as needed - max ~505)
    collect_all_assets(start_date='2020-01-01', num_stocks=100)