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

def collect_all_assets(start_date='2020-01-01'):
    """Collect data for all asset classes"""
    
    print("="*60)
    print("COLLECTING CROSS-ASSET DATA")
    print("="*60)
    
    # EQUITIES
    print("\n📈 EQUITIES:")
    equities = {
        'AAPL': 'Apple',
        'MSFT': 'Microsoft', 
        'GOOGL': 'Google',
        'SPY': 'S&P 500 ETF',
        'XLF': 'Financial Sector ETF'
    }
    
    for ticker, name in equities.items():
        data = fetch_stock_data(ticker, start_date=start_date)
        save_data(data, f'data/raw/{ticker}_stock.csv', name)
        time.sleep(1)  # Rate limiting
    
    # FX (FOREIGN EXCHANGE)
    print("\n💱 FOREIGN EXCHANGE:")
    fx_pairs = {
        'EURUSD=X': 'EUR/USD',
        'JPY=X': 'USD/JPY',
        'GBPUSD=X': 'GBP/USD',
        'DX-Y.NYB': 'US Dollar Index'
    }
    
    for ticker, name in fx_pairs.items():
        try:
            data = fetch_fx_data(ticker, start_date=start_date)
            save_data(data, f'data/raw/{ticker.replace("=", "").replace(".", "_")}_fx.csv', name)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"✗ Failed to fetch {name}: {e}")
    
    # COMMODITIES
    print("\n🏆 COMMODITIES:")
    commodities = {
        'GC=F': 'Gold Futures',
        'CL=F': 'Crude Oil Futures',
        'HG=F': 'Copper Futures'
    }
    
    for ticker, name in commodities.items():
        try:
            data = fetch_commodity_data(ticker, start_date=start_date)
            save_data(data, f'data/raw/{ticker.replace("=", "")}_commodity.csv', name)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"✗ Failed to fetch {name}: {e}")
    
    # RATES
    print("\n📊 RATES & TREASURIES:")
    rates = {
        '^TNX': '10-Year Treasury Yield',
        '^IRX': '13-Week Treasury Bill',
        'TLT': '20+ Year Treasury Bond ETF'
    }
    
    for ticker, name in rates.items():
        try:
            data = fetch_rates_data(ticker, start_date=start_date)
            save_data(data, f'data/raw/{ticker.replace("^", "")}_rates.csv', name)
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"✗ Failed to fetch {name}: {e}")
    
    # ECONOMIC INDICATORS
    print("\n📉 ECONOMIC INDICATORS:")
    indicators = {
        'UNRATE': 'Unemployment Rate',
        'GDP': 'GDP',
        'CPIAUCSL': 'Consumer Price Index',
        'FEDFUNDS': 'Federal Funds Rate'
    }
    
    for series_id, name in indicators.items():
        try:
            data = fetch_fred_data(series_id, start_date=start_date)
            data.to_csv(f'data/raw/{series_id}_economic.csv')
            print(f"✓ Saved {name}")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"✗ Failed to fetch {name}: {e}")
    
    print("\n" + "="*60)
    print("DATA COLLECTION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    collect_all_assets(start_date='2020-01-01')