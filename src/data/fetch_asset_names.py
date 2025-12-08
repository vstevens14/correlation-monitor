import yfinance as yf
import json
from pathlib import Path

def fetch_asset_names():
    """Fetch full names for all assets in data/raw"""
    data_dir = Path('data/raw')
    asset_names = {}
    
    print("Fetching asset names...")
    
    for file in data_dir.glob('*.csv'):
        ticker = file.stem.replace('_stock', '').replace('_etf', '').replace('_commodity', '').replace('_fx', '').replace('_rates', '').replace('_economic', '')
        
        try:
            # Try to get info from yfinance
            if '_stock' in file.stem or '_etf' in file.stem:
                info = yf.Ticker(ticker).info
                name = info.get('longName') or info.get('shortName') or ticker
            elif '_commodity' in file.stem:
                # Commodity mapping
                commodity_map = {
                    'GCF': 'Gold Futures',
                    'SIF': 'Silver Futures',
                    'CLF': 'Crude Oil Futures',
                    'NGF': 'Natural Gas Futures',
                    'HGF': 'Copper Futures',
                    'ZCF': 'Corn Futures',
                    'ZWF': 'Wheat Futures'
                }
                name = commodity_map.get(ticker, ticker)
            elif '_fx' in file.stem:
                # FX mapping
                fx_map = {
                    'EURUSDX': 'EUR/USD',
                    'JPYX': 'USD/JPY',
                    'GBPUSDX': 'GBP/USD',
                    'AUDUSDX': 'AUD/USD',
                    'USDCADX': 'USD/CAD',
                    'USDCHFX': 'USD/CHF',
                    'NZDUSDX': 'NZD/USD',
                    'DX-Y_NYB': 'US Dollar Index'
                }
                name = fx_map.get(ticker, ticker)
            elif '_rates' in file.stem:
                # Rates mapping
                rates_map = {
                    'TNX': '10-Year Treasury Yield',
                    'FVX': '5-Year Treasury Yield',
                    'IRX': '13-Week Treasury Bill',
                    'TLT': '20+ Year Treasury Bond ETF',
                    'IEF': '7-10 Year Treasury ETF',
                    'SHY': '1-3 Year Treasury ETF'
                }
                name = rates_map.get(ticker, ticker)
            elif '_economic' in file.stem:
                # Economic indicators mapping
                econ_map = {
                    'UNRATE': 'Unemployment Rate',
                    'GDP': 'Gross Domestic Product',
                    'CPIAUCSL': 'Consumer Price Index',
                    'FEDFUNDS': 'Federal Funds Rate',
                    'DGS10': '10-Year Treasury Rate',
                    'DGS2': '2-Year Treasury Rate',
                    'T10Y2Y': '10Y-2Y Treasury Spread'
                }
                name = econ_map.get(ticker, ticker)
            else:
                name = ticker
            
            asset_names[ticker] = name
            print(f"  {ticker}: {name}")
            
        except Exception as e:
            print(f"  {ticker}: Could not fetch name - {e}")
            asset_names[ticker] = ticker
    
    # Save to JSON
    output_file = 'src/data/asset_names.json'
    with open(output_file, 'w') as f:
        json.dump(asset_names, f, indent=2)
    
    print(f"\n✅ Saved {len(asset_names)} asset names to {output_file}")
    return asset_names

if __name__ == "__main__":
    fetch_asset_names()