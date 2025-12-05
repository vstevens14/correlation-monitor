import pandas as pd
import numpy as np
from pathlib import Path
from rolling_correlation import load_asset_data, calculate_rolling_correlation, calculate_correlation_statistics, detect_anomalies

def get_all_available_assets():
    """Scan data/raw directory and return all available assets"""
    data_dir = Path('data/raw')
    assets = {}
    
    for file in data_dir.glob('*.csv'):
        asset_name = file.stem.replace('_stock', '').replace('_etf', '').replace('_fx', '').replace('_commodity', '').replace('_rates', '').replace('_economic', '')
        assets[asset_name] = str(file)
    
    return assets

def scan_all_correlations(window=90, lookback=252, threshold=1.5, max_pairs=500):
    """
    Scan all possible asset pairs and find anomalies
    Returns DataFrame with anomalous pairs sorted by z-score
    """
    assets = get_all_available_assets()
    asset_list = list(assets.keys())
    
    print(f"📊 Scanning {len(asset_list)} assets...")
    print(f"   Total possible pairs: {len(asset_list) * (len(asset_list) - 1) // 2}")
    print(f"   Analyzing up to {max_pairs} pairs...")
    
    results = []
    pairs_analyzed = 0
    
    # Analyze pairs
    for i, asset1 in enumerate(asset_list):
        if pairs_analyzed >= max_pairs:
            break
            
        for asset2 in asset_list[i+1:]:
            if pairs_analyzed >= max_pairs:
                break
                
            try:
                # Load data
                data1 = load_asset_data(assets[asset1])
                data2 = load_asset_data(assets[asset2])
                
                # Calculate rolling correlation
                rolling_corr = calculate_rolling_correlation(data1, data2, window=window)
                
                if len(rolling_corr) < window:
                    continue
                
                # Get statistics
                stats = calculate_correlation_statistics(rolling_corr, lookback_period=lookback)
                
                # Check if anomalous
                if abs(stats['z_score']) > threshold:
                    # Determine asset categories
                    def get_category(asset_name):
                        if any(x in assets[asset_name] for x in ['_stock', '_etf']):
                            return 'Equity/ETF'
                        elif '_fx' in assets[asset_name]:
                            return 'FX'
                        elif '_commodity' in assets[asset_name]:
                            return 'Commodity'
                        elif '_rates' in assets[asset_name]:
                            return 'Rates'
                        elif '_economic' in assets[asset_name]:
                            return 'Economic'
                        return 'Other'

                    results.append({
                        'Asset 1': asset1,
                        'Category 1': get_category(asset1),
                        'Asset 2': asset2,
                        'Category 2': get_category(asset2),
                        'Current Correlation': stats['current'],
                        'Historical Mean': stats['mean'],
                        'Z-Score': stats['z_score'],
                        'Std Dev': stats['std'],
                        'Status': 'Anomaly'
                    })
                
                pairs_analyzed += 1
                
                if pairs_analyzed % 50 == 0:
                    print(f"   Analyzed {pairs_analyzed} pairs... Found {len(results)} anomalies")
                    
            except Exception as e:
                continue
    
    print(f"\n✅ Analysis complete!")
    print(f"   Pairs analyzed: {pairs_analyzed}")
    print(f"   Anomalies found: {len(results)}")
    
    # Convert to DataFrame and sort by absolute z-score
    if results:
        df = pd.DataFrame(results)
        df['Abs Z-Score'] = df['Z-Score'].abs()
        df = df.sort_values('Abs Z-Score', ascending=False)
        df = df.drop('Abs Z-Score', axis=1)
        return df
    else:
        return pd.DataFrame()

def get_top_anomalies(n=10, window=90, threshold=1.5):
    """Get top N most anomalous pairs"""
    all_anomalies = scan_all_correlations(window=window, threshold=threshold, max_pairs=500)
    
    if len(all_anomalies) > 0:
        return all_anomalies.head(n)
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    print("="*60)
    print("CROSS-ASSET ANOMALY SCANNER")
    print("="*60)
    
    top_anomalies = get_top_anomalies(n=20, window=90, threshold=1.5)
    
    if len(top_anomalies) > 0:
        print("\n🚨 TOP 20 CORRELATION ANOMALIES:\n")
        print(top_anomalies.to_string(index=False))
        
        # Save to CSV
        top_anomalies.to_csv('docs/top_anomalies.csv', index=False)
        print("\n💾 Saved to docs/top_anomalies.csv")
    else:
        print("\n✅ No significant anomalies detected")