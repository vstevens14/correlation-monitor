import pandas as pd
import numpy as np
from rolling_correlation import load_asset_data, calculate_rolling_correlation

def load_multiple_assets(asset_dict):
    """
    Load multiple assets into a single DataFrame
    asset_dict: dict with asset names as keys and filepaths as values
    """
    data = {}
    
    for name, filepath in asset_dict.items():
        try:
            data[name] = load_asset_data(filepath)
            print(f"  Loaded {name}: {len(data[name])} days")
        except Exception as e:
            print(f"  Warning: Could not load {name}: {e}")
    
    # Combine into single DataFrame with outer join (keep all dates)
    df = pd.DataFrame(data)
    
    # Only drop rows where ALL values are NaN
    df = df.dropna(how='all')
    
    # Then forward fill missing values (common for daily financial data)
    df = df.fillna(method='ffill').dropna()
    
    return df

def calculate_correlation_matrix(df, method='pearson'):
    """Calculate correlation matrix for all assets"""
    return df.corr(method=method)

def calculate_rolling_correlation_matrix(df, window=90):
    """Calculate rolling correlation matrix"""
    rolling_corrs = {}
    
    assets = df.columns.tolist()
    
    for i, asset1 in enumerate(assets):
        for asset2 in assets[i+1:]:  # Only upper triangle
            pair_name = f"{asset1}_vs_{asset2}"
            rolling_corrs[pair_name] = df[asset1].rolling(window=window).corr(df[asset2])
    
    return pd.DataFrame(rolling_corrs)

def detect_matrix_anomalies(current_matrix, historical_matrix, threshold=1.5):
    """
    Detect which asset pairs have anomalous correlations
    Returns dict of anomalous pairs with their z-scores
    """
    anomalies = {}
    
    for i in range(len(current_matrix)):
        for j in range(i+1, len(current_matrix)):
            asset1 = current_matrix.index[i]
            asset2 = current_matrix.columns[j]
            
            current_corr = current_matrix.iloc[i, j]
            hist_mean = historical_matrix.iloc[i, j]
            
            # Calculate std from rolling correlations (simplified)
            # In production, you'd track this properly
            hist_std = 0.3  # Placeholder - would calculate from rolling data
            
            z_score = (current_corr - hist_mean) / hist_std
            
            if abs(z_score) > threshold:
                anomalies[f"{asset1}_vs_{asset2}"] = {
                    'current': current_corr,
                    'historical_mean': hist_mean,
                    'z_score': z_score
                }
    
    return anomalies

def plot_correlation_matrix(corr_matrix, title="Asset Correlation Matrix", 
                           save_path=None, annot=True):
    """Plot correlation matrix as heatmap"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Plot heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=annot, fmt='.2f', 
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()

def plot_correlation_comparison(current_matrix, historical_matrix, 
                               save_path=None):
    """Plot current vs historical correlation matrices side by side"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(current_matrix, dtype=bool), k=1)
    
    # Current correlations
    sns.heatmap(current_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax1)
    ax1.set_title('Current Correlations (90-Day)', fontsize=14, fontweight='bold')
    
    # Historical correlations
    sns.heatmap(historical_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax2)
    ax2.set_title('Historical Avg (1-Year)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Loading cross-asset data...")
    print("="*60)
    
    # Define asset universe
    assets = {
        'SPY': 'data/raw/SPY_stock.csv',
        'Gold': 'data/raw/GCF_commodity.csv',
        'Oil': 'data/raw/CLF_commodity.csv',
        'USD_Index': 'data/raw/DX-Y_NYB_fx.csv',
        '10Y_Yield': 'data/raw/TNX_rates.csv',
        'EUR/USD': 'data/raw/EURUSDX_fx.csv'
    }
    
    # Load all assets
    df = load_multiple_assets(assets)

    # DEBUG: Check what we actually loaded
    print("\nDEBUG INFO:")
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame index type: {type(df.index)}")
    print(f"First few index values: {df.index[:5].tolist() if len(df) > 0 else 'Empty'}")
    print(f"DataFrame head:")
    print(df.head())
    
    print(f"\nLoaded {len(df.columns)} assets")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Trading days: {len(df)}")
    
    # Calculate current correlation (last 90 days)
    print("\nCalculating current correlations (90-day)...")
    current_data = df.tail(90)
    current_corr = calculate_correlation_matrix(current_data)
    
    # Calculate historical correlation (1 year)
    print("Calculating historical correlations (1-year)...")
    historical_data = df.tail(252)  # ~1 trading year
    historical_corr = calculate_correlation_matrix(historical_data)
    
    # Detect anomalies
    print("\nDetecting correlation anomalies...")
    anomalies = detect_matrix_anomalies(current_corr, historical_corr, threshold=1.5)
    
    if anomalies:
        print(f"\n⚠️  Found {len(anomalies)} anomalous correlations:")
        for pair, data in anomalies.items():
            print(f"\n  {pair}:")
            print(f"    Current: {data['current']:.3f}")
            print(f"    Historical: {data['historical_mean']:.3f}")
            print(f"    Z-Score: {data['z_score']:.2f}")
    else:
        print("\n✓ No significant correlation anomalies detected")
    
    print("\n" + "="*60)
    print("\nCreating visualizations...")
    
    # Plot current correlation matrix
    plot_correlation_matrix(current_corr, 
                           title="Current Cross-Asset Correlations (90-Day Window)",
                           save_path='docs/current_correlation_matrix.png')
    
    # Plot comparison
    plot_correlation_comparison(current_corr, historical_corr,
                               save_path='docs/correlation_comparison.png')
    
    print("\nAnalysis complete!")