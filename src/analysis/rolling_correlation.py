import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_asset_data(filepath, return_full=False):
    """
    Load asset price data from CSV
    
    Args:
        filepath: Path to CSV file
        return_full: If True, return full DataFrame with volume, etc.
                    If False, return only close prices (default for backward compatibility)
    
    Returns:
        Series of close prices or full DataFrame
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    
    # Handle timezone - check if it's a DatetimeIndex first
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Standardize column names
    df.columns = df.columns.str.lower()
    
    if return_full:
        return df
    else:
        # Return close prices only (backward compatible)
        if 'close' in df.columns:
            return df['close']
        elif 'value' in df.columns:  # For economic data
            return df['value']
        else:
            return df.iloc[:, 0]  # Return first column as fallback
        
def calculate_rolling_correlation(asset1_data, asset2_data, window=90):
    """
    Calculate rolling correlation between two assets
    
    Args:
        asset1_data: Price series for asset 1
        asset2_data: Price series for asset 2
        window: Rolling window size in days
    
    Returns:
        Series of rolling correlations
    """
    # Ensure data is aligned
    df = pd.DataFrame({
        'asset1': asset1_data,
        'asset2': asset2_data
    })
    
    # Remove timezone if present
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    # Drop NaN values
    df = df.dropna()
    
    # Calculate rolling correlation
    rolling_corr = df['asset1'].rolling(window=window).corr(df['asset2'])
    
    return rolling_corr

def calculate_correlation_statistics(rolling_corr, lookback_period=252):
    """
    Calculate statistics for rolling correlation
    
    Args:
        rolling_corr: Series of rolling correlations
        lookback_period: Period for calculating baseline statistics (days)
    
    Returns:
        Dictionary with correlation statistics
    """
    # Use the lookback period for baseline
    baseline = rolling_corr.iloc[-lookback_period:] if len(rolling_corr) > lookback_period else rolling_corr
    
    stats = {
        'current': rolling_corr.iloc[-1],
        'mean': baseline.mean(),
        'std': baseline.std(),
        'min': baseline.min(),
        'max': baseline.max(),
        'z_score': (rolling_corr.iloc[-1] - baseline.mean()) / baseline.std() if baseline.std() > 0 else 0
    }
    
    return stats

def detect_anomalies(rolling_corr, threshold=1.5, lookback_period=None):
    """
    Detect correlation anomalies using z-score
    threshold: number of standard deviations (default 1.5)
    lookback_period: days to calculate baseline stats (None = all history)
    """
    if lookback_period:
        rolling_corr_baseline = rolling_corr.iloc[-lookback_period:]
    else:
        rolling_corr_baseline = rolling_corr
    
    mean = rolling_corr_baseline.mean()
    std = rolling_corr_baseline.std()
    
    z_scores = (rolling_corr - mean) / std
    
    anomalies = rolling_corr[abs(z_scores) > threshold]
    
    return anomalies, z_scores

def plot_multi_window_correlation(asset1_data, asset2_data, asset1_name, asset2_name, 
                                  windows=[30, 90], save_path=None):
    """Plot multiple rolling correlation windows for comparison"""
    fig, axes = plt.subplots(len(windows), 1, figsize=(14, 5*len(windows)))
    
    if len(windows) == 1:
        axes = [axes]
    
    for idx, window in enumerate(windows):
        rolling_corr = calculate_rolling_correlation(asset1_data, asset2_data, window=window)
        
        # Calculate statistics using 365-day lookback for baseline
        stats = calculate_correlation_statistics(rolling_corr, lookback_period=365)
        anomalies, z_scores = detect_anomalies(rolling_corr, threshold=1.5, lookback_period=365)
        
        ax = axes[idx]
        
        # Plot rolling correlation
        ax.plot(rolling_corr.index, rolling_corr, 
                label=f'{window}-day Rolling Correlation', 
                linewidth=2, color='blue', alpha=0.7)
        
        # Add mean line (1-year lookback)
        mean = stats['mean']
        ax.axhline(y=mean, color='green', linestyle='--', alpha=0.5, 
                   label=f'1-Year Mean ({mean:.3f})')
        
        # Add +/- 1.5 std dev bands (1-year lookback)
        std = stats['std']
        upper_band = mean + 1.5*std
        lower_band = mean - 1.5*std
        
        ax.axhline(y=upper_band, color='orange', linestyle=':', alpha=0.5, 
                   label=f'+1.5 Std Dev ({upper_band:.3f})')
        ax.axhline(y=lower_band, color='orange', linestyle=':', alpha=0.5,
                   label=f'-1.5 Std Dev ({lower_band:.3f})')
        
        # Highlight anomalies
        if len(anomalies) > 0:
            ax.scatter(anomalies.index, anomalies.values, color='red', 
                      s=80, zorder=5, label=f'Anomalies ({len(anomalies)})', 
                      marker='o', edgecolors='darkred', linewidth=1.5)
        
        # Add current value annotation
        current = stats['current']
        z_score = stats['z_score']
        ax.annotate(f'Current: {current:.3f}\nZ-score: {z_score:.2f}',
                   xy=(rolling_corr.index[-1], current),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Correlation', fontsize=11)
        ax.set_title(f'{asset1_name} vs {asset2_name}: {window}-Day Rolling Correlation\n(Baseline: 1-year lookback, Threshold: ±1.5 std)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()

def plot_rolling_correlation(rolling_corr, asset1_name, asset2_name, 
                            window=90, anomalies=None, save_path=None):
    """Plot single rolling correlation with anomaly highlights"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Calculate stats using 1-year lookback
    stats = calculate_correlation_statistics(rolling_corr, lookback_period=365)
    
    # Plot rolling correlation
    ax.plot(rolling_corr.index, rolling_corr, label=f'{window}-day Rolling Correlation', 
            linewidth=2, color='blue', alpha=0.7)
    
    # Add mean line (1-year lookback)
    mean = stats['mean']
    ax.axhline(y=mean, color='green', linestyle='--', alpha=0.5, 
               label=f'1-Year Mean ({mean:.3f})')
    
    # Add +/- 1.5 std dev bands (1-year lookback)
    std = stats['std']
    ax.axhline(y=mean + 1.5*std, color='orange', linestyle=':', alpha=0.5, 
               label='+1.5 Std Dev')
    ax.axhline(y=mean - 1.5*std, color='orange', linestyle=':', alpha=0.5,
               label='-1.5 Std Dev')
    
    # Highlight anomalies
    if anomalies is not None and len(anomalies) > 0:
        ax.scatter(anomalies.index, anomalies.values, color='red', 
                  s=100, zorder=5, label=f'Anomalies ({len(anomalies)})', marker='o')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Correlation', fontsize=12)
    ax.set_title(f'{asset1_name} vs {asset2_name}: Rolling Correlation Analysis\n(Baseline: 1-year lookback)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Loading asset data...")
    
    # Example: Analyze Gold vs S&P 500
    gold = load_asset_data('data/raw/GCF_commodity.csv')
    spy = load_asset_data('data/raw/SPY_stock.csv')
    
    print("\nAnalyzing with multiple time windows...")
    print("="*60)
    
    # Analyze with both 30-day and 90-day windows
    for window in [30, 90]:
        print(f"\n{window}-Day Rolling Correlation:")
        rolling_corr = calculate_rolling_correlation(gold, spy, window=window)
        
        # Use 1-year lookback for baseline statistics
        stats = calculate_correlation_statistics(rolling_corr, lookback_period=365)
        
        print(f"  Current: {stats['current']:.4f}")
        print(f"  1-Year Mean: {stats['mean']:.4f}")
        print(f"  1-Year Std Dev: {stats['std']:.4f}")
        print(f"  Z-Score: {stats['z_score']:.4f}")
        
        anomalies, z_scores = detect_anomalies(rolling_corr, threshold=1.5, lookback_period=365)
        print(f"  Anomalies detected: {len(anomalies)}")
    
    print("\n" + "="*60)
    print("\nCreating multi-window visualization...")
    plot_multi_window_correlation(gold, spy, 'Gold', 'S&P 500', 
                                 windows=[30, 90],
                                 save_path='docs/gold_spy_multi_window_correlation.png')