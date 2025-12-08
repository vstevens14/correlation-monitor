import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import sys
sys.path.append('src/analysis')
from rolling_correlation import load_asset_data, calculate_rolling_correlation

def fetch_vix_data(start_date='2020-01-01'):
    """Fetch VIX (Volatility Index) data"""
    try:
        vix = yf.Ticker('^VIX')
        data = vix.history(start=start_date)
        data.columns = data.columns.str.lower()
        # Remove timezone
        if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        return data['close']
    except Exception as e:
        print(f"Error fetching VIX: {e}")
        return None

def calculate_volume_features(asset_data):
    """
    Calculate volume-based features
    
    Args:
        asset_data: DataFrame with 'close' and 'volume' columns
    
    Returns:
        DataFrame with volume features
    """
    features = pd.DataFrame(index=asset_data.index)
    
    # Volume moving averages
    features['volume_ma_20'] = asset_data['volume'].rolling(20).mean()
    features['volume_ma_50'] = asset_data['volume'].rolling(50).mean()
    
    # Volume ratio (current vs average)
    features['volume_ratio'] = asset_data['volume'] / features['volume_ma_20']
    
    # Volume trend
    features['volume_trend'] = features['volume_ma_20'] / features['volume_ma_50']
    
    return features

def calculate_price_momentum(asset_data, windows=[5, 10, 20, 50]):
    """
    Calculate price momentum features
    
    Args:
        asset_data: DataFrame with 'close' column
        windows: List of lookback periods
    
    Returns:
        DataFrame with momentum features
    """
    features = pd.DataFrame(index=asset_data.index)
    
    for window in windows:
        # Returns
        features[f'return_{window}d'] = asset_data['close'].pct_change(window)
        
        # Moving averages
        features[f'ma_{window}'] = asset_data['close'].rolling(window).mean()
        
        # Price vs MA
        features[f'price_vs_ma_{window}'] = asset_data['close'] / features[f'ma_{window}'] - 1
    
    # Volatility
    features['volatility_20'] = asset_data['close'].pct_change().rolling(20).std()
    
    return features

def load_macro_indicators():
    """Load all available macro indicators"""
    data_dir = Path('data/raw')
    indicators = {}
    
    for file in data_dir.glob('*_economic.csv'):
        indicator_name = file.stem.replace('_economic', '')
        try:
            df = pd.read_csv(file, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index)
            # Remove timezone if present
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            indicators[indicator_name] = df.iloc[:, 0]
        except Exception as e:
            print(f"Could not load {indicator_name}: {e}")
    
    return indicators

def create_enhanced_features(asset1_data, asset2_data, rolling_corr, 
                            include_vix=True, include_volume=True, 
                            include_macro=True, include_momentum=True):
    """
    Create comprehensive feature set for correlation prediction
    
    Args:
        asset1_data: Price data for asset 1 (DataFrame)
        asset2_data: Price data for asset 2 (DataFrame)
        rolling_corr: Historical rolling correlation
        include_vix: Include VIX features
        include_volume: Include volume features
        include_macro: Include macro indicators
        include_momentum: Include price momentum
    
    Returns:
        DataFrame with all features aligned to correlation dates
    """
    # Remove timezone from all inputs
    if isinstance(rolling_corr.index, pd.DatetimeIndex) and rolling_corr.index.tz is not None:
        rolling_corr = rolling_corr.copy()
        rolling_corr.index = rolling_corr.index.tz_localize(None)
    
    if isinstance(asset1_data.index, pd.DatetimeIndex) and asset1_data.index.tz is not None:
        asset1_data = asset1_data.copy()
        asset1_data.index = asset1_data.index.tz_localize(None)
    
    if isinstance(asset2_data.index, pd.DatetimeIndex) and asset2_data.index.tz is not None:
        asset2_data = asset2_data.copy()
        asset2_data.index = asset2_data.index.tz_localize(None)
    
    features = pd.DataFrame(index=rolling_corr.index)
    
    # Base feature: correlation itself
    features['corr'] = rolling_corr
    
    # Correlation momentum
    features['corr_change_5d'] = rolling_corr.diff(5)
    features['corr_change_20d'] = rolling_corr.diff(20)
    features['corr_volatility'] = rolling_corr.rolling(20).std()
    
    # VIX features
    if include_vix:
        try:
            vix = fetch_vix_data()
            if vix is not None:
                # Remove timezone from VIX before alignment
                if isinstance(vix.index, pd.DatetimeIndex) and vix.index.tz is not None:
                    vix.index = vix.index.tz_localize(None)
                
                # Align VIX to correlation dates
                vix_aligned = vix.reindex(rolling_corr.index, method='ffill')
                features['vix'] = vix_aligned
                features['vix_change_5d'] = vix_aligned.diff(5)
                features['vix_ma_20'] = vix_aligned.rolling(20).mean()
                print("✓ Added VIX features")
        except Exception as e:
            print(f"Could not add VIX features: {e}")
    
    # Volume features
    if include_volume:
        try:
            # Asset 1 volume
            vol_features_1 = calculate_volume_features(asset1_data)
            vol_features_1 = vol_features_1.reindex(rolling_corr.index, method='ffill')
            for col in vol_features_1.columns:
                features[f'asset1_{col}'] = vol_features_1[col]
            
            # Asset 2 volume
            vol_features_2 = calculate_volume_features(asset2_data)
            vol_features_2 = vol_features_2.reindex(rolling_corr.index, method='ffill')
            for col in vol_features_2.columns:
                features[f'asset2_{col}'] = vol_features_2[col]
            
            print("✓ Added volume features")
        except Exception as e:
            print(f"Could not add volume features: {e}")
    
    # Momentum features
    if include_momentum:
        try:
            # Asset 1 momentum
            momentum_1 = calculate_price_momentum(asset1_data)
            momentum_1 = momentum_1.reindex(rolling_corr.index, method='ffill')
            for col in momentum_1.columns:
                features[f'asset1_{col}'] = momentum_1[col]
            
            # Asset 2 momentum
            momentum_2 = calculate_price_momentum(asset2_data)
            momentum_2 = momentum_2.reindex(rolling_corr.index, method='ffill')
            for col in momentum_2.columns:
                features[f'asset2_{col}'] = momentum_2[col]
            
            print("✓ Added momentum features")
        except Exception as e:
            print(f"Could not add momentum features: {e}")
    
    # Macro indicators
    if include_macro:
        try:
            indicators = load_macro_indicators()
            for name, series in indicators.items():
                # Remove timezone from series before alignment
                if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
                    series = series.copy()
                    series.index = series.index.tz_localize(None)
                
                series_aligned = series.reindex(rolling_corr.index, method='ffill')
                features[f'macro_{name}'] = series_aligned
                features[f'macro_{name}_change'] = series_aligned.diff(1)
            print(f"✓ Added {len(indicators)} macro indicators")
        except Exception as e:
            print(f"Could not add macro features: {e}")
    
    # Remove rows with NaN (due to rolling windows)
    features = features.dropna()
    
    print(f"\n✅ Created {len(features.columns)} features for {len(features)} time periods")
    
    return features

if __name__ == "__main__":
    print("Testing Feature Engineering...\n")
    
    # Load sample data with full DataFrame
    spy_full = load_asset_data('data/raw/SPY_etf.csv', return_full=True)
    gold_full = load_asset_data('data/raw/GCF_commodity.csv', return_full=True)
    
    # Remove timezone from asset data
    if isinstance(spy_full.index, pd.DatetimeIndex) and spy_full.index.tz is not None:
        spy_full.index = spy_full.index.tz_localize(None)
    if isinstance(gold_full.index, pd.DatetimeIndex) and gold_full.index.tz is not None:
        gold_full.index = gold_full.index.tz_localize(None)
    
    # Calculate correlation using close prices
    rolling_corr = calculate_rolling_correlation(spy_full['close'], gold_full['close'], window=90)
    
    # Create features
    features = create_enhanced_features(
        spy_full, gold_full, rolling_corr,
        include_vix=True,
        include_volume=True,
        include_macro=True,
        include_momentum=True
    )
    
    print("\nFeature columns:")
    for col in features.columns:
        print(f"  - {col}")
    
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"Date range: {features.index.min()} to {features.index.max()}")