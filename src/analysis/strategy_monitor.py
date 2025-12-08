import pandas as pd
import numpy as np
from pathlib import Path
from rolling_correlation import load_asset_data, calculate_rolling_correlation, calculate_correlation_statistics, detect_anomalies
import itertools

def get_all_available_assets():
    """Scan and categorize all available assets"""
    data_dir = Path('data/raw')
    assets = {}
    
    for file in data_dir.glob('*.csv'):
        asset_name = file.stem.replace('_stock', '').replace('_etf', '').replace('_fx', '').replace('_commodity', '').replace('_rates', '').replace('_economic', '')
        
        # Determine category
        if '_stock' in file.stem or '_etf' in file.stem:
            category = 'Equity'
        elif '_fx' in file.stem:
            category = 'FX'
        elif '_commodity' in file.stem:
            category = 'Commodity'
        elif '_rates' in file.stem:
            category = 'Rates'
        elif '_economic' in file.stem:
            category = 'Economic'
        else:
            category = 'Other'
        
        assets[asset_name] = {
            'path': str(file),
            'category': category
        }
    
    return assets

def discover_stable_correlations(window=90, lookback=252, min_correlation=0.6, 
                                 max_pairs=200, sample_recent_days=30):
    """
    Automatically discover asset pairs with historically stable, strong correlations
    These are candidates for correlation-based trading strategies
    
    Args:
        window: Rolling window for correlation
        lookback: Historical baseline period
        min_correlation: Minimum average correlation to consider (absolute value)
        max_pairs: Maximum number of pairs to analyze
        sample_recent_days: Use recent data to speed up discovery
    
    Returns:
        DataFrame with discovered stable correlation pairs
    """
    assets = get_all_available_assets()
    asset_names = list(assets.keys())
    
    print(f"Discovering stable correlations from {len(asset_names)} assets...")
    
    stable_pairs = []
    pairs_analyzed = 0
    
    # Generate all possible pairs
    for asset1, asset2 in itertools.combinations(asset_names, 2):
        if pairs_analyzed >= max_pairs:
            break
        
        try:
            # Load data
            data1 = load_asset_data(assets[asset1]['path'])
            data2 = load_asset_data(assets[asset2]['path'])
            
            # Use recent data for faster discovery
            if len(data1) > sample_recent_days:
                data1 = data1.iloc[-sample_recent_days:]
            if len(data2) > sample_recent_days:
                data2 = data2.iloc[-sample_recent_days:]
            
            # Calculate correlation
            rolling_corr = calculate_rolling_correlation(data1, data2, window=min(window, len(data1)//2))
            
            if len(rolling_corr) < 10:
                continue
            
            # Calculate statistics
            mean_corr = rolling_corr.mean()
            std_corr = rolling_corr.std()
            current_corr = rolling_corr.iloc[-1]
            
            # Check if correlation is stable and strong
            if abs(mean_corr) >= min_correlation and std_corr < 0.25:
                stable_pairs.append({
                    'Asset 1': asset1,
                    'Category 1': assets[asset1]['category'],
                    'Asset 2': asset2,
                    'Category 2': assets[asset2]['category'],
                    'Avg Correlation': mean_corr,
                    'Std Dev': std_corr,
                    'Current Correlation': current_corr,
                    'Stability Score': abs(mean_corr) / (std_corr + 0.01),  # Higher = more stable
                    'Pair Type': f"{assets[asset1]['category']}-{assets[asset2]['category']}"
                })
            
            pairs_analyzed += 1
            
            if pairs_analyzed % 50 == 0:
                print(f"  Analyzed {pairs_analyzed} pairs, found {len(stable_pairs)} stable correlations...")
        
        except Exception as e:
            continue
    
    print(f"✅ Discovery complete! Found {len(stable_pairs)} stable correlation pairs")
    
    if stable_pairs:
        df = pd.DataFrame(stable_pairs)
        df = df.sort_values('Stability Score', ascending=False)
        return df
    else:
        return pd.DataFrame()

def monitor_correlation_strategies(strategy_pairs_df, window=90, lookback=252, 
                                   threshold=1.5, predict_future=False, forecast_days=30):
    """
    Monitor multiple correlation-based strategies for risk
    
    Args:
        strategy_pairs_df: DataFrame from discover_stable_correlations
        window: Rolling window
        lookback: Historical baseline
        threshold: Anomaly threshold
        predict_future: Include ML predictions
        forecast_days: Prediction horizon
    
    Returns:
        DataFrame with risk analysis for each pair
    """
    assets = get_all_available_assets()
    results = []
    
    print(f"Monitoring {len(strategy_pairs_df)} correlation strategies...")
    
    for idx, row in strategy_pairs_df.iterrows():
        try:
            asset1 = row['Asset 1']
            asset2 = row['Asset 2']
            expected_corr = row['Avg Correlation']
            
            # Load full historical data
            data1 = load_asset_data(assets[asset1]['path'])
            data2 = load_asset_data(assets[asset2]['path'])
            
            # Calculate current correlation
            rolling_corr = calculate_rolling_correlation(data1, data2, window=window)
            stats = calculate_correlation_statistics(rolling_corr, lookback_period=lookback)
            anomalies, z_scores = detect_anomalies(rolling_corr, threshold=threshold, lookback_period=lookback)
            
            # Calculate deviation from expected
            deviation = stats['current'] - expected_corr
            z_score_vs_expected = deviation / stats['std'] if stats['std'] > 0 else 0
            
            # Health score: 100 = perfect, 0 = severe breakdown
            health_score = max(0, 100 - abs(z_score_vs_expected) * 25)
            
            # Status
            if health_score >= 70:
                status = '🟢 Healthy'
            elif health_score >= 40:
                status = '🟡 Warning'
            else:
                status = '🔴 Critical'
            
            result = {
                'Asset 1': asset1,
                'Category 1': row['Category 1'],
                'Asset 2': asset2,
                'Category 2': row['Category 2'],
                'Pair Type': row['Pair Type'],
                'Expected Corr': expected_corr,
                'Current Corr': stats['current'],
                'Deviation': deviation,
                'Z-Score': z_score_vs_expected,
                'Health Score': health_score,
                'Status': status,
                'Anomalies': len(anomalies),
                'Std Dev': stats['std']
            }
            
            # Add ML prediction if requested
            if predict_future:
                try:
                    import sys
                    sys.path.append('src/ml')
                    from correlation_predictor import predict_correlation
                    
                    ml_result = predict_correlation(
                        data1, data2,
                        window=window,
                        days_ahead=forecast_days,
                        model_type='linear'
                    )
                    result['Predicted Corr'] = ml_result['predictions'][-1]
                    result['Forecast Change'] = ml_result['predictions'][-1] - stats['current']
                    
                    # Check if prediction suggests further breakdown
                    if abs(result['Forecast Change']) > 0.2:
                        result['Forecast Alert'] = '⚠️ Major change predicted'
                    else:
                        result['Forecast Alert'] = '✓ Stable'
                except:
                    result['Predicted Corr'] = None
                    result['Forecast Change'] = None
                    result['Forecast Alert'] = 'N/A'
            
            results.append(result)
            
        except Exception as e:
            print(f"Error monitoring {row['Asset 1']}-{row['Asset 2']}: {e}")
            continue
    
    print(f"✅ Monitoring complete! Analyzed {len(results)} strategies")
    
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('Health Score', ascending=True)  # Show most at-risk first
        return df
    else:
        return pd.DataFrame()

def get_critical_alerts(monitored_df, health_threshold=40):
    """Extract critical alerts from monitored strategies"""
    critical = monitored_df[monitored_df['Health Score'] < health_threshold]
    
    alerts = []
    for _, row in critical.iterrows():
        alert = {
            'Severity': '🔴 CRITICAL' if row['Health Score'] < 20 else '🟡 WARNING',
            'Pair': f"{row['Asset 1']} - {row['Asset 2']}",
            'Type': row['Pair Type'],
            'Issue': f"Correlation: {row['Current Corr']:.2f} (Expected: {row['Expected Corr']:.2f})",
            'Health': f"{row['Health Score']:.0f}/100",
            'Action': 'Review position sizing' if row['Health Score'] < 40 else 'Monitor closely'
        }
        alerts.append(alert)
    
    return pd.DataFrame(alerts) if alerts else pd.DataFrame()

if __name__ == "__main__":
    print("="*60)
    print("DYNAMIC STRATEGY RISK MONITOR")
    print("="*60)
    
    # Step 1: Discover stable correlations
    print("\nStep 1: Discovering stable correlation pairs...")
    stable_pairs = discover_stable_correlations(
        min_correlation=0.7,
        max_pairs=200,
        sample_recent_days=90
    )
    
    if len(stable_pairs) > 0:
        print(f"\nTop 10 Most Stable Correlations:")
        print(stable_pairs.head(10)[['Asset 1', 'Asset 2', 'Avg Correlation', 'Stability Score']].to_string(index=False))
        
        # Step 2: Monitor top strategies
        print(f"\nStep 2: Monitoring top {min(20, len(stable_pairs))} strategies for risk...")
        monitored = monitor_correlation_strategies(
            stable_pairs.head(20),
            predict_future=True
        )
        
        if len(monitored) > 0:
            print(f"\nStrategy Health Summary:")
            print(monitored[['Asset 1', 'Asset 2', 'Health Score', 'Status']].to_string(index=False))
            
            # Step 3: Generate alerts
            alerts = get_critical_alerts(monitored)
            if len(alerts) > 0:
                print(f"\n🚨 CRITICAL ALERTS ({len(alerts)}):")
                print(alerts.to_string(index=False))
            else:
                print("\n✅ No critical alerts - all strategies healthy")
    else:
        print("\n⚠️ No stable correlations discovered")