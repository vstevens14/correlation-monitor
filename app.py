import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Cross-Asset Correlation Anomaly Detector",
    page_icon="📊",
    layout="wide"
)

import sys
import os
from pathlib import Path

sys.path.append('src/analysis')
sys.path.append('src/ui')

from rolling_correlation import load_asset_data, calculate_rolling_correlation
from styling import get_custom_css, get_header_html
from sidebar import render_sidebar
from tab_pair_analysis import render_pair_analysis_tab
from tab_anomalies import render_anomalies_tab
from tab_strategy_monitor import render_strategy_monitor_tab
from tab_correlation_matrix import render_correlation_matrix_tab

# Create data directories if they don't exist
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Function to collect data on first run
def ensure_data_exists():
    """Check if data exists, if not, collect it"""
    if not os.path.exists('data/raw/MMM_stock.csv'):
        with st.spinner("📥 Collecting comprehensive dataset... This may take 5-10 minutes"):
            try:
                sys.path.append('src/data')
                from collect_all_assets import collect_all_assets
                collect_all_assets(start_date='2020-01-01', num_stocks=100)
                st.success("✅ Data collection complete!")
            except Exception as e:
                st.error(f"Error collecting data: {e}")
                st.stop()

# Run data collection check
ensure_data_exists()

# Function to dynamically load all available assets
def get_available_assets():
    """Scan data/raw directory and categorize assets"""
    data_dir = Path('data/raw')
    assets = {
        'Equities': {},
        'ETFs': {},
        'Commodities': {},
        'FX': {},
        'Rates': {},
        'Economic': {}
    }
    
    for file in data_dir.glob('*.csv'):
        filepath = str(file)
        filename = file.stem
        
        if '_stock' in filename:
            ticker = filename.replace('_stock', '')
            assets['Equities'][ticker] = filepath
        elif '_etf' in filename:
            ticker = filename.replace('_etf', '')
            assets['ETFs'][ticker] = filepath
        elif '_commodity' in filename:
            name = filename.replace('_commodity', '')
            assets['Commodities'][name] = filepath
        elif '_fx' in filename:
            name = filename.replace('_fx', '')
            assets['FX'][name] = filepath
        elif '_rates' in filename:
            name = filename.replace('_rates', '')
            assets['Rates'][name] = filepath
        elif '_economic' in filename:
            name = filename.replace('_economic', '')
            assets['Economic'][name] = filepath
    
    return assets

# Get all available assets
ASSETS = get_available_assets()

# Flatten assets for selection
all_assets = {}
for category, assets_dict in ASSETS.items():
    all_assets.update(assets_dict)
all_assets = dict(sorted(all_assets.items()))

# Apply custom styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Title and header
st.title("Cross-Asset Correlation Anomaly Detector")
st.markdown(get_header_html(), unsafe_allow_html=True)

# Render sidebar and get configuration
config = render_sidebar(ASSETS, all_assets)

# Extract configuration values
asset1_name = config['asset1_name']
asset2_name = config['asset2_name']
window = config['window']
lookback = config['lookback']
threshold = config['threshold']

# Load data
@st.cache_data
def load_data(filepath):
    return load_asset_data(filepath)

asset1_data = load_data(config['asset1_path'])
asset2_data = load_data(config['asset2_path'])

# Calculate rolling correlation (shared across tabs)
try:
    rolling_corr = calculate_rolling_correlation(asset1_data, asset2_data, window=window)
    
    # Debug: Check if data is valid
    if len(rolling_corr) == 0 or rolling_corr.empty:
        st.error(f"Correlation calculation returned empty results. Asset 1: {len(asset1_data)} points, Asset 2: {len(asset2_data)} points")
        st.stop()
        
except Exception as e:
    st.error(f"Error calculating correlation: {e}")
    st.info(f"Asset 1 ({asset1_name}): {len(asset1_data)} data points")
    st.info(f"Asset 2 ({asset2_name}): {len(asset2_data)} data points")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Pair Analysis", "Top Anomalies", "Strategy Monitor", "Correlation Matrix"])

# Render each tab
with tab1:
    render_pair_analysis_tab(asset1_data, asset2_data, asset1_name, asset2_name, 
                             window, lookback, threshold, rolling_corr)

with tab2:
    render_anomalies_tab(window, lookback, threshold)

with tab3:
    render_strategy_monitor_tab(window, lookback, threshold)

with tab4:
    render_correlation_matrix_tab()

# Footer
st.markdown("---")
st.markdown("""
**About:** This tool tracks rolling correlations across multiple asset classes to identify anomalies 
that may signal trading opportunities or regime changes in the market.

**Data Source:** Yahoo Finance (yfinance) | **Update Frequency:** Daily
""")