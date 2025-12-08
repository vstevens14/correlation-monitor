import streamlit as st
import json
from pathlib import Path

# Load asset names
def load_asset_names():
    """Load asset name mappings from JSON"""
    try:
        with open('src/data/asset_names.json', 'r') as f:
            return json.load(f)
    except:
        return {}

ASSET_NAMES = load_asset_names()

def format_asset_option(ticker):
    """Format asset ticker with full name"""
    name = ASSET_NAMES.get(ticker, ticker)
    if name != ticker:
        return f"{ticker} ({name})"
    return ticker

def parse_asset_selection(selection):
    """Extract ticker from formatted selection"""
    if ' (' in selection:
        return selection.split(' (')[0]
    return selection

def render_sidebar(ASSETS, all_assets):
    """
    Render the sidebar with asset selection and parameters
    
    Returns:
        dict: Configuration including asset names, paths, and parameters
    """
    
    # Header
    st.sidebar.markdown(
        "<h2 style='text-align: center; color: #1565C0; font-size: 1.4rem; margin-bottom: 1.5rem;'>Configuration</h2>", 
        unsafe_allow_html=True
    )
    
    # Asset Selection Section
    st.sidebar.markdown("**Asset Selection**")
    
    # Asset 1
    category_filter_1 = st.sidebar.selectbox(
        "Asset 1 Category",
        ["All", "Equities", "ETFs", "Commodities", "FX", "Rates", "Economic"],
        key="cat1"
    )
    
    # Filter assets based on category for Asset 1
    if category_filter_1 == "All":
        filtered_assets_1 = all_assets
    else:
        filtered_assets_1 = ASSETS[category_filter_1]
        filtered_assets_1 = dict(sorted(filtered_assets_1.items()))
    
    # Format options for display
    asset1_options = [format_asset_option(ticker) for ticker in filtered_assets_1.keys()]
    
    asset1_selection = st.sidebar.selectbox(
        "Select Asset 1", 
        asset1_options, 
        index=0
    )
    
    # Parse back to ticker
    asset1_name = parse_asset_selection(asset1_selection)
    
    st.sidebar.markdown("---")
    
    # Asset 2
    category_filter_2 = st.sidebar.selectbox(
        "Asset 2 Category",
        ["All", "Equities", "ETFs", "Commodities", "FX", "Rates", "Economic"],
        key="cat2"
    )
    
    # Filter assets based on category for Asset 2
    if category_filter_2 == "All":
        filtered_assets_2 = all_assets
    else:
        filtered_assets_2 = ASSETS[category_filter_2]
        filtered_assets_2 = dict(sorted(filtered_assets_2.items()))
    
    # Remove Asset 1 from Asset 2 options
    filtered_assets_2 = {k: v for k, v in filtered_assets_2.items() if k != asset1_name}
    
    # Format options for display
    asset2_options = [format_asset_option(ticker) for ticker in filtered_assets_2.keys()]
    
    if len(filtered_assets_2) > 0:
        asset2_selection = st.sidebar.selectbox(
            "Select Asset 2", 
            asset2_options, 
            index=0
        )
        
        # Parse back to ticker
        asset2_name = parse_asset_selection(asset2_selection)
    else:
        st.sidebar.error("Need at least 2 assets")
        st.stop()
    
    st.sidebar.markdown("---")
    
    # Analysis Parameters
    st.sidebar.markdown("**Analysis Parameters**")
    window = st.sidebar.selectbox("Rolling Window (days)", [30, 60, 90, 120], index=2)
    lookback = st.sidebar.selectbox("Baseline Period (days)", [180, 252, 365], index=1)
    threshold = st.sidebar.slider("Anomaly Threshold (σ)", 1.0, 3.0, 1.5, 0.1)
    
    # Get the original filtered_assets dicts to access paths
    if category_filter_1 == "All":
        filtered_assets_1_paths = all_assets
    else:
        filtered_assets_1_paths = ASSETS[category_filter_1]
        filtered_assets_1_paths = dict(sorted(filtered_assets_1_paths.items()))
    
    if category_filter_2 == "All":
        filtered_assets_2_paths = all_assets
    else:
        filtered_assets_2_paths = ASSETS[category_filter_2]
        filtered_assets_2_paths = dict(sorted(filtered_assets_2_paths.items()))
    
    filtered_assets_2_paths = {k: v for k, v in filtered_assets_2_paths.items() if k != asset1_name}
    
    # Return configuration
    return {
        'asset1_name': asset1_name,
        'asset2_name': asset2_name,
        'asset1_path': filtered_assets_1_paths[asset1_name],
        'asset2_path': filtered_assets_2_paths[asset2_name],
        'window': window,
        'lookback': lookback,
        'threshold': threshold
    }