import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
sys.path.append('src/analysis')
from rolling_correlation import load_asset_data, calculate_rolling_correlation, calculate_correlation_statistics, detect_anomalies
from correlation_matrix import load_multiple_assets, calculate_correlation_matrix

# Page configuration
st.set_page_config(
    page_title="Cross-Asset Correlation Anomaly Detector",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Cross-Asset Correlation Anomaly Detector")
st.markdown("""
Track rolling correlations across equities, FX, commodities, and rates. 
Detect when relationships break from historical norms to identify potential trading opportunities.
""")

# Sidebar - Asset Selection
st.sidebar.header("Configuration")

# Define available assets
ASSETS = {
    'Equities': {
        'SPY': 'data/raw/SPY_stock.csv',
        'AAPL': 'data/raw/AAPL_stock.csv',
        'MSFT': 'data/raw/MSFT_stock.csv',
        'GOOGL': 'data/raw/GOOGL_stock.csv',
        'XLF': 'data/raw/XLF_stock.csv'
    },
    'Commodities': {
        'Gold': 'data/raw/GCF_commodity.csv',
        'Oil': 'data/raw/CLF_commodity.csv',
        'Copper': 'data/raw/HGF_commodity.csv'
    },
    'FX': {
        'EUR/USD': 'data/raw/EURUSDX_fx.csv',
        'USD/JPY': 'data/raw/JPYX_fx.csv',
        'GBP/USD': 'data/raw/GBPUSDX_fx.csv',
        'USD Index': 'data/raw/DX-Y_NYB_fx.csv'
    },
    'Rates': {
        '10Y Yield': 'data/raw/TNX_rates.csv',
        '13W Bill': 'data/raw/IRX_rates.csv',
        'TLT': 'data/raw/TLT_rates.csv'
    }
}

# Flatten assets for selection
all_assets = {}
for category, assets in ASSETS.items():
    all_assets.update(assets)

# Asset pair selection
st.sidebar.subheader("Select Asset Pair")
asset1_name = st.sidebar.selectbox("Asset 1", list(all_assets.keys()), index=0)
asset2_name = st.sidebar.selectbox("Asset 2", list(all_assets.keys()), index=1)

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")
window = st.sidebar.selectbox("Rolling Window", [30, 60, 90, 120], index=2)
lookback = st.sidebar.selectbox("Historical Baseline", [180, 252, 365], index=1)
threshold = st.sidebar.slider("Anomaly Threshold (Std Dev)", 1.0, 3.0, 1.5, 0.1)

# Load data
@st.cache_data
def load_data(filepath):
    return load_asset_data(filepath)

try:
    asset1_data = load_data(all_assets[asset1_name])
    asset2_data = load_data(all_assets[asset2_name])
    
    # Calculate rolling correlation
    rolling_corr = calculate_rolling_correlation(asset1_data, asset2_data, window=window)
    
    # Calculate statistics
    stats = calculate_correlation_statistics(rolling_corr, lookback_period=lookback)
    
    # Detect anomalies
    anomalies, z_scores = detect_anomalies(rolling_corr, threshold=threshold, lookback_period=lookback)
    
    # Main content area
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Correlation", f"{stats['current']:.3f}")
    
    with col2:
        st.metric("Historical Mean", f"{stats['mean']:.3f}")
    
    with col3:
        delta_color = "normal" if abs(stats['z_score']) < threshold else "inverse"
        st.metric("Z-Score", f"{stats['z_score']:.2f}", 
                 delta=f"{'⚠️ Anomaly' if abs(stats['z_score']) > threshold else '✓ Normal'}")
    
    with col4:
        st.metric("Anomalies Detected", len(anomalies))
    
    # Rolling correlation chart
    st.subheader(f"{asset1_name} vs {asset2_name}: {window}-Day Rolling Correlation")
    
    fig = go.Figure()
    
    # Add rolling correlation line
    fig.add_trace(go.Scatter(
        x=rolling_corr.index,
        y=rolling_corr.values,
        mode='lines',
        name=f'{window}-day Correlation',
        line=dict(color='blue', width=2)
    ))
    
    # Add mean line
    fig.add_hline(y=stats['mean'], line_dash="dash", line_color="green",
                  annotation_text=f"Mean: {stats['mean']:.3f}")
    
    # Add threshold bands
    upper_band = stats['mean'] + threshold * stats['std']
    lower_band = stats['mean'] - threshold * stats['std']
    
    fig.add_hline(y=upper_band, line_dash="dot", line_color="orange",
                  annotation_text=f"+{threshold} Std")
    fig.add_hline(y=lower_band, line_dash="dot", line_color="orange",
                  annotation_text=f"-{threshold} Std")
    
    # Add anomaly points
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies.values,
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10, symbol='circle')
        ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Correlation",
        yaxis_range=[-1, 1],
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent anomalies table
    if len(anomalies) > 0:
        st.subheader("🚨 Recent Anomalies")
        recent_anomalies = anomalies.tail(10).to_frame()
        recent_anomalies.columns = ['Correlation']
        recent_anomalies['Z-Score'] = z_scores.loc[recent_anomalies.index]
        recent_anomalies = recent_anomalies.sort_index(ascending=False)
        st.dataframe(recent_anomalies.style.format({'Correlation': '{:.3f}', 'Z-Score': '{:.2f}'}))
    
    # Interpretation
    st.subheader("📖 Interpretation")
    
    if abs(stats['z_score']) > threshold:
        st.warning(f"""
        **⚠️ Anomaly Detected!**
        
        The current correlation ({stats['current']:.3f}) is {abs(stats['z_score']):.2f} standard deviations away 
        from the historical mean ({stats['mean']:.3f}).
        
        This suggests a **regime change** in the relationship between {asset1_name} and {asset2_name}.
        """)
    else:
        st.success(f"""
        **✓ Normal Correlation**
        
        The current correlation ({stats['current']:.3f}) is within normal range of the historical mean ({stats['mean']:.3f}).
        
        Z-Score: {stats['z_score']:.2f} (threshold: ±{threshold})
        """)
    
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure you've run the data collection script first: `python src/data/collect_all_assets.py`")

# Correlation Matrix Section
st.markdown("---")
st.header("🔥 Cross-Asset Correlation Matrix")

with st.spinner("Calculating correlations..."):
    try:
        # Select key assets for matrix
        matrix_assets = {
            'SPY': 'data/raw/SPY_stock.csv',
            'Gold': 'data/raw/GCF_commodity.csv',
            'Oil': 'data/raw/CLF_commodity.csv',
            'USD_Index': 'data/raw/DX-Y_NYB_fx.csv',
            '10Y_Yield': 'data/raw/TNX_rates.csv',
            'EUR/USD': 'data/raw/EURUSDX_fx.csv'
        }
        
        df = load_multiple_assets(matrix_assets)
        current_corr = df.tail(90).corr()
        
        # Create heatmap
        fig = px.imshow(current_corr,
                       labels=dict(color="Correlation"),
                       x=current_corr.columns,
                       y=current_corr.index,
                       color_continuous_scale='RdYlGn',
                       zmin=-1, zmax=1,
                       text_auto='.2f')
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate correlation matrix: {e}")

# Footer
st.markdown("---")
st.markdown("""
**About:** This tool tracks rolling correlations across multiple asset classes to identify anomalies 
that may signal trading opportunities or regime changes in the market.

**Data Source:** Yahoo Finance (yfinance) | **Update Frequency:** Daily
""")