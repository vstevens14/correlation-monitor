import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

sys.path.append('src/analysis')
from rolling_correlation import load_asset_data, calculate_rolling_correlation, calculate_correlation_statistics, detect_anomalies
from correlation_matrix import load_multiple_assets, calculate_correlation_matrix

# Create data directories if they don't exist
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Function to collect data on first run
def ensure_data_exists():
    """Check if data exists, if not, collect it"""
    if not os.path.exists('data/raw/SPY_stock.csv'):
        with st.spinner("📥 Collecting data for the first time... This may take 1-2 minutes"):
            try:
                # Import and run data collection
                sys.path.append('src/data')
                from collect_all_assets import collect_all_assets
                collect_all_assets(start_date='2020-01-01')
                st.success("✅ Data collection complete!")
            except Exception as e:
                st.error(f"Error collecting data: {e}")
                st.stop()

# Run data collection check before anything else
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
        
        # Categorize by suffix
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

# Flatten assets for selection
all_assets = {}
for category, assets_dict in ASSETS.items():
    all_assets.update(assets_dict)

# Sort for better UX
all_assets = dict(sorted(all_assets.items()))

# Asset pair selection with categories and search
st.sidebar.subheader("Select Asset Pair")

# Asset 1 selection
st.sidebar.markdown("**Asset 1**")
category_filter_1 = st.sidebar.selectbox(
    "Category",
    ["All Assets", "Equities", "ETFs", "Commodities", "FX", "Rates", "Economic"],
    key="cat1"
)

# Filter assets based on category for Asset 1
if category_filter_1 == "All Assets":
    filtered_assets_1 = all_assets
else:
    filtered_assets_1 = ASSETS[category_filter_1]
    filtered_assets_1 = dict(sorted(filtered_assets_1.items()))

# Search for Asset 1
search_term_1 = st.sidebar.text_input("🔍 Search", "", key="search1")
if search_term_1:
    filtered_assets_1 = {k: v for k, v in filtered_assets_1.items() if search_term_1.upper() in k.upper()}

st.sidebar.caption(f"Showing {len(filtered_assets_1)} assets")

if len(filtered_assets_1) > 0:
    asset1_name = st.sidebar.selectbox(
        "Select Asset 1", 
        list(filtered_assets_1.keys()), 
        index=0,
        label_visibility="collapsed"
    )
else:
    st.error("No assets found")
    st.stop()

st.sidebar.markdown("---")

# Asset 2 selection
st.sidebar.markdown("**Asset 2**")
category_filter_2 = st.sidebar.selectbox(
    "Category",
    ["All Assets", "Equities", "ETFs", "Commodities", "FX", "Rates", "Economic"],
    key="cat2"
)

# Filter assets based on category for Asset 2
if category_filter_2 == "All Assets":
    filtered_assets_2 = all_assets
else:
    filtered_assets_2 = ASSETS[category_filter_2]
    filtered_assets_2 = dict(sorted(filtered_assets_2.items()))

# Remove Asset 1 from Asset 2 options
filtered_assets_2 = {k: v for k, v in filtered_assets_2.items() if k != asset1_name}

# Search for Asset 2
search_term_2 = st.sidebar.text_input("🔍 Search", "", key="search2")
if search_term_2:
    filtered_assets_2 = {k: v for k, v in filtered_assets_2.items() if search_term_2.upper() in k.upper()}

st.sidebar.caption(f"Showing {len(filtered_assets_2)} assets")

if len(filtered_assets_2) > 0:
    asset2_name = st.sidebar.selectbox(
        "Select Asset 2", 
        list(filtered_assets_2.keys()), 
        index=0,
        label_visibility="collapsed"
    )
else:
    st.error("Need at least 2 assets to analyze correlation")
    st.stop()

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")
window = st.sidebar.selectbox("Rolling Window", [30, 60, 90, 120], index=2)
lookback = st.sidebar.selectbox("Historical Baseline", [180, 252, 365], index=1)
threshold = st.sidebar.slider("Anomaly Threshold (Std Dev)", 1.0, 3.0, 1.5, 0.1)

# Load data
@st.cache_data
def load_data(filepath):
    return load_asset_data(filepath)

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["📈 Pair Analysis", "🚨 Top Anomalies", "🔥 Correlation Matrix"])

# TAB 1: Pair Analysis
with tab1:
    try:
        asset1_data = load_data(filtered_assets_1[asset1_name])
        asset2_data = load_data(filtered_assets_2[asset2_name])
        
        # Calculate rolling correlation
        rolling_corr = calculate_rolling_correlation(asset1_data, asset2_data, window=window)
        
        # Calculate statistics
        stats = calculate_correlation_statistics(rolling_corr, lookback_period=lookback)
        
        # Detect anomalies
        anomalies, z_scores = detect_anomalies(rolling_corr, threshold=threshold, lookback_period=lookback)
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Correlation", f"{stats['current']:.3f}")
        
        with col2:
            st.metric("Historical Mean", f"{stats['mean']:.3f}")
        
        with col3:
            st.metric("Z-Score", f"{stats['z_score']:.2f}", 
                     delta=f"{'⚠️ Anomaly' if abs(stats['z_score']) > threshold else '✓ Normal'}")
        
        with col4:
            st.metric("Anomalies Detected", len(anomalies))
        
        # Rolling correlation chart
        st.subheader(f"{asset1_name} vs {asset2_name}: {window}-Day Rolling Correlation")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr.values,
            mode='lines',
            name=f'{window}-day Correlation',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_hline(y=stats['mean'], line_dash="dash", line_color="green",
                      annotation_text=f"Mean: {stats['mean']:.3f}")
        
        upper_band = stats['mean'] + threshold * stats['std']
        lower_band = stats['mean'] - threshold * stats['std']
        
        fig.add_hline(y=upper_band, line_dash="dot", line_color="orange",
                      annotation_text=f"+{threshold} Std")
        fig.add_hline(y=lower_band, line_dash="dot", line_color="orange",
                      annotation_text=f"-{threshold} Std")
        
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
        # ML Prediction Section
        st.markdown("---")
        st.subheader("🤖 ML Correlation Forecast")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("**Prediction Settings**")
            forecast_days = st.selectbox("Forecast Period", [7, 14, 30, 60, 90], index=2)
            model_type = st.selectbox("ML Model", ["linear"], index=0)
            st.caption("More models coming soon!")
        
        with col1:
            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner("Training ML model and generating predictions..."):
                    try:
                        sys.path.append('src/ml')
                        from correlation_predictor import predict_correlation
                        
                        # Generate predictions
                        ml_result = predict_correlation(
                            asset1_data, 
                            asset2_data, 
                            window=window,
                            days_ahead=forecast_days,
                            model_type=model_type
                        )
                        
                        # Display metrics
                        st.success("✅ Model trained successfully!")
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        with metric_col1:
                            st.metric("Current Correlation", f"{ml_result['current_correlation']:.3f}")
                        with metric_col2:
                            predicted_value = ml_result['predictions'][-1]
                            change = predicted_value - ml_result['current_correlation']
                            st.metric(
                                f"Predicted ({forecast_days}d)", 
                                f"{predicted_value:.3f}",
                                f"{change:+.3f}"
                            )
                        with metric_col3:
                            st.metric("Model R² Score", f"{ml_result['metrics']['test_r2']:.3f}")
                        
                        # Plot predictions
                        fig_pred = go.Figure()
                        
                        # Historical correlation
                        fig_pred.add_trace(go.Scatter(
                            x=rolling_corr.index[-60:],
                            y=rolling_corr.values[-60:],
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Predictions
                        fig_pred.add_trace(go.Scatter(
                            x=ml_result['dates'],
                            y=ml_result['predictions'],
                            mode='lines',
                            name='Predicted',
                            line=dict(color='orange', width=2, dash='dash')
                        ))
                        
                        # Confidence interval
                        if ml_result['confidence']:
                            std_error = ml_result['confidence']['std_error']
                            upper_bound = ml_result['predictions'] + 2 * std_error
                            lower_bound = ml_result['predictions'] - 2 * std_error
                            
                            fig_pred.add_trace(go.Scatter(
                                x=ml_result['dates'],
                                y=upper_bound,
                                mode='lines',
                                name='Upper Bound (95%)',
                                line=dict(width=0),
                                showlegend=False
                            ))
                            
                            fig_pred.add_trace(go.Scatter(
                                x=ml_result['dates'],
                                y=lower_bound,
                                mode='lines',
                                name='Confidence Interval',
                                fill='tonexty',
                                fillcolor='rgba(255, 165, 0, 0.2)',
                                line=dict(width=0)
                            ))
                        
                        fig_pred.update_layout(
                            title=f"Correlation Forecast: {asset1_name} vs {asset2_name}",
                            xaxis_title="Date",
                            yaxis_title="Correlation",
                            yaxis_range=[-1, 1],
                            hovermode='x unified',
                            height=400
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Model info
                        with st.expander("📊 Model Performance Details"):
                            st.markdown(f"""
                            **Model Type:** {ml_result['model_type'].title()}
                            
                            **Training Metrics:**
                            - Train R² Score: {ml_result['metrics']['train_r2']:.4f}
                            - Test R² Score: {ml_result['metrics']['test_r2']:.4f}
                            - Train MSE: {ml_result['metrics']['train_mse']:.4f}
                            - Test MSE: {ml_result['metrics']['test_mse']:.4f}
                            
                            **Prediction Confidence:**
                            - Mean Absolute Error: ±{ml_result['confidence']['mean_error']:.3f}
                            - Standard Deviation: ±{ml_result['confidence']['std_error']:.3f}
                            
                            *The model uses the past 30 days of correlation data to predict future values.*
                            """)
                        
                    except Exception as e:
                        st.error(f"Error generating predictions: {e}")
                        st.info("Make sure you have enough historical data (100+ days)")
    except Exception as e:
        st.error(f"Error loading data: {e}")

# TAB 2: Top Anomalies Scanner
with tab2:
    st.subheader("🚨 Most Anomalous Correlations")
    st.markdown("Automatically scanning asset pairs to find the most significant correlation anomalies...")
    
    # Scanner controls
    col1, col2 = st.columns(2)
    with col1:
        scan_pairs = st.number_input("Max pairs to scan", 100, 1000, 500, 100)
    with col2:
        top_n = st.number_input("Show top N anomalies", 5, 50, 20, 5)
    
    if st.button("🔍 Scan for Anomalies", type="primary"):
        with st.spinner(f"Scanning up to {scan_pairs} asset pairs..."):
            try:
                sys.path.append('src/analysis')
                from anomaly_scanner import scan_all_correlations
                
                anomalies_df = scan_all_correlations(
                    window=window,
                    lookback=lookback,
                    threshold=threshold,
                    max_pairs=scan_pairs
                )
                
                if len(anomalies_df) > 0:
                    st.success(f"Found {len(anomalies_df)} anomalous pairs!")
                    
                    # Display top anomalies
                    top_anomalies = anomalies_df.head(top_n)
                    
                    st.dataframe(
                        top_anomalies.style.format({
                            'Current Correlation': '{:.3f}',
                            'Historical Mean': '{:.3f}',
                            'Z-Score': '{:.2f}',
                            'Std Dev': '{:.3f}'
                        }).background_gradient(subset=['Z-Score'], cmap='RdYlGn_r'),
                        use_container_width=True
                    )
                    
                    # Download button
                    csv = anomalies_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Full Results (CSV)",
                        data=csv,
                        file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("✅ No significant anomalies detected in scanned pairs")
                    
            except Exception as e:
                st.error(f"Error scanning anomalies: {e}")
    else:
        st.info("👆 Click the button above to start scanning")

# TAB 3: Correlation Matrix
with tab3:
    st.subheader("Cross-Asset Correlation Matrix")
    
    with st.spinner("Calculating correlations..."):
        try:
            matrix_assets = {
                'SPY': 'data/raw/SPY_etf.csv',
                'GCF': 'data/raw/GCF_commodity.csv',
                'CLF': 'data/raw/CLF_commodity.csv',
                'DX-Y_NYB': 'data/raw/DX-Y_NYB_fx.csv',
                'TNX': 'data/raw/TNX_rates.csv',
                'EURUSDX': 'data/raw/EURUSDX_fx.csv'
            }
            
            df = load_multiple_assets(matrix_assets)
            current_corr = df.tail(90).corr()
            
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