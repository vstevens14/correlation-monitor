import streamlit as st
import plotly.express as px
import sys

sys.path.append('src/analysis')
from correlation_matrix import load_multiple_assets

def render_correlation_matrix_tab():
    """Render the Correlation Matrix tab"""
    
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