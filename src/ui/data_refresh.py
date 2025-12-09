import streamlit as st
from datetime import datetime
import sys

sys.path.append('src/data')

def render_data_refresh_section():
    """Render data refresh controls in sidebar"""
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Management**")
    
    # Show last data collection date
    try:
        import os
        from pathlib import Path
        
        spy_file = Path('data/raw/SPY_etf.csv')
        if spy_file.exists():
            mod_time = datetime.fromtimestamp(spy_file.stat().st_mtime)
            st.sidebar.caption(f"Last updated: {mod_time.strftime('%Y-%m-%d %H:%M')}")
    except:
        pass
    
    if st.sidebar.button("Refresh Data", help="Re-download all asset data with latest prices"):
        with st.spinner("Refreshing data... This may take 5-10 minutes"):
            try:
                from collect_all_assets import collect_all_assets
                collect_all_assets(start_date='2020-01-01', num_stocks=100)
                st.sidebar.success("Data refreshed successfully!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error refreshing data: {e}")