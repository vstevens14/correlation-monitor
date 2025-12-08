import streamlit as st
from datetime import datetime
import sys

sys.path.append('src/analysis')

def render_anomalies_tab(window, lookback, threshold):
    """Render the Top Anomalies tab"""
    
    st.subheader("Most Anomalous Correlations")
    st.markdown("Automatically scanning asset pairs to find the most significant correlation anomalies...")
    
    # Scanner controls
    col1, col2 = st.columns(2)
    with col1:
        scan_pairs = st.number_input("Max pairs to scan", 100, 1000, 500, 100)
    with col2:
        top_n = st.number_input("Show top N anomalies", 5, 50, 20, 5)
    
    if st.button("Scan for Anomalies", type="primary", use_container_width=True):
        with st.spinner(f"Scanning up to {scan_pairs} asset pairs..."):
            try:
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
                        label="Download Full Results (CSV)",
                        data=csv,
                        file_name=f"anomalies_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No significant anomalies detected in scanned pairs")
                    
            except Exception as e:
                st.error(f"Error scanning anomalies: {e}")
    else:
        st.info("Click the button above to start scanning")