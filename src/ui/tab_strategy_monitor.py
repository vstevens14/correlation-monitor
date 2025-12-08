import streamlit as st
from datetime import datetime
import sys

sys.path.append('src/analysis')

def render_strategy_monitor_tab(window, lookback, threshold):
    """Render the Strategy Risk Monitor tab"""
    
    st.subheader("Trading Strategy Risk Monitor")
    st.markdown("""
    Automatically discovers correlation-based trading strategies and monitors them for risk.
    Identifies when correlation assumptions break down before strategies fail.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_corr = st.slider("Min Correlation", 0.5, 0.9, 0.7, 0.05)
    with col2:
        max_discover = st.number_input("Max pairs to discover", 100, 500, 200, 50)
    with col3:
        health_threshold = st.slider("Alert threshold", 20, 80, 40, 10)
    
    if st.button("Discover & Monitor Strategies", type="primary", use_container_width=True):
        with st.spinner("Discovering stable correlation strategies..."):
            try:
                from strategy_monitor import discover_stable_correlations, monitor_correlation_strategies, get_critical_alerts
                
                # Discover stable pairs
                stable_pairs = discover_stable_correlations(
                    min_correlation=min_corr,
                    max_pairs=max_discover,
                    sample_recent_days=90
                )
                
                if len(stable_pairs) > 0:
                    st.success(f"Discovered {len(stable_pairs)} stable correlation strategies!")
                    
                    # Show discovered strategies
                    with st.expander(f"View All {len(stable_pairs)} Discovered Strategies"):
                        st.dataframe(
                            stable_pairs[['Asset 1', 'Category 1', 'Asset 2', 'Category 2', 
                                        'Avg Correlation', 'Std Dev', 'Stability Score']].style.format({
                                'Avg Correlation': '{:.3f}',
                                'Std Dev': '{:.3f}',
                                'Stability Score': '{:.2f}'
                            }),
                            use_container_width=True
                        )
                    
                    # Monitor strategies
                    with st.spinner("Monitoring strategies for risk..."):
                        monitored = monitor_correlation_strategies(
                            stable_pairs,
                            window=window,
                            lookback=lookback,
                            threshold=threshold,
                            predict_future=True,
                            forecast_days=30
                        )
                        
                        if len(monitored) > 0:
                            # Summary metrics
                            avg_health = monitored['Health Score'].mean()
                            critical_count = len(monitored[monitored['Health Score'] < 40])
                            warning_count = len(monitored[(monitored['Health Score'] >= 40) & (monitored['Health Score'] < 70)])
                            healthy_count = len(monitored[monitored['Health Score'] >= 70])
                            
                            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                            with metric_col1:
                                st.metric("Avg Health Score", f"{avg_health:.0f}/100")
                            with metric_col2:
                                st.metric("Healthy", healthy_count)
                            with metric_col3:
                                st.metric("Warning", warning_count)
                            with metric_col4:
                                st.metric("Critical", critical_count)
                            
                            # Critical alerts
                            alerts = get_critical_alerts(monitored, health_threshold=health_threshold)
                            
                            if len(alerts) > 0:
                                st.markdown("### Critical Alerts")
                                st.dataframe(alerts, use_container_width=True)
                            else:
                                st.success("No critical alerts - all strategies within acceptable risk levels")
                            
                            # Full strategy table
                            st.markdown("### Strategy Health Dashboard")
                            
                            # Color code by health
                            def color_health(val):
                                if val >= 70:
                                    return 'background-color: #90EE90'
                                elif val >= 40:
                                    return 'background-color: #FFD700'
                                else:
                                    return 'background-color: #FF6B6B'
                            
                            display_cols = ['Asset 1', 'Asset 2', 'Pair Type', 'Expected Corr', 
                                          'Current Corr', 'Deviation', 'Health Score', 'Status']
                            
                            if 'Predicted Corr' in monitored.columns:
                                display_cols.extend(['Predicted Corr', 'Forecast Alert'])
                            
                            st.dataframe(
                                monitored[display_cols].style.format({
                                    'Expected Corr': '{:.3f}',
                                    'Current Corr': '{:.3f}',
                                    'Deviation': '{:.3f}',
                                    'Predicted Corr': '{:.3f}',
                                    'Health Score': '{:.0f}'
                                }).applymap(color_health, subset=['Health Score']),
                                use_container_width=True
                            )
                            
                            # Download button
                            csv = monitored.to_csv(index=False)
                            st.download_button(
                                label="Download Strategy Report (CSV)",
                                data=csv,
                                file_name=f"strategy_monitor_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("Could not monitor strategies")
                
                else:
                    st.info(f"No stable correlations found with minimum correlation of {min_corr}")
                    
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
    else:
        st.info("Click the button above to discover and monitor correlation strategies")