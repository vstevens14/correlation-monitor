import streamlit as st
import plotly.graph_objects as go
import sys

sys.path.append('src/analysis')
from rolling_correlation import calculate_rolling_correlation, calculate_correlation_statistics, detect_anomalies

def render_pair_analysis_tab(asset1_data, asset2_data, asset1_name, asset2_name, window, lookback, threshold, rolling_corr):
    """Render the Pair Analysis tab"""
    
    try:
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
            status = "ANOMALY" if abs(stats['z_score']) > threshold else "NORMAL"
            st.metric("Z-Score", f"{stats['z_score']:.2f}", delta=status)
        
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
            st.subheader("Recent Anomalies")
            recent_anomalies = anomalies.tail(10).to_frame()
            recent_anomalies.columns = ['Correlation']
            recent_anomalies['Z-Score'] = z_scores.loc[recent_anomalies.index]
            recent_anomalies = recent_anomalies.sort_index(ascending=False)
            st.dataframe(recent_anomalies.style.format({'Correlation': '{:.3f}', 'Z-Score': '{:.2f}'}))
        
        # Interpretation
        st.subheader("Interpretation")
        
        if abs(stats['z_score']) > threshold:
            st.warning(f"""
            **Anomaly Detected**
            
            The current correlation ({stats['current']:.3f}) is {abs(stats['z_score']):.2f} standard deviations away 
            from the historical mean ({stats['mean']:.3f}).
            
            This suggests a **regime change** in the relationship between {asset1_name} and {asset2_name}.
            """)
        else:
            st.success(f"""
            **Normal Correlation**
            
            The current correlation ({stats['current']:.3f}) is within normal range of the historical mean ({stats['mean']:.3f}).
            
            Z-Score: {stats['z_score']:.2f} (threshold: ±{threshold})
            """)
        
        # ML Prediction Section
        render_ml_forecast(asset1_data, asset2_data, asset1_name, asset2_name, window, rolling_corr)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")


def render_ml_forecast(asset1_data, asset2_data, asset1_name, asset2_name, window, rolling_corr):
    """Render the ML forecast section"""
    
    st.markdown("---")
    st.subheader("ML Correlation Forecast")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("**Prediction Settings**")
        forecast_days = st.selectbox("Forecast Period", [7, 14, 30, 60, 90], index=2)
        model_type = st.selectbox("ML Model", ["linear", "lstm"], index=0)
        
        if model_type == "lstm":
            st.caption("Deep learning model (slower but more accurate)")
        else:
            st.caption("Fast baseline model")
    
    with col1:
        if st.button("Generate Forecast", type="primary", use_container_width=True):
            with st.spinner("Training ML model and generating predictions..."):
                try:
                    sys.path.insert(0, 'src/ml')
                    
                    if model_type == "linear":
                        from correlation_predictor import predict_correlation
                        
                        ml_result = predict_correlation(
                            asset1_data, 
                            asset2_data, 
                            window=window,
                            days_ahead=forecast_days,
                            model_type='linear'
                        )
                    else:  # LSTM
                        from lstm_predictor import predict_correlation_lstm
                        
                        ml_result = predict_correlation_lstm(
                            asset1_data, 
                            asset2_data, 
                            window=window,
                            days_ahead=forecast_days,
                            lookback=30,
                            epochs=50
                        )
                    
                    # Display metrics
                    st.success("Model trained successfully!")
                    
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
                    
                    fig_pred.add_trace(go.Scatter(
                        x=rolling_corr.index[-60:],
                        y=rolling_corr.values[-60:],
                        mode='lines',
                        name='Historical',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_pred.add_trace(go.Scatter(
                        x=ml_result['dates'],
                        y=ml_result['predictions'],
                        mode='lines',
                        name='Predicted',
                        line=dict(color='orange', width=2, dash='dash')
                    ))
                    
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
                    with st.expander("Model Performance Details"):
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