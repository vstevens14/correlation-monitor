"""
UI components for the Cross-Asset Correlation Anomaly Detector
"""

from .styling import get_custom_css, get_header_html
from .sidebar import render_sidebar
from .tab_pair_analysis import render_pair_analysis_tab
from .tab_anomalies import render_anomalies_tab
from .tab_strategy_monitor import render_strategy_monitor_tab
from .tab_correlation_matrix import render_correlation_matrix_tab
from .data_refresh import render_data_refresh_section

__all__ = [
    'get_custom_css',
    'get_header_html', 
    'render_sidebar',
    'render_pair_analysis_tab',
    'render_anomalies_tab',
    'render_strategy_monitor_tab',
    'render_correlation_matrix_tab'
    'render_data_refresh_section'
]