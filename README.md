# Cross-Asset Correlation Anomaly Detector

A real-time correlation monitoring system that tracks relationships across equities, FX, commodities, and rates to identify regime changes and potential trading opportunities.

## Features

- **Multi-Asset Coverage**: Analyze correlations across 15+ assets spanning equities, foreign exchange, commodities, and rates
- **Rolling Correlation Analysis**: Configurable time windows (30/60/90/120 days) to capture different market dynamics
- **Anomaly Detection**: Z-score based alerts when correlations deviate significantly from historical norms
- **Interactive Dashboard**: Web-based interface for real-time analysis and visualization
- **Correlation Matrix**: Visual heatmap showing all asset relationships at a glance

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/correlation-monitor.git
cd correlation-monitor
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys:
Create a `.env` file in the project root:
```
ALPHA_VANTAGE_API_KEY=your_key_here
FRED_API_KEY=your_key_here
```

5. Collect data:
```bash
python src/data/collect_all_assets.py
```

## Usage

Run the Streamlit dashboard:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure
```
correlation-monitor/
├── app.py                          # Main Streamlit dashboard
├── src/
│   ├── data/
│   │   ├── fetch_stock_data.py
│   │   ├── fetch_economic_data.py
│   │   ├── fetch_multi_asset_data.py
│   │   └── collect_all_assets.py
│   ├── analysis/
│   │   ├── rolling_correlation.py
│   │   └── correlation_matrix.py
│   └── visualization/
│       └── stock_plotter.py
├── data/
│   ├── raw/                        # Raw data files
│   └── processed/                  # Processed data
├── docs/                           # Documentation and charts
├── requirements.txt
└── README.md
```

## Methodology

The tool uses rolling window correlation analysis to track how asset relationships evolve over time. Anomalies are detected using z-score thresholds, comparing current correlations against historical baselines (typically 1-year lookback).

**Key Parameters:**
- Rolling Windows: 30, 60, 90, or 120 days
- Anomaly Threshold: 1.5 standard deviations (configurable)
- Historical Baseline: 180-365 days

## Technologies

- **Python 3.12+**
- **Streamlit**: Interactive web framework
- **Pandas/NumPy**: Data analysis
- **Plotly**: Interactive visualizations
- **yfinance**: Financial data API
- **FRED API**: Economic indicators

## License

MIT License

## Author

Victor - [Your LinkedIn/GitHub]