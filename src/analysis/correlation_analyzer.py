import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_align_data(stock_file, economic_file):
    """Load stock and economic data and align by date"""
    stock_df = pd.read_csv(stock_file)
    
    # Set Date column as index and convert to datetime
    if 'Date' in stock_df.columns:
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], utc=True)
        stock_df = stock_df.set_index('Date')
    else:
        stock_df.index = pd.to_datetime(stock_df.index, utc=True)
    
    # Resample stock data to monthly (end of month)
    stock_monthly = stock_df['close'].resample('ME').last()
    stock_monthly = stock_monthly.to_frame(name='close')
    
    # Convert to start of month to match economic data
    stock_monthly.index = stock_monthly.index.to_period('M').to_timestamp()
    stock_monthly.index = stock_monthly.index.tz_localize(None)
    
    econ_df = pd.read_csv(economic_file, index_col=0)
    econ_df.index = pd.to_datetime(econ_df.index)
    
    # Get the economic indicator name from filename
    econ_name = economic_file.split('/')[-1].replace('_economic_data.csv', '')
    econ_df.columns = [econ_name]
    
    # Align dates (inner join)
    combined = stock_monthly.join(econ_df, how='inner')
    combined = combined.dropna()
    
    return combined

def calculate_correlation(df, stock_column='close', econ_column=None):
    """Calculate correlation between stock price and economic indicator"""
    if econ_column is None:
        econ_column = df.columns[1]  # Use second column
    
    correlation = df[stock_column].corr(df[econ_column])
    
    # Statistical significance test
    n = len(df)
    if n > 2:
        t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
    else:
        p_value = 1.0
    
    return {
        'correlation': correlation,
        'p_value': p_value,
        'sample_size': n,
        'significant': p_value < 0.05
    }

def plot_correlation(df, stock_col='close', econ_col=None, 
                    stock_name='Stock', econ_name='Economic Indicator', save_path=None):
    """Create scatter plot showing correlation"""
    if econ_col is None:
        econ_col = df.columns[1]
        econ_name = econ_col.replace('_', ' ').title()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(df[econ_col], df[stock_col], alpha=0.6, s=50)
    
    # Add trend line
    z = np.polyfit(df[econ_col], df[stock_col], 1)
    p = np.poly1d(z)
    ax.plot(df[econ_col], p(df[econ_col]), "r--", alpha=0.8, linewidth=2, label='Trend')
    
    # Calculate and display correlation
    corr = df[stock_col].corr(df[econ_col])
    ax.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(econ_name, fontsize=12)
    ax.set_ylabel(f'{stock_name} Price ($)', fontsize=12)
    ax.set_title(f'{stock_name} vs {econ_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("Loading data...")
    df = load_and_align_data(
        'data/raw/AAPL_stock_data.csv',
        'data/raw/unemployment_rate_economic_data.csv'
    )
    
    print(f"\nData aligned: {len(df)} matching months")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    print("\nCalculating correlation...")
    result = calculate_correlation(df)
    
    print(f"\nCorrelation Analysis:")
    print(f"Correlation coefficient: {result['correlation']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Sample size: {result['sample_size']}")
    print(f"Statistically significant: {result['significant']}")
    
    print("\nCreating visualization...")
    plot_correlation(df, 'close', 'unemployment_rate', 
                    'AAPL', 'US Unemployment Rate',
                    save_path='docs/correlation_plot.png')