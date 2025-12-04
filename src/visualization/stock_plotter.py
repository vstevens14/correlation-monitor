import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_price_and_ma(df, symbol, save_path=None):
    """Plot stock price with moving averages"""
    fig, ax = plt.subplots()
    
    ax.plot(df.index, df['close'], label='Close Price', linewidth=2)
    ax.plot(df.index, df['MA_20'], label='20-day MA', linestyle='--', alpha=0.7)
    ax.plot(df.index, df['MA_50'], label='50-day MA', linestyle='--', alpha=0.7)
    
    ax.set_title(f'{symbol} Stock Price with Moving Averages')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()

def plot_returns(df, symbol, save_path=None):
    """Plot cumulative returns"""
    fig, ax = plt.subplots()
    
    ax.plot(df.index, df['cumulative_return'] * 100, linewidth=2, color='green')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_title(f'{symbol} Cumulative Returns')
    ax.set_xlabel('Date')
    ax.set_ylabel('Return (%)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Test the plotter
    print("Loading processed data...")
    df = pd.read_csv('data/processed/AAPL_processed.csv', index_col=0, parse_dates=True)
    
    print("Creating visualizations...")
    plot_price_and_ma(df, 'AAPL', save_path='docs/AAPL_price_chart.png')
    plot_returns(df, 'AAPL', save_path='docs/AAPL_returns_chart.png')