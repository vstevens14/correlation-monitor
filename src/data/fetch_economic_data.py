import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def fetch_fred_data(series_id, start_date=None):
    """
    Fetch economic data from FRED
    series_id: FRED series code (e.g., 'UNRATE' for unemployment)
    start_date: Start date in 'YYYY-MM-DD' format (optional)
    """
    api_key = os.getenv('FRED_API_KEY')
    fred = Fred(api_key=api_key)
    
    data = fred.get_series(series_id, observation_start=start_date)
    return data

if __name__ == "__main__":
    # Test with unemployment rate
    print("Fetching US Unemployment Rate...")
    unemployment = fetch_fred_data('UNRATE', start_date='2020-01-01')
    print(unemployment.tail())