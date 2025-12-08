def get_custom_css():
    """Return custom CSS styling for the app"""
    return """
    <style>
        /* Main container - remove padding for full-width header */
        .main .block-container {
            padding-top: 1rem;
            max-width: 100%;
        }
        
        /* Full-width header band */
        .main-header {
            font-size: 1.1rem;
            line-height: 1.6;
            padding: 2rem 3rem;
            background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
            color: white;
            border-radius: 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            margin-left: -5rem;
            margin-right: -5rem;
            margin-bottom: 2rem;
        }
        
        /* Metric cards - uniform sizing */
        .stMetric {
            background-color: #F5F7FA;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1565C0;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            height: 120px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        
        .stMetric label {
            font-size: 0.9rem;
            color: #546E7A;
            font-weight: 500;
        }
        
        .stMetric [data-testid="stMetricValue"] {
            font-size: 2rem;
            font-weight: 700;
            color: #1565C0;
        }
        
        /* Section headers */
        h1 {
            color: #263238;
            font-weight: 700;
        }
        
        h2, h3 {
            color: #1565C0;
            border-bottom: 2px solid #E3F2FD;
            padding-bottom: 0.5rem;
            margin-top: 2rem;
        }
        
        /* Buttons */
        .stButton>button {
            background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.75rem 2rem;
            font-weight: 600;
            box-shadow: 0 2px 4px rgba(21, 101, 192, 0.3);
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton>button:hover {
            box-shadow: 0 4px 8px rgba(21, 101, 192, 0.4);
            transform: translateY(-1px);
        }
        
        /* Tabs - add padding to tab text */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background-color: #F5F7FA;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            background-color: transparent;
            border-radius: 0.5rem;
            color: #546E7A;
            font-weight: 500;
            padding: 0 1.5rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
            color: white;
            padding: 0 1.5rem;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F5F7FA 0%, #FFFFFF 100%);
            border-right: 2px solid #E3F2FD;
        }
        
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #1565C0;
            border: none;
            font-size: 1.1rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        /* Info/Alert boxes */
        .stAlert {
            border-radius: 0.5rem;
            border-left: 4px solid #1565C0;
        }
        
        /* Dataframes */
        .dataframe {
            border-radius: 0.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }
        
        /* Selectbox styling */
        .stSelectbox label {
            color: #546E7A;
            font-weight: 500;
            font-size: 0.9rem;
        }
    </style>
    """

def get_header_html():
    """Return HTML for the main header"""
    return """
    <div class="main-header">
    Track rolling correlations across equities, FX, commodities, and rates. 
    Detect when relationships break from historical norms to identify potential trading opportunities.
    </div>
    """