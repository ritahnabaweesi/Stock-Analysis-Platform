import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from ta import momentum, trend
from datetime import datetime, timedelta
import warnings
import requests
import time
warnings.filterwarnings('ignore')
import streamlit as st



def create_realistic_stock_data(ticker, start_price=150, days=63):
    """Fallback: create realistic stock data"""
    import numpy as np
    from datetime import datetime, timedelta
    
    np.random.seed(hash(ticker) % 1000)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    returns = np.random.normal(0, 0.02, days)
    prices = start_price * (1 + returns).cumprod()
    
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, days))
    data['High'] = data[['Open', 'Close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.01, days)))
    data['Low'] = data[['Open', 'Close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.01, days)))
    data['Volume'] = np.random.randint(1000000, 50000000, days)
    data.ffill(inplace=True)
    data = data[data.index.dayofweek < 5]
    
    return data

#get_stock_data
def get_stock_data(ticker, period="3mo"):
    """Safely fetch stock data and return with simple column names"""

    for i in range(3):
        try:
            data = yf.download(ticker, period=period, progress=False)
            if not data.empty:
                # Handle MultiIndex columns explicitly
                if hasattr(data.columns, 'levels') and len(data.columns.levels) > 1:
                    # We have MultiIndex columns - extract just the price type level
                    simple_columns = [col[0] for col in data.columns]
                    data.columns = simple_columns
                return data
        except:
            time.sleep(2)
    print(f"Could not fetch {ticker}, using simulated data")
    return create_realistic_stock_data(ticker)

#calculate_technical_indicators()
def calculate_technical_indicators(data):
    """Calculate RSI, SMA, EMA and other indicators"""
    df = data.copy().reset_index() if hasattr(data.index, 'levels') else data.copy()

    #Simple moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    #Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    #RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta>0,0)).rolling(window=14).mean()
    loss = (delta.where(delta<0,0)).rolling(window=14).mean()
    rs = gain/loss
    df['RSI'] = 100 - (100/(1+rs))

    #MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    #bollinger bands
    bb_middle = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    bb_upper = bb_middle + (bb_std * 2)
    bb_lower = bb_middle -(bb_std * 2)
    
    #Assign to Dataframe
    df['BB_Middle'] = bb_middle
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    
    denominator = bb_upper - bb_lower
    denominator = denominator.replace(0, np.nan)
    bb_percent_b = (df['Close'] - bb_lower) / denominator
    
    df['BB_%B'] = bb_percent_b
    df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
    
    #Bollinger band signals
    df['BB_Signal'] = 'Neutral'
    df.loc[df['Close'] >= bb_upper, 'BB_Signal'] = 'Overbought'
    df.loc[df['Close'] <= bb_lower, 'BB_Signal'] = 'Oversold'
    
    #Squeeze indicator
    if len(df) > 50:
        squeeze_threshold = df['BB_Width'].rolling(50).quantile(0.25)
        df['BB_Squeeze'] = df['BB_Width'] < squeeze_threshold
    else:
        df['BB_Squeeze'] = False
    
    #set index back if we reset it
    if hasattr(data.index, 'levels'):
        df = df.set_index(data.index.names)
    return df

#compare_two_stocks()
def compare_two_stocks(ticker1, ticker2, period="3mo"):
    """Compare two stocks side by side"""
    #fetch data for both stocks
    stock1_data = get_stock_data(ticker1, period)
    stock2_data = get_stock_data(ticker2, period)

    #Calculate indicators
    stock1_tech = calculate_technical_indicators(stock1_data)
    stock2_tech = calculate_technical_indicators(stock2_data)

    return stock1_tech, stock2_tech

#calculate_statistics
def calculate_statistics(stock_data, ticker):
    """Calculate key statistics for a stock"""
    stats = {}

    #Basic stats
    stats['Ticker'] = ticker
    stats['Current Price'] = float(stock_data['Close'].iloc[-1]) #force to float
    stats['Price Change (%)'] = float(((stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[0])
                                 /stock_data['Close'].iloc[0])*100)

    #Volatility (standard deviation of returns)
    returns = stock_data['Close'].pct_change().dropna()
    stats['Volatility (%)'] = float(returns.std() *100*np.sqrt(252)) #Annualized

    #RSI status
    current_rsi = float(stock_data['RSI'].iloc[-1])
    if current_rsi > 70:
        rsi_status = "Overbought"
    elif current_rsi < 30:
        rsi_status = "Oversold"
    else:
        rsi_status = "Neutral"
    stats['RSI'] = f"{current_rsi:.1f} ({rsi_status})"

    #Moving average signals
    current_close = float(stock_data['Close'].iloc[-1])
    current_sma_20 = float(stock_data['SMA_20'].iloc[-1])
    
    if current_close > current_sma_20:
        ma_signal = "Above SMA20"
    else:
        ma_signal = "Below SMA20"
    stats['MA Signal'] = ma_signal
    
    #Bollinger Band statistics
    current_bb_signal = stock_data['BB_Signal'].iloc[-1]
    current_bb_percentb = float(stock_data['BB_%B'].iloc[-1])
    current_bb_width = float(stock_data['BB_Width'].iloc[-1])
    bb_squeeze = bool(stock_data['BB_Squeeze'].iloc[-1])
    
    #Bollinger Band interpretation
    if current_bb_signal == 'Overbought':
        bb_interpretation = "Price near upper band - potential resistance"
    elif current_bb_signal == 'Oversold':
        bb_interpretation = "Price near lower band - potential support"
    else:
        bb_interpretation = "Price within bands - normal range"
    
    stats['BB Position'] = f"{current_bb_signal} (%B: {current_bb_percentb:.2f})"
    stats['BB Width'] = f"{current_bb_width:.4f}"
    stats['BB Squeeze'] = "Yes" if bb_squeeze else "No"
    stats['BB Interpretation'] = bb_interpretation

    return stats
    
#display_statistics ()
def display_statistics_only(stock1_data, stock2_data, ticker1, ticker2):
    """Display only statistics in a clean format"""
    stats1 = calculate_statistics(stock1_data, ticker1)
    stats2 = calculate_statistics(stock2_data, ticker2)

    import pandas as pd
    stats_df = pd.DataFrame([stats1, stats2])

    print("COMPARISON STATISTICS")
    print("=" * 60)
    print(stats_df.to_string(index=False))
    print("=" * 60)

    return stats_df

#plot_price_comparison()
def plot_price_comparison(stock1_data, stock2_data, ticker1, ticker2, 
                         show_sma=True, show_ema=True, show_volume=True, show_bollinger=True):
    """Enhanced comparison plot with technical indicators and volume as lines"""
    
    # Determine grid layout
    nrows = 2 if show_volume else 1
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 5 * nrows))
    
    # Flatten axes for easier indexing
    if nrows == 2:
        ax1, ax2, ax3, ax4 = axes.flatten()
        ax_volume1, ax_volume2 = ax3, ax4
    else:
        ax1, ax2 = axes.flatten()
        ax_volume1, ax_volume2 = None, None
    
    # Plot 1: Stock 1 Price with Indicators
    ax1.plot(stock1_data.index, stock1_data['Close'], label=f'{ticker1} Close', 
             color='blue', linewidth=2)
    
    if show_sma:
        ax1.plot(stock1_data.index, stock1_data['SMA_20'], label=f'{ticker1} SMA 20',
                color='orange', linestyle='--', alpha=0.8)
        ax1.plot(stock1_data.index, stock1_data['SMA_50'], label=f'{ticker1} SMA 50',
                color='red', linestyle='--', alpha=0.8)
    
    if show_ema:
        ax1.plot(stock1_data.index, stock1_data['EMA_12'], label=f'{ticker1} EMA 12',
                color='green', linestyle='-.', alpha=0.8)
        ax1.plot(stock1_data.index, stock1_data['EMA_26'], label=f'{ticker1} EMA 26', 
                color='purple', linestyle='-.', alpha=0.8)
    
    # Add Bollinger Bands to stock 1 
    if show_bollinger and 'BB_Upper' in stock1_data.columns:
        ax1.fill_between(stock1_data.index, stock1_data['BB_Upper'], stock1_data['BB_Lower'],
                        alpha=0.2, color='gray', label=f'{ticker1} Bollinger Bands')
        ax1.plot(stock1_data.index, stock1_data['BB_Middle'], 
                color='gray', linestyle=':', alpha=0.7, label=f'{ticker1} BB Middle')
    
    # Add RSI as text annotation for stock 1
    current_rsi_1 = stock1_data['RSI'].iloc[-1]
    if current_rsi_1 > 70:
        rsi_color_1 = 'red'
        rsi_status_1 = 'Overbought'
    elif current_rsi_1 < 30:
        rsi_color_1 = 'green'
        rsi_status_1 = 'Oversold'
    else:
        rsi_color_1 = 'black'
        rsi_status_1 = 'Neutral'
    
    rsi_text_1 = f'RSI: {current_rsi_1:.1f} ({rsi_status_1})'
    ax1.text(0.98, 0.02, rsi_text_1, transform=ax1.transAxes, fontsize=12, horizontalalignment='right',
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=rsi_color_1),
             color=rsi_color_1, fontweight='bold')
    
    ax1.set_title(f'{ticker1} Price & Moving Averages')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stock 2 Price with Indicators
    ax2.plot(stock2_data.index, stock2_data['Close'], label=f'{ticker2} Close',
             color='darkred', linewidth=2)
    
    if show_sma:
        ax2.plot(stock2_data.index, stock2_data['SMA_20'], label=f'{ticker2} SMA 20',
                color='orange', linestyle='--', alpha=0.8)
        ax2.plot(stock2_data.index, stock2_data['SMA_50'], label=f'{ticker2} SMA 50',
                color='red', linestyle='--', alpha=0.8)
    
    if show_ema:
        ax2.plot(stock2_data.index, stock2_data['EMA_12'], label=f'{ticker2} EMA 12',
                color='green', linestyle='-.', alpha=0.8)
        ax2.plot(stock2_data.index, stock2_data['EMA_26'], label=f'{ticker2} EMA 26',
                color='purple', linestyle='-.', alpha=0.8)
    
    # Bollinger Bands for stock2
    if show_bollinger and 'BB_Upper' in stock2_data.columns:
        ax2.fill_between(stock2_data.index, stock2_data['BB_Upper'], stock2_data['BB_Lower'],
                        alpha=0.2, color='lightcoral', label=f'{ticker2} Bollinger Bands')
        ax2.plot(stock2_data.index, stock2_data['BB_Middle'], 
                color='darkred', linestyle=':', alpha=0.7, label=f'{ticker2} BB Middle')
    
    # Add RSI as text annotation for stock 2
    current_rsi_2 = stock2_data['RSI'].iloc[-1]
    if current_rsi_2 > 70:
        rsi_color_2 = 'red'
        rsi_status_2 = 'Overbought'
    elif current_rsi_2 < 30:
        rsi_color_2 = 'green'
        rsi_status_2 = 'Oversold'
    else:
        rsi_color_2 = 'black'
        rsi_status_2 = 'Neutral'
    
    rsi_text_2 = f'RSI: {current_rsi_2:.1f} ({rsi_status_2})'
    ax2.text(0.98, 0.02, rsi_text_2, transform=ax2.transAxes, fontsize=12,
             verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=rsi_color_2),
             color=rsi_color_2, fontweight='bold')
    
    ax2.set_title(f'{ticker2} Price & Moving Averages')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3 & 4: Volume as LINE PLOTS 
    if show_volume and ax_volume1 is not None and ax_volume2 is not None:
        # Stock 1 Volume as line plot
        ax_volume1.plot(stock1_data.index, stock1_data['Volume'], 
                       alpha=0.8, color='blue', linewidth=1.5, label=f'{ticker1} Volume')
        ax_volume1.set_title(f'{ticker1} Volume')
        ax_volume1.set_ylabel('Volume')
        ax_volume1.grid(True, alpha=0.3)
        ax_volume1.legend()
        
        # Stock 2 Volume as line plot
        ax_volume2.plot(stock2_data.index, stock2_data['Volume'],
                       alpha=0.8, color='darkred', linewidth=1.5, label=f'{ticker2} Volume')
        ax_volume2.set_title(f'{ticker2} Volume')
        ax_volume2.set_ylabel('Volume')
        ax_volume2.grid(True, alpha=0.3)
        ax_volume2.legend()
    
    plt.tight_layout()
    return fig

#search_stocks()
def search_stocks(query, max_results=10):
    """
    Search for stocks by company name or ticker using Yahoo Finance
    Returns list of (ticker, company_name, exchange) tuples
    """
    try:
        # Yahoo Finance search endpoint
        url = f"https://query1.finance.yahoo.com/v1/finance/search"
        params = {
            'q': query,
            'quotesCount': max_results,
            'newsCount': 0,
            'enableFuzzyQuery': False
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            for quote in data.get('quotes', []):
                ticker = quote.get('symbol', '')
                name = quote.get('longname') or quote.get('shortname', '')
                exchange = quote.get('exchange', '')
                
                # Filter for actual stocks (not indices, crypto, etc.)
                if (ticker and name and 
                    exchange in ['NYSE', 'NASDAQ', 'NMS', 'NYQ', 'BTS'] and
                    not any(x in ticker for x in ['^', '=', '-'])):
                    results.append((ticker, name, exchange))
            
            return results[:max_results]
        else:
            print(f"Search API returned status: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Search error: {e}")
        return []

#period_selection()
PERIOD_OPTIONS = {
    '1': ('1mo', '1 Month'),
    '2': ('3mo', '3 Months'),
    '3': ('6mo', '6 Months'),
    '4': ('1y', '1 Year'),
    '5': ('2y', '2 Years'),
    '6': ('5y', '5 Years'),
    '7': ('10y', '10 Years'),
    '8': ('max', 'Maximum Available')
}

def select_period():
    """Interactive period selection"""
    print("\n SELECT ANALYSIS PERIOD:")
    print("-" * 40)
    for key, (period_code, period_name) in PERIOD_OPTIONS.items():
        print(f"{key}. {period_name}")
    print("-" * 40)

    while True:
        choice = input("Choose period (1-8): ").strip()
        if choice in PERIOD_OPTIONS:
            period_code, period_name = PERIOD_OPTIONS[choice]
            print(f"Selected: {period_name}")
            return period_code, period_name
        else:
            print("Please enter a number between 1 and 8")

#interactive_stock_selection
def interactive_stock_selection(prompt_message="Select a stock:"):
    """
    Interactive stock selection with search
    Returns selected ticker
    """
    while True:
        print(f"\n{prompt_message}")
        search_term = input("Enter company name or ticker: ").strip()
        
        if not search_term:
            print("Please enter a search term.")
            continue
            
        print(f"Searching for '{search_term}'...")
        results = search_stocks(search_term)
        
        if not results:
            print("No results found. Try a different search term.")
            continue
            
        print(f"\n📈 Found {len(results)} results:")
        print("-" * 60)
        for i, (ticker, name, exchange) in enumerate(results, 1):
            print(f"{i}. {ticker} - {name} ({exchange})")
        print("-" * 60)
        
        try:
            selection = input(f"Select (1-{len(results)}) or 'r' to search again: ").strip().lower()
            
            if selection == 'r':
                continue
                
            selection_idx = int(selection) - 1
            if 0 <= selection_idx < len(results):
                selected_ticker, selected_name, exchange = results[selection_idx]
                print(f"Selected: {selected_ticker} - {selected_name}")
                return selected_ticker
            else:
                print(f"Please enter a number between 1 and {len(results)}")
                
        except ValueError:
            print("Please enter a valid number or 'r' to search again")

#interactive_stock_comparison()
def interactive_stock_comparison():
    """Interactive function to compare any two stocks"""
    print("STOCK COMPARISON PLATFORM")
    print("=" * 50)

    # Get stocks through search
    print("\nSELECT FIRST STOCK:")
    ticker1 = interactive_stock_selection("Select first stock to compare")

    print("\nSELECT SECOND STOCK:")
    ticker2 = interactive_stock_selection("Select second stock to compare")

    # Get period selection
    period_code, period_name = select_period()

    # Get visualization preferences
    print("\n Visualization Options:")
    show_sma = input("Show SMA? (y/n): ").lower() == 'y'
    show_ema = input("Show EMA? (y/n): ").lower() == 'y'
    show_volume = input("Show Volume? (y/n): ").lower() == 'y'

    print(f"\n Comparing {ticker1} vs {ticker2} over {period_name}.")

    #fetch data with selected period
    stock1, stock2 = compare_two_stocks(ticker1, ticker2,period_code)

    #Display statistics only
    #print("COMPARISON STATISTICS:")
    #print("=" * 60)
    #stats_df = display_statistics_only(stock1, stock2, ticker1, ticker2)

    
    #Plot comparison
    plot_price_comparison(stock1, stock2, ticker1, ticker2, show_sma=show_sma,
                         show_ema=show_ema, show_volume=show_volume)


    return stats_df

#robust_stock_search()
def robust_stock_search(query, max_results=10):
    """
    Enhanced search with multiple fallback strategies
    """
    # Try Yahoo Finance first
    results = search_stocks(query, max_results)
    
    if results:
        return results
    
    # Fallback: Try direct yfinance lookup for exact ticker matches
    try:
        stock = yf.Ticker(query.upper())
        info = stock.info
        if info.get('longName'):
            return [(query.upper(), info.get('longName'), info.get('exchange', 'Unknown'))]
    except:
        pass
    
    return []

def search_and_validate_ticker(ticker):
    """
    Validate that a ticker exists and get its full name
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        name = info.get('longName') or info.get('shortName')
        if name:
            return True, name
        return False, "Ticker not found"
    except:
        return False, "Invalid ticker"

# Page configuration
st.set_page_config(
    page_title="Stock Analysis Platform",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 0.5rem 0;
        background-color: #f9f9f9;
    }
    .stock-badge {
        background-color: #e1f5fe;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🎯 Stock Comparison Platform</h1>', unsafe_allow_html=True)
st.markdown("Compare technical indicators for any two stocks in real-time")

# Sidebar
with st.sidebar:
    st.header("🔧 Analysis Settings")
    
    # Stock selection
    st.subheader("Stock Selection")
    
    ticker1 = st.text_input("First Stock Ticker", value="AAPL", placeholder="e.g., AAPL")
    ticker2 = st.text_input("Second Stock Ticker", value="MSFT", placeholder="e.g., MSFT")
    
    # Stock search (optional enhancement)
    if st.checkbox("🔍 Use stock search"):
        st.info("Stock search feature - enter company names")
        search1 = st.text_input("Search first stock by name")
        search2 = st.text_input("Search second stock by name")
        # You can integrate your search function here later
    
    # Period selection
    st.subheader("Time Period")
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo", 
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "10 Years": "10y"
    }
    selected_period_name = st.selectbox("Select Period", list(period_options.keys()))
    period = period_options[selected_period_name]
    
    # Technical indicators
    st.subheader("Technical Indicators")
    show_sma = st.checkbox("Show Moving Averages", value=True)
    show_ema = st.checkbox("Show Exponential Moving Averages", value=True)
    show_volume = st.checkbox("Show Volume Analysis", value=True)
    show_bollinger = st.checkbox("Show Bollinger Bands", value=True)

    
    # Analysis button
    analyze_btn = st.button("🚀 Analyze Stocks", type="primary", use_container_width=True)

# Main content
if analyze_btn:
    if not ticker1 or not ticker2:
        st.error("❌ Please enter both stock tickers")
    elif ticker1.upper() == ticker2.upper():
        st.error("❌ Please select two different stocks")
    else:
        # Validate tickers first
        with st.spinner("🔍 Validating stock tickers..."):
            valid1, name1 = search_and_validate_ticker(ticker1)
            valid2, name2 = search_and_validate_ticker(ticker2)
            
            if not valid1:
                st.error(f"❌ Invalid ticker: {ticker1} - {name1}")
                st.stop()
            if not valid2:
                st.error(f"❌ Invalid ticker: {ticker2} - {name2}")
                st.stop()
        
        # Perform analysis
        with st.spinner(f"📊 Analyzing {ticker1} vs {ticker2}..."):
            try:
                # Get stock data
                stock1, stock2 = compare_two_stocks(ticker1, ticker2, period)
                
                # Display header with stock info
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<div class="stock-badge">📈 {ticker1} - {name1}</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="stock-badge">📈 {ticker2} - {name2}</div>', unsafe_allow_html=True)
                
                st.markdown(f"**Analysis Period:** {selected_period_name}")
                
                # Display statistics
                st.subheader("📋 Comparison Statistics")
                stats_df = display_statistics_only(stock1, stock2, ticker1, ticker2)
                st.dataframe(stats_df, use_container_width=True)
                
                # Display charts
                st.subheader("📊 Technical Analysis Charts")
                fig = plot_price_comparison(
                    stock1, stock2, ticker1, ticker2,
                    show_sma=show_sma, show_ema=show_ema, show_volume=show_volume,show_bollinger=show_bollinger
                )
                st.pyplot(fig)
                
                st.success("✅ Analysis complete!")
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("💡 Try selecting a different time period or check your internet connection")

else:
    # Welcome message
    st.markdown("""
    ### 📖 How to Use This Platform:
    1. **Enter stock tickers** in the sidebar (e.g., AAPL, TSLA, GOOGL)
    2. **Select analysis period** from the dropdown menu
    3. **Choose technical indicators** you want to see
    4. **Click 'Analyze Stocks'** to generate the comparison
    
    ### 📈 What You'll Get:
    - **Price charts** with technical indicators
    - **Volume analysis** and trends  
    - **Key statistics** and performance metrics
    - **RSI signals** and moving average analysis
    - **Professional-grade** technical analysis
    """)
    
    # Quick examples
    st.markdown("---")
    st.subheader("🚀 Quick Start Examples")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Tech Giants: AAPL vs MSFT", use_container_width=True):
            st.session_state.ticker1 = "AAPL"
            st.session_state.ticker2 = "MSFT"
    
    with col2:
        if st.button("EV Companies: TSLA vs NIO", use_container_width=True):
            st.session_state.ticker1 = "TSLA"
            st.session_state.ticker2 = "NIO"
    
    with col3:
        if st.button("Chip Makers: NVDA vs AMD", use_container_width=True):
            st.session_state.ticker1 = "NVDA"
            st.session_state.ticker2 = "AMD"