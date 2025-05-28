import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf

main_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(main_dir, "data")
results_dir = os.path.join(main_dir, "results")
data_fpath = os.path.join(data_dir, "crypto_data.csv")

# Function to fetch and save data for multiple tickers
def fetch_and_save_data(tickers, filename=data_fpath):
    try:
        existing_data = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")
        existing_tickers = list({col.split('_')[0] for col in existing_data.columns if '_' in col})
    except (FileNotFoundError, pd.errors.EmptyDataError):
        existing_data = pd.DataFrame()
        existing_tickers = []

    end = datetime.today()
    start = end - timedelta(days=365*5)

    for ticker in tickers:
        if ticker not in existing_tickers:
            print(f"Fetching data for {ticker}...")
            df = yf.download(ticker, start=start, end=end, progress=False)[["Low","Open","Close","High"]].dropna()
            df.columns = [f'{ticker}_Low', f'{ticker}_Open', f'{ticker}_Close', f'{ticker}_High']
            if existing_data.empty:
                existing_data = df
            else:
                existing_data = existing_data.join(df, how='outer')

    if not existing_data.empty:
        existing_data.to_csv(filename)
        print(f"Data saved to {filename}")
    else:
        print("No new data fetched.")


def load_ticker_data(ticker, filename):
    existing_data = pd.read_csv(filename, parse_dates=["Date"], index_col="Date")
    # Filter columns for the specified ticker
    ticker_columns = [col for col in existing_data.columns if col.startswith(f"{ticker}_")]
    if not ticker_columns:
        print(f"No data found for ticker {ticker}")
        return None
    df = existing_data[ticker_columns].copy()
    df.columns = [col.split('_')[1] for col in ticker_columns]  # Rename columns to 'Open', 'Close', etc.
    if df.empty:
        print(f"No data found for ticker {ticker}")
        return None
    return df

def get_indicators(ticker, df):
    # Calculate 50, 100 and 150-day SMA
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["SMA_100"] = df["Close"].rolling(window=100).mean()
    df["SMA_150"] = df["Close"].rolling(window=150).mean()

    # Define MACD params
    fast_len=30 
    slow_len=60
    atr_len=60
    atr_mult=0.30
    # Calculate Exponential Moving Averages (EMA)
    df["EMA_fast"] = df["Close"].ewm(span=fast_len, adjust=False).mean()
    df["EMA_slow"] = df["Close"].ewm(span=slow_len, adjust=False).mean()

    # True range and ATR (Average True Range) calculation
    df["H-L"] = df["Close"].shift(1).bfill() - df["Close"]
    df["TR"] = np.maximum.reduce([
        df["Close"] - df["Close"].shift(1).abs(),
        df["Close"].shift(1) - df["Close"]
    ])
    df["ATR"] = df["TR"].rolling(window=atr_len).mean()

    # Generate signals: +1 long, 0 flat
    diff = df["EMA_fast"] - df["EMA_slow"]
    df["signal"] = 0
    df.loc[diff > atr_mult * df["ATR"], "signal"] = 1
    df.loc[diff < -atr_mult * df["ATR"], "signal"] = 0  # flat on bearish
    # Shift signal into a position (enter next bar - next day after signal)
    df["long"] = df["signal"].shift(1).fillna(0)

    # Save the new df with indicators
    df = df.drop(columns=["H-L", "TR"])  # Drop intermediate columns

    # Save the DataFrame with indicators
    df.to_csv(data_ind_fpath)
    print(f"Indicators for {ticker} saved to {data_ind_fpath}")
    return df


def backtest(ticker, df):
    """
    Perform a backtest for a specific ticker using the preprocessed data.
    """
    def _simulate_strategy(df, initial_capital=1000.0):
        """
        Buy and hold strategy simulation:
        - Buy when long signal is triggered (1) and sell when it turns off (0).
        """
        df["strat_ret"] = initial_capital
        in_position = False
        cash = initial_capital
        shares = 0.0

        for i in range(1, len(df)):
            if df["long"].iloc[i] == 1 and df["long"].iloc[i - 1] == 0 and not in_position:
                # Buy with all available cash at close
                shares = cash / df["Close"].iloc[i]
                cash = 0.0
                in_position = True
            elif df["long"].iloc[i] == 0 and df["long"].iloc[i - 1] == 1 and in_position:
                # Sell all at close
                cash = shares * df["Close"].iloc[i]
                shares = 0.0
                in_position = False
            # Update portfolio value
            df.at[df.index[i], "strat_ret"] = cash + shares * df["Close"].iloc[i]
        return df

    def _calculate_lump_sum(df, initial_capital=1000.0):
        # Lump sum buy-and-hold: buy at first 1, sell at last 0
        df["strat_lump_sum"] = initial_capital
        first_buy_idx = df.index[df["long"].diff() == 1]
        last_sell_idx = df.index[df["long"].diff() == -1]
        if not first_buy_idx.empty and not last_sell_idx.empty:
            buy_idx = first_buy_idx[0]
            sell_idx = last_sell_idx[-1]
            buy_price = df.loc[buy_idx, "Close"]
            sell_price = df.loc[sell_idx, "Close"]
            shares_lump = initial_capital / buy_price
            for idx in df.index:
                if idx < buy_idx:
                    df.loc[idx, "strat_lump_sum"] = initial_capital
                elif buy_idx <= idx <= sell_idx:
                    df.loc[idx, "strat_lump_sum"] = shares_lump * df.loc[idx, "Close"]
                else:
                    df.loc[idx, "strat_lump_sum"] = shares_lump * sell_price
        return df

    def _calculate_performance_metrics(df, strategy_col="strat_ret"):
        """
        Compute performance metrics for any strategy column (e.g., 'strat_ret', 'strat_lump_sum').
        strategy_col: column name for strategy returns or portfolio value.
        If strategy_col is 'strat_ret', expects daily returns. If it's a portfolio value column, computes returns from it.
        """
        # Determine if strategy_col is returns or portfolio value
        if strategy_col in df.columns and "ret" in strategy_col:
            # It's a returns column (e.g., 'strat_ret')
            ret = df[strategy_col].fillna(0)
            total_return = (ret + 1).prod() - 1
        elif strategy_col in df.columns:
            # It's a portfolio value column (e.g., 'portfolio', 'strat_lump_sum')
            total_return = df[strategy_col].iloc[-1] / df[strategy_col].iloc[0] - 1
            # Synthesize daily returns for CAGR calculation
            ret = df[strategy_col].pct_change().fillna(0)
        else:
            raise ValueError(f"Column {strategy_col} not found in DataFrame.")

        years = (df.index[-1] - df.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else np.nan

        # Win rate: only meaningful for position-based strategies
        trades = []
        entry_price = None
        if "long" in df.columns and "Close" in df.columns:
            for i in range(1, len(df)):
                if df["long"].iloc[i] == 1 and df["long"].iloc[i - 1] == 0:
                    entry_price = df["Close"].iloc[i]
                if df["long"].iloc[i] == 0 and df["long"].iloc[i - 1] == 1 and entry_price is not None:
                    exit_price = df["Close"].iloc[i]
                    trades.append((exit_price / entry_price - 1))
                    entry_price = None
            win_rate = np.mean([1 if r > 0 else 0 for r in trades]) if trades else np.nan
            num_trades = len(trades)
        else:
            win_rate = np.nan
            num_trades = np.nan

        results = pd.DataFrame({
            "Metric": ["Backtest Period", "Strategy", "Total ROI", "CAGR", "Win Rate", "Number of Trades"],
            "Value": [
                f"{df.index[0].date()} to {df.index[-1].date()} ({years:.1f} years)",
                strategy_col,
                f"{total_return * 100:.2f}%",
                f"{cagr * 100:.2f}%",
                f"{win_rate * 100:.2f}%" if not np.isnan(win_rate) else "N/A",
                num_trades if not np.isnan(num_trades) else "N/A"
            ]
        })
        return results

    # Simulate the trading strategy
    initial_capital = 1000.0
    df = _simulate_strategy(df, initial_capital)
    df = _calculate_lump_sum(df, initial_capital)

    # Output results
    results_file = os.path.join(results_dir, f"{ticker}_backtest_results.csv")
    df.to_csv(results_file, index=True)

    # Calculate performance metrics
    results = _calculate_performance_metrics(df)
    summary_file = os.path.join(results_dir, f"{ticker}_performance_summary.csv")
    results.to_csv(summary_file, index=False)
    print(f"Backtesting results saved to {results_file}")

def ploter(ticker, df, date_range=None):

    if False:
        df = pd.read_csv(results_file, parse_dates=["Date"], index_col="Date")
        ticker = [col.split('_')[0] for col in df.columns if col.endswith("Close")][0]
        close_col = f"{ticker}_Close" if f"{ticker}_Close" in df.columns else "Close"
        if "EMA_fast" not in df.columns or "EMA_slow" not in df.columns:
            print("MACD lines not found in results file. Please run backtest first.")
            return

        plt.figure(figsize=(14, 7))
        plt.plot(df.index, df[close_col], label="Close Price", color="black")
        plt.plot(df.index, df["EMA_fast"], label="EMA Fast (MACD)", color="blue", linestyle="--")
        plt.plot(df.index, df["EMA_slow"], label="EMA Slow (MACD)", color="red", linestyle="--")
        plt.title(f"{ticker} Close Price and MACD Lines")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Plotting with mplfinance
    # Prepare DataFrame for mplfinance: columns must be ['Open', 'High', 'Low', 'Close']
    ohlc_cols = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in ohlc_cols):
        print("DataFrame does not contain required OHLC columns for mplfinance.")
        return
    
    # If date_range is provided, filter the DataFrame
    if date_range is not None:
        # Ensure date_range is in 'YYYY-MM-DD' format
        start_date = pd.to_datetime(date_range[0], format='%Y-%m-%d')
        end_date = pd.to_datetime(date_range[1], format='%Y-%m-%d')
        df_plot = df.loc[(df.index >= start_date) & (df.index <= end_date)].copy()
    else:
        df_plot = df

    # Now use df_plot for all subsequent plotting logic
    # Optionally add indicators to the plot (e.g., EMA_fast, EMA_slow, SMA_50)
    color_dict = {
        'golden': '#CFAC2D',
        'light_gold': '#FDFE7A',
        'dark_blue': "#311CE8",
        'light_blue': '#D1CAF4',
        'gray': '#8F8F8E'
    }
    addplots = []
    if True:
        if "EMA_fast" in df_plot.columns:
            addplots.append(mpf.make_addplot(df_plot["EMA_fast"], color=color_dict['golden'], width=1.0, linestyle='--', label='EMA Fast'))
        if "EMA_slow" in df_plot.columns:
            addplots.append(mpf.make_addplot(df_plot["EMA_slow"], color=color_dict['dark_blue'], width=1.0, linestyle='--', label='EMA Slow'))
    if False: 
        if "SMA_50" in df_plot.columns:
            addplots.append(mpf.make_addplot(df_plot["SMA_50"], color='green', width=1.0, linestyle='-', label='SMA 50'))
        if "SMA_100" in df_plot.columns:
            addplots.append(mpf.make_addplot(df_plot["SMA_100"], color='orange', width=1.0, linestyle='-', label='SMA 100'))
        if "SMA_150" in df_plot.columns:
            addplots.append(mpf.make_addplot(df_plot["SMA_150"], color='purple', width=1.0, linestyle='-', label='SMA 150'))

    # Plot buy/sell signals as markers
    signal_markers = []
    if False:
        buy_signals = df_plot[(df_plot["long"].diff() == 1)]
        sell_signals = df_plot[(df_plot["long"].diff() == -1)]
        if not buy_signals.empty:
            signal_markers.append(mpf.make_addplot(buy_signals["Close"], type='scatter', markersize=100, marker='^', color='lime', label='Buy'))
        if not sell_signals.empty:
            signal_markers.append(mpf.make_addplot(sell_signals["Close"], type='scatter', markersize=100, marker='v', color='red', label='Sell'))

    # Combine all addplots
    all_addplots = addplots + signal_markers

    mpf.plot(
        df_plot,
        type='candle',
        style='charles',
        title=f"{ticker} Price Chart with Indicators and Signals",
        ylabel='Price',
        volume=False,
        addplot=all_addplots,
        figratio=(16, 9),
        figscale=1.2,
        tight_layout=True,
        datetime_format='%Y-%m'
    )

if __name__ == "__main__":
    if False: # Fetch and save data for multiple tickers
        tickers = ["BTC-USD", "SOL-USD", "SUI-USD"]
        fetch_and_save_data(tickers)

    ticker = "BTC-USD"

    # Check if indicators file exists
    data_ind_fpath = os.path.join(data_dir, f"{ticker}_data_indicators.csv")
    if os.path.exists(data_ind_fpath):
        print(f"Indicators for {ticker} already exist at {data_ind_fpath}. Loading existing data.")
        df = pd.read_csv(data_ind_fpath, parse_dates=["Date"], index_col="Date")
    else:
        # Load ticker data
        df = load_ticker_data(ticker, data_fpath)
        # Compute indicator values
        df = get_indicators(ticker, df)

    if False:
        backtest(ticker, df)
    if True:
        date_range = ("2023-01-01", "2023-12-31")
        ploter(ticker, df, date_range=date_range)