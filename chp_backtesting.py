import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates

main_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(main_dir, "data")
results_dir = os.path.join(main_dir, "results")
data_fpath = os.path.join(data_dir, "crypto_data.csv")

# Function to fetch and save data for multiple tickers
def fetch_and_save_data(tickers, filename=data_fpath):
    """
    Fetch historical data for multiple tickers and save to a CSV file.
    tickers: List of ticker symbols to fetch data for.
    filename: Path to the CSV file where data will be saved.
    """
    print(f"Fetching data for tickers: {', '.join(tickers)}")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
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
    # Pine parameters
    fast_len, slow_len, atr_len, atr_mult = 30, 60, 60, 0.30

    # Calculate EMAs
    df["EMA_fast"] = df["Close"].ewm(span=fast_len, adjust=False).mean()
    df["EMA_slow"] = df["Close"].ewm(span=slow_len, adjust=False).mean()
    df["EMA_diff"] = df["EMA_fast"] - df["EMA_slow"]

    # Calculate ATR (PineScript style)
    prev_close = df["Close"].shift(1)
    tr = pd.DataFrame({
        "tr1": df["High"] - df["Low"],
        "tr2": (df["High"] - prev_close).abs(),
        "tr3": (df["Low"] - prev_close).abs()
    }).max(axis=1)
    df["ATR"] = tr.rolling(window=atr_len).mean()

    # Trend conditions for coloring
    atr_margin = atr_mult * df["ATR"]
    df["bull"] = df["EMA_diff"] >  atr_margin
    df["bear"] = df["EMA_diff"] < -atr_margin
    df["neutral"] = ~(df["bull"] | df["bear"])
    df["ema_color"] = np.where(df["bull"], "bull", np.where(df["bear"], "bear", "neutral"))

    # Save if needed
    data_ind_fpath = os.path.join(data_dir, f"{ticker}_data_indicators.csv")
    df.to_csv(data_ind_fpath)
    print(f"Indicators for {ticker} saved to {data_ind_fpath}")
    return df

def get_signals(df):
    df["long"] = 0
    df["signal"] = 0

    # Buy signal: when bull turns True (from not bull)
    buy_signal = (df["bull"] & ~df["bull"].shift(1).astype(bool).fillna(False))
    # Sell signal: when bear turns True (from not bear)
    sell_signal = (df["bear"] & ~df["bear"].shift(1).astype(bool).fillna(False))

    df.loc[buy_signal, "signal"] = 1   # Buy
    df.loc[sell_signal, "signal"] = -1 # Sell

    # Forward fill long position: in position after buy, out after sell
    in_position = False
    for i in range(len(df)):
        if df["signal"].iloc[i] == 1:
            in_position = True
        elif df["signal"].iloc[i] == -1:
            in_position = False
        df.at[df.index[i], "long"] = int(in_position)

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

class Ploter:
    def __init__(self, ticker, df, date_range=None):
        self.ticker = ticker
        self.df = df
        self.date_range = date_range

        # If date_range is provided, filter the DataFrame
        if date_range is not None:
            start_date = pd.to_datetime(date_range[0], format='%Y-%m-%d')
            end_date = pd.to_datetime(date_range[1], format='%Y-%m-%d')
            self.df = self.df.loc[(self.df.index >= start_date) & (self.df.index <= end_date)].copy()

        self.color_dict = {
            'dark_gold': '#CFAC2D',
            'light_gold': '#FDFE7A',
            'dark_blue': "#311CE8",
            'light_blue': '#D1CAF4',
            'dark_gray': '#4F4F4E',
            'light_gray': '#D9D9D8',
        }
    
    def plot_plt(self):
        """ Plotting with matplotlib using ema_color for segment coloring """

        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, self.df["Close"], label="Close Price", color="black")

        ema_fast = self.df["EMA_fast"]
        ema_slow = self.df["EMA_slow"]
        ema_color = self.df["ema_color"]
        x = mdates.date2num(self.df.index.to_pydatetime())

        # Map ema_color to color_dict
        color_map = {
            "bull": self.color_dict['dark_gold'],
            "bear": self.color_dict['dark_blue'],
            "neutral": self.color_dict['dark_gray'],
        }
        fill_color_map = {
            "bull": self.color_dict['light_gold'],
            "bear": self.color_dict['light_blue'],
            "neutral": self.color_dict['light_gray'],
        }


        # Plot EMA_fast and EMA_slow with coloring by ema_color
        def _plot_colored_segments(x, y, color_series, label_prefix):
            prev_color = None
            seg_start = 0
            for i in range(1, len(y)):
                color = color_series.iloc[i]
                if color != prev_color or i == len(y) - 1:
                    if prev_color is not None:
                        seg_end = i if color != prev_color else i + 1
                        plt.plot(
                            x[seg_start:seg_end],
                            y[seg_start:seg_end],
                            color=color_map[prev_color],
                            linewidth=2,
                            label=f"{label_prefix} ({prev_color})" if seg_start == 0 else None
                        )
                    seg_start = i - 1
                    prev_color = color
        _plot_colored_segments(x, ema_fast.values, ema_color, "EMA Fast")
        _plot_colored_segments(x, ema_slow.values, ema_color, "EMA Slow")

        # Fill between EMA_fast and EMA_slow with colors based on ema_color
        def _fill_between_emas(x, ema_fast, ema_slow, ema_color, fill_color_map):
            prev_color = None
            seg_start = 0
            for i in range(1, len(ema_fast)):
                color = ema_color.iloc[i]
                if color != prev_color or i == len(ema_fast) - 1:
                    if prev_color is not None:
                        seg_end = i if color != prev_color else i + 1
                        plt.fill_between(
                            x[seg_start:seg_end],
                            ema_fast[seg_start:seg_end],
                            ema_slow[seg_start:seg_end],
                            color=fill_color_map[prev_color],
                            alpha=0.4,
                            interpolate=True,
                            label=f"Fill ({prev_color})" if seg_start == 0 else None
                        )
                    seg_start = i - 1
                    prev_color = color
        _fill_between_emas(x, ema_fast.values, ema_slow.values, ema_color, fill_color_map)
        
        # Plot signals
        def _plot_signals():
            # Plot buy (green up) and sell (red down) arrows for signals
            buy_signals = self.df[self.df["signal"] == 1]
            sell_signals = self.df[self.df["signal"] == -1]

            plt.scatter(
                    mdates.date2num(buy_signals.index),
                    buy_signals["Close"],
                    marker="^",
                    color="green",
                    s=100,
                    label="Buy Signal"
                )
            plt.scatter(
                    mdates.date2num(sell_signals.index),
                    sell_signals["Close"],
                    marker="v",
                    color="red",
                    s=100,
                    label="Sell Signal"
                )
        _plot_signals()

        plt.title(f"{self.ticker} Price and EMA Trend Coloring")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_mpl(self):
        """ Plotting with mplfinance """

        df = self.df
        ticker = self.ticker

        # Prepare DataFrame for mplfinance: columns must be ['Open', 'High', 'Low', 'Close']
        ohlc_cols = ["Open", "High", "Low", "Close"]
        if not all(col in df.columns for col in ohlc_cols):
            print("DataFrame does not contain required OHLC columns for mplfinance.")
            return

        # Optionally add indicators to the plot (e.g., EMA_fast, EMA_slow, SMA_50)
        addplots = []
        if "EMA_fast" in df.columns:
            addplots.append(mpf.make_addplot(df["EMA_fast"], color=self.color_dict['dark_gold'], width=1.0, linestyle='--', label='EMA Fast'))
        if "EMA_slow" in df.columns:
            addplots.append(mpf.make_addplot(df["EMA_slow"], color=self.color_dict['dark_blue'], width=1.0, linestyle='--', label='EMA Slow'))
        # Optionally add SMAs (disabled by default)
        if False:
            if "SMA_50" in df.columns:
                addplots.append(mpf.make_addplot(df["SMA_50"], color='green', width=1.0, linestyle='-', label='SMA 50'))
            if "SMA_100" in df.columns:
                addplots.append(mpf.make_addplot(df["SMA_100"], color='orange', width=1.0, linestyle='-', label='SMA 100'))
            if "SMA_150" in df.columns:
                addplots.append(mpf.make_addplot(df["SMA_150"], color='purple', width=1.0, linestyle='-', label='SMA 150'))

        # Plot buy/sell signals as markers (disabled by default)
        signal_markers = []
        if False:
            buy_signals = df[(df["long"].diff() == 1)]
            sell_signals = df[(df["long"].diff() == -1)]
            if not buy_signals.empty:
                signal_markers.append(mpf.make_addplot(buy_signals["Close"], type='scatter', markersize=100, marker='^', color='lime', label='Buy'))
            if not sell_signals.empty:
                signal_markers.append(mpf.make_addplot(sell_signals["Close"], type='scatter', markersize=100, marker='v', color='red', label='Sell'))

        all_addplots = addplots + signal_markers

        mpf.plot(
            df,
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

    if True: # Backtest and plot for a specific ticker
        ticker = "BTC-USD"

        # Check if indicators file exists
        data_ind_fpath = os.path.join(data_dir, f"{ticker}_data_indicators.csv")
        overwrite = True
        if not overwrite and os.path.exists(data_ind_fpath):
            print(f"Indicators for {ticker} already exist. Loading existing data.")
            df = pd.read_csv(data_ind_fpath, parse_dates=["Date"], index_col="Date")
        else:
            # Load ticker data
            df = load_ticker_data(ticker, data_fpath)
            # Compute indicator values
            df = get_indicators(ticker, df)
            # Get trading signals
            df = get_signals(df)

        if False:
            backtest(ticker, df)
        if True:
            date_range = ("2024-01-01", "2025-01-01")
            ploter = Ploter(ticker, df, date_range=date_range)
            ploter.plot_plt()