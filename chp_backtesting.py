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
figs_dir = os.path.join(main_dir, "figures")
strategies_fpath = os.path.join(results_dir, "!strategies_performance_summary.csv")
tickers_fpath = os.path.join(data_dir, "tickers.csv")

class YF:
    def __init__(self, tickers=None, asset_type='stocks', start_date=None, end_date=None):
        self.tickers = tickers
        self.asset_data = os.path.join(data_dir, f"{asset_type}.csv")
        self.start_date = start_date if start_date else datetime(2020, 6, 1)
        self.end_date = end_date if end_date else datetime.today()
        self.data = None
        if type(tickers) is str:
            self.tickers = [tickers]
        else:
            self.tickers = pd.read_csv(tickers_fpath).get("tickers", tickers).tolist()[:7]

    def search_query(self, query):
        """
        Search for tickers based on a query string.
        Returns a list of ticker symbols that match the query.
        """
        results = yf.Search(query)
        # yfinance's Search returns a Search object, not a dict
        # Use .quotes to get the list of results
        if hasattr(results, "quotes"):
            quotes = results.quotes
            if quotes:
                # Print header
                print(f"{'Symbol':<15} {'Short Name':<40} {'Type':<15} {'Exchange':<15}")
                print("-" * 75)
                for q in quotes:
                    symbol = q.get("symbol", "")
                    shortname = q.get("shortname", "")
                    exchange = q.get("exchange", "")
                    quote_type = q.get("quoteType", "")
                    print(f"{symbol:<15} {shortname:<40} {quote_type:<15} {exchange:<15}")
            else:
                print("No quotes found for query.")
                return []
        else:
            print("No quotes found for query.")
            return []
            
    def fetch_and_save_data(self):
        """
        Fetch historical data for multiple tickers and save to a CSV file.
        tickers: List of ticker symbols to fetch data for.
        filename: Path to the CSV file where data will be saved.
        """
        print(f"Fetching data for tickers: {', '.join(self.tickers)}")
        if os.path.exists(self.asset_data):
            existing_data = pd.read_csv(self.asset_data, index_col="Date")
            existing_data.index = pd.to_datetime(existing_data.index, format='%Y-%m-%d', errors='coerce')
            existing_tickers = list({col.split('_')[0] for col in existing_data.columns if '_' in col})
        else:
            existing_data = pd.DataFrame()
            existing_tickers = []

        start = datetime(2020, 6, 2)
        end = datetime.today() - timedelta(days=1)

        for ticker in self.tickers:
            #ticker_cols = [f"{ticker}_Open", f"{ticker}_High", f"{ticker}_Low", f"{ticker}_Close"]
            if ticker not in existing_tickers:
                print(f"Fetching data for {ticker}...")
                df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)[["Low", "Open", "Close", "High"]].dropna()
                df.columns = [f'{ticker}_Low', f'{ticker}_Open', f'{ticker}_Close', f'{ticker}_High']
                if existing_data.empty:
                    existing_data = df
                else:
                    existing_data = existing_data.join(df, how='outer')
                    existing_data.sort_index(inplace=True)
                    existing_data.to_csv(self.asset_data)
                    print(f"Data for {ticker} saved to {os.path.basename(self.asset_data)}.")
            else:
                last_available_date = existing_data.index.max()
                # Update df with missing dates from last_available_date+1 up to end
                missing_dates = pd.date_range(start=last_available_date + pd.Timedelta(days=1), end=end, freq='B')
                if len(missing_dates) > 0:
                    print(f"Updating {ticker} with {len(missing_dates)} missing dates...")
                    df_new = yf.download(ticker, start=missing_dates[0], end=missing_dates[-1] + pd.Timedelta(days=1), progress=False, auto_adjust=True)[["Low", "Open", "Close", "High"]].dropna()
                    df_new.columns = [f'{ticker}_Low', f'{ticker}_Open', f'{ticker}_Close', f'{ticker}_High']
                    if not df_new.empty:
                        existing_data = existing_data.combine_first(df_new)
                        existing_data.sort_index(inplace=True)
                        existing_data.to_csv(self.asset_data)
                        print(f"Updated data for {ticker} saved to {os.path.basename(self.asset_data)}.")
                    else:
                        print(f"No new data found for {ticker}.")

class Calc:
    def __init__(self, ticker, asset_type, timeframe='1D', results_csv=None):
        self.ticker = ticker
        self.asset_data = os.path.join(data_dir, f"{asset_type}.csv")
        self.timeframe = timeframe
        self.results_csv = results_csv

    def get_ticker_data(self):
        # Read CSV with Date as string, then convert index to datetime with format '%d.%m.%Y'
        existing_data = pd.read_csv(self.asset_data, index_col="Date")
        existing_data.index = pd.to_datetime(existing_data.index, format='%Y-%m-%d', errors='coerce')
        # Filter columns for the specified ticker
        ticker_columns = [col for col in existing_data.columns if col.startswith(f"{self.ticker}_")]
        if not ticker_columns:
            print(f"No data found for ticker {self.ticker}")
            exit()
        df = existing_data[ticker_columns].copy()
        df.columns = [col.split('_')[1] for col in ticker_columns]  # Rename columns to 'Open', 'Close', etc.
        if df.empty:
            print(f"No data found for ticker {self.ticker}")
            exit()

        # Resample if timeframe is not '1D'
        if self.timeframe != '1D':
            if self.timeframe == '2D':
                df = df.resample('2D').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
            elif self.timeframe == '1W':
                df = df.resample('W').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last'
                }).dropna()
            else:
                print(f"Unsupported timeframe: {self.timeframe}")
                exit()
        self.df = df
        return df

    def get_default_indicators(self):
        df = self.df
        # Pine parameters
        fast_len, slow_len, atr_len, atr_mult = 30, 60, 60, 0.30

        # Only compute if columns do not already exist
        needed_cols = ["EMA_fast", "EMA_slow", "EMA_diff", 
                    "SMA_50", "SMA_100", "SMA_150", 
                    "ATR", "bull", "bear", "neutral", "ema_color"]
        if all(col in df.columns for col in needed_cols):
            print(f"-- Indicators for {self.ticker} already exist in DataFrame. Skipping calculation.")
            return df
        
        # Calculate SMAs
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_100"] = df["Close"].rolling(window=100).mean()
        df["SMA_150"] = df["Close"].rolling(window=150).mean()

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

        # Save
        df.to_csv(self.results_csv, index=True)
        print(f"--- Indicators for {self.ticker} saved with {self.timeframe} timeframe.")
        self.df = df
        return df

    def get_default_signals(self, prefix="str_def"):
        df = self.df
        # Skip if columns that start with "str_def" already exist
        if any(col.startswith(prefix) for col in df.columns):
            print(f"-- Signals for {self.ticker} already exist in DataFrame. Skipping calculation.")
            return df

        df[f"{prefix}_long"] = 0
        df[f"{prefix}_signal"] = 0

        # Buy signal: when bull turns True (from not bull)
        buy_signal = (df["bull"] & ~df["bull"].shift(1).astype(bool).fillna(False))
        # Sell signal: when bear turns True (from not bear)
        sell_signal = (df["bear"] & ~df["bear"].shift(1).astype(bool).fillna(False))

        df.loc[buy_signal, f"{prefix}_signal"] = 1   # Buy
        df.loc[sell_signal, f"{prefix}_signal"] = -1 # Sell

        # Forward fill long position: in position after buy, out after sell
        in_position = False
        for i in range(len(df)):
            if df[f"{prefix}_signal"].iloc[i] == 1:
                in_position = True
            elif df[f"{prefix}_signal"].iloc[i] == -1:
                in_position = False
            df.at[df.index[i], f"{prefix}_long"] = int(in_position)
        
        # Save
        df.to_csv(self.results_csv, index=True)
        print(f"--- Signals for {self.ticker} with {self.timeframe} timeframe saved.")
        self.df = df
        return df
    
    def append_additional_signals(self, df, prefix="str_def_2"):
        """
        Append additional signals to the DataFrame.
        This method is a placeholder for future signal calculations.
        """
        # Check if prefix already exists
        if any(col.startswith(prefix) for col in df.columns):
            print(f"-- Additional signals with prefix '{prefix}' already exist in DataFrame. Skipping calculation.")
            return df

        # Placeholder for additional signal logic
        df[f"{prefix}_signal"] = 0

        # If in position (str_def_long == 1), signal a buy when Close crosses above EMA_fast
        df[f"{prefix}_buy_cross"] = 0
        df[f"{prefix}_sell_cross"] = 0
        in_position = df["str_def_long"] == 1
        close = df["Close"]
        ema_fast = df["EMA_fast"]
        ema_slow = df["EMA_slow"]

        # Only detect crosses when ema_color is 'bull'
        bull_condition = df["ema_color"] == "bull"

        # Detect upward cross: previous Close <= previous EMA_fast and current Close > current EMA_fast
        cross_up = (close.shift(1) <= ema_fast.shift(1)) & (close > ema_fast) & bull_condition
        df.loc[in_position & cross_up, f"{prefix}_signal"] = 1

        # Detect downward cross: previous Close >= previous EMA_slow and current Close < current EMA_slow
        cross_down = (close.shift(1) >= ema_slow.shift(1)) & (close < ema_slow) & bull_condition
        df.loc[in_position & cross_down, f"{prefix}_signal"] = -1

        # Save
        df.to_csv(self.results_csv, index=True)
        print(f"--- Additional signals for {self.ticker} with {self.timeframe} timeframe saved.")
        self.df = df


class Backtest():
    def __init__(self, ticker, df, period=None, timeframe="1D", results_csv=None):
        self.ticker = ticker
        self.df = df
        self.period = period
        self.timeframe = timeframe
        self.results_csv = results_csv
        self.summary = None
        self.initial_capital = 1000.0
        self.str_prefix = "str_def"

        # Filter DataFrame by period if provided
        if self.period is not None:
            start_date = pd.to_datetime(self.period[0])
            end_date = pd.to_datetime(self.period[1])
            self.df = self.df.loc[(self.df.index >= start_date) & (self.df.index <= end_date)].copy()

    def _simulate_lump_sum(self):
        df = self.df
        initial_capital = self.initial_capital
        # Lump sum buy-and-hold: buy at first long==1, sell at last available date
        df["str_lump_sum"] = initial_capital
        long_diff = df["str_def_long"].diff().fillna(0)
        first_buy_idx = df.index[long_diff == 1]
        if not first_buy_idx.empty:
            buy_idx = first_buy_idx[0]
            sell_idx = df.index[-1]
            buy_price = df.at[buy_idx, "Close"]
            #sell_price = df.at[sell_idx, "Close"]
            shares_lump = initial_capital / buy_price
            for idx in df.index:
                if idx < buy_idx:
                    df.at[idx, "str_lump_sum"] = initial_capital
                elif buy_idx <= idx <= sell_idx:
                    df.at[idx, "str_lump_sum"] = shares_lump * df.at[idx, "Close"]
        self.df = df

    def _simulate_strategy(self):
        """ 
        Simulate a strategy:
         - buy/sell on signals
         """
        df = self.df
        str_prefix = self.str_prefix
        initial_capital = self.initial_capital
        df[f"{str_prefix}_ret"] = initial_capital
        allocated = False
        cash = initial_capital
        shares = 0.0

        for i in range(1, len(df)):
            signal = df[f"{str_prefix}_signal"].iloc[i]
            price = df["Close"].iloc[i]
            if signal == 1 and not allocated:
                # Allocate capital at Buy
                shares = cash / price
                cash = 0.0
                allocated = True
            elif signal == -1 and allocated:
                # Sell all at Sell
                cash = shares * price
                shares = 0.0
                allocated = False
            # Update portfolio value
            df.at[df.index[i], f"{str_prefix}_ret"] = cash + shares * price
        self.df = df
    
    def _simulate_strategy_50pct(self):
        """ 
        Simulate a strategy:
         - buy/sell on signals (with 50% of initial capital)
         - additional buy and sell signals based on EMA crosses
         """
        df = self.df
        str_prefix = self.str_prefix
        initial_capital = self.initial_capital
        alloc_percent = 0.5  # Allocate 50% of initial capital per signal
        df[f"{str_prefix}_50pct_ret"] = initial_capital

        cash = initial_capital
        shares = 0.0
        allocation_count = 0  # Number of active allocations (max 2 for 100%)

        for i in range(1, len(df)):
            signal = df[f"{str_prefix}_signal"].iloc[i]
            price = df["Close"].iloc[i]

            # Buy signal: allocate another alloc_percent if not fully allocated
            if signal == 1 and allocation_count < int(1 / alloc_percent):
                allocation_count += 1
                amount_to_allocate = initial_capital * alloc_percent
                if cash >= amount_to_allocate:
                    shares += amount_to_allocate / price
                    cash -= amount_to_allocate
                else:
                    # Allocate remaining cash if less than alloc_percent left
                    shares += cash / price
                    cash = 0.0

            # Sell signal: de-allocate one alloc_percent if allocated
            elif signal == -1 and allocation_count > 0:
                allocation_count -= 1
                amount_to_deallocate = initial_capital * alloc_percent
                shares_to_sell = amount_to_deallocate / price
                if shares >= shares_to_sell:
                    shares -= shares_to_sell
                    cash += shares_to_sell * price
                else:
                    # Sell all remaining shares if less than alloc_percent left
                    cash += shares * price
                    shares = 0.0

            # Update portfolio value
            df.at[df.index[i], f"{str_prefix}_50pct_ret"] = cash + shares * price

        self.df = df

    def _calculate_lump_sum_metrics(self):
        df = self.df
        lump_portfolio = df["str_lump_sum"].copy().ffill()
        total_return = lump_portfolio.iloc[-1] / lump_portfolio.iloc[0] - 1
        days = (df.index[-1] - df.index[0]).days
        years = days / 365.25 if days > 0 else np.nan
        cagr = (lump_portfolio.iloc[-1] / lump_portfolio.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
        self.lump_sum_metrics = {
            "Lump Sum ROI": total_return,
            "Lump Sum ROI %": f"{total_return * 100:.2f}%",
            "Lump Sum CAGR": cagr,
            "Lump Sum CAGR %": f"{cagr * 100:.2f}%" if years == years else "N/A",
            "Backtest Period": f"{df.index[0].date()} to {df.index[-1].date()} ({years:.1f} years)" if years == years else "N/A"
        }

    def _calculate_strategy_metrics(self):
        df = self.df
        ret_cols = [col for col in df.columns if col.endswith('_ret')]
        summaries = []
        lump_sum_roi = self.lump_sum_metrics.get("Lump Sum ROI", np.nan)
        lump_sum_cagr = self.lump_sum_metrics.get("Lump Sum CAGR", np.nan)
        backtest_period = self.lump_sum_metrics.get("Backtest Period", "N/A")

        for col in ret_cols:
            strat = col.replace('_ret', '')
            portfolio = df[col].copy().ffill()
            total_return = portfolio.iloc[-1] / portfolio.iloc[0] - 1
            days = (df.index[-1] - df.index[0]).days
            years = days / 365.25 if days > 0 else np.nan
            cagr = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan

            # Win rate and number of trades
            trades = []
            entry_price = None
            long_col = "str_def_long"
            if long_col in df.columns and "Close" in df.columns:
                for i in range(1, len(df)):
                    if df[long_col].iloc[i] == 1 and df[long_col].iloc[i - 1] == 0:
                        entry_price = df["Close"].iloc[i]
                    if df[long_col].iloc[i] == 0 and df[long_col].iloc[i - 1] == 1 and entry_price is not None:
                        exit_price = df["Close"].iloc[i]
                        trades.append(exit_price / entry_price - 1)
                        entry_price = None
                win_rate = np.mean([1 if r > 0 else 0 for r in trades]) if trades else np.nan
                num_trades = len(trades)
            else:
                win_rate = np.nan
                num_trades = np.nan

            # Calculate ratio of strategy ROI to lump sum ROI
            roi_ratio = (total_return / lump_sum_roi) if lump_sum_roi != 0 and not np.isnan(lump_sum_roi) else np.nan

            metrics = {
                "Ticker": self.ticker,
                "Strategy": strat,
                "Backtest Period": backtest_period,
                "Timeframe": self.timeframe,
                "Total ROI": f"{total_return * 100:.2f}%",
                "CAGR": f"{cagr * 100:.2f}%" if years == years else "N/A",
                "Win Rate": f"{win_rate * 100:.2f}%" if not np.isnan(win_rate) else "N/A",
                "Number of Trades": num_trades if not np.isnan(num_trades) else "N/A",
                "Lump Sum ROI": f"{lump_sum_roi * 100:.2f}%" if not np.isnan(lump_sum_roi) else "N/A",
                "Lump Sum CAGR": f"{lump_sum_cagr * 100:.2f}%" if not np.isnan(lump_sum_cagr) else "N/A",
                "ROI Ratio (Strategy/Lump Sum)": f"{roi_ratio:.2f}" if not np.isnan(roi_ratio) else "N/A"
            }
            summaries.append(metrics)

        self.summary = pd.DataFrame(summaries) if summaries else pd.DataFrame([{
            "Ticker": self.ticker,
            "Strategy": "N/A",
            "Backtest Period": "N/A",
            "Timeframe": self.timeframe,
            "Total ROI": "N/A",
            "CAGR": "N/A",
            "Win Rate": "N/A",
            "Number of Trades": "N/A",
            "Lump Sum ROI": "N/A",
            "Lump Sum CAGR": "N/A",
            "ROI Ratio (Strategy/Lump Sum)": "N/A"
        }])

    def run_backtest(self):
        if "str_lump_sum" in self.df.columns:
            print(f"-- Lump sum column already exists for {self.ticker}. Skipping simulation.")
        else:
            self._simulate_lump_sum()
            self._calculate_lump_sum_metrics()

        if "str_def_ret" in self.df.columns:
            print(f"-- Default strategy return column already exists for {self.ticker}. Skipping simulation.")
        else:
            self._simulate_strategy()
            self._simulate_strategy_50pct()
            self._calculate_strategy_metrics()

            # Save results to CSV
            if self.results_csv:
                self.df.to_csv(self.results_csv, index=True)
                # Append to existing summary CSV if it exists, else create new
                if os.path.exists(strategies_fpath):
                    existing = pd.read_csv(strategies_fpath)
                    combined = pd.concat([existing, self.summary], ignore_index=True)
                    combined.to_csv(strategies_fpath, index=False)
                else:
                    self.summary.to_csv(strategies_fpath, index=False)
                print(f"--- Backtesting results saved.")

        return self.df


class Ploter:
    def __init__(self, ticker, df, period=None, timeframe='1D', save_img=False):
        self.ticker = ticker
        self.df = df
        self.period = period
        self.timeframe = timeframe
        self.save_img = save_img

        # If date_range is provided, filter the DataFrame
        if period is not None:
            start_date = pd.to_datetime(period[0], format='%Y-%m-%d')
            end_date = pd.to_datetime(period[1], format='%Y-%m-%d')
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
            buy_signals = self.df[self.df["str_def_signal"] == 1]
            sell_signals = self.df[self.df["str_def_signal"] == -1]

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

        def _plot_add_signals(prefix="str_def_2"):
            # Plot additional buy/sell signals
            buy_signals = self.df[self.df[f"{prefix}_signal"] == 1]
            sell_signals = self.df[self.df[f"{prefix}_signal"] == -1]

            plt.scatter(
                mdates.date2num(buy_signals.index),
                buy_signals["Close"],
                marker="^",
                color="green",
                s=50,
                alpha=0.5,
                label=f"{prefix} Buy Signal"
            )
            plt.scatter(
                mdates.date2num(sell_signals.index),
                sell_signals["Close"],
                marker="v",
                color="red",
                s=50,
                alpha=0.5,
                label=f"{prefix} Sell Signal"
            )
        _plot_add_signals()

        def _plot_backtest_results(strategy_prefix="str_def"):
            # Plot backtest results on a secondary Y axis
            ax = plt.gca()
            ax2 = ax.twinx()
            ax2.set_ylabel("Initial capital + Profit & Loss")

            str_colors = {
                "str_lump_sum": "#808080",  # gray
                "str_def": "#FCA90E",  # purple
                "str_def_50pct": "#C96D03",  # orange
                }
            
            if "str_lump_sum" in self.df.columns:
                ax2.plot(self.df.index, self.df["str_lump_sum"], label="lump sum", color=str_colors["str_lump_sum"], linestyle='--')
            str_col = f"{strategy_prefix}_ret"
            if str_col in self.df.columns:
                ax2.plot(self.df.index, self.df[str_col], label="str_def", color=str_colors["str_def"], linestyle='--')
            str_col_50pct = f"{strategy_prefix}_50pct_ret"
            if str_col_50pct in self.df.columns:
                ax2.plot(self.df.index, self.df[str_col_50pct], label="str_def_50pct", color=str_colors["str_def_50pct"], linestyle='--')
            
            # Combine legends from both axes
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc="upper left")

            # Add horizontal line at the first point of the strategy portfolio value
            if str_col in self.df.columns:
                first_val = self.df[str_col].iloc[0]
                ax2.axhline(first_val, color="black", linestyle=":", linewidth=1.5, alpha=0.7, label="Initial Portfolio Value")

            # Display strategy summary
            summary = pd.read_csv(strategies_fpath)
            summary = summary[(summary["Ticker"] == self.ticker) & (summary["Timeframe"] == self.timeframe)]
            if not summary.empty:
                # Calculate the number of strategies to determine box positions
                strategies = list(summary["Strategy"].unique())
                n_strat = len(strategies)
                # Set box width and spacing
                box_width = 0.2 / max(n_strat, 1)
                for i, strategy in enumerate(strategies):
                    # Filter summary for the current strategy
                    strategy_summary = summary[summary["Strategy"] == strategy]
                    # Prepare summary text
                    s = strategy_summary.iloc[0]
                    box_text = (
                        "Strategy: " + s['Strategy'] + "\n"
                        f"Str. total ROI: {round(float(str(s['Total ROI']).replace('%',''))):.0f}%\n"
                        f"Win Rate: {round(float(str(s['Win Rate']).replace('%',''))):.0f}%\n"
                        f"Lump Sum ROI: {round(float(str(s['Lump Sum ROI']).replace('%',''))):.0f}%\n"
                        f"Str ROI Ratio: {s['ROI Ratio (Strategy/Lump Sum)']}"
                    )
                    # Calculate x position for each box
                    x_pos = 0.2 + i * (box_width + 0.05)
                    ax2.text(
                        x_pos, 0.98, box_text,
                        transform=ax2.transAxes,
                        fontsize=10,
                        verticalalignment='top',
                        horizontalalignment='left',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=str_colors[strategy], alpha=0.9)
                    )

        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.title(f"{self.ticker} {self.timeframe}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        #plt.legend()

        _plot_backtest_results()

        plt.tight_layout()

        if self.save_img:
            img_path = os.path.join(figs_dir, f"{self.ticker}_{self.timeframe}_plot.png")
            plt.savefig(img_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {img_path}")
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
            buy_signals = df[(df["str_def_long"].diff() == 1)]
            sell_signals = df[(df["str_def_long"].diff() == -1)]
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
    if True: # Fetch and save data for multiple tickers
        asset_type = "stocks"
        ticker = None #'MIGA.BE','TL0.DE','MIGA.BE','1X00.BE','1NW.BE','NVD.DE','TT8.DE','1QZ.DE','M44.BE','SGM.BE']
        # asset_type = "crypto-USD"
        # tickers = ['SUI20947-USD'] #["BTC-USD","SOL-USD"]

        yfin = YF(ticker, asset_type)
        if False: # Check if tickers are valid
            query = 'PTX.BE'
            yfin.search_query(query)
        if True:
            yfin.fetch_and_save_data()

    else: # Calculate, backtest and plot specific ticker
        asset_type = 'equity' #'crypto-USD'
        ticker = 'MIGA.BE' #'SOL-USD'
        timeframe = '1D'

        # Check if indicators file exists
        overwrite = False
        results_csv = os.path.join(results_dir, f"{ticker}_{timeframe}_data_indicators_signals.csv")
        if not overwrite and os.path.exists(results_csv):
            print(f"--- Indicators and signals for {ticker} already exist. Skipping calculation.")
            df = pd.read_csv(results_csv, index_col="Date")
            df.index = pd.to_datetime(df.index, format='%Y-%m-%d', errors='coerce')
        else:
            print(f"--- Calculating indicators and signals for {ticker} with {timeframe} timeframe.")
            calc = Calc(ticker, asset_type, timeframe, results_csv)
            # Load ticker data
            df = calc.get_ticker_data()
            # Compute indicator values
            df = calc.get_default_indicators()
            # Get trading signals
            df = calc.get_default_signals()
            # Append additional signals if needed
            calc.append_additional_signals(df, prefix="str_def_2")

        if True: # Get backtest results
            period = ("2022-01-02", "2025-05-30")
            bt = Backtest(ticker, df, period, timeframe, results_csv)
            df = bt.run_backtest()

        if True: # Plot results
            date_range = ("2022-01-02", "2025-05-30")
            ploter = Ploter(ticker, df, date_range, timeframe, save_img=True)
            ploter.plot_plt()