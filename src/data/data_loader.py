"""A class for loading data from various sources, such as CSV files, databases, or APIs."""
import pandas as pd
import requests
import yfinance


class DataLoader:
    """A class for loading data from various sources, such as CSV files, databases, or APIs."""

    def __init__(self) -> None:
        """Initialize the DataLoader."""

    def load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load data from a pickle file."""
        return pd.read_parquet(file_path)

    def load_csv(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        return pd.read_csv(file_path)

    def load_api(self, url: str) -> pd.DataFrame:
        """Load data from an API."""
        response = requests.get(url, timeout=600)
        response.raise_for_status()

        return pd.DataFrame(response.json())

    def load_yfinance(self, tickers: str, start: str, end: str, interval: str) -> pd.DataFrame:
        """"Load data from yfinance."""
        return yfinance.download (tickers = tickers, start = start,
                                  end = end, interval = interval)

    def load_spx(self, start: str, end: str, interval: str, path: str | None = None) -> pd.DataFrame:
        """Load SPX data, firstly trying to find in path."""
        if path is not None:
            try:
                return self.load_parquet(path)
            except Exception:
                pass

        return self.load_yfinance(tickers="^SPX", start=start, end=end, interval=interval)

    def load_option_chain(self, ticker: str = "^SPX", n_maturities: int = 5) -> pd.DataFrame:
        """Snapshot of the option chain for the nearest n_maturities expiries."""
        tk = yfinance.Ticker(ticker)
        expiries = tk.options[:n_maturities]

        frames = []
        for expiry in expiries:
            oc = tk.option_chain(expiry)
            for kind, df in (("call", oc.calls), ("put", oc.puts)):
                df: pd.DataFrame = df.copy()
                df["expiry"] = expiry
                df["type"] = kind
                frames.append(df)

        return pd.concat(frames, ignore_index=True)

