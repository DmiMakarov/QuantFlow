"""A class for loading data from various sources, such as CSV files, databases, or APIs."""
import pandas as pd
import pyarrow.parquet as pq
import requests
import yfinance


class DataLoader:
    """A class for loading data from various sources, such as CSV files, databases, or APIs."""

    def __init__(self) -> None:
        """Initialize the DataLoader."""

    def load_parquet(self, file_path: str) -> pd.DataFrame:
        """Load data from a pickle file."""
        return pq.read_table(file_path).replace_schema_metadata(None).to_pandas()

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

    @staticmethod
    def _underlying_snapshot(option_chain: object) -> tuple[float, pd.Timestamp]:
        """Pull the spot and snapshot time from an option_chain().underlying dict.

        Yahoo returns the underlying quote alongside every expiry; regularMarketPrice
        is the spot (s_0) consistent with these quotes and regularMarketTime is the
        epoch-second timestamp of the snapshot (the calibration valuation date).
        """
        underlying = option_chain.underlying or {}
        spot = float(underlying["regularMarketPrice"])
        snapshot_time = pd.to_datetime(underlying["regularMarketTime"], unit="s", utc=True)

        return spot, snapshot_time

    def load_option_chain(self, ticker: str = "^SPX", n_maturities: int = 5) -> pd.DataFrame:
        """Snapshot of the option chain for the nearest n_maturities expiries.

        Each row carries the spot and snapshot time captured atomically from Yahoo's
        underlying quote, plus time-to-maturity in years. This makes the frame fully
        self-contained for calibration: s_0, the valuation date, and t all live here,
        with no dependency on a separate price file.
        """
        tk = yfinance.Ticker(ticker)
        expiries = tk.options[:n_maturities]

        spot: float | None = None
        snapshot_time: pd.Timestamp | None = None

        frames = []
        for expiry in expiries:
            oc = tk.option_chain(expiry)
            if spot is None:
                spot, snapshot_time = self._underlying_snapshot(oc)
            for kind, side in (("call", oc.calls), ("put", oc.puts)):
                labelled = side.copy()
                labelled["expiry"] = expiry
                labelled["type"] = kind
                frames.append(labelled)

        chain = pd.concat(frames, ignore_index=True)
        chain["spot"] = spot
        chain["snapshot_time"] = snapshot_time
        # time to maturity in years, measured from the snapshot (not "today")
        ttm_days = (pd.to_datetime(chain["expiry"], utc=True) - snapshot_time).dt.total_seconds()
        chain["ttm"] = ttm_days / (365.0 * 24 * 3600)

        return chain

    @staticmethod
    def save_parquet(df: pd.DataFrame, file_path: str) -> None:
        """Write a DataFrame to parquet, preserving any DatetimeIndex as a column.

        The price series carries its dates in the index; resetting it first means the
        dates survive the round-trip instead of being dropped to a RangeIndex on save.
        """
        out = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df
        out.to_parquet(file_path, index=False)

