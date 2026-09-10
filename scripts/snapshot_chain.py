"""Capture one dated SPX option-chain snapshot (PROJECT_PLAN.md §7).

Yahoo serves only the *live* chain: `option_chain(date=...)` selects an expiry, not an
as-of date, and there is no history behind it. A trading day that is not captured is lost
permanently, which is why this runs daily rather than in one backfill.

Writes `data/chains/spx_option_chain_<YYYY-MM-DD>.parquet`. The Phase 1 snapshot at
`data/spx_option_chain.parquet` is frozen -- every benchmark number is calibrated to it --
and this script never touches it.

    uv run python scripts/snapshot_chain.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.data_loader import DataLoader  # noqa: E402

OUT_DIR = ROOT / "data" / "chains"
logger = logging.getLogger("snapshot_chain")


def snapshot_date(chain: pd.DataFrame) -> pd.Timestamp:
    """Read the capture date off the chain itself rather than off the wall clock.

    Yahoo stamps every row with `snapshot_time`, the instant the quotes were taken. Wall-clock
    UTC disagrees with it whenever the script runs after 20:00 ET, because that is already the
    next day in UTC -- so a Tuesday close would be filed under Wednesday and the day-over-day
    ordering would break silently. The quotes' own stamp is the valuation date by definition.
    """
    return pd.to_datetime(chain["snapshot_time"].iloc[0], utc=True).normalize()


def is_stale(chain: pd.DataFrame) -> bool:
    """Report whether Yahoo returned quotes that never traded on their own snapshot date.

    On weekends and market holidays the endpoint still answers, echoing the previous session's
    quotes. Saving those would write a duplicate under a fresh date and quietly inflate the
    count of distinct snapshots. `lastTradeDate` is per-contract; if not one contract traded on
    the snapshot date, this is a replay of an earlier session.
    """
    traded = pd.to_datetime(chain["lastTradeDate"], utc=True).dt.normalize()

    return bool((traded == snapshot_date(chain)).sum() == 0)


def main() -> int:
    """Fetch today's chain and persist it under its own snapshot date. Returns an exit code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    loader = DataLoader()
    chain = loader.load_option_chain()

    stamp = snapshot_date(chain)
    out_path = OUT_DIR / f"spx_option_chain_{stamp.date().isoformat()}.parquet"

    if out_path.exists():
        logger.info("%s already captured; nothing written.", out_path.name)
        return 0
    if is_stale(chain):
        logger.info("No contract traded on %s (weekend/holiday); nothing written.", stamp.date())
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    loader.save_parquet(chain, str(out_path))
    logger.info("%s -- %d quotes, %d expiries, spot %.2f",
                out_path.name, len(chain), chain["expiry"].nunique(), chain["spot"].iloc[0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
