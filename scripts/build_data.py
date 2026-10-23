"""Build the processed datasets from the Bloomberg workbook.

Writes (all gitignored, never commit):
  data/interim/bbg_long.parquet     every block of the workbook, long format
  data/processed/returns.parquet    contract-consistent daily returns per market
  data/processed/ohlc_m1.parquet    M1 open/high/low/settle with validity flags
  data/processed/implied_vol.parquet  front-month implied vol (HH, Brent, WTI)

Run:  python scripts/build_data.py [--refresh]
"""
import argparse

import pandas as pd

import etr
from etr.markets import MARKETS
from etr.returns import DEFAULT_ROLL_OFFSET, implied_vol, load_long, m1_ohlc, market_returns


def main(refresh: bool = False) -> None:
    long = load_long(refresh=refresh)
    returns = pd.concat([market_returns(long, m, DEFAULT_ROLL_OFFSET) for m in MARKETS.values()])
    ohlc = pd.concat([m1_ohlc(long, m) for m in MARKETS.values()])
    iv = pd.concat([s for m in MARKETS if (s := implied_vol(long, m)) is not None], axis=1)
    etr.PROCESSED.mkdir(parents=True, exist_ok=True)
    returns.to_parquet(etr.PROCESSED / "returns.parquet")
    ohlc.to_parquet(etr.PROCESSED / "ohlc_m1.parquet")
    iv.to_parquet(etr.PROCESSED / "implied_vol.parquet")
    summary = returns.groupby("market").agg(
        first=("ret", lambda r: r.dropna().index.min().date()),
        last=("ret", lambda r: r.dropna().index.max().date()),
        days=("ret", "size"), missing=("ret", lambda r: int(r.isna().sum())),
        rolls=("roll", "sum"))
    print(summary.to_string())
    print(f"wrote {etr.PROCESSED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-read the workbook even if cached")
    main(ap.parse_args().refresh)
