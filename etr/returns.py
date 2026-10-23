"""From the Bloomberg workbook to contract-consistent daily returns.

Adapted from energy-vol-targeting: src/evt/dataprep/load.py (workbook parser),
rolls.py (contract-consistent returns), quality.py (OHLC validity), detect.py (roll
verification) and src/evt/models/implied.py. The logic is unchanged; the modules are
merged because together they are one step, "data in, returns out".

Workbook layout (see its README sheet): each data sheet is a row of blocks
`Date | field1 | field2 ... | spacer`. Row 3 holds field names, row 4 the ticker, row 5 a
label, row 6 units, and data starts on row 7. Non-trading days are blank (BDH option
Fill=B). We keep them as NaN and never forward-fill, because a filled price is a fake zero
return that understates volatility.

Rolls. Bloomberg's `TZT1` is not one contract: on the business day after each expiry it
switches to the next contract, so log(M1_t / M1_{t-1}) on that day books the calendar
spread as a price move. We instead track the contract actually held and only ever compute
a return as that contract's price over its own previous price.
- Generic M_k on date t is the k-th listed contract whose LTD is on or after t.
- We roll at the close of the trading day `roll_offset` business days before the front
  contract's LTD (sell the old contract, buy the next one at that day's settlement), so the
  held contract is always M1 or M2. This avoids expiry-day disorder (WTI May-20 settled at
  -$37.63 the day before it expired while June stayed positive).
- The return on day t belongs to the contract held over (t-1, t], chosen at the close of
  t-1. Nothing about day t is needed to decide what is held into day t.
- Missing prices are never filled. The next valid return spans the gap (`span` > 1), so
  cumulative P&L stays correct.
"""
import datetime as dt
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

import etr
from etr.expiry import business_calendar, ltd_table, make_calendar
from etr.markets import Market

HEADER_ROW, TICKER_ROW, LABEL_ROW, UNIT_ROW, FIRST_DATA_ROW = 3, 4, 5, 6, 7
# PX_LAST appears on several sheets for the same ticker; prefer the curve sheets.
SHEET_PRIORITY = ["A_GasCurves", "D_OilCoalCarbon", "E_OHLC_Vol", "F_Liquidity"]

# Roll this many business days before the front contract's LTD: uniform and conventional,
# fixed in the previous project before any results existed. HH and Brent volume has mostly
# moved to M2 by 3-4 days before expiry; TTF and NBP front months stay the more liquid
# contract to the end, so for gas we hold M2 for the last week of each month.
DEFAULT_ROLL_OFFSET = 5


# ---- Workbook ----

def read_workbook(path: Path) -> pd.DataFrame:
    """Parse every block on every data sheet into columns
    [sheet, ticker, label, unit, field, date, value]. Blank or non-numeric cells -> NaN."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    frames = []
    for ws in wb.worksheets:
        rows = list(ws.iter_rows(values_only=True))
        if len(rows) < FIRST_DATA_ROW or "Date" not in (rows[HEADER_ROW - 1] or ()):
            continue  # README, Settings, Map, Log
        frames.extend(_parse_sheet(ws.title, rows))
    wb.close()
    out = pd.concat(frames, ignore_index=True)
    return out[out["date"].notna()].reset_index(drop=True)


def _parse_sheet(sheet: str, rows: list[tuple]) -> list[pd.DataFrame]:
    header = rows[HEADER_ROW - 1]
    meta = [rows[i - 1] for i in (TICKER_ROW, LABEL_ROW, UNIT_ROW)]
    body = rows[FIRST_DATA_ROW - 1:]
    width = len(header)

    def column(j):
        return [r[j] if j < len(r) else None for r in body]

    frames = []
    for start in (j for j, h in enumerate(header) if h == "Date"):
        # Keep only real datetimes; anything else (stray text) becomes NaT and is dropped.
        dates = pd.to_datetime(pd.Series([v if isinstance(v, dt.datetime) else None for v in column(start)]))
        j = start + 1
        while j < width and header[j] not in (None, "Date"):
            # Blanks ("") and Bloomberg error strings ("#N/A ...") both coerce to NaN.
            values = pd.to_numeric(pd.Series(column(j), dtype=object), errors="coerce")
            frames.append(pd.DataFrame({
                "sheet": sheet,
                "ticker": meta[0][start],
                "label": meta[1][start],
                "unit": meta[2][start],
                "field": header[j],
                "date": dates,
                "value": values.astype(float),
            }))
            j += 1
    return frames


def load_long(path: Path | None = None, refresh: bool = False) -> pd.DataFrame:
    """Long table, cached as data/interim/bbg_long.parquet (rebuilt if the workbook is newer)."""
    path = path or etr.raw_workbook()
    cache = etr.INTERIM / "bbg_long.parquet"
    if not refresh and cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cache)
    long = read_workbook(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    long.to_parquet(cache, index=False)
    return long


def wide(long: pd.DataFrame, field: str, tickers: list[str]) -> pd.DataFrame:
    """Date x ticker panel for one field. If a ticker/field is on several sheets, the
    first sheet in SHEET_PRIORITY wins. Rows where every ticker is blank are dropped."""
    sub = long[(long["field"] == field) & long["ticker"].isin(tickers)].copy()
    rank = {s: i for i, s in enumerate(SHEET_PRIORITY)}
    sub["rank"] = sub["sheet"].map(rank).fillna(len(rank))
    best = sub.groupby("ticker")["rank"].transform("min")
    sub = sub[sub["rank"] == best]
    out = sub.pivot_table(index="date", columns="ticker", values="value", aggfunc="first")
    return out.reindex(columns=tickers).dropna(how="all").sort_index()


# ---- Contract-consistent returns ----

def generic_panel(long: pd.DataFrame, market: Market) -> pd.DataFrame:
    """Settlement prices with integer tenor columns, on the market's trading days
    (dates where M1 or M2 has a price)."""
    tickers = [market.ticker(k) for k in market.tenors]
    px = wide(long, "PX_LAST", tickers)
    px.columns = list(market.tenors)
    return px[px[[1, 2]].notna().any(axis=1)]


def roll_schedule(ltd: pd.Series, cal: pd.DatetimeIndex, roll_offset: int) -> pd.Series:
    """Date on whose close we roll out of each contract: `roll_offset` business days
    before its LTD."""
    pos = cal.get_indexer(pd.DatetimeIndex(ltd.values))
    if (pos < 0).any():
        raise ValueError("every LTD must be a calendar business day")
    roll = pd.Series(cal[pos - roll_offset], index=ltd.index, name="roll_date")
    # The roll out of contract i must come after contract i-1 has expired, otherwise
    # we would need M3.
    if not (roll.values[1:] > ltd.values[:-1]).all():
        raise ValueError("roll_offset too large for this contract spacing")
    return roll


def build_returns(panel: pd.DataFrame, ltd: pd.Series, cal: pd.DatetimeIndex,
                  roll_offset: int = DEFAULT_ROLL_OFFSET) -> pd.DataFrame:
    """Daily contract-consistent log returns for a rolled futures position.

    Columns
      ret        log return of the contract held over (prev trading day, t]
      span       trading days that return covers (1 normally; >1 after a missing price)
      contract   delivery month of that contract
      tenor      its generic tenor on day t (1 or 2)
      px         its settlement on day t
      roll       True if we roll at the close of t (switch to a new contract)
      px_next    settlement on day t of the contract held after the close (== px unless roll)
      front_switch  True if Bloomberg's M1 changed contract on day t
      ret_naive  log(M1_t / M1_{t-1}): the wrong, unadjusted return, kept for comparison
    """
    dates = panel.index
    n = len(dates)
    ltd_v = pd.DatetimeIndex(ltd.values).values
    roll_v = pd.DatetimeIndex(roll_schedule(ltd, cal, roll_offset).values).values
    front = np.searchsorted(ltd_v, dates.values, side="left")
    held_after = np.searchsorted(roll_v, dates.values, side="right")  # held after close of t

    logp = np.log(panel.where(panel > 0)).to_numpy()   # non-positive prices -> NaN
    col = {k: j for j, k in enumerate(panel.columns)}

    def lp(i: int, c: int) -> float:
        k = c - front[i] + 1
        return logp[i, col[k]] if k in col else np.nan

    ret = np.full(n, np.nan)
    span = np.zeros(n, dtype=int)
    held = np.empty(n, dtype=int)
    px = np.full(n, np.nan)
    px_next = np.full(n, np.nan)
    anchor_c, anchor_lp, anchor_i = -1, np.nan, -1

    for i in range(n):
        c = held_after[i - 1] if i > 0 else held_after[0]
        held[i] = c
        p = lp(i, c)
        px[i] = np.exp(p)
        if i > 0 and anchor_c == c and not np.isnan(anchor_lp) and not np.isnan(p):
            ret[i] = p - anchor_lp
            span[i] = i - anchor_i
        # Position after today's close.
        c_next = held_after[i]
        p_next = lp(i, c_next)
        px_next[i] = np.exp(p_next)
        if not np.isnan(p_next):
            anchor_c, anchor_lp, anchor_i = c_next, p_next, i
        elif c_next != anchor_c:
            anchor_c, anchor_lp, anchor_i = c_next, np.nan, i   # entry price unknown
        # else: same contract, no price today -> keep the old anchor; next return spans the gap

    m1 = logp[:, col[1]]
    out = pd.DataFrame({
        "ret": ret,
        "span": span,
        "contract": ltd.index[held].astype(str),
        "tenor": held - front + 1,
        "px": px,
        "roll": held_after != held,   # position after the close differs from the one held into t
        "px_next": px_next,
        "front_switch": np.r_[False, front[1:] != front[:-1]],
        "ret_naive": np.r_[np.nan, np.diff(m1)],
    }, index=dates)
    out.index.name = "date"
    return out


_DEFAULT = object()


def market_returns(long: pd.DataFrame, market: Market, roll_offset: int = DEFAULT_ROLL_OFFSET, *,
                   ltd_shift: int = 0, expiry_holidays=_DEFAULT) -> pd.DataFrame:
    """Full pipeline for one market.

    Two calendars: expiry rules count business days on `business_calendar` (trading days
    minus the market's expiry holidays), while returns and our own roll dates use every
    trading day. `ltd_shift` and `expiry_holidays` exist only for verification, to test
    deliberately wrong expiry assumptions."""
    panel = generic_panel(long, market)
    cal = make_calendar(panel.index)
    if expiry_holidays is _DEFAULT:
        bcal = business_calendar(cal, market.expiry_holidays, market.expiry_holidays_since)
    else:
        bcal = business_calendar(cal, *expiry_holidays) if expiry_holidays else cal
    ltd = ltd_table(market, bcal, panel.index[0], panel.index[-1])
    if ltd_shift:
        pos = cal.get_indexer(pd.DatetimeIndex(ltd.values)) + ltd_shift
        ltd = pd.Series(cal[pos], index=ltd.index, name="ltd")
    out = build_returns(panel, ltd, cal, roll_offset)
    out.insert(0, "market", market.name)
    return out


# ---- Data quality and other inputs ----

def ohlc_valid(o: pd.Series, h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    """True where all four prices exist, are positive, and low <= open, close <= high.
    PX_LAST is the exchange settlement, which can sit outside the traded range when a
    contract barely trades (early TTF, EUA), so range-based estimators need this mask."""
    lo, hi = np.minimum(o, c), np.maximum(o, c)
    ok = pd.concat([o, h, l, c], axis=1).notna().all(axis=1) & (l > 0)
    return ok & (l <= lo) & (hi <= h)


def m1_ohlc(long: pd.DataFrame, market: Market) -> pd.DataFrame:
    """M1 open/high/low/settle with a validity flag. OHLC exists only for M1, which is not
    always the contract held (we hold M2 in the last days before expiry)."""
    t = market.ticker(1)
    fields = {"PX_OPEN": "open", "PX_HIGH": "high", "PX_LOW": "low", "PX_LAST": "close"}
    sub = long[(long.sheet == "E_OHLC_Vol") & (long.ticker == t) & long.field.isin(fields)]
    df = sub.pivot_table(index="date", columns="field", values="value").rename(columns=fields)
    df = df.reindex(columns=list(fields.values()))
    df["valid"] = ohlc_valid(df.open, df.high, df.low, df.close)
    df.insert(0, "market", market.name)
    return df


IV_TICKERS = {"HH": "NG1 Comdty", "BRENT": "CO1 Comdty", "WTI": "CL1 Comdty"}


def implied_vol(long: pd.DataFrame, market: Market, max_fill: int = 5) -> pd.Series | None:
    """Front-month option implied vol (mean of call and put, annualised fraction) on the
    market's trading days, forward-filled over at most `max_fill` missing days (past
    information only). It is missing about one day a month, around option expiry; those
    dates have no row at all in the workbook, so the fill must come after reindexing to the
    trading days. Only HH, Brent and WTI have it through 2026; TTF, NBP and EUA stop in
    2020-21."""
    t = IV_TICKERS.get(market.name)
    if t is None:
        return None
    sub = long[(long.sheet == "E_OHLC_Vol") & (long.ticker == t)
               & long.field.isin(["HIST_CALL_IMP_VOL", "HIST_PUT_IMP_VOL"])]
    iv = sub.pivot_table(index="date", columns="field", values="value").mean(axis=1) / 100
    days = generic_panel(long, market).index
    return iv.reindex(days).ffill(limit=max_fill).rename(market.name)


# ---- Roll verification (never used to build the schedule) ----

def oi_jump(oi: pd.Series, signed: bool = True) -> pd.Series:
    """Daily log change in M1 open interest. When M1 switches from an expiring contract
    (OI run down) to the next one, OI jumps *up*, so by default only rises score. Use
    signed=False where a switch can also be a fall (EUA: big Dec contract -> small Mar)."""
    oi = oi.where(oi > 0).dropna()
    chg = np.log(oi).diff()
    return chg if signed else chg.abs()


def event_offsets(events: pd.DatetimeIndex, score: pd.Series, dates: pd.DatetimeIndex,
                  window: int = 3) -> pd.Series:
    """For each event date, the offset (in trading days of `dates`) of the largest `score`
    within +/- window. NaN if the window has no score."""
    score = score.reindex(dates)
    pos = dates.get_indexer(events)
    offs = []
    for p in pos:
        lo, hi = max(p - window, 0), min(p + window, len(dates) - 1)
        w = score.iloc[lo:hi + 1]
        offs.append(np.nan if w.isna().all() else dates.get_loc(w.idxmax()) - p)
    return pd.Series(offs, index=events, name="offset")
