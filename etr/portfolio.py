"""Portfolios (PROTOCOL.md section 2): the registered books, their returns on the union
calendar, and joint filtered historical simulation.

A book has fixed weights w (gross notional sum |w| = 1), rebalanced daily to those weights
of its value (a constant-mix book), with each leg's P&L a percentage of its own notional
(FX-neutral). Its daily return is
    p_t = sum_i w_i (exp(r_it) - 1),
on the union of the trading days of the book's own markets (a spread of two markets does not
"trade" on a day both are closed): a market that is closed contributes nothing that day, and
its next return covers the gap. An h-day loss compounds the daily returns:
L = 1 - prod(1 + p). Writing the book's log return as log(1 + p_t), a book is then just
another "market" for the single-market engine (etr.engine), with one side: the book itself.

Two ways to forecast a book's loss:
  univariate  the engine on the book's own return history (what most desks do: revalue
              today's book on past returns). With fixed weights, EWMA on the book's returns
              equals w' (EWMA covariance) w, so this also covers the RiskMetrics-covariance
              Gaussian baseline.
  joint FHS   each market's own scale forecast times standardised residual *vectors*
              resampled by date, so whatever happened together on the same day (TTF and NBP
              move together; spreads depend on it) comes from the data, not from a fitted
              correlation or copula (session 1: joint tails too sparse to fit).
"""
import numpy as np
import pandas as pd

from etr import engine, tails
from etr.markets import MARKETS

ORDER = list(MARKETS)                               # TTF, NBP, HH, BRENT, WTI, EUA
VOL_WINDOW = ("2014-01-01", "2018-12-31")           # PROTOCOL: risk weights from 2014-18 vol
RANDOM_BOOKS, BOOK_SEED = 50, 20261024


def union_returns(rets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily log returns of each market on the union calendar (0 where the market is closed
    or has no price; a return spanning a gap lands on the day the market reopens), and the
    matching open mask (True where the market has a return that day)."""
    R = pd.DataFrame({n: rets[n]["ret"] for n in ORDER if n in rets}).sort_index()
    return R.fillna(0.0), R.notna()


def book_calendar(open_: pd.DataFrame, w: pd.Series) -> pd.DatetimeIndex:
    """The book's own trading days: days on which at least one of its markets trades."""
    return open_.index[open_[w.index[w != 0]].any(axis=1).to_numpy()]


def book_return(U: pd.DataFrame, w: pd.Series, open_: pd.DataFrame | None = None) -> pd.DataFrame:
    """A book's daily log return log(1 + sum_i w_i (exp(r_i) - 1)), in the single-market
    frame layout (ret, span), on the book's own calendar when the open mask is given."""
    idx = U.index if open_ is None else book_calendar(open_, w)
    p = (np.expm1(U.loc[idx, w.index]) * w).sum(axis=1)
    return pd.DataFrame({"ret": np.log1p(p), "span": 1}, index=idx)


def market_vols(U: pd.DataFrame, open_: pd.DataFrame, window=VOL_WINDOW) -> pd.Series:
    """Daily volatility of each market over the window, on its own trading days."""
    a, b = window
    sel = (U.index >= a) & (U.index <= b)
    return U[sel].where(open_[sel]).std()


def books(U: pd.DataFrame, open_: pd.DataFrame) -> dict[str, pd.Series]:
    """The registered books P1-P5 and the 50 random books R01-R50 (P6)."""
    vol = market_vols(U, open_)
    inv = 1 / vol

    def gross1(w):
        w = pd.Series(w, dtype=float).reindex(ORDER, fill_value=0.0)
        return w / w.abs().sum()

    out = {"P1": gross1(inv)}
    out["P2"] = -out["P1"]
    out["P3"] = gross1({"TTF": inv["TTF"], "NBP": -inv["NBP"]})
    out["P4"] = gross1({"BRENT": inv["BRENT"], "WTI": -inv["WTI"]})
    # P5: each leg risk-weighted inside, then the legs scaled to equal standalone risk.
    gas, oil = gross1({"TTF": inv["TTF"], "NBP": inv["NBP"]}), gross1({"BRENT": inv["BRENT"], "WTI": inv["WTI"]})
    a, b = VOL_WINDOW
    sel = (U.index >= a) & (U.index <= b)
    pnl_vol = lambda w: (np.expm1(U[sel]) * w).sum(axis=1).std()   # leg P&L is linear in its weights  # noqa: E731
    s_gas, s_oil = pnl_vol(gas), pnl_vol(oil)
    out["P5"] = gross1(gas * s_oil - oil * s_gas)                   # equal risk: weight inversely to leg vol
    W = np.random.default_rng(BOOK_SEED).normal(size=(RANDOM_BOOKS, len(ORDER)))
    for k, row in enumerate(W, start=1):
        out[f"R{k:02d}"] = gross1(dict(zip(ORDER, row)))
    return out


# ---- Joint filtered historical simulation ----

def residual_vectors(rets: dict[str, pd.DataFrame], insample: dict, year: int, scale_col: str) -> np.ndarray:
    """Training residual vectors for a forecast year: each market's standardised daily
    residual (from the scale model's year-Y parameters, as in etr.engine) placed on the date
    of the return, keeping only dates on which every market traded, so each row is one real
    day's joint move."""
    cut = pd.Timestamp(f"{year - 1}-12-31")
    cols = {}
    for n in ORDER:
        df = rets[n]
        span = df["span"].shift(-1)
        z = df["ret"].shift(-1) / np.sqrt(span.where(span > 0) * insample[(n, year, scale_col)].reindex(df.index))
        target = pd.Series(df.index, index=df.index).shift(-1)
        z.index = pd.DatetimeIndex(target)
        cols[n] = z[(z.index <= cut) & np.isfinite(z.to_numpy())]
    Z = pd.DataFrame(cols).dropna()
    return Z[ORDER].to_numpy()


def simulate_log_nav(sig: np.ndarray, opens: np.ndarray, pool: np.ndarray, Wm: np.ndarray,
                     rng: np.random.Generator, n_paths: int) -> np.ndarray:
    """Simulated log growth of each book over h days (origins x paths x books).
    sig: daily vol per origin and market (origins x markets); opens: whether each market
    trades on each simulated day (origins x h x markets); pool: residual vectors (rows = real
    days); Wm: weights (markets x books). Each simulated day draws one residual vector for
    all markets at once, so same-day dependence is kept; a closed market does not move."""
    n_o, h, _ = opens.shape
    log_nav = np.zeros((n_o, n_paths, Wm.shape[1]))
    for k in range(h):
        z = pool[rng.integers(0, len(pool), (n_o, n_paths))]                          # origins x paths x markets
        r = sig[:, None, :] * z * opens[:, k][:, None, :]
        log_nav += np.log1p(np.expm1(r) @ Wm)
    return log_nav


def joint_fhs(rets: dict[str, pd.DataFrame], fc: dict[str, pd.DataFrame], insample: dict, years: list[int],
              scale_name: str, weights: dict[str, pd.Series], h: int = 1, scale_col: str | None = None,
              n_paths: int = 2000, seed: int = 20261024, keep_draws: bool = True) -> dict:
    """h-day forecasts of every book by joint filtered historical simulation.

    Each market's daily volatility is its own scale forecast at its last trading day on or
    before the origin (for h = 10, the h-day variance forecast spread evenly over the days).
    Each simulated day draws one training residual vector (a real day's joint move) and
    applies it to the markets open that day (exchange holidays are known in advance); the
    book compounds daily over its own trading days. Books with the same markets share a
    calendar and the same simulated paths. Returns {(book, "long"): Forecast}."""
    U, open_ = union_returns(rets)
    scale_col = scale_col or f"{scale_name}_1"
    out = {(b, "long"): [] for b in weights}
    groups = {}
    for b, w in weights.items():
        groups.setdefault(tuple(w.index[w != 0]), []).append(b)
    var_all = pd.DataFrame({n: fc[n][scale_col] for n in ORDER}).reindex(U.index).ffill(limit=5)
    realised = {b: tails.forward_returns(book_return(U, w, open_), h) for b, w in weights.items()}
    for y in years:
        pool = residual_vectors(rets, insample, y, f"{scale_name}_1")
        for g, (markets, names) in enumerate(groups.items()):
            cal = book_calendar(open_, weights[names[0]])
            opn = open_.loc[cal, ORDER].to_numpy()
            days = cal[cal.year == y]
            pos = cal.get_indexer(days)
            ok = pos + h < len(cal)
            days, pos = days[ok], pos[ok]
            sig = np.sqrt(var_all.loc[days, ORDER].to_numpy() / h)                          # origins x markets
            valid = np.isfinite(sig[:, [ORDER.index(m) for m in markets]]).all(axis=1)
            days, pos, sig = days[valid], pos[valid], np.nan_to_num(sig[valid])
            opens = np.stack([opn[pos + k] for k in range(1, h + 1)], axis=1)              # origins x h x markets
            Wm = np.column_stack([weights[b].reindex(ORDER).to_numpy() for b in names])
            rng = np.random.default_rng([seed, y, h, g])
            log_nav = simulate_log_nav(sig, opens, pool, Wm, rng, n_paths)
            for j, b in enumerate(names):
                R = realised[b].loc[days]
                keep = R.notna().to_numpy()
                out[(b, "long")].append(engine.simulated_forecast(R[keep], log_nav[keep, :, j], "long", keep_draws))
    return {k: tails.concat(v) for k, v in out.items()}
