"""Volatility (scale) forecasts.

Phase 1 holds only what the baselines need: the RiskMetrics EWMA and the point-in-time
level calibration. Phase 2 adds weekday factors, GARCH, HAR and implied vol.

Timing: a forecast is indexed by its origin t (the close at which it is made) and uses
returns up to and including t. It forecasts the daily variance of the following days.
Adapted from energy-vol-targeting (src/evt/models/simple.py, walkforward.py).
"""
import numpy as np
import pandas as pd


def daily_variance(df: pd.DataFrame) -> pd.Series:
    """Squared log return per trading day covered; a multi-day return (after a missing
    price) is divided by its span. NaN where there is no return."""
    return df["ret"] ** 2 / df["span"].where(df["span"] > 0)


def ewma_variance(df: pd.DataFrame, lam: float = 0.94, min_periods: int = 40) -> pd.Series:
    """RiskMetrics exponentially weighted variance, s2_t = lam s2_{t-1} + (1 - lam) r_t^2:
    the forecast of next day's variance made at the close of t. lam = 0.94 is the
    RiskMetrics daily value (half-life about 11 days). NaN until `min_periods` returns."""
    v = daily_variance(df)
    s2 = v.ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean()
    return s2.where(v.notna().cumsum() >= min_periods)


def calibration_scale(df: pd.DataFrame, var_fc: pd.Series, min_obs: int = 120) -> pd.Series:
    """Point-in-time level calibration c for each origin: the mean of realised / forecast
    variance, r_s^2 / (span_s x var_fc_{s-1}), over all days s before 1 January of the
    origin's year. Refreshed once a year. Why: with estimation noise E[r^2 / f] > 1 even for
    an unbiased forecast f (1/f is convex), so uncalibrated forecasts under-state risk; the
    previous project found EWMA under-forecasts by 8-27% (weekly) and 9-16% (daily, 2011-18).
    Uses only past data (the calibration in energy-vol-targeting, applied to daily data)."""
    ratio = (daily_variance(df) / var_fc.shift(1)).replace([np.inf, -np.inf], np.nan)
    years = df.index.year
    c = pd.Series(np.nan, index=df.index)
    for y in np.unique(years):
        past = ratio[years < y].dropna()
        if len(past) >= min_obs:
            c[years == y] = past.mean()
    return c


def calibrated_ewma_variance(df: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    """EWMA variance times its point-in-time calibration scale."""
    v = ewma_variance(df, lam)
    return v * calibration_scale(df, v)
