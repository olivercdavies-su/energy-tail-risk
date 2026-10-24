"""From a forecast distribution of returns to position losses: VaR, ES, PIT and draws.

Positions. A unit long or short position in the rolled contract (contract-consistent
returns, so rolls inside the horizon are what a holder would earn). Over h trading days
after the origin t, with R the sum of daily log returns, the P&L as a fraction of notional
is exp(R) - 1 for a long and the negative for a short. Losses are the negative P&L:
    long:  L = 1 - exp(R)      short:  L = exp(R) - 1.
The long loses when prices fall (left tail of R), the short when they rise (right tail);
the two are evaluated separately because energy tails are asymmetric.

Every model produces a `Forecast`: a frame with, for each origin, the realised loss, VaR
and ES at each tail probability in ALPHAS, the PIT u = P(L >= realised loss), and a way
to simulate losses from each day's forecast distribution (for the Acerbi-Szekely test).

Models here (Phase 1): the scale x shape family with a normal shape (RiskMetrics and
calibrated EWMA-normal), and historical simulation. Phase 3 adds fat-tailed shapes.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

SIDES = ("long", "short")
ALPHAS = (0.05, 0.025, 0.01)
HS_WINDOW = 500


def col(kind: str, alpha: float) -> str:
    """Column name: col("var", 0.01) -> "var99", col("es", 0.025) -> "es975"."""
    return f"{kind}{f'{100 * (1 - alpha):g}'.replace('.', '')}"


def to_loss(logret, side: str):
    """Loss as a fraction of notional for a unit position, from an h-day log return."""
    if side == "long":
        return -np.expm1(logret)
    if side == "short":
        return np.expm1(logret)
    raise ValueError(side)


def forward_returns(df: pd.DataFrame, h: int) -> pd.Series:
    """Realised h-day log return after each origin t: r_{t+1} + ... + r_{t+h} in the market's
    trading days. NaN (excluded from evaluation) when the position cannot be marked: no
    price at t or at t+h, or, for h = 1, a return that spans a missing price."""
    r = df["ret"]
    cum = r.fillna(0.0).cumsum()
    fwd = cum.shift(-h) - cum
    ok = r.notna() & r.shift(-h).notna()
    if h == 1:
        ok &= df["span"].shift(-1).eq(1)
    return fwd.where(ok)


def past_returns(df: pd.DataFrame, h: int) -> pd.Series:
    """h-day log returns ending at each date s (known at the close of s): the history
    historical simulation draws from. Overlapping for h > 1."""
    r = df["ret"]
    cum = r.fillna(0.0).cumsum()
    past = cum - cum.shift(h)
    return past.where(r.notna() & r.shift(h).notna())


@dataclass
class Forecast:
    """One model's forecasts for one market, side and horizon, indexed by origin."""
    frame: pd.DataFrame                                               # loss, var.., es.., u
    sampler: Callable[[np.random.Generator, int, np.ndarray], np.ndarray]  # (rng, n, row positions) -> (n, k) losses

    def between(self, start, end) -> "Forecast":
        idx = np.flatnonzero((self.frame.index >= pd.Timestamp(start)) & (self.frame.index <= pd.Timestamp(end)))
        return Forecast(self.frame.iloc[idx], lambda rng, n, rows=None, _i=idx: self.sampler(rng, n, _i if rows is None else _i[rows]))

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return self.sampler(rng, n, None)


# ---- Scale x shape ----

_S = np.polynomial.legendre.leggauss(200)
_NODES, _WEIGHTS = (_S[0] + 1) / 2, _S[1] / 2                          # Gauss-Legendre on (0, 1)


def tail_mean(g: Callable[[np.ndarray], np.ndarray], shape, alpha: float, upper: bool) -> np.ndarray:
    """(1/alpha) * integral over the alpha tail of g(Q(u)) du, by Gauss-Legendre on the
    substitution u = alpha s^2, which puts nodes near the extreme (u -> 0) where Q diverges.
    upper=True integrates the upper tail (quantiles from the inverse survival function, so
    tiny tail probabilities keep their precision). g maps quantiles (broadcast along the
    last axis) to losses."""
    u = alpha * _NODES ** 2
    q = shape.isf(u) if upper else shape.ppf(u)
    return np.sum(g(q) * 2 * _NODES * _WEIGHTS, axis=-1)


def scale_shape_forecast(df: pd.DataFrame, daily_var: pd.Series, side: str, h: int = 1,
                         shape=stats.norm(), alphas=ALPHAS) -> Forecast:
    """Forecast R over the next h days as sigma_t * Z, with sigma_t^2 = h x the daily
    variance forecast made at t (the square-root-of-time rule; Phase 4 replaces it) and Z
    drawn from `shape`, a distribution with unit variance (`ppf`, `cdf`, `rvs`). Zero mean:
    at these horizons the drift is noise next to the volatility, and assuming none is the
    convention. VaR, ES and the PIT come from the exact loss mapping (no log-return
    approximation)."""
    fwd = forward_returns(df, h)
    sigma = np.sqrt(h * daily_var)
    ok = fwd.notna() & sigma.notna() & (sigma > 0)
    R, s = fwd[ok].to_numpy(), sigma[ok].to_numpy()
    out = {"loss": to_loss(R, side)}
    upper = side == "short"
    for a in alphas:
        q = shape.isf(a) if upper else shape.ppf(a)
        out[col("var", a)] = to_loss(s * q, side)
        out[col("es", a)] = tail_mean(lambda z: to_loss(s[:, None] * z[None, :], side), shape, a, upper)
    out["u"] = shape.sf(R / s) if upper else shape.cdf(R / s)
    frame = pd.DataFrame(out, index=fwd.index[ok])

    def sampler(rng, n, rows):
        ss = s if rows is None else s[rows]
        return to_loss(ss[None, :] * shape.rvs(size=(n, len(ss)), random_state=rng), side)

    return Forecast(frame, sampler)


# ---- Historical simulation ----

def empirical_var_es(sorted_desc: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """VaR and ES from losses sorted largest first (rows = origins, columns = window).
    VaR is the empirical (1-alpha) quantile, the smallest loss x with F_n(x) >= 1 - alpha:
    with n = 500 and alpha = 1% that is the 6th largest loss, exceeded by exactly 5 of 500.
    ES is the Acerbi-Tasche estimator, (1/(n alpha)) [sum of the floor(n alpha) largest +
    (n alpha - floor(n alpha)) x the next], the average of the worst alpha share."""
    n = sorted_desc.shape[1]
    k = n * alpha
    kf = int(np.floor(k + 1e-9))
    var = sorted_desc[:, kf] if kf < n else sorted_desc[:, -1]
    es = (sorted_desc[:, :kf].sum(1) + (k - kf) * sorted_desc[:, min(kf, n - 1)]) / k
    return var, es


def historical_forecast(df: pd.DataFrame, side: str, h: int = 1, window: int = HS_WINDOW,
                        alphas=ALPHAS) -> Forecast:
    """Historical simulation: the next h-day loss is drawn from the losses of the last
    `window` h-day periods (overlapping for h > 1), equally weighted, no volatility
    scaling. The usual desk default and the Basel reference point."""
    fwd = forward_returns(df, h)
    past = to_loss(past_returns(df, h), side)
    vals = past.to_numpy()
    valid_pos = np.flatnonzero(~np.isnan(vals))
    # Origin t can use the `window` most recent valid past losses ending at or before t.
    n_valid = np.searchsorted(valid_pos, np.arange(len(df)), side="right")
    ok = (n_valid >= window) & fwd.notna().to_numpy()
    origins = np.flatnonzero(ok)
    ends = n_valid[origins]                                           # windows = valid_pos[end-window:end]
    win = vals[valid_pos[ends[:, None] - window + np.arange(window)[None, :]]]
    srt = -np.sort(-win, axis=1)
    loss = to_loss(fwd.to_numpy()[origins], side)
    out = {"loss": loss}
    for a in alphas:
        out[col("var", a)], out[col("es", a)] = empirical_var_es(srt, a)
    out["u"] = (win >= loss[:, None]).mean(1)
    frame = pd.DataFrame(out, index=df.index[origins])

    def sampler(rng, n, rows):
        w = win if rows is None else win[rows]
        pick = rng.integers(0, window, (n, len(w)))
        return w[np.arange(len(w))[None, :], pick]

    return Forecast(frame, sampler)
