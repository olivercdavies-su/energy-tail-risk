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

Models here: the scale x shape family and historical simulation. A shape is the
distribution of standardised residuals z = r / sigma (unit variance by the scale's
calibration); the shapes are below, with their fitting rules.
"""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch.univariate import SkewStudent
from scipy import optimize, stats

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


def concat(parts: list[Forecast]) -> Forecast:
    """Join forecasts for consecutive periods (e.g. one per refit year) into one."""
    parts = [f for f in parts if len(f.frame)]
    frame = pd.concat([f.frame for f in parts])
    bounds = np.cumsum([0] + [len(f.frame) for f in parts])

    def sampler(rng, n, rows):
        rows = np.arange(len(frame)) if rows is None else np.asarray(rows)
        out = np.empty((n, len(rows)))
        for k, f in enumerate(parts):
            sel = (rows >= bounds[k]) & (rows < bounds[k + 1])
            if sel.any():
                out[:, sel] = f.sampler(rng, n, rows[sel] - bounds[k])
        return out

    return Forecast(frame, sampler)


# ---- Scale x shape ----

_S = np.polynomial.legendre.leggauss(200)
_NODES, _WEIGHTS = (_S[0] + 1) / 2, _S[1] / 2                          # Gauss-Legendre on (0, 1)


def tail_mean(g: Callable[[np.ndarray], np.ndarray], shape, alpha: float, upper: bool) -> np.ndarray:
    """(1/alpha) * integral over the alpha tail of g(Q(u)) du, by Gauss-Legendre on the
    substitution u = alpha s^2, which puts nodes near the extreme (u -> 0) where Q diverges.
    upper=True integrates the upper tail (quantiles from the inverse survival function, so
    tiny tail probabilities keep their precision). g maps quantiles (broadcast along the
    last axis) to losses."""
    if hasattr(shape, "tail_mean"):                                   # exact for empirical shapes
        return shape.tail_mean(g, alpha, upper)
    u = alpha * _NODES ** 2
    q = shape.isf(u) if upper else shape.ppf(u)
    return np.sum(g(q) * 2 * _NODES * _WEIGHTS, axis=-1)


def scale_shape_forecast(df: pd.DataFrame, var_h: pd.Series, side: str, h: int = 1,
                         shape=stats.norm(), alphas=ALPHAS) -> Forecast:
    """Forecast R over the next h days as sigma_t * Z, with sigma_t^2 = `var_h`, the
    forecast made at t of the variance of the h-day log return (h x a daily variance under
    the square-root-of-time rule; a term structure otherwise, see etr.scale), and Z drawn
    from `shape`, a distribution with unit variance (`ppf`, `cdf`, `rvs`). Zero mean: at
    these horizons the drift is noise next to the volatility, and assuming none is the
    convention. VaR, ES and the PIT come from the exact loss mapping (no log-return
    approximation)."""
    fwd = forward_returns(df, h)
    sigma = np.sqrt(var_h)
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


# ---- Shapes: distributions of standardised residuals z = r / sigma ----
#
# All have the interface of a scipy frozen distribution (ppf, isf, cdf, sf, rvs), so
# scale_shape_forecast works with any of them. Sign convention: z < 0 is a fall in price,
# so the left tail is the long position's loss tail and the right tail the short's.

class StudentT:
    """Student-t scaled to unit variance (nu > 2). Symmetric fat tails: the classic
    GARCH-t choice. One parameter, so it can be pinned down even per market."""

    def __init__(self, nu: float):
        self.nu = nu
        self._d = stats.t(nu, scale=np.sqrt((nu - 2) / nu))
        self.ppf, self.isf, self.cdf, self.sf, self.rvs = self._d.ppf, self._d.isf, self._d.cdf, self._d.sf, self._d.rvs

    def logpdf(self, z):
        return self._d.logpdf(z)


class SkewT:
    """Hansen (1994) skewed Student-t, zero mean and unit variance by construction
    (arch's SkewStudent). eta > 2 sets tail thickness, lambda in (-1, 1) the asymmetry:
    lambda < 0 puts more weight in the left tail (falls: long positions), lambda > 0 in the
    right (spikes: short positions). Session 1 found oil left-heavy and NBP right-heavy."""

    _D = SkewStudent()

    def __init__(self, eta: float, lam: float):
        self.eta, self.lam = eta, lam
        self._p = np.array([eta, lam])

    def ppf(self, u):
        u = np.asarray(u, float)                      # arch wants a 1-d array
        return self._D.ppf(np.atleast_1d(u).ravel(), self._p).reshape(u.shape)

    def isf(self, u):
        return self.ppf(1 - np.asarray(u, float))

    def cdf(self, x):
        x = np.asarray(x, float)
        return self._D.cdf(np.atleast_1d(x).ravel(), self._p).reshape(x.shape)

    def sf(self, x):
        return 1 - self.cdf(x)

    def rvs(self, size, random_state):
        return self.ppf(random_state.random(size))

    @classmethod
    def loglik(cls, params, z):
        return cls._D.loglikelihood(np.asarray(params, float), z, np.ones_like(z), individual=True)


class Empirical:
    """Filtered historical simulation: the training residuals themselves. Quantiles follow
    the same convention as historical simulation (the floor(n alpha)+1-th most extreme
    value, so it is breached by at most alpha), and ES is the exact Acerbi-Tasche average
    of the worst alpha share."""

    def __init__(self, z):
        self.z = np.sort(np.asarray(z, float))
        self.n = len(self.z)

    def _pick(self, u, from_top: bool):
        k = np.clip(np.floor(self.n * np.asarray(u, float) + 1e-9).astype(int), 0, self.n - 1)
        return self.z[::-1][k] if from_top else self.z[k]

    def ppf(self, u):
        return self._pick(u, False)

    def isf(self, u):
        return self._pick(u, True)

    def cdf(self, x):
        return np.searchsorted(self.z, np.asarray(x, float), side="right") / self.n

    def sf(self, x):
        return 1 - np.searchsorted(self.z, np.asarray(x, float), side="left") / self.n

    def rvs(self, size, random_state):
        return self.z[random_state.integers(0, self.n, size)]

    def tail_mean(self, g, alpha: float, upper: bool):
        k = self.n * alpha
        kf = int(np.floor(k + 1e-9))
        tail = (self.z[::-1] if upper else self.z)[:kf + 1]
        w = np.r_[np.ones(kf), k - kf] / k
        return np.sum(g(tail) * w, axis=-1)


GPD_TAIL = 0.10        # share of each side modelled by the generalised Pareto (McNeil & Frey 2000)


def gpd_nll(xi: float, beta: float, e: np.ndarray) -> float:
    """Negative log-likelihood of exceedances e under a generalised Pareto (xi, beta)."""
    if beta <= 0:
        return np.inf
    if abs(xi) < 1e-8:
        return len(e) * np.log(beta) + np.sum(e) / beta
    t = 1 + xi * e / beta
    if np.any(t <= 0):
        return np.inf
    return len(e) * np.log(beta) + (1 + 1 / xi) * np.sum(np.log(t))


def _gpd_beta(xi: float, e: np.ndarray) -> tuple[float, float]:
    """Profile out beta for a given xi: (best beta, its negative log-likelihood)."""
    lo = max(1e-6, -xi * e.max() * (1 + 1e-9)) if xi < 0 else 1e-6
    res = optimize.minimize_scalar(lambda lb: gpd_nll(xi, np.exp(lb), e),
                                   bounds=(np.log(lo) + 1e-9, np.log(100 * e.mean() + 1)), method="bounded")
    return float(np.exp(res.x)), float(res.fun)


def fit_gpd(exceedances: list[np.ndarray]) -> tuple[float, list[float]]:
    """Maximum likelihood with one tail index xi shared by all the exceedance sets and a
    scale beta for each (one set alone gives the usual per-tail fit). xi is restricted to
    (-0.5, 1): beyond those values the estimator's asymptotics fail, and session 1's
    estimates were -0.12 to 0.24."""
    def profile(xi):
        return sum(_gpd_beta(xi, e)[1] for e in exceedances)
    xi = float(optimize.minimize_scalar(profile, bounds=(-0.49, 0.99), method="bounded").x)
    return xi, [_gpd_beta(xi, e)[0] for e in exceedances]


class GPDTails:
    """Peaks over threshold (McNeil & Frey 2000): the empirical residual distribution in the
    middle, a generalised Pareto beyond each side's GPD_TAIL quantile. Extreme-value theory
    says exceedances over a high threshold are approximately GPD whatever the parent
    distribution, so the GPD extrapolates beyond the few observed extremes, where filtered
    historical simulation can only repeat them."""

    def __init__(self, z, left: tuple[float, float], right: tuple[float, float]):
        self.body = Empirical(z)
        self.uL, self.uR = threshold_pair(self.body.z)
        self.pL, self.pR = np.mean(self.body.z < self.uL), np.mean(self.body.z > self.uR)
        (self.xiL, self.bL), (self.xiR, self.bR) = left, right

    @staticmethod
    def _x(u_rel, xi, b):
        """Distance beyond the threshold with tail probability u_rel (share of the tail)."""
        return b * np.log(1 / u_rel) if abs(xi) < 1e-8 else b / xi * (u_rel ** -xi - 1)

    @staticmethod
    def _tail_sf(x, xi, b):
        return np.exp(-x / b) if abs(xi) < 1e-8 else np.maximum(1 + xi * x / b, 0) ** (-1 / xi)

    def ppf(self, u):
        u = np.asarray(u, float)
        body = self.body.ppf(u)
        left = self.uL - self._x(np.clip(u / self.pL, 1e-300, 1), self.xiL, self.bL)
        right = self.uR + self._x(np.clip((1 - u) / self.pR, 1e-300, 1), self.xiR, self.bR)
        return np.where(u < self.pL, left, np.where(u > 1 - self.pR, right, body))

    def isf(self, u):
        u = np.asarray(u, float)
        right = self.uR + self._x(np.clip(u / self.pR, 1e-300, 1), self.xiR, self.bR)
        return np.where(u < self.pR, right, self.ppf(1 - u))

    def cdf(self, x):
        x = np.asarray(x, float)
        body = self.body.cdf(x)
        left = self.pL * self._tail_sf(np.maximum(self.uL - x, 0), self.xiL, self.bL)
        right = 1 - self.pR * self._tail_sf(np.maximum(x - self.uR, 0), self.xiR, self.bR)
        return np.where(x < self.uL, left, np.where(x > self.uR, right, body))

    def sf(self, x):
        x = np.asarray(x, float)
        right = self.pR * self._tail_sf(np.maximum(x - self.uR, 0), self.xiR, self.bR)
        return np.where(x > self.uR, right, 1 - self.cdf(x))

    def rvs(self, size, random_state):
        return self.ppf(random_state.random(size))


def threshold_pair(z_sorted: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(z_sorted, GPD_TAIL)), float(np.quantile(z_sorted, 1 - GPD_TAIL))


def exceedances(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Left and right exceedances beyond the GPD_TAIL thresholds, as positive distances."""
    z = np.sort(np.asarray(z, float))
    uL, uR = threshold_pair(z)
    return uL - z[z < uL], z[z > uR] - uR


def fit_t(z: np.ndarray) -> StudentT:
    """Maximum-likelihood degrees of freedom of a unit-variance t (nu in (2.05, 200))."""
    res = optimize.minimize_scalar(lambda lnu: -StudentT(2 + np.exp(lnu)).logpdf(z).sum(),
                                   bounds=(np.log(0.05), np.log(198)), method="bounded")
    return StudentT(2 + float(np.exp(res.x)))


def fit_skewt(zs: list[np.ndarray], pooled: bool) -> list[SkewT]:
    """Hansen skewed-t by maximum likelihood, one per residual set; pooled: one eta shared by
    all sets, a lambda for each (tail thickness pooled, asymmetry market-specific)."""
    def nll_one(p, z):
        return -SkewT.loglik(p, z).sum()
    bounds_eta, bounds_lam = (2.05, 200.0), (-0.95, 0.95)
    if not pooled:
        out = []
        for z in zs:
            r = optimize.minimize(nll_one, x0=[8.0, 0.0], args=(z,), method="L-BFGS-B", bounds=[bounds_eta, bounds_lam])
            out.append(SkewT(*r.x))
        return out
    k = len(zs)
    r = optimize.minimize(lambda p: sum(nll_one([p[0], p[1 + i]], z) for i, z in enumerate(zs)),
                          x0=[8.0] + [0.0] * k, method="L-BFGS-B", bounds=[bounds_eta] + [bounds_lam] * k)
    return [SkewT(r.x[0], lam) for lam in r.x[1:]]


SHAPE_KINDS = ("normal", "t", "skewt", "fhs", "gpd")


def fit_shapes(zs: dict[str, np.ndarray], kind: str, pooled: bool) -> dict:
    """One shape per market from its training residuals. pooled=False: each market alone.
    pooled=True: tail thickness shared across markets (t: one nu; skewed-t: one eta with
    market-specific lambda; GPD: one xi for all twelve market-sides, with market-side
    thresholds and scales; filtered HS: all markets' residuals in one sample)."""
    names = list(zs)
    if kind == "normal":
        return {n: stats.norm() for n in names}
    if kind == "t":
        if pooled:
            s = fit_t(np.concatenate([zs[n] for n in names]))
            return {n: s for n in names}
        return {n: fit_t(zs[n]) for n in names}
    if kind == "skewt":
        return dict(zip(names, fit_skewt([zs[n] for n in names], pooled)))
    if kind == "fhs":
        if pooled:
            s = Empirical(np.concatenate([zs[n] for n in names]))
            return {n: s for n in names}
        return {n: Empirical(zs[n]) for n in names}
    if kind == "gpd":
        ex = {n: exceedances(zs[n]) for n in names}
        if pooled:
            xi, betas = fit_gpd([e for n in names for e in ex[n]])
            it = iter(betas)
            return {n: GPDTails(zs[n], (xi, next(it)), (xi, next(it))) for n in names}
        out = {}
        for n in names:
            xl, (bl,) = fit_gpd([ex[n][0]])
            xr, (br,) = fit_gpd([ex[n][1]])
            out[n] = GPDTails(zs[n], (xl, bl), (xr, br))
        return out
    raise ValueError(kind)
