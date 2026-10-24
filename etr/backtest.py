"""Backtests and scoring functions for Value-at-Risk and Expected Shortfall forecasts.

Conventions
- Everything is in **losses**: L = -(P&L as a fraction of notional), so positive = money
  lost. VaR and ES are positive loss numbers.
- `alpha` is the tail probability: VaR at alpha = 0.01 is the 99% VaR, i.e. the loss
  exceeded with probability 1%. ES at alpha is the mean loss beyond that VaR.
- A hit (breach, exception) is L_t > VaR_t.
- `u` is the forecast's probability integral transform on the loss tail:
  u_t = P(L >= realised L_t) under the forecast distribution. Small u = a bad day.
  If the forecast distribution is right, u is uniform.

Two kinds of test
1. Calibration tests ask "is this one model right?":
   - coverage: Kupiec (1995) unconditional coverage; a pooled, overlap-robust version;
   - independence: Christoffersen (1998), and Engle & Manganelli's (2004) dynamic
     quantile test, which can include regressors such as weekday dummies (a breach rate
     that is right on average but high on Mondays fails it, while breach counts pass);
   - ES: Acerbi & Szekely (2014) Z2, Du & Escanciano (2017) cumulative-violation test,
     McNeil & Frey (2000) exceedance residuals;
   - Basel traffic light.
   Small samples matter here: at 99% a year has only about 2.5 expected breaches, and
   the chi-squared approximations of the likelihood-ratio tests are poor. So Kupiec's
   p-value is exact (binomial) and the Christoffersen and DQ p-values are Monte Carlo
   (Dufour 2006): the statistic is simulated under the null of iid Bernoulli(alpha) hits.
2. Comparison tests ask "which model is better?": a consistent scoring function for VaR
   and ES jointly, the FZ0 loss of Patton, Ziegel & Chen (2019), compared with
   Diebold-Mariano tests and the model confidence set. FZ0 differences are unchanged when
   losses and forecasts are rescaled, so they can be averaged across markets and regimes
   (the role QLIKE played for variance forecasts in the previous project).

Pooling. Markets breach together (TTF-NBP and Brent-WTI correlate at about 0.9), so
stacking their hits as if independent overstates the evidence. The pooled tests average
across markets *per date* first, then use HAC standard errors on that daily series. The
same HAC treatment handles overlapping 10-day windows.

Adapted pieces: Kupiec, Christoffersen, Diebold-Mariano, the model confidence set and Holm
follow energy-vol-targeting (src/evt/evaluation/metrics.py and forecast.py), with exact or
Monte Carlo p-values added.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import MCS
from scipy import stats
from scipy.special import xlogy


# ---- Hits and coverage ----

def hits(loss, var) -> np.ndarray:
    """Breach indicator 1{L_t > VaR_t}."""
    return (np.asarray(loss, float) > np.asarray(var, float)).astype(int)


def _lr_uc(x: np.ndarray | int, n: int, alpha: float) -> np.ndarray:
    x = np.asarray(x, float)
    pi = x / n
    ll_alt = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    ll_null = (n - x) * np.log(1 - alpha) + x * np.log(alpha)
    return -2 * (ll_null - ll_alt)


def kupiec(hit, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures test. The p-value is exact: the probability,
    under Binomial(n, alpha), of a likelihood-ratio statistic at least as large as the one
    observed (the chi-squared(1) approximation is reported too)."""
    h = np.asarray(hit, int)
    n, x = len(h), int(h.sum())
    lr = float(_lr_uc(x, n, alpha))
    k = np.arange(n + 1)
    p = stats.binom.pmf(k, n, alpha)[_lr_uc(k, n, alpha) >= lr - 1e-9].sum()
    return {"n": n, "hits": x, "rate": x / n, "lr_uc": lr, "p": float(min(p, 1.0)),
            "p_asym": float(stats.chi2.sf(lr, 1))}


def _transition_counts(b: np.ndarray):
    """b: (..., n) 0/1 array. Counts of 0->0, 0->1, 1->0, 1->1 transitions."""
    prev, cur = b[..., :-1], b[..., 1:]
    n01 = ((1 - prev) * cur).sum(-1)
    n11 = (prev * cur).sum(-1)
    n00 = ((1 - prev) * (1 - cur)).sum(-1)
    n10 = (prev * (1 - cur)).sum(-1)
    return n00, n01, n10, n11


def _lr_ind(b: np.ndarray) -> np.ndarray:
    """Christoffersen independence LR for each row of b (0 when there is nothing to test)."""
    n00, n01, n10, n11 = (c.astype(float) for c in _transition_counts(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        pi0 = np.where(n00 + n01 > 0, n01 / (n00 + n01), 0.0)
        pi1 = np.where(n10 + n11 > 0, n11 / (n10 + n11), 0.0)
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll_alt = xlogy(n00, 1 - pi0) + xlogy(n01, pi0) + xlogy(n10, 1 - pi1) + xlogy(n11, pi1)
    return np.maximum(-2 * (ll_null - ll_alt), 0.0)


def christoffersen(hit, alpha: float, n_sims: int = 4999, seed: int = 0) -> dict:
    """Christoffersen (1998): independence (is a breach more likely after a breach?) and
    conditional coverage (right rate *and* independent). Monte Carlo p-values under iid
    Bernoulli(alpha) hits; chi-squared p-values reported as `_asym`."""
    h = np.asarray(hit, int)
    n = len(h)
    ind = float(_lr_ind(h))
    cc = float(_lr_uc(h.sum(), n, alpha)) + ind
    sim = (np.random.default_rng(seed).random((n_sims, n)) < alpha).astype(np.int8)
    ind_s = _lr_ind(sim)
    cc_s = _lr_uc(sim.sum(1), n, alpha) + ind_s
    mc = lambda obs, s: (1 + np.sum(s >= obs - 1e-9)) / (1 + n_sims)  # noqa: E731
    return {"lr_ind": ind, "p_ind": mc(ind, ind_s), "p_ind_asym": float(stats.chi2.sf(ind, 1)),
            "lr_cc": cc, "p_cc": mc(cc, cc_s), "p_cc_asym": float(stats.chi2.sf(cc, 2))}


def _dq_stat(h: np.ndarray, alpha: float, exog: np.ndarray | None, lags: int) -> float:
    y = h[lags:] - alpha
    cols = [np.ones(len(y))] + [h[lags - j:len(h) - j] for j in range(1, lags + 1)]
    if exog is not None:
        cols += [exog[lags:, k] for k in range(exog.shape[1])]
    X = np.column_stack(cols)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    fitted = X @ beta
    return float(y @ fitted / (alpha * (1 - alpha)))


def dq_test(hit, alpha: float, exog=None, lags: int = 4, n_sims: int = 1999, seed: int = 0) -> dict:
    """Engle & Manganelli (2004) dynamic quantile test. Regress hit_t - alpha on a constant,
    `lags` lagged hits and any `exog` columns known at the forecast origin (the VaR itself,
    weekday dummies, a vol-state indicator...). Under a correct model nothing predicts a
    breach, so DQ = fitted sum of squares / (alpha (1 - alpha)) is about chi-squared with as
    many degrees of freedom as regressors. Monte Carlo p-value: hits are simulated iid,
    exogenous columns held fixed, lagged-hit columns rebuilt from the simulated hits."""
    h = np.asarray(hit, float)
    ex = None if exog is None else np.asarray(exog, float).reshape(len(h), -1)
    stat = _dq_stat(h, alpha, ex, lags)
    df = 1 + lags + (0 if ex is None else ex.shape[1])
    rng = np.random.default_rng(seed)
    sims = np.array([_dq_stat((rng.random(len(h)) < alpha).astype(float), alpha, ex, lags)
                     for _ in range(n_sims)])
    return {"dq": stat, "df": df, "p": (1 + np.sum(sims >= stat - 1e-9)) / (1 + n_sims),
            "p_asym": float(stats.chi2.sf(stat, df))}


def traffic_light_bounds(n: int = 250, alpha: float = 0.01) -> tuple[int, int]:
    """Largest exception counts in the green and yellow zones. Basel: green while the
    cumulative binomial probability of the count is below 95%, red from 99.99% on
    (n = 250, alpha = 1%: green 0-4, yellow 5-9, red 10+)."""
    cdf = stats.binom.cdf(np.arange(n + 1), n, alpha)
    return int(np.sum(cdf < 0.95) - 1), int(np.sum(cdf < 0.9999) - 1)


def traffic_light(hit, alpha: float = 0.01, window: int = 250) -> dict:
    """Share of rolling `window`-day periods in each Basel zone, and the worst count."""
    h = np.asarray(hit, float)
    if len(h) < window:
        return {"windows": 0, "green": np.nan, "yellow": np.nan, "red": np.nan, "max_exceptions": np.nan}
    counts = np.convolve(h, np.ones(window), "valid")
    g, y = traffic_light_bounds(window, alpha)
    return {"windows": len(counts), "green": float(np.mean(counts <= g)),
            "yellow": float(np.mean((counts > g) & (counts <= y))), "red": float(np.mean(counts > y)),
            "max_exceptions": int(counts.max())}


# ---- Expected Shortfall tests ----

def acerbi_szekely_z2(loss, var, es, alpha: float, sim_losses: np.ndarray | None = None) -> dict:
    """Acerbi & Szekely (2014) Z2 = sum_t L_t 1{L_t > VaR_t} / (T alpha ES_t) - 1.
    If VaR and ES are right, E[L 1{L > VaR}] = alpha ES, so Z2 has mean zero; Z2 > 0 means
    ES (or VaR) is too small. The p-value (one-sided) is simulated: `sim_losses` holds
    draws of each day's loss from that day's forecast distribution (n_sims x T), and Z2 is
    recomputed on them with the forecasts held fixed."""
    L, V, E = (np.asarray(a, float) for a in (loss, var, es))
    T = len(L)
    z2 = float(np.sum(L * (L > V) / (alpha * E)) / T - 1)
    out = {"z2": z2, "p": np.nan}
    if sim_losses is not None:
        S = np.asarray(sim_losses, float)
        z2s = np.sum(S * (S > V) / (alpha * E), axis=1) / T - 1
        out["p"] = (1 + np.sum(z2s >= z2 - 1e-12)) / (1 + len(z2s))
    return out


def du_escanciano(u, alpha: float, lags: int = 5) -> dict:
    """Du & Escanciano (2017) tests from the PIT u_t (small = bad). The cumulative
    violation H_t = (alpha - u_t) / alpha * 1{u_t <= alpha} measures how deep into the tail
    each breach went. Under a correct forecast H is iid with mean alpha/2 and variance
    alpha (1/3 - alpha/4).
    - Unconditional: U = sqrt(n) (mean H - alpha/2) / sd -> N(0,1); one-sided p for U > 0
      (tail losses deeper than forecast, i.e. ES too small).
    - Conditional: C = n sum_{j<=lags} rho_j^2 -> chi-squared(lags), rho_j the
      autocorrelations of H (do deep breaches cluster?)."""
    u = np.asarray(u, float)
    n = len(u)
    H = (alpha - u) / alpha * (u <= alpha)
    sd = np.sqrt(alpha * (1 / 3 - alpha / 4))
    U = np.sqrt(n) * (H.mean() - alpha / 2) / sd
    d = H - alpha / 2
    g0 = alpha * (1 / 3 - alpha / 4)
    rho = np.array([np.sum(d[j:] * d[:-j]) / (n - j) / g0 for j in range(1, lags + 1)])
    C = n * np.sum(rho ** 2)
    return {"u_es": float(U), "p_u": float(stats.norm.sf(U)), "c_es": float(C),
            "p_c": float(stats.chi2.sf(C, lags))}


def mcneil_frey(loss, var, es, scale=None, n_boot: int = 4999, seed: int = 0) -> dict:
    """McNeil & Frey (2000): on breach days, the exceedance residual (L_t - ES_t) / s_t
    has mean zero if ES is right (s_t is any positive scale known at the origin; default
    ES_t itself). One-sided bootstrap test of a positive mean (ES too small), resampling
    the centred residuals (Efron & Tibshirani 1993)."""
    L, V, E = (np.asarray(a, float) for a in (loss, var, es))
    s = E if scale is None else np.asarray(scale, float)
    br = L > V
    e = (L[br] - E[br]) / s[br]
    k = len(e)
    if k < 3:
        return {"n_hits": k, "mean": float(e.mean()) if k else np.nan, "t": np.nan, "p": np.nan}
    t_obs = e.mean() / (e.std(ddof=1) / np.sqrt(k))
    rng = np.random.default_rng(seed)
    b = (e - e.mean())[rng.integers(0, k, (n_boot, k))]
    t_b = b.mean(1) / (b.std(1, ddof=1) / np.sqrt(k))
    return {"n_hits": k, "mean": float(e.mean()), "t": float(t_obs),
            "p": (1 + np.sum(t_b >= t_obs)) / (1 + n_boot)}


# ---- Scoring functions ----

def fz0(loss, var, es, alpha: float) -> np.ndarray:
    """FZ0 loss of Patton, Ziegel & Chen (2019), written for positive losses:
        FZ0 = 1{L > V} (L - V) / (alpha E) + V / E + log(E) - 1.
    Strictly consistent for (VaR, ES) jointly: its expectation is minimised by the true
    pair, so it can rank forecasts. It is the member of the Fissler-Ziegel family whose
    differences are unchanged by rescaling L, V and E together (degree-0 homogeneity), the
    property that makes averaging across markets and regimes meaningful. Requires E > 0."""
    L, V, E = (np.asarray(a, float) for a in (loss, var, es))
    if np.any(E <= 0):
        raise ValueError("FZ0 needs a positive ES")
    return (L > V) * (L - V) / (alpha * E) + V / E + np.log(E) - 1


def tick_loss(loss, var, alpha: float) -> np.ndarray:
    """Quantile (tick) loss for VaR alone, in loss units: consistent for the (1-alpha)
    quantile of L."""
    L, V = np.asarray(loss, float), np.asarray(var, float)
    return ((L > V) - alpha) * (L - V)


# ---- Comparing and pooling ----

def _hac_mean(x: np.ndarray, lags: int, kernel: str = "bartlett") -> tuple[float, float]:
    res = sm.OLS(x, np.ones_like(x)).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "kernel": kernel})
    return float(res.params[0]), float(res.bse[0])


def diebold_mariano(d, lags: int = 5) -> dict:
    """Is the mean loss differential d = loss_a - loss_b zero? HAC (Newey-West) t-test.
    p_less is one-sided for model a being better (mean d < 0). Simulated size on pooled FZ0
    differentials (results/p1_test_size.csv): about 4-5% at 1 day with 5 lags, and 5-6%
    for overlapping 10-day windows with 2h = 20 lags."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    m, se = _hac_mean(d, lags)
    t = m / se
    return {"n": len(d), "mean_diff": m, "se": se, "t": t, "p": float(2 * stats.norm.sf(abs(t))),
            "p_less": float(stats.norm.cdf(t))}


def date_mean(panel: pd.DataFrame) -> pd.Series:
    """Average across columns (markets, positions) on each date, skipping missing values.
    Pooled tests run on this daily series, so cross-market correlation is respected."""
    return panel.mean(axis=1, skipna=True).dropna()


def _pit_bootstrap(u_panel: pd.DataFrame, stat, n_boot: int, seed: int, block: int = 1) -> tuple[float, np.ndarray]:
    """Null distribution of a pooled statistic of the PITs. Each column's PITs are replaced
    by their exact uniform ranks, which imposes a correct forecast (uniform PIT) while
    keeping the observed dependence across columns on the same date; dates (or blocks of
    `block` dates) are then resampled. Returns the observed statistic and the draws.
    `stat` maps an (dates x columns) array with NaNs to a number."""
    U = u_panel.to_numpy(float)
    obs = stat(U)
    R = u_panel.rank(axis=0).sub(0.5).div(u_panel.notna().sum(axis=0), axis=1).to_numpy(float)
    T = len(R)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        draws[b] = stat(R[idx])
    return obs, draws


def _pooled_rate(U: np.ndarray, alpha: float) -> float:
    hit = np.where(np.isnan(U), np.nan, (U < alpha).astype(float))
    return float(np.nanmean(np.nanmean(hit, axis=1)))


def coverage_pooled(u_panel: pd.DataFrame, alpha: float, n_boot: int = 1999, seed: int = 0) -> dict:
    """Pooled unconditional coverage across markets: is the average breach rate (per-date
    mean of 1{u < alpha} over the columns, then averaged over dates) equal to alpha?
    Two-sided p-value from the PIT bootstrap above, so markets that breach together count
    as the partly redundant evidence they are, and the skew of a mean of rare events is
    respected. Dates must be independent under the null: 1-day forecasts, or non-overlapping
    multi-day windows (see `nonoverlapping`). Simulated size about 4-5%
    (results/p1_test_size.csv), against about 7% for a HAC t-test at the 99% level."""
    obs, draws = _pit_bootstrap(u_panel, lambda U: _pooled_rate(U, alpha), n_boot, seed)
    centre = draws.mean()
    p = (1 + np.sum(np.abs(draws - centre) >= abs(obs - alpha) - 1e-12)) / (1 + n_boot)
    return {"dates": len(u_panel), "rate": obs, "p": float(p)}


def es_pooled(u_panel: pd.DataFrame, alpha: float, n_boot: int = 1999, seed: int = 0) -> dict:
    """Pooled Du-Escanciano ES test: the per-date mean across columns of the cumulative
    violation H = (alpha - u) / alpha * 1{u <= alpha}, averaged over dates, against its null
    distribution from the PIT bootstrap. One-sided: a larger mean H means tail losses run
    deeper than forecast (ES too small). Reported as the ratio to its null mean alpha/2."""
    def stat(U):
        H = np.where(np.isnan(U), np.nan, (alpha - U) / alpha * (U <= alpha))
        return float(np.nanmean(np.nanmean(H, axis=1)))
    obs, draws = _pit_bootstrap(u_panel, stat, n_boot, seed)
    return {"dates": len(u_panel), "h_ratio": obs / (alpha / 2),
            "p": float((1 + np.sum(draws >= obs - 1e-12)) / (1 + n_boot))}


def coverage_hac(hit_panel: pd.DataFrame, alpha: float, lags: int = 5, kernel: str = "bartlett") -> dict:
    """HAC t-test that the per-date mean hit equals alpha. Kept for descriptive use with
    overlapping windows (kernel "uniform", lags h-1: the Hansen-Hodrick choice). Its size is
    poor for rare events (about 7% pooled and 13% for one market at 99%, 10 days,
    results/p1_test_size.csv), so registered tests use `coverage_pooled` and `kupiec` on
    non-overlapping windows instead."""
    x = date_mean(hit_panel.astype(float))
    m, se = _hac_mean(x.to_numpy() - alpha, lags, kernel)
    t = m / se if se > 0 else np.nan
    return {"days": len(x), "rate": m + alpha, "t": t, "p": float(2 * stats.norm.sf(abs(t)))}


def es_ratio_hac(loss_panel: pd.DataFrame, var_panel: pd.DataFrame, es_panel: pd.DataFrame,
                 alpha: float, lags: int = 5) -> dict:
    """Descriptive pooled ES ratio: the per-date mean of L 1{L > V} / (alpha E) - 1 (the
    Acerbi-Szekely Z2 summand) has mean zero under a correct forecast. HAC t-test, one-sided
    p for ES too small. Valid in large samples; for registered pooled tests use `es_pooled`."""
    z = loss_panel * (loss_panel > var_panel) / (alpha * es_panel) - 1
    x = date_mean(z.where(loss_panel.notna() & var_panel.notna() & es_panel.notna()))
    m, se = _hac_mean(x.to_numpy(), lags)
    t = m / se
    return {"days": len(x), "z2": m, "t": t, "p": float(stats.norm.sf(t))}


def nonoverlapping(frame: pd.DataFrame, h: int, anchor: str = "2010-01-01") -> pd.DataFrame:
    """Rows of an h-day forecast frame whose windows do not overlap, on a calendar shared by
    all markets: grid dates every h weekdays from `anchor`; for each grid date, the market's
    last origin on or before it (within 5 calendar days, so a long data gap never makes one
    origin serve two grid dates). The result is indexed by grid date, so
    markets line up for pooling even though their holidays differ. Holidays can make two
    consecutive windows share a day; that small overlap is ignored."""
    if frame.empty:
        return frame
    grid = pd.bdate_range(anchor, frame.index[-1])[::h]
    pos = frame.index.searchsorted(grid, side="right") - 1
    ok = pos >= 0
    grid, pos = grid[ok], pos[ok]
    picked = frame.index[pos]
    close = (grid - picked).days <= 5
    out = frame.iloc[pos[close]].copy()
    out.index = grid[close]
    return out[~out.index.duplicated()]


def model_confidence_set(losses: pd.DataFrame, size: float = 0.10, block: int = 10,
                         reps: int = 2000, seed: int = 0) -> pd.Series:
    """Hansen, Lunde & Nason (2011) model confidence set on per-date losses (one column per
    model), block bootstrap. Returns each model's MCS p-value; models with p >= size are in
    the set of models that cannot be told apart from the best."""
    m = MCS(losses.dropna(), size=size, reps=reps, block_size=block, method="R", seed=seed)
    m.compute()
    return m.pvalues.iloc[:, 0].reindex(losses.columns)


def holm(pvalues) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values: controls the family-wise error rate."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj
