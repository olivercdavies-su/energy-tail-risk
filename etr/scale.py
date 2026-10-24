"""Volatility (scale) forecasts at 1 and 10 days, walk-forward.

Timing: a forecast is indexed by its origin t (the close at which it is made) and uses
returns up to and including t. Horizon 1 forecasts the variance of the next trading day's
log return; horizon 10 the variance of the sum of the next 10 trading days' log returns
(what a 10-day P&L experiences, rolls inside the window included).

Models (PROTOCOL.md section 4)
  ewma     RiskMetrics EWMA, lambda 0.94. Its 10-day term structure is flat.
  ewma_wd  EWMA of weekday-deseasonalised returns, times the weekday factor of each target
           day. Session 1 found European-gas Mondays at 1.6x average variance and oil
           Wednesdays (EIA report) at 1.2-1.35x; at a daily horizon that is first-order.
  gjr      GJR-GARCH(1,1) with Student-t errors; the asymmetry may take either sign
           (European gas has inverse leverage). Analytic multi-step forecasts.
  har      log-linear HAR (Corsi 2009) on log mean variance over 1, 5, 22 and 66 days,
           fitted by minimising QLIKE: with daily squared returns as the target, log(r^2) is
           unusable (zero returns) and least squares is dominated by a few huge days. A
           separate direct fit on 10-day squared returns gives the 10-day forecast.
  iv       front-month implied vol (HH, Brent, WTI) after a plausibility filter, EWMA where
           it is missing or implausible and in the other markets.
  comb     equal-weight mean of the ewma_wd, gjr and har variances.
  comb_iv  equal-weight mean of ewma_wd, gjr, har and iv.

Each model gives three forecasts: `<m>_1` (1-day), `<m>_sqrt10` (10 x the 1-day forecast:
the square-root-of-time rule) and `<m>_10` (the model's own 10-day term structure).

Estimation: refits once a year on all data up to 31 December of the previous year
(expanding window from 2010); parameters stay fixed through the forecast year.
Calibration: each forecast (model x horizon x year) is multiplied by c = mean(realised /
forecast) over training origins whose target window ended before the forecast year, where
"forecast" is what the model with its *current* parameters says at those origins (the
previous project found that scaling by past out-of-sample forecasts fails for refitted
models, whose early refits were a different, worse model). With estimation noise
E[r^2 / f] > 1 even for an unbiased f, so uncalibrated forecasts understate risk.

Adapted from energy-vol-targeting: src/evt/models/simple.py (EWMA), garch.py (GJR fitting
and multi-step forecasts), har.py (HAR components and purging) and walkforward.py
(calibration from current parameters, restarting the GARCH recursion at the window start).
"""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model

FLOOR = 1e-10                 # floor on a daily variance before taking logs (a 0.001% move)
HAR_WINDOWS = (1, 5, 22, 66)
WEEKDAY_YEARS = 5             # weekday factors are estimated on the last 5 years (they drift)
IV_BAND = (1 / 3, 3.0)        # implied vol outside this multiple of EWMA vol is treated as a bad print
MODELS = ("ewma", "ewma_wd", "gjr", "har", "iv", "comb", "comb_iv")


# ---- Building blocks ----

def daily_variance(df: pd.DataFrame) -> pd.Series:
    """Squared log return per trading day covered; a multi-day return (after a missing
    price) is divided by its span. NaN where there is no return."""
    return df["ret"] ** 2 / df["span"].where(df["span"] > 0)


def ewma_variance(df: pd.DataFrame, lam: float = 0.94, min_periods: int = 40, v: pd.Series | None = None) -> pd.Series:
    """RiskMetrics exponentially weighted variance, s2_t = lam s2_{t-1} + (1 - lam) r_t^2:
    the forecast of next day's variance made at the close of t. lam = 0.94 is the
    RiskMetrics daily value (half-life about 11 days). NaN until `min_periods` returns.
    `v` replaces the daily variances (e.g. deseasonalised ones)."""
    v = daily_variance(df) if v is None else v
    s2 = v.ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean()
    return s2.where(v.notna().cumsum() >= min_periods)


def calibration_scale(df: pd.DataFrame, var_fc: pd.Series, min_obs: int = 120) -> pd.Series:
    """Point-in-time level calibration of a parameter-free 1-day forecast: for each origin,
    the mean of r_s^2 / (span_s x var_fc_{s-1}) over all days s before 1 January of the
    origin's year. Refreshed once a year."""
    ratio = (daily_variance(df) / var_fc.shift(1)).replace([np.inf, -np.inf], np.nan)
    years = df.index.year
    c = pd.Series(np.nan, index=df.index)
    for y in np.unique(years):
        past = ratio[years < y].dropna()
        if len(past) >= min_obs:
            c[years == y] = past.mean()
    return c


def calibrated_ewma_variance(df: pd.DataFrame, lam: float = 0.94) -> pd.Series:
    """EWMA variance times its point-in-time calibration scale (baseline B3)."""
    v = ewma_variance(df, lam)
    return v * calibration_scale(df, v)


def targets(df: pd.DataFrame) -> pd.DataFrame:
    """Realised quantities each forecast is calibrated and judged against, by origin t:
      y1     next day's squared return per day covered (the calibration_scale convention)
      y10    squared sum of the next 10 days' log returns
      end1, end10  the date each target is complete (for purging and calibration cut-offs)"""
    r = df["ret"]
    cum = r.fillna(0.0).cumsum()
    r10 = (cum.shift(-10) - cum).where(r.notna() & r.shift(-10).notna())
    dates = pd.Series(df.index, index=df.index)
    return pd.DataFrame({"y1": daily_variance(df).shift(-1), "y10": r10 ** 2,
                         "end1": dates.shift(-1), "end10": dates.shift(-10)})


def next_weekdays(index: pd.DatetimeIndex, h: int) -> np.ndarray:
    """Weekday (0 = Monday) of each of the next h trading days after each origin, from the
    market's own trading calendar (exchange holidays are published in advance). -1 where the
    calendar runs out."""
    wd = index.weekday.to_numpy()
    n = len(index)
    out = np.full((n, h), -1)
    for k in range(1, h + 1):
        out[:n - k, k - 1] = wd[k:]
    return out


# ---- Weekday factors ----

def weekday_stats(df: pd.DataFrame, cut: pd.Timestamp, years: int = WEEKDAY_YEARS) -> pd.DataFrame:
    """Raw weekday variance factors from the `years` years before `cut`: the mean of
    z^2 = r^2 / (span x EWMA_{t-1}) on each weekday over its overall mean, with standard
    errors. One EWMA pass on raw returns is enough: its level is the average over weekdays."""
    z2 = (daily_variance(df) / ewma_variance(df).shift(1)).loc[cut - pd.DateOffset(years=years):cut].dropna()
    z2 = z2[np.isfinite(z2)]
    g = z2.groupby(z2.index.weekday)
    m = z2.mean()
    return pd.DataFrame({"f_raw": g.mean() / m, "se": g.std() / np.sqrt(g.size()) / m, "share": g.size() / len(z2)}).reindex(range(5))


def shrink_weekday(stats_by_market: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
    """Empirical-Bayes (James-Stein) shrinkage of every market's weekday factors toward 1.
    The prior variance of true weekday effects, tau^2, is estimated across all markets and
    weekdays as the spread of the raw factors around 1 minus their average sampling
    variance; each factor keeps the share tau^2 / (tau^2 + se^2) of its deviation from 1.
    Noisy factors shrink most; nothing is tuned. Factors are renormalised to average 1 over
    the days they apply to."""
    allf = pd.concat(stats_by_market.values())
    tau2 = max(0.0, float(np.mean((allf["f_raw"] - 1) ** 2) - np.mean(allf["se"] ** 2)))
    out = {}
    for name, s in stats_by_market.items():
        w = tau2 / (tau2 + s["se"] ** 2) if tau2 > 0 else 0.0
        f = 1 + w * (s["f_raw"] - 1)
        out[name] = (f / np.sum(f * s["share"])).fillna(1.0)
    return out


# ---- GJR-GARCH ----

def daily_equivalent_returns(df: pd.DataFrame) -> pd.Series:
    """Returns in percent, a multi-day return scaled to one day's worth (r / sqrt(span)),
    missing returns set to 0 (a handful of days; GARCH needs a gap-free series)."""
    span = df["span"].where(df["span"] > 0, 1)
    return (100 * df["ret"] / np.sqrt(span)).fillna(0.0)


def _gjr(r: pd.Series):
    return arch_model(r, mean="Zero", vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)


def gjr_fit(r_train: pd.Series) -> pd.Series:
    """GJR-GARCH(1,1)-t by maximum likelihood. arch allows gamma < 0 (subject to
    alpha + gamma >= 0), which inverse leverage needs."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _gjr(r_train).fit(disp="off", options={"maxiter": 500}).params


def gjr_paths(r: pd.Series, params: pd.Series, start, horizon: int = 10) -> pd.DataFrame:
    """Daily-variance forecasts (decimal units) h = 1..horizon days ahead from every origin
    on or after `start`, parameters fixed. The recursion starts at r's first date."""
    res = _gjr(r).fix(params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = res.forecast(horizon=horizon, start=start, reindex=False, method="analytic")
    return f.variance / 1e4


def gjr_persistence(params: pd.Series) -> float:
    return params["alpha[1]"] + 0.5 * params["gamma[1]"] + params["beta[1]"]


# ---- HAR fitted by QLIKE ----

def har_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constant and log mean daily variance over the last 1, 5, 22 and 66 days (including
    t). Requires at least 80% of each window."""
    v = daily_variance(df).clip(lower=FLOOR)
    cols = {"const": 1.0}
    for w in HAR_WINDOWS:
        cols[f"lv{w}"] = np.log(v.rolling(w, min_periods=max(1, int(0.8 * w))).mean())
    return pd.DataFrame(cols, index=df.index)


def qlike_fit(X: np.ndarray, y: np.ndarray, tol: float = 1e-12, max_iter: int = 100) -> np.ndarray:
    """b minimising mean(y / exp(Xb) + Xb): QLIKE for a log-linear variance. The loss is
    convex in b, so Newton's method (with step halving) finds the unique minimum. At the
    optimum the constant's first-order condition gives mean(y / fitted) = 1: the fit is
    calibrated in-sample."""
    n = len(y)
    b = np.zeros(X.shape[1])
    b[0] = np.log(np.mean(y))
    loss = lambda b: np.mean(y * np.exp(-(X @ b)) + X @ b)  # noqa: E731
    cur = loss(b)
    for _ in range(max_iter):
        w = y * np.exp(-(X @ b))
        grad = X.T @ (1 - w) / n
        if np.max(np.abs(grad)) < tol:
            break
        step = np.linalg.solve((X * w[:, None]).T @ X / n, grad)
        t = 1.0
        while loss(b - t * step) > cur and t > 1e-8:
            t /= 2
        b = b - t * step
        cur = loss(b)
    return b


# ---- Implied vol ----

def filtered_iv_variance(iv: pd.Series, raw_ewma: pd.Series, max_fill: int = 5) -> pd.Series:
    """Daily variance from implied vol (annualised fraction), point-in-time: a print outside
    IV_BAND x the same day's EWMA vol is treated as bad and the last plausible value is
    carried for up to `max_fill` days; NaN beyond that (the caller falls back to EWMA). One bad
    HH print (1.2% on 2013-11-26, against 31% the day before) motivated the filter."""
    ratio = iv / np.sqrt(raw_ewma * 252)
    ok = (ratio > IV_BAND[0]) & (ratio < IV_BAND[1])
    return (iv.where(ok) ** 2 / 252).ffill(limit=max_fill)


# ---- Walk-forward ----

def _calibrate(ins: pd.Series, realised: pd.Series, end: pd.Series, cut: pd.Timestamp, min_obs: int = 120) -> float:
    ratio = (realised / ins).replace([np.inf, -np.inf], np.nan)
    ratio = ratio[(end <= cut) & ins.notna()].dropna()
    return float(ratio.mean()) if len(ratio) >= min_obs else np.nan


def _year_forecasts(df: pd.DataFrame, y: int, wd_factors: pd.Series, iv: pd.Series | None, info: list) -> dict:
    """Uncalibrated forecasts for one market and forecast year y: {(model, h): (oos, ins)},
    where oos covers origins in year y and ins the training origins (same parameters)."""
    cut = pd.Timestamp(f"{y - 1}-12-31")
    train = df.loc[:cut]
    in_y = (df.index.year == y)
    out = {}
    raw_ewma = ewma_variance(df)
    wd1, wd10 = next_weekdays(df.index, 1), next_weekdays(df.index, 10)

    # EWMA: flat term structure.
    out[("ewma", 1)] = (raw_ewma[in_y], raw_ewma.loc[:cut])
    out[("ewma", 10)] = (10 * raw_ewma[in_y], 10 * raw_ewma.loc[:cut])

    # Weekday EWMA: deseasonalise with this year's factors, re-apply for the target days.
    f_today = wd_factors.reindex(df.index.weekday).to_numpy()
    v_ds = daily_variance(df) / f_today
    ew_ds = ewma_variance(df, v=v_ds)
    fmap = np.r_[wd_factors.reindex(range(5)).to_numpy(), np.nan]        # index -1 -> NaN
    f1 = pd.Series(fmap[wd1[:, 0]], index=df.index)
    f10 = pd.Series(np.where((wd10 >= 0).all(1), fmap[wd10].sum(1), np.nan), index=df.index)
    out[("ewma_wd", 1)] = ((ew_ds * f1)[in_y], (ew_ds * f1).loc[:cut])
    out[("ewma_wd", 10)] = ((ew_ds * f10)[in_y], (ew_ds * f10).loc[:cut])

    # GJR: fit on the training window; the out-of-sample recursion starts at the window
    # start (2010) and runs through year y; the in-sample run uses the same parameters.
    r = daily_equivalent_returns(df)
    params = gjr_fit(r.loc[:cut])
    oos = gjr_paths(r.loc[:f"{y}-12-31"], params, start=f"{y}-01-01")
    ins = gjr_paths(r.loc[:cut], params, start=r.index[60])
    out[("gjr", 1)] = (oos["h.01"].reindex(df.index[in_y]), ins["h.01"])
    out[("gjr", 10)] = (oos.sum(1).reindex(df.index[in_y]), ins.sum(1))
    info.append({"year": y, "model": "gjr", **params.to_dict(), "persistence": gjr_persistence(params)})

    # HAR by QLIKE, separately for each horizon, on purged training origins.
    X = har_features(df)
    tg = targets(df)
    for h, ycol, ecol in ((1, "y1", "end1"), (10, "y10", "end10")):
        trn = pd.concat([X, tg[[ycol, ecol]]], axis=1).loc[:cut]
        trn = trn[(trn[ecol] <= cut)].replace([np.inf, -np.inf], np.nan).dropna()
        b = qlike_fit(trn[X.columns].to_numpy(), trn[ycol].to_numpy())
        pred = np.exp(X.to_numpy() @ b)
        pred = pd.Series(pred, index=df.index)
        out[("har", h)] = (pred[in_y], pred.loc[:cut])
        info.append({"year": y, "model": f"har_{h}", **dict(zip(X.columns, b)), "n_train": len(trn)})

    # Implied vol where available and plausible. Left NaN elsewhere: after calibration the
    # gaps (and markets without options) are filled with the calibrated EWMA forecast, so
    # implied vol's variance risk premium and EWMA's under-forecasting each get their own
    # calibration.
    if iv is not None:
        ivv = filtered_iv_variance(iv.reindex(df.index), raw_ewma)
        ratio = iv.reindex(df.index) / np.sqrt(raw_ewma * 252)
        info.append({"year": y, "model": "iv", "fallback_days": int(ivv[in_y].isna().sum()),
                     "filtered_days": int(((ratio <= IV_BAND[0]) | (ratio >= IV_BAND[1]))[in_y].sum())})
    else:
        ivv = pd.Series(np.nan, index=df.index)
    out[("iv", 1)] = (ivv[in_y], ivv.loc[:cut])
    out[("iv", 10)] = (10 * ivv[in_y], 10 * ivv.loc[:cut])
    return out


def walk_forward(rets: dict[str, pd.DataFrame], years: list[int], iv: dict[str, pd.Series] | None = None):
    """Calibrated scale forecasts for every market, model and horizon, at every origin in
    `years`. Returns ({market: frame of forecasts + targets}, diagnostics frame).
    Weekday factors are estimated per year from each market's last WEEKDAY_YEARS years and
    shrunk across markets (so all markets are processed together)."""
    iv = iv or {}
    frames = {name: [] for name in rets}
    diag = []
    tg = {name: targets(df) for name, df in rets.items()}
    for y in years:
        cut = pd.Timestamp(f"{y - 1}-12-31")
        wd = shrink_weekday({name: weekday_stats(df, cut) for name, df in rets.items()})
        for name, df in rets.items():
            info = []
            raw = _year_forecasts(df, y, wd[name], iv.get(name), info)
            t = tg[name]
            cal = {}
            for (m, h), (oos, ins) in raw.items():
                c = _calibrate(ins, t[f"y{h}"], t[f"end{h}"], cut)
                cal[(m, h)] = (oos * c, ins * c)
                diag.append({"market": name, "year": y, "model": m, "h": h, "scale_c": c})
            for h in (1, 10):                                              # implied-vol fallback
                cal[("iv", h)] = tuple(a.fillna(b) for a, b in zip(cal[("iv", h)], cal[("ewma", h)]))
            for h in (1, 10):
                for comb, parts in (("comb", ("ewma_wd", "gjr", "har")), ("comb_iv", ("ewma_wd", "gjr", "har", "iv"))):
                    oos = pd.concat([cal[(p, h)][0] for p in parts], axis=1).mean(axis=1, skipna=False)
                    ins = pd.concat([cal[(p, h)][1] for p in parts], axis=1).mean(axis=1, skipna=False)
                    c = _calibrate(ins, t[f"y{h}"], t[f"end{h}"], cut)
                    cal[(comb, h)] = (oos * c, ins * c)
                    diag.append({"market": name, "year": y, "model": comb, "h": h, "scale_c": c})
            idx = df.index[df.index.year == y]
            cols = {}
            for m in MODELS:
                cols[f"{m}_1"] = cal[(m, 1)][0].reindex(idx)
                cols[f"{m}_sqrt10"] = 10 * cols[f"{m}_1"]
                cols[f"{m}_10"] = cal[(m, 10)][0].reindex(idx)
            frames[name].append(pd.DataFrame(cols, index=idx))
            diag.extend({"market": name, **d} for d in info)
            diag.extend({"market": name, "year": y, "model": "weekday", "weekday": k, "factor": f}
                        for k, f in wd[name].items())
    out = {name: pd.concat(fr).join(tg[name][["y1", "y10"]]) for name, fr in frames.items()}
    return out, pd.DataFrame(diag)
