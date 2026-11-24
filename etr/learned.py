"""Learned VaR/ES model and stress early warning (PROTOCOL.md 4, Q5; design fixed in
PROJECT_LOG.md session 7 before fitting).

Inputs: scale-free quantities known at the forecast origin (`features`).

Learned linear model. Start from a configuration (a scale model and its shape) and multiply
its volatility by m = exp(x'beta): VaR and ES scale together and the model remains a full
forecast distribution (PIT, draws). beta is fitted per side, pooled over markets, by
minimising the mean FZ0 loss (alpha = 2.5%) over training origins whose target ended by the
cut-off, refitted every year on an expanding window. Inputs are standardised with the
training mean and standard deviation (and, as a numerical safeguard, limited to +/-5 of
them). Fitting FZ0 directly is the "dynamic semiparametric" idea of Patton, Ziegel & Chen
(2019): the loss that ranks VaR-ES forecasts is the one the model minimises.

Early warning. P(realised vol over the next 21 days >= k x the calibrated EWMA vol at the
origin), k = 1.5 or 2, by pooled logistic regression: a baseline on the vol state alone,
and a full model on all the inputs.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize

from etr import scale, tails

INPUTS = ["vol_state", "trend_move", "last_shock", "shock_size", "iv_ratio", "iv_flag", "curve_slope", "monday"]
CLIP = 5.0
EW_HORIZON = 21


def features(df: pd.DataFrame, panel: pd.DataFrame | None = None, iv: pd.Series | None = None) -> pd.DataFrame:
    """Inputs at each origin t, from data up to t (the Monday flag uses the market's
    trading calendar, which is published in advance)."""
    r = df["ret"].fillna(0.0)
    ew = scale.ewma_variance(df)
    sd = np.sqrt(ew)
    r5 = r.rolling(5).sum()
    trend = np.sign(r.rolling(252, min_periods=252).sum())
    out = pd.DataFrame(index=df.index)
    out["vol_state"] = 0.5 * np.log(ew / ew.rolling(252, min_periods=200).mean())
    out["trend_move"] = trend * r5 / (sd * np.sqrt(5))
    out["last_shock"] = (df["ret"] / np.sqrt(ew.shift(1))).fillna(0.0)
    out["shock_size"] = r5.abs() / (sd * np.sqrt(5))
    if iv is not None:
        ivv = scale.filtered_iv_variance(iv.reindex(df.index), ew)
        ratio = 0.5 * np.log(ivv / ew)
        out["iv_ratio"], out["iv_flag"] = ratio.fillna(0.0), ratio.notna().astype(float)
    else:
        out["iv_ratio"], out["iv_flag"] = 0.0, 0.0
    if panel is not None and {1, 2} <= set(panel.columns):
        p = panel.reindex(df.index)
        out["curve_slope"] = np.log(p[2] / p[1]).ffill(limit=5).fillna(0.0)
    else:
        out["curve_slope"] = 0.0
    nxt = pd.Series(df.index, index=df.index).shift(-1)
    out["monday"] = (nxt.dt.weekday == 0).astype(float)
    return out[INPUTS].replace([np.inf, -np.inf], np.nan)


def standardiser(X: pd.DataFrame):
    """Mean and sd from training rows; returns a function mapping inputs to clipped
    standardised values (constant columns stay 0)."""
    mu, sd = X.mean(), X.std().replace(0, 1.0).fillna(1.0)
    return lambda Z: ((Z - mu) / sd).clip(-CLIP, CLIP).fillna(0.0)


# ---- Learned volatility multiplier ----

def _tail_nodes(shape, alpha: float, upper: bool):
    """Quantile at alpha and the quadrature nodes and weights of the alpha tail (the same
    construction as etr.tails.tail_mean), in standardised units."""
    u = alpha * tails._NODES ** 2
    q = shape.isf(alpha) if upper else shape.ppf(alpha)
    z = np.clip(shape.isf(u) if upper else shape.ppf(u), -tails.Z_CAP, tails.Z_CAP)
    return float(q), z, 2 * tails._NODES * tails._WEIGHTS


def fz0_objective(beta: np.ndarray, X: np.ndarray, sigma: np.ndarray, R: np.ndarray, groups: list, side: str,
                  alpha: float = 0.025) -> float:
    """Mean FZ0 of forecasts sigma x exp(X beta) x Z over training rows. groups: list of
    (row index array, quantile, tail nodes, tail weights), one per distinct shape."""
    m = np.exp(np.clip(X @ beta, -3, 3))
    s = sigma * m
    L = tails.to_loss(R, side)
    V, E = np.empty_like(s), np.empty_like(s)
    for idx, q, z, w in groups:
        V[idx] = tails.to_loss(s[idx] * q, side)
        E[idx] = (tails.to_loss(s[idx, None] * z[None, :], side) * w).sum(1)
    return float(np.mean((L > V) * (L - V) / (alpha * E) + V / E + np.log(E) - 1))


def fit_multiplier(X: np.ndarray, sigma: np.ndarray, R: np.ndarray, groups: list, side: str,
                   alpha: float = 0.025) -> np.ndarray:
    """beta minimising mean FZ0, starting from beta = 0 (the starting configuration)."""
    X1 = np.c_[np.ones(len(X)), X]
    res = optimize.minimize(fz0_objective, np.zeros(X1.shape[1]), args=(X1, sigma, R, groups, side, alpha),
                            method="L-BFGS-B")
    return res.x


# ---- Early warning ----

def stress_events(df: pd.DataFrame, k: float, h: int = EW_HORIZON) -> pd.Series:
    """1 if realised vol over the next h trading days is at least k x the calibrated EWMA
    vol at the origin (PROTOCOL Q5); NaN where the window runs past the data."""
    v = (df["ret"] ** 2 / df["span"].where(df["span"] > 0)).fillna(0.0)
    fwd = v[::-1].rolling(h).mean()[::-1].shift(-1)
    cal = scale.calibrated_ewma_variance(df)
    return (fwd >= k ** 2 * cal).astype(float).where(fwd.notna() & cal.notna())


def fit_logit(X: pd.DataFrame, y: pd.Series):
    """Pooled logistic regression; returns a predict function on new inputs."""
    model = sm.Logit(y.to_numpy(), sm.add_constant(X.to_numpy(), has_constant="add")).fit(disp=0, maxiter=200)
    return lambda Z: model.predict(sm.add_constant(Z.to_numpy(), has_constant="add")), model


def brier_skill(p_model: pd.DataFrame, p_base: pd.DataFrame, y: pd.DataFrame, block: int = 21,
                reps: int = 2000, seed: int = 20261024) -> dict:
    """Brier scores and the Brier skill score 1 - BS(model)/BS(base), pooled across columns
    (markets) per date, with circular block-bootstrap intervals over dates."""
    e_m = ((p_model - y) ** 2).mean(axis=1, skipna=True)
    e_b = ((p_base - y) ** 2).mean(axis=1, skipna=True)
    ok = e_m.notna() & e_b.notna()
    a, b = e_m[ok].to_numpy(), e_b[ok].to_numpy()
    T = len(a)
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    bss = np.empty(reps)
    for k in range(reps):
        idx = ((rng.integers(0, T, nb)[:, None] + np.arange(block)[None, :]) % T).ravel()[:T]
        bss[k] = 1 - a[idx].mean() / b[idx].mean()
    return {"dates": T, "brier_model": a.mean(), "brier_base": b.mean(), "bss": 1 - a.mean() / b.mean(),
            "ci90": tuple(np.quantile(bss, [0.05, 0.95])), "ci95": tuple(np.quantile(bss, [0.025, 0.975]))}


def reliability(p: pd.Series, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    """Mean forecast and observed frequency by decile of the forecast."""
    d = pd.DataFrame({"p": p, "y": y}).dropna()
    d["bin"] = pd.qcut(d["p"].rank(method="first"), bins, labels=False)
    return d.groupby("bin").agg(n=("y", "size"), mean_forecast=("p", "mean"), observed=("y", "mean"))


# ---- Walk-forward assembly ----

def learned_forecasts(rets: dict[str, pd.DataFrame], fc: dict[str, pd.DataFrame], insample: dict,
                      feats: dict[str, pd.DataFrame], years: list[int], scale_name: str, shape_name: str,
                      h: int = 1, approach: str = "sqrt10", extra=None, alpha: float = 0.025,
                      sides=tails.SIDES) -> tuple[dict, list]:
    """Forecasts of the learned multiplier on top of one configuration (the scale model
    `scale_name` with shape `shape_name`; at 10 days by square-root-of-time or term
    structure, as etr.engine builds them). extra(market, year) may return a Series of one
    more input (the early-warning probability from that year's fit)."""
    from etr import engine
    kind, pooled = engine.SHAPES[shape_name]
    term = h > 1 and approach == "term"
    col = f"{scale_name}_1" if h == 1 else (f"{scale_name}_10" if term else f"{scale_name}_sqrt10")
    parts = {(n, s): [] for n in rets for s in sides}
    params = []
    fwd = {n: tails.forward_returns(df, h) for n, df in rets.items()}
    end = {n: pd.Series(df.index, index=df.index).shift(-h) for n, df in rets.items()}
    for y in years:
        cut = pd.Timestamp(f"{y - 1}-12-31")
        if term:
            zs = {n: engine.residuals_10(df, insample[(n, y, f"{scale_name}_10")], cut) for n, df in rets.items()}
        else:
            zs = {n: engine.residuals(df, insample[(n, y, f"{scale_name}_1")], cut) for n, df in rets.items()}
        shapes = tails.fit_shapes(zs, kind, pooled)
        Xs, rows = {}, []
        for n, df in rets.items():
            F = feats[n].copy()
            if extra is not None:
                F["overlay"] = extra(n, y).reindex(F.index)
            Xs[n] = F
            ins = insample[(n, y, f"{scale_name}_10" if term else f"{scale_name}_1")]
            var_ins = ins * (1 if (h == 1 or term) else h)
            d = pd.concat({"var": var_ins.reindex(df.index), "R": fwd[n], "end": end[n]}, axis=1).join(F)
            d = d[(d.index <= cut) & (d["end"] <= cut)].dropna()
            d["market"] = n
            rows.append(d)
        train = pd.concat(rows)
        cols = list(Xs[next(iter(Xs))].columns)
        std = standardiser(train[cols])
        X = std(train[cols]).to_numpy()
        sigma, R = np.sqrt(train["var"].to_numpy()), train["R"].to_numpy()
        for side in sides:
            upper = side == "short"
            groups = []
            for n in rets:
                idx = np.flatnonzero(train["market"].to_numpy() == n)
                q, z, w = _tail_nodes(shapes[n], alpha, upper)
                groups.append((idx, q, z, w))
            beta = fit_multiplier(X, sigma, R, groups, side, alpha)
            params.append({"year": y, "side": side, "n_train": len(train), **dict(zip(["const"] + cols, beta))})
            for n, df in rets.items():
                var = fc[n][col]
                var_y = var.where(var.index.year == y).reindex(df.index)
                Z = std(Xs[n].reindex(df.index)[cols]).to_numpy()
                m = pd.Series(np.exp(np.clip(np.c_[np.ones(len(Z)), Z] @ beta, -3, 3)), index=df.index)
                f = tails.scale_shape_forecast(df, var_y * m ** 2, side, h, shapes[n])
                f.frame["multiplier"] = m.reindex(f.frame.index)              # recorded for diagnostics
                parts[(n, side)].append(f)
    return {k: tails.concat(v) for k, v in parts.items()}, params


def early_warning(rets: dict[str, pd.DataFrame], feats: dict[str, pd.DataFrame], years: list[int], k: float,
                  h: int = EW_HORIZON):
    """Walk-forward stress probabilities. Returns (frame of origin-level forecasts with
    columns market, y, p_base, p_full; {year: predictor(market, inputs) -> p_full}, so a
    year's fitted model can be applied to any origin, e.g. to build the overlay input)."""
    events = {n: stress_events(df, k, h) for n, df in rets.items()}
    end = {n: pd.Series(df.index, index=df.index).shift(-h) for n, df in rets.items()}
    out, predictors = [], {}
    for y in years:
        cut = pd.Timestamp(f"{y - 1}-12-31")
        rows = []
        for n, df in rets.items():
            d = feats[n].join(events[n].rename("y")).join(end[n].rename("end"))
            rows.append(d[(d.index <= cut) & (d["end"] <= cut)].dropna().assign(market=n))
        train = pd.concat(rows)
        std = standardiser(train[INPUTS])
        base, _ = fit_logit(std(train[INPUTS])[["vol_state"]], train["y"])
        full, _ = fit_logit(std(train[INPUTS]), train["y"])
        predictors[y] = (lambda f, s=std: (lambda F: pd.Series(f(s(F[INPUTS])), index=F.index)))(full)
        for n, df in rets.items():
            F = feats[n][feats[n].index.year == y].dropna()
            p_b = base(std(F[INPUTS])[["vol_state"]])
            p_f = full(std(F[INPUTS]))
            out.append(pd.DataFrame({"market": n, "y": events[n].reindex(F.index), "p_base": p_b, "p_full": p_f}, index=F.index))
    return pd.concat(out), predictors
