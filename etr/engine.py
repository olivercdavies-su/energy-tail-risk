"""The VaR/ES engine: a scale (volatility) model times a shape, built walk-forward, and the
pre-registered rule that chooses among configurations (PROTOCOL.md section 4).

For each forecast year Y and configuration (scale, shape):
  1. Training residuals: z_t = r_{t+1} / sqrt(span_{t+1} x var_t) at the training origins
     whose target ended by 31 December Y-1, with var_t the scale model's calibrated
     forecast from its *year-Y* parameters (so z has mean square 1 in-sample, and the shape
     describes exactly the forecasts it will be combined with).
  2. The shape is fitted to those residuals, per market or pooled across markets.
  3. Year-Y forecasts: etr.tails.scale_shape_forecast with the scale's year-Y forecasts.
Nothing in a year's forecasts uses that year's returns, except each origin's own past.

10 days (PROTOCOL.md 4, Phase 4). Three ways to forecast the 10-day loss:
  sqrt10  the 1-day engine scaled by sqrt(10): variance 10 x the 1-day forecast, the
          shape fitted to daily residuals (the classic Basel scaling);
  term    each scale model's own calibrated 10-day forecast, with the shape fitted to
          10-day residuals R10 / sigma10 at training origins whose 10-day window ended by
          the cut-off (overlapping windows; the shape then carries whatever happens inside
          the horizon: rolls, volatility mean reversion, autocorrelation);
  paths   filtered historical simulation along GJR paths (Barone-Adesi, Giannopoulos &
          Vosper 1999): 2,000 ten-day paths per origin, each day's shock a resampled
          standardised GJR residual and the variance updated by the GJR recursion, so a
          fall raises the volatility of the rest of the path.
"""
import numpy as np
import pandas as pd

from etr import backtest as bt
from etr import tails

SHAPES = {"normal": ("normal", False), "t_pool": ("t", True), "t_mkt": ("t", False),
          "skewt_pool": ("skewt", True), "skewt_mkt": ("skewt", False), "fhs_pool": ("fhs", True),
          "fhs_mkt": ("fhs", False), "gpd_pool": ("gpd", True), "gpd_mkt": ("gpd", False)}
# PROTOCOL.md 4.1: lower is simpler. Baselines: riskmetrics and ewma_cal 1, hs500 3.
SCALE_RANK = {"ewma": 1, "ewma_wd": 2, "gjr": 3, "har": 3, "iv": 4, "comb": 5, "comb_iv": 5}
SHAPE_RANK = {"normal": 0, "t": 1, "skewt": 2, "fhs": 3, "gpd": 4}
BASELINE_RANK = {"hs500": 3, "riskmetrics": 1, "ewma_cal": 1}


def complexity(config: str) -> int:
    """PROTOCOL 4.1: scale + shape ranks; at 10 days a term structure or path simulation
    adds 1, square-root-of-time adds 0."""
    if config in BASELINE_RANK:
        return BASELINE_RANK[config]
    scale_name, shape_name = config.split("|")
    term = scale_name.endswith(("_10", "_paths"))
    shape_rank = SHAPE_RANK["fhs"] if shape_name == "joint_fhs" else SHAPE_RANK[SHAPES[shape_name][0]]
    return SCALE_RANK[scale_name.rsplit("_", 1)[0]] + shape_rank + int(term)


def residuals(df: pd.DataFrame, var_ins: pd.Series, cut: pd.Timestamp) -> np.ndarray:
    """Standardised next-day residuals at training origins whose target is on or before cut."""
    span = df["span"].shift(-1)
    z = df["ret"].shift(-1) / np.sqrt(span.where(span > 0) * var_ins.reindex(df.index))
    target = pd.Series(df.index, index=df.index).shift(-1)
    z = z[(target <= cut).to_numpy()]
    return z[np.isfinite(z)].to_numpy()


def residuals_10(df: pd.DataFrame, var_ins: pd.Series, cut: pd.Timestamp) -> np.ndarray:
    """Standardised 10-day residuals R10 / sqrt(var10) at training origins whose 10-day
    window ended on or before cut (overlapping windows)."""
    z = tails.forward_returns(df, 10) / np.sqrt(var_ins.reindex(df.index))
    end = pd.Series(df.index, index=df.index).shift(-10)
    z = z[(end <= cut).to_numpy()]
    return z[np.isfinite(z)].to_numpy()


def ten_day(rets: dict[str, pd.DataFrame], fc: dict[str, pd.DataFrame], insample: dict, years: list[int],
            scale_name: str, shape_name: str, approach: str, sides=tails.SIDES) -> tuple[dict, list]:
    """10-day forecasts of one configuration. approach "sqrt10": the 1-day engine (daily
    residuals, variance 10 x the 1-day forecast); "term": the model's own 10-day forecast and
    a shape fitted to 10-day residuals. scale_name is the model, e.g. "gjr"."""
    kind, pooled = SHAPES[shape_name]
    parts = {(n, s): [] for n in rets for s in sides}
    params = []
    col = f"{scale_name}_sqrt10" if approach == "sqrt10" else f"{scale_name}_10"
    for y in years:
        cut = pd.Timestamp(f"{y - 1}-12-31")
        if approach == "sqrt10":
            zs = {n: residuals(df, insample[(n, y, f"{scale_name}_1")], cut) for n, df in rets.items()}
        else:
            zs = {n: residuals_10(df, insample[(n, y, f"{scale_name}_10")], cut) for n, df in rets.items()}
        shapes = tails.fit_shapes(zs, kind, pooled)
        for n, df in rets.items():
            params.append({"year": y, "market": n, **describe(shapes[n])})
            var = fc[n][col]
            var_y = var.where(var.index.year == y).reindex(df.index)
            for side in sides:
                parts[(n, side)].append(tails.scale_shape_forecast(df, var_y, side, 10, shapes[n]))
    return {k: tails.concat(v) for k, v in parts.items()}, params


def gjr_path_returns(sigma2_1: np.ndarray, params: pd.Series, pool: np.ndarray, rng: np.random.Generator,
                     n_paths: int = 2000, h: int = 10) -> np.ndarray:
    """Simulated h-day log returns (origins x paths) from a GJR-GARCH recursion. sigma2_1 is
    each origin's one-step variance forecast (decimal units); each day's shock is a draw
    from `pool` (standardised residuals) times that path's current volatility, and
    s2_{k+1} = w + (a + g 1[r_k < 0]) r_k^2 + b s2_k."""
    w = params["omega"] / 1e4
    a, g, b = params["alpha[1]"], params["gamma[1]"], params["beta[1]"]
    s2 = np.repeat(np.asarray(sigma2_1, float)[:, None], n_paths, axis=1)
    total = np.zeros_like(s2)
    for _ in range(h):
        r = np.sqrt(s2) * pool[rng.integers(0, len(pool), s2.shape)]
        total += r
        s2 = w + (a + g * (r < 0)) * r ** 2 + b * s2
    return total


def gjr_paths(rets: dict[str, pd.DataFrame], diag: pd.DataFrame, years: list[int], pooled: bool,
              n_paths: int = 2000, seed: int = 20261024, keep_draws: bool = True, sides=tails.SIDES) -> tuple[dict, list]:
    """10-day forecasts by filtered historical simulation along GJR paths. Uses the same
    yearly GJR fits as etr.scale.walk_forward (from its diagnostics frame), the in-sample
    standardised residuals of those fits as the shock pool (per market, or pooled), and the
    GJR 10-day calibration scale, so its average level matches the calibrated gjr_10."""
    from etr import scale
    parts = {(n, s): [] for n in rets for s in sides}
    info = []
    for y in years:
        cut = pd.Timestamp(f"{y - 1}-12-31")
        fits, pools = {}, {}
        for n, df in rets.items():
            row = diag.query("market == @n and year == @y and model == 'gjr' and omega == omega").iloc[0]
            params = row[["omega", "alpha[1]", "gamma[1]", "beta[1]", "nu"]].astype(float)
            r = scale.daily_equivalent_returns(df)
            res = scale._gjr(r.loc[:cut]).fix(params)
            z = res.std_resid.iloc[60:].to_numpy()
            pools[n] = z[np.isfinite(z)]
            c10 = diag.query("market == @n and year == @y and model == 'gjr' and h == 10")["scale_c"].iloc[0]
            fits[n] = (params, r, c10)
        pooled_z = np.concatenate(list(pools.values()))
        for k, (n, df) in enumerate(rets.items()):
            params, r, c10 = fits[n]
            s1 = scale.gjr_paths(r.loc[:f"{y}-12-31"], params, start=f"{y}-01-01", horizon=1)["h.1"]
            fwd = tails.forward_returns(df, 10)
            origins = s1.index[s1.index.isin(fwd.dropna().index)]
            rng = np.random.default_rng([seed, y, k])
            pool = pooled_z if pooled else pools[n]
            R = gjr_path_returns(s1.loc[origins].to_numpy(), params, pool, rng, n_paths) * np.sqrt(c10)
            info.append({"year": y, "market": n, "c10": c10, "pool_size": len(pool), "pool_mean_square": float(np.mean(pool ** 2))})
            for side in sides:
                parts[(n, side)].append(simulated_forecast(fwd.loc[origins], R, side, keep_draws))
    return {k: tails.concat(v) for k, v in parts.items()}, info


def simulated_forecast(realised: pd.Series, sims: np.ndarray, side: str, keep_draws: bool = True) -> tails.Forecast:
    """A Forecast from simulated h-day log returns (origins x paths): VaR and ES by the
    historical-simulation convention on the simulated losses, the PIT as the share of
    simulated losses at least as large as the realised one, draws resampled from the paths.
    keep_draws=False drops the paths after summarising them (selection needs only VaR, ES
    and the PIT; the paths take about 16 kB per origin)."""
    L = tails.to_loss(sims, side).astype(np.float32)
    srt = -np.sort(-L, axis=1)
    loss = tails.to_loss(realised.to_numpy(), side)
    out = {"loss": loss}
    for a in tails.ALPHAS:
        out[tails.col("var", a)], out[tails.col("es", a)] = tails.empirical_var_es(srt.astype(float), a)
    out["u"] = (L >= loss[:, None]).mean(1)
    frame = pd.DataFrame(out, index=realised.index)

    if not keep_draws:
        def unavailable(rng, n, rows):
            raise RuntimeError("built with keep_draws=False")
        return tails.Forecast(frame, unavailable)

    def sampler(rng, n, rows):
        sub = L if rows is None else L[rows]
        pick = rng.integers(0, sub.shape[1], (n, len(sub)))
        return sub[np.arange(len(sub))[None, :], pick].astype(float)

    return tails.Forecast(frame, sampler)


def one_day(rets: dict[str, pd.DataFrame], fc: dict[str, pd.DataFrame], insample: dict, years: list[int],
            scale_col: str, shape_name: str, sides=tails.SIDES) -> tuple[dict, list]:
    """1-day forecasts of one configuration for every market and side, and the fitted shape
    parameters by year. `fc`, `insample` come from etr.scale.walk_forward(keep_insample=True);
    scale_col is e.g. "comb_iv_1"."""
    kind, pooled = SHAPES[shape_name]
    parts = {(n, s): [] for n in rets for s in sides}
    params = []
    for y in years:
        cut = pd.Timestamp(f"{y - 1}-12-31")
        zs = {n: residuals(df, insample[(n, y, scale_col)], cut) for n, df in rets.items()}
        shapes = tails.fit_shapes(zs, kind, pooled)
        for n, df in rets.items():
            params.append({"year": y, "market": n, **describe(shapes[n])})
            var = fc[n][scale_col]
            var_y = var.where(var.index.year == y).reindex(df.index)
            for side in sides:
                parts[(n, side)].append(tails.scale_shape_forecast(df, var_y, side, 1, shapes[n]))
    return {k: tails.concat(v) for k, v in parts.items()}, params


def describe(shape) -> dict:
    """The fitted parameters of a shape, for the diagnostics tables."""
    if isinstance(shape, tails.StudentT):
        return {"nu": shape.nu}
    if isinstance(shape, tails.SkewT):
        return {"eta": shape.eta, "lambda": shape.lam}
    if isinstance(shape, tails.GPDTails):
        return {"xi_left": shape.xiL, "xi_right": shape.xiR, "beta_left": shape.bL, "beta_right": shape.bR}
    if isinstance(shape, tails.Empirical):
        return {"n_residuals": shape.n}
    return {}


# ---- The selection rule (PROTOCOL.md 4.2) ----

def pooled_fz0(forecasts: dict, start, end, alpha: float = 0.025) -> pd.Series:
    """Per-date mean FZ0 over market-sides (TTF only from its liquid date, via the caller)."""
    v, e = tails.col("var", alpha), tails.col("es", alpha)
    cols = {}
    for key, f in forecasts.items():
        fr = f.between(start, end).frame
        cols[key] = pd.Series(bt.fz0(fr["loss"], fr[v], fr[e], alpha), index=fr.index)
    return bt.date_mean(pd.DataFrame(cols))


def pooled_calibration(forecasts: dict, start, end, h: int = 1, seed: int = 20261024) -> dict:
    """The six pooled tests of PROTOCOL Q1 (Q3 at 10 days, on non-overlapping windows):
    coverage at 99% and 97.5% and ES at 97.5%, per side. Books have one side (the book
    itself), so for them it is three tests."""
    out = {}
    for side in [s for s in tails.SIDES if any(k[1] == s for k in forecasts)]:
        U = pd.DataFrame({k: (bt.nonoverlapping(f.between(start, end).frame, h) if h > 1 else f.between(start, end).frame)["u"]
                          for k, f in forecasts.items() if k[1] == side})
        for alpha in (0.01, 0.025):
            r = bt.coverage_pooled(U, alpha, seed=seed)
            out[f"{side}_rate_{tails.col('var', alpha)}"], out[f"{side}_p_{tails.col('var', alpha)}"] = r["rate"], r["p"]
        e = bt.es_pooled(U, 0.025, seed=seed)
        out[f"{side}_es975_ratio"], out[f"{side}_p_es975"] = e["h_ratio"], e["p"]
    out["failures"] = int(sum(v < 0.05 for k, v in out.items() if "_p_" in k))
    return out


def select(scores: pd.DataFrame, failures: pd.Series, block: int, b3: str | None = None, size: float = 0.10,
           seed: int = 20261024) -> pd.DataFrame:
    """PROTOCOL 4.2 steps 2-4, with Amendment A1. scores: per-date pooled FZ0 (columns =
    configurations); failures: number of the six pooled calibration tests each fails at 5%;
    b3: the column holding baseline B3 (calibrated EWMA-normal).
    Amendment A1 (PROTOCOL section 10): a configuration is eligible only if its mean FZ0 on
    the validation window is no worse than B3's. The model confidence set keeps any model it
    cannot reject, and a model with very noisy losses is hard to reject however badly it
    scores (at 10 days historical simulation was shortlisted with a mean FZ0 0.135 worse
    than B3, and as the simplest survivor it would have been chosen). B3 is always eligible,
    so "nothing beats calibrated EWMA" remains a possible answer.
    Returns one row per configuration with its MCS p-value, the flags of each step,
    complexity, and the chosen one marked."""
    P = scores.dropna()
    # Identical forecasts are one model under several names (e.g. for a book, weekday EWMA
    # whose factors all shrink to 1 is plain EWMA): keep the simplest name. The model
    # confidence set cannot compare two identical columns (their difference has no variance).
    cols = list(P.columns)
    kept, twin = [], {}
    for c in sorted(cols, key=lambda c: (complexity(c), cols.index(c))):
        same = next((k for k in kept if np.array_equal(P[c].to_numpy(), P[k].to_numpy())), None)
        if same is None:
            kept.append(c)
        else:
            twin[c] = same
    mcs = bt.model_confidence_set(P[kept], size=size, block=block, reps=2000, seed=seed).reindex(P.columns)
    # Every column is indexed by configuration explicitly (a partial Series such as
    # duplicate_of would otherwise make pandas sort the index and misalign a plain list).
    t = pd.DataFrame({"mean_fz0": P.mean(), "mcs_p": mcs, "failures": failures.reindex(P.columns),
                      "complexity": pd.Series({c: complexity(c) for c in P.columns}),
                      "duplicate_of": pd.Series(twin, dtype=object).reindex(P.columns)}).reindex(P.columns)
    t["shortlisted"] = (t["mcs_p"] >= size) & t["duplicate_of"].isna()
    t["no_worse_than_b3"] = t["mean_fz0"] <= t.loc[b3, "mean_fz0"] + 1e-12 if b3 is not None else True
    t["passes_screen"] = t["shortlisted"] & t["no_worse_than_b3"] & (t["failures"] == 0)
    pool = t[t["passes_screen"]]
    if pool.empty:                                   # rule 4: fewest failures, then simplest
        pool = t[t["shortlisted"] & t["no_worse_than_b3"]]
        if pool.empty:
            pool = t.loc[[b3]]
        pool = pool[pool["failures"] == pool["failures"].min()]
    choice = pool.sort_values(["complexity", "mean_fz0"]).index[0]
    t["chosen"] = t.index == choice
    return t.sort_values(["shortlisted", "mean_fz0"], ascending=[False, True])
