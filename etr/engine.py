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
    if config in BASELINE_RANK:
        return BASELINE_RANK[config]
    scale_name, shape_name = config.split("|")
    return SCALE_RANK[scale_name.rsplit("_", 1)[0]] + SHAPE_RANK[SHAPES[shape_name][0]]


def residuals(df: pd.DataFrame, var_ins: pd.Series, cut: pd.Timestamp) -> np.ndarray:
    """Standardised next-day residuals at training origins whose target is on or before cut."""
    span = df["span"].shift(-1)
    z = df["ret"].shift(-1) / np.sqrt(span.where(span > 0) * var_ins.reindex(df.index))
    target = pd.Series(df.index, index=df.index).shift(-1)
    z = z[(target <= cut).to_numpy()]
    return z[np.isfinite(z)].to_numpy()


def one_day(rets: dict[str, pd.DataFrame], fc: dict[str, pd.DataFrame], insample: dict, years: list[int],
            scale_col: str, shape_name: str) -> tuple[dict, list]:
    """1-day forecasts of one configuration for every market and side, and the fitted shape
    parameters by year. `fc`, `insample` come from etr.scale.walk_forward(keep_insample=True);
    scale_col is e.g. "comb_iv_1"."""
    kind, pooled = SHAPES[shape_name]
    parts = {(n, s): [] for n in rets for s in tails.SIDES}
    params = []
    for y in years:
        cut = pd.Timestamp(f"{y - 1}-12-31")
        zs = {n: residuals(df, insample[(n, y, scale_col)], cut) for n, df in rets.items()}
        shapes = tails.fit_shapes(zs, kind, pooled)
        for n, df in rets.items():
            params.append({"year": y, "market": n, **describe(shapes[n])})
            var = fc[n][scale_col]
            var_y = var.where(var.index.year == y).reindex(df.index)
            for side in tails.SIDES:
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
    coverage at 99% and 97.5% and ES at 97.5%, per side."""
    out = {}
    for side in tails.SIDES:
        U = pd.DataFrame({k: (bt.nonoverlapping(f.between(start, end).frame, h) if h > 1 else f.between(start, end).frame)["u"]
                          for k, f in forecasts.items() if k[1] == side})
        for alpha in (0.01, 0.025):
            r = bt.coverage_pooled(U, alpha, seed=seed)
            out[f"{side}_rate_{tails.col('var', alpha)}"], out[f"{side}_p_{tails.col('var', alpha)}"] = r["rate"], r["p"]
        e = bt.es_pooled(U, 0.025, seed=seed)
        out[f"{side}_es975_ratio"], out[f"{side}_p_es975"] = e["h_ratio"], e["p"]
    out["failures"] = int(sum(v < 0.05 for k, v in out.items() if "_p_" in k))
    return out


def select(scores: pd.DataFrame, failures: pd.Series, block: int, size: float = 0.10, seed: int = 20261024) -> pd.DataFrame:
    """PROTOCOL 4.2 steps 2-4. scores: per-date pooled FZ0 (columns = configurations);
    failures: number of the six pooled calibration tests each fails at 5%. Returns one row
    per configuration with its MCS p-value, shortlist and screen flags, complexity, and the
    chosen one marked."""
    P = scores.dropna()
    mcs = bt.model_confidence_set(P, size=size, block=block, reps=2000, seed=seed)
    t = pd.DataFrame({"mean_fz0": P.mean(), "mcs_p": mcs, "failures": failures.reindex(P.columns),
                      "complexity": [complexity(c) for c in P.columns]})
    t["shortlisted"] = t["mcs_p"] >= size
    t["passes_screen"] = t["shortlisted"] & (t["failures"] == 0)
    pool = t[t["passes_screen"]]
    if pool.empty:                                   # rule 4: fewest failures, then simplest
        pool = t[t["shortlisted"]]
        pool = pool[pool["failures"] == pool["failures"].min()]
    choice = pool.sort_values(["complexity", "mean_fz0"]).index[0]
    t["chosen"] = t.index == choice
    return t.sort_values(["shortlisted", "mean_fz0"], ascending=[False, True])
