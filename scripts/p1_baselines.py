"""Phase 1: the three baselines on development data, run through the whole toolkit.

Purpose: an end-to-end check of the toolkit on real data, and the standard errors the
protocol needs for its minimum detectable effects. Data after 2018-12-31 is dropped
before anything is computed (the 2019+ test window stays sealed).

Baselines (fixed definitions, nothing tuned):
  hs500        historical simulation, last 500 trading days (overlapping h-day returns)
  riskmetrics  EWMA (lambda 0.94) variance, normal, sqrt(h) scaling
  ewma_cal     the same with the point-in-time level calibration (the previous project's
               most robust sizer), normal, sqrt(h) scaling
Positions: unit long and unit short in each of the six markets. TTF from 2014.
Windows: 2012-14 (early check) and 2015-18 (validation, where model choices will be made).

Writes results/p1_baselines_{series,pooled,scores}.csv.
Run:  python scripts/p1_baselines.py
"""
import numpy as np
import pandas as pd
from scipy import stats

import etr
from etr import backtest as bt
from etr import scale, tails
from etr.markets import MARKETS

DEV_END = "2018-12-31"
WINDOWS = {"2012-14": ("2012-01-01", "2014-12-31"), "2015-18": ("2015-01-01", "2018-12-31")}
MODELS = ("hs500", "riskmetrics", "ewma_cal")
TEST_DAYS = 1950          # 1-day origins per market in 2019-01 .. 2026-10
SEED = 20261024


def forecast(df: pd.DataFrame, model: str, side: str, h: int) -> tails.Forecast:
    if model == "hs500":
        return tails.historical_forecast(df, side, h)
    v = scale.ewma_variance(df) if model == "riskmetrics" else scale.calibrated_ewma_variance(df)
    return tails.scale_shape_forecast(df, v, side, h)


def build(rets: pd.DataFrame) -> dict:
    """{(model, market, side, h): Forecast} on data up to DEV_END only."""
    out = {}
    for name, m in MARKETS.items():
        df = rets[rets.market == name].loc[:DEV_END]
        for model in MODELS:
            for side in tails.SIDES:
                for h in (1, 10):
                    fc = forecast(df, model, side, h)
                    out[(model, name, side, h)] = fc.between(m.liquid_from or "2010-01-01", DEV_END)
    return out


def next_weekday(df_index: pd.DatetimeIndex, origins: pd.DatetimeIndex) -> np.ndarray:
    """Weekday dummies (Tue..Fri) of each origin's target day: known in advance."""
    pos = df_index.get_indexer(origins) + 1
    wd = df_index[np.minimum(pos, len(df_index) - 1)].weekday
    return np.column_stack([wd == k for k in range(1, 5)]).astype(float)


def series_tests(fcs: dict, rets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(SEED)
    for (model, name, side, h), fc in fcs.items():
        if h != 1:
            continue
        idx = rets[rets.market == name].loc[:DEV_END].index
        for win, (a, b) in WINDOWS.items():
            f = fc.between(a, b)
            fr = f.frame
            if len(fr) < 250:
                continue
            L = fr["loss"].to_numpy()
            row = {"window": win, "model": model, "market": name, "side": side, "days": len(fr)}
            for alpha in tails.ALPHAS:
                row[f"rate_{tails.col('var', alpha)}"] = np.mean(L > fr[tails.col("var", alpha)])
            for alpha in (0.01, 0.025):
                v = tails.col("var", alpha)
                hit = bt.hits(L, fr[v])
                row[f"kupiec_p_{v}"] = bt.kupiec(hit, alpha)["p"]
                row[f"christoffersen_cc_p_{v}"] = bt.christoffersen(hit, alpha, seed=SEED)["p_cc"]
                exog = np.column_stack([fr[v].to_numpy(), next_weekday(idx, fr.index)])
                row[f"dq_p_{v}"] = bt.dq_test(hit, alpha, exog=exog, seed=SEED)["p"]
            tl = bt.traffic_light(bt.hits(L, fr["var99"]), 0.01)
            row.update(tl_red_share=tl["red"], tl_yellow_share=tl["yellow"], tl_max_exceptions=tl["max_exceptions"])
            a975 = 0.025
            row["as_z2_975"], row["as_p_975"] = (lambda r: (r["z2"], r["p"]))(
                bt.acerbi_szekely_z2(L, fr["var975"], fr["es975"], a975, f.sample(rng, 1999)))
            row["de_p_975"] = bt.du_escanciano(fr["u"], a975)["p_u"]
            row["mf_p_975"] = bt.mcneil_frey(L, fr["var975"], fr["es975"], seed=SEED)["p"]
            rows.append(row)
    return pd.DataFrame(rows)


def panel(fcs: dict, model: str, h: int, field: str, win: tuple, sides=tails.SIDES, nonoverlap=False) -> pd.DataFrame:
    cols = {}
    for name in MARKETS:
        for side in sides:
            fr = fcs[(model, name, side, h)].between(*win).frame
            if nonoverlap:
                fr = bt.nonoverlapping(fr, h)
            cols[(name, side)] = fr[field]
    return pd.DataFrame(cols)


def pooled_tests(fcs: dict) -> pd.DataFrame:
    rows = []
    for win, w in WINDOWS.items():
        for h in (1, 10):
            for model in MODELS:
                for side in tails.SIDES:
                    U = panel(fcs, model, h, "u", w, sides=(side,), nonoverlap=h > 1)
                    row = {"window": win, "horizon": h, "model": model, "side": side, "dates": len(U)}
                    for alpha in (0.01, 0.025):
                        r = bt.coverage_pooled(U, alpha, seed=SEED)
                        row[f"rate_{tails.col('var', alpha)}"], row[f"p_{tails.col('var', alpha)}"] = r["rate"], r["p"]
                    e = bt.es_pooled(U, 0.025, seed=SEED)
                    row["es975_h_ratio"], row["es975_p"] = e["h_ratio"], e["p"]
                    rows.append(row)
    return pd.DataFrame(rows)


def fz0_panel(fcs, model, h, win, alpha):
    v, e = tails.col("var", alpha), tails.col("es", alpha)
    L, V, E = (panel(fcs, model, h, f, win) for f in ("loss", v, e))
    return pd.DataFrame(bt.fz0(L, V, E, alpha), index=L.index, columns=L.columns)


def scale_error_gap(alpha: float, deltas=np.linspace(0, 0.4, 81)) -> pd.Series:
    """Expected FZ0 penalty of a forecast whose scale is off by a factor (1 + delta), for a
    t(5) loss distribution (FZ0 is scale-free, so this does not depend on the level of vol).
    Translates a detectable FZ0 difference into a detectable scale error."""
    t5 = stats.t(5, scale=np.sqrt(3 / 5))
    x = t5.rvs(2_000_000, random_state=1)                  # losses in units of their scale
    q = t5.isf(alpha)
    es = t5.expect(lambda z: z, lb=q) / alpha
    base = bt.fz0(x, q, es, alpha).mean()
    # The smaller of the penalties for over- and under-stating the scale by (1 + delta).
    gaps = [min(bt.fz0(x, q * k, es * k, alpha).mean() for k in (1 + d, 1 / (1 + d))) - base for d in deltas]
    return pd.Series(gaps, index=deltas)


def scores(fcs: dict) -> pd.DataFrame:
    rows = []
    for win, w in WINDOWS.items():
        for h in (1, 10):
            lags = 5 if h == 1 else 2 * h
            for alpha in (0.025, 0.01):
                pooled = {m: bt.date_mean(fz0_panel(fcs, m, h, w, alpha)) for m in MODELS}
                P = pd.DataFrame(pooled).dropna()
                mcs = bt.model_confidence_set(P, block=10 if h == 1 else 2 * h)
                for m in MODELS:
                    row = {"window": win, "horizon": h, "alpha": alpha, "model": m, "dates": len(P),
                           "mean_fz0": P[m].mean(), "mcs_p": mcs[m]}
                    for ref in ("riskmetrics", "ewma_cal"):
                        if m != ref:
                            dm = bt.diebold_mariano(P[m] - P[ref], lags)
                            row[f"diff_vs_{ref}"], row[f"se_vs_{ref}"], row[f"p_vs_{ref}"] = dm["mean_diff"], dm["se"], dm["p"]
                    rows.append(row)
    out = pd.DataFrame(rows)
    # Minimum detectable FZ0 difference over the test window (5% two-sided, 80% power:
    # 2.8 standard errors), from the validation-window standard error of ewma_cal vs
    # riskmetrics, scaled by sqrt(dev days / test days).
    ses = out[(out.window == "2015-18") & (out.model == "ewma_cal")].set_index(["horizon", "alpha"])
    mde = {}
    for (h, alpha), r in ses.iterrows():
        n_test = TEST_DAYS if h == 1 else TEST_DAYS - 10
        mde[(h, alpha)] = 2.8 * r["se_vs_riskmetrics"] * np.sqrt(r["dates"] / n_test)
    out["mde_fz0_test_window"] = [mde.get((h, a)) for h, a in zip(out.horizon, out.alpha)]
    # The same for each model against ewma_cal: pairs of very different forecasts (hs500 vs
    # ewma_cal) have far larger loss-difference standard errors than near-identical ones.
    n_test = np.where(out.horizon == 1, TEST_DAYS, TEST_DAYS - 10)
    out["mde_vs_ewma_cal_test_window"] = 2.8 * out["se_vs_ewma_cal"] * np.sqrt(out["dates"] / n_test)
    return out


def main() -> None:
    rets = pd.read_parquet(etr.PROCESSED / "returns.parquet")
    rets = rets[rets.index <= DEV_END]                         # seal: nothing after 2018
    fcs = build(rets)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    s = series_tests(fcs, rets)
    s.round(4).to_csv(etr.RESULTS / "p1_baselines_series.csv", index=False)
    print(s.round(3).to_string(index=False))
    p = pooled_tests(fcs)
    p.round(4).to_csv(etr.RESULTS / "p1_baselines_pooled.csv", index=False)
    print(p.round(3).to_string(index=False))
    sc = scores(fcs)
    for alpha in (0.025, 0.01):
        gap = scale_error_gap(alpha)
        to_delta = lambda m: gap.index[np.argmax(gap.to_numpy() >= m)] if (gap >= m).any() else np.nan  # noqa: E731
        sel = sc.alpha == alpha
        sc.loc[sel, "mde_as_scale_error"] = sc.loc[sel, "mde_fz0_test_window"].map(to_delta)
        sc.loc[sel, "mde_vs_ewma_cal_as_scale_error"] = sc.loc[sel, "mde_vs_ewma_cal_test_window"].map(to_delta)
    sc.round(5).to_csv(etr.RESULTS / "p1_baselines_scores.csv", index=False)
    print(sc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
