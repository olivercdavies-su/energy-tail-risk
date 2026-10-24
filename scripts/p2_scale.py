"""Phase 2: scale models on development data (2012-18; model choices use 2015-18).

Data after 2018-12-31 is dropped before anything is computed. Every scale model is
evaluated with a normal shape (fat-tailed shapes are Phase 3), against calibrated EWMA
(baseline B3) at 1 day and its square-root-of-time version at 10 days.

Writes (results/):
  p2_scale_qlike.csv        variance accuracy: QLIKE difference vs B3, pooled DM t
  p2_scale_fz0.csv          joint VaR-ES score (normal shape): pooled FZ0 vs B3, DM, MCS
  p2_scale_calibration.csv  the six pooled calibration tests of PROTOCOL Q1 / Q3
  p2_scale_weekday.csv      realised / forecast variance by target weekday; DQ rejections
  p2_scale_term.csv         10-day: sd of R10 / forecast sd by vol state
  p2_scale_params.csv       GJR and HAR parameters and implied-vol filter counts, by market-year
  p2_scale_weekday_factors.csv  shrunk weekday factors, by market-year
  p2_scale_cal_scales.csv   point-in-time calibration scales, by market, model, horizon, year
  p2_scale_fz0.png
Forecasts are cached (gitignored) in data/processed/scale_forecasts.parquet for Phase 3.

Run:  python scripts/p2_scale.py
"""
import numpy as np
import pandas as pd
from scipy import stats

import etr
from etr import backtest as bt
from etr import scale, tails
from etr.markets import MARKETS

DEV_END = "2018-12-31"
YEARS = list(range(2012, 2019))
WINDOWS = {"2012-14": ("2012-01-01", "2014-12-31"), "2015-18": ("2015-01-01", "2018-12-31")}
VARIANTS = {1: ["1"], 10: ["sqrt10", "10"]}
B3 = {1: "ewma_1", 10: "ewma_sqrt10"}
SEED = 20261024
GROUPS = {"European gas": ("TTF", "NBP"), "HH": ("HH",), "oil": ("BRENT", "WTI"), "EUA": ("EUA",)}


def load():
    rets = pd.read_parquet(etr.PROCESSED / "returns.parquet")
    rets = rets[rets.index <= DEV_END]
    R = {n: rets[rets.market == n] for n in MARKETS}
    iv = pd.read_parquet(etr.PROCESSED / "implied_vol.parquet").loc[:DEV_END]
    return R, {n: iv[n] for n in iv.columns}


def start_of(name: str, a: str) -> str:
    return max(a, MARKETS[name].liquid_from or a)


def columns(h: int) -> list[str]:
    return [f"{m}_{v}" for m in scale.MODELS for v in VARIANTS[h]]


def qlike_table(fc: dict) -> pd.DataFrame:
    """QLIKE up to terms that do not depend on the forecast, L = y / f + log f, so
    differences are exact even on zero-return days (where log y is undefined). Per-market
    means of L(model) - L(B3), and a pooled DM test on the per-date mean across markets."""
    rows = []
    for win, (a, b) in WINDOWS.items():
        for h in (1, 10):
            y = f"y{h}"
            diffs = {}
            for col in columns(h):
                per = {}
                for name, f in fc.items():
                    d = f.loc[start_of(name, a):b].dropna(subset=[y, col, B3[h]])
                    loss = lambda c: d[y] / d[c] + np.log(d[c])  # noqa: E731
                    per[name] = loss(col) - loss(B3[h])
                diffs[col] = pd.DataFrame(per)
                pooled = bt.date_mean(diffs[col])
                row = {"window": win, "horizon": h, "forecast": col}
                row.update({f"diff_{n}": s.mean() for n, s in per.items()})
                if col != B3[h]:
                    dm = bt.diebold_mariano(pooled, lags=5 if h == 1 else 20)
                    row.update(pooled_diff=dm["mean_diff"], pooled_t=dm["t"], pooled_p=dm["p"])
                rows.append(row)
    return pd.DataFrame(rows)


def normal_forecasts(fc: dict, R: dict, h: int, col: str) -> dict:
    return {(name, side): tails.scale_shape_forecast(R[name], fc[name][col], side, h)
            for name in fc for side in tails.SIDES}


def panel(fcs: dict, field: str, win: tuple, sides=tails.SIDES, h: int = 1, nonoverlap: bool = False) -> pd.DataFrame:
    cols = {}
    for (name, side), f in fcs.items():
        if side not in sides:
            continue
        fr = f.between(start_of(name, win[0]), win[1]).frame
        cols[(name, side)] = bt.nonoverlapping(fr, h)[field] if nonoverlap else fr[field]
    return pd.DataFrame(cols)


def fz0_pooled(fcs: dict, win: tuple, alpha: float) -> pd.Series:
    v, e = tails.col("var", alpha), tails.col("es", alpha)
    L, V, E = (panel(fcs, f, win) for f in ("loss", v, e))
    return bt.date_mean(pd.DataFrame(bt.fz0(L, V, E, alpha), index=L.index, columns=L.columns))


def score_and_calibration(fc: dict, R: dict):
    fz_rows, cal_rows, dq_rows, wd_rows = [], [], [], []
    for h in (1, 10):
        lags = 5 if h == 1 else 20
        fcs = {col: normal_forecasts(fc, R, h, col) for col in columns(h)}
        for win, w in WINDOWS.items():
            for alpha in (0.025, 0.01):
                P = pd.DataFrame({col: fz0_pooled(f, w, alpha) for col, f in fcs.items()}).dropna()
                mcs = bt.model_confidence_set(P, block=10 if h == 1 else 20, reps=1000, seed=SEED)
                for col in P:
                    row = {"window": win, "horizon": h, "alpha": alpha, "forecast": col, "dates": len(P),
                           "mean_fz0": P[col].mean(), "mcs_p": mcs[col]}
                    if col != B3[h]:
                        dm = bt.diebold_mariano(P[col] - P[B3[h]], lags)
                        row.update(diff_vs_b3=dm["mean_diff"], se=dm["se"], t=dm["t"], p=dm["p"])
                    fz_rows.append(row)
            for col, f in fcs.items():
                row = {"window": win, "horizon": h, "forecast": col}
                for side in tails.SIDES:
                    U = panel(f, "u", w, sides=(side,), h=h, nonoverlap=h > 1)
                    for alpha in (0.01, 0.025):
                        r = bt.coverage_pooled(U, alpha, seed=SEED)
                        row[f"{side}_rate_{tails.col('var', alpha)}"], row[f"{side}_p_{tails.col('var', alpha)}"] = r["rate"], r["p"]
                    e = bt.es_pooled(U, 0.025, seed=SEED)
                    row[f"{side}_es975_ratio"], row[f"{side}_es975_p"] = e["h_ratio"], e["p"]
                ps = [v for k, v in row.items() if "_p_" in k or k.endswith("es975_p")]
                row["failures_at_5pct"] = int(np.sum(np.array(ps) < 0.05))
                cal_rows.append(row)
                if h == 1:
                    dq_rows.append(dq_counts(f, R, w, win, col))
                    wd_rows.extend(weekday_rows(fc, R, w, win, col))
    return (pd.DataFrame(fz_rows), pd.DataFrame(cal_rows),
            pd.DataFrame(dq_rows).merge(pd.DataFrame(wd_rows).pivot_table(
                index=["window", "forecast"], columns=["group", "weekday"], values="ratio").pipe(flatten),
                on=["window", "forecast"]))


def flatten(t: pd.DataFrame) -> pd.DataFrame:
    t.columns = [f"ratio_{g}_{d}" for g, d in t.columns]
    return t.reset_index()


def dq_counts(fcs: dict, R: dict, w: tuple, win: str, col: str) -> dict:
    """Market-sides (of 12) failing the DQ test with weekday regressors at 5%."""
    out = {"window": win, "forecast": col}
    for alpha in (0.01, 0.025):
        fails = 0
        for (name, side), f in fcs.items():
            fr = f.between(start_of(name, w[0]), w[1]).frame
            idx = R[name].index
            wd = idx[np.minimum(idx.get_indexer(fr.index) + 1, len(idx) - 1)].weekday
            exog = np.column_stack([fr[tails.col("var", alpha)].to_numpy()] + [wd == k for k in range(1, 5)]).astype(float)
            hit = bt.hits(fr["loss"], fr[tails.col("var", alpha)])
            fails += bt.dq_test(hit, alpha, exog=exog, n_sims=999, seed=SEED)["p"] < 0.05
        out[f"dq_weekday_fails_{tails.col('var', alpha)}"] = int(fails)
    return out


def weekday_rows(fc: dict, R: dict, w: tuple, win: str, col: str) -> list:
    """Mean realised / forecast 1-day variance by the target day's weekday, per market group
    (1 = right on average; the weekday pattern is what the weekday factors are for)."""
    rows = []
    for g, names in GROUPS.items():
        parts = []
        for name in names:
            f = fc[name].loc[start_of(name, w[0]):w[1]].dropna(subset=["y1", col])
            idx = R[name].index
            wd = idx[np.minimum(idx.get_indexer(f.index) + 1, len(idx) - 1)].weekday
            parts.append(pd.DataFrame({"ratio": f["y1"] / f[col], "wd": wd}))
        p = pd.concat(parts)
        for k, v in p.groupby("wd")["ratio"].mean().items():
            rows.append({"window": win, "forecast": col, "group": g, "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri"][k], "ratio": v})
    return rows


def term_table(fc: dict) -> pd.DataFrame:
    """sd of R10 / sqrt(forecast) by vol state (terciles of EWMA vol over its 1-year
    average at the origin), pooled across markets. 1 = right; session 1 found the
    square-root-of-time rule too low in calm states and about right in turbulent ones."""
    rows = []
    for win, (a, b) in WINDOWS.items():
        parts = []
        for name, f in fc.items():
            d = f.loc[start_of(name, a):b]
            ew = d["ewma_1"]
            state = np.sqrt(ew / ew.rolling(252, min_periods=200).mean())
            z = pd.DataFrame({c: np.sqrt(d["y10"] / d[c]) for c in columns(10)})
            z["state"] = state
            parts.append(z)
        p = pd.concat(parts).dropna()
        p["tercile"] = pd.qcut(p["state"], 3, labels=["calm", "middle", "turbulent"])
        for c in columns(10):
            row = {"window": win, "forecast": c, "all": np.sqrt(np.mean(p[c] ** 2))}
            for t, g in p.groupby("tercile", observed=True):
                row[str(t)] = np.sqrt(np.mean(g[c] ** 2))
            rows.append(row)
    return pd.DataFrame(rows)


def diagnostics_tables(diag: pd.DataFrame) -> None:
    """Three compact tables (results files stay summary-sized, under 200 rows)."""
    keys = ["market", "year"]
    gjr = diag[diag.model == "gjr"].dropna(axis=1, how="all").drop(columns=["model", "h", "scale_c"], errors="ignore")
    gjr = gjr.dropna(subset=["omega"]).set_index(keys).add_prefix("gjr_")
    parts = [gjr]
    for h in (1, 10):
        har = diag[diag.model == f"har_{h}"].dropna(axis=1, how="all").drop(columns=["model"]).set_index(keys)
        parts.append(har.add_prefix(f"har{h}_"))
    ivd = diag[(diag.model == "iv") & diag.get("filtered_days", pd.Series(dtype=float)).notna()]
    if len(ivd):
        parts.append(ivd.set_index(keys)[["filtered_days", "fallback_days"]].add_prefix("iv_"))
    pd.concat(parts, axis=1).round(5).to_csv(etr.RESULTS / "p2_scale_params.csv")
    wd = diag[diag.model == "weekday"].pivot_table(index=keys, columns="weekday", values="factor")
    wd.columns = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    wd.round(4).to_csv(etr.RESULTS / "p2_scale_weekday_factors.csv")
    cs = diag[diag.scale_c.notna()].pivot_table(index=["market", "model", "h"], columns="year", values="scale_c")
    cs.round(4).to_csv(etr.RESULTS / "p2_scale_cal_scales.csv")


def figure(fz: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURF, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6"
    d = fz[(fz.window == "2015-18") & (fz.alpha == 0.025) & fz.diff_vs_b3.notna()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURF, gridspec_kw={"width_ratios": [6, 12]})
    for ax, h in zip(axes, (1, 10)):
        s = d[d.horizon == h].iloc[::-1]
        y = np.arange(len(s))
        ax.errorbar(-s.diff_vs_b3, y, xerr=1.96 * s.se, fmt="o", color=BLUE, ms=5, lw=1.5, capsize=0)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks(y, [c.replace("_sqrt10", " (sqrt 10)").replace("_10", " (term)").replace("_1", "") for c in s.forecast])
        ax.set_title(f"{h}-day: FZ0 gain over calibrated EWMA", loc="left", fontsize=10, color=INK)
        ax.grid(True, axis="x", color=GRID, lw=0.6)
        ax.set_facecolor(SURF)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.set_xlabel("pooled FZ0 gain (95% CI); > 0 = better", color=INK2, fontsize=8)
    fig.suptitle("Scale models with a normal shape, 2015-2018 validation (12 market-sides)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(etr.RESULTS / "p2_scale_fz0.png", dpi=150)


def main() -> None:
    R, iv = load()
    fc, diag = scale.walk_forward(R, YEARS, iv=iv)
    pd.concat({n: f for n, f in fc.items()}, names=["market", "date"]).to_parquet(etr.PROCESSED / "scale_forecasts.parquet")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    q = qlike_table(fc)
    q.round(6).to_csv(etr.RESULTS / "p2_scale_qlike.csv", index=False)
    print(q[["window", "horizon", "forecast", "pooled_diff", "pooled_t", "pooled_p"]].round(4).to_string(index=False))
    fz, cal, wd = score_and_calibration(fc, R)
    fz.round(5).to_csv(etr.RESULTS / "p2_scale_fz0.csv", index=False)
    cal.round(4).to_csv(etr.RESULTS / "p2_scale_calibration.csv", index=False)
    wd.round(3).to_csv(etr.RESULTS / "p2_scale_weekday.csv", index=False)
    print(fz.round(4).to_string(index=False))
    print(cal.round(3).to_string(index=False))
    print(wd.round(2).to_string(index=False))
    t = term_table(fc)
    t.round(3).to_csv(etr.RESULTS / "p2_scale_term.csv", index=False)
    print(t.round(3).to_string(index=False))
    diagnostics_tables(diag)
    figure(fz)


if __name__ == "__main__":
    main()
