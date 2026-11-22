"""Phase 4: every 10-day configuration on development data, and the pre-registered selection
rule (PROTOCOL.md 4.2, Q3) on the 2015-18 validation window. Data after 2018-12-31 is
dropped before anything is computed. The selection is provisional, as in Phase 3.

Configurations: each scale model x shape, by square-root-of-time (the 1-day engine scaled)
and by term structure (the model's own 10-day forecast, shape fitted to 10-day residuals);
filtered historical simulation along GJR paths (per-market and pooled shock pools);
historical simulation on 10-day returns and RiskMetrics x sqrt(10). B3 at 10 days is
calibrated EWMA-normal scaled by sqrt(10): "ewma_sqrt10|normal".

Scoring uses every daily origin (overlapping windows; Diebold-Mariano with 20 lags, MCS
blocks of 20 days); calibration tests use non-overlapping windows (PROTOCOL Q3).

Writes (results/):
  p4_configs.csv     every configuration: pooled FZ0 and gain over B3 (both windows), MCS
                     p-value, the six pooled calibration tests, complexity, the rule's flags
  p4_vol_state.csv   pooled breach rates at 99% and 97.5% by vol state, selected configurations
  p4_supporting.csv  per-market-side breach rates and Kupiec on non-overlapping windows
  p4_configs.png     pooled FZ0 gain over B3: scale x shape, sqrt(10) vs term structure

Run:  python scripts/p4_ten_day.py
"""
import numpy as np
import pandas as pd

import etr
from etr import backtest as bt
from etr import engine, scale, tails
from etr.markets import MARKETS

DEV_END = "2018-12-31"
YEARS = list(range(2012, 2019))
WINDOWS = {"2015-18": ("2015-01-01", "2018-12-31"), "2012-14": ("2012-01-01", "2014-12-31")}
SEED = 20261024
B3 = "ewma_sqrt10|normal"
H = 10


def load():
    rets = pd.read_parquet(etr.PROCESSED / "returns.parquet")
    rets = rets[rets.index <= DEV_END]
    R = {n: rets[rets.market == n] for n in MARKETS}
    iv = pd.read_parquet(etr.PROCESSED / "implied_vol.parquet").loc[:DEV_END]
    return R, {n: iv[n] for n in iv.columns}


def trim(forecasts: dict) -> dict:
    return {k: f.between(MARKETS[k[0]].liquid_from or "2010-01-01", DEV_END) for k, f in forecasts.items()}


def build(R, iv):
    fc, diag, ins = scale.walk_forward(R, YEARS, iv=iv, keep_insample=True)
    configs, params = {}, []
    for approach, suffix in (("sqrt10", "sqrt10"), ("term", "10")):
        for m in scale.MODELS:
            for shape in engine.SHAPES:
                name = f"{m}_{suffix}|{shape}"
                configs[name], p = engine.ten_day(R, fc, ins, YEARS, m, shape, approach)
                params.extend({"config": name, **d} for d in p)
                print("built", name, flush=True)
    for pooled, shape in ((False, "fhs_mkt"), (True, "fhs_pool")):
        name = f"gjr_paths|{shape}"
        configs[name], info = engine.gjr_paths(R, diag, YEARS, pooled=pooled, seed=SEED)
        params.extend({"config": name, **d} for d in info)
        print("built", name, flush=True)
    for side in tails.SIDES:
        for n, df in R.items():
            configs.setdefault("hs500", {})[(n, side)] = tails.historical_forecast(df, side, H)
            configs.setdefault("riskmetrics", {})[(n, side)] = tails.scale_shape_forecast(df, H * scale.ewma_variance(df), side, H)
    return {c: trim(f) for c, f in configs.items()}, pd.DataFrame(params), fc


def evaluate(configs: dict) -> tuple[pd.DataFrame, str]:
    rows = {}
    chosen = None
    for win, (a, b) in WINDOWS.items():
        scores = pd.DataFrame({c: engine.pooled_fz0(f, a, b) for c, f in configs.items()})
        cal = {c: engine.pooled_calibration(f, a, b, h=H, seed=SEED) for c, f in configs.items()}
        fails = pd.Series({c: v["failures"] for c, v in cal.items()})
        sel = engine.select(scores, fails, block=2 * H, b3=B3, seed=SEED)
        for c in configs:
            dm = bt.diebold_mariano(scores[c] - scores[B3], lags=2 * H) if c != B3 else {"mean_diff": 0.0, "t": np.nan}
            r = rows.setdefault(c, {"config": c, "complexity": engine.complexity(c)})
            r.update({f"{win}_mean_fz0": sel.loc[c, "mean_fz0"], f"{win}_gain_vs_b3": -dm["mean_diff"],
                      f"{win}_t_vs_b3": -dm["t"], f"{win}_mcs_p": sel.loc[c, "mcs_p"], f"{win}_failures": fails[c]})
            if win == "2015-18":
                r.update({k: v for k, v in cal[c].items() if k != "failures"})
                r.update(shortlisted=sel.loc[c, "shortlisted"], no_worse_than_b3=sel.loc[c, "no_worse_than_b3"],
                         passes_screen=sel.loc[c, "passes_screen"], chosen=sel.loc[c, "chosen"])
        if win == "2015-18":
            chosen = sel.index[sel["chosen"]][0]
        print("evaluated", win, flush=True)
    return pd.DataFrame(rows.values()).sort_values("2015-18_mean_fz0"), chosen


def vol_state(configs: dict, names: list[str], fc: dict) -> pd.DataFrame:
    """Pooled breach rates (overlapping windows, descriptive) by vol state at the origin:
    terciles of EWMA vol over its trailing one-year average, pooled across markets. Phase 2
    found 10-day forecasts too low in calm states and slightly high in turbulent ones."""
    a, b = WINDOWS["2015-18"]
    state = {}
    for n, f in fc.items():
        ew = f["ewma_1"]
        state[n] = np.sqrt(ew / ew.rolling(252, min_periods=200).mean())
    allstate = pd.concat(state.values())
    cuts = allstate[(allstate.index >= a) & (allstate.index <= b)].quantile([1 / 3, 2 / 3]).to_numpy()
    rows = []
    for c in names:
        parts = []
        for (n, side), f in configs[c].items():
            fr = f.between(a, b).frame
            s = state[n].reindex(fr.index)
            parts.append(pd.DataFrame({"tercile": np.where(s.isna(), np.nan, np.digitize(s, cuts)),
                                       "b99": fr["loss"] > fr["var99"], "b975": fr["loss"] > fr["var975"]}).dropna())
        p = pd.concat(parts)
        for k, lab in enumerate(("calm", "middle", "turbulent")):
            g = p[p.tercile == k]
            rows.append({"config": c, "state": lab, "n": len(g), "breach_99": g["b99"].mean(), "breach_975": g["b975"].mean()})
    return pd.DataFrame(rows)


def supporting(configs: dict, names: list[str]) -> pd.DataFrame:
    a, b = WINDOWS["2015-18"]
    rows = []
    for c in names:
        for (n, side), f in configs[c].items():
            fr = f.between(a, b).frame
            nov = bt.nonoverlapping(fr, H)
            row = {"config": c, "market": n, "side": side, "windows": len(nov),
                   "rate_var99_overlapping": (fr["loss"] > fr["var99"]).mean(),
                   "rate_var975_overlapping": (fr["loss"] > fr["var975"]).mean()}
            for alpha in (0.01, 0.025):
                v = tails.col("var", alpha)
                k = bt.kupiec(bt.hits(nov["loss"], nov[v]), alpha)
                row[f"breaches_{v}_nonoverlapping"], row[f"kupiec_p_{v}_nonoverlapping"] = k["hits"], k["p"]
            rows.append(row)
    return pd.DataFrame(rows)


def figure(t: pd.DataFrame, chosen: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    cmap.set_bad(SURF)
    g = t[t.config.str.contains(r"\|")].copy()
    g["scale"], g["shape"] = g.config.str.split("|").str[0], g.config.str.split("|").str[1]
    rows = [f"{m}_sqrt10" for m in scale.MODELS] + [f"{m}_10" for m in scale.MODELS] + ["gjr_paths"]
    M = g.pivot(index="scale", columns="shape", values="2015-18_gain_vs_b3").reindex(index=rows, columns=list(engine.SHAPES))
    S = g.pivot(index="scale", columns="shape", values="shortlisted").reindex(index=rows, columns=list(engine.SHAPES))
    lim = np.nanmax(np.abs(M.to_numpy()))
    fig, ax = plt.subplots(figsize=(11, 7.4), facecolor=SURF)
    ax.imshow(np.ma.masked_invalid(M.to_numpy(float)), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i, r in enumerate(M.index):
        for j, c in enumerate(M.columns):
            if np.isnan(M.loc[r, c]):
                continue
            mark = "*" if S.loc[r, c] is True else ""
            weight = "bold" if f"{r}|{c}" == chosen else "normal"
            ax.text(j, i, f"{M.loc[r, c]:+.3f}{mark}", ha="center", va="center", fontsize=7.5, color=INK, fontweight=weight)
    labels = [r.replace("_sqrt10", "  (sqrt 10)").replace("_10", "  (term)").replace("_paths", "  (GJR paths)") for r in M.index]
    ax.set_yticks(range(len(M.index)), labels, fontsize=8, color=INK2)
    ax.set_xticks(range(len(M.columns)), [c.replace("_", " ") for c in M.columns], fontsize=8, color=INK2)
    ax.axhline(len(scale.MODELS) - 0.5, color=INK2, lw=1)
    ax.axhline(2 * len(scale.MODELS) - 0.5, color=INK2, lw=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("10-day pooled FZ0 gain over calibrated EWMA-normal x sqrt(10), 2015-18 validation\n"
                 "(blue = better; * = in the 10% model confidence set; bold = chosen)", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(etr.RESULTS / "p4_configs.png", dpi=150)


def main() -> None:
    R, iv = load()
    configs, params, fc = build(R, iv)
    t, chosen = evaluate(configs)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    t.round(5).to_csv(etr.RESULTS / "p4_configs.csv", index=False)
    print(t.round(4).to_string(index=False))
    print("chosen:", chosen, flush=True)
    # The registered choice and the two alternatives discussed in PROJECT_LOG session 5.
    alternatives = ["hs500", "har_sqrt10|t_mkt", "gjr_sqrt10|t_pool"]
    picks = [B3, chosen, *alternatives, "gjr_10|t_pool", "har_10|t_pool", "comb_iv_10|gpd_pool", "gjr_paths|fhs_pool"]
    picks = list(dict.fromkeys(picks))
    vs = vol_state(configs, picks, fc)
    vs.round(4).to_csv(etr.RESULTS / "p4_vol_state.csv", index=False)
    print(vs.round(4).to_string(index=False))
    sup = supporting(configs, list(dict.fromkeys([chosen, *alternatives, B3])))
    sup.round(4).to_csv(etr.RESULTS / "p4_supporting.csv", index=False)
    print(sup.round(3).to_string(index=False))
    figure(t, chosen)


if __name__ == "__main__":
    main()
