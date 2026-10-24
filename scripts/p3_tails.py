"""Phase 3: every 1-day configuration (7 scales x 9 shapes, plus the baselines) on development
data, and the pre-registered selection rule (PROTOCOL.md 4.2) on the 2015-18 validation
window. Data after 2018-12-31 is dropped before anything is computed.

This selection is provisional: the learned models and the stress overlay (Phase 6) are
still to be added to the candidate set, and the final choice is made once, at the freeze.

Writes (results/):
  p3_configs.csv       every configuration: pooled FZ0 (both windows) and gain over B3,
                       MCS p-value, the six pooled calibration tests, complexity, the rule's flags
  p3_supporting.csv    per-market-side tests (PROTOCOL Q1 supporting list) for the chosen
                       configuration and the baselines, 2015-18
  p3_shape_params.csv  fitted shapes for the chosen scale, 2015 and 2018 refits
  p3_configs.png       pooled FZ0 gain over calibrated EWMA, scale x shape

Run:  python scripts/p3_tails.py
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
B3 = "ewma_1|normal"


def load():
    rets = pd.read_parquet(etr.PROCESSED / "returns.parquet")
    rets = rets[rets.index <= DEV_END]
    R = {n: rets[rets.market == n] for n in MARKETS}
    iv = pd.read_parquet(etr.PROCESSED / "implied_vol.parquet").loc[:DEV_END]
    return R, {n: iv[n] for n in iv.columns}


def trim(forecasts: dict) -> dict:
    """Evaluate each market only from its liquid date (TTF from 2014)."""
    return {k: f.between(MARKETS[k[0]].liquid_from or "2010-01-01", DEV_END) for k, f in forecasts.items()}


def build(R, iv):
    fc, _, ins = scale.walk_forward(R, YEARS, iv=iv, keep_insample=True)
    configs, params = {}, []
    for m in scale.MODELS:
        for shape in engine.SHAPES:
            name = f"{m}_1|{shape}"
            configs[name], p = engine.one_day(R, fc, ins, YEARS, f"{m}_1", shape)
            params.extend({"config": name, **d} for d in p)
            print("built", name, flush=True)
    for side in tails.SIDES:
        for n, df in R.items():
            configs.setdefault("hs500", {})[(n, side)] = tails.historical_forecast(df, side, 1)
            configs.setdefault("riskmetrics", {})[(n, side)] = tails.scale_shape_forecast(df, scale.ewma_variance(df), side, 1)
    return {c: trim(f) for c, f in configs.items()}, pd.DataFrame(params)


def evaluate(configs: dict) -> tuple[pd.DataFrame, str]:
    rows = {}
    chosen = None
    for win, (a, b) in WINDOWS.items():
        scores = pd.DataFrame({c: engine.pooled_fz0(f, a, b) for c, f in configs.items()})
        cal = {c: engine.pooled_calibration(f, a, b, seed=SEED) for c, f in configs.items()}
        fails = pd.Series({c: v["failures"] for c, v in cal.items()})
        sel = engine.select(scores, fails, block=10, seed=SEED)
        for c in configs:
            dm = bt.diebold_mariano(scores[c] - scores[B3], lags=5) if c != B3 else {"mean_diff": 0.0, "t": np.nan}
            r = rows.setdefault(c, {"config": c, "complexity": engine.complexity(c)})
            r.update({f"{win}_mean_fz0": sel.loc[c, "mean_fz0"], f"{win}_gain_vs_b3": -dm["mean_diff"],
                      f"{win}_t_vs_b3": -dm["t"], f"{win}_mcs_p": sel.loc[c, "mcs_p"], f"{win}_failures": fails[c]})
            if win == "2015-18":
                r.update({k: v for k, v in cal[c].items() if k != "failures"})
                r.update(shortlisted=sel.loc[c, "shortlisted"], passes_screen=sel.loc[c, "passes_screen"],
                         chosen=sel.loc[c, "chosen"])
        if win == "2015-18":
            chosen = sel.index[sel["chosen"]][0]
    t = pd.DataFrame(rows.values()).sort_values("2015-18_mean_fz0")
    return t, chosen


def supporting(configs: dict, names: list[str], R: dict) -> pd.DataFrame:
    a, b = WINDOWS["2015-18"]
    rng = np.random.default_rng(SEED)
    rows = []
    for c in names:
        for (n, side), f in configs[c].items():
            g = f.between(a, b)
            fr = g.frame
            L = fr["loss"].to_numpy()
            idx = R[n].index
            wd = idx[np.minimum(idx.get_indexer(fr.index) + 1, len(idx) - 1)].weekday
            row = {"config": c, "market": n, "side": side, "days": len(fr)}
            for alpha in (0.01, 0.025):
                v = tails.col("var", alpha)
                hit = bt.hits(L, fr[v])
                row[f"rate_{v}"] = hit.mean()
                row[f"kupiec_p_{v}"] = bt.kupiec(hit, alpha)["p"]
                exog = np.column_stack([fr[v].to_numpy()] + [wd == k for k in range(1, 5)]).astype(float)
                row[f"dq_p_{v}"] = bt.dq_test(hit, alpha, exog=exog, seed=SEED)["p"]
            row["christoffersen_cc_p_var99"] = bt.christoffersen(bt.hits(L, fr["var99"]), 0.01, seed=SEED)["p_cc"]
            row["as_p_975"] = bt.acerbi_szekely_z2(L, fr["var975"], fr["es975"], 0.025, g.sample(rng, 1999))["p"]
            row["de_p_975"] = bt.du_escanciano(fr["u"], 0.025)["p_u"]
            row["mf_p_975"] = bt.mcneil_frey(L, fr["var975"], fr["es975"], seed=SEED)["p"]
            tl = bt.traffic_light(bt.hits(L, fr["var99"]), 0.01)
            row.update(tl_yellow=tl["yellow"], tl_red=tl["red"])
            rows.append(row)
    out = pd.DataFrame(rows)
    # Holm across the 12 market-sides within each test, as PROTOCOL Q1 prescribes.
    pcols = [k for k in out.columns if "_p_" in k]
    for c in names:
        sel = out.config == c
        for col in pcols:
            out.loc[sel, f"{col}_holm"] = bt.holm(out.loc[sel, col].to_numpy())
    return out


def figure(t: pd.DataFrame, chosen: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    g = t[t.config.str.contains(r"\|")].copy()
    g["scale"] = g.config.str.split("|").str[0].str.replace("_1", "", regex=False)
    g["shape"] = g.config.str.split("|").str[1]
    M = g.pivot(index="scale", columns="shape", values="2015-18_gain_vs_b3").loc[list(scale.MODELS), list(engine.SHAPES)]
    S = g.pivot(index="scale", columns="shape", values="shortlisted").loc[M.index, M.columns]
    lim = np.nanmax(np.abs(M.to_numpy()))
    fig, ax = plt.subplots(figsize=(11, 4.8), facecolor=SURF)
    ax.imshow(M.to_numpy(), cmap=cmap, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
    for i, r in enumerate(M.index):
        for j, c in enumerate(M.columns):
            mark = "*" if S.loc[r, c] else ""
            weight = "bold" if f"{r}_1|{c}" == chosen else "normal"
            ax.text(j, i, f"{M.loc[r, c]:+.3f}{mark}", ha="center", va="center", fontsize=8, color=INK, fontweight=weight)
    ax.set_xticks(range(len(M.columns)), [c.replace("_", " ") for c in M.columns], fontsize=8, color=INK2)
    ax.set_yticks(range(len(M.index)), M.index, fontsize=8, color=INK2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("1-day pooled FZ0 gain over calibrated EWMA-normal, 2015-18 validation "
                 "(blue = better; * = in the 10% model confidence set; bold = chosen)", loc="left", fontsize=9.5, color=INK)
    fig.tight_layout()
    fig.savefig(etr.RESULTS / "p3_configs.png", dpi=150)


def main() -> None:
    R, iv = load()
    configs, params = build(R, iv)
    t, chosen = evaluate(configs)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    t.round(5).to_csv(etr.RESULTS / "p3_configs.csv", index=False)
    print(t.round(4).to_string(index=False))
    print("chosen:", chosen)
    sup = supporting(configs, [chosen, B3, "hs500"], R)
    sup.round(4).to_csv(etr.RESULTS / "p3_supporting.csv", index=False)
    print(sup.round(3).to_string(index=False))
    sc = chosen.split("|")[0]
    p = params[params.config.str.startswith(sc + "|") & params.year.isin([2015, 2018])].dropna(axis=1, how="all")
    p.round(4).to_csv(etr.RESULTS / "p3_shape_params.csv", index=False)
    figure(t, chosen)


if __name__ == "__main__":
    main()
