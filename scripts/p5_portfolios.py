"""Phase 5: portfolios on development data (2012-18; choices on 2015-18).

The registered books P1-P5 (PROTOCOL.md section 2) at 1 and 10 days, by:
  - the single-market engine on each book's own return history (every scale x shape;
    implied vol does not exist for a book, so the iv scales are left out);
  - joint filtered historical simulation (each market's scale forecast x residual vectors
    resampled by date), for every market scale; at 10 days with the square-root-of-time or
    the term-structure variance spread evenly over the days;
  - GJR path simulation on the book (10 days);
  - the baselines on the book: historical simulation, RiskMetrics, calibrated EWMA (B3).
The selection rule (PROTOCOL 4.2 with Amendment A1) is applied over P1-P5. The 50 random
books (P6) are then forecast with each horizon's chosen method. Data after 2018-12-31 is
dropped before anything is computed. Books holding TTF are evaluated from 2014.

Writes (results/):
  p5_configs_1d.csv, p5_configs_10d.csv   every configuration and the rule's flags
  p5_supporting.csv     per-book tests (PROTOCOL Q1 list) for the 1-day choice, B3 and HS
  p5_random_books.csv   P6: breach rates and Kupiec for the 50 random books, chosen methods
  p5_dependence.csv     correlation of standardised residuals by year (TTF-NBP, Brent-WTI)
  p5_configs.png        1-day pooled FZ0 gain over B3 by method

Run:  python scripts/p5_portfolios.py
"""
import numpy as np
import pandas as pd
from scipy import stats

import etr
from etr import backtest as bt
from etr import engine, scale, tails
from etr import portfolio as pf
from etr.markets import MARKETS

DEV_END = "2018-12-31"
YEARS = list(range(2012, 2019))
WINDOWS = {"2015-18": ("2015-01-01", "2018-12-31"), "2012-14": ("2012-01-01", "2014-12-31")}
SEED = 20261024
BOOKS = ["P1", "P2", "P3", "P4", "P5"]
BOOK_SCALES = ["ewma", "ewma_wd", "gjr", "har", "comb"]          # iv and comb_iv collapse to these for a book
B3 = {1: "ewma_1|normal", 10: "ewma_sqrt10|normal"}
LONG = ("long",)          # a book is one position: its "short side" would be another book (P1 short = P2)


def load():
    rets = pd.read_parquet(etr.PROCESSED / "returns.parquet")
    rets = rets[rets.index <= DEV_END]
    R = {n: rets[rets.market == n] for n in pf.ORDER}
    iv = pd.read_parquet(etr.PROCESSED / "implied_vol.parquet").loc[:DEV_END]
    return R, {n: iv[n] for n in iv.columns}


def book_start(w: pd.Series) -> str:
    return max(MARKETS[m].liquid_from or "2010-01-01" for m in w.index[w != 0])


def trim(forecasts: dict, W: dict) -> dict:
    return {k: f.between(book_start(W[k[0]]), DEV_END) for k, f in forecasts.items()}


def build(R, iv, W):
    U, open_ = pf.union_returns(R)
    fc, _, ins = scale.walk_forward(R, YEARS, iv=iv, keep_insample=True)          # market level
    Rb = {b: pf.book_return(U, W[b], open_) for b in BOOKS}
    fcb, diagb, insb = scale.walk_forward(Rb, YEARS, keep_insample=True)          # book level
    five = {b: W[b] for b in BOOKS}
    out = {1: {}, 10: {}}
    for m in BOOK_SCALES:
        for shape in engine.SHAPES:
            out[1][f"{m}_1|{shape}"] = engine.one_day(Rb, fcb, insb, YEARS, f"{m}_1", shape, sides=LONG)[0]
            out[10][f"{m}_sqrt10|{shape}"] = engine.ten_day(Rb, fcb, insb, YEARS, m, shape, "sqrt10", sides=LONG)[0]
            out[10][f"{m}_10|{shape}"] = engine.ten_day(Rb, fcb, insb, YEARS, m, shape, "term", sides=LONG)[0]
        print("built univariate", m, flush=True)
    for m in scale.MODELS:
        out[1][f"{m}_1|joint_fhs"] = pf.joint_fhs(R, fc, ins, YEARS, m, five, h=1, keep_draws=False)
        for v in ("sqrt10", "10"):
            out[10][f"{m}_{v}|joint_fhs"] = pf.joint_fhs(R, fc, ins, YEARS, m, five, h=10, scale_col=f"{m}_{v}", keep_draws=False)
        print("built joint FHS", m, flush=True)
    for pooled, shape in ((False, "fhs_mkt"), (True, "fhs_pool")):
        out[10][f"gjr_paths|{shape}"] = engine.gjr_paths(Rb, diagb, YEARS, pooled=pooled, seed=SEED, keep_draws=False, sides=LONG)[0]
    for h in (1, 10):
        for b in BOOKS:
            out[h].setdefault("hs500", {})[(b, "long")] = tails.historical_forecast(Rb[b], "long", h)
            out[h].setdefault("riskmetrics", {})[(b, "long")] = tails.scale_shape_forecast(Rb[b], h * scale.ewma_variance(Rb[b]), "long", h)
    return {h: {c: trim(f, W) for c, f in cfg.items()} for h, cfg in out.items()}, (fc, ins, fcb, insb, Rb)


def evaluate(configs: dict, h: int) -> tuple[pd.DataFrame, str]:
    rows = {}
    chosen = None
    for win, (a, b) in WINDOWS.items():
        scores = pd.DataFrame({c: engine.pooled_fz0(f, a, b) for c, f in configs.items()})
        cal = {c: engine.pooled_calibration(f, a, b, h=h, seed=SEED) for c, f in configs.items()}
        fails = pd.Series({c: v["failures"] for c, v in cal.items()})
        sel = engine.select(scores, fails, block=10 if h == 1 else 20, b3=B3[h], seed=SEED)
        for c in configs:
            dm = bt.diebold_mariano(scores[c] - scores[B3[h]], lags=5 if h == 1 else 20) if c != B3[h] else {"mean_diff": 0.0, "t": np.nan}
            r = rows.setdefault(c, {"config": c, "complexity": engine.complexity(c)})
            r.update({f"{win}_mean_fz0": sel.loc[c, "mean_fz0"], f"{win}_gain_vs_b3": -dm["mean_diff"],
                      f"{win}_t_vs_b3": -dm["t"], f"{win}_mcs_p": sel.loc[c, "mcs_p"], f"{win}_failures": fails[c]})
            if win == "2015-18":
                r.update({k: v for k, v in cal[c].items() if k != "failures"})
                r.update(duplicate_of=sel.loc[c, "duplicate_of"], shortlisted=sel.loc[c, "shortlisted"],
                         no_worse_than_b3=sel.loc[c, "no_worse_than_b3"], passes_screen=sel.loc[c, "passes_screen"],
                         chosen=sel.loc[c, "chosen"])
        if win == "2015-18":
            chosen = sel.index[sel["chosen"]][0]
        print(f"evaluated {h}-day {win}", flush=True)
    return pd.DataFrame(rows.values()).sort_values("2015-18_mean_fz0"), chosen


def supporting(configs: dict, names: list[str], Rb: dict) -> pd.DataFrame:
    a, b = WINDOWS["2015-18"]
    rng = np.random.default_rng(SEED)
    rows = []
    for c in names:
        for (bk, side), f in configs[c].items():
            g = f.between(a, b)
            fr = g.frame
            L = fr["loss"].to_numpy()
            idx = Rb[bk].index
            wd = idx[np.minimum(idx.get_indexer(fr.index) + 1, len(idx) - 1)].weekday
            row = {"config": c, "book": bk, "days": len(fr)}
            for alpha in (0.01, 0.025):
                v = tails.col("var", alpha)
                hit = bt.hits(L, fr[v])
                row[f"rate_{v}"], row[f"kupiec_p_{v}"] = hit.mean(), bt.kupiec(hit, alpha)["p"]
                exog = np.column_stack([fr[v].to_numpy()] + [wd == k for k in range(1, 5)]).astype(float)
                row[f"dq_p_{v}"] = bt.dq_test(hit, alpha, exog=exog, seed=SEED)["p"]
            row["christoffersen_cc_p_var99"] = bt.christoffersen(bt.hits(L, fr["var99"]), 0.01, seed=SEED)["p_cc"]
            try:
                row["as_p_975"] = bt.acerbi_szekely_z2(L, fr["var975"], fr["es975"], 0.025, g.sample(rng, 1999))["p"]
            except RuntimeError:
                row["as_p_975"] = np.nan
            row["de_p_975"] = bt.du_escanciano(fr["u"], 0.025)["p_u"]
            row["mf_p_975"] = bt.mcneil_frey(L, fr["var975"], fr["es975"], seed=SEED)["p"]
            row["tl_red"] = bt.traffic_light(bt.hits(L, fr["var99"]), 0.01)["red"]
            rows.append(row)
    return pd.DataFrame(rows)


def random_books(R, iv, W, chosen: dict, cache) -> pd.DataFrame:
    """P6: the 50 random books forecast by each horizon's chosen method; per book, the 99%
    and 97.5% breach rates and exact Kupiec p-values (1-day on daily origins, 10-day on
    non-overlapping windows), 2015-18."""
    fc, ins, *_ = cache
    U, open_ = pf.union_returns(R)
    rand = {k: w for k, w in W.items() if k.startswith("R")}
    a, b = WINDOWS["2015-18"]
    rows = []
    for h, name in chosen.items():
        if name in ("hs500", "riskmetrics"):
            Rr = {k: pf.book_return(U, w, open_) for k, w in rand.items()}
            if name == "hs500":
                f = {(k, "long"): tails.historical_forecast(df, "long", h) for k, df in Rr.items()}
            else:
                f = {(k, "long"): tails.scale_shape_forecast(df, h * scale.ewma_variance(df), "long", h) for k, df in Rr.items()}
            scale_part, shape = name, None
        else:
            scale_part, shape = name.split("|")
        if shape is None:
            pass
        elif shape == "joint_fhs":
            m = scale_part.rsplit("_", 1)[0]
            f = pf.joint_fhs(R, fc, ins, YEARS, m, rand, h=h, scale_col=scale_part, keep_draws=False)
        else:
            Rr = {k: pf.book_return(U, w, open_) for k, w in rand.items()}
            fcr, diagr, insr = scale.walk_forward(Rr, YEARS, keep_insample=True)
            m, v = scale_part.rsplit("_", 1)
            if h == 1:
                f = engine.one_day(Rr, fcr, insr, YEARS, scale_part, shape, sides=LONG)[0]
            elif m == "gjr" and v == "paths":
                f = engine.gjr_paths(Rr, diagr, YEARS, pooled=shape == "fhs_pool", seed=SEED, keep_draws=False, sides=LONG)[0]
            else:
                f = engine.ten_day(Rr, fcr, insr, YEARS, m, shape, "sqrt10" if v == "sqrt10" else "term", sides=LONG)[0]
        for (bk, _), g in f.items():
            fr = g.between(max(a, book_start(W[bk])), b).frame
            ev = bt.nonoverlapping(fr, h) if h > 1 else fr
            row = {"horizon": h, "config": name, "book": bk, "obs": len(ev)}
            for alpha in (0.01, 0.025):
                v = tails.col("var", alpha)
                k = bt.kupiec(bt.hits(ev["loss"], ev[v]), alpha)
                row[f"rate_{v}"], row[f"kupiec_p_{v}"] = k["rate"], k["p"]
            rows.append(row)
    return pd.DataFrame(rows)


def dependence(R, cache) -> pd.DataFrame:
    """Correlation of the two legs' standardised daily residuals (calibrated EWMA), by year:
    is the dependence joint filtered HS resamples from history stable?"""
    rows = []
    for a, b in (("TTF", "NBP"), ("BRENT", "WTI")):
        z = {}
        for n in (a, b):
            df = R[n]
            v = scale.calibrated_ewma_variance(df)
            z[n] = df["ret"] / np.sqrt(v.shift(1) * df["span"].where(df["span"] > 0))
        Z = pd.DataFrame(z).dropna()
        for y, g in Z.groupby(Z.index.year):
            if y >= 2011:
                rows.append({"pair": f"{a}-{b}", "year": y, "days": len(g), "corr": g[a].corr(g[b]),
                             "spread_sd_ratio": (g[a] - g[b]).std() / np.sqrt(2)})
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
    uni = g[g["shape"] != "joint_fhs"].pivot(index="scale", columns="shape", values="2015-18_gain_vs_b3")
    uni = uni.reindex(index=[f"{m}_1" for m in BOOK_SCALES], columns=list(engine.SHAPES))
    jt = g[g["shape"] == "joint_fhs"].set_index("scale")["2015-18_gain_vs_b3"].reindex([f"{m}_1" for m in scale.MODELS])
    sl = t.set_index("config")["shortlisted"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURF, gridspec_kw={"width_ratios": [9, 1.6]})
    lim = np.nanmax(np.abs(np.r_[uni.to_numpy().ravel(), jt.to_numpy()]))
    norm = TwoSlopeNorm(0, -lim, lim)
    for ax, M, xl in ((axes[0], uni, [c.replace("_", " ") for c in uni.columns]), (axes[1], jt.to_frame(), ["joint FHS"])):
        ax.imshow(np.ma.masked_invalid(M.to_numpy(float)), cmap=cmap, norm=norm, aspect="auto")
        for i, r in enumerate(M.index):
            for j, c in enumerate(M.columns):
                val = M.iloc[i, j]
                if np.isnan(val):
                    continue
                name = f"{r}|{c}" if M is uni else f"{r}|joint_fhs"
                ax.text(j, i, f"{val:+.3f}{'*' if sl.get(name, False) else ''}", ha="center", va="center", fontsize=7.5,
                        color=INK, fontweight="bold" if name == chosen else "normal")
        ax.set_yticks(range(len(M.index)), [r.replace("_1", "") for r in M.index], fontsize=8, color=INK2)
        ax.set_xticks(range(len(xl)), xl, fontsize=8, color=INK2)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_title("Books P1-P5, 1-day pooled FZ0 gain over calibrated EWMA-normal on each book, 2015-18\n"
                      "left: the engine on the book's own returns; right: joint filtered HS with each market's scale "
                      "(* = in the 10% MCS; bold = chosen)", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(etr.RESULTS / "p5_configs.png", dpi=150)


def main() -> None:
    R, iv = load()
    U, open_ = pf.union_returns(R)
    W = pf.books(U, open_)
    configs, cache = build(R, iv, W)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    chosen = {}
    for h in (1, 10):
        t, chosen[h] = evaluate(configs[h], h)
        t.round(5).to_csv(etr.RESULTS / f"p5_configs_{h}d.csv", index=False)
        print(t.round(4).head(25).to_string(index=False))
        print(f"chosen {h}-day:", chosen[h], flush=True)
        if h == 1:
            figure(t, chosen[h])
    # Supporting tests at 1 day; a joint-FHS choice is rebuilt with its paths for the AS test.
    fc, ins, fcb, insb, Rb = cache
    sup_cfg = {c: configs[1][c] for c in (chosen[1], B3[1], "hs500")}
    if chosen[1].endswith("joint_fhs"):
        m = chosen[1].split("|")[0].rsplit("_", 1)[0]
        sup_cfg[chosen[1]] = trim(pf.joint_fhs(R, fc, ins, YEARS, m, {b: W[b] for b in BOOKS}, h=1), W)
    sup = supporting(sup_cfg, list(sup_cfg), Rb)
    sup.round(4).to_csv(etr.RESULTS / "p5_supporting.csv", index=False)
    print(sup.round(3).to_string(index=False))
    rb = random_books(R, iv, W, chosen, cache)
    rb.round(4).to_csv(etr.RESULTS / "p5_random_books.csv", index=False)
    for h, g in rb.groupby("horizon"):
        n = len(g)
        fails = int((g["kupiec_p_var99"] < 0.05).sum())
        print(f"P6 {h}-day: {fails} of {n} books fail Kupiec 99% at 5%; binomial 95% range under a correct model:",
              stats.binom.ppf([0.025, 0.975], n, 0.043))
    dep = dependence(R, cache)
    dep.round(4).to_csv(etr.RESULTS / "p5_dependence.csv", index=False)
    print(dep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
