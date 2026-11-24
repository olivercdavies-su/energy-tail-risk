"""Phase 6: stress early warning and the learned VaR/ES model on development data
(2012-18; decisions on 2015-18), following the design committed in PROJECT_LOG.md
session 7 before anything was fitted. Data after 2018-12-31 is dropped first.

Steps:
  1. Early warning (PROTOCOL Q5): stress events at 1.5x and 2x, baseline (vol state) and
     full (all inputs) pooled logistic models, Brier skill with block-bootstrap intervals,
     reliability by decile. Rule 4.3 decides whether the overlay may enter.
  2. Learned linear multiplier on each horizon's provisional choice (1 day: gjr_1|t_pool;
     10 days: har_sqrt10|t_mkt), and with the overlay probability as one more input if
     rule 4.3 passed.
  3. Comparison with the starting configuration and B3 (pooled FZ0, Diebold-Mariano, the
     six pooled calibration tests), and the committed rule for building a network.

Writes (results/): p6_early_warning.csv, p6_reliability.csv, p6_learned.csv,
p6_learned_params.csv, p6_reliability.png.
Run:  python scripts/p6_learned.py
"""
import numpy as np
import pandas as pd

import etr
from etr import backtest as bt
from etr import engine, learned, returns, scale, tails
from etr.markets import MARKETS

DEV_END = "2018-12-31"
YEARS = list(range(2012, 2019))
WINDOWS = {"2015-18": ("2015-01-01", "2018-12-31"), "2012-14": ("2012-01-01", "2014-12-31")}
SEED = 20261024
BASE = {1: ("gjr", "t_pool", "sqrt10"), 10: ("har", "t_mkt", "sqrt10")}
B3 = {1: "ewma_1|normal", 10: "ewma_sqrt10|normal"}


def load():
    rets = pd.read_parquet(etr.PROCESSED / "returns.parquet")
    rets = rets[rets.index <= DEV_END]
    R = {n: rets[rets.market == n] for n in MARKETS}
    iv = pd.read_parquet(etr.PROCESSED / "implied_vol.parquet").loc[:DEV_END]
    long = returns.load_long()
    panels = {n: returns.generic_panel(long, m).loc[:DEV_END] for n, m in MARKETS.items()}
    feats = {n: learned.features(R[n], panels[n], iv[n] if n in iv else None) for n in MARKETS}
    return R, {n: iv[n] for n in iv.columns}, feats


def trim(forecasts: dict) -> dict:
    return {k: f.between(MARKETS[k[0]].liquid_from or "2010-01-01", DEV_END) for k, f in forecasts.items()}


def early_warning_tables(R, feats):
    rows, rel, decision, preds_out = [], [], None, None
    for k in (1.5, 2.0):
        ew, preds = learned.early_warning(R, feats, YEARS, k)
        ew = ew[[d >= pd.Timestamp(MARKETS[m].liquid_from or "2010-01-01") for d, m in zip(ew.index, ew.market)]]
        for win, (a, b) in WINDOWS.items():
            w = ew[(ew.index >= a) & (ew.index <= b)]
            piv = {c: w.pivot_table(index=w.index, columns="market", values=c) for c in ("y", "p_base", "p_full")}
            s = learned.brier_skill(piv["p_full"], piv["p_base"], piv["y"], seed=SEED)
            clim = pd.DataFrame(np.full(piv["y"].shape, np.nan), index=piv["y"].index, columns=piv["y"].columns)
            for m in clim.columns:                                     # base rate from training years only
                clim[m] = ew[(ew.market == m) & (ew.index < a)]["y"].mean() if (ew.index < a).any() else w[w.market == m]["y"].mean()
            vs_clim = learned.brier_skill(piv["p_base"], clim, piv["y"], seed=SEED)
            rows.append({"event": f"{k}x", "window": win, "dates": s["dates"], "event_rate": float(np.nanmean(piv["y"].to_numpy())),
                         "brier_base": s["brier_base"], "brier_full": s["brier_model"], "bss_full_vs_base": s["bss"],
                         "ci90_low": s["ci90"][0], "ci90_high": s["ci90"][1], "ci95_low": s["ci95"][0], "ci95_high": s["ci95"][1],
                         "bss_base_vs_base_rate": vs_clim["bss"]})
            if win == "2015-18":
                for model in ("p_base", "p_full"):
                    r = learned.reliability(w[model], w["y"]).assign(event=f"{k}x", model=model)
                    rel.append(r.reset_index())
                if k == 1.5:
                    decision = s["bss"] > 0 and s["ci90"][0] > 0
                    preds_out = preds
    return pd.DataFrame(rows), pd.concat(rel), decision, preds_out


def build_learned(R, iv, feats, overlay_preds):
    fc, _, ins = scale.walk_forward(R, YEARS, iv=iv, keep_insample=True)
    out, params = {1: {}, 10: {}}, []
    for h, (m, shape, approach) in BASE.items():
        base_name = f"{m}_1|{shape}" if h == 1 else f"{m}_{approach}|{shape}"
        if h == 1:
            out[h][base_name] = engine.one_day(R, fc, ins, YEARS, f"{m}_1", shape)[0]
            out[h][B3[h]] = engine.one_day(R, fc, ins, YEARS, "ewma_1", "normal")[0]
        else:
            out[h][base_name] = engine.ten_day(R, fc, ins, YEARS, m, shape, approach)[0]
            out[h][B3[h]] = engine.ten_day(R, fc, ins, YEARS, "ewma", "normal", "sqrt10")[0]
        name = f"learn_1|{shape}" if h == 1 else f"learn_{approach}|{shape}"
        out[h][name], p = learned.learned_forecasts(R, fc, ins, feats, YEARS, m, shape, h, approach)
        params.extend({"horizon": h, "config": name, **d} for d in p)
        print("built", name, flush=True)
        if overlay_preds is not None:
            extra = lambda n, y: overlay_preds[y](feats[n].dropna())                    # noqa: E731
            name = f"learn_ew_1|{shape}" if h == 1 else f"learn_ew_{approach}|{shape}"
            out[h][name], p = learned.learned_forecasts(R, fc, ins, feats, YEARS, m, shape, h, approach, extra=extra)
            params.extend({"horizon": h, "config": name, **d} for d in p)
            print("built", name, flush=True)
    return {h: {c: trim(f) for c, f in cfg.items()} for h, cfg in out.items()}, pd.DataFrame(params)


def compare(configs: dict) -> pd.DataFrame:
    rows = []
    for h, cfg in configs.items():
        m, shape, approach = BASE[h]
        base_name = f"{m}_1|{shape}" if h == 1 else f"{m}_{approach}|{shape}"
        for win, (a, b) in WINDOWS.items():
            S = pd.DataFrame({c: engine.pooled_fz0(f, a, b) for c, f in cfg.items()}).dropna()
            for c in cfg:
                row = {"horizon": h, "window": win, "config": c, "mean_fz0": S[c].mean()}
                for ref, tag in ((base_name, "base"), (B3[h], "b3")):
                    if c != ref:
                        d = bt.diebold_mariano(S[c] - S[ref], lags=5 if h == 1 else 20)
                        row.update({f"gain_vs_{tag}": -d["mean_diff"], f"t_vs_{tag}": -d["t"], f"p_better_than_{tag}": d["p_less"]})
                if win == "2015-18":
                    cal = engine.pooled_calibration(cfg[c], a, b, h=h, seed=SEED)
                    row.update({k: v for k, v in cal.items()})
                rows.append(row)
    return pd.DataFrame(rows)


def figure(rel: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    COL = {"p_base": "#2a78d6", "p_full": "#eb6834"}
    LAB = {"p_base": "baseline (vol state)", "p_full": "full (all inputs)"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), facecolor=SURF)
    for ax, ev in zip(axes, ("1.5x", "2.0x")):
        g = rel[rel.event == ev]
        top = max(g.mean_forecast.max(), g.observed.max()) * 1.1
        ax.plot([0, top], [0, top], color=INK2, lw=1, ls="--")
        for model in ("p_base", "p_full"):
            s = g[g.model == model]
            ax.plot(s.mean_forecast, s.observed, "o-", color=COL[model], lw=2, ms=6, label=LAB[model])
        ax.set_title(f"vol reaches {ev} within 21 days", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("forecast probability (decile mean)", fontsize=8, color=INK2)
        ax.set_ylabel("observed frequency", fontsize=8, color=INK2)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_facecolor(SURF)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Stress early warning: reliability, 2015-2018 validation (six markets pooled)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(etr.RESULTS / "p6_reliability.png", dpi=150)


def main() -> None:
    R, iv, feats = load()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    ew, rel, passes, preds = early_warning_tables(R, feats)
    ew.round(5).to_csv(etr.RESULTS / "p6_early_warning.csv", index=False)
    rel.round(5).to_csv(etr.RESULTS / "p6_reliability.csv", index=False)
    print(ew.round(4).to_string(index=False))
    print("rule 4.3 (overlay may enter):", passes, flush=True)
    figure(rel)
    configs, params = build_learned(R, iv, feats, preds if passes else None)
    params.round(5).to_csv(etr.RESULTS / "p6_learned_params.csv", index=False)
    cmp_ = compare(configs)
    cmp_.round(5).to_csv(etr.RESULTS / "p6_learned.csv", index=False)
    print(cmp_.round(4).to_string(index=False))
    v = cmp_[(cmp_.window == "2015-18") & cmp_.config.str.startswith("learn_") & ~cmp_.config.str.contains("_ew_")]
    helps = bool(((v["gain_vs_base"] > 0) & (v["p_better_than_base"] < 0.10)).any())
    print("linear helps (committed rule for building a network):", helps)


if __name__ == "__main__":
    main()
