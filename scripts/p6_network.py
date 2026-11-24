"""Phase 6, second part: the small neural network on top of the learned linear multiplier,
at 1 day on gjr_1|t_pool, following the design committed in PROJECT_LOG.md session 7
before any training. Built because the committed rule fired (the linear model's pooled
FZ0 gain over its starting configuration had one-sided DM p < 0.10 at 1 day).
Data after 2018-12-31 is dropped first.

Compares, on 2015-18 (decisions) and 2012-14: the starting configuration, B3, the
learned linear model and the network (pooled FZ0, Diebold-Mariano, the six pooled
calibration tests), and reports what the network learned: the early-stopping step counts
per refit and the partial dependence of log m on each input in the last refit (2018).

Writes (results/): p6_network.csv, p6_network_params.csv, p6_network_pd.csv,
p6_network.png.
Run:  python scripts/p6_network.py            (about an hour)
      python scripts/p6_network.py --smoke DIR (two years, one small network; to DIR)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import etr
from etr import backtest as bt
from etr import engine, learned, scale
from p6_learned import B3, SEED, WINDOWS, YEARS, compare, load, trim

NAME = "nn_1|t_pool"
LINEAR = "learn_1|t_pool"
BASE = "gjr_1|t_pool"


def build(R, iv, feats, years):
    fc, _, ins = scale.walk_forward(R, years, iv=iv, keep_insample=True)
    cfg = {BASE: engine.one_day(R, fc, ins, years, "gjr_1", "t_pool")[0],
           B3[1]: engine.one_day(R, fc, ins, years, "ewma_1", "normal")[0]}
    cfg[LINEAR], _ = learned.learned_forecasts(R, fc, ins, feats, years, "gjr", "t_pool", h=1)
    print("built", LINEAR, flush=True)
    keep = {}
    cfg[NAME], params = learned.learned_forecasts(R, fc, ins, feats, years, "gjr", "t_pool", h=1,
                                                  model="network", seed=SEED, keep=keep)
    print("built", NAME, flush=True)
    return {c: trim(f) for c, f in cfg.items()}, pd.DataFrame(params), keep


def versus_linear(cfg, windows) -> pd.DataFrame:
    rows = []
    for win, (a, b) in windows.items():
        S = pd.DataFrame({c: engine.pooled_fz0(cfg[c], a, b) for c in (LINEAR, NAME)}).dropna()
        d = bt.diebold_mariano(S[NAME] - S[LINEAR], lags=5)
        rows.append({"window": win, "config": NAME, "gain_vs_linear": -d["mean_diff"], "t_vs_linear": -d["t"],
                     "p_better_than_linear": d["p_less"]})
    return pd.DataFrame(rows)


def partial_dependence(keep, feats, year) -> pd.DataFrame:
    """log m as each input moves over its training range (1st-99th percentile, 21 points;
    0 and 1 for the flags), the others held at their training mean, for the linear model
    and the network of the `year` refit. One row per (input, value), one column per side
    and model."""
    cut = pd.Timestamp(f"{year - 1}-12-31")
    raw = pd.concat([f.loc[:cut] for f in feats.values()]).dropna()
    parts = []
    for side in ("long", "short"):
        rows = []
        k = keep[(year, side)]
        cols, beta, members = k["cols"], k["beta"], k["members"]
        for i, c in enumerate(cols):
            grid = np.array([0.0, 1.0]) if c in ("iv_flag", "monday") else np.linspace(*raw[c].quantile([0.01, 0.99]), 21)
            G = pd.DataFrame(np.tile(raw[cols].mean().to_numpy(), (len(grid), 1)), columns=cols)
            G[c] = grid
            Z = np.zeros((len(grid), len(cols)))
            Z[:, i] = k["std"](G)[c].to_numpy()                      # the refit's own standardisation
            lin = np.clip(np.c_[np.ones(len(Z)), Z] @ beta, -3, 3)
            net = learned.network_eta(members, Z)
            rows += [{"input": c, "value": g, f"{side}_linear": a, f"{side}_network": b} for g, a, b in zip(grid, lin, net)]
        parts.append(pd.DataFrame(rows).set_index(["input", "value"]))
    return pd.concat(parts, axis=1).reset_index()


def figure(pdp: pd.DataFrame, out: Path, year: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    COL = {"long": "#2a78d6", "short": "#eb6834"}
    inputs = list(dict.fromkeys(pdp["input"]))
    fig, axes = plt.subplots(2, 4, figsize=(12, 5.6), facecolor=SURF)
    for ax, c in zip(axes.ravel(), inputs):
        g = pdp[pdp.input == c]
        mk = "o" if len(g) == 2 else None
        for side in ("long", "short"):
            ax.plot(g.value, np.exp(g[f"{side}_linear"]), ls="--", lw=1.2, color=COL[side], marker=mk, ms=4)
            ax.plot(g.value, np.exp(g[f"{side}_network"]), ls="-", lw=2, color=COL[side], marker=mk, ms=5,
                    label=f"{side}: network (solid), linear (dashed)")
        ax.axhline(1, color=INK2, lw=0.8)
        ax.set_title(c.replace("_", " "), loc="left", fontsize=9, color=INK)
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_facecolor(SURF)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("volatility multiplier", fontsize=8, color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle(f"What the {year} refit learned: multiplier as one input moves, the others at their training mean",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out / "p6_network.png", dpi=150)


def main() -> None:
    smoke = "--smoke" in sys.argv
    out = Path(sys.argv[sys.argv.index("--smoke") + 1]) if smoke else etr.RESULTS
    years = [2017, 2018] if smoke else YEARS
    if smoke:
        learned.NN_MEMBERS, learned.NN_MAX_STEPS = 1, 50
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    R, iv, feats = load()
    cfg, params, keep = build(R, iv, feats, years)
    params.round(5).to_csv(out / "p6_network_params.csv", index=False)
    print(params[["year", "side", "n_train", "k_star", "inner_valid_gain", "train_gain"]].round(4).to_string(index=False))
    windows = {"2015-18": ("2018-01-01", "2018-12-31")} if smoke else WINDOWS     # smoke: 2018 under the same label
    table = compare({1: cfg}, windows).merge(versus_linear(cfg, windows), on=["window", "config"], how="left")
    table.round(5).to_csv(out / "p6_network.csv", index=False)
    print(table.round(4).to_string(index=False))
    pdp = partial_dependence(keep, feats, years[-1])
    pdp.round(5).to_csv(out / "p6_network_pd.csv", index=False)
    figure(pdp, out, years[-1])


if __name__ == "__main__":
    main()
