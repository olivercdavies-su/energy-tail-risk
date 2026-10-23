"""Session 1: re-verify the earlier findings this project depends on.

Each check recomputes a published number from energy-vol-targeting with independent code
(not that repo's backtester) and puts it next to the published value:

  1. Rolls: open-interest jumps on the rule's switch days, and the size of the roll
     adjustment (naive vs contract-consistent drift and vol). Full sample: data quality only.
  2. Return dynamics on 2010-18: fat tails, the share of variance in the top 1% of days,
     lag-1 autocorrelation and 5-day variance ratios, GJR asymmetry sign.
  3. Weekly VaR calibration of vol-scaled trend positions (signal at the week's last close,
     trade at the next settlement, 10% target, Gaussian VaR):
       - 2011-18 with raw EWMA (Phase 2 development table);
       - 2019-26 with no vol targeting and with calibrated EWMA (lock-box table).
     2019-26 is touched only to reproduce these already-published rows.

Writes results/p0_reverify_*.csv. Run:  python scripts/p0_reverify.py
"""
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

import etr
from etr.markets import MARKETS
from etr.returns import event_offsets, load_long, market_returns, oi_jump, wide

ANN, TARGET, LEV = 252, 0.10, 5.0
OLD = etr.ROOT.parent / "energy-vol-targeting" / "results" / "tables"


def ewma_var(df: pd.DataFrame, lam: float = 0.94, min_periods: int = 40) -> pd.Series:
    """Daily EWMA variance known at the close of t (same recursion as RiskMetrics)."""
    v = df["ret"] ** 2 / df["span"].where(df["span"] > 0)
    out = v.ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean()
    return out.where(v.notna().cumsum() >= min_periods)


def weekly_periods(df: pd.DataFrame) -> pd.DataFrame:
    """Signal day = last trading day of each week; execution = next trading day; holding
    period = the days after one execution up to and including the next."""
    d = df.index
    last = pd.Series(d, index=d).groupby(d.to_period("W-FRI")).max().values
    sig = d.get_indexer(last)
    sig = sig[sig + 1 < len(d)]
    ex = sig + 1
    hp = pd.DataFrame({"signal": d[sig[:-1]], "first": ex[:-1] + 1, "last": ex[1:]})
    lr = df["ret"].fillna(0.0).to_numpy()
    v = (df["ret"] ** 2 / df["span"].where(df["span"] > 0)).to_numpy()
    hp["logret"] = [lr[f:l + 1].sum() for f, l in zip(hp["first"], hp["last"])]
    hp["rv"] = [ANN * np.nanmean(v[f:l + 1]) for f, l in zip(hp["first"], hp["last"])]
    hp["days"] = hp["last"] - hp["first"] + 1
    hp["start"] = d[ex[:-1]]
    hp["end"] = d[hp["last"].to_numpy()]
    return hp.set_index("signal")


def calibrated(vol_at_signal: pd.Series, hp: pd.DataFrame, year: int) -> float:
    """Point-in-time level calibration (previous project): c = mean(realised / forecast
    variance) over holding periods that ended before `year`."""
    past = hp[hp["end"] < pd.Timestamp(f"{year}-01-01")]
    ratio = (past["rv"] / vol_at_signal.reindex(past.index) ** 2).dropna()
    return float(ratio.mean()) if len(ratio) >= 52 else np.nan


def kupiec(br: np.ndarray, p: float) -> float:
    n, x = len(br), int(br.sum())
    ll_alt = 0.0 if x in (0, n) else (n - x) * np.log(1 - x / n) + x * np.log(x / n)
    ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
    return stats.chi2.sf(-2 * (ll_null - ll_alt), 1)


def sleeve_var(df: pd.DataFrame, vol: pd.Series, start: str, end: str) -> dict:
    """Weekly trend-sleeve VaR backtest: z = holding return / (target x sqrt(days/252))."""
    hp = weekly_periods(df)
    cum = df["ret"].fillna(0.0).rolling(252, min_periods=252).sum()
    signal = np.sign(cum).where(df["ret"].notna().cumsum() >= 252)
    s, v = signal.reindex(hp.index), vol.reindex(hp.index)
    w = np.clip(s * TARGET / v, -LEV, LEV)
    hp = hp[(hp["start"] >= pd.Timestamp(start)) & (hp["end"] <= pd.Timestamp(end))]
    w = w.reindex(hp.index)
    z = (w * np.expm1(hp["logret"]) / (TARGET * np.sqrt(hp["days"] / ANN))).dropna()
    out = {"periods": len(z), "z_sd": z.std()}
    for lev in (95, 99):
        br = (z < -stats.norm.ppf(lev / 100)).to_numpy()
        out[f"breach_rate_{lev}"] = br.mean()
        out[f"kupiec_p_{lev}"] = kupiec(br, 1 - lev / 100)
    return out


def check_var(rets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    old_dev = pd.read_csv(OLD / "phase2_dev_performance.csv")
    old_dev = old_dev[old_dev.window == "development"].set_index(["sizer", "market"])
    old_lb = pd.read_csv(OLD / "phase6_q3_q4_sleeves.csv").set_index(["sizer", "market"])
    rows = []
    for name, df in rets.items():
        ev = np.sqrt(ewma_var(df) * ANN)
        start = "2014-01-01" if name == "TTF" else "2011-01-01"
        rows.append({"window": "2011-18", "sizer": "ewma", "market": name,
                     **sleeve_var(df, ev, start, "2018-12-31"),
                     **{f"published_{k}": old_dev.loc[("ewma", name), k] for k in ("z_sd", "breach_rate_95", "breach_rate_99")}})
        # No targeting: one vol number from all data to 2018-12-31.
        v = df["ret"] ** 2 / df["span"].where(df["span"] > 0)
        const = pd.Series(np.sqrt(v[:"2018-12-31"].mean() * ANN), index=df.index)
        rows.append({"window": "2019-26", "sizer": "constant", "market": name,
                     **sleeve_var(df, const, "2019-01-01", "2026-12-31"),
                     **{f"published_{k}": old_lb.loc[("constant", name), k] for k in ("z_sd", "breach_rate_95", "breach_rate_99")}})
        # Calibrated EWMA: scale fixed at each year-end from past holding periods.
        hp = weekly_periods(df)
        ev_sig = ev.reindex(hp.index)
        cal = ev.copy()
        for y in range(2019, 2027):
            mask = cal.index.year == y
            cal[mask] = ev[mask] * np.sqrt(calibrated(ev_sig, hp, y))
        rows.append({"window": "2019-26", "sizer": "ewma_cal", "market": name,
                     **sleeve_var(df, cal, "2019-01-01", "2026-12-31"),
                     **{f"published_{k}": old_lb.loc[("ewma_cal", name), k] for k in ("z_sd", "breach_rate_95", "breach_rate_99")}})
    return pd.DataFrame(rows).round(4)


def check_rolls(long, rets) -> pd.DataFrame:
    old = pd.read_csv(OLD / "phase1_roll_checks.csv").set_index("market")
    old_imp = pd.read_csv(OLD / "phase1_coverage_and_impact.csv").set_index("market")
    rows = []
    for name, df in rets.items():
        m = MARKETS[name]
        row = {"market": name, "switch_days": int(df.front_switch.sum())}
        if name != "WTI":
            oi = wide(long, "OPEN_INT", [m.ticker(1)])[m.ticker(1)]
            offs = event_offsets(df.index[df.front_switch], oi_jump(oi, signed=name != "EUA"), df.index)
            row.update(oi_checked=int(offs.notna().sum()), oi_exact=int((offs == 0).sum()),
                       published_oi_exact=int(old.loc[name, "oi_exact_match"]))
        for kind, col in (("adjusted", "ret"), ("naive", "ret_naive")):
            r = df[col].dropna()
            row[f"ann_drift_{kind}"] = r.mean() * ANN
            row[f"ann_vol_{kind}"] = r.std() * np.sqrt(ANN)
        row["published_drift_adjusted"] = old_imp.loc[name, "ann_mean_logret_adjusted"]
        row["published_drift_naive"] = old_imp.loc[name, "ann_mean_logret_naive"]
        row["naive_vol_overstatement"] = row["ann_vol_naive"] / row["ann_vol_adjusted"] - 1
        rows.append(row)
    return pd.DataFrame(rows).round(4)


def check_dynamics(rets) -> pd.DataFrame:
    rows = []
    for name, df in rets.items():
        d = df.loc[:"2018-12-31"]
        r = (d["ret"] / np.sqrt(d["span"].where(d["span"] > 0))).dropna()
        r2 = (r ** 2).sort_values(ascending=False)
        row = {"market": name, "excess_kurtosis": stats.kurtosis(r), "skew": stats.skew(r),
               "max_abs_move_sd": (r.abs() / r.std()).max(),
               "var_share_top1pct_days": r2.iloc[:int(np.ceil(0.01 * len(r2)))].sum() / r2.sum()}
        for era, (a, b) in {"2010-13": ("2010", "2013"), "2014-18": ("2014", "2018")}.items():
            x = r.loc[a:b]
            row[f"ac1_{era}"] = x.autocorr(1)
            row[f"vr5_{era}"] = x.rolling(5).sum().var() / (5 * x.var())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = arch_model(100 * r.loc["2012":], mean="Zero", vol="GARCH", p=1, o=1, q=1,
                             dist="t", rescale=False).fit(disp="off")
        row.update(gjr_gamma=fit.params["gamma[1]"], gjr_nu=fit.params["nu"],
                   gjr_persistence=fit.params["alpha[1]"] + fit.params["gamma[1]"] / 2 + fit.params["beta[1]"])
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def main() -> None:
    long = load_long()
    rets = {name: market_returns(long, m) for name, m in MARKETS.items()}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    for tag, fn in (("rolls", lambda: check_rolls(long, rets)), ("dynamics", lambda: check_dynamics(rets)),
                    ("var", lambda: check_var(rets))):
        t = fn()
        t.to_csv(etr.RESULTS / f"p0_reverify_{tag}.csv", index=False)
        print(f"\n== {tag}\n{t.to_string(index=False)}")


if __name__ == "__main__":
    etr.RESULTS.mkdir(exist_ok=True)
    main()
