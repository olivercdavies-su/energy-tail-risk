"""Session 1: explore tail behaviour, on development data only (2011-2018; TTF 2014-2018).

2019-2026 is deliberately not used here, so it stays as clean as possible for tail
questions (see PROJECT_LOG.md, holdout design).

Everything is in the units a VaR engine works in: daily returns standardised by the
volatility forecast known the evening before,
    z_t = r_t / sqrt(c * sigma2_{t-1} * span_t),
with sigma2 the RiskMetrics EWMA (lambda 0.94) and c a point-in-time level calibration
(mean z^2 over all earlier years). If the scale were perfect, z would have unit variance
and its quantiles would describe the shape of the conditional distribution.

Tables (results/p0_tails_*.csv):
  shape      per market and tail: Gaussian-VaR breach rates, empirical quantiles and ES,
             GPD tail index of z, kurtosis of raw returns vs z
  state      pooled breach rates by vol state, weekday/report day, held tenor, season
  horizon    10-day returns: variance ratio with robust SE, sd and tails of z10, rolls
  dependence co-exceedances between markets vs a Gaussian copula with the same correlation
  stress     base rate and number of episodes of "vol doubles within a month"
  iv         implied vol vs EWMA as the scale (HH, Brent, WTI)
  power      expected breaches and Kupiec power for the sample sizes this project has
Figure: results/p0_tail_curves.png

Run:  python scripts/p0_tails.py
"""
import numpy as np
import pandas as pd
from scipy import stats

import etr
from etr.markets import MARKETS

ANN = 252
DEV_START, DEV_END = "2011-01-01", "2018-12-31"
START = {"TTF": "2014-01-01"}      # TTF futures were illiquid, settlements smoothed, before 2014
REPORT_DAY = {"TTF": 0, "NBP": 0, "HH": 3, "BRENT": 2, "WTI": 2, "EUA": None}  # Mon weather, Thu EIA gas, Wed EIA oil
LEVELS = (0.95, 0.99)


def standardise(df: pd.DataFrame, lam: float = 0.94) -> pd.DataFrame:
    """z_t with the EWMA variance of t-1 and a point-in-time calibration scale per year."""
    span = df["span"].where(df["span"] > 0)
    v = df["ret"] ** 2 / span
    s2 = v.ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean().where(v.notna().cumsum() >= 40)
    s2_prev = s2.shift(1)
    z_raw = df["ret"] / np.sqrt(s2_prev * span)
    c = pd.Series(np.nan, index=df.index)
    for y in range(df.index.year.min() + 1, df.index.year.max() + 1):
        past = z_raw[z_raw.index.year < y].dropna()
        c[df.index.year == y] = (past ** 2).mean() if len(past) > 120 else np.nan
    out = pd.DataFrame({"ret": df["ret"], "span": span, "z_raw": z_raw, "c": c,
                        "z": z_raw / np.sqrt(c), "s2_prev": s2_prev * c, "s2_raw_prev": s2_prev, "tenor": df["tenor"],
                        "roll": df["roll"]})
    # Vol state: today's EWMA vol relative to its trailing one-year average (known at t-1).
    out["vol_ratio"] = np.sqrt(s2_prev / s2_prev.rolling(252, min_periods=200).mean())
    return out


def dev(name: str, x: pd.DataFrame) -> pd.DataFrame:
    return x.loc[START.get(name, DEV_START):DEV_END]


def gpd_tail(x: np.ndarray, q: float = 0.95) -> tuple[float, float, int]:
    """Peaks over threshold: fit a generalised Pareto to the exceedances of x over its q
    quantile (x = losses, positive = bad). Returns (xi, approx SE, n exceedances).
    xi > 0 means a power-law tail (t-like, xi = 1/nu); xi = 0 is exponential (normal-ish
    tails decay faster still)."""
    u = np.quantile(x, q)
    exc = x[x > u] - u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)
    se = (1 + xi) / np.sqrt(len(exc))          # asymptotic SE of the MLE (Smith 1987)
    return xi, se, len(exc)


def shape_table(Z: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, x in Z.items():
        d = dev(name, x)
        z = d["z"].dropna().to_numpy()
        r = (d["ret"] / np.sqrt(d["span"])).dropna().to_numpy()
        for tail, loss in (("left (long loses)", -z), ("right (short loses)", z)):
            row = {"market": name, "tail": tail, "n": len(z), "sd_z": z.std(),
                   "kurt_raw": stats.kurtosis(r), "kurt_z": stats.kurtosis(z)}
            for lev in LEVELS:
                row[f"breach_gauss_{lev:.0%}"] = np.mean(loss > stats.norm.ppf(lev))
                row[f"breach_raw_gauss_{lev:.0%}"] = np.mean(
                    (-d["z_raw"].dropna().to_numpy() if tail.startswith("left") else d["z_raw"].dropna().to_numpy())
                    > stats.norm.ppf(lev))
            row["q99_emp"] = np.quantile(loss, 0.99)
            row["es975_emp"] = loss[loss >= np.quantile(loss, 0.975)].mean()
            row["es975_over_q99"] = row["es975_emp"] / row["q99_emp"]
            row["max_loss_z"] = loss.max()
            row["xi"], row["xi_se"], row["n_exc"] = gpd_tail(loss)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["normal"] = {"q99": stats.norm.ppf(0.99), "es975": stats.norm.expect(lambda t: t, lb=stats.norm.ppf(0.975)) / 0.025}
    return out


def state_table(Z: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Pooled across markets: two-sided 99% Gaussian breach rate (|z| > 2.576, nominal 1%)
    and mean z^2 (nominal 1) by state. Each market weighted by its days."""
    frames = []
    for name, x in Z.items():
        d = dev(name, x).dropna(subset=["z"]).copy()
        d["market"] = name
        d["report_day"] = (d.index.weekday == REPORT_DAY[name]) if REPORT_DAY[name] is not None else False
        d["winter"] = d.index.month.isin([10, 11, 12, 1, 2, 3])
        d["after_gap"] = d["span"] > 1
        frames.append(d)
    p = pd.concat(frames)
    p["vol_state"] = pd.qcut(p["vol_ratio"], 3, labels=["low vol vs past year", "middle", "high vol vs past year"])
    p["breach"] = p["z"].abs() > stats.norm.ppf(0.995)
    rows = []

    def add(group: str, key, sub: pd.DataFrame, market: str = "all"):
        n, k = len(sub), int(sub["breach"].sum())
        rows.append({"grouping": group, "state": str(key), "market": market, "n": n, "breaches": k,
                     "breach_rate": k / n if n else np.nan, "mean_z2": (sub["z"] ** 2).mean(),
                     "p_vs_1pct": stats.binomtest(k, n, 0.01).pvalue if n else np.nan})

    for key, sub in p.groupby("vol_state", observed=True):
        add("vol state (EWMA / 1y avg, terciles)", key, sub)
    for key, sub in p.groupby("report_day"):
        add("scheduled report / weather day", key, sub)
    for name in MARKETS:
        sub = p[p.market == name]
        for wd, s in sub.groupby(sub.index.weekday):
            add("weekday", ["Mon", "Tue", "Wed", "Thu", "Fri"][wd], s, name)
    for key, sub in p.groupby("tenor"):
        add("held contract", f"M{key}", sub)
    gas = p[p.market.isin(["TTF", "NBP", "HH"])]
    for key, sub in gas.groupby("winter"):
        add("gas season", "Oct-Mar" if key else "Apr-Sep", sub, "gas")
    return pd.DataFrame(rows)


def lo_mackinlay(z: np.ndarray, q: int) -> tuple[float, float]:
    """Variance ratio VR(q) from overlapping q-sums, with the heteroskedasticity-robust
    standard error of Lo & MacKinlay (1988). VR < 1 = mean reversion: sqrt(q) scaling of
    the daily risk overstates q-day risk."""
    z = z - z.mean()
    n = len(z)
    s = np.convolve(z, np.ones(q), "valid")
    vr = (s ** 2).mean() / (q * (z ** 2).mean())
    d = z ** 2
    theta = 0.0
    for j in range(1, q):
        delta = n * np.sum(d[j:] * d[:-j]) / np.sum(d) ** 2
        theta += (2 * (q - j) / q) ** 2 * delta
    return vr, np.sqrt(theta / n)


def horizon_table(Z: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """10-day horizon. z10 = 10-day log return / sqrt(10 x calibrated EWMA variance at t):
    exactly what a sqrt(10) x one-day VaR assumes has unit variance."""
    rows = []
    for name, x in Z.items():
        d = dev(name, x)
        r = x["ret"].fillna(0.0)
        r10 = r[::-1].rolling(10).sum()[::-1].shift(-1)          # returns t+1..t+10
        roll10 = x["roll"].astype(float)[::-1].rolling(10).max()[::-1].shift(-1)  # roll at a close in t+1..t+10
        s2 = x["s2_prev"].shift(-1)                               # forecast known at close of t
        origins = d.index[:-10]                                   # windows end inside the dev period
        z10 = (r10 / np.sqrt(10 * s2)).loc[origins].dropna()
        roll10 = roll10.reindex(z10.index)
        zd = d["z"].dropna().to_numpy()
        raw = (d["ret"] / np.sqrt(d["span"])).dropna().to_numpy()
        row = {"market": name, "n_days": len(zd)}
        for q in (5, 10):
            row[f"vr{q}_z"], row[f"vr{q}_se"] = lo_mackinlay(zd, q)
        row["vr10_raw"], row["vr10_raw_se"] = lo_mackinlay(raw, 10)
        row["ac1_raw"] = pd.Series(raw).autocorr(1)
        row["ac1_z"] = pd.Series(zd).autocorr(1)
        row["sd_z10"] = z10.std()
        state = pd.qcut(x["vol_ratio"].shift(-1).reindex(z10.index), 3, labels=False)
        for k, lab in enumerate(("low", "mid", "high")):
            row[f"sd_z10_vol_{lab}"] = z10[state == k].std()
        row["sd_z10_with_roll"] = z10[roll10 == 1].std()
        row["sd_z10_no_roll"] = z10[roll10 == 0].std()
        row["kurt_z10"] = stats.kurtosis(z10)
        for tail, loss in (("left", -z10), ("right", z10)):
            row[f"breach_gauss_99_{tail}"] = np.mean(loss > stats.norm.ppf(0.99))
            row[f"q99_emp_{tail}"] = np.quantile(loss, 0.99)
        row["n_nonoverlap_10d"] = len(z10) // 10
        rows.append(row)
    return pd.DataFrame(rows)


def dependence_table(Z: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Do markets crash together more often than their correlation implies? Joint
    exceedance of the 2.5% tail in both markets on the same day, observed vs a Gaussian
    copula with the same (Spearman-implied) correlation."""
    zz = pd.DataFrame({n: dev(n, x)["z"] for n, x in Z.items()}).loc["2014-01-01":DEV_END]
    pairs = [("TTF", "NBP"), ("BRENT", "WTI"), ("NBP", "BRENT"), ("TTF", "BRENT"), ("HH", "WTI"),
             ("HH", "NBP"), ("TTF", "EUA"), ("BRENT", "EUA")]
    rows = []
    q = 0.025
    for a, b in pairs:
        x = zz[[a, b]].dropna()
        ra, rb = x[a].rank(pct=True), x[b].rank(pct=True)
        rho_s = stats.spearmanr(x[a], x[b]).statistic
        rho = 2 * np.sin(np.pi * rho_s / 6)                       # Gaussian-copula correlation
        cop = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])
        k = stats.norm.ppf(q)
        gauss_joint = cop.cdf([k, k])                             # same for both tails by symmetry
        for tail, both in (("both fall", (ra <= q) & (rb <= q)), ("both rise", (ra > 1 - q) & (rb > 1 - q))):
            rows.append({"pair": f"{a}-{b}", "tail": tail, "n": len(x), "corr_pearson": x[a].corr(x[b]),
                         "copula_rho": rho, "joint_obs": both.mean(), "joint_gauss": gauss_joint,
                         "ratio": both.mean() / gauss_joint, "cond_prob_obs": both.mean() / q,
                         "cond_prob_gauss": gauss_joint / q})
    return pd.DataFrame(rows)


def stress_table(Z: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Event: realised vol over the next 21 trading days >= 2 x today's calibrated EWMA vol.
    Episodes: runs of event days separated by more than 21 days (one 'crisis' counts once)."""
    rows = []
    for name, x in Z.items():
        v = (x["ret"] ** 2 / x["span"]).fillna(0.0)
        fwd = v[::-1].rolling(21).mean()[::-1].shift(-1)
        s2 = x["s2_prev"].shift(-1)                                 # calibrated forecast at t
        ev = (np.sqrt(fwd / s2) >= 2).where(fwd.notna() & s2.notna())
        d = ev.loc[START.get(name, DEV_START):DEV_END].iloc[:-21].dropna()   # windows end inside dev
        days = d.index[d.astype(bool)]
        gaps = np.diff(days.values).astype("timedelta64[D]").astype(int) if len(days) > 1 else np.array([])
        episodes = int(len(days) > 0) + int((gaps > 30).sum())
        ratio = np.sqrt(fwd / s2).loc[d.index]
        rows.append({"market": name, "days": len(d), "event_days": int(d.sum()), "base_rate": d.mean(),
                     "episodes": episodes, "first_days": ", ".join(str(t.date()) for t in days[np.r_[True, gaps > 30]][:6]) if len(days) else "",
                     "p90_fwd_vol_ratio": ratio.quantile(0.9), "share_ratio_ge_1.5": (ratio >= 1.5).mean()})
    return pd.DataFrame(rows)


def iv_table(Z: dict[str, pd.DataFrame], iv: pd.DataFrame) -> pd.DataFrame:
    """Implied vol (front-month options, known at t-1) as the scale instead of EWMA, with the
    same point-in-time calibration. Lower kurtosis of z = a better scale for tails."""
    rows = []

    def z_and_c(x, vol_annual):
        s2 = (vol_annual ** 2 / ANN).shift(1)
        z_raw = x["ret"] / np.sqrt(s2 * x["span"])
        c = pd.Series(np.nan, index=x.index)
        for y in range(2011, 2019):
            past = z_raw[z_raw.index.year < y].dropna()
            c[x.index.year == y] = (past ** 2).mean()
        return (z_raw / np.sqrt(c)).loc[DEV_START:DEV_END], c.loc[DEV_START:DEV_END].mean()

    for name in ("HH", "BRENT", "WTI"):
        x = Z[name]
        raw_iv = iv[name].reindex(x.index)
        ewma_vol = np.sqrt(x["s2_raw_prev"].shift(-1) * ANN)   # uncalibrated EWMA vol known at close of t
        ratio = raw_iv / ewma_vol
        # Plausibility filter (point-in-time): ignore an implied vol below 1/3 or above 3x the
        # EWMA vol of the same day and carry the last plausible value. Catches bad prints such as
        # HH on 2013-11-26 (1.2%, against 31% the day before).
        filt = raw_iv.where((ratio > 1 / 3) & (ratio < 3)).ffill(limit=5)
        scales = {"iv": z_and_c(x, raw_iv), "iv_filtered": z_and_c(x, filt),
                  "ewma": (x["z"].loc[DEV_START:DEV_END], x["c"].loc[DEV_START:DEV_END].mean())}
        both = pd.concat({k: v[0] for k, v in scales.items()}, axis=1).dropna()
        for k, (_, mean_c) in scales.items():
            z = both[k]
            rows.append({"market": name, "scale": k, "n": len(z), "sd_z": z.std(), "kurt_z": stats.kurtosis(z),
                         "breach99_left": np.mean(-z > stats.norm.ppf(0.99)),
                         "breach99_right": np.mean(z > stats.norm.ppf(0.99)), "mean_c": mean_c,
                         "days_filtered": int(((ratio <= 1 / 3) | (ratio >= 3)).loc[DEV_START:DEV_END].sum()) if k == "iv_filtered" else None,
                         "iv_over_ewma_p01": ratio.loc[DEV_START:DEV_END].quantile(0.01) if k == "iv" else None,
                         "iv_over_ewma_p99": ratio.loc[DEV_START:DEV_END].quantile(0.99) if k == "iv" else None})
    return pd.DataFrame(rows)


def power_table() -> pd.DataFrame:
    """Kupiec test (5%, two-sided) power against true breach rates 1.5x and 2x nominal, for
    the sample sizes available: per market (dev ~2,000 days; 2019-26 ~1,950 days), pooled
    over ~4.5 effective markets, and 10-day non-overlapping windows."""
    rows = []
    for label, n in (("one market, 1-day, 4 years", 1000), ("one market, 1-day, 8 years", 2000),
                     ("4.5 effective markets, 1-day, 8 years", 9000), ("one market, 10-day non-overlapping, 8 years", 200)):
        for p0 in (0.05, 0.01):
            ks = np.arange(n + 1)
            lr = np.zeros(n + 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                ph = ks / n
                ll_alt = np.where((ks > 0) & (ks < n), (n - ks) * np.log(1 - ph) + ks * np.log(np.where(ph > 0, ph, 1)), 0.0)
            ll_null = (n - ks) * np.log(1 - p0) + ks * np.log(p0)
            lr = -2 * (ll_null - ll_alt)
            reject = stats.chi2.sf(lr, 1) < 0.05
            row = {"sample": label, "n": n, "nominal": p0, "expected_breaches": n * p0}
            for mult in (1.5, 2.0):
                row[f"power_true_{mult}x"] = stats.binom.pmf(ks, n, p0 * mult)[reject].sum()
            rows.append(row)
    return pd.DataFrame(rows)


def tail_figure(Z: dict[str, pd.DataFrame]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    LEFT, RIGHT = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex=True, sharey=True, facecolor="#fcfcfb")
    xs = np.linspace(1, 8, 200)
    for ax, (name, x) in zip(axes.ravel(), Z.items()):
        z = dev(name, x)["z"].dropna().to_numpy()
        for loss, col, lab in ((-z, LEFT, "falls (long loses)"), (z, RIGHT, "rises (short loses)")):
            s = np.sort(loss)[::-1]
            p = np.arange(1, len(s) + 1) / len(s)
            keep = s > 1
            ax.step(s[keep], p[keep], where="post", color=col, lw=1.6, label=lab)
        ax.plot(xs, stats.norm.sf(xs), color=INK2, lw=1, ls="--", label="normal")
        ax.set_yscale("log")
        ax.set_ylim(2e-4, 0.3)
        ax.set_xlim(1, 8 if name != "EUA" else 8)
        ax.set_title(f"{name}  ({dev(name, x).index[0].year}-18)", color=INK, fontsize=10, loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        for s_ in ax.spines.values():
            s_.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
    for ax in axes[1]:
        ax.set_xlabel("standardised move z (in forecast sds)", color=INK2, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of days beyond z", color=INK2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Daily tails after scaling by the calibrated EWMA forecast, 2011-2018",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(etr.RESULTS / "p0_tail_curves.png", dpi=150)


def main() -> None:
    rets = pd.read_parquet(etr.PROCESSED / "returns.parquet")
    iv = pd.read_parquet(etr.PROCESSED / "implied_vol.parquet")
    Z = {name: standardise(rets[rets.market == name]) for name in MARKETS}
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    tables = {"shape": shape_table(Z), "state": state_table(Z), "horizon": horizon_table(Z),
              "dependence": dependence_table(Z), "stress": stress_table(Z), "iv": iv_table(Z, iv),
              "power": power_table()}
    for tag, t in tables.items():
        t = t.round(4)
        t.to_csv(etr.RESULTS / f"p0_tails_{tag}.csv", index=False)
        print(f"\n== {tag}\n{t.to_string(index=False)}")
    print("normal reference:", tables["shape"].attrs["normal"])
    tail_figure(Z)


if __name__ == "__main__":
    main()
