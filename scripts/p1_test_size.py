"""Phase 1: size and power of every registered test, by simulation, at this project's
sample sizes. Writes results/p1_test_size.csv (rejection rates at the 5% level).

Design (chosen to resemble the data, decided before any tail model exists):
- 2,000 trading days (2019-01 to 2026-10 is about 1,950), six markets in three pairs with
  daily correlation 0.9 (TTF-NBP, Brent-WTI, and a third pair standing in for HH/EUA;
  cross-pair correlation 0), via a Gaussian copula.
- Marginals: Student-t with 5 degrees of freedom scaled to unit variance (conditional
  kurtosis about 6, inside the 0.8-4.3 range found in 2011-18 plus margin).
- Losses are the standardised moves themselves (one tail); the other tail is symmetric.
Scenarios:
  H0          the forecast is the true distribution: rejection rate = size
  scale_low   forecast scale 13% too small (variance 25% too small): the uncalibrated
              EWMA / regime-shift error
  normal_tail truth t(4) with unit variance, forecast normal with unit variance: right
              volatility, wrong tails
  monday      Monday variance 1.6x, other days scaled so the average is 1; the forecast
              ignores weekdays (the European-gas effect found in session 1)
The 10-day block uses sums of 10 days, non-overlapping windows for the calibration tests.

Run:  python scripts/p1_test_size.py [reps]     (default 200 reps; about 15 minutes)
      python scripts/p1_test_size.py primary    (1,000-rep size check of the primary tests; ~4 minutes)
"""
import sys

import numpy as np
import pandas as pd
from scipy import stats

import etr
from etr import backtest as bt

N_DAYS, N_SIMS, RHO = 2000, 499, 0.9
DATES = pd.bdate_range("2019-01-01", periods=N_DAYS)
WEEKDAYS = np.column_stack([DATES.weekday == k for k in range(1, 5)]).astype(float)


class Dist:
    """A loss distribution scaled by s: VaR, ES, PIT and draws (for the AS test)."""

    def __init__(self, base, s=1.0):
        self.base, self.s = base, s

    def var(self, a):
        return self.s * self.base.isf(a)

    def es(self, a):
        q = self.base.isf(a)
        return self.s * self.base.expect(lambda x: x, lb=q) / a

    def pit(self, x):
        return self.base.sf(x / self.s)

    def draws(self, rng, n, k):
        return self.s * self.base.rvs(size=(n, k), random_state=rng)


def unit_t(nu):
    return stats.t(nu, scale=np.sqrt((nu - 2) / nu))


class Empirical:
    """Distribution known only by a large sample (10-day sums of t variables)."""

    def __init__(self, sample):
        self.x = np.sort(sample)
        self.n = len(sample)

    def isf(self, a):
        return np.quantile(self.x, 1 - a)

    def expect(self, f, lb):
        tail = self.x[self.x >= lb]
        return tail.sum() / self.n

    def sf(self, v):
        return 1 - np.searchsorted(self.x, v, side="left") / self.n

    def rvs(self, size, random_state):
        return self.x[random_state.integers(0, self.n, size)]


def correlated_uniforms(rng, n):
    z = rng.standard_normal((n, 3))
    x = np.repeat(z, 2, axis=1) * np.sqrt(RHO) + rng.standard_normal((n, 6)) * np.sqrt(1 - RHO)
    return stats.norm.cdf(x)


def daily_case(scenario, rng):
    """(losses n x 6, forecast Dist per column, per-day scale multiplier of the truth)."""
    U = correlated_uniforms(rng, N_DAYS)
    nu = 4 if scenario == "normal_tail" else 5
    X = unit_t(nu).ppf(U)
    if scenario == "monday":
        mult = np.where(DATES.weekday == 0, np.sqrt(1.6), np.sqrt((5 - 1.6) / 4))
        X = X * mult[:, None]
    fc = {"H0": Dist(unit_t(5)), "scale_low": Dist(unit_t(5), 0.87), "normal_tail": Dist(stats.norm()),
          "monday": Dist(unit_t(5))}[scenario]
    return X, fc


def run_daily(scenario, rng):
    X, fc = daily_case(scenario, rng)
    out = {}
    U = pd.DataFrame(fc.pit(X), index=DATES)
    x0 = X[:, 0]
    seed = int(rng.integers(1 << 31))
    for a, tag in ((0.01, "99"), (0.025, "975")):
        v = fc.var(a)
        h0 = bt.hits(x0, np.full(N_DAYS, v))
        out[f"kupiec_{tag}"] = bt.kupiec(h0, a)["p"]
        out[f"pooled_coverage_{tag}"] = bt.coverage_pooled(U, a, n_boot=N_SIMS, seed=seed)["p"]
        out[f"pooled_coverage_hac_{tag}"] = bt.coverage_hac((U < a).astype(int), a)["p"]
    h99 = bt.hits(x0, np.full(N_DAYS, fc.var(0.01)))
    ch = bt.christoffersen(h99, 0.01, n_sims=N_SIMS, seed=seed)
    out["christoffersen_cc_99"] = ch["p_cc"]
    out["dq_weekdays_99"] = bt.dq_test(h99, 0.01, exog=WEEKDAYS, n_sims=N_SIMS, seed=seed)["p"]
    h975 = bt.hits(x0, np.full(N_DAYS, fc.var(0.025)))
    out["dq_weekdays_975"] = bt.dq_test(h975, 0.025, exog=WEEKDAYS, n_sims=N_SIMS, seed=seed)["p"]
    a = 0.025
    V, E = np.full(N_DAYS, fc.var(a)), np.full(N_DAYS, fc.es(a))
    out["acerbi_szekely_z2_975"] = bt.acerbi_szekely_z2(x0, V, E, a, fc.draws(rng, N_SIMS, N_DAYS))["p"]
    out["du_escanciano_975"] = bt.du_escanciano(fc.pit(x0), a)["p_u"]
    out["mcneil_frey_975"] = bt.mcneil_frey(x0, V, E, n_boot=N_SIMS, seed=seed)["p"]
    out["pooled_es_975"] = bt.es_pooled(U, a, n_boot=N_SIMS, seed=seed)["p"]
    return out


TEN = {}


def ten_day_dist(nu, rng):
    if nu not in TEN:
        TEN[nu] = Empirical(unit_t(nu).rvs(size=(400_000, 10), random_state=rng).sum(1))
    return TEN[nu]


def run_ten_day(scenario, rng):
    """Non-overlapping 10-day windows (200 per market) for calibration tests."""
    nu = 4 if scenario == "normal_tail" else 5
    U = correlated_uniforms(rng, N_DAYS)
    X = unit_t(nu).ppf(U)
    S = np.add.reduceat(X, np.arange(0, N_DAYS, 10), axis=0)                    # 200 x 6
    truth = ten_day_dist(nu, rng)
    fc = {"H0": Dist(truth), "scale_low": Dist(truth, 0.87),
          "normal_tail": Dist(stats.norm(scale=np.sqrt(10)))}[scenario]
    Upit = pd.DataFrame(fc.pit(S))
    seed = int(rng.integers(1 << 31))
    out = {}
    for a, tag in ((0.01, "99"), (0.025, "975")):
        out[f"10d_kupiec_nonoverlap_{tag}"] = bt.kupiec(bt.hits(S[:, 0], np.full(len(S), fc.var(a))), a)["p"]
        out[f"10d_pooled_coverage_nonoverlap_{tag}"] = bt.coverage_pooled(Upit, a, n_boot=N_SIMS, seed=seed)["p"]
    out["10d_pooled_es_nonoverlap_975"] = bt.es_pooled(Upit, 0.025, n_boot=N_SIMS, seed=seed)["p"]
    return out


def dm_size(h, rng, reps):
    """Two forecasts that are equally good by symmetry (true scale times independent
    lognormal noise); pooled FZ0 at 2.5%, overlapping windows when h = 10."""
    a, t5 = 0.025, unit_t(5)
    qv = t5.isf(a)
    qe = t5.expect(lambda x: x, lb=qv) / a
    shift = 3 * np.sqrt(h)                                                       # keeps ES > 0
    rej = 0
    for _ in range(reps):
        X = t5.ppf(correlated_uniforms(rng, N_DAYS + h))
        L = np.stack([np.convolve(X[:, j], np.ones(h), "valid")[:N_DAYS] for j in range(6)], 1) + shift
        s = np.sqrt(h) * np.exp(0.1 * rng.standard_normal((2, N_DAYS, 6)))
        la = bt.fz0(L, shift + s[0] * qv, shift + s[0] * qe, a)
        lb = bt.fz0(L, shift + s[1] * qv, shift + s[1] * qe, a)
        rej += bt.diebold_mariano((la - lb).mean(1), lags=5 if h == 1 else 2 * h)["p"] < 0.05
    return rej / reps


def main(reps: int = 200) -> None:
    rng = np.random.default_rng(20261024)
    rows = []
    for scenario in ("H0", "scale_low", "normal_tail", "monday"):
        res = [run_daily(scenario, rng) for _ in range(reps)]
        if scenario != "monday":
            res = [{**d, **run_ten_day(scenario, rng)} for d in res]
        for test in res[0]:
            p = np.array([r[test] for r in res])
            rows.append({"scenario": scenario, "test": test, "rejection_rate": np.mean(p < 0.05), "reps": reps})
        print(scenario, "done", flush=True)
    for h in (1, 10):
        rows.append({"scenario": "H0", "test": f"dm_fz0_pooled_{h}d", "rejection_rate": dm_size(h, rng, reps), "reps": reps})
    t = pd.DataFrame(rows).pivot(index="test", columns="scenario", values="rejection_rate")
    t = t[["H0", "scale_low", "normal_tail", "monday"]].rename(columns={"H0": "size (H0)"})
    t.round(3).to_csv(etr.RESULTS / "p1_test_size.csv")
    print(t.round(3).to_string())


def primary_size(reps: int = 1000) -> None:
    """A tighter size check (Monte Carlo SE about 0.7 points) of the three pooled tests that
    carry the primary verdict in PROTOCOL.md Q1. Writes results/p1_primary_size.csv."""
    rng = np.random.default_rng(777)
    rej = {"pooled_es_975": [], "pooled_coverage_99": [], "pooled_coverage_975": []}
    for k in range(reps):
        X, fc = daily_case("H0", rng)
        U = pd.DataFrame(fc.pit(X))
        rej["pooled_es_975"].append(bt.es_pooled(U, 0.025, n_boot=N_SIMS, seed=k)["p"] < 0.05)
        rej["pooled_coverage_99"].append(bt.coverage_pooled(U, 0.01, n_boot=N_SIMS, seed=k)["p"] < 0.05)
        rej["pooled_coverage_975"].append(bt.coverage_pooled(U, 0.025, n_boot=N_SIMS, seed=k)["p"] < 0.05)
    t = pd.DataFrame([{"test": k, "size": np.mean(v), "mc_se": np.sqrt(np.mean(v) * (1 - np.mean(v)) / reps),
                       "reps": reps} for k, v in rej.items()])
    t.round(4).to_csv(etr.RESULTS / "p1_primary_size.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "primary":
        primary_size()
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
