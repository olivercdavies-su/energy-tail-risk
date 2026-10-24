"""Backtests and scoring functions, checked on synthetic data where the right answer is
known: a correct forecast must pass at about the nominal rate, a wrong one must fail."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from etr import backtest as bt

RNG = np.random.default_rng(12345)


def markov_hits(n, alpha, persistence, rng):
    """Hits with stationary rate alpha but P(hit | hit yesterday) = persistence (clustering)."""
    p01 = alpha * (1 - persistence) / (1 - alpha)
    h = np.zeros(n, int)
    for t in range(1, n):
        h[t] = rng.random() < (persistence if h[t - 1] else p01)
    return h


# ---- coverage and independence ----

def test_kupiec_exact_pvalue_and_size():
    assert bt.kupiec(np.r_[np.ones(10), np.zeros(990)], 0.01)["p"] > 0.9
    assert bt.kupiec(np.r_[np.ones(25), np.zeros(975)], 0.01)["p"] < 0.001
    # Exact size, summed over every possible breach count: at most 5% by construction, and
    # not far below it (a chi-squared(1) approximation gives 5.5%, i.e. over-rejects).
    n, a = 1000, 0.01
    k = np.arange(60)
    p = np.array([bt.kupiec(np.r_[np.ones(x), np.zeros(n - x)], a)["p"] for x in k])
    size = (stats.binom.pmf(k, n, a) * (p < 0.05)).sum()
    assert 0.035 <= size <= 0.05


def test_christoffersen_passes_iid_and_catches_clustering():
    rng = np.random.default_rng(2)
    ps = [bt.christoffersen(rng.random(1000) < 0.02, 0.02, n_sims=499, seed=k)["p_ind"] for k in range(200)]
    assert 0.0 < np.mean(np.array(ps) < 0.05) <= 0.10
    clustered = markov_hits(1000, 0.02, 0.3, np.random.default_rng(3))
    res = bt.christoffersen(clustered, 0.02, n_sims=999)
    assert res["p_ind"] < 0.01 and res["p_cc"] < 0.01


def test_dq_catches_a_weekday_effect_that_breach_counts_miss():
    """Right on average (1%) but 4% on Mondays: Kupiec passes, DQ with weekday dummies fails."""
    rng = np.random.default_rng(4)
    days = pd.bdate_range("2012-01-02", periods=2500)
    p = np.where(days.weekday == 0, 0.04, 0.0025)
    h = (rng.random(len(days)) < p).astype(int)
    assert bt.kupiec(h, 0.01)["p"] > 0.05
    dummies = np.column_stack([days.weekday == k for k in range(1, 5)]).astype(float)
    assert bt.dq_test(h, 0.01, exog=dummies, n_sims=499)["p"] < 0.01
    iid = (rng.random(len(days)) < 0.01).astype(int)
    assert bt.dq_test(iid, 0.01, exog=dummies, n_sims=499)["p"] > 0.01


def test_traffic_light_matches_basel():
    assert bt.traffic_light_bounds(250, 0.01) == (4, 9)
    h = np.zeros(300)
    h[:10] = 1                                         # 10 exceptions in the first window
    tl = bt.traffic_light(h)
    assert tl["max_exceptions"] == 10 and tl["red"] > 0


# ---- ES tests ----

@pytest.fixture(scope="module")
def normal_case():
    """1000 days of N(0, 1) losses with the true 97.5% VaR and ES, and a forecast whose
    VaR and ES are 20% too small."""
    a = 0.025
    L = RNG.standard_normal(1000)
    v = stats.norm.ppf(1 - a)
    e = stats.norm.pdf(v) / a
    return a, L, np.full(1000, v), np.full(1000, e)


def test_acerbi_szekely_z2(normal_case):
    a, L, V, E = normal_case
    sims = np.random.default_rng(5).standard_normal((1999, len(L)))
    good = bt.acerbi_szekely_z2(L, V, E, a, sims)
    assert abs(good["z2"]) < 0.5 and good["p"] > 0.01
    bad = bt.acerbi_szekely_z2(L, 0.8 * V, 0.8 * E, a, 0.8 * sims)
    assert bad["z2"] > good["z2"] and bad["p"] < 0.01


def test_du_escanciano():
    rng = np.random.default_rng(6)
    u = rng.random(2000)
    assert bt.du_escanciano(u, 0.025)["p_u"] > 0.01
    # A forecast whose scale is 20% too small: true losses N(0,1), forecast N(0, 0.8^2).
    L = rng.standard_normal(2000)
    assert bt.du_escanciano(stats.norm.sf(L / 0.8), 0.025)["p_u"] < 0.001


def test_mcneil_frey(normal_case):
    a, L, V, E = normal_case
    assert bt.mcneil_frey(L, V, E)["p"] > 0.01
    L2 = np.random.default_rng(7).standard_t(3, 3000) / np.sqrt(3)   # fatter tails, same variance
    v, e = np.full(3000, V[0]), np.full(3000, E[0])
    assert bt.mcneil_frey(L2, v, e)["p"] < 0.05


# ---- scoring ----

def test_fz0_is_minimised_by_the_true_var_and_es():
    a = 0.025
    nu = 5
    t = stats.t(nu, scale=np.sqrt((nu - 2) / nu))
    L = t.rvs(400_000, random_state=8) + 3.0                       # positive ES needed
    v = t.isf(a) + 3.0
    e = 3.0 + t.expect(lambda x: x, lb=t.isf(a)) / a
    best = bt.fz0(L, v, e, a).mean()
    for dv, de in [(1.05, 1), (0.95, 1), (1, 1.05), (1, 0.95), (1.03, 1.03), (0.97, 0.97)]:
        assert bt.fz0(L, v * dv, e * de, a).mean() > best


def test_fz0_differences_are_scale_free():
    L, V, E = RNG.standard_normal(100) + 3, np.full(100, 4.0), np.full(100, 4.5)
    np.testing.assert_allclose(bt.fz0(7 * L, 7 * V, 7 * E, 0.025), bt.fz0(L, V, E, 0.025) + np.log(7))
    with pytest.raises(ValueError):
        bt.fz0(L, V, -E, 0.025)


def test_tick_loss_minimised_by_the_quantile():
    L = RNG.standard_normal(200_000)
    q = stats.norm.ppf(0.99)
    best = bt.tick_loss(L, q, 0.01).mean()
    assert bt.tick_loss(L, q * 1.05, 0.01).mean() > best and bt.tick_loss(L, q * 0.95, 0.01).mean() > best


# ---- comparison and pooling ----

def test_diebold_mariano():
    d = -0.1 + RNG.standard_normal(1000)
    res = bt.diebold_mariano(d)
    assert res["p_less"] < 0.05 and res["mean_diff"] < 0
    assert bt.diebold_mariano(RNG.standard_normal(1000))["p"] > 0.001


def correlated_hits(n, alpha, rng, rho=0.9):
    """Six markets in three highly correlated pairs (like TTF-NBP, Brent-WTI), Gaussian."""
    z = rng.standard_normal((n, 3))
    eps = rng.standard_normal((n, 6))
    x = np.repeat(z, 2, axis=1) * np.sqrt(rho) + eps * np.sqrt(1 - rho)
    return pd.DataFrame(x > stats.norm.isf(alpha)).astype(int)


def test_pooled_coverage_respects_cross_market_correlation():
    rng = np.random.default_rng(9)
    rej = np.mean([bt.coverage_hac(correlated_hits(2000, 0.01, rng), 0.01)["p"] < 0.05 for _ in range(300)])
    assert rej <= 0.10
    # A 1.5x breach rate in every market is caught.
    assert bt.coverage_hac(correlated_hits(2000, 0.015, rng), 0.01)["p"] < 0.05


def test_pooled_pit_bootstrap_tests_are_sized_and_catch_errors():
    """Six correlated markets with correct forecasts: pooled coverage and ES tests reject at
    about their nominal size. Forecasts 20% too narrow are rejected."""
    rng = np.random.default_rng(11)

    def panel(n, scale):
        z = rng.standard_normal((n, 3))
        x = np.repeat(z, 2, axis=1) * np.sqrt(0.9) + rng.standard_normal((n, 6)) * np.sqrt(0.1)
        return pd.DataFrame(stats.norm.sf(x / scale))          # PIT of losses x under N(0, scale^2)

    rej_cov = [bt.coverage_pooled(panel(1000, 1.0), 0.01, n_boot=199, seed=k)["p"] < 0.05 for k in range(150)]
    rej_es = [bt.es_pooled(panel(1000, 1.0), 0.025, n_boot=199, seed=k)["p"] < 0.05 for k in range(150)]
    assert np.mean(rej_cov) <= 0.10 and np.mean(rej_es) <= 0.10
    narrow = panel(2000, 0.8)
    assert bt.coverage_pooled(narrow, 0.01, n_boot=499)["p"] < 0.01
    assert bt.es_pooled(narrow, 0.025, n_boot=499)["p"] < 0.01


def test_pooled_tests_handle_missing_columns():
    rng = np.random.default_rng(12)
    u = pd.DataFrame(rng.random((800, 3)))
    u.iloc[:400, 0] = np.nan                                     # a market that starts later
    res = bt.coverage_pooled(u, 0.05, n_boot=199)
    assert 0.03 < res["rate"] < 0.07 and res["dates"] == 800


def test_overlapping_hits_break_iid_tests():
    """10-day overlapping hits are autocorrelated by construction: treating them as iid
    (Kupiec) over-rejects badly; non-overlapping windows restore the size."""
    rng = np.random.default_rng(10)
    alpha, h = 0.05, 10
    q = stats.norm.isf(alpha) * np.sqrt(h)
    rej_iid, rej_non = [], []
    days = pd.bdate_range("2010-01-04", periods=2000)
    for _ in range(300):
        r = rng.standard_normal(2000 + h)
        hit = pd.DataFrame({"hit": (np.convolve(r, np.ones(h), "valid")[:2000] > q).astype(int)}, index=days)
        rej_iid.append(bt.kupiec(hit["hit"], alpha)["p"] < 0.05)
        rej_non.append(bt.kupiec(bt.nonoverlapping(hit, h)["hit"], alpha)["p"] < 0.05)
    assert np.mean(rej_iid) > 0.2 and np.mean(rej_non) <= 0.07


def test_nonoverlapping_grid_lines_markets_up():
    us = pd.bdate_range("2020-01-01", "2020-12-31").drop(pd.to_datetime(["2020-07-03", "2020-11-26"]))
    uk = pd.bdate_range("2020-01-01", "2020-12-31").drop(pd.to_datetime(["2020-04-10", "2020-04-13"]))
    a = bt.nonoverlapping(pd.DataFrame({"x": 1}, index=us), 10)
    b = bt.nonoverlapping(pd.DataFrame({"x": 1}, index=uk), 10)
    assert a.index.equals(b.index)
    assert (np.diff(a.index.values).astype("timedelta64[D]").astype(int) == 14).all()


def test_es_ratio_hac(normal_case):
    a, L, V, E = normal_case
    one = lambda x: pd.DataFrame({"m": x})  # noqa: E731
    assert bt.es_ratio_hac(one(L), one(V), one(E), a)["p"] > 0.01
    assert bt.es_ratio_hac(one(L), one(0.8 * V), one(0.8 * E), a)["p"] < 0.01


def test_holm():
    np.testing.assert_allclose(bt.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_model_confidence_set_keeps_the_best_and_drops_the_bad():
    n = 1000
    base = RNG.standard_normal(n)
    losses = pd.DataFrame({"best": base, "same": base + 0.01 * RNG.standard_normal(n), "bad": base + 0.5})
    p = bt.model_confidence_set(losses, reps=500)
    assert p["best"] >= 0.10 and p["bad"] < 0.10
