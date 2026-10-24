"""Tail shapes on data with known tails, and the engine that assembles scale x shape."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from etr import engine, scale, tails
from test_scale import gjr_market

LEFT_HEAVY = tails.SkewT(6.0, -0.3)          # falls fatter than spikes, like oil


def tail_truth(dist, alpha):
    """True alpha quantile and ES of z, both tails, by brute force on 4 million draws."""
    z = dist.rvs(size=4_000_000, random_state=np.random.default_rng(0))
    out = {}
    for name, x in (("left", np.sort(z)), ("right", np.sort(-z))):
        k = int(alpha * len(x))
        out[name] = (-x[k], -x[:k].mean())
    return out


@pytest.fixture(scope="module")
def skewed_sample():
    return LEFT_HEAVY.rvs(size=30_000, random_state=np.random.default_rng(1))


@pytest.mark.parametrize("kind", ["skewt", "fhs", "gpd"])
def test_shapes_recover_known_asymmetric_tails(skewed_sample, kind):
    """1% quantile and ES of both tails within 5% of the truth (30,000 residuals)."""
    s = tails.fit_shapes({"m": skewed_sample}, kind, pooled=False)["m"]
    truth = tail_truth(LEFT_HEAVY, 0.01)
    est = {"left": (-s.ppf(0.01), -tails.tail_mean(lambda z: z, s, 0.01, False)),
           "right": (s.isf(0.01), tails.tail_mean(lambda z: z, s, 0.01, True))}
    for side in ("left", "right"):
        assert est[side][0] == pytest.approx(truth[side][0], rel=0.05)
        assert est[side][1] == pytest.approx(truth[side][1], rel=0.05)


def test_symmetric_t_cannot_see_the_asymmetry(skewed_sample):
    t = tails.fit_shapes({"m": skewed_sample}, "t", pooled=False)["m"]
    truth = tail_truth(LEFT_HEAVY, 0.01)
    assert -t.ppf(0.01) < 0.97 * truth["left"][0]                # understates the heavy side
    assert t.isf(0.01) > 1.03 * truth["right"][0]               # overstates the light side


def test_skewt_fit_recovers_parameters_and_pooling_shares_eta_only():
    rng = np.random.default_rng(2)
    a = tails.SkewT(5.0, -0.3).rvs(size=40_000, random_state=rng)
    b = tails.SkewT(5.0, 0.25).rvs(size=40_000, random_state=rng)
    sa = tails.fit_shapes({"a": a}, "skewt", False)["a"]
    assert sa.eta == pytest.approx(5.0, rel=0.1) and sa.lam == pytest.approx(-0.3, abs=0.03)
    pooled = tails.fit_shapes({"a": a, "b": b}, "skewt", True)
    assert pooled["a"].eta == pooled["b"].eta == pytest.approx(5.0, rel=0.1)
    assert pooled["a"].lam == pytest.approx(-0.3, abs=0.03) and pooled["b"].lam == pytest.approx(0.25, abs=0.03)


def test_gpd_fit_recovers_xi_and_pools_it_across_tails():
    rng = np.random.default_rng(3)
    e1 = stats.genpareto(0.2, scale=0.5).rvs(5000, random_state=rng)
    xi, (beta,) = tails.fit_gpd([e1])
    assert xi == pytest.approx(0.2, abs=0.05) and beta == pytest.approx(0.5, rel=0.08)
    e2 = stats.genpareto(0.2, scale=1.5).rvs(5000, random_state=rng)
    xi, betas = tails.fit_gpd([e1, e2])
    assert xi == pytest.approx(0.2, abs=0.04)
    assert betas[0] == pytest.approx(0.5, rel=0.08) and betas[1] == pytest.approx(1.5, rel=0.08)


@pytest.mark.parametrize("kind", ["skewt", "gpd"])
def test_quantile_and_probability_functions_invert_each_other(skewed_sample, kind):
    s = tails.fit_shapes({"m": skewed_sample}, kind, pooled=False)["m"]
    u = np.array([1e-5, 1e-3, 0.01, 0.05])
    np.testing.assert_allclose(s.cdf(s.ppf(u)), u, rtol=1e-6)
    np.testing.assert_allclose(s.sf(s.isf(u)), u, rtol=1e-6)


def test_gpd_tails_join_the_empirical_body_at_the_thresholds(skewed_sample):
    s = tails.fit_shapes({"m": skewed_sample}, "gpd", pooled=False)["m"]
    assert s.cdf(s.uL - 1e-9) == pytest.approx(s.pL, rel=1e-6)
    assert s.sf(s.uR + 1e-9) == pytest.approx(s.pR, rel=1e-6)
    assert s.pL == pytest.approx(tails.GPD_TAIL, abs=0.001)


def test_empirical_shape_exact_es_and_breach_consistency():
    rng = np.random.default_rng(4)
    z = rng.standard_t(4, 1000) / np.sqrt(2)
    s = tails.Empirical(z)
    srt = np.sort(z)
    assert tails.tail_mean(lambda x: x, s, 0.025, False) == pytest.approx((srt[:25].sum()) / 25)
    assert tails.tail_mean(lambda x: x, s, 0.0125, False) == pytest.approx((srt[:12].sum() + 0.5 * srt[12]) / 12.5)
    # In a forecast, a breach is exactly u <= alpha (the historical-simulation convention).
    n = 3000
    dates = pd.bdate_range("2010-01-04", periods=n)
    df = pd.DataFrame({"ret": np.r_[np.nan, 0.02 * rng.standard_t(4, n - 1) / np.sqrt(2)],
                       "span": np.r_[0, np.ones(n - 1, int)]}, index=dates)
    f = tails.scale_shape_forecast(df, pd.Series(0.02 ** 2, index=dates), "long", 1, s).frame
    np.testing.assert_array_equal(f["loss"] > f["var99"], f["u"] <= 0.01)


def test_concat_keeps_each_period_sampler():
    dates = pd.bdate_range("2010-01-04", periods=600)
    df = pd.DataFrame({"ret": np.r_[np.nan, np.full(599, 0.001)], "span": np.r_[0, np.ones(599, int)]}, index=dates)
    v1 = pd.Series(np.where(dates.year == 2010, 1e-4, np.nan), index=dates)
    v2 = pd.Series(np.where(dates.year == 2011, 1e-2, np.nan), index=dates)
    f = tails.concat([tails.scale_shape_forecast(df, v1, "long"), tails.scale_shape_forecast(df, v2, "long")])
    draws = f.sample(np.random.default_rng(5), 2000)
    small, big = draws[:, f.frame.index.year == 2010].std(), draws[:, f.frame.index.year == 2011].std()
    assert big / small == pytest.approx(10, rel=0.1)              # sd 1% vs 10%


# ---- engine ----

@pytest.fixture(scope="module")
def walk():
    mk = {"A": gjr_market(seed=11), "B": gjr_market(seed=12)}
    fc, _, ins = scale.walk_forward(mk, [2015, 2016], keep_insample=True)
    return mk, fc, ins


def test_training_residuals_have_mean_square_one(walk):
    """Calibration makes the in-sample residuals' mean square exactly 1."""
    mk, _, ins = walk
    z = engine.residuals(mk["A"], ins[("A", 2016, "gjr_1")], pd.Timestamp("2015-12-31"))
    assert np.mean(z ** 2) == pytest.approx(1.0, rel=1e-9)


def test_engine_has_no_lookahead(walk):
    """Shock every return after mid-2016: VaR and ES at earlier origins must not move, and
    the shapes fitted for 2016 must not either."""
    mk, fc, ins = walk
    base, p0 = engine.one_day(mk, fc, ins, [2015, 2016], "comb_1", "skewt_pool")
    cut = pd.Timestamp("2016-06-15")
    shocked = {k: v.copy() for k, v in mk.items()}
    for v in shocked.values():
        late = v.index > cut
        v.loc[late, "ret"] = v.loc[late, "ret"] * 3
    fc2, _, ins2 = scale.walk_forward(shocked, [2015, 2016], keep_insample=True)
    alt, p1 = engine.one_day(shocked, fc2, ins2, [2015, 2016], "comb_1", "skewt_pool")
    cols = [c for c in base[("A", "long")].frame.columns if c.startswith(("var", "es"))]
    for k in base:
        b = base[k].frame.loc[:cut - pd.offsets.BDay(1), cols]
        pd.testing.assert_frame_equal(b, alt[k].frame.loc[b.index, cols])
    assert p0 == p1


def test_selection_rule():
    rng = np.random.default_rng(6)
    base = rng.normal(size=800)
    noise = lambda: 0.3 * rng.normal(size=800)  # noqa: E731
    scores = pd.DataFrame({"ewma_1|normal": base + noise(), "ewma_1|t_pool": base + 0.001 + noise(),
                           "comb_1|skewt_pool": base + noise(), "hs500": base + 0.5 + noise()})
    fails = pd.Series({"ewma_1|normal": 2, "ewma_1|t_pool": 0, "comb_1|skewt_pool": 0, "hs500": 3})
    t = engine.select(scores, fails, block=10)
    assert t.loc["ewma_1|t_pool", "chosen"]              # shortlisted, passes, simplest (1 + 1)
    assert not t.loc["hs500", "shortlisted"]
    t2 = engine.select(scores, fails.replace(0, 1), block=10)   # nothing passes: fewest failures, simplest
    assert t2.loc["ewma_1|t_pool", "chosen"]
    assert engine.complexity("comb_iv_1|gpd_pool") == 9 and engine.complexity("hs500") == 3


def test_shapes_for_a_year_ignore_that_years_returns(walk):
    """Shock the first week of 2016: the shapes fitted for 2016 must not move (the training
    residual of 31 December targets 4 January, which belongs to 2016)."""
    mk, fc, ins = walk
    _, p0 = engine.one_day(mk, fc, ins, [2016], "gjr_1", "gpd_mkt")
    shocked = {k: v.copy() for k, v in mk.items()}
    for v in shocked.values():
        jan = (v.index >= "2016-01-01") & (v.index <= "2016-01-08")
        v.loc[jan, "ret"] *= 10
    fc2, _, ins2 = scale.walk_forward(shocked, [2016], keep_insample=True)
    _, p1 = engine.one_day(shocked, fc2, ins2, [2016], "gjr_1", "gpd_mkt")
    assert p0 == p1
