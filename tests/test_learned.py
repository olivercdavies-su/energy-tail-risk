"""Learned multiplier and early warning: known relations recovered, noise ignored, and no
look-ahead in inputs, fits or forecasts."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from etr import learned, scale, tails
from test_scale import gjr_market

YEARS = [2015, 2016]


@pytest.fixture(scope="module")
def walk():
    mk = {"A": gjr_market(seed=31), "B": gjr_market(seed=32)}
    fc, _, ins = scale.walk_forward(mk, YEARS, keep_insample=True)
    feats = {n: learned.features(df) for n, df in mk.items()}
    return mk, fc, ins, feats


def groups_for(shape, n, side, alpha=0.025):
    q, z, w = learned._tail_nodes(shape, alpha, side == "short")
    return [(np.arange(n), q, z, w)]


@pytest.mark.parametrize("side", tails.SIDES)
def test_multiplier_recovers_a_known_effect(side):
    """True vol = base vol x exp(0.3 x): minimising FZ0 recovers the 0.3 (and a zero
    constant), because FZ0 is minimised by the true VaR and ES."""
    rng = np.random.default_rng(1)
    n = 40_000
    x = rng.standard_normal(n)
    sigma = np.full(n, 0.02)
    shape = tails.StudentT(5)
    R = sigma * np.exp(0.3 * x) * shape.rvs(size=n, random_state=rng)
    beta = learned.fit_multiplier(x[:, None], sigma, R, groups_for(shape, n, side), side)
    assert beta[1] == pytest.approx(0.3, abs=0.06)
    assert beta[0] == pytest.approx(0.0, abs=0.06)


def test_multiplier_ignores_an_irrelevant_input():
    rng = np.random.default_rng(2)
    n = 40_000
    shape = tails.StudentT(5)
    sigma = np.full(n, 0.02)
    R = sigma * shape.rvs(size=n, random_state=rng)
    beta = learned.fit_multiplier(rng.standard_normal((n, 2)), sigma, R, groups_for(shape, n, "long"), "long")
    assert np.abs(beta).max() < 0.06


def test_features_use_only_the_past(walk):
    mk, _, _, feats = walk
    df = mk["A"]
    cut = pd.Timestamp("2016-06-15")
    shocked = df.copy()
    shocked.loc[shocked.index > cut, "ret"] *= 3
    pd.testing.assert_frame_equal(feats["A"].loc[:cut], learned.features(shocked).loc[:cut])


def test_learned_forecasts_have_no_lookahead(walk):
    mk, fc, ins, feats = walk
    base, p0 = learned.learned_forecasts(mk, fc, ins, feats, YEARS, "gjr", "t_pool", h=1)
    cut = pd.Timestamp("2016-06-15")
    shocked = {k: v.copy() for k, v in mk.items()}
    for v in shocked.values():
        v.loc[v.index > cut, "ret"] *= 3
    fc2, _, ins2 = scale.walk_forward(shocked, YEARS, keep_insample=True)
    feats2 = {n: learned.features(df) for n, df in shocked.items()}
    alt, p1 = learned.learned_forecasts(shocked, fc2, ins2, feats2, YEARS, "gjr", "t_pool", h=1)
    cols = [c for c in base[("A", "long")].frame.columns if c.startswith(("var", "es"))]
    for k in base:
        b = base[k].frame.loc[:cut - pd.offsets.BDay(1), cols]
        pd.testing.assert_frame_equal(b, alt[k].frame.loc[b.index, cols])
    assert p0 == p1                                     # the 2016 fits end in 2015


def test_learned_fit_for_a_year_ignores_that_years_returns(walk):
    mk, fc, ins, feats = walk
    _, p0 = learned.learned_forecasts(mk, fc, ins, feats, [2016], "har", "t_pool", h=10)
    shocked = {k: v.copy() for k, v in mk.items()}
    for v in shocked.values():
        jan = (v.index >= "2016-01-01") & (v.index <= "2016-01-15")
        v.loc[jan, "ret"] *= 10
    fc2, _, ins2 = scale.walk_forward(shocked, [2016], keep_insample=True)
    feats2 = {n: learned.features(df) for n, df in shocked.items()}
    _, p1 = learned.learned_forecasts(shocked, fc2, ins2, feats2, [2016], "har", "t_pool", h=10)
    assert p0 == p1


def test_stress_events_look_21_days_ahead(walk):
    mk, _, _, _ = walk
    df = mk["A"]
    ev = learned.stress_events(df, 1.5)
    cal = scale.calibrated_ewma_variance(df)
    t = df.index.get_loc(pd.Timestamp("2016-03-01"))
    fwd = (df["ret"].iloc[t + 1:t + 22] ** 2).mean()
    assert ev.iloc[t] == float(fwd >= 1.5 ** 2 * cal.iloc[t])
    assert ev.iloc[-21:].isna().all()


def test_brier_skill_sees_information_and_not_noise():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2015-01-01", periods=1000)
    x = rng.standard_normal((1000, 3))
    p_true = 1 / (1 + np.exp(-(-2 + 1.5 * x)))
    y = pd.DataFrame((rng.random((1000, 3)) < p_true).astype(float), index=dates)
    base = pd.DataFrame(np.full((1000, 3), y.to_numpy().mean()), index=dates)
    good = learned.brier_skill(pd.DataFrame(p_true, index=dates), base, y)
    assert good["bss"] > 0.1 and good["ci90"][0] > 0
    noise = pd.DataFrame(np.clip(base.to_numpy() + 0.02 * rng.standard_normal((1000, 3)), 0, 1), index=dates)
    bad = learned.brier_skill(noise, base, y)
    assert bad["ci90"][0] < 0.01


def test_learned_forecast_scales_the_base_by_its_multiplier(walk):
    """The forecast applies the fitted multiplier to volatility (variance x m^2), as the fit
    does: VaR is the starting configuration's VaR times m, to first order."""
    from etr import engine
    mk, fc, ins, feats = walk
    lf, _ = learned.learned_forecasts(mk, fc, ins, feats, [2016], "gjr", "t_pool", h=1)
    bf, _ = engine.one_day(mk, fc, ins, [2016], "gjr_1", "t_pool")
    a, b = lf[("A", "long")].frame, bf[("A", "long")].frame
    c = a.index.intersection(b.index)
    np.testing.assert_allclose(a.loc[c, "var99"] / b.loc[c, "var99"], a.loc[c, "multiplier"], rtol=0.03)
    assert a.loc[c, "multiplier"].std() > 0
