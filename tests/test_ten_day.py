"""10-day engine: square-root-of-time and term-structure configurations, GJR path
simulation, and their timing."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from etr import engine, scale, tails
from test_scale import gjr_market

YEARS = [2015, 2016]


@pytest.fixture(scope="module")
def walk():
    mk = {"A": gjr_market(seed=21), "B": gjr_market(seed=22, params=(0.05, 0.03, 0.08, 0.90, 6.0))}
    fc, diag, ins = scale.walk_forward(mk, YEARS, keep_insample=True)
    return mk, fc, diag, ins


def test_ten_day_residuals_have_mean_square_one(walk):
    mk, _, _, ins = walk
    z = engine.residuals_10(mk["A"], ins[("A", 2016, "har_10")], pd.Timestamp("2015-12-31"))
    assert np.mean(z ** 2) == pytest.approx(1.0, rel=1e-9)


def test_sqrt10_is_the_one_day_engine_scaled_in_log_space(walk):
    """VaR10 = 1 - exp(sqrt(10) sigma1 q) where VaR1 = 1 - exp(sigma1 q): the same shape
    and the 1-day sigma, so log(1 - VaR) scales exactly by sqrt(10)."""
    mk, fc, _, ins = walk
    one, _ = engine.one_day(mk, fc, ins, YEARS, "gjr_1", "t_pool")
    ten, _ = engine.ten_day(mk, fc, ins, YEARS, "gjr", "t_pool", "sqrt10")
    a, b = one[("A", "long")].frame, ten[("A", "long")].frame
    common = a.index.intersection(b.index)
    np.testing.assert_allclose(np.log1p(-b.loc[common, "var99"]), np.sqrt(10) * np.log1p(-a.loc[common, "var99"]), rtol=1e-10)


def test_gjr_paths_match_the_analytic_variance_and_the_leverage_direction():
    """With a normal shock pool the simulated 10-day variance matches the analytic GJR
    term structure; positive gamma (oil: falls raise vol) skews 10-day returns left,
    negative gamma (gas) skews them right."""
    rng = np.random.default_rng(0)
    pool = rng.standard_normal(200_000)
    for gamma, sign in ((0.10, -1), (-0.06, 1)):
        p = pd.Series({"omega": 0.04, "alpha[1]": 0.08, "gamma[1]": gamma, "beta[1]": 0.88, "nu": 8.0})
        s1 = np.full(4, 4.0 / 1e4)                                   # 2% daily vol
        R = engine.gjr_path_returns(s1, p, pool, rng, n_paths=100_000)
        w, a, b = p["omega"] / 1e4, p["alpha[1]"] + gamma / 2, p["beta[1]"]
        path, v = [], s1[0]
        for _ in range(10):
            path.append(v)
            v = w + (a + b) * v
        assert R.var(axis=1).mean() == pytest.approx(sum(path), rel=0.02)
        assert np.sign(stats.skew(R.ravel())) == sign


def test_simulated_forecast_breach_equals_small_pit():
    rng = np.random.default_rng(1)
    sims = 0.05 * rng.standard_t(5, (300, 2000))
    realised = pd.Series(0.05 * rng.standard_t(5, 300), index=pd.bdate_range("2016-01-04", periods=300))
    for side in tails.SIDES:
        f = engine.simulated_forecast(realised, sims, side).frame
        np.testing.assert_array_equal(f["loss"] > f["var99"], f["u"] <= 0.01)


@pytest.mark.parametrize("kind", ["term", "paths"])
def test_ten_day_has_no_lookahead(walk, kind):
    """Shock every return after mid-2016: VaR and ES at earlier origins must not move."""
    mk, fc, diag, ins = walk
    cut = pd.Timestamp("2016-06-15")
    shocked = {k: v.copy() for k, v in mk.items()}
    for v in shocked.values():
        late = v.index > cut
        v.loc[late, "ret"] = v.loc[late, "ret"] * 3
    fc2, diag2, ins2 = scale.walk_forward(shocked, YEARS, keep_insample=True)

    def run(m, f, d, i):
        if kind == "term":
            return engine.ten_day(m, f, i, YEARS, "comb", "gpd_pool", "term")[0]
        return engine.gjr_paths(m, d, YEARS, pooled=True, n_paths=500)[0]

    base, alt = run(mk, fc, diag, ins), run(shocked, fc2, diag2, ins2)
    cols = [c for c in base[("A", "long")].frame.columns if c.startswith(("var", "es"))]
    for k in base:
        b = base[k].frame.loc[:cut, cols]
        b = b[b.index.isin(alt[k].frame.index)]
        assert len(b) > 300
        pd.testing.assert_frame_equal(b, alt[k].frame.loc[b.index, cols])


def test_ten_day_shapes_ignore_the_forecast_years_returns(walk):
    """Shock the first two weeks of 2016: the 10-day shapes fitted for 2016 must not move
    (a training window ending in late December would otherwise run into January)."""
    mk, fc, _, ins = walk
    _, p0 = engine.ten_day(mk, fc, ins, [2016], "gjr", "gpd_mkt", "term")
    shocked = {k: v.copy() for k, v in mk.items()}
    for v in shocked.values():
        jan = (v.index >= "2016-01-01") & (v.index <= "2016-01-15")
        v.loc[jan, "ret"] *= 10
    fc2, _, ins2 = scale.walk_forward(shocked, [2016], keep_insample=True)
    _, p1 = engine.ten_day(shocked, fc2, ins2, [2016], "gjr", "gpd_mkt", "term")
    assert p0 == p1


def test_ten_day_complexity():
    assert engine.complexity("gjr_sqrt10|t_pool") == engine.complexity("gjr_1|t_pool") == 4
    assert engine.complexity("gjr_10|t_pool") == 5 and engine.complexity("gjr_paths|fhs_pool") == 7


def test_gjr_paths_carry_the_ten_day_calibration(walk):
    """The simulated 10-day variance matches the calibrated GJR 10-day forecast on average
    (the paths are scaled by the same point-in-time calibration)."""
    mk, fc, diag, _ = walk
    f, _ = engine.gjr_paths(mk, diag, [2016], pooled=False, n_paths=4000)
    fr = f[("A", "long")].frame
    draws = f[("A", "long")].sample(np.random.default_rng(2), 4000)          # losses ~ -R for small R
    ratio = draws.var(axis=0) / fc["A"]["gjr_10"].reindex(fr.index).to_numpy()
    assert np.mean(ratio) == pytest.approx(1.0, abs=0.05)
