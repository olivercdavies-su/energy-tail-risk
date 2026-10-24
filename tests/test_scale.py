"""Scale models on synthetic markets with known answers, and the walk-forward's timing:
nothing at an origin may depend on returns after it, and nothing fitted for a year may
depend on that year's returns. Adapted in part from energy-vol-targeting/tests/test_forecasting.py."""
import numpy as np
import pandas as pd
import pytest
from arch.univariate import GARCH, StudentsT, ZeroMean

from etr import scale

FORECAST_COLS = [f"{m}_{h}" for m in scale.MODELS for h in ("1", "sqrt10", "10")]


def gjr_market(years=7, seed=0, params=(0.05, 0.08, -0.04, 0.92, 6.0), monday=1.0, start="2010-01-04"):
    """GJR-GARCH-t returns (inverse leverage: gamma < 0) on a business-day calendar, shaped
    like etr.returns output. `monday` multiplies Monday variance (others rescaled to keep
    the average)."""
    dates = pd.bdate_range(start, periods=252 * years)
    # arch draws from the distribution's own generator, so seed that generator directly.
    sim = ZeroMean(volatility=GARCH(p=1, o=1, q=1), distribution=StudentsT(seed=np.random.default_rng(seed)))
    r = sim.simulate(list(params), len(dates), burn=500)["data"].to_numpy() / 100
    if monday != 1.0:
        mult = np.where(dates.weekday == 0, monday, (5 - monday) / 4)
        r = r * np.sqrt(mult)
    return pd.DataFrame({"ret": r, "span": 1}, index=dates)


@pytest.fixture(scope="module")
def two_markets():
    return {"A": gjr_market(seed=1), "B": gjr_market(seed=2, monday=1.6)}


@pytest.fixture(scope="module")
def forecasts(two_markets):
    return scale.walk_forward(two_markets, [2015, 2016], iv={"A": synthetic_iv(two_markets["A"])})


def synthetic_iv(df):
    """An implied vol with its own dynamics (a lagged, smoothed EWMA vol with a premium), so
    calibration cannot turn it back into the EWMA forecast."""
    return np.sqrt(252 * scale.ewma_variance(df, lam=0.97).shift(3)) * 1.15


def test_gjr_recovers_a_negative_asymmetry():
    df = gjr_market(years=14, seed=3)
    p = scale.gjr_fit(scale.daily_equivalent_returns(df))
    assert p["gamma[1]"] == pytest.approx(-0.04, abs=0.025)
    assert p["beta[1]"] == pytest.approx(0.92, abs=0.03)
    assert scale.gjr_persistence(p) == pytest.approx(0.98, abs=0.015)


def test_gjr_multi_step_matches_the_recursion():
    """E_t[s2_{t+h}] = w + (a + g/2 + b) E_t[s2_{t+h-1}] for h >= 2 (symmetric errors)."""
    df = gjr_market(years=4, seed=4)
    r = scale.daily_equivalent_returns(df)
    p = scale.gjr_fit(r.loc[:"2012-12-31"])
    paths = scale.gjr_paths(r, p, start="2013-01-01") * 1e4
    t = paths.index[50]
    s2 = scale._gjr(r).fix(p).conditional_volatility ** 2
    pos = r.index.get_loc(t)
    w, a, g, b = p["omega"], p["alpha[1]"], p["gamma[1]"], p["beta[1]"]
    e = r.iloc[pos]
    path = [w + (a + g * (e < 0)) * e ** 2 + b * s2.iloc[pos]]
    for _ in range(9):
        path.append(w + (a + g / 2 + b) * path[-1])
    np.testing.assert_allclose(paths.loc[t].to_numpy(), path, rtol=1e-6)


def test_qlike_fit_recovers_a_log_linear_variance_and_is_calibrated():
    rng = np.random.default_rng(5)
    n = 40_000
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    b = np.array([-9.0, 0.5, -0.3])
    y = np.exp(X @ b) * rng.standard_t(6, n) ** 2 / 1.5          # E[t6^2] = 1.5
    est = scale.qlike_fit(X, y)
    np.testing.assert_allclose(est, b, atol=0.05)
    assert np.mean(y / np.exp(X @ est)) == pytest.approx(1.0, abs=1e-6)


def test_weekday_factors_find_a_monday_effect_and_shrink_noise():
    cut = pd.Timestamp("2016-12-31")
    stats = {"mon": scale.weekday_stats(gjr_market(seed=6, monday=1.6), cut),
             "none": scale.weekday_stats(gjr_market(seed=7), cut)}
    f = scale.shrink_weekday(stats)
    assert f["mon"][0] > 1.3 and f["mon"][1:].max() < 1.0
    assert (f["none"] - 1).abs().max() < (stats["none"]["f_raw"] - 1).abs().max()   # shrunk
    for k, s in stats.items():
        assert np.sum(f[k] * s["share"]) == pytest.approx(1.0)


def test_weekday_ewma_beats_plain_ewma_where_the_effect_exists():
    """A market with Monday variance 2x: the weekday model has lower 1-day QLIKE over six
    forecast years, and its errors no longer depend on the weekday. (With a 1.6x effect and
    two years the gain, about 0.04 per day, is inside the noise: weekday factors from five
    years of fat-tailed data carry standard errors of 0.1-0.2.)"""
    mk = {"mon": gjr_market(years=12, seed=8, monday=2.0), "none": gjr_market(years=12, seed=9)}
    fc, _ = scale.walk_forward(mk, list(range(2016, 2022)))
    b = fc["mon"].dropna(subset=["y1", "ewma_1", "ewma_wd_1"])
    ql = lambda f: np.mean(b["y1"] / f + np.log(f))  # noqa: E731
    assert ql(b["ewma_wd_1"]) < ql(b["ewma_1"]) - 0.03
    target_wd = mk["mon"].index.to_series().shift(-1).reindex(b.index).dt.weekday
    ratio = (b["y1"] / b["ewma_wd_1"]).groupby(target_wd).mean()
    assert ratio.max() - ratio.min() < (b["y1"] / b["ewma_1"]).groupby(target_wd).mean().pipe(lambda r: r.max() - r.min())


def test_ewma_candidate_equals_baseline_b3(two_markets, forecasts):
    fc, _ = forecasts
    b3 = scale.calibrated_ewma_variance(two_markets["A"]).reindex(fc["A"].index)
    np.testing.assert_allclose(fc["A"]["ewma_1"], b3, rtol=1e-12)


def test_implied_vol_filter_and_fallback():
    idx = pd.bdate_range("2020-01-01", periods=12)
    ewma = pd.Series(0.3 ** 2 / 252, index=idx)
    iv = pd.Series(0.3, index=idx)
    iv.iloc[3] = 0.012                                              # a bad print
    iv.iloc[6:] = np.nan                                            # options data stops
    v = scale.filtered_iv_variance(iv, ewma, max_fill=2)
    assert v.iloc[3] == pytest.approx(0.3 ** 2 / 252)               # last plausible value
    assert v.iloc[7] == pytest.approx(0.3 ** 2 / 252) and np.isnan(v.iloc[8])


def test_implied_vol_model_falls_back_to_calibrated_ewma(forecasts):
    fc, _ = forecasts
    np.testing.assert_allclose(fc["B"]["iv_1"], fc["B"]["ewma_1"])           # no options in B
    assert not np.allclose(fc["A"]["iv_1"], fc["A"]["ewma_1"])


def test_combination_is_the_calibrated_mean(forecasts):
    fc, diag = forecasts
    f = fc["A"].loc["2016"]
    c = diag.query("market == 'A' and year == 2016 and model == 'comb' and h == 1")["scale_c"].iloc[0]
    np.testing.assert_allclose(f["comb_1"], c * f[["ewma_wd_1", "gjr_1", "har_1"]].mean(1, skipna=False), rtol=1e-12)


def test_term_structure_has_its_own_calibration(forecasts):
    """The square-root-of-time variant is exactly 10x the 1-day forecast; the term
    structure is not."""
    fc, _ = forecasts
    f = fc["A"]
    np.testing.assert_allclose(f["gjr_sqrt10"], 10 * f["gjr_1"])
    assert not np.allclose(f["gjr_10"], f["gjr_sqrt10"])


def test_walk_forward_has_no_lookahead(two_markets, forecasts):
    """Perturb every return after a mid-2016 cut-off: every forecast at origins up to the
    cut-off must be unchanged, in both markets."""
    base, _ = forecasts
    cut = pd.Timestamp("2016-06-15")
    shocked = {k: v.copy() for k, v in two_markets.items()}
    for v in shocked.values():
        late = v.index > cut
        v.loc[late, "ret"] = v.loc[late, "ret"] * 3 + 0.01
    alt, _ = scale.walk_forward(shocked, [2015, 2016], iv={"A": synthetic_iv(shocked["A"])})
    for k in base:
        pd.testing.assert_frame_equal(base[k].loc[:cut, FORECAST_COLS], alt[k].loc[:cut, FORECAST_COLS])
        assert not np.allclose(base[k].loc["2016-07":, "gjr_1"], alt[k].loc["2016-07":, "gjr_1"])


def test_fits_for_a_year_ignore_that_years_returns(two_markets):
    """Shock the first week of 2016: the 2016 GJR, HAR (both horizons, so the purge works),
    weekday factors and calibration scales must not move."""
    _, d0 = scale.walk_forward(two_markets, [2016])
    shocked = {k: v.copy() for k, v in two_markets.items()}
    for v in shocked.values():
        jan = (v.index >= "2016-01-01") & (v.index <= "2016-01-08")
        v.loc[jan, "ret"] *= 10
    _, d1 = scale.walk_forward(shocked, [2016])
    num = d0.select_dtypes("number").columns
    pd.testing.assert_frame_equal(d0[num], d1[num])
