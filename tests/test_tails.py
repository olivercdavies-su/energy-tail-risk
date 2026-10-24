"""Losses, forecasts and baselines on synthetic markets with known answers, plus
look-ahead tests: changing returns after a date must not change any forecast made on or
before it."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from etr import scale, tails
from etr.tails import col


def synthetic_market(n=1500, seed=0, missing=(300,)):
    """Returns with slowly varying volatility; a missing price at the given positions (that
    day NaN, the next return spans two days, as etr.returns builds them)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)
    vol = 0.02 * np.exp(0.5 * np.sin(np.arange(n) / 120))
    ret = vol * rng.standard_normal(n)
    span = np.ones(n, int)
    for m in missing:
        ret[m + 1] += ret[m]
        ret[m] = np.nan
        span[m], span[m + 1] = 0, 2
    ret[0], span[0] = np.nan, 0
    return pd.DataFrame({"ret": ret, "span": span}, index=dates)


def test_losses_and_column_names():
    assert tails.to_loss(-0.1, "long") == pytest.approx(1 - np.exp(-0.1))
    assert tails.to_loss(0.1, "short") == pytest.approx(np.exp(0.1) - 1)
    assert [col("var", a) for a in tails.ALPHAS] == ["var95", "var975", "var99"]


def test_forward_returns_alignment_and_gaps():
    df = synthetic_market()
    f1, f10 = tails.forward_returns(df, 1), tails.forward_returns(df, 10)
    t = 100
    assert f1.iloc[t] == pytest.approx(df.ret.iloc[t + 1])
    assert f10.iloc[t] == pytest.approx(df.ret.iloc[t + 1:t + 11].sum())
    assert np.isnan(f1.iloc[300]) and np.isnan(f1.iloc[299])   # no price at 300; 301 spans the gap
    assert np.isnan(f10.iloc[300]) and np.isnan(f10.iloc[290])   # no mark at the origin / at the end
    assert f10.iloc[295] == pytest.approx(np.nansum(df.ret.iloc[296:306]))  # gap inside the window is fine


def test_normal_forecast_matches_closed_form_and_its_pit_is_uniform():
    """Data drawn from the forecast itself: VaR/ES must match the lognormal closed form,
    breach rates must match alpha, and the PIT must be uniform."""
    rng = np.random.default_rng(1)
    n = 20_000
    dates = pd.bdate_range("2000-01-03", periods=n)
    sig = 0.02 * np.exp(rng.normal(0, 0.3, n))
    var_fc = pd.Series(sig ** 2, index=dates)
    ret = np.r_[np.nan, sig[:-1] * rng.standard_normal(n - 1)]       # r_{t+1} ~ N(0, var_fc_t)
    df = pd.DataFrame({"ret": ret, "span": np.r_[0, np.ones(n - 1, int)]}, index=dates)
    for side in tails.SIDES:
        f = tails.scale_shape_forecast(df, var_fc, side).frame
        s = np.sqrt(var_fc.reindex(f.index))
        a = 0.025
        if side == "long":
            q = s * stats.norm.ppf(a)
            es = 1 - np.exp(s ** 2 / 2) * stats.norm.cdf(q / s - s) / a
        else:
            q = s * stats.norm.isf(a)
            es = np.exp(s ** 2 / 2) * stats.norm.sf(q / s - s) / a - 1
        np.testing.assert_allclose(f["es975"], es, rtol=1e-8)
        assert np.mean(f["loss"] > f["var99"]) == pytest.approx(0.01, abs=0.0025)
        assert stats.kstest(f["u"], "uniform").pvalue > 0.01
        # The PIT and the VaR must describe the same tail: a breach is exactly u < alpha.
        np.testing.assert_array_equal(f["loss"] > f["var99"], f["u"] < 0.01)


def test_sampler_draws_from_the_forecast():
    df = synthetic_market()
    v = scale.ewma_variance(df)
    fc = tails.scale_shape_forecast(df, v, "short", h=10)
    draws = fc.sample(np.random.default_rng(2), 4000)
    assert draws.shape == (4000, len(fc.frame))
    assert np.mean(draws > fc.frame["var95"].to_numpy()) == pytest.approx(0.05, abs=0.003)
    sub = fc.between("2012-01-01", "2012-12-31")
    assert sub.sample(np.random.default_rng(3), 10).shape == (10, len(sub.frame))


def test_historical_simulation_order_statistics():
    """With a 500-day window: VaR99 is the 6th largest loss, ES99 the mean of the 5
    largest, ES97.5 the Acerbi-Tasche average (12 largest + half the 13th) / 12.5."""
    df = synthetic_market(n=700, missing=())
    fc = tails.historical_forecast(df, "long").frame
    t = fc.index[0]
    i = df.index.get_loc(t)
    w = np.sort(tails.to_loss(df.ret.iloc[i - 499:i + 1].to_numpy(), "long"))[::-1]
    row = fc.iloc[0]
    assert row["var99"] == pytest.approx(w[5])
    assert row["es99"] == pytest.approx(w[:5].mean())
    assert row["es975"] == pytest.approx((w[:12].sum() + 0.5 * w[12]) / 12.5)
    assert row["loss"] == pytest.approx(tails.to_loss(df.ret.iloc[i + 1], "long"))
    assert row["u"] == pytest.approx(np.mean(w >= row["loss"]))
    for side in tails.SIDES:
        f = tails.historical_forecast(df, side).frame
        np.testing.assert_array_equal(f["loss"] > f["var99"], f["u"] <= 0.01)


FORECAST_COLS = [c for c in ["var95", "var975", "var99", "es95", "es975", "es99"]]


@pytest.mark.parametrize("model", ["riskmetrics", "ewma_cal", "hs"])
@pytest.mark.parametrize("h", [1, 10])
def test_no_lookahead(model, h):
    """Perturb (or delete) every return after a cut-off: VaR and ES at origins up to the
    cut-off must not move. Their realised loss legitimately does (it is the future)."""
    df = synthetic_market(n=1800)
    cut = df.index[1500]
    shocked = df.copy()
    late = shocked.index > cut
    shocked.loc[late, "ret"] *= np.exp(np.random.default_rng(4).normal(0, 1, late.sum()))

    def run(d):
        if model == "hs":
            return tails.historical_forecast(d, "long", h).frame
        v = scale.ewma_variance(d) if model == "riskmetrics" else scale.calibrated_ewma_variance(d)
        return tails.scale_shape_forecast(d, v, "long", h).frame

    base, pert = run(df), run(shocked)
    common = base.index[base.index <= cut - pd.offsets.BDay(h + 1)]
    pd.testing.assert_frame_equal(base.loc[common, FORECAST_COLS], pert.loc[common, FORECAST_COLS])
    v0, v1 = scale.calibrated_ewma_variance(df), scale.calibrated_ewma_variance(df.loc[:cut])
    pd.testing.assert_series_equal(v0.loc[:cut], v1)


def test_calibration_uses_only_earlier_years():
    df = synthetic_market(n=1500)
    v = scale.ewma_variance(df)
    c = scale.calibration_scale(df, v)
    y = 2013
    ratio = (scale.daily_variance(df) / v.shift(1)).dropna()
    assert c[c.index.year == y].iloc[0] == pytest.approx(ratio[ratio.index.year < y].mean())
    assert c[c.index.year == y].nunique() == 1
    assert c[c.index.year == 2010].isna().all()            # no earlier year to learn from
