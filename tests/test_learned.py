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


# ---- The small network (design: PROJECT_LOG session 7) ----

@pytest.fixture
def small_net(monkeypatch):
    """One member and at most 300 steps, so the walk-forward tests run in seconds."""
    monkeypatch.setattr(learned, "NN_MEMBERS", 1)
    monkeypatch.setattr(learned, "NN_MAX_STEPS", 300)


def simulated(n, rng, f, d=3, shape=tails.StudentT(5)):
    X = rng.standard_normal((n, d))
    sigma = np.full(n, 0.02)
    return X, sigma, sigma * np.exp(f(X)) * shape.rvs(size=n, random_state=rng)


def out_of_sample(beta, members, f, side="long", n=20_000, seed=9):
    """FZ0 of the linear model and the network on fresh data, and their log multipliers."""
    Xt, st, Rt = simulated(n, np.random.default_rng(seed), f)
    g = groups_for(tails.StudentT(5), n, side)
    lin = np.clip(np.c_[np.ones(n), Xt] @ beta, -3, 3)
    net = learned.network_eta(members, Xt)
    score = lambda eta: learned.fz0_and_grad(st * np.exp(eta), Rt, g, side)[0]          # noqa: E731
    return score(lin), score(net), lin, net, f(Xt)


@pytest.mark.parametrize("side", tails.SIDES)
def test_network_gradient_matches_finite_differences(side):
    """Backpropagation through FZ0, the tanh layer and the linear part agrees with central
    differences for every parameter (including the weight decay)."""
    rng = np.random.default_rng(4)
    n, d = 400, 3
    X, sigma, R = simulated(n, rng, lambda X: 0 * X[:, 0])
    g = groups_for(tails.StudentT(5), n, side)
    p = learned.init_network(rng.normal(0, 0.1, d + 1), d, 1)
    p["v"], p["c"] = rng.normal(0, 0.2, learned.NN_HIDDEN), rng.normal(0, 0.3, learned.NN_HIDDEN)
    _, grad = learned.network_loss_grad(p, X, sigma, R, g, side)
    eps = 1e-6
    for k in p:
        for i in np.ndindex(p[k].shape):
            up = {kk: vv.copy() for kk, vv in p.items()}
            dn = {kk: vv.copy() for kk, vv in p.items()}
            up[k][i] += eps
            dn[k][i] -= eps
            num = (learned.network_loss_grad(up, X, sigma, R, g, side)[0]
                   - learned.network_loss_grad(dn, X, sigma, R, g, side)[0]) / (2 * eps)
            assert grad[k][i] == pytest.approx(num, rel=1e-4, abs=1e-8), (k, i)


def test_untrained_network_is_the_linear_model(monkeypatch):
    """With no training steps the fitted ensemble is exactly the linear model it was given
    (fitted on all rows), whatever its random hidden weights."""
    monkeypatch.setattr(learned, "NN_MAX_STEPS", 0)
    rng = np.random.default_rng(5)
    n = 2000
    X, sigma, R = simulated(n, rng, lambda X: 0.3 * X[:, 0])
    g = groups_for(tails.StudentT(5), n, "long")
    beta = learned.fit_multiplier(X, sigma, R, g, "long")
    dates = pd.bdate_range("2010-01-01", periods=n).to_numpy()
    members, diag = learned.fit_network(X, sigma, R, g, "long", dates, dates, beta, seed=1)
    assert diag["k_star"] == " ".join(["0"] * learned.NN_MEMBERS)
    np.testing.assert_allclose(learned.network_eta(members, X), np.clip(np.c_[np.ones(n), X] @ beta, -3, 3))


def test_network_learns_a_nonlinear_effect(small_net):
    """True vol = base vol x exp(0.4 (|x| - 0.8)): bigger moves either way widen the tails.
    No linear multiplier can see it; the network does, out of sample."""
    f = lambda X: 0.4 * (np.abs(X[:, 0]) - 0.8)                                       # noqa: E731
    n = 4000
    X, sigma, R = simulated(n, np.random.default_rng(6), f)
    g = groups_for(tails.StudentT(5), n, "long")
    beta = learned.fit_multiplier(X, sigma, R, g, "long")
    dates = pd.bdate_range("2010-01-01", periods=n).to_numpy()
    members, _ = learned.fit_network(X, sigma, R, g, "long", dates, dates, beta, seed=1)
    lin, net, _, eta, truth = out_of_sample(beta, members, f)
    assert net < lin - 0.01
    assert np.corrcoef(eta, truth)[0, 1] > 0.8


def test_network_stays_near_linear_on_noise(monkeypatch):
    """With nothing to learn, early stopping keeps the network close to the linear model
    (training the full 1,000 steps on noise moves it about six times further)."""
    monkeypatch.setattr(learned, "NN_MEMBERS", 1)
    f = lambda X: 0 * X[:, 0]                                                          # noqa: E731
    n = 4000
    X, sigma, R = simulated(n, np.random.default_rng(5), f)
    g = groups_for(tails.StudentT(5), n, "short")
    beta = learned.fit_multiplier(X, sigma, R, g, "short")
    dates = pd.bdate_range("2010-01-01", periods=n).to_numpy()
    members, _ = learned.fit_network(X, sigma, R, g, "short", dates, dates, beta, seed=1)
    _, _, lin, net, _ = out_of_sample(beta, members, f, side="short")
    assert np.std(net - lin) < 0.1


def test_inner_validation_is_the_latest_dates_and_purged():
    """Early stopping validates on the latest quarter of training dates; inner training rows
    are earlier and their (10-day) targets end before validation starts."""
    days = pd.bdate_range("2012-01-02", periods=800)
    dates = np.r_[days.to_numpy(), days[100:].to_numpy()]                          # two markets
    ends = np.r_[days.shift(10).to_numpy(), days[100:].shift(10).to_numpy()]
    tr, va = learned.inner_split(dates, ends)
    start = dates[va].min()
    assert dates[tr].max() < start and ends[tr].max() < start
    assert set(np.flatnonzero(dates >= start)) == set(va)
    assert len(np.unique(dates[va])) == pytest.approx(0.25 * 800, abs=2)
    assert len(tr) + len(va) < len(dates)                                          # the purge removed rows


def test_network_forecasts_have_no_lookahead(walk, small_net):
    """Network forecasts before a shock are unchanged by it, and the fits (made at the start
    of each year) are identical."""
    mk, fc, ins, feats = walk
    base, p0 = learned.learned_forecasts(mk, fc, ins, feats, YEARS, "gjr", "t_pool", h=1, model="network")
    cut = pd.Timestamp("2016-06-15")
    shocked = {k: v.copy() for k, v in mk.items()}
    for v in shocked.values():
        v.loc[v.index > cut, "ret"] *= 3
    fc2, _, ins2 = scale.walk_forward(shocked, YEARS, keep_insample=True)
    feats2 = {n: learned.features(df) for n, df in shocked.items()}
    alt, p1 = learned.learned_forecasts(shocked, fc2, ins2, feats2, YEARS, "gjr", "t_pool", h=1, model="network")
    cols = [c for c in base[("A", "long")].frame.columns if c.startswith(("var", "es"))]
    for k in base:
        b = base[k].frame.loc[:cut - pd.offsets.BDay(1), cols]
        pd.testing.assert_frame_equal(b, alt[k].frame.loc[b.index, cols])
    assert p0 == p1
    assert all("k_star" in row for row in p0)
