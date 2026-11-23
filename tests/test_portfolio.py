"""Portfolios: the registered books, union-calendar returns, and joint filtered historical
simulation (that it keeps same-day dependence, respects closed markets and looks only back)."""
import numpy as np
import pandas as pd
import pytest

from etr import portfolio as pf
from etr import scale, tails
from test_scale import gjr_market

YEARS = [2015, 2016]


def synthetic_markets(rho=0.95, seed=0, holidays=True):
    """Six GJR markets; TTF/NBP and BRENT/WTI built from shared shocks with correlation rho.
    With holidays, HH and WTI (US) are closed on a few days the others trade."""
    base = {n: gjr_market(seed=seed + k) for k, n in enumerate(pf.ORDER)}
    for a, b in (("TTF", "NBP"), ("BRENT", "WTI")):
        base[b] = base[b].copy()
        base[b]["ret"] = rho * base[a]["ret"] + np.sqrt(1 - rho ** 2) * base[b]["ret"]
    if holidays:
        closed = base["HH"].index[100::250]
        for n in ("HH", "WTI"):
            df = base[n].drop(closed).copy()
            base[n] = df
    return base


@pytest.fixture(scope="module")
def setup():
    R = synthetic_markets()
    U, open_ = pf.union_returns(R)
    W = pf.books(U, open_)
    fc, _, ins = scale.walk_forward(R, YEARS, keep_insample=True)
    return R, U, open_, W, fc, ins


def test_books_follow_the_protocol(setup):
    _, U, open_, W, _, _ = setup
    vol = pf.market_vols(U, open_)
    for b in ("P1", "P3", "P4", "P5", "R01", "R50"):
        assert W[b].abs().sum() == pytest.approx(1.0)
    np.testing.assert_allclose(W["P1"] * vol, (W["P1"] * vol).iloc[0])          # equal risk: w ~ 1/vol
    pd.testing.assert_series_equal(W["P2"], -W["P1"])
    assert W["P3"]["TTF"] * vol["TTF"] == pytest.approx(-W["P3"]["NBP"] * vol["NBP"])
    assert set(W["P4"][W["P4"] != 0].index) == {"BRENT", "WTI"} and W["P4"]["BRENT"] > 0 > W["P4"]["WTI"]
    gas = W["P5"][["TTF", "NBP"]]
    oil = W["P5"][["BRENT", "WTI"]]
    assert (gas > 0).all() and (oil < 0).all()
    a, b = pf.VOL_WINDOW
    sel = (U.index >= a) & (U.index <= b)
    s_gas = (np.expm1(U[sel]) * gas.reindex(pf.ORDER, fill_value=0)).sum(axis=1).std()
    s_oil = (np.expm1(U[sel]) * oil.reindex(pf.ORDER, fill_value=0)).sum(axis=1).std()
    assert s_gas == pytest.approx(s_oil, rel=1e-9)                                 # legs carry equal P&L risk
    W2 = pf.books(U, open_)
    pd.testing.assert_series_equal(W["R17"], W2["R17"])                           # fixed seed


def test_union_calendar_and_book_return(setup):
    R, U, open_, W, _, _ = setup
    closed = R["TTF"].index.difference(R["HH"].index)[0]
    assert U.loc[closed, "HH"] == 0 and not open_.loc[closed, "HH"]
    nxt = R["HH"].index[R["HH"].index > closed][0]
    assert U.loc[nxt, "HH"] == R["HH"].loc[nxt, "ret"]                           # the gap lands on reopening
    w = W["R03"]
    d = U.index[500]
    assert pf.book_return(U, w).loc[d, "ret"] == pytest.approx(np.log1p(np.sum(w * np.expm1(U.loc[d]))))


def test_book_loss_compounds_daily_returns(setup):
    _, U, _, W, _, _ = setup
    b = pf.book_return(U, W["P1"])
    loss = tails.to_loss(tails.forward_returns(b, 10), "long")
    t = 300
    p = np.expm1(b["ret"].iloc[t + 1:t + 11])
    assert loss.iloc[t] == pytest.approx(1 - np.prod(1 + p))


def test_joint_fhs_keeps_same_day_dependence():
    """Two markets that move almost together: the hedged spread's simulated risk must be
    far below the legs' (it would be near sqrt(2) x a leg if the markets were resampled
    independently)."""
    R = synthetic_markets(rho=0.99, holidays=False)
    U, open_ = pf.union_returns(R)
    fc, _, ins = scale.walk_forward(R, [2016], keep_insample=True)
    spread = pd.Series({"TTF": 0.5, "NBP": -0.5}).reindex(pf.ORDER, fill_value=0.0)
    leg = pd.Series({"TTF": 1.0}).reindex(pf.ORDER, fill_value=0.0)
    f = pf.joint_fhs(R, fc, ins, [2016], "ewma", {"spread": spread, "leg": leg})
    v_spread, v_leg = f[("spread", "long")].frame["var99"].mean(), f[("leg", "long")].frame["var99"].mean()
    assert v_spread < 0.25 * v_leg


def test_closed_markets_do_not_move_in_a_simulation():
    """Same seed, same draws: changing the weight of a market that is closed on every
    simulated day changes nothing; once it opens, it does."""
    rng_pool = np.random.default_rng(3)
    pool = rng_pool.standard_normal((500, 3))
    sig = np.full((4, 3), 0.02)
    opens = np.ones((4, 5, 3), bool)
    opens[:, :, 0] = False                                        # market 0 closed throughout
    W1 = np.array([[0.0], [0.5], [0.5]])
    W2 = np.array([[0.9], [0.5], [0.5]])
    a = pf.simulate_log_nav(sig, opens, pool, W1, np.random.default_rng(1), 300)
    b = pf.simulate_log_nav(sig, opens, pool, W2, np.random.default_rng(1), 300)
    np.testing.assert_array_equal(a, b)
    opens[:, 2, 0] = True
    c = pf.simulate_log_nav(sig, opens, pool, W2, np.random.default_rng(1), 300)
    assert not np.allclose(a, c)


def test_books_live_on_their_own_markets_calendar(setup):
    """A book of US markets has no day on which all its markets are closed, so it never
    gets a certain-zero loss (whose ES would be 0); its returns are its market's own."""
    R, U, open_, W, fc, ins = setup
    hh = pd.Series({"HH": 1.0}).reindex(pf.ORDER, fill_value=0.0)
    cal = pf.book_calendar(open_, hh)
    assert cal.equals(R["HH"].index[R["HH"]["ret"].notna()])
    b = pf.book_return(U, hh, open_)
    np.testing.assert_allclose(b["ret"], R["HH"]["ret"].reindex(cal))
    f = pf.joint_fhs(R, fc, ins, [2015], "ewma", {"hh": hh})[("hh", "long")].frame
    assert (f["es975"] > 0).all() and (f["var99"] > 0).all()


def test_joint_fhs_has_no_lookahead(setup):
    R, U, open_, W, fc, ins = setup
    books = {b: W[b] for b in ("P1", "P3")}
    base = pf.joint_fhs(R, fc, ins, YEARS, "gjr", books, h=10, scale_col="gjr_sqrt10", n_paths=500)
    cut = pd.Timestamp("2016-06-15")
    shocked = {k: v.copy() for k, v in R.items()}
    for v in shocked.values():
        late = v.index > cut
        v.loc[late, "ret"] = v.loc[late, "ret"] * 3
    fc2, _, ins2 = scale.walk_forward(shocked, YEARS, keep_insample=True)
    alt = pf.joint_fhs(shocked, fc2, ins2, YEARS, "gjr", books, h=10, scale_col="gjr_sqrt10", n_paths=500)
    cols = [c for c in base[("P1", "long")].frame.columns if c.startswith(("var", "es"))]
    for k in base:
        b = base[k].frame.loc[:cut, cols]
        assert len(b) > 300
        pd.testing.assert_frame_equal(b, alt[k].frame.loc[b.index, cols])


def test_residual_pool_ignores_the_forecast_years_returns(setup):
    """Shock the first two weeks of 2016: the joint residual pool for 2016 must not move."""
    R, _, _, _, _, ins = setup
    p0 = pf.residual_vectors(R, ins, 2016, "gjr_1")
    shocked = {k: v.copy() for k, v in R.items()}
    for v in shocked.values():
        jan = (v.index >= "2016-01-01") & (v.index <= "2016-01-15")
        v.loc[jan, "ret"] *= 10
    _, _, ins2 = scale.walk_forward(shocked, [2016], keep_insample=True)
    np.testing.assert_array_equal(p0, pf.residual_vectors(shocked, ins2, 2016, "gjr_1"))
    assert p0.shape[1] == 6 and np.isfinite(p0).all()


def test_books_are_forecast_long_only(setup):
    """A book is one position; its 'short side' would be a different book (P1 short is P2),
    so the engine can be asked for the long side only."""
    from etr import engine
    R, _, _, _, fc, ins = setup
    sub = {k: R[k] for k in ("TTF", "NBP")}
    one, _ = engine.one_day(sub, fc, ins, YEARS, "ewma_1", "t_pool", sides=("long",))
    ten, _ = engine.ten_day(sub, fc, ins, YEARS, "ewma", "t_pool", "term", sides=("long",))
    assert set(one) == set(ten) == {("TTF", "long"), ("NBP", "long")}
