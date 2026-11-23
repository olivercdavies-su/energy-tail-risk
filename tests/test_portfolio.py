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


def test_joint_fhs_closed_markets_do_not_move(setup):
    """A book entirely in HH: on a day HH is closed the simulated move is zero."""
    R, U, open_, W, fc, ins = setup
    hh = pd.Series({"HH": 1.0}).reindex(pf.ORDER, fill_value=0.0)
    f = pf.joint_fhs(R, fc, ins, [2015], "ewma", {"hh": hh})[("hh", "long")]
    fr = f.frame
    closed_next = [d for d in fr.index if not open_.loc[U.index[U.index.get_loc(d) + 1], "HH"]]
    assert closed_next
    for d in closed_next:
        assert fr.loc[d, "var99"] == pytest.approx(0.0, abs=1e-12)


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
