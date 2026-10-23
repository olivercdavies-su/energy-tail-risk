"""Roll logic on a synthetic market where every contract's true price path is known.

Each contract carries a large level offset (10% per month along the curve), so any
return that mixes two contracts is off by about 10%, far above the daily noise. Adapted from energy-vol-targeting/tests/test_rolls.py.
"""
import numpy as np
import pandas as pd
import pytest

from etr.expiry import ice_gas
from etr.returns import build_returns, roll_schedule

HOLIDAYS = pd.DatetimeIndex(["2020-04-10", "2020-04-13", "2020-12-25", "2021-01-01"])
CAL = pd.bdate_range("2019-06-01", "2022-06-30")
CAL = CAL[~CAL.isin(HOLIDAYS)]
DATES = CAL[(CAL >= "2020-01-02") & (CAL <= "2021-06-30")]
MONTHS = pd.period_range("2019-12", "2022-03", freq="M")
LTD = pd.Series([ice_gas(m, CAL) for m in MONTHS], index=MONTHS)


def true_log_prices(seed=0):
    """Contract x date log prices: common random walk + contract offset + small idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.02, len(DATES)))
    offset = 0.10 * np.arange(len(MONTHS))
    idio = rng.normal(0, 0.003, (len(MONTHS), len(DATES)))
    return pd.DataFrame(common[None, :] + offset[:, None] + idio + np.log(20), index=MONTHS, columns=DATES)


def generic_from_truth(truth, tenors=(1, 2, 3)):
    """What Bloomberg would show: M_k on date t is the k-th contract with LTD >= t."""
    front = np.searchsorted(LTD.values, DATES.values, side="left")
    data = {k: [np.exp(truth.iat[front[i] + k - 1, i]) for i in range(len(DATES))] for k in tenors}
    return pd.DataFrame(data, index=DATES)


@pytest.fixture
def case():
    truth = true_log_prices()
    return truth, generic_from_truth(truth)


def check_against_truth(out, truth):
    """Every non-NaN return must equal the true move of the held contract over its span."""
    pos = {d: i for i, d in enumerate(DATES)}
    checked = 0
    for d, row in out.dropna(subset=["ret"]).iterrows():
        i = pos[d]
        c = pd.Period(row.contract, "M")
        expected = truth.at[c, DATES[i]] - truth.at[c, DATES[i - row.span]]
        assert row.ret == pytest.approx(expected, abs=1e-12), d
        checked += 1
    return checked


def test_returns_match_true_contract_returns(case):
    truth, panel = case
    out = build_returns(panel, LTD, CAL, roll_offset=5)
    assert check_against_truth(out, truth) == len(DATES) - 1   # every day except the first
    assert set(out.tenor) <= {1, 2}
    assert (out.span.iloc[1:] == 1).all()


def test_rolls_happen_offset_days_before_ltd(case):
    _, panel = case
    out = build_returns(panel, LTD, CAL, roll_offset=5)
    sched = roll_schedule(LTD, CAL, 5)
    expected = sched[(sched >= DATES[0]) & (sched <= DATES[-1])]
    assert list(out.index[out.roll]) == list(expected)
    # After each roll we hold M2; on Bloomberg's switch day that contract becomes M1.
    for d in expected:
        i = out.index.get_loc(d)
        assert out.tenor.iloc[i] == 1 and out.tenor.iloc[i + 1] == 2
        assert out.px_next.iloc[i] != out.px.iloc[i]


def test_naive_returns_are_contaminated_on_switch_days(case):
    _, panel = case
    out = build_returns(panel, LTD, CAL)
    sw = out.front_switch
    assert sw.sum() == 18                                # one per month, Jan-20..Jun-21
    assert (out.ret_naive[sw] > 0.05).all()              # the 10% calendar spread
    assert out.ret[sw].abs().max() < 0.1                 # adjusted: ordinary daily moves
    assert (out.ret_naive[~sw] - out.ret[~sw]).abs().iloc[1:].max() < 0.02  # small idio wedge only


def test_missing_price_spans_the_gap(case):
    truth, panel = case
    out0 = build_returns(panel, LTD, CAL)
    d = out0.index[(out0.tenor == 1) & ~out0.roll & ~out0.front_switch][40]
    panel = panel.copy()
    panel.loc[d, 1] = np.nan                             # held contract (M1) has no price
    out = build_returns(panel, LTD, CAL)
    i = out.index.get_loc(d)
    assert np.isnan(out.ret.iloc[i])
    assert out.span.iloc[i + 1] == 2
    check_against_truth(out, truth)
    # Cumulative P&L is unchanged by the gap.
    assert out.ret.sum() == pytest.approx(out0.ret.sum(), abs=1e-12)


def test_negative_expiring_contract_does_not_touch_held_m2(case):
    truth, panel = case
    panel = panel.copy()
    ltd_day = LTD[LTD > DATES[100]].iloc[0]
    panel.loc[ltd_day - pd.offsets.BDay(1), 1] = -37.63  # expiring M1 goes negative
    out = build_returns(panel, LTD, CAL)
    assert np.isfinite(out.ret.iloc[1:]).all()
    check_against_truth(out, truth)


def test_no_lookahead(case):
    """Changing prices after date T must not change anything up to and including T."""
    truth, panel = case
    cut = DATES[200]
    out = build_returns(panel, LTD, CAL)
    shocked = panel.copy()
    shocked.loc[shocked.index > cut] *= np.exp(np.random.default_rng(1).normal(0, 0.5, shocked.loc[shocked.index > cut].shape))
    out2 = build_returns(shocked, LTD, CAL)
    pd.testing.assert_frame_equal(out.loc[:cut], out2.loc[:cut])
    # Truncating the data entirely gives the same history too.
    out3 = build_returns(panel.loc[:cut], LTD, CAL)
    pd.testing.assert_frame_equal(out.loc[:cut], out3)


def test_offset_too_large_is_rejected():
    with pytest.raises(ValueError):
        roll_schedule(LTD, CAL, roll_offset=25)
