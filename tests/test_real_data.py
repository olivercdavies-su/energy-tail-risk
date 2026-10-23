"""Regression checks on the licensed Bloomberg data, skipped when the workbook is absent
(e.g. on a public clone). Thresholds sit just below the previous project's verification
results, so a change that breaks the roll schedule fails here. Adapted from
energy-vol-targeting/tests/test_real_data.py."""
import numpy as np
import pytest

import etr
from etr.markets import MARKETS

try:
    etr.raw_workbook()
    HAVE_DATA = True
except FileNotFoundError:
    HAVE_DATA = False

pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="Bloomberg workbook not available")


@pytest.fixture(scope="module")
def long():
    from etr.returns import load_long
    return load_long()


@pytest.mark.parametrize("name, min_share", [("NBP", 0.96), ("HH", 0.99), ("BRENT", 0.99), ("EUA", 0.97), ("TTF", 0.95)])
def test_open_interest_jumps_on_rule_switch_days(long, name, min_share):
    from etr.returns import event_offsets, market_returns, oi_jump, wide
    m = MARKETS[name]
    out = market_returns(long, m)
    oi = wide(long, "OPEN_INT", [m.ticker(1)])[m.ticker(1)]
    offs = event_offsets(out.index[out.front_switch], oi_jump(oi, signed=(name != "EUA")), out.index).dropna()
    assert (offs == 0).mean() >= min_share


@pytest.mark.parametrize("name", list(MARKETS))
def test_wrong_schedules_are_worse(long, name):
    from etr.returns import market_returns
    m = MARKETS[name]
    base = market_returns(long, m)
    pos = base.index.get_indexer(base.index[base.front_switch])

    def ss(o):
        r2 = (o.ret ** 2).fillna(0.0)
        return sum(r2.iloc[max(p - 3, 0):p + 4].sum() for p in pos)

    for shift in (-1, 1):
        assert ss(market_returns(long, m, ltd_shift=shift)) > ss(base)


def test_held_contract_is_always_m1_or_m2_and_returns_are_finite(long):
    from etr.returns import market_returns
    for m in MARKETS.values():
        out = market_returns(long, m)
        assert set(out.tenor) <= {1, 2}
        assert np.isfinite(out.ret.dropna()).all()
        assert out.ret.isna().sum() <= 4      # first day + a few partial-holiday rows
