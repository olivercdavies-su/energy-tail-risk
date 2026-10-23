"""Exchange expiry rules and calendars. Adapted from energy-vol-targeting/tests/test_expiry.py."""
import pandas as pd
import pytest

from etr import expiry as ex
from etr.markets import MARKETS

T = pd.Timestamp
M = lambda s: pd.Period(s, "M")  # noqa: E731

HOLIDAYS = [T(d) for d in ["2015-12-25", "2015-12-28", "2016-01-01",
                           "2023-12-25", "2023-12-26", "2024-01-01",
                           "2024-12-25", "2024-12-26", "2025-01-01"]]


def calendar(holidays=HOLIDAYS):
    days = pd.bdate_range("2009-01-01", "2027-12-31")
    return days[~days.isin(holidays)]


CAL = calendar()


def test_bday_helpers():
    assert ex.bday_before(CAL, T("2024-01-02"), 1) == T("2023-12-29")   # skips 1 Jan holiday
    assert ex.bday_before(CAL, T("2023-12-27"), 1) == T("2023-12-22")   # skips 25-26 Dec
    assert ex.bday_before(CAL, T("2024-01-06"), 1) == T("2024-01-05")   # date not a bday itself
    assert ex.last_bday_of_month(CAL, M("2024-03")) == T("2024-03-29")
    assert ex.is_bday(CAL, T("2024-01-02")) and not ex.is_bday(CAL, T("2024-01-01"))
    with pytest.raises(ValueError):
        ex.bday_before(CAL, T("2024-01-02"), 0)


@pytest.mark.parametrize("rule, month, expected", [
    # Published expiries that are well documented:
    ("nymex_ng", "2024-01", "2023-12-27"),   # NG F24: 3 bdays before 1 Jan (29, 28, 27 Dec)
    ("nymex_cl", "2020-05", "2020-04-21"),   # CL K20, the negative-price contract: 25 Apr was a Saturday
    ("nymex_cl", "2024-02", "2024-01-22"),   # CL G24: 25 Jan a Thursday -> 3 bdays before
    ("ice_brent", "2024-03", "2024-01-31"),  # new rule: last bday of second month before delivery
    # Rule mechanics:
    ("ice_gas", "2024-02", "2024-01-30"),    # 2 bdays before Thu 1 Feb
    ("ice_gas", "2024-01", "2023-12-28"),    # 2 bdays before Mon 1 Jan: 29 Dec, then 28 Dec
    ("ice_brent", "2016-02", "2016-01-14"),  # old rule: 15 days before 1 Feb = Sun 17 Jan -> 2 bdays back
    ("ice_brent", "2016-03", "2016-01-29"),  # first contract under the new rule
    ("ice_eua", "2026-09", "2026-09-28"),    # last Monday, no holidays after it
    ("ice_eua", "2023-12", "2023-12-18"),    # last Monday is Christmas -> penultimate Monday
    ("ice_eua", "2024-12", "2024-12-16"),    # 30 Dec fails (1 Jan), 23 Dec fails (25 Dec) -> 16 Dec
    ("ice_eua", "2015-12", "2015-12-14"),    # 28 Dec is a holiday, 21 Dec fails (25 Dec) -> 14 Dec
])
def test_rules(rule, month, expected):
    assert ex.RULES[rule](M(month), CAL) == T(expected)


def test_brent_rule_change_gives_two_january_2016_expiries():
    ltd = ex.ltd_table(MARKETS["BRENT"], CAL, T("2015-10-01"), T("2016-03-01"))
    jan16 = ltd[(ltd >= "2016-01-01") & (ltd <= "2016-01-31")]
    assert list(jan16.index.astype(str)) == ["2016-02", "2016-03"]


def test_ltd_table_is_strictly_increasing_and_on_business_days():
    for m in MARKETS.values():
        ltd = ex.ltd_table(m, CAL, T("2015-01-01"), T("2024-12-31"))
        assert ltd.is_monotonic_increasing and not ltd.duplicated().any()
        assert all(ex.is_bday(CAL, d) for d in ltd)
        assert set(ltd.index.month) <= set(m.cycle)


def test_out_of_range_raises_instead_of_clamping():
    with pytest.raises(ValueError):
        ex.bday_before(CAL, T("2030-01-01"), 1)
    with pytest.raises(ValueError):
        ex.last_bday_of_month(CAL, M("2030-01"))


def test_make_calendar_pads_with_weekdays():
    obs = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-07"])  # 6 Jan missing (holiday)
    cal = ex.make_calendar(obs, pad_years=1)
    assert T("2020-01-06") not in cal              # observed gap respected inside the sample
    assert T("2020-01-08") in cal and T("2019-12-31") in cal   # weekdays outside the sample
    assert T("2020-01-11") not in cal              # Saturday


def test_business_calendar_removes_uk_bank_holidays_optionally_from_a_date():
    trading = pd.bdate_range("2015-01-01", "2021-12-31")     # exchange open on bank holidays
    full = ex.business_calendar(trading, "GB")
    assert T("2020-08-31") not in full and T("2015-08-31") not in full
    assert T("2020-08-28") in full
    later = ex.business_calendar(trading, "GB", since="2018-01-01")
    assert T("2015-08-31") in later and T("2020-08-31") not in later
    assert ex.business_calendar(trading, None) is trading


def test_brent_new_year_clause():
    # Feb-17: last bday of Dec-16 is Fri 30 Dec, the bday before New Year -> Thu 29 Dec.
    assert ex.ice_brent(M("2017-02"), CAL) == T("2016-12-29")
    # Feb-25: last bday of Dec-24 is Tue 31 Dec -> Mon 30 Dec.
    assert ex.ice_brent(M("2025-02"), CAL) == T("2024-12-30")
    # Other months are untouched.
    assert ex.ice_brent(M("2025-03"), CAL) == T("2025-01-31")


def test_brent_uses_uk_business_days():
    trading = pd.bdate_range("2019-01-01", "2021-12-31")
    bcal = ex.business_calendar(trading, "GB")
    # Oct-20 contract: last bday of Aug-20. 31 Aug was a bank holiday -> Fri 28 Aug.
    assert ex.ice_brent(M("2020-10"), bcal) == T("2020-08-28")
    assert ex.ice_brent(M("2020-10"), trading) == T("2020-08-31")


def test_overrides_are_applied():
    ltd = ex.ltd_table(MARKETS["HH"], CAL, T("2010-06-01"), T("2011-06-01"))
    assert ltd[M("2010-12")] == T("2010-11-24") and ltd[M("2011-01")] == T("2010-12-28")
