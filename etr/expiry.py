"""Last trading day (LTD) of each futures contract, from the exchanges' published rules.

Copied from energy-vol-targeting (src/evt/dataprep/expiry.py), where each rule and
calendar quirk was verified against open-interest jumps and curve shifts.

Why rules and not the data: expiry dates are fixed by the contract specification and
known years in advance, so a roll schedule built from them cannot leak future
information. Detecting rolls from price jumps would use the data to decide how to
treat the data, and it fails exactly when it matters (in 2022, daily gas moves were
bigger than the calendar spreads, so jump detection missed rolls). We still run a
data-based detector, but only as an independent check (see etr.returns.oi_jump).

Business days: each rule counts business days on the relevant exchange calendar. We
start from the market's own observed trading days (dates with a settlement price), since
exchange holidays are published in advance and a trader would have known them. For the
ICE contracts (TTF, NBP, Brent, EUA) we then also remove English bank holidays. ICE often
trades and settles on those days, but the data shows they do not count as business days
for expiry: on 31 Aug 2020, for example, Bloomberg's Brent, TTF and NBP generics all
rolled a day earlier than a count including that bank holiday would give (verified with
the curve-shift and open-interest checks in the previous project). Outside the sample
we fall back to weekdays minus those holidays.
"""
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from etr.markets import Market

with warnings.catch_warnings():  # holidays 0.6x prints a versioning notice on import
    warnings.simplefilter("ignore", DeprecationWarning)
    import holidays


def make_calendar(observed, pad_years: int = 4) -> pd.DatetimeIndex:
    """Observed trading days, padded with weekdays before the first and after the last."""
    obs = pd.DatetimeIndex(observed).normalize().unique().sort_values()
    lo, hi = obs[0], obs[-1]
    before = pd.bdate_range(lo - pd.DateOffset(years=pad_years), lo - pd.Timedelta(days=1))
    after = pd.bdate_range(hi + pd.Timedelta(days=1), hi + pd.DateOffset(years=pad_years))
    return before.append(obs).append(after)


def holiday_dates(country: str, years: range) -> pd.DatetimeIndex:
    """Public holidays: "GB" = England and Wales bank holidays."""
    kw = {"subdiv": "ENG"} if country == "GB" else {}
    return pd.DatetimeIndex(sorted(holidays.country_holidays(country, years=years, **kw)))


def business_calendar(trading_cal: pd.DatetimeIndex, country: str | None,
                      since: str | None = None) -> pd.DatetimeIndex:
    """Calendar used to count days in expiry rules: trading days minus `country` holidays
    (only holidays on or after `since`, if given)."""
    if country is None:
        return trading_cal
    hol = holiday_dates(country, range(trading_cal[0].year, trading_cal[-1].year + 1))
    if since is not None:
        hol = hol[hol >= pd.Timestamp(since)]
    return trading_cal[~trading_cal.isin(hol)]


def last_bday_on_or_before(cal: pd.DatetimeIndex, date: pd.Timestamp) -> pd.Timestamp:
    return cal[cal.searchsorted(date, side="right") - 1]


def is_bday(cal: pd.DatetimeIndex, date: pd.Timestamp) -> bool:
    i = cal.searchsorted(date)
    return i < len(cal) and cal[i] == date


def bday_before(cal: pd.DatetimeIndex, date: pd.Timestamp, n: int) -> pd.Timestamp:
    """The n-th business day strictly before `date` (n=1 is the previous business day)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    i = cal.searchsorted(date, side="left") - n
    if i < 0 or date > cal[-1]:
        raise ValueError(f"{date.date()} is outside the calendar")
    return cal[i]


def last_bday_of_month(cal: pd.DatetimeIndex, month: pd.Period) -> pd.Timestamp:
    if month.end_time > cal[-1] + pd.Timedelta(days=1):
        raise ValueError(f"{month} is outside the calendar")
    return cal[cal.searchsorted(month.end_time.normalize(), side="right") - 1]


# ---- Rules. Each takes the delivery month and the calendar and returns the LTD. ----

def ice_gas(month: pd.Period, cal: pd.DatetimeIndex) -> pd.Timestamp:
    """ICE Endex TTF and ICE NBP monthly futures: trading ceases at the close of business
    two business days before the first calendar day of the delivery month."""
    return bday_before(cal, month.start_time, 2)


def nymex_ng(month: pd.Period, cal: pd.DatetimeIndex) -> pd.Timestamp:
    """NYMEX Henry Hub (NG): three business days before the first calendar day of the
    delivery month."""
    return bday_before(cal, month.start_time, 3)


BRENT_RULE_CHANGE = pd.Period("2016-03", "M")


def ice_brent(month: pd.Period, cal: pd.DatetimeIndex) -> pd.Timestamp:
    """ICE Brent. From the March 2016 contract: last business day of the second month
    before delivery (so the Mar-16 contract expired at the end of Jan-16).
    Before that: the business day before the 15th calendar day prior to the first day
    of delivery, or, if that 15th day is not a business day, the business day before
    the business day that precedes it. The switch means two contracts (Feb-16 and Mar-16)
    both expired in January 2016."""
    if month >= BRENT_RULE_CHANGE:
        ltd = last_bday_of_month(cal, month - 2)
    else:
        fifteenth = month.start_time - pd.Timedelta(days=15)
        ltd = bday_before(cal, fifteenth, 1 if is_bday(cal, fifteenth) else 2)
    return _brent_holiday_clause(ltd, cal)


def _brent_holiday_clause(ltd: pd.Timestamp, cal: pd.DatetimeIndex) -> pd.Timestamp:
    """ICE Brent: if expiry would fall on the business day immediately before Christmas Day
    or New Year's Day, it moves to the business day before that. Under the post-2016 rule
    this always applies to the February contract (last business day of December)."""
    nxt = cal[cal.searchsorted(ltd, side="right")]
    for holiday in (pd.Timestamp(ltd.year, 12, 25), pd.Timestamp(ltd.year + 1, 1, 1)):
        if ltd < holiday <= nxt:
            return bday_before(cal, ltd, 1)
    return ltd


def nymex_cl(month: pd.Period, cal: pd.DatetimeIndex) -> pd.Timestamp:
    """NYMEX WTI (CL): three business days before the 25th calendar day of the month
    before delivery; if the 25th is not a business day, three business days before the
    last business day preceding the 25th (i.e. the 4th business day before the 25th)."""
    t25 = (month - 1).start_time + pd.Timedelta(days=24)
    return bday_before(cal, t25, 3 if is_bday(cal, t25) else 4)


def ice_eua(month: pd.Period, cal: pd.DatetimeIndex) -> pd.Timestamp:
    """ICE EUA futures: the last Monday of the contract month, unless that Monday is not a
    business day or a non-business day falls in the four days after it, in which case the
    penultimate Monday. The exchange applies this repeatedly: the LTD is the latest Monday
    that is a business day with four business days after it. For December that skips both
    the New Year and the Christmas weeks (16 Dec 2024, 15 Dec 2025), which the open-interest
    collapse in MO1 confirms."""
    days = pd.date_range(month.start_time, month.end_time.normalize())
    for monday in days[days.weekday == 0][::-1]:
        following = pd.date_range(monday + pd.Timedelta(days=1), periods=4)
        if is_bday(cal, monday) and all(is_bday(cal, d) for d in following):
            return monday
    raise ValueError(f"no valid EUA expiry Monday in {month}")


RULES: dict[str, Callable[[pd.Period, pd.DatetimeIndex], pd.Timestamp]] = {
    "ice_gas": ice_gas,
    "nymex_ng": nymex_ng,
    "ice_brent": ice_brent,
    "nymex_cl": nymex_cl,
    "ice_eua": ice_eua,
}


# Expiries where the rule and Bloomberg's generics disagree and two independent checks
# (curve shift and the open-interest jump) agree on the actual date. Each is the trading
# day before the date Bloomberg's M1 switched. Cause unknown (both late 2010; the same
# calendar situations in later years follow the rule).
LTD_OVERRIDES: dict[str, dict[str, str]] = {
    "HH": {"2010-12": "2010-11-24", "2011-01": "2010-12-28"},
}


def contract_months(cycle: tuple[int, ...], first: pd.Timestamp, last: pd.Timestamp) -> pd.PeriodIndex:
    months = pd.period_range(first, last, freq="M")
    return months[np.isin(months.month, cycle)]


def ltd_table(market: Market, cal: pd.DatetimeIndex, first: pd.Timestamp, last: pd.Timestamp,
              cycle: tuple[int, ...] | None = None) -> pd.Series:
    """LTD for every listed contract that can be a generic tenor between `first` and
    `last`. Returns a Series indexed by delivery month (Period), strictly increasing."""
    cycle = cycle or market.cycle
    rule = RULES[market.expiry_rule]
    step = 12 // len(cycle)  # months between listed contracts
    # Far enough ahead that the deepest tenor on the last date is covered, plus a buffer.
    months = contract_months(cycle, first - pd.DateOffset(months=step),
                             last + pd.DateOffset(months=step * (max(market.tenors) + 1) + 3))
    ltd = pd.Series([rule(m, cal) for m in months], index=months, name="ltd")
    for month, date in LTD_OVERRIDES.get(market.name, {}).items():
        if pd.Period(month, "M") in ltd.index:
            ltd[pd.Period(month, "M")] = pd.Timestamp(date)
    if not ltd.is_monotonic_increasing or ltd.duplicated().any():
        raise ValueError(f"{market.name}: LTDs are not strictly increasing")
    return ltd
