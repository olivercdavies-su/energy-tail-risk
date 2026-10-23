"""The six markets and their contract conventions.

Adapted from energy-vol-targeting (src/evt/markets.py). The trading-cost fields are dropped
(a risk engine measures positions, it does not trade them). Excluded markets, for the
reasons found there: JKM (assessed prices, stale on up to 48% of days), gasoil QS1 and
heating oil HO1 (no second generic month in this workbook, so roll days cannot be made
contract-consistent).
"""
from dataclasses import dataclass

MONTHLY = tuple(range(1, 13))
QUARTERLY = (3, 6, 9, 12)


@dataclass(frozen=True)
class Market:
    name: str
    root: str                  # Bloomberg generic root: "TZT" -> "TZT1 Comdty"
    tenors: tuple[int, ...]    # generic tenors loaded from the workbook
    cycle: tuple[int, ...]     # delivery months that are listed contracts
    expiry_rule: str           # key into etr.expiry.RULES
    exchange: str
    currency: str
    unit: str
    expiry_holidays: str | None = None        # holidays removed when counting expiry days
    expiry_holidays_since: str | None = None  # ...only from this date (see TTF)
    liquid_from: str | None = None            # first date the futures traded in size

    def ticker(self, k: int) -> str:
        return f"{self.root}{k} Comdty"


MARKETS = {
    m.name: m
    for m in [
        # TTF counted UK bank holidays as business days in 2015-16 and had stopped by 2020. No
        # TTF expiry depends on them between Jun 2016 and Aug 2020, so any cut-over date in that
        # gap gives the same schedule. TTF futures barely traded before 2014 (median 60-190
        # lots/day in 2010-13), and their settlements look smoothed (positive autocorrelation).
        Market("TTF", "TZT", MONTHLY, MONTHLY, "ice_gas", "ICE Endex", "EUR", "EUR/MWh",
               expiry_holidays="GB", expiry_holidays_since="2018-01-01", liquid_from="2014-01-01"),
        Market("NBP", "FN", MONTHLY, MONTHLY, "ice_gas", "ICE Futures Europe", "GBP", "p/therm",
               expiry_holidays="GB"),
        Market("HH", "NG", MONTHLY, MONTHLY, "nymex_ng", "NYMEX", "USD", "USD/MMBtu"),
        Market("BRENT", "CO", (1, 2, 3), MONTHLY, "ice_brent", "ICE Futures Europe", "USD", "USD/bbl",
               expiry_holidays="GB"),
        Market("WTI", "CL", (1, 2), MONTHLY, "nymex_cl", "NYMEX", "USD", "USD/bbl"),
        # EUA generics follow the quarterly cycle; the quarterly contracts are thinly traded but
        # track the December contract almost exactly (correlation 0.9998).
        Market("EUA", "MO", (1, 2), QUARTERLY, "ice_eua", "ICE Endex", "EUR", "EUR/t",
               expiry_holidays="GB"),
    ]
}
