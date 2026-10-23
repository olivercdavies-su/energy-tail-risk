"""energy-tail-risk: Value-at-Risk and Expected Shortfall for energy futures positions.

Filesystem locations live here. Everything under data/ is gitignored and blocked by
scripts/hooks/pre-commit (Bloomberg licence): raw or derived data is never committed.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
RESULTS = ROOT / "results"


def raw_workbook() -> Path:
    """The Bloomberg workbook: $ETR_RAW_XLSX if set, else the single .xlsx in data/raw/."""
    if env := os.environ.get("ETR_RAW_XLSX"):
        if not Path(env).is_file():
            raise FileNotFoundError(env)
        return Path(env)
    files = sorted(RAW.glob("*.xlsx"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly one .xlsx in {RAW}, found {len(files)}")
    return files[0]
