"""Mutation check: inject one realistic bug at a time, run the tests, restore the file.

Every mutation must make at least one test fail; one that doesn't shows a gap in the tests,
not a bug in the code. Add mutations for each new module.

Run from the repo root:  python scripts/mutation_check.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MUTATIONS = [
    ("etr/scale.py", "s2 = v.ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean()",
     "s2 = v.shift(-1).ewm(alpha=1 - lam, adjust=False, ignore_na=True).mean()", "EWMA peeks at tomorrow"),
    ("etr/scale.py", "past = ratio[years < y].dropna()", "past = ratio[years <= y].dropna()", "calibration uses its own year"),
    ("etr/tails.py", "    fwd = cum.shift(-h) - cum\n", "    fwd = cum.shift(-h + 1) - cum.shift(1)\n", "forward return off by one day"),
    ("etr/tails.py", "win = vals[valid_pos[ends[:, None] - window + np.arange(window)[None, :]]]",
     "win = vals[np.minimum(valid_pos[ends[:, None] - window + np.arange(window)[None, :]] + 1, len(vals) - 1)]", "HS window shifted one day forward"),
    ("etr/tails.py", "out[\"u\"] = shape.sf(R / s) if upper else shape.cdf(R / s)", "out[\"u\"] = shape.cdf(R / s)", "PIT ignores the side"),
    ("etr/backtest.py", "return (L > V) * (L - V) / (alpha * E) + V / E + np.log(E) - 1", "return (L > V) * (L - V) / E + V / E + np.log(E) - 1", "FZ0 missing 1/alpha"),
    ("etr/backtest.py", "    R = u_panel.rank(axis=0).sub(0.5).div(u_panel.notna().sum(axis=0), axis=1).to_numpy(float)\n",
     "    R = U\n", "bootstrap does not impose the null"),
    ("etr/backtest.py", "    return {\"n\": n, \"hits\": x, \"rate\": x / n, \"lr_uc\": lr, \"p\": float(min(p, 1.0)),",
     "    return {\"n\": n, \"hits\": x, \"rate\": x / n, \"lr_uc\": lr, \"p\": float(stats.chi2.sf(lr, 2)),", "Kupiec with wrong distribution"),
]
failed_to_catch = []
for path, old, new, label in MUTATIONS:
    f = ROOT / path
    src = f.read_text()
    assert src.count(old) == 1, (label, "pattern not found")
    f.write_text(src.replace(old, new))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "tests/test_tails.py", "tests/test_backtest.py"],
                           capture_output=True, text=True, cwd=ROOT)
    finally:
        f.write_text(src)
    caught = r.returncode != 0
    first = next((l for l in r.stdout.splitlines() if l.startswith("FAILED")), "")
    print(f"{'CAUGHT ' if caught else 'MISSED '} {label:40s} {first[:90]}")
    if not caught: failed_to_catch.append(label)
print("all caught" if not failed_to_catch else f"missed: {failed_to_catch}")
sys.exit(1 if failed_to_catch else 0)
