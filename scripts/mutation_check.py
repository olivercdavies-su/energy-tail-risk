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
    ("etr/scale.py", "        trn = trn[(trn[ecol] <= cut)].replace([np.inf, -np.inf], np.nan).dropna()",
     "        trn = trn.replace([np.inf, -np.inf], np.nan).dropna()", "HAR training targets not purged"),
    ("etr/scale.py", "    params = gjr_fit(r.loc[:cut])", "    params = gjr_fit(r.loc[:f\"{y}-12-31\"])", "GJR fitted on the forecast year"),
    ("etr/scale.py", "    ratio = ratio[(end <= cut) & ins.notna()].dropna()",
     "    ratio = ratio[(end <= cut + pd.DateOffset(years=1)) & ins.notna()].dropna()", "calibration sees the forecast year"),
    ("etr/scale.py", "    f1 = pd.Series(fmap[wd1[:, 0]], index=df.index)",
     "    f1 = pd.Series(fmap[df.index.weekday.to_numpy()], index=df.index)", "weekday factor of the origin, not the target"),
    ("etr/scale.py", "    z2 = (daily_variance(df) / ewma_variance(df).shift(1)).loc[cut - pd.DateOffset(years=years):cut].dropna()",
     "    z2 = (daily_variance(df) / ewma_variance(df).shift(1)).loc[cut - pd.DateOffset(years=years):cut + pd.DateOffset(months=6)].dropna()",
     "weekday factors see the forecast year"),
    ("etr/scale.py", "                cal[(\"iv\", h)] = tuple(a.fillna(b) for a, b in zip(cal[(\"iv\", h)], cal[(\"ewma\", h)]))",
     "                cal[(\"iv\", h)] = cal[(\"iv\", h)]", "implied vol without its EWMA fallback"),
    ("etr/engine.py", "    z = z[(target <= cut).to_numpy()]", "    z = z[(target <= cut + pd.DateOffset(months=3)).to_numpy()]",
     "shape residuals run past the training cut"),
    ("etr/tails.py", "        return self.ppf(1 - np.asarray(u, float))\n\n    def cdf(self, x):\n        x = np.asarray(x, float)\n        return self._D.cdf(",
     "        return self.ppf(np.asarray(u, float))\n\n    def cdf(self, x):\n        x = np.asarray(x, float)\n        return self._D.cdf(",
     "skewed-t upper tail read from the wrong end"),
    ("etr/tails.py", "            out[n] = GPDTails(zs[n], (xl, bl), (xr, br))", "            out[n] = GPDTails(zs[n], (xr, br), (xl, bl))",
     "GPD left and right tails swapped"),
    ("etr/tails.py", "        w = np.r_[np.ones(kf), k - kf] / k", "        w = np.r_[np.ones(kf), 0.0] / kf",
     "filtered-HS ES without the fractional weight"),
    ("etr/engine.py", "        cut = pd.Timestamp(f\"{y - 1}-12-31\")\n        zs =", "        cut = pd.Timestamp(f\"{y}-12-31\")\n        zs =",
     "shapes fitted on the forecast year"),
    ("etr/tails.py", "    return [SkewT(r.x[0], lam) for lam in r.x[1:]]", "    return [SkewT(r.x[0], r.x[1]) for lam in r.x[1:]]",
     "pooled skewed-t forces one asymmetry"),
    ("etr/tails.py", "    xi = float(optimize.minimize_scalar(profile, bounds=(0.0, 0.49), method=\"bounded\").x)",
     "    xi = float(optimize.minimize_scalar(profile, bounds=(-0.49, 0.99), method=\"bounded\").x)", "GPD allows a finite maximum loss"),
    ("etr/tails.py", "    q = np.clip(shape.isf(u) if upper else shape.ppf(u), -Z_CAP, Z_CAP)",
     "    q = shape.isf(u) if upper else shape.ppf(u)", "no cap on extreme moves in ES"),
    ("etr/engine.py", "    end = pd.Series(df.index, index=df.index).shift(-10)\n    z = z[(end <= cut).to_numpy()]",
     "    end = pd.Series(df.index, index=df.index).shift(-10)\n    z = z[(end <= cut + pd.DateOffset(days=20)).to_numpy()]",
     "10-day residual windows run past the cut"),
    ("etr/engine.py", "        s2 = w + (a + g * (r < 0)) * r ** 2 + b * s2", "        s2 = w + (a + g * (r > 0)) * r ** 2 + b * s2",
     "GJR paths with the leverage sign flipped"),
    ("etr/engine.py", "            R = gjr_path_returns(s1.loc[origins].to_numpy(), params, pool, rng, n_paths) * np.sqrt(c10)",
     "            R = gjr_path_returns(s1.loc[origins].to_numpy(), params, pool, rng, n_paths)", "GJR paths not calibrated"),
    ("etr/engine.py", "    col = f\"{scale_name}_sqrt10\" if approach == \"sqrt10\" else f\"{scale_name}_10\"",
     "    col = f\"{scale_name}_10\" if approach == \"sqrt10\" else f\"{scale_name}_sqrt10\"", "sqrt10 and term scales swapped"),
    ("etr/engine.py", "    t[\"passes_screen\"] = t[\"shortlisted\"] & t[\"no_worse_than_b3\"] & (t[\"failures\"] == 0)",
     "    t[\"passes_screen\"] = t[\"shortlisted\"] & (t[\"failures\"] == 0)", "selection without the A1 guard"),
    ("etr/portfolio.py", "        z = pool[rng.integers(0, len(pool), (n_o, n_paths))]",
     "        z = pool[rng.integers(0, len(pool), (n_o, n_paths, pool.shape[1])), np.arange(pool.shape[1])]",
     "joint FHS resamples markets independently"),
    ("etr/portfolio.py", "        r = sig[:, None, :] * z * opens[:, k][:, None, :]",
     "        r = sig[:, None, :] * z", "joint FHS ignores closed markets"),
    ("etr/portfolio.py", "    var_all = pd.DataFrame({n: fc[n][scale_col] for n in ORDER}).reindex(U.index).ffill(limit=5)",
     "    var_all = pd.DataFrame({n: fc[n][scale_col] for n in ORDER}).reindex(U.index).ffill(limit=5).shift(-1)",
     "joint FHS reads volatility a day ahead"),
    ("etr/portfolio.py", "    inv = 1 / vol\n", "    inv = vol\n", "books weighted by vol, not 1/vol"),
    ("etr/portfolio.py", "        cols[n] = z[(z.index <= cut) & np.isfinite(z.to_numpy())]",
     "        cols[n] = z[(z.index <= cut + pd.DateOffset(months=3)) & np.isfinite(z.to_numpy())]",
     "joint residual pool runs past the cut"),
    ("etr/engine.py", "        same = next((k for k in kept if np.array_equal(P[c].to_numpy(), P[k].to_numpy())), None)",
     "        same = None", "selection does not collapse identical forecasts"),
    ("etr/backtest.py", "            i = int(np.argwhere(included_loss_diffs == test_stat)[0, 0])",
     "            i = np.argwhere(included_loss_diffs == test_stat).squeeze()[0]", "MCS not tie-safe (arch's elimination)"),
    ("etr/engine.py", "                      \"complexity\": pd.Series({c: complexity(c) for c in P.columns}),",
     "                      \"complexity\": list(reversed([complexity(c) for c in P.columns])),",
     "selection with misaligned complexities"),
    ("etr/learned.py", "            d = d[(d.index <= cut) & (d[\"end\"] <= cut)].dropna()\n            d[\"market\"] = n",
     "            d = d[(d.index <= cut)].dropna()\n            d[\"market\"] = n", "learned training targets run past the cut"),
    ("etr/learned.py", "    return float(np.mean((L > V) * (L - V) / (alpha * E) + V / E + np.log(E) - 1))",
     "    return float(np.mean((L > V) * (L - V) / E + V / E + np.log(E) - 1))", "learned FZ0 objective missing 1/alpha"),
    ("etr/learned.py", "    r5 = r.rolling(5).sum()\n", "    r5 = r.rolling(5).sum().shift(-1)\n", "inputs peek a day ahead"),
    ("etr/learned.py", "    fwd = v[::-1].rolling(h).mean()[::-1].shift(-1)", "    fwd = v[::-1].rolling(h).mean()[::-1]",
     "stress window includes the origin day"),
    ("etr/learned.py", "                f = tails.scale_shape_forecast(df, var_y * m ** 2, side, h, shapes[n])",
     "                f = tails.scale_shape_forecast(df, var_y * m, side, h, shapes[n])", "multiplier applied to variance, not vol"),
]
failed_to_catch = []
for path, old, new, label in MUTATIONS:
    f = ROOT / path
    src = f.read_text()
    assert src.count(old) == 1, (label, "pattern not found")
    f.write_text(src.replace(old, new))
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "tests/test_tails.py", "tests/test_backtest.py", "tests/test_scale.py",
                            "tests/test_shapes.py", "tests/test_ten_day.py",
                            "tests/test_portfolio.py", "tests/test_learned.py"],
                           capture_output=True, text=True, cwd=ROOT)
    finally:
        f.write_text(src)
    caught = r.returncode != 0
    first = next((l for l in r.stdout.splitlines() if l.startswith("FAILED")), "")
    print(f"{'CAUGHT ' if caught else 'MISSED '} {label:40s} {first[:90]}")
    if not caught: failed_to_catch.append(label)
print("all caught" if not failed_to_catch else f"missed: {failed_to_catch}")
sys.exit(1 if failed_to_catch else 0)
