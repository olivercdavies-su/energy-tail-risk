# Project log

What was decided, why, and what comes next, so any session can pick up from here. Newest session at the top. The phased plan (approved 2026-10-24) is at the bottom.

---

## Session 3 (2026-10-24): Phase 2, scale models at 1 and 10 days. Done.

### Done
- **`etr/scale.py`: the seven scale models the protocol lists.**
  - The models: calibrated EWMA (= baseline B3), weekday EWMA, GJR-GARCH-t, HAR, implied vol, and two equal-weight combinations (`comb` = weekday EWMA + GJR + HAR; `comb_iv` adds implied vol).
  - Each gives a 1-day forecast, a √10 version, and its own 10-day term structure.
  - Walk-forward with annual refits on data up to the previous year-end (expanding from 2010).
  - Point-in-time calibration of every model, horizon and year, from the model's current parameters run over its training window.
  - Ported from energy-vol-targeting: EWMA, GJR fitting and multi-step forecasts, HAR components and purging, calibration from current parameters.
- **`etr/tails.scale_shape_forecast` now takes the h-day variance directly**, so a term structure can replace √h. The Phase 1 baseline outputs are byte-identical after the change.
- **87 tests** (12 new). They cover:
  - GJR recovering a *negative* asymmetry, and its multi-step forecasts matching the recursion;
  - the QLIKE fit recovering known coefficients and being exactly calibrated in-sample;
  - weekday factors finding a planted Monday effect and shrinking noise;
  - the implied-vol filter and fallback;
  - combination arithmetic;
  - the EWMA candidate equalling B3 exactly;
  - walk-forward look-ahead, and that a year's fits ignore that year's returns.
- **The mutation check now injects 14 bugs; all are caught.** The six new ones: no HAR purge, GJR fitted on the forecast year, calibration or weekday factors seeing the forecast year, the origin's weekday used instead of the target's, and no implied-vol fallback.
- **`scripts/p2_scale.py`** evaluates every model on 2012–18 (data after 2018 dropped first) with a normal shape. Forecasts are cached in `data/processed/` for Phase 3.

### Decisions and why
1. **HAR is fitted by minimising QLIKE, not by least squares on log variance.**
   - With daily squared returns as the target, log(r²) is undefined on zero-return days and extremely noisy, and least squares on r² is dominated by a few huge days.
   - QLIKE for a log-linear variance is convex, so Newton's method finds the optimum exactly, and the fit is calibrated in-sample by construction.
   - A separate fit on 10-day squared returns gives a direct 10-day HAR.
   - The fitted weight on yesterday's squared return is about zero in every market. One day's r² is noise; the 5-, 22- and 66-day averages carry the information.
2. **Weekday factors use the last five years, shrunk toward 1 by empirical Bayes across all markets.** The prior spread of true effects is estimated from the data, so nothing is tuned.
   - A unit test showed how noisy they are: five years of fat-tailed data give each factor a standard error of 0.1–0.2. A planted 1.6× Monday effect was not detectable over two forecast years; 2× over six was.
3. **Implied vol and its EWMA fallback are calibrated separately.** Implied vol needs shrinking (a variance risk premium: c = 0.69–0.97) while EWMA needs scaling up (1.04–1.22). One constant for a mix of the two would be wrong for both.
4. **No candidate was dropped on validation results.** All of them go to Phase 3, where the protocol's rule chooses among full scale × shape configurations.

### Results (validation 2015–18; 2012–14 in brackets where it differs)
**1-day, normal shape:** pooled FZ0 gain over calibrated EWMA at α = 2.5%, with t-statistics. The minimum detectable effect is about 0.010.

| | comb_iv | comb | GJR | implied vol | HAR | weekday EWMA |
|---|---|---|---|---|---|---|
| FZ0 gain | **0.024 (t 4.8)** | 0.019 (4.2) | 0.016 (2.1) | 0.017 (1.5) | 0.014 (1.6) | −0.005 (−0.8) |
| QLIKE ranking | same order: comb_iv best (t 6.4) | | | | | none |
| 2012–14 | | | worse (t −1.3) | | worse | **better (t 3.2)** |

- **The combinations are best and the most stable across both windows.**
- **Calibrated EWMA (B3) and weekday EWMA are outside the 10% model confidence set; every other model is inside.**
- **GJR and HAR lose to EWMA in 2012–14.** Their early refits had only 2–4 years of data, as the previous project found. From 2015 they have 5+ years, and the test window will have 9+.
- **No scale model fixes 1-day calibration with a normal shape.**
  - Long-side 99% breaches run at 1.4–1.7% for every model, and long-side ES is 30–45% too deep.
  - The better-scoring scales fail slightly *more* often.
  - So the extreme tail is a **shape problem**, which is exactly Phase 3's job. The scale's job is accuracy.
- **Weekday factors fix European gas but hurt oil.**
  - European-gas Monday realised/forecast variance goes from 1.32 to 1.00 in 2015–18 (2.20 to 1.74 in 2012–14).
  - But oil inherits stale factors: a Thursday effect from 2011–14 that has since vanished (oil Thursday ratio 0.71).
  - Net validation gain: none. In 2012–14, when the effects were fresh, a clear gain.
  - DQ weekday failures fall from 5 of 12 market-sides to 3–4.

**10-day:**
- **Every model's own term structure beats its √10 version.**
  - FZ0 gains: implied-vol term 0.063, comb_iv term 0.056, HAR term 0.046, comb term 0.045, GJR term 0.042, flat EWMA with 10-day calibration 0.016.
  - The 10-day minimum detectable effect is about 0.026.
- **Calibration:** with a normal shape, the EWMA, weekday-EWMA, GJR and comb_iv term structures pass all six pooled non-overlapping tests; √10 calibrated EWMA fails two.
- **The 10-day calibration scales say why.** Realised 10-day variance is 1.2–1.8× ten times the EWMA variance in TTF, NBP and EUA (positive autocorrelation, as session 1 found), and about 1.0 in HH.
- **The vol-state bias survives:** even term structures are too low in calm and middle states (realised/forecast sd 1.04–1.17) and slightly high in turbulent ones (0.88–1.03). The direct 10-day HAR is best in calm states (1.04). This is Phase 4's problem.

**Parameters** (`results/p2_scale_params.csv`):
- GJR asymmetry is negative for TTF, NBP and HH (−0.024 to −0.036) and positive for Brent, WTI and EUA (+0.05 to +0.07). Persistence is about 1.000 in TTF, NBP and EUA.
- HH's conditional tails are thin (Student-t ν ≈ 20); the others' ν is 5–8.
- The implied-vol filter replaced 3 prints in 2012–18 (all HH); no fallback was needed.

### Things I'm unsure about
1. **The weekday model's window may be the wrong design.** Effects drift, so a five-year window carries stale ones (oil Thursday).
   - A design that only uses the structural weekend effect (European gas Mondays, weather updates) might work better.
   - Choosing it now would be tuned on the validation window, so I have not changed anything. It is a possible amendment, for you to decide; the protocol allows amendments decided on development data.
2. **The 1-day gains are real but small,** about 1–2× the minimum detectable effect, so the test window may not confirm them.
3. **Combinations use equal weights by design.** Fitted weights would have to be estimated on the same few years.
4. **For TTF, NBP and EUA, `comb_iv` is just `comb`:** there are no options data for them in 2012–18, so implied vol falls back to EWMA.

### Next: Phase 3 (tail shapes and the 1-day engine)
1. **Shapes on the standardised residuals of each scale** (in-sample, from current parameters, point-in-time):
   - Student-t;
   - Hansen skewed-t;
   - filtered historical simulation;
   - peaks-over-threshold GPD tails.
   Each is pooled across markets with market-specific asymmetry, and also per market.
2. **Tests:** synthetic skewed fat-tailed data with known quantiles and ES; GPD recovering ξ; look-ahead.
3. **Run the protocol's selection rule (§4.2) on 2015–18** over all scale × shape configurations, and report the shortlist, the calibration screen and the chosen 1-day configuration.

---

## Session 2 (2026-10-24): Phase 1, the yardstick and the pre-registration. Done.

Oliver approved the session 1 plan as proposed, including its defaults:
- the semi-blind 2019→ test window;
- long and short positions evaluated separately;
- 1.5× as the main early-warning event;
- the proposed portfolios;
- VaR 95/97.5/99% with ES 97.5%.

### Done
- **`etr/backtest.py`, the evaluation toolkit:**
  - exact Kupiec;
  - Christoffersen and Engle–Manganelli DQ with Monte Carlo p-values (DQ takes extra regressors, e.g. weekday dummies);
  - Basel traffic light;
  - ES tests: Acerbi–Székely Z2 (simulated under the model), Du–Escanciano, McNeil–Frey;
  - the FZ0 joint VaR–ES score and tick loss;
  - Diebold–Mariano, the model confidence set, Holm;
  - pooled coverage and ES tests;
  - a non-overlapping window selector on a calendar grid shared by all markets.
- **`etr/tails.py`:**
  - unit long/short losses from contract-consistent returns, using the exact simple-return mapping (no log approximation);
  - a `Forecast` object carrying VaR, ES, the PIT and a sampler, so every test sees the same distribution;
  - scale × shape forecasts, with ES by Gauss–Legendre quadrature on the quantile function (accurate to 1e-10 against the closed form);
  - historical simulation.
- **`etr/scale.py`:** EWMA and its point-in-time calibration.
- **Three baselines, fixed before any model exists:** `hs500`, `riskmetrics`, `ewma_cal`.
- **75 tests**, all on synthetic data with known answers or as look-ahead perturbations. `scripts/mutation_check.py` injects eight realistic bugs (EWMA peeking a day ahead, calibration using its own year, an off-by-one forward return, a shifted HS window, a PIT on the wrong tail, FZ0 without 1/α, a bootstrap that doesn't impose the null, Kupiec with the wrong distribution); **each is caught**. The first run caught only six. The PIT-direction and Kupiec tests were too weak and have been strengthened.
- **`scripts/p1_test_size.py`:** size and power of every registered test by simulation (`results/p1_test_size.csv`).
- **`scripts/p1_baselines.py`:** the baselines on 2012–18 through the whole toolkit (`results/p1_baselines_*.csv`). It drops all data after 2018 before computing anything.
- **[`PROTOCOL.md`](PROTOCOL.md) registered in commit `e636450`**, before any tail model exists. Any later change goes in its section 10 as a labelled amendment.

### Decisions and why
1. **Small-sample p-values throughout.**
   - At 99% a year has about 2.5 expected breaches, and the χ² approximations are poor: χ²-Kupiec has size 5.5% at n = 1,000, the exact version 4.3%.
   - So Kupiec is exact (binomial), and Christoffersen and DQ are Monte Carlo under iid Bernoulli hits (Dufour 2006).
2. **Pooled tests bootstrap the PITs by date.**
   - Each market's PITs are replaced by their exact uniform ranks, which imposes "the model is right" while keeping the observed dependence between markets on the same day; whole dates are then resampled.
   - TTF–NBP and Brent–WTI breach together, so stacking their hits would double-count. A HAC t-test on the per-date mean ran at 7.4% size at 99%; the PIT bootstrap at 5.0%.
3. **10-day calibration tests use non-overlapping windows; 10-day ranking uses all windows.**
   - Overlapping 10-day hits are autocorrelated by construction. For one market at 99%, every HAC variant tried rejected a correct model 13% of the time. Pooled HAC and block bootstraps ran at 7–10%.
   - The mean of rare, clustered events is too skewed for a normal approximation with about 200 independent windows. Non-overlapping windows are independent under the null, so their tests are exact.
   - Diebold–Mariano on overlapping FZ0 differences keeps its size (3–6% across the simulations, 20 lags), because loss differences are continuous.
4. **FZ0 for ranking.** It is the Fissler–Ziegel member whose differences are unchanged by rescaling, so it can be pooled across markets and regimes (the role QLIKE played for variance).
5. **Selection rule (PROTOCOL §4.2):** the model confidence set on validation FZ0, then a calibration screen, then the simplest survivor.
   - I chose "simplest in the confidence set" over "lowest point estimate" because validation has 1,031 days and the minimum detectable effect between similar models is about a 9% scale error. A lower point estimate would mostly reflect noise.
   - The calibration screen makes selection serve the primary question (is it calibrated?), not only ranking.

### Results (development data only)
**Baselines on the 2015–18 validation window** (`results/p1_baselines_*.csv`):

| | hs500 | riskmetrics | ewma_cal |
|---|---|---|---|
| Pooled FZ0, 1-day, α = 2.5% (lower is better) | −2.68 | −2.87 | **−2.88** |
| Long 99% breach rate (pooled p) | 1.3% (0.07) | 1.8% (0.000) | 1.4% (0.008) |
| Short 99% breach rate (pooled p) | 1.2% (0.15) | 1.4% (0.010) | 1.2% (0.20) |
| Pooled ES 97.5%, long / short (p) | 0.006 / 0.004 | 0.000 / 0.002 | 0.018 / 0.16 |
| Market-sides failing DQ at 99% | 11 of 12 | 8 of 12 | 5 of 12 |

- **Calibrated EWMA ranks first** and is the only baseline in the 10% model confidence set at both horizons and both levels. Historical simulation is far behind at 1 day (FZ0 +0.20, p < 0.001); at 10 days the gap is not significant (p = 0.20).
- **No baseline is calibrated.**
  - Every one fails somewhere, mainly on the long side and on ES: oil's heavy left tail, as session 1 found.
  - Calibrated EWMA's short side passes.
  - DQ failures for calibrated EWMA come from European gas and EUA, consistent with the weekday effects.
- **So there is room for the tail model, and the bar is calibrated EWMA, not RiskMetrics or historical simulation.**
- **What the test window can detect** (2.8 standard errors, from validation, scaled to about 1,950 test days):
  - Similar models (e.g. calibrated EWMA vs RiskMetrics): an FZ0 difference of 0.010 at 1 day, α = 2.5%, roughly a **9% scale error**. At α = 1% it is 0.020 (12%). At 10 days, 0.026 (15%) and 0.047 (20%).
  - Very different models (historical simulation vs calibrated EWMA): 0.073 at 1 day (about a 27% scale error). At 10 days the required difference exceeds anything observed.

**Size and power of the tests** (`results/p1_test_size.csv`; full table in PROTOCOL §6):
- **Every test rejects a correct model at 0.5–8% at the 5% level** (200 replications, ±1.5 points).
  - The pooled ES test's 8% was noise: with 1,000 replications the three primary pooled tests run at 3.6–4.3% (`results/p1_primary_size.csv`).
  - Diebold–Mariano is slightly conservative (3–3.5%).
- **McNeil–Frey is the test that sees thin tails:** it has 100% power against normal tails on t(4) data, against 38% for Acerbi–Székely and 52% for Du–Escanciano.
  - So its rejection of calibrated EWMA in 7 of 12 development market-sides is real evidence that the normal shape is too thin, not an oversized test (its size is 4%).
  - It is weak against scale errors (17%), where Acerbi–Székely and Du–Escanciano are strong (93–94%). The tests are complementary.
- **Only DQ with weekday regressors detects a Monday effect** (47% at 97.5%); breach counts can't.
- **Coverage at 97.5% is blind to tail shape**, because normal and t(4) quantiles nearly coincide there.
- **10-day tests detect scale errors when pooled (73–84%) but hardly tail shape.**

### Things I'm unsure about
1. **The simulations assume a Gaussian copula** (pairs at 0.9) and t(5) marginals. Real joint tails may cluster more, which would make the pooled tests somewhat over-sized. Session 1 found joint tails close to Gaussian for the correlated pairs, so I expect this to be small.
2. **The "scale error" translation of the MDEs assumes a pure scale error under t(5) losses.** Real model differences are not pure scale errors; treat it as a rough guide.
3. **The selection rule's complexity order is a judgement.** It is fixed now, before any candidate exists, which is what matters.

### Next: Phase 2 (scale models, 1-day and 10-day)
1. Port GARCH and HAR, the walk-forward and the calibration from the previous repo, with their tests.
2. Add weekday factors (point-in-time, shrunk), filtered implied vol with an EWMA fallback, and multi-step term structures for 10 days.
3. Judge them on the validation window by QLIKE at 1 and 10 days, by normal-VaR breach rates by weekday and vol state, and by FZ0 with a normal shape. Shapes come in Phase 3.

---

## Session 1 (2026-10-24): setup, re-verification, tail exploration, proposed plan

**Status: approved by Oliver as proposed (start of session 2).**

### Done
- **Repo and data guard.** `git init`; `.gitignore` excludes `data/` and every data-file format; `scripts/hooks/pre-commit` (adapted from energy-vol-targeting) blocks anything under a `data/` folder, data-file extensions, notebooks, CSVs outside `results/`, results files long enough to be a daily series, and files over 2 MB. A new `scripts/hooks/commit-msg` rejects AI attribution lines. Each rule was tested by trying to commit a violating file (the workbook, a parquet, a notebook, stray and nested CSVs, a 500-row results table, a file under `etr/data/`, a 3 MB file, a message with a `Co-Authored-By` trailer): all were blocked. **A fresh clone must run `git config core.hooksPath scripts/hooks`.**
- **Read the previous repo:** README, findings brief, lock-box protocol and results, Phase 1 review, project log, and the code to be reused.
- **Ported the data pipeline** into `etr/` (`markets.py`, `expiry.py`, `returns.py`), credited in docstrings, with its tests (44 pass, including real-data roll checks). The rebuilt returns are **identical, row for row, to the previous project's** `returns.parquet`.
- **Fixed a latent loader bug:** implied vol is missing about one day a month (option expiry) and those dates have no workbook row at all, so the old `implied_vol()` forward-fill never reached them. The previous project re-filled after reindexing, so its results were not affected. Test added.
- **Re-verification** (`scripts/p0_reverify.py`, `results/p0_reverify_*.csv`) and **tail exploration on 2011–18 only** (`scripts/p0_tails.py`, `results/p0_tails_*.csv`, `results/p0_tail_curves.png`).

### Re-verification: the inherited findings this project relies on
All recomputed with independent code, not the old backtester.

| Finding | Published | Reproduced |
|---|---|---|
| Open-interest jump on the rule's switch day | TTF 152/156, NBP 195/201, HH 201/201, Brent 202/202, EUA 66/67 | identical |
| TTF drift, adjusted vs naive | −10.7% vs +10.7%/yr | −10.70% vs +10.67% |
| HH vol overstated by naive returns | 52% vs 60% | 52.4% vs 60.5% (+15.4%) |
| EUA excess kurtosis, top-1% variance share (2010–18) | 18.5, 27% | 18.5, 27.3% |
| GJR asymmetry γ (2012–18 fit) | gas ≈ −0.03, oil ≈ +0.07 | TTF −0.026, NBP −0.030, HH −0.035, Brent +0.059, WTI +0.068 |
| Weekly 99% VaR breaches, EWMA-sized trend sleeves 2011–18 | 1.2–2.2% | 1.2–1.9% (each market within one breach) |
| 2019–26 without vol targeting, TTF/NBP | z_sd 3.2–3.3, 99% breaches 17% | 3.19–3.28, 16.5–16.8% |
| 2019–26 calibrated EWMA | z_sd 0.95–1.07, 99% breaches 0.7–2.2% | identical to 3 d.p. |

2019–26 was touched only to reproduce these already-published rows.

**One inherited number needs a caveat.** The 5-day variance ratios "0.80–0.98" were computed from non-overlapping 5-day sums, which carry a standard error of about ±0.1. With overlapping sums I get 0.87–1.08. The broader claim built on them, that √time scaling overstates multi-day risk, does not survive in the form a VaR engine needs (finding 4 below).

### Tail exploration (development data only: 2011–18, TTF 2014–18)
Daily returns are standardised by the volatility forecast known the evening before: z = r / (calibrated EWMA vol). If the scale were perfect, z would have unit variance, so z isolates the **shape** problem from the **scale** problem. Its standard deviation is 1.01–1.02 in every market, so the point-in-time calibration works at the daily horizon. Before calibration, EWMA under-forecasts daily variance by 9–16% over 2011–18; the point-in-time scales applied were 1.07–1.12.

1. **The 99% tail problem is real, moderate, and asymmetric, and the asymmetry differs by market.** Gaussian 99% VaR breach rates per tail (nominal 1%):

   | | TTF | NBP | HH | Brent | WTI | EUA |
   |---|---|---|---|---|---|---|
   | falls (long position loses) | 1.6% | 0.9% | 1.4% | **2.2%** | **2.1%** | 1.7% |
   | rises (short position loses) | 1.6% | **1.8%** | 1.4% | 1.1% | 0.7% | 1.4% |

   - At 95%, Gaussian VaR is fine everywhere (3.9–5.7%).
   - Oil's left tail is heavy: its empirical 99% quantile is 2.96–3.00 sds against the normal's 2.33, so long-oil 99% VaR needs to be about 27% larger. Oil's right tail is, if anything, thinner than normal.
   - The previous project's tail numbers mixed the two tails through trend positions (mostly short gas, long or short oil). **A risk engine has to be judged on both tails separately.**
2. **Vol scaling removes some fat-tailedness, not all.** Excess kurtosis, raw → standardised: HH 3.1 → 0.8 (HH's tails are almost all vol clustering); TTF 3.0 → 2.9, NBP 2.9 → 2.7, Brent 3.2 → 2.3, WTI 2.9 → 2.4, EUA 17.3 → 4.3.
3. **Per market and tail, the tail index cannot be pinned down.** Peaks-over-threshold on the top 5% of each tail gives GPD ξ between −0.12 and 0.24, each with a standard error of about 0.1 (65–104 exceedances). The estimates are indistinguishable from each other and mostly from zero. **Market-by-market extreme-value tails would mostly fit noise; the shape has to be pooled across markets** (scale-free residuals make that legitimate), with only the asymmetry allowed to differ.
4. **10-day horizon: the error comes from volatility mean-reverting, not returns.**
   - Raw returns do mean-revert in HH and oil (lag-1 autocorrelation −0.08 to −0.10; 10-day variance ratio 0.90–1.02, SE ≈ 0.11). But that comes from the high-vol periods.
   - After vol scaling, the 10-day variance ratio is 1.00–1.18 (SE 0.07–0.10): no significant mean reversion anywhere, and positive autocorrelation in TTF, NBP and EUA.
   - √10 × calibrated EWMA **understates** 10-day risk on average (sd of z10 1.02–1.16).
   - Split by vol state (terciles of EWMA vol relative to its past year): in the calmest third it understates (sd 1.02–1.19); in the most turbulent third it is about right or overstated (0.90–1.04). That is volatility mean-reverting within the horizon, which a flat EWMA forecast ignores.
   - **Implication:** 10-day VaR needs a volatility term structure (GARCH/HAR multi-step, or a direct 10-day scale), not a return-autocorrelation haircut.
5. **Calendar time matters at the daily horizon.**
   - On scheduled report and weather days, standardised variance is 1.39× and two-sided 99% breaches are 3.2% (1.7% on other days).
   - **European gas Mondays** (weekend weather updates) are the largest effect: 4.7–5.2% breaches, variance 1.56–1.65×. It is present in both halves of the period but weakening (NBP 1.84 → 1.27).
   - Oil Wednesdays (EIA report) run at 1.16–1.35 in both halves. HH Thursdays (EIA storage) fell from 1.52 to 1.12. Oil Thursdays appear only in 2011–14.
   - The previous project found calendar inputs useless, but at a weekly horizon, where every week contains each weekday once. **For a daily engine, weekday factors are a likely gain**, estimated point-in-time and shrunk, because they drift.
6. **State dependence of the shape is weak in this sample.**
   - Breaches are a little more frequent when vol is low relative to its past year (2.2% vs 1.8% in the high-vol tercile, p = 0.12). That "calm before the storm" pattern is what an early-warning model would exploit, but it isn't established here.
   - Holding M2 vs M1 makes no difference. Gas winter vs summer: 2.1% vs 1.5%.
7. **Joint tails.**
   - TTF–NBP and Brent–WTI crash together about as often as a Gaussian copula with their correlation (0.92) implies (1.0–1.3×).
   - Cross-commodity pairs show 0.6–2.9× the Gaussian rate, but on 1–6 days each: not estimable.
   - So portfolio dependence should come from the data non-parametrically (resampling same-day residual vectors), not from a parametric copula this sample can't calibrate.
8. **"Vol doubles within a month" is rare.**
   - Defined as next-21-day realised vol ≥ 2 × today's calibrated EWMA vol: 0.8% (Brent) to 4.7% (EUA) of days, in only **3–9 episodes per market** over 2011–18.
   - Several episodes are shared (TTF and NBP on 2014-01-31 and 2016-03-30). Brier scores and calibration plots over ~20 independent episodes are almost uninformative.
   - At 1.5×, the base rate is 6–12% of days.
9. **Implied vol is a better scale for tails, once bad prints are removed.**
   - One HH print of 1.2% (26 Nov 2013, against 31% the day before) produced a 10-σ standardised move.
   - With a point-in-time plausibility filter (implied vol within 1/3–3× EWMA), implied-vol-scaled residuals have lower kurtosis than EWMA-scaled ones in all three markets: HH 0.5 vs 0.8, Brent 1.9 vs 2.3, WTI 1.5 vs 2.4.
   - Its level needs shrinking by 10–27% in variance terms (the variance risk premium).
10. **Power.** Kupiec test at 99%, chance of detecting a true breach rate of 1.5× / 2× nominal:

    | Sample | 1.5× | 2× |
    |---|---|---|
    | one market, 4 years daily | 34% | 78% |
    | one market, 8 years daily | 52% | 96% |
    | 4.5 effective markets pooled, 8 years | 99% | 100% |
    | one market, 10-day non-overlapping, 8 years | 13% | 23% |

    **Per-market 10-day breach tests are essentially powerless.**

### Where I'd change the framing
1. **Evaluate both tails: long and short unit positions, not one direction.** The asymmetry is market-specific (oil left-heavy, NBP right-heavy), and a desk holds both.
2. **"Volatility forecasting is solved" holds at the weekly horizon, not the daily one.** Calendar-time effects and implied vol as a tail scale are first-order for daily VaR. The engine should be **scale × shape**, and both parts get evaluated.
3. **Drop "√time overstates multi-day risk".** For vol-scaled VaR it understates in calm states and overstates in turbulent ones. The 10-day model needs a vol term structure.
4. **10-day VaR can't be validated market by market.** Evaluate it pooled across markets, with overlap-robust inference (HAC, block bootstrap) and the joint VaR–ES score. Say in advance that a null 10-day result is uninformative.
5. **Pool the tail shape across markets** (finding 3); per-market EVT is mostly noise.
6. **Early-warning: use 1.5× as the main event and report 2×.** Benchmark it against a vol-state logistic, not just the base rate, because vol mean reversion alone makes "doubling" likelier from calm states. Feed it into VaR only if it passes a pre-set rule on validation data.
7. **Portfolio dependence:** joint filtered historical simulation, plus the desk-standard univariate model on the portfolio's own P&L history as a baseline.

### Holdout design (proposal)
| Option | Clean? | Power | Verdict |
|---|---|---|---|
| New markets from a fresh pull (gasoil, ULSD, RBOB; coal/power if available) | Fully | Moderate: products correlate about 0.8–0.9 with crude, so they add perhaps 1–1.5 effective markets | Best confirmation, if the pull happens |
| Forward data after Oct 2026 | Fully | Very low: about 1 expected 99% breach per market per 100 days | Descriptive live track record only |
| 2019–26 on the six markets | Partly: examined for weekly vol levels and weekly trend-sleeve VaR, **not** for daily tails, ES, 10-day risk, joint tails or stress probabilities | High: pooled about 90 expected daily 99% breaches; covers COVID, negative WTI and the 2021–23 gas crisis | Main test, labelled semi-blind |
| Pre-2010 history (fresh pull from 2005) | Fully | Low for testing: models need 5+ years of burn-in | Better as extra training history (2008) |

**Recommendation:**
- **Seal 2019-01-01 onwards now for all tail work.** This session's exploration stopped at 2018. The final evaluation runs once on 2019-01-01 → end of data, under a protocol registered before any tail model is built.
- Results for the six markets are labelled **semi-blind**. Any new markets from a fresh pull are run through the frozen engine as a **blind replication**; their roll schedules are verified with open interest only, without looking at returns' tails.
- Data after Oct 2026 is appended as a **descriptive live check**.
- Development keeps the previous split: expanding-window annual refits from 2010, model selection on **2015–18**, and 2012–14 as an earlier check (GARCH needs about 5 years, so earlier refits are weak).

---

## Plan (approved by Oliver, 2026-10-24)

### Layout (a handful of modules)
```
etr/        markets.py, expiry.py, returns.py   (done: data in, returns out)
            scale.py      vol forecasts: EWMA (+ weekday factors), GJR-GARCH, HAR, implied vol
                          with filter and fallback; 1-day and 10-day; walk-forward and calibration
            tails.py      shape models on standardised residuals: normal, t, skewed-t, filtered HS,
                          pooled peaks-over-threshold; scale x shape -> VaR/ES; multi-day
            learned.py    scale-free learned VaR/ES model and the stress early-warning
            backtest.py   coverage, independence, DQ, traffic light, ES tests, joint VaR-ES score,
                          Diebold-Mariano, model confidence set, Holm, overlap-robust inference
            portfolio.py  portfolio P&L and joint filtered historical simulation
scripts/    build_data.py, p<phase>_*.py, run_holdout.py
tests/      synthetic-data, look-ahead and mutation tests; real-data tests skip without the workbook
results/    p<phase>_*.csv and *.png (summary tables and charts only)
PROTOCOL.md (the pre-registration), PROJECT_LOG.md, README.md
```

### Phases (one per session; each ends with tests passing, a log entry and commits)

**Phase 1: the yardstick, then pre-registration.** Build the evaluation before any model, so the tests can't be bent to fit a favourite.
- `backtest.py`:
  - Kupiec;
  - Christoffersen (independence and conditional coverage);
  - Engle–Manganelli dynamic-quantile test, with weekday and vol-state regressors (it catches the Monday effect, which breach counts can't);
  - Basel traffic light per 250 days;
  - ES backtests: Acerbi–Székely Z2 (simulated under each model's own forecast distribution) and McNeil–Frey;
  - the FZ0 joint VaR–ES loss (Patton, Ziegel & Chen 2019): the zero-homogeneous member of the Fissler–Ziegel family, scale-free like QLIKE, so markets and regimes can be pooled;
  - Diebold–Mariano with HAC, model confidence set, Holm;
  - overlapping 10-day handling.
- **Two baselines a desk would recognise:** historical simulation (500 days) and RiskMetrics (EWMA-normal).
- **Verified by simulation:** each test rejects at about its nominal rate when the model is right, and rejects reliably when it is wrong. FZ0 is minimised by the true VaR/ES. Mutation tests (shift the forecast by a day, use σ_t instead of σ_{t−1}) must fail a test.
- **Then write and commit `PROTOCOL.md` before any tail model exists.** It fixes the test window, markets, positions and portfolios, horizons and levels (VaR 95/97.5/99%, ES 97.5%), the primary and secondary tests, decision rules, multiple-testing corrections, minimum detectable effects (simulated), the candidate model families, and the rule that will pick among them on 2015–18.

**Phase 2: scale models, 1-day and 10-day.**
- Port GARCH, HAR, walk-forward and calibration from the previous repo.
- Add weekday factors (point-in-time, shrunk), implied vol with the plausibility filter and an EWMA fallback, and multi-step term structures.
- Judged by QLIKE at 1 and 10 days, and by Gaussian-VaR breach rates by weekday and vol state.
- *Verified by:* parameter recovery, look-ahead perturbation, purge tests, and synthetic weekday effects recovered.

**Phase 3: tail (shape) models and the 1-day engine.**
- Candidates, all on the best one or two scales:
  - normal, Student-t and Hansen skewed-t (pooled, with per-market asymmetry);
  - filtered historical simulation (pooled and per market);
  - peaks-over-threshold GPD per tail with a pooled tail index (McNeil–Frey).
- Full backtests on 2015–18 for long and short positions; selection by the registered rule.
- *Verified by:* synthetic data with known skewed fat tails, where each estimator must recover the true quantile and ES; GPD must recover ξ.

**Phase 4: 10-day horizon.**
- Compare three approaches:
  - √10 × the 1-day model;
  - multi-step simulation (vol term structure plus filtered-HS paths);
  - a direct 10-day scale with a pooled empirical distribution of 10-day residuals.
- Rolls inside the horizon come for free from contract-consistent returns; holiday gaps are handled through spans.
- Pooled, overlap-robust backtests and the FZ0 score.

**Phase 5: portfolio.**
- Registered portfolios: an equal-risk long basket, a short basket, TTF–NBP and Brent–WTI spreads, a gas-vs-oil book, and a fixed-seed set of random long/short books.
- Methods: univariate engine on the portfolio's own P&L history (the desk standard), joint filtered HS, and an EWMA-covariance Gaussian/t baseline.
- P&L as a percentage of gross notional, FX-neutral (as before).

**Phase 6: learned model and stress early-warning.**
- (a) A linear model whose VaR and ES are multiples of the scale, depending on scale-free inputs (vol state, weekday, implied/realised ratio, trend strength, curve slope), fitted by minimising FZ0.
- (b) A small network with the same inputs and loss, only if (a) shows the inputs carry information. I'll explain every choice as it's made.
- Early warning: a pooled logistic model for P(vol ≥ 1.5× / 2× within 21 days), judged by Brier skill against a vol-state baseline and by reliability, with block bootstrap. It feeds VaR only if it passes the registered rule.

**Phase 7: freeze, holdout, write-up.**
- Freeze the selected configuration (commit), then commit the holdout script unrun. Run it once on 2019→ (plus new markets if pulled) and report everything, labelled.
- Robustness checks and a README write-up.

**Cross-cutting checks:**
- every timing-sensitive function gets a look-ahead perturbation test;
- every estimator and test is checked on synthetic data with a known answer;
- seed stability for anything simulated;
- the attribution check before every push.

### Open questions for Oliver
1. **Holdout:** do you approve "seal 2019→ now, semi-blind main test, new markets as blind replication, forward data descriptive"? Can you get terminal time for the fresh pull (see the previous repo's `handover/data_checklist.md`)? Constant-maturity implied vol for all six markets would also fix the front-month implied-vol weaknesses found here.
2. **Both tails:** OK to evaluate long and short positions separately?
3. **Early-warning event:** 1.5× as the main definition, 2× reported?
4. **Portfolios:** are the proposed books the ones that matter for the roles you're targeting, or do you have a desk-style book in mind?
5. **Levels:** VaR 95 / 97.5 / 99% and ES 97.5% (the Basel FRTB pairing), with the traffic light at 99%?

### Things I'm unsure about
1. **Weekday effects drift** (NBP Monday 1.84 → 1.27; oil Thursday vanished). They may not survive out of sample; Phase 2 must show they do on 2015–18 before they enter the engine.
2. **The implied-vol filter band (1/3–3×) is my choice,** made after seeing one bad print. It's a data-cleaning rule, so I think that's acceptable, but it's fixed now and won't be tuned further.
3. **Joint-tail and early-warning conclusions rest on handfuls of events.** They're hypotheses, not findings.
4. **The GPD standard errors are asymptotic,** and rough at about 100 exceedances.
5. **2019–26 is only semi-blind.** General lessons from it (the 2021–23 gas regime, the scale-free design) already shape choices here. The protocol will say so.

### Next
Wait for Oliver's approval of the framing, the holdout design and the phases. Then Phase 1: `etr/backtest.py` with simulation-verified tests, then `PROTOCOL.md`, committed before any tail model is built.
