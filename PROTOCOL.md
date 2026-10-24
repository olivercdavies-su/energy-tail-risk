# Evaluation protocol (pre-registered)

**Registered at the end of Phase 1, before any tail model exists.** The commit that adds this file is the registration. What existed at that point: the data pipeline, the evaluation toolkit (`etr/backtest.py`), the loss mapping (`etr/tails.py`) and three baselines (`etr/scale.py`, `etr/tails.py`). The only results seen are on development data up to 2018-12-31: session 1's tail exploration and Phase 1's baseline backtests (`results/p0_*`, `results/p1_*`).

The final evaluation runs **once**, by a script committed unrun after the configuration is frozen (Phase 7), and every result is reported, favourable or not. A departure from this protocol is labelled as a departure, with its reason, next to the result it affects.

---

## 1. Data and windows

- **Markets:** TTF, NBP, Henry Hub, Brent, WTI, EUA. Contract-consistent returns from exchange expiry rules, rolling 5 business days before expiry (`etr/returns.py`, verified in session 1).
- **Development:** data up to 2018-12-31.
  - Models are estimated on an expanding window from 2010, refitted once a year on data up to the previous 31 December.
  - **Model choice uses the validation window: forecast origins 2015-01-01 to 2018-12-31.** Origins 2012–2014 are a secondary check.
  - TTF is evaluated from 2014 (its futures were illiquid before).
- **Test window: forecast origins from 2019-01-01 to the last origin whose target lies inside the data** (the workbook ends 2026-10-13).
  - It is labelled **semi-blind**. The previous project examined 2019–26 for weekly volatility forecasts and for weekly Gaussian VaR of trend positions. This project has not looked at it for any tail question: daily or 10-day VaR/ES, tail shape, joint tails or stress probabilities.
  - During the run, models refit annually exactly as in development.
- **Blind replication:** markets from a new Bloomberg pull, if one arrives before the freeze, are run through the frozen engine with the same statistics and reported as a separate group.
  - Their roll schedules are verified with open interest only, without looking at their returns' tails.
  - The six-market verdicts do not depend on them.
- **Forward data** after 2026-10-13 is reported descriptively, with no tests.
- **Exclusions:** a 1-day target whose return spans a missing price (at most 4 days per market).

## 2. Positions and portfolios

- **Market-sides:** a unit long and a unit short position in each market (12 in all).
  - P&L is the simple return on notional over the horizon; the loss is its negative.
  - Long and short are evaluated separately, because energy tails are asymmetric.
- **Portfolios** (Phase 5): fixed notional weights with gross notional 1, rebalanced daily. Each leg's P&L is a percentage of its own notional (FX-neutral). Risk weights use each market's realised vol over 2014–2018.
  - **P1 long basket:** all six long, weights ∝ 1/vol.
  - **P2 short basket:** −P1.
  - **P3 TTF–NBP spread:** long TTF, short NBP, equal risk.
  - **P4 Brent–WTI spread:** long Brent, short WTI, equal risk.
  - **P5 European gas vs oil:** long TTF and NBP, short Brent and WTI; equal risk within each leg, equal risk between legs.
  - **P6 random books:** 50 weight vectors with independent N(0, 1) entries, scaled to gross notional 1, drawn with `numpy.random.default_rng(20261024)`.
- **Calendars:** portfolio P&L is on the union of the markets' trading days. A market that is closed contributes nothing that day; its next return covers the gap.

## 3. Forecasts

- A forecast is made at each close (the origin) for the **next trading day (1-day)** and the **next 10 trading days (10-day; one per origin, so windows overlap)**.
- **VaR** at 95%, 97.5% and 99%. **ES** at 97.5% (primary, the Basel FRTB level) and 99% (secondary).
- Every model supplies, for each origin: VaR, ES, the PIT u = P(L ≥ realised loss) and draws from its forecast distribution (`etr.tails.Forecast`).

## 4. Models and how one is chosen

**Baselines** (fixed definitions, in code at the registration commit):

| | Definition |
|---|---|
| **B1 `hs500`** | Historical simulation over the last 500 trading days (overlapping 10-day returns at 10 days). Empirical-quantile VaR; Acerbi–Tasche ES. |
| **B2 `riskmetrics`** | EWMA variance (λ = 0.94), normal, √h scaling. |
| **B3 `ewma_cal`** | B2 with the point-in-time level calibration (the previous project's most robust choice). |

**Candidates**, built in Phases 2–6, each fully specified in code before selection:
- **Scale:**
  - calibrated EWMA;
  - EWMA × point-in-time weekday factors;
  - GJR-GARCH-t with the asymmetry free to take either sign;
  - HAR;
  - implied vol (plausibility-filtered, EWMA fallback; HH, Brent, WTI);
  - equal-weight combinations of these.
  - At 10 days, additionally: GARCH/HAR term structures and a direct 10-day scale.
- **Shape:** normal, Student-t, Hansen skewed-t, filtered historical simulation, peaks-over-threshold GPD tails. Each is pooled across markets (market-specific asymmetry allowed) or per market.
- **Learned:** a linear model of VaR and ES multiples fitted by minimising FZ0, and a small neural network with the same inputs and loss (built only if the linear one helps).
- **Stress overlay:** an early-warning adjustment, a candidate only if it passes rule 4.3.

**4.1 Complexity order** (lower is simpler):
- **Scale:** calibrated EWMA (1) < weekday EWMA (2) < GJR (3) = HAR (3) < implied-vol based (4) < combination (5) < learned (6).
- **Shape:** normal (0) < t (1) < skewed-t (2) < filtered HS (3) < GPD tails (4).
- **Overlay:** +1.
- **10-day scaling:** √10 adds 0; a term structure or a direct 10-day scale adds 1.
- **Baselines:** `riskmetrics` and `ewma_cal` count as 1, `hs500` as 3.
- A configuration's complexity is the sum of its parts.

**4.2 Selection rule** (validation window, 1-day and 10-day separately, before the freeze):
1. **Score:** pooled FZ0 at α = 2.5% for every candidate and baseline. Pooled means the per-date mean over the 12 market-sides.
2. **Shortlist:** the model confidence set at 10% (block bootstrap; blocks of 10 days at 1-day, 20 at 10-day).
3. **Calibration screen:** keep the shortlisted configurations that pass all six pooled calibration tests of Q1 (Q3 at 10 days) on the validation window at the 5% level, unadjusted.
4. **Choose:** the lowest-complexity survivor; ties go to the lower mean FZ0. If none passes step 3, take the shortlisted configuration with the fewest failures, then lowest complexity.
5. **Portfolios:** chosen by the same rule over P1–P5.

**4.3 Stress overlay.** It may enter as a candidate only if, on validation, its Brier skill score against the vol-state logistic baseline (Q5) is positive with a 90% block-bootstrap interval that excludes zero.

## 5. Questions, tests and decision rules (test window)

**General settings:**
- All tests are as implemented in `etr/backtest.py` at the registration commit.
- Simulation counts: 4,999 (Christoffersen), 1,999 (DQ, Acerbi–Székely draws, pooled PIT bootstraps), 4,999 (McNeil–Frey bootstrap). The seed is 20261024 throughout.
- Pooled tests average over markets per date before testing, so correlated markets are not double-counted.

**Q1 (primary). Is the selected 1-day engine calibrated?**
- For each side (long, short), across the six markets:
  - pooled coverage at 99% and at 97.5% (`coverage_pooled`, two-sided);
  - pooled ES at 97.5% (`es_pooled`, one-sided for ES too small).
- These six p-values form one family, Holm-corrected.
- **Verdict:**
  - "**calibrated at 1 day**" if no Holm-adjusted p is below 0.05;
  - otherwise "not calibrated", naming the side, level and test.
- **Supporting results per market-side** (12; Holm across the 12 within each test; reported as counts of rejections):
  - exact Kupiec at 99% and 97.5%;
  - Christoffersen conditional coverage at 99%;
  - DQ at 99% and 97.5% (4 lagged hits, the VaR, and weekday dummies of the target day);
  - Acerbi–Székely Z2 at 97.5% (simulated under the model);
  - Du–Escanciano at 97.5%;
  - McNeil–Frey at 97.5%;
  - Basel traffic light at 99% (share of 250-day windows in yellow and red).
- A market-side is called **desk-usable** if it has no Holm-adjusted rejection in Kupiec 99%, Christoffersen or Acerbi–Székely, and no 250-day window in the red zone.

**Q2 (primary). Is it better than the baselines?**
- Pooled FZ0 at α = 2.5% (per-date mean over the 12 market-sides).
- One-sided Diebold–Mariano (HAC, 5 lags) of the engine against each of B1, B2 and B3; Holm over the three.
- **Verdict:** "**better than** Bx" if its Holm-adjusted p is below 0.05. Otherwise "not detectably better" (see the MDEs, section 6), or "worse" if the two-sided test rejects in the other direction.
- **Secondary, descriptive:**
  - FZ0 at α = 1%;
  - tick loss at 99%;
  - per-market-side Diebold–Mariano tests;
  - the model confidence set over all candidates and baselines.

**Q3 (secondary). The 10-day engine.**
- **Calibration** on non-overlapping windows (`nonoverlapping`: a grid every 10 weekdays from 2010-01-01, shared by all markets):
  - pooled coverage at 99% and 97.5%, and pooled ES at 97.5%, per side;
  - Holm over the six;
  - verdict as in Q1.
- **Ranking:** pooled FZ0 at α = 2.5% on all overlapping origins; one-sided Diebold–Mariano (HAC, 20 lags) against the 10-day versions of B1–B3; Holm over the three.
- Per-market-side Kupiec on non-overlapping windows is reported descriptively only, because its power is 13–23% (session 1).

**Q4 (secondary). Portfolios.**
- For P1–P5 at 1 and 10 days: the Q1 per-series tests and Q2's Diebold–Mariano comparison.
- Baselines are applied to each portfolio's own P&L history (historical simulation, RiskMetrics, calibrated EWMA on the portfolio return series).
- For P6: the share of the 50 random books that fail exact Kupiec at 99% (5% level), against the binomial range expected under a correct model.

**Q5 (secondary). Stress early warning.**
- **Event:** realised vol over the next 21 trading days ≥ 1.5 × the calibrated EWMA vol at the origin. The 2× event is also reported.
- **Score:** Brier skill score against a logistic model on the vol state (log of EWMA vol over its 1-year average), pooled across markets.
- **Uncertainty:** 95% block-bootstrap interval (21-day blocks), plus a reliability table by forecast decile.
- Reported whether or not the overlay entered the engine.

## 6. What the tests can detect

**Calibration tests** (`results/p1_test_size.csv`). The simulation design:
- 2,000 days, six markets in three pairs correlated at 0.9, Student-t(5) losses;
- 200 replications, so rejection rates carry a Monte Carlo error of about ±1.5 points;
- the three primary pooled tests rechecked with 1,000 replications (`results/p1_primary_size.csv`).

Each cell is the rejection rate at the 5% level.

| Test | Size (correct model) | Variance 25% too small | Normal tails on t(4) data | Monday variance 1.6× |
|---|---|---|---|---|
| **Pooled coverage 99%** (Q1) | 0.055 (1,000 reps: 0.036) | 1.00 | 0.98 | 0.10 |
| **Pooled coverage 97.5%** (Q1) | 0.045 (0.041) | 1.00 | 0.05 | 0.08 |
| **Pooled ES 97.5%** (Q1) | 0.080 (0.043) | 1.00 | 0.93 | 0.09 |
| Kupiec 99%, one market | 0.045 | 0.74 | 0.51 | 0.06 |
| Kupiec 97.5%, one market | 0.035 | 0.91 | 0.03 | 0.05 |
| Christoffersen CC 99% | 0.045 | 0.73 | 0.48 | 0.05 |
| DQ 99% with weekdays | 0.035 | 0.55 | 0.33 | 0.22 |
| DQ 97.5% with weekdays | 0.045 | 0.73 | 0.06 | **0.47** |
| Acerbi–Székely Z2 97.5% | 0.030 | 0.94 | 0.38 | 0.06 |
| Du–Escanciano 97.5% | 0.040 | 0.93 | 0.52 | 0.09 |
| McNeil–Frey 97.5% | 0.040 | 0.17 | **1.00** | 0.07 |
| 10-day pooled coverage 99% (Q3) | 0.060 | 0.73 | 0.15 | |
| 10-day pooled coverage 97.5% (Q3) | 0.070 | 0.80 | 0.04 | |
| 10-day pooled ES 97.5% (Q3) | 0.065 | 0.84 | 0.12 | |
| 10-day Kupiec 99%, one market | 0.005 | 0.19 | 0.02 | |
| Diebold–Mariano, pooled FZ0, 1-day / 10-day | 0.030 / 0.035 | | | |

What this means:
- **Pooled tests catch scale errors reliably, and normal tails at 99%.** At 97.5%, coverage tests cannot see tail shape (normal and t(4) quantiles nearly coincide there), so the ES tests carry the shape question. McNeil–Frey is the most sensitive to thin tails; Acerbi–Székely and Du–Escanciano are the most sensitive to scale.
- **A weekday effect is visible only to the DQ test with weekday regressors.** Breach counts are right on average, so the pooled coverage tests miss it.
- **At 10 days, calibration tests detect scale errors but hardly tail shape:** sums of ten days have thinner tails. Per-market 10-day tests are almost powerless.

**Ranking** (`results/p1_baselines_scores.csv`). Minimum detectable FZ0 differences over the test window, at 5% two-sided and 80% power (2.8 standard errors, from validation, scaled to about 1,950 days):

| Comparison | 1-day, α = 2.5% | 1-day, α = 1% | 10-day, α = 2.5% |
|---|---|---|---|
| Similar models (e.g. calibrated EWMA vs RiskMetrics) | 0.010 (≈ 9% scale error) | 0.020 (≈ 12%) | 0.026 (≈ 15%) |
| Very different models (historical simulation vs calibrated EWMA) | 0.073 (≈ 27%) | 0.118 (≈ 35%) | 0.21 (beyond anything seen in development) |

A difference below these is reported as "**not detectable** at this sample size", not "no difference". The scale-error translations assume a pure scale error under t(5) losses and are a rough guide only.

## 7. Robustness (descriptive, all reported)

- **Sub-periods:** 2019–20 (COVID, negative WTI), 2021–23 (European gas crisis), 2024–26.
- **Markets:** all tests without European gas (TTF, NBP), whose 2021–23 vol lies far outside the development range.
- **Estimation and rolls:** a rolling 5-year estimation window; a 2-day roll for TTF and NBP.
- **Implied vol:** without the plausibility filter, if implied vol is in the engine.
- **Levels:** rankings at α = 1% and 5%.
- **Runner-up:** the second-choice configuration under rule 4.2, in full.

## 8. Not allowed

- Changing sections 1–5 after any 2019+ result has been seen.
- Choosing or tuning any configuration on test-window results.
- Dropping markets, sides, horizons, levels or sub-periods from the registered tests.
- Reporting only favourable results.
- Re-running the test after seeing results, except to fix a bug. In that case both runs are reported and the bug is described.

## 9. Limitations stated in advance

- **The test window is semi-blind.** General lessons from it (the 2021–23 gas regime, the case for scale-free modelling) already shape this design.
- **Two shocks dominate the window:** COVID with negative WTI, and the 2021–23 gas crisis. Effective breadth is about 4.5 markets (TTF–NBP and Brent–WTI correlate at about 0.9).
- **Implied vol** exists only for HH, Brent and WTI, and refers to front-month options.
- **Aggregation is FX-neutral;** the trading costs of the positions are ignored.
- **Breach-count tests have little power at 99%** for one market, and almost none at 10 days (section 6). The pooled and score-based tests carry the verdicts.

## 10. Amendments before the freeze

None yet. Each amendment must be decided on development data only and record its date, commit and reason. None may change a test, a decision rule or the test window.
