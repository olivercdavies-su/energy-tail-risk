# Project log

What was decided, why, and what comes next, so any session can pick up from here. Newest session at the top. The proposed phased plan is at the bottom.

---

## Session 1 (2026-10-24): setup, re-verification, tail exploration, proposed plan

**Status: waiting for Oliver's approval of the plan below.** Nothing beyond setup, the data pipeline port and exploration has been built.

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

## Proposed plan

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
