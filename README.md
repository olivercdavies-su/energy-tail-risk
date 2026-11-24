# Energy Tail Risk

**Question.** Can a daily Value-at-Risk and Expected Shortfall engine for energy futures positions (TTF, NBP, Henry Hub, Brent, WTI, EU carbon; 1-day and 10-day; single markets and a small portfolio) pass the standard risk-management backtests well enough for a risk desk to use?

It builds on [energy-vol-targeting](https://github.com/olivercdavies-su/energy-vol-targeting). That project found that volatility forecasts are good enough for sizing, but that the tails beyond the 95% level are not modelled well. Its data pipeline (contract-consistent returns built from exchange expiry rules) is reused here.

> **Status:** Phase 6 complete. The evaluation toolkit is built and checked by simulation, the final test is pre-registered in [`PROTOCOL.md`](PROTOCOL.md) (one amendment, A1), and the volatility models, tail shapes, the 1-day and 10-day engines, the portfolio methods, the learned models and the stress early warning are built and evaluated on development data. The 2019–26 test window stays sealed until the frozen engine is run once. See [`PROJECT_LOG.md`](PROJECT_LOG.md).

## How the engine will be judged

Every forecast is scored on both tails: a unit long position (loses when prices fall) and a unit short (loses when they rise). There are two questions:
- **Is it calibrated?** Breach rates at 99% and 97.5%, and Expected Shortfall, are tested pooled across markets. Markets that crash together count as the partly redundant evidence they are. Supporting per-market tests: Kupiec, Christoffersen, dynamic quantile, Acerbi–Székely, Du–Escanciano, McNeil–Frey and the Basel traffic light.
- **Is it better than what desks use?** Historical simulation, RiskMetrics and calibrated EWMA are ranked by FZ0, a scoring function for VaR and ES jointly, with Diebold–Mariano tests and a model confidence set.

Each test's size and power at this project's sample sizes were checked by simulation first ([`results/p1_test_size.csv`](results/p1_test_size.csv)).

## First look at the tails (2011–2018, development data only)

![Daily tails after scaling by the calibrated EWMA forecast](results/p0_tail_curves.png)

After scaling by a volatility forecast, a Gaussian 99% VaR is breached on 0.7–2.2% of days per tail instead of 1%. Which tail is heavy depends on the market: oil's left tail (long positions) and NBP's right tail (short positions). Details and the other findings are in the project log.

## Scale models (2015–2018 validation, development data only)

![FZ0 gain of each scale model over calibrated EWMA](results/p2_scale_fz0.png)

- **1-day:** combinations of GJR-GARCH, HAR, weekday EWMA and implied vol beat calibrated EWMA on the joint VaR–ES score. A volatility model alone does not fix the 99% tail: with normal tails, long positions still breach 1.4–1.7% of the time. That is the job of the tail-shape models (Phase 3).
- **10-day:** each model's own term structure beats the square-root-of-time rule.

## Tail shapes and the 1-day engine (2015–2018 validation)

![FZ0 gain by scale and shape](results/p3_configs.png)

- Every volatility model was combined with every tail shape (normal, Student-t, skewed-t, filtered historical simulation, extreme-value GPD tails; each pooled across markets or per market). That makes 63 configurations.
- **Fat tails fix calibration:** with any fat-tailed shape and a non-EWMA volatility model, all six pooled calibration tests pass. With normal tails, 2–5 fail.
- **Which fat-tailed shape is used matters far less than which volatility model.**
- The pre-registered rule picks the simplest configuration that is statistically tied with the best and passes calibration: **GJR-GARCH with a pooled Student-t**. Across the 12 market-sides it has no Holm-adjusted rejection in any backtest. The choice is provisional until the learned models join the candidate set.

## The 10-day horizon (2015–2018 validation)

![10-day FZ0 gain by configuration](results/p4_configs.png)

- Three ways to get a 10-day loss distribution were compared:
  - scaling the 1-day engine by √10 (the Basel convention);
  - each model's own 10-day forecast with a shape fitted to 10-day residuals;
  - filtered historical simulation along GJR-GARCH paths.
- **At 10 days the data can say much less.** 84 of 130 configurations are statistically indistinguishable from the best. Whether the term structures beat √10 reverses between 2012–14 and 2015–18.
- **As registered, the selection rule picked historical simulation**, which scores worse than calibrated EWMA. In turbulent markets it breached its 99% VaR 3.4% of the time, against 1% expected. A guard now requires the chosen engine to be no worse than calibrated EWMA (protocol Amendment A1, recorded before any test data is seen). Under it the 10-day choice is **HAR volatility × √10 with a per-market Student-t**.

## Portfolios (2015–2018 validation)

![Book-level FZ0 gain by method](results/p5_configs.png)

- **The books:** five registered books (long and short baskets, TTF–NBP and Brent–WTI spreads, European gas vs oil) and 50 random books.
- **Two ways to forecast them:**
  - the engine on each book's own return history;
  - joint filtered historical simulation, which applies each market's volatility to whole days of past joint moves.
- **For books, simple wins.** EWMA on the book with a Student-t tail is the 1-day choice. At 10 days nothing beats calibrated EWMA × √10, so it is kept. With these, 1 and 3 of the 50 random books fail the 99% Kupiec test, inside the 0–5 expected by chance.
- **Joint simulation does well on outright books but badly on spreads.** The Brent–WTI relationship tightened (correlation of standardised moves 0.76–0.87 in 2011–14, 0.93–0.98 in 2015–18), and resampling the whole history carries the loose years forward. Its spread VaR comes out about 50% too high.

## Learned models and the stress early warning (2015–2018 validation)

![Reliability of the stress early warning](results/p6_reliability.png)

- **Early warning:** the probability that volatility over the next 21 days reaches 1.5× its calibrated forecast. It is a pooled logistic model on eight inputs known at the forecast date: vol state, recent moves, implied/realised vol, curve slope and Monday.
  - Its Brier score is 3.8% better than a model using the vol state alone (90% interval 1.2–6.2%), so it passes the pre-registered bar.
  - The gain comes from ranking: on the 10% of days it rates riskiest, the event happens 22% of the time.
  - At 2× the events are too rare (2% of days) to judge.
- **Learned VaR/ES model:** the chosen engine's volatility times a multiplier, exp(linear function of the same inputs), fitted by minimising FZ0 directly.
  - At 1 day it scores slightly better (p = 0.08) but reverses in 2012–14 and fails two short-side calibration tests. At 10 days it is worse.
  - The clearest input is implied vol above realised, which widens both tails.
- **A small neural network** (one layer of 8 units, started from the linear model and stopped early on the latest training data) learns nothing beyond the linear model. Early stopping halts it after a median of 4 steps, and it is slightly worse than the linear model in both windows.
- All of these go into the final selection as registered. As things stand, the calibration screen would keep the simpler engines.

## Data and licensing

Daily Bloomberg futures data, 2010–2026. **Bloomberg's licence forbids redistributing the data, so none of it is in this repository, raw or derived.** The repo holds only code, aggregated results tables and charts.

Safeguards:
- `.gitignore` excludes `data/` and every common data-file format.
- `scripts/hooks/pre-commit` blocks commits of anything under a `data/` folder, data-file formats, notebooks, CSVs outside `results/`, results files long enough to be a daily series, and files over 2 MB.
- `scripts/hooks/commit-msg` keeps authorship clean.

Enable both once per clone:
```
git config core.hooksPath scripts/hooks
```

## Reproducing

Requires the Bloomberg workbook (not distributed) and Python 3.11+.

```
pip install -e ".[dev]"
git config core.hooksPath scripts/hooks
cp <your workbook>.xlsx data/raw/
python scripts/build_data.py        # -> data/processed/ (gitignored)
python scripts/p0_reverify.py       # re-verification tables
python scripts/p0_tails.py          # tail exploration tables and chart
python scripts/p1_test_size.py      # size and power of every test, by simulation (~15 min)
python scripts/p1_baselines.py      # baselines on 2012-18 development data
python scripts/p2_scale.py          # volatility models, 2012-18
python scripts/p3_tails.py          # tail shapes, all configurations, the selection rule
python scripts/p4_ten_day.py        # the 10-day horizon (about 40 minutes)
python scripts/p5_portfolios.py     # portfolios P1-P5 and the random books (about an hour)
python scripts/p6_learned.py        # stress early warning and the learned linear model
python scripts/p6_network.py        # the small neural network (about 5 minutes)
pytest                              # real-data tests skip without the workbook
python scripts/mutation_check.py    # injects known bugs; each must make a test fail
```

## Licence

Code: MIT (to be added). Data: not included, see above.
