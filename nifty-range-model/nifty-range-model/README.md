# NIFTY Range Model

**We can forecast how far NIFTY will move, better than the industry-standard models. We could not forecast which way it will move from its own price history, and this repository shows both results honestly.**

This project started as one volatility indicator for TradingView and became a 20-year research study on the NIFTY 50 index (2005 → September 2026). Every idea for predicting *direction* failed honest testing. Every test of the *range* (how big the move will be) passed. The final model, **COMBO**, blends three volatility sources with Monte Carlo tails, and on 15 years of out-of-sample data it beat RiskMetrics EWMA, GARCH(1,1), historical volatility and India VIX on standard forecasting and risk-model tests.

> **Not investment advice.** This is research and educational software. It does not predict market direction and does not guarantee profits. Read [Limitations](#11-limitations) before using any number from it.

---

## Table of contents

1. [Results at a glance](#1-results-at-a-glance)
2. [The core idea: magnitude vs direction](#2-the-core-idea-magnitude-vs-direction)
3. [Repository structure](#3-repository-structure)
4. [Quick start](#4-quick-start)
5. [Data](#5-data)
6. [Concepts you need (5-minute primer)](#6-concepts-you-need-5-minute-primer)
7. [The models](#7-the-models)
8. [How everything was tested](#8-how-everything-was-tested)
9. [Experiments and results](#9-experiments-and-results)
10. [What the results mean](#10-what-the-results-mean)
11. [Limitations](#11-limitations)
12. [What you can and cannot claim from this repo](#12-what-you-can-and-cannot-claim-from-this-repo)
13. [Using the TradingView indicator](#13-using-the-tradingview-indicator)
14. [Practical rules for traders](#14-practical-rules-for-traders)
15. [Reproducing every result](#15-reproducing-every-result)
16. [Troubleshooting](#16-troubleshooting)
17. [Mistakes made and corrected](#17-mistakes-made-and-corrected)
18. [Future work](#18-future-work)
19. [Glossary](#19-glossary)
20. [License and disclaimer](#20-license-and-disclaimer)

---

## 1. Results at a glance

### Headline (1-day forecasts, 2011–2026, 3,859 out-of-sample days, walk-forward)

| Model | CRPS (bps, lower = better) | 5% VaR breaches (target 5%) | 1% VaR breaches (target 1%) | Passes all VaR tests? |
|---|---|---|---|---|
| Historical vol (1 year) | 53.58 | 4.85% | 1.76% | No |
| RiskMetrics EWMA (industry standard) | 52.59 | 5.65% | 1.94% | No |
| GARCH(1,1) (academic standard) | 52.67 | 4.30% | 1.35% | No |
| India VIX, raw (market's forecast) | 52.76 | 3.11% | 0.88% | No |
| **COMBO (this repo)** | **52.04** | **4.95%** | **1.06%** | **Yes** |
| COMBO no-VIX (this repo) | 52.41 | 4.98% | 1.17% | Yes |

COMBO beat **every** benchmark on all three accuracy metrics (CRPS, QLIKE, 90% interval score) at the 1-day horizon, with Diebold–Mariano t-statistics between **4 and 10**.

### Everything we tested

| Idea | Question | Verdict |
|---|---|---|
| EWMA volatility range | How far will NIFTY move? | ✅ Works, beats constant volatility |
| Monte Carlo (resampling real past surprises) | What does the range really look like? | ✅ Fixes fat tails |
| India VIX | Does the option market forecast better? | ✅ Wins on scheduled events; overstates volatility by ~14% |
| **COMBO** (EWMA + VIX + long-run + Monte Carlo) | Can we combine them? | ✅ **Best at every horizon** |
| Volatility-breakout strategy | Can volatility give buy/sell signals? | ❌ No edge |
| Markov chains on 5-min candles | Can past candles predict direction? | ❌ No usable skill |
| Support/resistance + price action + ADX | Does a real trader's method have an edge? | ❌ No edge; ATM option buying loses to theta |
| Markov + Monte Carlo forecaster | Pure direction forecast, no trading | ❌ No skill beyond 5 minutes |

**One-line conclusion:** NIFTY's direction was not predictable from its own price history at 5 minutes to 1 hour, but its range is predictable, and COMBO forecasts that range measurably better than the standard tools.

---

## 2. The core idea: magnitude vs direction

Every trade depends on three things:

1. **Direction** — will price go up or down?
2. **Magnitude** — how far will it move?
3. **Cost of time** — for options, theta and the premium built into option prices.

**Magnitude (volatility) is partly predictable** because markets move in moods: calm days follow calm days, wild days follow wild days (*volatility clustering*).

**Direction is very hard to predict** from price history because NIFTY is heavily traded, and any simple repeatable pattern gets traded away.

Think of the range model as a **ruler, not a compass**. It tells you how far, never which way. A ruler is still extremely useful: it sets stop distances, target realism, position size, and whether options are cheap or expensive.

---

## 3. Repository structure

```
nifty-range-model/
├── README.md                    ← you are here
├── requirements.txt             ← Python packages
├── .env.example                 ← template for Angel One credentials (never commit .env)
├── .gitignore
├── indicator/
│   └── ewma_volatility.pine     ← TradingView indicator (Pine Script v6)
├── src/
│   ├── ewma_lab.py              ← data download (Angel One), token lookup, first range/direction tests
│   ├── range_backtest.py        ← 20-year walk-forward range test + the COMBO model
│   └── benchmark.py             ← formal benchmark vs RiskMetrics, GARCH, historical vol, India VIX
├── data/                        ← CSV market data (not committed; create it with `fetch`)
├── results_range/               ← output of range_backtest.py (charts + tables)
└── results_benchmark/           ← output of benchmark.py (benchmark_report.md + tables)
```

### What each file does

| File | Purpose | Commands |
|---|---|---|
| `indicator/ewma_volatility.pine` | Live range, IV comparison and cheap/expensive verdict on TradingView | Paste into Pine Editor |
| `src/ewma_lab.py` | Downloads candles from Angel One; looks up instrument tokens; runs the hit-probability test and the first Markov direction test on 5-min data | `fetch`, `tokens`, `hitprob`, `markov`, `all`, `demo` |
| `src/range_backtest.py` | Walk-forward range test on daily or intraday data; fits and scores COMBO | `run`, `demo` |
| `src/benchmark.py` | Scores COMBO against the standard models on CRPS, QLIKE, interval score and VaR tests | `run`, `demo` |
| `results_range/` | Tables and the 4-panel chart from the 2005–2026 run with India VIX | — |
| `results_benchmark/` | `benchmark_report.md` and tables behind the headline claims | — |

### How the files connect

```
ewma_lab.py fetch ──► data/*.csv ──► range_backtest.py ──► results_range/
                                 └─► benchmark.py ─────► results_benchmark/
                                       (imports range_backtest.py)
```

- `src/benchmark.py` imports `src/range_backtest.py`, so keep them in the same folder.
- `src/range_backtest.py` and `src/ewma_lab.py` stand alone.

Every script has a **`demo`** command that runs on synthetic data, so you can check the code works before downloading anything.

### About the direction experiments

Sections 9.2, 9.5 and 9.6 document three direction experiments (a TradingView vol-breakout strategy, a support/resistance + price action + ADX backtest, and a Markov + Monte Carlo forecaster). All three found **no edge**. Their code is not part of this repository, which contains only the parts that worked; their results are kept here because they explain why the project focuses on range rather than direction.

---

## 4. Quick start

### 4.1 Install

Python 3.10+ (tested on 3.12).

```bash
pip install -r requirements.txt
```

### 4.2 Try it with no data at all

```bash
python src/range_backtest.py demo
python src/benchmark.py demo
```

These run on synthetic, NIFTY-like data and print full reports. They prove the code works; the numbers are not NIFTY results.

### 4.3 Run on real NIFTY data

If you already have `data/nifty_daily.csv` and `data/vix_daily.csv`, skip to step 3.

1. Set your Angel One SmartAPI credentials (see [Data → Credentials](#53-credentials)).
2. Download data:

```bash
python src/ewma_lab.py fetch --interval ONE_DAY --start 2005-01-01 --end 2026-09-29 --out data/nifty_daily.csv
python src/ewma_lab.py tokens --search "INDIA VIX" --exch NSE               # find the VIX token
python src/ewma_lab.py fetch --interval ONE_DAY --exchange NSE --token <VIX token> --start 2008-01-01 --end 2026-09-29 --out data/vix_daily.csv
```

3. Run the main results:

```bash
python src/range_backtest.py run --csv data/nifty_daily.csv --vix-csv data/vix_daily.csv
python src/benchmark.py     run --csv data/nifty_daily.csv --vix-csv data/vix_daily.csv
```

Results print to the console and are saved in `results_range/` and `results_benchmark/` (including `benchmark_report.md`).

---

## 5. Data

### 5.1 Datasets used

| File | Instrument | Bar size | Period | Size | Used for |
|---|---|---|---|---|---|
| `nifty_daily.csv` | NIFTY 50 index (Angel token `99926000`, NSE) | 1 day | 2005-01-03 → 2026-09-29 | 5,350 bars | Range test, COMBO, benchmark |
| `vix_daily.csv` | India VIX (NSE) | 1 day | 2008-01-02 → 2026-09-29 | 4,573 values | VIX models, COMBO, benchmark |
| `nifty_5m.csv` | NIFTY 50 index | 5 min | 2023-01-02 → 2026-09-29 | 69,215 bars, 926 sessions | Markov tests, hit probabilities, S/R backtest |
| `niftybees_5m.csv` | NIFTYBEES ETF | 5 min | 2023 → 2026 | 69,208 matched bars | Volume proxy for the S/R experiment only (not needed for this repo) |

**CSV format** (any of these work): a time column named `datetime`, `timestamp`, `time` or `date` (ISO strings with or without `+05:30`, or UNIX seconds), plus `open, high, low, close` and optionally `volume`. Files saved by `fetch` work directly.

### 5.2 Why a volume proxy? (S/R experiment only)

The NIFTY **index** has zero volume (an index is not traded). The S/R boxes need volume. Angel One only serves **live** futures contracts (~3 months of history), and TradingView export needs a paid plan. So the S/R backtest takes prices from the index and volume from **NIFTYBEES**, an ETF that tracks NIFTY, matched candle by candle. It is a proxy, not the same as futures volume.

### 5.3 Credentials

The `fetch` command reads four environment variables (a `.env` file in the working folder fills any that are missing):

| Variable | What it is |
|---|---|
| `ANGEL_API_KEY` | SmartAPI app key |
| `ANGEL_CLIENT_ID` | Your Angel One client code |
| `ANGEL_PASSWORD` | Your MPIN / password |
| `ANGEL_TOTP_SECRET` | The **base32 key** shown under the QR code on SmartAPI's "Enable TOTP" page — *not* the 6-digit code |

PowerShell (current window only):

```powershell
$env:ANGEL_API_KEY="..."
$env:ANGEL_CLIENT_ID="..."
$env:ANGEL_PASSWORD="..."
$env:ANGEL_TOTP_SECRET="..."
$env:ANGEL_TOTP_SECRET -match '^[A-Za-z2-7=]+$'    # must print True
```

Linux/macOS: `export ANGEL_API_KEY=...` etc.

**Never commit `.env` or real credentials.** Add `.env` to `.gitignore` and share only `.env.example` with placeholder values.

### 5.4 Fetch behaviour

- Downloads in chunks sized to Angel One limits (~60 days for 5-min, ~1,800 days for daily).
- Waits after login and **retries with back-off** (2s, 4s, 8s… up to 60s) on "exceeding access rate".
- **Saves after every chunk and resumes** from the last saved bar if re-run. Because of this, always use a **different file name for each dataset**.

### 5.5 Cleaning rules

- Intraday bars outside 09:15–15:29 IST are dropped; timestamps converted to IST; duplicates removed.
- On intraday data, the **first bar of each session** (which contains the overnight gap) is excluded from volatility estimates.
- Daily data uses close-to-close returns, including overnight moves.

---

## 6. Concepts you need (5-minute primer)

**Log return.** r = ln(Pₜ / Pₜ₋₁). For small moves, 0.01 ≈ +1%. Log returns add up across time.

**Volatility (σ).** The typical size of a move. NIFTY at 22,700 with daily σ = 0.8% → a normal day is about ±180 points.

**Annualising.** σ_day = σ_year / √252 ≈ σ_year / 16. India VIX 13.4 → about 0.84% per day. For 5-min bars: 75 bars/day, so σ_bar = σ_year / √(252 × 75) ≈ σ_year / 137.5.

**Square-root-of-time.** σ over N bars = σ_bar × √N. (On NIFTY intraday this slightly *overstates* moves, because 5-min moves partly snap back.)

**What ±1σ means.** Under a bell curve, price stays inside ±1σ about 68% of the time (2 days in 3), ±2σ about 95%, ±3σ about 99.7%. A probability, never a guarantee.

**Standardised move (surprise).** z = r / σ_forecast. A z of −6 means the move was six times bigger than expected.

**Fat tails and skew.** Real markets have far more extreme moves than a bell curve predicts, and big falls are more common than big rises.

**Clustering and mean reversion.** Big moves follow big moves (clustering), but volatility eventually drifts back to a normal level (mean reversion — the κ(θ − v) term in Heston/Bates models).

**Calibration / coverage.** A 95% range should contain 95% of actual moves. More is too wide; less is too narrow.

**Implied vs realised.** Realised = how much the market actually moved. Implied (IV) = volatility priced into options; India VIX is NIFTY's 30-day IV. Implied usually sits above realised — the *volatility risk premium* option buyers pay.

---

## 7. The models

### 7.1 Historical volatility (CONST) — baseline
Standard deviation of returns over a fixed window (252 days daily; ~3,750 bars on 5-min), every day weighted equally. Simple but slow: in March 2020 it was still mostly measuring calm 2019.

### 7.2 EWMA / RiskMetrics — the indicator's engine

σ²ₜ = λ·σ²ₜ₋₁ + (1 − λ)·r²ₜ, with **λ = 0.94** (J.P. Morgan RiskMetrics, 1996).

- Each step back, a move's weight shrinks by 6%. **Half-life ≈ 11.2 bars** (11 days daily; ~56 minutes on 5-min).
- Seeded with the average of the first 20 squared returns.
- σₜ is known at the close of bar t and forecasts bar t+1.
- **Strength:** reacts fast. **Weakness:** backward-looking (can't see scheduled events) and no mean reversion.

### 7.3 GARCH(1,1) — academic standard

σ²ₜ = ω + α·r²ₜ₋₁ + β·σ²ₜ₋₁, with α + β < 1. Reverts to a long-run level ω / (1 − α − β). Fitted by maximum likelihood, refitted every year on earlier data only (≥ 750 days).

### 7.4 India VIX

σ_day = VIX / (100 × √252). VIX 13.41 → ±0.845% → about ±192 points at NIFTY 22,716.
**Strength:** forward-looking; prices elections, budgets, RBI decisions. **Weakness:** carries a risk premium (~14% above actual volatility in a typical year); daily only; NIFTY only.

### 7.5 Monte Carlo — filtered historical simulation (FHS)

Instead of assuming a bell curve:

1. Compute every past standardised move z = r / σ_previous.
2. Keep only moves from **before** the year being forecast (pool rebuilt yearly; ≥ 250 values).
3. For an H-bar forecast, draw H surprises at random, add them up; repeat **20,000** times.
4. Multiply by the current σ. The spread of outcomes is the forecast range.

This keeps the real fat tails and downside skew.

### 7.6 Markov chains (direction attempt — failed)

| Model | States | Definition |
|---|---|---|
| Markov-3, order 1 | 3 | Last move z < −0.5 / −0.5…0.5 / > 0.5 |
| Markov-3, order 2 | 9 | Last two moves |
| Markov-5, order 1 | 5 | Boundaries −1.2, −0.4, 0.4, 1.2 |
| Regime \| candle \| side | 12 | Vol regime (Low/Normal/High) × last candle (U/D) × above/below EWMA line |

Transition probabilities learned on training data (+1 smoothing); multi-bar horizons walked forward with 20,000 Monte Carlo paths.

### 7.7 S/R + price action + ADX strategy (failed)

A rule-based rebuild of a discretionary 5-min method:
- **Boxes (ChartPrime-style "high-volume boxes"):** 20-bar pivots, confirmed only 20 bars later (no hindsight); created on strong delta volume; height = ATR(200). *Rebuilt from memory of the indicator's logic, not its source code.*
- **Play by ADX(14):** ≥ 25 → breakout then retest (direction from +DI/−DI); < 20 → bounce; 20–25 → no trade.
- **Trigger:** rejection candle (close in top/bottom 40% of its range).
- **Risk:** stop beyond box + 1 × bar σ; target 2R; time stop 3 hours; entries 09:30–14:45; square-off 15:15.
- **P&L:** futures points (1-pt cost) and **ATM option buyer** (Black–Scholes, IV 13%, 2 days to expiry, 1 premium-point cost, theta included).

### 7.8 COMBO — the integrated model ⭐

```
ln σ²_COMBO = 2·ln(scale) + b_E·ln σ²_EWMA + b_V·ln σ²_VIX + b_L·ln σ²_long-run,   with b_E + b_V + b_L = 1
```

Equivalently: **σ_COMBO = scale × σ_EWMA^bE × σ_VIX^bV × σ_LR^bL**, then Monte Carlo on past moves measured in COMBO units.

| Ingredient | Adds | Covers the blind spot of |
|---|---|---|
| EWMA | What the market is doing now | VIX's premium and slowness |
| India VIX | What the option market expects, incl. events | EWMA can't see the calendar |
| Long-run vol | Pull back to normal (mean reversion) | EWMA assumes today lasts forever |
| Scale | Corrects the overall level | Systematic bias (e.g. VIX premium) |
| Monte Carlo | Real fat tails and skew | The bell curve's thin tails |

**How it's fitted:**
- Shares **sum to 1** (if every input doubles, the forecast doubles) → stable fits.
- **Maximum likelihood** of the actual H-bar moves: minimise mean(ln σ² + y²/σ²). Convex → one stable answer.
- **Refitted every year** using only forecasts whose outcomes ended before that year (≥ 500).
- Separate shares for 1, 5 and 21 days.
- **COMBO no-VIX** uses only EWMA + long-run → works on any instrument and on intraday data.

**Learned shares on NIFTY (examples):**

| Horizon | Model | Fitted for | Scale | EWMA | VIX | Long-run |
|---|---|---|---|---|---|---|
| 1 day | COMBO | 2026 | 0.881 | 0.281 | 0.974 | −0.256 |
| 5 days | COMBO | 2026 | 0.934 | 0.192 | 0.938 | −0.130 |
| 21 days | COMBO | 2026 | 1.089 | 0.460 | 0.439 | 0.101 |
| 1 day | no-VIX | 2026 | 1.038 | 0.921 | — | 0.079 |
| 5 days | no-VIX | 2026 | 1.095 | 0.807 | — | 0.193 |
| 21 days | no-VIX | 2020 | 1.133 | 0.580 | — | 0.420 |

---

## 8. How everything was tested

**Ground rules applied everywhere:**

- **No look-ahead.** Every forecast uses only information available at that moment (volatility timing, pivot confirmation, yearly refits).
- **Train and test on different periods.** 5-min tests: first 60% of days train, last 40% test. Daily tests: walk-forward, refit every year.
- **Non-overlapping samples** for significance at multi-bar horizons.
- **Beat a no-skill baseline.** Base rate / memory-less Monte Carlo for direction; constant volatility, RiskMetrics, GARCH and VIX for ranges; random entries for strategies.
- **Significance:** |t| ≥ 2 required. Multiple comparisons noted.

### Metrics

| Metric | What it measures | Better is |
|---|---|---|
| Accuracy / confident-call accuracy | Correct up/down calls | Higher |
| Brier score / Brier skill | Quality of probability forecasts | Lower score / higher skill |
| Coverage | % of moves inside 50/80/90/95/99% ranges | Close to the target |
| Tails | % of moves beyond 2σ, 3σ, 4σ | Close to model's claim |
| CRPS | Accuracy of the whole forecast distribution | Lower |
| QLIKE | Volatility-forecast loss: ln σ² + y²/σ² | Lower |
| Interval score (90%) | Range width + penalty for misses | Lower |

### Statistical tests

| Test | Question | Pass rule |
|---|---|---|
| t-statistic | Is an average effect real? | \|t\| ≥ 2 |
| Binomial (Bonferroni) | Is a state's up-rate different from base? | p < 0.05 / #states |
| Shuffle test | Could the best state's edge be chance? | Beats 95% of shuffles |
| Random-entry placebo | Does the strategy beat random trades? | Beats 95% of runs |
| Diebold–Mariano | Is one forecast's loss really lower? | \|t\| ≥ 2, non-overlapping |
| Kupiec | Right number of VaR breaches? | p ≥ 0.05 |
| Christoffersen | Are breaches spread out, not bunched? | p ≥ 0.05 |
| Basel traffic light | 99% breaches per 250 days | Green ≤ 4, red ≥ 10 |

---

## 9. Experiments and results

### 9.1 Reading the indicator live (29 Sep 2026)
- NIFTY 5-min: EWMA 8.17%, next-bar ±1σ 0.059% ≈ **±13 points**; full session ≈ ±116 points.
- India VIX 13.41 → ±192 points/day: options priced ~65% more movement than the session delivered (partly overnight/event risk).
- Futures on expiry day: EWMA jumped to 13.7% from the closing spike — matched VIX by coincidence. **Compare VIX with the *daily* EWMA, not the 5-min one.**

### 9.2 Vol-breakout strategy (TradingView, Jul–Sep 2026) ❌
*Code not included in this repository; see [About the direction experiments](#about-the-direction-experiments).*

64 trades, 42.19% win rate, **profit factor 1.082**, max drawdown >3× profit. Essentially breakeven. (Results changed with capital because Pine v6 defaults to no leverage; fix with `margin_long = 20, margin_short = 20`.)

### 9.3 Markov direction + trade test (5-min, 2023–2026) ❌
*Reproduce with `python src/ewma_lab.py markov --csv data/nifty_5m.csv`.*

- 5 min: after two down candles, next candle up **52.4%** in test (real) — but worth only ~0.26 points vs a 1-point cost.
- Rule "short after two up candles" (train t = −3.68) **flipped** in test: 6,625 trades, **−8,350 points**, t = −6.54.
- 1 hour: a weak trend tilt (above EWMA line in normal regime drifts slightly up); trade test **+0.30 pts/trade, t = 0.20** → zero edge.

### 9.4 Hit probabilities (1-hour window, 57,463 origins) ✅ range / ❌ 2R targets
*Reproduce with `python src/ewma_lab.py hitprob --csv data/nifty_5m.csv`.*

- EWMA beat constant volatility at every point target (skill +0.03 to +0.05 vs −0.01 to −0.05).
- Typical 1-hour range is 10–15% narrower than √time predicts.
- Moves beyond 3σ: **1.65%** vs 0.27% bell curve (~6×).
- **Target 60 before stop 30 within 1 hour:** hit first only **13.0%** (4.1% in calm markets).

### 9.5 S/R + price action + ADX (5-min + NIFTYBEES volume) ❌
*Code not included in this repository; see [About the direction experiments](#about-the-direction-experiments).*


| Variant | Period | Trades | Futures avg (pts) | Option avg (premium pts) | Option t |
|---|---|---|---|---|---|
| Core | Train | 229 | +0.28 | −4.04 | −2.08 |
| Core | Test | 123 | −6.90 | −9.04 | −3.63 |
| + skip high-vol (chosen on train) | Test | 95 | −1.60 | **−6.60** | −2.35 |
| + both filters | Test | 77 | −0.08 | −5.92 | −1.90 |

**Why options lose (per trade, 2-DTE ATM):** premium ~104; **theta −6.8**; gain from move +2.0; costs −1.0 → **net −5.8**. NIFTY must move ~15 points in your favour just to break even over ~90 minutes. Breakout-retest made +9 to +11 pts/trade in training and **−9 to −11 in testing**.

### 9.6 Markov + Monte Carlo pure forecast (no trading) ❌
*Code not included in this repository; see [About the direction experiments](#about-the-direction-experiments).*


| Horizon | Best Markov accuracy | Markov vs no-memory Monte Carlo |
|---|---|---|
| 5 min | 50.8% | +0.0004 Brier skill, t = 1.32 (not significant) |
| 15 min | 50.7% | **negative** for every model |
| 1 hour | 50.2% | **negative** for every model |

Markov models never produced odds beyond 45/55.

### 9.7 20-year range test (EWMA only, 2007–2026) ✅

| 1-day coverage target | EWMA-normal | Historical vol | EWMA + Monte Carlo |
|---|---|---|---|
| 50% | 52.9% | 57.8% | **49.9%** |
| 80% | 81.0% | 83.3% | **79.7%** |
| 95% | 93.8% | 94.3% | **94.9%** |
| 99% | 97.8% | 97.7% | **98.9%** |

- 4σ days happened **~45×** more often than a bell curve says.
- Weekly −2σ touched **8.4%** vs +2σ 4.7% (crash skew).
- Held through 2008 (93%) and 2020 (94%) at the 95% level.
- At 21 days EWMA had no edge over constant volatility (no mean reversion).

**Biggest 1-day surprises** (date = day the forecast was made; move happened next trading day):

| Forecast date | Move | EWMA 1σ | Surprise | Event |
|---|---|---|---|---|
| 2009-05-15 | +17.74% | ±2.26% | +7.2σ | 2009 election-result upper circuit |
| 2015-08-21 | −5.92% | ±0.88% | −6.9σ | China-led global sell-off |
| 2024-06-03 | −5.93% | ±1.01% | −6.1σ | 2024 election-result day |
| 2019-09-19 | +5.32% | ±0.97% | +5.4σ | Corporate tax cut |
| 2020-03-11 | −8.30% | ±1.72% | −5.0σ | COVID crash |

### 9.8 Adding India VIX (2010–2026) ✅
- VIX-MC beat EWMA-MC at **1 day** (+0.5%, **t = 3.20**); tied at 5 and 21 days.
- Raw VIX ranges too wide (61% of moves inside its 50% range); median VIX/actual ≈ **1.14**.
- **VIX wins on scheduled events:** on election eves VIX surprises were 1.4–2.1σ vs 4.6–4.9σ for EWMA.

### 9.9 COMBO (2011–2026) ✅

| CRPS skill vs constant vol | 1 day | 5 days | 21 days |
|---|---|---|---|
| EWMA + Monte Carlo | +2.1% | +1.4% | +2.0% |
| VIX + Monte Carlo | +2.8% | +2.2% | +2.5% |
| **COMBO** | **+2.9%** (t 10.0) | **+2.3%** (t 3.4) | **+3.1%** (t 2.9) |
| COMBO no-VIX | +2.2% | +1.5% | +2.6% |
| COMBO with bell curve | +2.5% | +1.6% | +0.5% |

Best at every horizon; margin over VIX-MC small and not significant (+0.1% to +0.6%).

![Range backtest: daily moves vs ±2σ bands, 95% coverage by year, forecast vs actual volatility, and skill by model](results_range/range_backtest.png)

### 9.10 Formal benchmark ✅

**Diebold–Mariano t-stats, COMBO vs each benchmark (positive = COMBO better):**

| vs | 1d CRPS | 1d QLIKE | 1d Interval | 5d CRPS | 5d QLIKE | 5d Interval | 21d CRPS |
|---|---|---|---|---|---|---|---|
| Historical vol | +9.99 | +5.79 | +6.99 | +3.42 | +3.01 | +2.13 | +2.88 |
| RiskMetrics EWMA | +7.41 | +4.71 | +4.09 | +3.43 | +2.97 | +2.94 | +2.50 |
| GARCH(1,1) | +8.15 | +5.12 | +4.67 | +3.27 | +2.39 | +2.20 | +2.23 |
| India VIX (raw) | +8.81 | +6.16 | +6.29 | +2.52 | +0.69 | +1.19 | +1.93 |

At 1 day COMBO was the **only** model passing Kupiec (95% and 99%) and Christoffersen. Full tables are in `results_benchmark/benchmark_report.md`.

---

## 10. What the results mean

- **Why direction failed:** NIFTY is heavily traded; simple price patterns get arbitraged away. Small real patterns (5-min snap-back) are too small to beat costs, and patterns that look strong in training often flip.
- **Why the range works:** volatility clusters. Weighting recent moves captures it, and this held for 20 years including 2008 and COVID.
- **Why COMBO wins:** each source covers another's blind spot; a likelihood-fitted blend keeps their strengths.
- **What the learned shares show:**
  - VIX carries the most weight at short horizons — the option market's forward information matters most.
  - The long-run share **grows with horizon** (≈0.08 → 0.19 → 0.24–0.42): mean reversion, measured directly.
  - At 1 day the long-run share is negative: short-term volatility has *momentum*.
  - Scale ≈ 0.88 at 1 day (removes VIX premium), ≈ 1.1 at 21 days (monthly ranges need extra width).
- **Market facts found:** VIX overprices movement (~14%); option buyers pay that premium on average; tails are very fat; crashes are faster than rallies; unscheduled shocks surprise every model.

---

## 11. Limitations

1. **One market.** Only NIFTY was tested, and COMBO's *structure* was chosen after earlier NIFTY tests. An untouched market (e.g. S&P 500 + US VIX) is needed to remove this doubt.
2. **Many comparisons.** 72 benchmark comparisons; a few |t| ≈ 2 wins could be chance. The 1-day wins (t = 4–10) survive strict correction; t = 2.1–2.5 wins are "moderate".
3. **Basic benchmarks.** GARCH(1,1) and RiskMetrics used bell curves; stronger variants (GJR-GARCH-t, HAR, GARCH + Monte Carlo) weren't tested.
4. **Small gains.** 1–3% CRPS, 2–9% interval score — typical for volatility forecasting.
5. **21-day results** rest on only 183 independent months.
6. **Strategy test approximations:** ChartPrime logic rebuilt from memory; NIFTYBEES volume proxy; constant IV 13% and fixed 2 DTE; fills at candle close.
7. **No event awareness** without VIX (intraday and no-VIX versions).
8. **Unscheduled shocks** (COVID, Feb–Mar 2020) remain unforecastable.

---

## 12. What you can and cannot claim from this repo

**Strong (safe to state):**

> On 15 years of out-of-sample NIFTY data (2011–2026, 3,859 days, walk-forward with yearly refits), the COMBO model — blending EWMA, India VIX and long-run volatility with filtered historical simulation — produced 1-day forecasts that beat RiskMetrics EWMA, GARCH(1,1), 1-year historical volatility and raw India VIX on CRPS, QLIKE and 90% interval score (Diebold–Mariano t = 4–10). It was the only model to pass the Kupiec and Christoffersen VaR tests at both 95% and 99% (breach rates 4.95% and 1.06% vs 5% and 1% targets).

**Moderate (state with t-stats):** at 5 days, beats RiskMetrics and GARCH(1,1) on all three metrics (t = 2.2–3.4); at 21 days, lower CRPS than both (t = 2.2–2.5) and tied with VIX.

**Do not claim:** that it predicts direction or makes money; that it beats VIX at 1 month; that it beats "GARCH models" in general.

---

## 13. Using the TradingView indicator

1. Open TradingView → **Pine Editor** → paste `indicator/ewma_volatility.pine` → **Save** → **Add to chart**.
2. Recommended: NIFTY 5-min chart for intraday ranges; daily chart for 1-day / expiry ranges and VIX comparison.

**Pane:**
- EWMA vol line (annualised %), coloured by regime: **teal** calm (≤ 20th percentile), **orange** normal, **red** wild (≥ 80th).
- Grey = rolling realised vol; purple = India VIX (or manual IV).

**Price chart:**
- Blue line = EWMA of price (trend so far, lagging — not a prediction).
- Purple ±2σ bands for the current candle (from the previous candle's σ).
- Triangles = candle closed outside its expected range (a mood change, **not** a buy/sell signal).

**Table:**
- EWMA vol (chart timeframe and daily), implied vol, realised vol, percentile.
- **IV ÷ daily EWMA** verdict: **EXPENSIVE** (≥ 1.3), **FAIR**, **CHEAP** (≤ 1.0).
- Expected ±1σ move in **% and points** for: next bar, 1 hour, 1 day, and to expiry — for both EWMA and IV.

**Key inputs:** λ (0.94), skip overnight gap (on), trading days (252), session minutes (375; 1,440 for crypto), IV symbol (`NSE:INDIAVIX`), manual IV (enter your expiry's ATM IV), days to expiry, cheap/expensive thresholds.

**Alerts:** vol entering high regime, vol entering low regime, expected-move breach, options turned cheap.

---

## 14. Practical rules for traders

| Situation | Rule from the data |
|---|---|
| Stop placement | 1.5–2× the range for your holding time, beyond a level |
| Target realism | Within ~1× the holding-time range; 60+ points in an hour happened ~13% of the time |
| Position size | Lots = ₹ risk ÷ (stop distance × lot size) |
| Option cheap or expensive? | IV ÷ EWMA ≈ 1.14 is normal; < 1.0 genuinely cheap; > 1.3 expensive |
| Weekly expiry range | COMBO 5-day range, or EWMA widened ~10%; extra room on the downside |
| Before elections, budget, RBI | Trust VIX / IV over EWMA |
| After an expiry-day close | 5-min reading starts "hot"; give it ~1 hour |
| Direction | Always your own analysis — the model can't tell you |

**Option-buying arithmetic:** a 2-DTE ATM option loses roughly 4–5 premium points per hour to theta; with delta ≈ 0.5, NIFTY must move ~15 points your way over 90 minutes just to break even.

---

## 15. Reproducing every result

Run from the repository root.

```bash
# --- data ---
python src/ewma_lab.py tokens --search "INDIA VIX" --exch NSE        # look up the VIX token
python src/ewma_lab.py fetch --interval ONE_DAY     --start 2005-01-01 --end 2026-09-29 --out data/nifty_daily.csv
python src/ewma_lab.py fetch --interval ONE_DAY     --exchange NSE --token <VIX token> --start 2008-01-01 --end 2026-09-29 --out data/vix_daily.csv
python src/ewma_lab.py fetch --interval FIVE_MINUTE --start 2023-01-01 --end 2026-09-29 --out data/nifty_5m.csv

# --- main results (Sections 9.7–9.10) ---
python src/range_backtest.py run --csv data/nifty_daily.csv --out results_range_ewma_only   # EWMA-only run (9.7)
python src/range_backtest.py run --csv data/nifty_daily.csv --vix-csv data/vix_daily.csv   # with VIX + COMBO (9.8–9.9)
python src/benchmark.py     run --csv data/nifty_daily.csv --vix-csv data/vix_daily.csv   # formal benchmark (9.10)

# --- intraday tests (Sections 9.3–9.4) ---
python src/ewma_lab.py all --csv data/nifty_5m.csv

# --- sanity checks on synthetic data (no download needed) ---
python src/ewma_lab.py demo
python src/range_backtest.py demo
python src/benchmark.py demo
```

The benchmark is deterministic: with the same `nifty_daily.csv` and `vix_daily.csv`, `benchmark.py` reproduces `results_benchmark/benchmark_report.md` exactly (fixed random seeds, yearly refits on past data only).

Data fetched later will extend past 29 Sep 2026, so the latest year's numbers will differ slightly; pass `--end 2026-09-29` to match the published results.

**Key parameters (fixed before testing):**

| Parameter | Value |
|---|---|
| EWMA λ / seed | 0.94 / 20 returns |
| Regime percentiles | 20th / 80th over 750 bars |
| Monte Carlo paths / min pool | 20,000 / 250 |
| COMBO min training | 500 forecasts |
| GARCH min training | 750 days |
| 5-min train/test split | 60% / 40% of days |
| S/R | 20-bar pivots, ATR(200) boxes, ADX 14 (≥25 trend, <20 range), RR 2, 3-hour time stop |
| Option model | ATM, strike step 50, 2 DTE, IV 13%, 1-pt cost |

---

## 16. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `error: the following arguments are required: cmd` | Script run without a command (e.g. VS Code ▶) | Run `python src/<script>.py demo` or `run …` |
| `unrecognized arguments: --interval` | An old copy of the script is running | Check with `Select-String src/ewma_lab.py -Pattern "ONE_DAY"`; replace the file; delete `ewma_lab (1).py` duplicates |
| `Non-base32 digit found` | `ANGEL_TOTP_SECRET` holds a placeholder or a 6-digit code | Use the base32 key from SmartAPI's TOTP page; close and reopen the terminal if placeholders were set |
| `Access denied because of exceeding access rate` | Angel rate limit | Wait a few minutes and re-run; the fetch resumes |
| `ModuleNotFoundError: No module named 'range_backtest'` | `benchmark.py` moved away from `range_backtest.py` | Keep both in `src/` |
| `ModuleNotFoundError: No module named 'SmartApi'` | Fetch dependencies missing | `pip install -r requirements.txt` |

---

## 17. Mistakes made and corrected

Kept here because they show why the final tests are strict.

| Stage | Mistake | Correction |
|---|---|---|
| First Markov test | Verdict needed only positive profit; p-values ignored overlapping windows → false "EDGE" | Significant t-stat required; non-overlapping samples |
| Markov + MC test | Baseline was a drifting base rate → false "REAL" at 15 min | Baseline = memory-less Monte Carlo; accuracy > 50% required |
| COMBO development | Least-squares blend unstable | Likelihood fit, shares summing to 1 |
| Fetch | No rate-limit handling; lost data | Back-off, per-chunk saves, resume |
| Prediction | Expected VIX to win at 21 days, EWMA at 1 day | It was the reverse; reported as found |
| S/R boxes | TradingView draws pivots with hindsight | Backtest uses confirmation time |

---

## 18. Future work

- [ ] **Independent replication** on S&P 500 + US VIX, code unchanged.
- [ ] **Forward test:** log COMBO forecasts from October 2026 and score after 6–12 months.
- [ ] **Stronger benchmarks:** GJR-GARCH with fat tails, GARCH + Monte Carlo, HAR on intraday realised vol.
- [ ] **COMBO inside the TradingView indicator.**
- [ ] **Volatility-targeted NIFTY:** exposure = target σ / COMBO σ — a strategy that needs no direction forecast.
- [ ] **Direction research done properly** with external data (flows, global cues, breadth, fundamentals), stock ranking, and an agent + algorithm system where LLMs read and explain while code computes and validation decides.

**Traps for direction research:** look-ahead in fundamental data (use point-in-time values), LLM leakage (LLMs may know historical outcomes — validate LLM signals forward only), too many features, survivorship bias.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| ADX | Average Directional Index — trend strength (+DI/−DI give direction) |
| ATM | At the money — strike nearest the current price |
| ATR | Average True Range — typical candle range |
| Basel traffic light | Bank rule: 99% VaR breaches in 250 days, green ≤ 4, red ≥ 10 |
| Brier score | Squared error of probability forecasts |
| Calibration | A forecast is right as often as it claims |
| COMBO | This repo's blend of EWMA, VIX and long-run volatility with Monte Carlo tails |
| CRPS | Score for a whole forecast distribution; lower is better |
| Delta | Option price change per 1-point move in the underlying |
| Diebold–Mariano | Test of whether one forecast's loss is really lower |
| DTE | Days to expiry |
| EWMA | Exponentially weighted moving average |
| Fat tails | Extreme moves more frequent than a bell curve predicts |
| FHS | Filtered historical simulation — Monte Carlo from past standardised moves |
| GARCH | Volatility model with reaction, persistence and mean reversion |
| Half-life | Bars until a shock's influence halves |
| IV | Implied volatility from option prices |
| Kupiec / Christoffersen | VaR tests: right number of breaches / breaches not bunched |
| Look-ahead bias | Accidentally using future information in a backtest |
| Mean reversion | Tendency to return to a long-run level |
| QLIKE | Volatility-forecast loss; lower is better |
| RiskMetrics | J.P. Morgan's EWMA standard, λ = 0.94 |
| σ (sigma) | Typical size of a move (standard deviation) |
| Theta | Value an option loses with time |
| VaR | Value at Risk — loss level breached only a set % of the time |
| Volatility risk premium | IV minus realised volatility — what option buyers pay on average |
| Walk-forward | Refit on past data, test on the next period, repeat |

---

## 20. License and disclaimer

**License:** the TradingView indicator carries a Mozilla Public License 2.0 header. Add a `LICENSE` file stating the license for the whole repository (MPL-2.0 keeps it consistent with the indicator).

**Disclaimer:** This repository is for research and education only. It is **not investment advice**, does not predict market direction, and makes no promise of returns. Past performance in backtests does not guarantee future results. Trading derivatives involves substantial risk of loss. Market data belongs to its providers; check their terms before redistributing. If you build a commercial product on this work in India, check SEBI regulations (Research Analyst / Investment Adviser rules) with a qualified professional.
