#!/usr/bin/env python3
"""
range_backtest.py — 20-year walk-forward test of the EWMA + Monte Carlo RANGE forecast (NIFTY)

Works on DAILY data (2005 → today: 2008 crash, 2009 election circuit, 2016 demonetisation,
2020 COVID, 2024 election day …) and on INTRADAY data (5-min etc.). Timeframe is auto-detected.

Forecast models for the move over the next H bars (all use ONLY information available at the time):
  EWMA-normal : σ from EWMA (λ = 0.94), bell-curve range                  ← your TradingView indicator
  EWMA-MC     : σ from EWMA, Monte Carlo that resamples PAST standardized moves
                (filtered historical simulation, refitted each year → real fat tails, no look-ahead)
  CONST       : σ = plain 1-year rolling std (what most people use)        ← baseline
  VIX         : σ from India VIX (optional --vix-csv)                      ← the market's own forecast

Checks, per horizon:
  • Coverage   : does the 50/80/90/95/99% range contain the actual move that often?
  • Tails      : how often moves exceed 2σ, 3σ, 4σ vs the bell curve
  • Touch      : chance of touching ±1σ / ±2σ levels (using real highs/lows) vs prediction
  • Score      : CRPS (one number for the whole forecast distribution — lower = better) + skill vs CONST
  • By year    : is it stable across 20 years of very different markets?
  • Worst days : the biggest surprises, in σ units, with dates

Usage
  python ewma_lab.py fetch --interval ONE_DAY --start 2005-01-01 --end 2026-09-29 --out nifty_daily.csv
  python range_backtest.py run --csv nifty_daily.csv                       # H = 1, 5, 21 days
  python ewma_lab.py fetch --interval ONE_DAY --token <INDIA VIX token> --start 2008-01-01 --end 2026-09-29 --out vix_daily.csv
  python range_backtest.py run --csv nifty_daily.csv --vix-csv vix_daily.csv
  python range_backtest.py run --csv nifty_5m.csv                          # intraday: H = 1, 12, 36 bars
  python range_backtest.py demo
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def _parse_time(t: pd.Series) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(t):
        ts = pd.to_datetime(t, unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    else:
        s = t.astype(str)
        ts = (pd.to_datetime(s, utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
              if s.str.contains(r"\+|Z$", regex=True).any() else pd.to_datetime(s))
    return pd.DatetimeIndex(ts)


def load_any(path: str):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in ("datetime", "timestamp", "time", "date") if c in df.columns), None)
    if tcol is None:
        raise SystemExit("CSV needs a datetime/timestamp/time/date column")
    idx = _parse_time(df[tcol])
    cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    out = df[cols].astype(float)
    out.index = idx
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for c in ("open", "high", "low"):
        if c not in out:
            out[c] = out["close"]
    step = pd.Series(out.index).diff().median()
    daily = step >= pd.Timedelta(hours=20)
    if daily:
        out.index = out.index.normalize()
        out = out[~out.index.duplicated(keep="last")]
        bar_min = None
    else:
        out = out.between_time("09:15", "15:29")
        bar_min = int(round(step.total_seconds() / 60))
    out = out[out["close"] > 0]
    return out, daily, bar_min


def load_vix(path: str) -> pd.Series:
    v, _, _ = load_any(path)
    return v["close"].rename("vix")


def synthetic_daily(years=21, seed=5) -> pd.DataFrame:
    """GARCH + jumps + occasional crash clusters, ~NIFTY-like. For checking the machinery only."""
    rng = np.random.default_rng(seed)
    n = 252 * years
    lr = (0.18 / np.sqrt(252)) ** 2
    a, b = 0.08, 0.90
    w = lr * (1 - a - b)
    h, p = lr, 2500.0
    O, H, L, C, TV = (np.empty(n) for _ in range(5))
    for i in range(n):
        z = rng.standard_t(5) / np.sqrt(5 / 3)
        jump = rng.normal(-0.04, 0.03) if rng.random() < 0.004 else 0.0
        r = np.sqrt(h) * z + jump + 0.0004
        o = p * np.exp(rng.normal(0, 0.3) * np.sqrt(h))
        c = p * np.exp(r)
        hi = max(o, c) * np.exp(abs(rng.normal(0, 0.5)) * np.sqrt(h))
        lo = min(o, c) * np.exp(-abs(rng.normal(0, 0.5)) * np.sqrt(h))
        O[i], H[i], L[i], C[i] = o, hi, lo, c
        h = w + a * (r - 0.0004) ** 2 + b * h
        TV[i] = np.sqrt(h * 252) * 100          # true next-day vol, annualized %
        p = c
    idx = pd.bdate_range("2005-01-03", periods=n)
    df = pd.DataFrame({"open": O, "high": H, "low": L, "close": C}, index=idx)
    # demo "VIX": knows the true vol, adds a 15% risk premium and noise, starts in 2008 like India VIX
    vix = pd.Series(TV * 1.15 * rng.lognormal(0, 0.10, n), index=idx, name="vix")
    return df, vix[vix.index >= "2008-03-03"]


# ──────────────────────────────────────────────────────────────────────────────
# Volatility forecasts (no look-ahead)
# ──────────────────────────────────────────────────────────────────────────────
def prepare(df, daily, bar_min, lam=0.94, seed_len=20, tdays=252, sess_min=375, vix=None):
    d = df.copy()
    d["date"] = d.index.normalize()
    d["r"] = np.log(d["close"]).diff()
    if daily:
        d["r_use"] = d["r"]
        bars_year = tdays
        const_win = 252
    else:
        first = d["date"].ne(d["date"].shift())
        d["r_use"] = d["r"].where(~first)                     # skip overnight gap bars
        bars_year = tdays * sess_min / bar_min
        const_win = int(252 * sess_min / bar_min / 5)          # ≈ 50 trading days of bars
    var = np.full(len(d), np.nan)
    v, cnt, s = np.nan, 0, 0.0
    for i, x in enumerate(d["r_use"].to_numpy()):
        if not np.isnan(x):
            cnt += 1
            if cnt <= seed_len:
                s += x * x
                if cnt == seed_len:
                    v = s / seed_len
            else:
                v = lam * v + (1 - lam) * x * x
        var[i] = v
    d["sig"] = np.sqrt(var)                                    # forecast for the NEXT bar
    d["sig_c"] = d["r_use"].rolling(const_win, min_periods=const_win // 3).std()
    d["z"] = d["r_use"] / d["sig"].shift(1)                    # standardized surprise
    if vix is not None:
        v = pd.merge_asof(pd.DataFrame({"t": d.index}),
                          vix.rename("vix").rename_axis("t").reset_index().sort_values("t"),
                          on="t", direction="backward")["vix"].to_numpy()
        # VIX close is known at the close of the same bar; stale values older than 5 days are dropped
        vt = pd.merge_asof(pd.DataFrame({"t": d.index}),
                           pd.DataFrame({"t": vix.index, "vt": vix.index}).sort_values("t"),
                           on="t", direction="backward")["vt"]
        stale = (pd.Series(d.index) - vt).dt.days.to_numpy() > 5
        v = np.where(stale, np.nan, v)
        d["vix"] = v
        d["sig_v"] = v / 100 / np.sqrt(bars_year)
        d["z_v"] = d["r_use"] / d["sig_v"].shift(1)
    d.attrs.update(daily=daily, bars_year=bars_year, bar_min=bar_min, lam=lam)
    return d


# ──────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ──────────────────────────────────────────────────────────────────────────────
def crps_normal(y, s):
    z = y / s
    return s * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def crps_sorted(xs, y):
    n = len(xs)
    cs = np.concatenate([[0.0], np.cumsum(xs)])
    k = np.searchsorted(xs, y)
    e_abs = ((y * k - cs[k]) + ((cs[n] - cs[k]) - y * (n - k))) / n
    i = np.arange(1, n + 1)
    return e_abs - 0.5 * (2 * np.sum((2 * i - n - 1) * xs) / (n * n))


def fhs_by_year(d, zcol, H, years, n_paths, rng, min_pool=250):
    """For each year: Monte Carlo sum of H past standardized moves, pool = everything BEFORE that year."""
    out = {}
    z, dates = d[zcol].to_numpy(), d["date"].to_numpy()
    for y in years:
        pool = z[(dates < np.datetime64(f"{y}-01-01")) & np.isfinite(z)]
        if len(pool) < min_pool:
            continue
        out[y] = np.sort(pool[rng.integers(len(pool), size=(n_paths, H))].sum(1))
    return out


def model_specs(d):
    """name → (σ column, standardized-move column for Monte Carlo or None for bell curve)"""
    m = {"EWMA-normal": ("sig", None), "EWMA-MC": ("sig", "z"), "CONST": ("sig_c", None)}
    if "sig_v" in d:
        m["VIX-normal"] = ("sig_v", None)
        m["VIX-MC"] = ("sig_v", "z_v")
    return m



def fit_combos(d, H, idx, years, min_rows=500):
    """COMBINED model, refitted every year on PAST data only.

    Volatility forecast for the next H bars (a log-linear blend, like HAR/GARCH-X models):
        log σ²_combo = a + bE·log σ²_EWMA + bV·log σ²_VIX + bL·log σ²_long-run
    • bE → today's realized volatility (EWMA)
    • bV → the option market's forward view incl. scheduled events (VIX); 'a' removes VIX's premium
    • bL → pull toward the long-run level = mean reversion (the κθ idea from Heston/Bates), grows with H
    Fit: maximum likelihood of the actual H-bar moves (convex → one stable answer, robust to crashes).
    Monte Carlo: resample PAST H-bar moves measured in combo-σ units (real fat tails AND skew).
    """
    from scipy.optimize import minimize
    daily = d.attrs["daily"]
    n = len(d)
    c, dates = d["close"].to_numpy(), d["date"].to_numpy()
    t = np.arange(n - H)
    y_all = np.log(c[t + H] / c[t])
    win_ok = np.ones(len(t), bool)
    if not daily:
        win_ok &= dates[t + H] == dates[t]
    end_date = dates[t + H]

    logf = {"EWMA": np.log(d["sig"].to_numpy() ** 2), "LR": np.log(d["sig_c"].to_numpy() ** 2)}
    if "sig_v" in d:
        logf["VIX"] = np.log(d["sig_v"].to_numpy() ** 2)
    variants = ({"COMBO (EWMA+VIX+LR)": ["EWMA", "VIX", "LR"], "COMBO no-VIX (EWMA+LR)": ["EWMA", "LR"]}
                if "VIX" in logf else {"COMBO (EWMA+LR)": ["EWMA", "LR"]})

    def nll(theta, D, off, y2):
        eta = off + D @ theta                            # log per-bar variance
        e = y2 / (H * np.exp(eta))
        return np.mean(eta + e), D.T @ (1 - e) / len(y2)

    out = {}
    for name, cols in variants.items():
        # shares sum to 1 (if every input doubles, the forecast doubles) → stable fits even on short history
        others = [k for k in cols if k != "LR"]
        off_all = logf["LR"]
        D_all = np.column_stack([np.ones(n)] + [logf[k] - logf["LR"] for k in others])
        good = win_ok & np.isfinite(D_all[t]).all(1) & np.isfinite(off_all[t]) & np.isfinite(y_all)
        sb = np.full(len(idx), np.nan)
        fhs, weights = {}, {}
        theta0 = np.r_[0.0, np.full(len(others), 1.0 / len(cols))]
        for Y in np.unique(years):
            tr = t[good & (end_date < np.datetime64(f"{Y}-01-01"))]
            if len(tr) < min_rows:
                continue
            D, off, y2 = D_all[tr], off_all[tr], y_all[tr] ** 2
            th = minimize(nll, theta0, args=(D, off, y2), jac=True, method="L-BFGS-B").x
            theta0 = th
            s_tr = np.sqrt(np.exp(off + D @ th))
            fhs[Y] = np.sort(y_all[tr] / s_tr)                  # past moves in combo-σ units
            m = years == Y
            sb[m] = np.sqrt(np.exp(off_all[idx[m]] + D_all[idx[m]] @ th))
            w = dict(zip(others, th[1:]))
            w["LR"] = 1 - th[1:].sum()
            weights[Y] = {"scale": np.exp(th[0] / 2), **{k: w[k] for k in cols}}
        out[name] = dict(sb=sb, fhs=fhs, weights=weights)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Test one horizon — every model scored on the SAME forecasts
# ──────────────────────────────────────────────────────────────────────────────
def test_horizon(d, H, n_paths=20000, seed=0):
    rng = np.random.default_rng(seed)
    daily, bars_year = d.attrs["daily"], d.attrs["bars_year"]
    specs = model_specs(d)
    c, hi, lo = d["close"].to_numpy(), d["high"].to_numpy(), d["low"].to_numpy()
    dates, n = d["date"].to_numpy(), len(d)

    ok = np.zeros(n, bool)
    ok[:n - H] = True
    if not daily:
        ok[:n - H] &= dates[H:] == dates[:n - H]
    for col, _ in specs.values():
        ok &= np.isfinite(d[col].to_numpy())
    idx = np.where(ok)[0]
    idx = idx[idx < n - H]
    years = pd.DatetimeIndex(dates[idx]).year.to_numpy()

    # base models
    M = {}
    for name, (col, zc) in specs.items():
        M[name] = dict(sb=d[col].to_numpy()[idx],
                       fhs=fhs_by_year(d, zc, H, np.unique(years), n_paths, rng) if zc else None)
    # combined models
    combos = fit_combos(d, H, idx, years)
    res_weights = {}
    for name, cm in combos.items():
        M[name] = dict(sb=cm["sb"], fhs=cm["fhs"])
        res_weights[name] = cm["weights"]
    full = next(iter(combos))
    M[full.replace("COMBO", "COMBO-normal").split(" (")[0] + " (bell curve)"] = dict(sb=combos[full]["sb"], fhs=None)

    common = set(np.unique(years))
    for m_ in M.values():
        if m_["fhs"] is not None:
            common &= set(m_["fhs"])
    keep = np.isin(years, list(common))
    for m_ in M.values():
        keep &= np.isfinite(m_["sb"])
    idx, years = idx[keep], years[keep]
    for m_ in M.values():
        m_["sb"] = m_["sb"][keep]

    S = c[idx]
    y = np.log(c[idx + H] / S)
    mx = sliding_window_view(hi[1:], H)[idx].max(1)
    mn = sliding_window_view(lo[1:], H)[idx].min(1)
    res = dict(H=H, n=len(idx), idx=idx, years=years, y=y, S=S, models=list(M), weights=res_weights,
               combo_name=full)

    cov, tails, touch, crps, by_year_cols = {}, {}, {}, {}, {}
    for name, m_ in M.items():
        sb, fh = m_["sb"], m_["fhs"]
        zc = fh
        sH = sb * np.sqrt(H)
        zH = y / sH
        res[f"sH_{name}"] = sH
        # coverage and CRPS
        if zc is None:
            cov[name] = {lv: np.mean(np.abs(zH) <= norm.ppf(0.5 + lv / 2)) for lv in LEVELS}
            cr = crps_normal(y, sH)
            inside95 = np.abs(zH) <= norm.ppf(0.975)
        else:
            cov[name], cr, inside95 = {}, np.empty(len(idx)), np.empty(len(idx), bool)
            u = y / sb
            ins = {lv: np.empty(len(idx), bool) for lv in LEVELS}
            for yr in np.unique(years):
                m = years == yr
                sims = fh[yr]
                cr[m] = sb[m] * crps_sorted(sims, u[m])
                for lv in LEVELS:
                    a, b = np.quantile(sims, [0.5 - lv / 2, 0.5 + lv / 2])
                    ins[lv][m] = (u[m] >= a) & (u[m] <= b)
            cov[name] = {lv: ins[lv].mean() for lv in LEVELS}
            inside95 = ins[0.95]
        crps[name] = cr
        if zc is None:
            tails[name] = {k: np.mean(np.abs(zH) > k) for k in (2, 3, 4)}
            touch[name] = {f"±{k:g}σ": (np.mean(mx >= S * np.exp(k * sH)) + np.mean(mn <= S * np.exp(-k * sH))) / 2
                           for k in (1.0, 2.0)}
        by_year_cols[name] = (sH, inside95)

    res["coverage"] = pd.DataFrame(cov)
    t = pd.DataFrame(tails)
    t.insert(0, "bell curve", [2 * norm.sf(k) for k in t.index])
    res["tails"] = t
    tc = pd.DataFrame(touch)
    tc.insert(0, "bell curve", [2 * norm.sf(k) for k in (1.0, 2.0)])
    res["touch"] = tc

    # scores + significance (non-overlapping forecasts)
    sub = np.arange(0, len(idx), H)
    base = crps["CONST"]
    rows = {}
    for name, cr in crps.items():
        diff = base[sub] - cr[sub]
        t_ = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if name != "CONST" else np.nan
        rows[name] = dict(crps_bps=cr.mean() * 1e4, skill_vs_const=1 - cr.mean() / base.mean(), t_vs_const=t_)
    res["scores"] = pd.DataFrame(rows).T
    res["crps_all"] = crps
    singles = [k for k in ("EWMA-normal", "EWMA-MC", "VIX-normal", "VIX-MC", "CONST") if k in crps]
    best_single = min(singles, key=lambda k: crps[k].mean())
    diff = crps[best_single][sub] - crps[full][sub]
    res["combo_vs_best"] = (best_single, 1 - crps[full].mean() / crps[best_single].mean(),
                            diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))
    if "VIX-MC" in crps:
        best_ewma = min(("EWMA-normal", "EWMA-MC"), key=lambda k: crps[k].mean())
        best_vix = min(("VIX-normal", "VIX-MC"), key=lambda k: crps[k].mean())
        diff = crps[best_ewma][sub] - crps[best_vix][sub]
        res["head_to_head"] = (best_ewma, best_vix, 1 - crps[best_vix].mean() / crps[best_ewma].mean(),
                               diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))

    # by year
    rows = []
    for yr in np.unique(years):
        m = years == yr
        row = dict(year=yr, n=int(m.sum()), real_vol=np.sqrt(np.mean(y[m] ** 2) / H * bars_year) * 100)
        for name in ("EWMA-normal", "VIX-normal", full):
            if name in by_year_cols:
                lab = "COMBO" if name == full else name.split('-')[0]
                row[f"pred_{lab}"] = np.mean(by_year_cols[name][0][m]) / np.sqrt(H) * np.sqrt(bars_year) * 100
        for name in ("EWMA-MC", "VIX-MC", "CONST", full):
            if name in by_year_cols:
                row[f"cov95_{'COMBO' if name == full else name}"] = by_year_cols[name][1][m].mean()
        rows.append(row)
    res["by_year"] = pd.DataFrame(rows).set_index("year")

    # biggest surprises vs EWMA
    zE = y / res["sH_EWMA-normal"]
    ev = pd.DataFrame(dict(forecast_date=d.index[idx], move_pct=(np.exp(y) - 1) * 100,
                           ewma_1sig_pct=res["sH_EWMA-normal"] * 100, z_ewma=zE))
    if "sH_VIX-normal" in res:
        ev["vix_1sig_pct"] = res["sH_VIX-normal"] * 100
        ev["z_vix"] = y / res["sH_VIX-normal"]
    ev["abs_z"] = ev["z_ewma"].abs()
    ev["date"] = ev["forecast_date"].dt.normalize()
    res["events"] = ev.sort_values("abs_z", ascending=False).drop_duplicates("date").head(12).drop(
        columns=["abs_z", "date"])
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────
def _block(df, fmt):
    txt = df.apply(lambda col: col.map(lambda v: "—" if pd.isna(v) else fmt(col.name, v)))
    return txt.to_string().replace("\n", "\n    ").join(["    ", ""])


def report(d, horizons, outdir="results_range", tag=""):
    os.makedirs(outdir, exist_ok=True)
    daily = d.attrs["daily"]
    unit = "day" if daily else f"{d.attrs['bar_min']}-min bar"
    has_vix = "sig_v" in d
    print("\n" + "═" * 100)
    print(f" EWMA + MONTE CARLO RANGE BACKTEST {tag}")
    print(f" {d.index[0].date()} → {d.index[-1].date()} │ {len(d):,} bars ({'daily' if daily else 'intraday'}) │ "
          f"λ = {d.attrs['lam']} │ Monte Carlo refitted every year on past data only")
    if has_vix:
        print(" India VIX included → ALL models are scored on the same dates (only where VIX exists)")
    print("═" * 100)

    all_res = {}
    for H in horizons:
        r = test_horizon(d, H)
        all_res[H] = r
        print(f"\n── Horizon {H} {unit}{'s' if H > 1 else ''} │ {r['n']:,} forecasts │ {r['years'].min()}–{r['years'].max()}")

        print("\n  Coverage: share of actual moves inside the range (should equal the row label)")
        print(_block(r["coverage"], lambda c, v: f"{v:.1%}"))

        print("\n  Tails: share of moves beyond kσ")
        print(_block(r["tails"], lambda c, v: f"{v:.2%}"))

        print("\n  Touch ±kσ at any point (real highs/lows, avg of up and down)")
        print(_block(r["touch"], lambda c, v: f"{v:.1%}"))

        print("\n  Forecast score — CRPS (lower = better), skill and t vs CONST on non-overlapping forecasts")
        print(_block(r["scores"], lambda c, v: f"{v:.2f}" if c == "crps_bps" else
                     (f"{v:+.1%}" if c == "skill_vs_const" else f"{v:+.2f}")))
        bs, sk, tt = r["combo_vs_best"]
        print(f"\n  COMBO vs best single model ({bs}): {sk:+.1%}, t = {tt:+.2f} → "
              f"{'COMBO better' if sk > 0 else 'single model better'}"
              f"{' (significant)' if abs(tt) >= 2 else ' (not significant)'}")
        for cname, W in r["weights"].items():
            if not W:
                continue
            yrs = sorted(W)
            show = sorted(set(yrs[::max(1, len(yrs) // 5)] + [yrs[-1]]))
            wt = pd.DataFrame({y_: W[y_] for y_ in show}).T
            wt.index.name = "fitted for"
            print(f"\n  {cname} — share of each ingredient (sums to 1; scale = overall multiplier on σ), "
                  f"learned from data before each year:")
            print(_block(wt, lambda c_, v: f"{v:.3f}"))
        if "head_to_head" in r:
            be, bv, sk, tt = r["head_to_head"]
            who = "VIX better" if sk > 0 else "EWMA better"
            print(f"\n  HEAD TO HEAD: best VIX ({bv}) vs best EWMA ({be}): {sk:+.1%}, t = {tt:+.2f} → "
                  f"{who}{' (significant)' if abs(tt) >= 2 else ' (not significant)'}")

        print("\n  By year")
        print(_block(r["by_year"], lambda c, v: f"{v:,.0f}" if c == "n" else
                     (f"{v:.1f}%" if c.startswith(("pred", "real")) else f"{v:.0%}")))

        print("\n  Biggest surprises (forecast date = the day BEFORE the move for H=1)")
        ev = r["events"].copy()
        ev["forecast_date"] = ev["forecast_date"].dt.strftime("%Y-%m-%d" if daily else "%Y-%m-%d %H:%M")
        for col in ev.columns[1:]:
            ev[col] = ev[col].map(("{:+.1f}σ" if col.startswith("z") else "{:+.2f}%" if col == "move_pct" else "±{:.2f}%").format)
        print(ev.to_string(index=False).replace("\n", "\n    ").join(["    ", ""]))

        for k in ("coverage", "tails", "scores", "by_year", "events"):
            r[k].to_csv(os.path.join(outdir, f"{k}_h{H}{tag}.csv"))

    _plots(d, all_res, outdir, tag)
    _verdict(all_res)
    print(f"\n  Tables and charts saved in ./{outdir}/")


def _verdict(all_res):
    print("\n  VERDICT  (coverage error = average distance from the 50/80/90/95/99% targets)")
    for H, r in all_res.items():
        cv = r["coverage"]
        err = {m: (cv[m] - cv.index).abs().mean() for m in cv.columns}
        best_cov = min(err, key=err.get)
        sc = r["scores"]
        best_score = sc["crps_bps"].idxmin()
        errs = " │ ".join(f"{m} {e:.1%}" for m, e in err.items())
        print(f"   H={H:<3} coverage error: {errs}")
        print(f"         best coverage: {best_cov}   best score: {best_score} "
              f"(skill vs CONST {sc.loc[best_score, 'skill_vs_const']:+.1%})")


def _plots(d, all_res, outdir, tag):
    H0 = min(all_res)
    r = all_res[H0]
    has_vix = "VIX-normal" in r["models"]
    fig, ax = plt.subplots(4, 1, figsize=(13, 16))
    t = d.index[r["idx"]]
    sE = r["sH_EWMA-normal"]
    ax[0].plot(t, r["y"] * 100, lw=0.4, color="k", label="actual move")
    ax[0].plot(t, 2 * sE * 100, color="tab:orange", lw=0.8, label="±2σ EWMA")
    ax[0].plot(t, -2 * sE * 100, color="tab:orange", lw=0.8)
    if has_vix:
        sV = r["sH_VIX-normal"]
        ax[0].plot(t, 2 * sV * 100, color="tab:purple", lw=0.8, alpha=0.8, label="±2σ VIX")
        ax[0].plot(t, -2 * sV * 100, color="tab:purple", lw=0.8, alpha=0.8)
    br = np.abs(r["y"] / sE) > 2
    ax[0].scatter(t[br], r["y"][br] * 100, s=6, color="red", zorder=3, label="outside EWMA ±2σ")
    ax[0].set_title(f"Next-{H0}-bar moves vs ±2σ ranges")
    ax[0].set_ylabel("%")
    ax[0].legend(fontsize=8, loc="lower left")

    by = r["by_year"]
    for col, style in (("cov95_EWMA-MC", "-o"), ("cov95_VIX-MC", "-s"), ("cov95_CONST", ":."), ("cov95_COMBO", "-D")):
        if col in by:
            ax[1].plot(by.index, by[col] * 100, style, label=col.replace("cov95_", ""))
    ax[1].axhline(95, color="gray", ls="--")
    ax[1].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[1].set_title(f"95% range coverage by year, H={H0} (should sit near the dashed line)")
    ax[1].set_ylabel("% of moves inside")
    ax[1].legend(fontsize=8)

    if has_vix:
        ax[2].plot(by.index, by["real_vol"], "k-o", label="actual vol")
        ax[2].plot(by.index, by["pred_EWMA"], "-o", color="tab:orange", label="EWMA forecast")
        ax[2].plot(by.index, by["pred_VIX"], "-s", color="tab:purple", label="India VIX")
        if "pred_COMBO" in by:
            ax[2].plot(by.index, by["pred_COMBO"], "-D", color="tab:green", label="COMBO forecast")
        ax[2].set_title("Average forecast vs actual volatility by year (annualized %)")
        ax[2].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax[2].legend(fontsize=8)
    else:
        z = r["y"] / sE
        bins = np.linspace(-8, 8, 161)
        ax[2].hist(z, bins=bins, density=True, alpha=0.5, color="gray", label="actual (in EWMA σ units)")
        xx = np.linspace(-8, 8, 400)
        ax[2].plot(xx, norm.pdf(xx), color="tab:blue", label="bell curve")
        ax[2].set_yscale("log")
        ax[2].set_ylim(1e-5, 1)
        ax[2].set_title(f"Standardized moves, H={H0}: log scale shows the fat tails")
        ax[2].legend(fontsize=8)
    # skill of every model vs CONST, per horizon
    Hs = list(all_res)
    names = list(all_res[Hs[0]]["scores"].index)
    wbar = 0.8 / len(Hs)
    for k, H in enumerate(Hs):
        sk = all_res[H]["scores"]["skill_vs_const"].reindex(names) * 100
        ax[3].bar(np.arange(len(names)) + k * wbar, sk.values, width=wbar, label=f"H={H}")
    ax[3].set_xticks(np.arange(len(names)) + wbar * (len(Hs) - 1) / 2, names, rotation=15, fontsize=8)
    ax[3].axhline(0, color="k", lw=0.8)
    ax[3].set_ylabel("% better than CONST")
    ax[3].set_title("Forecast skill (CRPS) vs constant volatility — higher is better")
    ax[3].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"range_backtest{tag}.png"), dpi=110)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="20-year walk-forward test of EWMA + Monte Carlo ranges")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("demo")
    r = sp.add_parser("run")
    r.add_argument("--csv", required=True)
    r.add_argument("--vix-csv", default=None)
    r.add_argument("--horizons", default=None, help="default: 1,5,21 (daily) or 1,12,36 (intraday)")
    r.add_argument("--lam", type=float, default=0.94)
    r.add_argument("--out", default="results_range")
    a = ap.parse_args()

    if a.cmd == "demo":
        df, vix = synthetic_daily()
        print("SYNTHETIC daily data (GARCH + fat tails + crash jumps) with a demo 'VIX' that knows the true vol")
        print("plus a 15% premium and noise → VIX-MC should win. Checks the machinery, not NIFTY.")
        report(prepare(df, True, None, vix=vix), (1, 5, 21), tag="_demo")
        return

    df, daily, bar_min = load_any(a.csv)
    print(f"Loaded {len(df):,} {'daily' if daily else f'{bar_min}-min'} bars: {df.index[0]} → {df.index[-1]}")
    horizons = tuple(int(h) for h in a.horizons.split(",")) if a.horizons else ((1, 5, 21) if daily else (1, 12, 36))
    vix = load_vix(a.vix_csv) if a.vix_csv else None
    if vix is not None:
        print(f"India VIX: {len(vix):,} values, {vix.index[0].date()} → {vix.index[-1].date()}")
    report(prepare(df, daily, bar_min, lam=a.lam, vix=vix), horizons, outdir=a.out)


if __name__ == "__main__":
    main()
