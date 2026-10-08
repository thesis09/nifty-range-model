#!/usr/bin/env python3
"""
benchmark.py — formal out-of-sample benchmark of the COMBO range model (NIFTY, daily)

The question it answers: "On which standard metrics does COMBO beat which recognised benchmark,
and is the difference statistically real?"  Everything is walk-forward: every model is refitted
each year using only data from before that year.

Models
  Historical vol (1y)       what most retail tools use                       (baseline)
  RiskMetrics EWMA 0.94     J.P. Morgan's industry standard since 1996        (industry)
  GARCH(1,1)                the standard academic volatility model            (academic)
  India VIX (raw)           the option market's own forecast                  (market)
  COMBO no-VIX (ours)       EWMA + long-run + Monte Carlo — works on any instrument
  COMBO (ours)              EWMA + India VIX + long-run + Monte Carlo

Metrics (the ones risk managers and forecasting papers use)
  CRPS              whole-distribution accuracy (lower = better)
  QLIKE             volatility-forecast accuracy, robust to noisy proxies — Patton (2011) (lower = better)
  Interval score    90% range: rewards narrow ranges, punishes misses — Gneiting & Raftery (2007)
  Coverage          does the 80/95% range hold 80/95% of moves? + average 95% width (sharpness)
  VaR backtests     95%/99% downside: breach rate, Kupiec (right number of breaches),
                    Christoffersen (breaches not clustered), Basel traffic light (99%, 250-day windows)
  Significance      Diebold–Mariano t-stat on non-overlapping forecasts, COMBO vs each benchmark

Usage
  python benchmark.py run --csv nifty_daily.csv --vix-csv vix_daily.csv
  python benchmark.py demo
Needs range_backtest.py in the same folder. Writes benchmark_report.md with the claims you can make.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import chi2, norm

from range_backtest import (crps_normal, crps_sorted, fit_combos, load_any, load_vix, prepare,
                            synthetic_daily)

OURS = ("COMBO (ours)", "COMBO no-VIX (ours)")


# ──────────────────────────────────────────────────────────────────────────────
# GARCH(1,1), refitted yearly on past data
# ──────────────────────────────────────────────────────────────────────────────
def garch_filter(r, w, a, b, v0):
    """s2[t] = variance forecast for bar t made at t-1 (so s2[t+1] is known at close of t)."""
    x = np.r_[0.0, r[:-1] ** 2]
    n = len(r)
    k = np.arange(n)
    return lfilter([a], [1, -b], x) + w * (1 - b ** k) / (1 - b) + b ** k * v0


def garch_fit(r):
    r = r * 100                                             # percent units for numerical comfort
    v0 = r.var()

    def nll(p):
        w, a, b = p
        if a + b >= 0.999:
            return 1e10
        s2 = np.maximum(garch_filter(r, w, a, b, v0), 1e-8)
        return 0.5 * np.mean(np.log(s2) + r ** 2 / s2)

    best = minimize(nll, [0.05, 0.08, 0.90], method="L-BFGS-B",
                    bounds=[(1e-5, 5), (1e-4, 0.4), (0.3, 0.998)]).x
    return best[0] / 1e4, best[1], best[2]                  # back to decimal units


def garch_forecasts(d, H, years):
    """Per-bar σ for the next H bars at every origin, parameters fitted on data before each year."""
    r = d["r_use"].to_numpy()
    fin = np.isfinite(r)
    rr = np.where(fin, r, 0.0)
    dates = d["date"].to_numpy()
    out = np.full(len(d), np.nan)
    params = {}
    for Y in years:
        past = fin & (dates < np.datetime64(f"{Y}-01-01"))
        if past.sum() < 750:
            continue
        w, a, b = garch_fit(r[past])
        params[Y] = (w, a, b)
        s2 = garch_filter(rr, w, a, b, np.var(r[past]))
        s2_next = np.r_[s2[1:], np.nan]                     # forecast for t+1 made at t
        lr = w / (1 - a - b)
        k = np.arange(H)
        avg = lr + np.mean((a + b) ** k) * (s2_next - lr)   # average per-bar variance over H bars
        m = pd.DatetimeIndex(dates).year == Y
        out[m] = np.sqrt(np.maximum(avg[m], 1e-12))
    return out, params


# ──────────────────────────────────────────────────────────────────────────────
# Forecast objects: bell curve or Monte Carlo sample
# ──────────────────────────────────────────────────────────────────────────────
class Forecast:
    """Distribution of the H-bar log return at each origin."""

    def __init__(self, sH=None, sb=None, samples=None, years=None):
        self.sH, self.sb, self.samples, self.years = sH, sb, samples, years

    def _groups(self):
        for yr in np.unique(self.years):
            yield yr, self.years == yr, self.samples[yr]

    def quantile(self, p):
        if self.samples is None:
            return norm.ppf(p) * self.sH
        q = np.empty(len(self.sb))
        for yr, m, s in self._groups():
            q[m] = self.sb[m] * np.quantile(s, p)
        return q

    def variance(self):
        if self.samples is None:
            return self.sH ** 2
        v = np.empty(len(self.sb))
        for yr, m, s in self._groups():
            v[m] = self.sb[m] ** 2 * np.mean(s ** 2)
        return v

    def crps(self, y):
        if self.samples is None:
            return crps_normal(y, self.sH)
        c = np.empty(len(y))
        for yr, m, s in self._groups():
            c[m] = self.sb[m] * crps_sorted(s, y[m] / self.sb[m])
        return c


# ──────────────────────────────────────────────────────────────────────────────
# VaR backtests
# ──────────────────────────────────────────────────────────────────────────────
def kupiec(hits, p):
    n, x = len(hits), int(hits.sum())
    if x in (0, n):
        ll = n * np.log(1 - p) if x == 0 else n * np.log(p)
        return float(1 - chi2.cdf(-2 * ll, 1))
    pi = x / n
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - pi) - x * np.log(pi))
    return float(1 - chi2.cdf(lr, 1))


def christoffersen(hits):
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    if n01 + n11 == 0 or n00 + n10 == 0:
        return np.nan
    p01, p11 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    p = (n01 + n11) / (n00 + n01 + n10 + n11)

    def ll(q, k0, k1):
        return (k0 * np.log(1 - q) if k0 else 0) + (k1 * np.log(q) if k1 and q > 0 else 0)

    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return float(1 - chi2.cdf(lr, 1))


def traffic_light(hits99, window=250):
    if len(hits99) < window:
        return (np.nan,) * 3
    c = np.convolve(hits99.astype(int), np.ones(window, int), "valid")
    return float((c <= 4).mean()), float(((c >= 5) & (c <= 9)).mean()), float((c >= 10).mean())


def dm_t(loss_a, loss_b, step):
    """Positive t = model A has LOWER loss than B. Non-overlapping sub-sample for multi-bar horizons."""
    diff = (loss_b - loss_a)[::step]
    return float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))))


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark one horizon
# ──────────────────────────────────────────────────────────────────────────────
def run_horizon(d, H, n_paths=20000, seed=0):
    rng = np.random.default_rng(seed)
    c, dates, n = d["close"].to_numpy(), d["date"].to_numpy(), len(d)
    has_vix = "sig_v" in d
    ok = np.zeros(n, bool)
    ok[:n - H] = True
    cols = ["sig", "sig_c"] + (["sig_v"] if has_vix else [])
    for col in cols:
        ok &= np.isfinite(d[col].to_numpy())
    idx = np.where(ok)[0]
    idx = idx[idx < n - H]
    years = pd.DatetimeIndex(dates[idx]).year.to_numpy()

    g_all, g_params = garch_forecasts(d, H, np.unique(years))
    combos = fit_combos(d, H, idx, years)
    keep = np.isfinite(g_all[idx])
    for cm in combos.values():
        keep &= np.isfinite(cm["sb"])
        keep &= np.isin(years, list(cm["fhs"]))
    idx, years = idx[keep], years[keep]
    y = np.log(c[idx + H] / c[idx])

    F = {"Historical vol (1y)": Forecast(sH=d["sig_c"].to_numpy()[idx] * np.sqrt(H)),
         "RiskMetrics EWMA": Forecast(sH=d["sig"].to_numpy()[idx] * np.sqrt(H)),
         "GARCH(1,1)": Forecast(sH=g_all[idx] * np.sqrt(H))}
    if has_vix:
        F["India VIX (raw)"] = Forecast(sH=d["sig_v"].to_numpy()[idx] * np.sqrt(H))
    for name, cm in combos.items():
        label = "COMBO no-VIX (ours)" if "no-VIX" in name or not has_vix else "COMBO (ours)"
        F[label] = Forecast(sb=cm["sb"][keep], samples=cm["fhs"], years=years)

    rows, losses = {}, {}
    for name, f in F.items():
        cr = f.crps(y)
        v = f.variance()
        ql = np.log(v) + y ** 2 / v
        lo90, hi90 = f.quantile(0.05), f.quantile(0.95)
        isc = (hi90 - lo90) + (2 / 0.10) * (np.maximum(lo90 - y, 0) + np.maximum(y - hi90, 0))
        lo80, hi80 = f.quantile(0.10), f.quantile(0.90)
        lo95, hi95 = f.quantile(0.025), f.quantile(0.975)
        var95, var99 = f.quantile(0.05), f.quantile(0.01)
        sub = slice(None, None, H)                                     # non-overlapping for VaR tests
        h95, h99 = (y < var95)[sub], (y < var99)[sub]
        g, yel, red = traffic_light(h99) if H == 1 else (np.nan,) * 3
        rows[name] = {"CRPS (bps)": cr.mean() * 1e4, "QLIKE": ql.mean(), "Interval score 90% (bps)": isc.mean() * 1e4,
                      "Coverage 80%": np.mean((y >= lo80) & (y <= hi80)),
                      "Coverage 95%": np.mean((y >= lo95) & (y <= hi95)),
                      "Avg 95% width (%)": np.mean(hi95 - lo95) * 100,
                      "VaR95 breaches": h95.mean(), "Kupiec p (95)": kupiec(h95, 0.05),
                      "VaR99 breaches": h99.mean(), "Kupiec p (99)": kupiec(h99, 0.01),
                      "Christoffersen p (99)": christoffersen(h99) if H == 1 else np.nan,
                      "Basel green": g, "Basel red": red}
        losses[name] = {"CRPS": cr, "QLIKE": ql, "Interval": isc}

    table = pd.DataFrame(rows).T
    dm = {}
    for ours in [o for o in OURS if o in F]:
        for bench in [b for b in F if b not in OURS]:
            dm[(ours, bench)] = {m: dm_t(losses[ours][m], losses[bench][m], H) for m in ("CRPS", "QLIKE", "Interval")}
            dm[(ours, bench)].update({f"{m} gain": 1 - losses[ours][m].mean() / losses[bench][m].mean()
                                      if m != "QLIKE" else losses[bench][m].mean() - losses[ours][m].mean()
                                      for m in ("CRPS", "QLIKE", "Interval")})
    return dict(H=H, n=len(idx), n_indep=len(idx[::H]), start=int(years.min()), end=int(years.max()),
                table=table, dm=dm, garch=g_params)


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────
def _md(df):
    try:
        return df.to_markdown()                     # needs `pip install tabulate`
    except ImportError:
        return "```\n" + df.to_string() + "\n```"


def fmt_table(t: pd.DataFrame) -> pd.DataFrame:
    f = {"CRPS (bps)": "{:.2f}", "QLIKE": "{:.4f}", "Interval score 90% (bps)": "{:.1f}",
         "Coverage 80%": "{:.1%}", "Coverage 95%": "{:.1%}", "Avg 95% width (%)": "{:.2f}",
         "VaR95 breaches": "{:.2%}", "Kupiec p (95)": "{:.3f}", "VaR99 breaches": "{:.2%}",
         "Kupiec p (99)": "{:.3f}", "Christoffersen p (99)": "{:.3f}", "Basel green": "{:.0%}", "Basel red": "{:.0%}"}
    return t.apply(lambda col: col.map(lambda v: "—" if pd.isna(v) else f[col.name].format(v)))


def claims(res_by_h, alpha_t=2.0):
    out = []
    for H, r in res_by_h.items():
        for (ours, bench), s in r["dm"].items():
            wins = [m for m in ("CRPS", "QLIKE", "Interval") if s[m] >= alpha_t]
            losses = [m for m in ("CRPS", "QLIKE", "Interval") if s[m] <= -alpha_t]
            if wins:
                gains = ", ".join(f"{m} {'−' if m != 'QLIKE' else ''}{abs(s[m + ' gain']):.1%}"
                                  f"{'' if m != 'QLIKE' else ' (Δ)'} (t={s[m]:.1f})" if m != "QLIKE"
                                  else f"QLIKE Δ {s['QLIKE gain']:.4f} (t={s[m]:.1f})" for m in wins)
                out.append(("WIN", H, ours, bench, gains))
            if losses:
                out.append(("LOSS", H, ours, bench, ", ".join(f"{m} (t={s[m]:.1f})" for m in losses)))
            if not wins and not losses:
                out.append(("TIE", H, ours, bench, "no significant difference"))
    return out


def report(d, horizons, outdir="results_benchmark", tag=""):
    os.makedirs(outdir, exist_ok=True)
    res = {}
    md = [f"# COMBO range model — out-of-sample benchmark {tag}\n",
          f"Data: {d.index[0].date()} → {d.index[-1].date()}, {len(d):,} daily bars. "
          "Walk-forward: every model refitted each year on earlier data only.\n"]
    print("\n" + "═" * 110)
    print(f" BENCHMARK REPORT {tag} — every model refitted yearly on past data only; claims need |t| ≥ 2")
    print("═" * 110)
    for H in horizons:
        r = run_horizon(d, H)
        res[H] = r
        t = fmt_table(r["table"])
        print(f"\n── Horizon {H} day{'s' if H > 1 else ''} │ {r['n']:,} forecasts ({r['n_indep']:,} non-overlapping) │ "
              f"{r['start']}–{r['end']}")
        show1 = t[["CRPS (bps)", "QLIKE", "Interval score 90% (bps)", "Coverage 80%", "Coverage 95%", "Avg 95% width (%)"]]
        show2 = t[["VaR95 breaches", "Kupiec p (95)", "VaR99 breaches", "Kupiec p (99)", "Christoffersen p (99)",
                   "Basel green", "Basel red"]]
        print("\n  Accuracy (lower is better for the first three)")
        print(show1.to_string().replace("\n", "\n    ").join(["    ", ""]))
        print("\n  Downside risk backtests (target breaches 5% / 1%; p < 0.05 = model FAILS the test)")
        print(show2.to_string().replace("\n", "\n    ").join(["    ", ""]))
        print("\n  Diebold–Mariano t-stats (positive = ours better)")
        dm = pd.DataFrame({f"{o} vs {b}": {m: v[m] for m in ("CRPS", "QLIKE", "Interval")}
                           for (o, b), v in r["dm"].items()}).T
        print(dm.map(lambda v: f"{v:+.2f}").to_string().replace("\n", "\n    ").join(["    ", ""]))
        md += [f"\n## Horizon {H} day{'s' if H > 1 else ''} — {r['n']:,} forecasts, {r['start']}–{r['end']}\n",
               _md(show1), "\n", _md(show2), "\n", _md(dm.map(lambda v: f"{v:+.2f}")), "\n"]
        r["table"].to_csv(os.path.join(outdir, f"table_h{H}{tag}.csv"))

    cl = claims(res)
    print("\n" + "═" * 110)
    print(" WHAT YOU CAN HONESTLY CLAIM (Diebold–Mariano |t| ≥ 2 on non-overlapping out-of-sample forecasts)")
    print("═" * 110)
    md.append("\n## Claims supported by the data (|t| ≥ 2)\n")
    for kind, H, ours, bench, txt in cl:
        line = f"[{kind}] H={H:<2} {ours} vs {bench}: {txt}"
        print("  " + line)
        md.append(f"- **{kind}** — H={H}: {ours} vs {bench}: {txt}")
    with open(os.path.join(outdir, f"benchmark_report{tag}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\n  Full report: ./{outdir}/benchmark_report{tag}.md")


def main():
    ap = argparse.ArgumentParser(description="Formal benchmark of the COMBO range model")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("demo")
    r = sp.add_parser("run")
    r.add_argument("--csv", required=True)
    r.add_argument("--vix-csv", default=None)
    r.add_argument("--horizons", default="1,5,21")
    r.add_argument("--out", default="results_benchmark")
    a = ap.parse_args()
    if a.cmd == "demo":
        df, vix = synthetic_daily()
        report(prepare(df, True, None, vix=vix), (1, 5, 21), tag="_demo")
        return
    df, daily, bar_min = load_any(a.csv)
    if not daily:
        raise SystemExit("benchmark.py is for DAILY data (GARCH and VIX are daily). Use range_backtest.py for intraday.")
    vix = load_vix(a.vix_csv) if a.vix_csv else None
    report(prepare(df, True, None, vix=vix), tuple(int(h) for h in a.horizons.split(",")), outdir=a.out)


if __name__ == "__main__":
    main()
