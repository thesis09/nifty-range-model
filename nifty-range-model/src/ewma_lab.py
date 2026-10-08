#!/usr/bin/env python3
"""
EWMA Lab — two honest tests on intraday data (built for NIFTY 5-min, NSE session)

  1) MARKOV DIRECTION TEST
     Does the current "state" (last candle, last two candles, or vol-regime|candle|EWMA-side)
     change the odds of the NEXT move being up? Train on the first part of history, verify on the
     rest, correct for testing many states, check against a shuffle baseline, and trade it with costs.

  2) HIT-PROBABILITY TEST (formula + Monte Carlo)
     Using EWMA σ, predict the chance NIFTY touches / finishes beyond a level within H bars, and the
     chance a target is hit before a stop. Then check those predictions against what actually happened
     (using real highs/lows), vs a constant-vol baseline and optionally India VIX.

Usage
  python ewma_lab.py demo                                   # synthetic sanity check, no data needed
  python ewma_lab.py fetch --start 2023-01-01 --end 2026-09-29 --out nifty_5m.csv
  python ewma_lab.py tokens --search "INDIA VIX" --exch NSE  # look up an instrument token
  python ewma_lab.py markov  --csv nifty_5m.csv
  python ewma_lab.py hitprob --csv nifty_5m.csv --H 12 --target 60 --stop 30 [--vix-csv vix_5m.csv]
  python ewma_lab.py all     --csv nifty_5m.csv

CSV format: a datetime/timestamp column + open, high, low, close (volume optional).
Angel One getCandleData output saved by `fetch` works directly.
fetch uses ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET (same as run_analysis_bates.py).

Requires: numpy, pandas, scipy, matplotlib   (fetch also needs: smartapi-python pyotp)
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import binomtest, chi2_contingency, norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    lam: float = 0.94          # EWMA decay (same as the TradingView indicator)
    seed: int = 20             # bars used to seed σ²
    pct_len: int = 750         # regime lookback (~10 days of 5-min bars)
    lo_q: float = 0.20         # below this quantile → Low regime (teal)
    hi_q: float = 0.80         # above this quantile → High regime (red)
    pe_span: int = 20          # price EWMA span (blue line)
    tdays: int = 252
    sess_min: int = 375
    bar_min: int = 5

    @property
    def bars_per_day(self) -> float:
        return self.sess_min / self.bar_min


# ──────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next((c for c in ("datetime", "timestamp", "date", "time") if c in df.columns), None)
    if tcol is None:
        raise SystemExit("CSV needs a datetime/timestamp column")
    ts = pd.to_datetime(df[tcol], errors="coerce")
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df.index = ts
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.between_time("09:15", "15:29")
    return df


def load_vix(path: str) -> pd.Series:
    v = load_csv(path)["close"]
    v.name = "vix"
    return v


# Credentials come from environment variables — same as run_analysis_bates.py:
#   ANGEL_API_KEY, ANGEL_CLIENT_ID, ANGEL_PASSWORD, ANGEL_TOTP_SECRET
# Set them in your shell before running, e.g. (PowerShell):
#   $env:ANGEL_API_KEY="xxxx"
CREDS = dict(
    api_key=os.environ.get("ANGEL_API_KEY"),
    client_id=os.environ.get("ANGEL_CLIENT_ID"),
    password=os.environ.get("ANGEL_PASSWORD"),
    totp_secret=os.environ.get("ANGEL_TOTP_SECRET"),
)


def _load_dotenv_quietly():
    """Env vars already set in the shell win (same as run_analysis_bates.py); .env only fills gaps."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    for p in (".env", here):
        if os.path.exists(p):
            load_dotenv(p, override=False)
            break


def _get_candles(api, params, tries=7):
    """getCandleData with back-off: Angel returns 'exceeding access rate' as non-JSON text."""
    wait = 2.0
    for k in range(tries):
        try:
            resp = api.getCandleData(params) or {}
            msg = str(resp.get("message", ""))
            if resp.get("status") is False and "rate" in msg.lower():
                raise RuntimeError(msg)
            if resp.get("status") is False and msg:
                print(f"    server said: {msg}")
            return resp.get("data") or []
        except Exception as e:
            text = str(e).lower()
            if "rate" in text or "couldn't parse" in text or "timed out" in text:
                print(f"    rate-limited → waiting {wait:.0f}s (retry {k + 1}/{tries})")
                time.sleep(wait)
                wait = min(wait * 2, 60)
            else:
                raise
    return None


def fetch_angel(start: str, end: str, out: str, token: str = "99926000",
                exchange: str = "NSE", interval: str = "FIVE_MINUTE", chunk_days: int = 60,
                pause: float = 1.2):
    """Download candles from Angel One SmartAPI using the ANGEL_* env vars (see CREDS above).
    Saves after every chunk and resumes from the last saved bar if you run it again.
    NIFTY 50 index token = 99926000. For India VIX look up its token in the instrument master."""
    try:
        from SmartApi import SmartConnect
        import pyotp
    except ImportError:
        raise SystemExit("pip install smartapi-python pyotp")

    _load_dotenv_quietly()
    creds = dict(api_key=os.environ.get("ANGEL_API_KEY"), client_id=os.environ.get("ANGEL_CLIENT_ID"),
                 password=os.environ.get("ANGEL_PASSWORD"), totp_secret=os.environ.get("ANGEL_TOTP_SECRET"))
    missing = [k for k in ("api_key", "client_id", "password", "totp_secret") if not creds[k]]
    if missing:
        raise SystemExit("Missing env vars for: " + ", ".join("ANGEL_" + k.upper() for k in missing)
                         + "\nSet them before running (see comments above CREDS).")

    secret = creds["totp_secret"].strip().strip('"').strip("'").replace(" ", "").upper()
    try:
        otp = pyotp.TOTP(secret).now()
    except Exception:
        bad = sorted({ch for ch in secret if ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567="})
        raise SystemExit(f"ANGEL_TOTP_SECRET is not a valid TOTP secret (length {len(secret)}, "
                         f"invalid characters: {' '.join(bad) or 'none'}).")

    # Resume: continue after the last bar already saved
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame(columns=cols)
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    if len(old):
        last = pd.to_datetime(old["datetime"].iloc[-1])
        if last.tzinfo is not None:
            last = last.tz_convert("Asia/Kolkata").tz_localize(None)
        s = max(s, last.normalize() + pd.Timedelta(days=1))
        print(f"Resuming: {len(old):,} bars already in {out}, continuing from {s.date()}")
    if s > e:
        print("Nothing to download — file is already up to date.")
        return

    api = SmartConnect(api_key=creds["api_key"])
    sess = api.generateSession(creds["client_id"], creds["password"], otp)
    if not sess or not sess.get("status"):
        raise SystemExit(f"Login failed: {sess}")
    time.sleep(1.5)  # Angel often rate-limits a request fired immediately after login

    rows = []
    while s <= e:
        stop = min(s + pd.Timedelta(days=chunk_days), e)
        params = {"exchange": exchange, "symboltoken": token, "interval": interval,
                  "fromdate": s.strftime("%Y-%m-%d 09:15"), "todate": stop.strftime("%Y-%m-%d 15:30")}
        data = _get_candles(api, params)
        if data is None:
            print("Still rate-limited. Progress is saved — wait a few minutes and run the same command again.")
            break
        print(f"  {params['fromdate']} → {params['todate']}: {len(data)} bars")
        rows.extend(data)
        new_df = pd.concat([old, pd.DataFrame(rows, columns=cols)], ignore_index=True)
        new_df.drop_duplicates("datetime").to_csv(out, index=False)   # save progress every chunk
        s = stop + pd.Timedelta(days=1)
        time.sleep(pause)
    final = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    print(f"Saved {len(final):,} bars → {out}")


def list_tokens(search="NIFTY", exch=None):
    """Look up Angel One instrument tokens (e.g. INDIA VIX, NIFTYBEES, NIFTY futures)."""
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    print("Downloading Angel One instrument master (~40 MB) …")
    m = pd.read_json(url, dtype={"token": str})
    q = search.upper()
    f = m[(m["symbol"].str.upper().str.startswith(q)) | (m["name"].str.upper() == q)]
    if exch:
        f = f[f["exch_seg"] == exch.upper()]
    cols = [c for c in ("exch_seg", "token", "symbol", "name", "instrumenttype", "expiry", "lotsize") if c in f.columns]
    print(f[cols].head(40).to_string(index=False) if len(f) else "No match.")


def make_synthetic(days=500, bars=75, seed=7, plant=False) -> pd.DataFrame:
    """NIFTY-like 5-min bars: GARCH vol clustering, intraday U-shape, overnight gaps.
    plant=True hides a small momentum edge (only in calm regimes) so we can check the test finds it."""
    rng = np.random.default_rng(seed)
    lr_var = (0.13 / np.sqrt(252 * bars)) ** 2
    a, b = 0.06, 0.925
    w = lr_var * (1 - a - b)
    dates = pd.bdate_range("2024-01-01", periods=days)
    n = days * bars
    O, H, L, C = (np.empty(n) for _ in range(4))
    ts = []
    price, h, prev_r, i = 22000.0, lr_var, 0.0, 0
    for di, day in enumerate(dates):
        if di > 0:
            price *= np.exp(rng.normal(0, 0.004))           # overnight gap
        for k in range(bars):
            u = 1 + 0.9 * np.exp(-k / 5) + 0.35 * np.exp(-(bars - 1 - k) / 5)  # U-shape
            sig = np.sqrt(h * u)
            mu = 0.15 * np.sign(prev_r) * sig if (plant and k > 0 and h < lr_var) else 0.0
            path = price * np.exp(np.cumsum(rng.normal(mu / 5, sig / np.sqrt(5), 5)))
            O[i], C[i] = price, path[-1]
            H[i], L[i] = max(price, path.max()), min(price, path.min())
            r = np.log(C[i] / O[i])
            e = r / np.sqrt(u)
            h = w + a * e * e + b * h
            prev_r, price = r, C[i]
            ts.append(day + pd.Timedelta(hours=9, minutes=15 + 5 * k))
            i += 1
    return pd.DataFrame({"open": O, "high": H, "low": L, "close": C}, index=pd.DatetimeIndex(ts))


# ──────────────────────────────────────────────────────────────────────────────
# Features (mirror the TradingView indicator)
# ──────────────────────────────────────────────────────────────────────────────
def add_features(df: pd.DataFrame, cfg: Cfg) -> pd.DataFrame:
    d = df.copy()
    d["pos"] = np.arange(len(d))
    d["date"] = d.index.normalize()
    d["first"] = d["date"].ne(d["date"].shift())
    d["r"] = np.log(d["close"]).diff()
    d["r_use"] = d["r"].where(~d["first"])                  # drop overnight-gap return

    var = np.full(len(d), np.nan)
    v, cnt, s = np.nan, 0, 0.0
    for i, x in enumerate(d["r_use"].to_numpy()):
        if not np.isnan(x):
            cnt += 1
            if cnt <= cfg.seed:
                s += x * x
                if cnt == cfg.seed:
                    v = s / cfg.seed
            else:
                v = cfg.lam * v + (1 - cfg.lam) * x * x
        var[i] = v
    d["sig"] = np.sqrt(var)                                  # per-bar σ forecast for next bar

    q_lo = d["sig"].rolling(cfg.pct_len, min_periods=cfg.pct_len).quantile(cfg.lo_q)
    q_hi = d["sig"].rolling(cfg.pct_len, min_periods=cfg.pct_len).quantile(cfg.hi_q)
    reg = np.where(d["sig"] <= q_lo, "L", np.where(d["sig"] >= q_hi, "H", "N"))
    d["regime"] = pd.Series(reg, index=d.index, dtype=object).where(q_lo.notna())

    d["pe"] = d["close"].ewm(span=cfg.pe_span, adjust=False).mean()
    d["side"] = np.where(d["close"] >= d["pe"], "A", "B")    # Above / Below blue line
    d["dir"] = np.where(d["r"] > 0, "U", "D")

    d["s1"] = d["dir"]
    d["s2"] = d["dir"].shift(1).fillna("D") + d["dir"]
    d["s3"] = d["regime"] + "|" + d["dir"] + "|" + d["side"]
    return d


def fwd_return(d: pd.DataFrame, h: int) -> pd.Series:
    """Log return over the next h bars, only if it stays inside the same session."""
    c, dates = d["close"].to_numpy(), d["date"].to_numpy()
    out = np.full(len(d), np.nan)
    same = np.zeros(len(d), bool)
    out[:-h] = np.log(c[h:] / c[:-h])
    same[:-h] = dates[h:] == dates[:-h]
    out[~same] = np.nan
    return pd.Series(out, index=d.index)


def split_by_date(d: pd.DataFrame, frac: float):
    days = d["date"].unique()
    cut = days[int(len(days) * frac)]
    return d[d["date"] < cut], d[d["date"] >= cut], pd.Timestamp(cut)


# ──────────────────────────────────────────────────────────────────────────────
# 1) Markov direction test
# ──────────────────────────────────────────────────────────────────────────────
FAMILIES = [("s1", "Last candle (U/D)"),
            ("s2", "Last two candles"),
            ("s3", "Regime | candle | side of EWMA")]


def state_stats(tr, te, col):
    base_tr, base_te = tr["up"].mean(), te["up"].mean()
    rows = []
    for s, g in tr.groupby(col):
        t = te[te[col] == s]
        n_tr, n_te = len(g), len(t)
        sd_tr = g["fwd"].std()
        t_tr = g["fwd"].mean() / (sd_tr / np.sqrt(n_tr)) if n_tr > 1 and sd_tr > 0 else 0.0
        sd_te = t["fwd"].std() if n_te > 1 else np.nan
        t_te = t["fwd"].mean() / (sd_te / np.sqrt(n_te)) if n_te > 1 and sd_te > 0 else np.nan
        pv = binomtest(int(t["up"].sum()), n_te, base_te).pvalue if n_te else np.nan
        rows.append(dict(state=s, n_tr=n_tr, p_up_tr=g["up"].mean(), mean_bps_tr=g["fwd"].mean() * 1e4,
                         t_tr=t_tr, n_te=n_te, p_up_te=t["up"].mean() if n_te else np.nan,
                         mean_bps_te=t["fwd"].mean() * 1e4 if n_te else np.nan, t_te=t_te, p_value_te=pv))
    tab = pd.DataFrame(rows).set_index("state")
    tab["edge_tr"] = tab["p_up_tr"] - base_tr
    tab["edge_te"] = tab["p_up_te"] - base_te
    tab["same_sign"] = np.sign(tab["edge_tr"]) == np.sign(tab["edge_te"])
    tab["bonferroni_ok"] = tab["p_value_te"] < 0.05 / len(tab)
    return tab, base_tr, base_te


def snoop_test(tr, col, min_n, n_perm, rng):
    """How big would the BEST state's edge be if the market were pure noise? (shuffle labels)"""
    codes, uniq = pd.factorize(tr[col])
    up = tr["up"].to_numpy().astype(float)
    n = np.bincount(codes, minlength=len(uniq))
    valid = n >= min_n
    if not valid.any():
        return np.nan, np.nan, np.nan

    def best_edge(y):
        p = np.bincount(codes, weights=y, minlength=len(uniq)) / np.maximum(n, 1)
        return np.max(np.abs(p[valid] - y.mean()))

    obs = best_edge(up)
    null = np.array([best_edge(rng.permutation(up)) for _ in range(n_perm)])
    return obs, float((null >= obs).mean()), float(np.percentile(null, 95))


def trade_test(te, col, rules, h, cost_pts):
    """Trade test-period states using rules learned on train. Non-overlapping trades."""
    pnl, next_free = [], -1
    for pos, st, f, c in zip(te["pos"].to_numpy(), te[col].to_numpy(),
                             te["fwd"].to_numpy(), te["close"].to_numpy()):
        if pos < next_free or st not in rules:
            continue
        pnl.append(rules[st] * (np.exp(f) - 1) * c)
        next_free = pos + h
    pnl = np.array(pnl)
    if len(pnl) == 0:
        return dict(trades=0)
    net = pnl - cost_pts
    t = net.mean() / (net.std(ddof=1) / np.sqrt(len(net))) if len(net) > 1 and net.std() > 0 else np.nan
    return dict(trades=len(pnl), hit_rate=float((pnl > 0).mean()), avg_gross_pts=float(pnl.mean()),
                avg_net_pts=float(net.mean()), total_net_pts=float(net.sum()), t_stat_net=float(t))


def markov_test(d, cfg, horizons=(1, 3, 12), train_frac=0.6, min_n=300, cost_pts=1.0,
                n_perm=300, outdir="results", tag="", seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    print("\n" + "═" * 78)
    print(f" MARKOV DIRECTION TEST {tag}")
    print("═" * 78)
    verdicts = []
    for h in horizons:
        x = d.assign(fwd=fwd_return(d, h)).dropna(subset=["fwd", "s3"])
        x = x[x["fwd"] != 0].copy()
        x["up"] = (x["fwd"] > 0).astype(int)
        tr, te, cut = split_by_date(x, train_frac)
        print(f"\n── Horizon: next {h} bar(s) ({h * cfg.bar_min} min) │ train {len(tr):,} │ "
              f"test {len(te):,} (from {cut.date()}) │ base P(up) train {tr['up'].mean():.3f}, "
              f"test {te['up'].mean():.3f}")
        for col, name in FAMILIES:
            tab, _, _ = state_stats(tr, te, col)
            tab.to_csv(os.path.join(outdir, f"markov_{col}_h{h}{tag}.csv"))
            ct = pd.crosstab(te[col], te["up"])
            chi_p = chi2_contingency(ct)[1] if ct.shape[0] > 1 and ct.shape[1] > 1 else np.nan
            obs, p_snoop, null95 = snoop_test(tr, col, min_n, n_perm, rng)

            rules = {s: int(np.sign(r.mean_bps_tr)) for s, r in tab.iterrows()
                     if r.n_tr >= min_n and abs(r.t_tr) >= 2}
            tt = trade_test(te, col, rules, h, cost_pts)

            passed = tab[(tab.n_tr >= min_n) & (tab.n_te >= min_n) & (tab.t_tr.abs() >= 2)
                         & tab.same_sign & tab.bonferroni_ok]
            print(f"\n  [{name}]  states={len(tab)}   chi² independence p (test) = {chi_p:.3g}")
            print(f"   best train edge {obs:+.3f} vs noise-95% {null95:.3f}  → shuffle p = {p_snoop:.3f}")
            show = tab[["n_tr", "p_up_tr", "t_tr", "n_te", "p_up_te", "mean_bps_te", "p_value_te"]]
            print(show.round(3).to_string(max_rows=20).replace("\n", "\n   ").join(["   ", ""]))
            if tt.get("trades", 0):
                print(f"   trade test (rules from train, {cost_pts} pt cost): {tt['trades']} trades, "
                      f"hit {tt['hit_rate']:.1%}, avg net {tt['avg_net_pts']:+.2f} pts, "
                      f"total {tt['total_net_pts']:+.0f} pts, t={tt['t_stat_net']:.2f}")
            else:
                print("   trade test: no state had |t| ≥ 2 on train → nothing to trade")
            ok = len(passed) > 0 and tt.get("trades", 0) and tt.get("avg_net_pts", -1) > 0
            verdicts.append((h, name, list(passed.index), bool(ok)))

            if col == "s3":
                _plot_edges(tab, min_n, os.path.join(outdir, f"markov_s3_h{h}{tag}.png"),
                            f"Regime|candle|side → P(up) edge, next {h} bar(s) {tag}")

    print("\n  VERDICT")
    any_ok = False
    for h, name, states, ok in verdicts:
        mark = "EDGE" if ok else "none"
        any_ok |= ok
        print(f"   h={h:<3} {name:<34} {mark:<5} {', '.join(states) if states else ''}")
    print("   → " + ("Some states survived train→test, multiple-testing and costs. Paper-trade before real money."
                     if any_ok else "No state survived train→test + multiple-testing + costs: no tradable direction edge."))
    return verdicts


def _plot_edges(tab, min_n, path, title):
    t = tab[(tab.n_tr >= min_n) & (tab.n_te > 0)].sort_values("edge_te")
    if t.empty:
        return
    ci = 1.96 * np.sqrt(t.p_up_te * (1 - t.p_up_te) / t.n_te)
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(t) + 1.5))
    y = np.arange(len(t))
    ax.barh(y, t.edge_te, xerr=ci, color=np.where(t.edge_te > 0, "#2a9d8f", "#e76f51"), alpha=0.8,
            label="test edge ±95% CI")
    ax.scatter(t.edge_tr, y, color="k", s=18, zorder=3, label="train edge")
    ax.axvline(0, color="gray", lw=1)
    ax.set_yticks(y, t.index)
    ax.set_xlabel("P(up | state) − base P(up)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# 2) Hit-probability test (formula + Monte Carlo)
# ──────────────────────────────────────────────────────────────────────────────
def mc_target_stop_grid(ratio, H, xs, n_paths=20000, sub=10, seed=1):
    """Monte Carlo: zero-drift log-price paths over H bars (sub-steps approximate intrabar highs/lows).
    Distances in units of σ√H. For stop distance x and target distance ratio·x, return
    P(target first), P(stop first). Common random numbers across xs."""
    rng = np.random.default_rng(seed)
    steps = H * sub
    W = np.cumsum(rng.standard_normal((n_paths, steps)), axis=1) / np.sqrt(steps)
    pt, ps = [], []
    for x in xs:
        ht, hs = W >= ratio * x, W <= -x
        ft = np.where(ht.any(1), ht.argmax(1), steps)
        fs = np.where(hs.any(1), hs.argmax(1), steps)
        pt.append(np.mean(ft < fs))
        ps.append(np.mean(fs < ft))
    return np.array(pt), np.array(ps)


def mc_touch(k, H, n_paths=20000, sub=10, seed=2):
    rng = np.random.default_rng(seed)
    W = np.cumsum(rng.standard_normal((n_paths, H * sub)), axis=1) / np.sqrt(H * sub)
    return float((W.max(1) >= k).mean())


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def hitprob_test(d, cfg, H=12, pts=(20, 40, 60, 100), ks=(0.5, 1.0, 1.5, 2.0),
                 target=60.0, stop=30.0, vix: pd.Series | None = None, outdir="results", tag=""):
    os.makedirs(outdir, exist_ok=True)
    print("\n" + "═" * 78)
    print(f" HIT-PROBABILITY TEST {tag}   window = next {H} bars ({H * cfg.bar_min} min)")
    print("═" * 78)

    N = len(d)
    c, hi, lo = (d[k].to_numpy() for k in ("close", "high", "low"))
    sig, dates = d["sig"].to_numpy(), d["date"].to_numpy()
    sig_c = d["r_use"].rolling(3750, min_periods=750).std().to_numpy()   # constant-ish vol baseline

    valid = np.zeros(N, bool)
    valid[:N - H] = dates[H:] == dates[:N - H]
    valid &= ~np.isnan(sig) & ~np.isnan(sig_c)
    idx = np.where(valid)[0]
    idx = idx[idx < N - H]

    Whi = sliding_window_view(hi[1:], H)[idx]
    Wlo = sliding_window_view(lo[1:], H)[idx]
    S, fin = c[idx], c[idx + H]
    sH, sHc = sig[idx] * np.sqrt(H), sig_c[idx] * np.sqrt(H)
    mx, mn = Whi.max(1), Wlo.min(1)

    sHi = None
    if vix is not None:
        v = pd.merge_asof(pd.DataFrame({"t": d.index[idx]}),
                          vix.rename("vix").rename_axis("t").reset_index().sort_values("t"),
                          on="t", direction="backward")["vix"].to_numpy()
        sHi = v / 100 / np.sqrt(cfg.tdays * cfg.bars_per_day) * np.sqrt(H)
    print(f" origins tested: {len(idx):,}   median ±1σ over window: {np.median(S * sH):.1f} pts")

    # A) σ-multiple levels: is σ the right size?
    print("\n A) Levels at k·σ√H — predicted (normal) vs actual")
    rows = []
    for k in ks:
        up_lv, dn_lv = S * np.exp(k * sH), S * np.exp(-k * sH)
        rows.append(dict(k=k, pred_touch=2 * (1 - norm.cdf(k)), mc_touch=mc_touch(k, H),
                         real_touch_up=np.mean(mx >= up_lv), real_touch_dn=np.mean(mn <= dn_lv),
                         pred_finish=1 - norm.cdf(k), real_finish_up=np.mean(fin >= up_lv),
                         real_finish_dn=np.mean(fin <= dn_lv)))
    ta = pd.DataFrame(rows).set_index("k")
    print(ta.round(3).to_string().replace("\n", "\n   ").join(["   ", ""]))
    ta.to_csv(os.path.join(outdir, f"hit_sigma_levels{tag}.csv"))
    print("   (pred = formula 2·(1−Φ(k)) ; mc = Monte Carlo of the same — they should agree)")

    # B) fixed-point targets: does EWMA σ discriminate better than constant vol / VIX?
    print("\n B) Fixed-point targets — calibration & skill (touch within window, up and down pooled)")
    all_p = {"EWMA": [], "CONST": []} | ({"VIX": []} if sHi is not None else {})
    all_y, rows = [], []
    for D in pts:
        b_up, b_dn = np.log1p(D / S), -np.log1p(-D / S)
        y = np.r_[mx >= S + D, mn <= S - D].astype(float)
        preds = {"EWMA": np.r_[2 * norm.sf(b_up / sH), 2 * norm.sf(b_dn / sH)],
                 "CONST": np.r_[2 * norm.sf(b_up / sHc), 2 * norm.sf(b_dn / sHc)]}
        if sHi is not None:
            preds["VIX"] = np.r_[2 * norm.sf(b_up / sHi), 2 * norm.sf(b_dn / sHi)]
        clim = brier(np.full_like(y, y.mean()), y)
        row = dict(target_pts=D, actual_rate=y.mean())
        for m, p in preds.items():
            p = np.clip(p, 0, 1)
            row[f"{m}_mean_pred"] = p.mean()
            row[f"{m}_skill"] = 1 - brier(p, y) / clim
            all_p[m].append(p)
        all_y.append(y)
        rows.append(row)
    tb = pd.DataFrame(rows).set_index("target_pts")
    print(tb.round(3).to_string().replace("\n", "\n   ").join(["   ", ""]))
    print("   skill > 0 = better than always guessing the average rate; higher = better")
    tb.to_csv(os.path.join(outdir, f"hit_points{tag}.csv"))
    _plot_reliability({m: np.concatenate(v) for m, v in all_p.items()}, np.concatenate(all_y),
                      os.path.join(outdir, f"hit_reliability{tag}.png"), f"Touch-probability calibration {tag}")

    # C) fat tails
    z = np.log(fin / S) / sH
    print("\n C) Tails of the H-bar move (in σ units)")
    for kk, pn in ((2, 2 * norm.sf(2)), (3, 2 * norm.sf(3))):
        print(f"   |move| > {kk}σ: actual {np.mean(np.abs(z) > kk):.3%}  vs normal {pn:.3%}")

    # D) target before stop — Monte Carlo vs reality (random long entries: NO direction edge)
    print(f"\n D) Target {target:g} pts before stop {stop:g} pts within {H} bars — long at every bar")
    x_stop = -np.log1p(-stop / S) / sH
    ratio = float(np.median(np.log1p(target / S) / -np.log1p(-stop / S)))
    xs = np.linspace(0.02, 6, 80)
    g_t, g_s = mc_target_stop_grid(ratio, H, xs)
    p_t, p_s = np.interp(x_stop, xs, g_t), np.interp(x_stop, xs, g_s)

    ht, hs = Whi >= (S + target)[:, None], Wlo <= (S - stop)[:, None]
    ft = np.where(ht.any(1), ht.argmax(1), H)
    fs = np.where(hs.any(1), hs.argmax(1), H)
    tgt_first, stp_first = ft < fs, (fs < ft) | ((fs == ft) & (fs < H))   # same-bar tie → stop
    neither = ~tgt_first & ~stp_first
    ambiguous = np.mean((fs == ft) & (fs < H))
    pnl = np.where(tgt_first, target, np.where(stp_first, -stop, fin - S))

    q = pd.qcut(x_stop, 5, labels=False, duplicates="drop")
    tdf = pd.DataFrame(dict(q=q, stop_in_sigma=x_stop, pred_target=p_t, real_target=tgt_first,
                            pred_stop=p_s, real_stop=stp_first, real_neither=neither))
    td = tdf.groupby("q").mean().rename_axis("vol bucket (0=wild,4=calm)")
    print(td.round(3).to_string().replace("\n", "\n   ").join(["   ", ""]))
    print(f"   overall: pred target-first {p_t.mean():.3f}, actual {tgt_first.mean():.3f} │ "
          f"pred stop-first {p_s.mean():.3f}, actual {stp_first.mean():.3f} │ same-bar ties {ambiguous:.2%}")
    print(f"   no-time-limit theory P(target first) = stop/(target+stop) = {stop / (target + stop):.3f}")
    print(f"   avg P&L per random long: {pnl.mean():+.2f} pts (before costs, before option theta)")
    td.to_csv(os.path.join(outdir, f"hit_target_stop{tag}.csv"))
    return dict(sigma_levels=ta, points=tb, target_stop=td)


def _plot_reliability(preds, y, path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    for m, p in preds.items():
        bins = pd.qcut(p, 10, labels=False, duplicates="drop")
        g = pd.DataFrame(dict(b=bins, p=p, y=y)).groupby("b").mean()
        ax.plot(g.p, g.y, marker="o", label=m)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("actual frequency")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="EWMA Lab: Markov direction test + hit-probability test")
    sp = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--csv", required=True)
        p.add_argument("--out", default="results")
        p.add_argument("--lam", type=float, default=0.94)

    f = sp.add_parser("fetch")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--out", default="nifty_5m.csv")
    f.add_argument("--token", default="99926000")
    f.add_argument("--exchange", default="NSE", help="NSE for index, NFO for futures")
    f.add_argument("--interval", default="FIVE_MINUTE",
                   help="ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, ONE_HOUR, ONE_DAY …")

    for name in ("markov", "hitprob", "all"):
        p = sp.add_parser(name)
        common(p)
        p.add_argument("--horizons", default="1,3,12")
        p.add_argument("--train-frac", type=float, default=0.6)
        p.add_argument("--min-n", type=int, default=300)
        p.add_argument("--cost-pts", type=float, default=1.0)
        p.add_argument("--H", type=int, default=12)
        p.add_argument("--target", type=float, default=60)
        p.add_argument("--stop", type=float, default=30)
        p.add_argument("--points", default="20,40,60,100")
        p.add_argument("--vix-csv", default=None)

    sp.add_parser("demo")
    tk = sp.add_parser("tokens", help="look up an Angel One instrument token")
    tk.add_argument("--search", default="NIFTY", help='e.g. "INDIA VIX", NIFTYBEES')
    tk.add_argument("--exch", default=None, help="NSE or NFO")
    a = ap.parse_args()

    if a.cmd == "tokens":
        list_tokens(a.search, a.exch)
        return

    if a.cmd == "fetch":
        max_days = {"ONE_MINUTE": 25, "THREE_MINUTE": 55, "FIVE_MINUTE": 60, "TEN_MINUTE": 90,
                    "FIFTEEN_MINUTE": 180, "THIRTY_MINUTE": 180, "ONE_HOUR": 360, "ONE_DAY": 1800}
        fetch_angel(a.start, a.end, a.out, token=a.token, exchange=a.exchange, interval=a.interval,
                    chunk_days=max_days.get(a.interval, 60))
        return

    if a.cmd == "demo":
        cfg = Cfg()
        for plant, tag in ((False, "_demo_noise"), (True, "_demo_planted")):
            print(f"\n\n######## SYNTHETIC DATA — {'hidden momentum edge' if plant else 'pure noise'} ########")
            d = add_features(make_synthetic(plant=plant), cfg)
            markov_test(d, cfg, horizons=(1,), tag=tag)
            if not plant:
                hitprob_test(d, cfg, tag=tag)
        return

    cfg = Cfg(lam=a.lam)
    d = add_features(load_csv(a.csv), cfg)
    print(f"Loaded {len(d):,} bars, {d['date'].nunique()} sessions: {d.index[0]} → {d.index[-1]}")
    if a.cmd in ("markov", "all"):
        markov_test(d, cfg, horizons=tuple(int(h) for h in a.horizons.split(",")),
                    train_frac=a.train_frac, min_n=a.min_n, cost_pts=a.cost_pts, outdir=a.out)
    if a.cmd in ("hitprob", "all"):
        vix = load_vix(a.vix_csv) if a.vix_csv else None
        hitprob_test(d, cfg, H=a.H, pts=tuple(float(x) for x in a.points.split(",")),
                     target=a.target, stop=a.stop, vix=vix, outdir=a.out)
    print(f"\nTables and charts saved in ./{a.out}/")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:          # ▶ Run button with no arguments → run the demo
        sys.argv.append("demo")
    main()
