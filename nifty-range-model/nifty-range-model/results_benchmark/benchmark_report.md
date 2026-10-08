# COMBO range model — out-of-sample benchmark 

Data: 2005-01-03 → 2026-09-29, 5,350 daily bars. Walk-forward: every model refitted each year on earlier data only.


## Horizon 1 day — 3,859 forecasts, 2011–2026

```
                    CRPS (bps)    QLIKE Interval score 90% (bps) Coverage 80% Coverage 95% Avg 95% width (%)
Historical vol (1y)      53.58  -8.2143                    455.9        83.0%        94.4%              3.89
RiskMetrics EWMA         52.59  -8.3819                    422.9        80.5%        93.9%              3.68
GARCH(1,1)               52.67  -8.3921                    423.2        84.8%        95.6%              3.99
India VIX (raw)          52.76  -8.3856                    426.7        87.9%        97.3%              4.30
COMBO (ours)             52.04  -8.4337                    413.0        81.0%        95.6%              3.91
COMBO no-VIX (ours)      52.41  -8.3866                    423.5        79.5%        95.2%              3.91
```


```
                    VaR95 breaches Kupiec p (95) VaR99 breaches Kupiec p (99) Christoffersen p (99) Basel green Basel red
Historical vol (1y)          4.85%         0.659          1.76%         0.000                 0.008         56%        7%
RiskMetrics EWMA             5.65%         0.070          1.94%         0.000                 0.664         55%        3%
GARCH(1,1)                   4.30%         0.042          1.35%         0.039                 0.733         80%        3%
India VIX (raw)              3.11%         0.000          0.88%         0.448                 0.437         92%        0%
COMBO (ours)                 4.95%         0.885          1.06%         0.700                 0.348         81%        0%
COMBO no-VIX (ours)          4.98%         0.944          1.17%         0.312                 0.555         83%        0%
```


```
                                             CRPS  QLIKE Interval
COMBO (ours) vs Historical vol (1y)         +9.99  +5.79    +6.99
COMBO (ours) vs RiskMetrics EWMA            +7.41  +4.71    +4.09
COMBO (ours) vs GARCH(1,1)                  +8.15  +5.12    +4.67
COMBO (ours) vs India VIX (raw)             +8.81  +6.16    +6.29
COMBO no-VIX (ours) vs Historical vol (1y)  +7.99  +5.17    +6.09
COMBO no-VIX (ours) vs RiskMetrics EWMA     +2.42  +1.32    -0.66
COMBO no-VIX (ours) vs GARCH(1,1)           +2.92  -0.83    -0.18
COMBO no-VIX (ours) vs India VIX (raw)      +2.93  +0.07    +0.95
```



## Horizon 5 days — 3,855 forecasts, 2011–2026

```
                    CRPS (bps)    QLIKE Interval score 90% (bps) Coverage 80% Coverage 95% Avg 95% width (%)
Historical vol (1y)     125.14  -6.5370                   1005.6        80.6%        94.1%              8.71
RiskMetrics EWMA        124.10  -6.6268                    977.5        78.1%        92.5%              8.24
GARCH(1,1)              123.83  -6.6748                    962.6        83.1%        95.6%              9.08
India VIX (raw)         123.47  -6.6905                    955.0        85.8%        96.9%              9.61
COMBO (ours)            122.30  -6.7041                    940.5        80.1%        95.7%              9.03
COMBO no-VIX (ours)     123.24  -6.6552                    969.4        80.2%        94.7%              8.85
```


```
                    VaR95 breaches Kupiec p (95) VaR99 breaches Kupiec p (99) Christoffersen p (99) Basel green Basel red
Historical vol (1y)          5.97%         0.232          2.08%         0.009                     —           —         —
RiskMetrics EWMA             7.26%         0.007          2.33%         0.001                     —           —         —
GARCH(1,1)                   5.45%         0.574          1.69%         0.081                     —           —         —
India VIX (raw)              4.02%         0.197          1.04%         0.917                     —           —         —
COMBO (ours)                 5.32%         0.689          1.30%         0.428                     —           —         —
COMBO no-VIX (ours)          5.32%         0.689          1.04%         0.917                     —           —         —
```


```
                                             CRPS  QLIKE Interval
COMBO (ours) vs Historical vol (1y)         +3.42  +3.01    +2.13
COMBO (ours) vs RiskMetrics EWMA            +3.43  +2.97    +2.94
COMBO (ours) vs GARCH(1,1)                  +3.27  +2.39    +2.20
COMBO (ours) vs India VIX (raw)             +2.52  +0.69    +1.19
COMBO no-VIX (ours) vs Historical vol (1y)  +1.91  +1.97    +1.01
COMBO no-VIX (ours) vs RiskMetrics EWMA     +1.37  +2.23    +1.01
COMBO no-VIX (ours) vs GARCH(1,1)           +1.19  -1.18    -0.75
COMBO no-VIX (ours) vs India VIX (raw)      +0.49  -1.93    -1.20
```



## Horizon 21 days — 3,839 forecasts, 2011–2026

```
                    CRPS (bps)    QLIKE Interval score 90% (bps) Coverage 80% Coverage 95% Avg 95% width (%)
Historical vol (1y)     258.32  -4.9696                   2093.4        81.0%        94.6%             17.86
RiskMetrics EWMA        258.34  -4.9942                   2084.7        77.2%        92.5%             16.92
GARCH(1,1)              257.94  -5.0842                   2061.7        85.2%        97.0%             19.67
India VIX (raw)         256.97  -5.0593                   2058.4        86.0%        97.1%             19.73
COMBO (ours)            250.25  -5.0252                   2032.5        81.1%        96.1%             19.00
COMBO no-VIX (ours)     251.56  -5.0471                   2088.8        83.3%        96.2%             19.57
```


```
                    VaR95 breaches Kupiec p (95) VaR99 breaches Kupiec p (99) Christoffersen p (99) Basel green Basel red
Historical vol (1y)          6.01%         0.543          2.73%         0.052                     —           —         —
RiskMetrics EWMA             8.20%         0.068          3.28%         0.014                     —           —         —
GARCH(1,1)                   4.37%         0.690          1.64%         0.426                     —           —         —
India VIX (raw)              4.92%         0.959          2.19%         0.163                     —           —         —
COMBO (ours)                 6.56%         0.355          1.64%         0.426                     —           —         —
COMBO no-VIX (ours)          6.56%         0.355          1.64%         0.426                     —           —         —
```


```
                                             CRPS  QLIKE Interval
COMBO (ours) vs Historical vol (1y)         +2.88  +1.51    +1.44
COMBO (ours) vs RiskMetrics EWMA            +2.50  +1.24    +1.67
COMBO (ours) vs GARCH(1,1)                  +2.23  -0.30    +0.37
COMBO (ours) vs India VIX (raw)             +1.93  -0.95    -0.79
COMBO no-VIX (ours) vs Historical vol (1y)  +1.62  +1.18    +0.75
COMBO no-VIX (ours) vs RiskMetrics EWMA     +1.48  +0.92    +1.39
COMBO no-VIX (ours) vs GARCH(1,1)           +1.33  -0.84    -0.11
COMBO no-VIX (ours) vs India VIX (raw)      +1.05  -1.53    -1.51
```



## Claims supported by the data (|t| ≥ 2)

- **WIN** — H=1: COMBO (ours) vs Historical vol (1y): CRPS −2.9% (t=10.0), QLIKE Δ 0.2194 (t=5.8), Interval −9.4% (t=7.0)
- **WIN** — H=1: COMBO (ours) vs RiskMetrics EWMA: CRPS −1.0% (t=7.4), QLIKE Δ 0.0518 (t=4.7), Interval −2.4% (t=4.1)
- **WIN** — H=1: COMBO (ours) vs GARCH(1,1): CRPS −1.2% (t=8.2), QLIKE Δ 0.0417 (t=5.1), Interval −2.4% (t=4.7)
- **WIN** — H=1: COMBO (ours) vs India VIX (raw): CRPS −1.4% (t=8.8), QLIKE Δ 0.0481 (t=6.2), Interval −3.2% (t=6.3)
- **WIN** — H=1: COMBO no-VIX (ours) vs Historical vol (1y): CRPS −2.2% (t=8.0), QLIKE Δ 0.1722 (t=5.2), Interval −7.1% (t=6.1)
- **WIN** — H=1: COMBO no-VIX (ours) vs RiskMetrics EWMA: CRPS −0.3% (t=2.4)
- **WIN** — H=1: COMBO no-VIX (ours) vs GARCH(1,1): CRPS −0.5% (t=2.9)
- **WIN** — H=1: COMBO no-VIX (ours) vs India VIX (raw): CRPS −0.7% (t=2.9)
- **WIN** — H=5: COMBO (ours) vs Historical vol (1y): CRPS −2.3% (t=3.4), QLIKE Δ 0.1671 (t=3.0), Interval −6.5% (t=2.1)
- **WIN** — H=5: COMBO (ours) vs RiskMetrics EWMA: CRPS −1.5% (t=3.4), QLIKE Δ 0.0773 (t=3.0), Interval −3.8% (t=2.9)
- **WIN** — H=5: COMBO (ours) vs GARCH(1,1): CRPS −1.2% (t=3.3), QLIKE Δ 0.0293 (t=2.4), Interval −2.3% (t=2.2)
- **WIN** — H=5: COMBO (ours) vs India VIX (raw): CRPS −0.9% (t=2.5)
- **TIE** — H=5: COMBO no-VIX (ours) vs Historical vol (1y): no significant difference
- **WIN** — H=5: COMBO no-VIX (ours) vs RiskMetrics EWMA: QLIKE Δ 0.0283 (t=2.2)
- **TIE** — H=5: COMBO no-VIX (ours) vs GARCH(1,1): no significant difference
- **TIE** — H=5: COMBO no-VIX (ours) vs India VIX (raw): no significant difference
- **WIN** — H=21: COMBO (ours) vs Historical vol (1y): CRPS −3.1% (t=2.9)
- **WIN** — H=21: COMBO (ours) vs RiskMetrics EWMA: CRPS −3.1% (t=2.5)
- **WIN** — H=21: COMBO (ours) vs GARCH(1,1): CRPS −3.0% (t=2.2)
- **TIE** — H=21: COMBO (ours) vs India VIX (raw): no significant difference
- **TIE** — H=21: COMBO no-VIX (ours) vs Historical vol (1y): no significant difference
- **TIE** — H=21: COMBO no-VIX (ours) vs RiskMetrics EWMA: no significant difference
- **TIE** — H=21: COMBO no-VIX (ours) vs GARCH(1,1): no significant difference
- **TIE** — H=21: COMBO no-VIX (ours) vs India VIX (raw): no significant difference