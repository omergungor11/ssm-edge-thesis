"""İstatistik özetleme — protokol: medyan/ortalama/std/P90/P99 (Bölüm 3.5.5)."""
from __future__ import annotations

import statistics


def summarize(samples_ms: list[float]) -> dict:
    s = sorted(samples_ms)
    n = len(s)

    def pct(p: float) -> float:
        return s[min(n - 1, max(0, round(p / 100 * n) - 1))]

    return {
        "n": n,
        "median_ms": round(statistics.median(s), 4),
        "mean_ms": round(statistics.fmean(s), 4),
        "std_ms": round(statistics.stdev(s), 4) if n > 1 else 0.0,
        "min_ms": round(s[0], 4),
        "p90_ms": round(pct(90), 4),
        "p99_ms": round(pct(99), 4),
        "max_ms": round(s[-1], 4),
    }
