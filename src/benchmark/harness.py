"""Ölçüm çekirdeği: ısınma → termal bekleme → ölçüm → ham JSON (results/raw/).

Protokol (Bölüm 3.5): her koşu öncesi termal durum kontrol edilir; nominal değilse
soğuma beklenir. Enerji: powermetrics şifresiz sudo yoksa 'unavailable' yazılır —
ölçüm engellenmez (dizüstünde fişte olmak kullanıcının sorumluluğunda).
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import env as env_mod
from .stats import summarize

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"


@dataclass
class MeasureConfig:
    warmup: int = 20
    runs: int = 100
    thermal_wait_s: int = 120  # nominal değilse en fazla bu kadar bekle
    cooldown_poll_s: int = 10
    tags: dict = field(default_factory=dict)


def _wait_thermal_nominal(cfg: MeasureConfig) -> str:
    waited = 0
    state = env_mod.thermal_state()
    while state != "nominal" and waited < cfg.thermal_wait_s:
        time.sleep(cfg.cooldown_poll_s)
        waited += cfg.cooldown_poll_s
        state = env_mod.thermal_state()
    return state


def _energy_available() -> bool:
    try:
        return subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


def run_benchmark(name: str, runner, x: np.ndarray, cfg: MeasureConfig | None = None) -> dict:
    """Tek (model, yığın, girdi) hücresini ölç ve ham JSON'a ekle (append-safe)."""
    cfg = cfg or MeasureConfig()
    thermal = _wait_thermal_nominal(cfg)

    runner.prepare(x)
    for _ in range(cfg.warmup):
        runner.run_once()
    runner.sync()

    samples = []
    for _ in range(cfg.runs):
        t0 = time.perf_counter()
        runner.run_once()
        runner.sync()
        samples.append((time.perf_counter() - t0) * 1000)

    rec = {
        "name": name,
        "stack": runner.label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_shape": list(x.shape),
        "input_dtype": str(x.dtype),
        "config": {"warmup": cfg.warmup, "runs": cfg.runs, **cfg.tags},
        "thermal_at_start": thermal,
        "energy": "available" if _energy_available() else "unavailable (sudo yok)",
        "latency": summarize(samples),
        "runner": runner.stats(),
        "env": env_mod.capture(),
        "raw_samples_ms": [round(s, 4) for s in samples],
    }

    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{name}.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{name}] {runner.label}: medyan {rec['latency']['median_ms']} ms "
          f"(n={cfg.runs}, termal={thermal})", flush=True)
    return rec
