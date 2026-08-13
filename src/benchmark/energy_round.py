"""TASK-021 enerji turu — powermetrics eşzamanlı, pencere-eşleştirmeli.

Önkoşul: sudo powermetrics -i 200 --samplers cpu_power,gpu_power,ane_power \
         -o /tmp/tez-powermetrics.log   (kullanıcı oturumunda çalışıyor olmalı)

Her hücre: ısınma → [t0, t1] penceresinde ölçüm döngüsü → kayıt.
Sonra log'daki örnekler pencereyle eşleştirilir; boşta taban çizgisi düşülür.
Çıktı: results/raw/energy_matrix.json
"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark.runners import CoreMLRunner, TorchRunner
from models import vmamba_upernet as vu

RAW = ROOT / "results" / "raw"
LOG = Path("/tmp/tez-powermetrics.log")
SIZE = 512


def run_cell(label: str, runner, x, runs: int, warmup: int = 5) -> dict:
    runner.prepare(x)
    for _ in range(warmup):
        runner.run_once()
    runner.sync()
    t0 = time.time()
    tp = time.perf_counter()
    for _ in range(runs):
        runner.run_once()
        runner.sync()
    dur = time.perf_counter() - tp
    rec = {"label": label, "t0": t0, "t1": time.time(), "runs": runs,
           "ms_per_inf": round(dur / runs * 1000, 2)}
    print(f"[{label}] {rec['ms_per_inf']} ms/inf, pencere {dur:.1f} s", flush=True)
    return rec


def idle_cell(seconds: int = 15) -> dict:
    print(f"[idle] {seconds} s boşta taban çizgisi...", flush=True)
    t0 = time.time()
    time.sleep(seconds)
    return {"label": "idle_baseline", "t0": t0, "t1": time.time(), "runs": 0, "ms_per_inf": None}


def parse_log() -> list[tuple[float, float, float, float]]:
    """(epoch, cpu_mw, gpu_mw, ane_mw) örnekleri."""
    samples, ts, vals = [], None, {}
    pat_hdr = re.compile(r"\*\*\* Sampled system activity \((.+?)\)")
    pat_pow = re.compile(r"^(CPU|GPU|ANE) Power:\s+(\d+)\s*mW")
    for line in LOG.read_text(errors="ignore").splitlines():
        m = pat_hdr.search(line)
        if m:
            if ts is not None and len(vals) == 3:
                samples.append((ts, vals["CPU"], vals["GPU"], vals["ANE"]))
            try:
                ts = datetime.strptime(m.group(1).rsplit(" ", 1)[0],
                                       "%a %b %d %H:%M:%S %Y").timestamp()
            except ValueError:
                ts = None
            vals = {}
            continue
        m = pat_pow.match(line.strip())
        if m and ts is not None:
            vals[m.group(1)] = float(m.group(2))
    if ts is not None and len(vals) == 3:
        samples.append((ts, vals["CPU"], vals["GPU"], vals["ANE"]))
    return samples


def main() -> None:
    assert LOG.exists(), "powermetrics logu yok"
    vu.PSPModule.export_mode = True  # MPS uyumluluğu
    x = np.random.rand(1, 3, SIZE, SIZE).astype(np.float32)
    xt = x
    cells = [idle_cell(15)]

    for m in ("swin", "convnext"):
        if m == "swin":
            from models.swin_upernet import load_pretrained
        else:
            from models.convnext_upernet import load_pretrained
        model = load_pretrained()
        for cu in ("ALL", "CPU_AND_GPU", "CPU_ONLY"):
            runs = 20 if cu != "CPU_ONLY" else 10
            cells.append(run_cell(f"{m}/coreml_{cu.lower()}",
                         CoreMLRunner(str(RAW / f"{m}_upernet_{SIZE}.mlpackage"), cu), xt, runs))
        cells.append(run_cell(f"{m}/torch_mps", TorchRunner(model, "mps"), xt, 15))
        model.to("cpu")
        cells.append(run_cell(f"{m}/torch_cpu", TorchRunner(model, "cpu"), xt, 8, warmup=2))
        del model
        torch.mps.empty_cache()

    from models.vmamba_upernet import load_pretrained as lv
    model = lv()
    cells.append(run_cell("vmamba/torch_mps", TorchRunner(model, "mps"), xt, 8, warmup=2))
    model.to("cpu")
    cells.append(run_cell("vmamba/torch_cpu", TorchRunner(model, "cpu"), xt, 5, warmup=1))

    print("hücreler bitti; log işleniyor (powermetrics artık kapatılabilir)", flush=True)
    time.sleep(3)
    samples = parse_log()
    print(f"log örneği: {len(samples)}", flush=True)

    idle = next(c for c in cells if c["label"] == "idle_baseline")
    def window_avg(t0, t1):
        w = [(c, g, a) for ts, c, g, a in samples if t0 <= ts <= t1]
        if not w:
            return None
        arr = np.array(w)
        return {"cpu_mw": round(float(arr[:, 0].mean()), 1),
                "gpu_mw": round(float(arr[:, 1].mean()), 1),
                "ane_mw": round(float(arr[:, 2].mean()), 1),
                "n_samples": len(w)}

    base = window_avg(idle["t0"], idle["t1"]) or {"cpu_mw": 0, "gpu_mw": 0, "ane_mw": 0}
    out = {"timestamp": datetime.now().isoformat(), "idle_baseline": base, "cells": []}
    for c in cells:
        if c["label"] == "idle_baseline":
            continue
        p = window_avg(c["t0"], c["t1"])
        if p:
            total = p["cpu_mw"] + p["gpu_mw"] + p["ane_mw"]
            net = total - (base["cpu_mw"] + base["gpu_mw"] + base["ane_mw"])
            c["power"] = p
            c["net_power_mw"] = round(net, 1)
            c["mj_per_inf"] = round(net * c["ms_per_inf"] / 1000, 1)
        out["cells"].append(c)
        pw = c.get("power", {})
        print(f"{c['label']:26s} {c['ms_per_inf']:>8} ms | cpu {pw.get('cpu_mw','-'):>7} "
              f"gpu {pw.get('gpu_mw','-'):>7} ane {pw.get('ane_mw','-'):>6} mW "
              f"| {c.get('mj_per_inf','-'):>7} mJ/inf", flush=True)

    (RAW / "energy_matrix.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("ENERJI-TURU-TAMAM", flush=True)


if __name__ == "__main__":
    main()
