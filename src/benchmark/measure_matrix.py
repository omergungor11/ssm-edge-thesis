"""TASK-021 ilk tur: gecikme matrisi + powermetrics eşzamanlı enerji penceresi.

25 dakikalık kullanıcı penceresine sığacak hücre seçimi (512×512, batch 1):
- swin / convnext: CoreML(ALL, CPU_ONLY, CPU_AND_GPU), ORT CPU, torch MPS, torch CPU
- vmamba: torch CPU, torch MPS (ORT yüklemesi 12 dk sürdüğü için bu tura alınmadı)

Her kayıt UTC zaman damgalı → /tmp/tez-powermetrics.log ile sonradan eşleştirilir.
Çıktı: results/raw/latency_matrix.jsonl (harness üzerinden, ortam bilgisi gömülü).
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark import CoreMLRunner, MeasureConfig, OrtRunner, TorchRunner, run_benchmark
from benchmark.harness import RAW

NAME = "latency_matrix"
SIZE = 512
FAST = MeasureConfig(warmup=10, runs=50, tags={"resolution": SIZE})
SLOW = MeasureConfig(warmup=3, runs=15, tags={"resolution": SIZE})  # CPU'da s-mertebesi hücreler


def get_model(name: str):
    if name == "vmamba":
        from models.vmamba_upernet import load_pretrained
    elif name == "swin":
        from models.swin_upernet import load_pretrained
    else:
        from models.convnext_upernet import load_pretrained
    return load_pretrained()


def cell(fn, label: str) -> None:
    try:
        fn()
    except Exception as e:
        print(f"[{label}] HATA: {type(e).__name__}: {str(e)[:200]}", flush=True)


def main() -> None:
    x = np.random.rand(1, 3, SIZE, SIZE).astype(np.float32)

    for m in ("swin", "convnext"):
        model = get_model(m)
        for cu in ("ALL", "CPU_ONLY", "CPU_AND_GPU"):
            cell(lambda cu=cu: run_benchmark(f"{NAME}_{m}",
                 CoreMLRunner(str(RAW / f"{m}_upernet_{SIZE}.mlpackage"), cu), x, FAST),
                 f"{m}/coreml_{cu}")
        cell(lambda: run_benchmark(f"{NAME}_{m}",
             OrtRunner(str(RAW / f"{m}_upernet_{SIZE}.onnx")), x, SLOW), f"{m}/ort_cpu")
        cell(lambda: run_benchmark(f"{NAME}_{m}", TorchRunner(model, "mps"), x, FAST),
             f"{m}/torch_mps")
        model.to("cpu")
        cell(lambda: run_benchmark(f"{NAME}_{m}", TorchRunner(model, "cpu"), x, SLOW),
             f"{m}/torch_cpu")
        del model
        gc.collect()
        torch.mps.empty_cache()

    model = get_model("vmamba")
    cell(lambda: run_benchmark(f"{NAME}_vmamba", TorchRunner(model, "mps"), x, SLOW),
         "vmamba/torch_mps")
    model.to("cpu")
    cell(lambda: run_benchmark(f"{NAME}_vmamba", TorchRunner(model, "cpu"), x, SLOW),
         "vmamba/torch_cpu")
    print("MATRIS-TURU-TAMAM", flush=True)


if __name__ == "__main__":
    main()
