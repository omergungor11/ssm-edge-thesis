"""TASK-021 çözünürlük taraması — eager yığınlar (AS2 ölçekleme eğrileri).

3 model × {torch_cpu, torch_mps} × {256, 768, 1024}. 512 verisi önceki turdan var.
Kayıtlar: results/raw/latency_matrix_*.jsonl (harness, resolution tag'li).
"""
from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark import MeasureConfig, TorchRunner, run_benchmark
from models import vmamba_upernet as vu

vu.PSPModule.export_mode = True  # MPS adaptive-pool kısıtı
RESOLUTIONS = (256, 768, 1024)


def get_model(name: str):
    if name == "vmamba":
        from models.vmamba_upernet import load_pretrained
    elif name == "swin":
        from models.swin_upernet import load_pretrained
    else:
        from models.convnext_upernet import load_pretrained
    return load_pretrained()


def main() -> None:
    for m in ("convnext", "swin", "vmamba"):
        model = get_model(m)
        for res in RESOLUTIONS:
            x = np.random.rand(1, 3, res, res).astype(np.float32)
            runs = 8 if (m == "vmamba" or res >= 768) else 15
            cfg = MeasureConfig(warmup=2, runs=runs, tags={"resolution": res})
            for dev in ("mps", "cpu"):
                try:
                    model.to(dev)
                    run_benchmark(f"latency_matrix_{m}", TorchRunner(model, dev), x, cfg)
                except Exception as e:
                    print(f"[{m}/{dev}/{res}] HATA: {type(e).__name__}: {str(e)[:150]}",
                          flush=True)
        model.to("cpu")
        del model
        gc.collect()
        torch.mps.empty_cache()
    print("TARAMA-TAMAM", flush=True)


if __name__ == "__main__":
    main()
