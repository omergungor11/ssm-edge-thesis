"""TASK-010: Harness doğrulaması — ResNet-50 (bilinen model, bilinen davranış).

Beklenti (M-serisi, 224x224, batch=1): torch CPU onlarca ms, MPS < CPU (paralel
dostu model — MiniMamba'nın tersi), CoreML ANE birkaç ms. Bu desen tutmazsa
harness şüphelidir, Mamba sonuçlarına güvenilemez.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import CoreMLRunner, MeasureConfig, OrtRunner, TorchRunner, run_benchmark
from benchmark.harness import RAW

NAME = "validate_resnet50"


def main() -> None:
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    x = np.random.rand(1, 3, 224, 224).astype(np.float32)
    xt = torch.from_numpy(x)
    cfg = MeasureConfig(warmup=15, runs=60)

    run_benchmark(NAME, TorchRunner(model, "cpu"), x, cfg)
    if torch.backends.mps.is_available():
        run_benchmark(NAME, TorchRunner(model, "mps"), x, cfg)

    onnx_path = RAW / "resnet50.onnx"
    if not onnx_path.exists():
        t0 = time.perf_counter()
        torch.onnx.export(model.cpu(), (xt,), str(onnx_path), opset_version=17, dynamo=False)
        print(f"ONNX export: {time.perf_counter()-t0:.1f} s, "
              f"{onnx_path.stat().st_size/1e6:.1f} MB", flush=True)
    run_benchmark(NAME, OrtRunner(str(onnx_path)), x, cfg)
    try:
        run_benchmark(NAME, OrtRunner(str(onnx_path),
                      ["CoreMLExecutionProvider", "CPUExecutionProvider"]), x, cfg)
    except Exception as e:
        print(f"ORT CoreML EP başarısız: {type(e).__name__}: {str(e)[:200]}", flush=True)

    ml_path = RAW / "resnet50.mlpackage"
    if not ml_path.exists():
        import coremltools as ct

        t0 = time.perf_counter()
        mlm = ct.convert(torch.jit.trace(model.cpu(), xt),
                         inputs=[ct.TensorType(shape=x.shape, dtype=float)],
                         minimum_deployment_target=ct.target.macOS15,
                         compute_units=ct.ComputeUnit.ALL)
        mlm.save(str(ml_path))
        print(f"CoreML dönüşüm: {time.perf_counter()-t0:.1f} s", flush=True)
    for cu in ("CPU_ONLY", "CPU_AND_GPU", "ALL"):
        try:
            run_benchmark(NAME, CoreMLRunner(str(ml_path), cu), x, cfg)
        except Exception as e:
            print(f"CoreML {cu} başarısız: {type(e).__name__}: {str(e)[:200]}", flush=True)

    print("TAMAM", flush=True)


if __name__ == "__main__":
    main()
