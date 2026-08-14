"""TASK-025 ilk tur: ağırlık-nicemleme (kalibrasyonsuz) — üretim + gecikme + sapma.

Kapsam (bilinçli hafif):
- CoreML (swin, convnext): W8 linear + W4 palettization → .mlpackage üret,
  CoreML ALL/CPU+GPU gecikmesi + fp32 CoreML'e karşı çıktı sapması (256 örnek piksel)
- ORT (3 model): dinamik INT8 (quantize_dynamic, ağırlık-yalnız) → gecikme + sapma
  (vmamba INT8 ORT: yükleme yine uzun sürebilir — kayıtlanır)

mIoU etkisi (TASK-026) sonraki tur; burada amaç boru hattının çalışırlığı ve
boyut/gecikme/sapma üçlüsü. Ham kayıt: results/raw/quant_matrix.jsonl
"""
from __future__ import annotations

import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark import CoreMLRunner, MeasureConfig, OrtRunner, run_benchmark
from benchmark.harness import RAW

OUT = RAW / "quant_matrix.jsonl"
SIZE = 512
X = np.random.rand(1, 3, SIZE, SIZE).astype(np.float32)


def log(**data) -> None:
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(), **data}
    with OUT.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[quant] {data}", flush=True)


def out_diff(a, b) -> dict:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"max_abs": round(float(d.max()), 4), "mean_abs": round(float(d.mean()), 5),
            "argmax_match_pct": round(float((a.argmax(1) == b.argmax(1)).mean()) * 100, 2)}


def coreml_out(mlm) -> np.ndarray:
    spec = mlm.get_spec()
    name = spec.description.output[0].name
    return mlm.predict({spec.description.input[0].name: X})[name]


def main() -> None:
    import coremltools as ct
    import coremltools.optimize.coreml as cto

    # --- CoreML ağırlık nicemleme (klasikler; VMamba CoreML'e giremiyor) ---
    for m in ("convnext", "swin"):
        base = ct.models.MLModel(str(RAW / f"{m}_upernet_{SIZE}.mlpackage"),
                                 compute_units=ct.ComputeUnit.CPU_AND_GPU)
        ref = coreml_out(base)

        for mode, make_cfg in {
            "w8_linear": lambda: cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)),
            "w4_palette": lambda: cto.OptimizationConfig(global_config=cto.OpPalettizerConfig(nbits=4, weight_threshold=512)),
        }.items():
            try:
                t0 = time.perf_counter()
                if mode == "w8_linear":
                    q = cto.linear_quantize_weights(base, make_cfg())
                else:
                    q = cto.palettize_weights(base, make_cfg())
                qs = round(time.perf_counter() - t0, 1)
                path = RAW / f"{m}_upernet_{SIZE}_{mode}.mlpackage"
                q.save(str(path))
                size_mb = round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6, 1)
                qd = out_diff(coreml_out(q), ref)
                log(model=m, backend="coreml", mode=mode, ok=True, quantize_s=qs,
                    size_mb=size_mb, diff_vs_fp32=qd)
                for cu in ("ALL", "CPU_AND_GPU"):
                    run_benchmark(f"latency_matrix_{m}", CoreMLRunner(str(path), cu), X,
                                  MeasureConfig(warmup=8, runs=30,
                                                tags={"resolution": SIZE, "quant": mode}))
                del q
            except Exception as e:
                log(model=m, backend="coreml", mode=mode, ok=False,
                    error=f"{type(e).__name__}: {e}"[:300])
            gc.collect()
        del base
        gc.collect()

    # --- ORT dinamik INT8 (3 model) ---
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for m in ("convnext", "swin", "vmamba"):
        src = RAW / f"{m}_upernet_{SIZE}.onnx"
        dst = RAW / f"{m}_upernet_{SIZE}_int8.onnx"
        try:
            t0 = time.perf_counter()
            quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
            qs = round(time.perf_counter() - t0, 1)
            log(model=m, backend="ort", mode="dynamic_int8", ok=True, quantize_s=qs,
                size_mb=round(dst.stat().st_size / 1e6, 1))
            runner = OrtRunner(str(dst))
            log(model=m, backend="ort", mode="dynamic_int8", stage="load",
                load_s=runner.stats()["load_s"])
            # sapma: fp32 ORT çıktısına karşı
            import onnxruntime as ort
            ref_sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]) \
                if m != "vmamba" else None  # vmamba fp32 yüklemesi 12 dk — sapma atlanır
            if ref_sess is not None:
                ref = ref_sess.run(None, {ref_sess.get_inputs()[0].name: X})[0]
                qo = runner.sess.run(None, {runner.sess.get_inputs()[0].name: X})[0]
                log(model=m, backend="ort", mode="dynamic_int8", diff_vs_fp32=out_diff(qo, ref))
                del ref_sess
            runs = 10 if m == "vmamba" else 15
            run_benchmark(f"latency_matrix_{m}", runner, X,
                          MeasureConfig(warmup=3, runs=runs,
                                        tags={"resolution": SIZE, "quant": "ort_int8"}))
            del runner
        except Exception as e:
            log(model=m, backend="ort", mode="dynamic_int8", ok=False,
                error=f"{type(e).__name__}: {e}"[:300])
        gc.collect()

    print("QUANT-TUR1-TAMAM", flush=True)


if __name__ == "__main__":
    main()
