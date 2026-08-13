"""TASK-021 kalan hücreler + TASK-022 operatör profili.

Sıra:
1. ORT CoreML EP (swin, convnext) — mevcut .onnx dosyalarıyla
2. torch.compile (3 model, CPU; başarısızlık da kaydedilir)
3. VMamba CPU@768 yeniden doğrulama (önceki kayıt kirlilik şüpheli)
4. VMamba ORT CPU: tek oturum — yükleme (yeniden ölçüm) + gecikme + profil (2 koşu)
5. ORT profil (swin, convnext, 10 koşu) → results/raw/ort_profile_<model>_top.json

Bellek tavanı ~7 GB (VMamba ORT). Ağır export YOK.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from benchmark import MeasureConfig, OrtRunner, TorchRunner, run_benchmark
from benchmark.harness import RAW
from models import vmamba_upernet as vu

vu.PSPModule.export_mode = True
SIZE = 512
X = np.random.rand(1, 3, SIZE, SIZE).astype(np.float32)


def get_model(name: str):
    if name == "vmamba":
        from models.vmamba_upernet import load_pretrained
    elif name == "swin":
        from models.swin_upernet import load_pretrained
    else:
        from models.convnext_upernet import load_pretrained
    return load_pretrained()


def profile_ort(model_name: str, runs: int) -> None:
    """ORT profillemesi: op-tipine göre kümülatif süre özeti (TASK-022)."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.enable_profiling = True
    sess = ort.InferenceSession(str(RAW / f"{model_name}_upernet_{SIZE}.onnx"), so,
                                providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: X}
    for _ in range(2):  # ısınma (profile dahil olur ama op-tipi oranlarını bozmaz)
        sess.run(None, feed)
    for _ in range(runs):
        sess.run(None, feed)
    prof = Path(sess.end_profiling())
    agg: dict[str, list[float]] = {}
    for ev in json.loads(prof.read_text()):
        if ev.get("cat") == "Node" and ev.get("dur") is not None:
            op = ev.get("args", {}).get("op_name", "?")
            agg.setdefault(op, [0.0, 0])
            agg[op][0] += ev["dur"] / 1000.0  # ms
            agg[op][1] += 1
    total = sum(v[0] for v in agg.values())
    top = sorted(agg.items(), key=lambda kv: -kv[1][0])[:15]
    out = {"model": model_name, "runs": runs, "total_node_ms": round(total, 1),
           "top_ops": [{"op": k, "ms": round(v[0], 1), "count": v[1],
                        "pct": round(100 * v[0] / total, 1)} for k, v in top]}
    (RAW / f"ort_profile_{model_name}_top.json").write_text(json.dumps(out, indent=2))
    prof.unlink()  # ham profil dosyası devasa olabilir; özeti sakladık
    print(f"[profil/{model_name}] toplam {total:.0f} ms/koşu-kümülatif; "
          f"ilk3: {[(o['op'], o['pct']) for o in out['top_ops'][:3]]}", flush=True)


def main() -> None:
    # 1) ORT CoreML EP
    for m in ("swin", "convnext"):
        try:
            run_benchmark(f"latency_matrix_{m}",
                          OrtRunner(str(RAW / f"{m}_upernet_{SIZE}.onnx"),
                                    ["CoreMLExecutionProvider", "CPUExecutionProvider"]),
                          X, MeasureConfig(warmup=10, runs=30, tags={"resolution": SIZE}))
        except Exception as e:
            print(f"[{m}/ort_coreml_ep] HATA: {type(e).__name__}: {str(e)[:200]}", flush=True)

    # 2) torch.compile (CPU, inductor)
    for m in ("swin", "convnext", "vmamba"):
        model = get_model(m)
        try:
            runs = 8 if m == "vmamba" else 15
            run_benchmark(f"latency_matrix_{m}",
                          TorchRunner(model, "cpu", compile_mode="default"), X,
                          MeasureConfig(warmup=3, runs=runs,
                                        tags={"resolution": SIZE, "compile": "inductor-cpu"}))
        except Exception as e:
            print(f"[{m}/torch_compile] HATA: {type(e).__name__}: {str(e)[:300]}", flush=True)
        del model
        gc.collect()

    # 3) VMamba CPU@768 yeniden doğrulama
    model = get_model("vmamba")
    x768 = np.random.rand(1, 3, 768, 768).astype(np.float32)
    run_benchmark("latency_matrix_vmamba", TorchRunner(model, "cpu"), x768,
                  MeasureConfig(warmup=2, runs=8, tags={"resolution": 768, "reverify": True}))
    del model
    gc.collect()

    # 4) VMamba ORT (yükleme + gecikme + kısa profil)
    try:
        t0 = time.perf_counter()
        runner = OrtRunner(str(RAW / f"vmamba_upernet_{SIZE}.onnx"))
        print(f"[vmamba/ort_load] {time.perf_counter()-t0:.0f} s", flush=True)
        run_benchmark("latency_matrix_vmamba", runner, X,
                      MeasureConfig(warmup=3, runs=15, tags={"resolution": SIZE}))
        del runner
        gc.collect()
        profile_ort("vmamba", runs=2)
    except Exception as e:
        print(f"[vmamba/ort] HATA: {type(e).__name__}: {str(e)[:200]}", flush=True)

    # 5) Klasiklerin profili
    for m in ("swin", "convnext"):
        try:
            profile_ort(m, runs=10)
        except Exception as e:
            print(f"[profil/{m}] HATA: {type(e).__name__}: {str(e)[:200]}", flush=True)

    print("TUR3-TAMAM", flush=True)


if __name__ == "__main__":
    main()
