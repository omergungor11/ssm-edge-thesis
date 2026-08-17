"""TASK-030 Adım 5 — form başına export + ölçüm matrisi.

Kullanım: python run_matrix.py <form> [stage]
  form : seq | blocked64 | blocked128 | ane64 | ane128
  stage: torch | onnx | coreml | all (varsayılan)

Her aşama results/raw/reform_matrix.jsonl'a ANINDA yazılır (süreç ölürse iz
kalır). Gecikme ölçümleri benchmark harness'ı ile results/raw/reform_<form>.jsonl
dosyalarına da düşer. Başarısızlık da veridir — hata metni kaydedilir.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW = ROOT / "results" / "raw"
MATRIX = RAW / "reform_matrix.jsonl"
INPUT_SHAPE = (1, 384, 32, 32)  # 512² görüntüde stage-2 ara aktivasyonu


def log(form: str, stage: str, **data) -> None:
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "form": form,
           "stage": stage, "input": list(INPUT_SHAPE),
           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
           **data}
    with MATRIX.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{form}/{stage}] {data}", flush=True)


def build(form: str) -> torch.nn.Module:
    cap = torch.load(RAW / "reform_input.pt", weights_only=False)
    if form == "seq":
        from ss2d_seq import SS2DSeq
        m = SS2DSeq()
    elif form.startswith("blocked"):
        from ss2d_blocked import SS2DBlocked
        m = SS2DBlocked(block_size=int(form.removeprefix("blocked")), mode="decay")
    elif form.startswith("ane"):
        from ss2d_ane import SS2DAne
        m = SS2DAne(block_size=int(form.removeprefix("ane")))
    else:
        raise ValueError(form)
    m.load_state_dict(cap["op_state"], strict=True)
    return m.eval()


def ref_io() -> tuple[np.ndarray, np.ndarray]:
    cap = torch.load(RAW / "reform_input.pt", weights_only=False)
    return cap["x_op"].numpy(), cap["y_op"].numpy()


def stage_torch(form: str, model: torch.nn.Module) -> None:
    from benchmark import MeasureConfig, TorchRunner, run_benchmark

    x, _ = ref_io()
    cfg = MeasureConfig(warmup=5, runs=30, tags={"form": form})
    for device in ("cpu", "mps"):
        try:
            rec = run_benchmark(f"reform_{form}", TorchRunner(build(form), device), x, cfg)
            log(form, f"torch_{device}", ok=True,
                median_ms=rec["latency"]["median_ms"], p90_ms=rec["latency"].get("p90_ms"))
        except Exception as e:
            log(form, f"torch_{device}", ok=False, error=f"{type(e).__name__}: {e}"[:400])


def stage_onnx(form: str, model: torch.nn.Module) -> None:
    from benchmark import MeasureConfig, OrtRunner, run_benchmark

    x, y_ref = ref_io()
    path = RAW / f"reform_{form}.onnx"
    log(form, "onnx_export", status="started")
    t0 = time.perf_counter()
    try:
        torch.onnx.export(model, (torch.from_numpy(x),), str(path),
                          opset_version=17, dynamo=False)
        log(form, "onnx_export", ok=True, s=round(time.perf_counter() - t0, 1),
            size_mb=round(path.stat().st_size / 1e6, 2))
    except Exception as e:
        log(form, "onnx_export", ok=False, s=round(time.perf_counter() - t0, 1),
            error=f"{type(e).__name__}: {e}"[:400])
        return

    import onnx

    g = onnx.load(str(path), load_external_data=False).graph
    ops: dict[str, int] = {}
    for n in g.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    log(form, "onnx_graph", num_nodes=len(g.node),
        transpose=ops.get("Transpose", 0), reshape=ops.get("Reshape", 0),
        cumsum=ops.get("CumSum", 0), matmul=ops.get("MatMul", 0),
        top_ops=dict(sorted(ops.items(), key=lambda kv: -kv[1])[:8]))

    try:
        t0 = time.perf_counter()
        runner = OrtRunner(str(path))
        load_s = round(time.perf_counter() - t0, 2)
        runner.prepare(x)
        y = runner.run_once()[0]
        diff = float(np.abs(y - y_ref).max())
        rec = run_benchmark(f"reform_{form}", runner, x,
                            MeasureConfig(warmup=5, runs=30, tags={"form": form}))
        log(form, "ort_cpu", ok=True, load_s=load_s,
            median_ms=rec["latency"]["median_ms"], max_abs_diff_vs_ref=diff)
    except Exception as e:
        log(form, "ort_cpu", ok=False, error=f"{type(e).__name__}: {e}"[:400])


def stage_coreml(form: str, model: torch.nn.Module) -> None:
    import coremltools as ct

    from benchmark import CoreMLRunner, MeasureConfig, run_benchmark

    x, y_ref = ref_io()
    xt = torch.from_numpy(x)
    log(form, "coreml_trace", status="started")
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            traced = torch.jit.trace(model, xt)
        log(form, "coreml_trace", ok=True, s=round(time.perf_counter() - t0, 1))
    except Exception as e:
        log(form, "coreml_trace", ok=False, s=round(time.perf_counter() - t0, 1),
            error=f"{type(e).__name__}: {e}"[:400])
        return

    log(form, "coreml_convert", status="started")
    t0 = time.perf_counter()
    path = RAW / f"reform_{form}.mlpackage"
    try:
        mlm = ct.convert(traced, inputs=[ct.TensorType(shape=INPUT_SHAPE, dtype=float)],
                         minimum_deployment_target=ct.target.macOS15,
                         compute_units=ct.ComputeUnit.ALL)
        mlm.save(str(path))
        log(form, "coreml_convert", ok=True, s=round(time.perf_counter() - t0, 1))
    except Exception as e:
        log(form, "coreml_convert", ok=False, s=round(time.perf_counter() - t0, 1),
            error=f"{type(e).__name__}: {e}"[:400])
        return

    for cu in ("CPU_AND_GPU", "ALL"):
        try:
            runner = CoreMLRunner(str(path), compute_units=cu)
            runner.prepare(x)
            out = runner.run_once()
            y = np.asarray(next(iter(out.values())))
            diff = float(np.abs(y - y_ref).max())
            rec = run_benchmark(f"reform_{form}", runner, x,
                                MeasureConfig(warmup=5, runs=30, tags={"form": form}))
            log(form, f"coreml_{cu.lower()}", ok=True, load_s=runner.stats()["load_s"],
                median_ms=rec["latency"]["median_ms"], max_abs_diff_vs_ref=diff)
        except Exception as e:
            log(form, f"coreml_{cu.lower()}", ok=False, error=f"{type(e).__name__}: {e}"[:400])

    # op → cihaz dökümü (ct 8+ compute_plan API'si)
    try:
        from coremltools.models.compute_plan import MLComputePlan

        mlm = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        compiled = mlm.get_compiled_model_path()
        plan = MLComputePlan.load_from_path(path=compiled, compute_units=ct.ComputeUnit.ALL)
        program = plan.model_structure.program
        main_fn = program.functions["main"]
        counts = {"NeuralEngine": 0, "GPU": 0, "CPU": 0, "other": 0, "none": 0}
        total = 0
        for op in main_fn.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is None:
                counts["none"] += 1
            else:
                dev = type(usage.preferred_compute_device).__name__
                if "NeuralEngine" in dev:
                    counts["NeuralEngine"] += 1
                elif "GPU" in dev:
                    counts["GPU"] += 1
                elif "CPU" in dev:
                    counts["CPU"] += 1
                else:
                    counts["other"] += 1
            total += 1
        ane_pct = round(100 * counts["NeuralEngine"] / max(total - counts["none"], 1), 1)
        log(form, "coreml_compute_plan", ok=True, total_ops=total,
            device_counts=counts, ane_op_pct=ane_pct)
    except Exception as e:
        log(form, "coreml_compute_plan", ok=False, error=f"{type(e).__name__}: {e}"[:400])


def main() -> None:
    form = sys.argv[1]
    stage = sys.argv[2] if len(sys.argv) > 2 else "all"
    model = build(form)
    n_params = sum(p.numel() for p in model.parameters())
    log(form, "build", ok=True, params_m=round(n_params / 1e6, 3))

    if stage in ("torch", "all"):
        stage_torch(form, model)
    if stage in ("onnx", "all"):
        stage_onnx(form, model)
    if stage in ("coreml", "all"):
        stage_coreml(form, model)
    print("FORM-TAMAM", flush=True)


if __name__ == "__main__":
    main()
