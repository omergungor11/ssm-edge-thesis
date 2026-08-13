"""Export matrisi — tek hücre (TASK-020/021).

Kullanım: python export_cell.py <model> <format> [boyut=512]
  model : vmamba | swin | convnext
  format: onnx | coreml

Her aşama results/raw/export_matrix.jsonl'a ANINDA yazılır (başarısızlık/asılma
durumunda iz kalır). Süre, graf boyutu, düğüm sayısı, hata metni — hepsi veridir.
Tam gecikme ölçümü TASK-021'in işi; burada yalnızca export edilebilirlik +
yükleme/dönüşüm maliyeti ölçülür.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "results" / "raw"
OUT = RAW / "export_matrix.jsonl"
SIZE = 512  # argv[3] ile değişir (çözünürlük taraması)


def log(model: str, fmt: str, stage: str, **data) -> None:
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": model,
           "format": fmt, "stage": stage, "input": [1, 3, SIZE, SIZE],
           "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
           **data}
    with OUT.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{model}/{fmt}/{stage}] {data}", flush=True)


def load_model(name: str):
    if name == "vmamba":
        from models.vmamba_upernet import load_pretrained
    elif name == "swin":
        from models.swin_upernet import load_pretrained
    else:
        from models.convnext_upernet import load_pretrained
    return load_pretrained()


def main() -> None:
    global SIZE
    model_name, fmt = sys.argv[1], sys.argv[2]
    if len(sys.argv) > 3:
        SIZE = int(sys.argv[3])
    x = torch.randn(1, 3, SIZE, SIZE)

    t0 = time.perf_counter()
    model = load_model(model_name)
    log(model_name, fmt, "model_load", ok=True, s=round(time.perf_counter() - t0, 1))

    # Export-dostu mod: statik PSP + shape-okumasız boyutlar; numerik eşdeğerlik kanıtı
    from models import vmamba_upernet as vu

    with torch.no_grad():
        ref = model(x)
        vu.PSPModule.export_mode = True
        vu.EXPORT_INPUT_SIZE = SIZE
        alt = model(x)
    max_diff = float((ref - alt).abs().max())
    log(model_name, fmt, "psp_export_mode", max_abs_diff=round(max_diff, 8),
        ok=max_diff < 1e-4)

    if fmt == "onnx":
        path = RAW / f"{model_name}_upernet_{SIZE}.onnx"
        log(model_name, fmt, "export", status="started")
        t0 = time.perf_counter()
        try:
            torch.onnx.export(model, (x,), str(path), opset_version=17, dynamo=False)
            log(model_name, fmt, "export", ok=True, s=round(time.perf_counter() - t0, 1),
                size_mb=round(path.stat().st_size / 1e6, 1))
        except Exception as e:
            log(model_name, fmt, "export", ok=False, s=round(time.perf_counter() - t0, 1),
                error=f"{type(e).__name__}: {e}"[:400])
            return
        import onnx

        g = onnx.load(str(path), load_external_data=False).graph
        ops: dict[str, int] = {}
        for n in g.node:
            ops[n.op_type] = ops.get(n.op_type, 0) + 1
        log(model_name, fmt, "graph", num_nodes=len(g.node),
            loop_scan=ops.get("Loop", 0) + ops.get("Scan", 0),
            top_ops=dict(sorted(ops.items(), key=lambda kv: -kv[1])[:6]))

        import onnxruntime as ort

        log(model_name, fmt, "ort_load", status="started")
        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        log(model_name, fmt, "ort_load", ok=True, s=round(time.perf_counter() - t0, 1))
        feed = {sess.get_inputs()[0].name: x.numpy()}
        t0 = time.perf_counter()
        sess.run(None, feed)
        log(model_name, fmt, "ort_first_run", ok=True, s=round(time.perf_counter() - t0, 2))

    elif fmt == "coreml":
        import coremltools as ct

        log(model_name, fmt, "trace", status="started")
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, x)
            log(model_name, fmt, "trace", ok=True, s=round(time.perf_counter() - t0, 1))
        except Exception as e:
            log(model_name, fmt, "trace", ok=False, s=round(time.perf_counter() - t0, 1),
                error=f"{type(e).__name__}: {e}"[:400])
            return
        log(model_name, fmt, "convert", status="started")
        t0 = time.perf_counter()
        try:
            mlm = ct.convert(traced, inputs=[ct.TensorType(shape=x.shape, dtype=float)],
                             minimum_deployment_target=ct.target.macOS15,
                             compute_units=ct.ComputeUnit.ALL)
            path = RAW / f"{model_name}_upernet_{SIZE}.mlpackage"
            mlm.save(str(path))
            log(model_name, fmt, "convert", ok=True, s=round(time.perf_counter() - t0, 1))
        except Exception as e:
            log(model_name, fmt, "convert", ok=False, s=round(time.perf_counter() - t0, 1),
                error=f"{type(e).__name__}: {e}"[:400])
            return
        t0 = time.perf_counter()
        mlm.predict({mlm.get_spec().description.input[0].name: x.numpy()})
        log(model_name, fmt, "first_predict", ok=True, s=round(time.perf_counter() - t0, 2))

    print("HUCRE-TAMAM", flush=True)


if __name__ == "__main__":
    main()
