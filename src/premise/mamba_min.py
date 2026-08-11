"""Öncül doğrulama mikrobenchmark'ı (TASK-004..006, Mac ayağı).

Kullanım: python mamba_min.py <seq_len>
Her aşama sonrası sonuç ANINDA diske yazılır (results/raw/premise_L<L>.json) —
bir aşama asılı kalırsa öncekiler kaybolmaz. İlk koşuda L=1024'ün ORT/CoreML
aşaması >1 saat %0 CPU ile takıldı; bu sürüm aşama bazlı ilerleme kaydı tutar.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

D_MODEL, D_STATE, N_LAYERS = 96, 16, 4
WARMUP, RUNS = 10, 50
ORT_OPT = os.environ.get("ORT_OPT", "all")  # all|basic|none — hang teşhisi için


class SelectiveScan(nn.Module):
    """Saf-PyTorch ardışık selective scan. Bilinçli döngülü — ONNX'in bu
    ardışıklığı nasıl temsil ettiği (Loop mu, unroll mu) ölçülen şeyin kendisi."""

    def __init__(self, d: int, n: int):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(d, n) * 0.1)
        self.x_proj = nn.Linear(d, n * 2 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        B, L, D = x.shape
        A = -torch.exp(self.A_log)
        proj = self.x_proj(x)
        dt = torch.nn.functional.softplus(proj[..., 0:1])
        Bm = proj[..., 1 : 1 + D_STATE]
        Cm = proj[..., 1 + D_STATE :]
        h = x.new_zeros(B, D, D_STATE)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            dBx = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    def __init__(self, d: int, n: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, d * 2)
        self.conv = nn.Conv1d(d, d, 3, padding=2, groups=d)
        self.scan = SelectiveScan(d, n)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        x = torch.nn.functional.silu(x)
        x = self.scan(x)
        x = x * torch.nn.functional.silu(gate)
        return self.out_proj(x) + r


class MiniMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(MambaBlock(D_MODEL, D_STATE) for _ in range(N_LAYERS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x


def measure(fn, sync=lambda: None, runs: int = RUNS) -> dict:
    for _ in range(WARMUP):
        fn()
    sync()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {
        "median_ms": round(statistics.median(ts), 3),
        "mean_ms": round(statistics.fmean(ts), 3),
        "std_ms": round(statistics.stdev(ts), 3),
        "p99_ms": round(ts[max(0, int(len(ts) * 0.99) - 1)], 3),
        "runs": runs,
    }


def main() -> None:
    L = int(sys.argv[1])
    out_path = RAW / f"premise_L{L}.json"
    rec: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": {"platform": platform.platform(), "python": platform.python_version(),
                 "torch": torch.__version__},
        "model": {"d_model": D_MODEL, "d_state": D_STATE, "layers": N_LAYERS},
        "seq_len": L,
        "ort_opt_level": ORT_OPT,
        "stages": {},
    }

    def save(stage: str, data) -> None:
        rec["stages"][stage] = data
        out_path.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
        print(f"[{stage}] kaydedildi", flush=True)

    torch.manual_seed(0)
    model = MiniMamba().eval()
    rec["model"]["params"] = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, L, D_MODEL)

    # 1) torch CPU
    with torch.no_grad():
        r = measure(lambda: model(x))
    save("torch_cpu", r)
    print(f"torch CPU: {r['median_ms']} ms", flush=True)

    # 2) torch MPS
    if torch.backends.mps.is_available():
        m_mps, x_mps = model.to("mps"), x.to("mps")
        with torch.no_grad():
            r = measure(lambda: m_mps(x_mps), torch.mps.synchronize)
        model.to("cpu")
        save("torch_mps", r)
        print(f"torch MPS: {r['median_ms']} ms", flush=True)

    # 3) ONNX export
    onnx_path = RAW / f"mini_mamba_L{L}.onnx"
    t0 = time.perf_counter()
    try:
        torch.onnx.export(model, (x,), str(onnx_path), opset_version=17, dynamo=False)
        save("onnx_export", {"ok": True, "export_s": round(time.perf_counter() - t0, 2)})
    except Exception as e:
        save("onnx_export", {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]})
        return

    import onnx

    graph = onnx.load(str(onnx_path)).graph
    ops: dict[str, int] = {}
    for node in graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    save("onnx_graph", {
        "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "num_nodes": len(graph.node),
        "top_ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:8]),
        "loop_nodes": ops.get("Loop", 0) + ops.get("Scan", 0),
    })
    print(f"ONNX graf: {len(graph.node)} düğüm", flush=True)

    # 4) ORT CPU — yükleme süresi dahil (ilk koşuda burada hang şüphesi)
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }[ORT_OPT]
    save("ort_load", {"status": "started"})  # asılı kalırsa iz bırak
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    load_s = round(time.perf_counter() - t0, 2)
    save("ort_load", {"status": "ok", "load_s": load_s})
    print(f"ORT yükleme: {load_s} s (opt={ORT_OPT})", flush=True)

    feed = {sess.get_inputs()[0].name: x.numpy()}
    r = measure(lambda: sess.run(None, feed))
    r["slowdown_vs_torch_cpu"] = round(r["median_ms"] / rec["stages"]["torch_cpu"]["median_ms"], 2)
    save("ort_cpu", r)
    print(f"ORT CPU: {r['median_ms']} ms → oran {r['slowdown_vs_torch_cpu']}x", flush=True)

    # 5) CoreML
    try:
        import coremltools as ct

        traced = torch.jit.trace(model, x)
        save("coreml", {"status": "converting"})  # asılı kalırsa iz bırak
        t0 = time.perf_counter()
        mlm = ct.convert(traced, inputs=[ct.TensorType(shape=x.shape, dtype=float)],
                         minimum_deployment_target=ct.target.macOS15,
                         compute_units=ct.ComputeUnit.ALL)
        conv_s = round(time.perf_counter() - t0, 2)
        mlm.save(str(RAW / f"mini_mamba_L{L}.mlpackage"))
        feed_ml = {mlm.get_spec().description.input[0].name: x.numpy()}
        r = measure(lambda: mlm.predict(feed_ml))
        r["convert_s"] = conv_s
        r["slowdown_vs_torch_cpu"] = round(r["median_ms"] / rec["stages"]["torch_cpu"]["median_ms"], 2)
        save("coreml", {"status": "ok", **r})
        print(f"CoreML: dönüşüm {conv_s} s, medyan {r['median_ms']} ms", flush=True)
    except Exception as e:
        save("coreml", {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]})

    print("TAMAM", flush=True)


if __name__ == "__main__":
    main()
