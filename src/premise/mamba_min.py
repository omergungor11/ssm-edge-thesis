"""Öncül doğrulama mikrobenchmark'ı (TASK-004..006, Mac ayağı).

Minimal, saf-PyTorch bir Mamba benzeri model (selective scan dahil):
1. PyTorch eager gecikmesi (CPU + MPS)
2. ONNX export — graf boyutu, düğüm sayısı, op dökümü, yükleme süresi
3. ONNX Runtime CPU gecikmesi → yavaşlama oranı
4. CoreML dönüşüm denemesi (başarı/başarısızlık da veridir)

Ham sonuçlar: results/raw/premise_<tarih>.json
"""
from __future__ import annotations

import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

D_MODEL = 96
D_STATE = 16
N_LAYERS = 4
SEQ_LENS = [196, 1024]  # 14x14 patch (224px ViT-vari) ve yüksek çözünürlük senaryosu
WARMUP, RUNS = 10, 50


class SelectiveScan(nn.Module):
    """Saf-PyTorch ardışık selective scan: h_t = exp(dt*A) ⊙ h_{t-1} + dt*B_t x_t.

    Bilinçli olarak döngülü yazıldı — ONNX'in bu ardışıklığı nasıl temsil ettiği
    (Loop düğümü mü, devasa unroll mu) tezin ölçmek istediği şeyin ta kendisi.
    """

    def __init__(self, d: int, n: int):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(d, n) * 0.1)
        self.x_proj = nn.Linear(d, n * 2 + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, L, D)
        B, L, D = x.shape
        A = -torch.exp(self.A_log)  # (D, N)
        proj = self.x_proj(x)  # (B, L, 2N+1)
        dt = torch.nn.functional.softplus(proj[..., 0:1])  # (B, L, 1)
        Bm = proj[..., 1 : 1 + D_STATE]  # (B, L, N)
        Cm = proj[..., 1 + D_STATE :]  # (B, L, N)
        h = x.new_zeros(B, D, D_STATE)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)  # (B, D, N)
            dBx = dt[:, t].unsqueeze(-1) * Bm[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))  # (B, D)
        return torch.stack(ys, dim=1)  # (B, L, D)


class MambaBlock(nn.Module):
    def __init__(self, d: int, n: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.in_proj = nn.Linear(d, d * 2)
        self.conv = nn.Conv1d(d, d, 3, padding=2, groups=d)  # nedensel depthwise
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


def measure(fn, sync=lambda: None) -> dict:
    for _ in range(WARMUP):
        fn()
    sync()
    ts = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {
        "median_ms": round(statistics.median(ts), 3),
        "mean_ms": round(statistics.fmean(ts), 3),
        "std_ms": round(statistics.stdev(ts), 3),
        "p99_ms": round(ts[int(len(ts) * 0.99) - 1], 3),
    }


def main() -> None:
    torch.manual_seed(0)
    model = MiniMamba().eval()
    n_params = sum(p.numel() for p in model.parameters())
    out: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "model": {"d_model": D_MODEL, "d_state": D_STATE, "layers": N_LAYERS, "params": n_params},
        "runs": {},
    }
    print(f"MiniMamba: {n_params/1e3:.0f}K param, {N_LAYERS} katman")

    for L in SEQ_LENS:
        x = torch.randn(1, L, D_MODEL)
        rec: dict = {}

        with torch.no_grad():
            rec["torch_cpu"] = measure(lambda: model(x))
        print(f"L={L} torch CPU   : {rec['torch_cpu']['median_ms']} ms")

        if torch.backends.mps.is_available():
            m_mps, x_mps = model.to("mps"), x.to("mps")
            with torch.no_grad():
                rec["torch_mps"] = measure(lambda: m_mps(x_mps), torch.mps.synchronize)
            model.to("cpu")
            print(f"L={L} torch MPS   : {rec['torch_mps']['median_ms']} ms")

        # --- ONNX export (TorchScript trace yolu) ---
        onnx_path = RAW / f"mini_mamba_L{L}.onnx"
        t0 = time.perf_counter()
        try:
            torch.onnx.export(model, (x,), str(onnx_path), opset_version=17, dynamo=False)
            rec["onnx_export"] = {"ok": True, "export_s": round(time.perf_counter() - t0, 2)}
        except Exception as e:  # başarısızlık da veridir
            rec["onnx_export"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
            out["runs"][f"L{L}"] = rec
            print(f"L={L} ONNX export BAŞARISIZ: {type(e).__name__}")
            continue

        import onnx

        graph = onnx.load(str(onnx_path)).graph
        ops: dict[str, int] = {}
        for node in graph.node:
            ops[node.op_type] = ops.get(node.op_type, 0) + 1
        rec["onnx_graph"] = {
            "size_mb": round(onnx_path.stat().st_size / 1e6, 2),
            "num_nodes": len(graph.node),
            "top_ops": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:8]),
            "loop_nodes": ops.get("Loop", 0) + ops.get("Scan", 0),
        }
        print(f"L={L} ONNX graf   : {rec['onnx_graph']['num_nodes']} düğüm, "
              f"{rec['onnx_graph']['size_mb']} MB, Loop/Scan={rec['onnx_graph']['loop_nodes']}")

        import onnxruntime as ort

        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        rec["ort_load_s"] = round(time.perf_counter() - t0, 2)
        feed = {sess.get_inputs()[0].name: x.numpy()}
        rec["ort_cpu"] = measure(lambda: sess.run(None, feed))
        ratio = rec["ort_cpu"]["median_ms"] / rec["torch_cpu"]["median_ms"]
        rec["slowdown_ort_vs_torch_cpu"] = round(ratio, 2)
        print(f"L={L} ORT CPU     : {rec['ort_cpu']['median_ms']} ms "
              f"(yükleme {rec['ort_load_s']} s) → torch CPU'ya oran: {ratio:.2f}x")

        # --- CoreML dönüşüm denemesi ---
        try:
            import coremltools as ct

            traced = torch.jit.trace(model, x)
            t0 = time.perf_counter()
            mlm = ct.convert(
                traced,
                inputs=[ct.TensorType(shape=x.shape, dtype=float)],
                minimum_deployment_target=ct.target.macOS15,
                compute_units=ct.ComputeUnit.ALL,
            )
            conv_s = round(time.perf_counter() - t0, 2)
            mlpath = RAW / f"mini_mamba_L{L}.mlpackage"
            mlm.save(str(mlpath))
            pred_feed = {mlm.get_spec().description.input[0].name: x.numpy()}
            rec["coreml"] = {"ok": True, "convert_s": conv_s,
                             **{"latency_" + k: v for k, v in measure(lambda: mlm.predict(pred_feed)).items()}}
            print(f"L={L} CoreML      : dönüşüm {conv_s} s, "
                  f"medyan {rec['coreml']['latency_median_ms']} ms (ALL compute units)")
        except Exception as e:
            rec["coreml"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
            print(f"L={L} CoreML BAŞARISIZ: {type(e).__name__}: {str(e)[:200]}")

        out["runs"][f"L{L}"] = rec

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    path = RAW / f"premise_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nHam sonuç: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
