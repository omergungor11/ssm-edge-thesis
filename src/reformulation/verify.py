"""TASK-030 doğrulama — üç formun gerçek SS2D çıktısıyla karşılaştırılması.

Referans: capture.py'nin kaydettiği gerçek ara-aktivasyon girdisi + üçüncü-parti
SS2D çıktısı (results/raw/reform_input.pt). Her form aynı ağırlıklarla yüklenir,
max mutlak fark + kanal başına göreli hata raporlanır ve reform_matrix.jsonl'a
'verify' aşaması olarak yazılır.

Kullanım: python verify.py [seq|blocked|ane|all]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW = ROOT / "results" / "raw"
MATRIX = RAW / "reform_matrix.jsonl"


def log(form: str, stage: str, **data) -> None:
    rec = {"timestamp": datetime.now(timezone.utc).isoformat(),
           "form": form, "stage": stage, **data}
    with MATRIX.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[{form}/{stage}] {data}", flush=True)


def compare(y: torch.Tensor, ref: torch.Tensor) -> dict:
    diff = (y - ref).abs()
    per_ch_max = diff.amax(dim=(0, 2, 3))                      # (C,)
    per_ch_scale = ref.abs().amax(dim=(0, 2, 3)).clamp_min(1e-8)
    rel = per_ch_max / per_ch_scale
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "per_channel_rel_max": float(rel.max()),
        "per_channel_rel_median": float(rel.median()),
    }


def stability_stats(model, x: torch.Tensor, P: int) -> dict:
    """Gerçek aktivasyonlarda blok-içi log-cumsum aralığı (fp32 güvenlik payı)."""
    with torch.no_grad():
        h = F.silu(model.conv2d(F.conv2d(x, model.in_proj.weight[:, :, None, None])))
        from common import cross_scan, K_GROUP, D_INNER, DT_RANK
        B, _, H, W = x.shape
        L = H * W
        u = cross_scan(h).reshape(B, K_GROUP * D_INNER, L)
        x_dbl = F.conv1d(u, model.x_proj_weight.view(-1, D_INNER, 1), groups=K_GROUP)
        dts = x_dbl.view(B, K_GROUP, DT_RANK + 2, L)[:, :, :DT_RANK]
        dts = F.conv1d(dts.reshape(B, -1, L),
                       model.dt_projs_weight.reshape(K_GROUP * D_INNER, DT_RANK, 1),
                       groups=K_GROUP)
        delta = F.softplus(dts + model.dt_projs_bias.reshape(1, -1, 1))
        a_log = delta * (-torch.exp(model.A_logs.reshape(1, -1, 1)))
        S = torch.cumsum(a_log.view(B, -1, L // P, P), dim=-1)
    import math
    min_s = float(S.min())
    try:
        e = math.exp(-min_s)
    except OverflowError:
        e = float("inf")
    return {"P": P, "min_S": round(min_s, 2), "exp_neg_S_max": e,
            "fp32_overflow": -min_s > math.log(torch.finfo(torch.float32).max)}


def build(form: str, state: dict, P: int = 128):
    if form == "seq":
        from ss2d_seq import SS2DSeq
        m = SS2DSeq()
    elif form == "blocked":
        from ss2d_blocked import SS2DBlocked
        m = SS2DBlocked(block_size=P)
    elif form == "ane":
        from ss2d_ane import SS2DAne
        m = SS2DAne(block_size=P)
    else:
        raise ValueError(form)
    missing, unexpected = m.load_state_dict(state, strict=True), None
    return m.eval()


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cap = torch.load(RAW / "reform_input.pt", weights_only=False)
    x, y_ref, state = cap["x_op"], cap["y_op"], cap["op_state"]

    if which in ("seq", "all"):
        m = build("seq", state)
        with torch.no_grad():
            y = m(x)
        log("seq", "verify", **compare(y, y_ref), target="<1e-4 (uccuncu-parti ile)")

    if which in ("blocked", "all"):
        for P in (64, 128):
            for mode in ("cumsum", "decay"):
                m = build("blocked", state, P)
                m.mode = mode
                with torch.no_grad():
                    y = m(x)
                stab = stability_stats(m, x, P)
                log(f"blocked{P}_{mode}", "verify", **compare(y, y_ref), **stab,
                    target="<1e-3")

    if which in ("ane", "all"):
        for P in (64, 128):
            m = build("ane", state, P)
            with torch.no_grad():
                y = m(x)
            log(f"ane{P}", "verify", **compare(y, y_ref), target="<1e-3")


if __name__ == "__main__":
    main()
