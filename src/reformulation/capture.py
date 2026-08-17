"""TASK-030 Adım 1 — Gerçek SS2D bloğu girdi/çıktı yakalama.

VMamba-T + UPerNet modelini yükler, gerçek bir ADE20K 512² görüntüsünü ileri
geçirir ve `backbone.layers[2].blocks[0].op` (SS2D) modülünün girdi/çıktısını
forward hook ile yakalar. Tekrarlanabilirlik için girdi + referans çıktı +
op ağırlıkları `results/raw/reform_input.pt` dosyasına yazılır.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "raw" / "reform_input.pt"


def main() -> None:
    from data.ade20k import ADE20KVal
    from models.vmamba_upernet import load_pretrained

    model = load_pretrained()
    op = model.backbone.layers[2].blocks[0].op
    print("op tipi:", type(op).__name__)
    print("op alt modüller:", [n for n, _ in op.named_children()])
    print("op parametreler:", {n: tuple(p.shape) for n, p in op.named_parameters()})
    print("d_model:", op.d_model, "| d_inner:", op.d_inner, "| d_state:", op.d_state,
          "| dt_rank:", op.dt_rank, "| k_group:", op.k_group,
          "| channel_first:", op.channel_first, "| disable_z:", op.disable_z)

    ds = ADE20KVal(512)
    img, _ = ds[0]
    x = torch.from_numpy(img)[None]  # (1, 3, 512, 512)

    captured: dict = {}

    def hook(_mod, args, output):
        captured["x_op"] = args[0].detach().clone()
        captured["y_op"] = output.detach().clone()

    h = op.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    h.remove()

    x_op, y_op = captured["x_op"], captured["y_op"]
    print("op girdi:", tuple(x_op.shape), x_op.dtype, "| op çıktı:", tuple(y_op.shape))

    torch.save({
        "image_id": ds.ids[0],
        "x_op": x_op,
        "y_op": y_op,
        "op_state": {k: v.detach().clone() for k, v in op.state_dict().items()},
        "meta": {
            "d_model": op.d_model, "d_inner": op.d_inner, "d_state": op.d_state,
            "dt_rank": op.dt_rank, "k_group": op.k_group,
            "channel_first": op.channel_first, "module_path": "backbone.layers[2].blocks[0].op",
            "forward_type": "v05_noz", "input_image_size": 512,
        },
    }, OUT)
    print("kaydedildi:", OUT)


if __name__ == "__main__":
    main()
