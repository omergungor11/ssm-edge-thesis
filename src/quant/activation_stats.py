"""TASK-027: Aktivasyon istatistikleri × çözünürlük (aykırı değer analizi).

Her modelde seçilmiş katman çıkışlarına forward-hook takar; N gerçek ADE20K
görüntüsüyle 256/512/768'de kanal-bazlı istatistik toplar:
maks |akt|, %99.9 yüzdelik, kurtosis, kanal-maks / kanal-medyan oranı (aykırılık).

OuroMamba/PTQ4VM bulgularıyla karşılaştırma tabanı: çözünürlük büyüdükçe SSM
aktivasyonlarının kuyruğu klasiklere göre nasıl davranıyor? (AS3)
Çıktı: results/raw/activation_stats_<model>.json
Kullanım: python activation_stats.py <model> [n_img=16]
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.ade20k import ADE_DIR, MEAN, STD

RAW = ROOT / "results" / "raw"
RESOLUTIONS = (256, 512, 768)


def get_model(name: str):
    from models import vmamba_upernet as vu

    vu.PSPModule.export_mode = True
    if name == "vmamba":
        from models.vmamba_upernet import load_pretrained
    elif name == "swin":
        from models.swin_upernet import load_pretrained
    else:
        from models.convnext_upernet import load_pretrained
    return load_pretrained()


def pick_layers(model, model_name: str) -> dict[str, nn.Module]:
    """Omurga boyunca ~8 temsilî nokta: her stage'den giriş/çıkış yakını modüller."""
    picks: dict[str, nn.Module] = {}
    for name, mod in model.backbone.named_modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and any(
            k in name for k in ("in_proj", "out_proj", "qkv", "proj", "pointwise_conv2",
                                 "reduction", "dwconv", "depthwise_conv")):
            picks[name] = mod
    # seyrelt: en fazla 10 katman, derinlik boyunca eşit aralıklı
    names = list(picks)
    step = max(1, len(names) // 10)
    return {n: picks[n] for n in names[::step][:10]}


def channel_stats(t: torch.Tensor) -> dict:
    x = t.detach().float().abs()
    if x.ndim == 4:      # (B,C,H,W)
        flat = x.permute(1, 0, 2, 3).reshape(x.shape[1], -1)
    elif x.ndim == 3:    # (B,L,C)
        flat = x.permute(2, 0, 1).reshape(x.shape[-1], -1)
    else:
        flat = x.reshape(1, -1)
    ch_max = flat.max(dim=1).values
    med = float(ch_max.median())
    all_v = flat.flatten()
    q999 = float(torch.quantile(all_v[:: max(1, all_v.numel() // 100000)], 0.999))
    m = all_v.mean(); s = all_v.std()
    kurt = float((((all_v - m) / (s + 1e-9)) ** 4).mean())
    return {"tensor_max": round(float(all_v.max()), 3), "p999": round(q999, 3),
            "kurtosis": round(kurt, 1),
            "chmax_over_median": round(float(ch_max.max()) / (med + 1e-9), 1)}


def main() -> None:
    model_name = sys.argv[1]
    n_img = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    model = get_model(model_name)
    layers = pick_layers(model, model_name)
    print(f"{model_name}: {len(layers)} katman izleniyor", flush=True)

    ids = sorted(p.stem for p in (ADE_DIR / "images" / "validation").glob("*.jpg"))[:n_img]
    out: dict = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": model_name,
                 "n_img": n_img, "layers": list(layers), "stats": {}}

    for res in RESOLUTIONS:
        acc: dict[str, list[dict]] = {n: [] for n in layers}
        hooks = []
        for lname, mod in layers.items():
            hooks.append(mod.register_forward_hook(
                lambda m, i, o, ln=lname: acc[ln].append(channel_stats(o))))
        with torch.no_grad():
            for iid in ids:
                img = Image.open(ADE_DIR / "images" / "validation" / f"{iid}.jpg") \
                    .convert("RGB").resize((res, res), Image.BILINEAR)
                x = ((np.asarray(img, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
                model(torch.from_numpy(x))
        for h in hooks:
            h.remove()
        out["stats"][str(res)] = {
            ln: {k: round(float(np.mean([s[k] for s in v])), 3) for k in v[0]}
            for ln, v in acc.items() if v}
        worst = max(out["stats"][str(res)].items(), key=lambda kv: kv[1]["chmax_over_median"])
        print(f"  {res}px: en aykırı katman {worst[0]} → chmax/med {worst[1]['chmax_over_median']}, "
              f"kurtosis {worst[1]['kurtosis']}", flush=True)

    (RAW / f"activation_stats_{model_name}.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("TAMAM", flush=True)


if __name__ == "__main__":
    main()
