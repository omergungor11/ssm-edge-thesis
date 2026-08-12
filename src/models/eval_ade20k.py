"""ADE20K val mIoU doğrulaması (TASK-014).

Kullanım: python eval_ade20k.py [N]   (N: alt-küme boyutu, varsayılan 250)

Protokol: en-boy korumalı yeniden boyutlandırma (kısa kenar 512), /32'ye
yansıma pad'i, logit'ler pad'siz bölgeden alınır. mmseg 'whole' moduna yakın;
kare-resize sapması yok. Bildirilen değer (VMamba-T v2seg): mIoU ~48.
İlerleme her 25 görüntüde results/raw/ade20k_miou_progress.json'a yazılır.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.ade20k import ADE_DIR, IGNORE_INDEX, MEAN, NUM_CLASSES, STD
from models.vmamba_upernet import load_pretrained

RAW = ROOT / "results" / "raw"
SHORT = 512
LONG_MAX = 2048


def prep(img: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
    w, h = img.size
    scale = min(SHORT / min(w, h), LONG_MAX / max(w, h))
    nw, nh = round(w * scale), round(h * scale)
    img = img.resize((nw, nh), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)
    t = torch.from_numpy(x)[None]
    ph, pw = (32 - nh % 32) % 32, (32 - nw % 32) % 32
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    return t, (nh, nw)


def main() -> None:
    n_subset = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    model = load_pretrained()
    img_dir = ADE_DIR / "images" / "validation"
    ann_dir = ADE_DIR / "annotations" / "validation"
    ids = sorted(p.stem for p in img_dir.glob("*.jpg"))[:n_subset]

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    t_start = time.perf_counter()
    for i, iid in enumerate(ids):
        img = Image.open(img_dir / f"{iid}.jpg").convert("RGB")
        ann = Image.open(ann_dir / f"{iid}.png")
        x, (nh, nw) = prep(img)
        with torch.no_grad():
            logits = model(x)[:, :, :nh, :nw]
        pred = logits.argmax(1)[0].numpy().astype(np.int16)
        y = np.asarray(ann.resize((nw, nh), Image.NEAREST), dtype=np.int16)
        y = np.where(y == 0, IGNORE_INDEX, y - 1)
        valid = y != IGNORE_INDEX
        conf += np.bincount(
            (y[valid] * NUM_CLASSES + pred[valid]).ravel(), minlength=NUM_CLASSES**2
        ).reshape(NUM_CLASSES, NUM_CLASSES)

        if (i + 1) % 25 == 0 or i + 1 == len(ids):
            inter = np.diag(conf).astype(np.float64)
            union = conf.sum(0) + conf.sum(1) - np.diag(conf)
            iou = inter[union > 0] / union[union > 0]
            miou = float(iou.mean()) * 100
            acc = float(inter.sum() / conf.sum()) * 100
            el = time.perf_counter() - t_start
            prog = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "done": i + 1, "total": len(ids),
                "mIoU": round(miou, 2), "aAcc": round(acc, 2),
                "sec_per_img": round(el / (i + 1), 2),
                "classes_seen": int((union > 0).sum()),
            }
            (RAW / "ade20k_miou_progress.json").write_text(json.dumps(prog, indent=2))
            print(f"{i+1}/{len(ids)}  mIoU={miou:.2f}  aAcc={acc:.2f}  "
                  f"({el/(i+1):.1f} s/img)", flush=True)

    np.save(RAW / f"ade20k_conf_n{len(ids)}.npy", conf)
    print("TAMAM", flush=True)


if __name__ == "__main__":
    main()
