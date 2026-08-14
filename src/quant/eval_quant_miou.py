"""TASK-026: Nicemlenmiş modellerin mIoU etkisi (backend-parametrik değerlendirici).

Kullanım:
  python eval_quant_miou.py --backend coreml --path results/raw/convnext_upernet_512_w8_linear.mlpackage --tag convnext_w8 [--n 250]
  python eval_quant_miou.py --backend ort --path results/raw/swin_upernet_512_int8.onnx --tag swin_int8 [--n 250]

Protokol farkı (dürüstlük notu): export edilen modeller SABİT 512×512 girdi alır;
bu değerlendirici bu yüzden kare-512 resize kullanır — torch değerlendiricisinin
en-boy korumalı protokolünden farklıdır. Karşılaştırma tabanı olarak aynı betik
fp32 export'la da koşulmalıdır (fp32-export vs quant-export farkı = nicemleme etkisi;
torch-fp32'ye karşı değil). Çıktı: results/raw/quant_miou_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from data.ade20k import ADE_DIR, IGNORE_INDEX, MEAN, NUM_CLASSES, STD

RAW = ROOT / "results" / "raw"
SIZE = 512


class CoreMLBackend:
    def __init__(self, path: str):
        import coremltools as ct

        self.m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
        spec = self.m.get_spec()
        self.in_name = spec.description.input[0].name
        self.out_name = spec.description.output[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.m.predict({self.in_name: x})[self.out_name])


class OrtBackend:
    def __init__(self, path: str):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.in_name: x})[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("coreml", "ort"), required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=250)
    args = ap.parse_args()

    model = CoreMLBackend(args.path) if args.backend == "coreml" else OrtBackend(args.path)
    img_dir = ADE_DIR / "images" / "validation"
    ann_dir = ADE_DIR / "annotations" / "validation"
    ids = sorted(p.stem for p in img_dir.glob("*.jpg"))[: args.n]

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    t0 = time.perf_counter()
    for i, iid in enumerate(ids):
        img = Image.open(img_dir / f"{iid}.jpg").convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)
        x = ((np.asarray(img, np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        pred = model(x.astype(np.float32)).argmax(1)[0].astype(np.int16)
        y = np.asarray(Image.open(ann_dir / f"{iid}.png").resize((SIZE, SIZE), Image.NEAREST), np.int16)
        y = np.where(y == 0, IGNORE_INDEX, y - 1)
        v = y != IGNORE_INDEX
        conf += np.bincount((y[v] * NUM_CLASSES + pred[v]).ravel(),
                            minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)
        if (i + 1) % 50 == 0 or i + 1 == len(ids):
            inter = np.diag(conf).astype(np.float64)
            union = conf.sum(0) + conf.sum(1) - np.diag(conf)
            miou = float((inter[union > 0] / union[union > 0]).mean()) * 100
            print(f"{i+1}/{len(ids)} mIoU={miou:.2f} ({(time.perf_counter()-t0)/(i+1):.1f} s/img)",
                  flush=True)

    inter = np.diag(conf).astype(np.float64)
    union = conf.sum(0) + conf.sum(1) - np.diag(conf)
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag, "backend": args.backend, "path": args.path, "n": len(ids),
        "protocol": "square-512 (export protokolü; en-boy korumalı DEĞİL)",
        "mIoU": round(float((inter[union > 0] / union[union > 0]).mean()) * 100, 2),
        "aAcc": round(float(inter.sum() / conf.sum()) * 100, 2),
    }
    (RAW / f"quant_miou_{args.tag}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("TAMAM:", out["tag"], out["mIoU"], flush=True)


if __name__ == "__main__":
    main()
