"""VMamba-T + UPerNet (ADE20K) — mmseg'siz, saf-torch yükleyici (TASK-014).

mmseg checkpoint'i anahtar-uyumlu modüllerle birebir yüklenir (missing=0 hedefi).
Backbone: third_party/VMamba (2 CPU yaması uygulanmış — bkz. tez-docs/vmamba-yamalari.md).
Baş: mmseg UPerHead'in inference-eşdeğeri (aux başlık eğitim içindir, atlanır).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
# NOT: 'classification' değil, içindeki 'models' dizini eklenir — aksi hâlde
# third_party'nin 'models' paketi bizim src/models ile çakışıyor.
sys.path.insert(0, str(ROOT / "third_party" / "VMamba" / "classification" / "models"))

CKPT = ROOT / "checkpoints" / "upernet_vssm_ade20k_tiny_iter160k.pth"
NUM_CLASSES = 150


class ConvBNReLU(nn.Module):
    """mmseg ConvModule (conv + bn + relu) anahtar düzeniyle: .conv, .bn"""

    def __init__(self, cin: int, cout: int, k: int, pad: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# Export modunda tüm shape okumalarını ortadan kaldıran statik girdi boyutu.
# torch.jit.trace, girdiye bağlı int(x.shape[i]) okumalarını aten::Int düğümüne
# çevirir ve coremltools tam-model grafında bunlardan birinde çöker (TypeError).
# Girdi boyutu export'ta sabit olduğundan boyutlar buradan türetilir.
EXPORT_INPUT_SIZE: int | None = None


def _adaptive_avg_pool_static(x: torch.Tensor, s: int) -> torch.Tensor:
    """adaptive_avg_pool2d'nin statik-dilimli eşdeğeri (sabit H,W için birebir aynı).

    Export modunda kullanılır: adaptive_avg_pool2d, çıktı girdiyi tam bölmediğinde
    (16→3, 16→6) ne ONNX TorchScript exporter'ı ne CoreML tarafından destekleniyor.
    PyTorch tanımıyla aynı pencere sınırları: start=floor(i*H/s), end=ceil((i+1)*H/s).
    """
    if EXPORT_INPUT_SIZE:
        H = W = EXPORT_INPUT_SIZE // 32  # PSP yalnızca en derin ölçekte çalışır
    else:
        H, W = int(x.shape[2]), int(x.shape[3])
    rows = []
    for i in range(s):
        h0, h1 = (i * H) // s, -((-(i + 1) * H) // s)
        cols = []
        for j in range(s):
            w0, w1 = (j * W) // s, -((-(j + 1) * W) // s)
            cols.append(x[:, :, h0:h1, w0:w1].mean(dim=(2, 3), keepdim=True))
        rows.append(torch.cat(cols, dim=3))
    return torch.cat(rows, dim=2)


class PSPModule(nn.Module):
    """mmseg PPM hücresi: AdaptiveAvgPool + ConvModule → anahtar '{i}.1.*'

    export_mode=True iken statik-dilimli pooling kullanılır (numerik eşdeğer).
    """

    export_mode: bool = False  # sınıf düzeyinde global anahtar

    def __init__(self, scale: int, cin: int, cout: int):
        super().__init__()
        self.scale = scale
        self.add_module("0", nn.AdaptiveAvgPool2d(scale))
        self.add_module("1", ConvBNReLU(cin, cout, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if PSPModule.export_mode:
            p = _adaptive_avg_pool_static(x, self.scale)
        else:
            p = getattr(self, "0")(x)
        y = getattr(self, "1")(p)
        if EXPORT_INPUT_SIZE:
            tgt = (EXPORT_INPUT_SIZE // 32, EXPORT_INPUT_SIZE // 32)
        else:
            tgt = (int(x.shape[2]), int(x.shape[3]))
        return F.interpolate(y, size=tgt, mode="bilinear", align_corners=False)


class UPerHead(nn.Module):
    def __init__(self, in_channels=(96, 192, 384, 768), channels=512, pool_scales=(1, 2, 3, 6)):
        super().__init__()
        cin_last = in_channels[-1]
        self.psp_modules = nn.ModuleList(PSPModule(s, cin_last, channels) for s in pool_scales)
        self.bottleneck = ConvBNReLU(cin_last + len(pool_scales) * channels, channels, 3, pad=1)
        self.lateral_convs = nn.ModuleList(ConvBNReLU(c, channels, 1) for c in in_channels[:-1])
        self.fpn_convs = nn.ModuleList(ConvBNReLU(channels, channels, 3, pad=1) for _ in in_channels[:-1])
        self.fpn_bottleneck = ConvBNReLU(len(in_channels) * channels, channels, 3, pad=1)
        self.conv_seg = nn.Conv2d(channels, NUM_CLASSES, 1)

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        psp = torch.cat([feats[-1]] + [m(feats[-1]) for m in self.psp_modules], dim=1)
        laterals = [conv(feats[i]) for i, conv in enumerate(self.lateral_convs)]
        laterals.append(self.bottleneck(psp))
        for i in range(len(laterals) - 1, 0, -1):
            if EXPORT_INPUT_SIZE:
                e = EXPORT_INPUT_SIZE // (4 * 2 ** (i - 1))
                tgt = (e, e)
            else:
                tgt = (int(laterals[i - 1].shape[2]), int(laterals[i - 1].shape[3]))
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=tgt, mode="bilinear", align_corners=False)
        outs = [self.fpn_convs[i](laterals[i]) for i in range(len(laterals) - 1)] + [laterals[-1]]
        if EXPORT_INPUT_SIZE:
            size0 = (EXPORT_INPUT_SIZE // 4, EXPORT_INPUT_SIZE // 4)
        else:
            size0 = (int(outs[0].shape[2]), int(outs[0].shape[3]))
        outs = [outs[0]] + [F.interpolate(o, size=size0, mode="bilinear", align_corners=False)
                            for o in outs[1:]]
        return self.conv_seg(self.fpn_bottleneck(torch.cat(outs, dim=1)))


class VMambaUPerNet(nn.Module):
    def __init__(self):
        super().__init__()
        import vmamba  # third_party (yamalı) — sys.path üzerinden

        self.backbone = vmamba.Backbone_VSSM(
            out_indices=(0, 1, 2, 3), pretrained=None, dims=96, depths=(2, 2, 5, 2),
            ssm_d_state=1, ssm_dt_rank="auto", ssm_ratio=2.0, ssm_conv=3,
            ssm_conv_bias=False, forward_type="v05_noz", mlp_ratio=4.0,
            downsample_version="v3", patchembed_version="v2", drop_path_rate=0.0,
            norm_layer="ln2d",
        )
        self.decode_head = UPerHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.decode_head(self.backbone(x))
        if EXPORT_INPUT_SIZE:
            tgt = (EXPORT_INPUT_SIZE, EXPORT_INPUT_SIZE)
        else:
            tgt = (int(x.shape[2]), int(x.shape[3]))
        return F.interpolate(logits, size=tgt, mode="bilinear", align_corners=False)


def load_pretrained(ckpt_path: Path = CKPT) -> VMambaUPerNet:
    model = VMambaUPerNet()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k: v for k, v in sd.items() if not k.startswith("auxiliary_head.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"eksik anahtar: {missing[:5]}"
    assert not unexpected, f"fazla anahtar: {unexpected[:5]}"
    return model.eval()


if __name__ == "__main__":
    import time

    model = load_pretrained()
    n = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, 512, 512)
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    print(f"param: {n/1e6:.1f}M | cikti: {tuple(y.shape)} | ileri gecis {time.perf_counter()-t0:.2f} s")
    assert y.shape == (1, NUM_CLASSES, 512, 512)
    print("OK — checkpoint eksiksiz yuklendi")
