"""ConvNeXt-T + UPerNet (ADE20K) — mmseg'siz, saf-torch yükleyici (TASK-016).

Backbone: mmcls.ConvNeXt anahtar düzeni (downsample_layers, stages.X.Y,
norm0..3). layer_scale (gamma) aktif, LN eps=1e-6, gap_before_final_norm=False.
Bildirilen: mIoU 46.11 — DİKKAT: bildirilen değer 'slide' (512 kayan pencere)
test modu iledir; bu tezin protokolü tüm omurgalarda 'whole' moddur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from models.vmamba_upernet import NUM_CLASSES, UPerHead  # noqa: E402
CKPT = ROOT / "checkpoints" / "upernet_convnext_tiny_ade20k.pth"


class LayerNorm2d(nn.LayerNorm):
    """Kanal-önce (NCHW) LayerNorm — mmcls LN2d eşdeğeri."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x.permute(0, 2, 3, 1))
        return x.permute(0, 3, 1, 2).contiguous()


class ConvNeXtBlock(nn.Module):
    """mmcls ConvNeXtBlock (linear_pw_conv=True): dw7x7 → LN → MLP → gamma → +res"""

    def __init__(self, dim: int):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pointwise_conv1 = nn.Linear(dim, 4 * dim)
        self.pointwise_conv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.depthwise_conv(x).permute(0, 2, 3, 1)  # NHWC
        x = self.pointwise_conv2(F.gelu(self.pointwise_conv1(self.norm(x))))
        return shortcut + (self.gamma * x).permute(0, 3, 1, 2)


class ConvNeXtBackbone(nn.Module):
    def __init__(self, dims=(96, 192, 384, 768), depths=(3, 3, 9, 3)):
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(3, dims[0], 4, stride=4), LayerNorm2d(dims[0], eps=1e-6)))
        for i in range(1, len(dims)):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm2d(dims[i - 1], eps=1e-6),
                nn.Conv2d(dims[i - 1], dims[i], 2, stride=2)))
        self.stages = nn.ModuleList(
            nn.Sequential(*(ConvNeXtBlock(dims[i]) for _ in range(depths[i])))
            for i in range(len(dims)))
        for i, c in enumerate(dims):
            self.add_module(f"norm{i}", LayerNorm2d(c, eps=1e-6))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for i in range(len(self.stages)):
            x = self.stages[i](self.downsample_layers[i](x))
            outs.append(getattr(self, f"norm{i}")(x))
        return outs


class ConvNeXtUPerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXtBackbone()
        self.decode_head = UPerHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.decode_head(self.backbone(x))
        from models.vmamba_upernet import EXPORT_INPUT_SIZE
        tgt = (EXPORT_INPUT_SIZE, EXPORT_INPUT_SIZE) if EXPORT_INPUT_SIZE else (int(x.shape[2]), int(x.shape[3]))
        return F.interpolate(logits, size=tgt, mode="bilinear", align_corners=False)


def load_pretrained(ckpt_path: Path = CKPT) -> ConvNeXtUPerNet:
    model = ConvNeXtUPerNet()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k: (v.float() if v.is_floating_point() else v)  # fp16 meta → fp32 garanti
          for k, v in sd.items() if not k.startswith("auxiliary_head.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing, f"eksik anahtar: {missing[:5]}"
    assert not unexpected, f"fazla anahtar: {unexpected[:5]}"
    return model.eval()


if __name__ == "__main__":
    import sys
    import time

    sys.path.insert(0, str(ROOT / "src"))
    model = load_pretrained()
    n = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 3, 512, 512)
    t0 = time.perf_counter()
    with torch.no_grad():
        y = model(x)
    print(f"param: {n/1e6:.1f}M | cikti: {tuple(y.shape)} | ileri gecis {time.perf_counter()-t0:.2f} s")
    assert y.shape == (1, NUM_CLASSES, 512, 512)
    print("OK — checkpoint eksiksiz yuklendi")
