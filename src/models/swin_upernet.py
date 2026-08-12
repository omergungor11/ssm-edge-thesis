"""Swin-T + UPerNet (ADE20K) — mmseg'siz, saf-torch yükleyici (TASK-015).

mmseg (refactor sonrası) SwinTransformer anahtar düzenine birebir uyumlu
modül isimleriyle implement edildi; hedef missing=0 / unexpected=0.
Bildirilen: mIoU 44.41 (512x512, 160k iter, 'whole' test).

mmseg'e özgü iki tuzak (tez notu):
- PatchMerging, orijinal Swin'in [0::2,0::2]-concat sırası yerine nn.Unfold
  kanal-major sırasını kullanır (mmseg swin2mmseg.py ağırlıkları buna göre
  permüte etmiştir) — orijinal sırayla yüklemek sessizce yanlış sonuç verir.
- relative_position_index checkpoint'te buffer olarak kayıtlıdır; kendimiz
  hesaplamak yerine doğrudan yüklenir.
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
CKPT = ROOT / "checkpoints" / "upernet_swin_tiny_ade20k.pth"


class WindowMSA(nn.Module):
    """mmseg WindowMSA — anahtarlar: qkv, proj, relative_position_bias_table/index."""

    def __init__(self, dim: int, num_heads: int, window: int = 7):
        super().__init__()
        self.num_heads = num_heads
        self.window = window
        self.scale = (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window - 1) ** 2, num_heads))
        self.register_buffer(
            "relative_position_index",
            torch.zeros(window * window, window * window, dtype=torch.long))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None,
                dim: int, nw: int) -> torch.Tensor:
        # Trace-dostu: x.shape okunmaz — N/C/nw Python int olarak dışarıdan gelir,
        # batch boyutu her yerde -1 (aksi hâlde aten::Int → CoreML çöküyor).
        N = self.window * self.window
        hd = dim // self.num_heads
        qkv = self.qkv(x).reshape(-1, N, 3, self.num_heads, hd)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(N, N, -1)
        attn = attn + bias.permute(2, 0, 1)[None]
        if mask is not None:
            attn = attn.view(-1, nw, self.num_heads, N, N) + mask[None, :, None]
            attn = attn.view(-1, self.num_heads, N, N)
        x = (attn.softmax(-1) @ v).transpose(1, 2).reshape(-1, N, dim)
        return self.proj(x)


class ShiftWindowMSA(nn.Module):
    """mmseg ShiftWindowMSA — pad + cyclic shift + pencere maskesi. Anahtar: w_msa.*"""

    def __init__(self, dim: int, num_heads: int, window: int, shift: int):
        super().__init__()
        self.window, self.shift = window, shift
        self.w_msa = WindowMSA(dim, num_heads, window)

    def _partition(self, x: torch.Tensor, H: int, W: int, C: int) -> torch.Tensor:
        """(B,H,W,C) -> (nW*B,ws*ws,C) — H/W/C Python int, shape okunmaz."""
        ws = self.window
        x = x.view(-1, H // ws, ws, W // ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(-1, ws * ws, C)

    def _reverse(self, wins: torch.Tensor, H: int, W: int, C: int) -> torch.Tensor:
        ws = self.window
        x = wins.view(-1, H // ws, W // ws, ws, ws, C).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(-1, H, W, C)

    def forward(self, x: torch.Tensor, hw: tuple[int, int], dim: int) -> torch.Tensor:
        H, W = hw
        C = dim
        ws, sh = self.window, self.shift
        x = x.view(-1, H, W, C)
        pad_r, pad_b = (ws - W % ws) % ws, (ws - H % ws) % ws
        if pad_r or pad_b:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        mask = None
        if sh > 0:
            x = torch.roll(x, (-sh, -sh), dims=(1, 2))
            img_mask = torch.zeros(1, Hp, Wp, 1, device=x.device)
            cnt = 0
            for hs in (slice(0, -ws), slice(-ws, -sh), slice(-sh, None)):
                for wsl in (slice(0, -ws), slice(-ws, -sh), slice(-sh, None)):
                    img_mask[:, hs, wsl] = cnt
                    cnt += 1
            mw = self._partition(img_mask, Hp, Wp, 1).view(-1, ws * ws)
            mask = mw[:, None] - mw[:, :, None]
            mask = mask.masked_fill(mask != 0, -100.0)

        nw = (Hp // ws) * (Wp // ws)
        x = self._reverse(self.w_msa(self._partition(x, Hp, Wp, C), mask, C, nw), Hp, Wp, C)
        if sh > 0:
            x = torch.roll(x, (sh, sh), dims=(1, 2))
        return x[:, :H, :W].reshape(-1, H * W, C)


class FFN(nn.Module):
    """mmcv FFN anahtar düzeni: layers.0.0 (Linear) + layers.1 (Linear)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU()), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, window: int, shift: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ShiftWindowMSA(dim, num_heads, window, shift)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FFN(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), hw, self.dim)
        return x + self.ffn(self.norm2(x))


class PatchMerging(nn.Module):
    """mmcv PatchMerging: nn.Unfold(k=2,s=2) kanal-major concat → norm → reduction."""

    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, hw: tuple[int, int]) -> tuple[torch.Tensor, tuple[int, int]]:
        H, W = hw
        C = self.norm.normalized_shape[0] // 4
        x = x.transpose(1, 2).view(-1, C, H, W)
        if H % 2 or W % 2:  # corner pad (mmcv AdaptivePadding)
            x = F.pad(x, (0, W % 2, 0, H % 2))
        x = F.unfold(x, kernel_size=2, stride=2)  # (B, 4C, L') kanal-major
        x = self.reduction(self.norm(x.transpose(1, 2)))
        return x, ((H + 1) // 2, (W + 1) // 2)


class SwinStage(nn.Module):
    """mmseg SwinBlockSequence: blocks + opsiyonel downsample."""

    def __init__(self, dim: int, depth: int, num_heads: int, window: int, downsample: bool):
        super().__init__()
        self.blocks = nn.ModuleList(
            SwinBlock(dim, num_heads, window, 0 if i % 2 == 0 else window // 2)
            for i in range(depth))
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, hw):
        for blk in self.blocks:
            x = blk(x, hw)
        if self.downsample is not None:
            x_down, hw_down = self.downsample(x, hw)
            return x_down, hw_down, x, hw
        return x, hw, x, hw


class PatchEmbed(nn.Module):
    def __init__(self, dim: int = 96):
        super().__init__()
        self.projection = nn.Conv2d(3, dim, 4, stride=4)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.projection(x)
        # int(...) zorunlu: trace'te 0-dim tensör kalırsa downstream view/pad
        # hesapları aten::Int düğümü üretiyor ve CoreML dönüşümü çöküyor.
        # Export modunda tamamen statik boyut kullanılır.
        from models.vmamba_upernet import EXPORT_INPUT_SIZE
        if EXPORT_INPUT_SIZE:
            hw = (EXPORT_INPUT_SIZE // 4, EXPORT_INPUT_SIZE // 4)
        else:
            hw = (int(x.shape[2]), int(x.shape[3]))
        return self.norm(x.flatten(2).transpose(1, 2)), hw


class SwinBackbone(nn.Module):
    def __init__(self, dim=96, depths=(2, 2, 6, 2), heads=(3, 6, 12, 24), window=7):
        super().__init__()
        self.patch_embed = PatchEmbed(dim)
        dims = [dim * 2 ** i for i in range(len(depths))]
        self.stages = nn.ModuleList(
            SwinStage(dims[i], depths[i], heads[i], window, i < len(depths) - 1)
            for i in range(len(depths)))
        for i, c in enumerate(dims):
            self.add_module(f"norm{i}", nn.LayerNorm(c))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x, hw = self.patch_embed(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw, out, out_hw = stage(x, hw)
            out = getattr(self, f"norm{i}")(out)
            outs.append(out.view(-1, *out_hw, out.shape[-1]).permute(0, 3, 1, 2).contiguous())
        return outs


class SwinUPerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SwinBackbone()
        self.decode_head = UPerHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.decode_head(self.backbone(x))
        from models.vmamba_upernet import EXPORT_INPUT_SIZE
        tgt = (EXPORT_INPUT_SIZE, EXPORT_INPUT_SIZE) if EXPORT_INPUT_SIZE else (int(x.shape[2]), int(x.shape[3]))
        return F.interpolate(logits, size=tgt, mode="bilinear", align_corners=False)


def load_pretrained(ckpt_path: Path = CKPT) -> SwinUPerNet:
    model = SwinUPerNet()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k: (v.float() if v.is_floating_point() else v)
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
