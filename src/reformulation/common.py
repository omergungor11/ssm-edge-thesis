"""TASK-030 — SS2D yeniden formülasyonu için ortak parçalar.

Üç formun (seq / blocked / ane) paylaştığı iskelet: ağırlık düzeni orijinal
SS2D (v05_noz, channel_first, d_state=1) state_dict anahtarlarıyla birebir
uyumludur; `load_state_dict(op_state)` doğrudan çalışır.

SS2D (op) alt yapısı — capture.py çıktısıyla doğrulandı:
  in_proj.weight   (768, 384)   Linear2d → 1x1 conv eşdeğeri
  conv2d.weight    (768, 1, 3, 3) depthwise, bias yok
  x_proj_weight    (4, 26, 768)  yön başına [dt_rank=24 | B=1 | C=1] projeksiyonu
  dt_projs_weight  (4, 768, 24)  dt_rank → d_inner
  dt_projs_bias    (4, 768)
  A_logs           (3072, 1)     A = -exp(A_logs)  (d_state=1 → kanal başına skaler)
  Ds               (3072,)       skip katsayısı
  out_norm.{weight,bias} (768,)  LayerNorm2d
  out_proj.weight  (384, 768)

EXPORT NOTU: forward hiçbir yerde x.shape OKUMAZ — tüm boyutlar `spatial`
parametresinden türetilen python sabitleridir. Tam-model export'unda çöken
aten::Int düğümleri (bkz. tez-docs/export-matrisi.md) böyle önlenir; batch
boyutu reshape'lerde -1 ile serbest bırakılır.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 384
D_INNER = 768
DT_RANK = 24
K_GROUP = 4
KD = K_GROUP * D_INNER


def cross_scan(x: torch.Tensor) -> torch.Tensor:
    """(B, D, H, W) → (B, 4, D, L). Yönler: satır, sütun, ters-satır, ters-sütun."""
    row = x.flatten(2)                     # (B, D, L) satır-öncelikli
    col = x.transpose(2, 3).flatten(2)     # (B, D, L) sütun-öncelikli
    return torch.stack([row, col, row.flip(-1), col.flip(-1)], dim=1)


def cross_merge(ys: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """(B, 4, D, L) → (B, D, L). Ters yönler geri çevrilir, sütun yönü satıra hizalanır."""
    y = ys[:, 0:2] + ys[:, 2:4].flip(-1)
    y_row, y_col = y[:, 0], y[:, 1]
    y_col = y_col.reshape(-1, D_INNER, W, H).transpose(2, 3).reshape(-1, D_INNER, H * W)
    return y_row + y_col


class SS2DBase(nn.Module):
    """Ön/son işleme ortak; çekirdek tarama alt sınıfta (`_scan_core`)."""

    def __init__(self, spatial: int = 32) -> None:
        super().__init__()
        self.spatial = spatial
        self.in_proj = nn.Linear(D_MODEL, D_INNER, bias=False)
        self.conv2d = nn.Conv2d(D_INNER, D_INNER, 3, padding=1, groups=D_INNER, bias=False)
        self.x_proj_weight = nn.Parameter(torch.empty(K_GROUP, DT_RANK + 2, D_INNER))
        self.dt_projs_weight = nn.Parameter(torch.empty(K_GROUP, D_INNER, DT_RANK))
        self.dt_projs_bias = nn.Parameter(torch.empty(K_GROUP, D_INNER))
        self.A_logs = nn.Parameter(torch.empty(KD, 1))
        self.Ds = nn.Parameter(torch.empty(KD))
        self.out_norm = nn.LayerNorm(D_INNER)
        self.out_proj = nn.Linear(D_INNER, D_MODEL, bias=False)

    def _scan_core(self, a_log: torch.Tensor, bu: torch.Tensor, L: int) -> torch.Tensor:
        """h çöz: h_t = exp(a_log_t)·h_{t-1} + bu_t. Girdi/çıktı (B, K*D, L)."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 384, H, W)
        H = W = self.spatial
        L = H * W
        x = F.conv2d(x, self.in_proj.weight[:, :, None, None])
        x = self.conv2d(x)
        x = F.silu(x)

        xs = cross_scan(x)                                   # (B, 4, D, L)
        u = xs.reshape(-1, KD, L)
        x_dbl = F.conv1d(u, self.x_proj_weight.reshape(-1, D_INNER, 1), groups=K_GROUP)
        x_dbl = x_dbl.reshape(-1, K_GROUP, DT_RANK + 2, L)
        dts, Bs, Cs = torch.split(x_dbl, [DT_RANK, 1, 1], dim=2)
        dts = F.conv1d(dts.reshape(-1, K_GROUP * DT_RANK, L),
                       self.dt_projs_weight.reshape(KD, DT_RANK, 1),
                       groups=K_GROUP)                       # (B, K*D, L)

        delta = F.softplus(dts + self.dt_projs_bias.reshape(1, -1, 1))
        A = -torch.exp(self.A_logs.reshape(1, -1, 1))        # (1, K*D, 1) skaler A
        Bk = Bs.expand(-1, K_GROUP, D_INNER, L).reshape(-1, KD, L)
        Ck = Cs.expand(-1, K_GROUP, D_INNER, L).reshape(-1, KD, L)

        a_log = delta * A                                    # log(a_t) ≤ 0
        bu = delta * Bk * u
        h = self._scan_core(a_log, bu, L)
        y = Ck * h + u * self.Ds.reshape(1, -1, 1)

        y = cross_merge(y.reshape(-1, K_GROUP, D_INNER, L), H, W)
        y = y.transpose(1, 2)                                # (B, L, D)
        y = self.out_norm(y)
        y = y.transpose(1, 2).reshape(-1, D_INNER, H, W)
        return F.conv2d(y, self.out_proj.weight[:, :, None, None])
