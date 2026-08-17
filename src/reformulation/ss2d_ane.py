"""TASK-030 Adım 4 — `ane` formu: blocked/decay matematiği, Apple ANE reçetesiyle.

ANE reçetesi (Apple "Deploying Transformers on the Apple Neural Engine"):
  - Tüm ara tensörler 4D (B, C, 1, S) ya da (B, C, nb, P) düzeninde kalır
  - Projeksiyonlar 1x1 conv2d (Linear/conv1d yerine)
  - LayerNorm kanal ekseninde permute'suz hesaplanır (mean/var dim=1)
  - transpose sayısı minimize: yalnızca 2 gerçek transpose kalır
    (sütun-yönlü tarama girişte 1, birleştirmede 1); blok içi (B,C,1,P)→(B,C,P,1)
    geçişleri bitişik-bellek reshape'idir, transpose değil.

Sayım (statik, kod üzerinden):
  base (seq/blocked) yolu : 4 transpose + ~14 reshape/view + LN çevresinde 2 permute
  ane yolu                : 2 transpose + ~12 reshape (+ blok başına 2 bitişik reshape)

Ağırlık düzeni SS2DBase ile aynı → aynı state_dict yüklenir.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from common import D_INNER, DT_RANK, K_GROUP, SS2DBase


class SS2DAne(SS2DBase):
    def __init__(self, block_size: int = 128, spatial: int = 32) -> None:
        super().__init__(spatial=spatial)
        self.block_size = block_size
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size)), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, 384, H, W)
        H = W = self.spatial
        L = H * W
        P = self.block_size
        nb = L // P
        KD = K_GROUP * D_INNER

        x = F.conv2d(x, self.in_proj.weight[:, :, None, None])
        x = self.conv2d(x)
        x = F.silu(x)

        # --- çapraz tarama → (B, 4*D, 1, L); yön kanala katlanır -------------
        row = x.reshape(-1, D_INNER, 1, L)
        col = x.transpose(2, 3).reshape(-1, D_INNER, 1, L)          # transpose #1
        xs = torch.cat([row, col, row.flip(-1), col.flip(-1)], dim=1)

        # --- projeksiyonlar: gruplu 1x1 conv2d -------------------------------
        x_dbl = F.conv2d(xs, self.x_proj_weight.reshape(-1, D_INNER, 1, 1),
                         groups=K_GROUP)                            # (B, 104, 1, L)
        x_dbl = x_dbl.reshape(-1, K_GROUP, DT_RANK + 2, L)
        dts = x_dbl[:, :, :DT_RANK].reshape(-1, K_GROUP * DT_RANK, 1, L)
        Bs = x_dbl[:, :, DT_RANK:DT_RANK + 1].expand(-1, -1, D_INNER, -1).reshape(-1, KD, 1, L)
        Cs = x_dbl[:, :, DT_RANK + 1:].expand(-1, -1, D_INNER, -1).reshape(-1, KD, 1, L)

        dts = F.conv2d(dts, self.dt_projs_weight.reshape(KD, DT_RANK, 1, 1),
                       groups=K_GROUP)                              # (B, KD, 1, L)
        delta = F.softplus(dts + self.dt_projs_bias.reshape(1, KD, 1, 1))
        a_log = delta * (-torch.exp(self.A_logs.reshape(1, KD, 1, 1)))
        bu = delta * Bs * xs

        # --- bloklu çürüme-matrisi taraması, tamamı 4D -----------------------
        S = torch.cumsum(a_log.reshape(-1, KD, nb, P), dim=3)
        b4 = bu.reshape(-1, KD, nb, P)
        tril = self.tril
        h0 = torch.zeros_like(b4[:, :, 0:1, 0:1])
        outs = []
        for k in range(nb):
            S_row = S[:, :, k:k + 1, :]                    # (B, KD, 1, P)
            S_col = S_row.reshape(-1, KD, P, 1)             # bitişik reshape
            T = torch.exp((S_col - S_row) * tril + (tril - 1.0) * 1e4)
            b_k = b4[:, :, k:k + 1, :].reshape(-1, KD, P, 1)
            h_k = T @ b_k + torch.exp(S_col) * h0          # (B, KD, P, 1)
            outs.append(h_k.reshape(-1, KD, 1, P))
            h0 = h_k[:, :, P - 1:P, :]
        h = torch.cat(outs, dim=3)                         # (B, KD, 1, L)

        y = Cs * h + xs * self.Ds.reshape(1, KD, 1, 1)

        # --- birleştirme: ters yönler çevrilir, sütun satıra hizalanır -------
        y_row = y[:, :D_INNER] + y[:, 2 * D_INNER:3 * D_INNER].flip(-1)
        y_col = y[:, D_INNER:2 * D_INNER] + y[:, 3 * D_INNER:].flip(-1)
        y_col = y_col.reshape(-1, D_INNER, W, H).transpose(2, 3)     # transpose #2
        y = y_row.reshape(-1, D_INNER, H, W) + y_col

        # --- LayerNorm kanal ekseninde, permute'suz --------------------------
        mu = y.mean(dim=1, keepdim=True)
        var = (y * y).mean(dim=1, keepdim=True) - mu * mu
        y = (y - mu) * torch.rsqrt(var + self.out_norm.eps)
        y = y * self.out_norm.weight.reshape(1, D_INNER, 1, 1) \
            + self.out_norm.bias.reshape(1, D_INNER, 1, 1)

        return F.conv2d(y, self.out_proj.weight[:, :, None, None])
