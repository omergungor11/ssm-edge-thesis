"""TASK-030 Adım 3 — `blocked` kapalı form (iki varyant).

d_state=1 → özyineleme kanal başına skaler: h_t = a_t·h_{t-1} + b_t,
a_t = exp(Δ_t·A). Blok uzunluğu P için blok-içi kapalı form + bloklar arası
h_0 taşıma (L/P adımlık kısa döngü; trace'te L/P kez unroll olur).

Varyant 1 — `cumsum` (görev tanımındaki reçete):
    S_t = Σ_{i≤t} Δ_i·A   (log-cumprod; cumprod'a gerek yok, ONNX'te zaten yok)
    h_t = e^{S_t}·(h_0 + Σ_{i≤t} b_i·e^{-S_i})
  SORUN: gerçek aktivasyonlarda blok içi min(S) ≈ -513 (P=64) ölçüldü →
  e^{-S} fp32'de taşar (üst sınır ~e^88) → NaN. Bu varyant kayıt için tutulur;
  başarısızlık da veridir (bkz. reform_matrix.jsonl 'verify' kayıtları).

Varyant 2 — `decay` (sayısal olarak stabil, varsayılan):
    T[t,i] = e^{S_t - S_i}  (i ≤ t; her giriş ≤ 1 → taşma imkânsız)
    h_blok = T @ b,  taşıma: h_t += e^{S_t}·h_0  (e^{S_t} alttan taşarsa 0 —
    zararsız: gerçek katkı zaten ihmal edilebilir)
  Bu, Mamba2/SSD'nin "chunked" formülasyonunun d_state=1 özel hâlidir:
  blok-köşegen dikkat matrisi + skaler taşıma. cumsum + exp + matmul —
  hepsi ONNX/CoreML'in yerli op'ları.
"""
from __future__ import annotations

import torch

from common import KD, SS2DBase


def blocked_scan_cumsum(a_log: torch.Tensor, bu: torch.Tensor, P: int, L: int) -> torch.Tensor:
    """Reçetedeki e^{-S} bölmeli form — fp32'de gerçek veride taşar (kayıt için)."""
    nb = L // P
    S = torch.cumsum(a_log.reshape(-1, KD, nb, P), dim=-1)
    cum_a = torch.exp(S)
    b4 = bu.reshape(-1, KD, nb, P)
    t = torch.cumsum(b4 * torch.exp(-S), dim=-1)
    h0 = torch.zeros_like(b4[:, :, 0, 0:1])
    outs = []
    for k in range(nb):
        h_k = cum_a[:, :, k] * (h0 + t[:, :, k])
        outs.append(h_k)
        h0 = h_k[:, :, -1:]
    return torch.stack(outs, dim=2).reshape(-1, KD, L)


def blocked_scan_decay(a_log: torch.Tensor, bu: torch.Tensor, P: int, L: int,
                       tril: torch.Tensor) -> torch.Tensor:
    """Stabil çürüme-matrisi formu: blok içi T@b (T alt-üçgen, girişler ≤ 1)."""
    nb = L // P
    S = torch.cumsum(a_log.reshape(-1, KD, nb, P), dim=-1)
    b4 = bu.reshape(-1, KD, nb, P)
    h0 = torch.zeros_like(b4[:, :, 0:1, 0:1])              # (B, C, 1, 1)
    outs = []
    for k in range(nb):
        S_k = S[:, :, k]                                    # (B, C, P)
        # Maske exp'ten ÖNCE eklenir: üst üçgende S_t - S_i > 0 olabilir (+513
        # ölçüldü) → önce exp sonra maske inf·0 = NaN üretir. -1e4 → exp → 0.
        diff = S_k.unsqueeze(-1) - S_k.unsqueeze(-2)
        T = torch.exp(diff * tril + (tril - 1.0) * 1e4)
        h_k = T @ b4[:, :, k].unsqueeze(-1)                 # (B, C, P, 1)
        h_k = h_k + torch.exp(S_k).unsqueeze(-1) * h0
        outs.append(h_k.squeeze(-1))
        h0 = h_k[:, :, P - 1:P, :]
    return torch.stack(outs, dim=2).reshape(-1, KD, L)


class SS2DBlocked(SS2DBase):
    def __init__(self, block_size: int = 128, mode: str = "decay", spatial: int = 32) -> None:
        super().__init__(spatial=spatial)
        self.block_size = block_size
        self.mode = mode
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size)), persistent=False)

    def _scan_core(self, a_log: torch.Tensor, bu: torch.Tensor, L: int) -> torch.Tensor:
        if self.mode == "cumsum":
            return blocked_scan_cumsum(a_log, bu, self.block_size, L)
        return blocked_scan_decay(a_log, bu, self.block_size, L, self.tril)
