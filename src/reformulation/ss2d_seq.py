"""TASK-030 Adım 2 — `seq` referans formu.

Üçüncü-parti selective_scan_torch ile matematiksel olarak birebir aynı
ardışık özyineleme; d_state=1 olduğundan durum kanal başına skalerdir:
    h_t = a_t · h_{t-1} + b_t,   a_t = exp(Δ_t · A) ∈ (0, 1]
Trace/export edildiğinde L adımlık python döngüsü tamamen açılır (unroll) —
tezin gösterdiği graf patlamasının tek-blok ölçekli hali.
"""
from __future__ import annotations

import torch

from common import SS2DBase


class SS2DSeq(SS2DBase):
    def _scan_core(self, a_log: torch.Tensor, bu: torch.Tensor, L: int) -> torch.Tensor:
        a = torch.exp(a_log)
        h = torch.zeros_like(bu[:, :, 0])
        hs = []
        for t in range(L):
            h = a[:, :, t] * h + bu[:, :, t]
            hs.append(h)
        return torch.stack(hs, dim=2)
