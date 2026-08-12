# Session Notes

## 2026-08-11/12 — Session 3 (uzun oturum: revizyon + Faz 0 + Faz 1 açılışı)

### Completed
- [x] TASK-003: İPTAL — 5070'e erişim yok; plan **Mac-only**'ye revize edildi (`tez-plans/revizyon-mac-only.md`)
- [x] TASK-004/005/006: Öncül mikrobenchmark (MiniMamba, L=196 + L=1024)
- [x] TASK-008: KARAR NOKTASI — öncül **rafine haliyle doğrulandı**: darboğaz çıkarımda değil araç zincirinde (`tez-docs/oncul-dogrulama.md`)
- [x] TASK-009: Ölçüm harness'ı `src/benchmark/` (termal kontrol, yığın-agnostik runner'lar, JSONL)
- [x] TASK-010: ResNet-50 doğrulaması — tekrar sapması ≤%1.2. Kilit kontrast: ResNet-50 CoreML dönüşümü 2.5 sn vs MiniMamba 94 dk
- [x] TASK-011: ADE20K + `src/data/ade20k.py` (20210+2000)
- [x] TASK-012: Bölüm 3.5 taslağı `tez/bolum-3.5-olcum-protokolu.md` (4 katmanlı maliyet modeli)
- [x] TASK-013 (kısmen) + TASK-014: VMamba-T+UPerNet Mac'te çalışıyor — mmseg'siz yükleyici, missing=0/unexpected=0. Alt-küme (250): mIoU 39.0 (yanlı, tırmanıyor), **aAcc 83.3 ≈ bildirilen** ✓

### In Progress (arka plan)
- [ ] TASK-014: Tam 2000-görüntü mIoU (~2 sa) — ilerleme `results/raw/ade20k_miou_progress.json`
- [ ] TASK-015/016: Swin-T + ConvNeXt-T checkpoint indirmeleri → `checkpoints/`

### Next Session
- [ ] Tam mIoU sonucu (bildirilen ~48'e yakınsadı mı?) → TASK-014 kapat
- [ ] Swin-T / ConvNeXt-T anahtar-uyumlu yükleyiciler (UPerHead hazır, sadece backbone eşlemesi)
- [ ] TASK-017: EfficientViM checkpoint araştır; DeiT-S mmseg checkpoint var mı bak
- [ ] Enerji ölçümü için kullanıcıdan sudo'lu oturum iste (powermetrics)

### Notes / Kararlar
- **VMamba upstream 2 CPU yaması** → `tez-docs/vmamba-yamalari.md`; `[TEZ YAMASI]` etiketli, third_party git dışı. Bulgu: resmî repo CPU-only'de import edilemiyor
- **Config tuzağı:** v2seg tiny gerçek mimari depths=(2,2,5,2), ssm_ratio=2.0 — yayınlanan config yanıltıcı, ağırlık şekillerinden çıkarıldı
- VMamba-T 512² CPU ~3.6 s/img (saf-torch scan); CNN ~50-200 ms — özel çekirdek bedeli gerçek modelde doğrulandı
- Cityscapes hesabı kullanıcıda bekliyor (Faz 2 sonuna kadar lazım değil)

## 2026-08-11 — Session 2

### Completed
- [x] Konu ve proje seçimi özeti kullanıcıya sunuldu (Teoriden Silikona + 2 portfolyo projesi)
- [x] TASK yapısı 16 haftalık deney planıyla senkronlandı: task-index.md yeniden yazıldı (38 task, 6 faz), phase-0..5.md oluşturuldu
- [x] MEMORY.md, tech-stack.md gerçek içerikle dolduruldu (template placeholder'lar atıldı)
- [x] Ortam keşfi: bu makine M5 Pro Mac — nvidia-smi YOK. Python 3.14.5 (sistem), uv var, torch 2.11.0 MPS aktif

### In Progress
- [ ] TASK-001: git init + ilk commit

### Next Session
- [ ] TASK-002: Mac ortamı — `uv venv --python 3.12` + torch/coremltools/onnxruntime
- [ ] TASK-003 (🧑 Ömer, RTX 5070 makinesinde): `nvidia-smi` + `mamba-ssm` derleme testi — çıktılar buraya yapıştırılacak
- [ ] VRAM doğrulanınca deney matrisini güncelle (12 vs 16 GB)

### Notes
- Python 3.14 tuzağı: coremltools/mamba-ssm tekerlek yok → 3.12 venv şart
- RTX 5070 ayrı sistem; oradaki komutlar sadece Ömer tarafından çalıştırılabilir
- `mamba-ssm` Blackwell'de derlenmezse = tezin ilk bulgusu, log saklanacak

## 2026-08-10 — Session 1

### Completed
- [x] Konu seçimi: 4 aday analiz edildi, "Teoriden Silikona" kilitlendi
- [x] Literatür taraması (~40 kaynak), konu seçim analizi, tez iskeleti, 16 haftalık deney planı yazıldı
- [x] Proje dizini template'ten kuruldu (tez- prefix)

### Notes
- SAM 3 çıktı → aday 2 trivialleşti; 3DGS kalabalık → elendi; Edge AI tek başına özgün değil → 3+6 birleşti
- ONNX Runtime #27796 öncül kanıtı olarak not edildi
