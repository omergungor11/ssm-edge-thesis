# Session Notes

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
