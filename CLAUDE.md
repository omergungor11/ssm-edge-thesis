# Teoriden Silikona - SSM Goru Omurgalarinin Uc Cihaz Verimliligi

## Proje

YL tezi: SSM tabanlı görü omurgalarının (VMamba vb.) teorik verimlilik avantajının gerçek dağıtım
yığınlarında (PyTorch/CUDA, torch.compile, ONNX Runtime, CoreML/ANE) ne kadarının gerçekleştiğinin
ampirik analizi. 16 hafta, hedef bitiş ~30 Kasım 2026.

- **GitHub**: (henüz remote yok)
- **Donanım**: RTX 5070 (ayrı sistem — eğitim + CUDA ölçüm) + Apple M5 Pro (bu makine — ANE ölçüm)

## Slash Commandlar

| Command | Ne yapar |
|---------|----------|
| `/cold-start` | Session baslangici — projeyi oku, durumu raporla |
| `/git-full` | Stage, commit, push — task durumlarini guncelle |
| `/local-testing` | Tum servisleri ayaga kaldir ve dogrula |
| `/turn-off` | Session notu yaz, tasklari isaretle, push, kapat |

---

## Mevcut Durum

**Progress**: 0/38 task (%0) — Faz 0 (öncül doğrulama), Hafta 1.

> Her yeni session'da `tez-tasks/task-index.md` oku veya `/cold-start` calistir.

---

## Workspace

```
src/benchmark/   → Ölçüm harness'ı (TASK-009) — tezin en çok kullanılan kodu
src/models/      → Omurga + UPerNet boru hattı (Faz 1)
results/raw/     → Ham ölçümler (asla silinmez, git'e dahil)
checkpoints/     → Eğitilmiş ağırlıklar (git dışı)
tez/             → Tez metni bölümleri
```

## Temel Komutlar

```bash
uv venv --python 3.12 && source .venv/bin/activate   # ortam (sistem 3.14 KULLANMA)
uv pip install -r requirements.txt
```

---

## Code Conventions (Kisa)

- **Python 3.12**, type hint'li; dosyalar `snake_case.py`
- Ham ölçüm daima `results/raw/` — sadece grafik saklamak yasak
- Her ölçüm betiği ortam sürümlerini otomatik kaydeder
- **Commit**: `feat(TASK-XXX): aciklama` — **Claude attribution eklenmez** (global kural)

Detaylar → `tez-config/conventions.md`

## Parallel Agent Orchestration

Birden fazla sub-agent paralel calistirilirken:
- Her agent sadece kendi modul dizininde dosya duzenler (dizin izolasyonu)
- Paket kurulumu sadece ana agent (orchestrator) tarafindan yapilir
- Paylasilan dosyalarda retry pattern uygulanir
- Bagimli task'lar sirali, bagimsiz olanlar paralel calistirilir

Detaylar → `tez-config/agent-instructions.md`

---

## Referans Dizinleri

| Dizin | Icerik |
|-------|--------|
| `tez-tasks/` | Task takip — dashboard + tum task'lar |
| `tez-tasks/task-index.md` | Master task listesi |
| `tez-tasks/phases/` | Phase bazli detayli task aciklamalari |
| `tez-tasks/active/session-notes.md` | Session notlari |
| `tez-config/workflow.md` | Task workflow kurallari |
| `tez-config/conventions.md` | Kod standartlari |
| `tez-config/tech-stack.md` | Teknolojiler + versiyonlar |
| `tez-config/agent-instructions.md` | Sub-agent sorumluluklari |
| `tez-docs/MEMORY.md` | Kalici hafiza |
| `tez-docs/CHANGELOG.md` | Degisiklik kaydi |
| `tez-plans/` | Uygulama planlari |

---

## Hooks (Otomatik Kurallar)

| Hook | Tetikleyici | Ne yapar |
|------|------------|----------|
| `protect-files.sh` | PreToolUse (Edit/Write) | .env, lock files, .git/ duzenlemeyi bloklar |

---

## Notlar

- Hafiza dosyasi `tez-docs/MEMORY.md`'de — her session'da oku, gerektiginde guncelle
