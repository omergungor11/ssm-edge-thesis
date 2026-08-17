# Teoriden Silikona - SSM Goru Omurgalarinin Uc Cihaz Verimliligi

## Proje

YL tezi: SSM tabanlı görü omurgalarının (VMamba vb.) teorik verimlilik avantajının **Apple Silicon**
dağıtım yığınlarında (PyTorch eager/compile, ONNX Runtime CPU+CoreML EP, CoreML CPU/GPU/ANE) ne
kadarının gerçekleştiğinin ampirik analizi. 16 hafta, hedef bitiş ~30 Kasım 2026.

- **GitHub**: (henüz remote yok)
- **Donanım**: Sadece bu makine — baz Apple M5, 24 GB (NVIDIA yok; revizyon → `tez-plans/revizyon-mac-only.md`)
- **Eğitim yok**: yayınlanmış ADE20K checkpoint'leri + doğruluk-gecikme Pareto düzlemi

## Slash Commandlar

| Command | Ne yapar |
|---------|----------|
| `/cold-start` | Session baslangici — projeyi oku, durumu raporla |
| `/git-full` | Stage, commit, push — task durumlarini guncelle |
| `/local-testing` | Tum servisleri ayaga kaldir ve dogrula |
| `/turn-off` | Session notu yaz, tasklari isaretle, push, kapat |

---

## Mevcut Durum

**Progress**: 34/37 task (%92) — Faz 0-4 TAMAM + tez metni komple (Özet+Bölüm 1-6+kaynakça+ekler).
**GitHub**: github.com/omergungor11/ssm-edge-thesis (public, MIT).
**Sıradaki**: TASK-037 bütünsel okuma (🧑 Ömer — kritik!), TASK-038 repo cilası.
Faz 4 sonucu: blok-form CoreML kapısını açtı (düğüm 20×↓); kök teşhis: engel op-uyumu değil graf ölçeği.
**Ana bulgular**: `tez-docs/export-matrisi.md` + `tez-docs/nicemleme-sonuclari.md` —
"SSM'in uç engeli sayısal değil yapısal" (CoreML ∅, 1024² bellek duvarı, ORT paradoksu,
ANE: CNN %100 / ViT %0 / SSM ∅; W8 bedava).

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
