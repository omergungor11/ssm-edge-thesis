# Session Notes

## 2026-08-14 — Session 5 (Faz 4 + Faz 5 yazım maratonu)

### Completed
- [x] GitHub yayını: github.com/omergungor11/ssm-edge-thesis (public, MIT, kapsamlı README; >45MB model artefaktları git geçmişinden temizlendi — 207MiB→6.9MiB)
- [x] TASK-030..032: Faz 4 prototipi — çürüme-matrisi blok formu (naif cumsum NaN bulgusuyla), düğüm 20×↓, CoreML kapısı açıldı; MEKANİZMA: engel op-uyumu değil GRAF ÖLÇEĞİ (tek-blok seq bile ANE'de 6.7ms/%99.9)
- [x] TASK-033: Bölüm 4.5 — FAZ 4 TAMAM, AS1-AS4 hepsi cevaplı
- [x] TASK-034: Bölüm 5 (4.0K — uygulayıcı tablosu, konjonktürel/yapısal ayrımı) + Bölüm 6 (1.8K)
- [x] TASK-035: Bölüm 1 Giriş (2.5K) + Özet/Abstract — TEZ METNİ KOMPLE (~25K kelime)
- [x] TASK-036: Kaynakça 45 künye (48/48 atıf çözüldü, uydurma yok) + EK A-D
- [x] Şekil 4.5-4.7 → 7/7 şekil tamam
- [x] Faz 4 kapsam kararı (kullanıcı): prototip-önce

### Next Session
- [ ] **TASK-037 (🧑 Ömer): bütünsel okuma** — önerilen sıra: Özet → Bölüm 1 → Bölüm 5, sonra 4'ler; notlar gelince ben işlerim
- [ ] TASK-038: repo son cilası (README şekil önizlemeleri, smoke-test)
- [ ] TASK-007 kalıntısını kapat (formalite)
- [ ] Kaynakça tutarlılık raporundaki ayıklama (9 atıfsız tarama-kaynağı, 4 taramasız metin-kaynağı) + QMamba venue teyidi
- [ ] Yer tutucu atıfları metne bağlama (kaynakca.md eşleme tablosu hazır)

### Notlar
- Agent alt-agent bekleme deseni İKİNCİ kez askı yarattı (kaynakça agent'ı) — SendMessage dürtmesiyle çözüldü; agent'lara "alt-agent'a devretme, senkron bitir" talimatı verilecek
- Tez %92: kalan 3 iş, ikisi insanlı

## 2026-08-13 — Session 4 (maraton: Faz 2 kapanış + Faz 3 komple)

### Completed
- [x] TASK-021/022: Ölçüm matrisi + profiller — ORT paradoksu (VMamba ORT 618ms=eager 0.30x; yük yüklemede: 725s), 1024² ONNX bellek duvarı (~65GB, kullanıcı durdurdu), compile fiyaskosu (ConvNeXt 2.9x yavaş, VMamba süreç çöker), CoreML EP 94-parça bölünme
- [x] TASK-023: Xcode ANE kanıtı — ConvNeXt 353/353 ANE, Swin 0/631; ekran görüntüleri `results/raw/xcode/`
- [x] TASK-024: Bölüm 4.2-4.3 v2 (5.6K kelime, Tablo 4.5-4.15) — Faz 2 TAMAM
- [x] TASK-025..028: Nicemleme — W8 bedava (mIoU ±0, ANE %29 hızlanma), W4 asimetrik (-3.9/-2.8), ORT INT8 6-17x pesimizasyon, VMamba INT8 858→715MB (yapı nicemlenemiyor); TASK-027: VMamba aktivasyonları EN ILIMLI ve sabit (literatürün tersi!)
- [x] TASK-029: Bölüm 4.4 (2.5K kelime) — Faz 3 TAMAM
- [x] Bölüm 3 (Yöntem) taslak v1 (3.5K kelime, agent)
- [x] 4 tez şekli (`results/figures/`), enerji matrisi (powermetrics, kullanıcıyla)

### Next Session
- [ ] Faz 4 tasarım kararı: TASK-030 scan yeniden formülasyonu (SSD/blok) — kapsam konuşulacak
- [ ] Şekil 4.5-4.6 üretimi (nicemleme + aykırı değer grafikları) — make_figures'a ekle
- [ ] Ara doğrulamalar: tam-val mIoU'lar (kare-512 fp32/W8), enerji turu tekrarı (temiz koşullarda)
- [ ] Cityscapes hesabı hâlâ kullanıcıda (opsiyonel — 1024 bellek duvarı bulgusundan sonra önemi azaldı)

### Kilit kararlar/dersler
- Ölçüm sırasında Xcode/kullanıcı işi çakışması → duraklat-devam protokolü uygulandı; kirli kayıtlar (768) yeniden ölçüldü
- zsh word-splitting tuzağı (set -- $cfg) mIoU zincirini sessiz düşürdü — bash döngülerinde dikkat
- Rastgele-girdi argmax vekili gerçek mIoU hasarını abartıyor — vekil metrik dersi tezde


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
