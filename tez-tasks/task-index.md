# Task Index

> **Tez:** Teoriden Silikona — Durum-Uzayı Tabanlı Görü Omurgalarının **Apple Silicon** Uç Donanımında Gerçekleşen Verimliliğinin Ampirik Analizi
> **Başlangıç:** 10 Ağustos 2026 · **Hedef bitiş:** 30 Kasım 2026 (16 hafta)
> Kaynak plan: [`tez-plans/deney-plani.md`](../tez-plans/deney-plani.md) · **Revizyon:** [`tez-plans/revizyon-mac-only.md`](../tez-plans/revizyon-mac-only.md) (11 Ağu — NVIDIA ekseni çıkarıldı)

## Dashboard

| Faz | Ad | Hafta | Total | Done | In Progress | Pending | Blocked |
|-----|-----|-------|-------|------|-------------|---------|---------|
| 0 | Öncül doğrulama + ölçüm altyapısı | 1-2 | 11 | 10 | 1 | 0 | 0 |
| 1 | Model edinimi + doğruluk doğrulama *(revize)* | 3-4 | 7 | 7 | 0 | 0 | 0 |
| 2 | Apple dağıtım yığını matrisi ← *kalp* | 5-7 | 5 | 5 | 0 | 0 | 0 |
| 3 | Nicemlemenin yoğun tahmine transferi | 8-10 | 5 | 5 | 0 | 0 | 0 |
| 4 | ANE-dostu yeniden formülasyon ← *riskli* | 11-13 | 4 | 0 | 0 | 4 | 0 |
| 5 | Yazım ve toparlama | 14-16 | 5 | 0 | 0 | 5 | 0 |
| **Total** | | | **37** | **27** | **0** | **10** | **0** |

**Progress**: 27/37 (%73) — **FAZ 3 TAMAM** → Faz 4 (özgün katkı) ya da Faz 5 hazırlığı · **Öncül doğrulandı** (`tez-docs/oncul-dogrulama.md`) · *TASK-006: CoreML EP hücresi Faz 2'ye devredildi

### Kim sütunu

| İşaret | Anlamı |
|---|---|
| 🧑 | Sadece Ömer — RTX 5070 makinesinde fiziksel erişim gerekiyor |
| 🤝 | Ortak — Claude hazırlar, Ömer çalıştırır/karar verir |
| 🤖 | Claude tek başına yapabilir |

---

## Faz 0: Öncül doğrulama + ölçüm altyapısı *(Hafta 1-2)*

> **Amaç tezi yazmak değil, tezin öncülünün doğru olduğunu kendi makinende görmek.**
> ONNX'te Mamba yavaş değilse konu değişmeli — bunu 16. haftada değil 1. haftada öğren.

| ID | Task | Kim | Complexity | Status | Dependencies |
|----|------|-----|-----------|--------|-------------|
| TASK-001 | Git repo init + .gitignore + ilk commit | 🤖 | S | ✅ DONE | - |
| TASK-002 | Mac ortam kurulumu (Python 3.12 venv, torch/MPS, coremltools, onnxruntime) | 🤖 | M | ✅ DONE | TASK-001 |
| TASK-003 | ~~RTX 5070 ortamı~~ **İPTAL** — 5070'e erişim yok (bkz. revizyon). 5070 dönerse arşivden geri açılır | — | — | ❌ CANCELLED | - |
| TASK-004 | Referans gecikme ölçümü — PyTorch eager (CPU+MPS), saf-torch selective scan | 🤖 | M | ✅ DONE | TASK-002 |
| TASK-005 | ONNX export denemesi — graf boyutu, unroll davranışı, yükleme süresi | 🤖 | M | ✅ DONE | TASK-004 |
| TASK-006 | ONNX Runtime ölçümü (CPU EP + **CoreML EP**) → **ilk yavaşlama oranı** | 🤖 | M | ✅ DONE* | TASK-005 |
| TASK-007 | CoreML export + M5 / ANE ölçümü (CPU/GPU/ANE compute unit ayrımı) | 🤝 | M | 🔶 PARTIAL (ALL units ölçüldü; unit ayrımı + Xcode → Faz 2) | TASK-002, TASK-005 |
| TASK-008 | **KARAR NOKTASI** — `tez-docs/oncul-dogrulama.md`: öncül doğrulandı mı? | 🤝 | S | ✅ DONE — **doğrulandı (rafine)** | TASK-006, TASK-007 |
| TASK-009 | Ölçüm harness'ı: ısınma, senkronizasyon, termal bekleme, `powermetrics` (ops.), istatistik | 🤖 | L | ✅ DONE — `src/benchmark/` | TASK-008 |
| TASK-010 | Harness doğrulaması — ResNet-50 ile bilinen sayılar üretiliyor mu? | 🤝 | M | ✅ DONE — desen + %1.2 tekrar | TASK-009 |
| TASK-011 | Veri kümesi boru hattı: ADE20K ✅ (20210+2000, loader doğrulandı) · Cityscapes → 🧑 hesap · ImageNet alt kümesi → Faz 3 başında | 🤖 | M | ✅ DONE | TASK-002 |
| TASK-012 | Bölüm 3.5 (Ölçüm protokolü) taslağı — ~4 sayfa | 🤖 | M | ✅ DONE (v1) — `tez/bolum-3.5-olcum-protokolu.md` | TASK-009 |

> **Karar noktası (TASK-008):** Yavaşlama gözlemlendi mi? **Evet** → devam. **Hayır** → konuyu ANE/CoreML eksenine daralt (bkz. `deney-plani.md` çıkış yolları). Bu bir başarısızlık değil, ucuz bir keşif.

---

## Faz 1: Model edinimi ve doğruluk doğrulama *(Hafta 3-4)* — *revize: eğitim yok*

> Mac'te VMamba eğitimi fiilen imkânsız (mamba-ssm MPS'te yok). Yayınlanmış ADE20K checkpoint'leri
> kullanılır; "eşit doğruluk" yerine **doğruluk-gecikme Pareto düzlemi** raporlanır.
> Kazanılan ~2 hafta Faz 2'ye tampon.

| ID | Task | Kim | Complexity | Status | Dependencies |
|----|------|-----|-----------|--------|-------------|
| TASK-013 | Değerlendirme boru hattı: ADE20K val + mIoU hesabı | 🤖 | L | ✅ DONE — `src/models/eval_ade20k.py` | TASK-011 |
| TASK-014 | VMamba-T checkpoint doğrulaması | 🤖 | M | ✅ DONE — **mIoU 48.33 ≈ bildirilen 48.3** | TASK-013 |
| TASK-015 | Swin-T doğrulaması | 🤖 | M | ✅ DONE — **mIoU 44.32 ≈ 44.41** | TASK-013 |
| TASK-016 | ConvNeXt-T doğrulaması | 🤖 | M | ✅ DONE — **mIoU 45.42** (46.11 slide; −0.69 protokol farkı) | TASK-013 |
| TASK-017 | EfficientViM | 🤖 | M | ✅ DONE — **gerekçeyle kapsam dışı** (FPN başlık, reçete uyumsuz; bkz. Tablo 4.1) | TASK-013 |
| TASK-018 | **Tablo 4.1 — model kartları** + Pareto çerçevesi | 🤖 | M | ✅ DONE — `tez/bolum-4.1-model-kartlari.md` (GFLOPs → Faz 2 başı) | TASK-014..017 |
| TASK-019 | Bölüm 2 (Kuramsal temeller) yazımı | 🤖 | L | ✅ DONE (v1) — `tez/bolum-2-kuramsal-temeller.md` 6.9K kelime | - |

**Risk:** VMamba'nın saf-torch fallback'i Mac'te çok yavaşsa mIoU doğrulaması alt-küme (ör. 500 görüntü) üzerinden yapılır — tam val seti yerine, gerekçesi yazılır.

---

## Faz 2: Dağıtım yığını matrisi *(Hafta 6-8)* ← **Tezin kalbi**

> Bu fazın sonunda tezin ana iddiası kanıtlanmış ya da çürütülmüş olur.

| ID | Task | Kim | Complexity | Status | Dependencies |
|----|------|-----|-----------|--------|-------------|
| TASK-020 | Export matrisi | 🤝 | L | ✅ DONE — `tez-docs/export-matrisi.md` (VMamba: ONNX 390K düğüm/12dk yükleme, CoreML ❌; compile/CoreML-EP → 021'e) | TASK-018 |
| TASK-021 | Tam gecikme/bellek/enerji ölçümü + çözünürlük taraması | 🤝 | L | ✅ DONE — 7 yığın × 4 çözünürlük + enerji; 1024 ONNX bellek duvarı | TASK-020, TASK-010 |
| TASK-022 | Operatör seviyesi profilleme | 🤝 | L | ✅ DONE — ORT profilleri: optimizer unroll'u eritiyor, %82 conv; yük 'yükleme' katmanında | TASK-021 |
| TASK-023 | **ANE yürütme oranı doğrulaması** (Xcode) | 🧑 | M | ✅ DONE — **ConvNeXt %100 ANE, Swin %0** (353/353 vs 0/631) | TASK-021 |
| TASK-024 | Bölüm 4.2 (AS1) + 4.3 (AS2) yazımı | 🤖 | L | ✅ DONE (v2) — 5.6K kelime, Tablo 4.5-4.15, tüm ölçümler işlendi | TASK-022, TASK-023 |

---

## Faz 3: Nicemlemenin yoğun tahmine transferi *(Hafta 9-11)*

| ID | Task | Kim | Complexity | Status | Dependencies |
|----|------|-----|-----------|--------|-------------|
| TASK-025 | Nicemleme boru hattı (Aşama I: ağırlık-yalnız) | 🤝 | L | ✅ DONE — W8/W4 CoreML + ORT INT8; `tez-docs/nicemleme-sonuclari.md` | TASK-021 |
| TASK-026 | Nicemleme mIoU etkisi | 🤝 | L | ✅ DONE — **W8 ±0, W4 −2.8/−3.9** (250-alt-küme) | TASK-025 |
| TASK-027 | Aykırı değer × çözünürlük analizi | 🤝 | M | ✅ DONE — **VMamba en ılımlı, sabit; ConvNeXt 768'de kurtosis 64** | TASK-026 |
| TASK-028 | Nicemlenmiş modellerin yığın ölçümü | 🤝 | M | ✅ DONE — W8 ANE %29 hızlanma; ORT INT8 6-17× pesimizasyon; VMamba 715MB/691s değişmedi | TASK-026 |
| TASK-029 | Bölüm 4.4 (AS3) yazımı | 🤖 | M | ✅ DONE — 2.5K kelime, Tablo 4.16-4.19 | TASK-027, TASK-028 |

---

## Faz 4: Yeniden formülasyon *(Hafta 12-13)* ← **Özgün katkı, riskli**

> **Bu fazı asla Faz 1-3'ün önüne alma.** Faz 4 tamamen başarısız olsa bile Faz 1-3 tek başına bir tez oluşturur.

| ID | Task | Kim | Complexity | Status | Dependencies |
|----|------|-----|-----------|--------|-------------|
| TASK-030 | Scan yeniden formülasyonu — **PROTOTİP kapsamı** (kullanıcı kararı 14 Ağu): tek SS2D bloğu × 3 form (seq/blocked/ane) | 🤝 | L | 🔶 IN PROGRESS | TASK-022 |
| TASK-031 | Apple ANE reçetesi: (B,C,1,S) düzeni, split/concat parçalama, reshape/transpose minimizasyonu | 🤝 | L | PENDING | TASK-030 |
| TASK-032 | Kazanç ölçümü — graf boyutu, yükleme süresi, gecikme, doğruluk ödünleşimi | 🤝 | M | PENDING | TASK-031 |
| TASK-033 | Bölüm 4.5 (AS4) yazımı. Kazanç yoksa "denenen yaklaşım ve neden yetersiz kaldığı" olarak yaz | 🤖 | M | PENDING | TASK-032 |

---

## Faz 5: Yazım ve toparlama *(Hafta 14-16)*

| ID | Task | Kim | Complexity | Status | Dependencies |
|----|------|-----|-----------|--------|-------------|
| TASK-034 | Bölüm 5 (Tartışma) + Bölüm 6 (Sonuç). Eksik ablasyonları tamamla | 🤖 | L | PENDING | TASK-033 |
| TASK-035 | Bölüm 1 (Giriş) + Özet/Abstract — **en son yazılır** | 🤖 | M | PENDING | TASK-034 |
| TASK-036 | Kaynakça düzeni (60-80 kaynak) + Ekler A-D | 🤖 | M | PENDING | TASK-034 |
| TASK-037 | Bütünsel okuma, tutarlılık kontrolü, şekil/tablo numaralandırma, dil düzeltmesi | 🤝 | L | PENDING | TASK-035, TASK-036 |
| TASK-038 | Kod deposunu temizle ve yayınla (tekrarlanabilirlik paketi) | 🤖 | M | PENDING | TASK-037 |

---

## Sürekli işler (her hafta, task ID'siz)

- [Awesome-Vision-Mamba](https://github.com/ReaFly/Awesome-Vision-Mamba) kontrolü — alan hızlı, çakışma riskini erken gör
- Ölçüm sonuçlarını **ham halde** sakla (`results/raw/`), asla sadece grafiği tutma
- Ortam sürümlerini her ölçümde kaydet — tekrarlanabilirlik için
- Haftalık 1 saat: o haftanın sonuçlarını tez metnine yaz. **Biriktirme.**

## Altın kurallar

1. Her hafta **tek bir somut çıktı**: bir tablo, bir grafik, bir çalışan betik. "Okudum, düşündüm" çıktı değildir.
2. Bir grafik **üretildiği hafta** yorumlanır. Üç ay sonra kendi grafiğini hatırlamazsın.
3. Başarısız export, çalışmayan derleme, desteklenmeyen operatör — **hepsi bulgudur.** Sil değil, kaydet.
