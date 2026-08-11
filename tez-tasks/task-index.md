# Task Index

> **Tez:** Teoriden Silikona — Durum-Uzayı Tabanlı Görü Omurgalarının Uç Cihazlarda Gerçekleşen Verimliliğinin Ampirik Analizi
> **Başlangıç:** 10 Ağustos 2026 · **Hedef bitiş:** 30 Kasım 2026 (16 hafta)
> Kaynak plan: [`tez-plans/deney-plani.md`](../tez-plans/deney-plani.md) · İskelet: [`tez-plans/tez-iskeleti.md`](../tez-plans/tez-iskeleti.md)

## Dashboard

| Faz | Ad | Hafta | Total | Done | In Progress | Pending | Blocked |
|-----|-----|-------|-------|------|-------------|---------|---------|
| 0 | Öncül doğrulama + ölçüm altyapısı | 1-2 | 12 | 2 | 0 | 10 | 0 |
| 1 | Doğruluk eşitleme | 3-5 | 7 | 0 | 0 | 7 | 0 |
| 2 | Dağıtım yığını matrisi ← *kalp* | 6-8 | 5 | 0 | 0 | 5 | 0 |
| 3 | Nicemlemenin yoğun tahmine transferi | 9-11 | 5 | 0 | 0 | 5 | 0 |
| 4 | Yeniden formülasyon ← *riskli* | 12-13 | 4 | 0 | 0 | 4 | 0 |
| 5 | Yazım ve toparlama | 14-16 | 5 | 0 | 0 | 5 | 0 |
| **Total** | | | **38** | **0** | **0** | **38** | **0** |

**Progress**: 2/38 (0%) — Faz 0, Hafta 1

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
| TASK-003 | RTX 5070 ortamı: `nvidia-smi` VRAM doğrulama + CUDA/torch + **`mamba-ssm` derleme testi** | 🧑 | M | PENDING | - |
| TASK-004 | VMamba-T referans gecikme ölçümü (PyTorch + özel CUDA çekirdeği) | 🤝 | M | PENDING | TASK-003 |
| TASK-005 | ONNX export denemesi — hata mesajları, graf boyutu, yükleme süresi belgelenir | 🤝 | M | PENDING | TASK-004 |
| TASK-006 | ONNX Runtime ölçümü (CPU + CUDA EP) → **ilk yavaşlama oranı** | 🤝 | M | PENDING | TASK-005 |
| TASK-007 | CoreML export + M5 Pro / ANE ölçümü | 🤝 | M | PENDING | TASK-002, TASK-005 |
| TASK-008 | **KARAR NOKTASI** — `tez-docs/oncul-dogrulama.md`: öncül doğrulandı mı? | 🤝 | S | PENDING | TASK-006, TASK-007 |
| TASK-009 | Ölçüm harness'ı: ısınma, senkronizasyon, termal bekleme, NVML/`powermetrics`, istatistik (medyan/std/P99) | 🤖 | L | PENDING | TASK-008 |
| TASK-010 | Harness doğrulaması — ResNet-50 ile bilinen sayılar üretiliyor mu? | 🤝 | M | PENDING | TASK-009 |
| TASK-011 | Veri kümesi boru hattı: ADE20K, Cityscapes, ImageNet-1k alt kümesi | 🤖 | M | PENDING | TASK-002 |
| TASK-012 | Bölüm 3.5 (Ölçüm protokolü) taslağı — ~4 sayfa | 🤖 | M | PENDING | TASK-009 |

> **Karar noktası (TASK-008):** Yavaşlama gözlemlendi mi? **Evet** → devam. **Hayır** → konuyu ANE/CoreML eksenine daralt (bkz. `deney-plani.md` çıkış yolları). Bu bir başarısızlık değil, ucuz bir keşif.

---

## Faz 1: Doğruluk eşitleme *(Hafta 3-5)*

> Farklı doğruluktaki modellerin hızını karşılaştırmak anlamsızdır. Önce aynı çizgiye getir.

| ID | Task | Kim | Complexity | Status | Dependencies |
|----|------|-----|-----------|--------|-------------|
| TASK-013 | Segmentasyon boru hattı (UPerNet başlığı sabit, reçete dondurulur) | 🤖 | L | PENDING | TASK-011 |
| TASK-014 | VMamba-T'yi ADE20K'da eğit (512×512, AMP + gradient checkpointing) | 🤝 | L | PENDING | TASK-013 |
| TASK-015 | ViT-S / DeiT-S'i **birebir aynı reçeteyle** eğit | 🤝 | L | PENDING | TASK-013 |
| TASK-016 | ConvNeXt-T'yi **birebir aynı reçeteyle** eğit | 🤝 | L | PENDING | TASK-013 |
| TASK-017 | EfficientViM ekle (verimli SSM referansı) | 🤝 | M | PENDING | TASK-013 |
| TASK-018 | **Tablo 4.1 — doğruluk eşitleme.** Gerekirse epoch/lr ile çizgiyi hizala | 🤝 | M | PENDING | TASK-014..017 |
| TASK-019 | Bölüm 2 (Kuramsal temeller) yazımı — literatür okurken paralel, biriktirme | 🤖 | L | PENDING | - |

**12 GB notu:** AMP + gradient checkpointing zorunlu, batch 8-16. Sığmazsa 448×448'e in — bilimsel geçerlilik bozulmaz, yeter ki **tüm omurgalar için aynı** olsun.
**Risk:** 12 GB eğitime yetmezse → dondurulmuş ImageNet ön-eğitimli omurga + sadece başlık eğitimi.

---

## Faz 2: Dağıtım yığını matrisi *(Hafta 6-8)* ← **Tezin kalbi**

> Bu fazın sonunda tezin ana iddiası kanıtlanmış ya da çürütülmüş olur.

| ID | Task | Kim | Complexity | Status | Dependencies |
|----|------|-----|-----------|--------|-------------|
| TASK-020 | Export matrisi: 4 omurga × 4 yığın (PyTorch/CUDA, `torch.compile`, ONNX Runtime, CoreML). **Başarısız export'lar da belgelenir — başarısızlık da veridir** | 🤝 | L | PENDING | TASK-018 |
| TASK-021 | Tam gecikme/bellek/enerji ölçümü + çözünürlük taraması (256/512/768/1024) | 🤝 | L | PENDING | TASK-020, TASK-010 |
| TASK-022 | Operatör seviyesi profilleme: `Loop` yükü, bellek kopyaları, füzyon kaçırmaları | 🤝 | L | PENDING | TASK-021 |
| TASK-023 | **ANE yürütme oranı doğrulaması** (Xcode profili) — model gerçekten ANE'de mi çalışıyor? | 🧑 | M | PENDING | TASK-021 |
| TASK-024 | Bölüm 4.2 (AS1) + 4.3 (AS2) yazımı | 🤖 | L | PENDING | TASK-022, TASK-023 |

---

## Faz 3: Nicemlemenin yoğun tahmine transferi *(Hafta 9-11)*

| ID | Task | Kim | Complexity | Status | Dependencies |
|----|------|-----|-----------|--------|-------------|
| TASK-025 | PTQ4VM implementasyonu — sınıflandırma referansıyla doğrula, makaledeki sayıları üret | 🤝 | L | PENDING | TASK-021 |
| TASK-026 | Aynı PTQ'yu segmentasyon modellerine uygula, doğruluk kaybını karşılaştır | 🤝 | L | PENDING | TASK-025 |
| TASK-027 | Çözünürlüğün aykırı değer dağılımına etkisi (aktivasyon histogramları, kanal istatistikleri) | 🤝 | M | PENDING | TASK-026 |
| TASK-028 | Nicemlenmiş modelleri tekrar dört yığında ölç | 🤝 | M | PENDING | TASK-026 |
| TASK-029 | Bölüm 4.4 (AS3) yazımı | 🤖 | M | PENDING | TASK-027, TASK-028 |

---

## Faz 4: Yeniden formülasyon *(Hafta 12-13)* ← **Özgün katkı, riskli**

> **Bu fazı asla Faz 1-3'ün önüne alma.** Faz 4 tamamen başarısız olsa bile Faz 1-3 tek başına bir tez oluşturur.

| ID | Task | Kim | Complexity | Status | Dependencies |
|----|------|-----|-----------|--------|-------------|
| TASK-030 | `selective scan`'i parçalı/sabit-blok forma yeniden yaz (Mamba-2 SSD ikiliğinden yararlan) | 🤝 | L | PENDING | TASK-022 |
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
