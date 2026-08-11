# Faz 1: Doğruluk Eşitleme *(Hafta 3-5)*

> Amaç: omurgaları **aynı doğruluk çizgisine** getirmek.
> Farklı mIoU'daki modellerin hızını karşılaştırmak bilimsel olarak anlamsızdır — Faz 2'nin tüm geçerliliği buna dayanıyor.

**Metodolojik ilke:** "Eşit parametre" değil **"eşit doğruluk"** karşılaştırması. Gerekçe → `tez-iskeleti.md` §3.2.

---

## TASK-013: Segmentasyon Boru Hattı

**Kim**: 🤖 Claude · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-011

### Kabul kriterleri
- [ ] UPerNet başlığı — **tüm omurgalar için sabit**
- [ ] Eğitim reçetesi dondurulur: optimizer, lr programı, epoch, veri artırma, kayıp fonksiyonu
- [ ] Omurga-değiştirilebilir arayüz (aynı kod, farklı backbone)
- [ ] AMP + gradient checkpointing varsayılan açık (12 GB kısıtı)
- [ ] mIoU değerlendirme kodu doğrulandı
- [ ] Tohum kontrolü + deterministik mod

---

## TASK-014 — TASK-017: Omurga Eğitimleri

**Kim**: 🤝 Ortak · **Complexity**: L (017: M) · **Status**: PENDING · **Dependencies**: TASK-013

| ID | Omurga | Aile |
|---|---|---|
| TASK-014 | VMamba-T | SSM |
| TASK-015 | ViT-S / DeiT-S | Transformer |
| TASK-016 | ConvNeXt-T | CNN |
| TASK-017 | EfficientViM | Verimli SSM |

### Her biri için kabul kriterleri
- [ ] ADE20K, 512×512, **TASK-013'teki reçeteyle birebir aynı**
- [ ] Eğitim logu + eğri kaydedildi
- [ ] Nihai mIoU raporlandı
- [ ] Ağırlıklar `checkpoints/` altında, sürümlü
- [ ] Eğitim süresi ve tepe VRAM kullanımı kaydedildi

> **Tek bir hiperparametreyi bir omurga için değiştirirsen, hepsi için değiştir.** Aksi halde Faz 2 çöker.

---

## TASK-018: Tablo 4.1 — Doğruluk Eşitleme

**Kim**: 🤝 Ortak · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-014..017

### Kabul kriterleri
- [ ] Dört omurga × (parametre, GFLOPs, mIoU, eğitim süresi) tablosu
- [ ] mIoU'lar karşılaştırılabilir aralıkta mı? Değilse epoch/lr ile hizala
- [ ] Hizalama için yapılan her müdahale **belgelendi** (gizlenmez)
- [ ] Tablo tez formatında, Bölüm 4.1'e yerleştirildi

---

## TASK-019: Bölüm 2 (Kuramsal Temeller) Yazımı

**Kim**: 🤖 Claude · **Complexity**: L · **Status**: PENDING · **Dependencies**: -

### Açıklama
Literatür okunurken **paralel** yazılır, sona biriktirilmez. Kaynak: `tez-docs/literatur-taramasi.md`.

### Kabul kriterleri
- [ ] 2.1 Durum-uzayı modelleri (S4 → Mamba, seçicilik, **CUDA bağımlılığı**, Mamba-2/SSD)
- [ ] 2.2 Görü için SSM omurgaları (Vim, VMamba, hibritler, yoğun tahmin)
- [ ] 2.3 Karşılaştırma mimarileri (ViT, Swin, ConvNeXt)
- [ ] 2.4 Nicemleme (PTQ/QAT, SSM'e özgü zorluklar, dinamik aykırı değer)
- [ ] 2.5 Dağıtım yığınları (ONNX `Loop`, TensorRT, CoreML/ANE, uyumsuzluk problemi)
- [ ] 2.6 **Literatürdeki boşluk ve tezin konumu**
- [ ] ~18 sayfa

---

## Faz 1 riskleri

| Risk | Çıkış yolu |
|---|---|
| 12 GB eğitim için yetmiyor | Dondurulmuş ImageNet ön-eğitimli omurga + sadece başlık eğitimi. Doğruluk düşer, karşılaştırma geçerliliği korunur |
| 512×512 sığmıyor | 448×448'e in — **tüm omurgalar için aynı** olmak kaydıyla |
| Bir omurga diğerlerinden çok geride kalıyor | Reçeteyi değiştirme; farkı raporla. Bu da bir bulgudur |
