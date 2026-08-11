# Faz 3: Nicemlemenin Yoğun Tahmine Transferi *(Hafta 9-11)*

> AS3: Mevcut PTQ yöntemleri sınıflandırmadan yoğun tahmine taşındığında doğruluk kaybı nasıl davranır?
> **Yeni nicemleme algoritması icat etmiyoruz** — o kapı kapalı (PTQ4VM, QMamba, OuroMamba, Ternary Mamba). Mevcut yöntemleri *taşıyoruz*.

---

## TASK-025: PTQ4VM Implementasyonu + Doğrulama

**Kim**: 🤝 Ortak · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-021

### Kabul kriterleri
- [ ] PTQ4VM uygulandı (token-bazlı varyans, kanal aykırı değer ele alımı)
- [ ] **Makaledeki sınıflandırma sayıları üretilebiliyor** — yöntem doğru mu?
- [ ] Sapma varsa sebebi bulundu ve belgelendi
- [ ] Kalibrasyon kümesi ve prosedürü sabitlendi (ImageNet-1k alt kümesi)

> Kendi implementasyonun makalenin sayısını tutturmuyorsa, segmentasyon sonuçların da güvenilmez. Bu adımı atlama.

---

## TASK-026: PTQ'nun Segmentasyona Uygulanması

**Kim**: 🤝 Ortak · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-025

### Kabul kriterleri
- [ ] Aynı PTQ, Faz 1'de eğitilen dört segmentasyon modeline uygulandı
- [ ] mIoU kaybı ölçüldü (FP32 → INT8)
- [ ] **Sınıflandırma kaybı vs segmentasyon kaybı** karşılaştırma tablosu
- [ ] Omurga ailesine göre kayıp farkı analiz edildi (SSM daha mı kırılgan?)

### Beklenen bulgu
Yoğun tahmin, sınıflandırmadan daha hassas olmalı — piksel-başına karar, hata birikimi. Ne kadar?

---

## TASK-027: Çözünürlük × Aykırı Değer Analizi

**Kim**: 🤝 Ortak · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-026

### Kabul kriterleri
- [ ] Aktivasyon histogramları — çözünürlük başına (256/512/768/1024)
- [ ] Kanal-bazlı istatistikler (maks/varyans/kurtosis)
- [ ] Aykırı değerlerin çözünürlükle nasıl kaydığı gösterildi
- [ ] OuroMamba'nın "dinamik aykırı değer" bulgusuyla karşılaştırma
- [ ] Statik PTQ'nun yüksek çözünürlükte neden bozulduğuna dair mekanizma açıklaması

---

## TASK-028: Nicemlenmiş Modellerin Yığın Ölçümü

**Kim**: 🤝 Ortak · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-026

### Kabul kriterleri
- [ ] Nicemlenmiş modeller dört yığında yeniden ölçüldü
- [ ] **Nicemleme kazancı yığına göre değişiyor mu?** (INT8 desteği her yığında aynı değil)
- [ ] CoreML'de INT8/palettization davranışı ayrıca incelendi
- [ ] Kazanç-kayıp Pareto eğrisi çizildi

---

## TASK-029: Bölüm 4.4 (AS3) Yazımı

**Kim**: 🤖 Claude · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-027, TASK-028

### Kabul kriterleri
- [ ] 4.4.1 Sınıflandırma vs segmentasyon doğruluk kaybı
- [ ] 4.4.2 Çözünürlüğün aykırı değer dağılımına etkisi
- [ ] 4.4.3 Nicemleme sonrası gecikme kazancının yığına göre değişimi
