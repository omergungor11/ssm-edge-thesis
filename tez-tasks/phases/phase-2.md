# Faz 2: Dağıtım Yığını Matrisi *(Hafta 6-8)* ← **Tezin kalbi**

> **Bu fazın sonunda tezin ana iddiası kanıtlanmış ya da çürütülmüş olur.**
> AS1 ve AS2 burada cevaplanıyor.

---

## TASK-020: Export Matrisi (4 omurga × 4 yığın)

**Kim**: 🤝 Ortak · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-018

### Matris

| | PyTorch/CUDA | `torch.compile` | ONNX Runtime | CoreML |
|---|---|---|---|---|
| VMamba-T | | | | |
| ViT-S | | | | |
| ConvNeXt-T | | | | |
| EfficientViM | | | | |

### Kabul kriterleri
- [ ] 16 hücrenin her biri denendi
- [ ] Başarılı export'lar: graf boyutu, düğüm sayısı, yükleme süresi
- [ ] **Başarısız export'lar: tam hata + hangi operatörde takıldığı + neden**
- [ ] Sayısal denklik kontrolü — export edilen model referansla aynı çıktıyı veriyor mu? (tolerans belirlenmiş)
- [ ] `tez-docs/export-matrisi.md` tamamlandı

> **Başarısızlık da veridir.** Boş hücre yok — her hücre ya sayı ya gerekçe içerir.

---

## TASK-021: Tam Ölçüm + Çözünürlük Taraması

**Kim**: 🤝 Ortak · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-020, TASK-010

### Kabul kriterleri
- [ ] Her başarılı hücre için: gecikme (medyan/std/P99), bellek tepesi, enerji
- [ ] Çözünürlük taraması: **256 / 512 / 768 / 1024**
- [ ] Her iki donanımda (RTX 5070, M5 Pro) uygulanabilir olanlar ölçüldü
- [ ] Ham sonuçlar `results/raw/` altında, ortam sürümleriyle birlikte
- [ ] **Ana sonuç tablosu ve eğrileri** üretildi

### Beklenen kilit grafik
Çözünürlük artarken SSM'in teorik lineer avantajı hangi yığında görünüyor, hangisinde kayboluyor?

---

## TASK-022: Operatör Seviyesi Darboğaz Analizi

**Kim**: 🤝 Ortak · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-021

### Açıklama
**Tezin "neden" cevabı.** "Yavaş" demek yetmez; nerede ve niçin yavaş olduğu gösterilmeli.

### Kabul kriterleri
- [ ] ONNX Runtime profili: operatör-başına süre dökümü
- [ ] `Loop` operatörünün toplam süredeki payı — omurga ve çözünürlük başına
- [ ] Bellek kopyası / düzen değişimi (transpose, reshape) maliyeti
- [ ] Kaçırılan çekirdek füzyonları
- [ ] PyTorch/CUDA referansında aynı analiz — fark nereden geliyor?
- [ ] Darboğaz sıralaması (Pareto): ilk 3 maliyet kalemi

---

## TASK-023: ANE Yürütme Oranı Doğrulaması

**Kim**: 🧑 **Sadece Ömer** (Xcode gerekiyor) · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-021

### Açıklama
CoreML derleyicisi neyin nerede çalışacağına **şeffaf olmayan** kararlar veriyor. "ANE'de çalıştırdım" demek yetmez — gerçekten ANE'de mi çalıştığı kanıtlanmalı.

### Kabul kriterleri
- [ ] Xcode Core ML Performance Report her model için çalıştırıldı
- [ ] Katman-başına yürütme yeri (ANE / GPU / CPU) tablolandı
- [ ] **ANE yürütme oranı** yüzdesi hesaplandı
- [ ] ANE'ye düşmeyen operatörler ve gerekçeleri listelendi
- [ ] Ekran görüntüleri / rapor çıktıları EK C için saklandı

---

## TASK-024: Bölüm 4.2 + 4.3 Yazımı

**Kim**: 🤖 Claude · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-022, TASK-023

### Kabul kriterleri
- [ ] 4.2 (AS1): Referans yığında verimlilik profilleri, çözünürlük eğrileri
- [ ] 4.3.1 Yığın-başına gecikme karşılaştırması
- [ ] 4.3.2 **Teorik FLOPs vs ölçülen gecikme sapması**
- [ ] 4.3.3 Operatör seviyesi darboğaz analizi
- [ ] 4.3.4 ANE yürütme oranı analizi
- [ ] Her şekil/tablo numaralı ve metinde referanslı
