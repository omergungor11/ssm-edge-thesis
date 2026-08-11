# Faz 4: Yeniden Formülasyon *(Hafta 12-13)* ← **Özgün katkı, RİSKLİ**

> **Bu fazı asla Faz 1-3'ün önüne alma.**
> Faz 4 tamamen başarısız olsa bile Faz 1-3 tek başına bir tez oluşturur. Sırayı bozmak, tezi tek bir bahse bağlamaktır.

> **Risk yönetimi:** Beklenen kazanç çıkmazsa Bölüm 4.5 *"denenen yaklaşım ve neden yetersiz kaldığı"* olarak yazılır. Tez ayakta kalır.

---

## TASK-030: `selective scan`'in Derleyici-Dostu Yeniden Formülasyonu

**Kim**: 🤝 Ortak · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-022

### Açıklama
Problem: `selective scan` ardışık → ONNX'te `Loop` düğümü → yorumlayıcı yükü → çöküş.
Fikir: Mamba-2'nin **durum-uzayı ikiliğinden (SSD)** yararlanarak taramayı parçalı/sabit-blok forma çevir; `Loop` yerine **matris çarpımı ağırlıklı** bir graf üret.

### Kabul kriterleri
- [ ] Parçalı/sabit-blok tarama implementasyonu (PyTorch)
- [ ] Referans implementasyonla **sayısal denklik** doğrulandı (tolerans belirtilmiş)
- [ ] Blok uzunluğu bir hiperparametre — tarama yapıldı
- [ ] ONNX export sonrası: `Loop` düğüm sayısı, toplam düğüm sayısı, graf boyutu **öncesi/sonrası**
- [ ] Yükleme süresi karşılaştırması

---

## TASK-031: ANE-Dostu Tensör Düzeni Uyarlaması

**Kim**: 🤝 Ortak · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-030

### Açıklama
Apple'ın ANE optimizasyon reçetesi **Transformer için** yazılmış; SSM için karşılığı yok. Bu tezin özgün katkısının ikinci ayağı bu boşluk.

### Kabul kriterleri
- [ ] **(B, C, 1, S) veri düzeni** uygulandı
- [ ] Ara tensörler split/concat ile parçalandı (ANE bellek kısıtı)
- [ ] `reshape` / `transpose` sayısı minimize edildi — öncesi/sonrası sayıldı
- [ ] Desteklenmeyen operatörler desteklenenlerle değiştirildi
- [ ] Her adımın ANE yürütme oranına etkisi ayrı ayrı ölçüldü (ablasyon)

---

## TASK-032: Kazanç Ölçümü

**Kim**: 🤝 Ortak · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-031

### Kabul kriterleri
- [ ] Gecikme: yeniden formüle edilmiş vs orijinal, tüm yığınlarda
- [ ] ANE yürütme oranı: öncesi/sonrası (Xcode ile doğrulanmış)
- [ ] Doğruluk ödünleşimi — mIoU kaybı var mı?
- [ ] **Kazanç tablosu** üretildi
- [ ] `Loop` darboğazının ne kadarının kapandığı sayısal olarak ifade edildi

---

## TASK-033: Bölüm 4.5 (AS4) Yazımı

**Kim**: 🤖 Claude · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-032

### Kabul kriterleri
- [ ] 4.5.1 Parçalı tarama: graf boyutu ve yükleme süresi
- [ ] 4.5.2 ANE-dostu tensör düzeni uyarlaması
- [ ] 4.5.3 Kazanç-doğruluk ödünleşimi
- [ ] Kazanç yetersizse: **dürüst negatif sonuç bölümü** — ne denendi, neden yetmedi, ne öğrenildi
