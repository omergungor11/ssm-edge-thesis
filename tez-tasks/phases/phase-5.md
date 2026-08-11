# Faz 5: Yazım ve Toparlama *(Hafta 14-16)*

> Bu faza gelindiğinde Bölüm 2, 3 ve 4 **zaten yazılmış olmalı** (her faz kendi bölümünü yazdı).
> Faz 5 sadece kalanları ve bütünlüğü kapatıyor. Buraya "her şeyi yazacağım" diye gelirsen plan çökmüş demektir.

---

## TASK-034: Bölüm 5 (Tartışma) + Bölüm 6 (Sonuç)

**Kim**: 🤖 Claude · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-033

### Kabul kriterleri
- [ ] 5.1 Mimari verimliliği bir **donanım-yazılım eşleşmesi** problemidir
- [ ] 5.2 "Teorik verimlilik" metriklerinin (FLOPs, parametre sayısı) yetersizliği
- [ ] 5.3 Uygulayıcılar için mimari seçim kılavuzu
- [ ] 5.4 Genellenebilirlik ve sınırlılıklar (tek GPU sınıfı, tek NPU ailesi, sınırlı model ölçeği)
- [ ] 5.5 **"Araç zincirleri olgunlaşırsa bu bulgular geçersizleşir mi?"** — bu soruya dürüst cevap
- [ ] 6.1 Araştırma sorularına (AS1-AS4) tek tek cevaplar
- [ ] 6.2 Katkıların özeti · 6.3 Gelecek çalışmalar
- [ ] Eksik kalan ablasyonlar tamamlandı

---

## TASK-035: Bölüm 1 (Giriş) + Özet/Abstract

**Kim**: 🤖 Claude · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-034

### Açıklama
**En son yazılır.** Ne bulduğunu bilmeden giriş yazmak, iki kez yazmak demektir.

### Kabul kriterleri
- [ ] 1.1 Problem tanımı — verimlilik iddiaları ile dağıtım gerçekliği arasındaki uçurum
- [ ] 1.2 Motivasyon · 1.3 Araştırma soruları ve hipotezler
- [ ] 1.4 Katkılar (5 madde, `tez-iskeleti.md`'deki liste)
- [ ] 1.5 Tezin organizasyonu
- [ ] Özet (TR) + Abstract (EN) — **gerçek sayılar içerir**, genel laf değil

---

## TASK-036: Kaynakça + Ekler

**Kim**: 🤖 Claude · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-034

### Kabul kriterleri
- [ ] 60-80 kaynak, tutarlı stil, hepsi metinde atıflı
- [ ] EK A: Tam deney sonuç tabloları
- [ ] EK B: Ölçüm altyapısı kodu ve kullanımı
- [ ] EK C: Ortam ve sürüm belgelemesi (+ Xcode ANE raporları)
- [ ] EK D: Yeniden formüle edilmiş tarama operatörünün implementasyonu

---

## TASK-037: Bütünsel Okuma ve Tutarlılık Kontrolü

**Kim**: 🤝 Ortak · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-035, TASK-036

### Kabul kriterleri
- [ ] Baştan sona tek oturumda okuma
- [ ] Şekil/tablo numaralandırma ve metin referansları doğru
- [ ] Terminoloji tutarlı (TR karşılıklar sabit — sözlük tutuldu mu?)
- [ ] Bölüm 1'deki iddialar Bölüm 4'teki sonuçlarla **birebir** örtüşüyor
- [ ] Dil düzeltmesi
- [ ] Sayfa hedefi: 70-90 sayfa ana metin

---

## TASK-038: Kod Deposu Temizliği + Yayın

**Kim**: 🤖 Claude · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-037

### Kabul kriterleri
- [ ] README: kurulum, çalıştırma, sonuçların yeniden üretimi
- [ ] Ortam sürümleri sabitlenmiş (`requirements.lock` / `uv.lock`)
- [ ] Ham sonuçlar (`results/raw/`) dahil
- [ ] Ölçüm harness'ı bağımsız kullanılabilir durumda
- [ ] Lisans seçildi
- [ ] Tekrarlanabilirlik: temiz makinede tek komutla en az bir sonuç üretilebiliyor
