# Tez İskeleti

**Başlık:** Teoriden Silikona: Durum-Uzayı Tabanlı Görü Omurgalarının Uç Cihazlarda Gerçekleşen Verimliliğinin Ampirik Analizi

**İngilizce başlık:** From Theory to Silicon: An Empirical Analysis of the Realized Efficiency of State-Space Vision Backbones on Edge Devices

**Hedef uzunluk:** 70-90 sayfa (ana metin), YL tezi standardı

---

## Tezin tek cümlelik iddiası

> Durum-uzayı tabanlı görü omurgalarının literatürde bildirilen verimlilik avantajı, özel CUDA çekirdeklerine bağımlıdır ve genel amaçlı dağıtım yığınlarında (ONNX Runtime, CoreML/ANE) büyük ölçüde kaybolur; bu kaybın kaynağı `selective scan` işleminin ardışık yapısı ile graf derleyicilerinin yürütme modeli arasındaki uyumsuzluktur.

Bu iddia **doğrulanırsa** → mimari seçimi için dağıtım-farkındalıklı bir kılavuz.
Bu iddia **çürütülürse** → SSM'lerin uç cihaz olgunluğunun kanıtı, yine değerli bir sonuç.

**Her iki durumda da tez yazılır.** Konunun seçilme sebebi budur.

---

## Araştırma soruları

| Kod | Soru | Hangi fazda cevaplanır |
|---|---|---|
| **AS1** | Eşit doğruluk bütçesinde SSM / ViT / CNN omurgalarının yüksek çözünürlüklü semantik segmentasyondaki gerçek gecikme, bellek tepe noktası ve enerji profili nedir? | Faz 1-2 |
| **AS2** | Teorik FLOPs avantajı ile ölçülen duvar-saati gecikmesi arasındaki fark, dağıtım yığınına (PyTorch/CUDA → ONNX Runtime → CoreML/ANE) göre nasıl değişir? Avantaj nerede buharlaşır? | Faz 2 |
| **AS3** | Mevcut PTQ yöntemleri sınıflandırmadan yoğun tahmine taşındığında doğruluk kaybı nasıl davranır? Yüksek çözünürlük aykırı değer profilini değiştirir mi? | Faz 3 |
| **AS4** | `selective scan`'in derleyici-dostu yeniden formülasyonu (parçalı/paralel tarama, sabit uzunluklu bloklar) ONNX `Loop` darboğazını ne kadar kapatır? | Faz 4 |

**AS1-AS3 ölçüm, AS4 katkı.** İlk üçü kesin sonuç verir; AS4 riskli ama kısmi başarı bile raporlanabilir.

---

## İçindekiler

### ÖZET / ABSTRACT

### 1. GİRİŞ *(~8 sayfa)*
- 1.1 Problem tanımı — verimlilik iddiaları ile dağıtım gerçekliği arasındaki uçurum
- 1.2 Motivasyon — uç cihazda görü modellerinin artan önemi
- 1.3 Araştırma soruları ve hipotezler
- 1.4 Tezin katkıları
  - Dört dağıtım yığını × üç mimari ailesi × iki donanım için sistematik verimlilik ölçümü
  - Teorik-gerçekleşen verimlilik farkının nicel karakterizasyonu
  - PTQ yöntemlerinin yoğun tahmine transfer edilebilirliğinin analizi
  - `selective scan` için derleyici-dostu yeniden formülasyon ve ANE uyarlaması
  - Açık kaynak ölçüm altyapısı ve tekrarlanabilir kıyaslama paketi
- 1.5 Tezin organizasyonu

### 2. KURAMSAL TEMELLER VE İLGİLİ ÇALIŞMALAR *(~18 sayfa)*
- 2.1 Durum-uzayı modelleri
  - 2.1.1 S4'ten Mamba'ya: seçicilik mekanizması
  - 2.1.2 Donanım-farkındalıklı paralel tarama — **ve CUDA bağımlılığı**
  - 2.1.3 Mamba-2 ve durum-uzayı ikiliği (SSD)
- 2.2 Görü için SSM omurgaları
  - 2.2.1 Vim, VMamba ve tarama stratejileri
  - 2.2.2 Hibrit yaklaşımlar (MambaVision, VCMamba) — saf SSM'in sınırları
  - 2.2.3 Yoğun tahmin görevlerinde SSM
- 2.3 Karşılaştırma mimarileri: ViT, Swin, ConvNeXt
- 2.4 Model nicemleme
  - 2.4.1 Genel PTQ/QAT çerçevesi
  - 2.4.2 SSM'e özgü nicemleme zorlukları: token-bazlı varyans, kanal aykırı değerleri, uzun kuyruklu aktivasyonlar
  - 2.4.3 Dinamik aykırı değer problemi
- 2.5 Dağıtım yığınları ve derleyiciler
  - 2.5.1 ONNX ve graf temsili; `Loop` operatörü
  - 2.5.2 TensorRT ve çekirdek füzyonu
  - 2.5.3 CoreML ve Apple Neural Engine mimarisi
  - 2.5.4 Derleyici-model uyumsuzluğu problemi
- 2.6 **Literatürdeki boşluk ve tezin konumu**

### 3. YÖNTEM *(~14 sayfa)*
- 3.1 Deneysel tasarım ve değişken kontrolü
  - Sabit tutulanlar: segmentasyon başlığı, eğitim reçetesi, veri artırma, epoch sayısı
  - Değişkenler: omurga ailesi, dağıtım yığını, nicemleme seviyesi, giriş çözünürlüğü, donanım
- 3.2 Model seçimi ve doğruluk eşitleme protokolü
  - Neden "eşit parametre" değil, **"eşit doğruluk"** karşılaştırması
- 3.3 Veri kümeleri ve görevler
  - ADE20K (semantik segmentasyon, 150 sınıf) — birincil
  - Cityscapes (yüksek çözünürlük, 1024×2048) — çözünürlük ölçeklendirme deneyi
  - ImageNet-1k alt kümesi — sınıflandırma referansı ve PTQ kalibrasyonu
- 3.4 Dağıtım yığınları
  - PyTorch + özel CUDA çekirdeği (referans / üst sınır)
  - PyTorch derlenmiş (`torch.compile`)
  - ONNX Runtime (CPU, CUDA EP)
  - CoreML (CPU / GPU / ANE)
- 3.5 **Ölçüm protokolü** *(bilimsel geçerliliğin kalbi)*
  - 3.5.1 Isınma ve termal stabilizasyon (3 W / 30 sn / 65 °C)
  - 3.5.2 Senkronizasyon ve zamanlama doğruluğu
  - 3.5.3 İş parçacığı ve sistem gürültüsü izolasyonu
  - 3.5.4 Enerji ölçümü: NVML ve `powermetrics`
  - 3.5.5 İstatistiksel raporlama: medyan, ortalama, std, P99
  - 3.5.6 **Yürütme yeri doğrulaması** — modelin gerçekten ANE'de çalıştığının Xcode profili ile kanıtlanması
- 3.6 Nicemleme protokolü (PTQ4VM, QMamba uyarlamaları)
- 3.7 Tekrarlanabilirlik: tohum kontrolü, sürüm sabitleme, ortam belgeleme

### 4. DENEYSEL SONUÇLAR *(~22 sayfa)*
- 4.1 Doğruluk eşitleme sonuçları — omurgalar aynı çizgiye getirildi mi?
- 4.2 **AS1:** Referans yığında verimlilik profilleri
  - Gecikme, bellek tepe noktası, enerji; çözünürlük ölçeklendirme eğrileri
- 4.3 **AS2:** Dağıtım yığını matrisi — avantajın buharlaşması
  - 4.3.1 Yığın-başına gecikme karşılaştırması
  - 4.3.2 Teorik FLOPs vs ölçülen gecikme sapması
  - 4.3.3 Operatör seviyesinde darboğaz analizi (`Loop` yükü, bellek kopyaları)
  - 4.3.4 ANE'de yürütme oranı analizi
- 4.4 **AS3:** Nicemlemenin yoğun tahmine transferi
  - 4.4.1 Sınıflandırma vs segmentasyon doğruluk kaybı
  - 4.4.2 Çözünürlüğün aykırı değer dağılımına etkisi
  - 4.4.3 Nicemleme sonrası gecikme kazancının yığına göre değişimi
- 4.5 **AS4:** Yeniden formülasyon sonuçları
  - 4.5.1 Parçalı tarama: graf boyutu ve yükleme süresi
  - 4.5.2 ANE-dostu tensör düzeni uyarlaması
  - 4.5.3 Kazanç-doğruluk ödünleşimi
- 4.6 Ablasyon çalışmaları
- 4.7 Sonuçların özet sentezi ve karar tablosu

### 5. TARTIŞMA *(~10 sayfa)*
- 5.1 Bulguların yorumu: mimari verimliliği bir donanım-yazılım eşleşmesi problemidir
- 5.2 "Teorik verimlilik" metriklerinin (FLOPs, parametre sayısı) yetersizliği
- 5.3 Uygulayıcılar için mimari seçim kılavuzu
- 5.4 Sonuçların genellenebilirliği ve sınırlılıklar
  - Tek GPU sınıfı, tek NPU ailesi, sınırlı model ölçeği
- 5.5 Tehditler ve karşı-argümanlar
  - "Araç zincirleri olgunlaşırsa bu bulgular geçersizleşir mi?" — **bu soruya dürüst cevap ver**

### 6. SONUÇ VE GELECEK ÇALIŞMALAR *(~5 sayfa)*
- 6.1 Araştırma sorularına cevaplar
- 6.2 Katkıların özeti
- 6.3 Gelecek çalışmalar

### KAYNAKLAR *(60-80 kaynak hedefi)*

### EKLER
- EK A: Tam deney sonuç tabloları
- EK B: Ölçüm altyapısı kodu ve kullanımı
- EK C: Ortam ve sürüm belgelemesi
- EK D: Yeniden formüle edilmiş tarama operatörünün implementasyonu

---

## Yazım stratejisi

**Bölümleri sırayla yazma.** Önerilen sıra:

1. **Bölüm 3 (Yöntem)** — deneyler başlamadan yazılır. Deney tasarımını yazmak, tasarımdaki hatayı bulmanın en ucuz yoludur.
2. **Bölüm 2 (Kuramsal temeller)** — literatür okurken paralel yazılır, biriktirilmez
3. **Bölüm 4 (Sonuçlar)** — her faz bittiğinde ilgili alt bölüm yazılır, sonuna bırakılmaz
4. **Bölüm 5, 6** — sonuçlar tamamlanınca
5. **Bölüm 1 (Giriş)** — **en son yazılır.** Ne bulduğunu bilmeden giriş yazmak, iki kez yazmak demektir
6. **Özet** — en son

**Altın kural:** Bir grafik üretildiği hafta yorumlanır. Üç ay sonra kendi grafiğini hatırlamazsın.
