# 4.1 Model Kartları ve Doğruluk Doğrulaması *(TASLAK v1 — 12 Ağustos 2026, TASK-018)*

Bu bölüm, deney matrisine giren omurgaların kimliklerini ve yayınlanmış doğruluklarının
bu tezin donanımında bağımsız olarak yeniden üretildiğini belgeler. Tüm modeller aynı
segmentasyon başlığını (UPerNet, 512 kanal — üç modelde de birebir 31.5M parametre) ve
aynı eğitim reçetesini (mmseg 160k iterasyon, ADE20K 512×512) kullanır; değişen tek
bileşen omurgadır. Bu, Bölüm 3.1'deki değişken kontrolü ilkesinin uygulamasıdır.

## Tablo 4.1 — Model kartları ve mIoU doğrulaması

| Omurga | Aile | Omurga param | Başlık param | Bildirilen mIoU | **Doğrulanan mIoU** | Fark |
|---|---|---|---|---|---|---|
| VMamba-T (v2seg) | SSM | 29.9M | 31.5M | 48.3 | **48.33** | +0.03 |
| Swin-T | Transformer | 27.5M | 31.5M | 44.41 | **44.32** | −0.09 |
| ConvNeXt-T | CNN | 27.8M | 31.5M | 46.11¹ | **45.42** | −0.69¹ |

¹ ConvNeXt-T'nin bildirilen değeri mmseg "slide" (kayan pencere) test protokolüyledir;
bu tezin tüm doğrulamaları tek protokolde ("whole", en-boy korumalı, kısa kenar 512,
/32 yansıma pad'i) yapılmıştır. Kayan pencere değerlendirmesi tipik olarak 0.5-1.0 puan
avantaj sağlar; gözlenen −0.69 bu banttadır. Protokol ayrıntıları: Bölüm 3.5.

**Doğrulama koşulları:** ADE20K val (2 000 görüntü, 150 sınıf), fp32, PyTorch eager,
Apple M5 CPU (saf-torch `selective scan` — özel CUDA çekirdeği yok). Yükleme
kontrolü: üç modelde de checkpoint `missing=0 / unexpected=0` ile, mmseg'e bağımlı
olmayan anahtar-uyumlu implementasyonlara yüklendi (EK B). Ham karışıklık matrisleri:
`results/raw/ade20k_conf_*_n2000.npy`.

## Değerlendirme hızı yan bulgusu

Aynı boru hatta görüntü başına ortalama süreler (CPU, fp32, değerlendirme koşusu):
VMamba-T **~2.7-3.6 s**, Swin-T **~1.4 s**, ConvNeXt-T **~1.6 s**. Henüz kontrollü
gecikme ölçümü değildir (Bölüm 4.2'nin harness'ı ayrı); ancak özel çekirdek yokluğunda
SSM omurgasının ~2× bedel ödediğinin ilk işaretidir ve AS2'nin motivasyonunu kurar.

## EfficientViM'in kapsam dışı bırakılması (TASK-017 kararı)

Verimli-SSM temsilcisi olarak değerlendirilen EfficientViM (CVPR'25), iki nedenle ana
matrise dahil edilmemiştir: (1) yayınlanmış tek ADE20K checkpoint'i (M4-450, mIoU 41.3)
UPerNet değil **Semantic FPN** başlığı kullanır — başlığı sabitleyen deney tasarımını
bozar; (2) eğitim reçetesi (450 epoch ImageNet ön-eğitimi) diğer üçünün 300 epoch
sınıfıyla hizalı değildir. Hibrit/verimli SSM tasarımları Bölüm 2.2.2'de literatür
düzeyinde ele alınmakta; EfficientViM'e Pareto düzleminde literatür-değeri olarak
(kendi bildirilen sayılarıyla, ayrı işaretlenmiş) yer verilmesi Bölüm 4.7'de
değerlendirilecektir.

## Pareto çerçevesi (Bölüm 4.2-4.3'e köprü)

Üç omurga aynı doğrulukta değildir (44.3-48.3 bandı); bu nedenle sonuç bölümleri tekil
"hız" karşılaştırması yerine **doğruluk-gecikme Pareto düzlemi** raporlar: her (omurga ×
dağıtım yığını × çözünürlük) hücresi düzlemde bir noktadır. Bir omurganın "üstünlüğü",
Pareto sınırında yer alıp almadığıyla; SSM iddiası ise VMamba noktalarının yığın
değiştikçe sınırdan ne kadar uzaklaştığıyla test edilir. GFLOPs sütunu (teorik eksen,
AS2) Faz 2 başında fvcore ile eklenecektir — VMamba'nın tarama operatörü fvcore
tarafından sayılamadığından o hücre için literatür değeri + elle hesap kullanılacak
ve ayrıca işaretlenecektir.
