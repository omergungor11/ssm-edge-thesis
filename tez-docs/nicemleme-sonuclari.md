# Nicemleme Sonuçları — Faz 3 Özeti *(13 Ağustos 2026, TASK-025..028)*

Ham kayıtlar: `results/raw/quant_matrix.jsonl`, `quant_miou_*.json`, `activation_stats_*.json`,
gecikmeler `latency_matrix_*.jsonl` (quant tag'li). Protokol: Bölüm 3.6 (Aşama I —
kalibrasyonsuz ağırlık-nicemleme).

## Ana tablo (512², CoreML; mIoU: 250-görüntü, kare-512 protokolü — deltalar esastır)

| | fp32-export | W8 linear | W4 palette |
|---|---|---|---|
| ConvNeXt boyut | 118.8 MB | 59.7 MB | 30.0 MB |
| ConvNeXt mIoU | 34.93 | **35.00 (±0)** | 31.08 (**−3.9**) |
| ConvNeXt CoreML ALL | 91.4 ms | **64.7 ms (hızlandı!)** | 59.3 ms |
| Swin boyut | 127.5 MB | 64.1 MB | 32.3 MB |
| Swin mIoU | 36.01 | **36.05 (±0)** | 33.26 (−2.8) |

## ORT dinamik INT8 (CPU)

| | ConvNeXt | Swin | VMamba |
|---|---|---|---|
| Boyut | 237→60 MB | 239→63 MB | **858→715 MB (yalnız 1.2×)** |
| Gecikme | **10 863 ms (17× YAVAŞ)** | 7 094 ms (10×) | 3 752 ms (6×) |
| Yükleme | 0.2 s | 0.3 s | **691 s (değişmedi)** |

## Bulgular

1. **W8 bedava, hatta kârlı:** sıfır mIoU kaybı, boyut ½, ConvNeXt'in ANE yolu %29
   hızlandı (ANE'nin düşük-hassasiyet tercihiyle uyumlu). Uçta varsayılan W8 olmalı.
2. **Nicemleme kazancı yığına bağlı:** aynı W8/INT8 fikri CoreML'de kazanç, ORT-CPU'da
   6-17× *pesimizasyon* (dequant yükü). "INT8 = hız" varsayımı platformsuz anlamsız.
3. **Yapısal şişkinlik nicemlenemiyor:** VMamba INT8'de 858→715 MB — grafın 614 MB'ı
   ağırlık değil unroll yapısı; yükleme süresi de aynı kaldı (691 s). Nicemleme,
   SSM'in gerçek dağıtım engeline (yapı) dokunamıyor.
4. **W4'ün faturası aykırı değer profiline göre:** ConvNeXt −3.9 (depthwise conv
   kanalları, 768'de kurtosis 64), Swin −2.8. 4-bit palet aykırı kanalları temsil edemiyor.
5. **Aykırı değer sürprizi (TASK-027):** VMamba-T aktivasyonları üçünün EN ILIMLISI
   (chmax/med 3.2-3.8, kurtosis ~14 sabit; çözünürlükle BOZULMUYOR). Literatürün SSM
   aykırı-değer anlatısı bu omurga/görevde gözlenmedi → **tezin ironisi: nicemlemeye
   en dayanıklı aday, nicemlenecek formata zaten dönüşemiyor. Engel sayısal değil, yapısal.**
6. **Metodolojik not:** rastgele-girdi argmax uyuşması (W4 ConvNeXt %41.6) gerçek-veri
   mIoU kaybını (−3.9) çok abartıyor — vekil metrik uyarısı Bölüm 3.6'ya işlendi.

## Kapsam sınırları (dürüstlük)
- Aşama I yalnız ağırlık-nicemleme; aktivasyon nicemleme (W8A8) + kalibrasyon yapılmadı
  (ImageNet alt kümesi gerekecek — planlanan)
- mIoU'lar 250-görüntü alt-küme + kare-512; tam-val doğrulaması Faz 5 öncesi tekrarlanabilir
- VMamba CoreML nicemlemesi tanım gereği boş hücre (dönüşemiyor)
