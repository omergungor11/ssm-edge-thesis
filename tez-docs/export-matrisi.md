# Export Matrisi — TASK-020 Ana Sonuçları *(12 Ağustos 2026)*

**Koşullar:** 512×512, batch 1, fp32. torch 2.13 / onnxruntime 1.28 / coremltools 9.0, baz M5, 24 GB.
Ham kayıt: `results/raw/export_matrix.jsonl` (başarısız denemeler dahil, aşama aşama).
Üç model de aynı UPerNet başlığını taşır (31.5M); tek değişken omurga.

## Ana tablo — ONNX yolu

| | ConvNeXt-T (CNN) | Swin-T (Transformer) | **VMamba-T (SSM)** | SSM/CNN oranı |
|---|---|---|---|---|
| Export süresi | 1.2 s | 2.2 s | **537.8 s (9 dk)** | **448×** |
| Graf boyutu | 237.4 MB | 238.8 MB | **858.4 MB** | 3.6× |
| Graf düğümü | 843 | 8 667 | **390 758** | **463×** |
| ORT yükleme | 0.1 s | 0.2 s | **724.9 s (12 dk)** | **7 249×** |
| ORT ilk koşu | 0.89 s | 0.91 s | 1.2 s | 1.3× |

Ağırlıklar ~244 MB; VMamba grafının **614 MB'ı saf graf yapısı** (unroll edilmiş tarama).
`Loop`/`Scan` düğümü yok — TorchScript exporter dört yönlü taramaları tamamen açıyor.
En kalabalık op: `Gather` ×139 798, `Einsum` ×46 614. Tepe RSS: 6.65 GB.

> **Tezin ana bulgusunun ilk tam kanıtı:** Çıkarım gecikmesi hayatta kalıyor
> (1.2 s, klasiklerin ~1.3 katı) ama araç zinciri katmanları çöküyor: yükleme
> klasiklerden **~3 600-7 000×** yavaş. FLOPs tabanlı hiçbir analiz bunu göremez.

## Ana tablo — CoreML yolu

| | ConvNeXt-T | Swin-T | **VMamba-T** |
|---|---|---|---|
| Trace | 3.5 s | 3.8 s | **952.4 s (16 dk)** |
| Dönüşüm | 4.1 s ✅ | ~4 s ✅ | **❌ 574.8 s sonra TypeError** |
| İlk tahmin | 0.17 s | 4.41 s | — |

VMamba dönüşümü, 390k-düğümlü grafın derinliklerinde üçüncü-parti koddaki
(`third_party/VMamba`) dinamik shape okumalarının ürettiği `aten::Int` düğümünde
düşüyor — bizim kontrolümüzdeki baş/sarmalayıcı kodda bu sınıf hata giderilmişti
(aşağıya bkz.). **"CoreML'e dönüştürülemiyor" bugünkü resmî durumdur**; scan'in
export-dostu yeniden formülasyonu Faz 4'ün konusu ve oradaki kazanç ölçümünün
temel çizgisi bu başarısızlıktır.

## Yol boyu bulgular (dağıtım sürtünmesi SSM'den önce başlıyor)

Klasik omurgalar bile "kutudan çıktığı gibi" export edilemedi; üç cerrahi gerekti:

1. **UPerNet PSP:** `adaptive_avg_pool2d(3/6)` (çıktı, girdiyi bölmüyor) ONNX
   TorchScript exporter'da desteklenmiyor → statik-dilimli numerik eşdeğer yazıldı
   (max sapma ~2e-5).
2. **mmseg kalıbı `size=x.shape[2:]`:** CoreML dönüştürücüsünü anında düşürüyor
   (`aten::Int`'e 2-elemanlı dizi). Tüm sarmalayıcılarda statik boyutla değiştirildi.
3. **Swin pencere matematiği:** pad/pencere hesaplarındaki shape okumaları aynı
   hatayı üretti → batch=-1 + Python-int boyutlarla yeniden yazıldı (numerik
   eşdeğerlik düzenleme-öncesi ONNX'e karşı 5e-5 ile kanıtlı).

## Matris durumu

| Omurga | PyTorch eager | ONNX/ORT CPU | CoreML |
|---|---|---|---|
| ConvNeXt-T | ✅ | ✅ | ✅ |
| Swin-T | ✅ | ✅ | ✅ |
| VMamba-T | ✅ (saf-torch scan) | ✅ *(yükleme 12 dk)* | ❌ dönüşemiyor |

Sonraki adım (TASK-021): başarılı hücrelerde tam gecikme/bellek/enerji ölçümü +
çözünürlük taraması; ORT CoreML EP hücreleri; `torch.compile` denemeleri.
