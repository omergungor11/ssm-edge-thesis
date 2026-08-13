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

## TASK-021 ilk tur sonuçları (13 Ağustos)

### Gecikme, 512², medyan

| Yığın | ConvNeXt-T | Swin-T | VMamba-T |
|---|---|---|---|
| CoreML CPU+GPU | **63.9 ms** | **63.9 ms** | ❌ |
| CoreML ALL | 91.4 ms | 86.0 ms | ❌ |
| CoreML CPU_ONLY | 311 ms | 315 ms | ❌ |
| torch MPS (statik PSP) | 152 ms | 172 ms | **1 008 ms** |
| torch CPU | 571 ms | 535 ms | **2 032 ms** |
| ORT CPU | 644 ms | 714 ms | *(ayrı tur — yükleme 12 dk)* |

**Dağıtılabilir-en-iyi uçurumu:** klasikler 64 ms (CoreML) — VMamba 1 008 ms (MPS eager) → **16×.**

### Enerji (powermetrics, 200 ms örnekleme, boşta-düşülmüş)

| Hücre | ANE gücü | mJ/çıkarım |
|---|---|---|
| ConvNeXt CoreML ALL | **3 879 mW — ANE AKTİF** | **458** |
| ConvNeXt CoreML CPU+GPU | 0 | 494 |
| Swin CoreML ALL | 0 — `ANECCompile FAILED` → GPU | 607 |
| Swin CoreML CPU+GPU | 0 | 329 |
| VMamba torch MPS | 0 | 4 844 |
| VMamba torch CPU | 0 | **11 720** |

**Mimari başına bir kademe:** CNN → ANE (en verimli). Transformer → CoreML tamam ama
ANE derleyicisi reddediyor (sessiz GPU fallback; ALL'un CPU+GPU'dan yavaş oluşu imzası).
SSM → CoreML'e giremiyor; eager'da ConvNeXt-ANE'nin **25 katı** enerji.

**Ek bulgu:** `adaptive_avg_pool2d` (bölünmeyen çıktı) üçüncü platformu da vurdu —
MPS eager desteklemiyor; statik-dilim eşdeğeriyle çözüldü. Aynı operatör: ONNX
exporter ✗, CoreML ✗, MPS ✗.

### Kalan hücreler
Çözünürlük taraması (256/768/1024, eager) ve VMamba ONNX 256/1024 export'ları koşuyor;
VMamba ORT ölçümü, `torch.compile`, ORT CoreML EP sıradaki turlarda. ANE resmî kanıtı:
TASK-023 Xcode raporu (kullanıcı).
