# Öncül Doğrulama Raporu *(TASK-008 — Hafta 1 karar noktası)*

**Tarih:** 12 Ağustos 2026 · **Donanım:** baz Apple M5, 24 GB, macOS 26.5.2
**Model:** MiniMamba — saf-PyTorch selective scan, 4 katman, d=96, n=16, ~400K param
**Ham veri:** `results/raw/premise_L196.json`, `results/raw/premise_L1024.json`
**Ortam:** Python 3.12.13, torch 2.13.0, onnxruntime 1.28.0 (CPU EP), coremltools 9.0

> Bu bir mikrobenchmark'tır: amaç VMamba'yı temsil etmek değil, **selective scan'in ardışık
> yapısının araç zincirlerinde nasıl tezahür ettiğini** kendi makinemizde görmek. Gerçek
> modellerle tam ölçüm Faz 2'nin işi.

---

## Sonuç tablosu

| Metrik | L=196 | L=1024 | Oran (L 5.2×) | Yorum |
|---|---|---|---|---|
| torch CPU (eager) | 17.5 ms | 59.3 ms | 3.4× | referans |
| torch MPS | 24.0 ms | 91.8 ms | 3.8× | **her iki ölçekte CPU'dan yavaş** |
| ONNX graf düğümü | 17,671 | 91,363 | 5.2× | **L ile lineer — tam unroll** |
| ONNX graf boyutu | 3.0 MB | 13.4 MB | 4.5× | ağırlık sabit (~1.6 MB); fazlası graf |
| ORT yükleme | 2.0 s | 35.4 s | **17.9×** | **süperlineer — ölçek kırılması** |
| ORT CPU çıkarım | 13.5 ms | 40.0 ms | 3.0× | 0.77× / 0.68× — torch'tan **hızlı** |
| CoreML dönüşüm | 1,576 s (26 dk) | 5,668 s (94 dk) | 3.6× | **400K param için 1.5 saat** |
| CoreML çıkarım | 0.79 ms | 3.83 ms | 4.8× | 22× / 15× **hızlanma** (ALL units) |

Not: `torch.onnx.export` (TorchScript trace, opset 17) `Loop` düğümü üretmedi — döngüyü
tamamen açtı (unroll). ONNX #27796'daki `Loop` yorumlayıcı yükü bu yolda **graf patlaması**
olarak tezahür ediyor; aynı hastalığın iki farklı semptomu.

## Karar: ✅ Öncül doğrulandı — rafine haliyle

Naif hipotez "SSM'ler uçta yavaş çalışır" **değil**. Ölçtüğümüz gerçek şu:

1. **Darboğaz çıkarımda değil, araç zincirinde.** Ardışık tarama, graf temsiline L ile lineer
   büyüyen yapı olarak giriyor; bunun bedeli derleme/dönüşüm/yükleme aşamasında ödeniyor:
   ORT yüklemesi süperlineer (17.9×), CoreML dönüşümü oyuncak modelde bile 1.5 saat.
   Gerçek bir VMamba-T'de (512×512 → çift yönlü 4 tarama, L≈16K) bu maliyetin
   **pratikte derlenemezliğe** dönüşmesi beklenir → Faz 2'nin ana sorusu.
2. **Çıkarım hızı hayatta kalabiliyor, hatta kazanabiliyor:** CoreML'e bir kez dönüşen model
   15-22× hızlı (yürütme yerinin ANE olup olmadığı Xcode ile doğrulanacak — TASK-023).
   ORT bile eager torch'tan hızlı. "Yavaş" anlatısı eksik; doğru anlatı **"dönüştürülemez /
   ölçeklenemez araç zinciri"**.
3. **Paralel donanım ardışık taramayı sevmiyor:** MPS her iki ölçekte CPU'dan yavaş —
   Mamba'nın el yazması CUDA çekirdeğinin varlık sebebinin Apple tarafındaki izdüşümü.

### Tez anlatısına etkisi

AS2 ("avantaj nerede buharlaşıyor?") sorusunun cevabı katmanlanıyor:
**(a)** dönüşüm/derleme süresi, **(b)** yükleme süresi, **(c)** graf boyutu (dağıtım paketi),
**(d)** çıkarım gecikmesi. Literatür yalnızca (d)'yi raporluyor; (a)-(c) bu tezin
katkı alanı. Ölçüm protokolüne (Bölüm 3.5) bu dört katman ayrı ayrı girecek.

### Sınırlılıklar (dürüstlük notu)

- Oyuncak model; gerçek VMamba mimarisi değil (2D çapraz tarama, büyük d_state yok)
- CoreML "ALL" compute unit — hangi işlemcide koştuğu **henüz kanıtlanmadı** (TASK-023)
- Dinamik dizi uzunluğu denenmedi (statik şekil); dinamik şekilde unroll mümkün değil,
  export ya `Loop`'a düşer ya başarısız olur — Faz 2'de denenecek
- Tek koşu günü, termal kontrol harness'ı henüz yok (TASK-009)

## Sonraki adım

Faz 0 Hafta 2'ye geçiş: ölçüm harness'ı (TASK-009), ResNet-50 doğrulaması (TASK-010),
veri boru hattı (TASK-011). Karar noktası geçildi — plan aynen devam.
