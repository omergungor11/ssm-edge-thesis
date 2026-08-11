# Plan Revizyonu — Mac-Only *(11 Ağustos 2026)*

## Neden

RTX 5070 sistemine erişim **yok**. Hibrit plan (11 Ağu sabahı) CUDA eksenini "sonra eklenir"
diye tutuyordu; bu revizyon onu tamamen çıkarıp tezi Apple Silicon eksenine oturtuyor.

**Konu değişmiyor.** Gerekçe:
- Öncül kanıtı (ONNX Runtime #27796 — Mamba, M3'te 17× yavaş) zaten Apple donanımında ölçülmüştü
- Tezin özgün boşluğu — Apple'ın ANE optimizasyon reçetesi Transformer için var, **SSM için yok** — tamamen Mac'te kapanıyor
- Faz 4 (ANE-dostu yeniden formülasyon) zaten %100 Mac işiydi
- İlk kendi ölçümümüz de mekanizmayı doğruladı: trace export'ta ONNX graf boyutu dizi uzunluğuyla ölçekleniyor (L=196→3 MB, L=1024→13.4 MB; ağırlıklar sabit)

## Başlık revizyonu

> **"Teoriden Silikona: Durum-Uzayı Tabanlı Görü Omurgalarının Apple Silicon Uç Donanımında Gerçekleşen Verimliliğinin Ampirik Analizi"**

"Uç cihazlar" → "Apple Silicon uç donanımı". İddia daralıyor ama derinleşiyor: tek platformda
katman-katman analiz (CPU / GPU-MPS / ANE), yürütme yeri kanıtlı (Xcode profili).

## Metodolojik değişiklikler

| Eski | Yeni | Gerekçe |
|---|---|---|
| 4 omurgayı sıfırdan eğit (eşit reçete) | **Yayınlanmış ADE20K checkpoint'leri** (mmseg/resmî repo: VMamba, Swin/DeiT, ConvNeXt + EfficientViM) | Mac'te VMamba eğitimi fiilen imkânsız (mamba-ssm MPS'te yok) |
| Eşit doğruluk karşılaştırması | **Doğruluk-gecikme Pareto düzlemi** | Hazır checkpoint'lerin mIoU'ları farklı; ikisi birlikte çizilir — bilimsel olarak geçerli |
| Referans: PyTorch + özel CUDA çekirdeği | Referans: **PyTorch eager (CPU + MPS)**; CUDA sayıları literatürden alıntılanır | Kendi CUDA ölçümü yok — literatür sayısı "bildirilen", bizimki "gerçekleşen" olarak ayrışır |
| Yığınlar: CUDA / compile / ORT / CoreML | Yığınlar: **PyTorch eager (CPU, MPS) · torch.compile · ONNX Runtime (CPU EP, CoreML EP) · CoreML (CPU / GPU / ANE)** — 7+ ölçüm hücresi | Apple içi çeşitlilik arttı; ORT-CoreML EP yeni hücre |
| PTQ4VM reprodüksiyonu (CUDA kodu) | **coremltools nicemleme** (W8A8, W4 palettization) + ORT INT8; PTQ4VM literatür karşılaştırması | Apple yığını içinde tutarlı; yoğun tahmine transfer sorusu (AS3) aynen duruyor |
| Enerji: NVML + powermetrics | Enerji: **powermetrics** | |

## Kazanılan / kaybedilen

**Kazanç:** Eğitim fazı (3 hafta) → checkpoint doğrulamaya iniyor (1 hafta): **+2 hafta tampon.**
Tek makine = sıfır cihaz-arası sürtünme, her gün küçük adım atılabilir.

**Kayıp (dürüstçe, tez §5.4'e girecek):** Vim'in "2.8× hızlı" CUDA iddiası kendi ölçümümüz olmayacak —
"bildirilen" (literatür) vs "gerçekleşen" (bizim Apple ölçümü) karşıtlığı olarak kurgulanır.
Tek NPU ailesi sınırlılığı zaten planda vardı, şimdi tek GPU ailesi de eklendi.

## 5070 geri gelirse

CUDA ekseni matrise ek sütun olarak geri eklenebilir (eski TASK-003 tanımı arşivde duruyor).
Plan buna kapı bırakıyor ama **beklemiyor**.
