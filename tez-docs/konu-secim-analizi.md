# Tez Konusu Ön Literatür Taraması — 4 Aday Alan

**Tarih:** 10 Ağustos 2026
**Donanım kısıtı:** RTX 5070 (12 GB GDDR7, doğrulanacak) + Apple M5 Pro (ANE)
**Hedef çıktı:** Türkçe akademik tez formatı

---

## Değerlendirme kriteri

Her alan dört soruyla puanlandı:

1. **Doygunluk** — alan ne kadar dolu? Yüksek doygunluk = özgün katkı bulmak zor.
2. **Açık boşluk** — literatürün somut olarak *ölçmediği* / *çözmediği* ne var?
3. **Fizibilite** — 12 GB VRAM + M5 Pro ile 3-4 ayda bitirilebilir mi?
4. **Başarısızlık dayanıklılığı** — hipotez tutmazsa tez yine de yazılabilir mi?

Dördüncü kriter en önemlisi. Bir yüksek lisans tezi SOTA'yı yenmek zorunda değil; **doğru soruyu sorup dürüstçe ölçmek** yeterli. Hipotezi tutmadığında çöken bir tez konusu, kötü bir tez konusudur.

---

## 1) 3D Gaussian Splatting

### Alanın durumu (2026 Ağustos)

Alan **çok kalabalık ve hızlı**. 2024'te NeRF'i devirdi, 2026'da olgunlaştı:

- **Sparse-view / few-shot:** Kendi survey'i çıkmış durumda — [Sparse-View 3D Reconstruction: Recent Advances and Open Challenges](https://arxiv.org/pdf/2507.16406). Bilinen problemler: az görüntüyle geometrik bozulma, görülen açılara aşırı uyum, kararsız yeni-açı sentezi. İki ana çözüm hattı zaten oturmuş: (a) regülarizasyon tabanlı (derinlik önselleri, düzgünlük kısıtları, cost-volume rehberliği), (b) öğrenilmiş güçlü önseller / üretici modellerle girdi zenginleştirme.
- **Yüzey rekonstrüksiyonu:** [SolidGS](https://arxiv.org/pdf/2412.15400), [SparseSurf](https://arxiv.org/pdf/2511.14633) — "güzel görünüyor ama yüzey yanlış" problemine adanmış bir alt-alan var.
- **Sıkıştırma / mobil:** Burası da doldu. [Mobile-GS](https://huggingface.co/papers/2603.11531) Snapdragon 8 Gen 3 GPU'da 1600×1063'te **116 FPS** alıyor (SH damıtma + nöral vektör nicemleme + katkı-tabanlı budama). [MEGS2](https://arxiv.org/html/2509.07021) SH'i tamamen küresel Gaussian'larla değiştirip **8× VRAM sıkıştırma** sağlıyor. Ayrıca kendi survey'i var: [SUCCESS-GS](https://arxiv.org/pdf/2512.07197). Üstüne [coreset tabanlı ispatlanabilir budama](https://arxiv.org/pdf/2607.02721) ve [Tensor Core hızlandırma](https://arxiv.org/pdf/2605.17855).
- **Uygulama katmanı:** [Segmentasyon/düzenleme/üretim survey'i](https://arxiv.org/pdf/2508.09977) — dil-tabanlı GS (SLGaussian) dahil.

### Değerlendirme

| Kriter | Not |
|---|---|
| Doygunluk | **Çok yüksek.** Hem sparse-view hem sıkıştırma hem mobil, her birinin kendi survey'i var. |
| Açık boşluk | Dar. Apple Silicon / ANE üzerinde GS render'ı görece az işlenmiş, ama GS ağırlıklı olarak rasterizasyon işi — ANE'nin (nöral hızlandırıcı) doğal işi değil, kazanç şüpheli. |
| Fizibilite | **Riskli.** 12 GB VRAM büyük sahne eğitimi için sınırda. Mip-NeRF360 tarzı sahnelerde bellek sıkıntısı olur; küçük sahnelerle sınırlı kalırsın. |
| Dayanıklılık | Düşük. "116 FPS'i yenemedim" diye biten tez zor savunulur. |

**Karar: Tez olarak önerilmez.** Portfolyo projesi olarak mükemmel (görsel etkisi eşsiz), 2-3 haftalık bir demo repo'su olarak yap — ama tezini buna bağlama.

---

## 2) SAM 2 / açık-sözlükçe video segmentasyonu

### Alanın durumu — kritik gelişme var

**Bu konuyu önerirken bilmediğim şey: SAM 3 çıktı.**

[SAM 3: Segment Anything with Concepts](https://arxiv.org/pdf/2511.16719) artık açık-sözlükçe segmentasyonu **doğrudan modelin içinde** yapıyor: "sarı okul otobüsü", "çizgili kedi" gibi isim öbekleriyle ya da örnek görüntüyle prompt'lanıyor. Promptable Concept Segmentation'da mevcut sistemlere karşı **2× kazanç** bildiriyor ve [Ultralytics paketine tam entegre](https://docs.ultralytics.com/models/sam-3) — çoklu-nesne video takibi dahil.

Bu şu demek: benim önerdiğim "Grounding DINO ile tespit et → SAM 2'ye ver → takip et" mimarisi ([Grounded SAM 2](https://pyimagesearch.com/2026/01/19/grounded-sam-2-from-open-set-detection-to-segmentation-and-tracking/) olarak zaten paketlenmiş durumda) **artık tek modelin yerleşik özelliği**. Boru hattını kurmak birkaç günlük entegrasyon işi — tez değil.

### Kalan gerçek boşluklar

Yine de alan ölü değil, ağırlık merkezi kaydı:

- **Zayıf denetim:** [Weakly-Supervised RVOS through Text Supervision](https://arxiv.org/pdf/2604.17797) — maske etiketi olmadan sadece metinle öğrenme.
- **Zamansal muhakeme:** [Temporal Prompting Matters](https://arxiv.org/pdf/2510.07319), [MomentSeg](https://arxiv.org/pdf/2510.09274), [FeVOS](https://arxiv.org/pdf/2606.25585) — "ne zaman" sorusu "ne" sorusundan daha zor kalmış durumda.
- **Zorluk raporu:** [LSVOS 2025 Challenge Report](https://arxiv.org/pdf/2510.11063) — hangi durumların hâlâ çözülmediğini doğrudan listeliyor. Konu seçilirse **ilk okunacak belge budur.**

### Değerlendirme

| Kriter | Not |
|---|---|
| Doygunluk | Yüksek, ve SAM 3 ile giriş bariyeri düştü — "entegrasyon" projeleri değersizleşti. |
| Açık boşluk | Var ama dar ve zor: zamansal muhakeme, zayıf denetim. |
| Fizibilite | Orta. Video verisi + SAM 3 fine-tuning 12 GB'da sıkışır; dondurulmuş omurga + hafif adaptör ile mümkün. |
| Dayanıklılık | Orta. |

**Karar: Tez olarak zayıf, ikinci portfolyo projesi olarak güçlü.** SAM 3 tabanlı bir demo (metinle video nesne takibi) 1-2 haftada gösterişli bir repo verir.

---

## 3+6) SSM tabanlı görü omurgaları + uç cihaz dağıtımı — **ÖNERİLEN**

Bu ikisi ayrı konu değil; birleştiğinde tek ve savunulabilir bir tez oluyor.

### Kapalı kapılar (bunları yapma — dolu)

**Nicemleme algoritması icat etme.** Bu alan 2024 sonundan beri dolu ve doluyor:

| Çalışma | Katkı |
|---|---|
| [PTQ4VM](https://arxiv.org/abs/2412.20386) (Ara 2024) | Visual Mamba'da ilk PTQ. Üç problemi tanımladı: token-bazlı varyans, kanal-bazlı aykırı değerler, uzun kuyruklu aktivasyonlar. Per-Token Static nicemleme, GPU'da 1.83× hızlanma, <15 dk dönüşüm |
| [Mamba-PTQ](https://arxiv.org/pdf/2407.12397) | Mamba'nın nicemleme zorluğunun LLM'lerdeki gibi aktivasyon aykırı değerlerinden kaynaklandığını gösterdi |
| [OuroMamba](https://arxiv.org/html/2503.10959) (Mar 2025) | Veri-gerektirmeyen PTQ; aykırı değerlerin **zaman adımları arasında dinamik değiştiğini**, bu yüzden statik PTQ'nun çöktüğünü gösterdi |
| QMamba | Uzun-kuyruklu çarpıklık nicemlemesi (LtSQ) + zamansal grup nicemlemesi (TGQ) |
| [Q-MambaIR](https://arxiv.org/pdf/2503.21970) | Görüntü restorasyonunda nicemlenmiş Mamba |
| [ViM-Q](https://arxiv.org/pdf/2605.01935) (May 2026) | FPGA'da algoritma-donanım ortak tasarımı |
| [Ternary Mamba](https://arxiv.org/html/2606.18114v1) (Haz 2026) | W1.58A16 gruplu nicemleme-farkındalıklı eğitim |

**Yeni bir verimli Mamba omurgası tasarlama.** Bu da dolu: [MambaVision](https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf) (CVPR 2025, hibrit Mamba-Transformer), [EfficientViM](https://arxiv.org/pdf/2411.15241) (ADE20K'da 41.3 mIoU / 0.45 ms), [VCMamba](https://arxiv.org/pdf/2509.04669), [Dynamic Vision Mamba](https://arxiv.org/pdf/2504.04787), [token azaltma](https://arxiv.org/pdf/2507.14042), [Sigma](https://arxiv.org/html/2404.04256v2). Takip için: [Awesome-Vision-Mamba](https://github.com/ReaFly/Awesome-Vision-Mamba).

### Açık kapı — ve bu gerçek bir boşluk

Yukarıdaki çalışmaların **tamamının ortak kör noktası**: ölçümler ya A100/RTX sınıfı GPU'da, ya FPGA'da, ya da teorik FLOPs üzerinden. Gerçek dağıtım yığınında ne olduğu sistematik olarak ölçülmemiş.

Ve orada olan şey kötü. Somut kanıt:

> **ONNX Runtime, Issue #27796** — *"ONNX Loop op makes Mamba (SSM) models unusable on CPU and WebGPU"* ([microsoft/onnxruntime](https://github.com/microsoft/onnxruntime/issues/27796))
>
> 30 Mamba bloklu, 9.6M parametreli küçük bir konuşma modeli ONNX'e taşındığında **kullanılamaz** hale geliyor: selective scan, **Apple M3'te 0.1 saniyelik ses için 1.7 saniye** sürüyor — gerçek zamanın 17 katı yavaş. Scripting olmadan tracer, taramayı 298 MB'lık düz bir grafa açıyor ve oturumun yüklenmesi tek başına 445 saniye alıyor.

Bu tesadüf değil, yapısal: selective scan **ardışık** bir işlem, ONNX `Loop` yorumlayıcı yükü ise bunu öldürüyor. Uç cihazda Mamba çalıştırmayı başaranlar bu yüzden graf derleyicisini terk etmiş — [FEMBA](https://arxiv.org/pdf/2603.26716) (mikrodenetleyicide EEG Mamba) TFLite Micro yerine **el yazması özyinelemeli C++ runtime** kullanmış; [BabyMamba-HAR](https://arxiv.org/pdf/2602.09872) Brevitas'tan doğrudan optimize C çekirdeklerine kendi araç zincirini yazmış.

Yani: **Mamba'nın teorik lineer karmaşıklık avantajı ile gerçek uç cihaz performansı arasında ölçülmemiş bir uçurum var.** Vim'in [DeiT'e karşı yüksek çözünürlükte bildirdiği 2.8× hız ve %86.8 daha az bellek](https://arxiv.org/pdf/2406.16722) rakamları özel CUDA çekirdeğiyle alınmış rakamlar. Bu avantajın ne kadarı ANE'de, CoreML'de, ONNX Runtime'da hayatta kalıyor?

Bu soruyu **kimse Apple Neural Engine üzerinde sormamış** — üstelik sende M5 Pro var, yani donanımın kendisi yeni.

### Önerilen tez

> **"Teoriden Silikona: Durum-Uzayı Tabanlı Görü Omurgalarının Uç Cihazlarda Gerçekleşen Verimliliğinin Ampirik Analizi"**

**Araştırma soruları:**

- **AS1.** Eşit doğruluk bütçesinde VMamba / ViT / ConvNeXt omurgalarının yüksek çözünürlüklü yoğun tahmin görevindeki (semantik segmentasyon) gerçek gecikme, bellek tepe noktası ve enerji profili nedir?
- **AS2.** Teorik FLOPs avantajı ile ölçülen duvar-saati gecikmesi arasındaki fark, dağıtım yığınına (PyTorch/CUDA → ONNX Runtime → CoreML/ANE) göre nasıl değişir? Avantaj tam olarak nerede buharlaşıyor?
- **AS3.** Mevcut PTQ yöntemleri (PTQ4VM, QMamba) sınıflandırmadan **yoğun tahmine** taşındığında doğruluk kaybı nasıl davranıyor? Yüksek çözünürlük aykırı değer profilini değiştiriyor mu?
- **AS4.** Selective scan'in derleyici-dostu yeniden formülasyonu (parçalı/paralel tarama, sabit uzunluklu bloklar) ONNX `Loop` darboğazını ne kadar kapatır?

**Neden bu tez sağlam:**

- **Negatif sonuç da tezdir.** "SSM'lerin uç cihaz vaadi mevcut araç zincirlerinde gerçekleşmiyor, sebebi şu, kanıtı bu" — geçerli, yayınlanabilir ve savunulabilir bir sonuç. Hipotezin tutmaması tezi çökertmiyor.
- **Ölçüm işi, keşif işi değil.** Zaman planı öngörülebilir; "modelim yakınsamadı" riski düşük.
- **Senin profiline oturuyor.** 9 yıllık sistem/dağıtım/derleyici deneyimi burada avantaj; saf ML araştırmacısının zayıf olduğu yer tam olarak burası.
- **Tablo ve grafik üretir.** Tez formatı doğal olarak dolar: donanım × yığın × nicemleme × çözünürlük matrisi.
- **Mevcut projelerinden kategorik olarak farklı.** `car-tracking-yolo` "hazır modeli çalıştırdım"; bu "neden çalışmıyor, ölçtüm ve gösterdim".

### Fizibilite (12 GB + M5 Pro)

| Bileşen | Durum |
|---|---|
| VMamba-T / EfficientViM eğitimi, 512px ADE20K | 12 GB'da mixed precision + gradient checkpointing ile **uyar** |
| ViT-S / ConvNeXt-T karşılaştırma omurgaları | Uyar |
| PTQ (eğitim gerektirmez) | Rahat |
| CoreML / ANE ölçümü | M5 Pro'da doğrudan |
| ONNX Runtime CPU ölçümü | Her iki makinede |
| Enerji ölçümü | macOS `powermetrics`; NVIDIA tarafı `nvidia-smi` telemetri |

Büyük omurgalar (VMamba-B/L) eğitilemez ama **gerekmiyor** — tez küçük/orta ölçekte kontrollü karşılaştırma üzerine kurulu.

### İlk okunacaklar

1. [PTQ4VM](https://arxiv.org/abs/2412.20386) — problem taksonomisi buradan
2. [OuroMamba](https://arxiv.org/html/2503.10959) — dinamik aykırı değer analizi
3. [ONNX Runtime #27796](https://github.com/microsoft/onnxruntime/issues/27796) — tezin çıkış noktası
4. [FEMBA](https://arxiv.org/pdf/2603.26716) — uç cihazda Mamba'yı çalıştırmanın gerçek maliyeti
5. [MambaVision](https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf) — hibrit tasarım gerekçesi

---

## 6 tek başına) Genel uç cihaz AI / model sıkıştırma

Tek başına **tez olarak zayıf** — alan survey'lerle dolu ve herkes yapıyor: [On-Device AI Models survey (ACM CSUR)](https://dl.acm.org/doi/full/10.1145/3724420), [Efficient VLM survey](https://arxiv.org/pdf/2504.09724), [Onboard Optimization and Learning survey](https://arxiv.org/pdf/2505.08793), [QUAD](https://arxiv.org/abs/2603.29535) (nicemleme + adaptif damıtma, 6× bellek / 4× gecikme kazancı), [TileFuse](https://arxiv.org/pdf/2606.11357) (AMD NPU'da füzyonlu karma-hassasiyet çekirdekleri), Apple FastVLM.

Donanım zemini de hızlı kayıyor: Snapdragon 8 Elite 45 TOPS, Exynos 2600 38 TOPS donanımsal int4 desteğiyle ([2026 durumu](https://www.edge-ai-vision.com/2026/01/on-device-llms-in-2026-what-changed-what-matters-whats-next/)).

**Ama 3 ile birleştiğinde özgünleşiyor:** genel "modeli küçülttüm" değil, "**belirli bir mimari ailesinin dağıtım yığınıyla uyumsuzluğunu ölçtüm ve nedenini gösterdim**".

---

## Özet tablo

| # | Konu | Doygunluk | Açık boşluk | Fizibilite (12 GB) | Dayanıklılık | Sonuç |
|---|---|---|---|---|---|---|
| 1 | 3D Gaussian Splatting | Çok yüksek | Dar | Riskli | Düşük | Portfolyo projesi |
| 2 | SAM 2/3 açık-sözlükçe video | Yüksek (SAM 3 bariyeri düşürdü) | Dar, zor | Orta | Orta | İkinci portfolyo projesi |
| **3+6** | **SSM omurgaları × uç dağıtım** | **Nicemleme dolu, dağıtım yığını BOŞ** | **Somut ve doğrulanmış** | **Uygun** | **Yüksek** | **TEZ** |
| 6 | Genel edge AI | Çok yüksek | Yok | Uygun | Yüksek | Tek başına yetersiz |

---

## Önerilen yol haritası

1. **Ana tez:** 3+6 birleşimi — "Teoriden Silikona"
2. **Portfolyo projesi A** (tez sürerken, 1-2 hafta): SAM 3 tabanlı metin-komutlu video nesne takibi demo'su
3. **Portfolyo projesi B** (tez sonrası, 2-3 hafta): 3D Gaussian Splatting — küçük sahne, iPhone çekimi → web görüntüleyici

Üçü birlikte tutarlı bir anlatı kuruyor: **"verimli görü modelleri ve bunların gerçek cihazlarda çalıştırılması."** Dağınık üç proje değil, tek bir uzmanlık alanı.
