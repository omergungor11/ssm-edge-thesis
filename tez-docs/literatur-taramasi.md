# Literatür Taraması

**Tez:** Teoriden Silikona — Durum-Uzayı Tabanlı Görü Omurgalarının Uç Cihazlarda Gerçekleşen Verimliliğinin Ampirik Analizi
**Son güncelleme:** 10 Ağustos 2026

> **Not:** Klasik/temel çalışmaların (Mamba, ViT, ConvNeXt, ADE20K vb.) arXiv kimlikleri standart referanslardır ve doğrulanmıştır. 2025-2026 tarihli güncel çalışmalar bu tarama sırasında web araması ile bulunmuş ve linklenmiştir. **Tezde kullanmadan önce her kaynağın tam metnini oku** — bu tarama abstract seviyesindedir.

---

## A. Temel: Durum-uzayı modelleri

| Kaynak | Katkı | Tezle ilişkisi |
|---|---|---|
| Gu & Goel & Ré, **S4** (arXiv:2111.00396) | Yapılandırılmış durum-uzayı dizi modeli; uzun bağlamda O(n log n) | Teorik temel, Bölüm 2.1 |
| Gu & Dao, **Mamba** (arXiv:2312.00752) | Seçici (input-dependent) SSM + donanım-farkındalıklı paralel tarama algoritması | **Tezin merkezindeki `selective scan` burada tanımlanıyor.** Donanım-farkındalıklı çekirdeğin CUDA'ya özgü olduğu bu makalede açık — tezin çıkış noktası |
| Dao & Gu, **Mamba-2 / SSD** (arXiv:2405.21060) | State Space Duality: SSM ile attention arasında yapısal denklik; matris çarpımına indirgenebilen formülasyon | **Kritik.** Faz 4'teki "derleyici-dostu yeniden formülasyon" fikrinin teorik dayanağı buradan gelir |

**Boşluk notu:** Mamba'nın hız iddiası *donanım-farkındalıklı çekirdeğe* bağlıdır. Bu çekirdek CUDA'ya özgüdür. Literatür bu bağımlılığı belirtir ama sonucunu ölçmez.

---

## B. Görü için SSM omurgaları

| Kaynak | Katkı |
|---|---|
| Zhu vd., **Vision Mamba (Vim)** (arXiv:2401.09417) | İlk saf-SSM görü omurgası; çift yönlü tarama. DeiT'e karşı yüksek çözünürlükte **2.8× hız, %86.8 daha az GPU belleği** iddiası |
| Liu vd., **VMamba** (arXiv:2401.10166) | Cross-Scan Module (CSM) ile 2B uzamsal ilişkileri modelleme; lineer karmaşıklık + global alıcı alan |
| Hatamizadeh & Kautz, [**MambaVision**](https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf) (CVPR 2025) | Hibrit Mamba-Transformer omurga. **Saf SSM'in tek başına yetmediğinin itirafı** — son katmanlarda attention gerekiyor |
| [**EfficientViM**](https://arxiv.org/pdf/2411.15241) | Hidden State Mixer tabanlı SSD. ADE20K + SemanticFPN'de **41.3 mIoU / 0.45 ms** |
| [**VCMamba**](https://arxiv.org/pdf/2509.04669) | Konvolüsyon + çok yönlü Mamba köprüsü |
| [**Dynamic Vision Mamba**](https://arxiv.org/pdf/2504.04787) | Dinamik token/blok seçimi ile verimlilik |
| [**Training-free Token Reduction for Vision Mamba**](https://arxiv.org/pdf/2507.14042) | Eğitimsiz token azaltma |
| [**MAP**](https://arxiv.org/pdf/2410.00871) | Maskeli otoregresif ön-eğitim ile hibrit omurga potansiyeli |
| [**Sigma**](https://arxiv.org/html/2404.04256v2) | Siyam Mamba ağı, çok-modlu semantik segmentasyon |
| [**MambaSeg**](https://arxiv.org/html/2512.24243v1) | Görüntü-olay semantik segmentasyonu |
| [**AutoMamba**](https://www.mdpi.com/1424-8220/26/7/2227) | Otonom sürüş segmentasyonu; **RTMamba Jetson AGX Orin'de hız testi** |
| [**WinMamba**](https://arxiv.org/pdf/2511.13138) | 3B nesne tespitinde kaydırmalı pencere SSM |
| [Awesome-Vision-Mamba](https://github.com/ReaFly/Awesome-Vision-Mamba) | Alanın takip listesi — **tez süresince düzenli kontrol et** |

**Doygunluk değerlendirmesi:** Yeni omurga tasarlama kapısı kapalı. Bu çalışmalar *baseline* olarak kullanılacak, rakip olarak değil.

---

## C. Karşılaştırma temelleri (baseline omurgalar)

| Kaynak | Rol |
|---|---|
| Dosovitskiy vd., **ViT** (arXiv:2010.11929) | Karşılaştırmanın attention ayağı |
| Touvron vd., **DeiT** (arXiv:2012.12877) | Vim'in kendini kıyasladığı model — **aynı kıyası tekrarlamak zorunludur** |
| Liu vd., **Swin Transformer** (arXiv:2103.14030) | Hiyerarşik ViT; yoğun tahminde standart |
| Liu vd., **ConvNeXt** (arXiv:2201.03545) | Modern CNN ayağı. **Kontrol grubu olarak kritik** — "SSM mi iyi, yoksa sadece modern eğitim reçetesi mi?" sorusunu ayırır |
| Xie vd., **SegFormer** (arXiv:2105.15203) | Verimli segmentasyon başlığı alternatifi |
| Xiao vd., **UPerNet** (arXiv:1807.10221) | Standart segmentasyon başlığı; omurga karşılaştırmasında sabit tutulacak |

**Metodolojik ilke:** Segmentasyon başlığı ve eğitim reçetesi **tüm omurgalar için birebir aynı** olmalı. Aksi halde ölçtüğün şey mimari değil, hiperparametre şansıdır.

---

## D. SSM nicemleme — **kapalı kapı, ama araç kutusu**

Bu alanda yeni algoritma önerilmeyecek; mevcut yöntemler *kullanılacak*.

| Kaynak | Katkı | Kullanım |
|---|---|---|
| [**PTQ4VM**](https://arxiv.org/abs/2412.20386) (Ara 2024) | Visual Mamba'da ilk PTQ. Üç problem taksonomisi: **token-bazlı varyans, kanal-bazlı aykırı değerler, uzun kuyruklu aktivasyonlar**. Per-Token Static (PTS) nicemleme. GPU'da 1.83× hızlanma, <15 dk dönüşüm | **Ana PTQ aracı.** Problem taksonomisi tezin Bölüm 2.4'ünün iskeleti |
| [**Mamba-PTQ**](https://arxiv.org/pdf/2407.12397) | Nicemleme zorluğunun LLM'lerdeki gibi aktivasyon aykırı değerlerinden kaynaklandığını gösterir | Teorik gerekçe |
| [**OuroMamba**](https://arxiv.org/html/2503.10959) (Mar 2025) | Veri-gerektirmeyen PTQ. **Aykırı değerlerin zaman adımları arasında dinamik değiştiğini, statik PTQ'nun bu yüzden çöktüğünü** gösterir | **AS3 için kritik** — yüksek çözünürlükte bu dinamik daha da kötüleşiyor mu? |
| **QMamba** | Long-tailed Skewness Quantization (LtSQ) + Temporal Group Quantization (TGQ) | İkinci PTQ aracı, karşılaştırma |
| [**Q-MambaIR**](https://arxiv.org/pdf/2503.21970) | Görüntü restorasyonunda nicemlenmiş Mamba | Yoğun tahmine en yakın önceki iş — **konumlandırma için önemli** |
| [**ViM-Q**](https://arxiv.org/pdf/2605.01935) (May 2026) | FPGA'da algoritma-donanım ortak tasarımı | Alanın donanım ucu |
| [**Ternary Mamba**](https://arxiv.org/html/2606.18114v1) (Haz 2026) | W1.58A16 gruplu QAT | Alanın uç noktası; tez kapsamı dışı |

---

## E. Uç cihaz dağıtımı — **açık kapı**

Tezin özgün katkı alanı burası.

| Kaynak | Bulgu | Önem |
|---|---|---|
| [**ONNX Runtime Issue #27796**](https://github.com/microsoft/onnxruntime/issues/27796) | *"ONNX Loop op makes Mamba (SSM) models unusable on CPU and WebGPU."* 30 bloklu 9.6M parametreli model **Apple M3'te 0.1 sn ses için 1.7 sn** — gerçek zamanın 17× yavaşı. Scripting olmadan tracer taramayı **298 MB düz grafa** açıyor; oturum yüklemesi tek başına **445 sn** | **Tezin çıkış noktası.** Akademik değil, mühendislik kanıtı — tez bunu sistematikleştirecek |
| [**FEMBA**](https://arxiv.org/pdf/2603.26716) | Mikrodenetleyicide çift yönlü Mamba EEG modeli. **TFLite Micro yerine el yazması özyinelemeli C++ runtime** kullanmış; gerekçe: selective state space özyinelemesi genel amaçlı çıkarım motorunun yükünü kaldırmıyor | Graf derleyicisinin terk edilmesinin gerekçeli örneği |
| [**BabyMamba-HAR**](https://arxiv.org/pdf/2602.09872) | Kaynak-kısıtlı cihazda hafif SSM. Brevitas'tan doğrudan optimize C çekirdeklerine **kendi araç zinciri** | İkinci bağımsız kanıt — desen tesadüf değil |
| [Apple ML Research, **Deploying Transformers on the ANE**](https://machinelearning.apple.com/research/neural-engine-transformers) | ANE optimizasyon ilkeleri: **(B, C, 1, S) veri formatı** (ANE'nin 4B kanal-öncelikli mimarisine hizalama), büyük ara tensörleri **split/concat ile parçalama** (L2 önbellek yerleşimi), **reshape/transpose minimizasyonu** (bunlar bellek kopyası tetikliyor). `bchq,bkhc->bkhq` einsum formülü | **Faz 4'ün reçetesi.** ViT için var, **Mamba için yok** — tezin özgün katkısı bunu SSM'e taşımak |
| [apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers) | Referans implementasyon | Doğrudan uyarlama kaynağı |
| [Conformer-Based Speech Recognition on Extreme Edge](https://arxiv.org/pdf/2312.10359) | Uç cihazda dizi modeli dağıtımı | Metodoloji örneği |
| [ANE LLM inference: what actually works](https://insiderllm.com/guides/apple-neural-engine-llm-inference/) | **CoreML ANE'yi hedefler ama derleyici nerede ne çalışacağına dair kararları şeffaf değil; desteklenen op kümesi kısıtlı; dispatch zamanlaması ve IOSurface yerleşimi üzerinde doğrudan kontrol yok.** Kısa dizi uzunluğunda birçok Transformer konfigürasyonu bant-genişliği sınırlı hale geliyor | **Metodolojik uyarı.** "ANE'de çalıştı" demek yetmez; gerçekten ANE'de mi çalıştığı Xcode profili ile doğrulanmalı |

---

## F. Verimli çıkarım — genel bağlam

| Kaynak | Rol |
|---|---|
| [On-Device AI Models survey (ACM CSUR)](https://dl.acm.org/doi/full/10.1145/3724420) | Alanın genel haritası, Bölüm 2 girişi |
| [Efficient Vision-Language Models survey](https://arxiv.org/pdf/2504.09724) | Nicemleme / damıtma / budama taksonomisi |
| [Onboard Optimization and Learning: A Survey](https://arxiv.org/pdf/2505.08793) | Cihaz-üstü optimizasyon |
| [QUAD](https://arxiv.org/abs/2603.29535) | Nicemleme + uyarlanabilir damıtma; 6× bellek, 4× gecikme kazancı |
| [TileFuse](https://arxiv.org/pdf/2606.11357) | AMD NPU'da füzyonlu karma-hassasiyet çekirdekleri — **çekirdek füzyonunun önemi** |
| [PicoSAM2](https://arxiv.org/pdf/2506.18807) | Sensör-içi düşük gecikmeli segmentasyon; uç segmentasyonun alt sınırı |
| [On-Device LLMs in 2026](https://www.edge-ai-vision.com/2026/01/on-device-llms-in-2026-what-changed-what-matters-whats-next/) | Donanım zemini: Snapdragon 8 Elite 45 TOPS, Exynos 2600 38 TOPS donanımsal int4 |

---

## G. Ölçüm metodolojisi — **Bölüm 3'ün temeli**

Bu grup tezin bilimsel geçerliliğini belirler. Kötü ölçüm = geçersiz tez.

| Kaynak | Alınacak ders |
|---|---|
| [Watt Counts: Energy-Aware Benchmark for LLM Inference](https://arxiv.org/html/2604.09048v1) | Enerji-farkındalıklı kıyaslama protokolü |
| [Characterizing LLM Inference Energy-Performance Tradeoffs](https://arxiv.org/html/2501.08219v4) | **Termal stabilizasyon protokolü:** ısınma sonrası sistem boşta bekletilir; GPU güç çekimi **30 sn boyunca 3 W bandında** kalana ve sıcaklık **65 °C altına** inene kadar ölçüm başlatılmaz. Güç ve gecikme **NVML** ile kaydedilir |
| [WattGPU](https://arxiv.org/pdf/2607.02391) | Görülmemiş GPU'larda güç/gecikme tahmini |
| [Profiling Concurrent Vision Inference on Jetson](https://arxiv.org/pdf/2508.08430) | Uç cihazda eşzamanlı iş yükü profilleme |
| [DEEP-GAP](https://arxiv.org/pdf/2604.14552) | GPU mimari paralelliği değerlendirmesi |

### Ölçüm protokolü (literatürden damıtılmış)

1. **Isınma:** 10-50 iterasyon (GPU saatleri, bellek tahsisi, çekirdek önbelleği stabilizasyonu için)
2. **CPU gürültüsü izolasyonu:** PyTorch inter-op thread sayısı **1**'e sabitlenir — ölçülen şey GPU davranışı olmalı, CPU zamanlayıcı değişkenliği değil
3. **Termal kontrol:** yukarıdaki 3 W / 30 sn / 65 °C protokolü
4. **İstatistik:** ortalama tek başına yetersiz — **medyan, ortalama, standart sapma ve P99** birlikte raporlanır (kuyruk davranışı için)
5. **Senkronizasyon:** CUDA asenkron; `torch.cuda.synchronize()` olmadan ölçülen süre yalandır
6. **Boşta sistem:** ölçümler başka yük olmayan makinede, sabit batch ile

---

## Boşluk analizi — tezin konumu

```
Nicemleme algoritması          ████████████████████  DOLU (D grubu)
Verimli SSM omurga tasarımı    ████████████████████  DOLU (B grubu)
GPU/FPGA üzerinde ölçüm        ███████████████░░░░░  ÇOĞUNLUKLA DOLU
Yoğun tahminde nicemleme       ███████░░░░░░░░░░░░░  KISMİ (Q-MambaIR restorasyon)
Dağıtım yığını gerçekliği      ███░░░░░░░░░░░░░░░░░  BOŞ ← TEZ BURADA
ANE üzerinde SSM               ░░░░░░░░░░░░░░░░░░░░  BOŞ ← ÖZGÜN KATKI
```

**Tezin tek cümlelik konumu:**
Literatür SSM'lerin *ne kadar hızlı olabileceğini* özel CUDA çekirdekleriyle ölçmüştür; bu tez *gerçek dağıtım yığınlarında ne kadar hızlı olduklarını* ölçer ve aradaki farkın nedenini gösterir.

---

## Okuma sırası (ilk iki hafta)

**Hafta 1 — öncülü doğrula:**
1. [ONNX Runtime #27796](https://github.com/microsoft/onnxruntime/issues/27796) — tezin çıkış noktası
2. Mamba (arXiv:2312.00752) — özellikle Bölüm 3.3, donanım-farkındalıklı algoritma
3. [FEMBA](https://arxiv.org/pdf/2603.26716) — uç cihazda gerçek maliyet

**Hafta 2 — araç kutusunu kur:**
4. [PTQ4VM](https://arxiv.org/abs/2412.20386) — problem taksonomisi
5. [OuroMamba](https://arxiv.org/html/2503.10959) — dinamik aykırı değerler
6. [Apple ANE Transformers](https://machinelearning.apple.com/research/neural-engine-transformers) — Faz 4 reçetesi
7. Mamba-2 / SSD (arXiv:2405.21060) — yeniden formülasyonun teorisi

**Sürekli:** [Awesome-Vision-Mamba](https://github.com/ReaFly/Awesome-Vision-Mamba) haftalık kontrol — alan hızlı hareket ediyor.
