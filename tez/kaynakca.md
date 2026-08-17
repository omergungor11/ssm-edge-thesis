# Kaynakça

> **Durum (TASK-036):** Bölüm dosyalarındaki (`tez/bolum-*.md`, `ozet-abstract.md`) tüm
> atıf yer tutucuları taranmış, her biri künyeye bağlanmış ve **her künye arXiv/yayıncı
> sayfasından doğrulanmıştır** (doğrulama: Ağustos 2026; arXiv abs sayfaları + web
> araması). Metindeki yer tutucular yerinde bırakılmıştır; bağlama Faz 5 son geçişte
> yapılacaktır. Yazar listeleri 3'ten uzunsa "vd." ile kısaltılmıştır — son geçişte
> enstitü şablonuna göre tam listeye açılabilir (tümü arXiv sayfalarında mevcuttur).

## 1. Yer tutucu → künye eşlemesi

| Yer tutucu | Künye (aşağıdaki listede) | Not |
|---|---|---|
| [ADE20K-2017] | Zhou vd. (2017) | |
| [ANE-LLM-Inference] | InsiderLLM (2026) | |
| [Apple-ANE-2022] | Apple Machine Learning Research (2022) | |
| [Apple-ANE-Repo] | Apple Inc. (2022) — ml-ane-transformers | |
| [AutoMamba-2026] | Sun, Li ve Zhu (2026) | |
| [BabyMamba-2026] | Mandal (2026) | |
| [Cityscapes-2016] | Cordts vd. (2016) | |
| [ConvNeXt-2022] = [Liu-2022] | Liu, Mao vd. (2022) | **çift anahtar — birleştirilecek** |
| [Dao-2022] | Dao, Fu vd. (2022) — FlashAttention | |
| [Dao-Gu-2024] | Dao ve Gu (2024) — Mamba-2/SSD | |
| [Dosovitskiy-2020] | Dosovitskiy vd. (2020) — ViT | |
| [DynamicViM-2025] | Wu vd. (2025) | |
| [EfficientViM-2024] | Lee, Choi ve Kim (2024) | |
| [FEMBA-2026] | Tegon vd. (2026) | |
| [Gu-2021] | Gu, Goel ve Ré (2021) — S4 | |
| [Gu-Dao-2023] | Gu ve Dao (2023) — Mamba | |
| [Hatamizadeh-Kautz-2025] | Hatamizadeh ve Kautz (2025) — MambaVision | |
| [Hooker-2020] | Hooker (2020) | |
| [Jetson-Profiling-2025] | Chakraborty vd. (2025) | |
| [LLM-Energy-Tradeoffs-2025] | Maliakel vd. (2025) | |
| [Liu-2021] = [Swin-2021] | Liu, Lin vd. (2021) — Swin | **çift anahtar — birleştirilecek** |
| [Liu-2024] = [VMamba-2024] | Liu, Tian vd. (2024) — VMamba | **çift anahtar — birleştirilecek** |
| [MAP-2024] | Liu ve Yi (2024) | |
| [MambaPTQ-2024] | Pierro ve Abreu (2024) | şerh: LLM odaklı — bkz. §4 |
| [MambaSeg-2025] | Gu, Li vd. (2025) | |
| [ORT-27796] = [ORT-Issue-27796] | altunenes (2026) — ONNX Runtime #27796 | **çift anahtar — birleştirilecek** |
| [OnDeviceLLM-2026] | Chandra ve Krishnamoorthi (2026) | |
| [OuroMamba-2025] | Ramachandran vd. (2025) | |
| [PTQ4VM-2024] | Cho vd. (2024) | |
| [QMamba] | Li vd. (2025) | şerh: metindeki "[DOĞRULANACAK]" — bkz. §4 |
| [QMambaIR-2025] | Chen vd. (2025) | |
| [Sigma-2024] | Wan vd. (2024) | |
| [TernaryMamba-2026] | Ganesaraja vd. (2026) | |
| [TileFuse-2026] | Pang vd. (2026) | |
| [TokenReduction-2025] | Ma vd. (2025) | |
| [Touvron-2020] | Touvron vd. (2020) — DeiT | |
| [UPerNet-2018] = [Xiao-2018] | Xiao vd. (2018) | **çift anahtar — birleştirilecek** |
| [VCMamba-2025] | Munir vd. (2025) | |
| [ViM-Q-2026] | Lyu vd. (2026) | |
| [WattCounts-2026] | Argerich vd. (2026) | |
| [WinMamba-2025] | Zheng vd. (2025) | |
| [Xie-2021] | Xie vd. (2021) — SegFormer | |
| [Zhu-2024] | Zhu vd. (2024) — Vim | |
| *(anahtarsız)* | Apple Inc. — coremltools; MMSegmentation Contributors | yazılım künyeleri, §3 sonunda |

## 2. Kaynakça (alfabetik)

- altunenes (GitHub kullanıcısı). (2026). [Feature Request] ONNX Loop op makes Mamba (SSM) models unusable on CPU and WebGPU. microsoft/onnxruntime hata kaydı #27796 (Mart 2026; kapalı). https://github.com/microsoft/onnxruntime/issues/27796 (erişim: Ağustos 2026).
- Apple Inc. (2022). ml-ane-transformers: Reference implementation of the Transformer architecture optimized for Apple Neural Engine [Yazılım]. GitHub. https://github.com/apple/ml-ane-transformers (erişim: Ağustos 2026).
- Apple Inc. (2026). Core ML Tools (coremltools), sürüm 9.0 [Yazılım]. https://github.com/apple/coremltools (erişim: Ağustos 2026).
- Apple Machine Learning Research. (2022). Deploying Transformers on the Apple Neural Engine. https://machinelearning.apple.com/research/neural-engine-transformers (erişim: Ağustos 2026).
- Argerich, Fürst, Patiño-Martínez vd. (2026). Watt Counts: Energy-Aware Benchmark for Sustainable LLM Inference on Heterogeneous GPU Architectures. arXiv:2604.09048.
- Chakraborty, Tavernier, Kourtis vd. (2025). Profiling Concurrent Vision Inference Workloads on NVIDIA Jetson — Extended. arXiv:2508.08430.
- Chandra, V. ve Krishnamoorthi, R. (2026, 28 Ocak). On-Device LLMs in 2026: What Changed, What Matters, What's Next. Edge AI and Vision Alliance. https://www.edge-ai-vision.com/2026/01/on-device-llms-in-2026-what-changed-what-matters-whats-next/ (erişim: Ağustos 2026).
- Chen, Qin, Zhang vd. (2025). Q-MambaIR: Accurate Quantized Mamba for Efficient Image Restoration. arXiv:2503.21970.
- Cho, Lee, Kim vd. (2024). PTQ4VM: Post-Training Quantization for Visual Mamba. WACV 2025. arXiv:2412.20386.
- Cordts, M., Omran, M., Ramos, S. vd. (2016). The Cityscapes Dataset for Semantic Urban Scene Understanding. CVPR 2016. arXiv:1604.01685.
- Dao, T. ve Gu, A. (2024). Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. ICML 2024. arXiv:2405.21060. *(Mamba-2 / SSD)*
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A. ve Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. arXiv:2205.14135.
- Dosovitskiy, A., Beyer, L., Kolesnikov, A. vd. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021. arXiv:2010.11929. *(ViT)*
- Ganesaraja, Panse vd. (2026). Ternary Mamba: Grouped Quantization-Aware Training of W1.58A16 State Space Models. arXiv:2606.18114.
- Gu, A. ve Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.
- Gu, A., Goel, K. ve Ré, C. (2021). Efficiently Modeling Long Sequences with Structured State Spaces. ICLR 2022. arXiv:2111.00396. *(S4)*
- Gu, Li, Long vd. (2025). MambaSeg: Harnessing Mamba for Accurate and Efficient Image-Event Semantic Segmentation. AAAI 2026. arXiv:2512.24243.
- Hatamizadeh, A. ve Kautz, J. (2025). MambaVision: A Hybrid Mamba-Transformer Vision Backbone. CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/papers/Hatamizadeh_MambaVision_A_Hybrid_Mamba-Transformer_Vision_Backbone_CVPR_2025_paper.pdf
- Hooker, S. (2020). The Hardware Lottery. arXiv:2009.06489. *(Ayrıca: Communications of the ACM, 64(12), 2021.)*
- InsiderLLM. (2026, 5 Mart). Apple Neural Engine for LLM Inference: What Actually Works. https://insiderllm.com/guides/apple-neural-engine-llm-inference/ (erişim: Ağustos 2026).
- Lee, S., Choi, J. ve Kim, H. J. (2024). EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space Duality. CVPR 2025. arXiv:2411.15241.
- Li, Y. vd. (2025). QMamba: Post-Training Quantization for Vision State Space Models. arXiv:2501.13624. *(bkz. §4 şerhi)*
- Liu, Y. ve Yi, L. (2024). MAP: Unleashing Hybrid Mamba-Transformer Vision Backbone's Potential with Masked Autoregressive Pretraining. CVPR 2025. arXiv:2410.00871.
- Liu, Y., Tian, Y., Zhao, Y. vd. (2024). VMamba: Visual State Space Model. NeurIPS 2024. arXiv:2401.10166.
- Liu, Z., Lin, Y., Cao, Y. vd. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV 2021. arXiv:2103.14030.
- Liu, Z., Mao, H., Wu, C.-Y. vd. (2022). A ConvNet for the 2020s. CVPR 2022. arXiv:2201.03545. *(ConvNeXt)*
- Lyu, She, Hung vd. (2026). ViM-Q: Scalable Algorithm-Hardware Co-Design for Vision Mamba Model Inference on FPGA. FCCM 2026. arXiv:2605.01935.
- Ma, Zhang, Su vd. (2025). Training-free Token Reduction for Vision Mamba. arXiv:2507.14042.
- Maliakel, P. J., Ilager, S. ve Brandic, I. (2025). Characterizing LLM Inference Energy-Performance Tradeoffs across Workloads and GPU Scaling. arXiv:2501.08219.
- Mandal (2026). BabyMamba-HAR: Lightweight Selective State Space Models for Efficient Human Activity Recognition on Resource Constrained Devices. arXiv:2602.09872.
- MMSegmentation Contributors. (2020). MMSegmentation: OpenMMLab Semantic Segmentation Toolbox and Benchmark [Yazılım]. https://github.com/open-mmlab/mmsegmentation (erişim: Ağustos 2026).
- Munir, Zhang, Marculescu vd. (2025). VCMamba: Bridging Convolutions with Multi-Directional Mamba for Efficient Visual Representation. ICCV 2025 Workshops. arXiv:2509.04669.
- Pang, Jun, Liu vd. (2026). TileFuse: A Fused Mixed-Precision Kernel Library for Efficient Quantized LLM Inference on AMD NPUs. arXiv:2606.11357.
- Pierro, A. ve Abreu, S. (2024). Mamba-PTQ: Outlier Channels in Recurrent Large Language Models. ICML 2024 çalıştayı. arXiv:2407.12397.
- Ramachandran, Lee, Xu vd. (2025). OuroMamba: A Data-Free Quantization Framework for Vision Mamba. ICCV 2025. arXiv:2503.10959.
- Sun, H., Li, Z. ve Zhu, S. (2026). AutoMamba: Efficient Autonomous Driving Segmentation Model with Mamba. Sensors, 26(7), 2227. https://www.mdpi.com/1424-8220/26/7/2227
- Tegon, Lehmann, Li vd. (2026). FEMBA on the Edge: Physiologically-Aware Pre-Training, Quantization, and Deployment of a Bidirectional Mamba EEG Foundation Model on an Ultra-low Power Microcontroller. arXiv:2603.26716.
- Touvron, H., Cord, M., Douze, M. vd. (2020). Training data-efficient image transformers & distillation through attention. ICML 2021. arXiv:2012.12877. *(DeiT)*
- Wan, Zhang, Wang vd. (2024). Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation. WACV 2025. arXiv:2404.04256.
- Wu, Li, Liang vd. (2025). Dynamic Vision Mamba. arXiv:2504.04787.
- Xiao, T., Liu, Y., Zhou, B., Jiang, Y. ve Sun, J. (2018). Unified Perceptual Parsing for Scene Understanding. ECCV 2018. arXiv:1807.10221. *(UPerNet)*
- Xie, E., Wang, W., Yu, Z. vd. (2021). SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers. NeurIPS 2021. arXiv:2105.15203.
- Zheng, Xia, Chen vd. (2025). WinMamba: Multi-Scale Shifted Windows in State Space Model for 3D Object Detection. arXiv:2511.13138.
- Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A. ve Torralba, A. (2017). Scene Parsing through ADE20K Dataset. CVPR 2017.
- Zhu, L., Liao, B., Zhang, Q. vd. (2024). Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model. ICML 2024. arXiv:2401.09417. *(Vim)*

**Toplam: 45 girdi** — 43 atıflı eser (48 benzersiz yer tutucu anahtarının 5 çift-anahtar birleştirmesiyle tekilleşmiş hâli) + 2 yazılım künyesi (coremltools, MMSegmentation; metinde adla geçiyor, yer tutucusuz).

## 3. Not: yazar adı baş harfleri

Klasik/temel eserlerde (2016-2024 çekirdek literatür) tam ad-baş harf çiftleri
bilinen künyelerdir. 2025-2026 tarihli girdilerin bir kısmında doğrulama soyadı
düzeyinde yapılmıştır ve **baş harfler bilinçli olarak yazılmamıştır** — son geçişte
enstitü şablonuna (APA/IEEE) çevrilirken arXiv sayfasından tamamlanmalıdır. Başlık,
yıl, arXiv kimliği ve (varsa) yayın mekânı alanları tüm girdilerde doğrulanmıştır.

## 4. Çözülemeyen / şerhli atıflar

**Tam çözülemeyen atıf yoktur** — tüm yer tutucular bir künyeye bağlandı. İki şerh:

1. **[QMamba]** (`bolum-2-kuramsal-temeller.md:492`, yanında `*[DOĞRULANACAK: yayın yılı
   ve mekânı]*` notu var): Web araması ile arXiv:2501.13624 (Li vd., 2025, "QMamba:
   Post-Training Quantization for Vision State Space Models"; LtSQ + TGQ yöntemleri
   metindeki tanımla eşleşiyor) bulundu ve kaynakçaya alındı. **Yayın mekânı (konferans/dergi)
   hâlâ doğrulanamadı** — metindeki DOĞRULANACAK notu, mekân teyit edilene kadar kalmalı;
   atıf anahtarı [QMamba-2025] olarak güncellenmeli.
2. **[MambaPTQ-2024]** (`bolum-2-kuramsal-temeller.md`): Doğrulanan başlık "Mamba-PTQ:
   Outlier Channels in **Recurrent Large Language Models**" — makale görü değil LLM
   odaklıdır. Metindeki kullanım bağlamı ("nicemleme zorluğu LLM'lerdeki gibi aktivasyon
   aykırı değerlerinden kaynaklanır") bu içerikle **uyumludur**, atıf doğrudur; yalnızca
   makalenin görü makalesi sanılmaması için not düşüldü.

## 5. Tutarlılık raporu

**a) Aynı esere işaret eden çift anahtarlar** (Faz 5'te tek anahtara indirilmeli):

| Eser | Anahtarlar (geçtiği dosyalar) |
|---|---|
| Xiao vd. 2018 (UPerNet) | [Xiao-2018] (bolum-2, bolum-3) ↔ [UPerNet-2018] (bolum-3.5) |
| Liu vd. 2021 (Swin) | [Liu-2021] (bolum-2) ↔ [Swin-2021] (bolum-3) |
| Liu vd. 2022 (ConvNeXt) | [Liu-2022] (bolum-2) ↔ [ConvNeXt-2022] (bolum-3) |
| Liu vd. 2024 (VMamba) | [Liu-2024] (bolum-2) ↔ [VMamba-2024] (bolum-3) |
| ONNX Runtime #27796 | [ORT-27796] (bolum-1) ↔ [ORT-Issue-27796] (bolum-2, bolum-5) |

**b) Metinde atıflı olup literatür taramasında (`tez-docs/literatur-taramasi.md`) OLMAYAN
eserler** (tarama güncellenebilir): FlashAttention [Dao-2022], The Hardware Lottery
[Hooker-2020], ADE20K [ADE20K-2017], Cityscapes [Cityscapes-2016].

**c) Literatür taramasında olup metinde HİÇ atıf almayan kaynaklar** (ya Faz 5'te
metne bağlanmalı ya da kaynakçaya girmemeli — şimdilik kaynakçaya **alınmadılar**):
Conformer-Based Speech Recognition on Extreme Edge (arXiv:2312.10359), On-Device AI
Models survey (ACM CSUR), Efficient Vision-Language Models survey (arXiv:2504.09724),
Onboard Optimization and Learning survey (arXiv:2505.08793), QUAD (arXiv:2603.29535),
PicoSAM2 (arXiv:2506.18807), WattGPU (arXiv:2607.02391), DEEP-GAP (arXiv:2604.14552),
Awesome-Vision-Mamba (GitHub listesi — kaynakça gerektirmez).

**d) Genel yer tutucular:** `bolum-2:854` ve `bolum-3:485`'teki `[YAZAR-YIL]` ifadeleri
atıf değil, editoryal dipnottur ("atıflar yer tutucu biçimindedir") — işlem gerekmez.

**e) Kaynak gösterimi önerisi:** Mamba (Gu ve Dao, 2023) için yalnız arXiv verilmiştir;
COLM 2024'te yayımlandığı bilinmektedir ancak bu geçişte yayıncı sayfasından teyit
edilmediği için künyeye yazılmadı — son geçişte eklenebilir.
