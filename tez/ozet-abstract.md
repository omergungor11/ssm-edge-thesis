# ÖZET / ABSTRACT

*(TASLAK v1 — 14 Ağustos 2026, TASK-035)*

## ÖZET

**Teoriden Silikona: Durum-Uzayı Tabanlı Görü Omurgalarının Apple Silicon Uç
Donanımında Gerçekleşen Verimliliğinin Ampirik Analizi**

Durum-uzayı modelleri (SSM) tabanlı görü omurgaları, dizi uzunluğunda doğrusal
karmaşıklıkları sayesinde yüksek çözünürlüklü yoğun tahmin görevleri için güçlü bir
verimlilik vaadi taşır; ancak literatürdeki hız ve bellek kazanımları, donanım-farkındalıklı
özel CUDA çekirdekleriyle ölçülmüştür. Uç cihaz dağıtımının gerçekliği ise graf tabanlı
ara temsiller (ONNX, Core ML) ve derleyicilerden oluşur; bu vaadin genel amaçlı dağıtım
yığınlarında ne kadarının gerçekleştiği sistematik olarak ölçülmemiştir. Bu tez, üç
mimari ailesinin temsilcilerini — VMamba-T (SSM), Swin-T (Transformer), ConvNeXt-T
(CNN) — aynı UPerNet segmentasyon başlığı ve yayımlanmış ADE20K kontrol noktaları
altında sabitleyerek, Apple Silicon üzerinde yedi dağıtım hücresinde (PyTorch eager
CPU/MPS, `torch.compile`, ONNX Runtime CPU/CoreML EP, Core ML CPU/GPU/ANE)
karşılaştırmaktadır. Maliyet, tek bir gecikme sayısı yerine dışa aktarma, yükleme,
çıkarım ve enerji katmanlarına ayrıştırılarak ölçülmüş; yürütme yeri Xcode performans
profiliyle operatör düzeyinde doğrulanmıştır.

Bulgular, SSM'in uç dağıtım engelinin sayısal değil yapısal olduğunu göstermektedir.
VMamba-T Core ML'e hiç dönüşememekte; ONNX yolunda izleme tabanlı dışa aktarma, dört
yönlü taramayı 390 bin düğümlük düz bir grafa açarak oturum yüklemesini 12 dakikaya
çıkarmakta, 1024² çözünürlükte ise dışa aktarma bellek duvarına çarpmaktadır. Buna
karşılık çıkarımın kendisi hayatta kalmaktadır: ONNX Runtime, VMamba çıkarımını eager
temel çizgisinin 0,30 katına indirmekte — maliyet yanlış katmana, her süreç başlatımına
taşınmaktadır. Dağıtılabilir en iyi konfigürasyonlar karşılaştırıldığında klasik
omurgalar ile SSM arasındaki gecikme uçurumu 16 kata, çıkarım başına enerji farkı 25
kata ulaşmaktadır. Apple Neural Engine erişimi ise mimari ailelere göre keskin biçimde
kademelenmektedir: Xcode profiliyle doğrulanan ölçümlerde CNN operatörlerinin %100'ü
ANE'ye atanırken Transformer'da bu oran %0'dır (sessiz GPU geri dönüşü); SSM ise
dönüşüm kapısından girememektedir. 8-bit ağırlık nicemlemesi sıfır
doğruluk kaybıyla model boyutunu yarılamakta ve ANE yolunda çıkarımı hızlandırmakta,
fakat SSM grafının yapısal şişkinliğine dokunamamaktadır; üstelik SSM aktivasyonları
üç aile içinde nicemlemeye en elverişli profili sergilemektedir — nicemlemeye en
dayanıklı aday, nicemlenecek formata zaten dönüşememektedir. Son olarak, `selective scan`'in blok-kapalı-form yeniden
formülasyonu tek blok ölçeğinde graf düğüm sayısını 20 kat azaltarak Core ML kapısını
açmakta ve engelin operatör uyumsuzluğu değil graf ölçeği olduğunu doğrulamaktadır.

Tez, FLOPs tabanlı verimlilik anlatısının dağıtım maliyetinin yalnızca bir katmanını
gördüğünü ampirik olarak ortaya koymakta; mimari verimliliğin bir donanım-yazılım
eşleşmesi problemi olduğu tespitinden hareketle uygulayıcılar için dağıtım-farkındalıklı
bir mimari seçim kılavuzu ve açık, tekrarlanabilir bir ölçüm altyapısı sunmaktadır.

**Anahtar Kelimeler:** durum-uzayı modelleri, görü omurgaları, uç cihazda çıkarım,
Apple Neural Engine, model nicemleme, ONNX/Core ML dağıtım yığınları

---

## ABSTRACT

**From Theory to Silicon: An Empirical Analysis of the Realized Efficiency of
State-Space Vision Backbones on Apple Silicon Edge Hardware**

State-space model (SSM) based vision backbones promise linear complexity in sequence
length, making them attractive for high-resolution dense prediction; yet the reported
speed and memory gains are measured with hardware-aware custom CUDA kernels, whereas
edge deployment relies on graph-based intermediate representations (ONNX, Core ML) and
compilers. How much of the promise survives general-purpose deployment stacks has not
been measured systematically. This thesis benchmarks representatives of three
architecture families — VMamba-T (SSM), Swin-T (Transformer), and ConvNeXt-T (CNN) —
under an identical UPerNet segmentation head with published ADE20K checkpoints, across
seven deployment cells on Apple Silicon: PyTorch eager (CPU/MPS), `torch.compile`,
ONNX Runtime (CPU and CoreML execution providers), and Core ML (CPU/GPU/ANE).
Deployment cost is decomposed into four layers — export, load, inference, and energy —
and execution placement is verified at the operator level via Xcode performance
profiling.

The findings show that the SSM's edge-deployment barrier is structural rather than
numerical. VMamba-T fails to convert to Core ML at all; on the ONNX path,
tracing-based export unrolls the four-directional scan into a flat graph of roughly
390,000 nodes, inflating session load to twelve minutes, and export hits a memory wall
at 1024² resolution. Inference itself, however, survives: ONNX Runtime reduces VMamba
inference to 0.30× its eager baseline — the cost has migrated to the wrong layer,
being paid at every process launch. Access to the Apple Neural Engine is sharply
tiered by architecture family: 100% of CNN operators are mapped to the ANE, 0% for the
Transformer, while the SSM never passes conversion. Eight-bit weight quantization
halves model size at zero accuracy cost yet cannot touch the SSM graph's structural
bloat; strikingly, SSM activations exhibit the mildest outlier profile of the three
families. Finally, a block-closed-form reformulation of the selective scan reduces
graph node count by 20× at single-block scale and opens the Core ML gate, confirming
that the barrier is graph scale rather than operator incompatibility.

The thesis empirically demonstrates that FLOPs-based efficiency narratives capture
only one layer of deployment cost, and contributes a deployment-aware architecture
selection guide together with an open, reproducible measurement infrastructure.

**Keywords:** state-space models, vision backbones, on-device inference, Apple Neural
Engine, model quantization, ONNX/Core ML deployment stacks
