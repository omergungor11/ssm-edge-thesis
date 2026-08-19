# TEORİDEN SİLİKONA

**Durum-Uzayı Tabanlı Görü Omurgalarının Apple Silicon Uç Donanımında Gerçekleşen Verimliliğinin Ampirik Analizi**

*Ömer Faruk Güngör — Yüksek Lisans Tezi, BÜTÜNLEŞİK TASLAK (14 Ağustos 2026)*

*Bu dosya otomatik birleştirmedir; tek doğru kaynak tez/ altındaki bölüm dosyalarıdır.*


---

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


---

# 1. GİRİŞ

*(TASLAK v1 — 14 Ağustos 2026, TASK-035)*

## 1.1 Problem Tanımı: Verimlilik İddiaları ile Dağıtım Gerçekliği Arasındaki Uçurum

Derin öğrenme literatüründe bir mimarinin "verimli" ilan edilmesi ile o mimarinin bir
uç cihazda fiilen verimli çalışması, aynı önermenin iki hâli gibi sunulur; oysa ikisi
arasında, bu tezin konusunu oluşturan geniş ve büyük ölçüde ölçülmemiş bir boşluk
vardır. Durum-uzayı modelleri (state space models, SSM) bu boşluğun bugünkü en keskin
örneğidir. Mamba [Gu-Dao-2023] ile başlayan ve VMamba [Liu-2024], Vim [Zhu-2024] gibi
görü omurgalarıyla süren dalga, dizi uzunluğunda doğrusal karmaşıklık ve global alıcı
alanı aynı anda vaat eder; Vim, DeiT'e karşı yüksek çözünürlükte 2,8 kat hız ve
%86,8 daha az GPU belleği bildirir [Zhu-2024]. Bu sayılar doğrudur — fakat belirli bir
varsayım kümesi altında doğrudur: ölçümler, `selective scan` işleminin ardışık
özyinelemesini GPU'nun bellek hiyerarşisine el ile yerleştiren, donanım-farkındalıklı
özel CUDA çekirdekleriyle alınmıştır. Çekirdeğin kendisi mimarinin bir parçası gibi
görünmez, ama verimlilik iddiasının taşıyıcısı odur.

Bu durum, derin öğrenme tarihinde yeni değildir. Hooker'ın "donanım piyangosu"
kavramı, bir araştırma fikrinin başarısının kendi değerinden çok mevcut donanım ve
yazılım araçlarıyla uyumuna bağlı olduğunu savunur [Hooker-2020]: evrişimli ağlar
GPU'larla, Transformer'lar tensör çekirdekleriyle kazanmıştır. SSM'ler bu piyangoyu
tersinden oynamaktadır — mimari, verimliliğini var olan araç zincirlerinden değil,
kendisi için özel yazılmış bir çekirdekten alır. Böyle bir mimarinin verimlilik
iddiası, çekirdeğin taşınamadığı her ortamda yeniden sınanmak zorundadır; ve uç
cihazlar, tam da çekirdeğin taşınamadığı ortamlardır.

Uç cihaz dağıtımının gerçekliği ise bambaşka bir yürütme modeline dayanır. Model,
eğitildiği çerçeveden bir graf temsiline (ONNX, Core ML) dışa aktarılır; bir graf
derleyicisi bu temsili hedef donanımın operatör kümesine eşler; çıkarım, elle yazılmış
çekirdekler değil, derleyicinin ürettiği yürütme planı üzerinden koşar. Özel CUDA
çekirdeği bu boru hattında yoktur ve olamaz: dışa aktarma, modeli çerçevenin dışına,
çekirdeğin erişemeyeceği bir temsile taşır. Üstelik bu yürütme modeli, statik ve
ileri-beslemeli graflar — evrişimler, matris çarpımları, eleman-bazlı işlemler —
varsayımı üzerine inşa edilmiştir. `selective scan` ise özü gereği ardışık bir
özyinelemedir: her adımın çıktısı bir önceki adımın durumuna bağlıdır ve girdiye bağlı
(seçici) parametreler, özyinelemenin sabit bir evrişime katlanmasını engeller. Bir graf
derleyicisi bu yapıyı ya bir döngü operatörünün yorumlayıcı yüküyle ya da döngüyü dizi
uzunluğu kadar kopyalayıp düzleştirerek (unrolling) temsil etmek zorundadır — iki
seçenek de, mimarinin kâğıt üzerindeki karmaşıklık analizinde görünmeyen maliyetler
üretir. Dolayısıyla soru şudur: **bir mimarinin verimliliği hangi varsayımlar altında
geçerlidir ve bu varsayımlar — çekirdek, derleyici, donanım — değiştiğinde iddiaya ne
olur?**

Bu sorunun standart cevabı olan FLOPs ve parametre sayısı, tam da burada yetersiz
kalır. Bu metrikler donanımdan bağımsızdır; güçleri de zaafları da budur. Bir işlemin
kaç çarpma-toplama gerektirdiğini söylerler, fakat o işlemin hedef donanımda bir
operatöre eşlenip eşlenemeyeceğini, eşlenen grafın kaç düğüme açılacağını, o grafın
diske ve belleğe sığıp sığmayacağını, oturumun kaç saniyede yükleneceğini ve çıkarımın
hangi işlem biriminde — CPU, GPU, yoksa nöral hızlandırıcı — koşacağını söylemezler.
Dağıtım maliyeti tek katmanlı değildir: dışa aktarma/dönüşüm, yükleme, çıkarım ve
enerji ayrı ayrı ödenen, ayrı mekanizmalarla büyüyen kalemlerdir. Literatürün verimlilik
karşılaştırmaları bu kalemlerden yalnızca birine — çıkarım aritmetiğine — bakar.

Literatür bu soruyu sistematik olarak sormamıştır. SSM nicemlemesi hızla dolan bir alt
alandır [Cho-2024; Pierro-2024; OuroMamba-2025]; verimli SSM omurgası tasarımı da
öyledir [Hatamizadeh-2025; Lee-2024]. Fakat bu çalışmaların tamamının ortak kör noktası,
ölçümlerin ya A100/RTX sınıfı GPU'larda, ya FPGA prototiplerinde, ya da teorik FLOPs
üzerinden yapılmasıdır. Gerçek dağıtım yığınında — bir telefonun, bir dizüstünün
üzerindeki graf derleyicisi ve nöral hızlandırıcıda — ne olduğu ölçülmemiştir. Oysa
oradan gelen işaretler kötüdür: ONNX Runtime'ın hata izleyicisindeki #27796 numaralı
kayıt, 9,6 milyon parametrelik küçük bir Mamba modelinin ONNX'e taşındığında Apple M3
üzerinde 0,1 saniyelik ses için 1,7 saniye harcadığını, izleme (tracing) tabanlı dışa
aktarmanın taramayı 298 MB'lık düz bir grafa açtığını ve yalnızca oturum yüklemesinin
445 saniye sürdüğünü belgeler [ORT-27796]. Uç cihazda SSM çalıştırmayı başaran az
sayıda çalışmanın izlediği yol da aynı teşhisi doğrular: FEMBA, TFLite Micro yerine el
yazması özyinelemeli bir C++ çalışma zamanı yazmış [FEMBA-2026]; BabyMamba-HAR graf
derleyicisini tümüyle atlayıp kendi araç zincirini kurmuştur [BabyMamba-2026].
Uygulayıcılar graf derleyicisinden kaçıyorsa, mimari ile derleyici arasında yapısal bir
uyumsuzluk var demektir — fakat bu uyumsuzluğun boyutu, katmanları ve mekanizması
sistematik olarak karakterize edilmemiştir.

Bu tez o karakterizasyonu yapar: aynı görev, aynı segmentasyon başlığı ve aynı donanım
üzerinde, üç mimari ailesini yedi dağıtım hücresinde ölçerek teorik verimlilik ile
gerçekleşen verimlilik arasındaki farkın nerede, ne kadar ve neden oluştuğunu ortaya
koyar. Karakterizasyonun birimi tek bir gecikme sayısı değil, dört katmanlı bir maliyet
profilidir; ve Bölüm 4'te gösterildiği gibi, katmanlar birbirinden bağımsız
davranmakta — bir mimari çıkarım katmanında rekabetçi kalırken araç zincirinin başka
bir katmanında kullanılamaz hâle gelebilmektedir. Bu tablo, "verimli mimari" sorusunun
mimariden ibaret olmadığını, bir donanım-yazılım eşleşmesi sorusu olduğunu gösterir.

## 1.2 Motivasyon: Uçta Yoğun Tahmin ve Apple Silicon'un Temsil Gücü

Semantik segmentasyon türü yoğun tahmin (dense prediction) görevleri, uç cihazda
çalışması en çok istenen ve çalıştırılması en pahalı görü iş yükleri arasındadır:
otonom sürüş ve sürücü destek sistemleri, cihaz-üstü fotoğraf düzenleme, artırılmış
gerçeklik ve erişilebilirlik uygulamaları, piksel başına tahmini düşük gecikmeyle ve
buluta veri göndermeden — gizlilik, bağlantısızlık ve maliyet gerekçeleriyle — cihazda
ister. Yüksek çözünürlük bu görevlerin doğasıdır ve dizi uzunluğunu kare hızıyla
büyütür: 512×512 bir girdi, pencere boyutu 4 olan bir omurga için 16.384 token'lık bir
dizidir; 1024×1024'te dizi 65.536 token'a ulaşır. Öz-dikkat (self-attention)
mekanizmasının maliyeti bu uzunlukla kare hızında, SSM'inki doğrusal büyür — yani
SSM'lerin doğrusal karmaşıklık vaadinin en değerli olacağı rejim tam burasıdır ve
vaadin gerçekleşip gerçekleşmediğini sormak için doğru görev sınıfı da budur.
Sınıflandırma gibi düşük çözünürlüklü görevlerde mimariler arasındaki verimlilik farkı
küçülüp ölçüm gürültüsüne karışabilir; yoğun tahmin, farkı büyütecek ve mekanizmasını
görünür kılacak kadar uzun diziler üretir. Aynı zamanda çözünürlük, bu tezde bağımsız
bir deney değişkenidir: dizi uzunluğu ile dağıtım maliyetinin katman katman nasıl
ölçeklendiği, 256²'den 1024²'ye sistematik olarak izlenir (Bölüm 4.2, 4.3).

Deneysel platform olarak Apple Silicon seçilmiştir ve bu seçim bir uzlaşma değil, bir
temsil gücü argümanıdır. Birincisi, Apple Neural Engine (ANE) bugün tüketici
elektroniğindeki en yaygın nöral hızlandırıcılardan biridir: her güncel iPhone, iPad ve
Mac'te aynı mimari aile bulunur; ANE üzerinde çalışmayan bir model, yüz milyonlarca
cihazdan oluşan bir dağıtım yüzeyini kaybeder. İkincisi, Apple yığını tek makinede
katman katman ayrıştırılabilir bir merdiven sunar: PyTorch eager (CPU ve MPS),
`torch.compile`, ONNX Runtime (CPU ve CoreML yürütme sağlayıcıları) ve Core ML'in
CPU / GPU / ANE hedefleri — aynı donanım üzerinde yedi ölçüm hücresi. Üçüncüsü ve
tezin özgün boşluğu açısından belirleyici olanı: Apple, Transformer'ların ANE'ye
uyarlanması için ayrıntılı bir optimizasyon reçetesi yayımlamıştır
[Apple-ANE-2022] — SSM'ler için böyle bir reçete yoktur. Kimse Apple Neural Engine
üzerinde SSM sorusunu sormamıştır; öncül kanıt olan #27796 ölçümünün zaten Apple
donanımında (M3) alınmış olması, bu platform seçimini ayrıca doğal kılar.

Platformun tek olması, ölçümün derinliği için de bir kazançtır. Tüm hücrelerin aynı
fiziksel makinede koşması, cihazlar arası değişkenliği sıfırlar; `powermetrics` ile
işlem birimi başına güç telemetrisi, Xcode performans profili ile operatör düzeyinde
yürütme yeri doğrulaması mümkündür. Bu son nokta metodolojik olarak kritiktir: Core ML,
hesabı CPU, GPU ve ANE arasında şeffaf olmayan kararlarla dağıtır; "ANE'de çalıştı"
iddiası, güç imzası ve profil kanıtı olmadan doğrulanamaz. Bu tezde her yürütme yeri
iddiası bu iki kanıt katmanından en az biriyle desteklenmektedir (Bölüm 3.5.6).
Platform daralmasının bedeli ise dürüstçe kayıt altındadır: literatürün CUDA sayıları
bu çalışmada yeniden üretilmez; "bildirilen" (literatür, özel çekirdek) ile
"gerçekleşen" (bu tezin Apple Silicon ölçümleri) iki ayrı kategori olarak karşı karşıya
konur ve tek hızlandırıcı ailesiyle sınırlılık Bölüm 5.4'te tartışılır.

Bu bir ölçüm tezidir ve bunu bir kısıt değil, bir yöntem tercihi olarak sahiplenir.
Amaç yeni bir omurga ya da yeni bir nicemleme algoritması önermek değil — her iki kapı
da literatürde hızla kapanmaktadır — doğru soruyu kontrollü koşullarda sorup dürüstçe
ölçmektir. Ölçüm işinin öngörülebilirliği, tek makinede tekrarlanabilirliği ve her
hücresinin tablo/grafik olarak raporlanabilirliği, bir yüksek lisans tezinin zaman ve
kaynak bütçesiyle de örtüşür. Ölçümün titizliği ise pazarlık konusu değildir: dağıtım
yığını ölçümleri, termal durum, önbellek ısınması, asenkron yürütme ve arka plan yükü
gibi etkenlere karşı kırılgandır ve bu etkenler kontrol edilmediğinde kat farkları bile
gürültüye karışabilir. Bu nedenle tez, literatürden damıtılmış bir ölçüm protokolünü —
ısınma ve termal stabilizasyon, iş parçacığı izolasyonu, medyan/P99 birlikte raporlama,
yürütme yeri doğrulaması — bilimsel geçerliliğin ön koşulu olarak Bölüm 3.5'te ayrıntılı
biçimde tanımlar ve tüm ölçümleri bu protokole bağlar.

## 1.3 Araştırma Soruları ve Hipotez

Deneysel çerçeve şudur: üç mimari ailesinin yoğun tahminde yerleşik temsilcileri —
VMamba-T (SSM), Swin-T (Transformer) ve ConvNeXt-T (CNN) — aynı UPerNet segmentasyon
başlığı altında, yayımlanmış ADE20K kontrol noktalarıyla sabitlenir; tek değişken
omurgadır. ConvNeXt'in kontrol grubu olarak varlığı bilinçlidir: modern eğitim
reçetesiyle güncellenmiş bir CNN, "SSM mi iyi, yoksa yalnızca modern reçete mi?"
sorusunu ayrıştırır. Bu sabit çerçeve üzerinde tez, dört araştırma sorusu (AS)
etrafında örgütlenmiştir:

- **AS1.** Eşit doğruluk bütçesinde SSM / Transformer / CNN omurgalarının yüksek
  çözünürlüklü semantik segmentasyondaki gerçek gecikme, bellek tepe noktası ve enerji
  profili nedir?
- **AS2.** Teorik FLOPs avantajı ile ölçülen duvar-saati gecikmesi arasındaki fark,
  dağıtım yığınına (PyTorch eager → `torch.compile` → ONNX Runtime → CoreML/ANE) göre
  nasıl değişir? Avantaj nerede buharlaşır?
- **AS3.** Mevcut eğitim-sonrası nicemleme (PTQ) yöntemleri sınıflandırmadan yoğun
  tahmine taşındığında doğruluk kaybı nasıl davranır? Yüksek çözünürlük aykırı değer
  profilini değiştirir mi?
- **AS4.** `selective scan`'in derleyici-dostu yeniden formülasyonu (parçalı/paralel
  tarama, sabit uzunluklu bloklar) ONNX `Loop`/graf darboğazını ne kadar kapatır?

Dört soru bir merdiven oluşturur. AS1, referans yığında dürüst bir temel çizgi kurar:
karşılaştırma "eşit parametre" üzerinden değil, yayımlanmış kontrol noktalarının
doğruluk-gecikme Pareto düzlemi üzerinden yapılır; böylece bir omurganın hız avantajı,
doğruluk farkının arkasına saklanamaz (Bölüm 3.2, 4.2). AS2 tezin merkez sorusudur:
aynı model, yığın merdiveninin her basamağında yeniden ölçülür ve avantajın hangi
basamakta, hangi maliyet katmanında kaybolduğu izlenir (Bölüm 4.3). AS3, uç dağıtımın
standart aracı olan nicemlemenin bu tabloyu değiştirip değiştiremeyeceğini sorar —
literatürdeki SSM-PTQ çalışmaları sınıflandırma üzerinedir; yoğun tahminin uzun
dizileri ve yüksek çözünürlüğü, aykırı değer profilini ilkesel olarak değiştirebilir
(Bölüm 4.4). AS4 ise teşhisten tedaviye geçer: eğer engel taramanın graf temsilindeyse,
taramanın matematiksel olarak eşdeğer fakat derleyici-dostu bir yeniden formülasyonu
engeli kaldırabilmelidir — bu öngörünün sınanması, aynı zamanda teşhisin kendisinin de
sınanmasıdır (Bölüm 4.5). AS1–AS3 ölçüm sorularıdır ve kesin sonuç üretir; AS4 tezin
riskli özgün katkısıdır ve kısmi başarısı bile raporlanabilir niteliktedir.

Bu soruların arkasındaki hipotez şudur:

> Durum-uzayı tabanlı görü omurgalarının literatürde bildirilen verimlilik avantajı,
> özel CUDA çekirdeklerine bağımlıdır ve genel amaçlı dağıtım yığınlarında (ONNX
> Runtime, CoreML/ANE) büyük ölçüde kaybolur; bu kaybın kaynağı `selective scan`
> işleminin ardışık yapısı ile graf derleyicilerinin yürütme modeli arasındaki
> uyumsuzluktur.

Hipotez iki sınanabilir alt iddiaya ayrışır. Birinci alt iddia niceldir: avantajın
kaybolduğu, aynı modelin yığın merdiveni boyunca ölçülmesiyle doğrudan sınanır —
kaybolmuyorsa hipotez çürümüştür. İkinci alt iddia mekanizmaya dairdir: kaybın
kaynağının taramanın ardışıklığı ile derleyicinin yürütme modeli arasındaki uyumsuzluk
olduğu, operatör düzeyinde profil analiziyle (hangi katman, hangi operatör, hangi
maliyet) ve karşı-olgusal bir müdahaleyle (AS4'ün yeniden formülasyonu: uyumsuzluk
giderilirse engel kalkıyor mu?) sınanır. Bu ayrıştırma önemlidir, çünkü birinci iddia
doğrulanıp ikincisi rafine edilebilir — nitekim olan da budur.

Deney tasarımının kurucu ilkesi, hipotezin **her iki akıbetinin de değerli olmasıdır**.
Hipotez doğrulanırsa tez, mimari seçimi için dağıtım-farkındalıklı bir kılavuz ve
uyumsuzluğun mekanizma açıklamasını üretir; çürütülürse SSM'lerin uç cihaz olgunluğunun
sistematik ilk kanıtını üretir. Sonucun yönüne bağlı olarak çökmeyen bu tasarım, konunun
seçilme gerekçesidir. Nitekim Bölüm 4'te görüleceği gibi, gerçekleşen sonuç iki uçtan
daha ilginç bir yerdedir: hipotezin "avantaj kaybolur" öngörüsü doğrulanmış, fakat
mekanizma öngörüsü rafine edilmek zorunda kalmıştır — engel tek tek operatörlerin
uyumsuzluğundan çok, taramanın graf temsilinde yarattığı ölçek patlamasından
kaynaklanmaktadır ve maliyet çıkarım katmanından çok dışa aktarma, dönüşüm ve yükleme
katmanlarına yığılmaktadır.

## 1.4 Tezin Katkıları

Tezin katkıları beş başlıkta toplanır. Katkıların hiçbiri yeni bir omurga ya da yeni
bir nicemleme algoritması değildir; katkı, literatürün ölçmediği bir gerçekliğin —
dağıtım yığınının — sistematik, kanıtlı ve tekrarlanabilir haritasıdır. Aşağıdaki
maddeler, tezin bütününde gösterilen bulguların gerçekleşmiş hâlleridir; sayısal
ayrıntılar ilgili bölümlere bırakılmıştır.

1. **Apple Silicon üzerinde sistematik verimlilik matrisi ve dört-katmanlı maliyet
   modeli.** Üç mimari ailesinin temsilcileri (VMamba-T, Swin-T, ConvNeXt-T), aynı
   segmentasyon başlığı (UPerNet) altında, yedi dağıtım hücresinde (PyTorch eager
   CPU/MPS, `torch.compile`, ONNX Runtime CPU/CoreML EP, Core ML CPU/GPU/ANE)
   ölçülmüştür. Ölçüm, tek bir "gecikme" sayısı yerine maliyeti dört katmana ayırır:
   dışa aktarma/dönüşüm, yükleme, çıkarım ve enerji. Bölüm 4.3'te gösterildiği gibi bu
   ayrıştırma, FLOPs tabanlı analizin ilkesel olarak göremeyeceği bir tabloyu görünür
   kılar: SSM omurgasında çıkarım gecikmesi büyük ölçüde hayatta kalırken, maliyet araç
   zincirinin diğer katmanlarına taşınmaktadır.

2. **Teorik-gerçekleşen farkın mekanizma teşhisi: engel sayısal değil yapısaldır ve
   yapı, ölçek demektir.** Tez, SSM'in dağıtım engelinin kaynağını operatör
   uyumsuzluğu düzeyinden graf ölçeği düzeyine taşır: izleme tabanlı dışa aktarma,
   dört yönlü taramayı çözünürlükle ölçeklenen yüz binlerce düğümlük düz bir grafa
   açmakta; dönüşüm ve yükleme maliyeti düğüm sayısıyla süperlineer büyümekte; yüksek
   çözünürlükte dışa aktarma bir bellek duvarına çarpmaktadır (Bölüm 4.3 ve 4.5).
   Tek-blok ölçekte ardışık formun dahi Core ML'e dönüşüp ANE'de verimli çalışması
   (Bölüm 4.5), teşhisin belirleyici kanıtıdır.

3. **ANE erişiminin mimari ailelere göre kademelenmesinin ilk kanıtlı haritası.**
   Yürütme yeri, güç telemetrisiyle değil, Xcode performans profiliyle operatör
   düzeyinde doğrulanmıştır: CNN omurgası operatörlerinin %100'ü ANE'ye atanırken,
   Transformer omurgasında bu oran %0'dır (sessiz GPU geri dönüşü) ve SSM omurgası
   Core ML'e hiç dönüşememektedir (Bölüm 4.3.4). "ANE'de çalışır" iddiasının üç
   mimari aile için üç ayrı anlama geldiğini gösteren bu kademelenme, uygulayıcı
   için doğrudan mimari seçim bilgisidir: uçta enerji verimliliğinin belirleyicisi
   mimarinin aritmetik yoğunluğu değil, derleyicinin o mimariyi hızlandırıcıya kabul
   edip etmemesidir — ve bu kabul, ikili (var/yok) değil kademeli bir spektrumdur.

4. **Nicemlemenin yoğun tahmine transferi üzerine iki ayrıştırıcı bulgu.** Birincisi,
   8-bit ağırlık nicemlemesi bu görev sınıfında "bedava"dır — sıfır doğruluk kaybıyla
   model boyutunu yarılar ve ANE yolunda çıkarımı hızlandırır — fakat aynı fikrin
   kazancı yığına bağlıdır ve yanlış yığında pesimizasyona dönüşür (Bölüm 4.4).
   İkincisi, literatürün SSM'ler için bildirdiği ağır aykırı-değer profili bu
   omurga/görev biriminde gözlenmemiştir; incelenen SSM omurgası üç aile içinde
   nicemlemeye istatistiksel olarak en elverişli aktivasyonlara sahiptir. Bu ironi —
   nicemlemeye en dayanıklı adayın, nicemlenecek formata zaten dönüşememesi — engelin
   sayısal değil yapısal olduğu teşhisini bağımsız bir eksenden doğrular.

5. **Açık ölçüm altyapısı ve tekrarlanabilir kıyaslama paketi.** Termal
   stabilizasyon, yürütme yeri doğrulaması ve ortam sürümlerinin otomatik kaydını
   içeren ölçüm harness'ı, başarısız denemeler dahil tüm ham ölçüm kayıtlarıyla
   birlikte açık kaynak olarak yayımlanmıştır:
   `github.com/omergungor11/ssm-edge-thesis`. Ölçüm çalışmalarının değeri
   tekrarlanabilirliğiyle sınırlıdır; ve bu tezin bulgularının bir kısmı — dönüşüm
   hataları, bellek duvarları, yükleme süreleri — ancak sürecin kendisi kayıt altına
   alındığında görünür olan türdendir. Altyapı bu yüzden bir yan ürün değil, katkının
   parçasıdır; araç zinciri sürümleri değiştikçe aynı matrisin yeniden koşulabilmesi,
   Bölüm 5.5'te tartışılan "bulgular ne kadar kalıcı?" sorusunun da cevap mekanizmasıdır.

## 1.5 Tezin Organizasyonu

Tezin geri kalanı şöyle örgütlenmiştir. **Bölüm 2**, durum-uzayı modellerinin
matematiksel temellerinden görü omurgalarına, SSM'e özgü nicemleme zorluklarından
dağıtım yığınları ve derleyici mimarilerine uzanan kuramsal zemini kurar ve literatürün
boş bıraktığı alanı tanımlar. **Bölüm 3**, deneysel tasarımı — model seçimi ve
doğruluk-gecikme Pareto protokolü, veri kümeleri, yedi hücreli dağıtım matrisi — ile
tezin bilimsel geçerliliğinin kalbi olan ölçüm protokolünü (termal stabilizasyon,
yürütme yeri doğrulaması, istatistiksel raporlama) ayrıntılandırır. **Bölüm 4**,
dört araştırma sorusunun deneysel cevaplarını sunar: referans profiller (AS1), dağıtım
yığını matrisi ve avantajın buharlaşma noktaları (AS2), nicemleme transferi (AS3) ve
yeniden formülasyon prototipi (AS4). **Bölüm 5**, bulguları "mimari verimlilik bir
donanım-yazılım eşleşmesi problemidir" çerçevesinde yorumlar, uygulayıcılar için bir
seçim kılavuzu çıkarır ve sınırlılıklar ile karşı-argümanları — araç zincirleri
olgunlaştığında bulguların akıbeti dahil — dürüstçe tartışır. **Bölüm 6**, araştırma
sorularına verilen cevapları toplar ve gelecek çalışmaları çizer. Ekler, tam deney
sonuç tablolarını (EK A), ölçüm altyapısının kodu ve kullanımını (EK B), ortam ve sürüm
belgelemesini (EK C) ve yeniden formüle edilmiş tarama operatörünün implementasyonunu
(EK D) içerir.


---

# 2. KURAMSAL TEMELLER VE İLGİLİ ÇALIŞMALAR

*(TASLAK v1 — 12 Ağustos 2026, TASK-019)*

Bu bölüm, tezin deneysel sorularının üzerine oturduğu dört kuramsal katmanı sırayla inşa
eder. İlk katman, durum-uzayı modellerinin (state space models, SSM) matematiksel
evrimidir: S4'ün yapılandırılmış parametrizasyonundan Mamba'nın seçicilik mekanizmasına
ve bu mekanizmanın verimliliğini mümkün kılan — fakat aynı zamanda belirli bir donanıma
bağlayan — donanım-farkındalıklı tarama algoritmasına uzanır (§2.1). İkinci katman, bu
modellerin görü alanına taşınmasıdır: tarama stratejileri, hibrit mimariler ve yoğun
tahmin görevlerindeki kullanım (§2.2), ardından karşılaştırma eksenini oluşturan
Transformer ve evrişimli omurgalar (§2.3). Üçüncü katman, uç dağıtımın standart aracı
olan nicemlemenin SSM'lere uygulandığında ortaya çıkan kendine özgü zorluklardır (§2.4).
Dördüncü katman ise modellerin gerçek cihazlara ulaştığı yolun kendisidir: graf tabanlı
ara temsiller, derleyiciler ve Apple Neural Engine gibi özelleşmiş hızlandırıcılar
(§2.5). Bölüm, bu dört katmanın kesişiminde literatürün bugün boş bıraktığı alanı
tanımlayarak ve tezin bu boşluktaki konumunu netleştirerek kapanır (§2.6).

Bölüm boyunca izlenen kılavuz soru şudur: **bir mimarinin "verimli" olduğu iddiası,
hangi yazılım ve donanım varsayımları altında geçerlidir ve bu varsayımlar
değiştiğinde iddiaya ne olur?** Literatürün her alt alanı bu soruya kısmi cevaplar
verir; hiçbiri cevabı uçtan uca bir dağıtım yığını üzerinde ölçmez.

---

## 2.1 Durum-Uzayı Modelleri

### 2.1.1 S4'ten Mamba'ya: Seçicilik Mekanizması

Durum-uzayı modelleri, kontrol kuramının klasik sürekli-zaman formülasyonundan türetilir.
Tek girişli tek çıkışlı sürekli sistem,

$$x'(t) = \mathbf{A}\,x(t) + \mathbf{B}\,u(t), \qquad y(t) = \mathbf{C}\,x(t)$$

biçiminde, girdiler dizisi $u$'yu gizli durum $x \in \mathbb{R}^{N}$ üzerinden çıktı
$y$'ye eşler. Durum boyutu $N$ (literatürde `d_state`), modelin geçmişi ne kadar
zengin özetleyebildiğini belirleyen kapasite parametresidir; pratikte her model
kanalı kendi $N$ boyutlu durumunu taşıdığından, hesaplamanın ara tensörleri kanal
sayısı ile $N$'in çarpımı ölçeğinde genişler — §2.1.2'de ele alınacak bellek trafiği
probleminin tohumu bu genişlemededir. Ayrık dizilere uygulanabilmesi için sistem bir adım büyüklüğü
$\Delta$ ile ayrıklaştırılır (discretization); sıfırıncı derece tutucu (zero-order
hold) altında ayrık parametreler $\bar{\mathbf{A}} = \exp(\Delta \mathbf{A})$ ve
$\bar{\mathbf{B}}$ elde edilir ve model bir doğrusal özyineleme (recurrence) hâlini
alır:

$$h_t = \bar{\mathbf{A}}\,h_{t-1} + \bar{\mathbf{B}}\,x_t, \qquad y_t = \mathbf{C}\,h_t.$$

Bu formülasyonun dizi modellemesi için çekici iki yüzü vardır. Birincisi, model
**doğrusal zamanla-değişmez** (linear time-invariant, LTI) olduğu sürece özyineleme
bir evrişime (convolution) denktir: çıktı dizisi, parametrelerden türetilen sabit bir
evrişim çekirdeği ile girdinin evrişimi olarak yazılabilir ve eğitim sırasında tüm dizi
paralel işlenebilir. İkincisi, çıkarım sırasında aynı model adım adım özyinelemeli
çalıştırılabilir ve dizi uzunluğundan bağımsız sabit boyutlu bir durumla ilerler. S4
(Structured State Space Sequence Model) [Gu-2021], bu kuramsal iskeleti pratikte
eğitilebilir kılan çalışmadır: durum matrisi $\mathbf{A}$'ya, uzun geçmişi sıkıştırarak
saklama problemine ilkeli bir çözüm sunan özel bir başlangıç yapısı (HiPPO ailesi)
verilmiş ve evrişim çekirdeğinin hesabı, yapılandırılmış parametrizasyon sayesinde uzun
dizilerde $O(n \log n)$ karmaşıklığa indirilmiştir. S4 ve ardılları, uzun bağıml
dizi kıyaslamalarında Transformer'ların zorlandığı rejimlerde güçlü sonuçlar
bildirmiştir.

Aynı matematiksel nesnenin üç ayrı hesaplama kipine sahip olması — sürekli sistem,
ayrık özyineleme ve evrişim — SSM ailesinin kimliğini tanımlayan özelliktir ve her kip
farklı bir kullanım senaryosuna hizmet eder: evrişimsel kip eğitimde paralellik,
özyinelemeli kip çıkarımda sabit bellek, sürekli kip ise örnekleme hızından bağımsızlık
sağlar. Bu üçlü esneklik, literatürde çoğunlukla kuramsal bir zarafet olarak sunulur;
bu tezin perspektifinden ise kritik olan, **kiplerin donanım maliyetlerinin eşit
olmamasıdır**. Hangi kipin hangi yürütme ortamında fiilen kullanılabildiği, ilerleyen
bölümlerin ana sorusudur.

LTI varsayımı, verimliliğin kaynağı olduğu kadar ifade gücünün de sınırıdır. Sabit
$\bar{\mathbf{A}}, \bar{\mathbf{B}}, \mathbf{C}$ parametreleri, modelin her girdi
öğesine aynı dinamikle davranması demektir: model, *içeriğe bakarak* neyi hatırlayıp
neyi unutacağına karar veremez. Kısıtın pratikteki anlamı sentetik görevlerle
gösterilebilir: girdinin yalnızca belirli işaretli öğelerini kopyalamayı gerektiren
seçici kopyalama türü görevlerde, dinamiği içeriğe göre değiştiremeyen LTI modeli
ilkesel olarak çaresizdir — hangi öğenin önemli olduğu girdiye bağlıdır, model ise
tüm öğelere aynı ağırlıkla davranmak zorundadır [Gu-Dao-2023]. Mamba bu kısıtı
**seçicilik**
(selectivity) mekanizmasıyla kaldırır: $\mathbf{B}$, $\mathbf{C}$ ve adım büyüklüğü
$\Delta$, girdinin fonksiyonu hâline getirilir. $\Delta$'nın girdiye bağlı olması
özellikle kritiktir; büyük $\Delta$ durumun sıfırlanıp mevcut girdiye odaklanmasına,
küçük $\Delta$ ise mevcut girdinin yok sayılıp durumun korunmasına karşılık gelir —
yani model, içeriğe bağlı bir unutma/hatırlama kapısı kazanır. Bu mekanizma ile Mamba,
dil modellemesi dahil içerik-duyarlı görevlerde attention'sız bir mimarinin
Transformer kalitesine erişebildiğini göstermiştir.

Mimari düzeyde Mamba, seçici SSM'i kapılı (gated) bir blok içinde paketler: girdi iki
kola izdüşürülür; birinci kol kısa bir derinlemesine evrişimden (depthwise convolution)
ve doğrusal-olmayan etkinleştirmeden geçtikten sonra seçici taramaya girer, ikinci kol
çıktıyı çarpımsal kapıyla modüle eder. Bu blok yapısının dağıtım açısından önemli bir
sonucu vardır: bloktaki işlemlerin büyük çoğunluğu — izdüşümler, evrişim,
etkinleştirme, kapılama — her araç zincirinin sorunsuz desteklediği standart
operatörlerdir. Taşınabilirlik problemi bloğun bütününde değil, tam kalbindeki tek
işlemde, seçici taramada yoğunlaşır; bu yoğunlaşma, ilerleyen bölümlerde taşıma
başarısızlıklarının cerrahi biçimde tek operatöre atfedilebilmesini sağlar.

Ancak seçiciliğin yapısal bir bedeli vardır ve bu bedel tezin çıkış noktasıdır:
parametreler girdiye bağlı hâle gelince model artık zamanla-değişmez değildir ve
**evrişimsel eşdeğerlik ortadan kalkar**. Hesaplama, ilkesel olarak yalnızca
özyinelemeli formda — her adımın bir öncekinin çıktısına bağlı olduğu **ardışık bir
tarama** (selective scan) olarak — tanımlıdır. S4'te paralellik matematiksel formun
armağanıydı; Mamba'da paralellik, aşağıda ele alınan özel bir algoritma ve onun özel
bir donanımdaki gerçekleştirimiyle *geri kazanılmak* zorundadır.

*Tezle bağ:* Bu alt bölümün tanımladığı seçici tarama, tezin tüm ölçümlerinin odağındaki
işlemdir; "SSM verimliliği" iddialarının hangi matematiksel yapıya yaslandığını ve bu
yapının neden doğası gereği ardışık olduğunu burada netleştirmek, sonraki bölümlerdeki
dağıtım bulgularının nedensel zeminini kurar.

### 2.1.2 Donanım-Farkındalıklı Paralel Tarama — ve CUDA Bağımlılığı

Mamba makalesinin başlığındaki "hardware-aware" nitelemesi süsleme değil, iddianın
kurucu parçasıdır [Gu-Dao-2023]. Seçici taramanın naif gerçekleştirimi iki ayrı
nedenle yavaştır. Birincisi zamansal bağımlılıktır: $h_t$, $h_{t-1}$ olmadan
hesaplanamaz; dizi uzunluğu $L$ boyunca ardışık ilerlemek, on binlerce paralel iş
parçacığını aynı anda çalıştırmak üzere tasarlanmış modern hızlandırıcıların yürütme
modeline aykırıdır. İkincisi bellek trafiğidir: seçicilik nedeniyle her zaman adımının
kendi $\bar{\mathbf{B}}_t$, $\bar{\mathbf{A}}_t$ değerleri vardır; bunların ve genişletilmiş
durum tensörünün ($B \times L \times D \times N$ boyutunda) ana belleğe açıkça
yazılması, hesaplamayı bellek bant genişliğine boğar.

İkinci nedenin — bellek trafiğinin — modern hızlandırıcı programlamasındaki yeri ayrıca
vurgulanmalıdır. GPU'larda aritmetik işlem kapasitesi, bellek bant genişliğinden kat
kat hızlı büyümüştür; günümüz iş yüklerinin önemli kısmı hesap-sınırlı değil
**bellek-sınırlıdır** (memory-bound). Bu gerçeklik, "GÇ-farkındalıklı" (IO-aware)
çekirdek tasarımı adı verilebilecek bir mühendislik okulunu doğurmuştur: FlashAttention
[Dao-2022]'in attention için yaptığı gibi, ara sonuçları ana belleğe yazmak yerine
yonga-içi bellekte tutarak ve hesaplamayı bellek hiyerarşisine göre yeniden
düzenleyerek, *matematiksel olarak aynı* işlevin duvar-saati maliyetini katlarca
düşürmek. Mamba'nın donanım-farkındalıklı taraması bu okulun SSM'deki uygulamasıdır —
ve bu okulun tanımlayıcı özelliği, kazanımların donanımın bellek hiyerarşisi
parametrelerine (SRAM boyutu, warp genişliği, birleşik bellek erişim kuralları) sıkı
sıkıya bağlı olmasıdır. GÇ-farkındalıklı bir çekirdek, tanımı gereği taşınabilir
değildir; her yeni hedef için yeniden yazılması gerekir.

Mamba'nın çözümü üç tekniğin bileşimidir ve üçü de belirli bir donanım mimarisinin
özelliklerine yaslanır:

1. **Paralel (birleşmeli) tarama:** Özyineleme, birleşme özelliği taşıyan bir işlem
   olarak yeniden ifade edilir ve klasik paralel tarama (parallel/associative scan)
   algoritmalarıyla $O(L)$ ardışık adım yerine $O(\log L)$ derinlikte hesaplanır. Bu,
   ardışıklığı *algoritmik* düzeyde kırar; ancak paralel tarama, ilkel (primitive)
   olarak GPU programlama modellerinde vardır — ileride görüleceği gibi graf
   derleyicilerinin ve NPU'ların operatör kümelerinde genellikle yoktur.
2. **Çekirdek füzyonu ve SRAM-yerleşimi:** Ayrıklaştırma, tarama ve çıktı izdüşümü tek
   bir GPU çekirdeğinde (kernel) birleştirilir; genişletilmiş durum tensörü hiçbir
   zaman ana belleğe (HBM) yazılmaz, hesaplama boyunca hızlı yonga-içi bellekte (SRAM)
   tutulur. Verim kazancının önemli kısmı bu bellek hiyerarşisi yönetiminden gelir.
3. **Geri yayılımda yeniden hesaplama (recomputation):** Ara durumlar saklanmaz;
   geriye doğru geçişte yeniden üretilir. Bu, bellek ayak izini düşürür ve eğitimde
   uzun dizileri mümkün kılar.

Bu üçlünün ortak paydası şudur: hepsi **el yazması bir CUDA çekirdeği** olarak
gerçekleştirilmiştir. Kazanç, matematiksel formülasyonun değil, formülasyon ile NVIDIA
GPU mimarisinin (iş parçacığı blokları, paylaşımlı bellek, warp-düzeyi ilkeller)
bilinçli eşleştirilmesinin ürünüdür. Mamba deposundaki referans "PyTorch fallback"
yolu bu eşleştirmenin hiçbirini içermez: durumu açıkça materyalize eder ve taramayı
ardışık yürütür. Diğer bir deyişle, literatürde alıntılanan SSM hız avantajı bir
*mimari özelliği* gibi sunulur, ama teknik olarak bir *gerçekleştirim özelliğidir* —
ve bu gerçekleştirim tek bir donanım-yazılım ekosistemine (CUDA; türevi
gerçekleştirimlerde Triton) aittir. Apple Silicon tarafında bu çekirdeğin bir
karşılığı yoktur: `mamba-ssm` paketi Metal/MPS arka ucunda derlenmez ve Triton'un
macOS desteği bulunmaz [Gu-Dao-2023; ORT-Issue-27796].

Literatür bu bağımlılığı *belirtir* fakat *sonucunu ölçmez*. Mamba makalesi çekirdeğin
CUDA'ya özgü olduğunu açıkça yazar; onu izleyen görü-SSM çalışmaları (Bölüm 2.2) hız
tablolarını bu çekirdek üzerinde üretir; hiçbiri "bu çekirdeğin olmadığı bir hedefte
ne olur?" sorusunu sistematik biçimde sormaz. Bu tezin deneysel programı tam olarak bu
sorunun cevabıdır.

*Tezle bağ:* AS2'nin ("teorik avantaj nerede buharlaşıyor?") kuramsal öngörüsü bu alt
bölümde kurulur — avantaj çekirdeğe gömülüyse, çekirdeğin taşınamadığı her dağıtım
yığınında avantajın yeniden sınanması gerekir.

### 2.1.3 Mamba-2 ve Durum-Uzayı İkiliği (SSD)

Mamba-2 [Dao-Gu-2024], seçici SSM'ler ile attention arasında yapısal bir denklik kurar:
durum-uzayı ikiliği (state space duality, SSD). Durum matrisi belirli bir yapıya
(skaler-çarpan-birim biçimine) kısıtlandığında, SSM'nin dizi-dizi dönüşümü yarı-ayrılabilir
(semiseparable) bir matrisle çarpım olarak yazılabilir ve bu matris, maskeli attention
benzeri bir formülasyonla aynı cebirsel ailede yer alır. Bu denkliğin pratik sonucu,
hesaplamanın **blok bazlı matris çarpımlarına indirgenebilmesidir**: dizi sabit
uzunluklu parçalara (chunk) bölünür, parça içi hesap yoğun matris çarpımıyla, parçalar
arası aktarım küçük bir özyinelemeyle yapılır. Matris çarpımı modern hızlandırıcıların
(tensor core'lar dahil) en iyi optimize edilmiş yolu olduğundan, Mamba-2 aynı model
ailesini önemli ölçüde daha yüksek donanım verimiyle eğitmeyi mümkün kılar.

Parça-bazlı (chunked) hesaplamanın dağıtım açısından ihmal edilen bir yan ürünü vardır:
parça uzunluğu sabit seçildiğinde, parça içi hesaplamanın tensör biçimleri **statikleşir**.
Graf derleyicileri statik biçimli, tekrarlı yapıları iyi işler; dizi-uzunluğu-değişken
tek bir dev tarama yerine sabit boyutlu parça işlemlerinin dizilimi, derleyicinin
füzyon ve bellek planlama mekanizmalarına çok daha uygun bir hedeftir. SSD'nin özgün
motivasyonu tensor core kullanımı olsa da, aynı yapısal özellik graf-tabanlı dışa
aktarım için de umut vericidir — bu bağlantı literatürde kurulmamıştır.

Tez açısından SSD'nin önemi eğitim hızından bağımsızdır: SSD, seçici taramanın
**derleyici-dostu yeniden formülasyonlarının teorik olarak mümkün olduğunu** gösterir.
Aynı (ya da yakın) matematiksel işlev, "özel tarama çekirdeği gerektiren ardışık
özyineleme" yerine "standart matris çarpımları + kısa özyineleme" olarak ifade
edilebiliyorsa, graf derleyicilerinin ve NPU'ların iyi desteklediği operatörlerden
oluşan bir dışa aktarım yolu ilkece kurulabilir. Bu gözlem, tezin AS4 sorusunun
(parçalı/sabit-uzunluklu tarama formülasyonlarının ONNX ve Core ML hedeflerinde
sınanması) doğrudan dayanağıdır.

*Tezle bağ:* SSD, tezin katkı fazında (Bölüm 4.5) denenecek yeniden formülasyonun
kuramsal meşruiyetini sağlar; "tarama ilkeli yoksa model taşınamaz" hükmünün mutlak
olmadığını, uyarlamanın matematiksel alanının var olduğunu gösterir.

---

## 2.2 Görü için SSM Omurgaları

### 2.2.1 Vim, VMamba ve Tarama Stratejileri

SSM'leri görüye taşımanın temel problemi, görüntünün doğal bir dizi olmamasıdır.
Seçici tarama tek boyutlu ve yönlü bir işlemdir; iki boyutlu uzamsal komşuluğu bu
işleme beslemek için görüntü yamalarının (patch) bir veya birden çok **tarama
güzergâhına** (scan path) serileştirilmesi gerekir. Serileştirme kaçınılmaz olarak
uzamsal yapıyı bozar: satır-öncelikli bir güzergâhta dikey komşular, görüntü
genişliği kadar token arayla dizide yer alır; tek yönlü taramada her token yalnızca
güzergâhta kendinden önce gelenleri "görür". Görü-SSM tasarım uzayının büyük kısmı,
bu bozulmanın telafisi — güzergâh sayısı, yönü ve birleştirme biçimi — üzerine
kuruludur. Alandaki ilk iki çalışma problemi iki farklı stratejiyle çözer.

Vision Mamba (Vim) [Zhu-2024], ViT benzeri izotropik bir gövdede **çift yönlü tarama**
kullanır: yama dizisi hem ileri hem geri yönde taranır ve iki yönün çıktıları
birleştirilir. Çalışmanın verimlilik iddiası tezin motivasyonu açısından önemlidir:
Vim, yüksek çözünürlüklü girdilerde DeiT'e kıyasla **2.8× hız ve %86.8 daha az GPU
belleği** bildirir [Zhu-2024]. Bu rakamlar, dikkat mekanizmasının karesel
karmaşıklığına karşı SSM'in doğrusal karmaşıklığının somut kanıtı olarak literatürde
yaygın biçimde alıntılanır ve SSM omurgalarının "uç cihaz için doğal aday" olduğu
anlatısının çekirdeğini oluşturur. Ancak ölçümler, §2.1.2'de tanımlanan özel CUDA
çekirdeği üzerinde, NVIDIA GPU'da alınmıştır; iddianın başka yürütme ortamlarına
taşınabilirliği çalışmada ele alınmaz. Uç cihazların büyük bölümünün CUDA
çalıştırmadığı düşünüldüğünde, "uç için doğal aday" çıkarımı ile kanıtın alındığı
ortam arasında, bu tezin sınamayı üstlendiği örtük bir sıçrama vardır.

VMamba [Liu-2024] ise hiyerarşik (Swin benzeri, aşamalı çözünürlük düşüren) bir
gövdede **Çapraz-Tarama Modülü** (Cross-Scan Module, CSM) önerir: özellik haritası
dört güzergâhta — satır-öncelikli ve sütun-öncelikli, her ikisinin tersleriyle —
taranır ve dört taramanın çıktıları çapraz-birleştirme ile geri toplanır. Bu tasarım,
tek yönlü taramanın neden olduğu yön yanlılığını giderir ve doğrusal karmaşıklıkla
küresel alıcı alan (global receptive field) sağlar. VMamba, sınıflandırmanın yanı sıra
yoğun tahmin görevlerinde de güçlü sonuçlar bildirdiği ve resmî ADE20K
checkpoint'leri yayımlandığı için bu tezin birincil SSM omurgası olarak seçilmiştir
(Bölüm 3.2).

İki tasarım, dağıtım yükünün nerede biriktiği bakımından da ayrışır. Vim'in izotropik
gövdesi, tüm derinlik boyunca tek ve uzun bir dizi üzerinde çalışır: dağıtım maliyeti,
az sayıda ama çok uzun taramada toplanır. VMamba'nın hiyerarşik gövdesi ise
çözünürlüğü aşama aşama düşürür: erken aşamalarda diziler uzun, geç aşamalarda
kısadır ve her aşamada dört güzergâh yürütülür — maliyet, çok sayıda orta uzunlukta
taramaya dağılır. Graf temsili dizi uzunluğuyla ölçeklendiğinde (§2.5.1) bu iki
profil farklı arıza kipleri üretebilir: tek dev taramanın graf patlaması ile çok
sayıda taramanın birikimli dağıtım/dispatch yükü aynı toplam token sayısında farklı
davranabilir. Literatürde bu ayrım hiç incelenmemiştir; tezin ölçüm matrisinde omurga
seçimi bu yüzden yalnızca doğruluk temelli değil, tarama topolojisi temelli bir
değişkendir.

Dağıtım perspektifinden bakıldığında tarama stratejilerinin gözden kaçan bir maliyeti
daha vardır: her güzergâh ayrı bir seçici taramadır. Çift yönlü tarama işlem sayısını iki,
CSM dört katına çıkarır; hiyerarşik gövdelerde bu taramalar her aşamanın kendi dizi
uzunluğunda tekrarlanır. GPU'da füzyonlu çekirdek bu maliyeti büyük ölçüde emer;
ancak taramanın graf temsiline açıldığı dışa aktarım senaryolarında (Bölüm 2.5.1)
güzergâh sayısı, graf yapısının katsayısı hâline gelir. Literatürde tarama stratejileri
yalnızca doğruluk ve GPU hızıyla değerlendirilmiş, graf karmaşıklığına etkisi
raporlanmamıştır.

*Tezle bağ:* Vim'in DeiT kıyası, tezin Apple Silicon yığınında yeniden üretilmesi
metodolojik olarak zorunlu olan referans kıyastır; VMamba'nın CSM'i ise ölçülecek
birincil iş yükünün tarama topolojisini tanımlar.

### 2.2.2 Hibrit Yaklaşımlar — Saf SSM'in Sınırları

Saf SSM omurgalarını izleyen ikinci dalga, SSM bloklarını başka mekanizmalarla
harmanlar ve bu harmanlamanın kendisi alan hakkında bilgi vericidir. MambaVision
[Hatamizadeh-Kautz-2025], hiyerarşik bir omurganın son aşamalarına öz-dikkat
(self-attention) blokları yerleştirir ve bu hibrit tasarımla saf Mamba
varyantlarından daha iyi doğruluk-verim dengesi bildirir. Çalışmanın örtük mesajı
açıktır: en azından mevcut tarama stratejileriyle, **saf SSM görü omurgası tek başına
yeterli bulunmamıştır** — uzun menzilli, içerik-duyarlı etkileşim için son katmanlarda
attention'a geri dönülmektedir. VCMamba [VCMamba-2025] benzer bir harmanlamayı
evrişim tarafından yapar: erken aşamalarda evrişimli bloklar, geç aşamalarda çok yönlü
Mamba blokları.

Üçüncü bir damar, SSM bloğunun kendisini ucuzlatmayı hedefler. EfficientViM
[EfficientViM-2024], SSD formülasyonuna dayanan gizli-durum karıştırıcı (hidden state
mixer) tasarımıyla uç-odaklı bir omurga sunar ve ADE20K üzerinde SemanticFPN başlığıyla
41.3 mIoU / 0.45 ms gecikme bildirir. Dynamic Vision Mamba [DynamicViM-2025] dinamik
token/blok seçimiyle, eğitimsiz token azaltma çalışması [TokenReduction-2025] ise
çıkarım anında dizi kısaltmayla hesap tasarrufu sağlar; MAP [MAP-2024] maskeli
otoregresif ön-eğitimle hibrit omurgaların potansiyelini araştırır.

Bu literatürün tamamı için geçerli iki gözlem tezin konumunu keskinleştirir. Birincisi,
verimlilik iddiaları ya FLOPs/parametre sayısı üzerinden ya da NVIDIA GPU gecikmesiyle
desteklenir; dağıtım yığını değişkeni deney tasarımlarında yoktur (EfficientViM'in
0.45 ms değeri de GPU ölçümüdür). "Verimli" sıfatı bu çalışmalarda fiilen "CUDA
üzerinde verimli" anlamına gelir, fakat başlıklar ve özetler bu niteliği taşımaz;
verimlilik iddiası, kanıtlandığı ortamın sınırlarını aşan bir genellikte dolaşıma
girer. Bu terminolojik kayma masum değildir: uygulayıcıların mimari seçim kararları,
niteliksiz "verimli" etiketine dayanır. İkincisi, hibritleşme eğilimi dağıtım açısından
çift yönlü sonuç doğurur: attention ve evrişim blokları derleyici desteği olgun
operatörlerdir, dolayısıyla hibrit modellerin *SSM-olmayan kısmı* iyi taşınır — bu da
taşıma başarısızlıklarını izole biçimde SSM bloklarına atfetmeyi mümkün kılar.

*Tezle bağ:* Hibrit ve verimli varyantlar bu tezde rakip değil, ölçüm matrisinin
çeşitliliğini sağlayan aday iş yükleridir; alandaki "yeni omurga tasarımı" kapısının
doygunluğu (Bölüm 2.6), tezin katkısını tasarımdan ölçüme kaydıran bilinçli tercihin
gerekçesidir.

### 2.2.3 Yoğun Tahmin Görevlerinde SSM

Yoğun tahmin görevlerinde omurga, girdinin her pikseli için tahmin üreten bir boru
hattının parçasıdır: hiyerarşik omurganın ara aşamalarından alınan çok ölçekli özellik
haritaları, bir segmentasyon başlığında (bu tezde UPerNet [Xiao-2018]) birleştirilerek
piksel-bazlı sınıf haritasına çözülür. Sınıflandırmadan farklı olarak girdi
çözünürlüğü burada bir kalite değişkenidir: ince yapıların (direk, şerit, uzak nesne)
doğru bölütlenmesi yüksek çözünürlük ister ve kıyaslamalar bu nedenle 512×512
(ADE20K, 150 sınıf) ile 1024×2048 (Cityscapes) gibi büyük girdilerle yapılır.

Doğrusal karmaşıklığın en güçlü vaadi tam da burada, dizi uzunluğunun en hızlı
büyüdüğü yerde ortaya çıkar. Semantik segmentasyonda 512×512 girdi, tipik yama
boyutlarıyla binlerce token'lık dizilere karşılık gelir; Cityscapes'in 1024×2048
çözünürlüğünde bu sayı on binlere ulaşır.
Karesel karmaşıklıklı küresel attention bu rejimde pratik olmaktan çıkarken, SSM'in
doğrusal ölçeklenmesi kuramsal olarak belirleyici bir avantaja dönüşmelidir. Nitekim
görü-SSM literatürünün yoğun tahmin kolu hızla genişlemiştir: Sigma [Sigma-2024]
Siyam Mamba gövdesiyle çok-modlu semantik segmentasyon, MambaSeg [MambaSeg-2025]
görüntü-olay (event) verisi birleşimli segmentasyon, WinMamba [WinMamba-2025]
kaydırmalı pencere taramasıyla 3B nesne tespiti üzerine çalışır. Otonom sürüş odaklı
AutoMamba/RTMamba hattı [AutoMamba-2026], alandaki nadir gömülü-cihaz ölçümlerinden
birini içerir: Jetson AGX Orin üzerinde hız testi. Ancak Jetson bir CUDA cihazıdır;
ölçüm, özel çekirdek ekosisteminin *içinde* kalır.

Segmentasyon pratiğinin bir ayrıntısı, dağıtım tartışmasıyla doğrudan kesişir:
yüksek çözünürlüklü değerlendirme çoğunlukla **kayan pencere** (sliding window)
çıkarımıyla yapılır — büyük görüntü, sabit boyutlu (ör. 512×512) örtüşen kırpmalara
bölünür ve model her kırpmada ayrı çalıştırılır. Bu pratik, dağıtım senaryosunda girdi
biçimini fiilen sabitler; statik-biçimli dışa aktarımın (§2.5.1) segmentasyonda
sanıldığından daha uygulanabilir olmasının nedeni budur. Öte yandan pencere başına
maliyet, toplam gecikmeyi pencere sayısıyla çarpar: çıkarım katmanındaki küçük bir
verimsizlik, tam-sahne segmentasyonunda katlanarak büyür.

Yoğun tahmin, tez açısından yalnızca "SSM'in parladığı görev" değil, aynı zamanda
**dağıtım probleminin en sert hâlidir**. Çözünürlük büyüdükçe dizi uzunluğu büyür;
Bölüm 2.5.1'de gösterileceği gibi, seçici taramanın graf temsili dizi uzunluğuyla
ölçeklendiğinden, doğrusal karmaşıklığın çıkarım maliyetinde sağladığı kuramsal
tasarruf ile graf temsilinin dönüşüm/yükleme maliyetinde yarattığı büyüme aynı
değişkene bağlıdır. Literatür bu gerilimi görmez, çünkü iki tarafı iki ayrı topluluk
ölçmektedir: model çalışmaları GPU gecikmesini, dağıtım pratiği ise (akademik yayın
dışı kanallarda) araç zinciri arızalarını raporlar.

*Tezle bağ:* Tezin birincil görevi (ADE20K semantik segmentasyonu, Bölüm 3.3) bu alt
bölümdeki gerekçeyle seçilmiştir — yoğun tahmin, hem SSM avantaj iddiasının hem de
dağıtım maliyetinin aynı anda en büyük olduğu, dolayısıyla ayrıştırıcı gücü en yüksek
görevdir.

---

## 2.3 Karşılaştırma Mimarileri: ViT, Swin, ConvNeXt

SSM omurgalarının "neye göre" verimli olduğu sorusu, karşılaştırma eksenini tanımlar.
Bu tezde eksen üç mimari aileden oluşur ve her birinin seçilme gerekçesi işlevseldir.

**ViT ve DeiT.** Vision Transformer [Dosovitskiy-2020], görüntüyü yama dizisine çevirip
saf Transformer kodlayıcıdan geçiren mimaridir. Öz-dikkat (self-attention), her token
çiftinin etkileşimini açıkça hesapladığından token sayısında karesel — $N$ token için
$O(N^2)$ — bellek ve hesap maliyeti taşır; SSM literatürünün karşısına konumlandığı
temel maliyet budur. Buna karşılık attention'ın dağıtım profili SSM'in tam tersidir:
hesap, tümüyle yoğun matris çarpımlarından ve softmax'tan oluşur, zamansal bağımlılık
içermez ve her graf derleyicisinin birinci sınıf desteklediği operatörlere birebir
düşer. Yani ViT "pahalı ama taşınabilir", seçici SSM "ucuz ama taşınması zor"
kutuplarını temsil eder — tezin ölçtüğü gerilim bu iki eksenin çapraz ürünüdür. DeiT
[Touvron-2020], aynı mimariyi veri-verimli eğitim reçetesiyle ImageNet ölçeğinde
eğitilebilir kılmıştır ve Vim'in verimlilik iddiasının doğrudan kıyas hedefidir.
Vim–DeiT kıyasını bu tezin dağıtım yığınlarında tekrarlamak, "bildirilen" ile
"gerçekleşen" avantajı aynı model çifti üzerinden karşılaştırmayı sağlar.

**Swin.** Swin Transformer [Liu-2021], attention'ı sabit boyutlu yerel pencerelere
kısıtlayarak karmaşıklığı token sayısında doğrusala indirir; ardışık katmanlarda
pencere ızgarasını kaydırarak (shifted windows) pencereler-arası bilgi akışını sağlar
ve hiyerarşik piramidiyle yoğun tahminin fiilî standart omurgalarından biridir. Swin'in varlığı kıyası
keskinleştirir: SSM'in rakibi yalnızca karesel küresel attention değil, doğrusal
karmaşıklığa çoktan ulaşmış pencere-tabanlı attention'dır. Ayrıca Swin'in operatörleri
(pencere bölümleme, yoğun matris çarpımları) derleyici desteği olgun işlemlerdir —
"doğrusal karmaşıklıklı ama derleyici-dostu" bir referans noktası oluşturur.

**ConvNeXt.** ConvNeXt [Liu-2022], ResNet iskeletini ViT çağının tasarım ve eğitim
reçetesiyle — büyük çekirdekli derinlemesine evrişimler, ters darboğaz blokları,
LayerNorm, GELU ve modern veri artırma/düzenlileştirme şemaları — adım adım modernize
ederek saf evrişimli mimarinin hiyerarşik Transformer'larla rekabet edebildiğini
gösteren çalışmadır. Deney tasarımındaki rolü kontrol grubudur: SSM omurgası bir
kıyasta öne geçiyorsa, bu üstünlüğün mimari aileden mi yoksa modern eğitim reçetesinden
mi geldiğini ayırt etmeyi sağlar. Dağıtım açısından ConvNeXt ayrıca alt sınır işlevi
görür: evrişim, tüm graf derleyicilerinin ve NPU'ların en olgun biçimde desteklediği
operatördür; ConvNeXt'in bir yığındaki performansı, o yığının "en iyi ihtimalle" ne
verebildiğinin göstergesidir.

Segmentasyon tarafında UPerNet [Xiao-2018] tüm omurgalar için sabit başlık olarak
kullanılır (SegFormer [Xie-2021] verimli alternatif olarak not edilir); başlık ve
değerlendirme protokolünün sabitlenmesi, ölçülen farkların omurgaya atfedilebilmesinin
ön koşuludur. Karşılaştırmanın doğruluk ekseni ise şu pratik gerçekle şekillenir: bu
tez modelleri sıfırdan eğitmediği, yayımlanmış ADE20K checkpoint'lerini kullandığı
için omurgaların mIoU değerleri birebir eşit değildir. Bu nedenle karşılaştırma
"eşit doğrulukta gecikme" yerine **doğruluk–gecikme Pareto düzlemi** üzerinden yapılır
(Bölüm 3.2): her omurga, her yığında bir (mIoU, gecikme) noktası üretir ve sorular —
hangi mimari hangi yığında Pareto-önünde kalıyor, yığın değişince önsıra nasıl
değişiyor — düzlem üzerinden cevaplanır. Bu kurgu, farklı doğruluktaki modellerin
kıyasını bilimsel olarak geçerli kılar.

*Tezle bağ:* Bu üç aile, Pareto düzleminin (doğruluk–gecikme, Bölüm 3.2) SSM-dışı
noktalarını üretir; özellikle ConvNeXt, "derleyici-model uyumu" değişkeninin deneysel
kontrolüdür.

---

## 2.4 Model Nicemleme

### 2.4.1 Genel PTQ/QAT Çerçevesi

Nicemleme (quantization), model parametrelerinin ve/veya aktivasyonlarının düşük bit
genişlikli tam sayı ya da düşük hassasiyetli temsillere indirgenmesidir; uç dağıtımda
bellek ayak izini, bant genişliği ihtiyacını ve — donanım desteği varsa — hesap
süresini düşürmenin standart aracıdır. Yaygın çerçeve tekdüze afin nicemlemedir:
gerçel değer $r$, ölçek $s$ ve sıfır noktası $z$ ile $q = \mathrm{round}(r/s) + z$
tam sayısına eşlenir. Ölçeklerin hangi granülerlikte tutulduğu (tensör-başına,
kanal-başına, grup-başına) doğruluk–maliyet dengesinin ana ayar düğmesidir.

İki ana rejim ayrılır. **Eğitim-sonrası nicemleme** (post-training quantization, PTQ),
eğitilmiş modeli küçük bir kalibrasyon kümesiyle, yeniden eğitim olmaksızın dönüştürür;
ucuz ve hızlıdır, uç dağıtım pratiğinin varsayılan yoludur. Kalibrasyon, aktivasyon
aralıklarının temsili girdiler üzerinde gözlemlenerek ölçeklerin belirlenmesidir;
min–maks, yüzdelik-dilim kırpması ve hata-enküçültme gibi stratejiler, aralık
kapsayıcılığı ile gövde çözünürlüğü arasındaki dengeyi farklı noktalarda kurar.
**Nicemleme-farkındalıklı eğitim** (quantization-aware training, QAT) ise nicemleme
gürültüsünü eğitime dahil eder; daha dayanıklıdır ama tam eğitim maliyeti gerektirir.

Kapsam bakımından da iki seçenek ayrılır: yalnızca ağırlıkların nicemlendiği
**salt-ağırlık** şemaları (bellek ve paket boyutu kazancı; hesap hâlâ yüksek
hassasiyette) ve ağırlık ile aktivasyonların birlikte nicemlendiği **W8A8 benzeri**
şemalar (tam sayı aritmetiği; donanım desteği varsa hız kazancı). Apple yığınında buna
bir üçüncü mekanizma eklenir: **palettizasyon** (palettization) — ağırlıkların küçük
bir arama tablosundaki (lookup table) temsilcilere kümelenmesi; tekdüze nicemlemeden
farklı bir sıkıştırma ailesidir ve coremltools'un W4 seviyesindeki ana aracıdır. Vurgu
gereken nokta şudur: nicemlemenin *gecikme* kazancı otomatik değildir — düşük bit
genişliğinin hıza dönüşmesi, hedef donanımın o bit genişliğinde aritmetik yürütme
desteğine ve derleyicinin bunu kullanan çekirdekler üretmesine bağlıdır. Bu nedenle
tezde nicemleme sonuçları yalnızca doğruluk kaybıyla değil, yığın-başına gerçekleşen
gecikme değişimiyle birlikte raporlanır (AS3).

Bu tezin kapsamı, dağıtım gerçekliğiyle tutarlı olarak PTQ'dur ve Apple yığınının
kendi mekanizmaları (coremltools ağırlık/aktivasyon nicemlemesi, palettizasyon; ONNX
Runtime INT8) kullanılır (Bölüm 3.6). Aktivasyon nicemlemesinin başlıca düşmanı, LLM
literatüründen bilinen **aykırı değerlerdir** (outliers): birkaç kanaldaki aşırı büyük
aktivasyonlar, tensör ölçeğini domine ederek kalan değerlerin etkin çözünürlüğünü yok
eder.

*Tezle bağ:* PTQ, tezde bağımsız bir katkı alanı değil, ölçüm matrisinin bir eksenidir
(nicemleme seviyesi × yığın × model); bu alt bölüm eksenin terminolojisini sabitler.

### 2.4.2 SSM'e Özgü Nicemleme Zorlukları

SSM'lerin nicemlenmesi, Transformer pratiğinin doğrudan kopyalanamadığı bir alan
olarak 2024 sonundan itibaren kendi literatürünü oluşturmuştur. Alanı açan çalışma
PTQ4VM [PTQ4VM-2024], görsel Mamba omurgalarındaki aktivasyon istatistiklerini üç
başlıkta sınıflandırır ve bu taksonomi, sonraki çalışmaların ortak dilini kurmuştur:

1. **Token-bazlı varyans:** Aktivasyon dağılımı token'dan token'a güçlü biçimde
   değişir; tarama güzergâhındaki konum, istatistikleri sistematik olarak kaydırır.
   Tensör-başına tek ölçek bu değişimi temsil edemez.
2. **Kanal-bazlı aykırı değerler:** Belirli kanallar tutarlı biçimde aşırı büyük
   değerler taşır — LLM'lerdeki aykırı kanal olgusunun görsel SSM'deki karşılığı.
3. **Uzun kuyruklu aktivasyonlar:** Dağılımların kuyrukları ağırdır; kırpma
   (clipping) eşiği seçimi, gövde çözünürlüğü ile kuyruk sadakati arasında sert bir
   ödünleşim yaratır.

Bu üç olgunun SSM'de neden yapısal olduğu, seçicilik mekanizmasından izlenebilir.
Durum, tarama güzergâhı boyunca biriktiği için bir token'ın aktivasyon istatistiği
yalnızca kendi içeriğine değil, güzergâhta o âna dek biriken duruma da bağlıdır —
token-bazlı varyansın kaynağı budur. Girdiye bağlı $\Delta$ kapısı, belirli kanalları
girdi içeriğine göre keskin biçimde açıp kapattığından kanal ölçekleri arasındaki
makas büyür; ayrıklaştırmadaki üstel dönüşüm ise dağılım kuyruklarını doğal olarak
ağırlaştırır. Yani Transformer nicemleme pratiğinin aksine, SSM'deki nicemleme
zorlukları eğitim tesadüflerinin değil, mimarinin tanımlayıcı mekanizmalarının
ürünüdür ve mimari kullanıldığı sürece ortadan kalkmaları beklenmez.

PTQ4VM'in önerdiği Per-Token Static (PTS) nicemleme, token konumuna bağlı statik
ölçeklerle ilk iki problemi hedefler ve GPU'da 1.83× hızlanmayla birlikte 15 dakikanın
altında dönüşüm süresi bildirir. Mamba-PTQ [MambaPTQ-2024], zorluğun kökenini bağımsız
biçimde doğrular: Mamba'nın nicemleme direnci, LLM'lerde olduğu gibi aktivasyon aykırı
değerlerinden kaynaklanmaktadır. QMamba [QMamba] *[DOĞRULANACAK: yayın yılı ve mekânı]*,
uzun kuyruk problemine dağılım-farkındalıklı bir çözüm (Long-tailed Skewness
Quantization, LtSQ) ve zamansal boyutta grup nicemlemesi (Temporal Group Quantization,
TGQ) önerir — zamansal gruplama, bir sonraki alt bölümdeki dinamik problemin erken bir
kabulü olarak okunabilir.

Bu literatürün ortak sınırı görev kapsamıdır: değerlendirmeler ağırlıklı olarak
ImageNet sınıflandırması üzerindedir. Yoğun tahmine en yakın önceki iş, görüntü
restorasyonunda nicemlenmiş Mamba'yı inceleyen Q-MambaIR'dir [QMambaIR-2025]; semantik
segmentasyon gibi yüksek çözünürlüklü, uzun dizili görevlerde SSM nicemlemesinin
davranışı sistematik olarak raporlanmamıştır.

*Tezle bağ:* PTQ4VM taksonomisi, tezin AS3 sorusunun ("PTQ yoğun tahmine taşındığında
ne olur?") analiz çerçevesidir; segmentasyon deneylerindeki doğruluk kaybı bu üç
problem sınıfına geri haritalanarak yorumlanacaktır.

### 2.4.3 Dinamik Aykırı Değer Problemi

Statik kalibrasyonun örtük varsayımı, aktivasyon dağılımlarının girdiler ve zaman
adımları boyunca yeterince kararlı olduğudur: kalibrasyon kümesinde gözlenen aralık,
çıkarımda karşılaşılacak aralığın iyi bir kestirimi olmalıdır. OuroMamba
[OuroMamba-2025], SSM nicemlemesinde bu varsayımı doğrudan hedef alan en sert bulguyu
ekler: aykırı değerlerin *konumu ve büyüklüğü zaman adımları arasında dinamik olarak
değişir*.
Transformer'larda aykırı kanallar büyük ölçüde girdiden bağımsız ve kararlıdır; bu
kararlılık, kalibrasyon kümesinden çıkarılan statik ölçeklerin işe yaramasının
nedenidir. Seçici SSM'de ise durum özyinelemesi girdiye bağlı olduğundan aktivasyon
istatistikleri taramanın seyriyle birlikte evrilir; kalibrasyonda görülen aykırı deseni
çıkarımda tekrarlanmaz. OuroMamba, statik PTQ'nun bu nedenle çöktüğünü gösterir ve
veri-gerektirmeyen, dinamik aykırı değer tespitli bir alternatif önerir.

Bu bulgu, tez bağlamında iki soruyu doğurur. Birincisi ölçek sorusudur: dizi uzunluğu
büyüdükçe (segmentasyonda on binlerce token) zamansal istatistik kayması için daha
fazla alan açılır — sınıflandırma dizilerinde tolere edilebilen dinamiklik, yoğun
tahmin rejiminde büyüyor mu? AS3'ün "yüksek çözünürlük aykırı değer profilini
değiştiriyor mu?" alt sorusu doğrudan buradan türetilmiştir. İkincisi dağıtım
sorusudur: dinamik ölçekleme gerektiren bir çözüm, çıkarım anında istatistik toplayan
özel çekirdekler ister; oysa uç yığınların nicemleme mekanizmaları (coremltools, ORT
statik INT8) tam da statik ölçek varsayımı üzerine kuruludur. Yani SSM nicemleme
literatürünün önerdiği çözümler ile uç araç zincirlerinin sunduğu mekanizmalar
arasında, henüz kimsenin ölçmediği bir uyumsuzluk vardır — bu, derleyici-model
uyumsuzluğunun (§2.5.4) nicemleme düzlemindeki izdüşümüdür.

Alanın donanım ucundaki çalışmalar — FPGA üzerinde algoritma-donanım ortak tasarımı
yapan ViM-Q [ViM-Q-2026] ve aşırı düşük bit genişliğini QAT ile zorlayan Ternary Mamba
[TernaryMamba-2026] — problemi özel donanım tasarlayarak çözer; ticari, sabit işlevli
NPU'ların dünyasına bu çözümler taşınamaz ve tez kapsamı dışındadır. Kapsam sınırının
kendisi bilgi vericidir: SSM nicemleme literatürünün çözüm uzayı, ya özel çıkarım
çekirdeği (dinamik ölçekleme) ya özel donanım (FPGA) gerektirmektedir — yani alan,
farkında olmadan, standart dağıtım yığınlarının SSM'e yetmediği önermesini kendi
çözümlerinin ön koşullarında yeniden üretmektedir.

*Tezle bağ:* Dinamik aykırı değer bulgusu, tezin nicemleme deneylerinde beklenen
başarısızlık kipini önceden tanımlar; deney sonuçları bu kuramsal öngörünün yoğun
tahminde geçerli olup olmadığını sınayacaktır.

---

## 2.5 Dağıtım Yığınları ve Derleyiciler

Bir modelin eğitim ortamından uç cihaza yolculuğu, tipik olarak üç aşamalı bir boru
hattından geçer: modelin işlem grafının bir **ara temsile** (intermediate
representation) dışa aktarımı, bu temsilin hedef donanım için **derlenmesi/optimize
edilmesi** ve derlenmiş yapının bir **çıkarım motoru** (inference runtime) tarafından
yürütülmesi. Bu bölüm, tezin ölçtüğü yığınların her aşamada hangi varsayımları
yaptığını ve bu varsayımların seçici taramayla nerede çeliştiğini inceler.

### 2.5.1 ONNX ve Graf Temsili; `Loop` Operatörü

ONNX (Open Neural Network Exchange), modeli tensör operatörlerinden oluşan statik bir
veri-akış grafı olarak temsil eden, çerçeveler-arası fiilî standarttır. Temsilin gücü
statikliğinden gelir: graf, yürütmeden önce bütünüyle bilinir; çıkarım motoru operatör
füzyonu, bellek planlaması ve donanım-özgü çekirdek seçimi gibi optimizasyonları bu
bütünsel görünüm üzerinde yapar. Kontrol akışı — döngü ve koşul — bu resme sonradan
eklenmiş istisnadır: `Loop`, `Scan` ve `If` operatörleri, gövdelerini birer alt-graf
olarak taşır ve çıkarım motoru bu alt-grafları yorumlayıcı benzeri bir mekanizmayla,
her yinelemede yeniden yürütür.

Bir PyTorch modelinin ONNX'e hangi biçimde dışa aktarılacağını, dışa aktarım
mekanizması belirler. **İzleme** (tracing), modeli örnek girdiyle bir kez çalıştırıp
fiilen yürütülen operatörleri kaydeder; Python düzeyindeki kontrol akışı bu kayıtta
görünmez — döngü, kaydedilen işlemlerin L kez tekrarına, yani açılmış (unrolled) grafa
dönüşür. **Betikleme** (scripting) ve daha yeni `torch.export` tabanlı yollar ise
kontrol akışını sembolik olarak yakalayıp `Loop`/`Scan` operatörlerine çevirebilir.
Dolayısıyla aşağıdaki iki yol, aynı modelin iki farklı dışa aktarım mekanizmasından
geçmiş hâlidir ve pratisyen çoğu zaman ikisi arasında seçim yapmak zorundadır:

1. **Döngüyü korumak (`Loop`):** Tarama, gövdesi tek zaman adımı olan bir `Loop`
   düğümüne çevrilir. Her yineleme, motorun alt-graf yürütme mekanizmasından geçer:
   operatör dağıtım (dispatch) yükü, ara tensörlerin sınır aşırı kopyalanması ve
   füzyon imkânsızlığı, adım başına sabit bir vergi bindirir. Dizi uzunluğu L kadar
   yinelenen bu vergi, GPU çekirdeğinde nanosaniyeler mertebesinde olan adım
   maliyetini milisaniyelere taşıyabilir.
2. **Döngüyü açmak (unroll):** İzleme-tabanlı (tracing) dışa aktarım, döngüyü
   L kopya hâlinde grafa açar. Yorumlayıcı vergisi ödenmez; bedel graf boyutuna
   taşınır — düğüm sayısı ve dosya boyutu L ile doğrusal büyür, graf optimizasyon ve
   yükleme aşamaları bu boyutla (çoğu zaman süper-doğrusal) ölçeklenir.

ONNX'in `Scan` operatörü, ikilemin çözümü gibi görünebilir: sabit yineleme sayılı,
dizi-boyunca-tarama desenine özel bir döngü biçimidir ve tam da SSM taramasının
desenini adlandırır. Ancak `Scan`, temsil düzeyinde bir özelleşmedir, yürütme
düzeyinde değil: gövde alt-grafı yine yineleme başına yorumlayıcı mekanizmasından
geçer; hiçbir yaygın çıkarım motoru `Scan`'i paralel tarama algoritmasına ya da
füzyonlu bir çekirdeğe indirgemez. Desenin adının operatör kümesinde bulunması ile
deseni verimli yürüten bir arka ucun bulunması arasındaki bu fark, "operatör desteği
var" ifadesinin dağıtım tartışmalarında neden yanıltıcı olabildiğinin ders kitabı
örneğidir.

Bu ikilemin üretimdeki sonucu, akademik literatürde değil, mühendislik kanalında
belgelenmiştir. ONNX Runtime'ın 27796 numaralı sorunu [ORT-Issue-27796] — başlığıyla
"ONNX Loop op makes Mamba (SSM) models unusable on CPU and WebGPU" — 30 Mamba bloklu,
9.6M parametrelik küçük bir konuşma modelinin dışa aktarım sonrası davranışını
raporlar: seçici tarama, Apple M3 işlemcide 0.1 saniyelik ses için 1.7 saniye sürer —
gerçek zamanın 17 katı yavaş. Aynı raporda ikilemin öbür ucu da ölçülüdür: scripting
kullanılmadığında izleyici (tracer) taramayı 298 MB'lık düz bir grafa açar ve yalnızca
oturumun yüklenmesi 445 saniye alır. Tek bir vaka raporu olmasına karşın bu kayıt,
mekanizmanın iki semptomunu — yorumlayıcı vergisi ve graf patlaması — aynı model
üzerinde gösterdiği için tezin çıkış kanıtıdır.

İkilemin bir de biçim (shape) boyutu vardır. Açılmış graf, dizi uzunluğunun dışa
aktarım anında sabitlenmesini gerektirir: her girdi çözünürlüğü ayrı bir graf, ayrı
bir dönüşüm ve ayrı bir dağıtım paketi demektir. Dinamik dizi uzunluğu isteniyorsa
unroll seçeneği tümüyle devre dışı kalır ve dışa aktarım ya `Loop` yoluna ya da
başarısızlığa mahkûmdur. Evrişimli ve attention-tabanlı modellerde aynı seçim çok daha
yumuşaktır: graf yapısı girdi boyutundan bağımsızdır, değişen yalnızca tensör
boyutlarıdır. Seçici taramada ise **graf yapısının kendisi** dizi uzunluğunun
fonksiyonudur — mimari ile temsil arasındaki uyumsuzluğun en özlü ifadesi budur.

*Tezle bağ:* Bu alt bölümdeki ikilem, tezin dört katmanlı maliyet modelinin (§2.6)
kuramsal temelidir: `Loop` yolu maliyeti çıkarım gecikmesine, unroll yolu ise
dönüşüm süresi, paket boyutu ve yükleme süresine dağıtır — dördü birlikte
ölçülmedikçe tablo eksiktir.

### 2.5.2 TensorRT ve Çekirdek Füzyonu

NVIDIA ekosisteminde graf derleyiciliğinin olgun örneği TensorRT'dir: ONNX ya da
çerçeve grafını alır, katman/operatör füzyonu, hassasiyet kalibrasyonu (FP16/INT8) ve
donanıma özgü çekirdek seçimi (kernel autotuning) uygulayarak, hedef GPU'ya özgü bir
yürütme motoru (engine) üretir. Derleme çevrimdışı ve pahalıdır; karşılığında çıkarım
yolu, elle yazılmış çekirdeklere yaklaşan verimle çalışır. Bu model — ağır çevrimdışı
derleme, hafif çevrimiçi yürütme — tüm modern dağıtım yığınlarının paylaştığı
şablondur ve tezin dört katmanlı maliyet modelindeki "dönüşüm/derleme süresi"
katmanının neden bağımsız bir metrik olduğunu açıklar: şablon, derleme maliyetinin
*sınırlı* olduğu varsayımına dayanır ve seçici taramanın bu varsayımı zorladığına dair
işaretler (§2.5.1) şablonun kendisini sorgulatır.
Füzyonun verim üzerindeki belirleyiciliği yalnızca NVIDIA'ya özgü değildir; örneğin
AMD NPU'ları üzerinde füzyonlu karma-hassasiyet çekirdekleriyle elde edilen kazançlar
[TileFuse-2026], "operatörleri tek tek yürütmek ile birleşik yürütmek arasındaki fark"
olgusunun donanımlar-üstü olduğunu gösterir. Ancak CUDA ekosisteminin SSM'ler
açısından ayırt edici özelliği, derleyici yolunun *yanında* el yazması tarama
çekirdeklerinin de var olmasıdır: NVIDIA hedefinde pratisyen, graf derleyicisi
tarama ile baş edemediğinde özel çekirdeğe kaçabilir. Bu kaçış yolu, sorunun NVIDIA
platformlarında görünmez kalmasının nedenlerindendir. Bu tezin kapsamı Apple Silicon
olduğundan (Bölüm 1.4, [Revizyon]), TensorRT burada yalnızca genel arka plan olarak
anılır; CUDA tarafına ait sayılar tezde "bildirilen" değerler olarak literatürden
alıntılanacaktır.

*Tezle bağ:* TensorRT, "derleyici + kaçış yolu olarak özel çekirdek" modelinin
örneğidir; Apple yığınında bu kaçış yolunun yokluğu, tezin ölçtüğü sorunun neden
Apple tarafında sert biçimde görünür olduğunu açıklar.

### 2.5.3 Core ML ve Apple Neural Engine Mimarisi

Apple Silicon'da dağıtımın ana yolu Core ML'dir ve boru hattı ONNX'ten önemli
noktalarda ayrışır. PyTorch modeli, coremltools ile izlenip Apple'ın ara temsiline
(MIL — Model Intermediate Language) çevrilir ve `mlprogram` biçiminde paketlenir; bu
paket cihazda bir kez daha derlenerek (`.mlmodelc`) yürütülebilir biçime getirilir.
Dönüşüm zinciri, ONNX yolundakiyle aynı temel karaktere sahiptir: izleme-tabanlıdır ve
kontrol akışını graf yapısına açar; dolayısıyla §2.5.1'in ikilemi Core ML yolunda da
geçerlidir, üstelik dönüşüm katmanı (graf geçişleri, operatör eşleme, biçim çıkarımı)
ONNX'e göre daha ağırdır. Çalışma anında Core ML derleyicisi, grafı cihazdaki üç
hesaplama birimine — CPU, GPU ve Apple Neural Engine (ANE) — **katman bazında ve
şeffaf olmayan biçimde** dağıtır. Geliştirici `compute_units` parametresiyle yalnızca
bir *tercih* bildirir; hangi katmanın gerçekte nerede koştuğu ancak Xcode Core ML
Performance Report ile gözlemlenebilir. Bu şeffaflık eksikliği metodolojik bir tuzaktır:
ANE üzerinde LLM çıkarımını inceleyen pratik literatür [ANE-LLM-Inference], derleyicinin
yerleştirme kararlarının öngörülemezliğini, desteklenen operatör kümesinin
kısıtlılığını ve dispatch zamanlaması ile bellek yerleşimi (IOSurface) üzerinde
doğrudan kontrol bulunmadığını belgeler; kısa dizi uzunluklarında birçok Transformer
konfigürasyonunun hesap değil bellek bant genişliği sınırlı hâle geldiğini not eder.
Bu nedenle tezin ölçüm protokolü, "ANE'de çalışıyor" iddiasını yalnızca profil kanıtıyla
ileri sürer (Bölüm 3.5.6).

Apple Silicon'a giden ikinci bir yol, iki ekosistemi birleştirir: ONNX Runtime'ın
CoreML Yürütme Sağlayıcısı (Execution Provider, EP). Bu kipte model ONNX olarak
yüklenir; çalışma zamanı, grafın CoreML'in desteklediği alt-parçalarını Core ML'e
devreder, kalanını kendi CPU çekirdekleriyle yürütür. Graf böylece iki motor arasında
**bölümlenir** (partitioning) ve her bölümleme sınırında veri, motorlar arasında el
değiştirir. Desteklenen operatör adacıkları ne kadar parçalıysa sınır geçişi o kadar
sıklaşır; kontrol akışı ya da egzotik operatörler içeren graflarda EP'nin devredebildiği
pay küçülür ve yol, pratikte saf CPU yürütmesine yaklaşır. Bu hibrit kip, tezin ölçüm
matrisinde ayrı bir hücredir: seçici taramanın bölümleme davranışı — taramanın Core
ML'e devredilip devredilemediği, edilemiyorsa sınır geçişlerinin maliyeti — başka
hiçbir hücrede görünmeyen bir arıza kipini görünür kılar.

ANE'nin kendisi, genel amaçlı bir işlemci değil, evrişim ve yoğun matris işlemleri
etrafında tasarlanmış sabit işlevli bir sinir hızlandırıcısıdır; varlık nedeni ham
hız değil, **watt-başına iştir** — mobil ve dizüstü sınıfı cihazlarda sürekli çalışan
görü/dil iş yüklerini pil bütçesi içinde tutmak. Bu tasarım hedefi mimari
karakterini belirler: dar ama derin biçimde optimize edilmiş bir operatör kümesi,
sabit veri düzeni varsayımları ve ağırlıklı olarak düşük hassasiyetli (FP16 sınıfı)
aritmetik *[DOĞRULANACAK: M5 ANE'nin desteklediği hassasiyet kipleri — resmî belge
sınırlı]*. Genel programlanabilirlikten vazgeçmenin karşılığı, desteklenen desenlerde
CPU/GPU'ya kıyasla çarpıcı enerji verimidir; desteklenmeyen desenlerde ise derleyici
katmanları sessizce CPU/GPU'ya geri düşürür ve kazanç buharlaşır. Apple'ın kendi
araştırma yayını [Apple-ANE-2022], Transformer mimarilerinin ANE'de verimli
çalıştırılması için bir optimizasyon reçetesi tanımlar ve bu reçete, ANE mimarisinin
karakterini dolaylı ama güvenilir biçimde belgeler:

1. **(B, C, 1, S) veri düzeni:** Ara tensörler, ANE'nin 4-boyutlu, kanal-öncelikli
   (channels-first) mimarisine hizalanmalıdır; dizi ekseni son boyuta, kanal ekseni
   ikinci boyuta yerleştirilir. Standart Transformer'ın (B, S, C) düzeni bu hizaya
   uymaz ve doğrudan çevrimde verim kaybettirir.
2. **Büyük ara tensörlerin parçalanması:** Attention benzeri geniş ara sonuçlar,
   split/concat işlemleriyle parçalara bölünerek yonganın L2 önbelleğinde kalacak
   boyutlara indirilir; tek parça büyük tensörler önbellek yerleşimini bozar.
3. **Reshape/transpose minimizasyonu:** Boyut yeniden düzenleme işlemleri ANE'de
   bellek kopyası tetikler; reçete, hesaplamayı bu işlemlere gerek bırakmayan
   biçimde ifade etmeyi (ör. attention'ın `bchq,bkhc->bkhq` einsum formülasyonu)
   önerir.

Reçetenin referans gerçekleştirimi açık kaynak olarak yayımlanmıştır
[Apple-ANE-Repo]. Tezin konumu açısından belirleyici olgu şudur: **bu reçete
Transformer için vardır, SSM için yoktur.** Seçici taramanın ANE'nin operatör kümesi
ve veri düzeni kısıtları altında nasıl ifade edilmesi gerektiğine dair ne Apple'dan ne
akademik literatürden yayımlanmış bir kılavuz mevcuttur; SSD'nin matris-çarpım
formülasyonu (§2.1.3) ile bu reçetenin ilkeleri arasındaki köprü kurulmamıştır. Faz
0 ön gözlemlerimizin işaret ettiği dönüşüm-süresi davranışı bir yana, taramanın ANE'ye
hangi oranda yerleştirilebildiği dahi bilinmemektedir.

*Tezle bağ:* Bu alt bölüm, tezin özgün katkı alanının (Apple reçetesinin SSM'e
uyarlanması, AS4/Faz 4) hedef donanım kısıtlarını tanımlar; ANE reçetesinin üç ilkesi,
Bölüm 4.5'teki yeniden formülasyon denemelerinin tasarım kriterleridir.

### 2.5.4 Derleyici-Model Uyumsuzluğu Problemi

Önceki alt bölümlerin bulguları tek bir çerçevede toplanabilir; bu çerçeve, tekil
arıza raporlarını ortak bir nedene bağlar. Graf derleyicileri ve
NPU araç zincirleri, hedef iş yükü hakkında üç örtük varsayım yapar: **(i)** hesaplama,
statik ve çevrimsiz bir tensör-operatör grafıdır; **(ii)** operatörler, aralarında
zamansal bağımlılık olmayan, içsel olarak paralel yoğun işlemlerdir; **(iii)** operatör
kümesi, evrişim/matris-çarpımı/eleman-bazlı işlem üçgeninde kapalıdır. Bu varsayımlar
CNN'ler için tam, Transformer'lar için büyük ölçüde geçerlidir. Seçici tarama üçünü
birden ihlal eder: zamansal özyineleme çevrim demektir; adımlar arası bağımlılık içsel
paralelliği kırar; birleşmeli tarama (associative scan), bu araç zincirlerinin hiçbirinde
birinci sınıf ilkel değildir. Sonuç, taramanın ya kontrol akışına (yorumlayıcı vergisi)
ya da açılmış grafa (boyut patlaması) zorlanmasıdır — §2.5.1'in ikilemi, bu yapısal
uyumsuzluğun ONNX'teki özel hâlidir.

Uyumsuzluğun yapısal olduğunun en güçlü kanıtı, SSM'i uçta *başarıyla* çalıştıran
çalışmaların izlediği yoldur: hepsi graf derleyicisini terk etmiştir. FEMBA
[FEMBA-2026], mikrodenetleyici üzerinde çift yönlü Mamba tabanlı EEG modelini
dağıtırken TFLite Micro'yu kullanmamış, gerekçesini açıkça yazarak — seçici durum-uzayı
özyinelemesi genel amaçlı çıkarım motorunun yükünü kaldırmıyor — **el yazması
özyinelemeli bir C++ çalışma zamanı** geliştirmiştir. BabyMamba-HAR [BabyMamba-2026],
kaynak-kısıtlı cihazda insan aktivitesi tanıma için Brevitas nicemlemesinden doğrudan
optimize C çekirdeklerine inen **kendi araç zincirini** kurmuştur. İki bağımsız grubun
aynı kaçış desenine yönelmesi tesadüf değildir; VMamba resmî deposunun CUDA'sız bir
ortamda yamasız import dahi edilememesi gibi ekosistem gözlemleriyle birlikte
okunduğunda, desen tutarlıdır: **standart araç zinciri yolu SSM'ler için bugün
çalışmamaktadır ve çalışan her örnek, yolun dışına elle inşa edilmiştir.**

Uyumsuzluğun genel bağlamı, cihaz-üstü yapay zekâ literatüründe kısmen haritalanmıştır:
kapsamlı taramalar [OnDevice-Survey-2025; EfficientVLM-Survey-2025], uç dağıtımın
başarı hikâyelerinin ezici çoğunlukla evrişimli ve attention-tabanlı mimarilerden
geldiğini örtük biçimde belgeler — taranan sistemlerin operatör desteği, sıkıştırma
teknikleri ve donanım hedefleri hep bu iki ailenin etrafında olgunlaşmıştır. Donanım
zemini de aynı yönde ilerlemektedir: yeni nesil mobil NPU'lar düşük hassasiyetli yoğun
aritmetiğe (donanımsal int4 dahil) yatırım yapmakta [OnDeviceLLM-2026], ardışık
özyineleme desenine ise yatırım yapılmamaktadır. Başka bir deyişle ekosistem,
Transformer/CNN varsayımını her yıl biraz daha derinleştirmektedir; SSM'lerin
"gelecekte araç zincirleri olgunlaşır" beklentisi, bu eğilim veriyken kendiliğinden
gerçekleşecek bir öngörü değildir.

El yazması çalışma zamanı, araştırma prototipi için geçerli bir çözümdür; ancak ANE
gibi yalnızca sistem derleyicisi üzerinden erişilebilen kapalı hızlandırıcılarda bu
kaçış yolu mevcut değildir. Apple Silicon'da soru bu yüzden kaçınılmaz olarak şudur:
uyumsuzluk, modelin derleyicinin varsayımlarına yaklaştırılmasıyla — derleyici-dostu
yeniden formülasyonla — ne ölçüde kapatılabilir?

*Tezle bağ:* Bu alt bölüm, tezin ana hipotezini ("avantaj kaybının kaynağı, taramanın
ardışık yapısı ile derleyicilerin yürütme modeli arasındaki uyumsuzluktur") literatür
zeminine oturtur; Bölüm 4'ün deneyleri bu uyumsuzluğun her katmandaki nicel bedelini
ölçer.

---

## 2.6 Literatürdeki Boşluk ve Tezin Konumu

Önceki alt bölümlerin dökümü, alanın hangi kapılarının kapalı, hangilerinin açık
olduğunu netleştirir. **Kapalı kapılar:** SSM nicemleme algoritmaları (§2.4 —
PTQ4VM'den Ternary Mamba'ya uzanan yoğun bir hat) ve verimli SSM omurga tasarımı
(§2.2.2 — hibritler, dinamik token seçimi, uç-odaklı varyantlar) doymuş alanlardır;
bu tez ikisine de yeni öneri eklememektedir. **Kısmen açık:** nicemlemenin yoğun
tahmine transferi — en yakın iş görüntü restorasyonundadır [QMambaIR-2025], semantik
segmentasyon boştur. **Açık:** dağıtım yığını gerçekliğinin sistematik ölçümü ve ANE
üzerinde SSM.

Boşluğun kesin tarifi şudur. Görü-SSM literatürünün verimlilik kanıtları üç kaynaktan
gelir: **(i)** teorik vekil metrikler — FLOPs ve parametre sayısı; **(ii)** özel CUDA
çekirdekleriyle A100/RTX sınıfı GPU'larda alınan gecikme ölçümleri (Vim'in 2.8×
iddiası [Zhu-2024], EfficientViM'in 0.45 ms değeri [EfficientViM-2024], Jetson dahi
CUDA ekosisteminin içindedir [AutoMamba-2026]); **(iii)** özel tasarlanmış donanım —
FPGA ortak-tasarımları [ViM-Q-2026]. Üç kaynağın hiçbiri, bir pratisyenin bugün
elindeki gerçek dağıtım yolunu — çerçeveden dışa aktarım, graf derleyicisi, ticari
NPU — temsil etmez. Bu yolda ne olduğuna dair mevcut kanıt akademik değil,
mühendislik kanallarındadır: ONNX Runtime'ın Mamba'yı "kullanılamaz" ilan eden sorun
kaydı [ORT-Issue-27796] ve graf derleyicisini terk eden dağıtım çalışmaları
[FEMBA-2026; BabyMamba-2026]. Dağınık, tekil ve sistematikleştirilmemiş bu kanıtları
kontrollü bir deney matrisine dönüştüren yayımlanmış bir çalışma, bu taramanın
kapsamında tespit edilememiştir. Apple Neural Engine özelinde ise tablo daha da nettir:
ANE üzerinde SSM çalıştırmayı ölçen, raporlayan ya da optimize eden hiçbir çalışma
bulunamamıştır — Apple'ın ANE optimizasyon reçetesi [Apple-ANE-2022] Transformer'da
durur.

Ölçüm metodolojisi literatürü bu boşluğu kapatmaz, ama kapatacak çalışmanın nasıl
yapılması gerektiğini öğretir. Enerji-farkındalıklı kıyaslama protokolleri
[WattCounts-2026], termal stabilizasyon ve istatistiksel raporlama pratikleri
[LLM-Energy-Tradeoffs-2025] ile gömülü cihaz profilleme çalışmaları
[Jetson-Profiling-2025], güvenilir verimlilik ölçümünün asgari şartlarını — ısınma,
termal kontrol, senkronizasyon, medyan/P99 birlikte raporlama — belgelemiştir. Ancak
bu literatür mimari-agnostiktir: ölçüm nesnesi olarak LLM servisini ya da standart
CNN'leri alır; SSM'lerin dağıtım-katmanı davranışını hiçbir metodoloji çalışması
konu edinmemiştir. Bu tez, söz konusu protokolleri devralır (Bölüm 3.5) ve onları
literatürde ilk kez SSM × dağıtım-yığını matrisine uygular.

Boşluğun bir de *metrik* boyutu vardır. Literatürün tamamı verimliliği tek katmanda —
çıkarım gecikmesi (ve bazen bellek/enerji) — raporlar. Oysa §2.5.1'in ikilemi, seçici
taramanın dağıtım bedelinin çıkarım öncesi aşamalara kayabildiğini gösterir: aynı sorun
kaydında çıkarım yavaşlığı (17×), paket boyutu (298 MB) ve yükleme süresi (445 s) aynı
modelin üç ayrı semptomudur. Bu tez, bu gözlemi sistematik bir ölçüm çerçevesine
genelleştirir ve dağıtım maliyetini **dört katmanda** ayrıştırarak raporlar:
**(a)** dönüşüm/derleme süresi, **(b)** model yükleme süresi, **(c)** dağıtım paketi
boyutu ve **(d)** çıkarım gecikmesi (protokol: Bölüm 3.5). Literatürle
karşılaştırılabilirlik (d) üzerinden korunurken, (a)–(c) katmanları SSM'lerin uç
yaşayabilirliğini belirleyen ve bugüne dek ölçülmemiş boyutları görünür kılar: bir
modelin çıkarımı hızlı olsa bile, saatler süren dönüşüm, dakikalar süren yükleme ya da
yüzlerce megabaytlık paket, uç senaryoda aynı ölçüde diskalifiye edicidir.

Tezin konumu buna göre tek cümlede ifade edilebilir: **literatür SSM görü omurgalarının
özel CUDA çekirdekleriyle *ne kadar hızlı olabildiğini* ölçmüştür; bu tez, aynı
omurgaların Apple Silicon'un gerçek dağıtım yığınlarında — PyTorch eager/compile, ONNX
Runtime, Core ML'in CPU/GPU/ANE hedefleri — *ne kadar hızlı olduğunu* dört maliyet
katmanında ölçer ve aradaki farkın mekanizmasını gösterir.** Araştırma soruları bu
konumun işlemselleştirilmesidir: AS1 gerçekleşen profili çıkarır, AS2 farkın yığınlar
arası dağılımını izler, AS3 nicemleme literatürünün sınıflandırma-merkezli bulgularını
yoğun tahminde sınar, AS4 ise uyumsuzluğun yeniden formülasyonla kapatılabilirliğini —
Apple'ın Transformer reçetesinin SSM'e uyarlanması dahil — dener. Hipotez
doğrulanırsa sonuç, uygulayıcılar için dağıtım-farkındalıklı bir mimari seçim
kılavuzudur; çürütülürse SSM araç zincirlerinin olgunlaştığının ölçülmüş kanıtıdır.
Her iki durumda da boşluk kapanır — tezin bu konuya yaslanmasının nedeni budur.

Bölümün bütününden süzülen tablo şudur: matematiksel katman (§2.1) verimliliğin bir
gerçekleştirim özelliği olduğunu, model katmanı (§2.2–2.3) iddiaların tek ekosistemde
kanıtlandığını, nicemleme katmanı (§2.4) SSM'e özgü zorlukların mimariye içkin
olduğunu, dağıtım katmanı (§2.5) ise araç zincirlerinin varsayımlarıyla seçici
taramanın yapısı arasında sistematik bir uyumsuzluk bulunduğunu göstermektedir. Bu
dört gözlem bir arada, henüz kimsenin yürütmediği bir deneyi çağırır; izleyen bölüm
(Bölüm 3) bu deneyin tasarımını tanımlar.

---

*Sayfa hedefi: ~18. Atıflar `[YAZAR-YIL]` yer tutucu biçimindedir; kaynakça Faz 5'te
bağlanacaktır. `[DOĞRULANACAK]` etiketli noktalar kaynak metinler tam okunduğunda
kesinleştirilecektir.*


---

# 3. YÖNTEM *(TASLAK v1 — 13 Ağustos 2026)*

> **Dosya notu:** Bölüm 3.5 (Ölçüm Protokolü) ayrı dosyada tutulmaktadır:
> `tez/bolum-3.5-olcum-protokolu.md`. Bu dosyada 3.5'in içeriği tekrarlanmaz;
> ilgili yerlerde yalnızca "bkz. §3.5" atfı verilir.

Bölüm 2'nin vardığı sonuç, literatürün SSM görü omurgaları için *bildirdiği*
verimlilik ile bir pratisyenin gerçek dağıtım yolunda *bulacağı* verimlilik
arasındaki farkın hiç ölçülmemiş olduğuydu (§2.6). Bu bölüm, o farkı ölçülebilir
kılan deneyin tasarımını tanımlar. Yöntem bölümünün işi tarif değil savunmadır:
aşağıda her tasarım kararı, alternatifiyle birlikte ve gerekçesiyle sunulur.
Bölümün organizasyonu deney zincirini izler — önce neyin sabitlenip neyin
değiştirildiği (§3.1), sonra ölçüm nesnelerinin seçimi ve kimlik doğrulaması
(§3.2), veri (§3.3), ölçümün yapıldığı dağıtım yığınları (§3.4), ölçüm
protokolünün kendisi (§3.5), nicemleme deneylerinin protokolü (§3.6) ve
tekrarlanabilirlik altyapısı (§3.7).

## 3.1 Deneysel Tasarım ve Değişken Kontrolü

Tezin merkezi deneyi bir **kontrollü karşılaştırma matrisidir**: üç mimari
ailesinin birer temsilcisi (SSM, Transformer, CNN — §3.2), aynı yoğun-tahmin
görevi üzerinde, birden çok dağıtım yığını × giriş çözünürlüğü × nicemleme
seviyesi kombinasyonunda ölçülür. Matrisin bilimsel değeri, hücreler arasındaki
farkın yalnızca *ilgilenilen* değişkenlere atfedilebilmesine bağlıdır; bu da
geri kalan her şeyin sabitlenmesini gerektirir.

**Sabit tutulanlar:**

1. **Segmentasyon başlığı.** Üç modelde de aynı UPerNet başlığı [UPerNet-2018]
   kullanılır (512 kanal; üç modelde birebir 31.5M başlık parametresi).
   Böylece ölçülen her fark omurgaya atfedilebilir — başlık, matrisin
   "kontrol grubu"dur. Başlığı sabitleme kararı model seçimini de kısıtlar
   (bkz. §3.2'de EfficientViM'in dışlanması).
2. **Eğitim reçetesi.** Üç modelin ADE20K ağırlıkları da aynı standart mmseg
   reçetesiyle üretilmiştir: 160k iterasyon, 512×512 kırpma, ImageNet-1k
   ön-eğitimli omurga. Reçete farkı, doğruluk farkının mimariden mi eğitimden
   mi geldiğini ayırt edilemez kılacağından bu hizalama zorunludur.
3. **Test protokolü.** Tüm doğruluk doğrulamaları tek protokolde yapılır:
   "whole" (tam-görüntü) çıkarım, en-boy oranı korunarak kısa kenar 512'ye
   ölçekleme, /32 hizası için yansıma dolgusu. Modeller arasında protokol
   farkı (ör. kayan-pencere vs tam-görüntü) mIoU'da 0.5-1.0 puanlık yapay
   fark üretebildiğinden, bildirilen değerlerle karşılaştırmada protokol
   farkları açıkça not edilir (bkz. Bölüm 4.1, ConvNeXt dipnotu).
4. **Donanım ve ortam.** Tüm ölçümler tek makinede (Apple M5, 24 GB birleşik
   bellek, macOS) ve sabitlenmiş yazılım sürümleriyle (Python 3.12,
   torch 2.13, onnxruntime 1.28, coremltools 9.0 — §3.7) alınır. Tek-makine
   kısıtı bir sınırlılıktır (Bölüm 5.4); ama aynı zamanda cihazlar-arası
   değişkenliği sıfırlayan bir kontroldür.

**Değiştirilenler (bağımsız değişkenler):**

- **Omurga ailesi:** VMamba-T (SSM) / Swin-T (Transformer) / ConvNeXt-T (CNN).
- **Dağıtım yığını:** §3.4'te tanımlanan yedi-artı yürütme hücresi.
- **Giriş çözünürlüğü:** 256² → 512² (birincil) → 768²/1024² ölçekleme serisi;
  SSM'in dizi uzunluğuyla (L = H·W/16) ölçeklenen davranışını izole eder.
- **Nicemleme seviyesi:** fp32 taban çizgisi ile §3.6'daki ağırlık-nicemleme
  kipleri.

**Bağımlı değişkenler**, §2.6'da gerekçelendirilen dört maliyet katmanıdır:
dönüşüm/derleme süresi, model yükleme süresi, paket boyutu ve çıkarım gecikmesi;
bunlara bellek tepe noktası ve (telemetri mevcutken) çıkarım-başına enerji
eklenir. Ölçüm tanımları ve istatistiksel raporlama için bkz. §3.5.

### Araştırma sorularının tasarıma izdüşümü

Tasarımın her ekseni bir araştırma sorusuna hizmet eder ve bu eşleme baştan
sabitlenmiştir — deney bittikten sonra soruya uyan eksen aramak (HARKing)
bu şekilde dışlanır. **AS1** (gerçekleşen verimlilik profili), referans
yığındaki omurga × çözünürlük dilimiyle cevaplanır: yığın sabitlenir, mimari
ve dizi uzunluğu değişir. **AS2** (avantajın yığınlar arası buharlaşması),
matrisin yığın ekseni boyunca okunmasıdır: aynı model, aynı girdi, değişen
tek şey yürütme yolu — gözlenen her fark tanım gereği yığına aittir. **AS3**
(nicemlemenin yoğun tahmine transferi), nicemleme eksenini doğruluk ölçümüne
bağlar (§3.6). **AS4** (yeniden formülasyon) matrisin dışında ayrı bir
müdahale deneyidir; ancak başarı ölçütü yine bu matrisin hücreleriyle
tanımlanır — yeniden formüle edilmiş tarama, Faz 0'da başarısız olan hücreleri
(ör. VMamba × Core ML) açabildiği ve açtığı hücrede ölçülen maliyeti
düşürebildiği oranda başarılıdır. Bu kurgunun bir sonucu vurgulanmalıdır:
matristeki *boş* hücreler (dönüşemeyen, yüklenemeyen kombinasyonlar) deneyin
eksikliği değil, dört maliyet katmanının en üst basamağındaki sonuçlardır ve
Bölüm 4'te ölçülmüş değerlerle aynı statüde raporlanır.

### Eğitim yapılmaması kararı

Bu tezde hiçbir model eğitilmemiş; üç omurganın da **yayınlanmış resmî ADE20K
checkpoint'leri** kullanılmıştır. Bu karar iki gerekçeye dayanır ve ikisi de
açıkça yazılmalıdır.

Birincisi bilimseldir: kullanılan üç checkpoint zaten aynı mmseg reçetesiyle
(160k, UPerNet, 512×512) eğitilmiştir — yani "eşit reçete" kontrolü, yeniden
eğitim yapılmadan hâlihazırda sağlanmaktadır. Sıfırdan yeniden eğitim, aynı
reçeteyi hedeflese bile tohum, veri yükleme sırası ve donanım-bağımlı sayısal
farklar üzerinden **ek bir varyans kaynağı** katardı; üstelik üretilen
ağırlıklar literatürde bildirilen sonuçlarla doğrudan karşılaştırılamaz hâle
gelirdi. Yayınlanmış checkpoint kullanmak, tezin ölçümlerini literatürün kendi
referans noktalarına sabitler: Bölüm 4'te raporlanan her gecikme, okuyucunun
literatürden tanıdığı *o* modele aittir.

İkincisi pratiktir ve dürüstçe belirtilir: tezin donanımı yalnızca Apple
Silicon'dur ve VMamba'nın eğitimi, MPS arka ucunda karşılığı bulunmayan özel
CUDA çekirdeklerine (`mamba-ssm`, selective-scan çekirdekleri) bağımlıdır —
bu donanımda VMamba eğitimi fiilen olanaksızdır. Bu kısıt tesadüfen tezin
konusunun ta kendisidir: SSM ekosisteminin CUDA varsayımı, daha deney
tasarımı aşamasında bir tasarım kısıtı olarak kendini dayatmaktadır
(bkz. §3.4 ve Bölüm 4.3'teki kaynak-yaması bulguları).

## 3.2 Model Seçimi ve Doğruluk Doğrulama Protokolü

### Aile-başına temsilci ilkesi

Deney matrisi üç mimari aileden birer temsilci içerir: **VMamba-T**
[VMamba-2024] (SSM), **Swin-T** [Swin-2021] (hiyerarşik Transformer),
**ConvNeXt-T** [ConvNeXt-2022] (modern CNN). Seçim ölçütleri şunlardı:
(i) üçü de aynı "tiny" kapasite sınıfında olmalı (omurga parametreleri
27.5-29.9M bandında); (ii) üçünün de aynı reçeteyle eğitilmiş resmî
UPerNet+ADE20K checkpoint'i yayınlanmış olmalı (§3.1'in sabitleri);
(iii) her temsilci kendi ailesinin yoğun-tahmin literatüründeki standart
kıyas noktası olmalı. VMamba'nın SSM temsilcisi seçilmesinin ek gerekçesi,
görü-SSM literatüründe segmentasyon sonuçları en yaygın atıf alan ve resmî
deposu en etkin bakımda olan omurga olmasıdır (§2.2.1); Vim gibi düz (izotropik)
alternatiflerin hiyerarşik özellik piramidi üretmemesi, UPerNet sabitiyle
uyumsuzdur.

"Tiny" kapasite sınıfının seçilmesi de gerekçelendirilmelidir: uç dağıtım
bağlamında anlamlı olan sınıf budur (daha büyük varyantlar uç cihaz bellek
ve güç bütçelerini zorlar) ve üç ailenin en sık kıyaslanan, en iyi
belgelenmiş varyantları tiny/T sınıfındadır. Aile-başına *tek* temsilciyle
yetinilmesi ise bilinçli bir derinlik-genişlik ödünleşimidir: matrisin yığın
ekseni yedi-artı hücre içerdiğinden, her ek model matrisi çarpansal büyütür.
Tez, az modeli çok katmanda derinlemesine ölçmeyi, çok modeli tek katmanda
yüzeysel taramaya tercih eder; genellenebilirlik sınırı Bölüm 5.4'te açıkça
tartışılır.

### "Eşit doğruluk" yerine Pareto düzlemi

Özgün tasarım, omurgaları yeniden eğiterek doğruluğu eşitlemeyi ve tek eksenli
bir hız karşılaştırması yapmayı öngörüyordu. Eğitim yapılmaması kararıyla
(§3.1) bu yol kapanmıştır: yayınlanmış checkpoint'lerin mIoU değerleri farklıdır
(44.3-48.3 bandı) ve "VMamba daha yavaş ama daha doğru" gibi durumlar tek
eksende karşılaştırılamaz. Bunun yerine tez, her (omurga × yığın × çözünürlük)
hücresini **doğruluk-gecikme Pareto düzleminde bir nokta** olarak raporlar:
bir modelin üstünlüğü Pareto sınırında yer alıp almadığıyla, SSM iddiasının
testi ise VMamba noktalarının yığın değiştikçe sınırdan ne kadar uzaklaştığıyla
ifade edilir. Bu, bilgi kaybettiren bir ödün değil, iki değişkenli gerçekliği
tek değişkene indirgemeyi reddeden daha dürüst bir raporlama biçimidir;
kaldı ki nicemleme deneyleri (AS3) doğruluğu zaten hareketli bir eksen hâline
getirdiğinden, Pareto çerçevesi matrisin bütününe tek tip uygulanabilir.

### Yükleme doğruluğunun kanıtı: anahtar-uyumlu yeniden implementasyon

Hazır checkpoint kullanmanın metodolojik riski şudur: ağırlıklar doğru dosyadan
gelse bile *yanlış yüklenebilir* — eksik anahtar sessizce rastgele başlatmaya
düşer, mimari ayrıntı farkı (ör. yardımcı başlığın dahil edilmesi) çıktıyı
değiştirir. Bu riski kapatmak için üç model de mmseg çalışma zamanına bağımlı
olmadan, **checkpoint anahtar düzeniyle birebir uyumlu** saf-PyTorch modüller
olarak yeniden implemente edilmiştir (EK B; `src/models/`). mmseg'in kendisinin
kullanılmaması bilinçlidir: dağıtım deneyleri (export, trace, nicemleme)
modelin çıplak `nn.Module` hâlini gerektirir ve mmseg'in konfigürasyon/kayıt
katmanı hem export'a engel hem de bağımlılık yüzeyini büyüten bir etkendir.
Eğitime özgü yardımcı başlık (auxiliary FCN head) çıkarımda kullanılmadığından
yüklenmez; bu, mmseg'in kendi çıkarım davranışıyla eşdeğerdir.

Yeniden implementasyonun doğruluğu iki bağımsız kanıtla gösterilir:

1. **Yükleme bütünlüğü:** Üç modelde de checkpoint `missing=0 / unexpected=0`
   ile yüklenir — checkpoint'teki her tensör modelde bir karşılık bulur ve
   modelde karşılıksız kalan hiçbir parametre yoktur.
2. **Davranış eşdeğerliği:** Yüklenen her model, ADE20K doğrulama kümesinin
   tamamında (2 000 görüntü) değerlendirilir ve bildirilen mIoU yeniden
   üretilir: VMamba-T 48.33 (bildirilen 48.3), Swin-T 44.32 (44.41),
   ConvNeXt-T 45.42 (46.11; fark, bildirilen değerin kayan-pencere test
   protokolünden gelmesiyle açıklanır — ayrıntı ve tablo: Bölüm 4.1).

İki kanıt birlikte, "ölçtüğümüz model literatürdeki modeldir" iddiasını
yapısal (anahtar düzeyi) ve işlevsel (görev başarımı düzeyi) olarak destekler.
VMamba özelinde ek bir katman vardır: resmî depo CUDA'sız ortamda import dahi
edilemediğinden iki asgari yama gerekmiştir (Triton stub'ı ve koşullu CUDA
bağlamı; belgeleme §3.7 ve Bölüm 4.3). Yamalar yalnızca *hangi kod yolunun
seçileceğini* etkiler, hiçbir sayısal işlemi değiştirmez; mIoU'nun yeniden
üretilmesi bunun da doğrulamasıdır.

### EfficientViM'in kapsam dışı bırakılması

Dördüncü model adayı olarak "verimli-SSM" temsilcisi EfficientViM
[EfficientViM-2024] değerlendirilmiş ve ana matrise **dahil edilmemiştir**.
Gerekçe, §3.1'in değişken kontrolüne sadakattir: (i) modelin yayınlanmış tek
ADE20K checkpoint'i UPerNet değil Semantic FPN başlığı kullanır — başlığı
sabitleyen tasarımda bu, omurga etkisiyle başlık etkisini ayırt edilemez
kılar; (ii) ön-eğitim reçetesi (450 epoch) diğer üçünün 300-epoch sınıfıyla
hizalı değildir. Kirli bir hücre eklemektense matrisin temizliği tercih
edilmiştir. Hibrit ve verimli SSM tasarımları literatür düzeyinde §2.2.2'de
ele alınmakta; EfficientViM'in Pareto düzlemine kendi bildirdiği sayılarla,
ayrı işaretlenmiş bir "literatür noktası" olarak eklenmesi Bölüm 4.7'de
değerlendirilmektedir.

## 3.3 Veri Kümeleri ve Görevler

**ADE20K (birincil).** Tüm doğruluk doğrulamaları ve nicemleme-sonrası doğruluk
ölçümleri ADE20K [ADE20K-2017] üzerindedir: 150 sınıflı sahne ayrıştırma,
20 210 eğitim / 2 000 doğrulama görüntüsü; tezde yalnızca doğrulama bölümü
kullanılır (eğitim yapılmadığından eğitim bölümüne hiç dokunulmaz). ADE20K'nın
seçilmesinin nedeni pragmatiktir ve tasarımın geri kalanını belirlemiştir:
görü omurgası literatüründe segmentasyon kıyaslamasının fiilî standardı odur
ve üç aday ailenin de aynı başlıkla eğitilmiş resmî checkpoint'leri yalnızca
ADE20K için mevcuttur. Görev olarak semantik segmentasyonun seçilmesi ise
tezin sorusundan gelir: SSM'lerin teorik avantajı dizi uzunluğunda
doğrusallıktır ve bu avantajın anlamlı olduğu rejim, sınıflandırmanın 196
token'ı değil, yoğun tahminin binlerce-onbinlerce token'ıdır (512²'de
L = 16 384). Başka bir deyişle veri kümesi seçimi, tezin SSM'lere karşı
*hayırhah* olacak şekilde yapılmıştır: deney, SSM avantajının teoride en
güçlü olması gereken rejimde kurulur; avantaj orada da gerçekleşmiyorsa
bulgu güçlüdür, gerçekleşiyorsa tez hipotezi dürüstçe çürütülmüş olur.

Verimlilik ölçümlerinin kendisi (gecikme/bellek/enerji) veri içeriğinden
bağımsızdır ve §3.5 protokolü uyarınca sabit tohumlu sentetik girdilerle
(1×3×H×W, fp32) yürütülür; ADE20K görüntüleri yalnızca doğruluk
değerlendirmesinde kullanılır. Bu ayrım bilinçlidir: gecikme ölçümünü veri
yükleme ve ön-işleme boru hattından ayırmak, ölçülen büyüklüğü modelin ileri
geçişine indirger ve veri-G/Ç gürültüsünü matristen çıkarır.

**Cityscapes (planlanan, koşullu).** Çözünürlük ölçeklendirme sorusunun (AS1)
doğal uzantısı, gerçek yüksek-çözünürlüklü bir kümedir: Cityscapes
[Cityscapes-2016], 1024×2048 sürüş sahneleri. Ancak dürüst kapsam beyanı
şudur: Faz 0-2 bulguları, 1024² ve üzeri çözünürlüklerde SSM omurgasının bazı
yığınlara hiç taşınamadığını göstermiştir (Bölüm 4.3); Cityscapes deneyi bu
nedenle "çekirdek matris tamamlanır ve zaman bütçesi izin verirse" statüsünde
koşullu tutulmakta, çözünürlük ölçeklendirme ekseni ise ADE20K girdilerinin
sentetik olarak 256²-1024² serisinde ölçeklenmesiyle çekirdek matriste zaten
kapsanmaktadır. Bu ikame, doğruluk ekseni için değil yalnızca
verimlilik-ölçekleme ekseni için geçerlidir ve öyle raporlanır.

**ImageNet-1k alt kümesi (planlanan, kalibrasyon amaçlı).** Nicemleme
protokolünün ikinci aşaması (§3.6) aktivasyon istatistiği gerektirirse,
kalibrasyon verisi olarak omurgaların ön-eğitim dağılımından — ImageNet-1k
doğrulama bölümünden — küçük bir alt küme ayrılacaktır. İlk aşama
(kalibrasyonsuz ağırlık-nicemleme) veri gerektirmediğinden, bu küme henüz
kullanımda değildir; kapsam beyanının netliği için burada "planlanan" olarak
işaretlenir.

## 3.4 Dağıtım Yığınları

Tezin bağımsız değişkenlerinden en önemlisi dağıtım yığınıdır; çünkü tezin
iddiası tam olarak "verimlilik, mimarinin değil mimari-yığın eşleşmesinin
özelliğidir" biçimindedir. Matristeki yığınlar, bir pratisyenin Apple
Silicon'da bir görü modelini çalıştırmak için bugün önünde bulduğu gerçek
yolların tamamını kapsayacak şekilde seçilmiştir:

| Yığın | Yürütme hücreleri | Tezdeki rolü |
|---|---|---|
| PyTorch eager | CPU; MPS (Metal) | **Referans / üst taban çizgisi:** çerçevenin kendi yorumlayıcısı, graf dönüşümü yok — her modelin "dokunulmamış" davranışı |
| `torch.compile` | Inductor-CPU | Çerçeve-içi derleme: graf yakalamanın SSM'e maliyeti/kazancı |
| ONNX Runtime | CPU EP; CoreML EP | Çerçeveden bağımsız dağıtım standardı: export edilebilirlik + graf optimizasyonu + Apple hızlandırıcıya köprü |
| Core ML | CPU_ONLY; CPU+GPU; ALL (ANE dahil) | Apple'ın yerli dağıtım yolu ve tek ANE kapısı: uç dağıtımın "hedef" hücresi |

Bu küme yedi-artı ölçüm hücresi üretir ve kasıtlı bir **kademelenme** içerir:
soldan sağa gidildikçe esneklik azalır, graf derleyicisinin modele dayattığı
varsayımlar artar ve (klasik mimariler için) verimlilik artar. Tezin merkezi
sorusu (AS2), SSM omurgasının bu kademelerin neresinde ve hangi maliyet
katmanında (dönüşüm / yükleme / paket / çıkarım — §2.6) kademeden düştüğüdür.
CUDA yığını matriste yoktur; literatürün CUDA sayıları Bölüm 4-5'te
"bildirilen" değerler olarak alıntılanır ve tezin ölçtüğü "gerçekleşen"
değerlerle karşıtlık ekseni bilinçli olarak böyle kurulur (revizyon gerekçesi:
tek donanım kısıtı, Bölüm 5.4'te sınırlılık olarak ayrıca tartışılır).

Hücrelerin rolleri tek tek şöyle gerekçelendirilir:

**PyTorch eager (CPU, MPS)** referans seçilmiştir — "en hızlı" olduğundan
değil, **en az varsayım** içerdiğinden: eager, modeli graf'a dönüştürmeden
operatör operatör yürütür; dolayısıyla bir model eager'da çalışıp bir yığında
çalışmıyorsa, fark modelin kendisine değil yığının varsayımlarına
atfedilebilir. MPS hücresi ayrıca, "GPU var ama CUDA yok" koşulunun — Apple
Silicon'un tanımlayıcı koşulunun — eager içindeki karşılığıdır: SSM
literatürünün tüm GPU sayıları CUDA'dandır ve MPS hücresi, GPU hızlanmasının
ne kadarının donanıma, ne kadarının CUDA'ya özgü çekirdeklere ait olduğunu
ayrıştırır.

**`torch.compile` (Inductor)**, çerçeveyi terk etmeden graf yakalamanın
temsilcisidir: eager ile tam graf derleyicileri arasındaki ara basamak.
Tezdeki sorusu dardır — çerçeve-içi derleyici, seçici taramanın Python
düzeyindeki kontrol akışını yakalayabiliyor mu ve yakaladığında kazanç mı
kayıp mı üretiyor? Apple arka ucunda Inductor'un CPU yolu kullanılır; bu
hücrenin klasik modellerde dahi nötr/negatif çıkabileceği bilinerek matrise
alınmıştır, çünkü "derleme her zaman kazandırır" varsayımının kendisi test
edilen şeydir.

**ONNX Runtime**, çerçeveden bağımsız dağıtımın endüstri standardıdır ve
matriste iki rol üstlenir. **CPU EP** hücresi, export edilebilirlik ile graf
optimizasyonunun (sabit katlama, operatör füzyonu) birleşik etkisini ölçer;
SSM için kritik olan, bu hücrenin çıkarım *öncesi* katmanları — export
süresi, graf boyutu, oturum yükleme süresi — görünür kılmasıdır. **CoreML EP**
hücresi ise melez bir yoldur: ORT grafı bölümler ve desteklenen alt-grafları
Core ML'e devreder. Bu hücre, "ONNX üzerinden Apple hızlandırıcıya erişim"
vaadinin gerçekte kaç graf parçasına bölündüğünü ve bölümleme ek yükünün
kazancı yiyip yemediğini ölçer.

**Core ML**, Apple'ın yerli dağıtım biçimi ve **ANE'ye açılan tek programatik
kapıdır** — ANE'yi hedefleyebilen başka bir genel yol yoktur; tez ANE sorusunu
bu nedenle ancak Core ML üzerinden sorabilir. Üç hesaplama-birimi hücresi
kasıtlı bir ayrıştırma merdivenidir: `CPU_ONLY` derleyicinin graf
optimizasyonunu hızlandırıcıdan izole eder, `CPU_AND_GPU` Metal yolunu ekler,
`ALL` ANE'yi tercihe açar. `ALL` bir *istek*tir, garanti değil; hangi katmanın
gerçekte nerede çalıştığı §3.5.6'daki yürütme-yeri doğrulamasıyla kanıtlanır
ve `ALL`'un `CPU_AND_GPU`'dan yavaş çıkması gibi ters sıralamalar, sessiz
geri-düşüşün (fallback) imzası olarak ayrıca yorumlanır.

### Export-dostu eşdeğer dönüşümler metodolojisi

Faz 0'ın kritik metodolojik bulgusu, klasik omurgaların dahi ONNX/CoreML'e
"kutudan çıktığı gibi" aktarılamadığıdır: mmseg kalıplarındaki dinamik shape
okumaları (`size=x.shape[2:]`) ve UPerNet'in bölünmeyen-çıktılı
`adaptive_avg_pool2d` çağrıları (16→3, 16→6) hem ONNX TorchScript exporter'ını
hem CoreML dönüştürücüsünü hem de — bağımsız üçüncü bir platform olarak —
MPS eager'ı düşürmektedir. Bu engeller karşısında izlenen ilke şudur:

> **Modeli değiştirmeden grafı değiştir.** Ağırlıklara, katman yapısına veya
> hesaplanan fonksiyona dokunulmaz; yalnızca aynı fonksiyonun export-dostu bir
> ifadesi yazılır ve numerik eşdeğerlik kanıtlanır.

Uygulanan üç dönüşüm (ayrıntı ve ölçümler: Bölüm 4.3, EK B):

1. **Statik-dilimli PSP:** `adaptive_avg_pool2d`, PyTorch'un kendi pencere
   tanımıyla (start = ⌊iH/s⌋, end = ⌈(i+1)H/s⌉) birebir aynı sınırları
   kullanan sabit dilim + ortalama eşdeğeriyle değiştirildi.
2. **Shape-okumasız boyutlar:** Export sırasında girdi boyutu sabit olduğundan,
   `int(x.shape[i])` okumaları giriş çözünürlüğünden türetilen Python
   sabitleriyle değiştirildi; böylece trace'te `aten::Int` düğümü hiç oluşmaz.
3. **Swin pencere matematiği:** pad/pencere hesaplarındaki shape okumaları,
   batch=-1 ve Python-int boyutlarla yeniden yazıldı.

Her dönüşüm için **numerik eşdeğerlik doğrulaması** zorunludur: dönüştürülmüş
yol, orijinal yol ile aynı girdide karşılaştırılır ve azami mutlak sapmanın
fp32 yuvarlama gürültüsü mertebesinde (~2×10⁻⁵; Swin yeniden yazımında
düzenleme-öncesi ONNX'e karşı ~5×10⁻⁵) kaldığı gösterilir. Bu eşik, tek
`float32` işleminin göreli hassasiyetinin (~1.2×10⁻⁷) yüzlerce katman boyunca
birikimiyle tutarlıdır; sapma bu mertebeyi aşan hiçbir dönüşüm "eşdeğer"
sayılmaz. Doğrulama olmadan yapılan graf düzenlemesi, ölçülen nesnenin
kimliğini belirsizleştirir ve §3.2'deki doğrulama zincirini kırar.

Bu dönüşümlerin *eager* hücrelerine etkisi konusunda tutarlılık kuralı
uygulanır: eager ölçümleri modelin orijinal (dönüşümsüz) hâliyle alınır;
dönüşümler yalnızca export gerektiren yığınlarda etkindir. Tek istisna,
MPS'in `adaptive_avg_pool2d` desteksizliği nedeniyle MPS hücresinde statik
PSP'nin zorunlu kullanımıdır; numerik eşdeğerlik kanıtlı olduğundan bu,
hücreler arası karşılaştırılabilirliği bozmaz ve ham kayıtlarda etiketlenir.

VMamba özelinde export-dostu dönüşüm sorunu nitelik değiştirir: engel bizim
sarmalayıcı kodumuzda değil, üçüncü-parti tarama implementasyonunun kendisinde
yatar ve dört-yönlü seçici taramanın exporter tarafından tamamen açılması
(unroll) graf ölçeğini patolojik büyütür. Bu artık bir "cerrahi düzeltme" değil
tezin ölçüm *bulgusudur* ve Bölüm 4.3'te raporlanır; taramanın export-dostu
yeniden formülasyonu ise ayrı bir araştırma sorusu olarak Faz 4'ün (AS4)
konusudur. Yöntemsel sınır şöyle çizilir: §3.4'ün dönüşümleri yalnızca
*eşdeğerliği kanıtlanabilir, yerel* yeniden ifadeleri kapsar; hesaplama
düzenini değiştiren her müdahale AS4 kapsamına aittir.

## 3.5 Ölçüm Protokolü

Tüm verimlilik ölçümlerinin tabi olduğu protokol — dört maliyet katmanının
tanımı, ısınma ve termal stabilizasyon, senkronizasyon, gürültü izolasyonu,
`powermetrics` ile enerji ölçümü, istatistiksel raporlama, Xcode ile yürütme
yeri doğrulaması ve ResNet-50 kontrol modeliyle protokol doğrulaması — ayrı
bir bölüm olarak **§3.5'te** tanımlanmıştır (`tez/bolum-3.5-olcum-protokolu.md`).
Bu bölümdeki tüm ölçüm atıfları o protokole yapılır.

## 3.6 Nicemleme Protokolü

### Konumlandırma: algoritma reprodüksiyonu değil, transfer ölçümü

SSM'e özgü nicemleme literatürü (PTQ4VM, QMamba/Quamba hattı — §2.4.2)
yöntemlerini sınıflandırma görevinde ve CUDA implementasyonları üzerinde
kanıtlamıştır. Bu tez o yöntemleri **yeniden üretmez**; bunun iki nedeni
vardır. Birincisi pratiktir: söz konusu yöntemlerin yayınlanmış kodları özel
CUDA çekirdeklerine bağımlıdır ve bu tezin donanımında çalıştırılamaz.
İkincisi ve daha önemlisi konumsaldır: tezin sorusu (AS3) "en iyi SSM
nicemleme algoritması hangisi?" değil, "**bir pratisyenin bugün elindeki
endüstriyel PTQ araçları**, yoğun tahmine taşındığında ne verir?" sorusudur.
Bu nedenle nicemleme matrisi, dağıtım yığınlarının *kendi yerleşik* nicemleme
yollarıyla kurulur — tıpkı gecikme matrisinin özel çekirdeklerle değil
yerleşik yürütücülerle kurulması gibi. PTQ4VM/QMamba bulguları, Bölüm 4.4'te
sonuçların yorumlanmasında literatür karşılaştırma noktası olarak kullanılır:
onların sınıflandırmada raporladığı kayıp desenleri, bizim segmentasyonda
ölçtüğümüz desenlerle karşılaştırılır.

### Aşama I — kalibrasyonsuz ağırlık-nicemleme

İlk aşama, veri gerektirmeyen (kalibrasyonsuz) ağırlık-yalnız nicemlemeyi üç
yerleşik yolla uygular:

| Yol | Araç ve kip | Hedef modeller |
|---|---|---|
| Core ML W8 | `coremltools.optimize` doğrusal simetrik 8-bit ağırlık | ConvNeXt, Swin (VMamba Core ML'e dönüşemediğinden hariç — Bölüm 4.3) |
| Core ML W4 | `coremltools.optimize` 4-bit palettization (ağırlık kümeleme) | ConvNeXt, Swin |
| ORT INT8 | `quantize_dynamic`, ağırlık-yalnız QInt8 | ConvNeXt, Swin, VMamba |

Her nicemlenmiş model için raporlanan: üretim (nicemleme) süresi, paket boyutu,
§3.5 protokolüyle çıkarım gecikmesi ve fp32 referansa karşı **çıktı sapması**
(logit uzayında azami/ortalama mutlak fark ile piksel-başına argmax eşleşme
oranı). Çıktı sapması, tam mIoU değerlendirmesinden önce ucuz bir eleme
metriğidir: argmax eşleşmesi yüksekse mIoU etkisinin sınırlı kalacağı öngörülür,
düşükse tam değerlendirme önceliklendirilir. Nicemlemenin görev doğruluğuna
etkisi (ADE20K mIoU, 2 000 görüntü) ayrı bir ölçüm turudur ve Bölüm 4.4'te
raporlanır.

İki Core ML kipinin birlikte seçilmesi bit-genişliği taramasından fazlasını
amaçlar: W8 doğrusal nicemleme ile W4 palettization farklı *mekanizmalardır*
(tekdüze aralık nicemlemesi vs ağırlık kümeleme/arama tablosu) ve Apple
yığınında farklı yürütme yollarını tetikleyebilir; ikisinin doğruluk-boyut-
gecikme üçlüsündeki ayrışması, kazancın nereden geldiğine dair mekanizma
bilgisi taşır. ORT tarafında `quantize_dynamic`'in seçilmesi ise aynı
"yerleşik yol" ilkesinin sonucudur: statik (kalibrasyonlu) ORT nicemlemesi
Aşama II kapsamına bırakılır. VMamba'nın ORT-INT8 hücresi özel ilgiyle
izlenir: fp32'de gözlenen patolojik yükleme maliyetinin (Bölüm 4.3)
nicemlenmiş grafta nasıl davrandığı, dört-katmanlı maliyet çerçevesinin
nicemleme eksenindeki ilk testidir.

Ağırlık-yalnız başlangıcın gerekçesi üç katmanlıdır: (i) veri gerektirmediği
için nicemleme matrisinin tamamı, kalibrasyon kümesi seçiminden gelen ek bir
serbestlik derecesi olmadan kurulur — ilk turda değişken sayısını asgaride
tutmak §3.1'in kontrol ilkesinin devamıdır; (ii) uç dağıtımın ilk kazancı
olan paket boyutu küçülmesi tamamen ağırlıklardan gelir; (iii) SSM nicemleme
literatürünün işaret ettiği asıl kırılganlık *aktivasyon* tarafında olduğundan
(§2.4.2-2.4.3), ağırlık-yalnız sonuçlar "aktivasyonlara dokunmadan ne
kaybediliyor?" sorusunun temiz taban çizgisini oluşturur.

### Aşama II — aktivasyon istatistikleri ve çözünürlük bağımlılığı (planlanan)

İkinci aşama, AS3'ün asıl merak ettiği mekanizmaya iner: aktivasyon
nicemlemesi ve aykırı değer profili. Planlanan içerik: (i) omurga
aktivasyonlarının katman-başına dağılım istatistiklerinin (aykırı değer
oranı, kanal-bazlı dinamik aralık) 256²-1024² çözünürlük serisinde
çıkarılması — literatürün sınıflandırma (düşük L) rejiminde raporladığı
aykırı değer desenlerinin dizi uzunluğuyla nasıl değiştiği sorusu; (ii) bu
istatistiklere dayalı W8A8 nicemleme denemeleri (ImageNet-1k alt kümesiyle
kalibrasyon — §3.3). Bu aşama ayrı bir görev olarak planlanmıştır ve bu
taslağın yazıldığı tarihte henüz yürütülmemiştir; Bölüm 4.4'ün kapsamı
gerçekleşen ölçümlere göre kesinleşecektir. Aşamalandırmanın kendisi
metodolojik bir tercihtir: çalışan uçtan-uca bir ağırlık-nicemleme hattı
kurulmadan aktivasyon analizine girmek, hata ayıklama yüzeyini iki katmana
yayardı.

## 3.7 Tekrarlanabilirlik

Tezin tüm ölçüm altyapısı, üçüncü bir tarafın aynı donanım sınıfında sonuçları
yeniden üretebilmesi hedefiyle kurulmuştur. Protokol-düzeyi kurallar (ham
kayıtların içeriği, ±%5 yeniden üretim bandı) §3.5.8'de tanımlanır; burada
altyapının bütününe ait ilkeler belirtilir.

**Ham kayıt, git-commit'li ve eksiksiz.** Her ölçüm betiği sonuçlarını
`results/raw/` altına JSON-Lines biçiminde, satır başına bir olay olarak yazar;
her satır zaman damgası, ortam sürümleri (işletim sistemi, çip, Python ve tüm
çerçeve sürümleri) ve deponun git commit kimliğini gömülü taşır — ortam
bilgisini betiğin kendisinin kaydetmesi, "hangi sürümle alınmıştı?" sorusunu
insan hafızasından çıkarır. `results/raw/` git deposuna dahildir ve hiçbir
kayıt silinmez; grafikler ham kayıttan türetilir, hiçbir sonuç "yalnızca
grafik olarak" saklanmaz.

**Başarısızlık da veridir.** Başarısız denemeler — export hataları, dönüştürücü
çökmeleri, bellek duvarları, süreç ölümleri — başarılı ölçümlerle aynı kanala,
hata sınıfı ve mesajıyla birlikte kaydedilir. Bu tezde bu bir titizlik
ayrıntısı değil içerik gereğidir: araştırma sorularının cevabının bir kısmı
tam olarak *neyin çalışmadığıdır* (ör. "VMamba Core ML'e dönüşemiyor" Bölüm
4.3'ün bir hücresidir) ve kaydedilmeyen başarısızlık, yayın yanlılığının
ölçüm-altyapısı düzeyindeki karşılığı olurdu.

**Sürüm sabitleme ve ortam.** Ortam tek Python sürümüne (3.12) ve
`requirements.txt` ile sabitlenmiş paket sürümlerine (torch 2.13,
onnxruntime 1.28, coremltools 9.0 başta olmak üzere) kilitlidir; tam döküm
EK C'dedir. Rastgelelik içeren her adım (sentetik girdi üretimi, kalibrasyon
örneklemesi) sabit tohumla çalışır. Deterministik olamayan etkenler (macOS
zamanlayıcısı, termal durum) yok sayılmaz, ölçülür: §3.5'in termal ön-kontrol
ve çok-süreç tekrar mekanizmaları bunların etkisini nicelleştirir.

**Üçüncü-parti değişikliklerin izlenebilirliği.** Üçüncü-parti kodda (örn.
resmî VMamba deposu) zorunlu kalınan her değişiklik, kod içinde `[TEZ YAMASI]`
etiketiyle işaretlenir ve ayrı bir belgede gerekçesiyle listelenir; böylece
`grep -rn "TEZ YAMASI"` tek komutuyla tezin üçüncü-parti koda dokunduğu her
nokta dökülebilir. Yamalar asgari-müdahale ilkesine tabidir: yalnızca kod
yolunun seçimini düzeltir, sayısal davranışı değiştirmez ve doğrulaması
§3.2'nin mIoU yeniden-üretim zincirinden geçer. Ölçüm harness'ı, model
sarmalayıcıları ve tüm deney betikleri tezle birlikte açık kaynak olarak
yayımlanacaktır (EK B).

---

*Sayfa hedefi: ~14 (3.5 dahil). Atıflar `[YAZAR-YIL]` yer tutucu biçimindedir;
kaynakça Faz 5'te bağlanacaktır. §3.6 Aşama II ve Cityscapes hücreleri
"planlanan" statüsündedir; kapsam kesinleştiğinde bu taslak güncellenecektir.*


---

# 3.5 Ölçüm Protokolü *(TASLAK v1 — 12 Ağustos 2026)*

> Bu bölüm, tezdeki tüm verimlilik ölçümlerinin tabi olduğu protokolü tanımlar.
> Protokolün referans implementasyonu `src/benchmark/` paketidir ve ResNet-50 ile
> doğrulanmıştır (§3.5.7). Ham ölçüm kayıtları `results/raw/` altında, ortam
> bilgisiyle birlikte JSON-Lines biçiminde saklanır.

Verimlilik literatüründeki yaygın pratik, tek bir "gecikme" sayısı raporlamaktır.
Oysa Faz 0 ön bulgularımız (Bölüm 4.2), durum-uzayı modellerinde dağıtım maliyetinin
dört ayrı katmanda ortaya çıktığını göstermiştir: **(a)** dönüşüm/derleme süresi,
**(b)** model yükleme süresi, **(c)** dağıtım paketi boyutu ve **(d)** çıkarım
gecikmesi. Bu nedenle protokol, dört katmanı ayrı metrikler olarak tanımlar;
yalnızca (d)'yi raporlamak, örneğin 400K parametrelik bir SSM mikro-modelinin
CoreML dönüşümünün 94 dakika sürdüğü gerçeğini görünmez kılar.

## 3.5.1 Isınma ve Termal Stabilizasyon

Her ölçüm hücresi (model × yığın × girdi biçimi) şu sırayla yürütülür:

1. **Termal ön-kontrol:** macOS `pmset -g therm` çıktısı sorgulanır; termal veya
   performans uyarısı varsa sistem "nominal" durumuna dönene dek beklenir
   (10 sn aralıklı yoklama, üst sınır 120 sn). Ölçüm başlangıcındaki termal durum
   ham kayda yazılır.
2. **Isınma:** Ölçüme dahil edilmeyen ≥15 ileri geçiş. Bu, önbellek ısınması,
   JIT/derleyici geç-derlemesi (`torch.compile`, CoreML ilk-çağrı özelleştirmesi)
   ve güç yönetimi geçişlerinin ölçüm penceresine sızmasını engeller.
3. **Ölçüm:** ≥60 zamanlanmış geçiş (varsayılan 100). Tek geçiş = tek örnek;
   örnekler arası istatistik §3.5.5'e göre raporlanır.

Tüm ölçümler cihaz şebeke gücüne bağlıyken ve düşük güç modu kapalıyken alınır.

## 3.5.2 Senkronizasyon ve Zamanlama Doğruluğu

Zamanlayıcı `time.perf_counter()` (monotonik, ns çözünürlüklü) kullanır. Asenkron
yürütme kuyruklarında (Metal/MPS) süre, geçiş sonrası açık senkronizasyon
(`torch.mps.synchronize()`) dahil ölçülür; aksi hâlde ölçülen şey yalnızca kuyruğa
yazma süresidir. ONNX Runtime `run()` ve Core ML `predict()` çağrıları senkron
döner; ek bariyer gerekmez. Bu ayrım runner soyutlamasında (`runners.py`) her
yığın için ayrı `sync()` implementasyonuyla kodlanmıştır.

## 3.5.3 İş Parçacığı ve Sistem Gürültüsü İzolasyonu

Ölçüm sırasında etkileşimli kullanıcı işlemleri kapatılır; arka plan indirme,
dizinleme (Spotlight) ve benzeri işlemlerin etkin olmadığı doğrulanır. İş
parçacığı sayıları varsayılan bırakılır ve ortam kaydına yazılır — yapay
tek-iş-parçacığı kısıtlaması, gerçek dağıtım koşulunu temsil etmediği için
uygulanmaz. Aynı hücrenin üç bağımsız süreçte tekrarı (§3.5.7) sistem
gürültüsünün medyan üzerindeki etkisini sınar.

## 3.5.4 Enerji Ölçümü

Apple Silicon'da paket/ANE/GPU güç telemetrisi `powermetrics` aracıyla toplanır.
Araç ayrıcalıklı erişim gerektirdiğinden protokol iki kipte çalışır: telemetri
mevcutsa örnekleme ölçüm penceresiyle eşzamanlı yürütülür ve ortalama güç ile
geçiş-başına enerji (mJ) raporlanır; mevcut değilse ham kayda `energy: unavailable`
yazılır ve enerji sütunu boş bırakılır — hiçbir koşulda tahmini değer üretilmez.

## 3.5.5 İstatistiksel Raporlama

Her hücre için raporlanan: örnek sayısı, **medyan** (birincil metrik), ortalama,
standart sapma, minimum, P90, P99 ve maksimum. Medyanın birincil seçilmesinin
nedeni, işletim sistemi kaynaklı seyrek kesintilerin (arka plan işi, güç durumu
geçişi) uzun-kuyruklu dağılım oluşturmasıdır; kuyruk davranışı ayrıca P99 ile
görünür kılınır. Ham örnek listesi eksiksiz saklanır; böylece ileride farklı
istatistikler yeniden türetilebilir.

## 3.5.6 Yürütme Yeri Doğrulaması

Core ML derleyicisi, hesaplama birimini (CPU/GPU/ANE) katman bazında ve şeffaf
olmayan biçimde seçer; `compute_units=ALL` isteği bir *tercih*tir, garanti değil.
Bu nedenle "ANE'de çalışıyor" iddiası yalnızca Xcode Core ML Performance Report
ile katman-başına yürütme yeri dökümü alınarak ileri sürülür. Rapordan türetilen
**ANE yürütme oranı** (ANE'ye atanan katman yüzdesi) ana sonuç tablolarında ayrı
sütun olarak verilir. Dolaylı kanıt (ör. `CPU_AND_GPU` ile `ALL` arasındaki süre
farkı) yalnızca destekleyici gösterge olarak kullanılır.

## 3.5.7 Protokol Doğrulaması

Protokol, davranışı iyi bilinen bir kontrol modeliyle (ResNet-50, ImageNet
ağırlıkları, 224×224, yığın=1) doğrulanmıştır. Beklenen desenlerin tümü
gözlenmiştir: paralel-dostu evrişimli modelde MPS'in CPU'ya karşı ~2.9× hızlanması
(SSM mikro-modelindeki tersine davranışın karşıtı), Core ML hesaplama birimi
sıralaması CPU_ONLY > CPU+GPU > ALL ve iki bağımsız süreç tekrarında birincil
metrikte ≤%1.2 sapma (Core ML ALL: 1.194 → 1.208 ms). Kontrol modelinin dönüşüm
süresi (2.5 sn) ile SSM mikro-modelinin dönüşüm süresi (5 668 sn) arasındaki
~2 000× fark, ölçülen olgunun model boyutundan değil graf yapısından
kaynaklandığının ilk kanıtı olarak Bölüm 4'te ele alınır.

## 3.5.8 Tekrarlanabilirlik

Her ham kayıt şunları içerir: işletim sistemi ve çip kimliği, Python ve tüm
çerçeve sürümleri, git commit kısa-özeti, girdi biçimi/veri tipi, ölçüm
konfigürasyonu ve termal durum. Rastgelelik içeren adımlar sabit tohumla çalışır.
Deney betikleri tek komutla yeniden çalıştırılabilir ve aynı donanımda birincil
metriği ±%5 bandında yeniden üretmesi beklenir.

---
*Sayfa hedefi: ~4. Bu taslak deneyler ilerledikçe (özellikle enerji telemetrisi
ve Xcode doğrulaması devreye girince) somut sayılarla güncellenecek.*


---

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


---

# 4.2 – 4.3 Verimlilik Sonuçları

*(v2 — ölçümler tamamlandı, 13 Ağustos 2026)*

> Bu iki bölüm, Bölüm 4.1'de kimlikleri ve doğrulukları belgelenen üç omurganın
> (ConvNeXt-T [CNN], Swin-T [Transformer], VMamba-T [SSM]) verimlilik ölçümlerini
> raporlar. Tüm ölçümler Bölüm 3.5'teki protokole tabidir; ham kayıtlar
> `results/raw/` altındadır: `export_matrix.jsonl`, `energy_matrix.json`,
> `latency_matrix_{convnext,swin,vmamba}.jsonl` ve `ort_profile_*_top.json`.
> Deney matrisi tamamlanmıştır; v1 taslağındaki **[ölçüm sürüyor]** işaretlerinin
> tamamı bu sürümde final ölçümlerle doldurulmuştur.

---

## 4.2 AS1: Referans Yığında Verimlilik Profilleri

Birinci araştırma sorusu (AS1), eşit doğruluk bütçesi çerçevesinde üç omurga
ailesinin yüksek çözünürlüklü semantik segmentasyondaki gerçek gecikme, bellek ve
enerji profilini sorar. Bu bölümdeki "referans yığın", PyTorch eager yürütmesidir
(CPU ve MPS arka uçları): hiçbir dışa aktarma (export), derleme veya graf
optimizasyonu içermeyen, araştırma kodunun doğrudan çalıştırıldığı kip. Bu kip iki
nedenle referanstır: (i) literatürdeki doğruluk sonuçlarının üretildiği ortamdır,
(ii) dağıtım yığınlarının getirdiği ek maliyetlerin (Bölüm 4.3) karşılaştırma
tabanını oluşturur.

Önemli bir ortam notu: bu tezin donanımında (Apple M5) NVIDIA GPU'su ve dolayısıyla
Mamba ailesinin el yazması CUDA `selective scan` çekirdeği yoktur. VMamba-T,
saf-PyTorch tarama yoluyla (torch fallback) çalıştırılmıştır; resmî deponun bu
fallback'e CPU-only ortamda ancak iki kaynak yamasıyla ulaşabildiği Bölüm 4.3'ün
sonunda ele alınmaktadır. Dolayısıyla buradaki eager sayıları, "özel çekirdek
yokluğunda SSM'in ödediği bedelin" doğrudan ölçümüdür — AS2'nin motivasyonunu
kuran tablodur.

Tekrarlanabilirlik notu: bu bölümlerdeki her sayının ham kaydı, ortam bilgisi
(işletim sistemi ve çip kimliği, çerçeve sürümleri, git commit özeti, termal
durum) ile birlikte `results/raw/` altında saklanmaktadır (§3.5.8). Rapor edilen
koşullar aksi belirtilmedikçe şöyledir: 512×512 girdi, yığın boyutu 1, fp32,
torch 2.13 / onnxruntime 1.28 / coremltools 9.0, Apple M5 (24 GB), şebeke gücü.

### 4.2.1 Eager Gecikme Profilleri (512×512)

Tablo 4.2, 512×512 girdi, yığın boyutu 1, fp32 koşullarında ölçülen medyan
gecikmeleri verir. Protokol hatırlatması: her hücre termal ön-kontrol ve ≥15
geçişlik ısınmanın ardından ≥60 zamanlanmış geçişle ölçülmüş (§3.5.1), MPS
hücrelerinde süreye açık senkronizasyon dahil edilmiş (`torch.mps.synchronize()`,
§3.5.2 — aksi hâlde ölçülen şey yalnızca kuyruğa yazma süresi olurdu) ve birincil
metrik olarak medyan raporlanmıştır (§3.5.5).

**Tablo 4.2 — Eager PyTorch gecikmeleri, 512², yığın 1, fp32, medyan (ms)**

| Yığın | ConvNeXt-T | Swin-T | VMamba-T | VMamba/ConvNeXt |
|---|---|---|---|---|
| torch CPU | 571 | 535 | **2 032** | 3.6× |
| torch MPS | 152 | 172 | **1 008** | 6.6× |
| MPS hızlanması (CPU/MPS) | 3.8× | 3.1× | **2.0×** | — |

Tabloda üç şey görüyoruz. Birincisi, klasik omurgalar birbirine yakınken (CPU'da
535–571 ms, MPS'te 152–172 ms) VMamba-T her iki arka uçta da belirgin biçimde
ayrışıyor: CPU'da 3.6–3.8×, MPS'te 5.9–6.6× daha yavaş. İkincisi, bu fark Bölüm
4.1'deki değerlendirme-koşusu yan bulgusunun (görüntü başına VMamba ~2.7–3.6 s'ye
karşı klasiklerde ~1.4–1.6 s) kontrollü harness altında doğrulanmış hâlidir.
Üçüncüsü, MPS her üç modeli de hızlandırıyor, ancak hızlandırma oranı mimariye
göre eşit değil: klasiklerde 3.1–3.8×, VMamba'da yalnızca 2.0×. Paralel donanım,
modelin paralel-dostu kısımlarını hızlandırırken ardışık tarama payını aynı oranda
hızlandıramıyor; bu asimetri bir sonraki alt bölümün konusudur.

Klasiklerin kendi içindeki sıralama da not edilmeye değer: CPU'da Swin-T,
ConvNeXt-T'den bir miktar hızlıyken (535'e karşı 571 ms) MPS'te sıralama tersine
dönmektedir (172'ye karşı 152 ms). Fark her iki yönde de küçüktür (%7–13 bandı)
ve yorumlanmamaktadır; çözünürlük taraması (§4.2.4, Tablo 4.6) bu sıralamanın
kalıcı olmadığını göstermiştir — klasik ikilinin öncelik sırası arka uca ve
çözünürlüğe göre değişmektedir (ör. MPS'te 1024²'de Swin öne geçerken 256–768²
aralığında ConvNeXt öndedir). Analizin ana ekseni,
klasik ikilinin kendi içindeki küçük farklar değil, ikili ile VMamba arasındaki
kat düzeyindeki ayrışmadır.

### 4.2.2 MPS'in Ardışık Taramadaki Davranışı: Mikrobenchmark ile Tam Model Arasındaki Nüans

Faz 0 öncül doğrulamasında (TASK-008), `selective scan`'in ardışık yapısını izole
etmek için tasarlanan MiniMamba mikro-modelinde (saf-PyTorch selective scan,
4 katman, d=96, n=16, ~400K parametre) MPS **her iki dizi uzunluğunda da CPU'dan
yavaş** ölçülmüştü:

**Tablo 4.3 — MiniMamba mikrobenchmark: eager gecikme ve ölçekleme (TASK-008)**

| Metrik | L=196 | L=1024 | Oran (L 5.2×) |
|---|---|---|---|
| torch CPU (eager) | 17.5 ms | 59.3 ms | 3.4× |
| torch MPS | 24.0 ms | 91.8 ms | 3.8× |

Tam modelde ise tablo tersine dönüyor: Tablo 4.2'de VMamba-T MPS'te CPU'dan 2.0×
hızlı. Bu iki gözlem çelişki değildir; aynı mekanizmanın iki farklı karışım
oranındaki tezahürüdür. Mikrobenchmark neredeyse yalnızca tarama operatöründen
oluşur; taramanın adım-adım bağımlılık zinciri GPU'nun paralel yürütme modeline
uymadığından, çekirdek başlatma ve senkronizasyon yükü hesaplama kazancını aşar ve
MPS CPU'nun gerisine düşer. Tam VMamba-T + UPerNet boru hattında ise tarama,
büyük ve paralel-dostu bir bütünün (evrişimler, doğrusal katmanlar, 31.5M
parametrelik segmentasyon başlığı) içine gömülüdür; paralel kısımlar MPS'ten
yararlanır ve toplam süre düşer. Ancak tarama payı hızlanmaya direndiği için
VMamba'nın MPS kazancı (2.0×) klasiklerin kazancının (3.1–3.8×) belirgin altında
kalır. Kontrol modeliyle yapılan protokol doğrulaması da bu okumayı destekler:
paralel-dostu ResNet-50'de MPS ~2.9× hızlanma göstermiştir (§3.5.7) — SSM
mikro-modelindeki tersine davranışın karşıtı. Özetle: **MPS ardışık taramayı
cezalandırıyor; tam modelde bu ceza toplam süreye seyreltilmiş olarak yansıyor.**
Bu, Mamba'nın el yazması CUDA çekirdeğinin varlık sebebinin Apple tarafındaki
izdüşümüdür.

### 4.2.3 Enerji Profilleri

Enerji, `powermetrics` telemetrisiyle (200 ms örnekleme, §3.5.4) ölçüm
pencereleriyle eşzamanlı toplanmış; boşta taban çizgisi (CPU 3 287 mW, GPU
8 542 mW, ANE 0 mW; 70 örnek) düşülerek geçiş başına net enerji (mJ/çıkarım)
hesaplanmıştır. Ham kayıt: `results/raw/energy_matrix.json`.

**Tablo 4.4 — Enerji profilleri, 512², boşta-düşülmüş net mJ/çıkarım**

| Yığın | ConvNeXt-T | Swin-T | VMamba-T |
|---|---|---|---|
| CoreML ALL | **458** | 607 | ❌ (dönüşemiyor, §4.3) |
| CoreML CPU+GPU | 494 | **329** | ❌ |
| CoreML CPU_ONLY | 2 684 | 3 127 | ❌ |
| torch MPS | 937 | 1 183 | **4 844** |
| torch CPU | 4 268 | 4 166 | **11 720** |

*Not: Enerji hücrelerindeki koşular, telemetriyle eşzamanlı yürütülen kısa
serilerdir (hücre başına 5–20 geçiş); geçiş süreleri bu nedenle Tablo 4.8'deki
birincil gecikme medyanlarından bir miktar sapar. Gecikme için bağlayıcı değerler
Tablo 4.8'dir; bu tablo enerji içindir.*

Tabloda üç desen görüyoruz. Birincisi, eager kipte bile mimari farkı enerjiye
doğrudan yansıyor: VMamba-T'nin çıkarım başına enerjisi CPU'da 11 720 mJ ile
klasiklerin (~4 200 mJ) yaklaşık 2.8 katı, MPS'te 4 844 mJ ile 4–5 katıdır.
İkincisi, dağıtılabilir en verimli hücre ile karşılaştırıldığında uçurum
büyüyor: ConvNeXt-T CoreML ALL hücresi 458 mJ/çıkarım tüketirken VMamba-T'nin
bugün erişebildiği en verimli hücre (MPS eager, 4 844 mJ) bunun ~10.6 katı, CPU
hücresi ise **~25 katıdır**. Üçüncüsü, klasikler arasında CoreML hücrelerinde
görülen ALL/CPU+GPU sıralama farkı (ConvNeXt'te ALL daha verimli, Swin'de tersine
CPU+GPU daha verimli) rastgele değildir; §4.3.4'te gösterileceği gibi ANE
yürütmesinin enerji imzasıdır.

Enerjinin iki bileşenine — güç ve süre — ayrıştırılması, farkın kaynağını da
gösterir. Boşta-düşülmüş net ortalama güç, ölçülen hücrelerin çoğunda görece dar
bir bantta kalmaktadır (yaklaşık 4.3–6.7 W; ör. VMamba torch CPU 4 795 mW,
ConvNeXt torch CPU 5 825 mW, Swin CoreML ALL 6 731 mW). Yani VMamba, klasiklerden
belirgin biçimde daha *güçlü* çekmemektedir; enerji farkının baskın kaynağı
**süredir** — aynı işi kat kat daha uzun sürede yapmak, benzer güçte kat kat
daha çok enerji demektir. Bu bandın dikkat çekici istisnası ConvNeXt CoreML ALL
hücresidir: net güç 4 797 mW ile bandın alt ucunda *ve* süre kısadır; iş yükünün
bir bölümünün genel amaçlı çekirdekler yerine ANE'ye taşınması (ANE rayında
3 879 mW aktif çekim, §4.3.4) her iki bileşeni birden iyileştirmektedir.

### 4.2.4 Bellek Tepe Noktası ve Çözünürlük Ölçeklendirme

**Bellek.** Kontrollü, tüm-yığınlar bir çıkarım-anı tepe RSS turu bu ölçüm
kampanyasında yürütülmemiştir; bu paragraf, eldeki iki dolaylı göstergeyi kendi
sınırlarıyla raporlar. Birinci gösterge, MPS hücrelerinde harness'ın her koşuda
kaydettiği sürücü tahsis tepe noktasıdır (`torch.mps` ayırıcı tepe değeri —
süreç RSS'i değil, yalnızca MPS tensör tahsisleri):

**Tablo 4.5 — MPS tahsis tepe noktası (GB), çözünürlüğe göre**

| Çözünürlük | ConvNeXt-T | Swin-T | VMamba-T | VMamba/ConvNeXt |
|---|---|---|---|---|
| 512² | 1.23 | 1.23 | **2.43** | 2.0× |
| 768² | 2.31 | 2.33 | **4.69** | 2.0× |
| 1024² | 3.92 | 4.03 | **6.11** | 1.6× |

VMamba'nın çıkarım-anı tahsis tepesi her çözünürlükte klasiklerin ~1.5–2
katıdır: tarama, girdiyle birlikte büyüyen ara durum tensörleri taşımaktadır.
İkinci gösterge, dışa aktarma/yükleme aşamasının süreç RSS'idir ve §4.3.2'de
raporlanmaktadır (VMamba ONNX dışa aktarmada 512²'de tepe RSS 6.65 GB; 1024²'de
~65 GB'a şişerek başarısızlık — bellek duvarı, Tablo 4.11). İki gösterge aynı
şeyi ölçmez ve doğrudan karşılaştırılmamalıdır; ortak mesajları, bellek
maliyetinin de gecikme gibi çıkarım ve araç-zinciri katmanlarına asimetrik
dağıldığıdır.

**Çözünürlük ölçeklendirme.** SSM'lerin teorik cazibesi tam da burada yatar:
dikkat mekanizmasının O(L²) karmaşıklığına karşılık taramanın O(L) ölçeklenmesi,
çözünürlük büyüdükçe SSM lehine açılan bir makas vaat eder. Bu vaadin eager
kipte gerçekleşip gerçekleşmediğini sınamak için 256², 512², 768² ve 1024²
çözünürlüklerinde gecikme taraması tamamlanmıştır (ham kayıt:
`latency_matrix_{convnext,swin,vmamba}.jsonl`, `resolution` etiketiyle):

**Tablo 4.6 — Eager gecikme × çözünürlük, yığın 1, fp32, medyan (ms)**

| Arka uç | Çözünürlük | ConvNeXt-T | Swin-T | VMamba-T | VMamba/ConvNeXt |
|---|---|---|---|---|---|
| MPS | 256² | 38 | 48 | **350** | 9.3× |
| MPS | 512² | 152 | 172 | **1 008** | 6.6× |
| MPS | 768² | 356 | 401 | **2 593** | 7.3× |
| MPS | 1024² | 958 | 761 | **6 041** | 6.3× |
| CPU | 256² | 278 | 220 | **748** | 2.7× |
| CPU | 512² | 571 | 535 | **2 032** | 3.6× |
| CPU | 768² | 3 805ᵃ | 1 232 | **3 980**ᵇ | ~1.0×ᵃ |
| CPU | 1024² | 5 172 | 8 400ᵃ | **13 889**ᶜ | 2.7× |

ᵃ *Yüksek koşu-içi varyans (ConvNeXt 768² CPU: std ±1.8 s, örnekler 2.1–6.3 s
bandında; Swin 1024² CPU: std ±1.8 s): ≥768² CPU hücreleri bellek/termal baskıya
açıktır ve bu satırlardaki oranlar ihtiyatla okunmalıdır.*
ᵇ *İlk kayıt (13.3 s) kirli çıkmış; temiz yeniden-doğrulama koşusu 3 980 ms
vermiştir (512→768 ölçekleme ~lineer). Ham dosyada iki kayıt da korunmaktadır
(`reverify: true` etiketli kayıt geçerlidir).*
ᶜ *VMamba CPU 1024² değeri (13.9 s) bellek baskısı içerebilir (aynı çözünürlükte
MPS tahsis tepesi 6.1 GB, Tablo 4.5); üst sınır olarak okunmalıdır.*

Şekil 4.2'nin sorusunun cevabı bu tablodadır: **makas kapanmamaktadır.**
Güvenilir sinyalin bulunduğu MPS panelinde VMamba/ConvNeXt oranı 256²'den
1024²'ye 9.3× → 6.6× → 7.3× → 6.3× seyretmektedir — monoton bir kapanma yok,
~6–9× bandında salınım var; VMamba/Swin oranı 1024²'de tersine büyümektedir
(7.9×). Piksel sayısı 16× artarken VMamba'nın MPS süresi 17.3× artmıştır (350 →
6 041 ms): taramanın kendi ölçeklenmesi teoriyle uyumlu biçimde ~lineerdir,
ancak bu bir avantaj üretmemektedir, çünkü karşılaştırma tabanı da lineer
ölçeklenmektedir — Swin'in pencereli dikkati zaten O(L)'dir (aynı aralıkta
15.7×) ve ConvNeXt zaten evrişimseldir (25.4×). Teorik O(L)–O(L²) makası ancak
*global* dikkate karşı geçerlidir; buradaki pratik rakiplerin hiçbiri global
dikkat kullanmadığından, eager kipte SSM'e kalan tek fark sabit katsayıdır ve
özel çekirdek yokluğunda bu katsayı 6–9× aleyhtedir.

> **Şekil 4.2 — Çözünürlük-gecikme ölçeklendirme eğrileri** *(veri tamamlandı —
> Tablo 4.6; şekil bu tablodan üretilecektir)*. Üç omurganın 256²–1024²
> aralığındaki eager gecikme eğrileri (CPU ve MPS ayrı panellerde, log-log
> eksende). Sorunun cevabı: VMamba'nın O(L) eğimi gerçekleşmekte ancak makası
> kapatmamaktadır — eğriler yaklaşık paralel seyretmekte, aradaki dikey uzaklık
> (sabit katsayı) hiçbir çözünürlükte telafi edilmemektedir.

Mikrobenchmark bu çerçeve için bir ön ipucu vermişti (Tablo 4.3): L 5.2×
artarken MiniMamba'nın eager CPU süresi 3.4×, MPS süresi 3.8× artmıştı — izole
taramanın kendi ölçeklenmesi lineer-altı ve teoriyle uyumlu. Tam model taraması
bu ipucunu doğrulamıştır: buharlaşma çıkarım eğiminde değildir (eğim hayatta,
makas kapalı); kayıp, sabit katsayıda ve Bölüm 4.3'ün konusu olan
çıkarım-öncesi katmanlardadır.

### 4.2.5 Doğruluk-Verimlilik Düzlemi

Bölüm 4.1'de kurulan çerçeve gereği üç omurga aynı doğrulukta değildir
(mIoU 44.3–48.3 bandı; VMamba-T en doğru modeldir). Bu nedenle nihai sunum tekil
"hız" sıralaması değil, doğruluk-gecikme Pareto düzlemidir: her (omurga × yığın)
hücresi düzlemde bir noktadır.

> **Şekil 4.1 — Doğruluk-gecikme Pareto düzlemi, 512²** *(yer tutucu; matris
> hücreleri tamamlandı — Tablo 4.8; şekil bu verilerden üretilecektir)*.
> Yatay eksen medyan gecikme (ms, log),
> dikey eksen ADE20K val mIoU. Her omurga için eager CPU/MPS, ORT CPU ve CoreML
> hücreleri ayrı işaretçilerle; Pareto sınırı çizili. Beklenen okuma: eager
> düzlemde VMamba doğruluk avantajıyla sınırda yer bulurken, dağıtım yığınları
> eklendikçe VMamba noktalarının sınırdan uzaklaşması.

Eager veriler şimdiden şu nitel resmi veriyor: VMamba-T +2.9–4.0 mIoU puanlık
doğruluk avantajına karşılık MPS'te 5.9–6.6× gecikme ve ~4–5× enerji bedeli
ödemektedir. Bu bir "değmez" hükmü değildir — bazı uygulamalar için +4 mIoU bu
bedele değebilir; hüküm Bölüm 5'in işidir. Buradaki nesnel bulgu, bedelin
varlığı ve büyüklüğüdür.

---

## 4.3 AS2: Dağıtım Yığını Matrisi — Avantajın Buharlaşması

İkinci araştırma sorusu (AS2), teorik verimlilik ile ölçülen verimlilik
arasındaki farkın dağıtım yığınına göre nasıl değiştiğini ve avantajın *nerede*
buharlaştığını sorar. Bu bölümün ana bulgusu şudur: **buharlaşma çıkarım
gecikmesinde değil, araç zincirinin çıkarım-öncesi katmanlarında
gerçekleşmektedir** — dönüşüm süresi, graf boyutu ve yükleme süresi. Dahası,
final ölçümler çıkarım katmanının ORT yolunda SSM *lehine* olduğunu
göstermektedir (§4.3.1'deki ORT paradoksu); imkânsızlaştıran, diğer
katmanlardır. FLOPs tabanlı hiçbir analiz bu katmanları göremez, çünkü FLOPs
yalnızca çıkarımın aritmetik iş yükünü sayar.

Deney matrisinin nihai durumu:

**Tablo 4.7 — Dağıtım matrisi durumu (512², fp32; torch 2.13 / onnxruntime 1.28 / coremltools 9.0)**

| Omurga | PyTorch eager | ONNX / ORT CPU | CoreML |
|---|---|---|---|
| ConvNeXt-T | ✅ | ✅ | ✅ |
| Swin-T | ✅ | ✅ | ✅ |
| VMamba-T | ✅ (saf-torch scan) | ✅ *(yükleme 12 dk)* | **❌ dönüşemiyor** |

Tablo tek başına ilk sonucu veriyor: klasik omurgalar üç yığının üçünde de
çalışır durumdayken, VMamba-T CoreML'e — yani Apple Silicon'da ANE'ye giden tek
resmî yola — **hiç girememektedir**. "Dönüşemiyor" bu tezin bugünkü resmî
durumudur; taramanın export-dostu yeniden formülasyonu (AS4, Faz 4) bu
başarısızlığı temel çizgi olarak alacaktır. ONNX sütunundaki ✅ ise dipnotuyla
birlikte okunmalıdır: hücre "çalışıyor" — final ölçümde çıkarım medyanı eager'dan
bile hızlı çalışıyor (§4.3.1) — ancak §4.3.2'de gösterileceği gibi pratikte
dağıtılabilir olmaktan uzak bir maliyet profiliyle: bedel, çıkarımdan yükleme
katmanına taşınmıştır.

### 4.3.1 Yığın-Başına Gecikme: Dağıtılabilir-En-İyi Uçurumu

**Tablo 4.8 — Gecikme matrisi, 512², yığın 1, fp32, medyan (ms) — TASK-021 (tüm turlar)**

| Yığın | ConvNeXt-T | Swin-T | VMamba-T |
|---|---|---|---|
| CoreML CPU+GPU | **63.9** | **63.9** | ❌ |
| CoreML ALL | 91.4 | 86.0 | ❌ |
| CoreML CPU_ONLY | 311 | 315 | ❌ |
| torch MPS (statik PSP) | 152 | 172 | 1 008 |
| ORT CoreML EP | 318 | 331 | —² |
| torch CPU | 571 | 535 | 2 032 |
| ORT CPU | 644 | 714 | **618**¹ |
| torch.compile (inductor-CPU) | 1 670 | 531 | **✗ süreç çöküyor**³ |

¹ *§3.5 protokolüne uygun medyan (15 zamanlanmış geçiş). Ancak oturum yüklemesi
bu koşuda 620.7 s, dışa aktarma turunda 724.9 s sürmüştür; değer tek başına
değil, §4.3.2'deki yükleme maliyetiyle birlikte okunmalıdır.*
² *Denenmedi: her ORT oturum açılışı ~10–12 dk sürerken EP bölümleme denemesinin
ek maliyeti pratik bulunmamıştır.*
³ *İstisna dahi üretmeden süreç çökmesi (exit 1) — derleme katmanının VMamba'daki
üçüncü kırılma noktası (§4.3.2'deki ONNX unroll ve CoreML TypeError'dan sonra).*

Tablonun kritik okuması sütunlar arası değil, satırlar arasıdır. Her omurga için
"bugün gerçekten dağıtılabilir en iyi hücre"yi işaretleyelim: ConvNeXt-T ve
Swin-T için bu, CoreML CPU+GPU hücresidir — **63.9 ms**. VMamba-T sütunu ilk
bakışta bir sürpriz içerir: en düşük medyan artık ORT CPU hücresindedir
(618 ms). Ancak bu hücreye giriş bileti, her süreç başlatımında ödenen
~10–12 dakikalık oturum yüklemesidir (§4.3.2); uç dağıtım anlamında pratik
hücre MPS eager kalır — **1 008 ms**. Aradaki oran **~16×**'dir. Aynı iki model
eager CPU'da yalnızca
3.6–3.8× ayrışıyordu (Tablo 4.2): uçurumun 3.6×'ten 16×'e açılmasının kaynağı
VMamba'nın yavaşlaması değil, **klasiklerin erişebildiği hızlandırma yollarına
VMamba'nın erişememesidir**. Avantaj tam olarak burada buharlaşıyor: yığın
merdiveninin her basamağı (eager → ORT → CoreML → ANE) klasikler için bir
hızlanma adımıyken, VMamba merdivenin ilk basamağında kalmaktadır.

İkinci gözlem: klasiklerde CoreML ALL hücresi (91.4 / 86.0 ms) CPU+GPU
hücresinden (63.9 ms) *yavaştır*. "ALL" isteğinin ANE'yi de içeren bir tercih
olduğu, ancak garanti olmadığı hatırlanırsa (§3.5.6) bu sıralama şaşırtıcıdır ve
§4.3.4'te enerji imzasıyla birlikte açıklanmaktadır.

Üçüncü gözlem tablonun en öğretici satırındadır ve bu tezde **ORT paradoksu**
olarak adlandırılacaktır. Klasiklerde ORT CPU (644 / 714 ms) eager torch CPU'dan
(571 / 535 ms) hızlı değildir — ORT onlar için bir hızlandırma basamağı değil,
bir *taşınabilirlik* basamağıdır. VMamba'da ise ORT CPU 618 ms ile eager'ının
(2 032 ms) yalnızca **0.30×'udur** — dahası klasiklerin ORT hücrelerinden bile
hızlıdır. Naif beklenti tam tersiydi: 390 758 düğümlük patolojik grafın
yorumlanması matrisin en yavaş ORT hücresini vermeliydi. Mekanizma §4.3.3'te
operatör profiliyle gösterilmektedir: ORT'nin graf optimizer'ı, unroll edilmiş
taramanın düğüm kalabalığını **oturum yüklemesi sırasında** eritmekte —
12 dakikalık yüklemenin nedeni budur — ve geriye evrişim-ağırlıklı, hızlı bir
yürütme grafı kalmaktadır (mikrobenchmark'taki işaret, MiniMamba'da ORT'nin
eager'dan hızlı oluşu, tam modelde de korunmuştur). Yani maliyet buharlaşmamış,
**katman değiştirmiştir**: (d) çıkarımdan (b) yüklemeye. Bu, "maliyet yanlış
katmanda ödeniyor" tezinin en keskin örneğidir: yalnızca çıkarım medyanı
raporlayan bir benchmark bu hücreyi "VMamba ORT'de gayet hızlı" diye özetlerdi;
uç cihaz gerçekliğinde — uygulama açılışı, bellek baskısıyla model
boşaltma/yeniden yükleme — her başlatımda 10–12 dakika ödeyen bir modelin
çıkarım medyanının pratik değeri yoktur.

Dördüncü gözlem `torch.compile` satırıdır: Apple CPU'da inductor arka ucu ya
nötrdür (Swin 531 ≈ eager 535 ms) ya zararlıdır (ConvNeXt 1 670 ms — eager'dan
**2.9× yavaş**); VMamba'da ise derleme süreci istisna dahi üretmeden çökmektedir
(exit 1). PyTorch'un kendi derleme yolu bile SSM taramasını taşıyamamaktadır;
"derleme katmanı SSM için kırılma noktasıdır" bulgusu böylece üçüncü bağımsız
araç zincirinde (ONNX exporter, CoreML converter, TorchInductor) tekrarlanmıştır.

Beşinci gözlem ORT CoreML EP satırıdır (318 / 331 ms): CoreML'e ORT üzerinden
dolaylı erişim, saf CoreML hücresinin (63.9 ms) yaklaşık **5× yavaşındadır**.
Neden, bölümleme parçalanmasıdır: EP, grafı CoreML'in kabul ettiği alt-graflara
bölmekte — ConvNeXt'te 47 parça (427/498 düğüm CoreML'de), Swin'de **94 parça**
(600/821 düğüm) — ve her parça sınırında CPU↔CoreML veri aktarımı ödenmektedir.
Hızlandırıcıya *kısmî* erişimin, parçalanma yeterince yüksekse kazancı geri
yiyebildiğinin ölçülmüş örneğidir.

### 4.3.2 Dört Katmanlı Maliyet Modeli: FLOPs'un Kör Olduğu Katmanlar

Bölüm 3.5'te tanımlanan dört maliyet katmanı — **(a)** dönüşüm/derleme süresi,
**(b)** yükleme süresi, **(c)** dağıtım paketi boyutu, **(d)** çıkarım gecikmesi —
ONNX yolunda üç omurga için eksiksiz ölçülmüştür:

**Tablo 4.9 — ONNX yolu, dört katman (512², fp32; opset 17, TorchScript exporter)**

| Katman | ConvNeXt-T | Swin-T | **VMamba-T** | SSM/CNN oranı |
|---|---|---|---|---|
| (a) Export süresi | 1.2 s | 2.2 s | **537.8 s (9 dk)** | **448×** |
| (c) Graf boyutu | 237.4 MB | 238.8 MB | **858.4 MB** | 3.6× |
| (c′) Graf düğüm sayısı | 843 | 8 667 | **390 758** | **463×** |
| (b) ORT yükleme | 0.1 s | 0.2 s | **724.9 s (12 dk)** | **~7 249×** |
| (d) ORT ilk koşu | 0.89 s | 0.91 s | 1.2 s | **1.3×** |
| Export tepe RSS | 4.91 GB | 3.71 GB | **6.65 GB** | — |

Tablo, tezin ana bulgusunun ilk tam nicel kanıtıdır ve asimetrisi çarpıcıdır:
katman (d) — literatürün raporladığı tek katman — neredeyse hayatta kalmıştır
(ilk koşu 1.2 s, klasiklerin yalnızca 1.3 katı). Final medyan ölçümü bu gözlemi
keskinleştirmiştir: ORT çıkarım medyanı 618 ms ile eager'ın 0.30×'udur (Tablo
4.8) — katman (d) hayatta kalmakla kalmamış, kazanca dönüşmüştür. Buna karşılık
katman (a) 448×, katman (b) ~7 249× şişmiştir. Bir cümleyle: **çıkarım hızı
kazanıyor; araç zinciri çöküyor.** Üç modelin ağırlıkları karşılaştırılabilir
boyuttayken (~244 MB
düzeyi) VMamba grafının 858.4 MB'ının **614 MB'ı saf graf yapısıdır** — ağırlık
değil, serileştirilmiş operatör düğümleri. Katman (c) bu yüzden yalnızca bir
depolama sorunu değildir; katman (b)'nin nedenidir: ORT, 390 758 düğümlük grafı
her oturum açılışında ayrıştırıp optimize etmek zorundadır ve bunun bedeli her
model yüklemede yeniden ödenir. Uç cihaz senaryosunda — uygulama başlatma,
bellek baskısıyla model boşaltma/yeniden yükleme — 12 dakikalık yükleme süresi,
1.2 saniyelik çıkarım hızını pratikte anlamsızlaştırır.

Mekanizmanın ölçekle ilişkisi Faz 0 mikrobenchmark'ında ayrıca karakterize
edilmiştir (Tablo 4.3 ile aynı deney): MiniMamba'da dizi uzunluğu L 5.2× (196 →
1024) artırıldığında ONNX graf düğüm sayısı tam olarak 5.2× (17 671 → 91 363)
artmış — **L ile lineer, yani tam unroll** — buna karşılık ORT yükleme süresi
**17.9×** (2.0 s → 35.4 s) artmıştır. Yükleme maliyetinin graf boyutuna göre
süperlineer büyümesi, gerçek VMamba-T ölçeğinde (512² → çift yönlü 4 tarama)
gözlenen 12 dakikalık yüklemenin öngörülebilir sonucuydu. Model *boyutunun*
değil graf *yapısının* belirleyici olduğunun bir diğer kanıtı kontrol
modelindedir: 25.6M parametrelik ResNet-50'nin CoreML dönüşümü 2.5 s iken 400K
parametrelik SSM mikro-modelininki 5 668 s sürmüştür — ~2 000× fark, parametre
sayısının tersi yönünde (§3.5.7).

CoreML yolunda aynı katmanlar şöyle görünmektedir:

**Tablo 4.10 — CoreML yolu (512², fp32; coremltools 9.0)**

| Katman | ConvNeXt-T | Swin-T | **VMamba-T** |
|---|---|---|---|
| Trace süresi | 3.5 s | 3.8 s | **952.4 s (16 dk)** |
| Dönüşüm süresi | 4.1 s ✅ | 9.3 s ✅ | **❌ 574.8 s sonra TypeError** |
| İlk tahmin | 0.17 s | 4.41 s | — |

VMamba için CoreML yolu katman (a)'da sonlanmaktadır: 16 dakikalık trace'in
ardından dönüşüm, 390K düğümlü grafın derinliklerinde üçüncü-parti koddaki
(`third_party/VMamba`) dinamik shape okumalarının ürettiği bir `aten::Int`
düğümünde, 574.8 saniye çalıştıktan sonra `TypeError` ile düşmektedir. Aynı hata
sınıfı bizim kontrolümüzdeki başlık/sarmalayıcı kodunda giderilebilmişti
(§4.3.5); farkı yaratan, hatalı desenin bu kez elle erişilemeyecek kadar büyük ve
üretilmiş bir grafın içinde olmasıdır.

Çözünürlük ölçeklendirmenin dışa aktarma katmanına etkisi, VMamba ONNX yolunda
üç çözünürlükte karakterize edilmiştir:

**Tablo 4.11 — VMamba-T ONNX dışa aktarması × çözünürlük**

| Katman | 256² (L=4 096) | 512² (L=16 384) | 1024² (L=65 536) |
|---|---|---|---|
| Export süresi | 121 s | 538 s | **✗ tamamlanamadı** |
| Graf boyutu | 386.6 MB | 858.4 MB | — |
| Graf düğüm sayısı | 98 918 | 390 758 | — (beklenen ~1.5M) |
| ORT yükleme | 46.2 s | 724.9 s | — |
| Export tepe RSS | 4.5 GB | 6.4 GB | **~65 GB → manuel sonlandırma** |

Düğüm sayısı, mikrobenchmark'ın öngördüğü gibi L ile ~lineer büyümektedir
(4× piksel → 3.95× düğüm); ORT yüklemesi ise yine süperlineerdir (4× düğüm →
15.7× yükleme: 46.2 → 724.9 s). Kritik bulgu 1024² sütunundadır: dışa aktarma,
protobuf'un 2 GB serileştirme sınırına ulaşamadan **bellek duvarına**
çarpmıştır — süreç, 24 GB fiziksel RAM'in ~2.7 katına (~65 GB, yoğun swap)
şişerek makineyi kullanılamaz hâle getirdiği için 10 dakikanın ardından elle
sonlandırılmıştır (ham kayıt: `export_matrix.jsonl`, `terminated_by: "user"`).
Aynı işlem klasik omurgalarda her çözünürlükte saniyeler ve yüzlerce megabayt
mertebesindedir. Bulgunun tez açısından ağırlığı şudur: **SSM'in teorik avantaj
bölgesi — çözünürlük büyüdükçe açılması vaat edilen makas — tam da dışa
aktarmanın fiziksel olarak imkânsızlaştığı bölgedir.** 256²'de export 2 dakikaya,
yükleme 46 saniyeye inmektedir; ama 256²'de SSM'in ölçekleme avantajını arayan
da yoktur. Vaat ile araç zinciri, çözünürlük ekseninde ters yönde
ölçeklenmektedir.

Katman (d)'nin "hayatta kalması" salt bir teselli değildir; hem tam model hem
mikrobenchmark, dönüşümü *başarabilen* bir SSM'in çıkarımda ciddi kazanç elde
edebildiğini göstermektedir. Tam modelde ORT medyanı eager'ın 0.30×'udur
(618'e karşı 2 032 ms, §4.3.1); MiniMamba'da ORT CPU çıkarımı eager torch'un
0.77× / 0.68×'i sürede tamamlanmış (L=196'da 13.5'e karşı 17.5 ms, L=1024'te
40.0'a karşı 59.3 ms), CoreML'e dönüşen model ise ALL hesaplama birimleriyle
**15–22× hızlanmıştır** (L=196'da 0.79 ms, L=1024'te 3.83 ms). Yani doğru
anlatı "SSM'ler uçta yavaş çalışır" değildir; ödül kapının ardında durmaktadır
ve kapı — dönüşüm/derleme katmanı — mikro-modelde 94 dakikaya mal olmakta
(L=1024 CoreML dönüşümü 5 668 s), tam modelde ise hiç açılmamaktadır.

Bu dört katmanlı tablo, AS2'nin cevabının ilk yarısıdır: teorik FLOPs analizi
yalnızca katman (d)'yi modelleyebilir ve katman (d) SSM için neredeyse
sorunsuzdur — dahası, dönüşüm başarılabildiğinde kazanca bile dönüşebilmektedir.
Avantajın buharlaştığı yer (a)–(c) katmanlarıdır ve bu katmanlar literatürün
standart raporlama pratiğinde görünmezdir.

### 4.3.3 Operatör Dökümü: Ön Bulgular

Graf patlamasının anatomisi, dışa aktarılan ONNX graflarının operatör dökümünde
görülebilir (ham kayıt: `export_matrix.jsonl`, `graph` aşaması):

**Tablo 4.12 — ONNX graf operatör dökümü (en kalabalık altı op)**

| | ConvNeXt-T (843 düğüm) | Swin-T (8 667 düğüm) | **VMamba-T (390 758 düğüm)** |
|---|---|---|---|
| 1 | Constant ×346 | Constant ×3 682 | **Gather ×139 798** |
| 2 | Add ×75 | Shape ×702 | Constant ×63 686 |
| 3 | Slice ×73 | Unsqueeze ×564 | Add ×46 673 |
| 4 | Mul ×54 | Slice ×428 | Mul ×46 616 |
| 5 | ReduceMean ×50 | Cast ×348 | **Einsum ×46 614** |
| 6 | Transpose ×49 | Reshape ×335 | Unsqueeze ×46 592 |

Döküm, mekanizmayı operatör düzeyinde teşhis ettiriyor. Her üç grafta da
**tek bir `Loop`/`Scan` düğümü yoktur**: TorchScript exporter, VMamba'nın dört
yönlü taramalarını kontrol akışı olarak temsil etmek yerine **tamamen açmıştır**
(unroll). Ardışık taramanın her zaman adımı, grafa ayrı birer düğüm kümesi olarak
serilmiştir; VMamba sütunundaki desen bunun imzasıdır — zaman adımı başına bir
durum okuma (`Gather`), bir durum güncelleme aritmetiği (`Add`/`Mul`/`Einsum`
üçlüsünün neredeyse birebir eşit sayıları: 46 6xx) ve yardımcı şekil işlemleri.
139 798 adet `Gather`, taramanın adım adım bellek erişiminin grafa dökülmüş
hâlidir. Klasiklerin dökümü ise sağlıklı bir derlenmiş modelin görünümündedir:
ConvNeXt kompakt ve hesap-ağırlıklı, Swin'in kalabalığı pencere aritmetiğinin
şekil işlemlerinden ibarettir. ONNX topluluğunda `Loop` operatörünün
yorumlayıcı yükü bilinen bir sorundur; bu yolda aynı hastalık `Loop` yerine
**graf patlaması** olarak tezahür etmektedir — iki semptom, tek neden: graf
temsilleri ardışık yinelemeyi ifade etmekte yapısal olarak zorlanmaktadır.

Bu döküm statik graf analizidir; hangi operatörün *çalışma zamanında* ne kadar
süre tükettiği ayrı sorudur — ve cevabı, statik resmin neredeyse tam tersidir.
ORT profiler ile alınan düğüm-başına süre dökümü (TASK-022; ham kayıt:
`ort_profile_{model}_top.json`):

**Tablo 4.13 — ORT CPU çalışma-zamanı operatör profili (düğüm süresi payı, %)**

| Sıra | ConvNeXt-T | Swin-T | **VMamba-T** |
|---|---|---|---|
| 1 | NhwcFusedConv %78.7 | NhwcFusedConv %76.5 | **NhwcFusedConv %81.7** |
| 2 | Gemm %7.0 | Gemm %7.4 | FusedConv %3.3 |
| 3 | Conv %4.2 | LayerNormalization %3.1 | Conv %3.1 |
| 4 | LayerNormalization %2.7 | Transpose %2.7 | LayerNormalization %2.7 |
| 5 | Resize %2.1 | Resize %2.2 | Resize %2.7 |
| 6 | Gelu %1.9 | Concat %1.4 | Concat %2.6 |

Statik dökümün baş aktörü çalışma zamanında sahnede yoktur: VMamba grafındaki
139 798 `Gather`, profilin ilk sekiz — hatta listelenen on iki — operatörüne
dahi girememektedir; 46 614 `Einsum` da görünmemektedir. Çalışma zamanının
%81.7'si evrişimlerde geçmekte ve VMamba'nın profili klasiklerin profiliyle
neredeyse aynı şekli almaktadır. Profildeki işlem sayıları aynı şeyi söyler:
listelenen on iki operatör türü koşu başına yalnızca ~530 düğüm örneği
oluşturmaktadır — serileştirilmiş grafta 390 758 düğüm varken (döküm en
maliyetli operatörlerle sınırlı olsa da, yürütülen grafın serileştirilmiş
graftan yüzlerce kat küçük olduğu açıktır). Açıklama, §4.3.1'deki ORT
paradoksunun mekanizmasıdır: ORT'nin graf optimizer'ı (sabit katlama, füzyon),
unroll edilmiş taramanın düğüm kalabalığını oturum yüklemesi sırasında eritmekte
ve geriye klasiklere benzeyen, evrişim-ağırlıklı bir yürütme grafı
bırakmaktadır. 12 dakikalık yükleme bu eritmenin faturasıdır. Maliyet yok
olmamakta, katman (d)'den katman (b)'ye taşınmaktadır — ve katman (b), her
süreç başlatımında yeniden ödenmektedir.

### 4.3.4 ANE Yürütme Analizi: Enerji-İmzası Kanıtı

Apple Silicon'da verimlilik merdiveninin en üst basamağı ANE'dir (Apple Neural
Engine). Ancak §3.5.6'da belirtildiği gibi CoreML, hesaplama birimini şeffaf
olmayan biçimde seçer; `compute_units=ALL` bir tercihtir, garanti değil. Bu
nedenle "hangi model gerçekten ANE'de koşuyor" sorusu doğrudan ölçülmelidir. Bu
bölümde iki bağımsız kanıt kullanılmaktadır: `powermetrics`'in ANE güç rayı
telemetrisi (ANE kullanılmıyorsa rayın gücü sıfırdır, kullanılıyorsa aktif güç
çekimi görülür) ve §3.5.6'nın resmî kanıt saydığı Xcode Core ML Performance
Report'un katman-başına yürütme yeri dökümü (Tablo 4.15).

**Tablo 4.14 — ANE enerji imzası (powermetrics, 200 ms örnekleme, boşta-düşülmüş)**

| Hücre | ANE gücü (ölçüm penceresi) | mJ/çıkarım | Yorum |
|---|---|---|---|
| ConvNeXt CoreML ALL | **3 879 mW — ANE AKTİF** | **458** | ANE yürütmesi doğrulandı |
| ConvNeXt CoreML CPU+GPU | 0 mW | 494 | ANE'siz referans |
| Swin CoreML ALL | 0 mW — `ANECCompile FAILED` → GPU | 607 | sessiz GPU fallback |
| Swin CoreML CPU+GPU | 0 mW | 329 | — |
| VMamba torch MPS | 0 mW | 4 844 | CoreML'e giremiyor |
| VMamba torch CPU | 0 mW | **11 720** | — |

Tablo, üç mimarinin ANE karşısındaki konumunu üç ayrı kademe olarak gösteriyor —
bu tezin **"mimari başına bir kademe"** bulgusu:

1. **CNN (ConvNeXt-T): ANE'ye ulaşıyor.** ALL hücresinde ANE rayı 3 879 mW aktif
   güç çekmektedir ve bu hücre matristeki en düşük enerjiyi (458 mJ/çıkarım)
   vermektedir. ANE, gecikmede değil (Tablo 4.8'de ALL, CPU+GPU'dan yavaştır)
   enerjide kazandırmaktadır — uç cihaz için asıl önemli olan eksende.
2. **Transformer (Swin-T): CoreML'e giriyor, ANE'den dönüyor.** Dönüşüm
   başarılıdır; ancak ALL istendiğinde ANE derleyicisi modeli reddetmekte
   (`ANECCompile FAILED` günlük kaydı) ve yürütme sessizce GPU'ya düşmektedir.
   İmza tutarlıdır: ANE rayı 0 mW'ta kalmakta, ALL hücresi hem gecikmede
   (86.0 > 63.9 ms) hem enerjide (607 > 329 mJ) CPU+GPU'dan kötü çıkmaktadır —
   fallback'in ek yükünün ölçülebilir izi.
3. **SSM (VMamba-T): CoreML'e hiç giremiyor.** ANE sorusu sorulamamaktadır bile;
   model dönüşüm aşamasında elenmektedir (§4.3.2). Eager'da kalan model,
   ConvNeXt-ANE hücresinin **~25 katı** enerji tüketmektedir (11 720'ye karşı
   458 mJ).

Burada görüyoruz ki Apple Silicon'un verimlilik merdiveni mimari ailesine göre
farklı basamaklarda kesilmektedir: CNN son basamağa (ANE) çıkmakta, Transformer
bir alt basamakta (GPU) durmakta, SSM merdivene hiç binememektedir. Enerji
sıralaması da bu kademelenmeyi birebir izlemektedir.

Güç telemetrisi güçlü ama dolaylı bir kanıttır; §3.5.6 gereği "ANE'de
çalışıyor" iddiasının resmî kanıtı Xcode Core ML Performance Report'un
katman-başına yürütme yeri dökümüdür. Bu doğrulama TASK-023'te tamamlanmıştır
ve enerji-imzası okumasını katman düzeyinde, eksiksiz doğrulamaktadır:

**Tablo 4.15 — Xcode Core ML Performance Report (resmî katman-yeri dökümü; ekran görüntüleri EK C)**

| | ConvNeXt-T | Swin-T | VMamba-T |
|---|---|---|---|
| ANE'ye atanan op | **353 / 353 (%100)** | **0 / 631 (%0)** | — (dönüşemiyor) |
| CPU / GPU op | 0 / 0 | 138 / 493 | — |
| Prediction (medyan) | 115.6 ms | 98.3 ms | — |
| Load | 142.4 ms | **5 596.3 ms** | — |
| Compilation | 694.2 ms | 151.5 ms | — |

Rapor, "mimari başına bir kademe" bulgusunu dolaylı kanıttan resmî kanıta
taşımaktadır: ConvNeXt'in 353 operasyonunun **tamamı** ANE'ye atanmıştır
(%100); Swin'in 631 operasyonundan **tek biri bile** ANE'ye atanmamış, yürütme
138 CPU + 493 GPU operasyonuna bölünmüştür; VMamba için rapor üretilememektedir,
çünkü rapor edilecek bir CoreML modeli yoktur. Kademelenmenin resmî hâli budur:
**CNN → %100 ANE; Transformer → %0 ANE (GPU'da); SSM → rapor dahi üretilemiyor.**
İki yan bulgu not edilmelidir. Birincisi, ANE reddi yalnızca yürütmeye değil
yükleme katmanına da yansımaktadır: Swin'in model yüklemesi (5 596 ms)
ConvNeXt'inkinin (142 ms) ~40 katıdır — dört katmanlı maliyet modelinin (b)
katmanı, hesaplama birimi seçiminden bile etkilenmektedir. İkincisi, Xcode'un
prediction medyanları (115.6 / 98.3 ms) harness'ın ALL hücreleriyle
(91.4 / 86.0 ms) aynı mertebededir; ölçüm altyapıları farklı olduğundan bu,
birebir karşılaştırma değil tutarlılık kontrolü olarak okunmalıdır.

### 4.3.5 Dağıtım Sürtünmesi SSM'den Önce Başlıyor

Bu bölümün bulgularını doğru ölçeğe oturtmak için son bir gözlem gereklidir:
dağıtım sürtünmesi VMamba'ya özgü değildir; **klasik omurgalar bile "kutudan
çıktığı gibi" dışa aktarılamamıştır**. Matristeki her ✅ işaretinin arkasında
elle müdahale vardır (ham başarısız denemeler `export_matrix.jsonl`'de aşama
aşama kayıtlıdır):

1. **UPerNet PSP başlığı — `adaptive_avg_pool2d`:** Çıktı boyutu girdiyi
   bölmeyen adaptif havuzlama (3×3 ve 6×6 çıktılar), ONNX TorchScript
   exporter'da desteklenmemektedir; ilk dışa aktarma denemeleri üç omurgada da
   bu operatörde düşmüştür. Statik-dilimli numerik eşdeğer yazılarak çözülmüştür
   (orijinal başlığa karşı azami mutlak sapma ~2×10⁻⁵). Aynı operatör CoreML
   dönüştürücüsünü ve MPS eager arka ucunu da düşürmüştür: **tek bir operatör,
   üç platformda üç ayrı kırılma** — Tablo 4.8'deki "torch MPS (statik PSP)"
   ibaresinin nedeni budur.
2. **mmseg kalıbı — `size=x.shape[2:]`:** Segmentasyon ekosisteminin her yerinde
   bulunan bu dinamik yeniden-boyutlandırma deseni, CoreML dönüştürücüsünü
   anında düşürmektedir (`aten::Int`'e 2-elemanlı dizi hatası — jsonl'deki
   erken `TypeError` kayıtları). Sarmalayıcılarda statik boyutla değiştirilmiştir.
3. **Swin pencere cerrahisi:** Pencere bölümleme/birleştirme ve pad
   hesaplarındaki tensör-değerli shape okumaları aynı hata sınıfını üretmiş;
   ilgili matematik, batch=-1 ve Python-int boyutlarla yeniden yazılmıştır
   (numerik eşdeğerlik, düzenleme-öncesi ONNX çıktısına karşı 5×10⁻⁵ bandında
   doğrulanmıştır).

Bunlara VMamba'nın kendi katmanı eklenmelidir: resmî VMamba deposu, NVIDIA'sız
bir makinede **import dahi edilememektedir**. "PyTorch fallback mevcut" iddiası
pratikte iki CUDA varsayımına takılmaktadır — guard'sız `@triton.jit`
dekoratörleri (triton'un macOS wheel'i yoktur) ve fallback seçilmiş olsa bile
koşulsuz açılan `torch.cuda.device()` bağlamı. Tablo 4.7'deki eager ✅ işareti
dahi iki kaynak yamasının ürünüdür ("fallback var" ile "fallback çalışıyor"
arasındaki fark, ölçülebilir bir dağıtım engelidir).

Buradaki resim bir süreklilik olarak okunmalıdır: dağıtım araç zincirlerinde
sürtünme geneldir ve mimariden bağımsız olarak vardır; ancak klasik omurgalarda
bu sürtünme **sınırlı sayıda, yerelleştirilebilir ve elle giderilebilir**
noktadadır (birkaç satırlık cerrahi ile üç omurga da dışa aktarılabilmiştir).
SSM'de ise aynı sürtünme, taramanın grafa 390 758 düğüm olarak açılmasıyla
**elle müdahale edilemez ölçeğe** taşınmakta ve CoreML örneğinde düpedüz
**imkânsızlığa** dönüşmektedir. Nicel fark (448× export, ~7 249× yükleme) bir
noktadan sonra nitel farka dönüşmüştür; bu dönüşümün mimari kökeni ve
uygulayıcılar için sonuçları Bölüm 5'te tartışılmaktadır.

### 4.3.6 Ara Özet: AS2'nin Katmanlı Cevabı

Tamamlanmış matrisle AS2'nin cevabı şu şekilde katmanlanmaktadır. Teorik FLOPs
avantajı ile gerçekleşen verimlilik arasındaki fark, dağıtım yığını merdiveninde
basamak basamak büyümektedir: eager CPU'da 3.6–3.8× olan gecikme farkı (Tablo
4.2), her omurganın kendi dağıtılabilir-en-iyi hücresi karşılaştırıldığında 16×'e
(Tablo 4.8), enerji ekseninde ise ~25×'e (Tablo 4.4/4.14) açılmakta; çözünürlük
ekseninde makas kapanmamakta (Tablo 4.6) ve teorik avantaj bölgesi olan 1024²'de
dışa aktarma bellek duvarına çarpmaktadır (Tablo 4.11). Bu büyümenin mekanizması,
çıkarım aritmetiğinin yavaşlaması değildir — tam tersidir: ORT yolunda VMamba'nın
çıkarım medyanı eager'ının 0.30×'udur ve klasiklerin ORT hücrelerinden bile
hızlıdır (ORT paradoksu, §4.3.1); mikrobenchmark da dönüşümü başarabilen SSM'in
çıkarımda 15–22× kazanabildiğini göstermiştir. Naif okuyucunun "SSM uçta yavaş
çalışır" beklentisinin aksine, çıkarım katmanı (d) SSM için avantajlı
çıkmaktadır; imkânsızlaştıran, çıkarım-öncesi katmanlardır: ardışık taramanın
graf temsiline L ile lineer büyüyen yapı olarak girmesi (tam unroll, 390 758
düğüm, Gather ×139 798) ve bedelin dönüşümde (448×; 1024²'de bellek duvarıyla
düpedüz başarısızlık), pakette (614 MB saf graf yapısı), yüklemede (620–725 s,
her süreç başlatımında yeniden) ve CoreML örneğinde dönüşümün kendisinde
ödenmesi. Bu, "maliyet buharlaşmıyor, yanlış katmana taşınıyor" tezini
güçlendirir: FLOPs analizinin tek gördüğü katman kazanca dönüşürken, FLOPs'un
kör olduğu katmanlar dağıtımı imkânsızlaştırmaktadır. `torch.compile`'ın
VMamba'da süreç çökmesi, derleme katmanı kırılganlığının üçüncü bağımsız
zincirdeki tekrarıdır; ORT CoreML EP'nin bölümleme parçalanması (Swin'de 94
parça, saf CoreML'in ~5× yavaşı) kısmî hızlandırıcı erişiminin sınırını
çizmektedir. Klasik omurgaların da sürtünmesiz olmadığı (üç platformda üç
cerrahi, §4.3.5), ancak sürtünmenin yalnızca SSM'de imkânsızlığa dönüştüğü not
edilmelidir. Matris durumu (Tablo 4.7), dört katmanlı maliyet asimetrisi (Tablo
4.9) ve resmî ANE kademelenmesi (Tablo 4.15) birlikte, tezin ana iddiasının —
avantajın özel çekirdeklere ve araç zinciri desteğine bağımlı olduğu, genel
amaçlı yığınlarda maliyetin yok olmayıp yanlış katmana taşındığı — ampirik
çekirdeğini oluşturmaktadır. Yorum ve genelleme Bölüm 5'e bırakılmıştır.

---

*Sayfa hedefi: ~10–12. Deney matrisi kapanmıştır (TASK-020/021/022/023);
v1'deki dokuz [ölçüm sürüyor] işareti bu sürümde final verilerle doldurulmuştur.
Kalan iş: Şekil 4.1–4.2'nin Tablo 4.6 ve 4.8'deki verilerden üretilmesi.*


---

# 4.4 AS3: Nicemlemenin Yoğun Tahmine Transferi

*(TASLAK v1 — 13 Ağustos 2026, TASK-029)*

> Bu bölüm, üçüncü araştırma sorusunun (AS3) ölçümlerini raporlar: sınıflandırma
> görevinde kanıtlanmış PTQ pratiği, yüksek çözünürlüklü yoğun tahmine
> taşındığında ne verir? Protokol Bölüm 3.6'daki Aşama I'dir (kalibrasyonsuz
> ağırlık-nicemleme, yığınların yerleşik yollarıyla); ham kayıtlar `results/raw/`
> altındadır: `quant_matrix.jsonl`, `quant_miou_*.json`,
> `activation_stats_{vmamba,swin,convnext}.json` ve `latency_matrix_*.jsonl`
> (quant etiketli kayıtlar). Aktivasyon nicemlemesi (W8A8) ve kalibrasyonlu
> statik nicemleme Aşama II kapsamındadır ve bu bölümde raporlanmamaktadır.

Bölüm 4.3'ün ana bulgusu, SSM'in dağıtım engelinin çıkarım aritmetiğinde değil
araç zincirinin çıkarım-öncesi katmanlarında olduğuydu. Nicemleme, bu tabloya
kritik bir soruyla girer: uç dağıtımın standart sıkıştırma aracı, dört katmanlı
maliyet modelinin (§4.3.2) hangi katmanlarına dokunabilmektedir? Literatürün
vaadi katman (c) (paket boyutu) ve — donanım desteği varsa — katman (d)
(çıkarım gecikmesi) üzerinedir; bu bölümün ölçümleri vaadin her iki ayağını da
yığın-başına sınamakta ve bir üçüncü soruyu eklemektedir: SSM nicemleme
literatürünün merkezî anlatısı olan aykırı değer problemi (§2.4.2–2.4.3), yoğun
tahmin çözünürlüklerinde gerçekten gözleniyor mu?

## 4.4.1 Sınıflandırma vs Segmentasyon: Doğruluk Kaybının Dağılımı

Ağırlık-nicemleme matrisinin doğruluk-boyut-gecikme kesiti Tablo 4.16'dadır.
Protokol notu önemlidir: buradaki mIoU değerleri, dışa aktarılan modellerin
sabit-şekil kısıtı nedeniyle **kare-512, 250-görüntü** protokolüyle ölçülmüştür
(en-boy oranı korunmaz); bu nedenle Bölüm 4.1'in tam-değerlendirme mIoU
bandıyla (44.3–48.3) doğrudan karşılaştırılamaz ve mutlak değerler değil,
**fp32-export tabanına göre deltalar** esastır. fp32-export satırı, nicemleme
etkisini dışa aktarma etkisinden ayırmak için tabana konmuştur: her nicemli
model, aynı protokolle ölçülen kendi fp32 dışa aktarımıyla karşılaştırılır.

**Tablo 4.16 — Ağırlık-nicemleme matrisi: boyut, mIoU, gecikme (CoreML, 512²; mIoU: 250 görüntü, kare-512)**

| | fp32-export | W8 doğrusal | W4 palettizasyon |
|---|---|---|---|
| ConvNeXt-T boyut | 118.8 MB | 59.7 MB (0.50×) | 30.0 MB (0.25×) |
| ConvNeXt-T mIoU | 34.93 | **35.00 (±0)** | 31.08 (**−3.9**) |
| ConvNeXt-T CoreML ALL gecikme | 91.4 ms | **64.7 ms (−%29)** | 59.3 ms |
| Swin-T boyut | 127.5 MB | 64.1 MB (0.50×) | 32.3 MB (0.25×) |
| Swin-T mIoU | 36.01 | **36.05 (±0)** | 33.26 (−2.8) |
| Swin-T CoreML ALL gecikme | 86.0 ms | 84.1 ms (≈) | 92.9 ms |
| VMamba-T | — | — | — *(CoreML'e dönüşemiyor, §4.3.2)* |

Tablonun ilk bulgusu W8 satırındadır ve nettir: **8-bit ağırlık nicemlemesi
yoğun tahminde bedavadır.** Her iki modelde de mIoU değişimi ölçüm gürültüsü
içindedir (ConvNeXt +0.07, Swin +0.04 — pratikte ±0), paket boyutu yarıya
inmekte ve ConvNeXt'in ANE hücresinde gecikme de %29 *düşmektedir* (91.4 →
64.7 ms; mekanizma §4.4.3'te). Sınıflandırma literatürünün W8 için raporladığı
"neredeyse kayıpsız" deseni, segmentasyonda aynen — hatta gecikme tarafında
fazlasıyla — transfer olmaktadır. Uç dağıtım pratiği açısından sonuç
tartışmasızdır: bu yığında W8, fp32-export'a karşı her eksende eşit ya da
üstündür ve varsayılan dağıtım formatı olmalıdır.

İkinci bulgu W4 sütunundadır: 4-bit palettizasyonun faturası vardır ve
**modele göre değişmektedir** — ConvNeXt −3.9 mIoU puanı öderken Swin −2.8
ödemektedir. Boyut kazancı iki modelde de aynıdır (0.25×); farkı yaratan,
ağırlık/aktivasyon dağılımlarının 16 temsilcili bir arama tablosuna ne kadar
sığdığıdır. ConvNeXt'in daha ağır faturası, §4.4.2'deki katman
istatistikleriyle birleştirildiğinde mekanizmaya bağlanabilmektedir: ConvNeXt'in
depthwise evrişim katmanları üç omurganın en uzun kuyruklu (en yüksek
kurtosis'li) dağılımlarını taşımaktadır ve kurtosis çözünürlükle büyümektedir —
4-bit palet, kuyruktaki aykırı kanalları temsil etmekte zorlanan ilk şemadır.
Bulgunun tersten okuması da önemlidir: kayıp −2.8'den −3.9'a değişse de iki
modelde de *çökme* yoktur; kalibrasyonsuz, veri-görmemiş bir W4 dönüşümünün
segmentasyonda hâlâ 31–33 mIoU bandında çalışması, ağırlık tarafının literatürün
işaret ettiği asıl kırılganlık noktası olmadığı (§3.6) öngörüsüyle tutarlıdır.

Üçüncü bulgu metodolojiktir ve dürüstçe kaydedilmelidir. Nicemleme hattı, tam
mIoU değerlendirmesinden önce ucuz bir eleme metriği olarak rastgele girdiyle
fp32-nicemli çıktı karşılaştırması kullanmaktadır (§3.6): logit sapması ve
piksel-başına argmax eşleşme oranı. Tablo 4.17, bu vekil metriğin gerçek görev
kaybıyla ilişkisini göstermektedir:

**Tablo 4.17 — Vekil metrik (rastgele-girdi argmax eşleşmesi) ile gerçek kayıp (ΔmIoU) karşılaştırması**

| Model / kip | Argmax eşleşmesi (rastgele girdi) | ΔmIoU (250 görüntü, gerçek veri) |
|---|---|---|
| ConvNeXt W8 | %90.4 | **+0.07** |
| Swin W8 | %98.0 | +0.04 |
| ConvNeXt W4 | **%41.6** | −3.9 |
| Swin W4 | %86.3 | −2.8 |

İlişki monotondur (eşleşme düştükçe kayıp büyür) ama şiddeti sistematik olarak
abartılıdır: ConvNeXt W4'te rastgele-girdi eşleşmesi %41.6'ya düşmüşken —
naif okumayla "model bozuldu" — gerçek verideki kayıp −3.9 puandır ve model
31.08 mIoU ile çalışır durumdadır. Aynı yönde, W8'de %90.4'lük eşleşme sıfır
kayba karşılık gelmektedir. Mekanizma tahmin edilebilir: rastgele girdiler doğal
görüntü istatistiğinden uzaktır ve sınıf sınırlarındaki kararsız pikselleri
(logit farkları küçük olan bölgeleri) orantısız örnekler; doğal görüntülerde ise
kararlar büyük marjlarla verilir ve küçük logit sapmaları argmax'ı devirmez. Ders
şudur: **rastgele-girdi vekil metrikleri eleme sırası belirlemek için
kullanılabilir, hasar tahmini için kullanılamaz** — bu uyarı Bölüm 3.6'ya
işlenmiştir. Yoğun tahmin bağlamında ek bir incelik daha vardır: mIoU,
piksel-başına bir metriktir ve sınır piksellerindeki dalgalanmalara sınıf-içi
alan pikselleri kadar duyarlı değildir; argmax eşleşmesi ise her pikseli eşit
sayar.

## 4.4.2 Çözünürlüğün Aykırı Değer Dağılımına Etkisi: Beklenen Fırtına Gözlenmedi

AS3'ün mekanizma sorusu aktivasyon tarafındadır. SSM nicemleme literatürünün
merkezî anlatısı (§2.4.2), seçicilik mekanizmasının aktivasyonlarda yapısal
aykırı değerler ürettiğidir: token-bazlı varyans, kanal-bazlı aykırı kanallar,
uzun kuyruklar (PTQ4VM taksonomisi) ve bunların zaman adımları arasında dinamik
kayması (OuroMamba). §2.4.3'te bu anlatıdan türetilen ölçek hipotezi şuydu:
dizi uzunluğu büyüdükçe (segmentasyonda on binlerce token) zamansal istatistik
kayması için alan açılır — sınıflandırmada tolere edilen dinamiklik yoğun tahmin
rejiminde büyüyor olmalıdır. TASK-027 bu hipotezi doğrudan sınamıştır: üç
omurganın onar temsilci katmanında (VMamba'da SSM bloklarının `in_proj`
girişleri, ConvNeXt'te depthwise/pointwise evrişimler, Swin'de dikkat
projeksiyonları), 12 ADE20K görüntüsüyle, 256²/512²/768² çözünürlüklerinde
katman-başına dağılım istatistikleri çıkarılmıştır: tensör maksimumu, p99.9,
kurtosis (kuyruk ağırlığı) ve kanal-maksimumunun kanal-medyanına oranı
(chmax/med — aykırı *kanal* makasının ölçüsü).

**Tablo 4.18 — Aktivasyon istatistikleri × çözünürlük (10 katman üzerinden medyan ve en-kötü değer)**

| Omurga | Metrik | 256² | 512² | 768² | Eğilim |
|---|---|---|---|---|---|
| **VMamba-T** | kurtosis (medyan / maks) | 7.8 / 15.0 | 8.3 / 14.7 | 8.8 / **14.0** | **sabit** |
| | chmax/med (maks) | 3.82 | 3.42 | **3.21** | **daralıyor** |
| ConvNeXt-T | kurtosis (medyan / maks) | 18.1 / 63.0 | 23.5 / 71.0 | 27.2 / **75.0** | **büyüyor** |
| | chmax/med (maks) | 4.52 | 4.00 | 4.57 | sabit |
| Swin-T | kurtosis (medyan / maks) | 11.5 / 32.6 | 10.8 / 18.0 | 9.2 / 18.8 | sabit/azalıyor |
| | chmax/med (maks) | 4.64 | 4.07 | 3.95 | sabit |

Tablo, bu ölçüm kampanyasının en beklenmedik sonucunu içermektedir: **VMamba'nın
aktivasyon profili üç omurganın en ılımlısıdır ve çözünürlükle
bozulmamaktadır.** Kurtosis medyanı ~8, en kötü katmanda ~14 düzeyinde sabittir;
aykırı kanal makası (chmax/med) 3.8'den 3.2'ye *daralmaktadır*. Literatürün
kuramsal beklentisi tam tersiydi: girdiye bağlı kapılama ve üstel ayrıklaştırma
kuyrukları ağırlaştırmalı, dizi uzadıkça durum birikiminin varyansı istatistiği
savurmalıydı. Bu omurga/görev/katman kümesinde gözlenen budur: L'nin 4 096'dan
36 864'e (9×) çıkması, dağılım şeklini pratikte değiştirmemiştir. Beklenen
fırtınayı üreten omurga ise klasiklerden biridir: ConvNeXt'in depthwise evrişim
katmanları hem en yüksek kuyruk ağırlığını taşımakta (ilk-katman depthwise'ta
kurtosis 43 → 56 → 64; en kötü katman 768²'de 75) hem de üç omurga içinde
çözünürlükle *monoton büyüyen* tek kurtosis serisini vermektedir. Bu gözlem,
§4.4.1'deki W4 faturasının asimetrisini (ConvNeXt −3.9 > Swin −2.8) mekanizmaya
bağlar: 16 seviyeli palet, tam da bu uzun kuyruklu depthwise dağılımlarında
çözünürlük kaybetmektedir. (ConvNeXt aktivasyonlarındaki aykırı kanal olgusu
literatürde bilinmektedir ve LayerScale/GRN gibi tasarım yamalarının
motivasyonudur; buradaki katkı, aynı istatistiğin çözünürlük ekseninde
büyüdüğünün ve W4 kaybıyla eşleştiğinin ölçülmesidir.)

İki nüans bulguyu dengeler. Birincisi, VMamba'nın *mutlak* aktivasyon genlikleri
üç omurganın en büyüğüdür (tensör maksimumu ~29–35'e karşı ConvNeXt'te ~14,
Swin'de ~5.6). Yani VMamba'nın aktivasyon tensörleri geniş bir aralık kaplar;
ancak nicemleme açısından belirleyici olan aralığın kendisi değil, dağılımın o
aralığı ne kadar verimli doldurduğudur — ölçek genişliği tek başına bir bit
maliyetidir, aykırı değer problemi ise kuyruk/gövde ve kanal/kanal
*oransızlığıdır*. VMamba'da ölçülen düşük kurtosis ve dar kanal makası, tekdüze
bir nicemleme ızgarasının bu geniş aralığı görece verimli kullanabileceğine
işaret eder. İkincisi, bu ölçümler literatürle *çelişki* değil *kapsam
ayrışması* olarak okunmalıdır. PTQ4VM/OuroMamba bulguları sınıflandırma
omurgalarında, farklı model örneklerinde ve büyük ölçüde SSM bloğunun *iç*
aktivasyonlarında (durum güncellemeleri, kapı çıktıları) raporlanmıştır; bizim
ölçümümüz ADE20K ile eğitilmiş VMamba-T'nin blok *giriş* projeksiyonlarını
örneklemektedir. Segmentasyon eğitiminin kendisinin de dağılımları düzenlemiş
olması mümkündür. Kapsayıcı iddia şu şekilde daraltılarak kurulmalıdır: **bu
tezin ölçtüğü omurga, görev ve katman kümesinde, literatürün SSM'e atfettiği
aykırı değer patolojisi gözlenmemiş; çözünürlük büyüdükçe kötüleşme hipotezi
desteklenmemiştir.** Bu daraltılmış hâliyle bile bulgu önemlidir, çünkü AS3'ün
ölçek hipotezini — "yoğun tahmin rejimi SSM nicemlemesini sınıflandırmadan daha
da zorlaştırır" — eldeki veride yanlışlamaktadır: en azından ağırlık-nicemleme
ve aktivasyon-istatistiği düzeyinde, VMamba nicemlemeye üç omurganın en *uygun*
adayı görünmektedir.

> **Şekil 4.6 — Aykırı değer istatistikleri × çözünürlük** *(yer tutucu; veri
> tamamlandı — `activation_stats_*.json`; şekil bu kayıtlardan üretilecektir)*.
> Üç omurganın onar temsilci katmanı için katman-başına kurtosis (üst panel) ve
> chmax/med (alt panel), 256²/512²/768² çözünürlüklerinde. Beklenen okuma:
> ConvNeXt'in depthwise katmanlarında çözünürlükle yükselen kurtosis serisi ile
> VMamba'nın yatay, düşük-bantta seyreden profili arasındaki karşıtlık —
> literatürün SSM aykırı-değer beklentisinin bu veri kümesindeki tersine dönüşü.

## 4.4.3 Nicemleme Sonrası Gecikme Kazancı: Aynı Fikir, Üç Farklı Sonuç

Bölüm 2.4.1'deki uyarı — "nicemlemenin gecikme kazancı otomatik değildir" — bu
ölçüm kampanyasında beklenenden sert biçimde doğrulanmıştır: **aynı 8-bit
ağırlık-nicemleme fikri, çalıştığı yığına göre kazanca, etkisizliğe veya
katbekat kayba dönüşmektedir.**

Kazanç ucu CoreML/ANE hücresindedir. ConvNeXt'in ALL (ANE-dahil) gecikmesi
W8 ile 91.4'ten 64.7 ms'ye inmiştir (−%29) — fp32'de CPU+GPU hücresinin
gerisinde kalan ALL hücresi (§4.3.1, Tablo 4.8), W8 ile onunla eşitlenmiştir.
Mekanizma, §4.3.4'ün ANE analiziyle tutarlıdır: ANE düşük-hassasiyetli aritmetik
için tasarlanmış bir birimdir ve ağırlıkların 8-bit gelmesi hem bant genişliği
hem yürütme tarafında ANE'nin doğal rejimine denk düşer. Karşı-kanıt da kendi
içindedir: Swin'in ALL hücresi ANE'ye atanamadığı ve GPU'ya düştüğü için
(Tablo 4.15: 0/631 op ANE'de) W8'den neredeyse hiç kazanç görmemektedir (86.0 →
84.1 ms). Yani %29'luk kazanç "CoreML'in" değil, **ANE'nin** kazancıdır;
nicemleme kazancı yalnızca yığına değil, yığın içindeki hesaplama birimine bile
bağlıdır. Aynı ayrışma W4 palettizasyonda daha da keskindir: ALL hücresinde
palet ConvNeXt'i 59.3 ms'ye kadar indirirken, aynı paketin CPU+GPU hücresi
132.4 ms'ye *çıkmaktadır* (fp32 CPU+GPU tabanının 2.1 katı) — arama tablosunu
donanımda çözebilen birim için palet bir sıkıştırma, çözemeyen birim için her
çıkarımda ödenen bir açma maliyetidir.

Kaybın ucu ise ORT dinamik INT8'dedir:

**Tablo 4.19 — ORT dinamik INT8 (CPU, 512²): boyut, gecikme, yükleme**

| | ConvNeXt-T | Swin-T | **VMamba-T** |
|---|---|---|---|
| Boyut (fp32 → INT8) | 237 → 60 MB (0.25×) | 239 → 63 MB (0.26×) | **858 → 715 MB (0.83×)** |
| Gecikme (fp32 ORT → INT8) | 644 → **10 863 ms (17×)** | 714 → 7 094 ms (10×) | 618 → 3 752 ms (6×) |
| Oturum yüklemesi | 0.2 s | 0.3 s | **691 s (değişmedi)** |

Klasiklerde tablo, "INT8 = hız" varsayımının çıplak çürütülmesidir: dinamik
nicemleme boyutu dörtte bire indirirken gecikmeyi 10–17 katına çıkarmaktadır.
Mekanizma bilinen bir ORT-CPU davranışıdır: `quantize_dynamic` ağırlıkları INT8
saklar ama bu donanım/derleyici kombinasyonunda evrişim-ağırlıklı grafa verimli
tam sayı çekirdekleri sunulamadığında, çalışma zamanı her çıkarımda
nicemleme/dequantizasyon köprüleri ödemektedir; ayrıca Tablo 4.19'daki INT8
koşularının yüksek koşu-içi varyansı (ConvNeXt'te örnekler 4.6–22.4 s bandında)
bu yolun termal/bellek baskısına da açık olduğunu göstermektedir. Uygulayıcı
çıkarımı nettir ve Bulgu 2 olarak kayıtlıdır: nicemleme kararı, hedef yığında
*ölçülmeden* verilemez — aynı fikir bir yığında %29 kazanç, komşu yığında 17×
kayıptır.

Asıl tez-düzeyi bulgu ise VMamba sütunundadır. VMamba'nın ORT INT8 hücresi,
dört katmanlı maliyet modelinin nicemleme eksenindeki testi olarak özel ilgiyle
izleniyordu (§3.6) ve sonuç modelin öngörüsünü doğrulamaktadır: **yapısal
şişkinlik nicemlenememektedir.** Klasiklerin grafı 0.25×'e inerken VMamba'nın
858 MB'lık grafı yalnızca 715 MB'a (0.83×) inmiştir — çünkü Tablo 4.9'un
gösterdiği gibi bu grafın ~614 MB'ı ağırlık değil, unroll edilmiş taramanın
serileştirilmiş operatör düğümleridir ve nicemlemenin dokunabildiği tek bileşen
~244 MB'lık ağırlık payıdır. Belirleyici gösterge yükleme satırıdır: 691
saniyelik oturum yüklemesi, fp32'deki değerle (620–725 s bandı, §4.3.1–4.3.2)
pratikte aynıdır. Yükleme maliyetinin kaynağı 390 758 düğümlük graf yapısı
olduğundan ve nicemleme bu yapıya dokunmadığından, VMamba'nın gerçek dağıtım
engeli — her süreç başlatımında ödenen ~12 dakika — nicemlemeden *hiç*
etkilenmemiştir. Katman diliyle: nicemleme katman (c)'nin ağırlık payını ve
(donanım uygunsa) katman (d)'yi iyileştirebilen bir araçtır; VMamba'nın engeli
ise katman (a)/(b)'de, grafın *yapısındadır*.

> **Şekil 4.5 — Nicemleme düzlemi: boyut-mIoU-gecikme** *(yer tutucu; veri
> tamamlandı — Tablo 4.16/4.19; şekil bu tablolardan üretilecektir)*. Yatay
> eksen paket boyutu (MB, log), dikey eksen mIoU; işaretçi boyutu/etiketi CoreML
> ALL gecikmesi. Her model için fp32-export → W8 → W4 yörüngesi ok ile çizili;
> ORT INT8 hücreleri ayrı işaretçiyle (gecikme pesimizasyonu etiketli). Beklenen
> okuma: W8'in "sola bedava kayış" oluşu (boyut yarıya, mIoU ve gecikme aynı
> veya iyi), W4'ün modele göre değişen aşağı kırılımı ve VMamba INT8 noktasının
> hem boyut ekseninde hem gecikme etiketinde yörünge dışında kalışı.

## 4.4.4 Ara Özet: AS3'ün Cevabı ve Tezin İronisi

AS3'ün cevabı üç katmanda toplanmaktadır. **Doğruluk katmanında** transfer
sorunsuzdur: W8 ağırlık-nicemlemesi segmentasyonda sıfır mIoU kaybıyla boyutu
yarılamakta, W4 palettizasyon ise modelin aktivasyon/ağırlık kuyruk profiline
göre −2.8 ile −3.9 puan arasında değişen, çökme içermeyen bir fatura
kesmektedir. **Mekanizma katmanında** literatür beklentisi tersine dönmüştür:
çözünürlük büyüdükçe SSM aktivasyonlarının savrulacağı hipotezi bu veri
kümesinde desteklenmemiş; VMamba üç omurganın en ılımlı ve en
çözünürlük-kararlı aktivasyon profilini vermiş, beklenen uzun-kuyruk patolojisi
ConvNeXt'in depthwise katmanlarında ortaya çıkmıştır. **Gecikme katmanında**
kazanç tamamen yığına ve hesaplama birimine bağlıdır: ANE'de %29 kazanç, GPU
fallback'inde etkisizlik, ORT-CPU'da 6–17× pesimizasyon.

Bu üç katman birlikte, tezin ana anlatısına ironik bir kapanış eklemektedir.
Nicemlemeye — ölçülen aktivasyon istatistikleri itibarıyla — en uygun omurga
VMamba'dır; ama VMamba, nicemlemenin kazanca dönüştüğü tek hücreye (CoreML/ANE)
dönüşüm katmanında elendiği için hiç ulaşamamakta, ulaşabildiği tek nicemleme
yolunda (ORT INT8) ise kazanılacak olan şey — ağırlık baytları — zaten sorunun
küçük parçası olduğu için tablo değişmemektedir: boyut 0.83×, yükleme aynı 691
saniye. AS3'ün sonucu böylece AS2'nin sonucunu bağımsız bir eksenden
güçlendirmektedir: **SSM'in uç dağıtım engeli sayısal değil, yapısaldır.**
Aykırı değerler ehlileştirilebilir, ağırlıklar sıkıştırılabilir; ama taramanın
graf temsiline 390 758 düğüm olarak açılan yapısı, bit genişliğinden bağımsız
olarak orada durmaktadır. Nicemleme literatürünün SSM için öngördüğü savaş
(dinamik aykırı değerler, §2.4.3) bu cephede hiç başlamamış; savaş, literatürün
raporlamadığı cephede (dönüşüm ve yükleme katmanları) çoktan kaybedilmiştir.
Export-dostu yeniden formülasyonun (AS4, Faz 4) hedefi tam da bu yapıyı
değiştirmektir; nicemleme bulguları, o kapı açılabilirse ardında ANE'nin
düşük-hassasiyet kazancının SSM'i de bekliyor olabileceğine dair dolaylı bir
umut sunmaktadır.

### Kapsam sınırları

Bu bölümün iddiaları şu sınırlar içinde okunmalıdır:

1. **Yalnız ağırlık-nicemleme (Aşama I).** Aktivasyon nicemlemesi (W8A8) ve
   kalibrasyonlu statik nicemleme yapılmamıştır; literatürün asıl kırılganlık
   öngördüğü aktivasyon tarafı burada yalnızca *istatistik* düzeyinde
   incelenmiş, uçtan-uca W8A8 doğruluk etkisi ölçülmemiştir. §4.4.2'nin ılımlı
   profili W8A8 için umut vericidir ama kanıt değildir (Aşama II, ImageNet
   alt-kümesi kalibrasyonuyla planlanmıştır — §3.6).
2. **mIoU protokolü.** Doğruluk değerleri 250-görüntü alt-küme ve kare-512
   protokolüyledir; deltaların tam-değerlendirme (2 000 görüntü, en-boy
   korumalı) altında doğrulanması Faz 5 öncesine bırakılmıştır.
3. **Aktivasyon istatistiği örneklemi.** Omurga başına 10 temsilci katman,
   12 görüntü, 256²–768² aralığı; VMamba'da yalnızca blok giriş projeksiyonları
   (`in_proj`) örneklenmiştir — SSM bloğunun iç aktivasyonları (durum
   güncellemeleri, kapı çıktıları) enstrümante edilmemiştir ve literatürün
   aykırı değer raporları kısmen o iç noktalara aittir. 1024² çözünürlüğü
   bellek kısıtı nedeniyle istatistik taramasına dahil edilememiştir.
4. **Tek örnek, tek görev.** Bulgular VMamba-T + UPerNet + ADE20K
   kombinasyonuna aittir; segmentasyon eğitiminin dağılımları düzenlemiş olması
   ihtimali dışlanamadığından, "SSM aktivasyonları ılımlıdır" genellemesi değil,
   "bu kümede literatür anlatısı gözlenmedi" tespiti yapılmaktadır.
5. **VMamba'nın CoreML hücreleri tanım gereği boştur.** W8/W4 satırlarının
   VMamba sütunu ölçülememiş değil, *ölçülemez* durumdadır (dönüşüm
   başarısızlığı, §4.3.2); bu boşluk bir veri eksiği değil, §4.4.4'teki
   sentezin kendisidir.

---

*Sayfa hedefi: ~6–8. Aşama I matrisi kapanmıştır (TASK-025..028). Kalan iş:
Şekil 4.5–4.6'nın üretilmesi; Aşama II (W8A8 + kalibrasyon) ölçümleri
gerçekleşirse bölümün genişletilmesi.*


---

# 4.5 AS4: Seçici Taramanın Derleyici-Dostu Yeniden Formülasyonu

*(TASLAK v1 — 14 Ağustos 2026, TASK-033)*

> Bu bölüm, dördüncü araştırma sorusunun (AS4) sonuçlarını raporlar: seçici
> taramanın parçalı/sabit-uzunluklu (chunked) yeniden formülasyonları, ONNX ve
> Core ML hedeflerinde Bölüm 4.3'te belgelenen dağıtım engelini ne ölçüde
> açabilmektedir? Kuramsal dayanak §2.1.3'teki durum-uzayı ikiliğidir (SSD);
> implementasyon `src/reformulation/` altındadır (ayrıca EK D), ham kayıtlar
> `results/raw/reform_matrix.jsonl` dosyasındadır. Bu bölüm tezin *ölçüm*
> fazından *müdahale* fazına geçişidir: önceki bölümler engeli karakterize
> etmişti; bu bölüm engele kasıtlı bir tasarım değişikliğiyle dokunmakta ve
> sonucunu — kazanç ve kayıp ayaklarıyla birlikte — raporlamaktadır.

Deneyin kapsam kararı baştan ve bilinçli olarak **prototip-önce** verilmiştir:
yeniden formülasyon tam VMamba-T modeline değil, **tek bir gerçek SS2D bloğuna**
uygulanmıştır — `layers.2.blocks.0.op`, ADE20K checkpoint'inden yüklenen kendi
ağırlıkları ve boru hattından yakalanan gerçek bir ara-aktivasyon girdisiyle
(1×384×32×32; dört yönlü tarama ile L=1 024, kanal-katlamalı 3 072 skaler
özyineleme). Bu bir daraltma değil, deney tasarımıdır ve gerekçesi Bölüm 4.3'ün
bulgu yapısından gelmektedir. Tam modelde ölçülen patoloji — 390 758 düğümlük
graf, 12 dakikalık yükleme, CoreML dönüşüm hatası — 22 SS2D bloğunun, üçüncü
parti kod tabanının ve dekoder sarmalayıcısının bileşik ürünüdür; yeniden
formüle edilmiş bir taramanın etkisi bu bileşik yapı içinde ölçülürse, gözlenen
her değişim en az üç faktöre birden atfedilebilir hâlde kalır. Tek gerçek blok,
gerçek ağırlık ve gerçek aktivasyon ise **kontrollü atfedilebilirlik** sağlar:
graf boyutu, yükleme süresi, dönüşüm başarısı ve sayısal sadakat, yalnızca
tarama formülasyonunun fonksiyonu olarak okunabilir. Engelin kendisi de blok
başına çarpımsal olduğundan (tam modelin 390K düğümü, blok-düzeyi unroll'un 22
kopyasıdır), blok düzeyinde kırılamayan bir engelin model düzeyinde kırılması
zaten beklenemez — prototip, birinci-derece testin yapıldığı doğru ölçektir.
Ayrıca, aşağıda görüleceği gibi, bu ölçek seçimi tek başına bölümün en önemli
mekanizma bulgusunu üretmiştir.

Üç form karşılaştırılmıştır (adlandırma kod tabanıyla aynıdır):

1. **seq** — ardışık referans: üçüncü-parti `selective_scan` implementasyonuyla
   **bit-eşdeğer** (azami mutlak fark 0.0) doğrulanmış, L adımlık açık
   özyineleme. Trace edildiğinde tamamen unroll olur; tezin gösterdiği graf
   patlamasının tek-blok ölçekli hâlidir.
2. **blocked** — d_state=1 skaler özyinelemesinin blok-içi kapalı formu
   (P ∈ {32, 64, 128} blok uzunluklarıyla; metinde `blocked32` vb.).
3. **ane** — blocked ile aynı matematik, Apple'ın ANE dağıtım reçetesine göre
   yeniden düzenlenmiş tensör yerleşimiyle (§4.5.2).

## 4.5.1 Parçalı Tarama: Graf Boyutu ve Yükleme Süresi

### Kapalı form ve bir stabilite bulgusu

VMamba-T'nin kullandığı varyantta durum boyutu d_state=1'dir; seçici tarama bu
durumda kanal başına *skaler* bir özyinelemeye iner:

h_t = a_t · h_{t−1} + b_t,  a_t = exp(Δ_t · A) ∈ (0, 1]

Log-uzayında S_t = Σ_{i≤t} Δ_i·A birikimli toplamı tanımlandığında, P
uzunluklu bir blok içinde özyinelemenin cebirsel olarak eşdeğer bir kapalı formu
vardır ve ilk denenen reçete literatürdeki standart normalizasyonlu biçimdir:

h_t = e^{S_t} · (h_0 + Σ_{i≤t} b_i · e^{−S_i})

Bu naif form, sentetik girdilerde çalışırken **gerçek checkpoint ağırlıkları ve
gerçek aktivasyonlarla NaN üretmiştir** ve başarısızlığın mekanizması tezin
kendisi için bilgilendiricidir. Ölçüm (`reform_matrix.jsonl`, `verify`
kayıtları): blok-içi log-çürüme birikimi P=64'te −513.6'ya, P=128'de −982.9'a
inmektedir. Yani eğitilmiş model *hızlı unutmaktadır* — 64 adımda durumun
katkısı e^{−513} düzeyine sönmektedir ve bu, seçicilik mekanizmasının tam da
istenen davranışıdır. Ancak naif formdaki e^{−S_i} çarpanı bu çürümenin
*tersini* ister: e^{+513} ≈ 10^{223}, fp32'nin üst sınırının (~e^{88}) çok
ötesindedir ve toplam taşarak NaN'a çöker. Ders genelleştirilebilir niteliktedir:
**ardışık özyinelemede zararsız, hatta faydalı olan güçlü çürüme, kapalı forma
naif geçişte sayısal bombaya dönüşmektedir** — cebirsel eşdeğerlik, kayan-nokta
eşdeğerliği değildir. Bu başarısızlık silinmemiş, veri olarak kaydedilmiştir.

Çalışan form **çürüme-matrisi** formülasyonudur: blok içinde alt-üçgen bir

T[t,i] = e^{S_t − S_i}  (i ≤ t)

matrisi kurulur. Kritik özellik, i ≤ t için S_t − S_i ≤ 0 olmasıdır: T'nin *her
girişi ≤ 1'dir ve taşma yapısal olarak imkânsızdır*. İkinci incelik maskenin
yerindedir: üst üçgende S_t − S_i pozitiftir (+513'e kadar ölçülmüştür), bu
yüzden önce exp alıp sonra maskelemek inf·0 = NaN üretir — maske, exp'ten
*önce* toplamsal olarak (−10⁴) uygulanır ve exp altında sıfıra gider. Blok
çıktısı h = T·b + e^{S_t}·h_0 olarak hesaplanır (bloklar arası taşıma terimi
e^{S_t} alttan taşarsa sıfırlanır; gerçek katkı zaten ihmal edilebilir
olduğundan bu zararsızdır). Bu formülasyon, Mamba-2 SSD'nin parçalı (chunked)
hesabının d_state=1 özel hâlidir — §2.1.3'te "graf-tabanlı dışa aktarım için
umut verici ama literatürde kurulmamış" diye işaretlenen köprünün somut
kurulumudur — ve yalnızca `cumsum`, `exp` ve `matmul` gibi her iki hedef yığının
da yerli operatörlerinden oluşur. Sayısal doğrulama hedefi (<10⁻³) rahatça
sağlanmıştır: referansa karşı azami mutlak fark P=64'te 1.7×10⁻⁶, P=128'de
3.1×10⁻⁶'dır. Bu stabilite analizi — naif formun neden çöktüğü, çürüme
matrisinin neden çökemeyeceği ve maskenin exp'e göre konumu — katkının
implementasyon ayrıntısı değil, parçasıdır: aynı yolu deneyecek bir
uygulayıcının karşılaşacağı ilk duvar budur.

**Tablo 4.20 — Üç formun sayısal doğrulaması (referans: üçüncü-parti selective scan; gerçek ağırlık + gerçek aktivasyon)**

| Form | Azami mutlak fark | Not |
|---|---|---|
| seq | 0.0 | bit-eşdeğer |
| blocked, naif form (P=64) | **NaN** | blok-içi min S = −513.6; e^{−S} fp32'yi taşırıyor |
| blocked, naif form (P=128) | **NaN** | min S = −982.9; e^{−S} = ∞ |
| blocked, çürüme-matrisi (P=64) | 1.7×10⁻⁶ | hedef <10⁻³ ✓ |
| blocked, çürüme-matrisi (P=128) | 3.1×10⁻⁶ | ✓ |
| ane (P=64 / P=128) | 1.6×10⁻⁶ / 3.2×10⁻⁶ | ✓ |

### Graf katmanındaki sonuç

AS4'ün asıl hedefi, dört katmanlı maliyet modelinin (§4.3.2) (a)/(b)
katmanlarıydı ve sonuç oradadır:

**Tablo 4.21 — Graf yapısı ve yükleme: form karşılaştırması (tek SS2D bloğu, 1×384×32×32, fp32)**

| Form | ONNX düğüm | Transpose / Reshape | ORT yükleme | CoreML dönüşüm | CoreML ALL yükleme |
|---|---|---|---|---|---|
| seq | 7 285 | 4 / 11 | 0.99 s | ✓ 47.4 s | 16.2 s |
| blocked128 | **357 (20×↓)** | 4 / 14 | **0.04 s (25×↓)** | ✓ 16.4 s | 15.9 s |
| blocked64 | 599 | 4 / 14 | 0.01 s | ✓ 31.0 s | 30.2 s |
| blocked32 | 1 079 | 4 / 14 | 0.02 s | ✓ 58.8 s | 57.6 s |
| ane128 | 377 | **2** / 34 | 0.01 s | ✓ 16.2 s | 15.9 s |

Blok formu, graf patlamasını hedeflendiği gibi kırmaktadır. seq formunun 7 285
düğümü — L=1 024 adımın tam unroll'u, adım başına ~7 düğüm — blocked128'de
357'ye inmektedir (**20× küçülme**); düğüm sayısı artık L ile değil, blok
sayısı L/P ile ölçeklenmektedir (P=128 → 8 blok, P=64 → 16, P=32 → 32; grafın
`MatMul` sayısı birebir bu değerlerdir). ORT oturum yüklemesi 0.99 saniyeden
0.04 saniyeye inmiştir (**25× iyileşme**) — §4.3.2'de yükleme maliyetinin düğüm
sayısına göre süperlineer büyüdüğü gösterilmişti; düğüm sayısını kökten küçülten
bir müdahalenin yükleme kazancının düğüm oranını aşması bu mekanizmayla
tutarlıdır. Ve tam modelde hiç açılamayan kapı burada açılmıştır: **çürüme-matrisi
formu CoreML'e 16 saniyede dönüşmektedir** — tam VMamba'nın 574.8 saniye sonra
`TypeError` ile düştüğü (Tablo 4.10) yolda. Katman diliyle: yeniden formülasyon,
nicemlemenin dokunamadığı (§4.4.4) katman (a) ve (b)'ye doğrudan dokunmakta ve
blok ölçeğinde ikisini de kazanmaktadır.

### Mekanizma bulgusu: engel operatör değil, ölçek

Ölçüm matrisinin en değerli satırı ise planlanmamış olandı. Kontrol koşulu
olarak matrise dahil edilen **seq formu — yani tam modelde CoreML'i düşüren
unroll'lu taramanın kendisi — tek-blok ölçekte CoreML'e sorunsuz dönüşmüş**
(47 s), üstelik ANE'de tüm hücrelerin en hızlısı çıkmıştır: medyan **6.7 ms**,
compute-plan atamasında operatörlerin **%99.9'u** NeuralEngine'de (4 138/4 141).
Yani ANE, seçici taramanın unroll edilmiş ilkel operatörlerini (çarp-topla
zincirleri) yürütmekte hiçbir güçlük çekmemektedir; aynı operatör dizisinin 22
blok × 4 yön × L=16 384'lük birleşimi ise dönüşüm katmanından hiç geçememektedir.

Bu gözlem, Bölüm 4.3'ün "SSM'in engeli yapısaldır" bulgusundaki *yapısal*
sözcüğünü mekanizma düzeyinde netleştirir. İki aday açıklama vardı: (i) tarama,
graf hedeflerinin operatör kümesiyle *uyumsuzdur* — hangi ölçekte olursa olsun
dönüşemez; (ii) tarama operatör olarak uyumludur ama unroll'un ürettiği graf
*ölçeği*, dönüşüm/yükleme altyapısının süperlineer maliyet eğrisini aşılamaz
bölgeye taşır. Tek-blok matrisi (i)'i elemektedir: 7 285 düğümlük unroll hem
ONNX hem CoreML'den geçmekte, hem de ANE'ye neredeyse eksiksiz atanmaktadır.
Engel, düğüm sayısı 10⁴ mertebesinden 10⁵–10⁶ mertebesine büyürken bir yerde
kapanan bir ölçek penceresidir — §4.3.2'nin mikrobenchmark serisi (17 671 →
91 363 düğümde yüklemenin 17.9× büyümesi) ve ResNet-50/mikro-SSM dönüşüm
karşıtlığı (~2 000× fark) aynı eğrinin ara noktalarıdır. Dört katmanlı maliyet
modelinin diliyle bulgu şöyle sıkılaşır: **katman (a)/(b) maliyeti operatör
türünün değil, graf kardinalitesinin fonksiyonudur; SSM'i cezalandıran şey
taramanın *ne olduğu* değil, graf temsiline *kaç düğüm olarak* açıldığıdır.**
Bu, yeniden formülasyonun neden işe yaradığını da açıklar: çürüme-matrisi formu
taramanın matematiğini değil, düğüm sayısının L ile ölçeklenme *biçimini*
değiştirmektedir (O(L) → O(L/P)) — ve değişmesi gereken tek şeyin bu olduğu
ortaya çıkmıştır.

## 4.5.2 ANE-Dostu Tensör Düzeni Uyarlaması

İkinci müdahale, Apple'ın "Deploying Transformers on the Apple Neural Engine"
reçetesinin blok formuna uygulanmasıydı: tüm ara tensörlerin (B, C, 1, S) /
(B, C, nb, P) 4D düzeninde tutulması, projeksiyonların 1×1 `conv2d` olarak
ifadesi, LayerNorm'un permute'suz kanal-ekseni hesabı ve transpose sayısının
asgariye indirilmesi. Düzen hedefine ulaşılmıştır: gerçek transpose sayısı 4'ten
2'ye inmiştir (sütun-yönlü taramanın girişte ve birleştirmede zorunlu kıldığı
ikili). Sonuç ise dürüstçe raporlanması gereken bir **negatiftir: ölçülebilir
kazanç yoktur.** Transpose'dan kazanılan, Reshape/Slice tarafında geri
ödenmiştir (P=128'de reshape 14 → 34; P=32'de 106'ya kadar); düğüm toplamı
hafifçe *artmış* (357 → 377), ORT gecikmesi gürültü bandında kalmış (198.3'e
karşı 197.6 ms), CoreML ALL gecikmesi ise blocked karşılıklarından bir miktar
*kötüleşmiştir* (P=128'de 196 → 205 ms; P=32'de 40 → 66 ms) ve ANE atama yüzdesi
düşmüştür (97.7 → 95.2).

Negatifin mekanizması Tablo 4.22'nin ANE sütunlarında okunabilmektedir ve
reçetenin kendisinden çok reçetenin *varsayımlarıyla* ilgilidir. Apple reçetesi,
büyük yoğun matris çarpımlarının (transformer projeksiyon/attention
matrisleri) ANE'ye kesintisiz akmasını optimize eder. Blok formunun hesap
çekirdeği ise farklı bir desendir: P×P'lik (32–128) *küçük, maskeli* matris
çarpımları ve bloklar arası skaler taşıma. Compute-plan bu deseni %97–99
oranında ANE'ye atamaktadır — atama sorunu yoktur — ama uçtan-uca gecikme her
blocked/ane formunda GPU hücresinin (CPU+GPU) gerisindedir: blocked32'de GPU
6.6 ms iken ANE 40.0 ms'dir. Yani **ANE maskeli-küçük-matmul desenini kabul
etmekte ama sevmemektedir**; atama yüzdesi ile yürütme verimi arasındaki bu
makas, §4.3.4'te ViT için gözlenen "atanamıyor" durumundan farklı, daha ince
bir uyumsuzluk türüdür. Tensör düzeni uyarlaması bu tabloyu değiştirememektedir,
çünkü darboğaz düzen değil desendir. Bulgunun pratik değeri sınır çizmesidir:
ANE reçetesi, buradaki gibi zaten-4D, zaten-conv-tabanlı ve küçük-matmul'lu bir
operatörde uygulanabilir ancak kayda değer getirisi yoktur; maliyetli olan
transpose değilse, transpose'u azaltmak kazandırmaz.

**Tablo 4.22 — Çıkarım gecikmesi matrisi (tek SS2D bloğu, medyan ms; ANE op%: compute-plan ataması)**

| Form | Torch CPU | Torch MPS | ORT CPU | CoreML CPU+GPU | CoreML ALL (ANE) | ANE op% |
|---|---|---|---|---|---|---|
| seq | 59.6 | 16.7 | **47.3** | 16.8 | **6.7** | **99.9** |
| blocked32 | 123.0 | 60.1 | 91.1 | **6.6** | 40.0 | 99.3 |
| blocked64 | 175.0 | 112.4 | 126.9 | 10.6 | 76.6 | 98.7 |
| blocked128 | 217.7 | 221.3 | 198.3 | 17.1 | 196.3 | 97.7 |
| ane32 | 119.1 | 57.7 | 75.7 | 6.6 | 66.4 | 98.4 |
| ane64 | 175.0 | 112.6 | 124.1 | 10.5 | 83.2 | 97.1 |
| ane128 | 216.4 | 221.2 | 197.6 | 17.6 | 205.1 | 95.2 |

> **Şekil 4.7 — Form karşılaştırması: graf boyutu, yükleme ve gecikme** *(yer
> tutucu; veri tamamlandı — Tablo 4.21–4.22, `reform_matrix.jsonl`; şekil bu
> kayıtlardan üretilecektir)*. Üç panelli çubuk grafiği: (üst) ONNX düğüm
> sayısı (log ekseni) — seq'in 7 285'ine karşı blok formlarının 357–1 079
> bandı; (orta) ORT ve CoreML-ALL yükleme süreleri; (alt) yürütücü-başına
> medyan gecikme (ORT, CoreML CPU+GPU, CoreML ALL). Beklenen okuma: blok
> formlarının graf/yükleme panellerindeki ezici kazancına karşılık gecikme
> panelinde seq-ANE'nin (6.7 ms) ve blocked32-GPU'nun (6.6 ms) öne çıkışı —
> kazancın katmanlara asimetrik dağılımı.

## 4.5.3 Kazanç-Doğruluk Ödünleşimi

Yeniden formülasyonun faturası üç kalemde toplanmaktadır ve üçü de tablolardan
çıplak okunabilir durumdadır.

**Birinci kalem: FLOPs.** Çürüme-matrisi, blok başına P×P'lik bir matris kurup
çarptığından aritmetik maliyeti O(L)'den **O(L·P)'ye** çıkarır — graf
katmanındaki kazanç, hesap katmanından borç alınarak finanse edilmektedir.
Ölçüm bunu birebir yansıtmaktadır: ORT gecikmesi P ile monoton büyür (seq 47.3
→ blocked32 91.1 → blocked128 198.3 ms) ve eager PyTorch'ta da aynı desen
görülür. Yani katman (d), CPU yürütücülerde blok formunun *kaybettiği*
katmandır; kazancın tamamı katman (a)/(b)/(c′)'dedir. Bu asimetri tezin ana
anlatısıyla tutarlıdır — engel zaten katman (d)'de değildi (§4.3.2) — ama
ödünleşimin varlığı, "blok formu her eksende üstündür" gibi bir okumayı baştan
yasaklar.

**İkinci kalem: P seçimi.** P, düğüm sayısı (∝ L/P) ile FLOPs (∝ L·P) arasında
bir ayar düğmesidir ve iki uç da cezalıdır. Eldeki matriste **P=32 tatlı
noktadır**: 1 079 düğümlük grafı hâlâ seq'ten 6.8× küçüktür, ORT yüklemesi
0.02 saniyedir ve CoreML CPU+GPU hücresinde 6.6 ms ile *tüm matrisin en hızlı
GPU sonucunu* vermektedir. Buna karşılık blok sayısı arttıkça ANE derleme
maliyeti büyümektedir (CoreML dönüşümü blocked128'de 16 s iken blocked32'de
59 s) — P küçüldükçe graf, katman (a)'yı yeniden şişirmeye başlar. Ödünleşim
kaçınılmaz değil, ayarlanabilirdir; ama serbest öğle yemeği yoktur.

**Üçüncü kalem: sayısal sadakat.** fp32'de formlar arası fark 10⁻⁶
mertebesindeyken, CoreML'in fp16 yürütmesinde referansa karşı azami mutlak fark
0.007–0.07 bandına çıkmaktadır (CPU+GPU hücrelerinde 0.007–0.023, ANE
hücrelerinde 0.064–0.071; seq dahil tüm formlarda benzer). Tek blok için bu
düzey, W8'in sıfır-kayıp bulgusu (§4.4.1) ışığında muhtemelen zararsızdır;
ancak **22 bloğun ardışık birikiminin mIoU'ya etkisi bu prototipte
ölçülmemiştir** ve tam-model iddiası ancak o ölçümle kurulabilir — bu, bölümün
bilinçli sınırıdır.

Son olarak tam ölçeğe taşıma riski açıkça kaydedilmelidir. Prototipin L'si
1 024'tür (32×32 ara-aktivasyon); VMamba-T'nin stage-0'ı 512² girdide
L=16 384 ile çalışır. P=32'de bu, blok-taşıma döngüsünün **512 kez** unroll
olması demektir — düğüm sayısı O(L/P) ölçeklenmesiyle yeniden on binler
mertebesine, yani tam da §4.5.1'de tanımlanan ölçek penceresinin riskli
bölgesine yaklaşır. Muhtemel çıkış yolları bellidir — bloklar-arası taşımanın
kendisine ikinci bir kapalı-form katmanı (hiyerarşik/iki-seviyeli parçalama),
katmana göre değişen P, ya da uzun-L katmanlarında daha büyük blok — ama
bunların hiçbiri bu tezde ölçülmemiştir ve prototip, tam-ölçek genellemesini
bilinçli olarak gelecek çalışmaya bırakmaktadır. Bu sınırlama savunmacı bir
çekince değil, kapsam kararının simetrik yüzüdür: tek blokta *atfedilebilir*
kanıt üretmek, tam modelde *atfedilemez* bir sonuç üretmeye tercih edilmiştir.

## 4.5.4 Ara Özet: AS4'ün Cevabı

AS4 — "parçalı/sabit-uzunluklu tarama formülasyonları dağıtım engelini açabilir
mi?" — iki parçalı bir cevap almıştır.

**Birinci parça: kısmi evet.** SSD'nin d_state=1 özel hâli olan çürüme-matrisi
formu, sayısal olarak stabil (naif kapalı formun aksine), referansa 10⁻⁶
sadakatinde ve yalnızca yerli operatörlerden oluşan bir tarama vermektedir; tek
blok ölçeğinde graf 20× küçülmüş, ORT yüklemesi 25× hızlanmış ve tam modelde
kapalı olan CoreML kapısı açılmıştır. Engelin bulunduğu katmanlarda — (a) ve
(b) — müdahale çalışmaktadır. "Kısmi"nin gerekçeleri de aynı netlikte
ortadadır: FLOPs O(L·P)'ye çıkmakta (CPU'da katman (d) kaybı), ANE düzen
reçetesi kazanç getirmemekte, fp16 birikimi ve tam-ölçek davranışı ölçülmemiş
durumdadır. Tez planının "kazanç yoksa dürüst yaz" ilkesi bu bölümde iki kez
uygulanmıştır: naif formun NaN'ı ve ANE reçetesinin sıfır getirisi, sonuç
tablosunun eşit haklı satırlarıdır.

**İkinci parça — ve bölümün tezin bütününe asıl katkısı — mekanizma
aydınlatmasıdır.** Kontrol koşulu olarak matrise giren unroll'lu seq formunun
tek-blok ölçekte hem CoreML'e dönüşmesi hem de ANE'de en hızlı hücre olması
(6.7 ms, %99.9 atama), Bölüm 4.3'ün "yapısal engel" bulgusundaki mekanizmayı
teşhis etmiştir: engel taramanın operatör kümesinde değil, graf temsilinin
**ölçeğindedir** — *yapı = ölçek*. Bu teşhis iki yöne birden iş görür. Geriye
doğru, ana matrisin bulgularını tek bir eksene indirger: dönüşüm süresi,
yükleme süresi ve bellek duvarı, aynı süperlineer ölçek eğrisinin üç
görünümüdür. İleriye doğru, çözümün adresini belirler: SSM'i uca taşımak
isteyen bir uygulayıcının problemi operatör desteği beklemek değil, graf
kardinalitesini yönetmektir — ve bu tezin prototipi, bunun matematiği bozmadan
mümkün olduğunu tek blok ölçeğinde göstermiştir. Nicemleme bölümünün kapanışta
işaret ettiği dolaylı umut (§4.4.4) de bu bulguyla somutlaşmaktadır: kapı graf
ölçeğinde açılabilirse, ardında ANE'nin taramayı hızlı yürüttüğü ölçülmüş bir
gerçektir.

### Kapsam sınırları

Bu bölümün iddiaları şu sınırlar içinde okunmalıdır:

1. **Tek blok, tek girdi.** Tüm ölçümler tek SS2D bloğu (`layers.2.blocks.0.op`)
   ve tek bir gerçek ADE20K ara-aktivasyonu (1×384×32×32, L=1 024) üzerindedir;
   tam-model gecikme/yükleme kazancı bu sayılardan *çıkarsanamaz*, yalnızca
   mekanizma düzeyinde öngörülebilir.
2. **Doğruluk tensör-düzeyindedir.** Sadakat, referans çıktıya azami mutlak
   fark olarak ölçülmüştür; uçtan-uca mIoU etkisi — özellikle 22 bloğun fp16
   birikimi (tek blokta 0.007–0.07) — ölçülmemiştir.
3. **Tek dizi uzunluğu.** L=1 024 (stage-2 çözünürlüğü) test edilmiştir;
   stage-0'ın L=16 384'ü (P=32'de 512 blok) denenmemiştir ve §4.5.3'te
   tartışılan ölçekleme riski açık bir sorudur.
4. **ANE ölçümlerinin vekilleri.** ANE davranışı, compute-plan atama yüzdesi ve
   ALL/CPU+GPU hücre farkı üzerinden okunmaktadır; birim-içi profilleme (ANE
   sayaçları) mevcut araçlarla yapılamamaktadır.
5. **Tek donanım ve sürüm kümesi.** Bulgular tek makine (Apple M5, 24 GB) ve
   `reform_matrix.jsonl` içinde kayıtlı sürümlerle sınırlıdır; ölçek
   penceresinin sınırları (hangi düğüm sayısında kapandığı) araç zinciri
   sürümüne bağlı olabilir.

---

*Sayfa hedefi: ~6–7. Faz 4 prototip matrisi kapanmıştır (TASK-030..032).
Kalan iş: Şekil 4.7'nin üretilmesi; tam-ölçek parçalama (hiyerarşik taşıma)
gelecek çalışma olarak Bölüm 6.3'e taşınacaktır.*


---

# 5. TARTIŞMA

*(TASLAK v1 — 14 Ağustos 2026, TASK-034)*

> Bu bölüm, Bölüm 4'te raporlanan ölçümlerin yorumunu yapar. Bölüm 4'ün disiplini
> "sayıyı ver, hükmü verme" idi; bu bölümün işi hükümdür — ancak her hüküm,
> Bölüm 4'ün belirli bir tablosuna veya sayısına bağlanarak verilecektir. Bölüm
> 2.6'da tarif edilen literatür boşluğuna da burada dönülmekte ve tezin o boşluğu
> hangi kanıtlarla doldurduğu tartışılmaktadır. Zamansal çerçeve baştan
> sabitlenmelidir: buradaki tüm yorumlar, 12–14 Ağustos 2026 tarihlerinde, tek
> bir sürüm kümesiyle (torch 2.13.0, onnxruntime 1.28.0, coremltools 9.0,
> macOS 26.5.2, Python 3.12.13) ve tek bir çip (baz Apple M5, 24 GB) üzerinde
> alınmış ölçümlere dayanır. Bu bulgular **zaman damgalı bulgulardır**; hangi
> kısmının konjonktürel, hangi kısmının kalıcı olduğu §5.5'in konusudur.

## 5.1 Bulguların Yorumu: Mimari Verimlilik Bir Donanım-Yazılım Eşleşmesi Problemidir

Bu tezin ölçüm matrisinden çıkan en genel ders tek cümleye sığar: **"verimli
mimari hangisidir?" sorusu, "hangi dağıtım yığınında?" eki olmadan iyi tanımlı
bir soru değildir.** Matristeki hemen her satır bu cümlenin bir örneğidir.
Aynı ConvNeXt-T + UPerNet modeli, aynı makinede, aynı 512² girdiyle, yalnızca
yürütme yolu değiştirilerek 63.9 ms (CoreML CPU+GPU, Tablo 4.8) ile 1 670 ms
(torch.compile inductor-CPU, aynı tablo) arasında **26 kat** salınmaktadır.
Aynı VMamba-T, çıkarım medyanı ölçüldüğünde 618 ms ile klasiklerin ORT
hücrelerinden bile hızlıyken (ORT paradoksu, §4.3.1), aynı modelin CoreML'e
dönüşümü hiç mümkün değildir (Tablo 4.7) ve ORT hücresine giriş bileti her
süreç başlatımında ~12 dakikadır (Tablo 4.9). En uç örnek tarama operatörünün
kendisidir: aynı unroll'lu seçici tarama matematiği, tek blok ölçeğinde ANE
üzerinde 6.7 ms'de ve operatörlerin %99.9'u NeuralEngine'e atanmış hâlde
koşarken (Tablo 4.22), 22 blok × 4 yönlük tam-model birleşimi dönüşüm
katmanından hiç geçememektedir (Tablo 4.10). "Bu mimari verimli midir?"
sorusunun cevabı, aynı silikon üzerinde, milisaniyelerle imkânsızlık arasında
gezinmektedir.

Bu gözlemin kuramsal karşılığı, Bölüm 2.1'de kurulan ve Bölüm 4'ün doğruladığı
önermedir: verimlilik mimarinin içkin bir özelliği değil, **mimari ile
gerçekleştirim yığını arasındaki eşleşmenin** özelliğidir. Mamba ailesinin
literatürdeki hız iddiaları (§2.6'daki kaynak dökümü: Vim'in 2.8× iddiası,
EfficientViM'in milisaniye-altı değerleri) yanlış değildir; ancak bu iddialar,
mimarinin *bir* eşleşmesinin — el yazması CUDA çekirdeği + NVIDIA GPU —
ölçümleridir ve o eşleşmeden çıkarıldığında taşınmamaktadır. Bu tezin katkısı
tam bu taşınmama olayını nicelleştirmektir: eager CPU'da 3.6–3.8× olan
VMamba/klasik farkı (Tablo 4.2), her omurganın kendi dağıtılabilir-en-iyi
hücresi karşılaştırıldığında 16×'e (63.9'a karşı 1 008 ms, §4.3.1), enerji
ekseninde ~25×'e (458'e karşı 11 720 mJ/çıkarım, Tablo 4.14) açılmaktadır.
Uçurumun büyüme mekanizması dikkat çekicidir ve yanlış anlaşılmaya açıktır:
VMamba yığın merdiveninde *yavaşlamamakta*, klasiklerin tırmandığı merdivene
*binememektedir*. Klasikler için eager → ORT → CoreML → ANE zinciri her
basamakta ya hız ya enerji kazandırırken (ConvNeXt: 571 → 63.9 ms; 4 268 →
458 mJ), VMamba merdivenin ilk basamağında kalmaktadır.

Eşleşme probleminin en keskin kanıtı, üç mimari ailesinin ANE karşısındaki
"mimari başına bir kademe" tablosudur (Tablo 4.15): CNN'in 353 operasyonunun
tamamı ANE'ye atanmakta (%100), Transformer'ın 631 operasyonundan hiçbiri
atanmamakta (%0 — sessiz GPU fallback), SSM için rapor dahi üretilememektedir.
Aynı görevi aynı doğruluk bandında (mIoU 44.3–48.3, §4.2.5) çözen üç model,
aynı çipin en verimli biriminden üç ayrı muamele görmektedir. Bu tablo tek
başına, mimari karşılaştırmalarını donanım-yazılım bağlamından soyutlayan her
analizin — FLOPs tablolarının, parametre sayımlarının, tek-yığın
benchmark'larının — neden eksik kalacağının gösterimidir.

Eşleşmenin bir de *karışım* boyutu vardır ve mikrobenchmark ile tam model
arasındaki görünür çelişki bunu öğretmiştir. İzole taramada MPS her iki dizi
uzunluğunda CPU'dan yavaşken (Tablo 4.3), tam VMamba-T MPS'te CPU'dan 2.0×
hızlıdır (Tablo 4.2); §4.2.2'nin gösterdiği gibi bu, aynı mekanizmanın iki
karışım oranındaki tezahürüdür — tarama cezası, paralel-dostu bütünün içinde
seyreltilmiş olarak yansır. Bunun tartışma düzeyindeki önemi şudur: bir
modelin yığın uyumu, operatörlerinin uyumlarının *ağırlıklı* bileşkesidir ve
uyumsuz azınlık payı, toplam davranışı iki farklı biçimde belirleyebilir.
Yürütme katmanında azınlık payı toplamı yalnızca *frenler* (MPS kazancı
3.8×'ten 2.0×'e iner); temsil/dönüşüm katmanında ise azınlık payı grafa
hâkim olur ve toplamı *diskalifiye eder* (390 758 düğümün ezici çoğunluğu
taramadan gelir, Tablo 4.12). Aynı mimari kusurun katmana göre "sürtünme" ya
da "duvar" olarak tezahür etmesi, verimlilik analizinin neden katman katman
yapılmak zorunda olduğunun bir başka gösterimidir.

Eşleşme çerçevesi, tezin en beklenmedik iki bulgusunu da tek çatı altında
açıklar. Birincisi ORT paradoksudur: 390 758 düğümlük patolojik graf, ORT'nin
graf optimizer'ı tarafından oturum yüklemesi sırasında eritilmekte ve geriye
klasiklere benzeyen, %81.7'si evrişimlerde geçen bir yürütme grafı kalmaktadır
(Tablo 4.13). Maliyet yok olmamış, katman değiştirmiştir: çıkarımdan (d)
yüklemeye (b). İkincisi, yeniden formülasyon deneyinin mekanizma bulgusudur
(§4.5.1): engel taramanın operatör kümesinde değil, graf temsilinin
ölçeğindedir — *yapı = ölçek*. İki bulgu birlikte okunduğunda, SSM'in uç engeli
hakkındaki naif anlatıların ikisi de çürümektedir. "SSM'ler uçta yavaş çalışır"
anlatısı yanlıştır: dönüşümü başarabilen tarama, ANE'de matrisin en hızlı
hücresini vermektedir (6.7 ms) ve tam modelin ORT çıkarımı eager'ının
0.30×'udur. "SSM operatörleri graf hedefleriyle uyumsuzdur" anlatısı da
yanlıştır: aynı operatör dizisi küçük ölçekte üç hedeften de geçmektedir.
Doğru anlatı şudur: **ardışık taramanın graf temsiline L ile lineer büyüyen
kardinalitede açılması, araç zincirlerinin süperlineer maliyet eğrilerini
(yükleme: 4× düğüm → 15.7× süre, Tablo 4.11) aşılamaz bölgeye taşımaktadır.**
Sorun aritmetikte değil, temsilin ölçeğindedir; bu yüzden nicemleme gibi
sayısal araçlar engele dokunamazken (§4.4.4: boyut 0.83×, yükleme değişmedi)
graf kardinalitesini değiştiren blok-form dokunabilmektedir (§4.5.1: düğüm
20×↓, yükleme 25×↓, CoreML kapısı açık).

Bu noktada Bölüm 2.6'da tarif edilen boşluğa dönmek gerekir. Orada tezin
konumu tek cümleyle verilmişti: literatür SSM görü omurgalarının özel CUDA
çekirdekleriyle *ne kadar hızlı olabildiğini* ölçmüştür; bu tez, aynı
omurgaların gerçek dağıtım yığınlarında *ne kadar hızlı olduğunu* ölçer.
Ölçüm tamamlandığına göre iki soru cevaplanabilir durumdadır. Birincisi:
boşluk gerçek miydi? Evet — ve tahmin edilenden derindi. §2.6'nın dağınık
mühendislik kanıtı (ORT sorun kaydındaki 17× yavaşlama, M3 üzerinde) bu tezin
kontrollü matrisinde yalnızca doğrulanmakla kalmamış, tek bir semptomun —
çıkarım yavaşlığının — aslında dört katmanlı bir maliyet yapısının en hafif
yüzü olduğu ortaya çıkmıştır: aynı sorun kaydında yan not olarak geçen paket
boyutu ve yükleme süresi, bizim matrisimizde asıl hikâye çıkmıştır (Tablo
4.9). İkincisi: "bildirilen vs gerçekleşen" karşıtlığı ne verdi? Literatürün
bildirdiği CUDA sayıları ile bu tezin ölçtüğü Apple sayıları arasındaki fark,
iki donanımın hız farkına indirgenemez; fark *kategoriktir*. CUDA
ekosisteminde SSM bir "hızlı model"dir; ölçülen genel amaçlı yığınlarda ise
hızlı ya da yavaş olmaktan önce, çoğu hücrede **var olamayan** bir modeldir
(Tablo 4.7'deki ❌ işaretleri). Verimlilik tartışmasının birimi bile yığına
göre değişmektedir: CUDA tarafında milisaniye, bu tezin matrisinde
dönüşebilirlik. Bu kategorik fark, mimari verimliliğin eşleşme problemi
olduğu iddiasının en güçlü hâlidir — eşleşme yalnızca hız katsayısını değil,
modelin dağıtım evreninde var olup olmadığını belirlemektedir.

Son olarak bu çerçeve, doğruluk-verimlilik düzlemindeki resmi de netleştirir.
VMamba-T bu deney kümesinin en doğru modelidir (+2.9–4.0 mIoU, §4.2.5) ve bu
avantaj gerçektir. Eager düzlemde bu, savunulabilir bir Pareto noktasıdır:
+4 mIoU için 6.6× gecikme ve 4–5× enerji ödemeye razı uygulamalar vardır.
Ancak dağıtım yığınları düzleme eklendikçe VMamba'nın noktaları sınırdan
uzaklaşmakta, CoreML/ANE bölgesinde ise hiç var olamamaktadır. Mimari
verimlilik bir eşleşme problemi olduğu için, Pareto düzlemi de yığın-bağımsız
çizilememektedir: her yığın kendi Pareto sınırını üretir ve SSM bugün bu
sınırların en değerlisinde (ANE) temsil edilmemektedir.

## 5.2 FLOPs ve Parametre Sayısının Yetersizliği: Bir Raporlama Önerisi

Bölüm 2.6, literatürün verimlilik kanıtlarının üç kaynağını dökmüş ve
tamamının tek katmanı — çıkarım gecikmesini — raporladığını tespit etmişti.
Bu tezin ölçümleri, bu raporlama pratiğinin yalnızca eksik değil, **sistematik
olarak yanıltıcı** olabildiğini göstermektedir; çünkü ölçülen vakada maliyetin
baskın kısmı, raporlanan katmanın dışındadır.

Kanıtın çekirdeği Tablo 4.9'dur. VMamba-T'nin ONNX yolunda, literatürün
raporladığı tek katman olan çıkarım (d) neredeyse sorunsuzdur — ilk koşu
klasiklerin 1.3 katı, medyan ise eager'dan hızlıdır. Buna karşılık dönüşüm
süresi (a) 448×, yükleme süresi (b) ~7 249× şişmiştir ve paket boyutunun (c)
858.4 MB'lık toplamının 614 MB'ı ağırlık değil, serileştirilmiş graf
yapısıdır. FLOPs analizi bu üç katmanın üçüne de yapısal olarak kördür, çünkü
FLOPs yalnızca çıkarımın aritmetik iş yükünü sayar; grafın kaç düğüme
açıldığını, o düğümlerin her oturum açılışında yeniden ayrıştırılacağını,
dışa aktarma sürecinin 1024²'de fiziksel belleği ~2.7 kat aşarak duvara
çarpacağını (Tablo 4.11) modelleyemez. Aynı körlük parametre sayısı için de
geçerlidir ve kontrol deneyi bunu çıplak biçimde göstermiştir: 25.6M
parametrelik ResNet-50'nin CoreML dönüşümü 2.5 saniye sürerken 400K
parametrelik SSM mikro-modelininki 5 668 saniye sürmüştür (§4.3.2) — ~2 000×
fark, parametre sayısının *tersi* yönünde. Belirleyici değişken model boyutu
değil, graf kardinalitesidir.

Yanıltıcılığın ikinci yüzü, tek-katman raporlamanın *iyimser* hata yapmakla
kalmayıp yön de şaşırtabilmesidir. ORT paradoksu bunun örneğidir: yalnızca
çıkarım medyanı raporlayan bir benchmark, VMamba'nın ORT hücresini "618 ms —
klasiklerden hızlı" diye özetler ve teknik olarak doğru olan bu özet, pratikte
tam yanlış karara götürür; çünkü o hücrenin görünmeyen bedeli her süreç
başlatımında ödenen 620–725 saniyelik yüklemedir (§4.3.1–4.3.2). Ters yönlü
örnek nicemleme matrisindedir: "INT8 = küçük ve hızlı" varsayımıyla rapor
edilen boyut kazancı (0.25×) gerçektir, ama aynı paketin ORT-CPU gecikmesi
10–17× *kötüleşmiştir* (Tablo 4.19). Tek metrikli raporlama, bu iki vakanın
ikisinde de uygulayıcıyı yanlış hücreye yönlendirir.

FLOPs ile ölçülen gecikme arasındaki sapmanın kendisi de tek boyutlu
değildir ve bunun ayrıştırılması metrik tartışmasına netlik katar. Çıkarım
katmanı içinde kalındığında bile FLOPs, yürütme modeliyle etkileşimi
görmez: MiniMamba mikrobenchmark'ında aynı FLOPs bütçesi MPS'te CPU'dan
yavaş koşmuştur (Tablo 4.3) — aritmetik iş yükü aynıyken, ardışık bağımlılık
zinciri paralel donanımın çekirdek-başlatma ekonomisini cezalandırmaktadır.
Çözünürlük ekseninde FLOPs'un öngördüğü asimptotik davranış (O(L) eğim)
gerçekleşmiş, ama FLOPs'un *görmediği* sabit katsayı belirleyici olmuştur
(Tablo 4.6: makas kapanmıyor, 6–9× bant). Yani FLOPs, kendi ev sahasında —
çıkarım aritmetiğinde — bile ancak eğimi tahmin edebilmekte, katsayıyı ve
yürütme modelini kaçırmaktadır; çıkarım dışındaki üç katmanda ise hiçbir
öngörü gücü yoktur. Parametre sayısı için tablo daha da kötüdür: aynı
parametre bütçesi, graf temsiline göre 843 düğüm de (ConvNeXt) 390 758 düğüm
de (VMamba) üretebilmektedir — iki model arasındaki 463× düğüm farkı,
parametre sayılarına bakarak sıfır olarak tahmin edilirdi.

Bu gözlemlerden, alanın raporlama pratiğine somut bir öneri çıkmaktadır ve bu
tez önerinin hem tanımını hem doldurulmuş bir örneğini sunmaktadır: verimlilik
iddiaları **dört katmanlı maliyet modeliyle** raporlanmalıdır — **(a)**
dönüşüm/derleme süresi, **(b)** model yükleme süresi, **(c)** dağıtım paketi
boyutu (ağırlık/yapı ayrıştırmasıyla) ve **(d)** çıkarım gecikmesi. Model
literatürle karşılaştırılabilirliği (d) üzerinden korur; (a)–(c) ise uç
senaryonun fiilî eleme kriterlerini görünür kılar. Pratik biçimi mütevazıdır:
bir verimlilik tablosuna üç sütun eklemek (dönüşüm süresi, yükleme süresi,
paket boyutu) ve hedef yığını/sürümleri tabloya yazmak. Bu tezin deneyimi,
üç sütunun maliyetinin de düşük olduğunu göstermektedir — dönüşüm ve yükleme
süreleri, dışa aktarma betiğinin zaten ürettiği yan ürünlerdir; ölçülmemeleri
zorluktan değil, alışkanlıktan kaynaklanmaktadır. Bu tezin verisinde dört
katmanın dördü de en az bir vakada belirleyici olmuştur: (a) CoreML
dönüşüm başarısızlığı ve 1024² bellek duvarı; (b) 12 dakikalık ORT yüklemesi
ve Swin'in ANE reddiyle 40× büyüyen CoreML yüklemesi (Tablo 4.15); (c)
nicemlemenin dokunamadığı 614 MB'lık yapı payı; (d) W8'in ANE'de %29 kazancı
ile ORT dinamik INT8'in 17× kaybı. Katmanların bağımsız olmadığı da
not edilmelidir — (c′) graf kardinalitesi (b)'nin nedenidir (§4.3.2), (a)
ile (b) aynı süperlineer eğrinin iki kesitidir (§4.5.1) — ancak raporlama
açısından önemli olan ayrıştırılmaları değil, *görünür* olmalarıdır. FLOPs
ve parametre sayısı bu çerçevede tamamen değersiz değildir; (d) katmanının
kaba bir ön-eleme vekili olarak iş görmeye devam edebilirler. Değersiz olan,
onları dağıtım kararının *yeterli* girdisi saymaktır: bu tezin vakasında
FLOPs-optimal aday, dört katmanın üçünde diskalifiye olmuştur.

## 5.3 Uygulayıcılar İçin Mimari Seçim Kılavuzu

Bu bölüm, Bölüm 4'ün matrisini bir karar tablosuna indirger. Kapsam uyarısı
baştan: tablo, ölçülen evren içinde geçerlidir — Apple Silicon (baz M5),
yoğun tahmin (ADE20K semantik segmentasyon), tiny-ölçek omurgalar ve §5.1'de
sabitlenen sürüm kümesi. Bu evrenin dışına genelleme §5.4'ün sınırlarına
tabidir.

**Tablo 5.1 — Apple Silicon'da yoğun tahmin için dağıtım karar tablosu (Ağustos 2026 durumu)**

| Senaryo / öncelik | Öneri | Dayanak (Bölüm 4) |
|---|---|---|
| Enerji ve verim öncelikli uç dağıtım | **CNN omurga + CoreML W8, ANE hedefi (ALL)** | ConvNeXt: 353/353 op ANE (Tablo 4.15); W8 ile 64.7 ms, ±0 mIoU, 59.7 MB (Tablo 4.16); 458 mJ/çıkarım — matrisin en verimli hücresi (Tablo 4.14) |
| En düşük gecikme, GPU kabul edilebilir | **CNN veya Transformer + CoreML CPU+GPU** | 63.9 ms her ikisinde (Tablo 4.8); Swin CPU+GPU 329 mJ ile enerji açısından da güçlü (Tablo 4.14) |
| Transformer omurga şart | **CoreML CPU+GPU; ANE'ye güvenme** | Swin ALL: `ANECCompile FAILED`, 0/631 op ANE, ALL hücresi hem gecikme hem enerjide CPU+GPU'dan kötü (Tablo 4.14–4.15) |
| SSM doğruluğu şart (+3–4 mIoU değerli) | **torch MPS eager; dağıtım yığını yok** | VMamba tek çalışan pratik hücre: 1 008 ms, 4 844 mJ (Tablo 4.8, 4.14); CoreML ❌, torch.compile ✗, ORT yüklemesi 12 dk |
| SSM'i dışa aktarmak/dağıtmak isteyen | **Blok-form yeniden formülasyon + küçük L; tam model bugün dağıtılamaz** | Blok formu: graf 20×↓, yükleme 25×↓, CoreML kapısı açık (Tablo 4.21); tam-ölçek genellemesi ölçülmemiş (§4.5.3) |
| Yüksek çözünürlük (≥1024²) + SSM | **Bugün mümkün değil (export katmanı)** | 1024² ONNX dışa aktarması bellek duvarı: ~65 GB, tamamlanamadı (Tablo 4.11) |

Tabloya eşlik eden, yığın-seçimi düzeyinde beş operasyonel kural da verilerin
doğrudan sonucudur:

1. **W8 varsayılan dağıtım formatı olmalıdır.** Her iki klasik modelde mIoU
   değişimi ±0, boyut 0.50×, ConvNeXt-ANE'de gecikme −%29 (Tablo 4.16). Bu
   yığında W8'in fp32-export'a karşı kaybettiği tek eksen yoktur.
2. **W4 palettizasyon yalnızca boyut kritikse ve modele bakılarak seçilmelidir.**
   Fatura modele göre −2.8 ile −3.9 mIoU arasındadır ve uzun kuyruklu
   ağırlık/aktivasyon dağılımlarıyla (ConvNeXt depthwise, kurtosis 75'e kadar)
   büyümektedir (Tablo 4.16, 4.18). Ayrıca palet, çözemeyen birimde maliyete
   dönüşür: aynı W4 paketi ALL'da 59.3 ms, CPU+GPU'da 132.4 ms (§4.4.3).
3. **ORT dinamik INT8'i Apple CPU'da kullanmayın.** Boyut 0.25×'e inerken
   gecikme 10–17×'e çıkmaktadır (Tablo 4.19). Nicemleme kararı hedef yığında
   ölçülmeden verilemez.
4. **CoreML'e ORT CoreML EP üzerinden dolaylı erişim, doğrudan CoreML'in
   yerini tutmaz.** Bölümleme parçalanması (Swin'de 94 parça) kazancı geri
   yemekte, EP hücresi saf CoreML'in ~5× yavaşında kalmaktadır (§4.3.1).
5. **torch.compile'ı Apple CPU'da varsayılan iyileştirme sanmayın.** Ölçülen
   üç modelde sonuç nötr (Swin), zararlı (ConvNeXt 2.9× yavaş) veya süreç
   çökmesidir (VMamba) (Tablo 4.8).
6. **SSM'de çözünürlüğü dağıtım katmanının kısıtı olarak planlayın.**
   Çıkarım ~lineer ölçeklenir (Tablo 4.6) ama dışa aktarma maliyeti de L ile
   büyür: 256²'de export 2 dakika / yükleme 46 saniyeyken 512²'de 9 dakika /
   12 dakikadır ve 1024² hiç tamamlanamamaktadır (Tablo 4.11). SSM'i
   dışa aktarma niyeti, bugün ancak küçük çözünürlük + blok-form
   kombinasyonuyla gerçekçidir.

Tablonun soyut kalmaması için iki senaryo somutlaştırılabilir. *Senaryo A —
pilde çalışan cihazda sürekli segmentasyon (ör. erişilebilirlik uygulaması):*
belirleyici eksen enerji/çıkarımdır. Karar tablosunun ilk satırı geçerlidir:
ConvNeXt-sınıfı omurga + W8 + ANE, 458 mJ/çıkarım ile ikinci en iyi seçeneğin
(Swin CPU+GPU, 329 mJ ama daha düşük mIoU bandı ve W8 kazancı yok) yanında,
%29'luk W8 gecikme kazancıyla birlikte en dengeli hücredir; VMamba'nın en iyi
hücresi aynı işte ~10.6× enerji tüketir (Tablo 4.4). *Senaryo B — etkileşimli
düzenleme aracında istek-başına segmentasyon:* belirleyici eksen, soğuk
başlatma dahil uçtan-uca sürelerdir. Burada katman (b) tabloya girer: Swin'in
ANE reddi yüzünden 5.6 saniyelik CoreML yüklemesi (Tablo 4.15) ilk isteğin
gecikmesine eklenir; ConvNeXt 142 ms yüklemeyle bu eksende de öndedir.
VMamba'nın ORT hücresi bu senaryoda tanım gereği elenir — 618 ms'lik çıkarım
medyanının önünde her süreç başlatımında ~12 dakikalık yükleme vardır.
İki senaryonun ortak dersi, kılavuzun tek bir "en iyi model" üretmediğidir:
karar, senaryonun hangi maliyet katmanına duyarlı olduğuyla başlar.

Tablonun SSM satırları bir hükmü açıkça vermektedir ve bu hükmün dürüstçe
yazılması gerekir: **Ağustos 2026 itibarıyla, ölçülen yığın kümesinde, SSM
omurgası Apple Silicon'da "dağıtılabilir" kategorisinde değildir.** Bu hüküm
mimarinin aleyhine bir nitelik yargısı değildir — aynı ölçümler SSM'in en
doğru model olduğunu (§4.2.5), aktivasyon profilinin nicemlemeye en uygun
profil olduğunu (Tablo 4.18) ve taramanın ANE'de hızlı yürüyebildiğini
(Tablo 4.22) göstermektedir. Hüküm, araç zinciri eşleşmesinin bugünkü
durumuna dairdir; değişme koşulları §5.5'te tartışılmaktadır.

## 5.4 Sonuçların Genellenebilirliği ve Sınırlılıklar

Bu tezin iddiaları, deney evreninin sınırları içinde kurulmuştur ve bu
sınırların her biri genellemeyi belirli bir yönde kısıtlar. Aşağıdaki döküm,
Bölüm 4'ün bölüm-içi kapsam notlarını (özellikle §4.4 ve §4.5 sonlarındaki)
tez düzeyinde birleştirir.

**Tek çip, tek NPU ailesi.** Tüm ölçümler tek bir baz Apple M5 (24 GB)
üzerindedir. Plan revizyonuyla (Mac-only, 11 Ağustos 2026) CUDA ekseni deney
matrisinden çıkarılmıştır; NVIDIA tarafındaki verimlilik değerleri bu tezde
kendi ölçümümüz değil, literatürden alınan "bildirilen" değerlerdir ve tezin
karşıtlığı bilinçli olarak "bildirilen (CUDA, literatür) vs gerçekleşen
(Apple, bu tez)" olarak kurulmuştur. Bunun bedeli, karşılaştırmanın bir
ayağının farklı metodolojilerle üretilmiş sayılara dayanmasıdır. ANE
bulguları da tek bir NPU ailesine aittir: "mimari başına bir kademe"
tablosunun (Tablo 4.15) Qualcomm Hexagon, Intel NPU veya Google Edge TPU'da
aynı şekilde kademelenip kademelenmeyeceği bu veriden çıkarılamaz —
söylenebilecek olan, mekanizmanın (statik-graf derleyicisi + ardışık temsil
uyumsuzluğu) bu ailelere de taşınabilir *türden* olduğudur, taşındığı değil.

**Tek ölçek: tiny.** Üç omurga da tiny sınıfındadır (~28–31M omurga
parametresi). Graf patlaması L ile ölçeklendiğinden (Tablo 4.11) ve engel
ölçek-pencereli olduğundan (§4.5.1), daha büyük modellerin *daha kötü*
davranacağı güçlü bir beklentidir; ama bu bir ölçüm değil çıkarımdır. Küçük
yönde ise (mobil-ölçek, EfficientViM sınıfı) pencere içinde kalınabilir ve
tablo değişebilir.

**Tek görev, tek veri kümesi, kısıtlı değerlendirme protokolü.** Görev ADE20K
semantik segmentasyonudur; nicemleme mIoU'ları 250-görüntü, kare-512
protokolüyle ölçülmüştür ve mutlak değerler değil deltalar esastır (§4.4.1).
Cityscapes/1024² ekseni, tam da SSM'in teorik avantaj bölgesi olduğu için
planlanmış, ancak dışa aktarma katmanının 1024²'de duvara çarpması (Tablo
4.11) nedeniyle yalnızca eager kipte taranabilmiştir (Tablo 4.6). Bu, tezin
kendi bulgusunun tezin kapsamını kısıtladığı özel bir durumdur: yüksek
çözünürlük ekseni ölçülemediği için değil, *ölçülemez olduğu ölçüldüğü* için
eksiktir.

**Aktivasyon istatistiği ve nicemleme kapsamı.** AS3 bulguları Aşama I
(kalibrasyonsuz ağırlık-nicemleme) ile sınırlıdır; W8A8 ve kalibrasyonlu
statik nicemleme ölçülmemiştir. VMamba'nın "en ılımlı aktivasyon profili"
bulgusu (Tablo 4.18) blok *giriş* projeksiyonlarından örneklenmiştir; SSM
bloğunun iç aktivasyonları — literatürün aykırı değer raporlarının kısmen ait
olduğu noktalar — enstrümante edilmemiştir. Bu nedenle iddia bilinçli olarak
daraltılmıştır: "SSM aktivasyonları ılımlıdır" genellemesi değil, "bu
omurga/görev/katman kümesinde literatür anlatısı gözlenmedi" tespiti.

**Yeniden formülasyon: prototip ölçeği.** AS4'ün pozitif sonuçları tek SS2D
bloğu, tek girdi ve L=1 024 içindir; 22 bloğun fp16 birikiminin mIoU etkisi ve
stage-0'ın L=16 384'ündeki davranış (P=32'de 512 blok — ölçek penceresinin
riskli bölgesi) ölçülmemiştir (§4.5.3). Tam-model iddiası bu prototipten
*çıkarsanamaz*; prototipin kanıtladığı, mekanizma ve yönelimdir.

**Doğruluk eşitleme yerine Pareto düzlemi.** Mac-only revizyonun metodolojik
sonucu olarak omurgalar sıfırdan eşit reçeteyle eğitilmemiş, yayınlanmış
ADE20K checkpoint'leri kullanılmıştır; üç model bu yüzden aynı doğrulukta
değildir (mIoU 44.3–48.3) ve karşılaştırma doğruluk-gecikme Pareto düzlemi
üzerinden kurulmuştur (§4.2.5). Bu bilimsel olarak geçerli, ancak "eşit
doğrulukta X kat yavaş" türü cümleleri imkânsız kılan bir çerçevedir; tezin
oran raporları (3.6×, 16×, 25×) doğruluk farkı *lehte* olan bir SSM'e
karşıdır ve bu yönüyle SSM aleyhine değil lehine ihtiyatlıdır.

**Enerji ölçümünün çözünürlüğü.** Enerji, `powermetrics`'in 200 ms'lik
örnekleme penceresiyle ve kısa koşu serileriyle (hücre başına 5–20 geçiş)
ölçülmüştür; Tablo 4.4'ün dipnotunda belirtildiği gibi bu serilerin geçiş
süreleri birincil gecikme medyanlarından bir miktar sapar. Enerji
karşılaştırmaları bu nedenle mertebe düzeyinde (10×, 25×) okunmalı, yüzde
düzeyinde okunmamalıdır. ANE davranışı da doğrudan sayaçlarla değil, güç
rayı telemetrisi + Xcode raporu ikilisiyle doğrulanmıştır; birim-içi
profilleme mevcut araçlarla yapılamamaktadır (§4.5 kapsam notu 4).

**Tek sürüm kümesi, tek zaman kesiti.** Bulguların tamamı §5.1'de sabitlenen
sürümlerle alınmıştır. Ölçek penceresinin sınırları (hangi düğüm sayısında
kapandığı), CoreML dönüştürücüsünün hangi desende düştüğü ve ORT
optimizer'ının eritme davranışı araç zinciri sürümlerinin fonksiyonudur ve
sonraki sürümlerde kayabilir. Bu sınırlılık diğerlerinden farklı bir statüde
olduğu — tezin geçerlilik iddiasının kendisini ilgilendirdiği — için ayrı bir
alt bölümü hak etmektedir: §5.5.

**Yüksek çözünürlüklü CPU hücrelerinde varyans.** ≥768² CPU ölçümlerinin bir
kısmı yüksek koşu-içi varyans göstermiştir (ConvNeXt 768² CPU std ±1.8 s;
Swin 1024² benzer — Tablo 4.6 dipnotları) ve bu satırlardaki oranlar
Bölüm 4'te de ihtiyat kaydıyla verilmiştir. Tezin ana oranları (3.6×, 6.6×,
16×, 25×) bu hücrelere değil, varyansı düşük 512² ve MPS ölçümlerine
dayanmaktadır; yüksek-çözünürlük CPU satırları yalnızca eğilim düzeyinde
yorumlanmalıdır.

Bu sınırlılıkların hiçbiri sonradan keşfedilmiş değildir; Mac-only revizyon
kararıyla birlikte (11 Ağustos 2026) baştan kabul edilmiş ve deney tasarımı
buna göre kurulmuştur. Sınırlı evrende derin ölçüm, geniş evrende sığ ölçüme
bilinçli olarak tercih edilmiştir; tezin başlığındaki "Apple Silicon"
daraltması bu tercihin beyanıdır.

## 5.5 Tehditler ve Karşı-Argümanlar: Araç Zincirleri Olgunlaşırsa Bu Bulgular Geçersizleşir mi?

Bu tezin bulgularına yöneltilebilecek en güçlü itiraz şudur: "Ölçtüğünüz şey
mimarinin değil, 2026 yazındaki araç zincirlerinin fotoğrafıdır. TorchScript
exporter `Loop` üretmeyi öğrendiğinde, coremltools o `TypeError`'ı
düzelttiğinde, protobuf sınırı esnetildiğinde bu tez tarihe karışır." İtiraz
ciddiye alınmayı hak eder ve dürüst cevap iki parçalıdır: **kısmen evet,
büyük ölçüde hayır.** Ayrım, bulguları konjonktürel ve yapısal olarak
ayrıştırarak yapılmalıdır.

**Konjonktürel bulgular — yazılım güncellemesiyle geçersizleşebilirler.**
Şu bulgular, belirli araçların belirli sürümlerindeki eksikliklerin
ürünüdür ve öyle raporlanmıştır: CoreML dönüşümünü düşüren `aten::Int`
`TypeError`'ı (Tablo 4.10) bir dönüştürücü hatası sınıfıdır; benzer desenler
bizim kontrolümüzdeki kodda elle giderilebilmiştir (§4.3.5). TorchScript
exporter'ın taramayı `Loop`/`Scan` yerine tam unroll etmesi (Tablo 4.12) bir
exporter tercihi/eksikliğidir; farklı bir dışa aktarma yolu (ör. dynamo
tabanlı exporter — bu tezde denenmemiştir) farklı graf üretebilir.
`torch.compile`'ın istisnasız süreç çökmesi (Tablo 4.8) bir derleyici
hatasıdır. VMamba deposunun NVIDIA'sız makinede import edilememesi (§4.3.5)
iki satırlık guard eksikliğidir. Bu kalemlerin her biri, ilgili projelerin
bir sonraki sürümünde sessizce düzelebilir — ve düzelmelidir; tezin ekleri
(EK B–D) tam da bu düzeltmeleri kolaylaştıracak reprodüksiyon malzemesini
içermektedir.

**Yapısal bulgular — daha kalıcı olduklarını düşündüren ölçülmüş nedenleri
vardır.** Üç bulgu bu sınıftadır. Birincisi ve en önemlisi *yapı = ölçek*
teşhisidir (§4.5.1): dönüşüm/yükleme maliyetinin operatör türüne değil graf
kardinalitesine bağlı olduğu ve düğüm sayısıyla süperlineer büyüdüğü, üç
bağımsız seride ölçülmüştür (mikrobenchmark 17.9×, tam model 15.7×,
ResNet/mikro-SSM karşıtlığı ~2 000×). Bu eğri belirli bir araç hatası değil,
graf ayrıştırma/optimizasyon altyapılarının maliyet ekonomisidir; `Loop`
desteği gelse bile aynı hastalığın öbür semptomuna dönülür — ONNX
topluluğunun kendi kayıtları (§2.5.1, ORT-Issue-27796: M3'te 17× yavaşlama),
`Loop` yolunun yorumlayıcı yükünün de cezalandırıcı olduğunu göstermektedir.
İki semptom, tek neden: graf temsilleri ardışık yinelemeyi ifade etmekte
yapısal olarak zorlanmaktadır ve bu, tek bir sürümün değil temsil
paradigmasının özelliğidir. İkincisi, ANE'nin karakteridir: statik,
önceden-derlenmiş, sabit-şekilli graf tercihi ve dar operatör/desen yelpazesi
donanım-yazılım ortak tasarımının sonucudur — Transformer'ın 0/631'lik reddi
(Tablo 4.15) ve "atıyor ama sevmiyor" bulgusu (maskeli-küçük-matmul: %97–99
atama, GPU'nun 6×'i gecikme, §4.5.2) bu karakterin iki ölçümüdür. Üçüncüsü,
paralel donanımın ardışık taramayla temel uyumsuzluğudur: MPS'in izole
taramayı CPU'dan yavaş çalıştırması (Tablo 4.3) ve tam modelde hızlanma
oranının klasiklerin yarısında kalması (Tablo 4.2), Mamba'nın el yazması
CUDA çekirdeğinin varlık sebebinin Apple tarafındaki izdüşümüdür — çekirdek
mühendisliği gerektiren bu uyumsuzluk, dışa aktarma araçlarının olgunlaşmasıyla
kendiliğinden çözülmez.

Bu ayrıştırmanın pratik sonucu şudur: araç zincirleri olgunlaştığında
değişecek olan, engelin *konumu* ve *şiddetidir*; tezin teşhis ettiği
*mekanizma* — maliyetin katmanlar arasında taşınması ve graf kardinalitesinin
belirleyiciliği — geçerliliğini korur. Nitekim tezin kendi müdahale deneyi
bunun kanıtıdır: blok-form, hiçbir araç güncellemesi beklemeden, yalnızca
kardinaliteyi yöneterek CoreML kapısını açmıştır (Tablo 4.21). Araç
zincirlerinin gelecekteki iyileşmeleri de büyük olasılıkla aynı eksende —
temsil ölçeğini yönetme ekseninde — olacaktır ve bu tez o eksenin haritasını
çıkarmıştır.

Sınıflandırmanın gri bölgesi de dürüstçe işaretlenmelidir: bazı bulgular iki
sınıfın arasındadır. Dışa aktarmanın *tam unroll* stratejisi konjonktürel,
unroll'un cezalandırılması yapısaldır; bir `Loop`-tabanlı dışa aktarma yolu
semptomu değiştirir (graf küçülür, yorumlayıcı yükü büyür) ama ONNX
topluluğunun kendi ölçtüğü üzere hastalığı kaldırmaz. Benzer biçimde, statik
şekil zorunluluğu kısmen araç kısıtıdır; ancak ANE'nin sabit-şekil tercihi
donanım tarafında da vardır ve dinamik şekilli bir dışa aktarma yolu açılsa
bile ANE hedefinde yeniden statikleştirme gerekecektir. Bu gri bölge, "araçlar
düzelince her şey düzelir" iyimserliğinin neden temelsiz olduğunu gösterir:
düzeltilebilir kabuk ile kalıcı çekirdek iç içedir ve bu tezin katkılarından
biri ikisinin sınırını ölçümle çizmiş olmasıdır.

Bir diğer karşı-argüman mimari cepheden gelir: "Saf SSM omurgası zaten
terk ediliyor; hibrit mimariler (§2.2.2 — MambaVision sınıfı) tarama payını
küçülterek bu sorunu tasarımda çözüyor." Bu itiraz bulgularla çelişmez;
tersine, bu tezin mekanizma teşhisi hibrit eğilimin *nicel* gerekçesini
sağlar. Engel graf kardinalitesindeyse ve kardinalite tarama payıyla
ölçekleniyorsa, tarama paylarını azaltan her tasarım kararı dağıtım
maliyetini süperlineer eğri üzerinde aşağı kaydırır — hibritlerin CUDA
dışındaki yığınlarda daha iyi davranması beklenir, ancak bu tezde
ölçülmemiştir ve beklenti olarak işaretlenmelidir. Tezin blok-form katkısı
bu resimde hibritlere alternatif değil tamamlayıcıdır: hibrit, taramanın
*miktarını*; blok-form, taramanın *temsil ölçeğini* yönetir ve ikisi
birleştirilebilir.

İki karşı-argüman daha kısaca ele alınmalıdır. *"Yanlış dışa aktarma yolunu
ölçtünüz"* itirazına cevap: TorchScript exporter, ölçüm tarihinde resmî ve
varsayılan yoldur; alternatif yolların denenmemiş olması §5.4'te sınırlılık
olarak kayıtlıdır, ancak CoreML yolu exporter'dan bağımsız kendi
dönüştürücüsüyle düşmekte ve `torch.compile` üçüncü bağımsız zincir olarak
çökmektedir — üç zincirin aynı modelde, üç farklı mekanizmayla kırılması,
bulgunun tek bir aracın artefaktı olmadığının güçlü göstergesidir. *"Bir
sonraki çip/sürüm her şeyi değiştirir"* itirazına cevap: değiştirebilir ve bu
tez bunu ölçülebilir bir soruya dönüştürmüştür. Bulgular zaman damgalıdır ve
tezin açık ölçüm altyapısı (EK B), aynı matrisin gelecek sürümlerde yeniden
koşulmasını bir günlük işe indirger. "Bu bulgular geçersizleşir mi?" sorusunun
en dürüst cevabı budur: **geçersizleşmeleri dileğe değil ölçüme tabidir ve bu
tez, o ölçümün hem temel çizgisini hem aracını bırakmaktadır.** SSM araç
zincirleri olgunlaştığında bu tezin tabloları eskiyecektir; metodolojisi,
maliyet modeli ve mekanizma teşhisi ise tam da o olgunlaşmanın ölçüleceği
çerçeve olarak kalacaktır.

---

*Sayfa hedefi: ~10. Bölüm 4'ün tüm tablo referansları final ölçümlere
(TASK-020..033) dayanmaktadır. Tablo 5.1 karar tablosu Faz 5'te Şekil 4.1
(Pareto düzlemi) ile çapraz kontrol edilecektir.*


---

# 6. SONUÇ VE GELECEK ÇALIŞMALAR

*(TASLAK v1 — 14 Ağustos 2026, TASK-034)*

> Bu bölüm tezi kapatır: dört araştırma sorusunun ölçülmüş cevaplarını toplar
> (§6.1), katkıları listeler (§6.2) ve bu çalışmanın açtığı soruları gelecek
> çalışmalara devreder (§6.3). Bölüm 5'in yorum disiplini burada da geçerlidir:
> her cevap, Bölüm 4'ün belirli tablolarına dayanır.

## 6.1 Araştırma Sorularına Cevaplar

Sorulara tek tek geçmeden önce, tezin tek cümlelik iddiasının akıbeti
kaydedilmelidir. İddia şuydu: SSM tabanlı görü omurgalarının literatürde
bildirilen verimlilik avantajı özel CUDA çekirdeklerine bağımlıdır ve genel
amaçlı dağıtım yığınlarında büyük ölçüde kaybolur; kaybın kaynağı seçici
taramanın ardışık yapısı ile graf derleyicilerinin yürütme modeli arasındaki
uyumsuzluktur. Ölçümler iddianın birinci yarısını doğrulamış (kayıp:
dağıtılabilir-en-iyi hücrelerde 16×, enerjide ~25×, CoreML/ANE'de tam
imkânsızlık), ikinci yarısını ise doğrulamakla birlikte önemli biçimde
**rafine etmiştir**: uyumsuzluk, öngörülen gibi yürütme modelinde
(çıkarım aritmetiği hayatta, hatta ORT'de kazançlı) değil, ardışık yapının
graf *temsiline* açılma ölçeğindedir — ve bu ölçek, dönüşüm/yükleme
altyapılarının süperlineer maliyet eğrisini aşılamaz bölgeye taşımaktadır.
İddia böylece hem doğrulanmış hem keskinleşmiştir; aşağıdaki dört cevap bu
resmin soru-bazlı dökümüdür.

**AS1 — Eşit doğruluk bütçesinde SSM/ViT/CNN omurgalarının yüksek çözünürlüklü
semantik segmentasyondaki gerçek gecikme, bellek ve enerji profili nedir?**
Referans yığında (PyTorch eager) klasik omurgalar birbirine yakın, SSM ise kat
düzeyinde ayrık ölçülmüştür: 512²'de VMamba-T, CPU'da 3.6× (2 032'ye karşı
571 ms), MPS'te 6.6× (1 008'e karşı 152 ms) yavaştır (Tablo 4.2); MPS
hızlanması klasiklerde 3.1–3.8× iken VMamba'da 2.0×'te kalmaktadır — paralel
donanım ardışık taramayı aynı oranda hızlandıramamaktadır (§4.2.2). Enerji
aynı deseni izler: çıkarım başına 11 720 mJ (CPU) ve 4 844 mJ (MPS),
klasiklerin ~2.8–5 katı (Tablo 4.4); fark güçten değil süreden gelmektedir.
Bellek tarafında VMamba'nın MPS tahsis tepesi her çözünürlükte klasiklerin
~1.5–2 katıdır (Tablo 4.5). Kritik bulgu çözünürlük ekseninde negatiftir:
O(L) taramanın vaat ettiği makas kapanmamakta, VMamba/ConvNeXt oranı
256²–1024² aralığında ~6–9× bandında salınmaktadır (Tablo 4.6) — pratik
rakipler zaten O(L) olduğundan, SSM'e kalan tek fark 6–9× aleyhte bir sabit
katsayıdır. VMamba-T bu bedele karşılık deney kümesinin en doğru modelidir
(+2.9–4.0 mIoU, §4.2.5); AS1'in cevabı bir hüküm değil, bu ödünleşimin
nicelleştirilmesidir.

**AS2 — Teorik FLOPs avantajı ile ölçülen gecikme arasındaki fark dağıtım
yığınına göre nasıl değişir; avantaj nerede buharlaşır?** Cevap, sorunun
varsaydığı yerde değildir: avantaj çıkarım katmanında değil, araç zincirinin
çıkarım-öncesi katmanlarında buharlaşmaktadır. Çıkarım katmanı (d) SSM için
hayatta kalmakta, hatta kazanca dönüşmektedir — ORT çıkarım medyanı 618 ms
ile eager'ın 0.30×'u ve klasiklerin ORT hücrelerinden hızlıdır (ORT paradoksu,
§4.3.1). Buharlaşma katman (a)–(c)'dedir: ONNX dışa aktarması klasiklerin
448×'i (537.8 s), graf 463× kalabalık (390 758 düğüm; 858.4 MB'ın 614 MB'ı
saf graf yapısı), ORT oturum yüklemesi ~7 249× (724.9 s — her süreç
başlatımında yeniden) (Tablo 4.9). CoreML yolu dönüşüm aşamasında tamamen
kapalıdır; `torch.compile` süreci çökertmektedir; 1024²'de dışa aktarma ~65
GB'a şişerek bellek duvarına çarpmaktadır (Tablo 4.10, 4.8, 4.11) — teorik
avantaj bölgesi, dışa aktarmanın fiziksel olarak imkânsızlaştığı bölgeyle
çakışmaktadır. Net sonuç: eager'da 3.6× olan fark, dağıtılabilir-en-iyi
hücreler karşılaştırıldığında gecikmede 16×'e (63.9'a karşı 1 008 ms),
enerjide ~25×'e (458'e karşı 11 720 mJ) açılmaktadır; mekanizma VMamba'nın
yavaşlaması değil, klasiklerin tırmandığı hızlandırma merdivenine (eager →
ORT → CoreML → ANE; resmî kademelenme: CNN %100 ANE, ViT %0, SSM ∅ — Tablo
4.15) SSM'in binememesidir. Maliyet buharlaşmamakta, FLOPs'un kör olduğu
katmanlara taşınmaktadır.

**AS3 — PTQ yöntemleri yoğun tahmine taşındığında doğruluk kaybı nasıl
davranır; yüksek çözünürlük aykırı değer profilini değiştirir mi?** Doğruluk
katmanında transfer sorunsuzdur: W8 ağırlık-nicemlemesi segmentasyonda
bedavadır (ΔmIoU ±0, boyut 0.50×, ConvNeXt-ANE gecikmesi −%29 — Tablo 4.16);
W4 palettizasyon çökme olmaksızın modele bağlı bir fatura keser (ConvNeXt
−3.9, Swin −2.8 mIoU). Mekanizma katmanında literatür beklentisi tersine
dönmüştür: çözünürlük büyüdükçe SSM aktivasyonlarının savrulacağı hipotezi bu
veri kümesinde desteklenmemiş; VMamba üç omurganın en ılımlı ve en
çözünürlük-kararlı profilini vermiş (kurtosis ~8/14 sabit, aykırı kanal
makası 3.8 → 3.2 daralıyor), beklenen uzun-kuyruk patolojisi ConvNeXt'in
depthwise katmanlarında çıkmıştır (kurtosis 75'e kadar, monoton büyüyen —
Tablo 4.18). Gecikme katmanında kazanç tamamen yığına ve hesaplama birimine
bağlıdır: ANE'de %29 kazanç, GPU fallback'inde etkisizlik, ORT-CPU'da 6–17×
pesimizasyon (Tablo 4.19). Tez-düzeyi sonuç VMamba sütunundadır: yapısal
şişkinlik nicemlenememektedir (858 → 715 MB; yükleme 691 s, değişmedi) —
nicemlemeye ölçülen profillere göre en uygun omurga, nicemlemenin kazanca
dönüştüğü tek hücreye (CoreML/ANE) dönüşüm katmanında elendiği için hiç
ulaşamamaktadır. SSM'in uç engeli sayısal değil, yapısaldır.

**AS4 — Seçici taramanın derleyici-dostu yeniden formülasyonu ONNX/CoreML
darboğazını ne kadar kapatabilir?** Tek gerçek SS2D bloğu ölçeğinde
(checkpoint ağırlıkları, gerçek ADE20K ara-aktivasyonu) cevap iki parçalıdır.
Kısmi evet: Mamba-2 SSD'nin d_state=1 özel hâli olan çürüme-matrisi formu —
naif kapalı formun aksine sayısal olarak stabil (naif form gerçek ağırlıklarla
NaN; blok-içi log-çürüme −513.6'ya iniyor) ve referansa 10⁻⁶ sadakatinde —
grafı 20× küçültmüş (7 285 → 357 düğüm), ORT yüklemesini 25× hızlandırmış ve
tam modelde kapalı olan CoreML kapısını açmıştır (dönüşüm 16 s — Tablo 4.20,
4.21). Bedeli dürüstçe ölçülmüştür: FLOPs O(L·P)'ye çıkmakta (CPU yürütücülerde
katman (d) kaybı), ANE tensör-düzeni reçetesi ölçülebilir kazanç
getirmemekte, fp16 birikimi ve tam-ölçek davranışı prototip dışında
kalmaktadır (§4.5.2–4.5.3). Bölümün asıl katkısı ise planlanmamış mekanizma
bulgusudur: kontrol koşulu olan unroll'lu seq formu tek-blok ölçekte CoreML'e
sorunsuz dönüşmüş ve ANE'de tüm matrisin en hızlı hücresi olmuştur (6.7 ms,
%99.9 ANE ataması — Tablo 4.22). Engel taramanın operatör kümesinde değil,
graf temsilinin ölçeğindedir: *yapı = ölçek*. AS4'ün cevabı böylece hem bir
müdahale kanıtı (kapı açılabilir) hem AS2'nin mekanizma açıklamasıdır (kapıyı
kapatan, kardinalitedir).

## 6.2 Katkıların Özeti

Bu tezin katkıları, Bölüm 1.4'te taahhüt edilen beş kalemle hizalanarak şöyle
özetlenebilir:

1. **Sistematik verimlilik ölçümü.** Üç mimari ailesi (CNN/ConvNeXt-T,
   Transformer/Swin-T, SSM/VMamba-T) × Apple Silicon dağıtım yığınları
   (PyTorch eager CPU/MPS, torch.compile, ONNX Runtime CPU ve CoreML EP,
   CoreML CPU/GPU/ANE) matrisi, termal kontrol, senkronizasyon, medyan/P99
   raporlama ve Xcode ile yürütme-yeri doğrulaması içeren bir protokolle
   (Bölüm 3.5) eksiksiz ölçülmüştür. ANE üzerinde SSM'i konu edinen ilk
   sistematik ölçüm olması itibarıyla §2.6'da tespit edilen boşluğun
   doğrudan karşılığıdır.
2. **Teorik-gerçekleşen verimlilik farkının nicel karakterizasyonu.** Farkın
   büyüklüğü (eager 3.6× → dağıtılabilir-en-iyi 16× → enerji ~25×), konumu
   (çıkarım değil; dönüşüm 448×, yükleme ~7 249×, paket yapısı 614 MB) ve
   mekanizması (tam unroll → süperlineer yükleme eğrisi → 1024²'de bellek
   duvarı) dört katmanlı maliyet modeliyle ayrıştırılmıştır. Model, alanın
   raporlama pratiğine öneri olarak sunulmaktadır (§5.2).
3. **PTQ'nun yoğun tahmine transferinin analizi.** W8'in segmentasyonda
   bedava olduğu, W4 faturasının modelin kuyruk profiliyle eşleştiği,
   nicemleme kazancının hesaplama birimine bağlı olduğu (ANE %29 kazanç /
   ORT-CPU 17× kayıp) ve literatürün SSM aykırı-değer anlatısının bu
   omurga/görev/katman kümesinde gözlenmediği ölçülmüştür — son bulgu, alanın
   ölçek hipotezinin yoğun tahminde ilk doğrudan sınamasıdır.
4. **Blok-form yeniden formülasyon ve mekanizma teşhisi.** Seçici taramanın
   çürüme-matrisi formu (SSD'nin d_state=1 özel hâli) türetilmiş, sayısal
   stabilite koşulları belgelenmiş (naif formun NaN mekanizması dahil) ve tek
   blok ölçeğinde graf/yükleme/dönüşüm kazançları ölçülmüştür; kontrol
   koşulunun ürettiği *yapı = ölçek* teşhisi, dağıtım engelinin adresini
   operatör desteğinden graf kardinalitesi yönetimine taşımaktadır.
5. **Açık ölçüm altyapısı.** Ölçüm harness'ı, dışa aktarma/nicemleme/
   reformülasyon betikleri, ham ölçüm kayıtları (`results/raw/`, ortam
   sürümleriyle birlikte) ve reprodüksiyon belgeleri açık kaynak olarak
   yayımlanmaktadır (github.com/omergungor11/ssm-edge-thesis); aynı matrisin
   gelecek araç-zinciri sürümlerinde yeniden koşulması amaçlanan kullanım
   senaryosudur (§5.5).

Beş katkının ağırlığı eşit değildir ve dürüst bir özet bunu söylemelidir.
En sağlam ayak, ölçüm katkılarıdır (1–3): matris kapanmış, ham kayıtlar
ortam bilgisiyle arşivlenmiş, ana bulgular birden fazla bağımsız kanıt
hattıyla (güç telemetrisi + Xcode raporu; mikrobenchmark + tam model;
statik döküm + çalışma-zamanı profili) desteklenmiştir. Dördüncü katkı
bilinçli olarak prototip ölçeğindedir — kanıtladığı şey tam-model çözümü
değil, çözümün *mekanizması ve adresidir* — ve bu sınır §4.5'te ve §5.4'te
açıkça çizilmiştir. Beşinci katkının değeri ise zamanla artan türdendir:
bulgular zaman damgalı olduğundan (§5.5), aynı matrisin gelecek sürümlerde
yeniden koşulabilmesi, tezin tablolarını bir fotoğraftan bir izleme
serisinin ilk karesine dönüştürür.

## 6.3 Gelecek Çalışmalar

Bu tezin açık bıraktığı sorular beş başlıkta toplanmaktadır; ilk ikisi bu
çalışmanın doğrudan devamı, kalanlar kapsam genişletmesidir.

**Tam-model blok ölçekleme.** Prototipin en net devamı, blok formunun 22 SS2D
bloğunun tamamına ve stage-0'ın L=16 384'üne taşınmasıdır. §4.5.3'te
tanımlanan risk — P=32'de 512 bloklu taşıma döngüsünün grafı yeniden ölçek
penceresine yaklaştırması — muhtemel çözüm yollarıyla birlikte ortadadır:
bloklar-arası taşımaya ikinci bir kapalı-form katmanı (hiyerarşik/iki-seviyeli
parçalama), katmana göre değişen P, uzun-L katmanlarında büyük blok. Başarı
ölçütü de tezden devralınmalıdır: yalnızca çıkarım gecikmesi değil, dört
katmanın dördü.

**Uçtan-uca doğruluk doğrulaması.** Tek blokta 0.007–0.07 bandında ölçülen
CoreML fp16 sadakatinin 22 blok üzerindeki birikimi ve blok-form tam modelin
ADE20K mIoU'su ölçülmelidir — dönüşebilirlik kanıtı ancak görev başarımı
korunduğunda dağıtılabilirlik kanıtına dönüşür; nicemleme deltalarının tam-değerlendirme
protokolüyle (2 000 görüntü, en-boy korumalı) doğrulanması da aynı pakete
girer. "CoreML'e dönüşen SSM" hedefine ulaşılırsa, W8'in ANE kazancının
(§4.4.3) SSM'e de transfer olup olmadığı doğrudan sınanabilir.

**Aktivasyon nicemlemesi (Aşama II).** W8A8 ve kalibrasyonlu statik nicemleme
(ImageNet alt-kümesi kalibrasyonuyla) bu tezde ölçülmemiştir; §4.4.2'nin
ılımlı aktivasyon profili W8A8 için umut vericidir ama kanıt değildir. SSM
bloğunun iç aktivasyonlarının (durum güncellemeleri, kapı çıktıları)
enstrümantasyonu, literatürün aykırı-değer raporlarıyla doğrudan
karşılaştırmayı mümkün kılacaktır.

**Diğer NPU aileleri ve donanımlar.** "Mimari başına bir kademe" bulgusunun
(Tablo 4.15) Qualcomm Hexagon, Intel NPU, Google Edge TPU ve dönerse CUDA
ekseni (plan buna kapı bırakmaktadır) üzerinde sınanması, bulgunun tek NPU
ailesinden platform-genel bir teze yükselip yükselmeyeceğini belirleyecektir.
Dört katmanlı maliyet modeli bu genişletmenin hazır çerçevesidir; her yeni
platform, matrise yalnızca bir sütun ekler. Özellikle ilginç soru, başka bir
NPU derleyicisinin ölçek penceresinin *nerede* kapandığıdır — pencere sınırı
platformlar arasında ölçülebilirse, "graf kardinalitesi bütçesi" mimari
tasarımın nicel bir kısıtı hâline gelebilir.

**Yüksek çözünürlük alternatif yolları.** Cityscapes/1024² ekseni bu tezde
dışa aktarma bellek duvarı nedeniyle kapalı kalmıştır (Tablo 4.11). Blok-form
düğüm sayısını O(L/P)'ye indirdiğinden, tam-model blok ölçekleme başarılırsa
1024² dışa aktarması ilk kez erişilebilir hâle gelebilir; karo-tabanlı
(tiled) çıkarım ve dinamik-şekilli dışa aktarma yolları da aynı hedefin
alternatif rotalarıdır.

---

Kapanışı, tezin başında verilen sözle bağlamak gerekir. Bölüm 1'de bu konunun
seçilme gerekçesi, iddia doğrulansa da çürütülse de yazılabilir bir tez
olmasıydı: "negatif sonuç da sonuçtur." Bu ilke yalnızca konu seçimini değil,
çalışmanın her katmanını taşımıştır. Tezin ana bulgusu bir negatif sonuçtur —
SSM'in teorik verimlilik avantajı, ölçülen genel amaçlı yığınlarda dağıtım
katmanına ulaşamamaktadır — ve değeri tam da titizlikle ölçülmüş olmasından
gelir: 16× uçurumun kaynağının çıkarım değil araç zinciri olduğu, engelin
sayısal değil yapısal, yapının da operatör değil ölçek olduğu, ancak her
adımı ölçen bir zincirle teşhis edilebilmiştir. Ara negatifler de aynı
muameleyi görmüştür: beklenen aykırı-değer fırtınasının gözlenmemesi, naif
kapalı formun NaN'ı, ANE reçetesinin sıfır getirisi — hiçbiri silinmemiş,
her biri sonuç tablolarının eşit haklı satırları olarak raporlanmıştır ve
bunların birkaçı (VMamba'nın ılımlı aktivasyon profili, seq formunun ANE'de
hızlı oluşu) tezin en bilgilendirici bulguları çıkmıştır. Negatif sonucun bir de kurucu işlevi olmuştur: AS4'ün müdahalesi, ancak
AS2–AS3'ün negatiflerinin engeli doğru adrese — operatöre değil ölçeğe,
sayıya değil yapıya — yerleştirmesi sayesinde doğru hedefe nişan alabilmiştir.
Ölçmeden müdahale eden bir çalışma, büyük olasılıkla nicemlemeye ya da
operatör desteğine yatırım yapar ve ikisinin de engele dokunmadığını bu tezin
verisi göstermektedir. Alan, verimlilik iddialarının parlak sayılarla
duyurulduğu bir dönemden geçmektedir; bu tezin önerdiği düzeltme, iddiaların
küçültülmesi değil, maliyetin dört katmanıyla birlikte, zaman damgası ve ham
kayıtla raporlanmasıdır. Teoriden silikona
giden yol bugün SSM görü omurgaları için kapalıysa, bu tez o kapının nerede,
neden ve hangi ölçekte kapandığını ölçmüş; bir prototiple aralanabildiğini
göstermiş ve kapıyı izlemeye devam edecek altyapıyı açık bırakmıştır.

---

*Sayfa hedefi: ~5. AS cevapları Bölüm 4'ün final ölçümlerine (TASK-020..033),
katkı listesi Bölüm 1.4 taahhütlerine hizalıdır; Bölüm 1 yazılırken iki liste
karşılıklı senkronize edilecektir.*


---

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


---

# EKLER

> **Durum:** İskelet (TASK-036). Her ek, nihai içeriğin nereden üretileceğini işaret eder;
> tablo/şekil gövdeleri Faz 5 son geçişte bağlanacak. Yollar depo köküne görelidir.

---

## EK A — Tam Deney Sonuç Tabloları ve Ham Veri Envanteri

Tüm ham ölçümler `results/raw/` altındadır ve git'e dahildir ("başarısızlık da veridir"
ilkesiyle başarısız denemeler dahil). Her kayıt; ortam sürümleri, git commit'i ve termal
durumla damgalıdır. Aşağıdaki eşleme, metindeki her tabloyu besleyen ham dosyayı verir.

### A.1 Ham dosya → bölüm/tablo eşlemesi

| Ham dosya (`results/raw/`) | Üreten betik | Beslediği tablo/şekil |
|---|---|---|
| `ade20k_miou_progress.json`, `ade20k_swin_progress.json`, `ade20k_convnext_progress.json` | `src/models/eval_ade20k.py` | Tablo 4.1 (model kartları, mIoU doğrulaması) |
| `ade20k_conf_*_n250.npy`, `ade20k_conf_*_n2000.npy` | `src/models/eval_ade20k.py` | Tablo 4.1 (karışıklık matrisleri; n=250 hızlı / n=2000 tam) |
| `premise_L196.json`, `premise_L1024.json` (+ `mini_mamba_L196/L1024.onnx/.mlpackage`) | `src/premise/mamba_min.py` | Tablo 4.3 (MiniMamba öncül mikrobenchmark, TASK-008) |
| `latency_matrix_vmamba.jsonl`, `latency_matrix_swin.jsonl`, `latency_matrix_convnext.jsonl` | `src/benchmark/measure_matrix.py`, `src/benchmark/resolution_sweep.py`, `src/benchmark/round3_cells.py` | Tablo 4.2 (eager), 4.5 (MPS tepe bellek), 4.6 (gecikme × çözünürlük), 4.8 (gecikme matrisi, tüm turlar) |
| `energy_matrix.json` | `src/benchmark/energy_round.py` | Tablo 4.4 (net mJ/çıkarım), 4.14 (ANE enerji imzası, powermetrics) |
| `export_matrix.jsonl` | `src/export/export_cell.py` | Tablo 4.7 (dağıtım matrisi durumu), 4.9 (ONNX yolu), 4.10 (CoreML yolu), 4.11 (VMamba ONNX × çözünürlük), 4.12 (graf op dökümü) |
| `ort_profile_vmamba_top.json`, `ort_profile_swin_top.json`, `ort_profile_convnext_top.json` | `src/benchmark/round3_cells.py` | Tablo 4.13 (ORT CPU çalışma-zamanı operatör profili) |
| `xcode/xcode-perf-convnext-upernet-512-ane353of353.png`, `xcode/xcode-perf-swin-upernet-512-ane0of631.png` | Xcode Core ML Performance Report (elle) | Tablo 4.15 (resmî katman-yeri dökümü; görüntüler EK C'de) |
| `quant_matrix.jsonl` | `src/quant/quant_round1.py` | Tablo 4.16 (W-nicemleme matrisi), 4.17 (vekil metrik), 4.19 (ORT dinamik INT8) |
| `quant_miou_{convnext,swin}_{fp32export,w8,w4}.json` | `src/quant/eval_quant_miou.py` | Tablo 4.16-4.17 (gerçek ΔmIoU sütunları) |
| `activation_stats_{vmamba,swin,convnext}.json` | `src/quant/activation_stats.py` | Tablo 4.18 (aktivasyon istatistikleri × çözünürlük) |
| `reform_input.pt` | `src/reformulation/capture.py` | Tablo 4.20 (gerçek ağırlık + gerçek aktivasyonla doğrulama girdisi) |
| `reform_matrix.jsonl`, `reform_seq.jsonl`, `reform_{blocked,ane}{32,64,128}.jsonl` | `src/reformulation/run_matrix.py`, `verify.py` | Tablo 4.20 (sayısal doğrulama), 4.21 (graf/yükleme), 4.22 (gecikme + ANE op%) |
| `validate_resnet50.jsonl` (+ `resnet50.mlpackage`) | `src/benchmark/validate_resnet50.py` | §3.5 harness doğrulaması (≤%1.2 sapma) |

**Ölçüm artefaktları** (tablo değil, kanıt): `*_upernet_512.mlpackage`,
`*_w8_linear.mlpackage`, `*_w4_palette.mlpackage`, `*_int8.onnx`,
`reform_*.onnx/.mlpackage` — paket boyutu / yükleme süresi ölçümlerinin nesneleri.

### A.2 Şekiller

`results/figures/fig-4.1-pareto`, `fig-4.2-scaling`, `fig-4.3-cost-layers`,
`fig-4.4-energy`, `fig-4.5-quant` (PDF+PNG) — tümü `src/analysis/make_figures.py`
tarafından yalnızca `results/raw/` girdilerinden üretilir (elle düzenleme yok).

> **Faz 5 işi:** Tabloların tam (kırpılmamış) sürümleri — metinde yer kısıtıyla
> özetlenen satırlar dahil — bu eke JSONL kayıtlarından dökülecek.

---

## EK B — Ölçüm Altyapısı: Modül Haritası ve Yeniden Üretim

### B.1 Modül haritası (`src/`)

| Modül | İçerik |
|---|---|
| `src/benchmark/` | Ölçüm harness'ı: `harness.py` (ısınma + medyan/P99 istatistik + JSONL yazım), `runners.py` (yığın-agnostik torch/ORT/CoreML koşucuları), `env.py` (ortam sürümü damgalama), `stats.py`, `measure_matrix.py` (ana gecikme matrisi), `resolution_sweep.py` (çözünürlük taraması), `energy_round.py` (powermetrics enerji), `round3_cells.py` (ORT profilleri), `validate_resnet50.py` (harness doğrulaması) |
| `src/models/` | mmseg'siz, anahtar-uyumlu saf-PyTorch yükleyiciler: `vmamba_upernet.py`, `swin_upernet.py`, `convnext_upernet.py` + `eval_ade20k.py` (mIoU değerlendiricisi) |
| `src/export/` | `export_cell.py` — export matrisi hücre koşucusu (ONNX/CoreML; başarısızlıklar da kayıtlı) |
| `src/quant/` | `quant_round1.py` (CoreML W8/W4, ORT INT8), `eval_quant_miou.py`, `activation_stats.py` |
| `src/data/` | `ade20k.py` — ADE20K veri katmanı |
| `src/premise/` | `mamba_min.py` — Faz 0 MiniMamba öncül mikrobenchmark'ı |
| `src/reformulation/` | Faz 4 — EK D'ye bakınız |
| `src/analysis/` | `make_figures.py` — tez şekilleri |

### B.2 Tek komutluk yeniden üretim (README ile tutarlı)

Ortam: Apple Silicon Mac (test edilen: baz M5, 24 GB), Python 3.12.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

python src/models/eval_ade20k.py 2000 --model vmamba   # mIoU doğrulaması (Tablo 4.1)
python src/export/export_cell.py vmamba onnx 512       # Export matrisi hücresi (Tablo 4.7)
python src/benchmark/measure_matrix.py                 # Gecikme matrisi (Tablo 4.8)
python src/quant/quant_round1.py                       # Nicemleme (Tablo 4.16-4.19)
python src/reformulation/run_matrix.py                 # Yeniden formülasyon (Tablo 4.20-4.22)
```

Checkpoint'ler ve ADE20K git dışıdır (edinim adresleri README'de); VMamba üçüncü-parti
kodu `third_party/VMamba` olarak klonlanır ve CPU-only ortam için gereken iki satırlık
yamalar `tez-docs/vmamba-yamalari.md`'de `[TEZ YAMASI]` etiketiyle belgelidir.

---

## EK C — Ortam ve Sürüm Belgeleme

### C.1 Kilitli sürüm kümesi

Tüm ölçümler tek makinede, tek sürüm kümesiyle alınmıştır (tartışma için Bölüm 5 tehdit
analizi):

| Bileşen | Sürüm |
|---|---|
| Donanım | Apple baz M5, 24 GB birleşik bellek (şebeke gücü) |
| İşletim sistemi | macOS 26.5.2 |
| Python | 3.12.13 |
| PyTorch | 2.13.0 |
| ONNX Runtime | 1.28.0 |
| coremltools | 9.0 |

Tam paket dökümü: `requirements.txt` + her JSONL kaydına gömülü ortam bloğu
(`src/benchmark/env.py`). Makine/termal koşullar: `tez-docs/ortam-mac.md`.

### C.2 Xcode Core ML Performance Report kanıtları

Katman-yeri (CPU/GPU/ANE) atamaları Xcode'un resmî raporuyla doğrulanmıştır
(Tablo 4.15'in kaynağı):

- `results/raw/xcode/xcode-perf-convnext-upernet-512-ane353of353.png` — ConvNeXt-T+UPerNet: 353/353 op ANE (%100)
- `results/raw/xcode/xcode-perf-swin-upernet-512-ane0of631.png` — Swin-T+UPerNet: 0/631 op ANE (%0, GPU fallback)

*(VMamba-T için rapor yoktur: model Core ML'e dönüşemediğinden ölçülecek paket oluşmaz —
Bölüm 4.3.)*

> **Faz 5 işi:** Görüntüler ek sayfasına gömülecek; powermetrics örnekleme
> yapılandırması (200 ms) ve boşta-düşme prosedürü §3.5'ten buraya özetlenecek.

---

## EK D — Yeniden Formüle Edilmiş Tarama Operatörü (Faz 4)

### D.1 Dosya haritası (`src/reformulation/`)

| Dosya | Rol |
|---|---|
| `common.py` | `SS2DBase` — üç formun paylaştığı SS2D bloğu iskeleti; `KD` sabiti |
| `capture.py` | Gerçek VMamba ağırlıkları + gerçek ara aktivasyonların yakalanması → `results/raw/reform_input.pt` |
| `ss2d_seq.py` | `seq` formu — adım adım özyinelemeli referans (L adımlık döngü; trace'te tam unroll) |
| `ss2d_blocked.py` | `blocked` formu — blok kapalı form, iki varyant (aşağıda) |
| `ss2d_ane.py` | `ane` formu — ANE-dostu yerleşim (Apple ANE ilkeleri: (B,C,1,S) formatı, reshape/transpose minimizasyonu) |
| `verify.py` | Üç formun üçüncü-parti selective scan referansına karşı sayısal doğrulaması (Tablo 4.20) |
| `run_matrix.py` | Ölçüm matrisi: form × blok boyutu {32, 64, 128} → `reform_*.jsonl`, `reform_matrix.jsonl` (Tablo 4.21-4.22) |

### D.2 Çürüme-matrisi (decay) formülasyonunun özeti

`d_state=1` için özyineleme kanal başına skalerdir: `h_t = a_t·h_{t-1} + b_t`,
`a_t = exp(Δ_t·A)`. `S_t = Σ_{i≤t} Δ_i·A` (log-birikimli toplam) ile blok uzunluğu P
içinde iki kapalı form:

1. **`cumsum` varyantı** (reçetedeki bölmeli form): `h_t = e^{S_t}·(h_0 + Σ_{i≤t} b_i·e^{-S_i})`.
   Gerçek aktivasyonlarda blok içi min(S) ≈ −513 (P=64) ölçüldü → `e^{-S}` fp32'de taşar
   (üst sınır ~e^88) → NaN. **Kayıt için tutulur; başarısızlık da veridir**
   (`reform_matrix.jsonl` 'verify' kayıtları).
2. **`decay` varyantı** (stabil, varsayılan): alt-üçgen çürüme matrisi
   `T[t,i] = e^{S_t − S_i}` (i ≤ t; her giriş ≤ 1 → taşma imkânsız),
   `h_blok = T @ b`; bloklar arası taşıma `h_t += e^{S_t}·h_0` (alttan taşma → 0,
   zararsız). Maske exp'ten **önce** uygulanır (üst üçgende S_t − S_i > 0 → önce exp
   sonra maske `inf·0 = NaN` üretir; `−10⁴ → exp → 0`).

Bu, Mamba-2/SSD'nin "chunked" formülasyonunun `d_state=1` özel hâlidir: blok-köşegen
dikkat matrisi + skaler taşıma. Kullanılan op kümesi yalnızca `cumsum + exp + matmul` —
tümü ONNX/CoreML'in yerli operatörleridir; L uzunluklu unroll yerine L/P adımlık kısa
döngü kalır.

> **Faz 5 işi:** Formül türetimi Bölüm 4.5/§3'teki gösterimle hizalanacak; blok boyutu
> taraması (32/64/128) tam tabloları A.1'deki `reform_*` kayıtlarından dökülecek.
