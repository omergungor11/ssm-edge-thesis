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
