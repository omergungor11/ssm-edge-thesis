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
