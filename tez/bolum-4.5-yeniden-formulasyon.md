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
