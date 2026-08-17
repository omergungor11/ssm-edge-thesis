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
