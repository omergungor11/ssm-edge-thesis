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
