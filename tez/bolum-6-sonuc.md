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
