# 4.2 – 4.3 Verimlilik Sonuçları

*(TASLAK v1 — 13 Ağustos 2026, TASK-024 ön-taslak; ölçümler sürüyor)*

> Bu iki bölüm, Bölüm 4.1'de kimlikleri ve doğrulukları belgelenen üç omurganın
> (ConvNeXt-T [CNN], Swin-T [Transformer], VMamba-T [SSM]) verimlilik ölçümlerini
> raporlar. Tüm ölçümler Bölüm 3.5'teki protokole tabidir; ham kayıtlar
> `results/raw/export_matrix.jsonl` ve `results/raw/energy_matrix.json` dosyalarındadır.
> Deney matrisi henüz tamamlanmamıştır; eksik hücreler **[ölçüm sürüyor]** ile
> işaretlenmiş olup bu taslak, ölçümler tamamlandıkça güncellenecektir.

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
ve bu taslakta yorumlanmamaktadır; kalıcılığı, çözünürlük taraması
tamamlandığında (§4.2.4) yeniden değerlendirilecektir. Analizin ana ekseni,
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
serilerdir (hücre başına 5–20 geçiş); geçiş süreleri bu nedenle Tablo 4.6'daki
birincil gecikme medyanlarından bir miktar sapar. Gecikme için bağlayıcı değerler
Tablo 4.6'dır; bu tablo enerji içindir.*

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

**Bellek.** Çıkarım anındaki tepe bellek (peak RSS) profilleri için kontrollü
ölçüm turu henüz tamamlanmadı: **[ölçüm sürüyor]**. Eldeki ilk veriler dışa
aktarma aşamasına aittir ve §4.3.2'de raporlanmaktadır (VMamba ONNX dışa aktarma
sürecinde tepe RSS 6.65 GB'a ulaşmıştır — 24 GB'lık bir geliştirme makinesinde
dahi kayda değer bir pay).

**Çözünürlük ölçeklendirme.** SSM'lerin teorik cazibesi tam da burada yatar:
dikkat mekanizmasının O(L²) karmaşıklığına karşılık taramanın O(L) ölçeklenmesi,
çözünürlük büyüdükçe SSM lehine açılan bir makas vaat eder. Bu vaadin eager
kipte gerçekleşip gerçekleşmediğini sınamak için 256², 512², 768² ve 1024²
çözünürlüklerinde gecikme taraması yürütülmektedir: **[ölçüm sürüyor]** —
512² sütunu Tablo 4.2'deki değerlerdir; diğer üç çözünürlük tamamlandığında
aşağıdaki şekil doldurulacaktır.

> **Şekil 4.2 — Çözünürlük-gecikme ölçeklendirme eğrileri** *(yer tutucu; ölçüm
> sürüyor)*. Üç omurganın 256²–1024² aralığındaki eager gecikme eğrileri (CPU ve
> MPS ayrı panellerde, log-log eksende). Beklenen soru: VMamba'nın O(L) eğim
> avantajı, sabit-katsayı dezavantajını (özel çekirdek yokluğu) hangi çözünürlükte
> telafi ediyor — ediyor mu?

Mikrobenchmark bu çerçeve için bir ön ipucu veriyor (Tablo 4.3): L 5.2× artarken
MiniMamba'nın eager CPU süresi 3.4×, MPS süresi 3.8× artmıştır — yani izole
taramanın kendi ölçeklenmesi lineer-altı ve teoriyle uyumludur. Buharlaşmanın
çıkarım eğiminde değil, başka katmanlarda gerçekleştiği hipotezi (Bölüm 4.3) bu
gözlemle tutarlıdır.

### 4.2.5 Doğruluk-Verimlilik Düzlemi

Bölüm 4.1'de kurulan çerçeve gereği üç omurga aynı doğrulukta değildir
(mIoU 44.3–48.3 bandı; VMamba-T en doğru modeldir). Bu nedenle nihai sunum tekil
"hız" sıralaması değil, doğruluk-gecikme Pareto düzlemidir: her (omurga × yığın)
hücresi düzlemde bir noktadır.

> **Şekil 4.1 — Doğruluk-gecikme Pareto düzlemi, 512²** *(yer tutucu; matris
> hücreleri tamamlandığında üretilecek)*. Yatay eksen medyan gecikme (ms, log),
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
gerçekleşmektedir** — dönüşüm süresi, graf boyutu ve yükleme süresi. FLOPs
tabanlı hiçbir analiz bu katmanları göremez, çünkü FLOPs yalnızca çıkarımın
aritmetik iş yükünü sayar.

Deney matrisinin bugünkü durumu:

**Tablo 4.5 — Dağıtım matrisi durumu (512², fp32; torch 2.13 / onnxruntime 1.28 / coremltools 9.0)**

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
birlikte okunmalıdır: hücre "çalışıyor" ancak §4.3.2'de gösterileceği gibi
pratikte dağıtılabilir olmaktan uzak bir maliyet profiliyle çalışıyor.

### 4.3.1 Yığın-Başına Gecikme: Dağıtılabilir-En-İyi Uçurumu

**Tablo 4.6 — Gecikme matrisi, 512², yığın 1, fp32, medyan (ms) — TASK-021 ilk tur**

| Yığın | ConvNeXt-T | Swin-T | VMamba-T |
|---|---|---|---|
| CoreML CPU+GPU | **63.9** | **63.9** | ❌ |
| CoreML ALL | 91.4 | 86.0 | ❌ |
| CoreML CPU_ONLY | 311 | 315 | ❌ |
| torch MPS (statik PSP) | 152 | 172 | **1 008** |
| torch CPU | 571 | 535 | **2 032** |
| ORT CPU | 644 | 714 | [ölçüm sürüyor]¹ |
| torch.compile | [ölçüm sürüyor] | [ölçüm sürüyor] | [ölçüm sürüyor] |
| ORT CoreML EP | [ölçüm sürüyor] | [ölçüm sürüyor] | [ölçüm sürüyor] |

¹ *VMamba ORT hücresi ayrı bir tur gerektirmektedir: modelin ORT oturumuna
yüklenmesi tek başına ~12 dakika sürmektedir (§4.3.2) ve ölçüm bu nedenle
bağımsız bir koşuya planlanmıştır. ORT ilk-koşu değeri (1.2 s, ısınmasız) dışa
aktarma turundan mevcuttur ancak §3.5 protokolüne uygun medyan değildir.*

Tablonun kritik okuması sütunlar arası değil, satırlar arasıdır. Her omurga için
"bugün gerçekten dağıtılabilir en iyi hücre"yi işaretleyelim: ConvNeXt-T ve
Swin-T için bu, CoreML CPU+GPU hücresidir — **63.9 ms**. VMamba-T için CoreML
kapalı, ORT pratik-dışı olduğundan en iyi dağıtılabilir hücre MPS eager'dır —
**1 008 ms**. Aradaki oran **~16×**'dir. Aynı iki model eager CPU'da yalnızca
3.6–3.8× ayrışıyordu (Tablo 4.2): uçurumun 3.6×'ten 16×'e açılmasının kaynağı
VMamba'nın yavaşlaması değil, **klasiklerin erişebildiği hızlandırma yollarına
VMamba'nın erişememesidir**. Avantaj tam olarak burada buharlaşıyor: yığın
merdiveninin her basamağı (eager → ORT → CoreML → ANE) klasikler için bir
hızlanma adımıyken, VMamba merdivenin ilk basamağında kalmaktadır.

İkinci gözlem: klasiklerde CoreML ALL hücresi (91.4 / 86.0 ms) CPU+GPU
hücresinden (63.9 ms) *yavaştır*. "ALL" isteğinin ANE'yi de içeren bir tercih
olduğu, ancak garanti olmadığı hatırlanırsa (§3.5.6) bu sıralama şaşırtıcıdır ve
§4.3.4'te enerji imzasıyla birlikte açıklanmaktadır.

Üçüncü gözlem: klasiklerde ORT CPU hücresi (644 / 714 ms), eager torch CPU'dan
(571 / 535 ms) hızlı değildir. Bu ilk turda ORT, klasikler için bir hızlandırma
basamağı olmaktan çok bir *taşınabilirlik* basamağı olarak konumlanmaktadır;
CoreML hücrelerinin sağladığı asıl kazanç (535–571 ms → 63.9 ms) ile
karşılaştırıldığında bu, Apple Silicon'da hızlanmanın genel graf çalışma
zamanından değil platforma özgü derleyiciden geldiğini düşündürmektedir.
Mikrobenchmark'ta gözlenen ters yönlü işaret (MiniMamba'da ORT CPU eager'dan
hızlıydı: L=196'da 13.5'e karşı 17.5 ms) bu ilişkinin model bileşimine bağlı
olduğunu göstermektedir; tam modellerdeki kalıcılığı sonraki turlarda ve
`torch.compile` / ORT CoreML EP hücreleri kapandığında netleşecektir
**[ölçüm sürüyor]**.

### 4.3.2 Dört Katmanlı Maliyet Modeli: FLOPs'un Kör Olduğu Katmanlar

Bölüm 3.5'te tanımlanan dört maliyet katmanı — **(a)** dönüşüm/derleme süresi,
**(b)** yükleme süresi, **(c)** dağıtım paketi boyutu, **(d)** çıkarım gecikmesi —
ONNX yolunda üç omurga için eksiksiz ölçülmüştür:

**Tablo 4.7 — ONNX yolu, dört katman (512², fp32; opset 17, TorchScript exporter)**

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
(1.2 s, klasiklerin yalnızca 1.3 katı). Buna karşılık katman (a) 448×, katman (b)
~7 249× şişmiştir. Bir cümleyle: **çıkarım hızı hayatta kalıyor; araç zinciri
çöküyor.** Üç modelin ağırlıkları karşılaştırılabilir boyuttayken (~244 MB
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

**Tablo 4.8 — CoreML yolu (512², fp32; coremltools 9.0)**

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
üretilmiş bir grafın içinde olmasıdır. Çözünürlük ölçeklendirmenin dışa aktarma
katmanına etkisi (VMamba ONNX 256² ve 1024² exportları) ayrıca
karakterize edilmektedir: **[ölçüm sürüyor]**.

Katman (d)'nin "hayatta kalması" salt bir teselli değildir; mikrobenchmark
verisi, dönüşümü *başarabilen* bir SSM'in çıkarımda ciddi kazanç elde
edebildiğini göstermektedir. MiniMamba'da ORT CPU çıkarımı eager torch'un
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

**Tablo 4.9 — ONNX graf operatör dökümü (en kalabalık altı op)**

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
süre tükettiği ayrı sorudur. Operatör-seviyesi çalışma zamanı profillemesi
(ORT profiler ile düğüm-başına süre dökümü) **[derin profilleme TASK-022'de —
ölçüm sürüyor]**.

### 4.3.4 ANE Yürütme Analizi: Enerji-İmzası Kanıtı

Apple Silicon'da verimlilik merdiveninin en üst basamağı ANE'dir (Apple Neural
Engine). Ancak §3.5.6'da belirtildiği gibi CoreML, hesaplama birimini şeffaf
olmayan biçimde seçer; `compute_units=ALL` bir tercihtir, garanti değil. Bu
nedenle "hangi model gerçekten ANE'de koşuyor" sorusu doğrudan ölçülmelidir. Bu
taslakta birincil kanıt, `powermetrics`'in ANE güç rayı telemetrisidir: ANE
kullanılmıyorsa rayın gücü sıfırdır, kullanılıyorsa aktif güç çekimi görülür.

**Tablo 4.10 — ANE enerji imzası (powermetrics, 200 ms örnekleme, boşta-düşülmüş)**

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
   vermektedir. ANE, gecikmede değil (Tablo 4.6'da ALL, CPU+GPU'dan yavaştır)
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

Protokol notu: §3.5.6 gereği "ANE'de çalışıyor" iddiasının resmî kanıtı Xcode
Core ML Performance Report'un katman-başına yürütme yeri dökümüdür; buradaki güç
telemetrisi ve süre-farkı gözlemleri güçlü ama dolaylı kanıtlardır. Xcode
doğrulaması ve katman-başına ANE yürütme oranı: **[TASK-023 sürüyor]** —
tamamlandığında bu alt bölüme ANE yürütme oranı sütunu eklenecektir.

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
   üç platformda üç ayrı kırılma** — Tablo 4.6'daki "torch MPS (statik PSP)"
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
koşulsuz açılan `torch.cuda.device()` bağlamı. Tablo 4.5'teki eager ✅ işareti
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

Eldeki verilerle AS2'nin cevabı şu şekilde katmanlanmaktadır. Teorik FLOPs
avantajı ile gerçekleşen verimlilik arasındaki fark, dağıtım yığını merdiveninde
basamak basamak büyümektedir: eager CPU'da 3.6–3.8× olan gecikme farkı (Tablo
4.2), her omurganın kendi dağıtılabilir-en-iyi hücresi karşılaştırıldığında 16×'e
(Tablo 4.6), enerji ekseninde ise ~25×'e (Tablo 4.4/4.10) açılmaktadır. Bu
büyümenin mekanizması, çıkarım aritmetiğinin yavaşlaması değildir — ONNX yolunda
çıkarım farkı yalnızca 1.3×'tir ve mikrobenchmark, dönüşümü başarabilen SSM'in
çıkarımda 15–22× kazanabildiğini göstermektedir. Mekanizma, ardışık taramanın
graf temsiline L ile lineer büyüyen yapı olarak girmesi (tam unroll, 390 758
düğüm, Gather ×139 798) ve bunun bedelinin çıkarım-öncesi katmanlarda —
dönüşümde (448×), pakette (614 MB saf graf yapısı), yüklemede (~7 249×) ve
CoreML örneğinde düpedüz dönüşüm başarısızlığında — ödenmesidir. Klasik
omurgaların da sürtünmesiz olmadığı (üç platformda üç cerrahi, §4.3.5), ancak
sürtünmenin yalnızca SSM'de imkânsızlığa dönüştüğü not edilmelidir. Eksik
hücreler (VMamba ORT medyanı, `torch.compile`, ORT CoreML EP, çözünürlük
taraması, TASK-022 çalışma-zamanı profili, TASK-023 Xcode raporu) bu resmi
nicel olarak inceltebilir; ancak matris durumunun kendisi (Tablo 4.5) ve dört
katmanlı maliyet asimetrisi (Tablo 4.7) bugünkü haliyle dahi tezin ana iddiasının
— avantajın özel çekirdeklere bağımlı olduğu ve genel amaçlı yığınlarda büyük
ölçüde kaybolduğu — ampirik çekirdeğini oluşturmaktadır. Yorum ve genelleme
Bölüm 5'e bırakılmıştır.

---

*Sayfa hedefi: ~10–12. Eksik hücreler (çözünürlük taraması, VMamba ORT medyanı,
torch.compile, ORT CoreML EP, TASK-022 profilleme, TASK-023 Xcode raporu)
tamamlandıkça tablolar güncellenecek; Şekil 4.1–4.2 veriler kapandığında
üretilecektir.*
