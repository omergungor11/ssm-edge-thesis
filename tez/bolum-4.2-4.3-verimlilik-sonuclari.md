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
