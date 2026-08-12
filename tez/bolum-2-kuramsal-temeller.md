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
