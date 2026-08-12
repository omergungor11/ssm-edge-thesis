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
