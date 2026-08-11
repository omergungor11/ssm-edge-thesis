# Deney Planı — 16 Hafta

**Donanım:** RTX 5070 (12 GB varsayımı — `nvidia-smi` ile doğrulanacak) + Apple M5 Pro (ANE)
**Başlangıç:** 10 Ağustos 2026
**Hedef bitiş:** ~30 Kasım 2026

---

## Tasarım ilkesi

Her faz **tek başına raporlanabilir bir sonuç** üretir. Faz 4 (özgün katkı) tamamen başarısız olsa bile Faz 1-3 bir tez oluşturur. Risk sona itilmiştir, başa değil.

Her hafta sonunda **tek bir somut çıktı** olmalı: bir tablo, bir grafik, bir çalışan betik. "Okudum, düşündüm" bir çıktı değildir.

---

## FAZ 0 — Öncülü doğrula ve altyapıyı kur *(Hafta 1-2)*

Bu fazın amacı tezi yazmak değil, **tezin öncülünün doğru olduğunu kendi makinende görmek.** Eğer ONNX'te Mamba yavaş değilse, tez konusu değişmeli — bunu 16. haftada değil 1. haftada öğren.

### Hafta 1 — Öncül doğrulama

| Gün | İş | Çıktı |
|---|---|---|
| 1 | Ortam kurulumu: PyTorch + CUDA, `mamba-ssm`, `onnxruntime`, `coremltools`. `nvidia-smi` ile VRAM doğrulaması | Çalışan ortam, sürüm listesi |
| 2 | Küçük bir VMamba-T modelini yükle, PyTorch'ta CUDA çekirdeğiyle çalıştır, gecikmeyi ölç | Referans gecikme sayısı |
| 3 | **Aynı modeli ONNX'e export et.** Beklenen: sorun çıkacak. Hata mesajlarını, graf boyutunu, yükleme süresini kaydet | Export logu + graf boyutu |
| 4 | ONNX Runtime'da çalıştır (CPU + CUDA EP), gecikmeyi ölç. PyTorch ile karşılaştır | **İlk yavaşlama oranı** |
| 5 | Aynısını M5 Pro'da CoreML ile dene | ANE export raporu |
| 6-7 | Bulguları tek sayfalık bir notta topla: *"Öncül doğrulandı mı?"* | `tez-docs/oncul-dogrulama.md` |

> **Karar noktası (Hafta 1 sonu):** Yavaşlama gözlemlendi mi? Evet → devam. Hayır → konuyu yeniden değerlendir (bu bir başarısızlık değil, ucuz bir keşif).

### Hafta 2 — Ölçüm altyapısı

| İş | Çıktı |
|---|---|
| Ölçüm harness'ı yaz: ısınma, senkronizasyon, termal bekleme, NVML/`powermetrics` entegrasyonu, istatistik toplama (medyan/ortalama/std/P99) | `src/benchmark/` — **tezin en çok yeniden kullanılan kodu** |
| Harness'ı bilinen bir modelle (ResNet-50) doğrula — bilinen sayıları üretebiliyor mu? | Doğrulama raporu |
| Veri kümelerini indir ve hazırla: ADE20K, Cityscapes, ImageNet alt kümesi | Hazır veri boru hattı |
| Bölüm 3.5 (Ölçüm protokolü) taslağını yaz | Tez metni ~4 sayfa |

---

## FAZ 1 — Doğruluk eşitleme *(Hafta 3-5)*

Amaç: omurgaları **aynı doğruluk çizgisine** getirmek. Farklı doğruluktaki modellerin hızını karşılaştırmak anlamsızdır.

| Hafta | İş | Çıktı |
|---|---|---|
| 3 | Segmentasyon boru hattını kur (UPerNet başlığı sabit). VMamba-T'yi ADE20K'da eğit, 512×512 | İlk mIoU sayısı |
| 4 | ViT-S/DeiT-S ve ConvNeXt-T'yi **birebir aynı reçeteyle** eğit | Üç omurga, karşılaştırılabilir mIoU |
| 5 | EfficientViM ekle (verimli SSM referansı). Doğruluk eşitleme tablosunu tamamla. Gerekirse epoch/lr ayarıyla çizgiyi hizala | **Tablo 4.1 — doğruluk eşitleme** |

**12 GB notu:** mixed precision (AMP) + gradient checkpointing zorunlu. Batch 8-16 arası. Sığmazsa 448×448'e in — bilimsel geçerlilik bozulmaz, yeter ki **tüm omurgalar için aynı** olsun.

**Risk:** Eğitim 12 GB'a sığmazsa → dondurulmuş ImageNet ön-eğitimli omurga + sadece başlık eğitimi. Doğruluk düşer ama karşılaştırma geçerliliğini korur.

---

## FAZ 2 — Dağıtım yığını matrisi *(Hafta 6-8)*  ← **Tezin kalbi**

| Hafta | İş | Çıktı |
|---|---|---|
| 6 | Dört omurgayı dört yığına export et: PyTorch/CUDA, `torch.compile`, ONNX Runtime, CoreML. Başarısız export'ları da belgele — **başarısızlık da veridir** | Export matrisi (başarı/başarısızlık + gerekçe) |
| 7 | Tam gecikme/bellek/enerji ölçümü. Çözünürlük taraması: 256 / 512 / 768 / 1024 | **Ana sonuç tablosu ve eğrileri** |
| 8 | Operatör seviyesinde profilleme: zaman nerede harcanıyor? `Loop` yükü, bellek kopyaları, ANE'ye düşmeyen op'lar. Xcode ile ANE yürütme oranı doğrulaması | **Darboğaz analizi — tezin "neden" cevabı** |

**Bu fazın sonunda tezin ana iddiası kanıtlanmış ya da çürütülmüş olur.** Bölüm 4.2 ve 4.3 bu haftalarda yazılır.

---

## FAZ 3 — Nicemlemenin yoğun tahmine transferi *(Hafta 9-11)*

| Hafta | İş | Çıktı |
|---|---|---|
| 9 | PTQ4VM'i uygula (sınıflandırma referansı ile doğrula, makaledeki sayıları üret) | Yöntem doğrulaması |
| 10 | Aynı PTQ'yu segmentasyon modellerine uygula. Doğruluk kaybını sınıflandırmayla karşılaştır | **Tablo: sınıflandırma vs yoğun tahmin kaybı** |
| 11 | Çözünürlüğün aykırı değer dağılımına etkisini analiz et (aktivasyon histogramları, kanal-bazlı istatistikler). Nicemlenmiş modelleri tekrar dört yığında ölç | Aykırı değer analizi + nicemlenmiş gecikme matrisi |

---

## FAZ 4 — Yeniden formülasyon *(Hafta 12-13)*  ← **Özgün katkı, riskli**

| Hafta | İş | Çıktı |
|---|---|---|
| 12 | `selective scan`'i parçalı/sabit-blok forma yeniden yaz (Mamba-2 SSD ikiliğinden yararlan). ONNX `Loop` yerine matris çarpımı ağırlıklı graf üret | Yeni operatör implementasyonu |
| 13 | Apple'ın ANE reçetesini uygula: (B,C,1,S) düzeni, split/concat parçalama, reshape/transpose minimizasyonu. Ölç ve karşılaştır | **Kazanç tablosu** |

**Risk yönetimi:** Bu faz beklenen kazancı vermezse Bölüm 4.5 "denenen yaklaşım ve neden yetersiz kaldığı" olarak yazılır. Tez ayakta kalır. **Bu fazı asla Faz 1-3'ün önüne alma.**

---

## FAZ 5 — Yazım ve toparlama *(Hafta 14-16)*

| Hafta | İş |
|---|---|
| 14 | Bölüm 5 (Tartışma) + Bölüm 6 (Sonuç). Eksik ablasyonları tamamla |
| 15 | Bölüm 1 (Giriş) ve Özet. Kaynakça düzeni. Ekler |
| 16 | Bütünsel okuma, tutarlılık kontrolü, şekil/tablo numaralandırma, dil düzeltmesi. Kod deposunu temizle ve yayınla |

---

## Sürekli işler (her hafta)

- [Awesome-Vision-Mamba](https://github.com/ReaFly/Awesome-Vision-Mamba) kontrolü — alan hızlı, çakışma riskini erken gör
- Ölçüm sonuçlarını **ham halde** sakla (`results/raw/`), asla sadece grafiği tutma
- Ortam sürümlerini her ölçümde kaydet — tekrarlanabilirlik için
- Haftalık 1 saat: o haftanın sonuçlarını tez metnine yaz. **Biriktirme.**

---

## Kritik başarısızlık senaryoları ve çıkış yolları

| Senaryo | Çıkış yolu |
|---|---|
| ONNX'te yavaşlama gözlenmiyor (öncül yanlış) | Konuyu ANE/CoreML eksenine daralt — orada op desteği kısıtı kesin |
| 12 GB eğitim için yetmiyor | Dondurulmuş omurga + başlık eğitimi; ya da 448px |
| CoreML export hiç çalışmıyor | **Bu zaten bir bulgudur.** "Dağıtılamıyor" sonucu, "yavaş" sonucundan daha güçlüdür |
| Faz 4 kazanç vermiyor | Negatif sonuç olarak raporla; tez Faz 1-3 üzerine oturur |
| Rakip bir makale çıkıyor | Konumlandırmayı daralt (ANE + yoğun tahmin ekseni), tümüyle terk etme |

---

## ŞİMDİ YAPILACAK TEK ŞEY

Yarın sabah tek bir iş var, gerisi bekleyebilir:

> **`nvidia-smi` çalıştır, çıktıyı kaydet. Ardından Python ortamını kur ve `mamba-ssm` paketinin RTX 5070'te (Blackwell mimarisi) derlenip derlenmediğini gör.**

Bu ikincisi önemli: `mamba-ssm` özel CUDA çekirdekleri içerir ve yeni mimarilerde derleme sorunu çıkarabilir. Çıkarsa, bu **tezin ilk bulgusudur** — not al.
