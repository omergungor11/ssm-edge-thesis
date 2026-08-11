# Faz 0: Öncül Doğrulama + Ölçüm Altyapısı *(Hafta 1-2)*

> **Bu fazın amacı tezi yazmak değil.** Tezin öncülünün — "SSM'lerin verimlilik avantajı genel dağıtım yığınlarında kaybolur" — kendi makinende doğru olduğunu görmek.
> Öncül yanlışsa bunu 1. haftada öğren, 16. haftada değil.

---

## TASK-001: Git Repo Init + İlk Commit

**Kim**: 🤖 Claude · **Complexity**: S · **Status**: PENDING · **Dependencies**: -

### Açıklama
Proje dizini şu an git repo değil. Belgeler kayıt altına alınmadan deneye başlanmaz.

### Kabul kriterleri
- [ ] `.gitignore` — Python, veri kümeleri, model ağırlıkları, `results/raw/` hariç tutulmaz (ham veri saklanır), `.venv/`, `__pycache__/`, `*.onnx`, `*.mlpackage`
- [ ] `git init` + ilk commit: mevcut 4 belge + task yapısı
- [ ] Commit mesajı formatı: `docs(TASK-001): ...` (Claude attribution **yok**)
- [ ] Remote bağlanacaksa not düşülür (şimdilik opsiyonel)

---

## TASK-002: Mac Ortam Kurulumu

**Kim**: 🤖 Claude · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-001

### Açıklama
M5 Pro tarafı = CoreML/ANE ölçüm ayağı. Sistem Python'ı **3.14.5** — `coremltools` ve `mamba-ssm` bu sürümde tekerlek yayınlamıyor. Ayrı venv şart.

### Kabul kriterleri
- [ ] `uv venv --python 3.12` ile izole ortam (sistem Python'ına dokunulmaz)
- [ ] `torch` (MPS backend doğrulanmış), `coremltools`, `onnx`, `onnxruntime`
- [ ] `requirements.txt` / `pyproject.toml` — **sürümler sabitlenmiş**
- [ ] `tez-docs/ortam-mac.md`: tüm sürümler + macOS build + çip modeli kaydı

### Notlar
- `powermetrics` enerji ölçümü için `sudo` istiyor — harness tasarımında hesaba kat
- Ölçüm yapılırken Mac'in fişte ve düşük güç modunun kapalı olduğundan emin ol

---

## TASK-003: RTX 5070 Ortamı + `mamba-ssm` Derleme Testi

**Kim**: 🧑 **Sadece Ömer** · **Complexity**: M · **Status**: PENDING · **Dependencies**: -

### Açıklama
Bu makine Mac; NVIDIA GPU **ayrı bir sistemde**. Claude oraya erişemiyor, çıktıların yapıştırılması gerekiyor.

`mamba-ssm` özel CUDA çekirdekleri içeriyor ve RTX 5070 **Blackwell** mimarisi — derleme sorunu çıkabilir. **Çıkarsa panikleme: bu tezin ilk bulgusudur.**

### Kabul kriterleri
- [ ] `nvidia-smi` çıktısı kaydedildi → **VRAM doğrulaması** (12 GB mı, 16 GB mı?)
- [ ] CUDA toolkit + sürücü sürümü kaydedildi
- [ ] PyTorch CUDA build kuruldu, `torch.cuda.is_available()` True
- [ ] `pip install mamba-ssm` denendi — **başarılı olsa da olmasa da tam log kaydedildi**
- [ ] `causal-conv1d` derleme durumu kaydedildi
- [ ] `tez-docs/ortam-cuda.md` yazıldı

### Çalıştırılacak komutlar
```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
pip install mamba-ssm causal-conv1d 2>&1 | tee mamba-ssm-build.log
```

> **VRAM neden kritik:** 12 GB ise eğitim matrisi daralıyor (batch 8-16, gerekirse 448px). 16 GB ise genişletilebilir. Faz 1'in tüm tasarımı buna bağlı.

---

## TASK-004: VMamba-T Referans Gecikme Ölçümü

**Kim**: 🤝 Ortak · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-003

### Açıklama
Özel CUDA çekirdeğiyle çalışan PyTorch — bu **üst sınır**. Sonraki tüm yığınlar buna göre ne kadar kaybettiğiyle ölçülecek.

### Kabul kriterleri
- [ ] VMamba-T (veya Vim-T) ağırlıkları indirildi ve yüklendi
- [ ] İleri geçiş çalışıyor, çıktı şekli doğrulandı
- [ ] Gecikme ölçüldü: ısınma sonrası, `torch.cuda.synchronize()` ile, ≥100 tekrar
- [ ] Medyan / ortalama / std / P99 raporlandı
- [ ] Bellek tepe noktası (`torch.cuda.max_memory_allocated`) kaydedildi
- [ ] **Referans gecikme sayısı** belgeye yazıldı

### Notlar
- TASK-003'te `mamba-ssm` derlenmediyse: PyTorch saf-Python fallback ile ölç ve **bunu ayrıca not et** — zaten bulgunun bir parçası

---

## TASK-005: ONNX Export Denemesi

**Kim**: 🤝 Ortak · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-004

### Açıklama
**Beklenti: sorun çıkacak.** Amaç sorunu çözmek değil, karakterize etmek.

### Kabul kriterleri
- [ ] `torch.onnx.export` denendi (hem TorchScript hem dynamo yolu)
- [ ] Export başarılıysa: **graf boyutu (MB), düğüm sayısı, `Loop` düğümü sayısı** kaydedildi
- [ ] Export başarısızsa: **tam hata mesajı + hangi operatörde takıldığı** kaydedildi
- [ ] Model yükleme süresi ölçüldü (küçük değil — ONNX Runtime #27796'da darboğazın parçası)
- [ ] `tez-docs/export-matrisi.md` başlatıldı

> **Başarısızlık da veridir.** "Export edilemiyor" bulgusu, "yavaş" bulgusundan daha güçlüdür.

---

## TASK-006: ONNX Runtime Ölçümü → İlk Yavaşlama Oranı

**Kim**: 🤝 Ortak · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-005

### Açıklama
Tezin öncülünün ilk sayısal kanıtı burada çıkıyor.

### Kabul kriterleri
- [ ] ONNX Runtime CPU EP ile gecikme ölçüldü
- [ ] ONNX Runtime CUDA EP ile gecikme ölçüldü
- [ ] TASK-004 referansına oran hesaplandı → **yavaşlama katsayısı**
- [ ] ORT profilleme açık çalıştırıldı (`enable_profiling`), en pahalı operatörler listelendi
- [ ] `Loop` operatörünün toplam süredeki payı hesaplandı

### Karşılaştırma hedefi
ONNX Runtime issue #27796: 9.6M param Mamba, Apple M3'te gerçek zamanın **17 katı** yavaş. Bizim sayımız ne?

---

## TASK-007: CoreML Export + M5 Pro / ANE Ölçümü

**Kim**: 🤝 Ortak · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-002, TASK-005

### Kabul kriterleri
- [ ] `coremltools` ile dönüşüm denendi (ML Program formatı)
- [ ] Desteklenmeyen operatörler listelendi
- [ ] Export başarılıysa: CPU / GPU / ANE compute unit'lerinde ayrı ayrı ölçüldü
- [ ] **Hangi katmanların ANE'ye düştüğü** kaydedildi (tam doğrulama TASK-023'te)
- [ ] Export tamamen başarısızsa → **bu zaten bir bulgudur**, gerekçesiyle belgelendi

---

## TASK-008: KARAR NOKTASI — Öncül Doğrulama Raporu

**Kim**: 🤝 Ortak · **Complexity**: S · **Status**: PENDING · **Dependencies**: TASK-006, TASK-007

### Açıklama
Hafta 1'in tek çıktısı. Tek sayfa. Soru tek: **öncül doğrulandı mı?**

### Kabul kriterleri
- [ ] `tez-docs/oncul-dogrulama.md` yazıldı
- [ ] Yığın-başına gecikme tablosu (PyTorch/CUDA · ONNX CPU · ONNX CUDA · CoreML)
- [ ] Yavaşlama oranları
- [ ] Export başarı/başarısızlık matrisi
- [ ] **Net karar cümlesi**

### Karar ağacı

| Sonuç | Aksiyon |
|---|---|
| Belirgin yavaşlama var | ✅ Faz 0 Hafta 2'ye devam, plan aynen |
| Yavaşlama yok / marjinal | ⚠️ Konuyu **ANE/CoreML eksenine daralt** — orada op desteği kısıtı kesin. Tez ölmez, odak kayar |
| Export hiçbir yığında çalışmıyor | ✅ Bu **daha güçlü** bir sonuç — tez "dağıtılamıyor" ekseninde yeniden çerçevelenir |

---

## TASK-009: Ölçüm Harness'ı

**Kim**: 🤖 Claude · **Complexity**: L · **Status**: PENDING · **Dependencies**: TASK-008

### Açıklama
**Tezin en çok yeniden kullanılan kodu.** Her fazda bu çalışacak. Baştan doğru yazılır.

### Kabul kriterleri
- [ ] `src/benchmark/` paket yapısı
- [ ] Isınma turları + termal stabilizasyon bekleme (3 W / 30 sn / 65 °C eşikleri)
- [ ] Doğru senkronizasyon (CUDA event / MPS / CoreML'e özgü)
- [ ] İş parçacığı ve sistem gürültüsü izolasyonu
- [ ] Enerji ölçümü: NVML (NVIDIA) + `powermetrics` (Apple)
- [ ] Bellek tepe noktası takibi
- [ ] İstatistik: medyan, ortalama, std, P99 — **ham ölçümler `results/raw/` altına yazılır**
- [ ] Her çalıştırmada ortam sürümleri otomatik kaydedilir (tekrarlanabilirlik)
- [ ] Yığın-agnostik arayüz: aynı API ile PyTorch / ONNX / CoreML ölçülebilir

---

## TASK-010: Harness Doğrulaması (ResNet-50)

**Kim**: 🤝 Ortak · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-009

### Açıklama
Harness bilinen bir modelle bilinen sayıları üretemiyorsa, Mamba ölçümlerine güvenilemez.

### Kabul kriterleri
- [ ] ResNet-50 dört yığında da ölçüldü
- [ ] Sonuçlar yayınlanmış referans değerlerle karşılaştırıldı
- [ ] Sapma %10'un altında — değilse sebebi bulundu ve düzeltildi
- [ ] Ölçüm tekrarlanabilirliği doğrulandı (aynı koşulda 3 bağımsız çalıştırma, std makul)

---

## TASK-011: Veri Kümesi Boru Hattı

**Kim**: 🤖 Claude · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-002

### Kabul kriterleri
- [ ] ADE20K indirildi + doğrulandı (150 sınıf, semantik segmentasyon — **birincil**)
- [ ] Cityscapes indirildi (1024×2048 — çözünürlük ölçeklendirme deneyi)
- [ ] ImageNet-1k alt kümesi (sınıflandırma referansı + PTQ kalibrasyonu)
- [ ] Ortak `Dataset` / dönüşüm katmanı — **tüm omurgalar için birebir aynı artırma**
- [ ] Disk kullanımı belgelendi

---

## TASK-012: Bölüm 3.5 (Ölçüm Protokolü) Taslağı

**Kim**: 🤖 Claude · **Complexity**: M · **Status**: PENDING · **Dependencies**: TASK-009

### Açıklama
Deney tasarımını yazmak, tasarımdaki hatayı bulmanın en ucuz yoludur. Deneyler bitmeden yazılır.

### Kabul kriterleri
- [ ] 3.5.1 Isınma ve termal stabilizasyon
- [ ] 3.5.2 Senkronizasyon ve zamanlama doğruluğu
- [ ] 3.5.3 İş parçacığı ve sistem gürültüsü izolasyonu
- [ ] 3.5.4 Enerji ölçümü (NVML / `powermetrics`)
- [ ] 3.5.5 İstatistiksel raporlama
- [ ] 3.5.6 **Yürütme yeri doğrulaması** — ANE'de çalıştığının Xcode profiliyle kanıtı
- [ ] ~4 sayfa, tez formatında
