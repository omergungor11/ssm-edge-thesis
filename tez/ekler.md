# EKLER

> **Durum:** İskelet (TASK-036). Her ek, nihai içeriğin nereden üretileceğini işaret eder;
> tablo/şekil gövdeleri Faz 5 son geçişte bağlanacak. Yollar depo köküne görelidir.

---

## EK A — Tam Deney Sonuç Tabloları ve Ham Veri Envanteri

Tüm ham ölçümler `results/raw/` altındadır ve git'e dahildir ("başarısızlık da veridir"
ilkesiyle başarısız denemeler dahil). Her kayıt; ortam sürümleri, git commit'i ve termal
durumla damgalıdır. Aşağıdaki eşleme, metindeki her tabloyu besleyen ham dosyayı verir.

### A.1 Ham dosya → bölüm/tablo eşlemesi

| Ham dosya (`results/raw/`) | Üreten betik | Beslediği tablo/şekil |
|---|---|---|
| `ade20k_miou_progress.json`, `ade20k_swin_progress.json`, `ade20k_convnext_progress.json` | `src/models/eval_ade20k.py` | Tablo 4.1 (model kartları, mIoU doğrulaması) |
| `ade20k_conf_*_n250.npy`, `ade20k_conf_*_n2000.npy` | `src/models/eval_ade20k.py` | Tablo 4.1 (karışıklık matrisleri; n=250 hızlı / n=2000 tam) |
| `premise_L196.json`, `premise_L1024.json` (+ `mini_mamba_L196/L1024.onnx/.mlpackage`) | `src/premise/mamba_min.py` | Tablo 4.3 (MiniMamba öncül mikrobenchmark, TASK-008) |
| `latency_matrix_vmamba.jsonl`, `latency_matrix_swin.jsonl`, `latency_matrix_convnext.jsonl` | `src/benchmark/measure_matrix.py`, `src/benchmark/resolution_sweep.py`, `src/benchmark/round3_cells.py` | Tablo 4.2 (eager), 4.5 (MPS tepe bellek), 4.6 (gecikme × çözünürlük), 4.8 (gecikme matrisi, tüm turlar) |
| `energy_matrix.json` | `src/benchmark/energy_round.py` | Tablo 4.4 (net mJ/çıkarım), 4.14 (ANE enerji imzası, powermetrics) |
| `export_matrix.jsonl` | `src/export/export_cell.py` | Tablo 4.7 (dağıtım matrisi durumu), 4.9 (ONNX yolu), 4.10 (CoreML yolu), 4.11 (VMamba ONNX × çözünürlük), 4.12 (graf op dökümü) |
| `ort_profile_vmamba_top.json`, `ort_profile_swin_top.json`, `ort_profile_convnext_top.json` | `src/benchmark/round3_cells.py` | Tablo 4.13 (ORT CPU çalışma-zamanı operatör profili) |
| `xcode/xcode-perf-convnext-upernet-512-ane353of353.png`, `xcode/xcode-perf-swin-upernet-512-ane0of631.png` | Xcode Core ML Performance Report (elle) | Tablo 4.15 (resmî katman-yeri dökümü; görüntüler EK C'de) |
| `quant_matrix.jsonl` | `src/quant/quant_round1.py` | Tablo 4.16 (W-nicemleme matrisi), 4.17 (vekil metrik), 4.19 (ORT dinamik INT8) |
| `quant_miou_{convnext,swin}_{fp32export,w8,w4}.json` | `src/quant/eval_quant_miou.py` | Tablo 4.16-4.17 (gerçek ΔmIoU sütunları) |
| `activation_stats_{vmamba,swin,convnext}.json` | `src/quant/activation_stats.py` | Tablo 4.18 (aktivasyon istatistikleri × çözünürlük) |
| `reform_input.pt` | `src/reformulation/capture.py` | Tablo 4.20 (gerçek ağırlık + gerçek aktivasyonla doğrulama girdisi) |
| `reform_matrix.jsonl`, `reform_seq.jsonl`, `reform_{blocked,ane}{32,64,128}.jsonl` | `src/reformulation/run_matrix.py`, `verify.py` | Tablo 4.20 (sayısal doğrulama), 4.21 (graf/yükleme), 4.22 (gecikme + ANE op%) |
| `validate_resnet50.jsonl` (+ `resnet50.mlpackage`) | `src/benchmark/validate_resnet50.py` | §3.5 harness doğrulaması (≤%1.2 sapma) |

**Ölçüm artefaktları** (tablo değil, kanıt): `*_upernet_512.mlpackage`,
`*_w8_linear.mlpackage`, `*_w4_palette.mlpackage`, `*_int8.onnx`,
`reform_*.onnx/.mlpackage` — paket boyutu / yükleme süresi ölçümlerinin nesneleri.

### A.2 Şekiller

`results/figures/fig-4.1-pareto`, `fig-4.2-scaling`, `fig-4.3-cost-layers`,
`fig-4.4-energy`, `fig-4.5-quant` (PDF+PNG) — tümü `src/analysis/make_figures.py`
tarafından yalnızca `results/raw/` girdilerinden üretilir (elle düzenleme yok).

> **Faz 5 işi:** Tabloların tam (kırpılmamış) sürümleri — metinde yer kısıtıyla
> özetlenen satırlar dahil — bu eke JSONL kayıtlarından dökülecek.

---

## EK B — Ölçüm Altyapısı: Modül Haritası ve Yeniden Üretim

### B.1 Modül haritası (`src/`)

| Modül | İçerik |
|---|---|
| `src/benchmark/` | Ölçüm harness'ı: `harness.py` (ısınma + medyan/P99 istatistik + JSONL yazım), `runners.py` (yığın-agnostik torch/ORT/CoreML koşucuları), `env.py` (ortam sürümü damgalama), `stats.py`, `measure_matrix.py` (ana gecikme matrisi), `resolution_sweep.py` (çözünürlük taraması), `energy_round.py` (powermetrics enerji), `round3_cells.py` (ORT profilleri), `validate_resnet50.py` (harness doğrulaması) |
| `src/models/` | mmseg'siz, anahtar-uyumlu saf-PyTorch yükleyiciler: `vmamba_upernet.py`, `swin_upernet.py`, `convnext_upernet.py` + `eval_ade20k.py` (mIoU değerlendiricisi) |
| `src/export/` | `export_cell.py` — export matrisi hücre koşucusu (ONNX/CoreML; başarısızlıklar da kayıtlı) |
| `src/quant/` | `quant_round1.py` (CoreML W8/W4, ORT INT8), `eval_quant_miou.py`, `activation_stats.py` |
| `src/data/` | `ade20k.py` — ADE20K veri katmanı |
| `src/premise/` | `mamba_min.py` — Faz 0 MiniMamba öncül mikrobenchmark'ı |
| `src/reformulation/` | Faz 4 — EK D'ye bakınız |
| `src/analysis/` | `make_figures.py` — tez şekilleri |

### B.2 Tek komutluk yeniden üretim (README ile tutarlı)

Ortam: Apple Silicon Mac (test edilen: baz M5, 24 GB), Python 3.12.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt

python src/models/eval_ade20k.py 2000 --model vmamba   # mIoU doğrulaması (Tablo 4.1)
python src/export/export_cell.py vmamba onnx 512       # Export matrisi hücresi (Tablo 4.7)
python src/benchmark/measure_matrix.py                 # Gecikme matrisi (Tablo 4.8)
python src/quant/quant_round1.py                       # Nicemleme (Tablo 4.16-4.19)
python src/reformulation/run_matrix.py                 # Yeniden formülasyon (Tablo 4.20-4.22)
```

Checkpoint'ler ve ADE20K git dışıdır (edinim adresleri README'de); VMamba üçüncü-parti
kodu `third_party/VMamba` olarak klonlanır ve CPU-only ortam için gereken iki satırlık
yamalar `tez-docs/vmamba-yamalari.md`'de `[TEZ YAMASI]` etiketiyle belgelidir.

---

## EK C — Ortam ve Sürüm Belgeleme

### C.1 Kilitli sürüm kümesi

Tüm ölçümler tek makinede, tek sürüm kümesiyle alınmıştır (tartışma için Bölüm 5 tehdit
analizi):

| Bileşen | Sürüm |
|---|---|
| Donanım | Apple baz M5, 24 GB birleşik bellek (şebeke gücü) |
| İşletim sistemi | macOS 26.5.2 |
| Python | 3.12.13 |
| PyTorch | 2.13.0 |
| ONNX Runtime | 1.28.0 |
| coremltools | 9.0 |

Tam paket dökümü: `requirements.txt` + her JSONL kaydına gömülü ortam bloğu
(`src/benchmark/env.py`). Makine/termal koşullar: `tez-docs/ortam-mac.md`.

### C.2 Xcode Core ML Performance Report kanıtları

Katman-yeri (CPU/GPU/ANE) atamaları Xcode'un resmî raporuyla doğrulanmıştır
(Tablo 4.15'in kaynağı):

- `results/raw/xcode/xcode-perf-convnext-upernet-512-ane353of353.png` — ConvNeXt-T+UPerNet: 353/353 op ANE (%100)
- `results/raw/xcode/xcode-perf-swin-upernet-512-ane0of631.png` — Swin-T+UPerNet: 0/631 op ANE (%0, GPU fallback)

*(VMamba-T için rapor yoktur: model Core ML'e dönüşemediğinden ölçülecek paket oluşmaz —
Bölüm 4.3.)*

> **Faz 5 işi:** Görüntüler ek sayfasına gömülecek; powermetrics örnekleme
> yapılandırması (200 ms) ve boşta-düşme prosedürü §3.5'ten buraya özetlenecek.

---

## EK D — Yeniden Formüle Edilmiş Tarama Operatörü (Faz 4)

### D.1 Dosya haritası (`src/reformulation/`)

| Dosya | Rol |
|---|---|
| `common.py` | `SS2DBase` — üç formun paylaştığı SS2D bloğu iskeleti; `KD` sabiti |
| `capture.py` | Gerçek VMamba ağırlıkları + gerçek ara aktivasyonların yakalanması → `results/raw/reform_input.pt` |
| `ss2d_seq.py` | `seq` formu — adım adım özyinelemeli referans (L adımlık döngü; trace'te tam unroll) |
| `ss2d_blocked.py` | `blocked` formu — blok kapalı form, iki varyant (aşağıda) |
| `ss2d_ane.py` | `ane` formu — ANE-dostu yerleşim (Apple ANE ilkeleri: (B,C,1,S) formatı, reshape/transpose minimizasyonu) |
| `verify.py` | Üç formun üçüncü-parti selective scan referansına karşı sayısal doğrulaması (Tablo 4.20) |
| `run_matrix.py` | Ölçüm matrisi: form × blok boyutu {32, 64, 128} → `reform_*.jsonl`, `reform_matrix.jsonl` (Tablo 4.21-4.22) |

### D.2 Çürüme-matrisi (decay) formülasyonunun özeti

`d_state=1` için özyineleme kanal başına skalerdir: `h_t = a_t·h_{t-1} + b_t`,
`a_t = exp(Δ_t·A)`. `S_t = Σ_{i≤t} Δ_i·A` (log-birikimli toplam) ile blok uzunluğu P
içinde iki kapalı form:

1. **`cumsum` varyantı** (reçetedeki bölmeli form): `h_t = e^{S_t}·(h_0 + Σ_{i≤t} b_i·e^{-S_i})`.
   Gerçek aktivasyonlarda blok içi min(S) ≈ −513 (P=64) ölçüldü → `e^{-S}` fp32'de taşar
   (üst sınır ~e^88) → NaN. **Kayıt için tutulur; başarısızlık da veridir**
   (`reform_matrix.jsonl` 'verify' kayıtları).
2. **`decay` varyantı** (stabil, varsayılan): alt-üçgen çürüme matrisi
   `T[t,i] = e^{S_t − S_i}` (i ≤ t; her giriş ≤ 1 → taşma imkânsız),
   `h_blok = T @ b`; bloklar arası taşıma `h_t += e^{S_t}·h_0` (alttan taşma → 0,
   zararsız). Maske exp'ten **önce** uygulanır (üst üçgende S_t − S_i > 0 → önce exp
   sonra maske `inf·0 = NaN` üretir; `−10⁴ → exp → 0`).

Bu, Mamba-2/SSD'nin "chunked" formülasyonunun `d_state=1` özel hâlidir: blok-köşegen
dikkat matrisi + skaler taşıma. Kullanılan op kümesi yalnızca `cumsum + exp + matmul` —
tümü ONNX/CoreML'in yerli operatörleridir; L uzunluklu unroll yerine L/P adımlık kısa
döngü kalır.

> **Faz 5 işi:** Formül türetimi Bölüm 4.5/§3'teki gösterimle hizalanacak; blok boyutu
> taraması (32/64/128) tam tabloları A.1'deki `reform_*` kayıtlarından dökülecek.
