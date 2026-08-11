# Ortam Kaydı — Mac (ANE Ölçüm Ayağı)

> TASK-002 çıktısı · Kayıt tarihi: 11 Ağustos 2026
> Tekrarlanabilirlik: tam sürüm listesi → `requirements.lock`

## Donanım

| Alan | Değer |
|---|---|
| Model | MacBook Pro |
| Çip | **Apple M5** (system_profiler çıktısı — "M5 Pro" değil, baz M5) |
| Çekirdek | 10 (4 performans + 6 verimlilik) |
| RAM | 24 GB birleşik bellek |
| macOS | 26.5.2 (build 25F84), Darwin 25.5.0, arm64 (T8142) |

> ⚠️ Çip beklenenden farklı çıktı: donanım envanterinde "M5 Pro" yazıyordu, sistem **baz M5** raporluyor.
> Tez metninde donanım tablosuna doğru adıyla girilmeli. ANE mevcut — ölçüm ayağı etkilenmez.

## Yazılım ortamı

| Paket | Sürüm | Not |
|---|---|---|
| Python | 3.12.13 | `uv venv` — sistem 3.14.5 **kullanılmıyor** (coremltools/mamba-ssm tekerlek yok) |
| torch | 2.13.0 | MPS aktif ✅ |
| torchvision | 0.28.0 | |
| coremltools | 9.0 | ⚠️ uyarı: "torch 2.13 test edilmedi, en son test edilen 2.7.0" — aşağıya bak |
| onnx | 1.22.0 | |
| onnxruntime | 1.28.0 | EP'ler: **CoreMLExecutionProvider**, AzureEP, CPUExecutionProvider |
| timm | 1.0.28 | |
| numpy | 2.5.2 | |

Doğrulama: MPS smoke test geçti (64×64 matmul, device="mps").

## Bilinen riskler / notlar

1. **coremltools 9.0 × torch 2.13 uyumsuzluk uyarısı.** Resmî test 2.7.0'a kadar. CoreML export
   (TASK-007) beklenmedik hata verirse ilk deneme: `uv pip install "torch==2.7.0"` ile ikinci bir
   venv kurup export'u orada yapmak. Bu durumda hangi sürümle export edildiği ölçüm kaydına yazılır.
2. **onnxruntime'da CoreMLExecutionProvider var** — ONNX yolu üzerinden de ANE'ye erişim denenebilir.
   Bu, ölçüm matrisine ek bir hücre adayı: ORT-CoreML EP (TASK-006/007 sırasında değerlendir).
3. `powermetrics` sudo ister — enerji ölçümü interaktif oturum gerektirir (TASK-009 tasarım notu).
4. Ölçüm sırasında: fişte, Low Power Mode kapalı, ekran kapanması engellenmiş (`caffeinate`).
