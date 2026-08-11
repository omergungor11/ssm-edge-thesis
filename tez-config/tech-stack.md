# Tech Stack

> Bu bir ML araştırma projesi — web stack yok.

## Dil / Ortam
- Python **3.12** (uv ile izole venv — sistem 3.14.5 kullanılmaz: coremltools/mamba-ssm tekerlek yok)
- Paket yöneticisi: **uv** (`uv venv`, `uv pip`), sürümler lock'lu

## ML Çekirdek
- PyTorch ≥2.x (CUDA build 5070'te, MPS build Mac'te)
- `mamba-ssm` + `causal-conv1d` (özel CUDA çekirdekleri — Blackwell'de derleme riski, TASK-003)
- timm / mmsegmentation türevi eğitim kodu (Faz 1'de kesinleşecek)

## Modeller
- VMamba-T (SSM) · ViT-S/DeiT-S (Transformer) · ConvNeXt-T (CNN) · EfficientViM (verimli SSM)
- Başlık: UPerNet (sabit)

## Dağıtım yığınları (ölçüm hedefleri)
- PyTorch + özel CUDA çekirdeği (referans/üst sınır)
- `torch.compile`
- ONNX Runtime (CPU EP, CUDA EP)
- CoreML / ANE (`coremltools`, ML Program formatı)

## Veri kümeleri
- ADE20K (birincil, 150 sınıf) · Cityscapes (çözünürlük deneyi) · ImageNet-1k alt kümesi (PTQ kalibrasyonu)

## Ölçüm / Profilleme
- NVML (NVIDIA enerji) · `powermetrics` (Apple enerji, sudo ister)
- ONNX Runtime profiler · Xcode Core ML Performance Report (ANE doğrulaması)
- `src/benchmark/` — kendi harness'ımız (TASK-009)

## Donanım
- Eğitim + CUDA ölçüm: RTX 5070 (Blackwell; VRAM 12 GB varsayım, doğrulanacak)
- ANE ölçüm: Apple M5 Pro (bu makine)
