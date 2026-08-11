# Project Memory

## Project Info
- YL tezi: **"Teoriden Silikona"** — SSM görü omurgalarının (VMamba vb.) teorik verimlilik avantajının gerçek dağıtım yığınlarında (ONNX Runtime, CoreML/ANE) ne kadarının gerçekleştiğinin ampirik analizi. 16 hafta, hedef bitiş ~30 Kasım 2026.

## Project Status
- **Faz 0** (H1-2): IN PROGRESS — öncül doğrulama + ölçüm altyapısı. Task yapısı kuruldu (11 Ağu 2026)
- **Faz 1** (H3-5): PENDING — doğruluk eşitleme
- **Faz 2** (H6-8): PENDING — dağıtım yığını matrisi ← tezin kalbi
- **Faz 3** (H9-11): PENDING — nicemlemenin yoğun tahmine transferi
- **Faz 4** (H12-13): PENDING — yeniden formülasyon (riskli özgün katkı, sona itildi)
- **Faz 5** (H14-16): PENDING — yazım

## Key Technical Decisions
- **Eşit doğruluk** karşılaştırması, eşit parametre değil — farklı mIoU'daki modellerin hız kıyası anlamsız
- Segmentasyon başlığı sabit (UPerNet), reçete dondurulmuş — değişken sadece omurga
- Riskli faz (4) sona itildi — Faz 1-3 tek başına tez oluşturur
- Başarısız export = veri. Boş hücre yok, her hücre ya sayı ya gerekçe
- Python ortamı: **uv + 3.12 venv** (sistem Python 3.14.5, coremltools/mamba-ssm tekerlek yayınlamıyor)

## Important Patterns
- Ham ölçümler daima `results/raw/` — asla sadece grafik saklanmaz
- Her ölçümde ortam sürümleri otomatik kaydedilir
- ANE iddiası Xcode profili olmadan geçersiz — "ANE'de çalıştı" kanıt ister (3.5.6)
- Haftada tek somut çıktı; grafik üretildiği hafta yorumlanır

## Known Issues / Gotchas
- **Bu makine Mac (M5 Pro)** — nvidia-smi yok. RTX 5070 ayrı sistemde, oradaki işler (TASK-003, 023) sadece Ömer yapabilir
- RTX 5070 = Blackwell → `mamba-ssm` CUDA çekirdekleri derlenmeyebilir. Derleme hatası = tezin ilk bulgusu, panik değil
- VRAM doğrulanmadı: 12 GB varsayımıyla plan kuruldu (Ti ise 16 GB — nvidia-smi bekleniyor)
- ONNX Runtime #27796: Mamba, ONNX `Loop` yüzünden M3'te 17× yavaş — tezin öncül kanıtı
- `powermetrics` sudo ister — harness tasarımında hesaba kat

## Working Credentials (Dev)
- Yok — bu projede servis/DB yok
