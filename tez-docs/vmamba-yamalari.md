# VMamba Kaynak Yamaları — CPU-Only Ortam Bulguları *(12 Ağustos 2026)*

Resmî VMamba deposu (`MzeroMiko/VMamba`, shallow clone → `third_party/VMamba`), NVIDIA'sız
bir makinede **import dahi edilemiyor**. "PyTorch fallback mevcut" iddiası pratikte iki
CUDA varsayımına takılıyor. İkisi de tezin "ekosistem CUDA varsayımı" anlatısının birincil
kanıtı (Bölüm 4.3 / 5.1'e girecek).

## Yama 1 — `csm_triton.py`: guard'sız `@triton.jit`

**Belirti:** `NameError: name 'triton' is not defined` — modül import'ta çöküyor.
**Sebep:** Dosya başında `import triton` try/except ile korunuyor (`WITH_TRITON=False`),
ama satır ~278'den itibaren `@triton.jit` dekoratörleri ve `tl.constexpr`/`tl.tensor`
tip anotasyonları **guard'ın dışında, modül seviyesinde**. Triton yoksa (macOS'ta hiç
yok — triton'un macOS wheel'i bulunmuyor) fallback'e hiç ulaşılamıyor.
**Yama:** except bloğuna no-op stub (`triton.jit` = kimlik fonksiyonu, `tl.*` = `int`).
`WITH_TRITON=False` kaldığı için çalışma zamanında saf-torch yol seçiliyor; stub yalnızca
modülün import edilebilmesini sağlıyor.

## Yama 2 — `csm_triton.py`: koşulsuz `torch.cuda.device()` bağlamı

**Belirti:** `ValueError: Expected a cuda device, but got: cpu` — ileri geçişte.
**Sebep:** `cross_scan_fn` / `cross_merge_fn`, torch fallback'i **seçmiş olsa bile**
`with torch.cuda.device(x.device):` bağlamını koşulsuz açıyor.
**Yama:** `x.is_cuda` değilse bağlam atlanır, fonksiyon doğrudan çağrılır.

## Sonuç

İki yamayla VMamba-T + UPerNet (61.4M param, ADE20K checkpoint) Mac'te **eksiksiz
yükleniyor** (missing=0/unexpected=0) ve çalışıyor: 512²'de ~3.6 s/görüntü (CPU, saf-torch
scan). Karşılaştırma: aynı boyuttaki bir CNN/ViT CPU'da ~50-200 ms — fark, özel çekirdek
yokluğunda ödenen bedelin ta kendisi.

**Tez notu:** "Fallback var" ile "fallback çalışıyor" arasındaki fark ölçülebilir bir
dağıtım engeli. Resmî repo'nun CPU-only ortamda kutudan çıkmıyor olması, SSM'lerin uç
cihaz araç zinciri olgunluğu hakkında başlı başına bir veri noktası.

> Yamalar `third_party/` altında (git'e dahil değil). Tekrarlanabilirlik için: her iki
> değişiklik `[TEZ YAMASI]` yorum etiketiyle işaretli; `grep -rn "TEZ YAMASI" third_party/`
> ile bulunur. Üstyapıya PR açılması Faz 5'te değerlendirilecek.
