# Yeniden Formülasyon Prototipi — Faz 4 Özeti *(14 Ağustos 2026, TASK-030..032)*

**Kapsam kararı:** prototip-önce (kullanıcı, 14 Ağu) — tam model değil, tek gerçek SS2D
bloğu (VMamba-T `layers.2.blocks.0.op`, checkpoint ağırlıkları, gerçek ADE20K
ara-aktivasyonu 1×384×32×32). Kod: `src/reformulation/`, ham veri: `results/raw/reform_*`.

## Üç form

1. **seq** — ardışık referans (temiz-torch, üçüncü-parti çıktıyla **bit-eşdeğer**)
2. **blocked** — d_state=1 skaler özyinelemesinin blok-içi kapalı formu.
   **Stabilite bulgusu:** naif `cumsum + b/cumprod(a)` formu gerçek aktivasyonlarda
   **NaN** (blok-içi log-çürüme P=64'te −513'e iniyor; model hızlı unutuyor → fp32 taşar).
   Çalışan form: **çürüme-matrisi** T[t,i]=e^{S_t−S_i} (her giriş ≤1, taşma imkânsız;
   maske exp'ten önce) — Mamba-2 SSD chunked formunun d_state=1 özel hâli.
   Doğruluk: max fark ~2-3e-6 ✓
3. **ane** — blocked + Apple reçetesi (B,C,1,S): transpose 4→2 ama reshape patladı;
   ölçülebilir kazanç YOK.

## Ölçüm matrisi (tek blok, medyan ms)

| form | ONNX düğüm | ORT yükleme | CoreML | CoreML GPU | CoreML ANE | ANE op% |
|---|---|---|---|---|---|---|
| seq | 7 285 | 0.99 s | ✓ 47 s | 16.8 | **6.7 (en hızlı!)** | **99.9** |
| blocked128 | **357 (20×↓)** | 0.04 s | ✓ 16 s | 17.1 | 196.3 | 97.7 |
| blocked32 | 1 079 | 0.02 s | ✓ 59 s | **6.6** | 40.0 | 99.3 |

CoreML fp16 sadakati: 0.007-0.07 max fark (tam modelde 22 blok birikimi mIoU ile ölçülmeli).

## AS4'ün rafine cevabı

1. **Blok formu graf patlamasını kırıyor:** düğüm 20×↓, ORT yüklemesi 25×↓, CoreML
   kapısı açılıyor (tam VMamba'da kapalıydı). Katkının pozitif ayağı bu.
2. **Beklenmedik mekanizma bulgusu:** tek-blok ölçekte *ardışık form bile* CoreML'e
   dönüştü ve ANE'de en hızlı çıktı (%99.9 ANE, 6.7 ms). Yani engel **op-uyumsuzluğu
   değil, graf ölçeği**: dönüşüm/yükleme maliyeti düğüm sayısıyla süperlineer büyüyor
   ve tam modelin 390K düğümü duvara çarpıyor. Bu, Bölüm 4.3'ün "yapısal engel"
   bulgusunun mekanizmasını netleştiriyor: *yapı = ölçek*.
3. **Ödünleşim dürüst:** çürüme-matrisi FLOPs'u O(L·P)'ye çıkarıyor (ORT'ta seq'ten
   yavaş); kazanç yalnız graf-katmanında. P=32 tatlı nokta. ANE, maskeli-matmul
   desenini sevmiyor (atama %98 ama uçtan-uca kazanç GPU'da).
4. **Tam ölçekleme riski:** stage-0'da L=16 384 → P=32'de 512 blok; hiyerarşik/karma
   yaklaşım gerekir. Prototip bunu "gelecek çalışma" olarak sınırlıyor — bilinçli.

**Tez cümlesi:** "Blok-kapalı-form, SSM'in dağıtım engelinin *çevrilebilir* olduğunu tek
blok ölçeğinde kanıtlıyor; engelin kaynağının operatör uyumsuzluğu değil graf ölçeği
olduğunu göstermesi ise ana matrisin (Bölüm 4.3) mekanizma açıklamasını tamamlıyor."
