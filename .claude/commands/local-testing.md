Local dev ortamini ayaga kaldir ve tum servisleri dogrula:

0. **Port temizligi** (ONCELIKLI):
   - Eski process'leri kontrol et ve temizle

1. **Altyapi servisleri**:
   - Docker/container durumunu kontrol et
   - Kapaliysa baslat, healthy olmasini bekle

2. **Veritabani**:
   - ORM/migration client guncel mi?
   - Pending migration var mi?
   - Seed data var mi? Yoksa seed calistir

3. **Backend**:
   - Build kontrolu
   - Health endpoint testi

4. **Frontend**:
   - Build kontrolu
   - Derleme hatasi varsa bildir

5. **Ozet rapor ver**:
   - Her servisin durumu (OK/FAIL)
   - Erisim URL'leri
   - "Test ortami hazir." veya bulunan hatalari bildir

NOT: Sunuculari arka planda baslatma — sadece build ve health check yap.
