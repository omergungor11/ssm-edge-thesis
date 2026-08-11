Tum degisiklikleri stage, commit ve push et. Adimlar:

1. `git status` calistir — degisen dosyalari gor
2. `git diff --stat` calistir — degisiklik ozetini gor
3. `tez-tasks/task-index.md` oku — aktif/tamamlanmis task durumunu kontrol et
4. Eger IN_PROGRESS task varsa ve isi bittiyse:
   - task-index.md'de durumunu COMPLETED yap
   - Dashboard tablosundaki sayilari guncelle
   - CHANGELOG'a kisa bir kayit ekle
5. .env dosyalarinin stage'lenmediginden emin ol
6. Tum ilgili dosyalari stage et (.env, credentials haric)
7. Degisiklikleri analiz edip anlamli bir commit mesaji yaz:
   - Task ile ilgiliyse: `feat(TASK-XXX): aciklama`
   - Genel ise: `chore/docs/fix: aciklama`
   - Co-Authored-By satirini ekle
8. `git push` ile push et
9. Son commit'leri goster ve kisa ozet ver

NOT: .env, credentials veya secret iceren dosyalari ASLA commit etme.
