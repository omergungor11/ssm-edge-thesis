Yeni bir proje icin Claude Code workflow yapisi kuracaksin. MEVCUT DIZINDE calisacaksin (dosyalar zaten burada olusturulacak). Asagidaki adimlari sirasiyla uygula:

---

## Adim 1: Bilgi Topla

AskUserQuestion tool'unu kullanarak asagidaki sorulari TEK SEFERDE sor:

**Soru 1** (header: "Proje Adi"):
- Soru: "Projenin adi ne?"
- Secenekler: "My App", "Dashboard Pro", "API Gateway", (Other)
- multiSelect: false

**Soru 2** (header: "Prefix"):
- Soru: "Meta dizin prefix'i ne olsun? (ornek: myapp -> myapp-tasks/, myapp-docs/)"
- Secenekler: "app", "project", "my", (Other)
- multiSelect: false

**Soru 3** (header: "Tech Stack"):
- Soru: "Hangi tech stack kullanilacak?"
- multiSelect: false
- Secenekler:
  - "Next.js Full-Stack" — aciklama: Next.js App Router + API Routes + Prisma + Tailwind
  - "NestJS + Next.js" — aciklama: NestJS backend + Next.js frontend + Prisma + Tailwind
  - "Express + React" — aciklama: Express.js backend + React/Vite frontend

---

## Adim 2: Ek Detaylar

Ilk cevaplara gore 2. tur sorular sor (AskUserQuestion, TEK SEFERDE 4 soru):

**Soru 1** (header: "Monorepo"):
- Soru: "Monorepo yapisinda mi olacak?"
- multiSelect: false
- Secenekler:
  - "pnpm + Turborepo (Recommended)" — aciklama: pnpm workspaces + Turborepo build orchestration
  - "npm/bun workspaces" — aciklama: npm veya bun workspaces
  - "Tek paket" — aciklama: Monorepo yok, tek package.json

**Soru 2** (header: "Veritabani"):
- Soru: "Hangi veritabani kullanilacak?"
- multiSelect: false
- Secenekler:
  - "PostgreSQL + Prisma (Recommended)" — aciklama: En yaygin secim, type-safe ORM
  - "PostgreSQL + Drizzle" — aciklama: Lightweight, SQL-first ORM
  - "SQLite + Prisma" — aciklama: Dosya tabanli, basit projeler icin

**Soru 3** (header: "Ekstralar"):
- Soru: "Hangi ekstra ozellikler eklensin?"
- multiSelect: true
- Secenekler:
  - "Docker" — aciklama: docker-compose ile DB, Redis, vs.
  - "Redis/BullMQ" — aciklama: Cache + queue sistemi
  - "Auth (JWT)" — aciklama: Kimlik dogrulama altyapisi

**Soru 4** (header: "Phase Sayisi"):
- Soru: "Kac phase planlansin?"
- multiSelect: false
- Secenekler:
  - "3 phase" — aciklama: Setup + Core + Frontend (kucuk proje)
  - "5 phase" — aciklama: Setup + Core + Business + Frontend + Deploy (orta proje)
  - "8 phase" — aciklama: Setup + Core + Business + Advanced + Channels + Frontend + Test + Deploy (buyuk proje)

---

## Adim 3: Template Temizligi

Mevcut dizinde template'den kalma dosyalari temizle:

```bash
# Eski template dizinlerini sil
rm -rf _tasks _docs _config _plans

# Template yardimci dosyalarini sil
rm -f BLUEPRINT.md setup.sh

# Eski template CLAUDE.md'yi sil (yenisi olusturulacak)
rm -f CLAUDE.md
```

Bu adimi MUTLAKA dosya olusturmadan ONCE yap. Boylece eski ve yeni dosyalar cakismaz.

---

## Adim 4: Dosya Olusturma

Tum cevaplar toplandiktan ve temizlik yapildiktan sonra asagidaki dosyalari SIFIRDAN OLUSTUR.
Tum dosyalar MEVCUT DIZINDE (pwd) olusturulur — baska dizin kullanma.

Once dizin yapisini `mkdir -p` ile olustur, sonra dosyalari yaz.

### 4.1: Dizin Yapisi

```bash
mkdir -p .claude/{commands,hooks}
mkdir -p [PREFIX]-tasks/{phases,active}
mkdir -p [PREFIX]-config
mkdir -p [PREFIX]-docs
mkdir -p [PREFIX]-plans
```

`[PREFIX]` = kullanicinin Adim 1'de verdigi prefix degeri.

### 4.2: .claude/hooks/protect-files.sh

Asagidaki SABIT icerigi yaz ve `chmod +x` calistir:

```bash
#!/bin/bash
# PreToolUse hook: Protect sensitive files from being edited/written
# Exit 2 = BLOCK the action, Exit 0 = ALLOW

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

if [[ -z "$FILE_PATH" ]]; then
  exit 0
fi

# Block .env files (all variants)
if [[ "$FILE_PATH" =~ \.env($|\.local|\.production|\.staging|\.test|\.development) ]]; then
  echo "BLOCKED: .env files are protected. Never edit secrets directly." >&2
  exit 2
fi

# Block lock files
if [[ "$FILE_PATH" == *"pnpm-lock.yaml"* ]] || [[ "$FILE_PATH" == *"package-lock.json"* ]] || [[ "$FILE_PATH" == *"yarn.lock"* ]] || [[ "$FILE_PATH" == *"bun.lockb"* ]]; then
  echo "BLOCKED: Lock files should only be modified by the package manager." >&2
  exit 2
fi

# Block .git directory
if [[ "$FILE_PATH" == *"/.git/"* ]]; then
  echo "BLOCKED: .git directory should not be edited directly." >&2
  exit 2
fi

# Block credentials
if [[ "$FILE_PATH" == *"credentials"* ]] || [[ "$FILE_PATH" == *"secrets.yaml"* ]] || [[ "$FILE_PATH" == *"secrets.json"* ]] || [[ "$FILE_PATH" == *"service-account"* ]]; then
  echo "BLOCKED: Credential files are protected." >&2
  exit 2
fi

exit 0
```

### 4.3: .claude/settings.local.json

Tech stack'e gore `permissions.allow` listesini olustur. Temel izinler HER ZAMAN dahil:

```json
{
  "permissions": {
    "allow": [
      "Bash(ls:*)",
      "Bash(git add:*)",
      "Bash(git commit:*)",
      "Bash(git push:*)",
      "Bash(git branch:*)",
      "Bash(git log:*)",
      "Bash(git status:*)",
      "Bash(git diff:*)",
      "Bash(docker compose:*)",
      "Bash(lsof:*)",
      "Bash(kill:*)"
    ]
  },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/protect-files.sh",
            "timeout": 5
          }
        ]
      }
    ],
    "PostToolUse": []
  }
}
```

Tech stack'e gore permissions'a EKLE:
- pnpm secildiyse: `"Bash(pnpm:*)"`, `"Bash(npx:*)"`, `"Bash(npx turbo:*)"`
- npm secildiyse: `"Bash(npm:*)"`, `"Bash(npx:*)"`
- bun secildiyse: `"Bash(bun:*)"`, `"Bash(bunx:*)"`
- Prisma secildiyse: `"Bash(npx prisma generate:*)"`, `"Bash(npx prisma migrate dev:*)"`
- NestJS secildiyse: `"Bash(npx nest build:*)"`
- Next.js secildiyse: `"Bash(npx next build:*)"`

Prisma secildiyse PostToolUse'a auto-generate hook EKLE:
```json
{
  "matcher": "Edit",
  "hooks": [
    {
      "type": "command",
      "command": ".claude/hooks/post-edit-prisma.sh",
      "timeout": 30
    }
  ]
}
```
Ve `post-edit-prisma.sh` dosyasini da olustur + `chmod +x`:
```bash
#!/bin/bash
INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
if [[ "$FILE_PATH" == *"schema.prisma"* ]]; then
  cd "${CLAUDE_PROJECT_DIR:-$(pwd)}" || exit 0
  npx prisma generate 2>&1
fi
exit 0
```

### 4.4: Slash Commands (4 adet)

`.claude/commands/` altina 4 dosya yaz (mevcut new-project.md'ye DOKUNMA).
Iceriklerde `[PREFIX]` yerine kullanicinin verdigi GERCEK prefix degerini yaz.

**cold-start.md:**
```
Yeni bir session basliyor. Projeyi tam olarak anlamak icin adimlari sirasiyla uygula:

1. `CLAUDE.md` oku — proje ozeti, conventions, slash command referanslari
2. `[PREFIX]-docs/MEMORY.md` oku — kalici hafiza, teknik kararlar
3. `[PREFIX]-tasks/task-index.md` oku — dashboard + tum task durumlari
4. `[PREFIX]-tasks/active/session-notes.md` oku — onceki session notlari
5. Aktif (IN_PROGRESS) task var mi kontrol et → varsa phase dosyasindan detayini oku
6. Son birkac git commit'i incele (`git log --oneline -10`)
7. Kisa bir durum raporu ver:
   - Toplam ilerleme (X/Y)
   - Hangi phase'de oldugumuz
   - Aktif task, siradaki task(lar)
   - Onceki session'dan notlar (varsa)
8. "Hazirim, devam edebiliriz." mesaji ver

NOT: Kod degistirme. Sadece oku ve rapor ver.
```

**git-full.md:**
```
Tum degisiklikleri stage, commit ve push et:

1. `git status` — degisen dosyalari gor
2. `git diff --stat` — degisiklik ozeti
3. `[PREFIX]-tasks/task-index.md` oku — task durumu kontrol et
4. Bitmis IN_PROGRESS task varsa → COMPLETED yap, dashboard guncelle, CHANGELOG'a ekle
5. .env dosyalarinin stage'lenmediginden emin ol
6. Dosyalari stage et (.env, credentials haric)
7. Commit mesaji yaz: `feat(TASK-XXX): aciklama` + Co-Authored-By
8. `git push` ile push et
9. Son commit'leri goster

NOT: .env, credentials veya secret iceren dosyalari ASLA commit etme.
```

**turn-off.md:**
```
Session'i kapatmadan once:

1. [PREFIX]-tasks/task-index.md oku — aktif task var mi? Yapilan isleri listele
2. Task durumlarini guncelle (COMPLETED / IN_PROGRESS)
3. Session notu yaz → `[PREFIX]-tasks/active/session-notes.md`:
   - Tarih, yapilanlar, yarim kalanlar, siradaki, dikkat edilecekler
4. CLAUDE.md mevcut durum bolumunu guncelle
5. Git full calistir — stage, commit, push
6. Kapanis mesaji: ozet + siradakiler + "Session kapatildi."
```

**local-testing.md:**
```
Local dev ortamini dogrula:

0. Port temizligi — eski process'leri kontrol et
1. Docker/container durumu — kapaliysa baslat
2. Veritabani — ORM client guncel mi? Migration? Seed data?
3. Backend — build kontrolu + health endpoint testi
4. Frontend — build kontrolu
5. Ozet rapor: her servis OK/FAIL, erisim URL'leri

NOT: Sunuculari baslatma — sadece build ve health check yap.
```

### 4.5: Task Index ([PREFIX]-tasks/task-index.md)

Phase sayisina gore dashboard tablosu ve Phase 0 task'larini olustur.

Phase 0 HER PROJEDE AYNI (7 task):

```markdown
# [PROJE_ADI] - Task Index

## Dashboard

| Phase | Name | Total | Done | In Progress | Pending | Blocked |
|-------|------|-------|------|-------------|---------|---------|
| 0 | Project Setup | 7 | 0 | 0 | 7 | 0 |
| 1 | [Phase 1 adi] | 0 | 0 | 0 | 0 | 0 |
[... secilen phase sayisina gore ...]
| **Total** | | **7** | **0** | **0** | **7** | **0** |

**Progress**: 0/7 (0%)

---

## Phase 0: Project Setup

| ID | Task | Agent | Complexity | Status | Dependencies |
|----|------|-------|-----------|--------|-------------|
| TASK-001 | Monorepo + tooling init | devops | S | PENDING | - |
| TASK-002 | Meta directories | docs | S | PENDING | - |
| TASK-003 | .claude/ setup | devops | M | PENDING | TASK-001 |
| TASK-004 | CLAUDE.md configuration | docs | M | PENDING | TASK-002 |
| TASK-005 | Docker dev environment | devops | M | PENDING | TASK-001 |
| TASK-006 | Lint, format, TypeScript config | devops | S | PENDING | TASK-001 |
| TASK-007 | Git repo + first commit | devops | S | PENDING | TASK-001..006 |

## Phase 1: [Isim]
| ID | Task | Agent | Complexity | Status | Dependencies |
|----|------|-------|-----------|--------|-------------|
<!-- Task'lar sonra eklenir -->
```

Phase isimleri secilen phase sayisina gore:
- 3 phase: Setup, Core, Frontend
- 5 phase: Setup, Core Infrastructure, Business Logic, Frontend/UI, Deployment
- 8 phase: Setup, Core Infrastructure, Business Logic, Advanced Features, Channels, Frontend/UI, Test & QA, Deployment

### 4.6: Phase 0 Detay ([PREFIX]-tasks/phases/phase-0.md)

7 task'in her biri icin acceptance criteria ile birlikte detay yaz:

```markdown
# Phase 0: Project Setup

## TASK-001: Monorepo + Tooling Init
**Agent**: devops | **Complexity**: S | **Status**: PENDING | **Dependencies**: -

### Acceptance Criteria
- [ ] Package manager initialized
- [ ] Workspace configuration set up
- [ ] Build tool configured
- [ ] Root package.json with workspace scripts

---

## TASK-002: Meta Directories
**Agent**: docs | **Complexity**: S | **Status**: PENDING | **Dependencies**: -

### Acceptance Criteria
- [ ] [PREFIX]-tasks/ with task-index.md, phases/, active/
- [ ] [PREFIX]-docs/ with MEMORY.md, CHANGELOG.md
- [ ] [PREFIX]-config/ with workflow.md, conventions.md, tech-stack.md, agent-instructions.md
- [ ] [PREFIX]-plans/ directory created

---

## TASK-003: Claude Code Setup
**Agent**: devops | **Complexity**: M | **Status**: PENDING | **Dependencies**: TASK-001

### Acceptance Criteria
- [ ] protect-files.sh hook working
- [ ] 4 slash commands created
- [ ] settings.local.json configured

---

## TASK-004: CLAUDE.md Configuration
**Agent**: docs | **Complexity**: M | **Status**: PENDING | **Dependencies**: TASK-002

### Acceptance Criteria
- [ ] Project description and workspace layout
- [ ] Slash commands documented
- [ ] Code conventions summarized
- [ ] Reference directories table

---

## TASK-005: Docker Dev Environment
**Agent**: devops | **Complexity**: M | **Status**: PENDING | **Dependencies**: TASK-001

### Acceptance Criteria
- [ ] docker-compose.yml with required services
- [ ] Health checks configured
- [ ] Ports documented

---

## TASK-006: Lint, Format, TypeScript Config
**Agent**: devops | **Complexity**: S | **Status**: PENDING | **Dependencies**: TASK-001

### Acceptance Criteria
- [ ] Linter configured
- [ ] Formatter configured
- [ ] TypeScript strict mode

---

## TASK-007: Git Repo + First Commit
**Agent**: devops | **Complexity**: S | **Status**: PENDING | **Dependencies**: TASK-001..006

### Acceptance Criteria
- [ ] .gitignore with common patterns
- [ ] All Phase 0 files committed
- [ ] Remote connected (if applicable)
```

### 4.7: Session Notes ([PREFIX]-tasks/active/session-notes.md)

```markdown
# Session Notes
<!-- Her session icin tarih, yapilanlar, yarim kalanlar, siradakiler, notlar -->
```

### 4.8: Config Dosyalari

**[PREFIX]-config/workflow.md**: Task workflow kurallari yaz — pre-task (task-index oku, dependency kontrol, status IN_PROGRESS yap), during-task (acceptance criteria, validation), post-task (status COMPLETED, dashboard guncelle, CHANGELOG, git commit). Commit conventions (`feat/fix/refactor/docs/chore/test(TASK-XXX): desc`). Tech stack'e ozel validation komutlari.

**[PREFIX]-config/conventions.md**: Tech stack'e OZEL conventions yaz:
- Next.js Full-Stack secildiyse: App Router, Server Components default, API Routes, Tailwind, `use client` sadece gerekince
- NestJS + Next.js secildiyse: NestJS module pattern, DTO + class-validator, Swagger, dependency injection + Next.js App Router
- Express + React secildiyse: Router/middleware pattern + React/Vite component conventions
- HER DURUMDA: TypeScript strict, `any` yasak, kebab-case dosyalar, RESTful API, response format `{ data, meta? }`, commit conventions

**[PREFIX]-config/tech-stack.md**: Secilen teknolojilerin versiyonlariyla doldur (guncel versiyonlari kullan). Runtime, backend framework, frontend framework, ORM, DB, styling, state management, testing vs.

**[PREFIX]-config/agent-instructions.md**: Agent tipleri (backend, frontend, database, devops, docs) ve scope'lari yaz:
- Monorepo secildiyse: backend scope `apps/api/src/`, frontend scope `apps/web/src/`, db scope `packages/database/`
- Tek paket secildiyse: backend scope `src/api/` veya `src/server/`, frontend scope `src/` veya `src/client/`
- Paralel orkestrasyon kurallari (dizin izolasyonu, retry pattern, orchestrator sorumlulugu)
- Paylasilan dosya cakisma yonetimi (read-edit-retry, max 3)

### 4.9: Docs

**[PREFIX]-docs/MEMORY.md**: Proje adi, secilen tech stack, secilen ekstralar, project status (Phase 0 PENDING) ile doldurulmus baslat.

**[PREFIX]-docs/CHANGELOG.md**:
```markdown
# Changelog
<!-- Format: ## [DATE] / ### Added / ### Changed / ### Fixed -->
```

### 4.10: CLAUDE.md

TAMAMEN OZELLESTIRILMIS ana konfigurasyon dosyasi yaz. Hicbir placeholder ([PREFIX], [PROJE_ADI] vs.) KALMAYACAK — hepsi gercek degerlerle degistirilmis olacak.

Icerik:
- Proje adi ve 1 satirlik aciklama (kullanicinin verdigi ada gore)
- Slash command tablosu (cold-start, git-full, turn-off, local-testing — 4 adet)
- Mevcut durum: `0/7 task (%0) — Phase 0 basliyor`
- Workspace yapisi (monorepo secildiyse apps/packages, degilse src/ bazli — tech stack'e gore)
- Temel komutlar (tech stack'e uygun: pnpm/npm/bun dev, build, typecheck, lint, db komutlari)
- Code conventions ozeti (conventions.md'den 4-5 kisa madde)
- Parallel agent orchestration ozeti (agent-instructions.md'den kisa referans)
- Referans dizinleri tablosu (gercek prefix ile: myapp-tasks/, myapp-docs/ vs.)
- Hooks tablosu (protect-files + varsa prisma hook)
- Notlar (hafiza dosyasi yolu vs.)

### 4.11: .gitignore

Tech stack'e uygun .gitignore olustur:
```
node_modules/
dist/
build/
.next/
.turbo/
*.tsbuildinfo
.env
.env.*
!.env.example
coverage/
.DS_Store
*.log
```

---

## Adim 5: Dogrulama

Tum dosyalar olusturulduktan sonra:

1. `find . -type f -not -path './.git/*' | sort` ile dosya listesini goster
2. CLAUDE.md'nin TAMAM icerigini goster (Read tool ile)
3. Task dashboard'u goster (task-index.md'den)
4. Mesaj ver:
   ```
   Proje yapisi hazir! Siradaki adimlar:
   1. git init && git add -A && git commit -m "chore: initial project scaffold"
   2. /cold-start calistir
   3. Phase 0 task'larindan basla (TASK-001 ile)
   ```

---

## ONEMLI KURALLAR

- Hicbir adimi atlama, sirasiyla uygula
- Kullanici "Other" secerse, verdigi degeri kullan
- Tum dosyalar MEVCUT DIZINDE olusturulur (baska dizin sorma, kullanma)
- Tum icerikler SIFIRDAN olusturulur — harici template dosyasina KESINLIKLE bagimlilik yok
- HICBIR PLACEHOLDER birakma — `[PREFIX]`, `[PROJE_ADI]` gibi seyler gercek degerlerle degistirilmis olmali
- Adim 3'teki template temizligini MUTLAKA yap (eski tez-tasks/, tez-docs/ vs. silinmeli)
- protect-files.sh ve varsa post-edit-prisma.sh executable olsun (`chmod +x`)
- new-project.md slash command'ina DOKUNMA — oldugu gibi kalsin
- Dosya olusturma sirasinda hata olursa kullaniciya bildir ve dur
