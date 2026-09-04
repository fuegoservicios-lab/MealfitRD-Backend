# GitHub Actions — MealfitRD.IA (repo backend)

> **[P3-LIVE-1 · 2026-05-12 · reescrito P0-CI-VERDICT · 2026-09-04]** Este README documenta
> los workflows REALES del repo backend y, para el onboarding, el del repo hermano
> (frontend). La versión anterior describía un workflow monorepo de tres jobs
> (`backend-tests` / `frontend-tests` / `frontend-build`) que dejó de existir el
> 2026-08-22 (`P2-I18N-CI-HERMANOS-ROJO-PERMANENTE`): 44 corridas en rojo sin ejecutar un
> test porque pedía `backend/requirements.txt` en un checkout cuya raíz ES el backend.

---

## Los dos repos, los dos CI

| Repo | Workflow | Jobs (= status checks) | Bloquea merge? |
|---|---|---|---|
| `fuegoservicios-lab/MealfitRD-Backend` | [`workflows/ci.yml`](workflows/ci.yml) | **`Backend pytest`** | **Sí** |
| `fuegoservicios-lab/MealfitRD-Backend` | [`workflows/macro-benchmark-nightly.yml`](workflows/macro-benchmark-nightly.yml) | `macro-precision-no-regression` (nightly, se auto-salta sin secrets) | No |
| `fuegoservicios-lab/MealfitRD-Backend` | [`workflows/landing-benchmark-remote-guest.yml`](workflows/landing-benchmark-remote-guest.yml) | benchmark del landing (manual) | No |
| `fuegoservicios-lab/MealfitRD` (frontend) | `.github/workflows/ci.yml` | **`quality`**, **`audit`**, **`suministro`**, **`e2e (chromium)`**, **`e2e (firefox)`**, **`e2e (webkit)`** | **Sí** |

El job `quality` del frontend encadena, en este orden: `i18n:check:strict` → `eslint --max-warnings N`
→ `lint:count --gate` (techo **por regla**) → `typecheck` → `huerfanos.mjs --gate` (código
muerto) → `npm test` (vitest) → `npm run build` → `check:bundle-size` → `check:presupuestos`.
Detalle y cómo recalibrar cada gate: `frontend/docs/gates_de_ci.md`.

### Historia del gate de lint (para no "arreglarlo" hacia atrás)

- `P1-LIVE-2` (05-12) introdujo el CI; `P2-LIVE-1` puso el job de lint con
  `continue-on-error: true` porque la baseline eran 245 errores y bloquear habría paralizado
  el desarrollo. **Esa flag ya no existe en ningún workflow.**
- `P1-CI-GATE-PASSABLE` (08-14) y `P2-LINT-RATCHET-POR-REGLA` (08-18) la SUPERSEDEN: el
  lint es BLOQUEANTE, con techo global (`--max-warnings`) + techo por regla en
  `scripts/lint-count.mjs`, y los dos números se mueven juntos hacia abajo (trinquete).
  Un `continue-on-error` en lint hoy sería una regresión, no una restauración.

---

## `Backend pytest`: cómo da un veredicto sin el hermano ni la DB

El backend se clona en `backend/` para reproducir la disposición del workspace de desarrollo
(`<raíz>/backend`, `<raíz>/frontend`): ~400 tests construyen rutas al frontend por
`Path(__file__).parents[2]`.

1. **Hermano frontend**: se clona sólo si existe el secret `SIBLING_REPO_TOKEN` (PAT con
   `contents:read` sobre el repo privado). Sin él, los tests cross-repo se **SALTAN** con razón
   (`[P0-CI-VERDICT] repo hermano frontend ausente`) en vez de reventar — el resumen del job lo
   declara. Con el secret, corren de verdad (incluido el P0 clínico de alérgenos).
2. **Raíz del workspace emulada**: `CLAUDE.md` y `migrations/` viven en la raíz como copia
   SSOT de los ficheros del backend (`P3-MIGRATIONS-SSOT`); el job los enlaza
   (`ln -s backend/CLAUDE.md`, `ln -s backend/migrations`) y esos ~250 tests ejecutan contra
   el contenido real.
3. **Artefactos que ningún repo versiona** (`deploy-mealfit.ps1`, runbooks de
   `~/.claude/…/memory`, `scratch/README.md`, el `.env` local) y el **catálogo vivo** de
   `master_ingredients` (sin DB en el runner): `tests/conftest.py` convierte ese único motivo
   de fallo en un SKIP con razón. `-rs` los lista agrupados al final del log.
4. **`-m "not e2e"`** es load-bearing: los tests `@pytest.mark.e2e` necesitan Neon y además
   ese marker es la llave que autoriza escrituras (`P0-TEST-DB-ISOLATION`). El paso
   «Report unverified e2e count» dice cuántos NO corrieron para que un verde no se lea como
   cobertura. Sólo `test_p1_country_system_f2.py -m e2e` (read-only) corre si hay
   `NEON_DATABASE_URL`.

Medido antes de `P0-CI-VERDICT` (run 1451, 2026-09-03): 1.412 failed + 464 errors — CI sin
veredicto durante semanas. Ancla: `tests/test_p0_ci_verdict.py`.

---

## Activación de branch protection

Settings → Branches → rule para `main` (en CADA repo):

- **Require status checks to pass before merging** ✓
  - Backend: `Backend pytest`.
  - Frontend: `quality`, `audit`, `suministro`, `e2e (chromium)`, `e2e (firefox)`, `e2e (webkit)`.
- **Require branches to be up to date before merging** ✓
- **Do not allow bypassing the above settings** ✓ (también para admins).

Sin la regla, el workflow corre pero no bloquea: es telemetría, no gate.

---

## Wrappers locales

`scripts/run_ci.ps1` reproduce pytest + vitest + build antes del push (flags `-SkipBackend`,
`-BackendCrossRepoOnly`, `-SkipFrontend`, `-SkipBuild`); `scripts/run_ci.sh` es un fósil
deprecado que falla ruidosamente y apunta al `.ps1` (`P3-I18N-RUN-CI-SH-FOSIL`). Ver
[`../scripts/README.md`](../scripts/README.md).

---

## SOPs operacionales

### El CI falla solo en mi PR pero pasa localmente

- Sincroniza con `main`: `git pull origin main --rebase`.
- Backend: `rm -rf __pycache__ .pytest_cache`; frontend: `rm -rf node_modules && npm ci`.
- Compara `python --version` / `node --version` con el workflow (Python 3.12 + Node 22).
- Mira el resumen del job: si dice «Sin frontend hermano», el rojo NO puede venir de un test
  cross-repo (están saltados); si un test cross-repo te falla en local, ese es el veredicto real.

### Quiero re-correr un job / ver el log completo

Actions → run → «Re-run failed jobs» · Actions → run → job → expandir el step. Los skips por
razón están al final del step de pytest (`-rs`).

### Quiero skipear CI en un commit (no recomendado)

`[skip ci]` en el mensaje. Sólo para commits puramente documentales.

---

## Memoria relacionada

- `project_p1_live_2_ci_gate_2026_05_12.md` — cierre original del workflow.
- `project_p2_live_1_ci_lint_job_2026_05_12.md` — lint no-bloqueante (superseded).
- `project_p3_live_1_ci_docs_2026_05_12.md` — este README + scripts/README.md.
