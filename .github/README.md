# GitHub Actions — MealfitRD.IA

> **[P3-LIVE-1 · 2026-05-12]** Este README documenta el workflow CI
> introducido con P1-LIVE-2 y el lint job no-bloqueante de P2-LIVE-1.

---

## Estado actual del repo

`git status` desde la raíz reporta `not a git repository`. El workflow
[`workflows/ci.yml`](workflows/ci.yml) ya está commiteado en el árbol y
se activará **automáticamente** cuando alguien ejecute:

```bash
git init
git remote add origin https://github.com/<owner>/<repo>.git
git add -A
git commit -m "Initial commit"
git push -u origin main
```

GitHub detecta el workflow file en el primer push y lo registra. Tras
eso, cada push/PR ejecuta los jobs.

Mientras tanto, los wrappers locales [`scripts/run_ci.ps1`](../scripts/run_ci.ps1)
y [`scripts/run_ci.sh`](../scripts/run_ci.sh) reproducen los 3 jobs
principales offline.

---

## Jobs del workflow

| Job | Comando | Bloquea merge? | Propósito |
|---|---|---|---|
| `backend-tests` | `pytest tests/ -v --tb=short -m "not e2e" -x` | **Sí** | ~120+ tests parser-based + funcionales |
| `frontend-tests` | `npm test` (= `vitest run`) | **Sí** | Vitest unit + integration |
| `frontend-build` | `npm run build` | **Sí** | Tree-shaking + import resolution + bundle prod |
| `frontend-lint` | `npm run lint` | **No** (P2-LIVE-1) | eslint — `continue-on-error: true` mientras baseline 245 errores no se limpie |

### Por qué `frontend-lint` es no-bloqueante

Al introducir el gate (P1-LIVE-2), `npm run lint` reportaba 245 errores +
13 warnings pre-existentes (deuda técnica acumulada — ver
`project_p2_live_1_ci_lint_job_2026_05_12.md` en memoria). Bloquear
merges sobre esa baseline paralizaría todo el desarrollo. La flag
`continue-on-error: true` permite que el job EJECUTE y reporte status
en GitHub UI (visibilidad de trend) sin bloquear el merge.

**Roadmap**: tras cleanup incremental que reduzca el count a 0,
flippear `continue-on-error: false` y actualizar el test
`test_p2_live_1_ci_includes_lint_job::test_c_lint_job_is_non_blocking_initially`
para que enforce el modo bloqueante.

---

## Triggers

```yaml
on:
  push:
    branches: ["**"]            # cualquier branch
  pull_request:
    branches: [main, master]    # gate pre-merge
```

`concurrency` cancel-in-progress por `<workflow>-<ref>` evita acumular
runs viejos del mismo branch (ahorra minutos de GitHub Actions).

---

## Activación de branch protection (post-`git init`)

Para que el CI realmente bloquee merges a `main`, configurar branch
protection rule en GitHub UI tras el primer push:

1. **Settings → Branches → Add rule** (o "Add branch ruleset" en repos modernos).
2. Branch name pattern: `main`.
3. Marcar:
   - **Require status checks to pass before merging** ✓
     - Status checks required:
       - `Backend pytest`
       - `Frontend Vitest`
       - `Frontend build`
       - (**NO** marcar `Frontend lint (non-blocking)` hasta cleanup eslint.)
   - **Require branches to be up to date before merging** ✓
   - **Require linear history** ✓ (opcional pero recomendado).
   - **Do not allow bypassing the above settings** ✓ (también para admins).
4. Save.

Tras esto, cualquier PR a `main` requiere los 3 jobs verde antes de
mostrar el botón "Merge".

---

## Filtro `-m "not e2e"` en backend-tests

Los tests con marker `@pytest.mark.e2e` requieren Supabase live (DB real
+ service_role JWT). En GitHub runners no hay acceso a esa infraestructura
de prod. El filtro `-m "not e2e"` salta esos tests; los parser-based
+ funcionales puros (la inmensa mayoría) corren todos.

Si necesitas correr E2E en CI eventualmente (e.g. branch de staging con
project Supabase aparte), removerlos del filtro requiere:
1. Crear project Supabase staging.
2. Añadir secrets a GitHub (`SUPABASE_URL_STAGING`, `SUPABASE_KEY_STAGING`).
3. Crear job separado `backend-tests-e2e` que solo dispare en
   `pull_request` a `main` (no en cada push).

---

## SOPs operacionales

### El CI falla solo en mi PR pero pasa localmente

- Sincroniza con `main`: `git pull origin main --rebase`.
- Limpia caché Node: `cd frontend && rm -rf node_modules && npm ci`.
- Limpia caché Python: `cd backend && rm -rf __pycache__ .pytest_cache`.
- Si persiste, comparar versión `python --version` y `node --version`
  con las del workflow (Python 3.12 + Node 20).

### Quiero re-correr un job individual

GitHub UI → tab Actions → seleccionar run → click "Re-run failed jobs"
o "Re-run all jobs".

### Quiero ver el log completo de un job

GitHub UI → tab Actions → seleccionar run → click el job → expandir
el step que falló.

### Quiero skipear CI en un commit (no recomendado)

Añadir `[skip ci]` al mensaje del commit. Solo usar para commits
puramente docs (README, comments) que no tocan código.

---

## Memoria relacionada

- `project_p1_live_2_ci_gate_2026_05_12.md` — cierre original del workflow.
- `project_p2_live_1_ci_lint_job_2026_05_12.md` — cierre lint no-bloqueante.
- `project_p3_live_1_ci_docs_2026_05_12.md` — este README + scripts/README.md.
