# backend/scripts/

[P2-AUDIT-1 · 2026-05-12] Scripts operacionales de SRE / dev local.

NO son tests (esos viven en `backend/tests/`), NI parte del runtime de FastAPI
(eso vive en `backend/*.py` top-level). Cada script es invocado manualmente por
operador.

## Inventario

| Script | Propósito |
|---|---|
| `check_scaling.py` | Reemplazo del endpoint `/debug-scaling/{user_id}` eliminado en P1-AUDIT-NEW-1 (audit 2026-05-12). Read-only sobre `meal_plans` vía `execute_sql_query` (Neon). **Requiere `NEON_DATABASE_URL_POOLED` + filtrar por user_id explícito** — NO IDOR-fallback. |
| `check_schema.py` | Snapshot del schema actual (información para post-mortems de drift). |
| `publish_pfix_marker.py` | Helper para publicar `expected_last_known_pfix` en `app_kv_store` tras cierre de un P-fix (alternativa al MCP `execute_sql`). |
|  `demo_medical_reviewer.py` | Smoke test manual del review LLM (NO automated; llama al proveedor REAL, así que gasta — P0-LLM-PROVIDER-MIGRATION dejó obsoleta la GEMINI_API_KEY que citaba esta fila). |
|  `demo_semantic_cache.py` | Demostración manual del semantic cache. Invoca `run_plan_pipeline` ENTERO: gasta como una generación de plan. |
| `run_coverage.sh` / `run_coverage.ps1` | [I6 / P3-COVERAGE-HEATMAP · 2026-05-20] Genera coverage heatmap de los ~770 tests pytest. Auto-instala `pytest-cov` si falta. Output: `htmlcov/index.html` (gitignored) + summary terminal. Default filtro: `-m "not e2e"`. Uso: `./scripts/run_coverage.sh` o `--term` para skip HTML. |

## Wrappers locales del CI

[P3-LIVE-1 · 2026-05-12 · movido aquí P0-CI-VERDICT · 2026-09-04] Antes de cada `git push`,
reproduce los jobs del CI en local:

| Wrapper | Estado | Uso |
|---|---|---|
| `run_ci.ps1` | **VIVO** (PowerShell 7, multiplataforma) | `pwsh -File scripts/run_ci.ps1` — pytest (`-m "not e2e"`) + vitest + build del frontend hermano |
| `run_ci.sh` | **FÓSIL** deprecado (`P3-I18N-RUN-CI-SH-FOSIL`) | Falla ruidosamente (exit 2) y apunta al `.ps1`. No lo revivas: reproducía el CI monorepo que ya no existe. |

Flags de `run_ci.ps1` (se leen de su `param(...)`; una flag nueva sin documentar aquí falla
`test_p3_live_1_ci_docs`):

- `-SkipBackend` — sólo frontend (vitest + build).
- `-BackendCrossRepoOnly` — frontend + sólo los tests de paridad backend↔frontend
  (marker `frontend_cross_repo`); excluyente con `-SkipBackend`.
- `-SkipFrontend` — sólo backend.
- `-SkipBuild` — tests sin el build de producción.

Exit code 0 si todos los jobs no saltados pasan; 1 si alguno falla. Hook `pre-push` opcional:
`pwsh -File scripts/run_ci.ps1 -SkipBuild`.

## Sobre los ~140 scripts que no están en el inventario

Los `add_foods_*`, `seed_supermarket_*`, `micro_*`, `fix_*` y en general todo lo fechado
(`*_2026_MM_DD.py`) son **one-shot de datos** que ya corrieron contra Neon: se conservan como
pista de auditoría de la procedencia del catálogo (`docs/data_provenance_licenses.md`,
`docs/catalog_provenance_audit.md`), no como herramientas vivas. Regla: un script de datos no se
re-ejecuta sin leer su cabecera y sin `--dry-run`; los que mutan sin flag son los de fecha
anterior a la convención de abajo.

## Convenciones

- **Ningún script aquí debe quedar fijo en cron** — los crons viven en
  `backend/cron_tasks.py::register_plan_chunk_scheduler` (SSOT).
- **Ningún script debe escribir a `meal_plans` sin filtro `WHERE user_id = …`** —
  ver invariantes I2/I3/I6 de CLAUDE.md.
- Si necesitas un script destructivo (DELETE, UPDATE en masa): añadir flag
  `--dry-run` por default y `--commit` para activar mutación + log explícito
  de cada fila antes del UPDATE.
