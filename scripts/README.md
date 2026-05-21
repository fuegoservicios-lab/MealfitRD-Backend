# backend/scripts/

[P2-AUDIT-1 · 2026-05-12] Scripts operacionales de SRE / dev local.

NO son tests (esos viven en `backend/tests/`), NI parte del runtime de FastAPI
(eso vive en `backend/*.py` top-level). Cada script es invocado manualmente por
operador.

## Inventario

| Script | Propósito |
|---|---|
| `check_items.py` | Inspección rápida de `master_ingredients` desde shell local. |
| `check_scaling.py` | Reemplazo del endpoint `/debug-scaling/{user_id}` eliminado en P1-AUDIT-NEW-1 (audit 2026-05-12). Read-only sobre `meal_plans`. **Requiere `SUPABASE_DB_URL` + filtrar por user_id explícito** — NO IDOR-fallback. |
| `check_schema.py` | Snapshot del schema actual (información para post-mortems de drift). |
| `checkdb.py` | Sanity check de conectividad a la DB desde el venv local. |
| `publish_pfix_marker.py` | Helper para publicar `expected_last_known_pfix` en `app_kv_store` tras cierre de un P-fix (alternativa al MCP `execute_sql`). |
| `test_medical_reviewer.py` | Smoke test manual del review LLM (NO automated — requiere GEMINI_API_KEY). |
| `test_semantic_cache.py` | Smoke test del semantic cache (manual). |
| `run_coverage.sh` / `run_coverage.ps1` | [I6 / P3-COVERAGE-HEATMAP · 2026-05-20] Genera coverage heatmap de los ~770 tests pytest. Auto-instala `pytest-cov` si falta. Output: `htmlcov/index.html` (gitignored) + summary terminal. Default filtro: `-m "not e2e"`. Uso: `./scripts/run_coverage.sh` o `--term` para skip HTML. |

## Convenciones

- **Ningún script aquí debe quedar fijo en cron** — los crons viven en
  `backend/cron_tasks.py::register_plan_chunk_scheduler` (SSOT).
- **Ningún script debe escribir a `meal_plans` sin filtro `WHERE user_id = …`** —
  ver invariantes I2/I3/I6 de CLAUDE.md.
- Si necesitas un script destructivo (DELETE, UPDATE en masa): añadir flag
  `--dry-run` por default y `--commit` para activar mutación + log explícito
  de cada fila antes del UPDATE.
