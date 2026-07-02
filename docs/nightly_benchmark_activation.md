# Nightly benchmark de macros — activación y operación

[P2-NIGHTLY-BENCH-ACTIVATE · 2026-07-02] SOP del job `macro-benchmark-nightly`
(`.github/workflows/macro-benchmark-nightly.yml`): el gate E2E de **no-regresión de precisión de
macros** (corre el pipeline REAL contra el set held-out de 20 perfiles y falla si la precisión
regresa vs `tests/fixtures/macro_baseline.json`). Hoy se **auto-salta** sin secrets — el skip es
ruidoso (warning + step summary) desde P2-NIGHTLY-BENCH-ACTIVATE, pero la ACTIVACIÓN es acción del
owner (los secrets no pueden configurarse desde el repo).

## Activación (owner, una vez — ~5 min)

GitHub → repo `MealfitRD-Backend` → **Settings → Secrets and variables → Actions →
New repository secret**:

| Secret | Valor | Obligatorio |
|---|---|---|
| `DEEPSEEK_API_KEY` | la misma key del `.env` del VPS | ✅ |
| `NEON_DATABASE_URL_POOLED` | URL pooled de Neon (la del `.env`) | ✅ |
| `NEON_DATABASE_URL` | URL directa de Neon | ✅ |
| `COHERE_API_KEY` | key de Cohere (embeddings; sin ella degrada keyword) | opcional |

Verificación inmediata: **Actions → macro-benchmark-nightly → Run workflow** (workflow_dispatch).
Un run verde = gate activo; el cron nightly (03:00 RD) queda armado.

## Costo y cadencia

- N=20 con concurrency 2 ≈ 30-45 min y consume cuota DeepSeek compartida con prod (por eso corre
  a las 03:00 RD). Si la cuota aprieta, bajar el cron a 2-3 noches/semana editando el `schedule`.

## Post-activación (recomendado)

1. Tras estabilizar, refrescar el baseline con una corrida N=20 limpia:
   `PYTHONPATH=. python scripts/benchmark_macro_compliance.py 20 --concurrency 2 --write-baseline tests/fixtures/macro_baseline.json`
   (o copiar los números del summary al JSON a mano) y commitearlo.
2. Bajar tolerancias (`--max-mape-rise 5 --max-band-drop 10` → 3/7) cuando 2 semanas de runs
   estén estables.

## Diagnóstico de un run ROJO

1. ¿Regresión real o ruido? N=20 es razonablemente estable, pero UN run rojo aislado se re-corre
   (workflow_dispatch) antes de actuar — el swing run-to-run del LLM existe (lección
   [`project_macro_benchmark_baseline`]: N=8 oscila ±20pt; N=20 mucho menos, no cero).
2. Dos rojos seguidos = regresión: bisecar los commits del día en el eje motor
   (`_apply_macro_engine` / rebalanceador / quantize / closers) — la lección dura del proyecto es
   que la precisión FINAL la fija el MOTOR, no la generación (el motor "lava" el raw del LLM).
3. Rollback rápido: los levers del motor son knobs (`MEALFIT_MACRO_REBALANCE`,
   `MEALFIT_MACRO_SOLVER_ENABLED`, `MEALFIT_PORTION_QUANTIZE`…) — flip por `.env` sin redeploy
   mientras se bisecta.

## Qué NO cubre

- Micros (el panel es advisory; su serie vive en `_micro_floor_kpi_job` / pipeline_metrics).
- Calidad de plato (serie `_creativity_kpi_job`).
- El CI estándar ya corre el mínimo fixture-based (`test_p4_scoreboard.py`) sin LLM.
