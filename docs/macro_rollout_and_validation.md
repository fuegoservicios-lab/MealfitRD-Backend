# Rollout del motor de macros + validación clínica (P1-1 / P1-2 / P1-4)

[P1-MACRO-ROLLOUT-OBS · 2026-06-14] SOP para cerrar los P1 del audit de precisión que NO se activan
ciegos: requieren validación en vivo (benchmark + canary) y/o proceso humano. Este doc es el handoff.

---

## 1. Motor de macros determinista (P1-1 + P1-2)

### Qué es

El "cerebro dividido" determinista — solver LSQ de porciones (`portion_solver._box_lsq`) + protein-closer
(`_close_protein_gap_for_meal`) + techo de proteína (`_trim_day_protein_to_ceiling`) + reconcile
protein-preserving — vive en `_apply_macro_engine` ([`graph_orchestrator.py:12646`](../graph_orchestrator.py#L12646),
solver gate :12658 / cal-reconcile :12754) y está **gateado por `MEALFIT_MACRO_SOLVER_ENABLED`**.

> **[P1-MACRO-SOLVER-DEFAULT-ON · 2026-06-26] Default flipeado `False`→`True` en código** (tras cerrar el
> canary de abajo). Con el knob OFF, las porciones las decide el LLM "a ojo" → benchmark medido: **proteína
> 16% MAPE, solo ~24% de días con los 4 macros en banda [0.90,1.12]**. El closer/ceiling/reconcile cuelgan
> del MISMO gate (no son separables: comparten la DB de macros, slot fractions y egg-budget), así que P1-1 y
> P1-2 son **un solo rollout**. Rollback de una línea sin redeploy: `MEALFIT_MACRO_SOLVER_ENABLED=False`.

> Verificado 2026-06-14: la DB de macros (`master_ingredients`) está **100% poblada** (105/105 con
> macros completos; hierro 97, satfat 84, potasio 97, fdc 97). Activar es viable a nivel de datos.

### Por qué NO se flipea ciego

El bloque es fail-safe (cualquier error → plan legacy) y el coherence guard (modo `block`) atrapa
divergencias receta↔lista → retry. **Pero** el benchmark midió **45% de tasa de fallback sin explicar**,
y activar el solver podría (a) aumentar el fallback, (b) regresar coherencia receta↔lista. Eso solo se ve
en vivo. Por eso: validar con benchmark + canary ANTES de fijar el default.

### Procedimiento de activación (canary)

1. **Baseline OFF** (en el VPS, env de prod):
   ```bash
   PYTHONPATH=backend python backend/scripts/benchmark_macro_compliance.py 20 --concurrency 3
   ```
   Anota: all-4-macros-en-banda %, proteína MAPE, tasa de fallback.

2. **Medir ON** (mismo set held-out):
   ```bash
   MEALFIT_MACRO_SOLVER_ENABLED=True PYTHONPATH=backend \
     python backend/scripts/benchmark_macro_compliance.py 20 --concurrency 3
   ```
   **Criterio de aceptación**: all-4-macros sube (esperado ~24%→~50%), proteína MAPE baja, y la tasa de
   fallback NO sube respecto al baseline. Si el fallback sube, investigar antes de continuar (es el
   riesgo principal — el solver no resuelve platos criollos compuestos, gap P2 abierto).

3. **Flip en prod** (solo si (2) pasa): en `backend/.env` del VPS:
   ```
   MEALFIT_MACRO_SOLVER_ENABLED=True
   ```
   Redeploy/restart. Confirmar con `curl /health/version` que el binario está vivo.

4. **Watch (primeras 24-48h)** — dos crons ya vigilan la flota automáticamente:
   - `clinical_band_drift` (P4-SCOREBOARD): la precisión promedio debe SUBIR (umbral 0.45).
   - `plan_fallback_rate_high` (P1-MACRO-ROLLOUT-OBS, **nuevo**): la tasa de fallback NO debe cruzar 0.25.
   Revisar `system_alerts` (`resolved_at IS NULL`) + la métrica `clinical_band` en `pipeline_metrics`.

### Rollback (sin redeploy de código)

`MEALFIT_MACRO_SOLVER_ENABLED=False` (o quitar el env var) → vuelve al comportamiento legacy de inmediato.

### Knobs relacionados (ya existentes)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_MACRO_SOLVER_ENABLED` | `True` (P1-MACRO-SOLVER-DEFAULT-ON, 2026-06-26; era `False`) | **gate maestro del cerebro dividido** |
| `MEALFIT_MACRO_SOLVER_PROTEIN_TOPUP` | `True` | top-up legacy si el closer está off |
| `MEALFIT_MACRO_SOLVER_CAL_RECONCILE` | `True` | nivela kcal/día al target tras el solver |
| `MEALFIT_PROTEIN_FLOOR` | `True` | closer + techo de proteína (dentro del gate del solver) |
| `MEALFIT_FALLBACK_RATE_THRESHOLD` | `0.25` | umbral del alert `plan_fallback_rate_high` |
| `MEALFIT_FALLBACK_RATE_INTERVAL_H` | `6` | frecuencia del cron de tasa de fallback |

---

## 1b. Reconcile multi-macro (split C:F) — P1-MACRO-AWARE-RECONCILE

[2026-06-15] Con el solver ON en prod, el benchmark reveló que el problema de precisión NO era la
proteína (6% MAPE, resuelta) sino **carbos (19%) + grasas (22%)**: el reconcile de día legacy
(`_protein_preserving_day_reconcile`) es single-factor (escala C+F juntos para clavar kcal) → no
puede corregir el split C:F. El nuevo `_macro_aware_day_reconcile` escala carbo-dominantes→target_carbs
y grasa-dominantes→target_fats con factores SEPARADOS (preservando proteína; el target es consistente
kcal=4P+4C+9F → clavar C y F clava kcal). Knob `MEALFIT_MACRO_AWARE_RECONCILE`.

**Validación en VPS (N=20, solver ON, concurrencia 1, OFF vs ON):**

| | OFF | ON |
|---|---|---|
| all-4 macros en ±10% | 18.5% | **28.1%** |
| grasas MAPE / en-banda | 22.0% / 29.6% | **10.7% / 64.9%** |
| carbos MAPE | 19.4% | 17.9% |
| kcal MAPE | 10.3% | 9.3% |
| proteína MAPE | 6.2% | 7.4% (varianza de muestreo — el reconcile no toca proteína) |

**Estado: ENABLED en prod** (`MEALFIT_MACRO_AWARE_RECONCILE=True` en el `.env` del VPS, 2026-06-15).
Rollback de una línea: quitar/poner `False` en el `.env` + `sudo systemctl restart mealfit-backend`.

## 2. Validación clínica humana (P1-4)

Gap estructural del audit: la banda de precisión es **autoafirmada**, sin revisión por nutricionista
certificado. Esto es **PROCESO, no código** (`code_closeable=false`); el código solo lo facilita.

### Proceso

1. **Generar la muestra revisable** (código, ya existe):
   ```bash
   python backend/scripts/clinical_validation_export.py --n 15 --days 45 --out /tmp/clinical_review.csv
   ```
   Produce un CSV (`utf-8-sig`, abre en Excel) con una fila por día: `target` vs `claim`(LLM) vs
   `recomp`(catálogo) por macro + `band_score` + `res_pct` + columnas en blanco
   `nutricionista_aprobado`/`notas`.

2. **Revisión + firma**: un nutricionista certificado (RD) revisa la muestra, marca
   `nutricionista_aprobado` (sí/no) y `notas` por cada plan/día.

3. **Archivar la evidencia**: guardar el CSV firmado (fecha + nombre del profesional) como evidencia
   auditable del release. Criterio sugerido de release: ≥90% de días aprobados sin objeción de seguridad.

4. **Re-validar** tras cambios grandes del generador (nuevo modelo LLM, activar el solver, cambios de
   condition_rules).

### Lo que queda como follow-up (opcional, code-closeable)

- Ingesta del CSV revisado a una tabla `clinical_validation_signoff` + gate de release por % de
  aprobación (no implementado — YAGNI hasta que exista una primera revisión firmada).
- Benchmark externo (NutriBench LATAM / INCAP / LATINFOODS) — requiere adquirir el dataset (P2).

---

## Estado de los P1 del audit (2026-06-14)

| P1 | Estado |
|---|---|
| P1-5 derivados de alérgenos | ✅ implementado + test |
| P1-3 embarazo/lactancia + condiciones comunes | ✅ implementado + test |
| P1-1 + P1-2 motor de macros | ✅ **default ON en código** (P1-MACRO-SOLVER-DEFAULT-ON, 2026-06-26); canary cerrado (0% fallback, proteína ~7.3% MAPE, all-4-en-banda ↑). Rollback: `MEALFIT_MACRO_SOLVER_ENABLED=False` |
| P1-4 revisión humana | ⏳ proceso documentado; ejecución por nutricionista pendiente |
