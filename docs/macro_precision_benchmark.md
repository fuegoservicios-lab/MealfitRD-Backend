# Benchmark de precisión de macros (M2-MACRO-BENCHMARK)

[2026-06-14] Primer número MEDIDO de precisión de macros — reemplaza la banda "90-112%" autoafirmada.

Harness reproducible: [`scripts/benchmark_macro_compliance.py`](../scripts/benchmark_macro_compliance.py).
20 perfiles fijos held-out (diversos, no cherry-picked) → `arun_plan_pipeline` end-to-end contra
producción → desviación diaria |entregado − target| / target por macro. Mide **precisión de ENTREGA**
(el plan cumple su propio target), NO corrección clínica del target.

## Resultado (20 perfiles, 2026-06-14)

| | kcal | proteína | carbos | grasas |
|---|---|---|---|---|
| **Planes REALES (11)** MAPE | **0.5%** | **16.0%** | 12.6% | 12.2% |
| dentro ±10% | 100% | 48% | 52% | 55% |
| dentro ±20% | 100% | 61% | 70% | 79% |
| **4/4 macros en ±10%** | — | **solo 24% de los días** | — | — |
| **Planes FALLBACK (9)** | 100% en ±10% en TODO (matemático determinista, pero plan genérico) |

- **Tasa de fallback: 45%** (9/20 cayeron al plan matemático).
  - **[gap-audit G14 · 2026-06-15] Loop empírico cerrado (parcialmente):** query a `pipeline_metrics`
    (node='clinical_band', 21d) → 153 eventos, fallback global **28.8%**, PERO **147/153 son guests**
    (benchmark/smoke) y solo **6 eventos auth** (no significativo). Conclusión: la tasa real de usuarios
    AUTENTICADOS NO se puede confirmar por volumen insuficiente; el agregado está inflado por tráfico
    guest/benchmark. El fix G1 (None.dict() del planner → reintento en vez de fallback total, desplegado
    2026-06-15) debe bajar la tasa por CB. **Re-medir** cuando haya volumen auth real; el cron
    `plan_fallback_rate_high` (umbral 0.25) ya vigila la flota.
- **Lectura honesta:** las **calorías** se clavan (±0.5%), pero el **split de macros (proteína la peor, 16% MAPE)**
  cae dentro de ±10% solo ~la mitad de las veces. La banda "90-112%" NO se cumple para los macros individuales.

## Causa arquitectónica (análisis 6-lentes, 2026-06-14)

1. **El solver es GREEDY proporcional, no un optimizador multi-restricción** (`portion_solver.py:204-255`): escala
   cada grupo-de-macro por un factor; con ingredientes acoplados (pollo=P+grasa, arroz=C+P) no clava los 4 a la vez.
   El docstring lo admite: "si la telemetría muestra que grupos acoplados necesitan optimización conjunta, se añade LP".
   **Este benchmark ES esa telemetría.** → reemplazar por NNLS/LP (`scipy.optimize.nnls/linprog`, determinista, ~ms).
2. **El reconcile clava kcal+proteína y deja que carbo/grasa absorban el residual** (trade-off explícito) → carbo/grasa derivan.
3. **La cuantización corre AL FINAL** y reintroduce deriva sobre el target que el reconcile clavó (sin re-reconcile después).
4. **Ingredientes no-resueltos quedan sin contabilizar** (0-silencioso): el solver solo gobierna lo que resuelve en las 105 filas;
   platos criollos compuestos (sancocho/mangú/moro) caen fuera → la precisión REAL puede ser peor que la medida.

## Cómo correrlo

```bash
# En el VPS (env de prod) o local con .env:
PYTHONPATH=backend python backend/scripts/benchmark_macro_compliance.py 20 --concurrency 3
```

Pendiente (roadmap de validación): job de CI nightly contra un baseline commiteado + `clinical_band_score`
determinista persistido por-plan en prod (espejo del cron de coherencia).
