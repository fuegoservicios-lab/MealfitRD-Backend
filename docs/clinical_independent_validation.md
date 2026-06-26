# Validación INDEPENDIENTE del catálogo (P1-INDEPENDENT-VALIDATION)

[P1-INDEPENDENT-VALIDATION · 2026-06-26] Cierre del lado CODE-CLOSEABLE de la auditoría gap #2 ("cero
validación externa o humana — la validación existente recomputa desde el MISMO catálogo = auto-referencial").

## El problema raíz

La validación de precisión existente (`clinical_validation_export.py` / `benchmark_macro_compliance.py`)
recomputa los macros del plan desde el MISMO `master_ingredients` que usó el motor para generarlo. Si una
fila del catálogo tiene un valor errado, **el plan Y su "validación" comparten el error** → 0% de detección.
Sin una fuente de verdad INDEPENDIENTE del catálogo, cualquier número de precisión es autoafirmado.

## Qué cierra este harness (y qué NO)

`scripts/clinical_independent_validation.py` compara los macros per-100g del **catálogo vivo** contra una
tabla de referencia **USDA FoodData Central hardcodeada en el script** (independiente del catálogo) para una
muestra de ~24 staples es-DO. Un error de catálogo que la validación auto-referencial no ve, este SÍ lo caza.
Rompe la circularidad **para la muestra**.

**Lo que ESTO NO cierra (no es code-closeable — requiere acción del owner):**
- NO sustituye la revisión por un **nutricionista licenciado es-DO** (criterio clínico del TARGET, no solo
  precisión de entrega). Sigue pendiente (ver `clinical_validation.md`).
- NO es un benchmark externo COMPLETO (NutriBench / INCAP / LATINFOODS, dataset comercial del catálogo
  entero). Esta tabla es una muestra curada de 24 staples.

## Resultado de la corrida (2026-06-26, catálogo vivo en Neon)

96 celdas comparadas (24 ingredientes × 4 macros), **10 fuera de tolerancia** tras calibrar el ESTADO de la
referencia (no se dobla para ocultar errores — se corrige el estado: el catálogo guarda **arroz/lentejas
SECOS/crudos** y **atún light**, así que la referencia USDA usa el mismo estado).

**Hallazgos de convención (catálogo correcto, mi referencia estaba en otro estado):**
- `Arroz blanco` y `Lentejas` se guardan **CRUDOS/secos** (~365 y ~352 kcal/100g). ⚠️ **Implicación para el
  owner a investigar** (fuera del scope de gap #2): si una receta dice "1 taza de arroz cocido (150g)" pero
  el motor aplica 150g × 358 kcal/100g (valor SECO), sobre-cuenta ~2.7×. Verificar si los gramajes de receta
  son crudos o cocidos (el resolver/`available_sizes_g` puede manejarlo; confirmarlo).

**Candidatos GENUINOS a revisión (independiente diverge del catálogo):**
- `Papa.kcal` 61 vs USDA 77 (Δ-21%), `Papa.carbs` 12.4 vs 17.5 (Δ-29%): el catálogo subestima la papa cruda
  → **candidato #1 a corrección/revisión**.
- `Avena.protein` 13.2 vs 16.9 (Δ-22%): proteína de avena baja.
- Menores (varianza cultivar/medición, magnitud absoluta pequeña): `Manzana.kcal` +25%, `Tomate.kcal` +16%,
  `Piña.kcal` +20%, `Pechuga.fats`/`Arroz.fats`/`Carne.carbs` (deltas % grandes sobre valores ~0, clínicamente
  despreciables).

**Interpretación:** el catálogo es **mayormente consistente** con la referencia USDA independiente; la
circularidad queda rota para la muestra, con una lista triada de candidatos para revisión humana (el puente
al lado no-code del gap #2). NO se auto-corrige el catálogo desde aquí (un cambio de macro impacta todos los
planes; la dirección la confirma el nutricionista / el owner tras investigar el estado raw/cocido).

## Cómo correrlo

```bash
# En el VPS (env con NEON_DATABASE_URL):
/home/ubuntu/miniforge3/envs/mealfit/bin/python scripts/clinical_independent_validation.py
# CI / gate de release (exit 1 si hay drift fuera de tolerancia):
python scripts/clinical_independent_validation.py --strict
```

`_USDA_REF` (la referencia) y `_TOL` (tolerancias) viven en el script; la auto-consistencia de la referencia
(Atwater: kcal ≈ 4P+4C+9F) la ancla `tests/test_p1_independent_validation.py` (sin DB).

## Follow-up (code-closeable, YAGNI hasta primera revisión firmada)

- Ingesta de un CSV revisado por nutricionista a una tabla `clinical_validation_signoff` + gate de release
  por % de aprobación (ver `clinical_validation.md`). Diferido hasta que exista una primera revisión humana.
- Expandir `_USDA_REF` al catálogo entero si se adquiere un dataset de referencia (NutriBench/INCAP).
