# Base de macros nutricionales (`master_ingredients`) — integración USDA

[P2-MDDA-NUTRITION-AUDIT · 2026-06-13] Cimiento de datos del lado determinista del
"cerebro dividido" (MDDA). El solver de porciones ([`portion_solver.py`](../portion_solver.py))
y el lookup ([`nutrition_db.py`](../nutrition_db.py)) leen los macros reales por-100g de
estas columnas para computar y clavar los macros de cada comida, en vez de que el LLM los
adivine.

## Esquema (migración SSOT)

[`migrations/p1_food_db_nutrition_schema_2026_06_13.sql`](../migrations/p1_food_db_nutrition_schema_2026_06_13.sql)
añade a `master_ingredients` (todas NULLABLE → degradación grácil):

| Columna | Tipo | Nota |
|---|---|---|
| `kcal_per_100g` | numeric | **Atwater 4·P + 4·C + 9·F** (auto-consistente con el solver, NO la "Energy" de USDA) |
| `protein_g_per_100g` / `carbs_g_per_100g` / `fats_g_per_100g` | numeric | macros por 100g comestibles |
| `fiber_g_per_100g` / `sodium_mg_per_100g` | numeric | micros base (extensible) |
| `nutrition_source` | text CHECK | `usda` \| `off` \| `faoinfoods` \| `manual` |
| `nutrition_source_date` | date | fecha del import |
| `fdc_id` | bigint | FoodData Central id (trazabilidad). NULL para manual |
| `is_dominican_cultivar` | boolean | true = vianda/cultivar DD con macros estimados (pendiente validación nutricionista) |

> **Por qué Atwater y no la Energy de USDA**: el solver escala porciones por macro y deriva
> kcal de los macros. Usar la Energy metabolizable de USDA (que aplica factores específicos
> por alimento) rompería la identidad `kcal = 4P+4C+9F` que el solver asume. Ej.: mantequilla
> de maní da 640 kcal por Atwater vs 588 "real" — 640 es el correcto **para el solver**.

## Poblado — [`scripts/populate_nutrition_db.py`](../scripts/populate_nutrition_db.py)

One-shot operacional (NO corre en runtime de generación). Requiere USDA key propia
(gratis: fdc.nal.usda.gov/api-key-signup.html; DEMO_KEY topa 50/día).

```bash
export USDA_API_KEY=<tu key>
conda activate mealfit
python scripts/populate_nutrition_db.py                 # pobla los 105
python scripts/populate_nutrition_db.py --dry-run        # no escribe
python scripts/populate_nutrition_db.py --names "Avena,Leche"   # re-corre un subset
```

Tres fuentes, en orden de precedencia:
1. **`FDC_PIN`** {name: fdc_id} — fetch EXACTO por `fdc_id` (endpoint `/food/{id}`, determinista,
   sin ranking de búsqueda). Las correcciones post-auditoría viven aquí.
2. **`MANUAL_MACROS`** — 8 procesados/DD que USDA no cubre bien (Casabe, Longaniza/Salami
   dominicano, Queso de hoja, Proteína en polvo, Sal, Estevia, Vinagre blanco).
3. **`USDA_QUERY`** {name: query} — búsqueda USDA (Foundation/SR Legacy), `pageSize=5` →
   primer resultado con macros reales (salta filas Foundation vacías).

## Auditoría de calidad (workflow multi-agente)

[2026-06-13] El heurístico "primer resultado de búsqueda" es **poco fiable** (el ranking USDA
no siempre pone el alimento correcto primero; comas/paréntesis en el query rompen la búsqueda).
Un workflow de 7 auditores paralelos + 1 crítico de completitud revisó los 105 contra valores
de referencia es-DO. Encontró **8 errados de 105**; 7 corregidos vía `FDC_PIN` verificado:

| Ingrediente | Match errado | Corrección (fdc) |
|---|---|---|
| Avena | "Oil, oat" (900 kcal) | 173904 Cereals, oats |
| Batata | hojas de batata | 168482 Sweet potato, raw, root |
| Chinola | jugo de chinola | 169108 Passion-fruit, raw |
| Leche | queso mozzarella | 171265 Milk, whole 3.25% |
| Naranja | cáscara/jugo (~2×) | 169097 Oranges, raw |
| Guineo verde | ficha de plátano (152 kcal) | 173944 Bananas, raw |
| Mantequilla de maní | reduced-fat (F muy baja) | 172470 Peanut butter, full-fat |

**Pimentón** NO se corrige: el auditor lo leyó como ají fresco, pero el catálogo ya tiene
"Pimiento morrón" (rojo fresco) y "Ají cubanela" — así que "Pimentón" como entrada distinta es
la **especia paprika** (388 kcal), no un duplicado. Decisión documentada.

## Verificación

```bash
pytest backend/tests/test_food_db_population_coverage.py -v
```
- Capa parser (offline): mapeo cubre 105, las 7 correcciones ancladas en `FDC_PIN`, MANUAL sano.
- Capa integración (DB-gated, skip sin Neon): ≥95/105 con `kcal_per_100g` + identidad Atwater.

## Extender / re-validar

- Nuevo ingrediente → añadir a `USDA_QUERY` (o `MANUAL_MACROS` si USDA no lo cubre) + correr `--names`.
- Si un valor luce mal → verificar fdc candidato con el detail endpoint, pinear en `FDC_PIN`, `--names`.
- Viandas DD (`is_dominican_cultivar=true`) usan la especie USDA más cercana — pendiente curación
  por nutricionista con datos FAO/INFOODS Caribe cuando estén disponibles.
