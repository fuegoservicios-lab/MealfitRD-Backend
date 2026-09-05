# Dish Registry compilado — Fase 6 del roadmap 2.5 (capa V2.3)

[P1-ARQ25-F6-DISH-REGISTRY · 2026-09-05] Motor SSOT: [`backend/dish_registry.py`](../dish_registry.py). Curación DO:
[`scripts/build_dish_constituents_do.py`](../scripts/build_dish_constituents_do.py) → `data/dish_constituents_do.json`.
Compilador CLI: [`scripts/compile_dish_registry.py`](../scripts/compile_dish_registry.py) → `data/registry/`. Test ancla:
[`tests/test_p1_arq25_f6_dish_registry.py`](../tests/test_p1_arq25_f6_dish_registry.py). Roadmap §7 y Fase 6.

## Qué entrega

| Entregable (roadmap) | Dónde |
|---|---|
| `constituents` para las 87 plantillas DO | `data/dish_constituents_do.json`, compuestos desde los componentes con gramos de `dominican_dish_recipes.json` (las mismas 60 recetas del diario) + ítems con nombre EXACTO del catálogo. Lo que el catálogo no tiene (zapote, menta, chillo) se declara y el compilador lo lista como exclusión. |
| Compilador → snapshot inmutable por versión/país/cultura | `compile_library(lib)` → `data/registry/dish_registry_<lib>_v<versión>.json` (6 bibliotecas: do/es/mx/co/pr/us). JSON canónico (claves ordenadas, sin timestamps): misma fuente + mismo catálogo ⇒ mismos bytes. `snapshot_hash`, `source_hash` (plantillas + constituyentes), `catalog_fingerprint` (nombres + nutrición del catálogo). |
| Tags de riesgo derivados (§7.2), cero tags clínicos manuales | `derive_risk_attributes`: por porción, desde las columnas por 100 g del catálogo — `sodium_high` (≥600 mg), `potassium_high` (≥700), `phosphorus_high` (≥350), `sat_fat_high` (≥6 g), `sugar_high` (≥25 g), `glycemic_load_high` (carbohidrato neto ≥75 g), `energy_dense` (≥800 kcal), `processed_meat` (+ ítems), `allergens` (clases del vocabulario SSOT `graph_orchestrator._ALLERGEN_SYNONYMS`). Nunca `safe_for_*`: la elegibilidad se evalúa en runtime con el plato ya dimensionado. |
| Resolubilidad 100 % o exclusión explícita | Cada constituyente resuelve por nombre canónico o alias (sin acentos, singular/plural) o entra en `excluded[]` con `reason ∈ {not_in_catalog, no_grams, declared_unresolved}`. `status`: `ok` (todo resuelve) · `partial` (algo excluido) · `excluded` (nada). Solo `ok` se ofrece al allocator. |
| El allocator consume el snapshot; benchmarks guardan su hash | `build_blueprint` lleva `registry: {snapshot_hash, version, candidates[día:franja] → template_ids}` (`template_candidates`: franja + familia de proteína + sin las clases de alérgeno declaradas). `fidelity_report`/`emit_fidelity_metric` guardan `registry_hash` en `pipeline_metrics.plan_policy_fidelity`. |

## Knob

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_DISH_REGISTRY_SNAPSHOT` | `"1"` | Versión activa: `load_registry(country)` lee `dish_registry_<lib>_v<versión>.json`. Sin archivo ⇒ `None` (fail-open: el allocator sigue sin candidatos, nada bloquea). |

## Runbook

- **Recompilar** (tras editar plantillas, la curación DO o el catálogo): `python backend/scripts/compile_dish_registry.py`
  (abre el pool de Neon: necesita `backend/.env`). Commit de `data/registry/*.json` con el cambio de fuente.
- **Verificar reproducibilidad**: `python backend/scripts/compile_dish_registry.py --check` → exit 3 si algún hash difiere
  del disco. Un drift sin cambio de fuente significa que cambió el catálogo (`catalog_fingerprint` distinto): es
  información, no un bug — recompila y anota.
- **Nueva versión**: `--version 2` escribe `_v2.json` sin tocar la v1; el flip es el knob.
- **Qué NO hacer**: editar un snapshot a mano (`verify_snapshot` lo rechaza al cargar) o añadir tags clínicos a las
  plantillas (§7.2).

## Gate de la fase y cómo medirlo

1. 100 % de constituyentes resuelve o queda excluido: `stats.constituents == stats.resolved + Σ excluded` (test_e).
2. Reproducible bit a bit: `--check` en verde; test_c compila dos veces y compara bytes.
3. Sin regresión clínica ni de fidelidad: el snapshot NO altera prompts todavía (los candidatos viajan en el blueprint y
   la métrica lleva el hash); inyectarlos al prompt es la siguiente rebanada, con su medición.

## Fuera de esta rebanada

Candidatos del registry en el prompt del generador (hoy solo en el blueprint), metadata batch/freezer/shelf-life por
plantilla, nombre localizado y aliases editoriales, referencias a medios (Fase 8), editor DB que publique el mismo
snapshot (§7.3 punto 5).
