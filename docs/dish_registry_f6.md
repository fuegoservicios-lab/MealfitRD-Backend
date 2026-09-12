# Dish Registry compilado — Fase 6 del roadmap 2.5 (capa V2.3)

[P1-ARQ25-F6-DISH-REGISTRY · 2026-09-05] Motor SSOT: [`backend/dish_registry.py`](../dish_registry.py). Curación DO:
`data/dish_constituents_do.json` (tabla curada a mano, SSOT; validador
[`scripts/check_dish_constituents_do.py`](../scripts/check_dish_constituents_do.py)).
Compilador CLI: [`scripts/compile_dish_registry.py`](../scripts/compile_dish_registry.py) → `data/registry/`. Test ancla:
[`tests/test_p1_arq25_f6_dish_registry.py`](../tests/test_p1_arq25_f6_dish_registry.py). Roadmap §7 y Fase 6.

## Qué entrega

| Entregable (roadmap) | Dónde |
|---|---|
| `constituents` para las 87 plantillas DO | `data/dish_constituents_do.json`, compuestos EN ORIGEN (F6) desde los componentes con gramos de `dominican_dish_recipes.json` (las mismas 60 recetas del diario) + ítems con nombre EXACTO del catálogo. **[P1-PLAN-LOTE-13 · 2026-09-12] El JSON es el SSOT y se edita a mano**: el generador `build_dish_constituents_do.py` dejó de reproducirlo (75 entradas divergían tras las curaciones del 09-09/09-10 y C8) y se retiró (`git show e4528a22:scripts/build_dish_constituents_do.py`); `scripts/check_dish_constituents_do.py` valida la coherencia con las plantillas (sin catálogo) y `compile_dish_registry.py --check` la resolución. Las cuatro exclusiones históricas (zapote, menta, chillo, salami de pavo) las cerró el dueño en C8 (P1-PLAN-LOTE-11). |
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

## Rebanada 2 (P1-ARQ25-F6-REGISTRY-PROMPT · 2026-09-05)

- **Candidatos en el prompt**: `horizon.registry_prompt_lines` añade al bloque 📐 «Platos del registro curado para este
  bloque: Día N → almuerzo: A | B · cena: C | D» (2 por franja, familia programada, sin las clases de alérgeno
  declaradas). Knob `MEALFIT_DISH_REGISTRY_PROMPT` (True); apagado ⇒ prompt byte-idéntico al anterior. La métrica de
  fidelidad lleva `registry_in_prompt` para comparar antes/después (`pipeline_metrics.plan_policy_fidelity`).
- **Logística derivada** (`logistics`, `estimated: true`): `batch_friendly`/`freezer_friendly`/`difficulty_est` por
  técnica; `prep_minutes_est` con `prep_minutes_source` = `receta` (el tiempo que la receta congelada DECLARA,
  P1-MINUTOS-DE-LA-RECETA · 2026-09-10) o `tecnica` (estimación por técnica); el relleno `defecto` (30) existe en el
  snapshot pero **ningún lector lo sirve** (`recipe_library.prep_time_for_meal`, P1-AUDITORIA-ARQ-VERIFICADA);
  `min_shelf_life_days` = mínimo de la vida útil de sus constituyentes (catálogo).
- **Editorial** (`editorial`): `status=curated`, `source`, `display_name.es`, `aliases` (de `plan_policy.TEMPLATE_ALIASES`),
  `media: []` (Fase 8). Snapshot `schema_version` 2 / `compiler_version` 2 (recompilado: cambia el hash).

## Fuera de la fase

Referencias a medios reales (Fase 8) y el editor DB que publique el mismo snapshot (§7.3 punto 5).
