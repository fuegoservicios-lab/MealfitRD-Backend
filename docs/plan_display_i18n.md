# P1-PLAN-DISPLAY-I18N — el plan legible en el idioma del usuario

[P1-PLAN-DISPLAY-I18N · 2026-08-19] Capa de DISPLAY pura sobre el plan: el motor genera y persiste en **español canónico siempre** (los nombres de alimentos son identificadores del sistema — P1-I18N-DASHBOARD); un campo paralelo `_display.<locale>` por meal lleva nombre/descripción/receta traducidos e ingredientes **bilingües** («30 g dried red beans (Habichuelas rojas)»), y la lista de compras del PDF glosa con `master_ingredients.name_en` («Black beans (Habichuelas negras)»). **El motor jamás lee `_display`**. Spec: [`docs/superpowers/specs/2026-08-19-plan-display-i18n-design.md`](../../docs/superpowers/specs/2026-08-19-plan-display-i18n-design.md).

## Piezas

| Pieza | SSOT | Nota |
|---|---|---|
| Motor de enriquecimiento | [`backend/plan_display_i18n.py`](../plan_display_i18n.py) | flash por lotes (knob `..._BATCH_DAYS`=4), validación determinista (línea sin canónico se descarta; arrays desalineados descartan el meal), TOCTOU por name+huella de ingredients/recipe (la COPIA del snapshot es load-bearing), lock in-process+KV con day-hash, fail-open TOTAL |
| Disparadores (5) | TRIGGER-1A (persist chunked, services.py), 1B (no-chunked/tier gratis, routers/plans.py), 2 (chunk worker post-commit), 4 (cambio de locale, user_data.py) | best-effort try/except; es-DO/guest ⇒ no-op |
| DELETE-on-write | anchors `...-MUTATOR-*`: swap, regenday, chatmod, recipeexpand + **6 re-escritores de gramos** (macroengine, capdm2, capbariatric, quantize, carbtrim, qtysync, fatstrim) | el pop vive EN el punto de mutación (pop-at-mutation): cualquier re-cuantización mata la traducción de ese meal — mejor español temporal que gramos mintiendo |
| Frontend | `frontend/src/utils/displayMeal.js` (+`shoppingHelpers.js`) | fallback CAMPO A CAMPO devolviendo el original TAL CUAL (legacy string recipe incluido); identidad (swap/likes/keys) SIEMPRE por el name canónico |
| Catálogo | columna `name_en` (migración `p1_plan_display_i18n_name_en.sql`, 347/347 pobladas 2026-08-19 vía `scripts/fill_catalog_name_en.py`) | **DISPLAY-ONLY, guard escopeta**: cero `name_en` en matchers |

## Knobs

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_PLAN_DISPLAY_I18N` | `true` | kill switch TOTAL: motor + attach de `display_name_en` en el aggregator (FF-6) |
| `MEALFIT_PLAN_DISPLAY_I18N_MODEL` | flash | modelo del enriquecimiento |
| `MEALFIT_PLAN_DISPLAY_I18N_BATCH_DAYS` | 4 | días por llamada (evita truncamiento) |

Costo: a `llm_usage_events` con `node="plan_display_i18n"` — JAMÁS a `api_usage` (cero crédito del usuario).

## Decisiones de producto (FF-7/8/9 de la review de fase — no son bugs)

- **Gloss inglés para TODO locale ≠ es-DO** (fr-FR ve «Black beans (Habichuelas negras)»): decisión v1 — inglés como lingua franca del gloss de compra; columnas por idioma serían fase posterior si se pide.
- **Backfill de 2 bordes** (trigger 4 decide mirando primer y último día): un día intermedio puede quedar en español sin camino de recuperación automático hasta el próximo disparador natural (bloque nuevo, mutación, re-cambio de idioma). Aceptado: el fallback es español correcto, nunca contenido mintiendo. Compuesto con los pops colaterales de los re-escritores (que NO re-despachan a propósito), el estado estable es «ese día en español» — legal por spec.
- **es-DO y la clave `display_name_en`**: los items de lista persistidos llevan el campo también para usuarios es-DO (el aggregator no conoce locale) — inerte: el frontend solo glosa con locale ≠ es. Excepción documentada a la byte-identidad.
- La lista EN PANTALLA sigue en español para todos (solo el PDF glosa) — alcance fase 1b.

## Lecciones del ciclo (detalle en la memoria del proyecto)

`desc` vs `description` fue un REPEAT exacto de P1-DESC-KEY-DEAD (los meals persisten `desc`); la review de fase volvió a pagar el precedente F2 (3 HIGH cross-task que ninguna review por-task podía ver: re-escritores plan-wide sin pop, TOCTOU solo-name resucitando displays, y el helper colapsando recetas legacy). Tests: [`backend/tests/test_p1_plan_display_i18n.py`](../tests/test_p1_plan_display_i18n.py) (191).
