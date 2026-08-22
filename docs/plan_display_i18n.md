# P1-PLAN-DISPLAY-I18N — el plan legible en el idioma del usuario

[P1-PLAN-DISPLAY-I18N · 2026-08-19] Capa de DISPLAY pura sobre el plan: el motor genera y persiste en **español canónico siempre** (los nombres de alimentos son identificadores del sistema — P1-I18N-DASHBOARD); un campo paralelo `_display.<locale>` por meal lleva nombre/descripción/receta traducidos e ingredientes **bilingües** («30 g dried red beans (Habichuelas rojas)»), y la lista de compras del PDF glosa con `master_ingredients.name_en` («Black beans (Habichuelas negras)»). **El motor jamás lee `_display`**. Spec: [`docs/superpowers/specs/2026-08-19-plan-display-i18n-design.md`](../../docs/superpowers/specs/2026-08-19-plan-display-i18n-design.md).

## Piezas

| Pieza | SSOT | Nota |
|---|---|---|
| Motor de enriquecimiento | [`backend/plan_display_i18n.py`](../plan_display_i18n.py) | flash por lotes dimensionados por el TAMAÑO PROYECTADO de la salida (`_particionar_targets`; `..._BATCH_DAYS`=4 pasa a ser tope duro en días — P1-DISPLAY-LOTE-POR-COMIDAS), split-and-retry del lote que no parsea, validación determinista (línea sin canónico se descarta; arrays desalineados descartan el meal), TOCTOU por name+huella de ingredients/recipe (la COPIA del snapshot es load-bearing), lock in-process+KV con day-hash, fail-open TOTAL |
| Disparadores (5) | TRIGGER-1A (persist chunked, services.py), 1B (no-chunked/tier gratis, routers/plans.py), 2 (chunk worker post-commit), 4 (cambio de locale, user_data.py) | best-effort try/except; es-DO/guest ⇒ no-op |
| DELETE-on-write | anchors `...-MUTATOR-*`: swap, regenday, chatmod, recipeexpand + **6 re-escritores de gramos** (macroengine, capdm2, capbariatric, quantize, carbtrim, qtysync, fatstrim) | el pop vive EN el punto de mutación (pop-at-mutation): cualquier re-cuantización mata la traducción de ese meal — mejor español temporal que gramos mintiendo |
| Frontend | `frontend/src/utils/displayMeal.js` (+`shoppingHelpers.js`) | fallback CAMPO A CAMPO devolviendo el original TAL CUAL (legacy string recipe incluido); identidad (swap/likes/keys) SIEMPRE por el name canónico |
| Catálogo | columna `name_en` (migración `p1_plan_display_i18n_name_en.sql`, 347/347 pobladas 2026-08-19 vía `scripts/fill_catalog_name_en.py`) | **DISPLAY-ONLY, guard escopeta**: cero `name_en` en matchers |
| Slot ("Desayuno"/"Almuerzo"/...) — fase 1c | `frontend/src/utils/displayMeal.js::mealSlotLabel(slot, t)` | NUNCA lee `_display` (el slot es identificador de posición, no contenido del LLM) — mapeo directo canónico→`t(clave)`, case/acento-insensible, prefijo preservado en variantes ("Merienda AM"/"Merienda 1"/"Merienda Nocturna"); fallback al original si no reconoce. Consumido en Dashboard.jsx y History.jsx (modal). |
| Nombre del PLAN — fase 1c | `plan_data["_display"][locale]["name"]` (nivel PLAN, hermano del `_display` por-meal), escrito por `plan_display_i18n.py` en la MISMA llamada LLM que los meals (`_build_prompt(..., plan_name=...)`, addendum nativo por locale, contrato `{"meals":[...],"plan_name":"..."}`) | Intentado UNA vez por enriquecimiento (`plan_name_pending`, no una vez por lote); `_plan_name_already_translated` evita retraducir mientras la traducción vigente exista; TOCTOU por igualdad de `pd["name"]` (mismo patrón que meals, sin fingerprint de arrays porque no aplica). `/history-list` expone `plan_display_names` (clave ligera, `{locale: name}`); DELETE-on-write en `api_rename_plan` (`- '_display'` en el mismo UPDATE atómico, anchor `P1-PLAN-DISPLAY-I18N-MUTATOR-planrename`). |
| `/history-list::preview_meals[].display_names` — fase 1c | `routers/plans.py::api_plans_history_list` | Clave ligera `{locale: name}` extraída de `meal["_display"]`, SOLO names — jamás recipe/ingredients (el endpoint es polling del Historial). Key omitida por completo (no `null`/`{}`) cuando el meal no tiene `_display`. Frontend: `display_names[locale] ?? name`. |

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

`desc` vs `description` fue un REPEAT exacto de P1-DESC-KEY-DEAD (los meals persisten `desc`); la review de fase volvió a pagar el precedente F2 (3 HIGH cross-task que ninguna review por-task podía ver: re-escritores plan-wide sin pop, TOCTOU solo-name resucitando displays, y el helper colapsando recetas legacy). Tests: [`backend/tests/test_p1_plan_display_i18n.py`](../tests/test_p1_plan_display_i18n.py).

## Fase 1c — las 3 superficies que quedaban en español

[P1-PLAN-DISPLAY-I18N · fase 1c] Auditoría posterior a fase 1b encontró 3 superficies que renderizaban texto crudo español sin importar el idioma del dashboard: (A) la etiqueta de SLOT (`meal.meal`) en Dashboard.jsx/History.jsx — un valor CANÓNICO del dato, así que se resolvió con un helper de DISPLAY (`mealSlotLabel`), NO tocando el dato; (B) los chips del Historial (`m.name` truncado) y el modal del plan histórico (`meal.name`) — History.jsx no importaba el helper SSOT; (C) el nombre del PLAN (texto creativo del LLM, «Sazón Fuerte, Vida en Equilibrio»), mostrado en grande en card y modal del Historial, sin ninguna capa de traducción.

Nombre del plan: se enriquece EXACTAMENTE igual que los meals — mismo `_display` (nivel PLAN, hermano del de cada meal), misma llamada LLM (no una llamada aparte: costaría 2× por batch), misma frontera dura (nombres de alimentos dentro del título quedan en español canónico). Intentado UNA sola vez por enriquecimiento (no una vez por lote de días) y solo si `_plan_name_already_translated` dice que no hay traducción vigente — evita retraducir el mismo título en cada swap/regenday/chatmod que dispara `enrich_plan_display`. El rename manual (`PATCH /{plan_id}/name`) popea `plan_data["_display"]` (nivel plan) en el MISMO UPDATE atómico que cambia `name` — sin esto, renombrar a mano dejaría la traducción VIEJA visible.

`/history-list` gana dos claves ligeras — SOLO nombres, nunca `recipe`/`ingredients` (el endpoint es polling): `preview_meals[].display_names` (por meal) y `plan_display_names` (nivel plan, top-level de cada row). Ambas se omiten por completo (no `null`) cuando falta la traducción — el frontend cae con `?? name`.

Tests: [`backend/tests/test_p1_plan_display_i18n.py`](../tests/test_p1_plan_display_i18n.py) (sección "FASE 1c"), [`frontend/src/__tests__/displayMeal.test.js`](../../frontend/src/__tests__/displayMeal.test.js).

---

## Lo que cambió después de la fase 1c (2026-08-20 → 2026-08-22)

[P2-I18N-DOC-DISPLAY-CONGELADA · 2026-08-22] Esta doc se quedó fija en el 19-ago: citaba
cinco marcadores cuando el módulo ya llevaba veintidós, y la palabra «insights» no aparecía
ni una vez pese a existir `_INSIGHTS_ADDENDUM` y un disparador propio. Una doc SSOT
congelada es peor que ninguna — la de al lado (`i18n_dashboard.md`) ya provocó que una
auditoría dejara fuera la superficie i18n más cara del producto por creerle.

### Superficies que se añadieron

| Marcador | Qué entró |
|---|---|
| `P1-INSIGHTS-I18N` · 08-20 | El **razonamiento** del plan (`plan_data.insights`): el panel «Diagnóstico / Plan de Acción / Tip del Chef». Los títulos ya pasaban por `t()`; el cuerpo lo escribe el LLM y se quedaba en español con la app en inglés. Entra por `_INSIGHTS_ADDENDUM`, en la MISMA llamada que los meals. |
| `P1-PLAN-TITLE-I18N` · 08-20 | El título del plan no estaba a medias: estaba **INERTE**. `plan_data->>'name'` es `NULL` en todos los planes vivos —el nombre vive en la COLUMNA— así que al LLM nunca se le pedía `plan_name`. Sólo habría funcionado en planes RENOMBRADOS, que no había. |
| `P1-I18N-DISPLAY-NIVEL-PLAN-SIN-VIA` · 08-22 | Una línea: encolar un lote vacío cuando no hay días pendientes pero sí traducción de nivel plan. La rama estaba prevista y era inalcanzable. |

### Robustez del enriquecimiento

| Marcador | Qué cambió |
|---|---|
| `P2-DISPLAY-REDESPACHO-SIN-FILTRO` | Se exige **USABLE, no presente**. Misma lección que `P1-I18N-GATE-VALOR` dejó en el validador de catálogos: medir que la clave existe no es medir que sirve. Un display a medias dado por bueno deja esa comida en español **para siempre**, porque nadie la reintenta. |
| `P1-I18N-DISPLAY-LOTE-PERDIDO-SIN-SENAL` · 08-22 | Un lote que revienta al invocar se reintenta mientras quede presupuesto; si no, cuenta como perdido con `logger.error`. Y el resultado distingue `partial_loss` de éxito: antes, escribir 3 de 4 lotes se reportaba igual que escribir 4. |
| `P2-DISPLAY-VALIDADOR-SIN-CIFRAS` | Las cifras de la línea tienen que sobrevivir a la traducción. El separador decimal se normaliza (`1.5` → `1,5` es lo que un francés espera), y se comparan ORDENADAS porque el orden dentro de la frase sí puede cambiar. |
| `P3-DISPLAY-SUBSTRING-SIN-FRONTERA` | El validador comparaba con un `in` pelado y aprobaba por accidente. Es la clase de defecto que este repo ya pagó tres veces: «sal» dentro de Salami, «pollo» dentro de repollo, «res» dentro de fResco. |
| `P2-DISPLAY-ECO-NOMBRE` | Un nombre «traducido» que sólo cambia la caja o los acentos («HABICHUELAS guisadas») no es una traducción. Normaliza con NFKD y descarta combinantes. |
| `P1-DISPLAY-VOCAB-CERRADO` · `P2-RECIPE-NOTES-NOT-STEPS` | Un paso de receta no es prosa lisa: empieza por una etiqueta de sección, o es una nota. Una nota NUMERADA como acción de cocina convierte «Nota del nutricionista» en «Step 2». |
| `P1-I18N-DISPLAY-CANONICO-PARTITIVO` · 08-22 | Partitivos (`diente`, `ramita`, `puñado`, `lata`…) y fracciones vulgares (`⅓`, `⅔`, `⅛`…) entran en el prefijo de cantidad, que es lo que el validador exige conservar. |

### Coste y observabilidad

| Marcador | Qué cambió |
|---|---|
| `P2-DISPLAY-RETENCION-LOCALES` | `_display` sólo AÑADÍA idiomas: un plan de 30 días visitado en los cinco guardaba cinco copias completas del texto dentro de la misma fila. Ahora se evacúa el idioma abandonado. |
| `P2-MUTATOR-PURITY` | El `mutator` corre DENTRO del `SELECT … FOR UPDATE` de `update_plan_data_atomic`: **puro, CPU-only, sin IO ni LLM ni re-entrada al pool** — sostener el lock durante los segundos que dura la llamada LLM no es viable, así que `targets` se construye FUERA y el TOCTOU se cierra por huella. Acumular en los dicts `counters` (closures en memoria) no viola esa pureza. |
| `P2-DISPLAY-SIN-TELEMETRIA-RESULTADO` | El módulo instrumentaba lo que se GASTA (`llm_usage_events`) y nada de lo que PASA: cero referencias a `pipeline_metrics`. |

### ⚠️ Lo que ninguna de esas líneas dice, y es lo que más importa

[P1-I18N-SIN-EVIDENCIA-PRODUCCION] **Esta capa no ha traducido un plato en producción.**
Medido el 2026-08-22 sobre la base real: **5 ejecuciones en toda su historia** (contra 3.789
del generador de días), **1 plan de 44** con `_display`, **0 comidas** traducidas, **0 filas**
de telemetría, y **0 de 19 usuarios** con un locale distinto de `es-DO`.

Todo lo de arriba está verificado por tests y **ninguna cantidad de tests verdes puede
cerrar esa pregunta**: los tests miden el archivo, no el mundo. La condición de salida es
una ejecución real por idioma contra un plan real. Hasta entonces, cada afirmación de esta
doc sobre el comportamiento en vivo es una afirmación sobre el código, no sobre producción.
