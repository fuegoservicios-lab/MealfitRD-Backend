# Sistema de países — Fase 1 (el motor deja de forzar lo criollo en beta)

[P1-COUNTRY-SYSTEM-F1 · 2026-08-16] Con el knob maestro `MEALFIT_COUNTRY_SYSTEM` encendido, un
usuario de país beta (`COUNTRY_PROFILES[cc]['is_beta'] is True` — hoy ES/US/MX/PR/CO, ver
[`constants.py`](../constants.py)) recibe un plan sin imposición criolla: prompts por variante,
gates culturales suaves, fecha local independiente del país, presupuesto en su moneda, lista de
compras sin precios con aviso honesto. Con el knob apagado (default) el motor es **byte-idéntico**
a antes de esta fase — 8 tasks SDD, TDD completo, deploy en oscuro. Spec:
[`docs/superpowers/specs/2026-08-16-sistema-paises-design.md`](../../docs/superpowers/specs/2026-08-16-sistema-paises-design.md)
§Fase 1. Plan: [`docs/superpowers/plans/2026-08-16-paises-fase-1.md`](../../docs/superpowers/plans/2026-08-16-paises-fase-1.md).
Fase 0 (el dato, sin lectores): memoria `project_country_system_f0_2026_08_16.md`.

## La espina: `country_for_form_data`

[`constants.country_for_form_data(form_data) -> str`](../constants.py) (T1, constants.py:3209) es
la **ÚNICA** puerta de lectura de país del motor. Knob apagado (o `form_data` no-dict) ⇒ `'DO'`
SIEMPRE; encendido ⇒ `canonicalize_country(form_data.get('country'))`. Todo consumidor de país en
T2-T7 pasa por aquí — nunca lee `form_data['country']` crudo. El guard blanket (T8,
`test_p1_country_system_f1.py::test_country_for_form_data_es_el_unico_lector_en_los_6_modulos`)
enforza esto sobre `graph_orchestrator.py`/`cron_tasks.py`/`shopping_calculator.py`/
`nutrition_calculator.py`/`agent.py`/`tools.py` — `constants.py` queda fuera del barrido a
propósito (ahí vive la única lectura legítima). Extiende el guard homónimo de Fase 0
(`test_p1_country_system_f0.py`, scope=solo `graph_orchestrator.py`).

Hermanos del mismo patrón: `pricing_mode_for_country`/`pricing_mode_for_form_data` (T7,
constants.py:3239/3254 — SSOT del literal `'beta_no_prices'`).

## Surfaces wired en F1

| # | Surface | Mecanismo | Ubicación | Task |
|---|---|---|---|---|
| 1 | Prompt day-gen | `build_day_generator_system_prompt(diet, country)` — DO ⇒ retorno `is`-idéntico; beta ⇒ `_BETA_FRAGMENT_TABLE` (consciente de dieta) apilada sobre el render de dieta | [`prompts/day_generator.py`](../prompts/day_generator.py) | T2 |
| 2 | Derivación por chunk | `_day_system_instruction_for_diet(form_data)` deriva país una vez | [`graph_orchestrator.py:5251`](../graph_orchestrator.py#L5251) | T2 |
| 3 | Contexto compartido planner+day-gen | `_country_context_block(country)` — DO ⇒ `""`; beta ⇒ bloque breve. UNA sola llamada en `_build_shared_context` | [`graph_orchestrator.py:4720`](../graph_orchestrator.py#L4720) (block), [`:4736`](../graph_orchestrator.py#L4736) (`_build_shared_context`) | T3 |
| 4 | Juez culinario LLM | `_culinary_judge_rubric_for_country(country)` — DO ⇒ `_CULINARY_JUDGE_RUBRIC` byte-idéntico (cacheado); beta ⇒ sustituye "juez culinario dominicano experto" por variante con `COUNTRY_PROFILES[cc]['name_es']` | [`graph_orchestrator.py:6448`](../graph_orchestrator.py#L6448) (rubric), [`:6464`](../graph_orchestrator.py#L6464) (`run_culinary_judge(plan, country="DO")`) | T3 |
| 5 | Gates culturales suaves | `slot_rules_for_country(country)` — DO ⇒ el MISMO objeto `SLOT_INAPPROPRIATE_FOODS` (`is`); beta ⇒ tabla derivada memoizada, TODA regla `hardness='soft'` (mismos tokens — sigue midiendo por telemetría) | [`constants.py:2094`](../constants.py#L2094) | T4 |
| 6 | Detector determinista de apropiación horaria | `_detect_slot_appropriateness(days, form_data)` deriva país internamente (ver tabla de callers abajo) | [`graph_orchestrator.py:10298`](../graph_orchestrator.py#L10298) | T4 |
| 7 | Backstop de swap/chat-modify | `slot_coherence_backstop_for_meal(meal, meal_type, country="DO")` — DO conserva TODA violación (byte-idéntico); beta filtra a solo `hard` (hoy: ninguna) | [`graph_orchestrator.py:14709`](../graph_orchestrator.py#L14709) | T4 (fix-round 1) |
| 8 | Autofixes de arroz | `_night_rice_autofix`/`_breakfast_rice_autofix` ganan gate `country` — beta ⇒ no-op completo (el autofix ES criollo por diseño; la detección ya degrada a soft/telemetría aparte) | [`graph_orchestrator.py:27548`](../graph_orchestrator.py#L27548) / [`:29184`](../graph_orchestrator.py#L29184) | T4 |
| 9 | Finalizer de receta (swap/chat-modify) | `finalize_single_meal_recipe_coherence(..., country="DO")` — hilvana el gate anterior a su propio `_night_rice_autofix` interno | [`graph_orchestrator.py:26982`](../graph_orchestrator.py#L26982) | T4 (fix-round 1) |
| 10 | §16 contrato de horario en el prompt | `build_meal_timing_rules(meal_type, country="DO")` — DO idéntico byte a byte; beta omite la enumeración negativa (labels dominicanos intencionales) y usa `_SLOT_POSITIVE_HINT_NEUTRAL` | [`constants.py:2225`](../constants.py#L2225) | T4 |
| 11 | Fecha local del usuario | `user_tz_offset_min(user_id)` — los 4 SQL con `'America/Santo_Domingo'` hardcodeado pasan a `NOW() - make_interval(mins => offset)`; fallback 240 preservado | [`db_facts.py:181`](../db_facts.py#L181) (helper); call sites en `db_facts.py`/`tools.py`/`proactive_agent.py` | T5 |
| 12 | Presupuesto multi-moneda | `budget_floor_in_currency(days, currency, min_budget_dop)` + pisos EUR/MXN/COP (factores fijos provisionales, ver spec); `validate_budget_sufficient` emite el 422 en la moneda del usuario | [`nutrition_calculator.py:1894`](../nutrition_calculator.py#L1894) (floor), [`:2010`](../nutrition_calculator.py#L2010) (validate) | T6 |
| 13 | Piso de presupuesto en frontend | `effectiveBudgetCurrency(country, budgetCurrency, countrySystemUI)` — resuelve DOP/USD siempre; EUR/MXN/COP solo tras `COUNTRY_SYSTEM_UI` + país beta con esa moneda | [`frontend/src/config/formValidation.js:258`](../../frontend/src/config/formValidation.js#L258) | T6 |
| 14 | Flag de modo beta de precios | `pricing_mode_for_form_data(form_data)` estampado en `plan_data['_pricing_mode']` dentro de `assemble_plan_node`, ANTES del bloque de agregación de listas | [`graph_orchestrator.py:37797`](../graph_orchestrator.py#L37797) | T7 |
| 15 | Choke point del aggregator | `_strip_prices_for_beta_pricing_mode(res)` al final de `get_shopping_list_delta` — cubre los ~15 callers reales (agent/cron_tasks/routers/plans/tools) sin threadear un parámetro por función | [`shopping_calculator.py:12146`](../shopping_calculator.py#L12146) (strip), [`:12173`](../shopping_calculator.py#L12173) (`get_shopping_list_delta`) | T7 |
| 16 | Resumen de costo/reconciliación | `compute_shopping_cost_summary(..., pricing_mode=None)` — beta ⇒ `None` (nunca un dict de ceros); **8 call sites de producción**, todos gateados (ver CONTRATO en `graph_orchestrator.py`, reapuntado 5→8 en T8) | [`shopping_calculator.py:3953`](../shopping_calculator.py#L3953) | T7 (+ fix-round 2) |
| 17 | Sugerencias de presupuesto | `build_budget_suggestions` — sin código nuevo: filtra `estimated_cost_rd > 0`, y con el aggregator ya en `None` el resultado es `[]` siempre | [`shopping_calculator.py:3882`](../shopping_calculator.py#L3882) | T7 |
| 18 | Writer optimista del cliente | `_rebuildItemFromVariant(it, variant, suppressCost)` — `_suppressCost = plan?._pricing_mode === 'beta_no_prices'` | [`frontend/src/pages/Dashboard.jsx:335`](../../frontend/src/pages/Dashboard.jsx#L335) | T7 |
| 19 | Aviso PDF + panel Marcas | Cabecera/pie del PDF alternan con `_isBetaPricing`; panel `SupermarketBrands` oculto con el flag | [`frontend/src/pages/Dashboard.jsx:3247`](../../frontend/src/pages/Dashboard.jsx#L3247), [`:6918`](../../frontend/src/pages/Dashboard.jsx#L6918) | T7 |

## Barrido de callers: las 6 funciones derivadas de `SLOT_INAPPROPRIATE_FOODS`

[P1-COUNTRY-SYSTEM-F1 · T8] Mandato del controller tras el fix-round 1 de T4 (`progress.md`:
"barrido de TODOS los callers... antes de cerrar la fase" — 2 reviews sucesivas de T4 encontraron
caller tras caller no-gateado, la última vez en `routers/plans.py`, dos veces). Tabla completa,
verificada con un scanner tokenize-based (`test_p1_country_system_f1.py`, sección T8 —
`_scs_classify`) que enumera TODO call site de producción de cada función (comentarios/docstrings
enmascarados, alias de import resueltos, imports multi-línea parenthesized incluidos) y lo
clasifica `wired` (pasa el arg de país/tabla) o `exento` (marker `# [P1-COUNTRY-SYSTEM-F1 EXENTO:
<razón>]` a pocas líneas). **0 call sites sin clasificar** — el test falla si aparece uno nuevo sin
ninguna de las dos etiquetas.

| Función | Firma país-aware | Call sites | Wired | Exento | Exento: razón |
|---|---|---|---|---|---|
| `slot_rules_for_country` | `(country)` — la derivación misma | 3 | 3 | 0 | — |
| `_detect_slot_appropriateness` | `(days, form_data)` — deriva país internamente | 5 | 5 | 0 | — |
| `slot_coherence_backstop_for_meal` | `(meal, meal_type, country="DO")` | 2 | 1 | 1 | `routers/plans.py:7129` (`_swap_mutator`, dentro del `SELECT...FOR UPDATE` de `update_plan_data_atomic` — P2-MUTATOR-PURITY prohíbe reentrar al pool ahí para resolver país; requiere pre-fetch antes del lock, ver "Parqueado para Fase 2") |
| `build_meal_timing_rules` | `(meal_type, country="DO")` | 5 | 4 | 1 | `prompts/day_generator.py:244` (precompute import-time de `_SLOT_SSOT_RULES_BLOCK` — SIEMPRE DO por diseño, alimenta la constante estática `DAY_GENERATOR_SYSTEM_PROMPT`; la variante beta vive en `_SLOT_SSOT_RULES_BLOCK_BETA`, call site separado, SÍ wired) |
| `slot_violations_for_meal_name` | `(name, slot_key, rules_table=None)` | 8 | 7 | 1 | `plan_gym.py:138` (gym/benchmark OFFLINE — `scripts/plan_gym.py`/`scripts/landing_benchmark.py`, nunca corre en el request path de un usuario; puntúa deliberadamente contra la tabla nativa DO como vara de calidad fija) |
| `slot_ingredient_violations` | **(ingredients, slot_key) — SIN parámetro de país** | 2 | 0 | 2 | Estructural: la firma no tiene hook de país. `graph_orchestrator.py:10359` (dentro de `_detect_slot_appropriateness`) SÍ es país-consciente pero vía override en el dict del issue, no vía argumento — `"hard": v["hard"] and _country == "DO"`, 18 líneas después del call, documentado en el docstring de la función. `graph_orchestrator.py:14755` (dentro de `slot_coherence_backstop_for_meal`) es el residual disclosed-y-no-cerrado de T4 fix-round 1: sigue siempre `hard=True`, sin override — gap real, ver "Parqueado para Fase 2" |
| **Total** | | **25** | **20** | **5** | |

Cómo verificar:

```bash
pytest backend/tests/test_p1_country_system_f1.py -k "scs_ or country_for_form_data_es_el_unico" -v
```

## Parqueado para Fase 2 (no son gaps de T8 — decisiones explícitas con costo conocido)

| Item | Estado | Costo si no se toca |
|---|---|---|
| Nuance del retry-gate (`review_plan_node`) | **PARKED**, ruling del controller en T4 fix-round 1 (item 4a). Soft solo degrada a advisory en el intento FINAL — en attempts 1..N-1 CUALQUIER issue (hard o soft) fuerza retry igual. Para la regla del arroz, beta puede pagar **MÁS** retries que DO (el autofix ya no pre-arregla en silencio) | Retries extra en beta hasta que una task propia con batería completa toque el gate compartido |
| Interacción con `PROMPT_TRIM_FORM_DATA` | **CERRADO** en T1 (commit `2fc5c7f`) — la exclusión de `'country'` en `_sanitize_form_data_for_prompt` se movió FUERA de la rama de trim, incondicional en ambas ramas (kill-switch y passthrough). Ya no es un ítem parked | — |
| §9 taxonomía Mangú (enum de alimentos) | Deferred desde T2 — clase C, enum de ~40 ficheros. Territorio de Fase 2 (catálogo) | Ninguno hoy — T2 ya sustituye el §15 completo por variante beta genérica en el prompt |
| `get_avg_meal_hour` +8h | Bug PREEXISTENTE (doble `AT TIME ZONE` sobre timestamptz), forense contra Neon vivo (T5). Preservado A PROPÓSITO por el contrato de byte-identidad de T5 — cambiarlo mid-fase habría ensuciado la verificación | Ninguna regresión — la hora media sigue corrida 8h como HOY. P-fix propio pendiente, con medición |
| `routers/plans.py` — par MUTATOR-PURITY | Disclosed en T4 fix-round 1, formalizado con marker EXENTO en T8: `_swap_mutator` (`api_swap_meal_persist`) llama `slot_coherence_backstop_for_meal` Y `finalize_single_meal_recipe_coherence` país-blind (mismo motivo: corre dentro del `FOR UPDATE`); `api_recalculate_shopping_list` llama `finalize_single_meal_recipe_coherence` país-blind también. Fix real: pre-fetch país ANTES del lock + threading por el closure (patrón `_micro_form` ya usado ahí mismo) | Un swap persistido vía este endpoint específico no ablanda el backstop de horario para país beta — dos superficies puntuales, no las de mayor tráfico (esas SÍ están wired: `agent.py::swap_meal`, `tools.py::execute_modify_single_meal`). **[FINAL-FIX F2 · 2026-08-16]** Hasta esta fila el wiring estaba mecánicamente presente pero VACÍO en runtime: nada poblaba `country` en el `form_data`/`data` que esas dos funciones reciben, así que el argumento SIEMPRE llegaba `'DO'`. Ahora es comportamentalmente vivo: `agent.py::swap_meal` lo recibe porque `_enrich_clinical_from_profile` (`routers/plans.py:6096`, F2a) hidrata `data['country']` desde `health_profile` y el `meal_form` de `/regenerate-day` lo propaga (F2b); `tools.py::execute_modify_single_meal` lo recibe porque `services.merge_form_data_with_profile` ya mergea el `health_profile` completo al `form_data` del chat (F2c, sin fix de código — el mecanismo preexistente ya cubría este surface) |
| `slot_coherence_backstop_for_meal`'s pase ingredient-level | Disclosed en T4 fix-round 1 (item no cerrado, scope explícito de esa review) | Un swap/chat-modify con arroz oculto en ingredients (nombre inocuo) de un país beta puede seguir forzando retry vía este backstop específico, aunque `_detect_slot_appropriateness` (el productor S1) ya lo trata como soft/telemetría |

## Runbook del flip

- **Frontend**: `VITE_COUNTRY_SYSTEM=1|true` habilita `COUNTRY_SYSTEM_UI` ([`frontend/src/config/countries.js:36`](../../frontend/src/config/countries.js#L36)) — exige **REBUILD** (env inline-ada a build-time por Vite, no runtime).
- **Backend**: `MEALFIT_COUNTRY_SYSTEM=true` habilita `country_for_form_data`/`COUNTRY_SYSTEM_ENABLED` ([`constants.py:3175`](../constants.py#L3175)) — el helper lee el knob POR LLAMADA (`_env_bool` inline, no cacheado a import-time), así que exige **RESTART** del proceso, no redeploy de código.
- El corrimiento de índices del wizard (paso `QCountry`, Fase 0) ocurre en ese mismo deploy.
- Verificación viva post-flip: marker `_LAST_KNOWN_PFIX` (`/health/version`) + byte-identidad conductual de un plan DO real (los tests `..._do_control_...`/`..._byte_identico_...` de `test_p1_country_system_f1.py` son la evidencia offline; la verificación en vivo la corre el controller).

## Tests

- [`backend/tests/test_p1_country_system_f0.py`](../tests/test_p1_country_system_f0.py) — Fase 0 (el dato, sin lectores).
- [`backend/tests/test_p1_country_system_f1.py`](../tests/test_p1_country_system_f1.py) — Fase 1 completa, T1-T8 (232 tests a Task 8). Sección T8 al final del archivo.
- [`backend/tests/test_p3_claudemd_cap.py`](../tests/test_p3_claudemd_cap.py) / [`test_p3_1_last_known_pfix_freshness.py`](../tests/test_p3_1_last_known_pfix_freshness.py) / [`test_p2_hist_audit_14_marker_test_link.py`](../tests/test_p2_hist_audit_14_marker_test_link.py) — marker + CLAUDE.md, contrato genérico.
