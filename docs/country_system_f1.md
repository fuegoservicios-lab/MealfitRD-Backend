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
| Nuance del retry-gate (`review_plan_node`) | **CERRADO** en Fase 2 Task 9 (item f, "la más riesgosa"). Helper puro `_slot_appropriateness_advisory_decision` (extraído para ser unit-testable sin montar `review_plan_node` completo): país beta con issues de slot-appropriateness TODOS soft ⇒ advisory desde el intento 1 (antes: solo en el intento FINAL; 1..N-1 forzaba retry igual, hard o soft). DO/knob-off byte-idéntico (`beta_soft_only` colapsa a False, `advisory` = exactamente `is_final`, anclado con golden tests contra la fórmula legacy). Batería review/retry completa (60 archivos, 759 tests) verde. Ver `backend/tests/test_p1_country_system_f2.py` sección "f" | — |
| Interacción con `PROMPT_TRIM_FORM_DATA` | **CERRADO** en T1 (commit `2fc5c7f`) — la exclusión de `'country'` en `_sanitize_form_data_for_prompt` se movió FUERA de la rama de trim, incondicional en ambas ramas (kill-switch y passthrough). Ya no es un ítem parked | — |
| §9 taxonomía Mangú (enum de alimentos) | Deferred desde T2 — clase C, enum de ~40 ficheros. Territorio de Fase 2 (catálogo) | Ninguno hoy — T2 ya sustituye el §15 completo por variante beta genérica en el prompt |
| `get_avg_meal_hour` +8h | Bug PREEXISTENTE (doble `AT TIME ZONE` sobre timestamptz), forense contra Neon vivo (T5). Preservado A PROPÓSITO por el contrato de byte-identidad de T5 — cambiarlo mid-fase habría ensuciado la verificación | Ninguna regresión — la hora media sigue corrida 8h como HOY. P-fix propio pendiente, con medición |
| `routers/plans.py` — par MUTATOR-PURITY | **CERRADO** en Fase 2 Task 9 (item g) — dos superficies puntuales, no las de mayor tráfico (esas SÍ están wired: `agent.py::swap_meal`, `tools.py::execute_modify_single_meal`). **[FINAL-FIX F2 · 2026-08-16]** (histórico) ya había hecho ese wiring comportamentalmente vivo: `agent.py::swap_meal` lo recibe porque `_enrich_clinical_from_profile` (`routers/plans.py:6096`, F2a) hidrata `data['country']` desde `health_profile` y el `meal_form` de `/regenerate-day` lo propaga (F2b); `tools.py::execute_modify_single_meal` lo recibe porque `services.merge_form_data_with_profile` ya mergea el `health_profile` completo al `form_data` del chat (F2c). **[Task 9 · item g · 2026-08-17]** cierra los 2 call sites restantes con el MISMO patrón (pre-fetch de país ANTES del lock, `_micro_form`): `_swap_mutator` (`api_swap_meal_persist`) resuelve `_swap_country` una vez y lo threadea a `finalize_single_meal_recipe_coherence` Y `slot_coherence_backstop_for_meal`; `api_recalculate_shopping_list` resuelve `_recalc_country` y lo threadea a `finalize_single_meal_recipe_coherence` | — |
| `slot_coherence_backstop_for_meal`'s pase ingredient-level | **CERRADO** en Fase 2 Task 9 (item g). Mismo filtro `v.get("hard") and _is_do` que el pase name-level (con el ajuste correcto: AND, no OR — `slot_ingredient_violations` devuelve `hard=True` incondicional, así que un OR habría sido un no-op, verificado en vivo durante el desarrollo). Beta ya no fuerza retry vía este pase específico; DO byte-idéntico | — |

## Fase 2 (2026-08-18) — catálogo completo + cierre del "100%"

[P1-COUNTRY-SYSTEM-F2 · 2026-08-18] 9 tasks SDD (T1-T9) + Task 10 de cierre. Fase 1 dejó el
MOTOR sin imposición criolla estructural; Fase 2 responde la otra mitad de la pregunta — "¿el
país beta tiene SU comida en la base de datos?" — y cierra las dos advertencias parqueadas de
Fase 1 (tabla arriba: retry-gate nuance, par MUTATOR-PURITY, ambas **CERRADO** en Task 9).
Plan: [`docs/superpowers/plans/2026-08-17-paises-fase-2.md`](../../docs/superpowers/plans/2026-08-17-paises-fase-2.md).
Addendum del dueño §1: "catálogo por completitud MEDIDA, no por cuota" — cada alta de Fase 2
está respaldada por el harness `country_catalog_gap.py` (T1), no por una cuota arbitraria.

### Catálogo: 6 países, 347 filas en `master_ingredients`

| Tramo | Filas nuevas | Acumulado | Harness (`--country`, 0 silenciosas/0 drops) | Task |
|---|---|---|---|---|
| ES | 32 | 206→238 | 80/80 | T5 |
| MX + CO | 46 (43 USDA + 3 manual: Achiote/Flor de Jamaica/Hoja santa) | 238→284 | MX 76/76, CO 74/74 | T6 |
| PR + US | 62 (54 USDA directo + 8 manual, incl. 6 blends ponderados) | 284→346 | PR 67/67, US 78/78 | T7 |
| RD (top-up) | 1 (Hummus, unpriced a propósito) | 346→347 | — (catálogo nativo, sin harness `--country`) | T8 |

347 filas verificado en vivo (`SELECT COUNT(*) FROM master_ingredients`) al cierre de Task 10;
141 sin precio RD a propósito (`is_country_catalog_unpriced_item`/`is_baking_pantry_staple`,
mismo mecanismo). Barrido final Task 10 (pool abierto, tier semántico Cohere activo): los 5
países beta en 0 silenciosas / 0 drops — sin cambios de contenido desde el harness de su task de
catálogo (T5-T7). `--rd-drops` (telemetría de producción, 30 días) repite los mismos 7 items del
top-up de T8 — esperado, esos fixes aún no están desplegados; evidencia completa en el reporte de
Task 10 del ledger SDD.

### QA final con LLM vivo (Task 10, item a — única excepción a la directiva de gasto)

1 generación real por país beta (ES/MX/CO/PR/US) + 1 gemelo DO por cada uno, LOCAL contra Neon
prod con `MEALFIT_COUNTRY_SYSTEM=true` **solo para el proceso** (nunca escrito a `.env`), vía
`graph_orchestrator.arun_plan_pipeline` directo (mismo patrón que `scripts/plan_gym.py`, sin
pasar por el router HTTP/SSE — evita a propósito encolar `plan_chunk_queue`: verificado 0 filas
tras cada corrida). Usuarios sintéticos `qa-f2-<cc>-<uuid8>@test.local` /
`qa-f2-do-<cc>-<uuid8>@test.local`, plan persistido de verdad (`meal_plans` real, vía
`services._save_plan_and_track_background`), teardown completo al cerrar (barrido
`_tablas_con_user_id` de `tests/conftest.py` — el mismo mecanismo que P1-TEARDOWN-SWEEP).

Por país, 5 verificaciones contra código de producción real (no simulado): (1) prompt beta sin
criollo forzado — render real de `build_day_generator_system_prompt` re-escaneado con el guard
`_DOMINICAN_TOKEN_RX` de `test_p1_country_system_f1.py`; (2) `plan_data['_pricing_mode'] ==
'beta_no_prices'` + cero `estimated_cost_rd` no-nulos en `aggregated_shopping_list`; (3)
presupuesto en moneda local — `min_budget_for_goals` + `budget_floor_in_currency` reales
devuelven la moneda del país (EUR/MXN/COP/USD), nunca DOP; (4) gates soft — el helper real
`_slot_appropriateness_advisory_decision` entrega `advisory=True` desde el intento 1/N para
issues todo-soft, mientras el gemelo DO en la MISMA situación sigue forzando retry
(`beta_soft_only=False`); (5) gemelo DO — prompt intacto, `_pricing_mode is None`, precios reales
presentes. Evidencia completa (ids, costos, veredictos) en el reporte de Task 10 del ledger SDD
(`.superpowers/sdd/2026-08-17-paises-fase-2/task-10-report.md`).

### Aclaración: la prohibición "ARROZ DE NOCHE" no es criolla — su ENFORCEMENT sí

[T9 · nota de claridad del reviewer] El §15d del prompt del day-generator lista `paella, risotto`
(junto a `chofán, congrí, mamposteao`) como EJEMPLOS de platos con base de arroz prohibidos en la
cena — texto **byte-idéntico en TODOS los países** (fila 1 de la tabla "Surfaces wired", T2), a
propósito: es una regla nutricional universal (nada de arroz en la cena), no una imposición
dominicana, y paella/risotto son justamente los ejemplos NO-dominicanos que la ilustran para un
lector español o italiano. Lo que SÍ es DO-scoped es el **ENFORCEMENT determinista** — fila 8 de
esa misma tabla: `_night_rice_autofix`/`_breakfast_rice_autofix` son no-op completo para país
beta (el autofix reescribe recetas hacia platos criollos por diseño). Un país beta cuyo LLM
ignore la instrucción del prompt no tiene autofix que lo corrija — la detección degrada a
soft/telemetría (fila 5-6), consistente con el resto del gate cultural de Fase 1.

## Runbook del flip

El flip enciende DOS banderas independientes (backend + frontend) que deben moverse JUNTAS —
una sin la otra dispara el bug de UI-sin-motor o motor-sin-UI documentado en la spec §Fase 1.

> ⚠️ **El flip YA SE EJECUTÓ: el 2026-08-18.** `MEALFIT_COUNTRY_SYSTEM=true` vive en el `.env`
> del VPS y `VITE_COUNTRY_SYSTEM=true` en el build servido; el selector de 6 países está visible
> para cualquiera y hay planes beta reales persistidos. **Todo gap del sistema de países es de
> producción viva, no de código en oscuro.** Este runbook se conserva porque documenta el
> procedimiento exacto —y porque §5 lo necesita para el camino de vuelta—, pero léelo como
> historia de lo hecho, no como plan de lo pendiente. Los dos incidentes de §6 son POSTERIORES a
> esta fecha.
>
> *[P3-COUNTRY-DOC-TRUTH · 2026-08-22]* Hasta hoy esta cabecera declaraba el flip como pendiente
> de encender, ciento diez líneas por encima de una sección titulada «Incidente del día del flip».
> Una contradicción interna en el primer documento que lee quien va a operar el sistema.
> *La frase original no se reproduce aquí a propósito*: el guard la busca como literal, y citarla
> en la nota que la corrige pondría en rojo al propio arreglo — comentario-vence-guard, que este
> repo ya ha pagado ocho veces.

### 1. Backend — `MEALFIT_COUNTRY_SYSTEM=true`

- Habilita `country_for_form_data`/`COUNTRY_SYSTEM_ENABLED`
  ([`constants.py:3327`](../constants.py#L3327), función en
  [`constants.py:3361`](../constants.py#L3361)) — el helper lee el knob **POR LLAMADA**
  (`_env_bool` inline, no cacheado a import-time), así que basta **RESTART** del proceso, no
  redeploy de código.
- SSH al VPS Oracle (`ssh -i C:\Users\angel\.ssh\mealfit-vps.key ubuntu@132.145.160.173`), editar
  `/opt/mealfit/backend/.env` → añadir/actualizar `MEALFIT_COUNTRY_SYSTEM=true`.
- `sudo systemctl restart mealfit-backend`.
- Verificar: `curl https://mealfitrd.com/ready` → `{status:ready,plan_graph:compiled,db:true}`;
  `curl https://mealfitrd.com/health/version` → `last_known_pfix` coincide con HEAD, `drift:false`.

### 2. Frontend — `VITE_COUNTRY_SYSTEM=true`

- Habilita `COUNTRY_SYSTEM_UI`
  ([`frontend/src/config/countries.js:35`](../../frontend/src/config/countries.js#L35)) — env
  inline-ada a **build-time** por Vite (no runtime): exige **REBUILD**, no basta un restart.
- Añadir `VITE_COUNTRY_SYSTEM=true` a `frontend/.env.production` (repo local).
- Desde la raíz del workspace, con **pwsh 7** (no `powershell` 5.1 — el .ps1 es UTF-8 sin BOM y
  5.1 rompe el parser en los em-dash): `& .\deploy-mealfit.ps1 frontend` — `npm install` +
  `npm run build` + sube `dist/` al VPS vía nginx.
- El corrimiento de índices del wizard (paso `QCountry`, Fase 0) vive en el MISMO bundle — no
  requiere un segundo deploy.

### 3. Post-flip — smoke QA en vivo (checklist, ~10 min)

Repetir en miniatura el QA offline de Task 10 pero contra `mealfitrd.com` real:

1. Cuenta de prueba, país España (o cualquier beta) en el wizard → generar plan.
2. Wizard: aparece el paso de país (`QCountry`) y la lista incluye los 5 beta + DO.
3. Plan entregado: SIN "RD$" en la lista de compras, badge/aviso de "modo beta sin precios
   nativos" visible, presupuesto mostrado en la moneda del país elegido.
4. Cuenta de prueba DO (o sin país): plan CON "RD$", sin aviso beta — comportamiento idéntico a
   ayer.
5. `/health/version` sigue con `drift:false` 10-15 min después (el cron de deploy-lag no debe
   disparar `deploy_lag_drift_vs_expected`).

### 4. Lo que el flip NO enciende

- `MEALFIT_COUNTRY_COLDSTART_SEGMENT` **se queda OFF** — knob independiente (Fase 0) que
  segmenta el cold-start de `get_similar_user_patterns` por país; el flip de Fase 2 es sobre el
  MOTOR de generación, no sobre esa segmentación. Encenderlo es una decisión aparte, con su
  propia medición.
- `MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP` **se queda en su default `true`** — knob independiente
  (T5, `shopping_calculator._country_catalog_unpriced_keep_enabled`) que decide si un ingrediente
  de catálogo-país SIN precio RD (las 140 altas T5-T8: Jamón serrano, Chicharrón, Bacalaítos...)
  se CONSERVA en la lista de compras marcado "sin precio" o se DROPEA (comportamiento pre-T5). El
  flip de Fase 2 no lo toca; es un lever de rollback PARCIAL propio — si el "sin precio" en la
  lista confunde en producción sin querer apagar el motor entero, `=false` revierte SOLO ese
  comportamiento (drop + WARNING, ver `test_jamon_serrano_se_dropea_si_el_knob_de_keep_esta_apagado`
  en `test_p1_country_system_f2.py`), sin necesidad de tocar `MEALFIT_COUNTRY_SYSTEM`.

### 5. Rollback

> **⚠️ PASO PREVIO OBLIGATORIO — la cola de chunks NO vuelve sola.**
> [P2-ROLLBACK-RUNBOOK · 2026-08-21]
>
> La promesa de abajo —«byte-identidad DO en segundos»— es cierta para los planes **nuevos** y
> **falsa para los que ya existen**. Medido en la cola viva el 2026-08-21: **7 chunks
> `pending`/`pending_user_action` llevan `country='US'` en su `pipeline_snapshot`** y despiertan
> por su `execute_after`.
>
> Con el knob apagado el worker **sigue leyendo ese snapshot**, pero `country_for_form_data`
> devuelve `'DO'` incondicional. Resultado: las semanas 2-8 se generarían **criollas dentro de un
> plan estadounidense** que además sigue marcado `beta_no_prices` — un híbrido que ningún usuario
> pidió y que no es ni el estado nuevo ni el viejo.
>
> Antes de apagar el knob, decide qué hacer con esos chunks y hazlo:
>
> ```sql
> -- 1. ¿Cuántos hay? (si sale 0, el rollback es limpio y puedes seguir)
> SELECT status, pipeline_snapshot->'form_data'->>'country' AS pais, count(*)
> FROM plan_chunk_queue
> WHERE status NOT IN ('completed','cancelled','failed')
>   AND pipeline_snapshot->'form_data'->>'country' NOT IN ('DO')
> GROUP BY 1,2;
>
> -- 2a. OPCIÓN A — congelar: el plan se queda como está, sin días nuevos híbridos.
> --     Es la conservadora: el usuario conserva lo generado y no recibe nada incoherente.
> UPDATE plan_chunk_queue SET status = 'cancelled'
> WHERE status NOT IN ('completed','cancelled','failed')
>   AND pipeline_snapshot->'form_data'->>'country' NOT IN ('DO');
>
> -- 2b. OPCIÓN B — convertir a dominicano: el plan sigue rellenándose, en criollo y coherente
> --     con el motor apagado. Elige ésta sólo si el usuario ACEPTA que su plan pase a ser DO.
> UPDATE plan_chunk_queue
> SET pipeline_snapshot = jsonb_set(pipeline_snapshot, '{form_data,country}', '"DO"')
> WHERE status NOT IN ('completed','cancelled','failed')
>   AND pipeline_snapshot->'form_data'->>'country' NOT IN ('DO');
> ```
>
> Ninguna de las dos es obviamente correcta —una deja el plan incompleto, la otra le cambia la
> cocina a alguien que eligió otra— y por eso el paso es *decidir*, no *ejecutar un comando*. Lo
> que no es defendible es apagar el knob y dejar que el worker resuelva la ambigüedad solo.

- **Emergencia (motor)**: unset `MEALFIT_COUNTRY_SYSTEM` (o `=false`) en el `.env` del VPS +
  `systemctl restart mealfit-backend` — vuelve el motor a byte-identidad DO en segundos, AUNQUE
  el frontend siga mostrando el selector de país (UX degradada pero segura: cualquier país
  declarado se ignora, `country_for_form_data` cae a `'DO'` incondicional).
- **Completo**: además, revertir `VITE_COUNTRY_SYSTEM` + `& .\deploy-mealfit.ps1 frontend` para
  ocultar el selector.
- Verificación viva post-flip u post-rollback: marker `_LAST_KNOWN_PFIX` (`/health/version`) +
  byte-identidad conductual de un plan DO real (los tests `..._do_control_...`/
  `..._byte_identico_...` de `test_p1_country_system_f1.py`/`test_p1_country_system_f2.py` son la
  evidencia offline; la verificación en vivo la corre el controller).

### 6. Incidente del día del flip: la renovación pisaba el país de Configuración

[P1-COUNTRY-RENEWAL-PROFILE-WINS · 2026-08-18] Primer uso real post-flip: el dueño eligió
**España** en Configuración (PATCH → `health_profile.country='ES'`), pulsó «Renovar», y el plan
salió **dominicano** (RD$, crítica criolla) — y Configuración volvió sola a DO. Cadena: la
renovación reenvía el `formData` del dispositivo, cuyo `country` es el `'DO'` **sembrado por
`initialFormData`** (jamás elegido); la generación leyó ese valor stale, y el merge post-pipeline
(`hp_data.update(data)` en el persist compartido de /analyze y /analyze/stream) lo escribió de
vuelta al perfil, matando el 'ES'. **Dos setters del mismo dato sin jerarquía = last-writer-wins
silencioso.** La hidratación F2a no aplicaba: vive en las superficies de UPDATE
(`_enrich_clinical_from_profile`) y su `if not data.get("country")` asume que payload-con-país =
elección — **el default sembrado es indistinguible de una elección**, salvo por `update_reason`,
que solo las regens explícitas mandan.

Fix: `_hydrate_country_from_profile_for_submit` (routers/plans.py, llamado en LOS DOS entry
points tras `_close_medical_freetext_scope`, ANTES del pipeline y del merge): con
`update_reason` presente el **perfil GANA** (pisa la copia stale del payload); sin él (wizard
completo con QCountry recién elegido) el payload gana como siempre; fill-si-falta en ambos. Al
mutar `data` antes del persist compartido, `hp_data` re-escribe el valor correcto y el clobber
muere en la misma jugada. Complemento frontend: `Settings.handleSelectCountry` sincroniza también
`formData.country` (updateData) — mantiene coherente el dispositivo y la preselección de QCountry.
Test: [`test_p1_country_renewal_profile_wins.py`](../tests/test_p1_country_renewal_profile_wins.py).

### 7. Segunda renovación ES real: el sustituidor económico corría país-ciego

[P1-BUDGET-CHEAPEN-COUNTRY-GATE · 2026-08-18] Con el fix §6 ya vivo, la primera renovación
genuinamente ESPAÑOLA (plan 6a4321f5: pools Boquerones/Almejas/Garbanzos/Membrillo, review
aprobado, `_pricing_mode=beta_no_prices`) destapó una **3ª superficie de mutación país-ciega**
de la clase MUTATOR-PURITY que F2 cerró para swap/recalc: `_apply_budget_cheapen_pass`
(P1-BUDGET-TIER-LEVERS) sustituyó `habas → Habichuelas rojas` y `almendras → Maní` comparando
**RD$/lb** — y reescribió hasta el nombre del plato («Bowl Fresco de Habichuelas rojas»). Sus
dos piernas son DO-céntricas (price map solo-RD + `_BUDGET_CHEAP_EQUIVALENTS` apunta a filas
criollas). Gate en la CABECERA (cubre `force=True` de la convergencia T2 y los 3 call sites)
vía el literal SSOT `pricing_mode_for_country` — con el knob maestro apagado el gate es
inerte (byte-identidad DO). Test:
[`test_p1_budget_cheapen_country_gate.py`](../tests/test_p1_budget_cheapen_country_gate.py).

**Fast-follows con evidencia del mismo plan 6a4321f5** (warn-only, el plan se entrega):

- **Léxico del coherence guard ciego a filas beta (SISTÉMICO — confirmado en ES y US)**: la
  lista trae «Acelgas/Almejas/Judías pintas/Membrillo» (ES, plan 6a4321f5) o «Aderezo ranch/
  Provolone/Salsa inglesa/Tocineta» (US, plan 2245eb45) y las recetas también los traen, pero
  `expected_sum_from_recipes` no los matchea → 4 fantasmas `presence/aggregated_only` por plan
  y coherencia 35/100 (ES) y 25/100 (US) en el quality index — la nota castiga filas nuevas,
  no incoherencia real. El vocabulario del lado esperado del guard es DO-tuned; las filas beta
  de F2 no entraron. (Es el «espejo del coherence-guard» ya anotado como costura en la memoria
  de F2, ahora con costo medido en DOS países.) Mientras no se cierre, leer las notas de
  coherencia de planes beta con esta corrección mental.
- **Un mutator inserta ingredientes por su nombre de catálogo DO**: receta «Tortilla
  Mediterránea» (ES) con la línea literal «Orégano dominicano» mientras la lista dice
  «Orégano» — sospechoso primario: micro-closer/fat-swap añadiendo el carrier por nombre de
  fila DO. Misma clase MUTATOR-PURITY; auditar los closers que INSERTAN (no solo los que
  sustituyen).
- Menor: el boost del seeder por tercio-más-barato loguea en beta (precios NULL) — cosmético,
  el pool salió correcto.

## Tests

- [`backend/tests/test_p1_country_system_f0.py`](../tests/test_p1_country_system_f0.py) — Fase 0 (el dato, sin lectores).
- [`backend/tests/test_p1_country_system_f1.py`](../tests/test_p1_country_system_f1.py) — Fase 1 completa, T1-T8 (294 tests al cierre de Fase 2).
- [`backend/tests/test_p1_country_system_f2.py`](../tests/test_p1_country_system_f2.py) — Fase 2 completa, T1-T9 (504 tests `-m "not e2e"` + 61 e2e).
- [`backend/tests/test_p3_claudemd_cap.py`](../tests/test_p3_claudemd_cap.py) / [`test_p3_1_last_known_pfix_freshness.py`](../tests/test_p3_1_last_known_pfix_freshness.py) / [`test_p2_hist_audit_14_marker_test_link.py`](../tests/test_p2_hist_audit_14_marker_test_link.py) — marker + CLAUDE.md, contrato genérico.
