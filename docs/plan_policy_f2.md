# PlanPolicy — Fase 2 del roadmap 2.5 (capa V2.3), modo `shadow`

[P1-ARQ25-F2-PLANPOLICY · 2026-09-02] Motor SSOT: [`backend/plan_policy.py`](../plan_policy.py). Test ancla:
[`tests/test_p1_arq25_f2_planpolicy.py`](../tests/test_p1_arq25_f2_planpolicy.py).

## Qué entrega la fase

| Entregable (roadmap §Fase 2) | Dónde | Estado |
|---|---|---|
| Schema versionado de `PlanPolicy` (`schema_version=1`) | `policy_from_form` | ✅ |
| Compilador requested → effective con `relaxations[]` y reason codes (§6.2–6.4) | `compile_policy` | ✅ |
| Precedencia §6.3 como tabla ejecutable (rango 1..7 en cada relajación) | `compile_policy` (pasos 1–5; 6–7 sin reglas hasta Fase 3) | ✅ |
| Adapters desde el formulario V1 | `policy_from_form` (`stapleFoods`/`staple_foods` → anclas, `groceryDuration` → `main_cycle_days`, `dietType` → `canonicalize_diet_type`, país → `country_for_form_data`) | ✅ |
| Persistencia en `plan_generation_runs` (autenticados) | `create_or_replay_run(policy=…)` — mismo INSERT (`requested_policy`, `effective_policy`, `relaxations`, `policy_hash`, `policy_schema_version`, `engine_versions`) | ✅ |
| Persistencia para invitados | `plan_data["_plan_policy"]` en `_postprocess_pipeline_result` (todo plan: cola, SSE legado, invitados) | ✅ |
| `template_id` en las 6 bibliotecas | `attach_template_ids` al cargar (`dish_library.load_dish_templates`), alias en `TEMPLATE_ALIASES` | ✅ 338/338 únicos |
| `ingredient_id` → nombre canónico sin tocar los nombres del motor | `ingredient_id_for` / `canonical_name_for` | ✅ |
| Medición en `shadow` (distancia política ↔ plan V1) | `measure_plan_against_policy` → `plan_data["_plan_policy_shadow"]` + `pipeline_metrics` node `plan_policy_shadow` | ✅ |

## Knob

`MEALFIT_PLAN_POLICY_MODE` = `off` (default) · `shadow` (compila, persiste y mide; NO influye) ·
`enforce` (reservado a la Fase 3: hoy se comporta como `shadow`). Valor inválido ⇒ `off`.

## Decisión #4 del dueño (2026-09-02): presupuesto = límite duro

- Donde hay precios (`pricing_mode` nativo), `budget.mode = "hard"`. Si la cifra pedida queda por
  debajo del piso de las metas (`min_budget_for_goals`), el compilador **no la modifica**: emite
  `budget_below_floor` con `action = waiting_user` y la evidencia `{floor_dop, amount_dop}`, para
  que el usuario suba el presupuesto o ajuste las metas ANTES de gastar el crédito.
- En países sin precios (`beta_no_prices`) el presupuesto es orientativo (`budget_advisory_no_prices`),
  nunca fingido.
- Los tiers (`low/medium/high`) llevan la referencia `reference_dop` (piso × banda), como hoy.

## Precedencia (§6.3) y reason codes

| Rango | Regla | Reason code |
|---|---|---|
| 1 | Alergias declaradas vencen a las anclas | `anchor_conflicts_allergy` |
| 2 | Dieta canónica (vegan/vegetarian/pescatarian) vence a las anclas | `anchor_conflicts_diet` |
| 3 | Disponibilidad en el mercado (solo si el caller pasa `known_ingredients`; si no, `notes: market_check_skipped`) | `anchor_not_in_market` |
| 4 | Duras: presupuesto (decisión #4), ciclo sin congelador ni reposición | `budget_below_floor` (`waiting_user`), `budget_advisory_no_prices`, `cycle_shortened_no_freezer_no_topup` |
| 5 | Anclas y recurrencia: rango 0–7, tope de 8 anclas, slots canónicos | `recurrence_clamped`, `anchors_capped` |
| 6–7 | Preferencias suaves / optimización interna | (Fase 3) |

`explain_relaxations` da la frase en español de cada relajación (Fase 4 la pinta).

## Hash

`policy_hash` = sha256 del JSON canónico de la política efectiva: claves ordenadas, listas
ordenadas (el orden de anclas/alergias no es semántica), sin campos volátiles
(`policy_hash`, `compiled_at`, `notes`, `source`). Gate: misma entrada ⇒ mismo hash.

## Shadow: qué se mide

Por plan entregado: cobertura de anclas por 7 días (`min ≤ per_7d ≤ max`), violaciones de
exclusiones/alergias en los ingredientes, coincidencia del ciclo (`main_cycle_days` vs
`total_days_requested`), presupuesto (`budget_reconciliation`). `distance` = 1 − media de los
componentes disponibles (None cuando no hay nada que medir, nunca 0 falso).

## Gate de la fase

- misma entrada ⇒ mismo `policy_hash` (test);
- 100 % de las plantillas con `template_id` estable (test golden + alias);
- en `shadow` ≥ 20 planes sin excepción en el compilado (`pipeline_metrics` node `plan_policy_shadow`;
  las excepciones caen a fail-open y se ven como `error` en `_plan_policy`).

## Fuera de alcance (Fase 3/4)

El allocator del horizonte, que las superficies obedezcan la política (`enforce`), el formulario
progresivo (`mealOrganization`, `freezerMode`, `freshTopup`, `batchCooking` ya tienen adapter con
defaults) y la pantalla «solicitaste / aplicamos / por qué».
