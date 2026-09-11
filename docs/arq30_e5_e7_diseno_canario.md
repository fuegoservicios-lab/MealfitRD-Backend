# ARQ30 · E5–E7 — diseño y plan de canario (sin implementar a ciegas)

`[P1-PLAN-LOTE-10 · 2026-09-11]` · ítems E5, E6, E7 del plan de pendientes (`plan_pendientes_2026_09_11.md`)
· gaps `ARQ30-P1-01/03/04/05` y `ARQ30-P2-01…05` de `docs/audits/2026-09-06-generacion/GAPS.md`.

Los tres ítems mueven **autoridad** (quién decide la lista de compras, la asignación del horizonte, las porciones y
qué revisión del plan se considera válida). La regla que ya fijó el encargo y que este doc respeta: **`expand →
sombra → canario por cohorte → promoción → retirada`**, y *nunca se borra la vía anterior en el mismo paso que
transfiere su autoridad*. Lo que sigue es lo que haría, en qué orden, con qué medición decide cada paso y qué
parte es decisión del dueño. Nada de esto cambia comportamiento hasta que el dueño elige la cohorte.

---

## E5 · ARQ30-P1-01 — promover `IngredientLine` a compras reales

**Lo que hay (verificado):** [`canonical_recipe.py`](../canonical_recipe.py) — `IngredientLine`, `parse_line`,
`render_line`, `parse_meal`, `shopping_view` — es una representación de **solo lectura**; el test
`test_nadie_en_produccion_escribe_con_esta_representacion` (`test_p1_arq30_f4_canonical.py`) falla si un módulo de
producción la importa. Medido el 09-06 sobre 11.073 líneas de 96 planes: roundtrip exacto 0,5 %, equivalente 17,8 %
— por eso el dueño decidió que **sirve para COMPRAS y MACROS, no para el texto** (`arq30_f4_canonical.md`).

**Diseño en cuatro fases**

| Fase | Qué cambia para el usuario | Qué se mide | Sale a la siguiente si… |
|---|---|---|---|
| A · sombra | Nada. En `assemble_plan_node` y `_recompute_aggregates_after_swap`, además de la lista de hoy se calcula `shopping_view` por línea y se persiste SOLO la divergencia en `pipeline_metrics` (node `canonical_shopping_shadow`: líneas, `parse_fail`, alimentos con |Δ g| > 10 % frente a `aggregated_shopping_list`). | `parse_fail` / línea; fracción de alimentos divergentes; los 4 modos de fallo del guard (`cap_swallowed_modifier`, fantasmas, magnitud, bloque no consumido) | ≥ 30 planes con `parse_fail` < 1 % y divergencia > 10 % en < 5 % de los alimentos |
| B · canario | Para los uuids de `MEALFIT_CANONICAL_SHOPPING_USERS` (mismo patrón que `MEALFIT_DETERMINISTIC_DAY_USERS`: lista vacía CIERRA), la lista servida es la canónica; la anterior se guarda como `_shopping_legacy_shadow`. | Lo mismo que A, más el guard de coherencia sobre la lista servida | 2 semanas sin alerta `plan_quality_degraded` atribuible ni divergencia severa |
| C · promoción | `MEALFIT_CANONICAL_SHOPPING=1` global; la legacy sigue en sombra 2 semanas | Ídem | 2 semanas limpias |
| D · retirada | Se borra la agregación legacy; el test «nadie escribe» se invierte («nadie lee la legacy») | — | — |

**Tres trampas ya pagadas que el diseño evita.** (1) `pantry_names_match`, el guard de coherencia y el backstop de
alergias resuelven por el **nombre exacto del catálogo**: la canónica debe conservar ese nombre, nunca el texto
normalizado. (2) La lista y la referencia del guard no pueden cambiar de fuente **a la vez**: `P0-SHOPPING-CYCLE-DAYS`
enseñó que si los dos lados leen la misma fuente encogida, mutilar la lista MEJORA la métrica. La canónica consume
`shopping_source_days` (SSOT) y el guard sigue con `expected_sum_from_recipes` hasta la fase C. (3) El shift no
reconstruye la lista (prerrequisito 1 de `MEALFIT_TRIP_WINDOWED_PERISHABLES`): la sombra debe medirse también tras
un shift, no sólo al nacer el plan.

**Tamaño y decisión.** A ≈ 1 lote (sombra + métrica + test); B ≈ 1–2 lotes. **El dueño decide la cohorte** (él
primero) y el momento de C.

---

## E6 · ARQ30-P1-03 / P1-04 / P1-05

### P1-03 — asignación del horizonte por comidas viables

**Hoy:** `horizon.build_blueprint` asigna la familia de proteína por día con round-robin (`pool[d % len(pool)]`,
`horizon.py:759`), `repetition_limits_for` fija los topes y el día determinista añadió memoria entre días (B6) y
candidatos por franja. No resuelve el acoplamiento global (una franja sin candidato viable el día 23 se descubre el
día 23).

**Diseño:** un `allocator.py` que recibe el CandidateSet **fijado al run** (ARQ27-F3) y produce la asignación
día × franja como un CSP pequeño (30 × 4): variables = plantilla por franja; restricciones duras = sin alérgeno,
dieta, mercado, durabilidad bajo compra única, comidas **fijadas** (consumidas/editadas/swap); blandas = cuota de
repetición 7 d, familia de proteína del día, cocina del día, precio. Greedy + reparación local basta a ese tamaño; se
versiona (`allocator_version` entra en `blueprint_hash`, así la rebanada de un chunk viejo no se mezcla con la nueva).

**Canario:** primero **sombra sin tocar nada**: medir hoy cuántas franjas del horizonte se quedan SIN candidato viable
con el round-robin (el registry y el catálogo ya permiten calcularlo offline, como hizo
`scripts/measure_deterministic_day_macros.py`); si es 0, el allocator no urge y lo dice el número. Después, cohorte
por uuid, y sólo para el camino determinista (el LLM sigue recibiendo sus 2 candidatos por franja como hoy).

### P1-04 — una autoridad para porciones y reparaciones

**Hoy:** `portion_solver.solve_meal_macros` / `solve_portion_macros` / `refine_day_portions_integer` y, en el god
file, `_apply_macro_engine` (extraído como unidad llamable) con closers, caps y refinador que **vuelven sobre las
mismas cantidades**; el fix del mínimo cocinable (40 g) está preservado por fixture.

**Diseño:** `portion_authority.py` como controlador con (a) **presupuesto de reparaciones explícito** (knob, y al
agotarse se declara la causa de no convergencia en el informe, no se sigue cerrando), (b) **huella de estado** =
hash de la composición tras cada paso, para detectar ciclos (A→B→A) y garantizar idempotencia (correr dos veces =
misma salida), (c) `plan_data._portion_repairs[]` con causa, antes/después y hash — el rastro que hoy no existe.
Reutiliza el solver actual como candidato rápido; cada reparación local re-valida las restricciones previas. Fixtures
golden: mínimo 40 g, seco/cocido, sodio del día, volumen.

**Canario:** modo `shadow` (calcula y compara con lo que el motor actual entregó; mide divergencia y reparaciones
evitadas) → `enforce` por cohorte → promoción. Rollback = knob de modo.

### P1-05 — commit de la revisión validada en todas las superficies

**Hoy:** `generation_lifecycle` (fingerprint del run, fencing por `attempts`, revisión por trigger — F1),
`update_plan_data_atomic` (FOR UPDATE + callback, I7), y el outbox `plan_jobs` (F5) cuya `shopping_projection` ya se
liga a la revisión. Las guardas existen; lo que falta es **unirlas al hash exacto de composición y demanda**.

**Diseño:** `ValidationReport{plan_revision, ingredient_hash, demand_hash, versions{registry, allocator, solver}}`
persistido junto al plan; el commit compara `revision + content_hash + fencing` **dentro de la transacción y sin
red/LLM**; toda mutación nutricional (swap, regen, tool del chat, restock, fallback) invalida el reporte y encola
`revalidate`; las proyecciones consumen sólo revisiones con reporte válido. Una **suite única de invariantes** para
las 8 superficies (initial/chunk/renew/swap/regen/chat/fallback/compras) sustituye a los tests por superficie que hoy
las cubren por separado. Migración expand/contract: columna nullable + lector tolerante; los estados legacy no se
borran hasta 0 tráfico.

**Orden:** P1-04 antes que P1-05 (el reporte necesita el hash que la autoridad de porciones produce); P1-03 es
independiente y el más barato de medir en sombra.

---

## E7 · ARQ30-P2-01…05 — qué existe y cuál es el primer paso barato

| Gap | Lo que ya hay | Primer paso (medición, sin cambiar conducta) |
|---|---|---|
| P2-01 god files | Cap duro 53.100 líneas en `graph_orchestrator.py` (53.080 hoy), extracciones ya hechas (`_apply_macro_engine`, `deterministic_day`, `horizon`, `canonical_recipe`) | Mapa de dominios (elegibilidad, receta, porciones, validación, presentación) con conteo de líneas y llamadores; migrar tests parser → comportamiento SOLO al tocar cada contrato |
| P2-02 preferencias con evidencia | `user_facts` + Dreaming; `plan_meal_deviation` («comí otra cosa / todavía no») ya distingue registrado de prescrito | Registrar por separado elegido / cocinado / consumido / descartado / motivo del swap; **experimento offline** (reordenar candidatos seguros y comparar contra lo que el usuario eligió) antes de exponer nada |
| P2-03 disponibilidad y coste por mercado | `pricing_mode_for_country`, catálogo de 347 filas, `MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP`; `known_ingredients` ya entra al compilador (ARQ27-P1-07) | Medir cuántas filas beta tienen precio y cuántas «0» son en realidad **ausente**; el modelo con confianza y fecha viene después del dato |
| P2-04 latencia y coste por resultado útil | `llm_usage_events`, `/generation-eta` (p50/p90 real), timeouts por nodo | Instrumentar preflight vs generación vs reparaciones vs validación en `pipeline_metrics`; presupuesto de tokens por run como knob; comparar LLM necesario tras preflight |
| P2-05 adecuación ≠ cobertura de datos | El informe ya dice «sin dato» para el sodio (A4) en vez de 0 | Separar en el informe `adecuación` (frente al target) de `cobertura` (nutrientes con dato / aplicables); la política vegetal la revisa un clínico — **dueño** |

---

## Qué decide el dueño

1. **Cohortes**: quién entra primero en cada canario (E5-B, P1-03, P1-04). Recomendación: él, como con el día
   determinista.
2. **Orden**: recomiendo E5-A (sombra, sin riesgo, ≈ 1 lote) → P1-03 sombra (un número que dice si urge) → P1-04 →
   P1-05. E7 son mediciones primero, todas baratas.
3. **Lo clínico** de P2-05 (política vegetal) y **lo comercial** de P2-03 (precios por mercado) no son técnicos.
