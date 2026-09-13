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

### Estado 2026-09-12 · fase A (sombra) implementada — `P1-PLAN-LOTE-19`

**Dónde corre.** No en dos sitios sino en UNO que cubre las 6 superficies: al final de
`shopping_calculator.run_shopping_coherence_guard`, después de que el guard emite su propia métrica y antes de que
devuelva sus divergencias (tooltip-anchor `P1-PLAN-LOTE-19-CANONICAL-SHADOW-HOOK`). Recibe el MISMO multiplicador
efectivo que el guard aplica al lado esperado (`mult × _basis_scale`: hogar × 7/días fuente cuando compara contra la
lista semanal), así que compara like con like. Cero líneas en el god file.

**Qué compara** ([`canonical_shopping_shadow.py`](../canonical_shopping_shadow.py)): la lista canónica —gramos de cada
`IngredientLine`, sumados por alimento sobre `shopping_source_days` (SSOT del ciclo), con la identidad que decide el
canonicalizador del guard (`_canonicalize_for_coherence`) y espejando la POLÍTICA que la lista ya lleva sellada
(rendimiento de legumbres, `protein_yield_applied`, `pantry_deduction_applied`; preguntada al parser legacy, no
reinventada)— frente a la lista entregada (`aggregated_shopping_list_weekly`, cuyos items llevan `base_qty`/`base_unit`
en gramos; 46 de 47 en el plan medido, el 47.º es «cartón (20 uds.)» y queda como *no comparable*). Persiste SÓLO la
comparación en `pipeline_metrics` (node `canonical_shopping_shadow`: líneas, `parse_fail`, sin gramos y por qué,
comparables, divergentes > 10 % con ejemplos, sólo-canónica, sólo-lista, sellos, multiplicador, días archivados,
huella de contenido del plan). **No toca `plan_result`**: `test_la_sombra_no_escribe_en_el_plan` y el guard del expand
(`test_nadie_en_produccion_escribe_con_esta_representacion`, que ahora admite exactamente este módulo).

**Knob** `MEALFIT_CANONICAL_SHOPPING_SHADOW` (default True; apagarlo quita la sombra sin redeploy). **Gate** de salida
a la fase B: ≥ 30 planes distintos con `parse_fail` < 1 % y divergencia en < 5 % de los comparables —
`scripts/measure_canonical_shadow.py` (lee la métrica; `--offline` la calcula ahora sobre los planes vivos sin
escribir). Tres salidas: NO CONCLUYENTE / PASA / NO PASA.

**Primera medición, `--offline`, 2026-09-12** (5 planes, la flota tras la purga; 1 de ellos ya desplazado):

| | |
|---|---|
| líneas · `parse_fail` | 731 · **0** (0,0 %) |
| sin cantidad («al gusto») · sin gramos | 87 · 20 (cilantro ×9, perejil, canela y orégano en polvo, agua: hierbas y especias sin densidad en el catálogo) |
| gramos por autoridad | `to_base_amount` 195 · `nutrition_db.to_grams` 429 |
| alimentos comparables · divergentes > 10 % | 214 · **24 (11,2 %)** |
| sólo canónica · sólo lista · no comparables | 0 · 14 · 11 |
| veredicto | NO CONCLUYENTE (5 < 30) |

**Lo que dicen los 24 divergentes**: no es el parser. Rábano 524 g → **50 g**, puerro 312 → **50**, limón 938 → **201**,
tomate 3.500 → **750** son los **topes realistas del agregador** (caps P6/`_REALISM`); chinola 140 → 642 y maní 35 → 82
son **densidades** (la lista convierte piezas con otra tabla que `nutrition_db.to_grams`). Es decir: la
representación aguanta (0 fallos de parseo, identidad simétrica, 0 «sólo canónica»), y lo que separa a las dos listas
es **política de compra**. Para la fase B, el agregador canónico tiene que llevar los topes y las densidades del
legacy —o el dueño decide moverlos—; si no, el canario compraría 3,5 kg de tomate. Con 5 planes la divergencia
(11,2 %) no pasaría el 5 %; el veredicto real llega cuando la sombra acumule ≥ 30 planes.

**Decide el dueño**: la cohorte de la fase B (recomendación: él, como con el día determinista) y si los topes P6 se
espejan en la canónica o se mueven a ella.

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


#### Estado 2026-09-12 · P1-03 medido en sombra y allocator MÍNIMO tras knob — `P1-PLAN-LOTE-20`

**La sombra, primero.** Una franja sin candidato no dejaba rastro: la clave no existía en `registry.candidates` y el
modelo improvisaba. Ahora el blueprint anota `registry.empty_slots` (día, franja, familia, cocina y
`rescuable_by_family`: ¿OTRA familia de proteína sí tendría plato aquí, o no hay plato en la biblioteca con esos
filtros?) y la rebanada lleva los de sus días — sólo cuando los hay, así un blueprint sin huecos no cambia de forma ni
de hash. `scripts/measure_horizon_slots.py` construye blueprints para una matriz reproducible (6 países de mercado × 25
perfiles clínicos del landing × 4 escenarios de compra: semanal, quincenal, mensual, mensual SIN congelador; horizonte
= ciclo; 4 comidas) y cuenta.

**Lo medido (round-robin de hoy, 600 blueprints, 49.200 franjas):**

| | franjas | vacías | otra familia sí | hueco de biblioteca |
|---|---|---|---|---|
| total | 49.200 | **4.311 (8,8 %)** | **4.211** | 100 |
| semanal | 4.200 | 28 | 28 | 0 |
| quincenal | 9.000 | 69 | 69 | 0 |
| mensual, congelador limitado | 18.000 | 138 | 138 | 0 |
| **mensual sin congelador** | 18.000 | **4.076** | 3.976 | 100 |

El acoplamiento que ARQ30-P1-03 describe, con cifra: en compra mensual sin congelador, del día 8-9 en adelante
Res/Cerdo/Pollo no tienen almuerzo ni cena que aguante hasta el día, y el round-robin se los asignaba igual — mientras
otra familia del pool (legumbre, huevo, conserva) sí tenía plato. Vacías por franja: almuerzo 2.161, cena 2.040,
desayuno 110. Los 100 huecos de biblioteca son todos el mismo: desayuno del perfil alérgico a lácteo/gluten/huevo, del
día 11 en adelante, en los cinco mercados beta (ningún desayuno sin lácteo, gluten ni huevo aguanta más de 10 días).
Limitación de la matriz: los perfiles del landing no eligen cocina, así que ES/US/MX/PR/CO miden la biblioteca por
defecto bajo su mercado (por eso salen idénticos entre sí). **Veredicto: el allocator urge.**

**El allocator mínimo, tras knob.** Como el número lo pedía, se construyó la versión más pequeña que cierra el
acoplamiento, determinista y en el sitio donde ya se fijan los candidatos (`horizon._registry_block_for_country`):
el round-robin propone la familia del día; si alguna franja del día no tiene plato con ella, se toma —en orden
rotado desde la propuesta— la familia del pool que cubre MÁS franjas del día (no «todas»: un desayuno sin plato en
ninguna familia no puede condenar al almuerzo y la cena; exigir «todas» dejaba 120 franjas rescatables sin rescatar),
y sólo si mejora estrictamente. Mueve `d["protein"]`, así que candidatos, prompt, sembrador del día determinista y
gate de fidelidad ven la MISMA familia; queda anotado en `registry.family_reassignments` (y en la rebanada).
**Knob `MEALFIT_HORIZON_VIABLE_FAMILY`: nació OFF; ON por defecto desde `P1-PLAN-LOTE-21` (2026-09-12, decisión del
dueño)**: apagado, el blueprint es byte-idéntico al anterior salvo el diagnóstico. `blueprint_hash`/`slice_hash` cambian sólo para runs con huecos o con el knob encendido.

**Con el knob encendido (misma matriz, `--viable`):** vacías **4.311 → 100 (0,2 %)**, las 100 son los huecos de
biblioteca del desayuno alérgico; 0 rescatables sin rescatar; **2.315 días reasignados** de ~12.300 (2.140 en
mensual sin congelador, 22 en semanal). Con 3 candidatos: 41.656 → 45.910 franjas.

**Lo que NO hace, a propósito**: no resuelve cuotas de repetición ni cultura ni precio como CSP global (eso es el
`allocator.py` del diseño, versionado, con comidas fijadas y ventanas deslizantes); no cambia la familia de días que
ya tienen plato en todas sus franjas; no toca runs en curso (la rebanada del chunk fija lo que ya se fijó).

**Decidido (2026-09-12, `P1-PLAN-LOTE-21`)**: el dueño delegó («enciende el knob por mí») y `MEALFIT_HORIZON_VIABLE_FAMILY`
pasa a ON por defecto en el código — no por `.env` del VPS ni por cohorte de usuarios: el knob no tiene lista de usuarios
(la de `MEALFIT_PLAN_POLICY_ENFORCE_USERS` es de otra decisión) y no la necesita, porque los runs en vuelo conservan su
blueprint (`_run_blueprint_for_plan`) y sólo los runs NUEVOS ven la reasignación. Vuelta atrás sin redeploy:
`MEALFIT_HORIZON_VIABLE_FAMILY=0`. `scripts/measure_horizon_slots.py` apaga el knob explícitamente cuando mide sin
`--viable`, para que la cifra del round-robin puro siga siendo reproducible. **Queda del dueño** el hueco de biblioteca
del desayuno sin lácteo/gluten/huevo de larga duración (trabajo de plantillas, como E9).

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
| P2-01 god files | Cap duro en `graph_orchestrator.py`: 53.100 hasta el 13-sep (53.099 ese día); 52.600 tras `P1-PLAN-LOTE-32` (51.956), extracciones ya hechas (`_apply_macro_engine`, `deterministic_day`, `horizon`, `canonical_recipe`) | Mapa de dominios (elegibilidad, receta, porciones, validación, presentación) con conteo de líneas y llamadores; migrar tests parser → comportamiento SOLO al tocar cada contrato |
| P2-02 preferencias con evidencia | `user_facts` + Dreaming; `plan_meal_deviation` («comí otra cosa / todavía no») ya distingue registrado de prescrito | Registrar por separado elegido / cocinado / consumido / descartado / motivo del swap; **experimento offline** (reordenar candidatos seguros y comparar contra lo que el usuario eligió) antes de exponer nada |
| P2-03 disponibilidad y coste por mercado | `pricing_mode_for_country`, catálogo de 347 filas, `MEALFIT_COUNTRY_CATALOG_UNPRICED_KEEP`; `known_ingredients` ya entra al compilador (ARQ27-P1-07) | Medir cuántas filas beta tienen precio y cuántas «0» son en realidad **ausente**; el modelo con confianza y fecha viene después del dato |
| P2-04 latencia y coste por resultado útil | `llm_usage_events`, `/generation-eta` (p50/p90 real), timeouts por nodo | Instrumentar preflight vs generación vs reparaciones vs validación en `pipeline_metrics`; presupuesto de tokens por run como knob; comparar LLM necesario tras preflight |
| P2-05 adecuación ≠ cobertura de datos | El informe ya dice «sin dato» para el sodio (A4) en vez de 0 | Separar en el informe `adecuación` (frente al target) de `cobertura` (nutrientes con dato / aplicables); la política vegetal la revisa un clínico — **dueño** |

### P2-01 · primera extracción medida (`P1-PLAN-LOTE-32` · 2026-09-13)

**Medido primero.** 53.099 líneas con el tope en 53.100 (13 tests lo fijan): el siguiente arreglo del grafo no cabía. El
módulo tiene 577 definiciones de nivel superior y 1.062 globales. Las funciones grandes (`assemble_plan_node` 2.319 líneas,
`review_plan_node` 1.828, `arun_plan_pipeline` 1.523) leen entre 50 y 80 símbolos del módulo cada una: moverlas no es
mover, es reescribir. En cambio el bloque de resiliencia LLM (líneas 921-2599) sólo leía del grafo sus propios knobs
(medido por AST): ningún test nombraba los semáforos, y cuatro leían el FUENTE del breaker en el grafo.

**Qué se movió, tal cual** (mismo texto; sólo cambian los imports; el grafo re-exporta cada nombre):

| Módulo | Qué | Líneas |
|---|---|---|
| `llm_concurrency.py` | `DistributedLLMSemaphore`, `DistributedPerUserSemaphore`, `_LLM_BUDGET_STATS` + `_inc_budget_stat` + `get_llm_budget_stats_snapshot`, y los knobs `MEALFIT_LLM_PER_USER_LOCAL_CACHE_MAX` / `MEALFIT_LLM_LOCAL_MAX_WAIT_S` (sólo los leen ellos) | 541 |
| `llm_circuit_breaker.py` | `LLMCircuitBreaker`, `_BestEffortDBCircuitBreaker` + `_get_be_db_cb` + `_is_pool_timeout_error` (con sus dos knobs de `os.environ`), `LLMCircuitOpenError` | 682 |

**Qué se quedó, a propósito:** las instancias (`LLM_SEMAPHORE`, `PER_USER_LLM_SEMAPHORE`, `_circuit_breaker`), el registro
per-modelo `_get_circuit_breaker`, `acquire_user_and_global` y `_record_cb_failure_unless_transient`. Son la POLÍTICA
—qué umbral, qué fallo cuenta— y leen los knobs `MEALFIT_LLM_*` / `MEALFIT_CB_*` que el grafo define; moverlos arrastraba
una docena de knobs con sus anclas de test. Mecanismo fuera, política dentro.

**Tres cosas que «mover y re-exportar» rompe en silencio, y cómo se evitaron:**

1. *El nombre del logger.* El formato de producción imprime `%(name)s` y los `caplog` de la suite filtran por
   `graph_orchestrator`. Los dos módulos usan `logging.getLogger("graph_orchestrator")`: mover el código no mueve sus logs.
2. *Los parches.* `patch("graph_orchestrator.redis_client")` cambia el nombre en el grafo, no en el módulo que lo lee. El
   test del fallback atómico del breaker parchea ahora `llm_circuit_breaker.*`. Dos de sus parches
   (`graph_orchestrator.redis_async_client`) **ya no alcanzaban nada antes de mover**: el breaker lee el cliente per-loop
   `get_redis_async()` desde P1-REDIS-ASYNC-PERLOOP-CB, y el test seguía verde porque en la suite no hay Redis.
3. *Las lecturas del fuente.* Cinco tests leían el texto del grafo buscando código que ahora vive fuera
   (`_deadline = time.monotonic() + LLM_LOCAL_MAX_WAIT_S`, `class LLMCircuitBreaker`, los callsites
   `_get_be_db_cb("llm_cb_*")`…). Se encontraron comparando cada literal, regex y conteo de la suite contra el grafo de
   antes y el de después — no por nombre de símbolo, que no ve un `count(...) >= 5` que cae por debajo de su mínimo. Se
   re-apuntaron conservando su intención: al módulo nuevo, o a la unión de los dos cuando el contrato los abarca (los cinco
   callsites best-effort son tres en el grafo y dos en el breaker).

**Después.** 51.956 líneas (−1.143). El tope SSOT (`test_p3_shopping_projection_pkg`) baja de 53.100 a 52.600: 644 líneas de
aire para arreglos, y quien quiera volver a 53.100 tendrá que extraer. Cero cambios de conducta: el grafo resuelve los mismos
objetos (`go.LLMCircuitBreaker is llm_circuit_breaker.LLMCircuitBreaker`) y ningún nombre que el grafo usa quedó sin
definir (comprobado con `symtable` antes y después). Test: `tests/test_p1_plan_lote_32.py`.

**Lo que NO se hizo.** Las funciones grandes siguen dentro. El siguiente candidato natural es la caché LLM persistente
(`PersistentLLMCache`, que ya usa el breaker best-effort); su acoplamiento no se midió en este lote.

---

## Qué decide el dueño

1. **Cohortes**: quién entra primero en cada canario (E5-B, P1-03, P1-04). Recomendación: él, como con el día
   determinista.
2. **Orden**: recomiendo E5-A (sombra, sin riesgo, ≈ 1 lote) → P1-03 sombra (un número que dice si urge) → P1-04 →
   P1-05. E7 son mediciones primero, todas baratas.
3. **Lo clínico** de P2-05 (política vegetal) y **lo comercial** de P2-03 (precios por mercado) no son técnicos.
