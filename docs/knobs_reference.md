# Knobs operacionales `MEALFIT_*` — referencia de discovery

<!-- tooltip-anchor: P3-CONFIG-KNOBS-DOC-REFRESH -->


[P2-KNOBS-OPERATIONAL-DOC · 2026-05-23 · conteo refrescado P3-CONFIG-KNOBS-DOC-REFRESH · 2026-07-09]
Este documento existe para resolver el gap operacional del audit 2026-05-23: el backend tiene
**cientos de env vars `MEALFIT_*`** referenciadas en código, pero `backend/.env.example` solo
documenta ~10 overrides recomendados. Un operador nuevo no sabe dónde están los demás, y durante
un incidente puede no encontrar el knob de mitigación.

**Conteo actual (2026-07-09):** ~**420 knobs registrados** en `_KNOBS_REGISTRY` at import (via
`get_knobs_registry_snapshot()`); el universo estático de nombres `MEALFIT_*` en source es mayor
(~900, incluye lecturas in-function que se registran al ejecutarse + referencias en comentarios/tests).
El conteo previo del doc ("~161") era stale. **NO confíes en un número hardcodeado** — corre el
listado vivo:

```bash
PYTHONPATH=backend python backend/scripts/dump_knobs.py          # tabla legible (regenera desde el registry)
PYTHONPATH=backend python backend/scripts/dump_knobs.py --json   # JSON para tooling
```

Este doc **NO mirror-ea** los knobs (drift garantizado — precisamente lo que le pasó al "~161").
En lugar de eso:

1. **Explica el mecanismo de auto-registro** (cómo descubrirlos at runtime).
2. **Enumera knobs de alto valor para producción** (los que un SRE
   probablemente quiera tunear sin esperar redeploy).
3. **Linka el endpoint público que expone los activos** + el script `dump_knobs.py` para el listado completo.

## Mecanismo de auto-registro

Todo lector de env var con prefijo `MEALFIT_*` en el código pasa por uno
de los wrappers:

| Wrapper | Tipo | Archivo |
|---|---|---|
| `_env_int(name, default, validator=None)` | `int` | `graph_orchestrator.py` |
| `_env_float(name, default, validator=None)` | `float` | `graph_orchestrator.py` |
| `_env_bool(name, default)` | `bool` | `graph_orchestrator.py` |
| `_env_str(name, default, validator=None)` | `str` | `graph_orchestrator.py` |
| `_knob_env_float(name, default, validator=None)` | `float` | `app.py` |
| `_env_int_safe(name, default)` | `int` (no registry) | `rate_limiter.py` |

Cada llamada AÑADE una entry a `_KNOBS_REGISTRY` (módulo
`graph_orchestrator.py`) con `{name, type, default, current, validator,
caller_module}`. El `rate_limiter._env_int_safe` es la excepción
intencional (no se registra para evitar circular import upstream;
documentado vía CLAUDE.md o memoria del bundle correspondiente).

## Descubrir los knobs activos at runtime

### Endpoint público `/health/version` (sin auth)

```bash
curl -s https://<host>/health/version | jq '.knobs_registered_count, .knobs_registered_preview'
```

El response expone:
- `knobs_registered_count`: total de knobs en `_KNOBS_REGISTRY`.
- `knobs_registered_preview`: muestra de keys recientes (no el set completo
  — sería verbose en el JSON público).

### Python interactivo (dentro del proceso backend)

```python
from graph_orchestrator import get_knobs_registry_snapshot
snapshot = get_knobs_registry_snapshot()
# dict[str, dict] — `{knob_name: {type, default, current, caller_module}}`
print(sorted(snapshot.keys()))
```

### Grep cross-codebase (last resort, source-of-truth)

```bash
# Todos los lectores de env var MEALFIT_*:
grep -rn "_env_int\|_env_float\|_env_bool\|_env_str\|_knob_env_float" backend/ \
  --include="*.py" | grep -v test_

# O directo desde os.environ:
grep -rn 'os.environ.get("MEALFIT_\|os.getenv("MEALFIT_' backend/ \
  --include="*.py" | grep -v test_
```

## Knobs de alto valor para producción

Estos son los que un operador probablemente quiera tunear sin redeploy
durante un incidente. La lista no es exhaustiva — para el conjunto
completo, usar `/health/version` o `get_knobs_registry_snapshot()`.

### Coherence guard (recetas ↔ lista de compras)

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_SHOPPING_COHERENCE_GUARD` | `block` | Pasar a `warn` o `off` si guard está rechazando planes legítimos en masa (false-positive burst) |
| `MEALFIT_SHOPPING_COHERENCE_TOLERANCE_PCT` | `0.10` | Subir a 0.15-0.20 si el LLM tiende a sub/sobre-multiplier por household >2x |
| `MEALFIT_COHERENCE_T2_BLOCK_SEVERE_ONLY` | `True` | Flip a `False` para revertir al warn-only puro si el block-severe genera retry storms |

### Lista de compras — ventana del viaje

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_TRIP_WINDOWED_PERISHABLES` | **`False`** | Encender SOLO tras cerrar TODOS los prerequisitos de abajo y re-medir `n_days`. Hoy es inerte |

**[P1-TRIP-WINDOWED-PERISHABLES · 2026-08-02]** Con el knob en `True`, los PERECEDEROS de
la lista se agregan solo desde los 7 días del viaje activo (los ESTABLES siguen del
agregado del periodo completo) en vez de promediar todos los días materializados y
proyectar a 7. Arregla que la lista del viaje 1 de un plan de 15/30 días trajera una
fracción del pollo de la semana 1 y una fracción del pescado de la semana 3 (que se daña).

**Nace apagado — capacidad dormida, no código muerto.** La ventana solo entra cuando
`len(days) > 7`, y la medición contra producción (2026-08-02, 40 planes más recientes, 23
con datos) da `n_days=2` en 3 planes y `n_days=3` en 20: **cero por encima de 3**. El shift
poda los días consumidos a la misma velocidad a la que los chunks los añaden, así que
`len(days)` orbita 2-3 permanentemente y `active_trip_window_days` devuelve `None` siempre.
Encenderlo hoy no cambiaría ninguna lista, pero sí expondría sus riesgos.

Prerequisitos para encender (detalle en el bloque de cabecera del P-fix en
[`shopping_calculator.py`](../shopping_calculator.py), tooltip-anchor
`P1-TRIP-WINDOWED-PERISHABLES`):

1. El shift debe reconstruir —o marcar para recálculo— la lista (hoy ninguno de sus dos
   paths toca `aggregated_shopping_list`); una lista stale post-shift pasa de promedio
   viejo benigno a divergencia SEVERA del guard.
2. Medir el impacto sobre `budget_reconciliation`: `cycle_total_rd` pasa a extrapolar la
   semana 1 al ciclo, y un `status="excedido"` **sustituye alimentos** del plan.
3. El último chunk de un plan de 30 días no tiene rebuild posterior.
4. [review final · 2026-08-03] Los dos callsites read-only de `tools.py`
   (`check_shopping_list` y `mark_shopping_list_purchased`) NO están cableados y sí piden
   `structured=True`: con la ventana activa reconstruirían el promedio del plan mientras el
   usuario compró la lista ventaneada, y `mark_shopping_list_purchased` cargaría a la despensa
   una lista distinta de la comprada. Este cuarto prerequisito existía en el código desde la
   ronda 1 y faltaba aquí — el doc listaba 3 y el código enumeraba 4.

El knob gobierna **cómo se construyen listas nuevas**, nunca **cómo se interpretan las ya
construidas**: el espejo del guard se dispara por el sello `trip_window_days` de la propia
lista y no consulta este knob. Sin esa asimetría, apagarlo con listas selladas vivas en DB
fabricaba la divergencia severa que el espejo existe para evitar.

**[P1-PLAN-LOTE-10 · 2026-09-11 · D6] El «bug latente» de la rama ventaneada ya está cerrado**: la segunda
pasada (`aggregate_and_deduct_shopping_list` sobre la ventana del viaje) recibe `text_demand_g_map` desde la
ronda de revisión de `P1-VEG-BACKFILL-HONESTY` (el comentario junto al call site lo documenta). Lo abierto son
los 4 prerequisitos, decisión del dueño; el knob sigue `False`.

### Sentry sampling (costo)

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_SENTRY_TRACES_SAMPLE_RATE` | `0.1` | Subir a `1.0` SOLO para debug intensivo de un deploy específico; lineal con costo |
| `MEALFIT_SENTRY_PROFILES_SAMPLE_RATE` | `0.1` | Igual que traces — profiling es aún más caro |

### Circuit breaker LLM

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_CB_FAILURE_THRESHOLD` | `3` | Subir a 5-7 si el provider LLM está flap-eando 5xx pero recovery rápido (3 es agresivo) |
| `MEALFIT_CB_RESET_TIMEOUT_S` | `30` | Subir a 60-120s si los flaps son largos (evita thundering herd post-reset) |

### Rate limiters

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_RATE_LIMITER_BUCKET_LIMIT_WARN` | `100000` | Bajar a 10000 si sospecha botnet (alert temprano por bucket cardinality) |

### Deploy lag detection

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS` | `1` | Bajar a `0.25` post-deploy crítico para confirmación rápida `drift=false` |

### DB pool

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_DB_POOL_MIN_SIZE` | `10` | Subir si los crons + chunks + meta-learning concurrentes están timing out en `pool.checkout()` |
| `MEALFIT_DB_POOL_MAX_SIZE` | `60` | Subir si Supabase pooler está reportando connection saturation |
| `MEALFIT_DB_POOL_TIMEOUT_S` | `10` | Subir a 30 durante migrations grandes que mantienen rows lock-eadas |

### Coherence cron (knobs de frecuencia)

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_COHERENCE_METRICS_INTERVAL_MIN` | `60` | Bajar a 15 durante incidente para feedback rápido |
| `MEALFIT_COHERENCE_CRON_PERSIST_HISTORY` | `True` | Flip a `False` si el cron genera contención con write paths del usuario |

### LLM model selection

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_<FEATURE>_MODEL` | varios | Swap de modelo LLM sin redeploy. Patrón `MEALFIT_CHAT_AGENT_MODEL`, `MEALFIT_CRITIQUE_MODEL`, etc. El override per-feature gana sobre el router por tier (P0-LLM-PROVIDER-MIGRATION). |
| `MEALFIT_MODEL_FREE_TIER` / `MEALFIT_MODEL_PAID_TIER` | `glm-5.3-flash` / `glm-5.3-flash` (P1-FLASH-PRIMARY) | Router por tier de suscripción — ver `backend/docs/llm_tier_routing.md` |
| `MEALFIT_ZAI_BASE_URL` | `https://api.z.ai/api/paas/v4` | Proxy/endpoint alternativo OpenAI-compatible |
| `MEALFIT_GLM_REASONING_EFFORT` | `low` | [P0-GLM-MIGRATION · 2026-09-02] Esfuerzo de razonamiento por default de TODO `ChatGLM` (`low`/`high`/`max`; GLM no puede apagar el thinking). Las superficies con effort propio (day-gen por tier, reviewer, corrector Pro, juez) lo pisan. Subirlo multiplica latencia y tokens de salida facturados |
| `MEALFIT_TIER_CACHE_TTL_S` | `300` | TTL del cache de `plan_tier` por usuario (clamp [10, 3600]) |

### Pantry / chunk operacional

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_SWEEP_ORPHAN_PLANS_AGE_DAYS` | `7` | Bajar a 2-3 si los orphans plans están saturando metrics (clamp [1, 90]) |
| `MEALFIT_SWAP_RECIPE_COHERENCE_VALIDATE` | `True` | Flip a `False` para revertir al pre-P1-SWAP-RECIPE-COHERENCE behavior si validator genera FPs |

### Día determinista (scorer)

| Knob | Default | Cuándo cambiar |
|---|---|---|
| `MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS` | `1.0` | Peso del EXCESO de carbohidrato en el scorer de `elegir_plantillas` (clamp [0.5, 5]); `1.0` = simétrico (conducta anterior). Medido 2026-09-11 en tres dianas: `2.0` recorta el carbohidrato 6-7 pts en pérdida/estándar a costa de ~2 pts de proteína y 2-4 platos distintos — **decisión del dueño** vía canario [P1-PLAN-LOTE-10 · B7] |
| `MEALFIT_DETERMINISTIC_DAY_W_FAT_DEFICIT` | `1.0` | Peso del DÉFICIT de grasa (clamp [0.5, 5]). A `2.0` arregla la grasa en la diana estándar pero hunde la proteína en pérdida (−12 %): dejar en `1.0` |

### Knobs default-off sin plan de activación (inventario F5 · 2026-09-11)

[P1-PLAN-LOTE-6] La auditoría de guards inertes (`docs/audits/f5_guards_inertes_2026_09_11.md` §5) encontró 21 knobs
`_env_bool(..., False)` con rama real y sin una sola mención en la documentación. Ninguno está en el `.env` de producción
salvo `MEALFIT_CARB_TARGET_TRIM`. Aquí quedan con su criterio de activación; el que no tenga dueño ni criterio en seis
meses es candidato a borrarse (el de `MEALFIT_SLOT_AWARE_DAY_REPAIR`, que además no tenía rama, se retiró en este lote).

| Knob | Dónde | Qué enciende | Criterio de activación / dueño |
|---|---|---|---|
| `MEALFIT_HARDEN_MAIN_ARITY` | `graph_orchestrator.py` | Clase 6 de A1-HARDEN-POOLS: exige ≥3 proteínas gate-label distintas en el pool para las comidas principales | `prod_profile` lo pone `true`: producción YA lo tiene encendido; el default de código quedó atrás. Alinear default o documentar el porqué |
| `MEALFIT_MICRONUTRIENT_SOFT_REJECT` | `graph_orchestrator.py` | Rechazo suave por micros alcanzables bajo el piso DRI (fibra/K/Mg/Ca; vit D/hierro/B12 excluidos) | Medir tasa de rechazo con `low_micros` en `_quality_degraded_reason` antes de encender (reintentos extra ⇒ coste) |
| `MEALFIT_FAT_LEAN_SWAP` | `graph_orchestrator.py` | Swap a proteína magra cuando la grasa cae en la zona muerta (reescribe nombre/pasos/raw) | Cambia la identidad del plato: sólo con la serie P2-FAT-DEADZONE + A/B |
| `MEALFIT_VARIETY_GATE_BASE_DISH_REPEAT` | `graph_orchestrator.py` | Gate de variedad sobre el plato-base repetido el mismo día | El vocabulario de plato-base incluye técnicas legítimamente repetibles («plancha»): encender sólo tras depurarlo |
| `MEALFIT_CARB_TARGET_TRIM` | `graph_orchestrator.py` | Recorte del carbohidrato al objetivo con recompute honesto de macros | Único que SÍ está en el `.env` de prod. Validar por A/B; si se queda, subir el default |
| `MEALFIT_CORRECTOR_NONE_DIAGNOSTIC` | `graph_orchestrator.py` | Re-invoca el modelo RAW cuando el corrector devuelve `None` y loguea `finish_reason` | Sólo diagnóstico puntual (2.ª llamada = tokens). Apagar al terminar la investigación |
| `MEALFIT_EVALUATOR_USE_PRO` | `graph_orchestrator.py` | El evaluador usa el modelo PRO | Toca todos los planes: sólo tras validar calidad/latencia en canario |
| `MEALFIT_DAYGEN_LITE_FOR_EASY` | `graph_orchestrator.py` | Modelo lite para perfiles fáciles (nunca clínicos complejos ni retries) | El operador lo activa tras validar calidad; medir `self_critique` correcciones (anulan el ahorro) |
| `MEALFIT_INITIAL_CHUNK_PANTRY_GUARD` | `constants.py` | Restaura el guard estricto de despensa en la generación INICIAL | Decisión de producto (P1-RENEWAL-PANTRY-IGNORE eligió no bloquear el primer plan). Dueño |
| `MEALFIT_RENEWAL_PANTRY_AWARE_ENABLED` | `constants.py` | En renovación, bloque advisory de duraderos en `build_pantry_context` | Rollout incremental: encender con `_renewal_pantry_aware` en un canario y medir el prompt |
| `MEALFIT_PANTRY_COMPLETION_LIST_ENABLED` | `constants.py` | Lista de faltantes read-only post-plan (lo que el plan necesita y la nevera no cubre) | Fase 2 de la despensa; necesita superficie en el frontend antes de encender |
| `MEALFIT_PANTRY_SUFFICIENCY_MICROS_GATE` | `inventory_sufficiency.py` | Los micros pasan de advisory a GATE en la suficiencia de despensa | **needs-owner** (carga clínica: puede pausar chunks por micros) |
| `MEALFIT_REQUIRE_ATOMIC_POOL` | `db_profiles.py` | Exige pool atómico; sin pool, falla en vez de degradar | Sólo producción; en dev/scripts no hay pool. Encender en el VPS si se quiere fail-loud |
| `MEALFIT_INVENTORY_RPC_STRICT` | `db_inventory.py` | Falla en vez de caer al camino legacy (sin control de carrera) del incremento de inventario | Encender cuando el RPC atómico lleve un ciclo sin alertas |
| `MEALFIT_LEAK_DB_ERRORS` | `error_utils.py` | Devuelve el error de DB crudo en la respuesta HTTP | Sólo dev. JAMÁS en producción (filtra SQL/paths) |
| `MEALFIT_READY_REQUIRE_DB` | `app.py` | `/ready` exige DB viva (503 sin ella) | Encender si el balanceador debe sacar del pool a un proceso sin DB; hoy `/ready` mide sólo el grafo |
| `MEALFIT_LIGHT_PROTEIN_SEED` | `ai_helpers.py` | Sortea el ancla proteica de desayuno/merienda (con OFF el prompt es byte-idéntico) | A/B pendiente (audit solver+seeder v4) |
| `MEALFIT_GROCERY_CYCLE_LOCK` | `ai_helpers.py` | Renovación reutiliza las compras del ciclo en vez de elegir ingredientes nuevos | Decisión de producto: ahorro vs. variedad. Dueño |
| `MEALFIT_ANEMIA_CONDITION_TARGET` | `micronutrients.py` | Objetivo de hierro por condición (anemia) en panel/PDF | **needs-owner** (user-facing clínico; validar con la tabla de condiciones) |
| `MEALFIT_DISABLE_SEMANTIC_CACHE` | `shopping_calculator.py` | Apaga el semantic cache | Interruptor de emergencia; encender sólo ante un incidente del cache |

### Sombra de la lista canónica (E5-A · `P1-PLAN-LOTE-19` · 2026-09-12)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CANONICAL_SHOPPING_SHADOW` | `True` | `canonical_shopping_shadow.emit_canonical_shopping_shadow` corre al final de `run_shopping_coherence_guard` (las 6 superficies) y persiste en `pipeline_metrics` (node `canonical_shopping_shadow`) la distancia entre la lista entregada y la que saldría de `IngredientLine`. Solo lectura sobre el plan. `False` = sin sombra, sin redeploy. Lector: `scripts/measure_canonical_shadow.py` |

### Asignación del horizonte por comidas viables (E6 · `P1-PLAN-LOTE-20` · 2026-09-12)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_HORIZON_VIABLE_FAMILY` | `True` (nació `False` en el lote 20; ON desde `P1-PLAN-LOTE-21` · 2026-09-12, decisión del dueño) | `horizon.build_blueprint`: si la familia de proteína del round-robin deja alguna franja del día sin candidato del registry, el día pasa a la familia del pool que cubre MÁS franjas (orden rotado, determinista) y queda anotado en `registry.family_reassignments`. Medido: vacías 4.311 → 100 de 49.200 (las 100 son huecos de biblioteca). Apagado ⇒ blueprint byte-idéntico (sólo el diagnóstico `registry.empty_slots`). Apagar sin redeploy: `MEALFIT_HORIZON_VIABLE_FAMILY=0`; los runs en vuelo conservan su blueprint. Lector: `scripts/measure_horizon_slots.py [--viable]` (sin `--viable` lo apaga explícitamente en su proceso) |

### El contrato sobre la receta final (C2 · `P1-PLAN-LOTE-23` · 2026-09-12)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_RECIPE_FINAL_CONTRACT` | `repair` | `recipe_contract.apply_final_contract[_meal]`, ÚLTIMO paso de `finalize_plan_data_coherence` y `finalize_single_meal_recipe_coherence`: los pasos siguen a la lista (gramos, unidades, piezas; nunca cruza familias). `shadow` anota en `_recipe_contract_final` lo que habría reescrito sin tocar; `off` apaga. Medido sobre el corpus fijo: V7e 38 → 3, V6 11 → 0, V4 4 → 0, 63 cantidades en 36 de 64 comidas; idempotente. Lector: `scripts/medir_contrato_receta_final.py [--vivo N]`. **[`P1-PLAN-LOTE-24`]** El mismo knob gobierna las tres formas del huevo (`reconcile_meal`) y las tres colas nuevas del persist boundary (`db_plans._finalize_plan_data_for_insert`, swap, chat-modify: `P1-PLAN-LOTE-24-FINAL-CONTRACT-TAIL`) |

### Las tres formas del huevo (C3 · `P1-PLAN-LOTE-24` · 2026-09-12)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_EGG_STAPLE_HONORED` | `True` | El huevo declarado BÁSICO (`stapleFoods`/`stapleAnchors`) se honra en su forma: `_diversify_egg_pools` no le quita el huevo al planificador a partir del 3.º día, y quien declaró la **Clara de huevo** (y no el huevo entero) recibe en el prompt del día «HUEVOS: CLARAS PRIMERO» en lugar de «ENTEROS PRIMERO» (`prompts.day_generator.override_egg_form_preference`, misma técnica que el tope de claras: se sustituye la regla, no se añade una contradicción). Lectura única de la declaración: `plan_policy.egg_staple_forms`. Sin declaración, prompt byte-idéntico y diversificador intacto. `False` vuelve a la conducta anterior sin redeploy |

### Asignación paso↔ingrediente de la receta congelada (C4 · `P1-PLAN-LOTE-25` · 2026-09-12)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_RECIPE_USAGE_EXACT` | `True` | En una comida de receta CONGELADA (`_recipe_source == "library"`) con asignación vigente (`data/registry/recipe_usage_do_v1.json`, atada al texto de los pasos por `pasos_hash`), los checks V3, V6, V7a y V7e del escáner culinario leen las CUENTAS de `recipe_usage` (Σ de fracciones por constituyente: 0 → V3, entre 0 y 1 → V7a, > 1 o dos veces «el resto» → V6) en vez de adivinar por texto — sobre pasos sin cifras, la heurística callaba. Comidas del LLM, recetas reescritas en el plato o asignación caducada: heurística de siempre. El estado del scan cuenta las comidas así evaluadas (`exactas`). Medido en la biblioteca: 155 exactas / 35 estimadas / 3 a revisar; 1004 de 1017 constituyentes con Σ = 1. `False` devuelve la heurística sin redeploy. Lector: `scripts/asignar_uso_pasos.py [--write\|--verificar\|--revisar]` |

### Cultura, horario, equipo, tiempo y básicos como contexto (C5 · primera parte · `P1-PLAN-LOTE-26` · 2026-09-12)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CULINARY_CONTEXT` | `True` | `culinary_context`: lo declarado por la persona es CONTEXTO del plato. Un básico declarado para una franja no es violación de horario (`_detect_slot_appropriateness`) ni recibe el autofix del arroz de noche; el techo de comidas con huevo honra al huevo básico (≥ 1 por día, nunca menos que `max(3, 25 %)`) y una clara de aglutinante no cuenta; el equipo declarado en Súper Personalización llega al prompt del día, al selector determinista (poda plantillas que lo exigen) y al juez (`contexto`); el pareo chocante respeta «al lado». `False` ⇒ conducta anterior en los cinco enganches, sin redeploy. Los checks V8a (tiempo oculto) y V8b (equipo no disponible) del escáner y la relajación `portion_cap_default_not_enforced` NO dependen del knob (son aviso/escritura, no conducta) |

### Estructura del plato y cadena de reparación medida (C5 · segunda parte (a) · `P1-PLAN-LOTE-27` · 2026-09-12)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_REPAIR_STAGE_DIFF` | `True` | `repair_stage_diff`: el escáner culinario de capa 1 fotografía el plan a la ENTRADA de `db_plans._finalize_plan_data_for_insert`, TRAS los caps de realismo y a la SALIDA (tras el contrato final); `plan_data["_repair_stage_diff"]` guarda conteos por check y etapa, los hallazgos NUEVOS con la etapa que los introdujo, los resueltos y el coste (ms); `warning` en el log si la cadena introduce algo. No muta ni bloquea. Sin catálogo o con > 200 comidas no mide y lo dice. `False` apaga las tres fotos sin redeploy. V9 (estructura del plato: `dish_structure`) es un check de aviso y no depende de knob |

### El paso del gate con el perfil de producción (F8 · `P1-PLAN-LOTE-33` · 2026-09-13)

**El problema.** `tests/conftest.py` apaga a propósito cinco gates y la suite deja otros knobs en su default de código:
**la suite entera mide un producto distinto del que se entrega**. `prod_profile.py` tiene los knobs del `.env` del VPS
(leídos el 2026-09-06) y la batería los usaba para una matriz de cohortes, pero ningún test de la suite corría con
ellos.

**Medido.** La suite (`-m 'not e2e'`, sin la cuarentena) con `prod_profile.perfil_completo()` exportado al entorno:
**109 fallos de 24819 tests, en 34 ficheros** (artefacto `scripts/data/f8_prod_profile_2026_09_13.json`). Atribución
fichero a fichero —el conjunto mínimo de knobs cuya retirada deja el fichero en verde, corriéndolo SOLO—: 30 por
un knob, 4 por combinación, 0 que sólo fallan en la corrida completa y
0 que fallen también sin el perfil. Knobs que explican: `MEALFIT_VERIFIED_INGREDIENTS_ONLY` 18, `MEALFIT_COUNTRY_SYSTEM` 5, `MEALFIT_SODIUM_EXCESS_GATE` 3, `MEALFIT_PLAN_JOBS_ENABLED` 2, `MEALFIT_RECIPE_CONTRACT_GATE` 2, `MEALFIT_MICRO_CLOSER_PERDAY` 2. `MEALFIT_VERIFIED_INGREDIENTS_ONLY` sólo explica 90 de los
109: son harnesses que construyen planes con alimentos sintéticos fuera del catálogo, y el filtro de verificados de
producción los descarta. **No se «arreglan» a ciegas.**

**El paso.** `scripts/prod_profile_gate.py` corre la suite con el perfil exportado (el `setdefault` de conftest no pisa una
variable que ya existe) menos `tests/prod_profile_excluded.txt` menos la cuarentena de la CI, y después la batería con el
entorno normal (la batería aplica el perfil por dentro y afirma que la suite diverge: bajo el perfil exportado fallaría por
diseño). La lista de exclusión es NEGATIVA: un test nuevo entra al paso por defecto, y cada línea dice el knob que la
explica y por qué. En la CI es una segunda pata en paralelo (`matrix.perfil: [suite, produccion]`), porque en serie no
cabía en los 50 min; en local, la tercera fase del gate (`EXIT_PROD`). Tiempo del paso en local: ~19 min.

**Sin base de datos (bis 4 · 2026-09-13).** La pata `produccion` de la CI dio **76 fallos en 19 ficheros** que el gate local nunca vio: el gate local tiene base (el `.env` del dueño) y la CI no. Bajo el perfil, `shopping_calculator` pide `master_ingredients` a la base (28.611 líneas «No connection_pool available to fetch master_ingredients» en el log) y el agregado queda vacío. Atribución en un árbol local sin `.env` con las versiones de `requirements.txt`, que reproduce los mismos fallos por fichero: 18 por `MEALFIT_VERIFIED_INGREDIENTS_ONLY`, 1 por sólo en la corrida completa. Esos ficheros van a `tests/prod_profile_excluded_sin_base.txt`, que el paso aplica SÓLO sin base (la misma señal que `conftest._db_available`): con base corren y pasan. Artefacto `scripts/data/f8_sin_base_2026_09_13.json` (run 34781436422). La CI imprime además los 30 tests más lentos (`--durations=30`) y su techo sube a 120 min: las dos patas llegan al 99 % en 13-40 min y la suite tarda 45-64 min más en cerrar — un test lento sólo en Linux, aún sin nombre.

**La cola, con nombre (`P1-PLAN-LOTE-37` · 2026-09-13).** `--durations` la nombró en su primera corrida: los 7 tests de `test_p1_arq27_f3_candidateset.py` pasaban 630-750 s CADA UNO en el setup (`build_blueprint`), 80 de los 88 min de la pata. Aislado, el mismo setup tarda 3,5 s: la diferencia la ponía `import app`, que inicializa Sentry. Sin DSN no envía nada, pero su integración de logging convertía cada `logging.error` en un evento completo —serializando las variables locales de cada marco— para tirarlo al final (13-21 ms por evento), y sin base de datos `get_master_ingredients` registra un error por llamada: 32.564 por blueprint. Arreglo: sin DSN, `sentry_sdk.init` no instala integraciones (con DSN, idéntico), y `tests/conftest.py` vacía `SENTRY_DSN` antes de que nada cargue el entorno. Medido en local, sin base y con el perfil de producción, los 14 tests de las cuatro familias más lentas corridos juntos: 478 s → 11 s. En la CI (run 34799738022): la suite 88 → 10,5 min y la de producción 87 → 8,3; con eso el techo bajó de 120 a 30 min (`P1-PLAN-LOTE-37 (bis)` · 2026-09-14), tres veces la pata más larga — una regresión de la cola se ve como `cancelled` en media hora, no en dos.

**Boy scout.** Al tocar un fichero de la lista, migrarlo a alimentos del catálogo real (o a declarar el knob que prueba
con `monkeypatch`) y borrar su línea: el test `tests/test_p1_plan_lote_33.py` exige que la lista sea exactamente la de
ficheros que fallan bajo el perfil en el artefacto, así que una línea de más o de menos se nota.

**La mitad de la lista, cerrada (`P1-PLAN-LOTE-65` · 2026-09-16).** Los 35 ficheros marcados con
`MEALFIT_VERIFIED_INGREDIENTS_ONLY` en las dos listas no usaban «alimentos sintéticos» por descuido: parchean el catálogo
a vacío (`get_master_ingredients → []`) para aislar SU tema, y con el knob en el valor de producción el filtro de
verificados —que sólo deja pasar lo que resuelve a un `master_ingredients` con precio— los dropea TODOS y no queda nada
que medir. El arreglo es el que la propia nota del boy scout permitía: cada fichero DECLARA el knob que necesita
(`_f8_verified_only_off`, fixture autouse con `monkeypatch.setenv` y el motivo escrito) en vez de heredar el `setdefault`
global de `conftest.py`. Los 35 pasan con los dos valores del knob (788 tests).

Re-medida la suite entera bajo el perfil CON base: **19 fallos en 17 ficheros de 25.332** (eran 109 en 34 de 24.819);
artefacto `scripts/data/f8_prod_profile_2026_09_16.json`. La lista general baja de 34 a 17 y **no entra ninguno nuevo**.
Lo que queda son knobs de CONDUCTA, no arneses: `MEALFIT_COUNTRY_SYSTEM` (4 ficheros), `MEALFIT_SODIUM_EXCESS_GATE` (3),
`MEALFIT_PLAN_JOBS_ENABLED` (2), `MEALFIT_MICRO_CLOSER_PERDAY`, `MEALFIT_RECIPE_CONTRACT_GATE`,
`MEALFIT_DREAMING_RETRIEVAL_ENABLED`, `MEALFIT_HARDEN_MAIN_ARITY`, `MEALFIT_PANTRY_COMPLETION_LIST_ENABLED` y tres
combinaciones (más la batería, que aplica el perfil por dentro): cada uno prueba el camino con SU gate apagado, que es su
tema, y sacarlo exigiría reescribir lo que el test afirma.

**La lista SIN BASE no se tocó, a sabiendas.** Se mide en la pata `produccion` de la CI (Linux, sin `.env`, con el
layout completo) y en esta máquina no hay forma honesta de reproducirla: un worktree sin `frontend/` ni `migrations/` al
lado da **1.611 fallos en 344 ficheros** que son de layout —i18n, landing, iOS, Playwright—, no del perfil. Medido y
descartado; la lista se re-mide cuando la CI vuelva a correr esa pata.

### La tormenta de reintentos del catálogo con la base caída (tarea propuesta del 13-sep · `P1-PLAN-LOTE-41` · 2026-09-14)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CATALOG_NEGATIVE_CACHE_S` | `30` (clamp [1, 300]) | Ventana (segundos) durante la que `shopping_calculator.get_master_ingredients` NO vuelve a tocar el pool tras un fallo («sin pool» o excepción del driver): devuelve `_master_cache or []`, registra el error UNA vez por ventana (las demás llamadas salen a DEBUG con el conteo) y JAMÁS sella `_master_cache_ts` (P1-CATALOG-INDEX-NO-STICKY: cinco minutos de vacío servidos como catálogo). El pool se comprueba ANTES que la ventana: si aparece, o es otro objeto, la ventana no aplica y se lee la tabla en esa misma llamada. `catalog_capability` cachea su `None` por país con la MISMA ventana (`None` sigue siendo «capacidad desconocida, no cero»). `invalidate_master_cache()` y `catalog_capability.reset_cache()` limpian los dos sellos. Subir a 120-300 si Neon tarda en volver y los blueprints de varios chunks compiten por el pool; bajar a 5-10 sólo para diagnosticar. Test: `tests/test_p1_plan_lote_41.py` |

**Medido** (`horizon.build_blueprint` de un perfil del landing, 14 días × 4 comidas, la misma vara antes y después; script del scratchpad de la sesión, nada escrito en el repo ni en la base). **Antes:** 32.676 llamadas a `get_master_ingredients` por blueprint (32.564 con el `_EFF` de `test_p1_arq27_f3_candidateset`, la cifra de la CI del 13-sep), **32.676 líneas de error** sin pool y **32.676 intentos de conexión** con un pool que falla — `catalog_capability` sólo cacheaba snapshots no vacíos, así que cada ancla, constituyente y plantilla volvía a preguntar, y `get_master_ingredients` no sellaba nada. **Después:** 0 líneas de error y 0 intentos DURANTE el blueprint (1 y 1 en todo el proceso: los sella la compilación de la política, que corre antes, dentro de la misma ventana de 30 s; con el `_EFF` de la CI, 1 llamada, 1 error, 0 intentos); las 112 llamadas que quedan las hace `dish_cost.tabla_de_precios` (una por plantilla costeada) y se absorben en la ventana, a DEBUG. Tiempo del blueprint 0,92 → 0,37 s. En producción, un arranque en frío con Neon caído pasa de decenas de miles de intentos —cada uno esperando el timeout del pool— a uno cada 30 s.

### La prueba RD del dueño: receta en su orden, día determinista y tiempo de cocina (`P1-PLAN-LOTE-45` · 2026-09-14)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_RECIPE_TDF_IN_PLACE` | `True` | En una receta NARRATIVA (pasos sin rótulo, como las 193 de la biblioteca) `_repair_recipe_contract` ROTULA el pilar en su sitio: «El Toque de Fuego: » delante del primer paso de cocción (mejor si ya dice tiempo o temperatura) y «Montaje: » delante del último si empieza sirviendo. `False` ⇒ vuelve la extracción de oraciones, que sacaba de su orden a 149 de 193 recetas. |
| `MEALFIT_SKELETON_FIDELITY_SKIP_DETERMINISTIC` | `True` | El pool de proteínas del planificador deja de juzgar a los días `_day_source="deterministic"` en `_run_assembly_validations`: esos días eligen por la familia del blueprint y el reintento los arma igual. `False` ⇒ vuelve el rechazo HIGH que quemó dos reintentos en el plan 40535829. |
| `MEALFIT_DETERMINISTIC_MEMORY_SEES_RECYCLED` | `True` | En un reintento quirúrgico la memoria entre días del día determinista (`_det_prev`) empieza con los días reciclados, así el día rehecho no repite su plato. |
| `MEALFIT_DETERMINISTIC_DAY_BLUEPRINT_FAMILY` | `True` | Si el esqueleto del día no trae `protein`, la familia sale de la rebanada del blueprint (`_blueprint_slice.days[*].protein`), no del pool de alimentos del planificador. |
| `MEALFIT_DETERMINISTIC_DAY_POOL_FAMILY_CANON` | `True` | Sin familia del blueprint, cada alimento del pool que no se entiende como familia se lleva a la suya antes de preguntar al registro («tilapia» → `pescado`: 1 → 13 almuerzos DO). |
| `MEALFIT_DETERMINISTIC_DAY_PINNED_SLOT_ALIAS` | `False` | Lee los candidatos fijados al run con la franja del motor (`1:lunch`), que es como `horizon` los guarda. APAGADO por medición: con 3 candidatos por franja elegidos sin mirar el tiempo, el blueprint real del dueño dio 19 min de media, 8 platos fuera de presupuesto y 2 repetidos en 3 días frente a 17, 6 y 1 de la consulta viva. |
| `MEALFIT_DETERMINISTIC_DAY_COOKING_TIME` | `True` | El tiempo de cocina del formulario (`cookingTime`: `none`=10, `30min`=30, `1hour`=60 min) ordena los candidatos por tramos (≤1,25×, ≤2×, ≤3× y el resto); dentro de cada tramo manda el ajuste de macros y la cuota de repetición va antes que el tiempo. |

**Medido.** Biblioteca DO completa por `_repair_recipe_contract`: antes 149 de 193 recetas con oraciones fuera de su orden (281 oraciones), ahora 0; 73 terminan sirviendo y ya no reciben un segundo «sirve» genérico. Día determinista sobre el blueprint real del dueño (run `f0bfd772`, «Nada» de tiempo, 3 días, simulación de solo lectura): antes 32 min de media, 10 de 12 platos por encima del presupuesto y 4 platos repetidos; ahora 17 min, 6 y 1, con Pollo, Pescado y Huevo como pedía el blueprint. Los que siguen por encima son almuerzos y cenas: el registro DO tiene 2 almuerzos y 1 cena de 10 minutos o menos.

### La segunda prueba RD: repetición entre planes, identidad del plato y parches (`P1-PLAN-LOTE-46` · 2026-09-14)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_DETERMINISTIC_DAY_RECENT_PLANS` | `True` | En el primer bloque, las plantillas servidas en los últimos 3 planes del usuario (el mismo número que el revisor) van al final de la lista, como saturadas. Una consulta por usuario cada 5 minutos. |
| `MEALFIT_DETERMINISTIC_DAY_BLOCK_REPEAT` | `True` | Dentro del bloque manda su propio tope de repetición exacta (`balanced`, 3 días ⇒ 1), no el de 7 días. |
| `MEALFIT_DETERMINISTIC_DAY_FAMILY_MAIN_SLOT_ONLY` | `True` | La familia del blueprint filtra sólo la comida principal (almuerzo, o cena si no hay); las demás franjas eligen libres. |
| `MEALFIT_DETERMINISTIC_DAY_SAME_DAY_PROTEIN` | `True` | La puerta de proteína repetida en el día del día determinista, separada de la de base ligera (`MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY`, que sigue apagada). |
| `MEALFIT_ANTI_REPETITION_SKIP_DETERMINISTIC` | `True` | El revisor no rechaza por platos repetidos contra planes recientes cuando están en días deterministas: el reintento los arma igual. |
| `MEALFIT_VARIETY_GATE_SKIP_DETERMINISTIC` | `True` | El rechazo por proteína repetida el mismo día cuenta `same_day_protein_repeats_modelo` (sin días deterministas); el informe sigue contando todos. |
| `MEALFIT_PROTEIN_AUTOFIX_SKIP_LIBRARY` | `True` | El autofix de proteína repetida no reescribe recetas congeladas: sus pasos están escritos para esa proteína. |
| `MEALFIT_DISH_IDENTITY_FLOOR` | `True` | El ingrediente que da nombre a un plato de biblioteca (lo que el nombre nombra y lo más pesado de la plantilla) no lo recortan los re-trims de grasa y carbohidrato; si aun así falta, vuelve con el 25 % de los gramos de la plantilla por el factor del plato. |

**Medido.** Simulación de solo lectura del día determinista, 7 días en dos bloques sobre el blueprint del último run del dueño (tiempo «Nada»), contra sus 3 planes más recientes: platos de esos planes 15 → 5 de 28, repeticiones dentro del bloque 4 → 2, días con proteína repetida 4 → 0, platos distintos 19 → 21, minutos medios 19,6 → 20,9. Identidad sobre una copia del plan 63eedc6b: vuelven 4 alimentos (+8 g de maní, +24 g de leche evaporada, +6 g de dátiles, +38 g de aguacate) y la grasa de los tres días queda entre el 91 % y el 107 %. La primera versión (piso del 50 % y subir también lo que quedó pequeño) tocaba 7 platos, llevaba la grasa al 116-122 % y el salami de 5 a 41 g: por eso el piso es del 25 % y sólo vuelve lo que falta.

### La tercera prueba RD: re-elegir, no reescribir (`P1-PLAN-LOTE-47` · 2026-09-14)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CRITIQUE_REPICK_DETERMINISTIC` | `True` | La autocrítica y la regeneración quirúrgica REARMAN un día determinista señalado con el armador sin LLM, pidiéndole evitar lo señalado; se acepta si sus señales verificables bajan, se conserva si no hay nada verificable ni nombrado, y sólo el que no mejora va al corrector LLM. La autocrítica barre además los demás días deterministas con señal. |
| `MEALFIT_CRITIQUE_RESTORE_PROVENANCE` | `True` | Cuando el corrector LLM reescribe un día, los platos de biblioteca que dejó iguales (mismo nombre, franja y alimentos) vuelven con su receta congelada y su plantilla; si vuelven todos, el día vuelve a ser determinista. |
| `MEALFIT_DETERMINISTIC_DAY_PROTEIN_BY_CONTENT` | `True` | La puerta de proteína del día lee lo que el plato LLEVA con el detector del revisor (16 de 193 plantillas esconden huevo bajo otra etiqueta), y las franjas que no son la principal no gastan la familia del blueprint. |
| `MEALFIT_DETERMINISTIC_DAY_GAINMUSCLE_DINNER` | `True` | En ganancia muscular la cena prefiere una proteína animal magra (el detector de la autocrítica: queso de plato ⇒ reserva). |
| `MEALFIT_DETERMINISTIC_DAY_BLOCK_STAPLES` | `True` | Prefiere no repetir entre días del bloque un básico que la autocrítica cuenta (yuca, avena, queso blanco…). |
| `MEALFIT_DETERMINISTIC_DAY_BLOCK_HEAVY_PROTEIN` | `True` | Prefiere no llevar una proteína pesada a un 3.er día del bloque (monotonía de la autocrítica). |
| `MEALFIT_DETERMINISTIC_DAY_SLOT_COHERENCE` | `True` | Prefiere no abrir una incoherencia de franja: almuerzo y cena con la misma proteína o carbohidrato, merienda de plato fuerte, plato fuera de horario. |
| `MEALFIT_DETERMINISTIC_DAY_BLOCK_DISH_BASE` | `True` | Prefiere no llevar una cabeza de plato (guiso, revoltillo…) al día que la haría «plato-base repetido». |
| `MEALFIT_DETERMINISTIC_DAY_LIGHT_BASE` | `True` | La puerta de base ligera (misma base en desayuno y merienda), con knob propio; `MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY` sigue encendiendo las dos a la vez. |
| `MEALFIT_TIMETEMP_SKIP_COLD_STEP` | `True` | El tiempo por defecto de «El Toque de Fuego» no se añade a un paso que ENFRÍA sin calentar («pásalos a agua fría»). |
| `MEALFIT_CARB_SWAP_TECHNIQUE` | `True` | Tras cambiar el arroz de la cena por casabe, la frase que lo hierve pasa a tostarlo y la que lo enjuaga, a tenerlo a mano. |
| `MEALFIT_SWAP_CHEESE_WORDING` | `True` | Tras cambiar un huevo por queso, sus verbos no se heredan: «revuelve queso blanco» → «dora el queso blanco» (un queso que no se dora se incorpora o se mezcla). |

**Medido.** Simulación de solo lectura del día determinista sobre el blueprint del run del plan d8b10b05 (7 días en dos bloques, tiempo «Nada», contra los 2 planes que eran recientes a esa hora): con las reglas del lote 46 la autocrítica saltaba en los dos bloques —el primero por 4 detectores: avena en 2 días, almuerzo y cena con yuca, avena en desayuno y merienda, huevo dos veces el día 3—; con las del 47 el bloque 1 queda limpio y el 2 salta sólo por yuca en 2 días. Platos de planes recientes 5 → 2, exceso sobre el tope de 7 días 0 → 0, platos distintos 21 → 21, minutos medios 20,9 → 25,9 (el coste: con 10 min la biblioteca no tiene alternativa en almuerzos y cenas). Sobre los días del lote 46 con la sugerencia real del evaluador, la re-elección rehízo los 3 días sin LLM (el 3 por el barrido) y la autocrítica quedó en «yuca en 2 días». La primera medición, con sólo tres reglas, llevó el pollo a 3 días de 3 (monotonía): por eso entraron la proteína pesada, el plato-base, la franja y la base ligera; y la cuota de repetición pesa a medias porque, entera, empataba con lo que hace saltar la autocrítica.

### La cuarta prueba RD: lo que añaden los cerradores respeta la receta (`P1-PLAN-LOTE-48` · 2026-09-14)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_DETERMINISTIC_DAY_PROTEIN_FLOOR` | `True` | En ganancia muscular el armador del día determinista prefiere, en cada franja, el plato que ya llega al piso de proteína del cerrador (`LIGHT_SLOT_PROTEIN_MIN_PCT` del objetivo de la franja). Pesa 0,4: menos que la cuota de repetición agotada (0,5) y que cualquier regla de la autocrítica, así que sólo decide entre candidatos que no rompen nada más. |
| `MEALFIT_GAINMUSCLE_FLOOR_OWN_BASE` | `True` | El piso de calorías de ganancia muscular escala hasta ×1,5 la base de carbohidrato que la receta de biblioteca ya trae (plátano, yuca, papa, batata…), en la lista del plato y en la de compras, en vez de colgarle arroz o batata; si no cabe en el techo del día, esa comida se salta. |
| `MEALFIT_CLOSER_JUICE_DAIRY_ASIDE` | `True` | En un jugo (chinola, limonada, «agua de…») lo que añade el cerrador de proteína se sirve al lado: el lácteo no se licúa (el ácido lo corta) y nada se incorpora al vaso; lo que se cocina, se cocina aparte. Las batidas y licuados siguen a la licuadora. |
| `MEALFIT_GENERIC_CHEESE_FROM_NAME` | `True` | El «queso» a secas de una comida toma el nombre del único queso que el plato promete (en el nombre o, si no, en sus pasos sin la Mise), en `ingredients` y en `ingredients_raw`. Con ninguno o con dos, no se adivina. |
| `MEALFIT_DESALT_CLAUSE_ONLY` | `True` | Tras cambiar un curado por uno fresco, del desalado se quita la cláusula y no la frase: «(ya desalado y en trozos)» → «(en trozos)», «el bacalao desalado» → «el bacalao». Las frases que mandan desalar se siguen quitando enteras. |
| `MEALFIT_REALISM_PULP_CAP_G` | `120` | Techo propio de las frutas de pulpa (chinola, maracuyá, parcha, granadilla) en `_cap_unrealistic_portions`, lista y compra. Rango 60-300. 335 g eran unas 14 chinolas para un jugo. |

La re-elección re-mide además su cola (sin knob propio: va con `MEALFIT_CRITIQUE_REPICK_DETERMINISTIC`): un día determinista que iba al corrector LLM y ya no tiene señal, porque otro rearmado se la quitó, se conserva.

**Medido.** Réplica de solo lectura del día determinista sobre el blueprint del run del plan 358a2cdf (7 días en dos bloques, «Nada» de tiempo, presupuesto bajo, contra los planes que eran recientes a esa hora): sin el piso, 5 de 28 comidas quedaban por debajo del piso de proteína de su franja (el desayuno medio, al 76 % de su objetivo). Con el piso pesando 1,5 quedaban 0 de 28, pero servía «Sardinas en lata con casabe» 4 veces en 7 días (3 en su bloque): el piso le ganaba a la cuota de repetición que el formulario («equilibrado») pide respetar. Con 0,4 —por debajo de la cuota— quedan 3 de 28 y la variedad no se mueve (0 sobre el tope de 7 días, 2 repetidos en bloque, 21 platos distintos, 6 de planes recientes; minutos medios 28,2 → 29,3). Los cambios de receta, comprobados sobre los platos del plan: la batida «con queso cottage» compra queso cottage (antes, «queso» ⇒ queso blanco), el jugo de chinola sirve el queso al lado, el locrio conserva «el filete de pescado blanco (en trozos)… y añade el arroz blanco», la chinola baja de 335 a 120 g y el piso de calorías escala el plátano del mofongo (×1,5 como mucho) en vez de colgarle arroz.

### La quinta prueba RD: el plato, el día y la receta, hasta el final (`P1-PLAN-LOTE-49` · 2026-09-14)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_DISH_IDENTITY_RAISE` | `True` | En la cola del guardado (después de todos los recortes), el ingrediente que da nombre a un plato de biblioteca que quedó por debajo de su piso (25 % de la plantilla × el factor) sube AL piso, en la lista y en la compra, si el día cabe en su techo de kcal y de grasa (×1,05 del objetivo, por knob desde P1-PLAN-LOTE-178) — y si no cabe entero, lo que quepa cuando eso deja el alimento en al menos la mitad de su piso; lo que siga en migajas se paga dentro del día con lo que el nombre no menciona (`MEALFIT_DISH_IDENTITY_COMPENSATE`). Mide los gramos con el lector de la base y nunca baja una línea. |
| `MEALFIT_CLOSER_LIGHT_SLOT_CLEAN` | `True` | En desayuno y merienda, antes de repetir la proteína que el día ya tiene o de pegar pescado o carne, el cerrador de proteína busca un lácteo (yogurt, cottage, ricotta, queso) que el día no tenga; el pool denso no los traía. |
| `MEALFIT_CLOSER_NO_SALTCURED` | `True` | El cerrador de proteína no añade un curado (arenque, bacalao, salami, longaniza…): el tope de sodio lo dejaba en migajas. Si no queda otro candidato, se conserva el pool. |
| `MEALFIT_CLOSER_STEP_PLACEMENT` | `True` | En una receta de biblioteca, el paso 💪 del cerrador va donde la receta lo haría: «Sirve X al lado» al emplatado; lo demás, como paso propio después del último paso con fuego (no dentro del primero, que desde el lote 45 lleva el rótulo «El Toque de Fuego»). |
| `MEALFIT_EGGCAP_RAW_WHOLE_EGG` | `True` | Cuando el tope diario de huevos no puede emparejar la línea de la lista con la de la compra por alimento («3 huevos» ↔ «165 g de huevo cocido»), reescribe la única línea de huevo entero de la compra. |
| `MEALFIT_DETERMINISTIC_DAY_TIME_FAULT` | `False` (medido: cambia tiempo por variedad; se enciende cuando la biblioteca tenga almuerzos y cenas rápidos) | El tiempo de cocina del formulario como falta del armador, proporcional al exceso sobre el presupuesto (`_falta_tiempo`). |
| `MEALFIT_TIMETEMP_SKIP_BRIEF_STEP` | `True` | Un paso que ya dice que dura poco («brevemente», «un momento») recibe «~1-2 min a fuego medio» y no el tiempo de su técnica (10-12 min tostaban un casabe). |
| `MEALFIT_CANNED_SWAP_CLAUSES` | `True` | Tras cambiar un enlatado por uno fresco (tope de sodio), los pasos pierden «ya escurridas» y «el líquido de la lata». |

Los planes recientes (el día determinista y el revisor) cuentan sólo los planes con días: la fila del plan que se está generando ya existe, vacía, y ocupaba uno de los 3 huecos (sin knob: es una consulta).

**Medido.** Réplica de solo lectura sobre el plan entregado a059d7bb: en la cola del guardado el día 1 recupera 38 g de aguacate en el guacamole (tenía 5), 8 g de maní (5) y 7 g de mantequilla de maní (3), y pasa del 89 % al 92 % de sus kcal y del 71 % al 86 % de su grasa; los días 2 y 3 no cambian (el 2 ya está en su techo). La primera versión medía los gramos con el lector de la lista del contrato, que lee «57.2 g» como 2 g, y en la réplica «subió» la soya de la cena del día 3 de 57 a 18 g: ahora mide con el lector de la base y nunca baja. La falta de tiempo se midió en réplica del run (7 días, «Nada», contra los planes recientes) con cuatro pesos, y ninguno mejora el tiempo sin romper otra regla: con 0,35/1/1,5 los almuerzos y cenas bajan de 36,4 a 35,0 min de media y el primer bloque deja de hacer saltar la autocrítica, pero el segundo pasa de «guiso ×3» a «pollo ×4»; con 0/1/2,5 desaparece la cena de 65 min (máximo 35, media 28,9) pero dos platos pasan su tope de 7 días; con 0/0,5/2,5 el máximo baja a 45 y un plato pasa el tope. La biblioteca DO tiene 2 de 63 almuerzos y 1 de 56 cenas de 10 min: el armador no tiene con qué. Queda APAGADA, con los pesos 0,35/1/1,5 (los únicos que no pasan el tope de 7 días), lista para cuando haya platos rápidos.

### Las decisiones delegadas del 14-sep (lote 44 del plan · `P1-PLAN-LOTE-61` · 2026-09-15)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_BILLING_VERIFY_AMOUNT` | `block` (era `warn`) | `off`/`warn`/`block` para `_verify_subscription_amount` (`routers/billing.py`). En `block` se bloquea (409 + alerta crítica `billing_price_tampering:*`) el underpayment PROBADO bajo un cupón re-validado (como antes) y, desde este lote, el override por debajo del precio de lista cuando NO existe ningún cupón activo aplicable al tier (`_active_coupon_exists_for_tier`: un cupón agotado cuenta, y uno vencido hace menos de 24 h también). Si existe alguno, o la consulta falla, el caso sigue siendo ambiguo: alerta sin bloquear (`P1-BILLING-AMOUNT-FP-FIX`). Decisión del dueño delegada el 14-sep («block»), con el cierre que proponía `paypal_audit_2026_08_22.md` §1. Rollback sin redeploy: `warn`. Tests: `test_p1_plan_lote_61.py`, `test_p1_billing_amount_verification.py` |
| `MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS_CANARY` | `1.0` (clamp [0.5, 5.0]) | Peso del exceso de carbohidrato del scorer del día determinista **sólo** para quien está en `MEALFIT_DETERMINISTIC_DAY_USERS`; el global (`MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS`) no se toca. El default es INERTE y no el 2.0 de la decisión B7: medido en el perfil del canario (ganancia muscular, 2600 kcal · 180/300/80 g, 14 días, `measure_deterministic_day_macros.py`): con 1.0 el carbohidrato queda en −5,5 % y la grasa en +4,7 % (en banda: C 8, G 7 de 14); con 2.0, −16,5 % y +22,9 % (C 5, G 5 de 14); la proteína, 14/14 en los dos (−4,5 → −5,1 %). En ganancia la biblioteca ya se queda corta de carbohidrato, así que castigar su exceso empuja a platos grasos. 2.0 ayuda en pérdida/estándar (`P1-PLAN-LOTE-10`); para un canario con ese perfil es esta variable. Test: `test_p1_plan_lote_61.py` |

### La cocción que falta (C5 · lote 39 del plan · `P1-PLAN-LOTE-62` · 2026-09-15)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CULINARY_V7F` | `True` | V7f `coccion_faltante` en la capa 1 del escáner culinario (`culinary_coherence._v7f_coccion_faltante`): víveres y proteínas animales que la lista no declara cocidos y ningún paso cuece, o que un paso usa ya cocidos. `warn` (`minor`, no reparable), un hallazgo por comida. `False` lo quita sin redeploy (0 hallazgos V7f; el resto del escáner, igual). El día determinista (`verifica_comida`) descarta al candidato con cualquier hallazgo: con la biblioteca encendida, V7f aparta 2 desayunos de 192 (el huevo revuelto sin paso que lo revuelva). |

### El juez por código (C5/C6 · lote 40 del plan · `P1-PLAN-LOTE-63` · 2026-09-15)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CULINARY_JUDGE_OBSERVACION_CODES` | `paso_incoherente,tecnica_impropia,nombre_no_corresponde` | Códigos del juez culinario en OBSERVACIÓN: el post-proceso de `run_culinary_judge` les pone `certeza="dudosa"`. Se emiten y se guardan igual (`_culinary_judge_history`); no deciden `blocked` cuando `MEALFIT_CULINARY_JUDGE_GUARD=block` y el marcador estricto no los cuenta salvo `--con-dudosas`. Default medido: precisión estricta < 25 % con n ≥ 4 en la columna del lote 38 (los tres dan 0 % estricta y 60-100 % por sustancia; tabla en `culinary_coherence.md`, «El juez por código»). `""` = ninguno; lo que no sea uno de los 5 códigos del schema se ignora. |

### El piso de proteína: tolerancia del rechazo (`P1-PLAN-LOTE-82` · 2026-09-17)

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_PROTEIN_FLOOR_RETRY_TOLERANCE_PCT` | `0.05` | Puntos por debajo del piso de proteína (90 %) que NO valen otro intento del LLM: un día entre el 85 y el 90 % del target se entrega aprobado y queda escrito en `_protein_floor_tolerated`; por debajo se rechaza «high» como antes. Solo sobre el piso estándar (bariátrica 80 % y renal exento no cambian). Validador `[0, 0.10]`; `0` = conducta anterior. Medido: el bench del 17-sep rechazó 3 veces por 1,2 g y entregó degradado un plan que medía 184/199/201 g de 198. |

## Cómo añadir un knob nuevo

```python
# En graph_orchestrator.py (o el módulo donde aplica):
_MEALFIT_MI_KNOB = _env_int(
    "MEALFIT_MI_KNOB",
    default=42,
    validator=lambda v: 1 <= v <= 1000,
)
```

El wrapper `_env_int` auto-registra el knob en `_KNOBS_REGISTRY`. La
entry aparecerá en `/health/version` en el próximo `import` del módulo.

**Convención** (CLAUDE.md → "Convenciones del repo"):
- Prefijo `MEALFIT_` (los demás `*_API_KEY`, `SUPABASE_*`, `PAYPAL_*` son
  secretos y NO van en el registry).
- Default SEGURO (el knob debe ser opcional — el sistema arranca sin él).
- Validator si hay clamp (e.g., porcentajes deben estar en [0, 1]).
- Si el knob va a la doc "Knobs operacionales" de CLAUDE.md, añadir
  también allí. Si es interno (no SRE-tunable), basta el registry.

## Anti-patrones

- **NO** leer env var con `os.environ.get(...)` directo en módulo
  productivo (el knob no aparece en el registry → invisible para SRE).
- **NO** dar al knob un default INSEGURO (e.g., `MEALFIT_DISABLE_AUTH=False`
  default está bien; `MEALFIT_DISABLE_AUTH=True` default abriría el
  sistema si la env var se borra accidentalmente).
- **NO** hardcodear thresholds que pueden necesitar rollback sin redeploy
  (e.g., timeouts de LLM, tolerancias de coherence guard, tier limits).

Tooltip-anchor: `P2-KNOBS-OPERATIONAL-DOC-START` | knobs discovery 2026-05-23
| `MEALFIT_IDENTITY_TAIL_KCAL_CEIL` | `1.05` | [P1-PLAN-LOTE-178] Techo de kcal del día para la cola de identidad del guardado (subir lo que da nombre al plato). Clamp [1,0, 1,15]. |
| `MEALFIT_IDENTITY_TAIL_FAT_CEIL` | `1.05` | [P1-PLAN-LOTE-178] Techo de grasa del día para esa misma cola. Clamp [1,0, 1,20]. |
| `MEALFIT_CONTRACT_APPROX_GRAMS` | `True` | [P1-PLAN-LOTE-182] el contrato de receta recorta los gramos de un paso que pide más de lo que pesa la línea contable de la lista («1 pechuga (≈134 g)»). Sólo recorta: «el paso pide menos» es la decisión V7a del dueño. |
| `MEALFIT_DM2_SWEET_FRUIT_CAP_G` | `120` | [P1-PLAN-LOTE-182] DM2: tope por línea de fruta dulce (piña, mango, guineo maduro, uvas…) dentro del tope glucémico existente; el guineo verde no cuenta. 0 = apagado. |
| `MEALFIT_REVIEW_BAND_RECLOSE` | `True` | [P1-PLAN-LOTE-186] si la puerta de banda del revisor ve alguna celda fuera de banda, re-cierra con la misma cadena del guardado (`apply_plan_quality_finalize_chain`) y vuelve a medir: mide lo que se guarda. |
| `MEALFIT_LINE_POLISH_TAIL` | `True` | [P1-PLAN-LOTE-181] pulido final del display al final de la cola del escudo (mayúscula tras «de», «½ pizca», especias sin unidad, migajas de semillas, «1 pechugas»). Sólo `ingredients`. |
| `MEALFIT_DM2_CASABE_CAP` | `True` | [P1-PLAN-LOTE-178] DM2: deja el primer casabe del bloque y cambia los demás por pan integral (salvo alergia/rechazo al pan, trigo o gluten). |
| `MEALFIT_DISH_IDENTITY_COMPENSATE` | `True` | [P1-PLAN-LOTE-178] En la cola del guardado, lo que da nombre y sigue por debajo de la mitad de su piso sube pagando con lo que ningún nombre menciona en el mismo día y del mismo macro; se revierte si el día acaba por encima de lo que tenía y de su techo. |
