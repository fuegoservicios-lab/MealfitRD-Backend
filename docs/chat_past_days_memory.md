# Memoria de días pasados en el chat-agent

[P1-CHAT-PAST-DAYS · 2026-07-27] Spec de diseño. El coach no recuerda los días
que ya pasaron: ni lo que el plan prescribió, ni lo que el usuario realmente
comió. Este documento fija el contrato; el test ancla es
[`test_p1_chat_past_days_memory.py`](../tests/test_p1_chat_past_days_memory.py).

---

## 1. El problema, verificado

Síntoma reportado (2026-07-27, screenshots del chat): el usuario pregunta
"¿qué comí ayer?" un lunes y el agente responde con **el menú que el plan
prescribió** para el Día 1 (Domingo), presentado como si fuera lo que comió.
El usuario había comido otra cosa.

Seis causas verificadas, todas independientes:

| # | Causa | Anclaje |
|---|---|---|
| C1 | El chat lee `consumed_meals` **solo de hoy** | [`agent.py:4354`](../agent.py) (stream), [`agent.py:3983`](../agent.py) (no-stream) → `get_consumed_meals_today` ([`db_facts.py:665`](../db_facts.py)) |
| C2 | Los días pasados del plan se **podan del prompt** | `_archived_days` está en `_CHAT_PLAN_PRUNE_KEYS` ([`agent.py:149`](../agent.py)) |
| C3 | **Ningún día del plan tiene fecha.** Solo `day: 1` y `day_name: 'Lunes'`, y `day` se renumera a 1..N en cada shift | [`plans.py:2215`](../routers/plans.py); verificado contra Neon 2026-07-27 |
| C4 | **No hay guarda de ignorancia.** La rama vacía dice solo "no ha registrado ninguna comida el día de hoy" — nada sobre otros días | [`agent.py:4381`](../agent.py) |
| C5 | Fósil `REGLA CRUCIAL: … "Opción A", "Opción B" y "Opción C"` destruye la identidad de día, y se **contradice dentro del mismo mensaje** con `"Nunca digas 'Opción A/B/C'"` | [`chat_agent.py:43`](../prompts/chat_agent.py) y [`:71`](../prompts/chat_agent.py) vs [`agent.py:3617`](../agent.py) |
| C6 | `build_temporal_context()` usa `datetime.now()` — **reloj del servidor**, no la zona del usuario — mientras otro bloque del mismo prompt usa UTC-4 | [`chat_agent.py:84`](../prompts/chat_agent.py) vs [`agent.py:3601`](../agent.py) |

**Por qué el agente rellena con el plan:** el prompt no le da ninguna
estructura con forma de día salvo `plan_data.days[]`, y nada le dice que
carece de datos. Sustituir prescripción por observación es el fallo esperado.

### Estado de los datos (Neon, 2026-07-27)

- `consumed_meals`: **8 filas en toda la producción**, 2 usuarios, ninguna
  desde 2026-07-13. Sin `plan_id`, sin índice de día, sin slot de comida —
  no hay llave de unión con el plan.
- `plan_data.days[]`: cada día trae `day` (int, renumerado), `day_name`,
  `meals[]`. Cada meal trae `meal` (Desayuno/Almuerzo/Merienda/Cena), `name`,
  `desc`, `ingredients` (**con cantidades y gramos inline**), `ingredients_raw`,
  `recipe` (lista de pasos), `cals/protein/carbs/fats`, `time`, `prep_time`.
- Tamaños medidos: **un día = 7.8–9.7 KB ≈ 2.2–2.7k tokens**. El plan podado
  que ya entra en cada turno = 32–36 KB (plan de 3 días) hasta 248 KB
  (plan maduro de 27 días) ≈ 9k–69k tokens.

---

## 2. La decisión

**Híbrido.** Lo barato se inyecta siempre; lo caro se trae bajo demanda.

Se descartó la variante "solo tool" porque su fallo es **silencioso**: con 11
tools compitiendo, cuando el modelo no llama la tool vuelve a rellenar el hueco
con el plan — que es exactamente el bug de hoy. Una línea índice por día pasado
(~200 bytes) elimina ese modo de fallo: el agente *sabe* que el día existe.

Se descartó inyectar los días pasados a fidelidad completa: 7 días serían
15–19k tokens **por turno**, encima de los 9–69k que ya cuesta el plan.

Se descartó explícitamente **montar esto sobre `user_facts`/Dreaming**:
`'dieta'` está en `CRITICAL_CATEGORIES` ([`fact_extractor.py:572`](../fact_extractor.py)),
así que cada comida pasaría por el merge de contradicciones que soft-borra
"redundantes" — "el lunes comí arroz" y "el martes comí arroz" no son una
contradicción, pero se fusionarían. Además los facts **no tienen fecha de
evento** (solo `created_at`), y se inyectan bajo `LOS HECHOS PERMANENTES SON LEY`
([`agent.py:4241`](../agent.py)), lo que haría que una comida episódica pesara
más que el plan vivo. Dreaming además está OFF en sus tres gates y sin cablear.

---

## 3. El diseño

### Pieza 1 — `date` estampada en cada día del plan

Contrato: **todo objeto `day` producido o renumerado desde este P-fix lleva
`date` en ISO `YYYY-MM-DD` (fecha local del usuario, convención RD = UTC-4).**

Tres sitios de escritura:

| Sitio | Qué estampa | Nota |
|---|---|---|
| Shift de `/shift-plan` ([`plans.py:2212-2215`](../routers/plans.py)) | `day_obj['date'] = target_date.isoformat()` | `target_date` **ya se calcula ahí** y hoy solo se usa para `day_name`; se descartaba |
| Archivado en el mismo shift ([`plans.py:2201-2207`](../routers/plans.py)) | fecha del día archivado `j` = `today - (shift_amount - j)` | Se estampa **antes** de extender `_archived_days`, para que el archivo nazca fechado |
| Shift gemelo del cron (`cron_tasks.py`) | idéntico a los dos anteriores | Es un duplicado literal del bloque de `plans.py`; ambos deben moverse juntos |

**No hay backfill destructivo.** Los planes vivos no se reescriben: eso
exigiría un full-overwrite bajo I7 sobre 15+ planes por un beneficio que la
lectura puede derivar sola. En su lugar el lector degrada.

**Regla de inferencia** (para días sin `date`). El ancla NO puede ser
`cycle_start_date + idx`: tras un shift `days[0]` es **hoy**, no el inicio del
ciclo, así que esa fórmula desplaza el plan entero. El ancla correcta es
`grocery_start_date`, que es el campo que el shift **reescribe a hoy** en cada
rotación ([`plans.py:2575`](../routers/plans.py) y su gemelo de `cron_tasks.py`)
y por tanto el único que sigue a `days[0]`:

```
resolver_fechas(plan_data, hoy):        # `_live_anchor`, 4 tiers en orden
    1. primera days[i]['date'] estampada        -> ancla (i, esa fecha)   [autoritativa]
    2. grocery_start_date  (fecha LOCAL)        -> ancla (0, esa fecha)
    3. primer i con day_name == weekday(hoy)    -> ancla (i, hoy)         [fallback]
    4. cycle_start_date    (fecha LOCAL)        -> ancla (0, esa fecha)   [último recurso]

    fecha(days[i]) = fecha_ancla + (i - idx_ancla)

    # días archivados: son estrictamente anteriores a days[0], en orden
    A = len(_archived_days)
    fecha(_archived_days[k]) = fecha(days[0]) - (A - k)
```

[P1-CHAT-PAST-DAYS · 2026-07-28] Los tiers 2 y 4 nacieron de dos fallos
**reproducidos contra planes de producción**; el orden de arriba es el contrato,
no una preferencia estética:

| Tier | Por qué está donde está |
|---|---|
| 2 `grocery_start_date` | `cycle_start_date` es el ancla **inmutable de creación** ([`plans.py:7754`](../routers/plans.py)); `grocery_start_date` es la que el shift reescribe. Plan real `4d2c1111` (17 archivados, grocery=28-jul, cycle=11-jul): anclando por cycle el plan entero se fechaba 24-jun…12-jul, fuera de la ventana de 7 días → `build_past_plan_days_block` devolvía `""` y ese usuario se quedaba **sin la feature**. Se pasa por `_to_local_date` porque la flota lo persiste en dos shapes (date-only, ver marker `P3-SHIFT-DATEONLY-LOCAL`, y timestamp UTC completo): un plan shifteado a las 22:00 RD ancla un día tarde si se lee crudo. |
| 3 `day_name` (degradado a fallback) | El escaneo coge la **primera** coincidencia y en una ventana viva de más de 7 días el `day_name` se repite: un desfase de 1 día se vuelve de 7. Plan real `69f9e03d` (26 días vivos) emitía 7 líneas de "días que ya pasaron" de las cuales **6 eran días futuros** (28-jul…2-ago). |

Solo el tier 1 marca `inferred=False`; los tiers 2-4 siguen marcando
`inferred=True` (el hedge `~` en el prompt).

Toda fecha que no venga de `day['date']` se marca en el prompt con `~`
antepuesto (`~Domingo 26 jul`) para que el agente no afirme como exacta una
fecha que el sistema no garantiza. En archivos con agujeros (el clamp
`shift_amount = min(...)`) la inferencia hacia atrás se desalinea; es
precisamente el caso que el estampado de la Pieza 1 elimina de aquí en
adelante.

**Lo que esto NO arregla:** los cinco modos de rotura de `_archived_days` (cap
a `total_planned_days+31` en [`plans.py:2206`](../routers/plans.py); borrado en
renovación en [`plans.py:2348`](../routers/plans.py); agujeros cuando
`shift_amount = min(days_since_creation, len(shifted_days))`; reset de
`cycle_start_date` al regenerar desde el front; planes anteriores a 2026-05-31
sin archivo). La fecha los vuelve **detectables**, no imposibles.

### Pieza 2 — Índice de días pasados, siempre inyectado

Bloque nuevo `📖 DÍAS QUE YA PASARON (lo que tu plan mandaba)`. Una línea por
día, los últimos `MEALFIT_CHAT_HISTORY_DAYS` (default 7), en orden
cronológico descendente. Fuente: `_archived_days` + los días de `days[]` cuya
fecha resuelta sea anterior a hoy.

Formato por línea (~200 bytes):

```
- Domingo 26 jul: Desayuno "Revoltillo de Tayota con Atún" 793 kcal · Almuerzo "Pulpo al Horno con Croquetas" 549 · Merienda "Batido de Lechosa" 204 · Cena "Pescado Guisado con Yuca" 603
```

**Solo nombres y kcal.** Sin `ingredients`, sin `recipe`, sin `ingredients_raw`
— eso es lo que la tool de la Pieza 4 sirve bajo demanda.

**Posición: después del bloque de diario**, es decir dentro de la sección
volátil, nunca antes del prefijo estático que protege el prompt-cache
([`agent.py:280`](../agent.py)).

### Pieza 3 — Diario real multi-día

`get_consumed_meals_today(...)` → `get_consumed_meals_since(user_id, hoy − N
días, include_ingredients=True)` ([`db_facts.py:722`](../db_facts.py), ya existe
y ya la usan ~13 callsites de producción; el chat es el único que no la usaba).

- `DIARIO DE HOY` se **conserva intacto** como sub-bloque: hay comportamientos
  que dependen de él (alerta de micro-adaptación, `_macro_totals_line`, la
  heurística de no re-registrar una foto ya registrada).
- Se añade `DIARIO DE DÍAS ANTERIORES`, agrupado por fecha.
- **Cada día sin registro se declara explícitamente**: `Domingo 26: sin
  registro`. Esto es lo que impide que el modelo rellene el hueco.

### Pieza 4 — Tool `consultar_dia_del_plan(user_id, fecha)`

Devuelve el día completo a fidelidad total: los platos con `ingredients`
(cantidades y gramos), `recipe` (pasos), macros y `desc`. Busca por fecha
resuelta en `days[]` + `_archived_days`.

- Entra en `agent_tools` ([`tools.py`](../tools.py)) → hereda automáticamente el
  override de `user_id` de P0-AGENT-1 al tope del loop `execute_tools`.
- **Obligatorio**: fila nueva en
  [`agent_tools_user_id_table.md`](agent_tools_user_id_table.md), o
  `test_p2_chat_cleanup.py` falla por paridad bidireccional.
- Acepta `fecha` ISO o lenguaje relativo resuelto por el agente ("ayer" →
  el agente ya tiene HOY en el prompt).

### Pieza 5 — Guardas de honestidad

1. Instrucción explícita de separación de fuentes: el bloque DIARIO es la
   **única** fuente de lo que el usuario **comió**; el bloque del plan es lo que
   se le **prescribió**; nunca presentar lo segundo como lo primero; si un día
   no tiene registro, decirlo.
2. Eliminar el fósil `"Opción A" / "Opción B" / "Opción C"` de los **cuatro**
   prompts base de [`chat_agent.py`](../prompts/chat_agent.py):
   `CHAT_SYSTEM_PROMPT_BASE` (:15), `CHAT_STREAM_SYSTEM_PROMPT_BASE` (:24),
   `CHAT_AGENT_INLINE_PROMPT` (:43) y `CHAT_STREAM_INLINE_PROMPT` (:71).
   Es un fósil de un producto de 3 opciones rotativas; hoy `days[]` es el chunk
   de generación de 3 días (`PLAN_CHUNK_SIZE = 3`) de un plan de hasta 30.
   Se **conserva** la prohibición explícita de
   [`agent.py:3617`](../agent.py) (`"Nunca digas 'Opción A/B/C'"`): es la que
   corrige al modelo, y [`test_p1_chat_today_context.py:37`](../tests/test_p1_chat_today_context.py)
   la ancla. También se limpia el vocabulario en el docstring de
   `modify_single_meal` ([`tools.py:2207`](../tools.py)), hoy dormido tras
   `MEALFIT_CHAT_PLAN_TOOLS_ENABLED=False`, para que no reenseñe el mapeo
   viejo si el knob se enciende.
3. `build_temporal_context(local_date=None, tz_offset=None)`: usar la fecha
   local del cliente cuando llega (el stream ya la recibe en
   [`chat.py:640`](../routers/chat.py) y hoy la ignora en este bloque), con
   fallback a UTC-4 — nunca `datetime.now()` del servidor.

---

## 4. Knobs

| Knob | Default | Clamp | Efecto |
|---|---|---|---|
| `MEALFIT_CHAT_HISTORY_DAYS` | `7` | `[0, 30]` | Ventana de los bloques de Piezas 2 y 3. **`0` apaga ambos** (rollback sin redeploy) |
| `MEALFIT_CHAT_HISTORY_MAX_CHARS` | `3000` | `[500, 20000]` | Cap duro **por bloque**; al excederse se truncan los días más antiguos primero |
| `MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED` | `True` | — | Kill switch de la tool de la Pieza 4 |
| `MEALFIT_UPCOMING_DAYS_UI` | `True` | — | [P2-CHUNK-OVERDUE-SIGNAL · Ronda 4/B4] Kill switch de la señal de **días futuros** (PENDIENTE/ATRASADO) en sus **tres** superficies: el payload `upcoming_chunks`/`overdue` de `/chunk-status`, el cron horario `_chunk_overdue_alert_job` y el índice del coach. SSOT del knob: `chat_history_context.upcoming_days_signal_enabled()` — las tres lo leen de ahí, no cada una la env var. Apagarlo además **resuelve** las alertas `chunk_overdue` abiertas: el único uso del switch es cortar una inundación (pasó: 19 de 23 planes), así que dejarlas abiertas lo haría inútil |

### La ventana del ciclo (`plan_cycle_window`)

[Ronda 4 · B1+B3] El término "¿el plan aún debe días?" que usan el predicado
`compute_chunk_overdue` y el índice del coach **no es un conteo** de
`_archived_days + days`, sino una **ventana de fechas**: el ciclo vigente
pretende cubrir `total_days_requested` días desde su inicio, y solo hay atraso
si hoy cae dentro de esa ventana. Dos razones, ambas medidas:

- **Renovación.** `_archived_days` nunca se vacía, tampoco cuando el plan
  renueva (`P0-1 RENEWAL`): dos ciclos comparten el array, el conteo alcanza
  `total` y apagaba las tres superficies **para siempre**. Verificado aplicando
  la transformación real de la renovación a una fila real de producción: antes
  `(False, None)` con 27 días sin generar; después `(True, …)`.
- **Caducidad.** El conteo no vencía nunca ⇒ alerta ATRASADO sin ningún camino
  de resolución en un plan cuyo ciclo terminó. 44 divergencias sobre filas
  reales a +30/+60 días.

El inicio del ciclo sale de `_cycle_started_at`, que **estampa la renovación**.
No sirve ninguno de los campos que ya existían, medido sobre los 24 planes de
producción (2026-08-04): `grocery_start_date == days[0].date` en **23/23** (es
la ventana rolling, el shift la reescribe en cada rotación) y
`cycle_start_date == created_at` en **19/23** (ancla inmutable de creación, la
renovación no la toca). Planes sin la marca degradan a la primera fecha
entregada — correcto para todo plan que nunca renovó, mudo para uno que ya
renovó (limitación asumida y anclada en test).

[P1-CHAT-PAST-DAYS · 2026-07-28] `MEALFIT_CHAT_HISTORY_MAX_CHARS` es **por
bloque, no combinado**: `_assemble` se invoca una vez por bloque con el mismo
valor (su propio docstring ya decía "por bloque"). El techo efectivo de los dos
bloques juntos es **2× el knob** — 6000 chars con el default. Si necesitas un
techo combinado real, hay que repartir el presupuesto entre las dos llamadas,
no bajar el knob a la mitad (eso recorta cada bloque por separado y el bloque
del plan, que es el caro, se lleva el recorte igual que el del diario).

`MEALFIT_CHAT_PLAN_DAY_TOOL_ENABLED` está **implementado**
([`tools.py::_chat_plan_day_tool_enabled`](../tools.py) +
`_apply_chat_tool_knobs`): con el knob en `false` la tool se retira de
`agent_tools`. Nótese la asimetría con `MEALFIT_CHAT_PLAN_TOOLS_ENABLED`
(default `False`, el knob **añade**): como este default es `True`, la tool vive
dentro de la lista literal `agent_tools` y el knob la **quita** — sacarla de la
lista literal rompería la paridad bidireccional que
[`test_p2_chat_cleanup.py`](../tests/test_p2_chat_cleanup.py) enforza contra
[`agent_tools_user_id_table.md`](agent_tools_user_id_table.md).

Todos se auto-registran en `_KNOBS_REGISTRY` vía `_env_int`/`_env_bool`
(P3-NEW-D).

## 5. Costo

| Bloque | Tamaño | Tokens |
|---|---|---|
| Índice de días pasados (7 días) | ~1.4 KB | ~400 |
| Diario multi-día (7 días con registro real) | ~0.7–1.7 KB | ~200–470 |
| **Total añadido por turno** | **~2–3 KB** | **~600–900** |
| Techo duro (cap por bloque × 2 bloques, default 3000) | **6 KB** | **~1.7k** |

≈ +$0.0004/turno en `deepseek-v4-pro`. Frente a los 9–69k tokens que el plan
JSON ya mete en cada turno, es ruido. La tool solo cuesta cuando se invoca.

[P1-CHAT-PAST-DAYS · 2026-07-28] La fila del techo se añadió al corregir la
lectura de `MEALFIT_CHAT_HISTORY_MAX_CHARS` (§4): el cap es por bloque, así que
el peor caso es **2×** el knob, no 1×.

## 6. Invariantes que respeta

- **I6/I2**: no se añade ninguna escritura de `plan_data` desde el cliente. Las
  fechas se estampan en los sitios backend que ya escriben bajo la transacción
  del shift.
- **I7**: no hay full-overwrite nuevo. El estampado ocurre dentro del
  `shifted_data` que el shift ya persiste bajo su lock existente.
- **P0-AGENT-1**: la tool nueva hereda el override de `user_id` del loop; se
  documenta en la tabla canónica.
- **DDL**: cero. `date` es una clave jsonb, no una columna.
- **`_LAST_KNOWN_PFIX`**: bump a `P1-CHAT-PAST-DAYS · 2026-07-28`; el slug
  cruza con `tests/test_p1_chat_past_days_memory.py`. (La fecha se alineó al
  valor real de [`app.py:34`](../app.py) — el doc decía `2026-07-27` mientras el
  código ya iba por el 28, y el marker NUNCA se mueve hacia atrás: es lo que un
  operador compara contra `/health/version` para confirmar el deploy.)

## 7. Lo que este diseño NO resuelve

1. **No hace que el usuario registre comidas.** Sin logging, la mitad
   "realidad" dirá "sin registro" correctamente y para siempre. Hoy
   `POST /api/diary/consumed` ([`diary.py:508`](../routers/diary.py)) no acepta
   fecha, no existe borrado ni edición por fila en ninguna capa (API, frontend
   ni tools), y el card de Progreso muestra el conteo y las barras de macros
   pero **nunca lista los platos**
   ([`TrackingProgress.jsx:294`](../../frontend/src/components/dashboard/TrackingProgress.jsx)).
   Eso es trabajo de producto, fuera del alcance de este P-fix.
2. **No crea la llave de unión plato-a-plato** entre `consumed_meals` y el
   plan. El contraste "planeado vs real" lo hace el LLM en prosa a partir de
   los dos bloques del mismo día. Una métrica numérica de adherencia seguiría
   dependiendo del matching difuso que ya existe en los crons.
3. **No repara los cinco modos de rotura de `_archived_days`** (§3, Pieza 1).
