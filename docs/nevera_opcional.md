# La Nevera opcional en modo contador

[P1-NEVERA-OPCIONAL · 2026-09-23] Motor SSOT [`nevera_opcional.py`](../nevera_opcional.py). Spec: `docs/superpowers/specs/2026-09-23-nevera-opcional-y-compartir-dia-design.md` §2 (workspace-root). Test: [`tests/test_p1_nevera_opcional.py`](../tests/test_p1_nevera_opcional.py). Este doc describe el código TAL COMO quedó construido — donde diverge del plan original, lo dice explícitamente.

## 1. Qué es y por qué

El dueño: «cuando el generador de planes esté desactivado, que la Nevera también tenga una opción en Configuración para desactivarla: hay gente que solo quiere el contador y el agente». Y sobre el estado inicial: «encendida como hoy, pero si en 48 horas no se usa, igual que con la hidratación, que se desactive sola».

Dato que sostuvo la decisión (medido el 23-sep, solo lectura): **6 cuentas en modo contador, las 6 con la Nevera vacía**; de 6 en modo plan, 4 vacías. Quien usa la app solo como contador casi nunca la llena — mantenerla encendida por defecto para siempre es una tarjeta de Configuración y una entrada de navegación que no sirven a nadie en ese modo.

## 2. LA regla (una sola)

```
nevera_activa = NOT (nevera_enabled IS FALSE AND (plan_mode = 'tracking' OR MEALFIT_NEVERA_OFF_IN_PLAN_MODE))
```

[P1-PLAN-LOTE-217 · 2026-09-24] Desde el 24-sep la regla cubre también el modo plan (§8). Lo que este documento cuenta
del modo plan antes de §8 describe la regla del 23-sep, que vuelve con `MEALFIT_NEVERA_OFF_IN_PLAN_MODE=false`.

`nevera_enabled` es TRIESTADO en `user_profiles`:

| Valor | Significa |
|---|---|
| `NULL` | Automático: encendida, elegible para el apagado automático (§3) |
| `TRUE` | Encendida por el usuario — **nunca** se apaga sola |
| `FALSE` | Apagada — por el usuario o por el sistema |

Dónde vive: [`nevera_opcional.nevera_activa_de(perfil)`](../nevera_opcional.py) es la función **pura** (recibe un dict, no toca la DB); `nevera_opcional.nevera_activa(user_id)` la envuelve leyendo la fila, fallo abierto (sin usuario, invitado, columna sin migrar o DB caída ⇒ `True` — equivocarse hacia «activa» es la conducta de siempre, hacia «apagada» pierde un descuento real). `estado_nevera(user_id)` arma lo que pinta Configuración; `fijar_nevera` es la elección explícita (§3).

Cómo llega al frontend: `GET /api/profile` la calcula inline — **no** en `_normalize_profile_row` (ese helper de `db_profiles.py` solo normaliza tipos UUID/datetime/Decimal, no conoce la Nevera) — sino directo en [`api_get_profile`](../routers/user_data.py) (`routers/user_data.py`): `{**profile, "nevera_activa": nevera_activa_de(profile)}`. El frontend nunca reimplementa la regla: `neveraActiva(userProfile)` (`frontend/src/config/dashboardNav.js`) la LEE, en este orden: (1) fuera del modo contador (`!isTrackingMode(userProfile)`) ⇒ `true` siempre — `saveGeneratedPlan` pasa `plan_mode` a `'plan'` en memoria sin volver a pedir el perfil, así que el `nevera_activa: false` del contador seguiría ahí; (2) con perfil cargado ⇒ `userProfile.nevera_activa !== false` — un perfil SIN el campo (backend viejo, rollback) cuenta como activa y **no** cae al espejo, que podría ser de antes del rollback; (3) solo sin perfil (primer pintado) ⇒ el espejo `localStorage['mealfit_nevera_activa']` (lo siembran el perfil y cada toggle de Configuración). «No sé» ⇒ activa.

**En modo plan vuelve sola**: la regla ignora `nevera_enabled` cuando `plan_mode != 'tracking'` (lista de compras, reposición y «Me lo comí» la necesitan), sin tocar el flag — al volver al contador se respeta la última elección explícita.

Kill switch `MEALFIT_NEVERA_SWITCH` (default `True`): en `False` la regla devuelve siempre `True`, la tarjeta de Configuración no aparece y el cron de §3 no apaga a nadie.

## 3. Apagado automático

Cron horario `_nevera_auto_off_job` ([`cron_tasks.py`](../cron_tasks.py)), registrado en `register_plan_chunk_scheduler` con `id="nevera_auto_off"` cada 60 min, llama a `nevera_opcional.apagar_neveras_sin_uso()`: un solo `UPDATE … RETURNING` por lotes (`_SQL_APAGAR`, SQL exacto en §7).

Condición exacta: `plan_mode = 'tracking'` **y** `nevera_enabled IS NULL` (nunca eligió; `TRUE`/`FALSE` explícitos quedan fuera) **y** el reloj de 48 h vencido **y** `NOT EXISTS` una fila de `user_inventory` con `quantity > 0` **o** `updated_at` dentro de esa misma ventana. «Vacía» y no «sin añadir nada»: quien la llenó la semana pasada y lleva dos días sin tocarla sigue con cantidad > 0, así que no calza. El `UPDATE` externo repite sobre la fila que escribe (`p`) lo que el usuario puede cambiar entre el `SELECT` interno y el `UPDATE`: `nevera_enabled IS NULL` (una elección suya colada entre medias) y `plan_mode = 'tracking'` (encendió el generador entre medias). En READ COMMITTED Postgres re-evalúa sobre la versión nueva de la fila solo las condiciones del alias que se actualiza; las del subselect (`q`) se quedan con la foto del inicio de la sentencia.

El reloj: columna `nevera_reloj_desde` (`TIMESTAMPTZ NOT NULL DEFAULT now()`). La migración la estampó con SU propia hora de ejecución en las filas existentes — **las cuentas de antes del despliegue tienen 48 h de gracia desde ese momento**, no desde que se creó la cuenta. El punto de partida real es `GREATEST(nevera_reloj_desde, COALESCE(plan_mode_changed_at, nevera_reloj_desde))`: cambiar a modo contador reinicia el reloj.

La nota (una vez por dispositivo y por apagado, sin push): `DashboardTracking.jsx` dispara un toast al montar **solo cuando `modo === 'contador'`** — este mismo componente es también la pestaña «Progreso» del modo plan, donde la Nevera es obligatoria, y la guarda corta antes de mirar nada más — si `userProfile.nevera_activa === false` y `userProfile.nevera_auto_off_at` es una fecha nueva (`localStorage['mealfit_nevera_auto_off_visto']` guarda la última vista; un apagado posterior con otro timestamp vuelve a avisar). También queda como línea fija en la tarjeta «Nevera» de Configuración mientras `auto_off_at` siga puesto.

**Encenderla a mano es definitivo**: `fijar_nevera(user_id, True)` escribe `nevera_enabled = TRUE` explícito (nunca vuelve a `NULL`) y limpia `nevera_auto_off_at`. Sale del pool `IS NULL` para siempre — no hay camino de producto de regreso al automático.

`MEALFIT_NEVERA_AUTO_OFF` (default `True`) apaga solo el cron sin tocar el interruptor manual; `MEALFIT_NEVERA_SWITCH` en `False` apaga las dos cosas (§2).

## 4. Qué hace «apagada»

| Superficie | Qué cambia |
|---|---|
| Nav (lateral, barra móvil, menú ☰, menú del Agente) | `navItemsFor({ trackingMode, nevera: neveraActiva(userProfile) })` sin la entrada `pantry` (`dashboardNav.js`) |
| Rutas `/dashboard/pantry`, `/pantry`, `/mi-nevera` | `Pantry.jsx` redirige a `/dashboard` (patrón de `Recipes.jsx`) |
| Configuración → Capacidades | Tarjeta «Nevera» (toggle) solo si `enModoContador && neveraEstado?.disponible`; línea fija si la apagó el sistema |
| Escáner de comida (`ScanMealModal.jsx`) | No pide `GET /api/inventory`; «Descontar de tu Nevera» se sustituye por «Ingredientes que detectamos»; foto de compra ⇒ aviso sin «Escanear mi nevera». Sigue al perfil EN VIVO: el modal está siempre montado, así que si la Nevera se apaga con él abierto (el refresco tras el apagado automático, o apagarla en Configuración) el rótulo y los interruptores de descuento cambian en el acto, y al volver a encenderla pregunta el inventario de nuevo. **Corrección vs. el plan**: no existe un flag `deduct` que el cliente mande — `ConsumedMealRequest` (`routers/diary.py`) no lo tiene. El corte real es 100 % server-side, ver fila del diario |
| «Registrar comida» (manual/repetir) | Mismo `_persist_consumed_meal`, ver fila del diario |
| Agente (bienvenida del contador) | Sin «Dime qué hay en tu nevera» (`AgentPage.jsx`, `neveraOn = neveraActiva(userProfile)`) |
| Ayuda (widget + `prompts/help_bot.py`) | `HelpChatWidget.jsx` (`getSuggestions(t, contador, nevera)`) omite la sugerencia «¿Para qué sirve la Nevera?» cuando `nevera=false` (la pantalla ni existe); `prompts/help_bot.py` dice que la Nevera sale en la nav «si el usuario la tiene encendida (se apaga en Configuración → Capacidades)» |
| `_persist_consumed_meal` (diario — foto/manual/repetir, `routers/diary.py`) | `deduct AND nevera_activa(user_id)` decide el descuento; `tools.log_consumed_meal` comparte el mismo gate vía el mismo import a nivel de módulo |
| Coach: contexto (`agent._build_pantry_context`, `build_inventory_context`, `build_vision_context`) | Inventario vacío + `nevera_opcional.BLOQUE_PROMPT_NEVERA_APAGADA` en vez de la línea de inventario. **Cierra también el respaldo `current_pantry_ingredients`**: ese snapshot de la última generación vive en `form_data` y antes se leía siempre como último recurso; ahora `if not inventory_str and form_data and _nevera_on:` — apagada, NO hay respaldo, para no presentar una foto vieja como «lo que tiene ahora». Visión de compra sin «ofrece agregarlas a la Nevera» |
| Coach: tools (`tools.py`) | `check_current_pantry`/`modify_pantry_inventory` → `MENSAJE_NEVERA_APAGADA` sin tocar la DB; `log_consumed_meal` no descuenta; `correct_consumed_meal` con ingredientes nuevos **no toca el inventario** — ni devuelve lo que el registro original descontó (si lo hizo antes de apagarla) ni descuenta la lista corregida. **Cambio de la ola final (2026-09-24)**: antes devolvía sin volver a descontar, y el inventario oculto quedaba como si la comida nunca hubiera ocurrido — pero la comida SÍ ocurrió, solo cambió de ingredientes (borrarla es otra cosa, ver la fila de borrar); `proponer_comida` no lee `user_inventory`; `mark_shopping_list_purchased` responde `MENSAJE_NEVERA_APAGADA` y omite sugerir `modify_pantry_inventory` |
| Cron de descuentos fallidos (`_process_failed_inventory_deductions_queue`) | **Sin cambio, a propósito**: apagada no nacen fallos nuevos (no se descuenta), y reintentar los pendientes de ANTES de apagarla mantiene el inventario oculto coherente — mismo argumento que borrar una comida (fila siguiente) |
| Borrar una comida del diario (`DELETE /api/diary/consumed/{meal_id}`) | **Sin cambio**: `revert_consumption_events` corre incondicional, sigue devolviendo lo que se había descontado — una comida borrada no ocurrió, así que su descuento sobra (al revés que corregir, fila de las tools) |
| Inventario (`user_inventory`) | **No se borra** nunca por apagar la Nevera |

## 5. Knobs y rollback

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_NEVERA_SWITCH` | `True` | Kill switch del feature completo. En `False`: regla siempre `True`, sin tarjeta en Configuración, el cron no apaga a nadie, `PATCH /nevera` responde 409 — **todo como antes de este P-fix, sin redeploy** |
| `MEALFIT_NEVERA_AUTO_OFF` | `True` | En `False` desactiva SOLO el apagado automático; el toggle manual de Configuración sigue funcionando |
| `MEALFIT_NEVERA_AUTO_OFF_HOURS` | `48` | Horas de inactividad para el apagado automático. Validador `24 <= v <= 336`; fuera de rango cae al default 48 (rechazo-y-default, no clamp al borde — mismo patrón que el resto de knobs int de `knobs.py`) |

## 6. Migración

[`migrations/p1_nevera_opcional_2026_09_23.sql`](../migrations/p1_nevera_opcional_2026_09_23.sql) (copia idéntica en `backend/migrations/` y en `migrations/` del workspace-root, `P3-MIGRATIONS-SSOT`). Añade tres columnas a `public.user_profiles`, las tres `ADD COLUMN IF NOT EXISTS` + sanity `DO $$ RAISE EXCEPTION`: `nevera_enabled BOOLEAN` (triestado, §2), `nevera_auto_off_at TIMESTAMPTZ` (cuándo la apagó el sistema, §3), `nevera_reloj_desde TIMESTAMPTZ NOT NULL DEFAULT now()` (el reloj de las 48 h, §3).

Se aplica con `scripts/apply_migration.py --apply` **antes** de desplegar el backend que la lee. Sin la migración, las LECTURAS fallan abiertas (`nevera_activa`, `estado_nevera` y `GET /api/profile` ⇒ la Nevera queda activa para todos, sin error), pero no todo es inocuo: la tarjeta «Nevera» de Configuración **sí aparece** en modo contador (`disponible: true` sale solo del knob, no de la columna), y al pulsarla el `PATCH /api/user/preferences/nevera` responde **500** (`fijar_nevera` no encuentra `nevera_enabled`); además el cron de §3 deja un `logger.error` cada hora. Por eso la migración va primero.

## 7. SOP de verificación (solo lectura)

Candidatas al próximo apagado automático (el `WHERE` interno de `_SQL_APAGAR`, con el knob ya sustituido por su valor efectivo, hoy 48):

```sql
SELECT q.id FROM user_profiles q
 WHERE q.plan_mode = 'tracking'
   AND q.nevera_enabled IS NULL
   AND GREATEST(q.nevera_reloj_desde, COALESCE(q.plan_mode_changed_at, q.nevera_reloj_desde))
       < now() - (48 * interval '1 hour')
   AND NOT EXISTS (
         SELECT 1 FROM user_inventory i
          WHERE i.user_id = q.id
            AND (i.quantity > 0 OR i.updated_at > now() - (48 * interval '1 hour')));
```

Estado de una cuenta puntual:

```sql
SELECT plan_mode, nevera_enabled, nevera_auto_off_at, nevera_reloj_desde
  FROM user_profiles WHERE id = '<uuid>';
```

API equivalente (autenticada, mismo shape que `estado_nevera`): `GET /api/user/preferences/nevera` → `{enabled, activa, auto_off_at, disponible}`. `PATCH /api/user/preferences/nevera` con `{enabled: bool}` es la única escritura de la elección explícita del usuario (`routers/preferences.py`); cero costo LLM, `get_verified_user_id` sin `verify_api_quota` — mismo criterio que `water-tracker`.

## 8. Tests

- [`tests/test_p1_nevera_opcional.py`](../tests/test_p1_nevera_opcional.py) (backend): la regla, el kill switch, el fallo abierto, `fijar_nevera` filtrando por `id`, el SQL del apagado automático (incluido lo que el `UPDATE` externo repite), la migración, el endpoint, el perfil (la fuente y el `api_get_profile` real), el diario (con/sin descuento), el cron, las tools del coach (corregir apagada no toca el inventario), el texto neutro de la orden, los dos caminos del chat, el bot de ayuda y esta documentación.
- `frontend/src/__tests__/NeveraOpcional.contract.test.jsx`: `neveraActiva` (modo plan ⇒ activa; perfil sin el campo ⇒ activa; espejo solo sin perfil), nav, ruta y espejo, Configuración, nota única.
- `frontend/src/__tests__/NeveraOpcional.superficies.test.jsx`: registro manual, Ayuda, bienvenida del Agente y la fuente del escáner.
- `frontend/src/__tests__/ScanMealModal.photo_deducts.test.jsx` («con la Nevera apagada»): sin `GET /api/inventory`, rótulo neutro, aviso de compra y el cambio del perfil con el escáner montado (apagar ⇒ desaparecen los interruptores de descuento; encender ⇒ vuelve a preguntar el inventario).

## 8. También en modo plan (P1-PLAN-LOTE-217 · 2026-09-24)

El dueño: «lo de la nevera opcional, si consideras que es lo mejor, hazlo». El caso que lo motivó: `c7b90ca3` creó su
plan de 15 días el 17-sep, nunca abrió la Nevera y P1-PLAN-FREEZE congeló el plan el 19-sep con 12 días por generar.

- **La tarjeta de Configuración existe en los dos modos.** Apagada en modo plan: los bloques se generan sin mirarla
  (`_refresh_chunk_pantry` devuelve Nevera vacía + `_pantry_advisory_only` sin leer el inventario oculto), ninguna guarda
  de Nevera pausa, «Me lo comí» no descuenta, el coach no la usa y la lista del Dashboard no resta nada ni recupera lo
  suprimido (`getDeltaSourceList(..., { conNevera: false })`). Apagarla descongela al instante un plan congelado
  (`try_unfreeze_plan_for_user` desde `PATCH /api/user/preferences/nevera`).
- **Automática (NULL) en modo plan:** el barrido del congelado, donde antes CONGELABA a las 48 h vacía, apaga la Nevera
  (`apagar_por_plan_vacio`) si el usuario sigue activo (`activo_reciente`: chat, diario, Nevera o plan nuevo en
  `MEALFIT_NEVERA_ACTIVE_DAYS`, 14) y el plan sigue; si ya estaba congelado por eso, se reanuda. Sin uso reciente se
  congela como siempre: el congelado era el freno del gasto en cuentas abandonadas.
- **Encendida a mano (TRUE):** conserva el congelado — eligió cocinar con lo que tiene.
- **El relleno en segundo plano** (`trigger_background_rolling_refill`) no rellena a quien tiene la Nevera apagada y no
  usa la app: sin congelado, ese es el freno.
- **Compra única (P1-PLAN-LOTE-216):** con la Nevera apagada o vacía, el bloque 2+ de una compra única recibe como Nevera
  la compra del ciclo (`compra_unica.nevera_virtual`): lo comprado ES lo que hay en casa, y el revisor lo exige.

Tests: [`tests/test_p1_plan_lote_217.py`](../tests/test_p1_plan_lote_217.py),
[`tests/test_p1_plan_lote_216.py`](../tests/test_p1_plan_lote_216.py); frontend `NeveraOpcional.contract.test.jsx`.

