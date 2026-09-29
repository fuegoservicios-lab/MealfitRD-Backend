# Permiso para la IA de terceros — `P1-PLAN-LOTE-843`

**SSOT:** [`backend/consentimientos.py`](../consentimientos.py) · router [`backend/routers/consents.py`](../routers/consents.py) ·
migración [`p1_plan_lote_843_user_consents_2026_09_29.sql`](../migrations/p1_plan_lote_843_user_consents_2026_09_29.sql) ·
tests `tests/test_p1_plan_lote_843*.py`.

## Por qué

- **Apple, App Review 5.1.2(i)** (texto de nov-2025): hay que decir a qué IA de terceros van los datos personales y obtener
  permiso explícito **antes** del primer envío. El revisor lo prueba sobre una instalación limpia.
- **RGPD art. 9(2)(a)**: consentimiento explícito para tratar datos de salud.
- **RGPD art. 49(1)(a)**: la transferencia al proveedor de IA en China (sin decisión de adecuación ni cláusulas firmadas) necesita un
  consentimiento aparte, informado de sus riesgos.
- Auditoría 2026-09-29, fila 4 y §A.1 (`docs/superpowers/specs/2026-09-29-legal-appstore-auditoria.md`).

**Vigente** = la cuenta aceptó la versión actual (`AI_CONSENT_VERSION = "ia-2026-10"`) con las **dos** claves de IA
(`ai_processing` y `ai_transfer_cn`) y no lo retiró después. `analytics` va aparte, es opcional y no cuenta para la IA.

## Contrato para el frontend (Task 844)

Todos los errores del permiso tienen el mismo cuerpo **plano** (sin `{"detail": {...}}` alrededor):

```json
{"error_code": "<código>", "version": "ia-2026-10", "detail": "<frase legible en español>"}
```

`detail` es una frase para la persona: un frontend viejo que no conoce el 428 la enseña en vez de romperse.

### La versión

`ia-2026-10`. Espejo en `frontend/src/consent/version.js`; un test de paridad las ata (se salta mientras ese fichero no
exista). Subir la versión vuelve a pedir el permiso a todos y el backend rechaza la vieja (409 al concederla, 428 al usar
la IA).

### `GET /api/consents` (cuenta; 401 sin sesión)

```json
{
  "version": "ia-2026-10",
  "vigente": false,
  "ai_consent_version": null,
  "ai_consent_at": null,
  "ai_cn_transfer_at": null,
  "ai_consent_revoked_at": null,
  "analytics": null
}
```

Eso es exactamente lo que devuelve para una cuenta **sin permiso** (nunca preguntada). Las fechas son ISO-8601 UTC;
`analytics` es `true`, `false` o `null` (no preguntado). Con el permiso dado: `vigente: true`, `ai_consent_version:
"ia-2026-10"` y las dos fechas. Retirado: `vigente: false` y `ai_consent_revoked_at` con fecha.

**Lo que la app ya carga al arrancar:** `GET /api/profile` devuelve el mismo objeto en `profile.ai_consent`, calculado de
la fila que ya lee (cero consultas extra). No hace falta una llamada nueva al arrancar.

### `POST /api/consents` (cuenta)

```json
{"version": "ia-2026-10", "ai_processing": true, "ai_transfer_cn": true, "analytics": false,
 "locale": "es-DO", "platform": "ios", "app_build": "106", "text_sha256": "<64 hex, opcional>"}
```

- Conceder la IA exige **las dos** claves a `true`. Una sola, o una a `false`: 422 `ai_consent_incomplete` (para quitar el
  permiso se usa `/withdraw`).
- `analytics` es opcional e independiente: `{"version": "ia-2026-10", "analytics": true}` sin claves de IA anota solo la
  analítica.
- `version` distinta de la vigente: **409** `ai_consent_version_outdated`.
- Nada que anotar: 422 `ai_consent_nothing_to_record`. Campo con forma rara (`platform` fuera de `ios|android|web`,
  `locale`, `app_build` > 64, `text_sha256` que no es hex de 64): 422 `ai_consent_invalid_field`.
- `text_sha256`: el SHA-256 (hex en minúsculas) del texto EXACTO que se mostró, calculado por el cliente.
- Respuesta 200: el objeto de `GET /api/consents` más `"plan_reanudado": true|false` (si la retirada había pausado el
  generador, conceder lo reanuda; ver abajo).

### `POST /api/consents/withdraw` (cuenta)

Cuerpo opcional `{"locale", "platform", "app_build"}`. Respuesta 200: el objeto de `GET /api/consents` (con
`ai_consent_revoked_at`) más `"plan_pausado": true|false`. Retirar dos veces no escribe filas nuevas.

### `POST /api/consents/guest` (invitado, sin sesión)

El mismo cuerpo que `POST /api/consents` más `"session_id"`: el del invitado (`mealfit_guest_session_id`, 8-128
caracteres `[A-Za-z0-9_-]`). Se guarda `sha256(session_id)`, nunca el id. Respuesta 200:

```json
{"ok": true, "version": "ia-2026-10", "header": "X-Bioboros-AI-Consent", "ai": true, "analytics": false}
```

Sin `session_id` válido: 422 `ai_consent_invalid_session`. Si el invitado rota su `session_id` («Probar sin cuenta» de
nuevo), hay que volver a registrar el permiso con el id nuevo: la adopción busca por el id que se le pase.

### El 428

Todo endpoint de la tabla de abajo marcado **428** responde, sin permiso vigente:

```
HTTP/1.1 428 Precondition Required
{"error_code": "ai_consent_required", "version": "ia-2026-10", "detail": "Para usar la IA necesitamos tu permiso ..."}
```

La app abre la hoja del permiso. Hoy la cuota (402) y los cupos (429) se comprueban antes que el permiso, pero ese orden
no es parte del contrato. Si la base no se puede leer: 503 `ai_consent_unavailable` (no es «sin permiso»: no abras la
hoja).

### La cabecera del invitado

`X-Bioboros-AI-Consent: ia-2026-10` en **cada** llamada a la IA. Para el invitado es obligatoria (sin ella, o con una
versión vieja, 428). Con sesión se ignora (manda la base), pero conviene mandarla siempre que haya permiso local: si la
sesión caducó, la petición llega sin identidad y con la cabecera recibe el 401 de siempre en vez de un 428. Está en
`allow_headers` del CORS (en nativo toda llamada es cross-origin).

### Adopción del plan del invitado

`POST /api/plans/adopt-guest-plan` acepta `"session_id"` (el del invitado). El permiso que el invitado registró pasa a
la cuenta **con su fecha original**, antes de guardar el plan, y solo si la cuenta no tiene una decisión propia más
reciente. La respuesta de la adopción añade `"ai_consent": {"adoptadas": n, "estado_actualizado": bool}`. Si la cuenta
ya tenía plan (409), no se adopta nada.

### Donde la IA es un efecto lateral (se atiende sin ella)

- `PATCH /api/profile` cambiando `locale`: el perfil se guarda igual (sigue exento de cuota); la traducción del plan se
  salta y la respuesta añade `"translation_skipped": "ai_consent_required"`.
- `POST /api/i18n/textos`: `{"textos": null}` (se pinta el original).
- `POST /api/plans/guest-display` (invitado sin cabecera): `{"meals": [], "plan_name": null, "insights": null,
  "skipped": "ai_consent_required"}`.

## Endpoints con permiso (tabla canónica)

`tests/test_p1_plan_lote_843_cobertura.py` la lee: cada fila **428** lleva `Depends(requiere_consentimiento_ia)`, cada
fila **suave** usa `hay_permiso_ia` o `permite_ia`, y todo endpoint con la dependencia está en la tabla (en las dos
direcciones, también contra las rutas reales de la app).

| Método | Ruta | Fichero | Función | Sin permiso |
|---|---|---|---|---|
| POST | /api/plans/analyze | routers/plans.py | api_analyze | 428 |
| POST | /api/plans/analyze/stream | routers/plans.py | api_analyze_stream | 428 |
| POST | /api/plans/generation-runs | routers/plans_generation.py | api_create_generation_run | 428 |
| POST | /api/plans/swap-meal | routers/plans.py | api_swap_meal | 428 |
| POST | /api/plans/{plan_id}/regenerate-day | routers/plans.py | api_regenerate_day | 428 |
| POST | /api/plans/{plan_id}/fix-sodium-day | routers/plans.py | api_fix_sodium_day | 428 |
| POST | /api/plans/recipe/expand | routers/plans.py | api_expand_recipe | 428 |
| POST | /api/plans/{plan_id}/retry-chunk/{chunk_id} | routers/plans.py | api_retry_chunk | 428 |
| POST | /api/plans/{plan_id}/chunks/{chunk_id}/regenerate-simplified | routers/plans.py | api_regenerate_dead_lettered_simplified | 428 |
| POST | /api/plans/{plan_id}/regen-degraded | routers/plans.py | api_regen_degraded_chunks | 428 |
| POST | /api/chat/stream | routers/chat.py | api_chat_stream | 428 |
| POST | /api/chat | routers/chat.py | api_chat | 428 |
| POST | /api/chat/message | routers/chat.py | api_save_chat_message | 428 |
| POST | /api/chat/voz | routers/chat.py | api_chat_voz | 428 |
| POST | /api/diary/upload | routers/diary.py | api_diary_upload | 428 |
| POST | /api/diary/consumed/estimate-macros | routers/diary.py | api_estimate_macros | 428 |
| POST | /api/diary/consumed/estimate-plate | routers/diary.py | api_estimate_plate | 428 |
| POST | /api/diary/scan/ajuste-duda | routers/diary.py | api_ajuste_de_duda | 428 |
| POST | /api/diary/scan/ingrediente | routers/diary.py | api_ingrediente_corregido | 428 |
| POST | /api/inventory/photo-scan | routers/user_data.py | api_inventory_photo_scan | 428 |
| POST | /api/help/chat | routers/help_chat.py | api_help_chat | 428 |
| POST | /api/i18n/textos | routers/user_data.py | api_traducir_textos | suave |
| POST | /api/plans/guest-display | routers/plans.py | api_guest_display | suave |
| PATCH | /api/profile | routers/user_data.py | api_patch_profile | suave |

Notas: `/api/chat/message` no llama a la IA por sí mismo, pero un mensaje de rol `user` clasifica la respuesta a un aviso
pendiente (`handle_nudge_response` → IA); no tiene llamadores en el frontend. `/retry-chunk`, `/regenerate-simplified` y
`/regen-degraded` solo reencolan (la recogida ya los frenaría), pero así la persona ve la hoja en vez de un plan que no
avanza. `/swap-meal/persist` y los paneles de Configuración no son endpoints de IA: su efecto de IA (traducir, extraer
hechos) se frena dentro, en segundo plano.

## Segundo plano (sin petición delante)

Con `block`, nada de esto llama a un proveedor para una cuenta sin permiso vigente:

| Dónde | Qué manda al proveedor | Gate |
|---|---|---|
| Recogida del chunk worker (`cron_tasks.process_plan_chunk_queue`, las dos ramas) | el plan entero (IA de texto) | `fragmento_sql_permiso("q1.user_id")` en el WHERE, junto a la pausa de P1-PLAN-MODE y el congelado |
| Worker del outbox `plan_jobs` (traducción) | el plan (IA de traducción) | el claim salta los `display_i18n` sin permiso (se quedan en cola, sin reintentos) + `enrich_plan_display` devuelve `skipped: ai_consent` |
| `plan_display_i18n.enrich_plan_display` (todos los disparadores) | el plan | `permite_ia(user_id, "traduccion_del_plan")` |
| Coach proactivo (`proactive_agent.run_proactive_checks`) | dieta, objetivo, lo del día (IA) y el resumen de conducta (Cohere) | `permite_ia(user_id, "coach_proactivo")`: sin permiso, el aviso FIJO de su idioma, sin embedding ni IA |
| Respuesta a un aviso (`proactive_agent.handle_nudge_response`) | el mensaje | `permite_ia` antes de clasificar |
| Dreaming (`dreaming.consolidate_user`) | los hechos (IA) y el modelo sintetizado (Cohere) | `permite_ia` → `skipped_ai_consent` |
| Extractor de hechos (`fact_extractor.async_extract_and_save_facts`, la cola `process_pending_queue_sync`, el webhook) | el mensaje (IA) y los hechos (Cohere) | `permite_ia`; la cola se conserva como con la memoria pausada |
| Retrospectiva semanal (`cron_tasks._persist_nightly_learning_signals`, también desde el registro manual de comidas) | lo comido y lo que gustó | `permite_ia` antes de la IA |
| Aprendizaje del bloque (`cron_tasks._check_chunk_learning_ready`, también desde el cron de recuperación de la Nevera) | nombres de lo que anotó (Cohere) | `usar_embeddings=permite_ia(...)` |
| Título del plan (`services._titulo_del_plan`: guardado, parcial y diferido) | objetivo, calorías y platos | `permite_ia` → el título determinista |
| JIT de semana 2 (`proactive_agent._trigger_week2_background_generation`, código muerto) | el perfil | `permite_ia` por si alguien lo revive |

Y dos crons que no llaman a la IA pero hablaban de esa cola: el escalado de bloques atascados
(`_detect_and_escalate_stuck_chunks`, que mandaba «Optimizando tu plan… estará listo en breve») y la alerta de zombies
(`_alert_stuck_chunks`) saltan los bloques que esperan el permiso, con el mismo fragmento SQL.

## Knob `MEALFIT_AI_CONSENT_GATE`

| Valor | Peticiones | Segundo plano |
|---|---|---|
| `block` (default) | 428 sin permiso vigente | nada sin permiso vigente (SQL + `permite_ia`) |
| `log` | la falta de permiso solo se anota (warning); **una retirada explícita se respeta igual** (428) | igual: solo frena la retirada explícita |
| `off` | nada | nada |

`log` es el modo del despliegue gradual: lo que decide es que a la persona aún no se le preguntó. Una retirada es una
decisión suya y se respeta también ahí (la retirada, además, pausa la cola vía plan_mode en cualquier modo). `off` es el
interruptor de emergencia. La suite de tests corre con `off` (`tests/conftest.py`); el default de código es `block`.

## Retirar y volver a conceder

1. **La bandera primero**: `ai_consent_revoked_at = now()` y dos filas `granted=false` en la MISMA transacción. Desde ese
   COMMIT la recogida y los crons ya dicen que no.
2. **Después la cola**: `plan_mode.pause_plan_generation` (el patrón de P1-PLAN-MODE): `plan_mode='tracking'`, cancela la
   cola con su firma, suelta los locks y sella el plan `paused_by_user` (silencia los crons del plan). No borra datos:
   borrar sigue siendo «Eliminar cuenta».
3. Un bloque que YA estaba dentro del LLM termina esa generación (como en la pausa del modo plan); la validación previa al
   LLM (`_validate_chunk_pre_llm`) ve la fila cancelada y aborta los que aún no empezaron.
4. **Volver a conceder** reanuda (`resume_plan_generation`, que revive la cola firmada) solo si la pausa la puso la
   retirada: `plan_mode='tracking'` con `plan_mode_changed_at` dentro de los 5 minutos siguientes a la retirada. Quien ya
   estaba en modo contador sigue en modo contador.

## Datos

- `public.user_consents`: solo inserción (solo `consentimientos.py` escribe; un test vigila que nadie haga `UPDATE` ni
  `DELETE` sobre ella). Titular: `user_id` (FK a `user_profiles`, `ON DELETE CASCADE`) o `guest_hash`, exactamente uno.
  Índices `(user_id, consent_key, created_at DESC)` y `(guest_hash, created_at DESC)`. `REVOKE ALL FROM PUBLIC`.
- `user_profiles`: `ai_consent_version`, `ai_consent_at`, `ai_cn_transfer_at`, `ai_consent_revoked_at`,
  `analytics_consent` (NULL = no preguntado). Se escriben en la misma transacción que las filas. No están en
  `_PROFILE_SCALAR_WHITELIST`: `PATCH /api/profile` no puede tocarlas.
- **Exportación** (`GET /api/account/export`): `user_consents` de la cuenta, sin `guest_hash`; las columnas nuevas salen
  con `user_profiles`.
- **Borrado de cuenta**: el CASCADE del perfil se lleva las filas. La purga administrativa que conserva la cuenta
  (`include_profile=False`) no las toca: son la prueba de que el tratamiento de esa cuenta viva tiene base legal.
- Las filas de invitado no tienen plazo de conservación propio (no llevan datos personales más allá del hash de una
  sesión anónima); pendiente de decidir con el resto de la retención de invitados.

## Despliegue

1. Aplicar la migración con el libro (`scripts/apply_migration.py --apply`), ANTES del backend: el código lee las
   columnas nuevas en cada petición de IA y en la recogida.
2. Con `block` (default), toda cuenta existente queda sin IA hasta que acepte en la hoja (Task 844): sus bloques
   pendientes esperan en la cola (no se cancelan) y el coach proactivo manda el aviso fijo. El primer bloque tras aceptar
   sale con el retraso acumulado (`chunk_lag_excessive` puede dispararse una vez).
3. Sin el frontend de la Task 844, un invitado no puede usar la IA (no manda la cabecera) y una cuenta ve el `detail` del
   428. Para un despliegue escalonado: `MEALFIT_AI_CONSENT_GATE=log` hasta que el frontend esté fuera.
