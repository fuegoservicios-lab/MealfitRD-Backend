# Recordatorios de comida no registrada

[P1-PLAN-LOTE-72 · 2026-09-16] Contrato del cron `run_proactive_checks` ([`proactive_agent.py`](../proactive_agent.py)):
cuando el usuario no ha registrado una comida pasada su hora, el coach le escribe en el chat (y manda una notificación
push si tiene dispositivos suscritos). Test ancla: [`test_p1_plan_lote_72.py`](../tests/test_p1_plan_lote_72.py).

## Cuándo toca cada aviso

- El cron corre **una vez por hora, al minuto 30** (`app.py`, con jitter).
- Hora del aviso = hora habitual de la comida + retraso, en hora **local del usuario** (`user_tz_offset_min`).
  - Hora habitual: media circular de sus registros de los últimos 14 días (`db_facts.get_avg_meal_hour`), o la hora
    por defecto: desayuno 9:00, almuerzo 13:00, merienda 16:00, cena 19:30.
  - Retraso: 1,5 h (1 h si responde a más del 70 % de los avisos de esa comida, 2,5 h si a menos del 30 %).
- A las 23:00 locales, en vez de eso, el «Resumen del día» (solo si no registró nada).
- Tope anti-fatiga: `MEALFIT_PROACTIVE_MAX_NUDGES_PER_DAY` avisos por día local: **4**, uno por comida (decisión del
  dueño, 16-sep; era un 2 fijo, que dejaba sin recordatorio a la merienda y la cena). El Resumen de las 23:00 comparte
  el tope: a quien no registró nada y ya recibió los cuatro avisos no le llega.
- Anti-spam: no se escribe si el coach escribió en esa sesión hace menos de una hora.

## Lo que cambió (P1-PLAN-LOTE-72)

**Qué pasó.** El dueño recibió a las 10:30 el aviso del desayuno y lo contestó a las 12:58 registrándolo. El del
almuerzo, que tocaba a las 14:30, no llegó. Forense de solo lectura: un solo aviso en `nudge_outcomes` (Desayuno) y el
desayuno en `consumed_meals` con `consumed_at` a las 12:58 — la hora del REGISTRO.

**La cadena.**
1. `get_avg_meal_hour` promedia `consumed_at`: con ese único registro, «el dueño desayuna a las 12:58».
2. El aviso del desayuno pasó a 12:58 + 1,5 h = 14:28, la misma hora que el del almuerzo (13:00 + 1,5 h).
3. El bucle se quedaba con la PRIMERA comida que coincidía con la hora; al verla registrada, `continue` saltaba la hora
   entera sin mirar el almuerzo.
4. Al día siguiente el aviso del desayuno habría llegado a las 14:30, y a las 10:30 nada.

**Las tres piezas.**

| Pieza | Regla | Dónde |
|---|---|---|
| Franja de cada comida | Un registro fuera de la franja local de su comida no mueve la hora del aviso: desayuno 4:00-12:00, almuerzo 10:30-17:00, merienda 14:00-20:00, cena 17:00-3:00 (cruza la medianoche). Sin registros dentro, la hora por defecto. | `FRANJA_DE_COMIDA`, `get_avg_meal_hour(..., ventana=)` |
| Comida que falta | Se reúnen todas las comidas cuyo aviso toca y, con lo registrado delante, se avisa la primera que falte: la más reciente primero y, a igual hora, la más tardía del día. | `candidatas`, `_comida_ya_registrada` |
| Reintento | Un aviso toca desde su hora y durante `MEALFIT_PROACTIVE_NUDGE_RETRY_HOURS` horas (**3**; con 1, la conducta de antes), sin repetir una comida ya avisada hoy (`nudge_outcomes`) y sin cruzar la medianoche. Si no se puede leer qué se avisó hoy, solo la hora exacta. | `_horas_de_reintento`, `_comidas_avisadas_hoy` |

Antes, un aviso que no podía salir en su hora (el coach había respondido hace menos de una hora, la IA no contestó, un
despliegue a y media) se perdía para todo el día: el cron solo pasa una vez por cada hora.

**Lo que no cambia.** El anti-spam de una hora, el Resumen de las 23:00, el tono adaptativo
y el cruce de medianoche de las cenas tardías (`P3-AVG-MEAL-HOUR-CIRCULAR`). Sin `ventana`, `get_avg_meal_hour`
devuelve la media de siempre.

## El verbo de cada comida (P1-PLAN-LOTE-73)

El prompt solo nombraba la comida y el modelo tomó el verbo de otra: el dueño recibió «¿Ya cenaste tu merienda de la
tarde?». `PROACTIVE_PROMPT` recibe ahora `{verbo}` desde `VERBO_DE_COMIDA` (desayunaste, almorzaste, merendaste,
cenaste) y prohíbe el verbo de otra comida.

## El «¿ya cenaste?» de las 2:30 de la madrugada (`P1-PLAN-LOTE-83` · 2026-09-17)

Medido en producción (solo lectura): `nudge_outcomes` tiene el aviso de Cena a las 06:30 UTC del 17-sep (2:30 en RD) y el dueño lo vio a las 10:10 al abrir el chat, que no muestra la hora. La cadena: a las 00:38 anotó por el chat la «cena de ayer» y `consumed_at` quedó en las 00:38 del día 16 (la hora del registro, restada en días enteros); esa hora cae dentro de la franja de cena (17→3, holgada para cenas tardías), la hora media de cenar pasó a ser las 00:38 y el aviso (+1,5 h) cayó a las 2:08; el tick de las 2:30 buscó una cena «de hoy» (día 17) y no la había. El registro del almuerzo de ayer, a las 00:29, no hizo daño porque su franja (10:30→17) lo dejó fuera.

Dos cierres, los dos en el código real que decide el disparo:

- `get_avg_meal_hour` deja fuera todo registro anotado más de 18 h después de su `consumed_at` (un «ayer» resta días enteros): un registro de un día pasado no dice a qué hora se comió.
- Horas de silencio: ningún recordatorio de comida antes de las 6:00 locales (`MEALFIT_PROACTIVE_QUIET_UNTIL_HOUR`, 6; 0 = sin silencio). Es una decisión: quien cena de verdad a las 23:30 ya no recibe el aviso de la 1:00 que `P3-AVG-MEAL-HOUR-CIRCULAR` quiso conservar; un push a la 1:00 es peor que ninguno, y el Resumen de las 23:00 cubre ese día.

## Que el aviso LLEGUE a la pantalla (`P1-PLAN-LOTE-133` · 2026-09-20)

El dueño, con la captura del interruptor «Alertas Inteligentes · BETA»: «revisa a profundidad el sistema de notificaciones… quiero que le avise al usuario si no ha desayunado, almorzado, merendado… 100 % listo para producción, y quítale ese beta».

Medido en producción (solo lectura): el motor funciona —38 avisos en 5 días, a las 10:30 / 14:30 / 17:30 / 21:00— y `push_subscriptions` tiene UNA fila, cuyos envíos salen «exitosos» en el journal. La Web Push y sus llaves están bien: lo que fallaba era que casi nadie tenía CÓMO recibirlos. La app nativa de iOS es un WKWebView (sin Service Worker ni `PushManager`): ahí el interruptor daba «Tu navegador no soporta notificaciones Push».

| Canal | Dónde | Quién lo manda |
|---|---|---|
| Web Push | navegador de escritorio, Android, PWA de iOS instalada | el servidor (`utils_push.send_push_notification`), al generar el aviso del chat |
| Avisos locales | app nativa de iOS (`@capacitor/local-notifications`) | EL TELÉFONO: pide `GET /api/notifications/meal-reminders` y los programa para 7 días; salen con la app cerrada, sin APNs ni red, y el de hoy se cancela al registrar la comida (`mealfit:diary-changed`) |

- **Una sola cuenta**: `proactive_agent.hora_de_aviso` (hora habitual + espera, `% 24`) la usan el cron y el endpoint. El aviso local va a `floor(hora)`:35 — cinco minutos después del tick del cron (a y media), para que al tocarlo el mensaje del coach ya esté en el chat. Un aviso dentro de las horas de silencio no se programa.
- **Textos del aviso local y del aviso «fijo»**: `meal_reminders.py`, cortos (pantalla de bloqueo) y en los 5 idiomas.
- **El interruptor es un consentimiento explícito**: fuera el `send_push = False` que apagaba la pantalla para siempre a los 5 avisos «ignorados». La tasa de respuesta cambia el TONO y la espera, nada más.
- **«Respondió»** = contestó en el chat O registró una comida en la ventana tras el aviso (`_SQL_TASA_DE_RESPUESTA`); la ventana pasa de 60 min fijos a `MEALFIT_PROACTIVE_RESPONSE_WINDOW_MIN` (180). Cierra lo que aquí abajo figuraba como abierto desde el lote 72.
- **Suscriptor sin chat reciente**: la lista del cron salía solo de `agent_sessions` (3 días). Quien encendió las alertas y dejó de abrir el chat entra ahora con `id = None`: aviso corto y fijo a su pantalla, sin LLM y sin escribir en un chat viejo; queda en `nudge_outcomes` (`nudge_style = 'fijo'`), así que el tope diario y el «no repetir» siguen valiendo.
- **Etiqueta por comida** (`tag = comida-almuerzo`): la notificación nueva sustituye a la anterior en vez de apilarse; el Service Worker lleva la app ya abierta al chat en vez de abrir otra ventana.
- **Cerrar sesión** borra la suscripción de ESE navegador (y cancela los avisos locales): antes los recordatorios del usuario A le llegaban al B que entrara después.
- El cron tiene `id="proactive_meal_reminders"`.

Lo que el interruptor NO hace, a sabiendas: apagar el mensaje del coach en el chat. «Recibe avisos en tu pantalla» habla de la pantalla; el recordatorio dentro del chat es parte de la conversación.

## Abierto

- APNs (push remota en la app nativa) no está montada: haría falta la llave de Apple del dueño y el entitlement. Los avisos locales cubren el caso pedido sin ella. El plugin `@capacitor/push-notifications` sigue instalado y sin usar.
