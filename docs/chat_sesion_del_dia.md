# Qué chat abre el Agente

[P1-AGENT-SESSION-DAY · 2026-08-14 · P1-PLAN-LOTE-71 · 2026-09-16] Al entrar en el Agente, el usuario debe ver **su
chat de hoy** si ya existe, y uno **nuevo** si no. La regla vive en el frontend
(`frontend/src/utils/chatSessionDay.js`); este documento fija el contrato con el backend. Tests:
`frontend/src/__tests__/P1_agent_session_day.test.js` (la regla),
`frontend/src/__tests__/Agent.chat_del_dia_tras_login.test.jsx` (la página montada) y
[`test_p1_plan_lote_71.py`](../tests/test_p1_plan_lote_71.py) (el contrato entre repos).

## La frontera es el día (P1-AGENT-SESSION-DAY)

Dos peticiones del dueño que parecían opuestas: en mayo, «se refresca y molesta» (ir a la Nevera y volver le costaba la
conversación en curso); en agosto, que el Agente le abriera un chat nuevo (le resucitaba uno de trece días antes). La
persistencia no caducaba nunca. La frontera es el **día local**: dentro del día se conserva el hilo y cada día empieza
fresco. Día local, no UTC: en RD el corte en UTC caería a las 20:00, en plena cena.

## El chat del día vive en el servidor (P1-PLAN-LOTE-71)

**Qué pasó.** El dueño cerró sesión, volvió a entrar y el Agente le abrió un chat en blanco mientras «Recientes · Hoy»
listaba su conversación de esa mañana. En producción: una sola sesión, creada a las 08:26 RD, con 6 mensajes y el
último a las 12:58. El chat en blanco no existía en la base: nace al enviar el primer mensaje.

**Por qué.** La regla solo miraba `mealfit_current_session` en localStorage, y el logout la **borra a propósito**
(`P2-CHAT-CACHE-XUSER`: en un dispositivo compartido el siguiente usuario no puede heredarla). Pasaría igual en un
teléfono nuevo o tras limpiar el navegador.

**La regla ahora.**

1. La sesión que abre la regla del día (y no el usuario) queda marcada en `mealfit_current_session_auto`.
2. Con la **primera** lista del servidor de cada montaje (`GET /api/chat/sessions/{user_id}`, solo usuarios con
   sesión), esa sesión se cambia por **tu chat de hoy**: la sesión con mensajes cuya última actividad cae hoy (día
   local); de varias, la más reciente.
3. Nunca se cambia una sesión **elegida o usada**: `marcarActividad` (actividad real, «Nuevo chat», «Recientes»,
   borrar el chat abierto) quita la marca. Tampoco con un turno en vuelo, mensajes en pantalla o un borrador escrito.
4. Si el servidor ya conoce la sesión abierta con mensajes, se queda. Sin chat de hoy, se queda la nueva: el de ayer no
   resucita.
5. Mientras llega el historial se ve «Cargando mensajes…», no el saludo de un chat que ya no es el del usuario.

**Lo que el frontend lee del backend** (`db_chat._process_and_sort_sessions`):

| Campo | Significado para la regla |
|---|---|
| `last_activity` | `created_at` del **último** mensaje (texto de Postgres en GMT, `2026-09-16 16:58:13.337404+00`); sin mensajes, el `created_at` de la sesión. Chromium, WebKit y Firefox lo leen igual (medido). |
| `title_key: 'empty'` | La sesión no tiene mensajes del usuario ni título: no cuenta como «tu chat de hoy». |

**Lo que no cambia.** La lista es solo del dueño del token (IDOR, `routers/chat.py`), así que la adopción no puede abrir
un chat ajeno; y el logout sigue borrando la sesión guardada. Una sesión de invitado no se adopta (su lista sale de
los ids del propio navegador).
