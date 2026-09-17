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
5. **Un solo cambio en pantalla.** Sin nada local que enseñar (sesión abierta por la regla y ninguna lista en caché,
   que es justo lo que deja el logout) se ve «Cargando mensajes…» desde el principio, sin pedir el historial vacío de
   esa sesión, y luego la conversación o el saludo. Antes se veía el saludo, a los ~300 ms «Cargando…» y después la
   conversación: el parpadeo doble que el dueño ya había reportado. La espera termina siempre: al llegar la lista, si
   la lista falla o si el usuario elige un chat a mano. En la visita normal del día (lista en caché) el saludo sale al
   instante, como siempre.

**Lo que el frontend lee del backend** (`db_chat._process_and_sort_sessions`):

| Campo | Significado para la regla |
|---|---|
| `last_activity` | `created_at` del **último** mensaje (texto de Postgres en GMT, `2026-09-16 16:58:13.337404+00`); sin mensajes, el `created_at` de la sesión. Chromium, WebKit y Firefox lo leen igual (medido). |
| `title_key: 'empty'` | La sesión no tiene mensajes del usuario ni título: no cuenta como «tu chat de hoy». |

**Lo que no cambia.** La lista es solo del dueño del token (IDOR, `routers/chat.py`), así que la adopción no puede abrir
un chat ajeno; y el logout sigue borrando la sesión guardada. Una sesión de invitado no se adopta (su lista sale de
los ids del propio navegador).

## Se renueva solo, también con la pestaña abierta (P1-PLAN-LOTE-73)

**Qué pidió el dueño.** «Que no se tenga que dar a nuevo chat ni siquiera, que lo haga automático diario, y que lo diga
una cuenta regresiva donde dice nuevo chat». Al entrar ya era automático; faltaba el Agente que se queda abierto de un
día para otro.

**La regla** (`debeRenovarse`, comprobada al volver a la pestaña y una vez por minuto):

1. La sesión abierta no se ha usado ni elegido HOY: su día anotado es anterior. Si escribes pasada la medianoche, o
   abres a mano un chat viejo, el día anotado ya es hoy y no se toca.
2. Su conversación es de un día anterior por su último mensaje real, y ese mensaje tiene 15 minutos o más.
3. No hay turno en curso ni borrador, y si lo dispara el reloj (no la vuelta a la pestaña), no se ha tocado la página
   en 5 minutos.

La sesión nueva es automática, así que si otro dispositivo ya abrió el chat de hoy, la adopción del lote 71 la cambia
por ese al llegar la lista.

**Causa raíz de paso.** El día de actividad se anotaba como HOY cada vez que cambiaban los mensajes, y eso incluye
HIDRATAR: abrir pasada la medianoche un chat de ayer lo convertía en «el de hoy» y al volver a entrar resucitaba. Ahora
se anota el día del último mensaje real, y nunca retrocede (`diaDeActividad`), para que elegir a mano un chat viejo
siga contando como el de hoy.

**Cuenta regresiva.** Bajo «Nuevo chat»: «Nuevo chat automático en 6 h 21 min» (en la última hora, solo minutos; en el
último minuto, «en menos de un minuto»), por minutos y sin segundos, en su propio componente
(`CuentaRegresivaChat`) para no volver a pintar la página. El bloque conserva sus 84 px
(0.75rem + 2.75rem + 0.25rem + 1rem + 0.5rem), porque la barra de scroll del hilo arranca a esa altura
(`P2-CHAT-SCROLLBAR-TWINS`). Textos en los cuatro catálogos.

## El botón «Nuevo chat» queda bloqueado (P1-PLAN-LOTE-76)

Decisión del dueño («prefiero un bloqueo total hasta medianoche»): mientras el chat abierto es el de HOY (día anotado
= hoy, sea automático o elegido a mano), el botón va deshabilitado con el mismo tooltip de la cuenta regresiva, y
`handleNewChat` lo ignora aunque llegue por otro camino. Se habilita solo en el estado degenerado en que el día
anotado no es hoy (renovación imposible o almacenamiento borrado): es la salida de emergencia para no quedarse sin
chat. La regla vive en `nuevoChatBloqueado` (`utils/chatSessionDay.js`); la cabecera conserva sus 84 px.
