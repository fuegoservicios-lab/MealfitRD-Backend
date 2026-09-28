# Chat: la foto del plato, su registro y los avisos (lotes 690-695)

Auditoría del 28-sep («¿el agente del chat está al 100%?») sobre 14 días de conversaciones reales y 10 días de logs del
VPS: 39/39 turnos con HTTP 200, 0 errores del chat, p50 10 s / p90 17 s (texto) y 26 s (foto). Los fallos estaban en el
flujo de la foto y en el tono de los avisos.

## La foto con dudas (690, 694, 695)

| Paso | Dónde | Qué hace |
|---|---|---|
| Subida | `AgentPage.handleSend` | `/api/diary/upload` devuelve la descripción y hasta 2 `dudas` con opciones (cada una con su `ajuste` de macros). |
| Pausa | `hayQueEsperarRespuestas` (`utils/fotoAntesDelCoach.js`) | Con dudas, el turno TERMINA sin abrir `/api/chat/stream`: la burbuja queda `_esperaDudas` y sale la tarjeta «Antes de anotarlo, dime:». Knob `VITE_CHAT_DUDAS_ANTES` (true). |
| Persistencia | `guardarFotoPendiente` / `conFotoPendiente` | La foto pendiente se guarda por chat (12 h) y vuelve al rehidratar el historial: salir a la Nevera o cerrar la app no la pierde. |
| Respuesta | `_reanudarFoto` | UN envío: mismo `clientMessageId`, fotos ya subidas, texto «Mi cena\n2 huevos · Maduro», `vision.respuestas` + `vision.ajuste`. Lo tecleado (también tras «Otra…») es la respuesta y se junta con lo ya tocado. «Omitir preguntas» manda lo supuesto. |
| Contexto | `respuestas_de_la_foto.preparar_vision` | Quita «DUDAS (pregúntale solo esto)» (si no, la regla del 305 vuelve a preguntar) y aplica el ajuste a la «(Estimación…)» con UNA foto de plato. |
| Rótulo | `rotulo_de_comida` / `marcar_rotulo` | «Mi cena», «el almuerzo de hoy» = rótulo de la foto: regístralo, no propongas otro plato. «¿Esto sirve para la cena?», «para la cena», «voy a cenar esto» no lo son. |
| Red de seguridad | `agent.route_tools` → `nudge_photo_to_log` | Turno de anotar (respuestas o rótulo) que acaba sin `log_consumed_meal`/`correct_consumed_meal` ⇒ el grafo devuelve el turno al modelo UNA vez (`photo_log_retried`). Si el usuario dijo que aún no se lo come, no registra. |

Caso que lo originó (28-sep 00:42 UTC): «Mi cena» + foto → el coach propuso OTRA cena; «2 huevos · Maduro» llegó en un
segundo turno sin la foto → «¿Ya te los comiste?»; al final se anotaron 300 kcal en vez de ~570 (se perdió el salami).

## «Bioboros te respondió» (692)

`aviso_respuesta_chat.avisar_respuesta` en el `done` de `/api/chat/stream`: push en CADA respuesta con `solo_si_no_mira`
(el service worker la calla con una ventana visible; iOS no pinta pushes en primer plano —`presentationOptions: []`— y
Android las descarta con esa marca). Knob `MEALFIT_CHAT_REPLY_PUSH` (true). La respuesta ya se guardaba al salir de la app
(0 streams abortados por el cliente en 7 días); si algún día aparecen, el siguiente paso es desacoplar la generación de
la conexión (hilo productor + `/api/chat/stop` para «Detener»).

## Tono de los avisos de comida (693)

`tono_del_aviso.py`: «se salta esta comida» = días cerrados sin registro de esa comida (fecha local; tipo o nombre, la
misma regla que `_comida_ya_registrada`) ≥ 70 % con ≥ 5 días. Antes era la tasa de RESPUESTA al aviso, y el tono pedía
«pregúntale qué está fallando», contra la regla del 413 («nada de interrogatorios»). Replay en producción (0 IA): 10 casos
→ 1 (la merienda del dueño, 0 de 11 días). Ese tono invita, no pregunta, y menciona que el recordatorio de esa comida se
apaga en Configuración → Recordatorios de comida.

## Atajos (691)

«Escanear mi plato» y «Anotar comida» siempre en el teléfono con el chat quieto; las preguntas del momento solo con el
hilo corto (≤ 4 mensajes; el chat del día nace con los avisos del coach).

## Pruebas

Backend: `test_p1_plan_lote_690.py`, `_692`, `_693`, `_694`. Frontend: `lote690.test.jsx`, `lote691.test.js`,
`lote695.test.jsx`. Batería real en seco (`scripts/coach_battery`, casos P1-P5): rótulo, dudas contestadas, pregunta,
«todavía no me lo como», respuesta escrita.
