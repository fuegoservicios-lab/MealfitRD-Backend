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
- Tope anti-fatiga: `MEALFIT_PROACTIVE_MAX_NUDGES_PER_DAY` avisos por día local (**2** por defecto). Con cuatro comidas y
  el resumen, dos avisos dejan sin recordatorio a la merienda y la cena: subirlo es decisión del dueño.
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

**Lo que no cambia.** El tope de 2 avisos al día, el anti-spam de una hora, el Resumen de las 23:00, el tono adaptativo
y el cruce de medianoche de las cenas tardías (`P3-AVG-MEAL-HOUR-CIRCULAR`). Sin `ventana`, `get_avg_meal_hour`
devuelve la media de siempre.

## Abierto

- **Tope diario** (decisión del dueño): con 2, un día sin registrar nada recibe el aviso del desayuno y el del almuerzo, y
  ninguno más. Se puede subir sin desplegar con el knob.
- **Respuesta tardía**: `handle_nudge_response` solo enlaza la respuesta al aviso si llega en los 60 minutos siguientes.
  El dueño contestó el del desayuno 2 h 27 min después y el aviso quedó como «no respondido», y la tasa de respuesta
  decide el tono y si se manda push.
