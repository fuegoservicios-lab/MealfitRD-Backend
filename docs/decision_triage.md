# Triaje con Gemini Decision 1 (P1-DECISION-TRIAGE · P1-PLAN-LOTE-981 · 2026-11-11)

Tickets BB-074..BB-079. Módulo [`decision_triage.py`](../decision_triage.py); test
[`test_p1_plan_lote_981_decision_triage.py`](../tests/test_p1_plan_lote_981_decision_triage.py).

## Qué es

`gemini-decision-1` (Google, lanzado el 11-nov-2026) no genera texto: devuelve probabilidades calibradas para
preguntas cerradas sobre un contexto multimodal. Coste: USD 0,10 por 1M tokens de entrada; salida gratis.
Latencia anunciada: 150–250 ms.

## Contrato del API (verificado en la doc oficial, aistudio.google.com/docs/decisions-api)

- `POST https://generativelanguage.googleapis.com/v1beta/decisions`, cabecera `x-goog-api-key`.
- Body: `model`, `input` (string, bloque o lista de bloques `{type: text|image|audio|video|document, text|data|uri,
  mime_type, resolution}`), `questions` (mapa clave → `{type, instructions, criteria}`; 1 a 1.000).
- Tipos: **`noul`** (así se llama el sí/no; criteria opcional `{true_description, false_description}`), `choice`
  (criteria = mapa opción → descripción; **las opciones se ordenan alfabéticamente**: leer siempre por clave),
  `score` (criteria = lista ordenada de niveles).
- Respuesta: `answers[clave]` → `noul` (p de sí, 0,01–0,99) / `choice` + `probabilities` / `score` + `legend`;
  `confidence`; `usage.total_input_tokens` (salida siempre 0).
- Errores: HTTP 400 (límites, validación o filtros de seguridad con `error.code`).

## Qué hace y qué no

| Superficie | Modo `shadow` | Modo `guard` |
|---|---|---|
| Chat: `iniciar_turno` en `routers/chat.py` (los dos endpoints), en paralelo al agente | registra `decision_triage_shadow` | igual |
| Chat: freno en `agent.execute_tools` antes de `log_consumed_meal` | registra `decision_triage_tool`, no frena | frena **una vez** si `beverage_ambiguity ≥ 0,80` |
| Foto: `/api/diary/upload` | corre en paralelo a la visión y registra `decision_triage_media` | va antes (tope 0,4 s); si `p(comida) ≤ 0,15` no llama a la visión y devuelve el «no es comida» de siempre |

`routing_intent` y `ttl_fusion_valid` **solo se registran**. Ninguna rama escribe en la base por una probabilidad.

El freno NO aplica: con más de un `log_consumed_meal` en el turno (frenaría el plato junto al jugo), en turnos con
foto de plato, en repeticiones confirmadas, ni si en los últimos 10 mensajes ya hubo un freno (tope por comida).
El re-análisis de una foto con `aclaracion` no consulta el triaje.

Preguntas eliminadas del RFC a propósito: `criollo_fat_level` (la grasa criolla ya está calibrada en código; una
segunda calibración contaría doble) y `ready_to_commit_db` (depende de gramos, que este modelo no extrae).

## Fail-open

`None` ⇒ conducta actual ante: knob `off`, sin key de Gemini (`llm_provider._google_api_key`), usuario de la UE en
el chat (`region_ia`, por coherencia con el lote 937), timeout, HTTP ≠ 200, JSON sin las preguntas pedidas.
El chat espera el resultado como mucho `MEALFIT_DECISION_TRIAGE_CHAT_WAIT_S` y solo al llegar a la tool.

El turno viaja por un `ContextVar` (igual que `region_ia`). En `/stream` llega al agente porque
`turno_desacoplado.desacoplar` arranca el hilo con `inherit_context`; con `MEALFIT_CHAT_TURN_DETACHED=0` el generador
corre en otro contexto, el turno no se ve y el freno no actúa (fail-open, sin error).

## Knobs

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_DECISION_TRIAGE_MODE` | `off` | `off` / `shadow` / `guard` |
| `MEALFIT_DECISION_TRIAGE_MODEL` | `gemini-decision-1` | id del modelo |
| `MEALFIT_DECISION_TRIAGE_TIMEOUT_S` | `0.4` | tope del HTTP, clamp [0,05; 5] |
| `MEALFIT_DECISION_TRIAGE_CHAT_WAIT_S` | `0.3` | cuánto espera el chat al resultado en la tool |
| `MEALFIT_DECISION_BEVERAGE_BLOCK` | `0.80` | umbral del freno de bebida, clamp [0,5; 0,99] |
| `MEALFIT_DECISION_MEDIA_REJECT` | `0.15` | umbral de foto «no es comida», clamp [0,01; 0,5] |

Gasto en `llm_usage_events` (`node="decision_triage"`), nunca en `api_usage`.

## Promoción shadow → guard (decisión del dueño)

Con ≥ 500 turnos `decision_triage_shadow` (≥ 100 con bebida): revisar a mano 100, unidos por `turn_id` con
`decision_triage_tool`. Activar `guard` solo si la precisión con 0,80 es ≥ 95 %, la latencia p95 ≤ 400 ms y la tasa
de fallo < 1 %. Para la foto: comparar `p_food` con `vision_is_food` en `decision_triage_media`.
