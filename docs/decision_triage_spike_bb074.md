# Spike BB-074 — Gemini Decision 1: el API existe y este es su contrato (2026-11-11)

Entregable del primer ticket del plan (en el plan figuraba como BB-045; BB-045..073 ya estaban usados).
Implementación: [`decision_triage.md`](decision_triage.md).

## 1. ¿Existe?

Sí. Lanzado el 11-nov-2026: blog de Google (blog.google → «Gemini Decisions API») y referencia oficial en
`aistudio.google.com/docs/decisions-api` (la URL `ai.google.dev/gemini-api/docs/decisions` redirige ahí).

## 2. Contrato verificado

| Pieza | Valor |
|---|---|
| Endpoint | `POST https://generativelanguage.googleapis.com/v1beta/decisions` |
| Auth | cabecera `x-goog-api-key` |
| `model` | `gemini-decision-1` (obligatorio) |
| `input` | string, bloque o lista de bloques `{type: text\|image\|audio\|video\|document, text\|data(base64)\|uri, mime_type, resolution}` |
| `questions` | mapa clave → `{type, instructions, criteria}`; 1 a 1.000; clave `[A-Za-z0-9_-]{1,64}` |
| Sí/no | `type: "noul"` (sic); criteria opcional `{true_description, false_description}`; respuesta `noul` = p(sí) 0,01–0,99 |
| Elección | `type: "choice"`; criteria = mapa opción → descripción (2–255); **orden alfabético**; respuesta `choice` + `probabilities` |
| Rúbrica | `type: "score"`; criteria = lista ordenada; respuesta `score` (media ponderada), `legend`, `probabilities` |
| Uso | `usage.total_input_tokens`, `total_output_tokens` (siempre 0) |
| Límites | 262.144 tokens de contexto; media inline 100 MB (PDF 50 MB) |
| Errores | HTTP 400 (límites/validación/seguridad: `error.code` ∈ safety, prohibited_content, blocklist, image_safety, content_blocked) |
| Precio | USD 0,10 / 1M tokens de entrada; salida gratis |

Diferencias con el RFC: el sí/no no se llama `yes/no` sino `noul`; la rúbrica admite hasta 255 niveles (no 8);
el campo de contexto es `input`, no `contents`.

## 3. Medido en el VPS (Oracle, 11-nov, clave de producción)

| Llamada | Latencia | Resultado |
|---|---|---|
| «me tomé un jugo de chinola» + comida activa (frío, TLS nuevo) | 590 ms | bebida ambigua 0,972 · fusión 0,951 · intención `append_active_meal` 0,91 |
| la misma, conexión caliente (×2) | 253 / 257 ms | idéntico |
| «me tomé una malta», sin comida activa (`iniciar_turno`, fila en `pipeline_metrics`) | 417 ms | bebida ambigua 0,976 · intención `new_meal_log` 0,908 |

Conclusión: el contrato coincide con lo implementado; el tope de 0,4 s del plan es corto para la llamada en frío,
así que en el VPS va a 1,5 s (`MEALFIT_DECISION_TRIAGE_TIMEOUT_S`).
