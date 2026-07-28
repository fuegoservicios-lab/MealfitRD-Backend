# Escáner de comida: gemma local → Luna cloud (P1-VISION-LUNA · 2026-07-28)

Lee esto si "Escanear comida" (Dashboard) o el Diario Visual se comportan
raro, si un scan tarda 2 minutos en vez de 2 segundos, o si `llm_usage_events`
muestra un costo que no esperabas. SSOT del código: [`vision_agent.py`](../vision_agent.py)
(despacho + cascada), [`image_prep.py`](../image_prep.py) (resize),
[`routers/diary.py`](../routers/diary.py) (endpoint + contabilidad de costo).

## Qué cambió y por qué

Pre-fix, `vision_agent.py` enviaba la foto **cruda** al proveedor de visión
configurado y `/api/diary/upload` acepta hasta 20 MB — nunca había un cap de
resolución entre el móvil del usuario y el API. Con `gemma` local (Ollama,
costo cero) eso solo costaba latencia (30-120 s/scan). Encender un proveedor
cloud pago (`gpt-5.6-luna` vía API OpenAI-compatible) sobre esa misma ruta
hubiera facturado la foto completa cada vez.

Medido contra el API real el 2026-07-28 (no estimado):

| Resolución enviada | tokens entrada | USD/scan | 3 scans/día |
|---|---|---|---|
| 3024×4032 (lo que la app mandaba antes de este fix) | 12.406 | $0,0130 | $1,17/mes |
| 1024×1024 (default post-fix, `MEALFIT_VISION_MAX_SIDE_PX`) | 1.374 | $0,0020 | $0,18/mes |
| 512×512 | 453 | $0,0010 | $0,09/mes |

Pricing Luna: **$1,00/M tokens de entrada, $6,00/M de salida** (tabla
`_DEFAULT_LLM_PRICING_MICROS_PER_M` en [`db_profiles.py`](../db_profiles.py),
entrada `gpt-5.6-luna`, ya registrada por `P1-LUNA-PRICING · 2026-07-26` —
este fix no tocó el pricing, solo lo empieza a usar desde un surface nuevo).
Latencia medida: **1,7-2,7 s** (gemma local por túnel SSH: 30-120 s).

El fix son tres cambios en cadena, en orden de dependencia:

1. **Resize defensivo** ([`image_prep.py`](../image_prep.py)) — cap de lado
   mayor a `MEALFIT_VISION_MAX_SIDE_PX` px, re-encode JPEG calidad 85,
   fail-open ante cualquier error. Se invoca en el ÚNICO punto por el que
   pasan los bytes antes de salir a cualquier proveedor
   (`process_image_with_vision`, `vision_agent.py`) — cubre `ollama` Y
   `openai_compatible` con el mismo chokepoint, así que gemma también se
   beneficia (parte de sus 30-120 s es puro tamaño de imagen).
2. **Cascada de fallback** — `MEALFIT_VISION_FALLBACK_PROVIDER`: si el
   primario falla o devuelve un resultado no usable, reintenta UNA sola vez
   con el fallback configurado. Sin bucles: primario → fallback → si el
   fallback TAMBIÉN falla, se devuelve el resultado del PRIMARIO (no el del
   fallback), para que el log apunte al problema real.
3. **El gasto sale del libro de cuota de planes** ([`routers/diary.py`](../routers/diary.py))
   — ver sección dedicada abajo.

## Knobs

Todos vía `knobs._env_*` (auto-registrados en `_KNOBS_REGISTRY`,
`get_knobs_registry_snapshot()` los expone). Defaults confirmados leyendo el
source el 2026-07-28 — no asumidos del plan.

| Knob | Default | Clamp / choices | Dónde |
|---|---|---|---|
| `MEALFIT_VISION_PROVIDER` | `disabled` | `{disabled, off, openai_compatible, ollama}` — fuera del set cae al default con WARNING | `vision_agent._vision_provider` |
| `MEALFIT_VISION_MODEL` | `""` (vacío) | sin choices; `_env_str` normaliza a lowercase | `vision_agent._vision_model_name`. **Nota**: el path `ollama` usa un accessor DISTINTO (`_ollama_model_name`, lee el env var crudo `MEALFIT_VISION_MODEL`, default `gemma4:12b`, sin pasar por `_env_str`) — mismo nombre de env var, dos lecturas distintas según provider |
| `MEALFIT_VISION_BASE_URL` | `""` (vacío) | sin choices | `vision_agent._vision_base_url` (solo usado por `openai_compatible`) |
| `MEALFIT_VISION_MAX_SIDE_PX` | `1024` | `[256, 4096]` — fuera de rango cae al **default**, NO clampa al borde (así trata `_env_int` un `validator` que retorna False) | `image_prep.vision_max_side_px` |
| `MEALFIT_VISION_FALLBACK_PROVIDER` | `""` (vacío = sin cascada) | mismo choices-set que `MEALFIT_VISION_PROVIDER`; `disabled`/`off`/igual-al-primario también se tratan como "sin cascada" | `vision_agent._vision_fallback_provider` + `_vision_cascade_target` |
| `MEALFIT_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | env crudo, sin `_env_str` | `vision_agent._ollama_base_url` (mismo default que el escáner de Nevera en `routers/user_data.py`) |
| `MEALFIT_VISION_TIMEOUT_S` | `240` | `[30, 600]` — **este SÍ clampa al borde** (`min(600, max(30, v))`), a diferencia de `MEALFIT_VISION_MAX_SIDE_PX` que cae al default. Parse inválido → 240 | `vision_agent._ollama_timeout_s` (solo Ollama; gemma local es lento) |
| `MEALFIT_VISION_LLM_TIMEOUT_S` | `30.0` | `(0.0, 120.0]` — cae al default fuera de rango | `vision_agent._vision_llm_timeout_s` (solo `openai_compatible`) |

**API key del provider cloud — atención al nombre exacto**: el path
`openai_compatible` lee `VISION_API_KEY` (ver `.env.example` y
`vision_agent._dispatch_openai_compatible_vision`), **no** `OPENAI_API_KEY`.
Si `VISION_API_KEY` no está seteada, `ChatDeepSeek.__init__` no cae a
`OPENAI_API_KEY` — cae a `_deepseek_api_key()` (env `DEEPSEEK_API_KEY`), que
es la key EQUIVOCADA para el endpoint de OpenAI y produce un 401 silencioso
(el except de `_dispatch_openai_compatible_vision` lo captura y degrada a
`analysis_failed=True`, así que en el modal solo se ve "Error analizando
imagen" sin pista de la causa). **Al encender Luna en prod, setear
`VISION_API_KEY` explícitamente** — no asumir que basta con que
`OPENAI_API_KEY` ya exista en el `.env` para otro propósito.

## Rollback (sin redeploy)

```
# En /opt/mealfit/backend/.env del VPS:
MEALFIT_VISION_PROVIDER=ollama

# reiniciar el proceso backend (systemd/pm2/lo que corra en el VPS)
```

Esto vuelve el escáner a gemma local (costo cero, 30-120 s). No requiere
tocar código ni migraciones — es la misma palanca que ya existía desde
`P1-MEAL-SCAN-GEMMA`. Si el rollback es porque Luna está devolviendo
`analysis_failed` para todos los scans, también sirve dejar
`MEALFIT_VISION_FALLBACK_PROVIDER=ollama` puesto (cascada automática, sin
rollback completo) — pero si el primario está sistemáticamente caído, cada
scan paga el timeout del primario ANTES de caer al fallback, así que el
rollback completo (`MEALFIT_VISION_PROVIDER=ollama`) es más rápido para el
usuario mientras se investiga.

## Por qué el costo va a `llm_usage_events` y NO a `api_usage`

`get_monthly_api_usage` ([`db_profiles.py`](../db_profiles.py)) cuenta
**toda** fila de la tabla `api_usage` del mes sin filtrar por endpoint — así
es como funciona el paywall mensual compartido (gratis=15, basic=50,
plus=200, ultra sin cap real). Si un scan de Luna escribiera ahí, cada foto
de la cena consumiría un crédito de PLAN. Con el tier gratis (15/mes), 3
scans/día agotarían la cuota completa en 5 días — exactamente el bug que
`P1-MEAL-SCAN-GEMMA` cerró el 2026-07-12 (entonces con gemma gratis no
importaba si el gate fallaba; con Luna sí importa, porque encenderlo sin este
fix REABRE el bug).

El fix: `routers/diary.py` ya no llama `log_api_usage(...)` para visión (bajo
ningún provider). El gasto real de un scan CLOUD pago va a
`llm_usage_events` vía `log_llm_usage_event(user_id=..., model=_vision_model_name(),
node="vision_scan")` — el libro de **costo** (modelo + tokens + USD), no el
de **cuota**. Sigue siendo auditable (es el mismo sitio de donde salieron los
números de este documento, vía el canario de pricing `P1-LUNA-PRICING`) sin
cobrarle al usuario un plan por fotografiar su comida.

**No "arreglar" esto tocando `get_monthly_api_usage`.** Esa función cuenta
`api_usage` sin filtro de endpoint A PROPÓSITO — es el contrato de todos los
demás endpoints exentos (ver tabla "Historial-quota-exemption" en
`CLAUDE.md`). Filtrar por endpoint ahí cambiaría la facturación de TODO el
sistema, no solo de visión. El test
[`test_p1_vision_luna.py::test_get_monthly_api_usage_sigue_sin_filtro_de_endpoint`](../tests/test_p1_vision_luna.py)
ancla esto — si alguien "limpia" el SELECT agregándole `AND endpoint = ...`,
el test se pone rojo.

El path LOCAL (`ollama`/gemma, costo cero) tampoco escribe en
`llm_usage_events` — esa tabla es un libro de $ gastado, no de volumen de
scans. Una fila con `cost_usd_micros=NULL` por cada scan gratis es ruido. Si
algún día se necesita auditar volumen de scans locales, el lugar correcto es
`pipeline_metrics`, no el libro de costo.

## Fail-open del resize — consecuencia operacional

`image_prep.prepare_image_for_vision` es fail-open: si Pillow no está
instalado en el host (o no hay wheel para la arquitectura, riesgo real en el
VPS Oracle ARM), el resize se salta silenciosamente y **el escaneo sigue
funcionando** — solo que a 6,6× el costo por foto (foto cruda 3024×4032 en
vez de 1024px), sin ningún error visible en el modal ni en las métricas de
negocio.

**Cómo confirmar que el resize SÍ está corriendo en prod** — grep en los
logs por esta línea (nivel `info`, se emite en cada scan exitoso, sea cual
sea el resultado del análisis):

```
[P1-VISION-LUNA] resize original_bytes=... sent_bytes=... original_wh=... sent_wh=... resized=... skipped_reason=...
```

- `resized=True` + `skipped_reason=None` → Pillow está activo y la foto se
  redujo. Comportamiento esperado.
- `resized=False` + `skipped_reason=pillow_no_disponible` → Pillow AUSENTE
  del host. Cada scan cloud está pagando ~6,6× de más en silencio. Verificar
  `pip show Pillow` en el venv del VPS y que el deploy instaló
  `requirements.txt` correctamente (wheel manylinux aarch64, no debería
  requerir compilar desde fuente).
- `resized=False` + `skipped_reason=error_procesando_imagen:<Excepción>` →
  la imagen específica no se pudo decodificar (formato exótico, bytes
  corruptos). Revisar el `<Excepción>` para diagnosticar esa foto puntual;
  no es un problema de infraestructura como el caso anterior.
- `resized=False` + `skipped_reason=None` → la foto YA cabía dentro del cap
  (`sent_wh` == `original_wh`); comportamiento normal, no reencodear algo que
  ya cabe evita perder calidad sin ahorrar nada.

El `logger.warning` gemelo (`prepare_image_for_vision falló, fail-open a los
bytes originales`) solo aparece cuando el error ocurre DENTRO de un intento
de decode (Pillow presente pero la imagen es rara) — la ausencia total de
Pillow no pasa por esa rama porque nunca se intenta `Image.open`.

## Lo que este documento NO cubre

**No hay evidencia A/B de que la CALIDAD del análisis visual de Luna supere
la de gemma.** Los datos de A/B de Luna que existen en este repo
(`P1-LUNA-PRICING`, canario `project_daygen_luna_canary_2026_07_26`) son
sobre **day-gen** (generación de texto/planes), no sobre visión de fotos de
comida. Este P-fix se justifica enteramente por **costo y latencia**
(1,7-2,7 s vs 30-120 s, $0,002/scan vs gratis-pero-lento) — la promesa de que
Luna "ve mejor" los platos dominicanos que gemma4:12b es una pregunta
abierta, no una afirmación de este documento. Si alguna vez se necesita
decidir el proveedor por CALIDAD (no por costo/latencia), hace falta un A/B
dedicado sobre fotos de comida reales, comparando `meal_name`/`description`/
macros contra ground truth — eso no existe hoy.

## Tests

[`tests/test_p1_vision_luna.py`](../tests/test_p1_vision_luna.py) — ancla
completo: resize (contrato, aspecto, fail-open, clamp del knob), cascada
(orden resize→despacho, dispara solo con knob, propagación de error del
primario, "ambos fallan" devuelve el error del primario), el gate de cuota en
`diary.py` (ya no llama `log_api_usage`, sí llama `log_llm_usage_event` con
`node="vision_scan"`), anti-regresión de `get_monthly_api_usage`, y
supersession del marker `_LAST_KNOWN_PFIX`.
