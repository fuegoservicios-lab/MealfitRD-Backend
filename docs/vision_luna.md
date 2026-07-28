# Escáner de comida y de Nevera: Luna cloud, único provider (P1-VISION-LUNA · 2026-07-28, P1-VISION-NO-LOCAL · 2026-07-28)

Lee esto si "Escanear comida" (Dashboard), "Escanear mi nevera" (Pantry) o el
Diario Visual se comportan raro, si un scan tarda mucho más de 2-3 segundos, o
si `llm_usage_events` muestra un costo que no esperabas. SSOT del código:
[`vision_agent.py`](../vision_agent.py) (despacho), [`image_prep.py`](../image_prep.py)
(resize), [`routers/diary.py`](../routers/diary.py) (endpoint meal-scan +
contabilidad de costo), [`routers/user_data.py`](../routers/user_data.py)
(endpoint del escáner de Nevera, prompt/schema propios).

**[P1-VISION-NO-LOCAL · 2026-07-28]** El provider LOCAL (`ollama`, gemma
vía túnel SSH reverso desde el laptop del owner, P1-MEAL-SCAN-GEMMA) fue
ELIMINADO por completo — el laptop no podía sostener el servicio. Con eso
desaparecieron: el provider `ollama` y su transporte httpx propio, la
cascada de fallback `MEALFIT_VISION_FALLBACK_PROVIDER` (no tenía sentido con
un solo provider real), el single-flight lock que serializaba TODOS los
scans (limitación de una GPU de 4GB, ya no aplica a un provider cloud), y
`is_vision_local()`. El escáner de Nevera (antes su propio cliente Ollama en
`routers/user_data.py`) ahora reutiliza el MISMO transporte cloud que el
meal-scan (`vision_agent.analyze_image_structured`), con su prompt y schema
de negocio propios (items + marca, matcheados contra `master_ingredients`)
sin cambios. **No hay más provider al que hacer rollback** — ver la sección
de rollback más abajo.

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

El fix original (P1-VISION-LUNA) fueron tres cambios en cadena:

1. **Resize defensivo** ([`image_prep.py`](../image_prep.py)) — cap de lado
   mayor a `MEALFIT_VISION_MAX_SIDE_PX` px, re-encode JPEG calidad 85,
   fail-open ante cualquier error. Se invoca en el ÚNICO punto por el que
   pasan los bytes antes de salir al proveedor (`process_image_with_vision`,
   `vision_agent.py`). **Sigue vigente sin cambios.**
2. ~~Cascada de fallback~~ — `MEALFIT_VISION_FALLBACK_PROVIDER`: si el
   primario fallaba, reintentaba UNA vez con un provider distinto (pensado
   para cascar cloud→`ollama`). **ELIMINADA en P1-VISION-NO-LOCAL** — sin
   provider local no había a qué cascar, y la maquinaria (knob + retry en
   `_dispatch_vision_provider`) quedaba muerta. Ver "Rollback" abajo para lo
   que la reemplaza.
3. **El gasto sale del libro de cuota de planes** ([`routers/diary.py`](../routers/diary.py))
   — ver sección dedicada abajo. **Sigue vigente sin cambios.**

**[P1-VISION-NO-LOCAL · 2026-07-28]** añadió un cuarto cambio: el escáner de
Nevera (`routers/user_data.py`, antes su propio cliente Ollama) ahora
despacha a través de `vision_agent.analyze_image_structured` — la misma
resolución de cliente/modelo/key (`_resolve_vision_client`) y el mismo
resize defensivo que el meal-scan, con su prompt/schema de negocio propios
sin cambios (items + marca detectados en la foto, matcheados contra
`master_ingredients`). El único cliente cloud del repo (`vision_agent.py`)
ahora sirve AMBOS escáneres.

## Knobs

Todos vía `knobs._env_*` (auto-registrados en `_KNOBS_REGISTRY`,
`get_knobs_registry_snapshot()` los expone). Defaults confirmados leyendo el
source el 2026-07-28 — no asumidos del plan.

| Knob | Default | Clamp / choices | Dónde |
|---|---|---|---|
| `MEALFIT_VISION_PROVIDER` | `disabled` | `{disabled, off, openai_compatible}` — fuera del set cae al default con WARNING. **`ollama` salió del choices-set en P1-VISION-NO-LOCAL** — un valor `ollama` remanente en el `.env` ahora degrada a `disabled` con WARNING (apaga el feature) en vez de fallar silenciosamente. | `vision_agent._vision_provider` |
| `MEALFIT_VISION_MODEL` | `""` (vacío) | sin choices; `_env_str` normaliza a lowercase | `vision_agent._vision_model_name` — modelo del ÚNICO provider (`openai_compatible`) |
| `MEALFIT_VISION_BASE_URL` | `""` (vacío) | sin choices | `vision_agent._vision_base_url` |
| `MEALFIT_VISION_MAX_SIDE_PX` | `1024` | `[256, 4096]` — fuera de rango cae al **default**, NO clampa al borde (así trata `_env_int` un `validator` que retorna False) | `image_prep.vision_max_side_px` |
| `MEALFIT_VISION_LLM_TIMEOUT_S` | `30.0` | `(0.0, 120.0]` — cae al default fuera de rango | `vision_agent._vision_llm_timeout_s` |

**[P1-VISION-NO-LOCAL · 2026-07-28] Knobs ELIMINADOS** (ya no se leen en
ningún módulo de producción — si siguen en el `.env` del VPS, bórralos, son
ruido inerte): `MEALFIT_OLLAMA_BASE_URL`, `MEALFIT_OLLAMA_VISION_MODEL`,
`MEALFIT_VISION_FALLBACK_PROVIDER`, `MEALFIT_VISION_TIMEOUT_S` (era el
timeout del roundtrip Ollama, 240s/clamp [30,600] — el timeout cloud vigente
es `MEALFIT_VISION_LLM_TIMEOUT_S`, tabla arriba, un knob DISTINTO que ya
existía y sigue vivo).

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

**[P1-VISION-NO-LOCAL · 2026-07-28] No hay más provider al que hacer
rollback.** `ollama` (gemma local) fue eliminado del código, no solo
apagado — no existe una palanca de env var que lo reviva; revivirlo exigiría
revertir el commit de P1-VISION-NO-LOCAL (o cherry-pickear `vision_agent.py`
pre-fix) Y volver a levantar el túnel SSH del laptop del owner, que es
precisamente lo que este P-fix cerró porque no era sostenible.

El ÚNICO rollback disponible hoy es apagar el feature (soft-fail, sin costo,
sin escaneo):

```
# En /opt/mealfit/backend/.env del VPS:
MEALFIT_VISION_PROVIDER=disabled   # o "off" — mismo efecto

# reiniciar el proceso backend (systemd/pm2/lo que corra en el VPS)
```

Con esto, "Escanear comida"/"Escanear mi nevera"/Diario Visual responden
`analysis_failed=True` (meal-scan) o `503` (escáner de Nevera) — el frontend
ya distingue ese estado, así que no es un 500 sorpresivo. Si Luna está
devolviendo `analysis_failed` de forma intermitente (no sistemática), el
knob a revisar primero es `VISION_API_KEY`/`MEALFIT_VISION_BASE_URL` (ver
sección de arriba) o el circuito de rate-limit del proveedor — no hay
cascada automática a la que apostar mientras se investiga.

Si en el futuro se necesita un SEGUNDO provider real (otro modelo
OpenAI-compatible con visión, por ejemplo, para diversificar o abaratar
más), la cascada de P1-VISION-LUNA es un patrón razonable para reintroducir
— pero como fallback ENTRE DOS PROVIDERS CLOUD, no como vuelta a un modelo
local.

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

**[P1-VISION-NO-LOCAL]** El gate que antes excluía el path LOCAL
(`not is_vision_local()`, gemma/Ollama costo cero, para no escribir una fila
`cost_usd_micros=NULL` por scan gratis) se eliminó junto con `is_vision_local()`
— sin provider local esa condición nunca podía ser False, así que el
accessor quedaba muerto. Con un único provider (cloud), la fila se emite
SIEMPRE que hay usuario autenticado; ya no hay un "path gratis" que excluir.

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

- [`tests/test_p1_vision_luna.py`](../tests/test_p1_vision_luna.py) — ancla
  del resize (contrato, aspecto, fail-open, clamp del knob), el dispatch por
  MODELO (`ChatOpenAI` vs `ChatDeepSeek`), la precedencia de API key, el
  gate de cuota en `diary.py` (ya no llama `log_api_usage`, sí llama
  `log_llm_usage_event` con `node="vision_scan"`), anti-regresión de
  `get_monthly_api_usage`, y supersession del marker `_LAST_KNOWN_PFIX`.
  **Los tests de cascada (`ollama` de fallback) fueron eliminados en
  P1-VISION-NO-LOCAL junto con la cascada misma.**
- [`tests/test_p1_vision_no_local.py`](../tests/test_p1_vision_no_local.py)
  — ancla de la eliminación: el escáner de Nevera ya no requiere
  `provider == "ollama"` ni devuelve 503 bajo el provider cloud, blanket
  parser que falla si algún módulo de producción sigue referenciando
  `ollama`/`gemma`/`_ollama_`, y anti-regresión de la doctrina de cuota
  (costo a `llm_usage_events`, nunca a `api_usage`).
