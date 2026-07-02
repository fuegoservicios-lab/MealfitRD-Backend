import logging
import os
import math
from typing import List, Optional
# [P0-DEEPSEEK-MIGRATION · 2026-06-12] Embeddings ahora via capa pluggable
# (Gemini eliminado; DeepSeek no ofrece embeddings — provider pendiente).
from embeddings_provider import get_text_embedding

# [P2-1 · 2026-05-08] Helpers compartidos del registry de knobs. Antes los 5
# knobs `MEALFIT_*` de este módulo (POOL_FALLBACK_ALERT_*, LESSON_BUFFER_BACKLOG,
# CHILDREN_MULTIPLIER) eran raw `os.environ.get` y no aparecían en
# `/health/version` ni en `get_knobs_registry_snapshot()`. Importable a top-level
# porque `knobs.py` no depende de constants ni de graph_orchestrator (cero ciclo).
from knobs import _env_int, _env_float, _env_bool

logger = logging.getLogger(__name__)

PLAN_CHUNK_SIZE = 3  # Días generados por chunk en el pipeline de langgraph

# [P2-2] Política de programación temporal de chunks (`_compute_chunk_delay_days`).
# Define cuántos días ANTES del inicio del chunk N+1 se programa su `execute_after`,
# y por tanto cuánta señal de aprendizaje (logs reales del usuario sobre el chunk N)
# alcanza a observarse antes de generar el siguiente.
#
# Valores válidos:
#   - "strict" (default): chunk N+1 se programa para iniciar JUSTO después de que
#     termine el chunk N (delay_days = days_offset, modulo CHUNK_PROACTIVE_MARGIN_DAYS).
#     El aprendizaje del chunk previo es completo (el usuario tuvo todos los días
#     para registrar comidas). Tradeoff: si la generación falla cerca del momento de
#     ejecución, el usuario puede ver "no hay plan disponible para hoy" hasta que
#     pase el retry. Recomendado en producción para maximizar calidad del aprendizaje
#     (cuál es el punto de aprendizaje continuo si el LLM corre antes de tener señal).
#
#   - "safety_margin" (alternativo): chunk N+1 se adelanta ceil(days_count/2) días.
#     Para chunks típicos de 3 días eso son 2 días de adelanto: el chunk se genera
#     cuando el usuario solo ha vivido ~1 día del chunk anterior. Para planes ≥15
#     días, además el ÚLTIMO chunk se adelanta 3 días (heurística "GAP B"). Tradeoff:
#     buffer ante fallos de generación a costa de aprendizaje empobrecido. Útil en
#     entornos donde la latencia/fiabilidad del LLM no es estable o como prevención
#     durante incidentes (ramp-up controlado). NO recomendado en operación normal:
#     `_check_chunk_learning_ready` puede rechazar chunks adelantados con
#     `prev_chunk_day_not_yet_elapsed`, anulando el supuesto beneficio. Validar con
#     dashboard de deferrals antes de habilitar a más usuarios.
#
# Cualquier otro valor cae a "strict" defensivamente (sin warning porque la
# selección viene de env var en boot).
CHUNK_LEARNING_MODE = os.environ.get("CHUNK_LEARNING_MODE", "strict").strip().lower()
if CHUNK_LEARNING_MODE not in {"strict", "safety_margin"}:
    CHUNK_LEARNING_MODE = "strict"
# [P3-PROD-AUDIT-3 · 2026-05-30] Floor/ceiling clamp + auto-registro en
# _KNOBS_REGISTRY. ANTES era un `int(os.environ.get(...))` crudo sin clamp (a
# diferencia de los demás CHUNK_* en este archivo): `CHUNK_PIPELINE_TIMEOUT_SECONDS=0`
# (o negativo) hacía que `_fut.result(timeout=0)` lanzara TimeoutError inmediato en
# CADA chunk → todos degradaban a Smart Shuffle tras 3 intentos → calidad de plan
# fleet-wide degradada en silencio por un solo env var malo. El validator cae al
# default 180 si está fuera de [30, 1800].
CHUNK_PIPELINE_TIMEOUT_SECONDS = _env_int(
    "CHUNK_PIPELINE_TIMEOUT_SECONDS", 180, validator=lambda v: 30 <= v <= 1800
)
# [P0-A] Default 0: CHUNK_PROACTIVE_MARGIN_DAYS=0 significa "después del último día del chunk previo" (no "el último día").
# chunk N+1 se ejecuta DESPUÉS de que termine N para que el aprendizaje
# vea la adherencia completa del chunk previo. Subir a 1 solo si el usuario reporta gaps de
# disponibilidad al despertar (chunk listo recién a media mañana en vez de a primera hora).
CHUNK_PROACTIVE_MARGIN_DAYS = max(0, int(os.environ.get("CHUNK_PROACTIVE_MARGIN_DAYS", "0")))
CHUNK_SCHEDULER_INTERVAL_MINUTES = max(1, int(os.environ.get("CHUNK_SCHEDULER_INTERVAL_MINUTES", "1")))
CHUNK_LEARNING_READY_MIN_RATIO = float(os.environ.get("CHUNK_LEARNING_READY_MIN_RATIO", "0.5"))
CHUNK_LEARNING_READY_DELAY_HOURS = max(1, int(os.environ.get("CHUNK_LEARNING_READY_DELAY_HOURS", "12")))
CHUNK_LEARNING_READY_MAX_DEFERRALS = max(0, int(os.environ.get("CHUNK_LEARNING_READY_MAX_DEFERRALS", "2")))
# [P1-C] Cap de reintentos del temporal gate antes de escalación dura. Antes, si el cálculo
# de fechas del temporal gate decía "el día previo aún no concluyó" (típicamente por TZ
# desalineada o `_plan_start_date` corrompido), el chunk se difería para siempre. El cron
# `_detect_chronic_deferrals` notifica al usuario tras 5/48h, pero si el usuario nunca
# corrige su perfil el chunk seguía atascado indefinidamente — el plan no avanza.
#
# Con este cap, tras N reintentos consecutivos del mismo gate forzamos `ready=True` con
# `reason='temporal_gate_max_retries_exceeded'`, dejando que el chunk genere "tarde"
# en lugar de quedarse colgado para siempre. Mejor un plan generado con tz-error que un
# plan trabado. El override se telemetra vía `chunk_deferrals` para que el equipo de
# operaciones detecte estos forced overrides y arregle el origen de la desalineación.
#
# [P1-3 → P1-4] Default 8 (antes 5, antes 20). Historial:
#   - Cap=20 (original): el chunk caía en silent failure por ~20 min antes del
#     forced override; el usuario nunca veía que algo estaba mal hasta que
#     `_detect_chronic_deferrals` (cada 6h, umbral 5/48h) eventualmente notificaba.
#   - Cap=5 (P1-3): forzaba ready=True tras ~5 min, pero con TZ drift moderado
#     (>5 min, e.g., DST transition o cliente que reportó TZ tarde) el chunk
#     escalaba a `pending_user_action` prematuramente o forzaba override antes
#     de que el día previo hubiera realmente terminado en local.
#   - Cap=8 (P1-4): combinado con el backoff exponencial introducido abajo,
#     cubre ~4h de gracia (1+2+4+8+16+30+30+30 min ≈ 121 min ≈ 2h hasta el
#     forced override, dejando margen real para drifts moderados).
CHUNK_TEMPORAL_GATE_MAX_RETRIES = max(1, int(os.environ.get("CHUNK_TEMPORAL_GATE_MAX_RETRIES", "8")))
# [P1-4] Cap del backoff exponencial entre re-evaluaciones del temporal gate.
# Cada deferral persiste un nuevo `execute_after = NOW() + min(2^retry, cap)` minutos
# en plan_chunk_queue (junto al contador `_temporal_gate_retries`), reemplazando el
# patrón anterior donde el chunk se re-evaluaba cada `CHUNK_SCHEDULER_INTERVAL_MINUTES`
# linealmente. Con base 2 y cap 30:
#   retry=0 → 1 min   (después de la 1ra evaluación)
#   retry=1 → 2 min
#   retry=2 → 4 min
#   retry=3 → 8 min
#   retry=4 → 16 min
#   retry≥5 → 30 min (capped)
# Total acumulado hasta cap=8: 1+2+4+8+16+30+30+30 = 121 min ≈ 2h.
# Cap default 30 evita ventanas largas (>1h) que harían parecer al usuario que el
# plan está congelado. Subir si la flota tiene drifts de TZ recurrentes >2h.
CHUNK_TEMPORAL_GATE_BACKOFF_CAP_MINUTES = max(1, int(os.environ.get("CHUNK_TEMPORAL_GATE_BACKOFF_CAP_MINUTES", "30")))
# [P1-3 → P1-4] Push notification proactiva en el N-th deferral consecutivo del MISMO chunk.
# Antes la única señal al usuario era `_detect_chronic_deferrals` (cron cada 6h, umbral
# 5 deferrals en 48h). Eso significaba que un chunk con TZ desalineada deferia hasta
# CHUNK_TEMPORAL_GATE_MAX_RETRIES sin que el usuario supiera nada.
# Default movido a 5 (antes 3) por P1-4: con el backoff exponencial, retry=3 ocurre
# ~7 min después del primer deferral; el usuario aún no ha tenido tiempo razonable
# para revisar su TZ (especialmente si el plan está corriendo en background). Retry=5
# llega a ~31 min — momento más apropiado para una notificación que no se siente como
# spam. Para flotas con drifts crónicos, bajar a 4 reduce la latencia del aviso.
# Dedupe por (user_id, meal_plan_id, week_number) vía system_alerts con cooldown
# CHUNK_TEMPORAL_GATE_PUSH_COOLDOWN_HOURS.
CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY = max(1, int(os.environ.get("CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY", "5")))
# Cooldown entre pushes proactivos del N-th deferral para el MISMO chunk. Sin esto,
# si el usuario ignora la primera notificación y el chunk sigue defiriendo, cada
# tick desde el N-th hasta el cap dispararía un push. Default 6h ≈ ventana razonable
# para que el usuario actúe sin sentirse spameado.
CHUNK_TEMPORAL_GATE_PUSH_COOLDOWN_HOURS = max(1, int(os.environ.get("CHUNK_TEMPORAL_GATE_PUSH_COOLDOWN_HOURS", "6")))
# [P0-3] TTL de la pausa cuando el chunk N+1 quiere generarse pero el chunk N aún no
# ha concluido en el calendario del usuario. Tras CHUNK_LEARNING_READY_MAX_DEFERRALS
# deferrals (típicamente 2 × 12h = 24h), el chunk pasa a `pending_user_action` con
# reason=`prev_chunk_not_concluded`. Si pasa este TTL adicional sin que el día previo
# concluya, el recovery cron escala a flexible_mode. Mejor un plan en flexible que
# congelado, pero le damos al usuario una ventana explícita primero.
CHUNK_PREV_CHUNK_PAUSE_TTL_HOURS = max(1, int(os.environ.get("CHUNK_PREV_CHUNK_PAUSE_TTL_HOURS", "24")))
LIFETIME_LESSONS_WINDOW_DAYS = max(1, int(os.environ.get("LIFETIME_LESSONS_WINDOW_DAYS", "60")))
# [P1-4] Decaimiento semanal aplicado al ponderar lecciones del historial
# (`_lifetime_lessons_history`) cuando se recomputa `_lifetime_lessons_summary`.
# Sin esto, una lección del chunk 1 (8 semanas atrás) pesaba igual que una del
# chunk 9 (1 semana atrás): el LLM seguía recibiendo "no usar pollo" por meses
# después de que el usuario lo aceptara silentemente, generando UX de plan
# repetitivo sin variedad real. Ahora cada lección entra al summary con peso
# `LIFETIME_LESSON_WEEKLY_DECAY ** weeks_old`. Con default 0.9:
#   - Lección de hace 1 semana: peso 0.90.
#   - Lección de hace 4 semanas: peso 0.66.
#   - Lección de hace 8 semanas: peso 0.43.
#   - Lección de hace 22 semanas: peso 0.10 (filtrada por LIFETIME_LESSON_MIN_WEIGHT).
# El ranking de top_rejection_hits / top_repeated_bases / top_repeated_meal_names
# ahora prioriza recencia: un patrón de hace 1 semana desplaza a uno de hace 8
# en los caps de truncación (top 20, top 30) en vez de competir por inserción
# arbitraria (sets) o conteo plano de chunks.
LIFETIME_LESSON_WEEKLY_DECAY = max(
    0.01,
    min(1.0, float(os.environ.get("LIFETIME_LESSON_WEEKLY_DECAY", "0.9"))),
)
# [P1-4] Threshold mínimo de peso bajo el cual una lección se excluye del summary.
# Con LIFETIME_LESSON_WEEKLY_DECAY=0.9, peso 0.10 corresponde a ~22 semanas. Como
# LIFETIME_LESSONS_WINDOW_DAYS=60 ya filtra a ~8.5 semanas, el threshold por sí
# solo no recorta más en el caso default — pero protege contra ventanas amplias
# (env var override) o lecciones legacy con timestamp corrupto que escapen al
# filtro de cutoff. Forward-compatible y defense-in-depth.
LIFETIME_LESSON_MIN_WEIGHT = max(
    0.0,
    min(1.0, float(os.environ.get("LIFETIME_LESSON_MIN_WEIGHT", "0.1"))),
)
# [P2-PANTRY-FLOOR-VS-NUDGE · 2026-06-23] PISO DURO que gatea la PAUSA de la
# generación de mantenimiento (`_should_pause_chunk_for_insufficient_pantry`): si
# la nevera tiene MENOS ítems significativos que esto, no se puede generar un plan
# "desde la nevera" → se pausa. NO es un objetivo de nevera surtida (ese es
# PANTRY_RECOMMENDED_ITEMS abajo, solo nudge visual). MANTENER BAJO: subirlo alto
# congelaría el mantenimiento de casi todos (pocos usuarios reales tienen 20+
# ítems distintos). Bump 3→5: 3 ítems es casi-vacío; 5 es un piso más honesto.
CHUNK_MIN_FRESH_PANTRY_ITEMS = max(1, int(os.environ.get("CHUNK_MIN_FRESH_PANTRY_ITEMS", "5")))
# [P2-PANTRY-FLOOR-VS-NUDGE · 2026-06-23] OBJETIVO RECOMENDADO (nudge visual, NO
# bloquea NADA): número aspiracional que el banner "Mi Nevera" muestra para animar
# al usuario a surtir mejor su nevera y que sus planes la aprovechen. Desacoplado
# del piso duro de arriba — subirlo NO afecta la generación, solo el copy. Clamp
# [piso, 100]. Knob `MEALFIT_PANTRY_RECOMMENDED_ITEMS` para A/B sin redeploy.
PANTRY_RECOMMENDED_ITEMS = max(
    CHUNK_MIN_FRESH_PANTRY_ITEMS,
    min(100, int(os.environ.get("MEALFIT_PANTRY_RECOMMENDED_ITEMS", "20"))),
)
CHUNK_PANTRY_EMPTY_TTL_HOURS = max(1, int(os.environ.get("CHUNK_PANTRY_EMPTY_TTL_HOURS", "12")))
CHUNK_PANTRY_EMPTY_REMINDER_HOURS = max(1, int(os.environ.get("CHUNK_PANTRY_EMPTY_REMINDER_HOURS", "4")))
CHUNK_PANTRY_EMPTY_MAX_REMINDERS = max(0, int(os.environ.get("CHUNK_PANTRY_EMPTY_MAX_REMINDERS", "2")))
# [P2-3] Frecuencia del cron dedicado de limpieza de chunks huérfanos. Antes
# este cleanup vivía embebido en `process_plan_chunk_queue` (corre cada
# CHUNK_SCHEDULER_INTERVAL_MINUTES, default 1 min) y se ejecutaba ANTES de
# procesar el batch del tick. Si la query de orphan-detection era lenta o
# fallaba, bloqueaba el hot path del worker. Aislarla en su propio job:
#   - Decoupling: el processor no depende del cleanup para arrancar su tick.
#   - Frecuencia configurable independiente: cleanup más relajado (5 min)
#     mientras el processor mantiene latencia de 1 min.
#   - Aislamiento de fallas: timeout/error en el cleanup ya no afecta la
#     corrida normal del worker.
# Default 5 min: rápido para que el usuario no vea chunks fantasma de un plan
# cancelado por mucho tiempo, pero no spam contra la DB.
CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES = max(1, int(os.environ.get("CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES", "5")))
# [P1-2] Tope de intentos consecutivos del gate temporal sin ancla recuperable
# (`_plan_start_date` ausente y ni `grocery_start_date` ni `created_at` rescatables).
# Cada tick del scheduler que evalúa un chunk con ese gate incrementa el contador
# persistido en `plan_data._anchor_recovery_attempts`. Al exceder este umbral
# (default 3 ≈ ~45 min con scheduler de 15 min, tiempo razonable para que un blip
# de DB/snapshot se recupere), se invoca `_escalate_unrecoverable_chunk` con
# reason='unrecoverable_missing_anchor': el chunk se dead-letterea, se marca
# `meal_plans.plan_data._user_action_required` y se notifica al usuario para
# que regenere el plan manualmente. Antes el chunk quedaba pausado en
# 'pending_user_action' indefinidamente sin escalar.
CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS = max(1, int(os.environ.get("CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS", "3")))
# [P0-2] Cuando `_resolve_chunk_start_anchor` agota la cadena de fallback y devuelve
# source='forced_8am_utc' (ni snapshot, ni profile, ni último plan tienen TZ
# resoluble), el comportamiento legacy era encolar el chunk a 8am UTC + delay_days.
# Para usuarios en TZ negativas (UTC-5 Bogotá → 3am local; UTC-8 PST → midnight)
# eso disparaba la generación a horas inutilizables: el chunk se ejecutaba antes
# de que el usuario registrara nada del día previo, rompiendo el aprendizaje
# continuo. Ahora con flag activado (default True) el chunk se INSERTA pausado en
# 'pending_user_action' con reason='tz_unresolved' y se notifica al usuario; el
# recovery cron re-intenta `_resolve_chunk_start_anchor` en cada tick y reanuda
# en cuanto la TZ es resoluble (típicamente cuando el usuario abre la app y se
# escribe `tz_offset_minutes` en el perfil). Sin este flag, comportamiento legacy
# (chunk encolado a 8am UTC con riesgo de hora local equivocada).
CHUNK_REJECT_FORCED_UTC_ENQUEUE = (
    os.environ.get("CHUNK_REJECT_FORCED_UTC_ENQUEUE", "true").lower() == "true"
)
# [P0-2] Tope de intentos del recovery cron tratando de re-resolver TZ para un
# chunk pausado con reason='tz_unresolved'. Persistido en
# pipeline_snapshot._tz_recovery_attempts. Al exceder este umbral, escalamos via
# `_escalate_unrecoverable_chunk` con reason='unrecoverable_tz_unresolved' para
# evitar que un usuario sin TZ persistible quede atrapado en pausa indefinida.
# Default 6 (≈ 90 min con cron cada 15 min) es suficiente para cubrir el caso
# normal de "usuario abrirá la app pronto" sin atrapar bug de soporte.
CHUNK_TZ_RECOVERY_MAX_ATTEMPTS = max(
    1,
    int(os.environ.get("CHUNK_TZ_RECOVERY_MAX_ATTEMPTS", "6")),
)
# [P1-CHUNK-4] Piso wall-clock (minutos por intento) para escalar un chunk pausado
# (tz_unresolved / missing_anchor / corrupted_date) a dead_letter. El contador de
# intentos por sí solo está ACOPLADO a la frecuencia del cron: si el cron corre más
# seguido que los 15 min asumidos (o se dispara extra vía pre-pickup TZ sync, o un
# operador reduce el intervalo), los intentos se acumulan más rápido que el tiempo
# real y el chunk se dead-letterea ANTES de darle al usuario la ventana real (~45 min
# anchor, ~90 min tz) para accionar (abrir la app → persistir tz_offset_minutes).
# El piso wall-clock garantiza que NUNCA escalamos antes de `max_attempts × este
# valor` minutos transcurridos desde el primer intento de recuperación, sin importar
# cuántos ticks dispararon. Default 15 reconstruye exactamente la intención original
# documentada (3×15≈45 min, 6×15≈90 min) como garantía de tiempo real, no de conteo.
CHUNK_RECOVERY_MIN_WALL_MINUTES_PER_ATTEMPT = max(
    1,
    min(120, int(os.environ.get("CHUNK_RECOVERY_MIN_WALL_MINUTES_PER_ATTEMPT", "15"))),
)
# [P0-α/TZ-UNRESOLVED-ALERT] Alerta SRE cuando chunks llevan >Nh pausados con
# `_pantry_pause_reason='tz_unresolved'`. El recovery cron resuelve el caso
# normal (el usuario abre la app y `tz_offset_minutes` se persiste) pero hay un
# escenario degradado: el usuario nunca abre la app post-signup, o el flujo de
# update_user_health_profile falla silenciosamente. En ese hueco el chunk queda
# en pending_user_action sin que nadie lo vea hasta que el TTL del recovery
# (CHUNK_TZ_RECOVERY_MAX_ATTEMPTS × CHUNK_TZ_SYNC_INTERVAL_MINUTES ≈ 90 min)
# escala a dead-letter — pero ese fallback NO cubre la ventana intermedia donde
# el chunk lleva horas pausado sin progreso. Esta alerta llena ese hueco.
#
# CHUNK_TZ_UNRESOLVED_ALERT_HOURS: edad mínima del pause antes de alertar. 6h
#   alinea con la cadencia de `_detect_chronic_deferrals` y le da margen al
#   recovery normal (el chunk se reanuda en cuanto el usuario abre la app).
CHUNK_TZ_UNRESOLVED_ALERT_HOURS = max(1, int(os.environ.get("CHUNK_TZ_UNRESOLVED_ALERT_HOURS", "6")))
# CHUNK_TZ_UNRESOLVED_ALERT_INTERVAL_MINUTES: frecuencia del cron. 30 min es el
#   sweet spot entre detección rápida (operador no espera 6h tras crossing del
#   threshold para enterarse) y costo del scan (query barata pero suma con la
#   flota de crons existentes).
CHUNK_TZ_UNRESOLVED_ALERT_INTERVAL_MINUTES = max(15, int(os.environ.get("CHUNK_TZ_UNRESOLVED_ALERT_INTERVAL_MINUTES", "30")))
# CHUNK_TZ_UNRESOLVED_ALERT_COOLDOWN_HOURS: dedupe de la alerta agregada. 12h
#   evita spam si el problema persiste; al expirar, si hay nuevos chunks o los
#   antiguos siguen stuck, re-dispara.
CHUNK_TZ_UNRESOLVED_ALERT_COOLDOWN_HOURS = max(1, int(os.environ.get("CHUNK_TZ_UNRESOLVED_ALERT_COOLDOWN_HOURS", "12")))
# CHUNK_TZ_UNRESOLVED_ALERT_BATCH_LIMIT: cap de chunks examinados por corrida.
#   100 cubre flota mediana sin bloquear el scheduler.
CHUNK_TZ_UNRESOLVED_ALERT_BATCH_LIMIT = max(10, int(os.environ.get("CHUNK_TZ_UNRESOLVED_ALERT_BATCH_LIMIT", "100")))
# [P1-β/TZ-NUDGE] Re-push periódico a usuarios cuyo chunk lleva atascado en
# `_pantry_pause_reason='tz_unresolved'`. El push inicial se manda al ENQUEUE vía
# `_maybe_notify_user_tz_unresolved`, pero si el usuario nunca abre la app el
# helper tiene cooldown 24h y nadie le re-recuerda hasta que el recovery
# escala a dead-letter (~90min). Este cron busca usuarios stuck >Nh y re-invoca
# el helper (que respeta su propio cooldown 24h, así que no spamea).
#
# CHUNK_TZ_NUDGE_THRESHOLD_HOURS: edad mínima del pause antes de re-empujar.
#   12h = un día completo desde el primer push original sin que el usuario abriera
#   la app. Suficientemente paciente para no agobiar; suficientemente proactivo
#   para que el usuario no descubra el plan parado al tercer día.
CHUNK_TZ_NUDGE_THRESHOLD_HOURS = max(1, int(os.environ.get("CHUNK_TZ_NUDGE_THRESHOLD_HOURS", "12")))
# CHUNK_TZ_NUDGE_INTERVAL_MINUTES: frecuencia del cron. 60min porque el cooldown
#   del helper (24h) ya rate-limitea per-usuario; correr más a menudo no aporta.
CHUNK_TZ_NUDGE_INTERVAL_MINUTES = max(15, int(os.environ.get("CHUNK_TZ_NUDGE_INTERVAL_MINUTES", "60")))
# CHUNK_TZ_NUDGE_MAX_USERS: cap por corrida para no saturar webpush si la flota
#   crece. 50 cubre flotas medianas; el cron volverá a correr al siguiente tick.
CHUNK_TZ_NUDGE_MAX_USERS = max(1, int(os.environ.get("CHUNK_TZ_NUDGE_MAX_USERS", "50")))
# [P0-3] Detección PROACTIVA de zero-log al borde de chunk (cuando se enqueuea
# chunk N+1). Antes el sistema solo detectaba zero-log REACTIVAMENTE: el worker
# levantaba el chunk, computaba `learning_ready`, y si `zero_log` se confirmaba
# difería el chunk hasta CHUNK_LEARNING_READY_MAX_DEFERRALS×CHUNK_LEARNING_READY_DELAY_HOURS
# antes de pausar — un retraso de 2-4h en el que el chunk colgaba sin que el
# usuario supiera, y donde el sintetizador `_synthesize_last_chunk_learning_from_plan_days`
# producía un stub vacío que el LLM podía interpretar como "no hubo violaciones".
#
# Con flag activo (default True) probamos al ENQUEUE (no en pickup): si el
# chunk previo tuvo 0 logs explícitos Y 0 mutaciones de inventario en su
# ventana, marcamos chunk N+1 con `_zero_log_proactive_detected=True` +
# `_learning_ready_deferrals=MAX` para que el worker SALTE la cola de
# deferrals y pause inmediatamente con `learning_zero_logs`. Resultado: el
# usuario recibe el push 2-4h antes y el siguiente chunk no se genera con
# señal sintética.
#
# Solo se prueba para chunks no-iniciales (week_number > 1 y chunk_kind !=
# 'initial_plan') para evitar overhead en planes recién creados (donde aún no
# hay ventana previa que probar).
CHUNK_ZERO_LOG_PROACTIVE_DETECTION = (
    os.environ.get("CHUNK_ZERO_LOG_PROACTIVE_DETECTION", "true").lower() == "true"
)
# [P1-α/ZERO-LOG-AUTO-ESCALATE] Cuando un chunk detecta zero-log REACTIVO (ya en
# pickup), el flujo legacy hace hasta CHUNK_LEARNING_READY_MAX_DEFERRALS reintentos
# de 12h c/u antes de pausar — eso son hasta 24h sin plan visible para el usuario.
# El audit P1-α reportó esa cascada como degradación de UX inaceptable: el usuario
# que no logueó comidas ve un plan en blanco durante 1+ día con apenas un push
# inicial. Con knob activado (default true) saltamos directo al path de pausa con
# TTL corto (CHUNK_ZERO_LOG_RECOVERY_TTL_HOURS=4h por defecto) en el primer
# deferral cuando _is_zero_log=True y no hay inventory_proxy disponible: el
# usuario se entera 8-20h antes y el chunk queda en pending_user_action listo
# para resolver con cualquiera de las salidas existentes (logs, mutaciones de
# inventario, o expiración → flexible_mode).
#
# El proactive (CHUNK_ZERO_LOG_PROACTIVE_DETECTION) ya cubre el caso ENQUEUE; este
# knob cubre el caso REACTIVO donde el proactive no detectó el zero-log al
# encolar (e.g., chunk inicial, fallo transitorio del proactive probe).
CHUNK_ZERO_LOG_AUTO_ESCALATE = (
    os.environ.get("CHUNK_ZERO_LOG_AUTO_ESCALATE", "true").lower() == "true"
)
# [P0-4] Guard PROACTIVO de inventario al enqueue de chunks no-iniciales. Antes
# `_enqueue_plan_chunk` solo verificaba TZ (P0-2) y zero-log (P0-3); el chequeo
# de "items en nevera < CHUNK_MIN_FRESH_PANTRY_ITEMS" ocurría REACTIVAMENTE en el
# worker (~ línea 13193 de cron_tasks.py) cuando el chunk se levantaba para
# generar — típicamente días después del enqueue. El usuario veía un "plan
# fantasma" en la UI durante ese intervalo, hasta que el worker pausaba.
#
# Con flag activo (default True) probamos `get_user_inventory_net(user_id)` al
# ENQUEUE: si los items vivos están por debajo del mínimo, flipeamos el chunk
# a `pending_user_action` con `reason='empty_pantry_proactive'` y disparamos
# push inmediato pidiendo refrescar la nevera. El recovery cron existente
# (`_recover_pantry_paused_chunks`) lo reanuda en cuanto detecta items
# suficientes.
#
# Solo se prueba para chunks no-iniciales (week_number > 1 y chunk_kind !=
# 'initial_plan'). [P2-PANTRY-COMMENT-FIX · 2026-06-21] El plan INICIAL es
# intencionalmente EXENTO del mínimo de nevera: su lista de compras DEFINE el
# inventario futuro, así que un usuario nuevo con la nevera vacía DEBE poder
# generar su primer plan (onboarding). NO es que `routers/plans.py` valide un
# piso inicial — al contrario, `_run_pantry_validation_for_initial_chunk`
# (P1-PANTRY-GUARD-INITIAL-SKIP) SALTA la validación contra la nevera cuando hay
# < PANTRY_GUARD_MIN_ITEMS items. El mínimo de nevera (este guard) aplica SOLO al
# MANTENIMIENTO: cuando el usuario ya tiene un ciclo vivo y, con el tiempo, borra
# o agota los alimentos de su nevera — ahí el plan rolling SÍ debe respetar lo
# que hay y pausar/pedir reabastecer si cae bajo el mínimo.
CHUNK_PANTRY_PROACTIVE_GUARD = (
    os.environ.get("CHUNK_PANTRY_PROACTIVE_GUARD", "true").lower() == "true"
)
# [P0-2] TTL corto para cuando el worker no puede revalidar el inventario vivo al final
# de la generación. No aceptamos el chunk; pausamos y reintentamos pronto.
CHUNK_FINAL_VALIDATION_PAUSE_TTL_HOURS = max(1, int(os.environ.get("CHUNK_FINAL_VALIDATION_PAUSE_TTL_HOURS", "2")))
# [P0-C] Recovery proactivo de chunks pausados por validación final fallida cuando el
# usuario completa la lista de compras urgente. Antes este caso esperaba el TTL completo
# (2h por defecto) antes de escalar a flexible_mode, incluso si el usuario añadía los
# ingredientes faltantes a la nevera 5 minutos después de la pausa. Ahora el recovery
# cron compara `_pantry_pause_missing_ingredients` (persistido en pause_snapshot) contra
# la pantry actual cada tick; si está cubierto, re-encola como pending.
#
# CHUNK_FINAL_VALIDATION_MAX_RECOVERY_ATTEMPTS: tope de reintentos automáticos antes de
#   dejar al chunk seguir la ruta TTL existente. Sin tope, un chunk cuyo LLM siempre
#   genera ingredientes faltantes podría loopear (resume → pausa → resume → ...). 3 es
#   suficiente para cubrir reintentos legítimos sin que un loop empate al cron.
CHUNK_FINAL_VALIDATION_MAX_RECOVERY_ATTEMPTS = max(
    1,
    int(os.environ.get("CHUNK_FINAL_VALIDATION_MAX_RECOVERY_ATTEMPTS", "3")),
)
# CHUNK_FINAL_VALIDATION_RECOVERY_GRACE_MINUTES: tiempo mínimo de pausa antes de evaluar
#   recovery. Evita reintentar inmediatamente si el live fetch acababa de caer (puede
#   recuperarse solo en segundos sin necesidad de re-encolar el chunk).
CHUNK_FINAL_VALIDATION_RECOVERY_GRACE_MINUTES = max(
    0,
    int(os.environ.get("CHUNK_FINAL_VALIDATION_RECOVERY_GRACE_MINUTES", "5")),
)
# [P0-2] TTL específico para pausas por stale_snapshot (live fetch caído + snapshot vencido).
# Más corto que la pausa por nevera vacía porque el usuario no puede accionar nada — la causa
# es del lado servidor; conviene reintentar pronto y, si persiste, escalar rápido a flexible.
CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS = max(1, int(os.environ.get("CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS", "4")))
# [P0-1] Tope ABSOLUTO (en horas) que un chunk puede permanecer pausado por stale_snapshot
# antes de forzar escalada a flexible_mode + advisory_only. CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS
# (4h) sigue siendo cuándo enviamos el primer push recordatorio. Tras este tope (24h por
# defecto) escalamos sí o sí, con un push de advertencia distinto. Permite mantener el
# preview-only del primer 24h sin generar a ciegas.
CHUNK_STALE_MAX_PAUSE_HOURS = max(
    int(os.environ.get("CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS", "4")),
    int(os.environ.get("CHUNK_STALE_MAX_PAUSE_HOURS", "24"))
)
# [P0-1] Deeplink que se envía en push notifications cuando pedimos al usuario que refresque
# su nevera. Default apunta a la sección de inventario; configurable para apps mobile que
# usen scheme propio (e.g., "mealfit://nevera/refresh").
CHUNK_STALE_PANTRY_DEEPLINK = os.environ.get("CHUNK_STALE_PANTRY_DEEPLINK", "/mi-nevera")
# [P1-4] Deeplink para zero-log pause. Apunta al diario donde el banner expone
# el toggle "Continuar sin registrar" (PUT /api/diary/preferences/logging) que activa
# auto_proxy y desbloquea los chunks pausados sin requerir log explícito del usuario.
# El default `/diario?banner=zero_log` es una sugerencia para frontends web; mobile
# puede usar scheme propio (e.g., "mealfit://diario?banner=zero_log").
CHUNK_ZERO_LOG_DEEPLINK = os.environ.get("CHUNK_ZERO_LOG_DEEPLINK", "/diario?banner=zero_log")
CHUNK_MAX_FAILURE_ATTEMPTS = max(2, int(os.environ.get("CHUNK_MAX_FAILURE_ATTEMPTS", "5")))
CHUNK_RETRY_BASE_MINUTES = max(1, int(os.environ.get("CHUNK_RETRY_BASE_MINUTES", "2")))
CHUNK_RETRY_CRITICAL_MINUTES = max(5, int(os.environ.get("CHUNK_RETRY_CRITICAL_MINUTES", "30")))
# [P0-1-RECOVERY] Edad mínima (en minutos) que un chunk debe llevar en status='failed' antes de
# que `_recover_failed_chunks_for_long_plans` lo re-encole. Antes el cron recogía chunks que
# acababan de fallar en el mismo tick que su última caída — sin dejar que el backoff exponencial
# (CHUNK_RETRY_BASE_MINUTES * 2^n) cumpliera su función. 120 min = 2h cubre el peor caso del
# último retry crítico (30m) con margen y permite que problemas transitorios (caída de Gemini,
# rate limit, timeout DB) se recuperen solos antes de que la red de seguridad intervenga.
CHUNK_RECOVERY_MIN_AGE_MINUTES = max(15, int(os.environ.get("CHUNK_RECOVERY_MIN_AGE_MINUTES", "120")))
# [P0-1-RECOVERY] Tope de ciclos de recovery sobre el MISMO chunk antes de escalar. Un chunk
# que falla 5 veces (CHUNK_MAX_FAILURE_ATTEMPTS) se marca 'failed'; el cron lo reactiva a
# 'pending' (=1 recovery). Si vuelve a agotar los 5 intentos y a fallar, segunda recovery
# (=2). Tras `CHUNK_MAX_RECOVERY_ATTEMPTS` reactivaciones, asumimos que el chunk no es
# salvable automáticamente: pausamos el plan, mandamos push al usuario explicando que
# regenere y dejamos el chunk en dead_letter permanente. Sin este tope el cron loopea
# indefinidamente cada CHUNK_SCHEDULER_INTERVAL_MINUTES sin que el usuario lo sepa.
CHUNK_MAX_RECOVERY_ATTEMPTS = max(1, int(os.environ.get("CHUNK_MAX_RECOVERY_ATTEMPTS", "2")))
# [P0-1-RECOVERY] Cap por corrida del cron de recovery: cuántos chunks failed re-encolar
# como mucho en cada tick de 15 min. Antes era literal 20 hardcoded; subido a const
# tunable para no saturar la DB si una caída del LLM deja cientos de chunks failed
# de golpe.
# [P1-KNOB-CLAMPS · 2026-05-26] Migrado a `_env_int(..., validator=...)` con clamp
# [1, 1000]. Pre-fix `max(1, int(...))` tenía floor pero NO ceiling — operador
# que setea `CHUNK_RECOVERY_BATCH_LIMIT=10000` causaba OOM/timeout del cron.
# Auto-registro en `_KNOBS_REGISTRY` (visible en `/health/version`).
CHUNK_RECOVERY_BATCH_LIMIT = _env_int(
    "CHUNK_RECOVERY_BATCH_LIMIT", 20, validator=lambda v: 1 <= v <= 1000
)
# [P0-4] Máxima edad permitida para el snapshot de pantry antes de intentar un live-retry adicional.
# Pasado este TTL, si el live sigue fallando se marca como stale_snapshot en los logs.
CHUNK_PANTRY_SNAPSHOT_TTL_HOURS = max(1, int(os.environ.get("CHUNK_PANTRY_SNAPSHOT_TTL_HOURS", "6")))
# [P1-δ/PANTRY-STALENESS-ALERT] Alerta SRE cuando hay chunks pending cuyo snapshot
# de pantry tiene edad >Nh — entre el TTL operacional (6h) y el umbral de
# auto-flexible (24h) hay una zona donde el snapshot es viejo y el live-fetch
# debe re-correr al pickup, pero si hay un patrón sistémico (live-fetch caído
# para muchos usuarios, refresh proactivo no corriendo) los chunks acumulan
# riesgo silenciosamente. 20h es el sweet spot: ya quedan ≥4h al umbral de
# flexible_mode forzado, pero suficiente lead-time para investigar antes de
# que muchos chunks degraden.
CHUNK_PANTRY_STALENESS_ALERT_HOURS = max(
    1,
    int(os.environ.get("CHUNK_PANTRY_STALENESS_ALERT_HOURS", "20")),
)
# CHUNK_PANTRY_STALENESS_ALERT_INTERVAL_MINUTES: cada cuánto corre el cron.
#   60 min alinea con la cadencia del refresh proactivo y evita ruido.
CHUNK_PANTRY_STALENESS_ALERT_INTERVAL_MINUTES = max(
    15,
    int(os.environ.get("CHUNK_PANTRY_STALENESS_ALERT_INTERVAL_MINUTES", "60")),
)
# CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT: mínimo de chunks stale para disparar.
#   3 evita ruido por chunks aislados (timing del refresh, usuario en pausa);
#   sube señal real (degradación sistémica del live-fetch).
CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT = max(
    1,
    int(os.environ.get("CHUNK_PANTRY_STALENESS_ALERT_MIN_COUNT", "3")),
)
# CHUNK_PANTRY_STALENESS_ALERT_COOLDOWN_HOURS: dedupe.
CHUNK_PANTRY_STALENESS_ALERT_COOLDOWN_HOURS = max(
    1,
    int(os.environ.get("CHUNK_PANTRY_STALENESS_ALERT_COOLDOWN_HOURS", "6")),
)
# [P2-δ/BG-ROLLING-REFILL] Frecuencia del cron `trigger_background_rolling_refill`.
# El cron detecta usuarios inactivos ≥3 días y dispara shift_plan en background
# para que sus chunks de rolling refill no esperen al próximo login. Originalmente
# corría 1 vez al día (1am UTC), lo que dejaba ventana de hasta 24h donde un
# usuario que se vuelve inactivo a las 2am no era atendido hasta el día siguiente
# y el chunk se quedaba hambriento de adherence-window. 4h es el sweet spot:
# detecta inactividad pronto, no spamea (el filtro `≥3 días sin sesión` evita
# re-procesar al mismo usuario rápidamente), y se alinea con la cadencia
# operacional típica de turnos SRE.
CHUNK_BG_ROLLING_REFILL_INTERVAL_HOURS = max(
    1,
    int(os.environ.get("CHUNK_BG_ROLLING_REFILL_INTERVAL_HOURS", "4")),
)
# [P0-2] Pasada esta edad (24h), si el live falla, forzamos generación con flexible_mode=True
# para no bloquear el plan indefinidamente si el live fetch del usuario está roto.
CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS = max(6, int(os.environ.get("CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS", "8")))
# [P0-5] Hard-fail cap absoluto sobre la edad del snapshot de pantry. Las rutas de escalada
# `_live_degraded_now → flex+advisory_only` en _refresh_chunk_pantry permiten generar contra
# el snapshot incluso después de fallos repetidos del live para mantener al usuario corriendo.
# Pero en planes de 30 días con `_proactive_refresh_pending_pantry_snapshots` caído, ese
# snapshot puede tener semanas: generar comidas contra un inventario tan viejo rompe el
# contrato "solo lo que hay en la nevera ahora". Este cap es el techo absoluto: por encima
# de CHUNK_PANTRY_HARD_FAIL_AGE_HOURS pausamos siempre, sin importar el estado degradado.
# Por construcción (`max(...)`), nunca puede ser menor que CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS:
# el hard-fail debe estar más allá de la frontera de force-generate, no antes.
CHUNK_PANTRY_HARD_FAIL_AGE_HOURS = max(
    CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS,
    int(os.environ.get("CHUNK_PANTRY_HARD_FAIL_AGE_HOURS", "48")),
)
# [P0-5] Switch para desactivar el hard-fail (e.g. en tests que ejercitan la rama
# flex+advisory de propósito). Default: encendido en producción.
CHUNK_PANTRY_HARD_FAIL_ON_STALE = os.environ.get(
    "CHUNK_PANTRY_HARD_FAIL_ON_STALE", "true"
).strip().lower() in {"1", "true", "yes", "on"}
# [P0-3 / P0-4] Timeout final (en segundos) para el live fetch bloqueante antes de abortar.
# [P0-4] Subido de 30→60: APIs de usuarios lentas con jitter pausaban chunks ~10% de las
# veces innecesariamente con 30s. 60s cubre p99 de la mayoría de despensas; el retry con
# backoff (CHUNK_LIVE_FETCH_BACKOFF_TIMEOUTS_SECONDS) cubre el resto.
CHUNK_STALE_FINAL_LIVE_TIMEOUT_SECONDS = int(os.environ.get("CHUNK_STALE_FINAL_LIVE_TIMEOUT_SECONDS", "60"))
# [P0-4] Lista CSV de timeouts (en segundos) para el retry con backoff del live-fetch en los
# paths de fallback (TZ-drift mayor + stale snapshot). Cada intento usa el siguiente timeout;
# si todos fallan, se considera que el live está caído. Empezar con timeouts bajos para no
# atrasar el chunk si la API responde rápido normalmente.
CHUNK_LIVE_FETCH_BACKOFF_TIMEOUTS_SECONDS = os.environ.get("CHUNK_LIVE_FETCH_BACKOFF_TIMEOUTS_SECONDS", "30,60,90")
# [P0-2] Detección de fallos sistémicos del live-fetch a nivel de usuario. Antes, si la API
# del usuario estaba caída, cada chunk pausaba y esperaba hasta CHUNK_STALE_MAX_PAUSE_HOURS
# (24h) por separado antes de escalar a flexible_mode — el usuario no recibía plan durante
# horas. Ahora rastreamos fallos consecutivos en user_profiles.health_profile; al llegar al
# threshold dentro de la ventana, escalamos a flex+advisory_only inmediatamente sin esperar
# el preview window. Reseteamos el contador en cada live-fetch exitoso.
CHUNK_LIVE_FETCH_DEGRADED_FAILURES_THRESHOLD = max(2, int(os.environ.get("CHUNK_LIVE_FETCH_DEGRADED_FAILURES_THRESHOLD", "3")))
CHUNK_LIVE_FETCH_DEGRADED_WINDOW_HOURS = max(1, int(os.environ.get("CHUNK_LIVE_FETCH_DEGRADED_WINDOW_HOURS", "12")))
# [P0-B / P1-1] Validación de CANTIDADES (no solo existencia) contra la despensa. Modos:
#   - "off"      : solo validamos existencia (comportamiento previo). Útil para debugging y
#                  para usuarios cuya despensa tiene cantidades imprecisas (e.g., medidas a ojo).
#                  No usar en producción salvo flag temporal.
#   - "advisory" : sumamos uso vs. inventario, logueamos violaciones y las anotamos en
#                  form_data["_pantry_quantity_violations"] para telemetría, pero NO bloqueamos
#                  el chunk. Útil cuando se prefiere UX "siempre genera plan" a costo de algunos
#                  platos que pidan más de lo disponible.
#   - "hybrid"   : reintentamos con feedback al LLM igual que strict, pero si los retries se
#                  agotan anotamos la violación y continuamos (no fallamos ni pausamos). DEFAULT.
#                  Es el balance recomendado: pelea por planes precisos en cantidades, pero no
#                  bloquea al usuario si el LLM no converge tras N retries.
#   - "strict"   : reintentamos con feedback. Si tras CHUNK_PANTRY_MAX_RETRIES el LLM sigue
#                  excediendo, PAUSAMOS el chunk en pending_user_action y notificamos. Solo
#                  recomendado si la integridad de cantidades es crítica (e.g., usuarios
#                  con restricciones médicas estrictas).
#
# [P0-C] Default es "strict": si el LLM no logra ajustarse a las cantidades de la nevera
# tras CHUNK_PANTRY_MAX_RETRIES, el chunk se pausa en `pending_user_action` con push
# notification para que el usuario actualice su nevera. Esto cumple el contrato explícito
# "los platos solo usan los alimentos que hay en la nevera". Hybrid (default previo)
# rompía ese contrato porque tras agotar retries seguía generando platos que excedían
# el stock disponible — solo anotaba la violación en `_pantry_quantity_violations` y
# continuaba.
#
# Override per-usuario: `health_profile._pantry_quantity_mode = "hybrid"|"advisory"|"off"`
# para perfiles que prefieran "siempre genera plan" sobre "solo lo que hay en la nevera"
# (e.g., usuarios cuya despensa tiene cantidades imprecisas a ojo).
#
# Tests que patchean modos específicos (`test_hybrid_mode_*`, `test_advisory_mode_*`)
# usan `with patch("cron_tasks.CHUNK_PANTRY_QUANTITY_MODE", ...)` y siguen funcionando.
CHUNK_PANTRY_QUANTITY_MODE = os.environ.get("CHUNK_PANTRY_QUANTITY_MODE", "strict").strip().lower()
if CHUNK_PANTRY_QUANTITY_MODE not in {"off", "advisory", "hybrid", "strict"}:
    CHUNK_PANTRY_QUANTITY_MODE = "strict"
CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE = float(os.environ.get("CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE", "1.05"))
# [P1-1] Reintentos máximos de re-prompt al LLM cuando la validación pantry post-LLM falla.
# Cada reintento envía `_pantry_correction` con la lista de ingredientes inválidos al pipeline
# para que el LLM corrija. range(CHUNK_PANTRY_MAX_RETRIES + 1) = 3 attempts totales (0,1,2)
# por default. Tras agotar, el chunk se PAUSA en `pending_user_action` con reason
# `pantry_violation_after_retries`, NO se marca 'failed': si marcáramos 'failed', el cron
# `_recover_failed_chunks_for_long_plans` lo re-encolaría y el LLM volvería a hallucinarse
# los mismos ingredientes con la misma nevera → ciclo de retries quemando tokens sin
# convergencia. La pausa requiere acción del usuario (refrescar nevera, clarificar items)
# para que el `_recover_pantry_paused_chunks` lo reanude. Antes este valor estaba hardcoded
# inline como `_PANTRY_MAX_RETRIES = 2` en cron_tasks.py y la docstring de
# CHUNK_PANTRY_QUANTITY_MODE arriba ya referenciaba la constante como si existiera —
# este es el rename canónico que cierra esa inconsistencia documental.
CHUNK_PANTRY_MAX_RETRIES = max(1, int(os.environ.get("CHUNK_PANTRY_MAX_RETRIES", "2")))
# [P1-D] Bounds para `user_profiles.pantry_tolerance` (override per-usuario sobre el
# default global). Por debajo de 1.00 sería rechazar incluso recetas que respetan
# exactamente la nevera (ruido); por encima de 1.50 (50% over-budget) sería tan laxo
# que el guardrail pierde sentido. Cualquier valor fuera de estos límites se clampa
# al cargar la preferencia del perfil. El CHECK constraint en DDL espeja estos bounds.
CHUNK_PANTRY_TOLERANCE_MIN = float(os.environ.get("CHUNK_PANTRY_TOLERANCE_MIN", "1.00"))
CHUNK_PANTRY_TOLERANCE_MAX = float(os.environ.get("CHUNK_PANTRY_TOLERANCE_MAX", "1.50"))
# [P0-C] Cada N minutos se refresca el snapshot de despensa en chunks pending/stale cuyo
# _pantry_captured_at supere la mitad del TTL. Esto evita que un chunk de plan 15d/30d
# llegue a su execute_after con snapshot de hace días y se quede pausado por stale_snapshot
# si el live fetch del worker falla puntualmente. Default: 180 min (3h, mitad del TTL=6h).
CHUNK_PANTRY_PROACTIVE_REFRESH_MINUTES = max(15, int(os.environ.get("CHUNK_PANTRY_PROACTIVE_REFRESH_MINUTES", "180")))
# Horizonte máximo (en horas) para targetear chunks: en planes de 30d permitimos un
# barrido proactivo de hasta 7 días; los planes de 7d/15d siguen usando 48h vía lógica
# dinámica en cron_tasks.py.
CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS = max(48, int(os.environ.get("CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS", "168")))
# Tope de usuarios procesados por corrida para no saturar la DB en picos.
CHUNK_PANTRY_PROACTIVE_REFRESH_MAX_USERS = max(1, int(os.environ.get("CHUNK_PANTRY_PROACTIVE_REFRESH_MAX_USERS", "50")))
# [P0-D] Mínimo de mutaciones de inventario (rows con updated_at >= prev_chunk_start) para
# considerar que el usuario está consumiendo el plan aunque no loguee comidas explícitamente.
# Solo se usa como fallback cuando zero_log_proxy=True; evita pausar chunks de usuarios
# activos pero no-logueadores. Subir si hay falsos positivos por edits manuales triviales.
# [P0-2] Bajado de 5→2: 5 era demasiado alto y bloqueaba a usuarios activos durante 12h.
CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS = max(1, int(os.environ.get("CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS", "2")))
# [P0-6] Límite de chunks consecutivos que pueden pasar el gate de aprendizaje usando únicamente
# el proxy de inventario. Si se supera, se pausa el chunk para forzar el registro manual.
CHUNK_MAX_CONSECUTIVE_PROXY_LEARNING = int(os.environ.get("CHUNK_MAX_CONSECUTIVE_PROXY_LEARNING", "2"))
# [P0-3] Defensa secundaria contra el patrón [proxy, strong, proxy, strong, ...] que antes
# reseteaba duro el contador consecutivo. Si la proporción acumulada de chunks-proxy en el plan
# supera este umbral, se fuerza pausa por chronic_zero_logging. Mínimo 4 chunks totales antes
# de evaluar el ratio para no penalizar planes cortos.
CHUNK_MAX_LIFETIME_PROXY_RATIO = float(os.environ.get("CHUNK_MAX_LIFETIME_PROXY_RATIO", "0.6"))
CHUNK_LIFETIME_PROXY_MIN_TOTAL = int(os.environ.get("CHUNK_LIFETIME_PROXY_MIN_TOTAL", "4"))
# ──────────────────────────────────────────────────────────────────────────────
# [P3-1 · 2026-05-10] SSOT del catálogo de keys de "learning" del chunk system.
#
# Las 3 keys conviven en `meal_plans.plan_data` y se inyectan al prompt del LLM
# en momentos distintos del pipeline. Antes de P3-1 no había un documento único
# explicando la diferencia → riesgo de doble-inyección o fallback incorrecto
# en recovery paths.
#
# ┌─────────────────────────────┬───────────────────────────────────────────────────┐
# │ Key                         │ Semántica + retención + inyección                 │
# ├─────────────────────────────┼───────────────────────────────────────────────────┤
# │ `_last_chunk_learning`      │ Dict con la lección del chunk N−1 EXACTAMENTE     │
# │                             │ (estructura `{chunk, lesson, ...}`). Sobrescrita  │
# │                             │ en cada chunk que completa. Se inyecta como "lo   │
# │                             │ que aprendí del último bloque" en el prompt del   │
# │                             │ chunk N. Fuente principal: pipeline post-merge    │
# │                             │ (`graph_orchestrator`). Fallback sintetizado:     │
# │                             │ `_synthesize_last_chunk_learning_from_plan_days`  │
# │                             │ cuando `learning_metrics` falta (chunk failed o   │
# │                             │ schema corrupto). Tamaño: 1 entry. NO survive el  │
# │                             │ siguiente chunk (sobrescrita).                    │
# ├─────────────────────────────┼───────────────────────────────────────────────────┤
# │ `_recent_chunk_lessons`     │ Rolling window de los últimos M chunks (M ≤       │
# │                             │ CHUNK_RECENT_LESSONS_MAX, default 12 — ver        │
# │                             │ definición abajo si existe, o derivado del flow   │
# │                             │ de `_rebuild_recent_chunk_lessons_from_queue`).   │
# │                             │ Lista append-cap. Se inyecta como "tendencias     │
# │                             │ recientes" en el prompt — el LLM ve patrones      │
# │                             │ multi-chunk vs solo el último.                    │
# │                             │ Recovery: si la key se corrompe,                  │
# │                             │ `_regenerate_recent_chunk_lessons_from_plan_days` │
# │                             │ sintetiza desde plan_days (señal degradada,       │
# │                             │ marcada como `synthesized=True` en cada entry).   │
# ├─────────────────────────────┼───────────────────────────────────────────────────┤
# │ `_lifetime_chunk_lessons`   │ Lecciones CRÍTICAS permanentes que sobreviven al  │
# │ (también llamado            │ rolling window. Cubren alergias confirmadas,      │
# │  `_critical_lessons_permanent`│ rechazos repetidos (≥                          │
# │  en algunos call sites      │ `CHUNK_CRITICAL_REPEATED_REJECTION_IMMORTAL`,     │
# │  legacy)                    │ default 3), fatiga extrema. Cap blando:           │
# │                             │ `CHUNK_CRITICAL_LESSONS_MAX` (200). Cap duro      │
# │                             │ inmortales: `CHUNK_CRITICAL_LESSONS_IMMORTAL_     │
# │                             │ HARD_CAP` (≈ MAX-10). LRU sobre inmortales        │
# │                             │ viejas SIN re-validación reciente                 │
# │                             │ (`CHUNK_CRITICAL_LESSONS_REVALIDATION_DAYS`,      │
# │                             │ default 60). Se inyectan SIEMPRE para planes      │
# │                             │ largos (15d/30d) en TODOS los chunks.             │
# └─────────────────────────────┴───────────────────────────────────────────────────┘
#
# Orden de inyección al prompt (mayor especificidad → menor):
#   1. `_last_chunk_learning` (lo más fresco, el chunk inmediato anterior).
#   2. `_recent_chunk_lessons` (las últimas M iteraciones).
#   3. `_lifetime_chunk_lessons` (constantes inmortales — alergias/rechazos).
# El LLM las ve concatenadas, pero el orden ayuda al weighting interno.
#
# Persistencia atómica: las 3 keys se modifican vía
# `update_plan_data_atomic(mutator)` (db_plans.py:215) con `SELECT … FOR UPDATE`
# para que dos chunks concurrentes no se sobrescriban entre sí. P1-LOCK-1
# (2026-05-10) garantiza lock_timeout + statement_timeout para no colgarse.
#
# Tests de regresión cruzados que validan que las 3 conviven bien:
#   - tests/test_p0_1_learning_atomicity.py — atomicidad de la mutación.
#   - tests/test_p0_3_legacy_learning_atomicity.py — fallback path.
#   - tests/test_p0_7_critical_lessons_cap.py — LRU + immortal cap.
#   - tests/test_chunked_learning_propagation.py — inyección al prompt.
#
# Si añades una 4ª key de learning, documenta aquí su retención + inyección.
# Drift no atrapado por tests si la lista se desactualiza silenciosamente.
# ──────────────────────────────────────────────────────────────────────────────

# [P0-7] Máximo de lecciones críticas permanentes (rechazos fuertes, alergias, fatiga extrema).
# Estas lecciones sobreviven al rolling window de _recent_chunk_lessons y se inyectan siempre
# al LLM para planes largos (15d/30d). Cap para evitar bloat infinito.
CHUNK_CRITICAL_LESSONS_MAX = int(os.environ.get("CHUNK_CRITICAL_LESSONS_MAX", "200"))
# [P0-6] Cap absoluto de lecciones INMORTALES (alergias / rechazos repetidos). Antes podían
# crecer sin tope al exceder CHUNK_CRITICAL_LESSONS_MAX (la poda devolvía todas), causando
# bloat de memoria + prompt overflow al LLM. Si los inmortales superan este límite, se aplica
# LRU sobre las inmortales más viejas SIN re-validación reciente. Default = MAX - 10 (deja
# 10 slots libres para nuevas señales no-inmortales).
CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP = int(
    os.environ.get(
        "CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP",
        str(max(10, CHUNK_CRITICAL_LESSONS_MAX - 10))
    )
)
# [P0-6] Ventana (en días) para considerar una lección "re-validada" recientemente. Inmortales
# con last_validated_at o created_at más reciente que esta ventana son protegidas del LRU duro;
# las más viejas son candidatas a poda si superan el hard cap.
CHUNK_CRITICAL_LESSONS_REVALIDATION_DAYS = int(os.environ.get("CHUNK_CRITICAL_LESSONS_REVALIDATION_DAYS", "60"))
# [P0-6] Umbral de violaciones de rechazo en un solo chunk para considerar la lección "inmortal"
# (no elegible para eviction, igual que las alergias). Por defecto: 3 hits = patrón confirmado.
CHUNK_CRITICAL_REPEATED_REJECTION_IMMORTAL = int(os.environ.get("CHUNK_CRITICAL_REPEATED_REJECTION_IMMORTAL", "3"))
# [P0-2] TTL específico (en horas) para pausas por learning_zero_logs.
# Más corto que CHUNK_PANTRY_EMPTY_TTL_HOURS (12h) porque la causa no es falta de nevera
# sino falta de registro de comidas; el sistema puede generar un chunk razonable sin señal
# si tiene al menos actividad de inventario. 4h da tiempo a 1 reminder antes de forzar.
CHUNK_ZERO_LOG_RECOVERY_TTL_HOURS = max(1, int(os.environ.get("CHUNK_ZERO_LOG_RECOVERY_TTL_HOURS", "6")))
# [P0-5] Umbral de drift (en minutos) entre tz_offset del snapshot y tz_offset vivo del user_profile.
# Antes era 60 — perdía cambios de DST de 30-60 min y drifts menores pero significativos.
# 15 min cubre DST estándar, viajes a TZs adyacentes (e.g., +30m: India, +45m: Nepal) y permite
# detectar correcciones manuales del usuario en su perfil.
CHUNK_TZ_DRIFT_THRESHOLD_MINUTES = int(os.environ.get("CHUNK_TZ_DRIFT_THRESHOLD_MINUTES", "15"))
# [P0-5] Cada cuánto el cron `_sync_chunk_queue_tz_offsets` reescanea chunks pending/stale
# para reflejar cambios de tz del user_profile. Sin este sync, un chunk encolado con
# tz_offset=-240 (RD) se dispara con esa zona aunque el usuario haya viajado y actualizado
# su perfil a -300 (USA Central) — el `execute_after` queda desplazado 1h y el learning gate
# cree que el día previo no terminó.
# [P0-4] Bajado de 60 → 15 min para alinear con CHUNK_TZ_DRIFT_THRESHOLD_MINUTES=15.
# Antes había mismatch 4×: el sync solo corría cada hora pero podíamos detectar drifts en
# 15m, lo que dejaba ventana de 60 min sin detección cuando el cambio de TZ ocurría por una
# ruta que NO pasaba por update_user_health_profile (migración, edición admin directa, o
# fast-path roto). El fast-path en db_profiles.py:137-142 sigue siendo la primera línea de
# defensa; este cron es la red de seguridad.
CHUNK_TZ_SYNC_INTERVAL_MINUTES = max(5, int(os.environ.get("CHUNK_TZ_SYNC_INTERVAL_MINUTES", "15")))
# [P1-4] Factor de decaimiento temporal de fatiga por ingrediente. weight_today = count * (decay ^ days_ago).
# Con 0.9 un item eaten hoy pesa 1.0; hace 7d pesa 0.48; hace 14d pesa 0.23; hace 30d pesa 0.04.
# Antes este valor estaba hardcoded en cron_tasks.calculate_ingredient_fatigue Y en
# db_plans.get_user_ingredient_frequencies — riesgo de drift si alguien edita uno y olvida el otro.
# Centralizarlo aquí garantiza que ambas rutas usen el mismo factor y permite tuning sin redeploy.
INGREDIENT_FATIGUE_DECAY_FACTOR = float(os.environ.get("INGREDIENT_FATIGUE_DECAY_FACTOR", "0.9"))
# [P1-4] Umbral absoluto de peso decayado para marcar un ingrediente como fatigado. Con decay=0.9
# y days_back=14, weight=4.0 equivale aproximadamente a "presente en ~5 días recientes". Subir
# para reducir falsos positivos, bajar para que la rotación sea más agresiva.
INGREDIENT_FATIGUE_INDIVIDUAL_THRESHOLD = float(os.environ.get("INGREDIENT_FATIGUE_INDIVIDUAL_THRESHOLD", "4.0"))
# [P1-4] Umbral relativo: si un ingrediente acumula >35% del peso total decayado del histórico,
# es fatiga incluso aunque no llegue al threshold absoluto (caso usuarios con poca data).
INGREDIENT_FATIGUE_INDIVIDUAL_RATIO = float(os.environ.get("INGREDIENT_FATIGUE_INDIVIDUAL_RATIO", "0.35"))
# [P1-4] Umbral absoluto para fatiga por categoría nutricional (más alto que el individual porque
# las categorías agregan múltiples ingredientes; e.g., "proteína animal" suma pollo + res + cerdo).
INGREDIENT_FATIGUE_CATEGORY_THRESHOLD = float(os.environ.get("INGREDIENT_FATIGUE_CATEGORY_THRESHOLD", "6.0"))
# [P1-4] Umbral relativo para fatiga por categoría: 45% del peso total.
INGREDIENT_FATIGUE_CATEGORY_RATIO = float(os.environ.get("INGREDIENT_FATIGUE_CATEGORY_RATIO", "0.45"))
# [P1-4/AUTO-TUNE] Parámetros del auto-tuner de fatigue_decay
# (cron_tasks._auto_tune_fatigue_decay, antes hardcoded). El auto-tuner ajusta el
# decay por usuario observando la tasa de falsos positivos (ingredientes "fatigados"
# que el usuario sí comió) entre ciclos:
#   - Si FP rate > FP_HIGH_THRESHOLD → bajar decay (olvidar más rápido).
#   - Si FP rate < FP_LOW_THRESHOLD → subir decay (memoria más larga).
# Los clamps definen el rango válido para que el auto-tuner no arrastre el decay a
# valores absurdos (e.g., 0.5 borraría toda fatiga; 0.99 nunca olvidaría nada).
# Antes estos eran literales en cron_tasks.py:3514-3532 y un cambio del env
# INGREDIENT_FATIGUE_DECAY_FACTOR no se reflejaba en el fallback de usuarios nuevos.
INGREDIENT_FATIGUE_DECAY_LOWER_CLAMP = float(os.environ.get("INGREDIENT_FATIGUE_DECAY_LOWER_CLAMP", "0.70"))
INGREDIENT_FATIGUE_DECAY_UPPER_CLAMP = float(os.environ.get("INGREDIENT_FATIGUE_DECAY_UPPER_CLAMP", "0.98"))
INGREDIENT_FATIGUE_FP_HIGH_THRESHOLD = float(os.environ.get("INGREDIENT_FATIGUE_FP_HIGH_THRESHOLD", "0.6"))
INGREDIENT_FATIGUE_FP_LOW_THRESHOLD = float(os.environ.get("INGREDIENT_FATIGUE_FP_LOW_THRESHOLD", "0.2"))
INGREDIENT_FATIGUE_DECAY_STEP_DOWN = float(os.environ.get("INGREDIENT_FATIGUE_DECAY_STEP_DOWN", "0.03"))
INGREDIENT_FATIGUE_DECAY_STEP_UP = float(os.environ.get("INGREDIENT_FATIGUE_DECAY_STEP_UP", "0.02"))
# [P0-2-EXT] Drift "mayor" (>= 60 min). Indica cambio de país / viaje real (no DST ni ajuste fino).
# Cuando ocurre durante el refresh de pantry, escalamos: si el live falla, NO usamos el snapshot
# (su "hoy" del usuario ya cambió y validar cantidades contra inventario consumido en el día previo
# es riesgoso). En su lugar forzamos un live-retry con timeout extendido y, si también falla,
# pausamos el chunk con reason='tz_major_drift_live_unreachable' para que el usuario refresque.
CHUNK_TZ_MAJOR_DRIFT_MINUTES = int(os.environ.get("CHUNK_TZ_MAJOR_DRIFT_MINUTES", "60"))
# [P0-5] Tras N deferrals consecutivos por temporal_gate, forzar re-sync de TZ aunque el drift
# esté por debajo del umbral. Es nuestra red de seguridad: si el chunk lleva muchas horas sin
# avanzar, asumimos que el TZ del snapshot puede estar incorrecto y refrescamos.
CHUNK_TZ_FORCED_RESYNC_DEFERRALS = int(os.environ.get("CHUNK_TZ_FORCED_RESYNC_DEFERRALS", "3"))
# [P1-2] Detección crónica de deferrals: si un (user, plan, week) acumula >= MIN_COUNT
# deferrals en WINDOW_HOURS, enviamos un push proactivo sugiriendo revisar TZ.
# COOLDOWN_HOURS evita spam (1 push máx por user en esa ventana).
CHUNK_CHRONIC_DEFERRAL_MIN_COUNT = int(os.environ.get("CHUNK_CHRONIC_DEFERRAL_MIN_COUNT", "5"))
CHUNK_CHRONIC_DEFERRAL_WINDOW_HOURS = int(os.environ.get("CHUNK_CHRONIC_DEFERRAL_WINDOW_HOURS", "48"))
CHUNK_CHRONIC_DEFERRAL_NOTIFY_COOLDOWN_HOURS = int(os.environ.get("CHUNK_CHRONIC_DEFERRAL_NOTIFY_COOLDOWN_HOURS", "24"))
# [P1-2] Cron: cada cuántos minutos correr _detect_chronic_deferrals. 360 (6h) es suficiente
# porque el cooldown es 24h; correr más a menudo solo desperdicia queries.
CHUNK_CHRONIC_DEFERRAL_CHECK_INTERVAL_MINUTES = int(os.environ.get("CHUNK_CHRONIC_DEFERRAL_CHECK_INTERVAL_MINUTES", "360"))
# [P0-A] Telemetría de lecciones sintetizadas (low-confidence) vs. queue-based.
# Cada vez que el sistema cae a `_synthesize_last_chunk_learning_from_plan_days` o regenera
# `_recent_chunk_lessons` con entradas sintetizadas (porque `plan_chunk_queue.learning_metrics`
# está NULL para los chunks anteriores), registramos un evento en `chunk_lesson_telemetry`.
# Sin esta señal, el aprendizaje continuo podía estar degradado en producción sin alerta:
# el LLM recibe lecciones marcadas low_confidence con counters en cero, y los platos del
# chunk N+1 no respondían realmente a las repeticiones del N.
#
# CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD: porcentaje de chunks que pueden depender de
#   síntesis low-confidence antes de disparar alerta. 0.15 (15%) baja desde el 0.20
#   inicial tras el audit P0-β: el threshold previo permitía que 1 de cada 5 chunks
#   usara learning sintético antes de avisar — lo suficiente para que un bug latente
#   en el commit de learning_metrics afectara semanas de planes 7d antes de visibilizarse.
#   15% sigue siendo permisivo respecto al ruido natural (chunks recientes con la cola
#   aún sin commit, deferrals legítimos por zero-log) pero detecta degradación sistémica
#   (release que rompe la persistencia, schema downgrade) con ~5 chunks afectados de 30,
#   no 6. Per-user circuit breaker (CHUNK_SYNTH_PER_USER_RATIO_THRESHOLD) sigue en 0.30
#   porque la varianza individual es naturalmente mayor.
CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD = float(os.environ.get("CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD", "0.15"))
# CHUNK_LESSON_SYNTH_MIN_SAMPLES: mínimo de chunks procesados en la ventana antes de
#   evaluar el ratio. Evita falsos positivos en días de bajo tráfico (e.g., 1 chunk
#   sintetizado de 2 totales = 50% pero no significa nada con n=2).
CHUNK_LESSON_SYNTH_MIN_SAMPLES = int(os.environ.get("CHUNK_LESSON_SYNTH_MIN_SAMPLES", "10"))
# CHUNK_LESSON_SYNTH_WINDOW_HOURS: ventana sobre la cual se calcula el ratio. 24h es la
#   ventana natural: captura ciclos diarios de chunks (la mayoría se procesa en la madrugada
#   local del usuario) sin diluir señales con datos viejos.
CHUNK_LESSON_SYNTH_WINDOW_HOURS = int(os.environ.get("CHUNK_LESSON_SYNTH_WINDOW_HOURS", "24"))
# CHUNK_LESSON_SYNTH_ALERT_INTERVAL_MINUTES: cada cuánto corre el cron de evaluación.
#   360 (6h) alinea con el patrón de _detect_chronic_deferrals; el cooldown del alert
#   evita spam si el cron corre más a menudo que el ALERT_COOLDOWN_HOURS.
CHUNK_LESSON_SYNTH_ALERT_INTERVAL_MINUTES = int(os.environ.get("CHUNK_LESSON_SYNTH_ALERT_INTERVAL_MINUTES", "360"))
# CHUNK_LESSON_SYNTH_ALERT_COOLDOWN_HOURS: dedupe de la alerta. 24h es razonable: si el
#   ratio sigue alto al día siguiente, queremos re-alertar (no silenciar permanentemente).
CHUNK_LESSON_SYNTH_ALERT_COOLDOWN_HOURS = int(os.environ.get("CHUNK_LESSON_SYNTH_ALERT_COOLDOWN_HOURS", "24"))

# [P0-B] Per-user circuit breaker: cuando un USUARIO específico está cayendo en el
# camino de síntesis low-confidence con ratio alto, pausamos su próximo chunk antes
# de despacharlo al LLM con señal pobre. La alerta system-wide (CHUNK_LESSON_SYNTH_*)
# detecta el problema agregado pero no protege al usuario individual: el chunk seguía
# generándose y propagaba aprendizaje degradado al chunk N+2, N+3, etc.
#
# La acción es complementaria, no reemplaza la alerta system-wide:
#   - Alerta agregada → SRE investiga la causa raíz (commit de learning_metrics roto,
#     downgrade de schema, etc.).
#   - Pause per-usuario → ese usuario no genera más planes degradados hasta que
#     intervención manual (operador o usuario re-abre app) destrabe.
#
# CHUNK_SYNTH_PER_USER_RATIO_THRESHOLD: porcentaje de chunks sintetizados por usuario
#   que dispara el circuit breaker. 0.30 (30%) es más permisivo que el threshold
#   system-wide (CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD=0.15, ver línea ~654 —
#   bajado desde 0.20 inicial post audit P0-β) porque per-user tiene varianza
#   natural mayor: un usuario que no logueó comidas dos días seguidos puede
#   legítimamente cruzar 25% sin que sea degradación sistémica.
CHUNK_SYNTH_PER_USER_RATIO_THRESHOLD = float(os.environ.get("CHUNK_SYNTH_PER_USER_RATIO_THRESHOLD", "0.30"))
# CHUNK_SYNTH_PER_USER_MIN_SAMPLES: mínimo de chunks del usuario en la ventana antes
#   de evaluar el ratio. 4 cubre los primeros 2 chunks de un plan 7d (3+4) y evita
#   falsos positivos en planes recién creados (1/1 = 100% es trivialmente alto).
CHUNK_SYNTH_PER_USER_MIN_SAMPLES = int(os.environ.get("CHUNK_SYNTH_PER_USER_MIN_SAMPLES", "4"))
# CHUNK_SYNTH_PER_USER_WINDOW_HOURS: ventana de evaluación per-usuario. 72h captura
#   2-3 chunks completos de un plan típico, suficiente para detectar patrón sin
#   reaccionar a un blip aislado.
CHUNK_SYNTH_PER_USER_WINDOW_HOURS = int(os.environ.get("CHUNK_SYNTH_PER_USER_WINDOW_HOURS", "72"))
# CHUNK_SYNTH_PER_USER_PAUSE_COOLDOWN_HOURS: tras una pausa per-usuario, no volver a
#   evaluar/pausar la misma combinación (user, plan, week_number) por este período.
#   Evita re-pausar al mismo chunk si el cron lo recoge tras override manual y el
#   ratio histórico todavía está por encima del umbral.
CHUNK_SYNTH_PER_USER_PAUSE_COOLDOWN_HOURS = int(os.environ.get("CHUNK_SYNTH_PER_USER_PAUSE_COOLDOWN_HOURS", "12"))
# [P1-2/DEAD-LETTER-ALERT] Alerta proactiva sobre chunks dead-lettered acumulados.
# Antes los chunks que `_escalate_unrecoverable_chunk` (cron_tasks.py:5631) marcaba como
# dead_lettered_at se acumulaban invisibles a soporte: solo el usuario afectado recibía
# el push (con copy "regenera tu plan"). Operadores no veían el conteo agregado, ni los
# reasons predominantes — un release malo podía dead-letterar decenas de planes sin
# levantar ninguna señal hasta que el usuario abría un ticket.
#
# Este cron evalúa la ventana CHUNK_DEAD_LETTER_ALERT_WINDOW_HOURS, agrega por
# `dead_letter_reason` y, si el conteo ≥ CHUNK_DEAD_LETTER_ALERT_MIN_COUNT, inserta una
# alerta deduplicada en `system_alerts` (alert_key='dead_lettered_chunks_recent').
# El endpoint `/api/plans/admin/chunks/dead-lettered` permite inspeccionar las filas
# concretas tras recibir la alerta.
#
# CHUNK_DEAD_LETTER_ALERT_INTERVAL_MINUTES: cada cuánto corre el cron. 60 min es el
#   mínimo razonable: dead-letters son terminales (el usuario está bloqueado), pero
#   correr sub-hora satura logs sin acelerar la detección — el cooldown del alert
#   corta el spam de todos modos.
CHUNK_DEAD_LETTER_ALERT_INTERVAL_MINUTES = max(15, int(os.environ.get("CHUNK_DEAD_LETTER_ALERT_INTERVAL_MINUTES", "60")))
# CHUNK_DEAD_LETTER_ALERT_WINDOW_HOURS: cuánto hacia atrás escanea cada corrida.
#   1h captura el último ciclo del cron (con margen) sin acumular dead-letters viejos
#   ya conocidos. Si el operador pierde la alerta por un cooldown, el endpoint admin
#   permite consultar la ventana entera con ?window_hours=N.
CHUNK_DEAD_LETTER_ALERT_WINDOW_HOURS = max(1, int(os.environ.get("CHUNK_DEAD_LETTER_ALERT_WINDOW_HOURS", "1")))
# CHUNK_DEAD_LETTER_ALERT_COOLDOWN_HOURS: dedupe del alert. 6h evita re-alertar mientras
#   los mismos dead-lettered siguen acumulándose; al expirar, si la situación persiste,
#   se vuelve a disparar.
CHUNK_DEAD_LETTER_ALERT_COOLDOWN_HOURS = max(1, int(os.environ.get("CHUNK_DEAD_LETTER_ALERT_COOLDOWN_HOURS", "6")))
# CHUNK_DEAD_LETTER_ALERT_MIN_COUNT: mínimo de chunks dead-lettered en ventana para
#   disparar alerta. Default 1 (cualquier dead-letter es señal). Subir a 3-5 si la
#   flota es grande y un dead-letter aislado es ruido aceptable.
CHUNK_DEAD_LETTER_ALERT_MIN_COUNT = max(1, int(os.environ.get("CHUNK_DEAD_LETTER_ALERT_MIN_COUNT", "1")))

# [P3-CHUNK-GC-DEADLETTER · 2026-05-18] GC de chunks dead-lettered antiguos.
# `plan_chunk_queue` acumula filas `status='failed' AND dead_lettered_at IS NOT NULL`
# indefinidamente — el cron `_alert_new_dead_lettered_chunks` solo alerta, NO purga.
# Pre-fix, una flota con dead-letter rate sostenido (incidente de modelo LLM, racha
# de planes con pantry vacía) acumulaba filas para siempre, inflando index size
# (idx_plan_chunk_queue_status, idx_plan_chunk_queue_user_id) y degradando pickup
# perf O(N) en cada tick del worker.
#
# Patrón espejo del existente "purge cancelled >48h" (cron_tasks.py:20039), pero con
# TTL más conservador porque dead-lettered son forensic-interesantes (un SRE puede
# querer correlacionar dead-letter de hace 1-2 semanas con incidentes recientes).
# Default 30 días = 1 ciclo de plan mensual completo + margen.
#
# CHUNK_GC_DEAD_LETTER_TTL_DAYS: edad mínima antes de purgar. 30d = forensic window
#   razonable. Cap [7, 365] para evitar valores agresivos (<7d perderían contexto
#   post-mortem) o paranoicos (>365d ya no es GC, es archivado).
CHUNK_GC_DEAD_LETTER_TTL_DAYS = max(7, min(365, int(os.environ.get("CHUNK_GC_DEAD_LETTER_TTL_DAYS", "30"))))
# CHUNK_GC_DEAD_LETTER_INTERVAL_HOURS: frecuencia del cron. 24h es plenty — dead-letter
#   GC no es hot-path; correr más seguido satura el log sin acelerar nada. Cap [1, 168]
#   = [1h, 1 semana].
CHUNK_GC_DEAD_LETTER_INTERVAL_HOURS = max(1, min(168, int(os.environ.get("CHUNK_GC_DEAD_LETTER_INTERVAL_HOURS", "24"))))
# CHUNK_GC_DEAD_LETTER_BATCH: límite por corrida para evitar lock contention con
#   pickups en `plan_chunk_queue`. 1000 es seguro (DELETE de 1000 filas terminales
#   ~50ms en idx). Cap [10, 10000].
CHUNK_GC_DEAD_LETTER_BATCH = max(10, min(10000, int(os.environ.get("CHUNK_GC_DEAD_LETTER_BATCH", "1000"))))

# [P1-PANTRY-GUARD-INITIAL-SKIP · 2026-05-18] Umbral mínimo de items en la nevera
# del usuario para activar el pantry guard estricto en la generación inicial del
# plan (`/api/plans/analyze[/stream]`).
#
# Problema reportado por user 2026-05-18: el guard rechazaba planes recién
# generados cuando la nevera tenía pocos items (3-9), forzando 2-3 retries del
# pipeline LLM por plan (costo ~$0.30 + 5-10 min wall-clock). El usuario tampoco
# puede "llenar la nevera" mientras está dándole "regenerar plan" — ese flujo no
# tiene sentido porque la lista de compras del plan ES LA QUE DEFINE qué comprar.
#
# Comportamiento pre-fix:
#   - Nevera con 0 items → SKIP guard (corto-circuito sano).
#   - Nevera con 1-N items → APLICA guard estricto → rechaza ingredientes
#     legítimos del plan inicial → retries innecesarios.
#
# Comportamiento post-fix:
#   - Nevera con <PANTRY_GUARD_MIN_ITEMS → SKIP guard (mismo path que vacía).
#   - Nevera con ≥PANTRY_GUARD_MIN_ITEMS → APLICA guard (típico: ciclo de compras
#     vivo, swap/refill esperan compatibilidad).
#
# Default 10 = "nevera mínimamente poblada para un ciclo activo". Usuarios típicos
# tras su primera compra tienen 15-30 items. Por debajo de 10, está en generación
# inicial o nevera casi vacía → la lista de compras del plan define el inventario,
# no al revés.
#
# Knob ajustable sin redeploy. Si quieres desactivar el guard entero para flujos
# iniciales, sube a 100. Si quieres preservar comportamiento legacy estricto,
# bájalo a 1.
PANTRY_GUARD_MIN_ITEMS = max(0, min(500, _env_int("MEALFIT_PANTRY_GUARD_MIN_ITEMS", 10)))  # [P2-1-KNOBS-HYGIENE · 2026-06-15] vía helper, no os.environ raw

# [P1-RENEWAL-PANTRY-IGNORE · 2026-06-26] Variety-first en la generación de plan COMPLETO
# (`/api/plans/analyze[/stream]`): una renovación genera un menú NUEVO, y la lista de compras del
# plan ES LA QUE DEFINE qué comprar — NO se amarra a la nevera previa. Por eso el guard estricto de
# nevera NO debe correr en este endpoint por default (matchea la intención documentada "Renovar Ciclo
# ignora nevera"; los flujos pantry-aware reales —Cambiar Plato `/swap-meal`, día completo
# `/regenerate-day`— son endpoints SEPARADOS que NO pasan por `_run_pantry_validation_for_initial_chunk`).
#
# Por qué el default es OFF (skip): el skip previo (P1-PANTRY-GUARD-REGEN-SKIP) dependía de que el
# request llevara `update_reason` en el payload — señal LEAKY: si el usuario renovaba por un entry-point
# que no setea `update_reason` (no pasa por el modal Actualizar), el guard estricto se aplicaba a la
# nevera poblada → rechazaba el plan variado nuevo → tras agotar retries entregaba un plan matemático
# de emergencia band-0.0. Incidente real user d4bc3af5 (corr=9040fc1d, 2026-06-26 07:00): 39 ítems en
# nevera + update_reason ausente → `[PANTRY GUARD] RECHAZO unauthorized=15` → retries agotados → plan
# `_is_fallback` band-0.0 entregado. Una nevera ≥ PANTRY_GUARD_MIN_ITEMS SIEMPRE implica un plan previo +
# restock ("Ya compré la lista"), o sea SIEMPRE es un renew → debe ignorar la nevera.
#
# Flip a True → restaura el guard estricto en la generación inicial (comportamiento previo, gated por
# update_reason/min-items). Tooltip-anchor: P1-RENEWAL-PANTRY-IGNORE.
INITIAL_CHUNK_PANTRY_GUARD_ENABLED = _env_bool("MEALFIT_INITIAL_CHUNK_PANTRY_GUARD", False)

# [P1-RENEWAL-PANTRY-AWARE · 2026-06-28] Modo "completar nevera" en la renovación.
# Objetivo del owner: al Renovar plan → variedad (intacta) + REUSAR los DURADEROS
# sobrantes de la nevera como SUGERENCIA + (Fase 2) lista de SOLO los faltantes
# (perecederos) para tener la nevera al 100%. Diseño verificado contra el código
# (workflow renovar-plan-pantry-audit, 2026-06-28).
#
# Tensión resuelta (band-0.0, incidente d4bc3af5): el reuso de nevera como GATE
# colapsaba la variedad y degradaba planes (por eso P1-RENEWAL-PANTRY-IGNORE apagó
# la nevera en renovación). Aquí el reuso entra SOLO como prompt-hint ADVISORY:
# se mantiene update_reason='variety' (el reviewer SIGUE saltando la validación de
# despensa, P1-VARIETY-IGNORE-PANTRY) y los duraderos van como sugerencia capada,
# NUNCA 'OBLIGATORIO agotar'. Así es estructuralmente imposible reintroducir el gate.
#
# Master switch, default OFF (rollout incremental, rollback sin redeploy). Cuando ON
# Y el request trae `_renewal_pantry_aware=True`, build_pantry_context emite el bloque
# de duraderos advisory en vez de cadena vacía.
RENEWAL_PANTRY_AWARE_ENABLED = _env_bool("MEALFIT_RENEWAL_PANTRY_AWARE_ENABLED", False)
# Cap de duraderos inyectados al prompt: pocos = más variedad, más = más reuso.
# Default conservador 8 (evita inflar el prompt o colapsar variedad). Clamp [0, 50].
RENEWAL_DURABLE_HINT_MAX_ITEMS = max(0, min(50, _env_int("MEALFIT_RENEWAL_DURABLE_HINT_MAX_ITEMS", 8)))
# [Fase 2] Lista de faltantes (pantry_completion_list) derivada READ-ONLY post-plan:
# lo que el plan necesita y la nevera NO cubre (perecederos consumidos). Default OFF.
PANTRY_COMPLETION_LIST_ENABLED = _env_bool("MEALFIT_PANTRY_COMPLETION_LIST_ENABLED", False)

# [P2-6 · 2026-05-08] Alertas proactivas sobre fallback no-atómico del pool.
# `update_user_health_profile_atomic` cae al path legacy (get + update) cuando
# `connection_pool=None` y `MEALFIT_REQUIRE_ATOMIC_POOL≠1` (default). Ese
# fallback puede producir lost-updates bajo concurrencia silenciosamente.
# Hasta P2-6 el counter solo era visible vía `/api/system/atomic-pool-health`
# (polling manual). Este cron lo escala a alerta automática.
#
# POOL_FALLBACK_ALERT_INTERVAL_MINUTES: frecuencia del cron. Default 30 min
#   balanceando overhead (es solo lectura del snapshot in-memory + INSERT
#   condicional) y latencia de detección (≤30 min para ver primer fallback).
POOL_FALLBACK_ALERT_INTERVAL_MINUTES = max(5, _env_int("MEALFIT_POOL_FALLBACK_ALERT_INTERVAL_MINUTES", 30))
# POOL_FALLBACK_ALERT_WINDOW_MINUTES: cuán reciente debe ser el último
#   fallback (`last_at`) para considerar la situación "activa". 60 min cubre
#   2 ticks del cron por defecto — si el pool se recuperó hace >1h sin nuevos
#   fallbacks, la alerta se auto-resuelve via self-healing sweep.
POOL_FALLBACK_ALERT_WINDOW_MINUTES = max(5, _env_int("MEALFIT_POOL_FALLBACK_ALERT_WINDOW_MINUTES", 60))
# POOL_FALLBACK_ALERT_COOLDOWN_HOURS: dedupe entre re-emisiones. 1h porque la
#   situación es accionable (operator debe verificar pool init en deploy);
#   no queremos esperar 6h como con dead_letters.
POOL_FALLBACK_ALERT_COOLDOWN_HOURS = max(1, _env_int("MEALFIT_POOL_FALLBACK_ALERT_COOLDOWN_HOURS", 1))
# [P1-CHUNKS-3] Escalación de chunks pausados con `_pause_reason='missing_prior_lessons'`.
# Antes este pause reason tenía TTL "indefinido" en el state-machine (ver doc-string de
# `process_plan_chunk_queue` en cron_tasks.py:12750+): un chunk pausado por lecciones
# previas perdidas (P1-1 auto-recovery agotado + P1-2 regeneración insuficiente)
# quedaba esperando "revisión humana" sin alarma proactiva ni acción automática.
# `_alert_new_dead_lettered_chunks` solo cubre chunks ya `dead_lettered_at`, NO los
# que llevan días en `pending_user_action`. Resultado: usuario sin chunks nuevos +
# SRE sin señal hasta ticket de soporte.
#
# Este cron corre cada CHUNK_INDEFINITE_PAUSE_INTERVAL_MINUTES y aplica dos fases:
#   Fase 1 (>= ALERT_HOURS, < ESCALATE_HOURS): persiste alerta `system_alerts`
#     (severity=warning, alert_key per-chunk) para que SRE detecte el patrón.
#   Fase 2 (>= ESCALATE_HOURS): intenta unblock automático corriendo
#     `_rebuild_recent_chunk_lessons_from_queue` + `_regenerate_recent_chunk_lessons_from_plan_days`
#     una última vez. Si combined >= expected → flip status='pending'. Si insuficiente →
#     dead-letter directo vía `_escalate_unrecoverable_chunk` con
#     reason='missing_prior_lessons_unrecoverable' (ya hay copy específico
#     y el usuario recibe push + banner).
#
# CHUNK_INDEFINITE_PAUSE_INTERVAL_MINUTES: cada cuánto corre el cron. 60 min alinea con
#   `_alert_new_dead_lettered_chunks`: pausas indefinidas no son urgentes (el plan ya
#   está bloqueado), correr sub-hora solo satura logs.
CHUNK_INDEFINITE_PAUSE_INTERVAL_MINUTES = max(15, int(os.environ.get("CHUNK_INDEFINITE_PAUSE_INTERVAL_MINUTES", "60")))
# CHUNK_INDEFINITE_PAUSE_ALERT_HOURS: edad mínima del pause antes de emitir alert.
#   12h da margen razonable a operadores que estén investigando manualmente sin que
#   el chunk lleve días invisible. Bajar reduce el TTFR (time to first response) pero
#   sube el ruido si SRE ya está sobre el caso.
CHUNK_INDEFINITE_PAUSE_ALERT_HOURS = max(1, int(os.environ.get("CHUNK_INDEFINITE_PAUSE_ALERT_HOURS", "12")))
# CHUNK_INDEFINITE_PAUSE_ESCALATE_HOURS: edad mínima del pause para intentar unblock
#   automático y, si falla, dead-letterar. 24h da MÁS margen al operador antes del
#   auto-action (preferimos resolución manual cuando es posible). Subir si los
#   operadores reportan que necesitan más tiempo; bajar si el TTL del usuario
#   esperando un plan es prioridad.
CHUNK_INDEFINITE_PAUSE_ESCALATE_HOURS = max(2, int(os.environ.get("CHUNK_INDEFINITE_PAUSE_ESCALATE_HOURS", "24")))
# CHUNK_INDEFINITE_PAUSE_BATCH_LIMIT: cap por corrida. 50 cubre flota mediana sin
#   bloquear el scheduler en una corrida larga si el backlog crece tras un incidente.
CHUNK_INDEFINITE_PAUSE_BATCH_LIMIT = max(1, int(os.environ.get("CHUNK_INDEFINITE_PAUSE_BATCH_LIMIT", "50")))
# [P0-A/ZOMBIE-PARTIAL] Cierre de planes que se quedan en `generation_status='partial'`
# para siempre porque todos los chunks restantes terminaron en estados terminales
# (cancelled / dead-lettered failed) sin que ninguno haya commiteado el merge final.
# Sin este cron, el frontend mostraba "generando próximos días…" indefinidamente.
#
# CHUNK_ZOMBIE_PARTIAL_MIN_AGE_HOURS: edad mínima del plan antes de considerarlo zombie.
#   24h da margen al recovery loop normal (CHUNK_RECOVERY_MIN_AGE_MINUTES=120 + hasta
#   CHUNK_MAX_RECOVERY_ATTEMPTS=2 reintentos × backoff) para resolverse antes de que
#   intervengamos. Bajarlo dispararía finalizaciones prematuras sobre planes que
#   todavía pueden completarse.
CHUNK_ZOMBIE_PARTIAL_MIN_AGE_HOURS = max(1, int(os.environ.get("CHUNK_ZOMBIE_PARTIAL_MIN_AGE_HOURS", "24")))
# CHUNK_ZOMBIE_PARTIAL_FINALIZE_INTERVAL_MINUTES: frecuencia del cron. 60 min es suficiente:
#   los planes zombie no son urgentes (el usuario ya tiene parte de los días generados);
#   correr cada 5 min sería desperdicio.
CHUNK_ZOMBIE_PARTIAL_FINALIZE_INTERVAL_MINUTES = max(15, int(os.environ.get("CHUNK_ZOMBIE_PARTIAL_FINALIZE_INTERVAL_MINUTES", "60")))
# CHUNK_ZOMBIE_PARTIAL_BATCH_LIMIT: cap por corrida. 50 es alto pero realista en backlogs
#   tras incidentes de LLM down: cuando vuelve, podríamos tener decenas de planes a finalizar.
CHUNK_ZOMBIE_PARTIAL_BATCH_LIMIT = max(1, int(os.environ.get("CHUNK_ZOMBIE_PARTIAL_BATCH_LIMIT", "50")))
# [P1-A/ORPHAN-RESERVATIONS] Cron de cleanup de reservas de inventario huérfanas. Aunque
# `release_chunk_reservations` ahora es atómico (P1-A), pueden quedar reservas legacy
# de incidentes anteriores donde la liberación fue parcial. También cubre el caso
# patológico donde el caller olvidó invocar `release_chunk_reservations` antes de
# cancelar el chunk (defensa en profundidad).
#
# CHUNK_ORPHAN_RESERVATION_CLEANUP_INTERVAL_MINUTES: cada cuánto corre el cron. 60 min
#   es razonable: las reservas huérfanas no son urgentes (no rompen funcionalidad —
#   solo bloquean stock que el usuario podría haber reutilizado).
CHUNK_ORPHAN_RESERVATION_CLEANUP_INTERVAL_MINUTES = max(15, int(os.environ.get("CHUNK_ORPHAN_RESERVATION_CLEANUP_INTERVAL_MINUTES", "60")))
# CHUNK_ORPHAN_RESERVATION_BATCH_LIMIT: cuántas filas user_inventory escanea por corrida.
#   100 cubre flotas de tamaño medio; chunks por usuario × usuarios activos cabe en este
#   límite con margen. Subir si el cron acumula backlog.
CHUNK_ORPHAN_RESERVATION_BATCH_LIMIT = max(10, int(os.environ.get("CHUNK_ORPHAN_RESERVATION_BATCH_LIMIT", "100")))
# CHUNK_ORPHAN_RESERVATION_MIN_TERMINAL_AGE_HOURS: edad mínima del chunk en estado
#   terminal antes de limpiarlo. 1h evita race con `release_chunk_reservations` que
#   acaba de ejecutarse o con flujos de finalización en progreso.
CHUNK_ORPHAN_RESERVATION_MIN_TERMINAL_AGE_HOURS = max(1, int(os.environ.get("CHUNK_ORPHAN_RESERVATION_MIN_TERMINAL_AGE_HOURS", "1")))
# [P1-2/ZERO-LOG-NUDGE] Aviso PROACTIVO al usuario antes de que el chunk N+1 falle por
# zero-log. Antes el push solo se mandaba cuando el chunk N+1 ya estaba defiriéndose o
# pausado — el usuario descubría el problema cuando abría la app y no veía plan nuevo.
# Ahora un cron busca usuarios con plan activo + cero logs en los últimos N días y les
# manda un nudge antes de que el siguiente bloque tropiece. La idea es que el push llegue
# 1-2 días antes del execute_after de chunk N+1 para que el usuario tenga tiempo de loguear.
#
# CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS: ventana sin logs para considerar un usuario candidato.
#   2 días es el equilibrio: 1 día sería ruidoso (usuario que se saltó un día), 3 días sería
#   tarde (chunk N+1 ya está en risk).
CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS = max(1, int(os.environ.get("CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS", "2")))
# CHUNK_ZERO_LOG_NUDGE_COOLDOWN_HOURS: nunca más de 1 push por usuario en esta ventana.
#   24h evita spam; el usuario que ignora el primer push no necesita 4 más al día.
CHUNK_ZERO_LOG_NUDGE_COOLDOWN_HOURS = max(6, int(os.environ.get("CHUNK_ZERO_LOG_NUDGE_COOLDOWN_HOURS", "24")))
# CHUNK_ZERO_LOG_NUDGE_INTERVAL_MINUTES: frecuencia del cron. 360 (6h) alinea con
# CHUNK_CHRONIC_DEFERRAL: el cooldown de 24h hace que correr más a menudo no aporte.
CHUNK_ZERO_LOG_NUDGE_INTERVAL_MINUTES = max(15, int(os.environ.get("CHUNK_ZERO_LOG_NUDGE_INTERVAL_MINUTES", "360")))
# CHUNK_ZERO_LOG_NUDGE_MAX_USERS: cap por corrida para no saturar webpush si la flota crece.
CHUNK_ZERO_LOG_NUDGE_MAX_USERS = max(1, int(os.environ.get("CHUNK_ZERO_LOG_NUDGE_MAX_USERS", "100")))
# [P0-4] Locks zombie: el worker actualiza chunk_user_locks.heartbeat_at cada N segundos
# mientras procesa. Si el heartbeat queda más viejo que el threshold, el housekeeping lo
# considera huérfano y lo libera. Antes el cleanup miraba locked_at con TTL fijo de 20 min,
# lo que retenía locks huérfanos demasiado tiempo (worker crasheado bloqueaba al usuario).
CHUNK_LOCK_HEARTBEAT_INTERVAL_SECONDS = int(os.environ.get("CHUNK_LOCK_HEARTBEAT_INTERVAL_SECONDS", "60"))
CHUNK_LOCK_STALE_MINUTES = int(os.environ.get("CHUNK_LOCK_STALE_MINUTES", "3"))
# [P2-CHUNK-1 · 2026-05-28] Ventana del zombie-rescue (un chunk 'processing' cuyo
# worker murió). Antes era el literal hardcodeado `INTERVAL '10 minutes'` en el
# SQL del worker, violando la convención de knobs. Clamp [CHUNK_LOCK_STALE_MINUTES,
# 60]: el piso evita mirar más rápido que el staleness del heartbeat; el techo
# evita ventanas absurdas. El reclaim ahora es heartbeat-aware (P1-CHUNK-2) — esta
# ventana solo decide cuándo MIRAR, no a quién reclamar.
CHUNK_ZOMBIE_RESCUE_MINUTES = max(
    CHUNK_LOCK_STALE_MINUTES,
    min(60, int(os.environ.get("CHUNK_ZOMBIE_RESCUE_MINUTES", "10"))),
)
# [P0-γ/HEARTBEAT-ALERTS] Dos alertas que cubren el hueco donde el thread daemon
# de heartbeat muere o el zombie rescue se dispara sin razón legítima:
#
#   (1) lag excesivo en pickup (`effective_lag_seconds_at_pickup` > umbral): un
#       worker recogió un chunk muchos minutos tarde respecto a su execute_after.
#       Causa típica: scheduler saturado, DB slow, o el worker anterior crasheó
#       sin liberar el lock y el zombie rescue tardó. Sin esta alerta, los
#       usuarios reciben planes a horas inesperadas y nadie sabe por qué.
#
#   (2) dual-processing del mismo meal_plan_id: 2+ chunks en status='processing'
#       para el mismo plan. Esto no debería suceder nunca (el lock por user_id
#       previene paralelismo intra-usuario, y solo hay un worker activo por plan
#       en un momento dado). Si sucede, indica heartbeat thread muerto +
#       zombie rescue triggered + worker original todavía corriendo. Riesgo:
#       doble reserva de inventario, doble merge a plan_data.
#
# CHUNK_LAG_ALERT_THRESHOLD_SECONDS: umbral de lag (segundos) que dispara
#   alerta. 600 (10min) cubre el caso documentado en P0-γ y alinea con la idea
#   de que el SLA del scheduler es tick=1min: 10min tarde implica problema
#   sistémico, no jitter normal.
CHUNK_LAG_ALERT_THRESHOLD_SECONDS = max(60, int(os.environ.get("CHUNK_LAG_ALERT_THRESHOLD_SECONDS", "600")))
# CHUNK_LAG_ALERT_INTERVAL_MINUTES: cada cuánto corre el cron. 15min permite
#   detectar dentro del mismo turno de SRE sin saturar logs.
CHUNK_LAG_ALERT_INTERVAL_MINUTES = max(5, int(os.environ.get("CHUNK_LAG_ALERT_INTERVAL_MINUTES", "15")))
# CHUNK_LAG_ALERT_WINDOW_HOURS: cuánto hacia atrás escanea. 1h captura el
#   último ciclo del cron con margen.
CHUNK_LAG_ALERT_WINDOW_HOURS = max(1, int(os.environ.get("CHUNK_LAG_ALERT_WINDOW_HOURS", "1")))
# CHUNK_LAG_ALERT_MIN_COUNT: cuántos chunks con lag>threshold disparan la alerta
#   en la ventana. 1 = cualquier ocurrencia (lag 10+min es señal). Subir si la
#   flota es grande y un caso aislado es ruido aceptable.
CHUNK_LAG_ALERT_MIN_COUNT = max(1, int(os.environ.get("CHUNK_LAG_ALERT_MIN_COUNT", "1")))
# CHUNK_LAG_ALERT_COOLDOWN_HOURS: dedupe del alert agregado.
CHUNK_LAG_ALERT_COOLDOWN_HOURS = max(1, int(os.environ.get("CHUNK_LAG_ALERT_COOLDOWN_HOURS", "6")))
# CHUNK_DUAL_PROCESSING_ALERT_INTERVAL_MINUTES: frecuencia del cron de
#   dual-processing. 5min porque el costo de doble reserva es alto: cada
#   minuto que esperamos para detectarlo agrava el daño potencial.
CHUNK_DUAL_PROCESSING_ALERT_INTERVAL_MINUTES = max(1, int(os.environ.get("CHUNK_DUAL_PROCESSING_ALERT_INTERVAL_MINUTES", "5")))
# CHUNK_DUAL_PROCESSING_ALERT_COOLDOWN_HOURS: dedupe corto. Si el problema se
#   repite tras 1h, queremos saber.
CHUNK_DUAL_PROCESSING_ALERT_COOLDOWN_HOURS = max(1, int(os.environ.get("CHUNK_DUAL_PROCESSING_ALERT_COOLDOWN_HOURS", "1")))
# CHUNK_DUAL_PROCESSING_GRACE_SECONDS: ventana de gracia antes de alarmar.
#   Durante un handoff legítimo (zombie rescue + pickup nuevo) puede haber un
#   rango corto donde dos filas en `processing` coexistan antes de que el CAS
#   del rescue gane. 90s cubre ese flap sin perder señal real.
CHUNK_DUAL_PROCESSING_GRACE_SECONDS = max(30, int(os.environ.get("CHUNK_DUAL_PROCESSING_GRACE_SECONDS", "90")))
# [P1-B/HEARTBEAT-START-FAIL] Si el thread daemon de heartbeat NO arranca (límite de
# threads del proceso, OOM transient), el chunk queda sin protección y el zombie rescue
# lo mataría tras CHUNK_LOCK_STALE_MINUTES en pleno LLM call — perdiendo tokens y
# trabajo. Antes el código solo loguaba ERROR y continuaba; ahora abortamos el chunk
# diferiéndolo a NOW + este delay para reintentar cuando el sistema esté menos cargado.
#
# CHUNK_HEARTBEAT_START_FAIL_RETRY_MINUTES: delay del retry tras fallo de start. 5 min
#   da margen para que el límite de threads del proceso se libere (otros chunks
#   completen y devuelvan threads al pool del SO).
CHUNK_HEARTBEAT_START_FAIL_RETRY_MINUTES = max(1, int(os.environ.get("CHUNK_HEARTBEAT_START_FAIL_RETRY_MINUTES", "5")))
# [P1-6] Buffer local para telemetría de deferrals cuando la INSERT a chunk_deferrals
# falla (DB caída, schema corrupto, permisos). El cron `_flush_pending_deferrals`
# reintenta cada CHUNK_DEFERRALS_FLUSH_INTERVAL_MINUTES leyendo el archivo y borrando
# las líneas ya persistidas. Cap de líneas evita crecimiento ilimitado en outages largos.
CHUNK_DEFERRALS_BUFFER_PATH = os.environ.get(
    "CHUNK_DEFERRALS_BUFFER_PATH", "deferrals_pending.jsonl"
)
CHUNK_DEFERRALS_FLUSH_INTERVAL_MINUTES = max(
    1, int(os.environ.get("CHUNK_DEFERRALS_FLUSH_INTERVAL_MINUTES", "5"))
)
CHUNK_DEFERRALS_BUFFER_MAX_RECORDS = max(
    100, int(os.environ.get("CHUNK_DEFERRALS_BUFFER_MAX_RECORDS", "10000"))
)
# [P0-10] Buffer paralelo para `chunk_lesson_telemetry` cuando la INSERT falla
# durante un outage de DB. Asimétrico respecto a deferrals antes: el helper
# `_record_chunk_lesson_telemetry` solo logueaba el fallo y perdía el evento
# para siempre — eventos críticos para detectar degradación del aprendizaje
# (`lesson_synthesized_low_confidence`, `recent_lessons_partial_synthesis`,
# `synth_schema_invalid`, `indefinite_pause_unblocked`,
# `lifetime_proxy_ratio_exceeded`) quedaban invisibles para SRE en el
# rango horas-días posteriores al outage.
# Mismo cap, mismo intervalo de flush que el buffer de deferrals; un cron
# dedicado los reintenta cuando DB se recupera.
CHUNK_LESSON_TELEMETRY_BUFFER_PATH = os.environ.get(
    "CHUNK_LESSON_TELEMETRY_BUFFER_PATH", "lesson_telemetry_pending.jsonl"
)
CHUNK_LESSON_TELEMETRY_FLUSH_INTERVAL_MINUTES = max(
    1, int(os.environ.get("CHUNK_LESSON_TELEMETRY_FLUSH_INTERVAL_MINUTES", "5"))
)
CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS = max(
    100, int(os.environ.get("CHUNK_LESSON_TELEMETRY_BUFFER_MAX_RECORDS", "10000"))
)
# [P3-2 · 2026-05-07] Threshold de alerta para el backlog de lesson_telemetry
# buffered. Si tras un flush `remaining` >= threshold, se loggea WARNING y el
# row a pipeline_metrics queda con confidence=0.0 (visual gate). Default 500
# sobre cap de 10000 → alerta en torno al 5% de saturación del buffer, antes
# de que el FIFO cap empiece a descartar señales.
MEALFIT_LESSON_BUFFER_BACKLOG_THRESHOLD = max(
    1, _env_int("MEALFIT_LESSON_BUFFER_BACKLOG_THRESHOLD", 500)
)
# [P1-5] Timeout (ms) para adquirir el SELECT FOR UPDATE en _persist_nightly_learning_signals.
# Múltiples planes activos del mismo usuario pueden competir por el row de user_profiles;
# sin timeout un chunk puede quedar bloqueado indefinidamente esperando a que el LLM retro
# de otro chunk libere el lock. Si excede este timeout, salta la señal (best-effort).
CHUNK_LEARNING_LOCK_TIMEOUT_MS = int(os.environ.get("CHUNK_LEARNING_LOCK_TIMEOUT_MS", "10000"))

def split_with_absorb(total_days: int, base: int = 3) -> list[int]:
    """Divide total_days en chunks de tamaño >= base sin perder días.

    Invariantes garantizados:
      - sum(result) == total_days
      - todos los elementos >= base (o == total_days si total_days <= base+1)

    [P1-A] Para planes largos (>=5 chunks de tamaño base con rem=0), preferimos
    chunks de `base+1` (4 días) tras un primer chunk de `base` (3 días). Esto:
      1. Reduce el número total de chunks, aliviando carga del scheduler y de
         llamadas LLM (15d: 5→4 chunks, 30d: 10→8 chunks ≈ 20% menos overhead).
      2. Alinea con el modelo mental del usuario ("primero 3 días para arrancar
         rápido, luego bloques de 4 con aprendizaje del primero").
      3. Mejora la coherencia de la ventana rolling de lecciones: con menos
         chunks, el cap del rolling window cubre más historial relativo.

    Ejemplos:
      -  7d → [3, 4]                            (caso especial, sin cambio)
      -  9d → [3, 3, 3]                         (n_full=3 < umbral, sin cambio)
      - 14d → [3, 3, 4, 4]                      (rem!=0, lógica original)
      - 15d → [3, 4, 4, 4]                      (P1-A: antes [3,3,3,3,3])
      - 18d → [3, 4, 4, 4, 3]                   (P1-A)
      - 21d → [3, 4, 4, 4, 6]                   (P1-A: leftover absorbido)
      - 30d → [3, 4, 4, 4, 4, 4, 4, 3]          (P1-A: antes [3]*10)
    """
    if total_days == 7 and base == 3:
        return [3, 4]
    if total_days <= base + 1:
        return [total_days]
    n_full = total_days // base
    rem = total_days % base

    # [P1-A] Solo aplica al caso "rem=0 y muchos chunks": planes que la lógica
    # original dividía en >=5 chunks de base. Para casos con rem!=0 o pocos
    # chunks preservamos la distribución existente (no romper short plans).
    _P1A_LONG_PLAN_THRESHOLD = 5
    if rem == 0 and n_full >= _P1A_LONG_PLAN_THRESHOLD:
        target = base + 1  # 4
        rest = total_days - base
        n_target = rest // target
        leftover = rest % target
        if leftover == 0:
            return [base] + [target] * n_target
        if leftover >= base:
            return [base] + [target] * n_target + [leftover]
        # leftover ∈ [1, base-1]: absorberlo en el último chunk target para
        # mantener la invariante "todos los chunks >= base".
        return [base] + [target] * (n_target - 1) + [target + leftover]

    # Comportamiento original preservado para planes cortos / no divisibles.
    if rem == 0:
        return [base] * n_full
    if n_full == 1:
        # Solo un chunk: absorber todo el resto en él
        return [total_days]
    # Distribuir el resto (+1) entre los últimos `rem` chunks
    n_base = n_full - rem
    return [base] * n_base + [base + 1] * rem
# --- VECTOR SEARCH CACHE ---

# [P1-EMBEDDING-CACHE-BOUNDED · 2026-05-24] Caches de embeddings con bound LRU.
# Pre-fix `_embedding_cache = {}` + `_pantry_embeddings_cache = {}` crecían
# monotónicamente: `get_embedding(text)` indexaba por string del ingrediente,
# y `_pantry_embeddings_cache` se popula desde el matcher de pantry en el
# pipeline de generación. Con miles de planes/mes y ~50-200 strings únicos
# por plan, RAM del worker FastAPI subía ~1-5 MB/mes sin techo → OOM eventual
# en contenedor del VPS Oracle sin alerta previa.
#
# Diseño: dict-like LRU bound por maxsize. Preserva la API in/out actual
# (`text not in cache`, `cache[text] = emb`, `return cache[text]`) para no
# tocar callsites. Eviction LRU vía OrderedDict.move_to_end + popitem(last=False).
# Sin lock: la API del dict original tampoco era thread-safe; el GIL hace
# atómicos cada operación individual; secuencias multi-op compiten pero el
# failure mode es a-lo-sumo eviction stale, no data corruption.
#
# Knob `MEALFIT_EMBEDDING_CACHE_MAXSIZE` (default 5000, clamp [100, 100000])
# auto-registrado en `_KNOBS_REGISTRY`. Visible en `/health/version`.
#
# Tooltip-anchor: P1-EMBEDDING-CACHE-BOUNDED.
# Test: `test_p1_prod_final_3.py::test_embedding_caches_are_bounded`.
from knobs import _env_int as _knob_env_int_constants
# [P2-KNOBS-ENV-INT-NO-VALIDATOR · 2026-05-24] `_env_int` ahora acepta
# `validator=`; clamp manual post-lectura reemplazado por validator en la
# signature del helper. Si el operador setea MAXSIZE fuera de [100, 100k]
# el helper loguea WARNING + cae al default 5000 (visible en _KNOBS_REGISTRY).
_EMBEDDING_CACHE_MAXSIZE = _knob_env_int_constants(
    "MEALFIT_EMBEDDING_CACHE_MAXSIZE",
    5000,
    validator=lambda v: 100 <= v <= 100_000,
)


class _BoundedEmbeddingCache:
    """[P1-EMBEDDING-CACHE-BOUNDED] LRU cache dict-like, bound por maxsize.

    API mínima compatible con `dict`: `__contains__`, `__getitem__`,
    `__setitem__`, `__len__`, `clear`. `get(k, default)` + `keys()` para
    inspección desde tests / observability.
    """

    __slots__ = ("_d", "_maxsize")

    def __init__(self, maxsize: int):
        from collections import OrderedDict
        self._d = OrderedDict()
        self._maxsize = max(int(maxsize), 1)

    def __contains__(self, key) -> bool:
        return key in self._d

    def __getitem__(self, key):
        value = self._d[key]
        self._d.move_to_end(key)
        return value

    def __setitem__(self, key, value) -> None:
        if key in self._d:
            self._d.move_to_end(key)
            self._d[key] = value
            return
        self._d[key] = value
        while len(self._d) > self._maxsize:
            self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)

    def get(self, key, default=None):
        if key in self._d:
            return self[key]
        return default

    def keys(self):
        return self._d.keys()

    def clear(self) -> None:
        self._d.clear()


_embedding_cache = _BoundedEmbeddingCache(_EMBEDDING_CACHE_MAXSIZE)
_pantry_embeddings_cache = _BoundedEmbeddingCache(_EMBEDDING_CACHE_MAXSIZE)

# [P0-DEEPSEEK-MIGRATION · 2026-06-12] El knob legacy
# `MEALFIT_GEMINI_EMBEDDING_TEXT_MODEL` y el singleton Gemini fueron
# reemplazados por la capa pluggable `embeddings_provider.py`
# (`MEALFIT_EMBEDDINGS_PROVIDER` / `MEALFIT_EMBEDDINGS_MODEL` /
# `MEALFIT_EMBEDDINGS_BASE_URL` + env `EMBEDDINGS_API_KEY`). Con provider
# `disabled` (default actual), `get_embedding` retorna None y los callers
# degradan al matching no-semántico (regex/substring).


def get_embedding(text: str) -> Optional[List[float]]:
    """Embedding cacheado del texto, o None si el provider está disabled o
    falló. Los fallos NO se cachean — el provider puede activarse en runtime
    via env + restart, y un blip transitorio no debe envenenar el LRU.

    [P1-COHERE-EMBED-V4] La key del LRU incluye el model_id del provider
    (espacio vectorial). Este matching es texto-a-texto simétrico → ambos
    lados usan el default `query` del provider."""
    from embeddings_provider import get_embeddings_model_id

    cache_key = (get_embeddings_model_id(), text)
    if cache_key not in _embedding_cache:
        emb = get_text_embedding(text)
        if emb is None:
            return None
        _embedding_cache[cache_key] = emb
    return _embedding_cache[cache_key]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
# ---------------------------

# [P3-PAVO-SKELETON-VERIFIED-ALIGN · 2026-06-22 · rev2] "Pavo" → "Pechuga de pavo": el pool del esqueleto
# ofrecía "Pavo" (fresco genérico, NO verificado) → con verified-only ON el LLM escribía "pechuga de pavo"
# en la receta y la lista lo DROPEABA (modo visto en vivo corr=4a8d46e1, 2026-06-22). El owner agregó al
# catálogo "Pechuga de pavo" (RD$415/lb, owner_verified) — el pavo que la regla 69 del day-gen YA prefería
# (pechuga > jamón en lonjas). Ahora el pool ofrece el nombre verificado exacto: esqueleto → "Pechuga de
# pavo" → el LLM lo usa → normalize_name("pechuga de pavo")="Pechuga de pavo" (guard P3-PROTEIN-CAP-2)
# resuelve a la fila priced → aparece en la lista. Coherente end-to-end. (rev1 apuntaba a "Jamón de pavo"
# RD$255 como workaround cuando pechuga no estaba priced; superado al verificarse la pechuga.)
DOMINICAN_PROTEINS = [
    # [P3-PECHUGA-PAVO-REMOVE · 2026-06-23] "Pechuga de pavo" eliminada: el owner confirmó que La Sirena
    # NO vende pechuga de pavo fresca. El pool ya no ofrece pavo fresco (jamón de pavo es embutido
    # procesado, gateado aparte por _RESTRICTED_PROTEIN_KEYS, no proteína principal). Otras proteínas
    # cubren. canonicalize_pavo + PROTEIN_SYNONYMS quedan (defensivos/inertes; la pechuga ya no aparece).
    "Pollo", "Cerdo", "Res", "Pescado", "Atún", "Huevos", "Queso de Freír",
    "Salami Dominicano", "Camarones", "Chuleta", "Longaniza",
    "Habichuelas Rojas", "Habichuelas Negras", "Habichuelas Blancas",
    "Gandules", "Lentejas", "Garbanzos",
    "Queso Ricotta", "Queso Blanco", "Queso Mozzarella", "Yogurt",
    # [P1-VARIETY-CATALOG-POOLS · 2026-06-27] Proteínas verificadas del catálogo (202) añadidas a la rotación
    # para que cada renovación explote toda la variedad. El filtro de dieta (_get_fast_filtered_catalogs) y el
    # backstop P1-DIET-HARD-GUARD excluyen estas animales para vegano/vegetariano/pescetariano. Nombres EXACTOS
    # del catálogo (resuelven en la lista de compras). Tofu NO se añade (P3-TOFU-REMOVE).
    "Muslo de pollo", "Pavo molido", "Hígado de res", "Costilla de cerdo",
    "Mero", "Tilapia", "Salmón", "Bacalao", "Sardinas en lata", "Arenque",
    "Conejo", "Chivo", "Pulpo", "Calamar", "Mejillones", "Cangrejo", "Jamón de pavo",
    "Edamame", "Soya texturizada", "Frijoles pintos", "Habas", "Guisantes secos",
    "Queso de hoja", "Queso cottage", "Queso parmesano", "Queso cheddar", "Queso gouda",
    "Yogurt griego entero",
]
# [P3-TOFU-REMOVE · 2026-06-22] "Soya/Tofu" eliminado del pool de proteínas ofrecibles: el owner
# confirmó que La Sirena NO vende tofu (ni carne de soya como producto verificado). El esqueleto ya
# no lo ofrece; el master row "Tofu" fue borrado (no priced, fuera del catálogo VERIFIED-ONLY). Las
# refs de tofu en condition_rules/allergy maps SE CONSERVAN a propósito (defensa: si un usuario
# alérgico a soya menciona tofu en texto libre, el swap tofu→pollo sigue activo). Proteína vegana
# ahora = leguminosas (Gandules/Lentejas/Garbanzos/Habichuelas). Tooltip-anchor: P3-TOFU-REMOVE.

DOMINICAN_CARBS = [
    "Plátano Verde", "Plátano Maduro", "Yuca", "Batata", "Arroz Blanco",
    "Arroz Integral", "Avena", "Pan Integral", "Papas", "Guineítos Verdes", "Ñame", "Yautía",
    # [P1-VARIETY-CATALOG-POOLS · 2026-06-27] Carbos/granos verificados del catálogo añadidos a la rotación.
    "Quinoa", "Pasta integral", "Bulgur", "Cebada", "Casabe", "Mapuey", "Tortilla integral", "Harina de Negrito",
    # [P1-FLOURS-POOLS · 2026-07-01] (audit creatividad G2) Las HARINAS y el maíz verificados NO rotaban en los
    # pools → el ejemplo flagship del owner ("con la harina haces panqueques, bollos, arepas") casi no podía
    # ocurrir como plato principal fuera del desayuno, pese a que la regla 2.5 de creatividad los promueve.
    # El round-robin ahora los ofrece como carb-base; el day-gen los transforma (panqueques/bollos/arepitas/
    # empanadas al horno). Nombres = catálogo verificado (resuelven en la lista). tooltip-anchor: P1-FLOURS-POOLS
    "Harina de trigo", "Harina de maíz precocida", "Maíz dulce en granos", "Tortilla de trigo",
]

PROTEIN_SYNONYMS = {
    # [P1-FLOURS-POOLS · 2026-07-01] Backfill: los items del pool P1-VARIETY-CATALOG-POOLS que no resolvían
    # en el synonym system (ni key ni variant) se añaden como VARIANT de su base cuando son el mismo alimento
    # (muslo de pollo→pollo) o como base propia cuando son alimento distinto (conejo, chivo, mariscos, quesos).
    "pollo": ["pollo", "pechuga", "muslo", "alitas", "chicharrón de pollo", "filete de pollo", "muslo de pollo"],
    "cerdo": ["cerdo", "masita", "chicharrón de cerdo", "lomo", "pernil", "costilla", "costilla de cerdo"],
    "res": ["res", "carne molida", "bistec", "filete", "churrasco", "vaca", "picadillo", "carne de res",
            "hígado de res", "higado de res"],
    # [P2-COH-1] Lista expandida con peces comunes en RD/Caribe (merluza,
    # róbalo, pargo, corvina, mahi-mahi, lubina, carite, jurel, lambí). Sin
    # estos, el coherence check en `graph_orchestrator._run_assembly_validations`
    # producía falsos positivos HIGH cuando recetas como "Merluza Estofada"
    # mencionaban "el pescado" en su texto pero "merluza" no estaba registrada
    # como sinónimo (incidente 2026-05-05 — plan rechazado y entregado roto).
    # `mahi-mahi` y `mahi mahi` se incluyen ambos para tolerar la convención
    # de hyphenation que el LLM elige inconsistentemente.
    "pescado": ["pescado", "dorado", "chillo", "mero", "salmón", "salmon", "tilapia", "filete de pescado",
                "bacalao", "bacalao desalado", "bacalao salado", "filete de bacalao",
                "filete de mero", "filete de tilapia", "filete de chillo", "filete de dorado",
                "merluza", "filete de merluza",
                "róbalo", "robalo", "pargo", "corvina",
                "mahi-mahi", "mahi mahi", "mahimahi",
                "lubina", "carite", "jurel", "lambí", "lambi", "arenque"],
    "atún": ["atún", "atun", "atun en agua", "atun en lata"],
    "sardina": ["sardina", "sardinas", "sardina en lata", "sardinas en lata"],
    "huevos": ["huevos", "huevo", "tortilla", "revoltillo"],
    "queso de freír": ["queso de freír", "queso de freir", "queso frito", "queso de hoja"],
    "salami dominicano": ["salami dominicano", "salami", "salchichón"],
    "camarones": ["camarones", "camarón", "camaron"],
    "chuleta": ["chuleta", "chuletas", "chuleta frita", "chuleta al horno"],
    "longaniza": ["longaniza", "longanizas"],

    "habichuelas rojas": ["habichuelas rojas", "frijoles rojos", "habichuela roja"],
    "habichuelas negras": ["habichuelas negras", "frijoles negros", "habichuela negra"],
    "habichuelas blancas": ["habichuelas blancas", "frijoles blancos", "habichuela blanca", "porotos blancos", "alubias blancas"],
    "queso ricotta": ["queso ricotta", "ricotta", "queso ricotta descremado", "ricotta descremada", "ricotta light"],
    "yogurt": ["yogurt", "yogur", "yogurt griego", "yogurt natural", "yogurt griego natural",
               "yogurt griego sin azucar", "yogurt griego sin azúcar", "yogurt sin azucar",
               "yogurt descremado", "greek yogurt", "yogurt griego entero"],
    "queso blanco": ["queso blanco", "queso blanco fresco", "queso de pasta", "queso fresco",
                     "queso de mano", "queso crema", "queso campesino"],
    "queso mozzarella": ["queso mozzarella", "mozzarella", "mozzarella fresca", "queso mozarela",
                         "queso mozarella", "queso mozzarella fresco"],
    "gandules": ["gandules", "guandules", "gandul", "guandul"],
    "lentejas": ["lentejas", "lenteja"],
    "garbanzos": ["garbanzos", "garbanzo", "hummus", "puré de garbanzos"],
    "soya/tofu": ["soya/tofu", "soya", "tofu", "carne de soya", "tofu/soya", "tofu/soya firme", "tofu firme",
                  "soya firme", "soya texturizada"],
    # [P1-FLOURS-POOLS · 2026-07-01] Bases propias para items del pool sin base semántica existente.
    "conejo": ["conejo"],
    "chivo": ["chivo", "carne de chivo", "chivo guisado"],
    "pulpo": ["pulpo"],
    "calamar": ["calamar", "calamares"],
    "mejillones": ["mejillones", "mejillón", "mejillon"],
    "cangrejo": ["cangrejo", "jaiba"],
    "edamame": ["edamame", "edamames"],
    "frijoles pintos": ["frijoles pintos", "habichuelas pintas"],
    "habas": ["habas", "haba"],
    "guisantes secos": ["guisantes secos", "guisantes partidos", "arvejas secas"],
    "queso cottage": ["queso cottage", "cottage"],
    "queso parmesano": ["queso parmesano", "parmesano", "parmesano rallado"],
    "queso cheddar": ["queso cheddar", "cheddar"],
    "queso gouda": ["queso gouda", "gouda"],
    # [P3-PAVO-SYNONYM-KEY · 2026-06-22] Key renombrada "pavo"→"pechuga de pavo" para alinear con
    # el catálogo (DOMINICAN_PROTEINS usa "Pechuga de pavo" verificado desde P3-CONDIMENT-CONSOLIDATION).
    # test_synonyms::test_catalog_lists_match_synonym_keys exige que cada item del catálogo tenga key
    # en su synonym map; "Pechuga de pavo" no la tenía (era alias bajo "pavo"). "pavo" queda como alias.
    "pechuga de pavo": ["pechuga de pavo", "pavo", "pavo asado", "pavo desmenuzado", "jamón de pavo", "pavo molido", "carne de pavo"]
}

CARB_SYNONYMS = {
    "plátano verde": ["plátano verde", "platano verde", "mangú", "mangu", "tostones", "fritos verdes", "mangú de plátano", "mangu de platano"],
    "plátano maduro": ["plátano maduro", "platano maduro", "maduros", "plátano al caldero", "fritos maduros"],
    "yuca": ["yuca", "casabe", "arepitas de yuca", "puré de yuca"],
    "arroz blanco": ["arroz blanco", "arroz"],
    "arroz integral": ["arroz integral"],
    "avena": ["avena", "avena en hojuelas", "overnight oats", "avena instantanea"],
    "pasta": ["pasta", "espagueti", "espaguetis", "spaghetti", "macarrones", "coditos", "fideos",
              "pasta integral", "espagueti integral", "fideos integrales", "macarrones integrales"],
    "quinoa": ["quinoa", "quinua"],
    "pan integral": ["pan integral", "pan", "tostada integral", "tostada"],
    "papas": ["papas", "papa", "puré de papas", "papa hervida"],
    "guineítos verdes": ["guineítos verdes", "guineítos", "guineitos", "guineos verdes", "guineito verde", "guineitos verdes"],
    "ñame": ["ñame", "name", "ñame hervido"],
    "yautía": ["yautía", "yautia", "yautía hervida"],
    "batata": ["batata", "puré de batata", "batata hervida", "boniato"],
    # [P1-FLOURS-POOLS · 2026-07-01] Entradas para los items del pool que NO resolvían en el synonym system
    # (los del lote P1-VARIETY-CATALOG-POOLS que no eran variant de ninguna base + las harinas nuevas). Sin
    # entrada, la variedad/fatiga/coherencia no los reconocen. NO se añade "harina" ni "maíz" a secas como
    # variant: colapsarían preparaciones ("harina de avena") a la base equivocada (lección P1-NUT-BUTTER-DISTINCT).
    "bulgur": ["bulgur", "trigo bulgur"],
    "cebada": ["cebada", "cebada perlada"],
    "mapuey": ["mapuey"],
    "tortilla integral": ["tortilla integral", "wrap integral"],
    "harina de negrito": ["harina de negrito", "negrito"],
    "harina de trigo": ["harina de trigo", "harina blanca", "harina todo uso", "panqueques de harina",
                        "bollos de harina"],
    "harina de maíz precocida": ["harina de maíz precocida", "harina de maiz precocida", "harina de maíz",
                                 "harina de maiz", "harina pan", "arepa", "arepas", "arepitas de maíz",
                                 "arepitas de maiz"],
    "maíz dulce en granos": ["maíz dulce en granos", "maiz dulce en granos", "maíz dulce", "maiz dulce",
                             "maíz en granos", "maiz en granos"],
    "tortilla de trigo": ["tortilla de trigo", "wrap", "wraps", "tortillas de trigo"],
}

DOMINICAN_VEGGIES_FATS = [
    "Aguacate", "Berenjena", "Tayota", "Repollo", "Zanahoria",
    "Molondrones", "Brócoli", "Coliflor", "Tomate", "Vainitas",
    "Aceitunas", "Cebolla", "Ajíes", "Aceite de Oliva", "Nueces/Almendras",
    "Auyama",
    # [P1-VARIETY-CATALOG-POOLS · 2026-06-27] Vegetales + grasas/semillas verificados del catálogo (202).
    "Espinacas", "Pepino", "Lechuga", "Apio", "Espárragos", "Champiñones", "Remolacha", "Kale", "Rúcula",
    "Berro", "Calabacín", "Repollo morado", "Rábano", "Coles de Bruselas", "Puerro", "Bok choy", "Nabo",
    "Alcachofa", "Palmito", "Cebollín", "Cundeamor",
    "Maní", "Merey", "Pistachos", "Semillas de chía", "Semillas de girasol", "Semillas de calabaza",
    "Linaza", "Mantequilla de maní", "Mantequilla de almendras", "Ajonjolí", "Almendras fileteadas",
]

VEGGIE_FAT_SYNONYMS = {
    "aguacate": ["aguacate", "palta"],
    "berenjena": ["berenjena", "berenjenas", "berenjena rellena"],
    "tayota": ["tayota", "chayote", "tayotas", "chayotes", "cidra"],
    "espinaca": ["espinaca", "espinacas", "baby spinach"],
    "pepino": ["pepino", "pepinos"],
    "lechuga": ["lechuga", "lechugas", "lechuga romana", "lechuga iceberg"],
    "cilantro": ["cilantro", "culantro", "verdura", "recao"],
    "repollo": ["repollo"],
    "zanahoria": ["zanahoria", "zanahorias"],
    "molondrones": ["molondrones", "molondrón", "okra"],
    "brócoli": ["brócoli", "brocoli"],
    "coliflor": ["coliflor"],
    "tomate": ["tomate", "tomates", "pico de gallo"],
    "vainitas": ["vainitas", "judías verdes", "ejotes"],
    "aceitunas": ["aceitunas", "aceituna"],
    "cebolla": ["cebolla", "cebollas"],
    "ajíes": ["ajíes", "ají", "pimientos", "pimiento", "pimiento morrón", "pimiento morron", "ajies"],
    "auyama": ["auyama", "calabaza", "zapallo", "squash", "ahuyama"],
    "aceite de oliva": ["aceite de oliva", "aceite verde"],
    "nueces/almendras": ["nueces/almendras", "nueces", "almendras", "maní", "almendras fileteadas"],
    # [P1-FLOURS-POOLS · 2026-07-01] Backfill de los vegetales/grasas del pool P1-VARIETY-CATALOG-POOLS
    # que no resolvían en el synonym system (variedad/fatiga/coherencia eran ciegas a ellos).
    "apio": ["apio"],
    "espárragos": ["espárragos", "esparragos", "espárrago", "esparrago"],
    "champiñones": ["champiñones", "champinones", "champiñón", "champinon", "hongos", "setas"],
    "remolacha": ["remolacha", "betabel"],
    "kale": ["kale", "col rizada"],
    "rúcula": ["rúcula", "rucula", "arúgula", "arugula"],
    "berro": ["berro", "berros"],
    "calabacín": ["calabacín", "calabacin", "zucchini"],
    "repollo morado": ["repollo morado", "col morada"],
    "rábano": ["rábano", "rabano", "rábanos", "rabanos"],
    "coles de bruselas": ["coles de bruselas", "repollitos de bruselas"],
    "puerro": ["puerro", "puerros"],
    "bok choy": ["bok choy", "pak choi"],
    "nabo": ["nabo", "nabos"],
    "alcachofa": ["alcachofa", "alcachofas"],
    "palmito": ["palmito", "palmitos"],
    "cebollín": ["cebollín", "cebollin"],
    "cundeamor": ["cundeamor"],
    "merey": ["merey", "anacardo", "cajuil", "marañón", "maranon"],
    "pistachos": ["pistachos", "pistacho"],
    "semillas de chía": ["semillas de chía", "semillas de chia", "chía", "chia"],
    "semillas de girasol": ["semillas de girasol", "pipas de girasol"],
    "semillas de calabaza": ["semillas de calabaza", "pepitas de calabaza"],
    "linaza": ["linaza", "semillas de lino", "lino molido"],
    "mantequilla de maní": ["mantequilla de maní", "mantequilla de mani", "crema de maní", "crema de mani",
                            "peanut butter"],
    "mantequilla de almendras": ["mantequilla de almendras", "crema de almendras"],
    "ajonjolí": ["ajonjolí", "ajonjoli", "sésamo", "sesamo", "semillas de sésamo", "semillas de sesamo"],
}

DOMINICAN_FRUITS = [
    "Guineo", "Mango", "Piña", "Lechosa", "Chinola",
    "Limón", "Fresa", "Naranja", "Sandía", "Melón",
    # [P1-VARIETY-CATALOG-POOLS · 2026-06-27] Frutas FRESCAS verificadas añadidas. Se EXCLUYEN a propósito de la
    # rotación diaria los treats/secos (Cereza maraschino, Durazno en almíbar, Dátiles, Pasas, Ciruela pasa,
    # Tamarindo) y el Coco (alto en grasa) — son meriendas/endulzantes ocasionales, no fruta fresca del día.
    "Manzana", "Guayaba", "Guanábana", "Níspero", "Mandarina", "Toronja", "Uva", "Pera", "Kiwi",
    "Ciruela", "Arándanos",
]

FRUIT_SYNONYMS = {
    "guineo": ["guineo", "guineo maduro", "banana", "banano", "cambur"],
    "mango": ["mango", "mangos", "mango maduro"],
    "piña": ["piña", "pina", "piña natural"],
    "lechosa": ["lechosa", "papaya"],
    "chinola": ["chinola", "maracuyá", "maracuya", "parcha"],
    "limón": ["limón", "limon", "lima", "jugo de limón"],
    "fresa": ["fresa", "fresas", "frutilla"],
    "naranja": ["naranja", "naranjas", "jugo de naranja"],
    "sandía": ["sandía", "sandia", "patilla"],
    "melón": ["melón", "melon"],
    # [P1-FLOURS-POOLS · 2026-07-01] Backfill de las frutas frescas del pool P1-VARIETY-CATALOG-POOLS
    # que no resolvían en el synonym system (el gate de fruta repetida era ciego a ellas).
    "manzana": ["manzana", "manzanas", "manzana verde", "manzana roja"],
    "guayaba": ["guayaba", "guayabas"],
    "guanábana": ["guanábana", "guanabana"],
    "níspero": ["níspero", "nispero"],
    "mandarina": ["mandarina", "mandarinas"],
    "toronja": ["toronja", "pomelo"],
    "uva": ["uva", "uvas"],
    "pera": ["pera", "peras"],
    "kiwi": ["kiwi", "kiwis"],
    "ciruela": ["ciruela", "ciruelas"],
    "arándanos": ["arándanos", "arandanos", "arándano", "arandano", "blueberries"],
}

NUTRITIONAL_CATEGORIES = {
    "aves": ["pollo", "pavo"],
    "carnes rojas y embutidos": ["cerdo", "res", "chuleta", "longaniza", "salami dominicano"],
    "pescados y mariscos": ["pescado", "atún", "sardina", "camarones"],
    "huevos y lácteos": ["huevos", "queso de freír", "queso", "yogurt", "leche", "queso crema", "ricotta", "cottage", "yogurt griego"],
    "legumbres": ["habichuelas rojas", "habichuelas negras", "gandules", "lentejas", "garbanzos"],  # [P3-TOFU-REMOVE] soya/tofu fuera
    "víveres y almidones": ["plátano verde", "plátano maduro", "yuca", "batata", "guineítos verdes", "ñame", "yautía", "papas", "casabe"],
    "cereales": ["arroz blanco", "arroz integral", "avena", "pasta", "quinoa", "pan integral", "pan"]
}

def get_nutritional_category(base_ingredient: str) -> str:
    """Retorna la categoría nutricional amplia de un ingrediente base para detectar fatiga cruzada."""
    if not base_ingredient:
        return None
    for category, items in NUTRITIONAL_CATEGORIES.items():
        if base_ingredient in items:
            return category
    return None


# ──────────────────────────────────────────────────────────────────────────────
# [P1-SLOT-APPROPRIATENESS · 2026-06-27] (audit G4) Coherencia comida↔HORARIO (es-DO).
# ──────────────────────────────────────────────────────────────────────────────
# SSOT del mapa de apropiación por slot + helpers puros (sin deps de graph_orchestrator,
# así agent.py / tools.py / graph_orchestrator.py lo importan sin ciclos). Decisión de
# producto (owner 2026-06-27): el arroz/locrio/pasta en el DESAYUNO = SIEMPRE duro
# (no degrada); el "arroz de noche" / comida de desayuno en la CENA = duro CON degradación
# a advisory en el intento final (nunca cero-plan). El match es WORD-BOUNDARY sobre el
# NOMBRE del plato (NO los ingredientes) — anti-falso-positivo: así "Panqueques de harina
# de arroz" en desayuno NO se flagea como "arroz", protegiendo la creatividad (G5). Las
# exclusiones cubren los modificadores donde el token es parte de otro alimento (harina de
# arroz, pasta de maní, leche de arroz). Tokens YA normalizados (sin acentos, minúscula):
# "lasaña"→"lasana", "ñoqui"→"noqui". `hardness`: "hard"=rechaza en todos los intentos;
# "soft"=degrada a advisory en el intento final. tooltip-anchor: P1-SLOT-APPROPRIATENESS
_SLOT_RICE_EXCLUDE = (
    "harina de arroz", "leche de arroz", "vinagre de arroz", "galleta de arroz",
    "papel de arroz", "agua de arroz", "crema de arroz",
    # [P2-SLOT-RICE-TANGENTIAL · 2026-06-29] (audit objetivo · P2-9) Menciones TANGENCIALES de arroz que
    # NO hacen del plato un "arroz blanco" (el detector es name-based, ciego a cantidad): una cena
    # legítimamente ligera con "un toque de arroz" o un "crocante de arroz inflado" no debe disparar el
    # falso positivo de "arroz de noche". Reduce el ruido sin tocar el detector. tooltip-anchor: P2-SLOT-RICE-TANGENTIAL
    "toque de arroz", "crocante de arroz", "arroz inflado", "crujiente de arroz",
)
# [P2-SLOT-RICE-SYNONYMS · 2026-06-29] (audit objetivo · P2) El detector es name-based: un plato de arroz cuyo
# NOMBRE no contiene el substring "arroz" (chofán/chaufa = arroz frito; paella/risotto/congrí/mamposteao = arroz
# como base) se colaba como "arroz de noche" sin que el gate determinista lo cazara — solo el prompt (advisory)
# lo desalentaba. SSOT compartido por desayuno (hard) y cena (soft). Tokens normalizados (sin acento, minúscula).
# tooltip-anchor: P2-SLOT-RICE-SYNONYMS
_SLOT_RICE_TOKENS = ("arroz", "locrio", "moro", "morito", "chofan", "chaufa", "paella",
                     "congri", "mamposteao", "mampostea", "risotto")
SLOT_INAPPROPRIATE_FOODS = {
    "desayuno": [
        {"label": "arroz/locrio/moro", "tokens": _SLOT_RICE_TOKENS,
         "hardness": "hard", "exclude": _SLOT_RICE_EXCLUDE},
        {"label": "pasta/espaguetis/lasaña", "tokens": (
            "espagueti", "espaguetis", "macarron", "macarrones", "lasana", "coditos",
            "fideos", "tallarines", "pastelon", "ravioli", "penne", "noqui", "rigatoni"),
         "hardness": "hard"},
        {"label": "sopón de almuerzo (sancocho/asopao/mondongo)", "tokens": (
            "sancocho", "asopao", "mondongo"), "hardness": "hard"},
        # [P2-SLOT-DESAYUNO-GUISADOS · 2026-07-02] (audit v3 slots GAP-B) el prompt (plan_generator regla
        # cultural) prohíbe legumbres y proteínas GUISADAS pesadas al desayuno pero el validador no lo
        # enforzaba (prompt-only, sin red determinista). Tokens COMPUESTOS para las proteínas (salami
        # guisado / huevos guisados SÍ son desayuno RD legítimo — no van sueltos "guisado") + legumbres
        # standalone. Exclude: "habichuelas con dulce" (postre criollo — fuera del alcance de la regla).
        # hardness=soft (degrada a advisory en intento final). tooltip-anchor: P2-SLOT-DESAYUNO-GUISADOS
        {"label": "guiso pesado/legumbres de almuerzo en el desayuno", "tokens": (
            "pollo guisado", "pescado guisado", "cerdo guisado", "res guisada", "carne guisada",
            "chuleta guisada", "habichuela", "lenteja", "guandul", "gandule"),
         "hardness": "soft", "exclude": ("habichuelas con dulce", "habichuela con dulce")},
    ],
    "cena": [
        {"label": "arroz/locrio/moro (\"arroz de noche\")", "tokens": _SLOT_RICE_TOKENS,
         "hardness": "soft", "exclude": _SLOT_RICE_EXCLUDE},
        # [P1-SLOT-CENA-PASTA-OK · 2026-06-27] La PASTA/espagueti SÍ va en la cena dominicana (carbo ligero de
        # digestión rápida; "cenar espaguetis" es común) → NO se bloquea de noche. Solo queda inapropiada en el
        # DESAYUNO. Decisión del owner. Antes estaba como soft-block en cena (suposición cultural incorrecta).
        # [P2-SLOT-CENA-AVENA · 2026-07-02] (audit v3 slots GAP-D) + "avena": el almuerzo ya la tenía
        # (P3-SLOT-ALMUERZO-AVENA) pero la cena no — "Avena con frutas" de cena pasaba mientras la misma
        # de almuerzo flageaba (asimetría). Mismos excludes (harina/leche/costra de avena legítimos).
        {"label": "comida de desayuno en la cena (cereal/panqueque/waffle/avena)", "tokens": (
            "cereal", "hojuelas", "panqueque", "pancake", "waffle", "crepe", "crepa", "avena"),
         "hardness": "soft",
         "exclude": ("harina de avena", "leche de avena", "costra de avena", "empanizado de avena",
                     "empanizada de avena", "apanado de avena")},
        # [P2-SLOT-CENA-FRITURA · 2026-07-02] (audit v3 slots GAP-C) "frituras pesadas de noche" era
        # prompt-only (day_generator §15d) — 'frito'/'frita' sueltos se omiten A PROPÓSITO (falso positivo
        # en tostones/queso frito como acompañante). Tokens COMPUESTOS de fritura-de-proteína-como-plato
        # (inequívocos) + chicharrón (plato fuerte frito; como merienda sigue legítimo — regla solo cena).
        # hardness=soft. tooltip-anchor: P2-SLOT-CENA-FRITURA
        {"label": "fritura pesada de proteína como plato de la cena", "tokens": (
            "pollo frito", "pescado frito", "cerdo frito", "chuleta frita", "costilla frita",
            "salami frito", "carne frita", "chicharron"),
         "hardness": "soft"},
        # [P2-SLOT-CENA-HEAVY-SOUP · 2026-06-29] (re-audit objetivo · P2) El prompt prohíbe los sopones
        # pesados de noche pero el gate no los enforzaba (asimetría vs la regla "sopón de almuerzo" del
        # desayuno). sancocho/asopao/mondongo son culturalmente platos de mediodía, pesados para la cena.
        # hardness=soft (degrada a advisory en el intento final, nunca cero-plan). Conservador: NO añadimos
        # 'guisada'/'frito' (falso positivo en cenas ligeras legítimas). tooltip-anchor: P2-SLOT-CENA-HEAVY-SOUP
        {"label": "sopón/guiso pesado de noche (sancocho/asopao/mondongo)", "tokens": (
            "sancocho", "asopao", "mondongo"), "hardness": "soft"},
        # [P3-SLOT-NAME-HONESTY · 2026-07-01] (audit v2 slots nota P3, batch P3-AUDIT-V2-RESIDUALS) Un plato
        # de CENA literalmente nombrado "Desayuno ..." ("Cena: Desayuno criollo") pasaba el gate (token-de-
        # alimento, no token-de-slot). Token estrecho: solo la palabra "desayuno" en el NOMBRE. soft.
        {"label": "plato nombrado 'desayuno' servido en la cena", "tokens": ("desayuno",),
         "hardness": "soft"},
    ],
    # [P2-SLOT-ALMUERZO · 2026-06-29] (re-audit objetivo · P2) El almuerzo es el plato fuerte; un desayuno
    # (panqueque/cereal/granola) o un postre standalone (helado/flan) como PLATO PRINCIPAL del almuerzo es
    # incoherente. Set TIGHT y curado: NO incluye arroz/sopa/asopao/ensalada (todos legítimos en almuerzo — el
    # "no añadir almuerzo" del lote previo era por el arroz, no por un blocklist de desayuno/postre). hardness=
    # soft (nunca cero-plan). Estos tokens van en desayuno/merienda (F2 creatividad: panqueques de avena/harina),
    # pero como PLATO PRINCIPAL del almuerzo son incoherentes. tooltip-anchor: P2-SLOT-ALMUERZO
    "almuerzo": [
        # [P3-SLOT-ALMUERZO-AVENA · 2026-07-01] (audit v2 slots P3-2, batch P3-AUDIT-V2-RESIDUALS) + "avena":
        # el propósito declarado del set era cazar "comida de desayuno como plato principal" pero omitía la
        # más icónica RD. Excludes protegen usos legítimos de avena-como-INGREDIENTE en un plato fuerte
        # (costra/empanizado) y las formas harina/leche (F2 creatividad).
        {"label": "comida de desayuno como plato principal del almuerzo (cereal/panqueque/waffle/avena)", "tokens": (
            "cereal", "hojuelas", "granola", "panqueque", "pancake", "waffle", "avena"),
         "hardness": "soft",
         "exclude": ("harina de avena", "leche de avena", "costra de avena", "empanizado de avena",
                     "empanizada de avena", "apanado de avena")},
        {"label": "postre standalone como plato principal del almuerzo (helado/flan)", "tokens": (
            "helado", "flan"), "hardness": "soft"},
    ],
    # [P2-SLOT-MERIENDA · 2026-06-29] (audit objetivo · P2-8) Cierra el gap "el gate enforced solo cubre
    # desayuno/cena": un PLATO FUERTE disfrazado de merienda (mini-almuerzo) ahora se flagea en TODAS las
    # superficies (gate S1 + backstops swap/regenerate-day/chat-modify) porque todas leen este SSOT. Tokens =
    # técnicas/platos fuertes (espejo conceptual de `_HEAVY_TECHNIQUE_KEYWORDS` del detector advisory de S1).
    # NO incluye 'arroz'/'habichuela' sueltos: "Arroz con leche" es una merienda/postre dominicana legítima
    # (anclado por test_p1_slot_appropriateness). hardness=soft (degrada a advisory en el intento final, nunca
    # cero-plan — coherente con la filosofía de slot). tooltip-anchor: P2-SLOT-MERIENDA
    "merienda": [
        {"label": "plato fuerte (locrio/moro/guiso/sancocho) en la merienda", "tokens": (
            "locrio", "moro", "morito", "asopao", "sancocho", "mondongo", "mofongo", "pastelon",
            "salteado", "guisado", "guisada", "estofado", "encebollado", "croquetas"),
         "hardness": "soft"},
        # [P2-SLOT-MERIENDA-JUNK · 2026-06-29] (re-audit objetivo · P2) Comida chatarra / plato completo como
        # "merienda". CONSERVADOR a propósito: solo tokens INEQUÍVOCAMENTE de comida-completa/junk. EXCLUIDOS
        # deliberadamente: empanada/pastelito, chicharrón, frituras/yaniqueque/catibía (TODAS meriendas
        # dominicanas legítimas) y 'frito'/'frita' suelto (falso positivo en tostones/queso frito). hardness=soft.
        # tooltip-anchor: P2-SLOT-MERIENDA-JUNK
        {"label": "comida chatarra/plato completo como merienda (pizza/hamburguesa/yaroa)", "tokens": (
            "pizza", "hamburguesa", "yaroa"), "hardness": "soft"},
    ],
}

# Guía POSITIVA por slot (es-DO) inyectada a los prompts de UPDATE (swap/chat-modify) y usada
# en los mensajes de rechazo del gate S1 — describe qué SÍ va en cada horario.
SLOT_POSITIVE_HINT = {
    "desayuno": ("El desayuno dominicano va: mangú/víveres, avena/cereales calientes, pan/tostadas, "
                 "batido/bowl o revoltillo — con proteína y fruta."),
    "almuerzo": ("El almuerzo es el plato fuerte: arroz+habichuela+proteína+ensalada, locrio, moro, "
                 "asopao, pasta criolla, o pescado/carne con tubérculo y vegetal."),
    "cena": ("La cena es más ligera que el almuerzo: pescado/pollo a la plancha, tortilla/revoltillo de "
             "cena, sopa ligera, wrap o bowl de proteína + vegetales + un tubérculo (batata/yuca/casabe). "
             "Evita el \"arroz de noche\" y los guisos pesados."),
    "merienda": ("La merienda es un snack ligero (150-300 kcal): yogur+fruta, batido, casabe/galleta "
                 "integral con queso, fruta con maní, o huevo duro con fruta."),
}

# Mapa de canonicalización de meal_type es-DO → key del mapa de apropiación. SSOT propio para
# evitar dependencia circular con `graph_orchestrator._SLOT_KEY_MAP` (graph importa constants).
_SLOT_CANON_MAP = {
    "desayuno": "desayuno", "breakfast": "desayuno",
    "almuerzo": "almuerzo", "comida": "almuerzo", "lunch": "almuerzo",
    "cena": "cena", "dinner": "cena",
    "merienda": "merienda", "snack": "merienda", "merienda am": "merienda",
    "merienda pm": "merienda", "media manana": "merienda", "media tarde": "merienda",
    "merienda matutina": "merienda", "merienda vespertina": "merienda",
}


def canonical_slot_key(meal_type: str):
    """[P1-SLOT-APPROPRIATENESS] Normaliza un meal_type es-DO ('Desayuno'/'Cena'/'Merienda AM'…)
    a su key canónica del mapa de apropiación. Devuelve None si no reconoce el slot."""
    return _SLOT_CANON_MAP.get(strip_accents(str(meal_type or "").lower()).strip())


def slot_violations_for_meal_name(name: str, slot_key: str) -> list:
    """[P1-SLOT-APPROPRIATENESS] SSOT del detector de apropiación horaria. Devuelve
    [{label, hard}] de categorías de alimento que NO corresponden al `slot_key`
    (ya canonicalizado: desayuno/almuerzo/cena/merienda). Match WORD-BOUNDARY sobre el
    NOMBRE (anti-falso-positivo: no mira ingredientes; respeta exclusiones de modificadores
    como 'harina de arroz'). Pura → unit-testable. tooltip-anchor: P1-SLOT-APPROPRIATENESS"""
    rules = SLOT_INAPPROPRIATE_FOODS.get(slot_key)
    if not rules:
        return []
    nlow = strip_accents(str(name or "").lower())
    if not nlow:
        return []
    out = []
    for rule in rules:
        if any(strip_accents(ex.lower()) in nlow for ex in rule.get("exclude", ())):
            continue
        for tok in rule["tokens"]:
            try:
                if re.search(r"\b" + re.escape(strip_accents(tok.lower())), nlow):
                    out.append({"label": rule["label"], "hard": rule.get("hardness") == "hard"})
                    break
            except Exception:
                if tok in nlow:
                    out.append({"label": rule["label"], "hard": rule.get("hardness") == "hard"})
                    break
    return out


def slot_ingredient_violations(ingredients, slot_key) -> list:
    """[P2-SLOT-INGREDIENT-RICE · 2026-07-01] (audit v2 slots GAP-1, batch P2-AUDIT-V2-BATCH) Detector
    INGREDIENT-LEVEL solo para la regla más dura del owner: ARROZ EN EL DESAYUNO (hardness=hard). El
    detector de nombre (`slot_violations_for_meal_name`) es name-only A PROPÓSITO (anti-falso-positivo:
    "Panqueques de harina de arroz" no debe flagear), pero eso dejaba evadible el gate hard con un
    nombre inocuo: 'Bowl energético criollo' + '150g arroz blanco' en ingredients pasaba en TODAS las
    superficies. Scope deliberadamente ESTRECHO: solo desayuno + tokens SSOT de arroz + excludes SSOT
    (harina/leche/vinagre de arroz, toque de arroz...). La CENA NO va aquí: ya tiene su pase
    ingredient-driven con AUTOFIX (`_night_rice_autofix` sustituye en vez de rechazar, porque cena-arroz
    es soft). Devuelve [{'label','hard','ingredient'}]. Fail-safe → []. tooltip-anchor: P2-SLOT-INGREDIENT-RICE"""
    try:
        slot = canonical_slot_key(slot_key) or (slot_key if slot_key in SLOT_INAPPROPRIATE_FOODS else None)
        if slot != "desayuno" or not ingredients:
            return []
        out = []
        for ing in ingredients:
            ilow = strip_accents(str(ing).lower())
            if not ilow.strip():
                continue
            if any(exc in ilow for exc in _SLOT_RICE_EXCLUDE):
                continue
            for tok in _SLOT_RICE_TOKENS:
                if re.search(r"\b" + re.escape(tok), ilow):
                    out.append({"label": "arroz/locrio/moro en los INGREDIENTES del desayuno",
                                "hard": True, "ingredient": str(ing)[:80]})
                    break
        return out
    except Exception:
        return []


def build_meal_timing_rules(meal_type: str) -> str:
    """[P1-SLOT-APPROPRIATENESS] SSOT del directivo compacto de coherencia de HORARIO para los
    prompts de UPDATE (swap S3 / chat-modify): qué NO va en este slot + guía positiva es-DO.
    Devuelve '' si el slot no se reconoce. tooltip-anchor: P1-SLOT-APPROPRIATENESS"""
    slot = canonical_slot_key(meal_type)
    if not slot:
        return ""
    parts = []
    rules = SLOT_INAPPROPRIATE_FOODS.get(slot)
    if rules:
        prohibited = "; ".join(r["label"] for r in rules)
        parts.append(
            f"- 🕒 COHERENCIA DE HORARIO ({meal_type}): este plato es para el {slot}. "
            f"NO uses en este horario: {prohibited}."
        )
    hint = SLOT_POSITIVE_HINT.get(slot)
    if hint:
        parts.append(f"- 🍽️ {hint}")
    return ("\n    " + "\n    ".join(parts)) if parts else ""


import unicodedata
import re
from datetime import datetime, timezone, timedelta

def safe_fromisoformat(date_str: str) -> datetime:
    """Parche robusto para Python 3.10 que falla si los milisegundos no son exactamente 0, 3 o 6 dígitos.
    Soporta fechas con 'Z' al final o '+00:00'.
    """
    if date_str.endswith("Z"):
        date_str = date_str[:-1] + "+00:00"
    if "." in date_str:
        date_str = re.sub(r'\.(\d+)', lambda m: '.' + m.group(1).ljust(6, '0')[:6], date_str)
    return datetime.fromisoformat(date_str)


# [P0-A] Bounds para detectar `_plan_start_date` corrupto pero parseable.
# Año mínimo: nada antes de 2024 puede ser un plan legítimo (la app salió después).
# Margen futuro: el shift-plan permite renovar planes 7d/15d/30d, así que la fecha
# puede legítimamente estar hasta `total_days_requested` en el futuro al momento de
# crear el snapshot (caso límite: plan recién renovado). 90 días cubre 30d + holgura
# para no rechazar casos legítimos pero corta cualquier "2099-12-31" que congelaría
# el temporal gate hasta agotar P1-C max retries.
PLAN_START_DATE_MIN_YEAR = 2024
PLAN_START_DATE_MAX_FUTURE_DAYS = 90


def validate_plan_start_date(value, *, now=None):
    """[P0-A] Valida `_plan_start_date` y devuelve (datetime | None, reason | None).

    El temporal gate del chunk system asume que `_plan_start_date` es una fecha
    ISO parseable y razonable. Si llega corrupta:
      - Gibberish (`"abc"`, `"{}"`) → safe_fromisoformat raise → worker muere
        en bucle de retry hasta dead-letter.
      - Out-of-bounds (`"2099-12-31"`, `"1900-01-01"`) → parsea pero el gate se
        diferia hasta agotar `CHUNK_TEMPORAL_GATE_MAX_RETRIES` (~5 min de atasco).

    Esta función centraliza la sanitización: cualquier value que no represente
    una fecha "sana" devuelve (None, reason). El caller debe interpretar el
    None como "ausente" y caer a la cascada de recuperación
    (`grocery_start_date` → `created_at` → bloquear con `no_anchor`).

    Args:
        value: cualquier cosa (typically str). None / no-string / vacío → empty.
        now: datetime UTC opcional para tests; default `datetime.now(timezone.utc)`.

    Returns:
        (datetime UTC-aware, None) si es válida.
        (None, "empty") si es None / vacía / no-string.
        (None, "unparseable:<ExceptionClass>") si no parsea como ISO.
        (None, "out_of_bounds:past") si año < PLAN_START_DATE_MIN_YEAR.
        (None, "out_of_bounds:future") si está más de PLAN_START_DATE_MAX_FUTURE_DAYS adelante.
    """
    if value is None:
        return None, "empty"
    if not isinstance(value, str):
        return None, f"wrong_type:{type(value).__name__}"
    stripped = value.strip()
    if not stripped:
        return None, "empty"
    try:
        parsed = safe_fromisoformat(stripped)
    except (ValueError, TypeError) as exc:
        return None, f"unparseable:{type(exc).__name__}"
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed.year < PLAN_START_DATE_MIN_YEAR:
        return None, "out_of_bounds:past"
    if now is None:
        now = datetime.now(timezone.utc)
    if parsed > now + timedelta(days=PLAN_START_DATE_MAX_FUTURE_DAYS):
        return None, "out_of_bounds:future"
    return parsed, None

def strip_accents(s: str) -> str:
    """Remueve acentos de un string para comparaciones normalizadas.
    Esta es la definición CANÓNICA. Importar desde aquí en todos los módulos."""
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


# [P3-CONDITION-RULES · 2026-06-14] Términos canónicos (SSOT) de detección de condiciones del set
# Pareto clínico DR. Normaliza el input del usuario con strip_accents + lower ANTES de comparar.
# Importar desde aquí en graph_orchestrator (cap determinista de proteína / piso de fibra) Y en
# prompts/plan_generator (directiva de prompt) para que los DOS detectores NUNCA driften — un drift
# abre el modo peligroso "cap aplicado sin guía de prompt" (header capeado, comidas sin sesgo renal).
RENAL_CONDITION_TERMS = (
    "renal", "rinon", "kidney", "erc", "ckd", "nefro", "nephro", "dialisis", "dialysis",
    "insuficiencia renal", "enfermedad renal", "chronic kidney", "glomerul", "nefropat",
    "creatinina alta", "falla renal",
)
DIABETES_CONDITION_TERMS = (
    "diabet", "dm2", "dm-2", "dm 2", "t2dm", "prediabet", "pre-diabet", "hiperglucem",
    "resistencia a la insulina", "resistencia insulinica", "glucemia alta", "azucar alta",
    "intolerancia a la glucosa", "intolerancia a glucosa",
)
# [P3-CONDITION-ENGINE · 2026-06-14] Extensión del set Pareto cardiometabólico DR. Términos sin
# acento (el caller normaliza con strip_accents). Consumidos por el ConditionRuleEngine (condition_rules.py).
HTA_CONDITION_TERMS = (
    "hipertension", "hta", "presion alta", "presion arterial alta", "tension alta",
    "hipertenso", "high blood pressure", "blood pressure",
)
DYSLIPIDEMIA_CONDITION_TERMS = (
    "dislipidemia", "colesterol alto", "colesterol elevado", "trigliceridos altos",
    "hipercolesterolemia", "ldl alto", "hiperlipidemia", "high cholesterol",
)
ANEMIA_CONDITION_TERMS = (
    "anemia", "ferropenica", "ferropenia", "hierro bajo", "ferritina baja",
    "deficiencia de hierro", "iron deficiency",
)
# [P1-CONDITION-COVERAGE · 2026-06-14] Condiciones comunes que faltaban del modelado clínico
# (audit P1-de-precisión 2026-06-14). EMBARAZO/LACTANCIA es el caso de SEGURIDAD fail-hard: nunca
# entregar un déficit calórico (ver el gate en nutrition_calculator). El resto (hipotiroidismo, gota,
# hígado graso, SOP) son ADVISORY: prompt_block citable + gate de derivación FS9 — la regla fina la
# valida el profesional, no el motor (evita enforcement clínico sin revisión humana). Términos SIN
# acento (el caller normaliza con strip_accents) y sobre-inclusivos (dirección segura).
PREGNANCY_CONDITION_TERMS = (
    "embaraz", "gestac", "gestante", "lactan", "lactancia", "amamant",
    "pregnan", "pregnancy", "breastfeed", "postparto", "post parto", "puerperio",
)
HYPOTHYROID_CONDITION_TERMS = (
    "hipotiroid", "hashimoto", "hypothyroid", "tiroides baja", "tiroidea baja",
)
GOUT_CONDITION_TERMS = (
    "gota", "acido urico", "hiperuricemia", "gout", "uric acid",
)
NAFLD_CONDITION_TERMS = (
    "higado graso", "esteatosis", "nafld", "mafld", "fatty liver", "hepatica grasa",
)
PCOS_CONDITION_TERMS = (
    "sop", "ovario poliquistico", "ovarios poliquisticos", "poliquistico", "pcos",
    "sindrome de ovario", "ovarico poliquistico",
)
# [P1-GASTRITIS-RULE · 2026-06-26] (auditoría gap #8) El form expone el chip 'Gastritis'
# (InteractiveQuestions.jsx) → escribe a medicalConditions, pero NINGUNA ConditionRule lo enforzaba (solo
# disparaba el gate FS9 genérico advisory). Términos SIN acento (el caller normaliza con strip_accents) e
# inclusivos (gastritis + reflujo/ERGE + úlcera + acidez). Sobre-inclusión = dirección segura.
GASTRITIS_CONDITION_TERMS = (
    "gastritis", "reflujo", "erge", "gerd", "acidez", "ulcera", "ulcera peptica", "ulcera gastrica",
    "gastric", "gastrico", "agruras", "dispepsia", "esofagitis", "hernia hiatal",
)

# [P1-BARIATRIC-CLINICAL-RULES · 2026-06-27] Tokens de cirugía bariátrica (sleeve/bypass/manga/balón) — SSOT
# compartido por `nutrition_calculator.decide_meals_per_day` (→ 6 comidas pequeñas y frecuentes) y la
# `ConditionRule` bariátrica de `condition_rules.py` (reglas anti-dumping + topes de porción para el pouch).
# Normalizados (minúscula, sin acento); detección por substring (`t in c`). 'sleeve gastric'/'manga gastric'
# (no 'manga'/'sleeve' desnudos, que colisionarían con ropa). tooltip-anchor: P1-BARIATRIC-CLINICAL-RULES
BARIATRIC_CONDITION_TERMS = (
    "bariatr", "cirugia bariatrica", "bypass gastric", "manga gastric", "gastrectom",
    "sleeve gastric", "balon gastric", "balon intragastric", "cirugia de obesidad",
)

def get_reverse_synonyms_map():
    """Crea un diccionario inverso donde la clave es la variante ('pechuga') y el valor es el término base ('pollo').
    Incluye tanto las versiones con acento como sin acento para máxima resiliencia."""
    reverse_map = {}
    for synonyms_dict in [PROTEIN_SYNONYMS, CARB_SYNONYMS, VEGGIE_FAT_SYNONYMS, FRUIT_SYNONYMS]:
        for base, variants in synonyms_dict.items():
            for variant in variants:
                lowered = variant.lower()
                reverse_map[lowered] = base.lower()
                # Para asegurar match con textos sin acentos (e.g. "pure de papas"), agregamos la versión limpia:
                stripped = strip_accents(lowered)
                if stripped not in reverse_map:
                    reverse_map[stripped] = base.lower()
    return reverse_map

GLOBAL_REVERSE_MAP = get_reverse_synonyms_map()
# Ordenamos las variantes de mayor a menor longitud para no reemplazar subpalabras accidentalmente.
# Por ejemplo, reemplazar "habichuelas rojas" antes de "habichuelas".
SORTED_VARIANTS = sorted(GLOBAL_REVERSE_MAP.keys(), key=len, reverse=True)

IGNORED_TRACKING_TERMS = {
    # Condimentos y sazonadores básicos
    "sal", "pimienta", "agua", "ajo", "oregano", "cilantro",
    "limon", "aceite", "soya", "canela", "vinagre", "salsa", "salsas",
    # Hierbas, especias e ingredientes de procesado
    "albahaca", "hierbabuena", "mostaza", "comino", "paprika", "pimenton",
    "laurel", "tomillo", "romero", "jengibre", "curcuma", "perejil",
    "nuez", "moscada", "pimienton", "achiote", "sazon",
    "harina", "tortilla", "maicena", "almidón", "almidon",
    # Términos genéricos nutricionales
    "proteina", "proteinas", "carbohidrato", "carbohidratos",
    "vegetal", "vegetales", "fruta", "frutas", "grasa", "grasas",
    "aderezo", "aderezos", "caldo", "condimento", "condimentos",
    "especias", "hierbas", "azucar", "miel", "polvo", "cucharada", "cucharadita", "taza",
    # [P1-UNKNOWN-CATALOG-FILTER · 2026-06-15] Edulcorante "al gusto" es condimento de cero macros
    # (no hay fila de catálogo ni la merece) — ignorar como azucar/miel para que no contamine unknown_ingredients.
    "edulcorante", "edulcorantes", "estevia", "splenda"
}

# Pre-compilar patrones regex con \b al cargar el módulo (O(1) por llamada posterior)
_SYNONYM_PATTERNS = [
    (re.compile(rf'\b{re.escape(variant)}\b'), GLOBAL_REVERSE_MAP[variant])
    for variant in SORTED_VARIANTS
]

def apply_synonyms(text: str) -> str:
    """Reemplaza variantes por sus términos base en un texto usando patrones pre-compilados."""
    # Ahora que `_SYNONYM_PATTERNS` incluye sinónimos CON y SIN acento, si
    # enviamos un texto sin acento ("pure de papas") sí que hará match. 
    text = text.lower()
    for pattern, base in _SYNONYM_PATTERNS:
        text = pattern.sub(base, text)
    return text

# Regex pre-compilado para stripear cantidades/unidades al inicio de un string de ingrediente.
# Captura patrones como: "200g", "1/2", "3 cdas", "1 lb", "2 tazas de", "100 ml de", etc.
_QUANTITY_PATTERN = re.compile(
    r'^[\d\s/.,]+'                           # Números, fracciones, espacios iniciales
    r'(?:'
        r'g\b|gr\b|kg\b|mg\b|ml\b|l\b|lb\b|lbs\b|oz\b'   # Unidades métricas/imperiales
        r'|cdas?\b|cdtas?\b|cucharadas?\b|cucharaditas?\b'  # Cucharadas
        r'|tazas?\b|vasos?\b|porciones?\b|unidades?\b'      # Medidas de volumen
        r'|rodajas?\b|rebanadas?\b|lascas?\b|tiras?\b'      # Formas de corte
        r'|pizcas?\b|puñados?\b|ramitas?\b|ramas?\b'                  # Cantidades imprecisas
        r'|libras?\b|medias?\b|media\b'                      # Libras / fracciones
        r'|paquetes?\b|paqueticos?\b|fundas?\b|latas?\b|sobres?\b|sobrecitos?\b|chin\b|toques?\b|chorritos?\b|hojitas?\b' # Términos dominicanos y extremos
    r')?'
    r'\s*(?:de\s+)?',                        # "de " opcional que conecta cantidad con ingrediente
    re.IGNORECASE
)

def normalize_ingredient_for_tracking(raw: str) -> str:
    """Normaliza un string crudo de ingrediente para frequency tracking.
    
    Pipeline:  "200g Pechuga de Pollo deshuesada" 
            →  strip quantities →  "pechuga de pollo deshuesada"
            →  strip accents  →  "pechuga de pollo deshuesada" (sin acentos)
            →  apply synonyms  →  "pollo"
    
    Retorna el término base canónico (ej: "pollo", "platano verde", "aguacate").
    
    ⚠️ DIFERENTE a graph_orchestrator._normalize(), que normaliza NOMBRES DE PLATOS
    para anti-repetición y preserva las técnicas de cocción (ej: "plancha", "guisado")
    para poder distinguir preparaciones diferentes del mismo ingrediente.
    """
    if not raw or not raw.strip():
        return ""
    
    # 1. Normalizar minúsculas solamente, ¡NO quitar acentos todavía!
    # De esta manera _QUANTITY_PATTERN sí atrapa 'puñados' (con ñ).
    text = raw.lower().strip()
    
    # 2. Stripear cantidades y unidades del inicio
    text = _QUANTITY_PATTERN.sub('', text).strip()
    
    # Si quedó vacío tras stripear (ej: el input era solo "200g"), devolver el original
    if not text:
         text = raw.lower().strip()
    
    # 3. Ahora SÍ quitamos todos los acentos usando nuestra función canónica
    text = strip_accents(text)
    
    # 4. Aplicar mapa de sinónimos para colapsar a término base
    #    Como text ahora NO tiene acento ("pure de papas"), 
    #    hará match perfecto con la versión sin acentuar en GLOBAL_REVERSE_MAP
    #    Usamos n-gramas (de mayor a menor) para detectar multipalabra.
    words = text.split()
    for n in range(min(4, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            if ngram in GLOBAL_REVERSE_MAP:
                return GLOBAL_REVERSE_MAP[ngram]
    
    # 5. Fallback: si no matcheó ningún sinónimo, devolver el texto limpio
    return text



# ============================================================
# TÉCNICAS DE COCCIÓN Y SUPLEMENTOS
# ============================================================
TECHNIQUE_FAMILIES = {
    "seca": [
        "Horneado Saludable",
        "En Airfryer Crujiente",
        "Asado a la Parrilla",
        "A la Plancha con Cítricos"
    ],
    "húmeda": [
        "Guiso o Estofado Ligero",
        "En Salsa a base de Vegetales Naturales"
    ],
    "transformada": [
        "Desmenuzado (Ropa Vieja)",
        "En Puré o Majado",
        "Croquetas o Tortitas al Horno",
        "Relleno (Ej. Canoas, Vegetales rellenos)"
    ],
    "fresca": [
        "Estilo Ceviche o Fresco",
        "Salteado tipo Wok",
        "Al Vapor con Finas Hierbas"
    ],
    "fusión": [
        "Estilo Fusión Criolla",
        "Estilo Bowl/Poke Tropical",
        "Wrap o Burrito Dominicano"
    ]
}

ALL_TECHNIQUES = [t for techs in TECHNIQUE_FAMILIES.values() for t in techs]

TECH_TO_FAMILY = {}
for family, techs in TECHNIQUE_FAMILIES.items():
    for t in techs:
        TECH_TO_FAMILY[t] = family

# P1-11: Vocabulario de técnicas culinarias COMPLEJAS para análisis de complejidad
# del plan en graph_orchestrator._calculate_complexity_score. Antes el orquestador
# hardcodeaba una lista corta de 7 términos en español; si el LLM usaba sinónimos,
# conjugaciones o términos en inglés (planes bilingües), la métrica colapsaba a
# score=1 (falso negativo) y el guard de complejidad fallaba en silencio.
# Esta lista cubre vocabulario expandido + términos comunes en inglés.
COMPLEX_TECHNIQUE_KEYWORDS = [
    # ES — cocción lenta / al horno (alta complejidad temporal)
    "horno", "horneado", "horneada", "horneados", "horneadas",
    "asado", "asada", "asados", "asadas", "rostizado", "rostizada",
    "guiso", "guisado", "guisada",
    "estofado", "estofada", "estofar",
    "braseado", "braseada", "brasear",
    "lento", "lenta", "cocción lenta", "fuego lento",
    # ES — preparaciones que requieren múltiples pasos previos
    "marinado", "marinada", "marinar", "macerado", "macerar",
    "relleno", "rellena", "rellenar", "rellenado",
    "empanizado", "empanizada", "empanizar", "empanado",
    "desmenuzado", "desmenuzar", "deshilachado", "ropa vieja",
    "majado", "majar", "puré",
    "croqueta", "croquetas", "tortita", "tortitas",
    "fermentado", "fermentar", "fermentación",
    "confitado", "confitar",
    # EN — términos que el LLM bilingüe puede inyectar
    "roast", "roasted", "bake", "baked", "stew", "stewed",
    "braise", "braised", "marinate", "marinated",
    "stuffed", "breaded", "slow-cook", "slow cooked",
    "shredded", "ferment", "fermented", "confit",
]

# P1-11: Stopwords para extracción de "core noun" desde strings de ingredientes.
# Antes este set vivía hardcodeado dentro de `assemble_plan_node` en
# graph_orchestrator (~80 líneas inline). Eso dificultaba mantenimiento: si una
# unidad o adjetivo nuevo aparecía en planes generados (ej. "cdta", "lonjas"),
# había que editar el orquestador para evitar falsos positivos en la validación
# de coherencia receta↔ingredientes. Centralizado aquí para que la fuente única
# de verdad esté junto a otros vocabularios (COMPLEX_TECHNIQUE_KEYWORDS, etc.).
#
# IMPORTANTE: este set y COMPLEX_TECHNIQUE_KEYWORDS son CONCEPTUALMENTE distintos:
#   - RECIPE_INGREDIENT_STOPWORDS = palabras NO-significativas que se descartan al
#     extraer el sustantivo principal de un ingrediente (cantidades, adjetivos
#     de preparación, conectores). Aplica al string de INGREDIENTES.
#   - COMPLEX_TECHNIQUE_KEYWORDS = vocabulario para detectar TÉCNICAS culinarias
#     complejas en RECETAS (cocción al horno, marinado, etc.). Aplica al string
#     de RECETA. NO confundir uno con otro.
RECIPE_INGREDIENT_STOPWORDS = frozenset({
    "cdta", "semillas", "semilla", "guineo", "guineítos", "guineito", "guineitos",
    "esencia", "extracto", "polvo", "jugo", "zumo", "salsa", "pasta", "concentrado",
    "caldo", "gotas", "de", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "taza", "tazas", "cucharada", "cucharadas", "cucharadita", "cucharaditas",
    "cdita", "cditas", "g", "ml", "oz", "libra", "libras", "kg", "litro", "litros",
    "pizca", "al", "gusto", "para", "con", "y", "o",
    "fresco", "fresca", "frescos", "frescas",
    "picado", "picada", "molido", "molida", "rallado", "rallada",
    "cocido", "cocida", "crudo", "cruda", "mediano", "grande", "pequeño",
    "rebanada", "rebanadas", "diente", "dientes", "filete", "filetes",
    "porción", "porcion", "sobre", "proteína", "proteina",
    "carbohidratos", "carbohidrato", "vegetales", "vegetal",
    "grasas", "grasa", "macronutriente", "macronutrientes",
    "opcional", "acompañamiento", "acompañante",
    "unidad", "unidades", "lonja", "lonjas", "pote", "potes", "lata", "latas",
    "puñado", "manojo", "hoja", "hojas", "rama", "ramas",
    "vaso", "vasos", "botella", "botellas", "paquete", "paquetes", "bolsa", "bolsas",
    "gramos", "mililitros", "onzas", "pedazo", "pedazos", "trozo", "trozos",
    "mitad", "cuarto", "tercio", "entero", "entera",
    # Adjetivos de madurez y preparación — si aparecen solos (sin sustantivo) no son ingredientes
    "maduro", "madura", "maduros", "maduras",
    "verde", "verdes",  # excepción: "plátano verde" → "plátano" será el core noun
    "hervido", "hervida", "hervidos", "hervidas",
    "asado", "asada", "asados", "asadas",
    "frito", "frita", "fritos", "fritas",
    "desalado", "desalada", "remojado", "remojada",
    "fileteado", "fileteada",
    "natural", "naturales",
})

# [P1-FORM-11] SSOT con `frontend/src/components/assessment/questions/InteractiveQuestions.jsx`
# (componente `QSupplements` líneas ~1034-1046). El frontend ofrece exactamente
# estas 12 opciones; el backend DEBE conocerlas todas para que `build_supplements_context`
# las traduzca a su nombre legible al inyectarlas al prompt del LLM. ANTES había
# drift: el wizard ofrecía `vegan_protein`, `fat_burner`, `probiotics`, `electrolytes`
# que NO estaban acá → `SUPPLEMENT_NAMES.get(s, s)` devolvía la key snake_case
# cruda al LLM ("DEBES incluir: vegan_protein") y, peor, un cliente legacy podía
# inyectar strings arbitrarios como "esteroides anabolicos" — vector de
# prompt-injection ortogonal al patrón regex de P1-Q8 (que detecta patrones
# textuales, no enums).
#
# Mantenimiento: si se añade/quita una opción en el frontend, actualizar AMBOS
# lados Y `_SUPPLEMENT_ENUM` en `routers/plans.py`. El validador de boundary
# rechaza con 422 cualquier valor fuera del set; el filtro defensivo de
# `build_supplements_context` ignora con warning si el router se saltó.
#
# [P1-FORM-14] SSOT del frontend: `SUPPLEMENTS` en `frontend/src/config/formValidation.js`.
# Las claves de este dict DEBEN coincidir EXACTAMENTE con ese array (y con
# `_SUPPLEMENT_ENUM`). El test `backend/test_p1_form_14_supplements_sync.py`
# parsea ambos lados y falla en CI si detecta drift.
SUPPLEMENT_NAMES = {
    "whey_protein":  "Proteína Whey",
    "vegan_protein": "Proteína Vegana (guisante/arroz/cáñamo)",
    "creatine":      "Creatina Monohidrato",
    "bcaa":          "Aminoácidos BCAA / EAA",
    "pre_workout":   "Pre-Entreno (Cafeína + Beta-Alanina)",
    "fat_burner":    "Quemador de Grasa Termogénico",
    "collagen":      "Colágeno Hidrolizado",
    "multivitamin":  "Multivitamínico Completo",
    "omega3":        "Omega-3 (Aceite de Pescado)",
    "magnesium":     "Magnesio (Citrato o Glicinato)",
    "probiotics":    "Probióticos (Cepas Mixtas)",
    "electrolytes":  "Electrolitos (Sodio + Potasio + Magnesio)",
}

def _get_fast_filtered_catalogs(allergies: tuple, dislikes: tuple, diet: str):
    """Filtra el catálogo dominicano basado en restricciones del usuario O(N) sin Cache Thrashing volátil."""
    filtered_proteins = DOMINICAN_PROTEINS.copy()
    filtered_carbs = DOMINICAN_CARBS.copy()
    filtered_veggies = DOMINICAN_VEGGIES_FATS.copy()
    filtered_fruits = DOMINICAN_FRUITS.copy()
    
    restrictions = list(allergies) + list(dislikes)
    
    if diet in ["vegano", "vegan"]:
        restrictions.extend(["pollo", "cerdo", "res", "pescado", "atún", "huevos", "queso", "salami", "camarones", "chuleta", "longaniza", "carne", "marisco", "lácteo", "leche"])
    elif diet in ["vegetariano", "vegetarian"]:
        restrictions.extend(["pollo", "cerdo", "res", "pescado", "atún", "salami", "camarones", "chuleta", "longaniza", "carne", "marisco"])
    elif diet in ["pescetariano", "pescatarian"]:
        restrictions.extend(["pollo", "cerdo", "res", "salami", "chuleta", "longaniza", "carne"])
        
    if not restrictions:
        return filtered_proteins, filtered_carbs, filtered_veggies, filtered_fruits
        
    normalized_restrictions = [strip_accents(r.lower()) for r in restrictions]
    
    # Reglas genéricas CATCH-ALL de mariscos y carnes
    # [P1-VARIETY-CATALOG-POOLS · 2026-06-27] Pescados/mariscos y carnes/aves ESPECÍFICOS del catálogo (202
    # alimentos). Crítico de SEGURIDAD: el regex \bpescado\b/\bcarne\b NO matchea 'salmon'/'mero'/'pulpo'/
    # 'conejo'/'chivo'/'pavo' por nombre → sin esto, al expandir las pools de variedad un VEGANO/VEGETARIANO
    # recibiría esas proteínas. Vegano y vegetariano añaden "carne"+"marisco" → disparan AMBOS catch-alls;
    # pescetariano añade SOLO "carne" → se le excluyen carnes de tierra pero pescado/mariscos quedan permitidos.
    if any(r in ["mariscos", "seafood", "marisco"] for r in normalized_restrictions):
        normalized_restrictions.extend(["camaron", "camarones", "pescado", "atun",
            "mero", "tilapia", "salmon", "bacalao", "sardina", "sardinas", "arenque", "merluza",
            "pulpo", "calamar", "mejillones", "mejillon", "cangrejo", "langosta", "langostino", "lambi"])
    if any(r in ["carne", "carnes", "meat"] for r in normalized_restrictions):
        normalized_restrictions.extend(["pollo", "cerdo", "res", "chuleta", "longaniza", "salami",
            "pavo", "conejo", "chivo", "cabro", "higado", "costilla", "jamon", "muslo", "pernil"])
    # [P2-VARIETY-CATALOG-NOT-FILTERED · 2026-06-22] (audit fresco P2-4) Catch-all de categorías de alérgenos
    # cuyos chips llegan como CATEGORÍA ("lácteos"/"frutos secos"/"huevo"...) y NO matchean los nombres
    # concretos del catálogo (Queso, Yogurt, Nueces, Huevos) — análogo al catch-all de mariscos/carne de arriba.
    # Pre-fix solo mariscos/carne se expandían → un alérgico a lácteos/nueces/huevo veía esos alimentos en el
    # pool de variedad (el allergen guard determinista downstream los bloqueaba, pero subía rechazo→retry).
    # Sesgo a sobre-filtrar. tooltip-anchor: P2-VARIETY-CATALOG-NOT-FILTERED
    if any(r in ["lacteos", "lacteo", "lactosa", "dairy"] for r in normalized_restrictions):
        normalized_restrictions.extend(["leche", "queso", "yogur", "yogurt", "mantequilla", "crema",
                                        "ricotta", "mozzarella", "parmesano", "requeson", "suero de leche"])
    if any(r in ["frutos secos", "nueces", "nuts", "tree nuts"] for r in normalized_restrictions):
        normalized_restrictions.extend(["almendra", "nuez", "maranon", "pistacho", "avellana", "merey", "anacardo"])
    if any(r in ["mani", "cacahuate", "peanut", "peanuts"] for r in normalized_restrictions):
        normalized_restrictions.extend(["mani", "cacahuate", "mantequilla de mani"])
    if any(r in ["huevo", "huevos", "egg", "eggs"] for r in normalized_restrictions):
        normalized_restrictions.extend(["huevo", "clara", "yema", "mayonesa"])
    if any(r in ["gluten", "trigo", "wheat"] for r in normalized_restrictions):
        normalized_restrictions.extend(["trigo", "pan", "pasta", "harina de trigo", "galleta"])
    if any(r in ["soya", "soja", "soy"] for r in normalized_restrictions):
        normalized_restrictions.extend(["soya", "soja", "tofu", "edamame"])

    # [OPTIMIZACIÓN O(1)] Compilar un único patrón maestro ultra veloz
    import re
    or_pattern = '|'.join(map(re.escape, normalized_restrictions))
    fast_regex = re.compile(rf'\b({or_pattern})\b')
    
    def is_allowed(item):
        item_normalized = strip_accents(item.lower())
        return not fast_regex.search(item_normalized)
        
    filtered_proteins = [p for p in filtered_proteins if is_allowed(p)]
    filtered_carbs = [c for c in filtered_carbs if is_allowed(c)]
    filtered_veggies = [v for v in filtered_veggies if is_allowed(v)]
    filtered_fruits = [f for f in filtered_fruits if is_allowed(f)]
    
    return filtered_proteins, filtered_carbs, filtered_veggies, filtered_fruits

# ============================================================
# BASE DE DATOS CLINICA / DIGESTIVA (RAG CULINARIO)
# ============================================================
CULINARY_KNOWLEDGE_BASE = """
<biblioteca_culinaria_local>
[BASE DE DATOS CLÍNICA DE PLATOS DOMINICANOS]
Mofongo: 5-6 horas de digestión (Fritura profusa + almidón denso). Peligro de reflujo nocturno y pico insulínico si se consume antes de dormir.
Mangú (Los Tres Golpes): 4-5 horas de digestión aguda. Altísima carga de grasas saturadas (salami/queso frito) combinadas con carbohidrato puro.
La Bandera (arroz, habichuela, carne, concón): 4.5 horas. Demasiada carga glucémica para horarios sin desgaste físico posterior.
Yaroa: 6+ horas de digestión. Bomba de grasas trans/saturadas y carbohidratos fritos. Arruina el ciclo REM del sueño.
Pica Pollo / Chimi: 5+ horas. Exceso de aceites hidrogenados e irritantes gástricos.
Sancocho: 4-5 horas. Extrema condensación de viandas pesadas y caldos grasos.
</biblioteca_culinaria_local>
"""

def _to_base_unit(qty: float, unit: str):
    unit = unit.lower() if unit else 'unidad'
    # Weights -> g
    if unit in ['g', 'gr', 'gramos', 'gramo']: return qty, 'g'
    if unit in ['kg', 'kilo', 'kilos', 'kilogramos', 'kilogramo']: return qty * 1000.0, 'g'
    if unit in ['lb', 'lbs', 'libra', 'libras']: return qty * 453.592, 'g'
    if unit in ['oz', 'onza', 'onzas']: return qty * 28.3495, 'g'
    
    # Volumes -> ml
    if unit in ['ml', 'mililitro', 'mililitros']: return qty, 'ml'
    if unit in ['l', 'litro', 'litros']: return qty * 1000.0, 'ml'
    if unit in ['taza', 'tazas']: return qty * 236.588, 'ml'
    if unit in ['cda', 'cucharada', 'cucharadas']: return qty * 14.7868, 'ml'
    if unit in ['cdta', 'cucharadita', 'cucharaditas', 'cdita']: return qty * 4.92892, 'ml'
    
    # Extreme Abstract Dominican Terms -> nominal weight
    if unit in ['chin', 'pizca', 'pizcas', 'toque', 'toques', 'chorrito', 'chorritos', 'puñado', 'puñados', 'ramita', 'ramitas', 'hojita', 'hojitas']: return qty * 5.0, 'g'
    
    # Informal Containers -> always track as units for delta
    if unit in ['paquete', 'paquetes', 'paquetico', 'paqueticos', 'funda', 'fundas', 'sobre', 'sobres', 'sobrecito', 'sobrecitos', 'lata', 'latas', 'pote', 'potes']: return qty, 'unidad'
    
    return qty, unit

def _format_unit_qty(base_qty: float, base_unit: str) -> str:
    """Para mensajes de error legibles."""
    if base_unit == 'g':
        if base_qty >= 1000: return f"{round(base_qty/1000.0, 2)} kg"
        if base_qty >= 226: return f"{round(base_qty/453.592, 2)} lbs"
        return f"{round(base_qty)} g"
    if base_unit == 'ml':
        if base_qty >= 1000: return f"{round(base_qty/1000.0, 2)} L"
        if base_qty >= 220: return f"{round(base_qty/236.588, 1)} tazas"
        return f"{round(base_qty)} ml"
    return f"{round(base_qty, 2)} {base_unit}"

VOLUMETRIC_DENSITIES = {
    # Carbohidratos (Crudos y cocidos en volumen)
    "arroz blanco": 0.845,
    "arroz integral": 0.845,
    "arroz": 0.845,
    "avena": 0.380,
    "harina": 0.550,
    "harina de maiz": 0.550,
    "pasta": 0.420,
    "quinoa": 0.720,
    
    # Granos (Leguminosas en volumen)
    "lentejas": 0.810,
    "habichuelas rojas": 0.840,
    "habichuelas negras": 0.840,
    "garbanzos": 0.840,
    "habichuelas": 0.840,
    "gandules": 0.840,
    
    # Proteínas (Picadas/Desmenuzadas/Líquidas)
    "pollo": 0.634,
    "res": 0.634,
    "cerdo": 0.634,
    "pescado": 0.600,
    "carne molida": 0.850,
    "atun": 0.600,
    "tofu": 0.950,
    "clara de huevo": 1.03,
    
    # Lácteos y Similares
    "queso de freir": 0.500,
    "queso rallado": 0.450,
    "queso": 0.500,
    "yogurt": 1.050,
    "yogurt griego": 1.050,
    "leche": 1.030,
    "leche en polvo": 0.500,
    "queso crema": 0.950,
    "ricotta": 0.950,
    "cottage": 0.950,
    
    # Grasas y Semillas
    "aceite": 0.920,
    "aceite de oliva": 0.920,
    "aceite de coco": 0.920,
    "mantequilla": 0.960,
    "mayonesa": 0.960,
    "mantequilla de mani": 0.960,
    "mani": 0.600,
    "nueces": 0.600,
    "almendras": 0.600,
    "nueces/almendras": 0.600,
    "semillas de chia": 0.650,
    "semillas de linaza": 0.650,
    
    # Vegetales y Frutas (Picados en taza/volumen)
    "cebolla": 0.600,
    "tomate": 0.600,
    "aji": 0.600,
    "pimiento": 0.600,
    "espinaca": 0.200,
    "lechuga": 0.150,
    "fruta picada": 0.650,
    "pina": 0.650,
    "melon": 0.650,
    "manzana": 0.650,
    "salsa de tomate": 1.050,
    "pasta de tomate": 1.050,
    "salsa de soya": 1.050,
}

UNIT_WEIGHTS = {
    # Víveres y Carbohidratos (Unidad entera o porción estándar)
    "platano verde": 280.0,
    "platano maduro": 280.0,
    "guineito verde": 100.0,
    "guineitos verdes": 100.0,
    "guineos verdes": 100.0,
    "yuca": 400.0,
    "batata": 220.0,
    "papa": 150.0,
    "papas": 150.0,
    "name": 300.0,
    "yautia": 250.0,
    "pan integral": 30.0,
    "pan": 30.0,
    "pan de agua": 60.0,
    "casabe": 20.0,
    "tortilla": 45.0,
    "wrap": 45.0,
    "plantilla": 45.0,

    # Proteínas (Unidades y raciones rápidas)
    "huevos": 50.0,
    "huevo": 50.0,
    "chuleta": 150.0,
    "longaniza": 100.0,
    "salami": 40.0,
    "salami dominicano": 40.0,
    "queso": 25.0,
    "queso crema": 226.0, # 8pz
    "queso cottage": 453.0, # 16oz
    "yogurt griego": 453.0, # 16oz
    "yogurt": 453.0,
    "jamon": 20.0,
    "pechuga de pavo": 20.0,
    "pechuga de pollo": 200.0,
    "filete de pescado": 150.0,
    "lata de atun": 120.0,
    "atun": 120.0,
    "sardina": 106.0,

    # Frutas y Vegetales (Unidades enteras)
    "guineo": 120.0,
    "guineo maduro": 120.0,
    "manzana": 150.0,
    "naranja": 130.0,
    "limon": 60.0,
    "chinola": 90.0,
    "aguacate": 250.0,
    "tomate": 120.0,
    "cebolla": 110.0,
    "ajo": 5.0,
    "diente de ajo": 5.0,
    "aji": 100.0,
    "pimiento": 100.0,
    "zanahoria": 75.0,
    "pepino": 200.0,
    "berenjena": 250.0,
    "tayota": 250.0,
    "molondron": 15.0,
    "molondrones": 15.0,

    # Frutas tropicales (Unidades enteras - Mercado Dominicano)
    "mango": 300.0,
    "mango maduro": 300.0,
    "pina": 1500.0,
    "piña": 1500.0,
    "lechosa": 800.0,
    "papaya": 800.0,
    "fresa": 15.0,
    "sandia": 3000.0,
    "melon": 1200.0,

    # Vegetales grandes / Crucíferas (Unidades enteras)
    "auyama": 500.0,
    "brocoli": 300.0,
    "coliflor": 500.0,
    "repollo": 600.0,
    "vainitas": 200.0,
    "habichuelas verdes": 200.0,
    "aji dulce": 10.0,
    "ajies": 100.0,
}

# [P5-CASABE-UNIT-WEIGHT · 2026-06-23] Formas físicas que comparten término
# nutricional/sinónimo con otro alimento (vía CARB_SYNONYMS → normalize_ingredient_for_tracking)
# pero cuyo PESO-POR-UNIDAD físico es MUY distinto. Sin este override, el pantry guard
# heredaba el peso-por-unidad del término normalizado y rechazaba por `over_limit` falso.
# Caso observado en prod (swap cravings, corr=ae089712, 2026-06-23): el LLM pidió
# "1 unidad de Casabe (~30g, una hoja pequeña)" → casabe normaliza a "yuca" (es yuca
# deshidratada, misma nutrición — test_synonyms lo exige) → heredaba UNIT_WEIGHTS["yuca"]=400g
# (¡una raíz entera!) → 400g > 281g de tu paquete × 1.30 → rechazo → retry innecesario (~+10s).
# La nutrición/sinónimo de casabe→yuca se PRESERVA; SOLO se corrige el peso-por-unidad físico
# para la matemática del guard. Añade aquí cualquier forma análoga (arepita de yuca, etc.).
_PHYSICAL_UNIT_WEIGHT_OVERRIDES = {
    "casabe": 20.0,  # una hoja/torta de casabe ≈ 20g (NO 400g de yuca entera)
}


def _get_converted_quantity(req_qty: float, req_unit: str, dispo_unit: str, base_name: str, original_name: str | None = None) -> float | None:
    """Convierte matemáticamente entre familias de unidades incompatibles (Masa/Vol/Unidad).

    [P5-CASABE-UNIT-WEIGHT] `original_name` (opcional): nombre crudo del ingrediente ANTES de
    normalizar. Si contiene una forma de `_PHYSICAL_UNIT_WEIGHT_OVERRIDES`, su peso-por-unidad
    físico GANA sobre el del `base_name` normalizado (ej: casabe usa 20g, no los 400g de yuca).
    """
    if not base_name: return None
    density = VOLUMETRIC_DENSITIES.get(base_name)
    unit_weight = UNIT_WEIGHTS.get(base_name)

    if density is None:
        for k, v in VOLUMETRIC_DENSITIES.items():
            if k in base_name or base_name in k:
                density = v
                break
    if unit_weight is None:
        for k, v in UNIT_WEIGHTS.items():
            if k in base_name or base_name in k:
                unit_weight = v
                break

    # [P5-CASABE-UNIT-WEIGHT] Override del peso-por-unidad físico según el nombre ORIGINAL.
    if original_name:
        _orig = strip_accents(original_name.lower().strip())
        for _phys_key, _phys_w in _PHYSICAL_UNIT_WEIGHT_OVERRIDES.items():
            if _phys_key in _orig:
                unit_weight = _phys_w
                break

    if req_unit == 'g' and dispo_unit == 'ml' and density: return req_qty / density
    if req_unit == 'ml' and dispo_unit == 'g' and density: return req_qty * density
    if req_unit == 'g' and dispo_unit == 'unidad' and unit_weight: return req_qty / unit_weight
    if req_unit == 'unidad' and dispo_unit == 'g' and unit_weight: return req_qty * unit_weight
    if req_unit == 'ml' and dispo_unit == 'unidad' and density and unit_weight: return (req_qty * density) / unit_weight
    if req_unit == 'unidad' and dispo_unit == 'ml' and density and unit_weight: return (req_qty * unit_weight) / density
    return None

def validate_ingredients_against_pantry(generated_ingredients: list, pantry_ingredients: list, strict_quantities: bool = True, tolerance: float = 1.30, allow_external_count: int = 0) -> bool | str:
    """
    Función guardrail estricta y matemática. Comprueba:
    1. Que todos los ingredientes generados estén en la despensa.
    2. (Si strict_quantities=True) Que las CANTIDADES generadas no superen el Ledger de la despensa.

    En modo rotación (strict_quantities=False), solo se valida #1 (existencia),
    ya que el LLM redistribuye las mismas macros con proporciones distintas.

    [P2-SWAP-CONSISTENCY · 2026-05-22] `allow_external_count` (default 0):
    permite hasta N ingredientes "unauthorized" (= no encontrados en pantry,
    ni por substring ni por vector cosine match) sin abortar. Sirve para
    swap_reason ∈ {cravings, weekend} donde el user quiere un antojo /
    plato festivo y el sistema permite 1-2 compras externas pequeñas.
    NO relaja `over_limit` (cantidades excedidas siguen forzando retry —
    esas son problema cuantitativo distinto a "ingrediente nuevo").
    Tooltip-anchor: P2-SWAP-CONSISTENCY-ALLOW-EXTERNAL.
    """
    if not pantry_ingredients:
        logger.debug("⚠️ [PANTRY GUARD] Lista de despensa vacía — guardrail desactivado.")
        return True
        
    try:
        from shopping_calculator import _parse_quantity
    except ImportError:
        logger.error("❌ Falló import de _parse_quantity")
        _parse_quantity = lambda x: (0, 'u', x)

    pantry_ledger = {}
    pantry_bases = set()
    
    # [P3-PANTRY-GUARD-UNICODE-FRACTIONS · 2026-05-23] Caso real verificado
    # log 2026-05-23 00:58-01:01: el aggregator emite display strings con
    # fracciones Unicode tipo "1 paquete (½ lb) de Queso blanco". El regex
    # pre-fix solo capturaba `\d+` → `½` no matcheaba → cae al fallback
    # UNIT_WEIGHTS que no tiene "Queso blanco" → conversion devuelve
    # default conservador (~5g) → rechazo over_limit por 100g (incluso
    # 65g) cuando el user tiene ½ lb = 227g.
    #
    # Fix: extender el regex para capturar fracciones Unicode + helper que
    # las convierte a decimal.
    _container_weight_re = re.compile(
        r'\(([¼½¾⅓⅔⅛⅜⅝⅞⅙⅚]|\d+(?:\.\d+)?)\s*(g|gr|kg|oz|ml|l|lb|lbs)\)',
        re.IGNORECASE
    )
    # Map de fracciones Unicode → decimal. NumeroSet del bloque Unicode
    # "Number Forms" más comunes en es-DO (½, ¼, ¾, ⅓, ⅔ son los que el
    # aggregator emite). Los menos comunes (eighths, sixths) los añadimos
    # por completeness defensiva.
    _UNICODE_FRACTION_MAP = {
        "½": 0.5, "¼": 0.25, "¾": 0.75,
        "⅓": 1/3, "⅔": 2/3,
        "⅛": 0.125, "⅜": 0.375, "⅝": 0.625, "⅞": 0.875,
        "⅙": 1/6, "⅚": 5/6,
    }

    def _parse_fraction_or_number(s: str) -> float:
        """Parse un string que puede ser un Unicode fraction (`½`) O un
        número decimal (`0.5` / `1.5`). Si es fraction, lookup en el map.
        Si es número, ``float()``. Defensivo: si nada match, retorna 0.0
        (caller hará fallback)."""
        if not isinstance(s, str):
            return 0.0
        s_clean = s.strip()
        if s_clean in _UNICODE_FRACTION_MAP:
            return _UNICODE_FRACTION_MAP[s_clean]
        try:
            return float(s_clean)
        except (ValueError, TypeError):
            return 0.0
    
    for p in pantry_ingredients:
        norm = normalize_ingredient_for_tracking(p)
        if norm: pantry_bases.add(norm)
        pantry_bases.add(strip_accents(p.lower().strip()))
        
        qty, unit, name = _parse_quantity(p)
        base_norm = normalize_ingredient_for_tracking(name) or strip_accents(name.lower().strip())
        
        if not base_norm: continue
        
        # Inteligencia de Contenedor: Si el string contiene peso real embebido
        # (e.g., "1 Paquete (500g)"), usar ese peso en lugar de la unidad abstracta.
        # Esto evita que "1 Paquete de Pan integral" se registre como 30g (1 rebanada)
        # cuando en realidad el paquete pesa 500g.
        container_match = _container_weight_re.search(p)
        if container_match and unit in ['paquete', 'paquetes', 'pote', 'potes', 'cartón', 'carton',
                                         'funda', 'fundas', 'lata', 'latas', 'sobre', 'sobres',
                                         'fundita', 'funditas', 'botella', 'botellas']:
            # [P3-PANTRY-GUARD-UNICODE-FRACTIONS · 2026-05-23] Acepta
            # tanto decimales ("0.5") como fracciones Unicode ("½").
            container_qty = _parse_fraction_or_number(container_match.group(1))
            container_unit = container_match.group(2).lower()
            # El peso total = cantidad de contenedores × peso por contenedor
            real_weight = qty * container_qty
            base_qty, base_unit = _to_base_unit(real_weight, container_unit)
        else:
            base_qty, base_unit = _to_base_unit(qty, unit)
        
        if base_norm not in pantry_ledger:
            pantry_ledger[base_norm] = {}
        if base_unit not in pantry_ledger[base_norm]:
            pantry_ledger[base_norm][base_unit] = 0.0
            
        pantry_ledger[base_norm][base_unit] += base_qty

    if not pantry_bases:
        return True
        
    unauthorized = []
    over_limit = []
    
    for item in generated_ingredients:
        item = item.strip()
        if not item: continue
        
        gen_qty, gen_unit, gen_name = _parse_quantity(item)
        gen_base_qty, gen_base_unit = _to_base_unit(gen_qty, gen_unit)
        base = normalize_ingredient_for_tracking(gen_name) or strip_accents(gen_name.lower().strip())
        
        item_lower = strip_accents(item.lower())
        allowed_condiments = {
            "sal", "pimienta", "agua", "ajo", "oregano", "cilantro", 
            "limon", "aceite", "soya", "canela", "vinagre"
        }
        
        if base in allowed_condiments or any(c in item_lower for c in allowed_condiments):
            continue
            
        matched_pantry_key = None
        if base in pantry_ledger:
            matched_pantry_key = base
        else:
            # 1. Intentar Regex/Subcadena tradicional
            for pb in pantry_bases:
                if pb and len(pb) > 2 and (pb in item_lower or pb in base):
                    matched_pantry_key = pb if pb in pantry_ledger else None
                    if not matched_pantry_key:
                        for k in pantry_ledger.keys():
                            if k in pb or pb in k:
                                matched_pantry_key = k
                                break
                    break
                    
            # 2. Si falló el match tradicional, intentamos Similitud Coseno (Mejora 4)
            # [P0-DEEPSEEK-MIGRATION] `get_embedding` puede retornar None
            # (provider disabled/fallo) — en ese caso se omite el matching
            # semántico sin warning por-item y se cae al flujo no-vector.
            if not matched_pantry_key and len(base) > 2:
                try:
                    gen_emb = get_embedding(base)
                    if gen_emb is not None:
                        # [P1-COHERE-EMBED-V4] key del cache versionada por
                        # espacio vectorial (mismo patrón que _embedding_cache).
                        from embeddings_provider import get_embeddings_model_id
                        _emb_model_id = get_embeddings_model_id()
                        best_match = None
                        best_score = -1.0

                        for p_key in pantry_ledger.keys():
                            _p_cache_key = (_emb_model_id, p_key)
                            p_emb = _pantry_embeddings_cache.get(_p_cache_key)
                            if p_emb is None:
                                p_emb = get_embedding(p_key)
                                if p_emb is None:
                                    continue
                                _pantry_embeddings_cache[_p_cache_key] = p_emb
                            score = cosine_similarity(gen_emb, p_emb)

                            if score > best_score:
                                best_score = score
                                best_match = p_key

                        if best_score > 0.85: # Threshold estricto para evitar falsos positivos
                            logger.debug(f"🧠 [VECTOR MATCH] '{base}' -> '{best_match}' (score: {best_score:.3f})")
                            matched_pantry_key = best_match
                except Exception as e:
                    logger.warning(f"Error en Vector Search para '{base}': {e}")
                    
        if not matched_pantry_key:
            unauthorized.append(item)
            continue
            
        if not strict_quantities:
            continue  # En modo rotación, solo validamos existencia (arriba), no cantidades
            
        if gen_qty > 0 and matched_pantry_key in pantry_ledger:
            available_units_for_item = pantry_ledger[matched_pantry_key]
            
            # Tolerancia Inteligente: Si el inventario para TODAS las unidades de este ítem es 0,
            # no es un "exceso" sino un "faltante". Lo dejamos pasar — aparecerá en la lista de compras.
            total_available = sum(available_units_for_item.values())
            if total_available <= 0.01:
                logger.debug(f"🛒 [PANTRY GUARD] '{gen_name}' tiene inventario 0 — skip (se comprará).")
                continue
            
            if gen_base_unit in available_units_for_item:
                available_qty = available_units_for_item[gen_base_unit]
                if available_qty <= 0.01:
                    logger.debug(f"🛒 [PANTRY GUARD] '{gen_name}' en {gen_base_unit} tiene stock 0 — skip.")
                    continue
                if gen_base_qty > (available_qty * tolerance):
                    formatted_req = _format_unit_qty(gen_base_qty, gen_base_unit)
                    formatted_avail = _format_unit_qty(available_qty, gen_base_unit)
                    over_limit.append(f"[{item}] (Pediste {formatted_req}, límite: {formatted_avail})")
                else:
                    available_units_for_item[gen_base_unit] -= gen_base_qty
            else:
                # Conversión Matemática Activa (Mejora 1)
                converted = False
                for dispo_unit, available_qty in available_units_for_item.items():
                    if available_qty <= 0.01:
                        converted = True
                        break
                    req_qty_in_dispo_unit = _get_converted_quantity(gen_base_qty, gen_base_unit, dispo_unit, matched_pantry_key, original_name=gen_name)
                    if req_qty_in_dispo_unit is not None:
                        if req_qty_in_dispo_unit > (available_qty * tolerance):
                            formatted_req = _format_unit_qty(gen_base_qty, gen_base_unit)
                            formatted_avail = _format_unit_qty(available_qty, dispo_unit)
                            over_limit.append(f"[{item}] (Pediste {formatted_req}, convertido dinámicamente excede tu inventario de {formatted_avail})")
                        else:
                            available_units_for_item[dispo_unit] -= req_qty_in_dispo_unit
                        converted = True
                        break 
                
                if not converted:
                    # Solución 3: Conversor por default (5g aprox por porción imprecisa: "pizca", "ramita", "chorrito")
                    fallback_g = gen_base_qty * 5.0
                    for dispo_unit, available_qty in available_units_for_item.items():
                        req_qty_in_dispo_unit = None
                        
                        # Conversiones directas de masa desde fallback a la unidad de la despensa
                        if dispo_unit in ['g', 'kg', 'lb', 'oz', 'ml', 'unidad']:
                            req_qty_in_dispo_unit = _get_converted_quantity(fallback_g, 'g', dispo_unit, matched_pantry_key, original_name=gen_name)
                            if req_qty_in_dispo_unit is None:
                                if dispo_unit == 'g': req_qty_in_dispo_unit = fallback_g
                                elif dispo_unit == 'kg': req_qty_in_dispo_unit = fallback_g / 1000.0
                                elif dispo_unit == 'lb': req_qty_in_dispo_unit = fallback_g / 453.592
                                elif dispo_unit == 'oz': req_qty_in_dispo_unit = fallback_g / 28.3495
                                elif dispo_unit == 'ml': req_qty_in_dispo_unit = fallback_g # Asumir densidad agua para cosas raras
                        
                        if req_qty_in_dispo_unit is not None:
                            logger.debug(f"🔧 [PANTRY GUARD] Aplicando fallback de 5g/ut para '{gen_name}' ({gen_base_qty} {gen_base_unit} -> {req_qty_in_dispo_unit:.2f} {dispo_unit})")
                            if req_qty_in_dispo_unit > (available_qty * tolerance):
                                formatted_req = _format_unit_qty(gen_base_qty, gen_base_unit)
                                formatted_avail = _format_unit_qty(available_qty, dispo_unit)
                                over_limit.append(f"[{item}] (Pediste {formatted_req}, convertido con fallback [~{fallback_g}g] excede tu inventario de {formatted_avail})")
                            else:
                                available_units_for_item[dispo_unit] -= req_qty_in_dispo_unit
                            converted = True
                            break
                            
                    if not converted:
                        logger.debug(f"⚠️ [PANTRY GUARD] Unidades asintóticas TOTALMENTE irresolubles para {gen_name} (req: {gen_base_unit}). Aprobación flexible.")
            
    # [P2-SWAP-CONSISTENCY · 2026-05-22] Si el caller permite ingredientes
    # externos (cravings/weekend), consume `unauthorized` hasta el cap;
    # `over_limit` permanece intocado (cantidades excedidas siempre fallan).
    if allow_external_count > 0 and unauthorized and len(unauthorized) <= allow_external_count:
        logger.info(
            f"📦 [PANTRY GUARD] {len(unauthorized)} ingredientes externos "
            f"permitidos (allow_external_count={allow_external_count}): "
            f"{unauthorized}. swap_reason probablemente cravings/weekend."
        )
        unauthorized = []

    if unauthorized or over_limit:
        error_msg = "ERRORES DE DESPENSA HALLADOS OBLIGANDO A CORREGIR:\n"
        if unauthorized:
            error_msg += f"- Ingredientes COMPLETAMENTE INEXISTENTES en inventario: {', '.join(unauthorized)}.\n"
        if over_limit:
            error_msg += f"- Excediste tus CANTIDADES (Tu inventario restringe esto matemáticamente): {', '.join(over_limit)}.\n"

        error_msg += "Corrige tu respuesta bajando las porciones estrictamente numéricas al límite exacto, O eliminando/sustituyendo ingredientes."
        logger.warning(f"🚨 [PANTRY GUARD] RECHAZO | unauthorized={len(unauthorized)} | over_limit={len(over_limit)}")
        return error_msg
        
    logger.debug(f"✅ [PANTRY GUARD] APROBADO (Cantidades & Confiabilidad validadas)")
    return True


def compute_household_multiplier(source: dict | None) -> float:
    """[P1-3] Multiplier efectivo del hogar para escalar cantidades.

    Si `householdComposition: {adults, children}` está presente, devuelve
    `adults + children × MEALFIT_CHILDREN_MULTIPLIER` (default 0.6, clamp [0.3, 1.0]).
    Si no, fallback a `householdSize: int` legacy.

    Mínimo siempre 1.0 (planner requiere ≥1 persona).

    [P3-PDF-POLISH-4 · 2026-05-14] Cap superior vía knob
    `MEALFIT_MAX_HOUSEHOLD_SIZE` (default 20, clamp [1, 100]). Pre-fix no
    había límite: un POST adversarial con `householdSize=999999` producía
    listas absurdas + DB write bloated + 3× compute heavy en
    `/api/plans/recalculate-shopping-list`. El cap se aplica al resultado
    final (post-composition o post-householdSize) para que el clamp valga
    igual cuando viene de `{adults, children}` que del int legacy.
    Tooltip-anchor: P3-PDF-POLISH-4-B-MULTIPLIER.

    Acepta tanto `form_data` (assessment) como `health_profile` (DB) — ambos
    usan las mismas keys.
    """
    # [P3-PDF-POLISH-4-B-MULTIPLIER · 2026-05-14] Cap superior shared para
    # todos los callsites (recalc + planner + cron). Auto-registra knob.
    _max_household = _env_int("MEALFIT_MAX_HOUSEHOLD_SIZE", 20)
    _max_household = max(1, min(_max_household, 100))

    if not isinstance(source, dict):
        return 1.0

    composition = source.get("householdComposition")
    if isinstance(composition, dict):
        try:
            adults = max(0, int(composition.get("adults", 0) or 0))
            children = max(0, int(composition.get("children", 0) or 0))
            if adults + children > 0:
                # [P2-1 · 2026-05-08] `_env_float` registra en `_KNOBS_REGISTRY`.
                child_mult = _env_float("MEALFIT_CHILDREN_MULTIPLIER", 0.6)
                child_mult = max(0.3, min(child_mult, 1.0))
                raw = float(adults) + float(children) * child_mult
                return max(1.0, min(raw, float(_max_household)))
        except (TypeError, ValueError):
            pass

    try:
        raw = float(source.get("householdSize") or source.get("household_size") or 1)
        return max(1.0, min(raw, float(_max_household)))
    except (TypeError, ValueError):
        return 1.0


# ──────────────────────────────────────────────────────────────────────────────
# [P1-AUDIT-HIST-7 · 2026-05-09] SSOT del catálogo de events de
# `chunk_lesson_telemetry` que cuentan como LECCIONES semánticas (vs
# métricas mecánicas / de salud).
#
# Movido aquí desde `routers/plans.py:_LESSON_COUNT_EVENT_WHITELIST` para
# eliminar el riesgo de divergencia entre call sites. Antes:
#   - `routers/plans.py` definía la tupla in-place.
#   - `cron_tasks.py` emitía events sin importar la constante.
#   - El test `test_p1_hist_audit_5_lesson_event_whitelist.py` leía el
#     literal del módulo plans para validar; cualquier consumidor adicional
#     (admin tool, monitoring, dashboard) tendría que duplicar la lista.
#
# Ahora cualquier consumidor importa `LESSON_COUNT_EVENT_WHITELIST` desde
# `constants` y opera sobre la misma tupla — drift cero por construcción.
#
# Catálogo emitido por `cron_tasks.py` (search por `event="..."` en call
# sites de `_record_chunk_lesson_telemetry`):
#
#   LECCIONES (whitelist):
#     - lesson_synthesized_low_confidence: el sistema generó una lección
#       proxy desde plan_days porque el chunk no persistió learning_metrics.
#       Es learning real (aunque baja confianza).
#     - synth_propagated_to_prompt: lección sintetizada llegó al prompt del
#       LLM del próximo chunk → influyó en la generación.
#     - recent_lessons_partial_synthesis: síntesis parcial (algunos chunks
#       tenían learning_metrics, otros no) — entry SÍ contribuyó al
#       recent_lessons del LLM.
#     - indefinite_pause_unblocked: chunk pausado se desbloqueó con N
#       lecciones recuperadas del fallback de síntesis.
#
#   MÉTRICAS MECÁNICAS (excluidas — son señales de salud, no "aprendizaje"
#   desde la perspectiva del usuario):
#     - synth_schema_invalid / synth_schema_partial_invalid: descarte por
#       validación; el sistema NO aprendió esas lecciones.
#     - learning_rebuild_failed: fallo de la reconstrucción.
#     - failed_chunk_skipped_for_learning: chunk failed que NO contribuyó
#       al aprendizaje.
#     - lifetime_proxy_ratio_exceeded: alerta de degradación (ratio).
#
# Drift detection: `tests/test_p1_hist_audit_5_lesson_event_whitelist.py`
# parsea `cron_tasks.py` y verifica que cada event emitido está clasificado.
# Si se añade un event nuevo sin clasificar, falla el test.
# `tests/test_p1_audit_hist_7_lesson_whitelist_ssot.py` verifica que el
# import desde constants está intacto en todos los consumidores.
LESSON_COUNT_EVENT_WHITELIST = (
    "lesson_synthesized_low_confidence",
    "synth_propagated_to_prompt",
    "recent_lessons_partial_synthesis",
    "indefinite_pause_unblocked",
)


# [P2-NEW-2 · 2026-05-10] Whitelist completo de events VÁLIDOS escribibles
# a `chunk_lesson_telemetry`. Aplicado at-write en `_record_chunk_lesson_telemetry`
# (cron_tasks.py:~12224) como backstop a typos antes de tocar DB.
#
# Diferencia vs. `LESSON_COUNT_EVENT_WHITELIST` (línea 2192):
#   - `LESSON_COUNT_EVENT_WHITELIST` es el subset que cuenta como LECCIONES
#     semánticas en `/lessons-counts` (chip "X lecciones" del Historial).
#   - `CHUNK_LESSON_TELEMETRY_VALID_EVENTS` es la UNIÓN de lecciones +
#     métricas mecánicas (descartes, fallos de rebuild). Todos los events
#     que el codebase emite legítimamente.
#
# Bug original (audit 2026-05-10):
#   `_record_chunk_lesson_telemetry` aceptaba cualquier string en `event` y
#   confiaba en la CHECK constraint runtime de DB (P1-5: regex de formato).
#   Un typo (`leson_synthesized_low_confidence`) pasaría el regex de formato
#   pero NO cuenta como lección (whitelist miss en read-path); resultado:
#   row se persiste pero `/lessons-counts` lo ignora silenciosamente. El
#   usuario ve "0 lecciones" cuando realmente debería ver una.
#
# Fix:
#   Validar `event ∈ CHUNK_LESSON_TELEMETRY_VALID_EVENTS` ANTES de la INSERT.
#   Si no matchea: log error con contexto + return False (mismo contrato que
#   un INSERT fallido). El event inválido NO se persiste — el dev verá el log
#   y corregirá el typo antes de polucionar la tabla.
#
# Drift detection: `test_p2_new_2_chunk_lesson_telemetry_write_whitelist.py`
# parsea cron_tasks.py extrayendo todos los `event="..."` y verifica que
# cada literal está en esta tupla.
CHUNK_LESSON_TELEMETRY_VALID_EVENTS = (
    # Lecciones semánticas (subset que cuenta hacia /lessons-counts):
    "lesson_synthesized_low_confidence",
    "synth_propagated_to_prompt",
    "recent_lessons_partial_synthesis",
    "indefinite_pause_unblocked",
    # Métricas mecánicas (señales de salud, no lecciones):
    "synth_schema_invalid",
    "synth_schema_partial_invalid",
    "learning_rebuild_failed",
    "failed_chunk_skipped_for_learning",
    "lifetime_proxy_ratio_exceeded",
    # [P2-CHUNK-9] Override del gate temporal/aprendizaje por flexible_mode: el
    # chunk se generó SIN el aprendizaje continuo (gate not-ready bypaseado). Señal
    # de salud para cuantificar cuántos chunks degradan el loop de aprendizaje.
    "temporal_gate_override",
)


# [P2-NEW-3 · 2026-05-10] Canonical reasons aceptados por
# `_escalate_unrecoverable_chunk` (cron_tasks.py:~8622). Antes la función
# aceptaba cualquier string en `escalation_reason` y propagaba a:
#   - plan_chunk_queue.dead_letter_reason (persiste en DB).
#   - learning_metrics.escalation_reason (jsonb).
#   - meal_plans.plan_data._user_action_required.reason.
#   - push notification copy + deeplink URL query string.
#
# Un typo (`recover_exhausted` vs `recovery_exhausted`) o un valor
# arbitrario:
#   - Persiste como dead_letter_reason inválido → `/blocked_reasons`
#     reporta `_unknown` al frontend (línea ~3733 plans.py: "cualquier
#     otro caía al fallback empty_pantry/_unknown, mintiendo al usuario").
#   - El bloque if/elif del copy (cron_tasks.py:8678+) cae al else
#     genérico → mensaje y deeplink incorrectos.
#   - `_user_action_required.reason` desalineado con copy → frontend
#     muestra banner ambiguo.
#
# Fix:
#   Validar al entrar a `_escalate_unrecoverable_chunk` que
#   `escalation_reason ∈ ESCALATION_REASONS`. Si no: log error + early
#   return SIN persistir nada. El dev ve el log + corrige el callsite
#   antes de polucionar DB con valores no clasificables.
#
# Catálogo extraído de los call sites de `_escalate_unrecoverable_chunk`
# (`grep escalation_reason= cron_tasks.py`):
ESCALATION_REASONS = (
    # Default — chunk falló ≥CHUNK_MAX_RECOVERY_ATTEMPTS en pipeline LLM.
    "recovery_exhausted",
    # Plan sin fecha de inicio recuperable (anchor missing in 3 sources).
    "unrecoverable_missing_anchor",
    # Plan tiene anchors pero corruptos (safe_fromisoformat falla).
    "unrecoverable_corrupted_date",
    # Plan con tzOffset NULL + live TZ no resoluble tras N attempts.
    "unrecoverable_tz_unresolved",
    # Pausa indefinida `missing_prior_lessons`; unblock cron agotó vías.
    "missing_prior_lessons_unrecoverable",
    # [P1-NEW-D · 2026-05-11] Chunk `pending` con `execute_after` que cae
    # MÁS ALLÁ del horizonte temporal del plan (plan_start +
    # (total_days_requested + 7 días gracia) < execute_after). Indica que
    # el chunk fue anclado a una fecha que el plan ya no cubrirá: o el
    # plan se acortó (regeneración), o el cálculo de execute_after usó un
    # snapshot stale del `total_days_requested`. El chunk nunca disparará
    # útilmente — mejor escalarlo proactivamente que dejarlo "pending"
    # hasta su `execute_after` distante (zombie de larga duración).
    "execute_after_beyond_plan_window",
)


# [P2-HIST-AUDIT-D · 2026-05-09] Split por calidad de los events de
# `LESSON_COUNT_EVENT_WHITELIST`. Cada event semántico cae en una
# tier diferenciable para que el frontend pueda diferenciar el chip
# "X lecciones" plano por sub-conteos:
#
#   - HIGH (alta calidad): la lección llegó al prompt del próximo
#     chunk Y influenció la generación. Es el happy path completo.
#   - PARTIAL (mixta): síntesis parcial — algunos chunks tenían
#     learning_metrics y otros no. La entry SÍ contribuyó al
#     recent_lessons del LLM, con menos confianza que HIGH pero más
#     que LOW.
#   - LOW (proxy degradado): la lección se generó desde plan_days
#     porque el chunk no persistió learning_metrics (T2 fail u
#     otra corrupción). Aprendizaje real pero baja confianza.
#
# Drift detection: el test `test_p2_hist_audit_d_lesson_quality_tiers.py`
# valida que toda key de LESSON_COUNT_EVENT_WHITELIST está EXACTAMENTE
# en una tier (sin overlap, sin gaps). Si añades un event nuevo a
# la whitelist, DEBES decidir su tier — el test falla loud sino.
LESSON_QUALITY_TIERS = {
    "high": (
        "synth_propagated_to_prompt",
        "indefinite_pause_unblocked",
    ),
    "partial": (
        "recent_lessons_partial_synthesis",
    ),
    "low": (
        "lesson_synthesized_low_confidence",
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# [P2-HIST-AUDIT-13 · 2026-05-09] SSOT del set de `action_taken` que cuentan
# como AJUSTES anomalous del guard de coherencia recetas↔lista (chip "X
# ajustes" del Historial).
#
# Movido aquí desde la definición inline en `routers/plans.py:_ANOMALOUS_COHERENCE_ACTIONS`
# para alinearlo con el patrón establecido por P1-AUDIT-HIST-7 (lesson
# whitelist). Antes:
#   - `routers/plans.py` definía el set in-place.
#   - `frontend/src/pages/History.jsx` replicaba 4 string literals inline.
#   - Cuando `_aggregate_coherence_block_history_metrics` (P3-B) añadió
#     `post_swap_revalidation` y otros buckets, ambos sites necesitaron
#     actualización manual — drift latente en cualquier momento.
#
# Ahora consumidor backend importa `COHERENCE_ANOMALOUS_ACTIONS`; consumidor
# frontend importa de `frontend/src/utils/coherenceActions.js`. Drift
# detection cross-archivo (Python + JS) lo verifica con tests.
#
# Catálogo (decisión documentada del audit 2026-05-08 → P3-NEW-C):
#
#   ANOMALOUS (cuentan al chip — el sistema corrigió drift recetas↔lista):
#     - degrade: el guard mode=block degradó el plan (kill switch knob
#       MEALFIT_SHOPPING_COHERENCE_BLOCK_ACTION='degrade').
#     - reject_minor / reject_high: el guard rechazó el plan; review
#       node retorna severity minor/high según knob.
#     - hydration_error: bug del consumer (block_set=True pero
#       action_taken quedó None hasta que el fallback defensivo lo
#       hidrató). Cuenta como anomalous porque indica un fallo del
#       contrato P2-2 (action_taken jamás None tras review).
#
#   NO ANOMALOUS (NO cuentan al chip):
#     - not_applicable: warn-only, block_set=False (info pura).
#     - post_swap_revalidation (P2-B): observability tras swap; el cron
#       P3-B lo trata como bucket dedicado, NO anomalous.
#     - null: invariante violado (combinación reservada error — debería
#       ser hydration_error ya).
#
# Si el cron P3-B añade un nuevo bucket (e.g. `recoverable_drift`), el
# operador DEBE decidir si entra al SET aquí. El test
# `test_p2_hist_audit_13_coherence_anomalous_ssot.py` falla loud cuando
# alguien clasifica un action_taken sin actualizar la whitelist.
COHERENCE_ANOMALOUS_ACTIONS = (
    "degrade",
    "reject_minor",
    "reject_high",
    "hydration_error",
)
