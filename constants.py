import logging
import os
import math
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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
CHUNK_PIPELINE_TIMEOUT_SECONDS = int(os.environ.get("CHUNK_PIPELINE_TIMEOUT_SECONDS", "180"))
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
CHUNK_MIN_FRESH_PANTRY_ITEMS = max(1, int(os.environ.get("CHUNK_MIN_FRESH_PANTRY_ITEMS", "3")))
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
# 'initial_plan') para evitar duplicar la validación que ya hace
# `routers/plans.py` antes de aceptar la creación inicial del plan.
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
# tunable para no saturar Supabase si una caída de Gemini deja cientos de chunks failed
# de golpe.
CHUNK_RECOVERY_BATCH_LIMIT = max(1, int(os.environ.get("CHUNK_RECOVERY_BATCH_LIMIT", "20")))
# [P0-4] Máxima edad permitida para el snapshot de pantry antes de intentar un live-retry adicional.
# Pasado este TTL, si el live sigue fallando se marca como stale_snapshot en los logs.
CHUNK_PANTRY_SNAPSHOT_TTL_HOURS = max(1, int(os.environ.get("CHUNK_PANTRY_SNAPSHOT_TTL_HOURS", "6")))
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
# Tope de usuarios procesados por corrida para no saturar Supabase en picos.
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
#   síntesis low-confidence antes de disparar alerta. 0.20 = 20% es suficientemente alto
#   para no disparar por casos puntuales (chunks recientes donde la cola aún no commiteó
#   learning_metrics) y suficientemente bajo para detectar degradación sistémica (e.g.,
#   bug que dejó la columna NULL para todos los chunks de un release).
CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD = float(os.environ.get("CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD", "0.20"))
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
#   system-wide (0.20) porque per-user tiene varianza natural mayor: un usuario que
#   no logueó comidas dos días seguidos puede legítimamente cruzar 25% sin que sea
#   degradación sistémica.
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
_embedding_model = None
_embedding_cache = {}
_pantry_embeddings_cache = {}

def get_embedding(text: str) -> List[float]:
    global _embedding_model
    if not _embedding_model:
        _embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-2-preview", 
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
    if text not in _embedding_cache:
        emb = _embedding_model.embed_query(text)
        _embedding_cache[text] = emb
    return _embedding_cache[text]

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
# ---------------------------

DOMINICAN_PROTEINS = [
    "Pollo", "Cerdo", "Res", "Pavo", "Pescado", "Atún", "Huevos", "Queso de Freír",
    "Salami Dominicano", "Camarones", "Chuleta", "Longaniza",
    "Habichuelas Rojas", "Habichuelas Negras", "Habichuelas Blancas",
    "Gandules", "Lentejas", "Garbanzos", "Soya/Tofu",
    "Queso Ricotta", "Queso Blanco", "Queso Mozzarella", "Yogurt"
]

DOMINICAN_CARBS = [
    "Plátano Verde", "Plátano Maduro", "Yuca", "Batata", "Arroz Blanco", 
    "Arroz Integral", "Avena", "Pan Integral", "Papas", "Guineítos Verdes", "Ñame", "Yautía"
]

PROTEIN_SYNONYMS = {
    "pollo": ["pollo", "pechuga", "muslo", "alitas", "chicharrón de pollo", "filete de pollo"],
    "cerdo": ["cerdo", "masita", "chicharrón de cerdo", "lomo", "pernil", "costilla"],
    "res": ["res", "carne molida", "bistec", "filete", "churrasco", "vaca", "picadillo", "carne de res"],
    "pescado": ["pescado", "dorado", "chillo", "mero", "salmón", "tilapia", "filete de pescado",
                "bacalao", "bacalao desalado", "bacalao salado", "filete de bacalao",
                "filete de mero", "filete de tilapia", "filete de chillo", "filete de dorado"],
    "atún": ["atún", "atun", "atun en agua", "atun en lata"],
    "sardina": ["sardina", "sardinas", "sardina en lata"],
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
               "yogurt descremado", "greek yogurt"],
    "queso blanco": ["queso blanco", "queso blanco fresco", "queso de pasta", "queso fresco",
                     "queso de mano", "queso crema", "queso campesino"],
    "queso mozzarella": ["queso mozzarella", "mozzarella", "mozzarella fresca", "queso mozarela",
                         "queso mozarella", "queso mozzarella fresco"],
    "gandules": ["gandules", "guandules", "gandul", "guandul"],
    "lentejas": ["lentejas", "lenteja"],
    "garbanzos": ["garbanzos", "garbanzo", "hummus", "puré de garbanzos"],
    "soya/tofu": ["soya/tofu", "soya", "tofu", "carne de soya", "tofu/soya", "tofu/soya firme", "tofu firme", "soya firme"],
    "pavo": ["pavo", "pechuga de pavo", "pavo asado", "pavo desmenuzado", "jamón de pavo", "pavo molido", "carne de pavo"]
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
    "batata": ["batata", "puré de batata", "batata hervida", "boniato"]
}

DOMINICAN_VEGGIES_FATS = [
    "Aguacate", "Berenjena", "Tayota", "Repollo", "Zanahoria",
    "Molondrones", "Brócoli", "Coliflor", "Tomate", "Vainitas",
    "Aceitunas", "Cebolla", "Ajíes", "Aceite de Oliva", "Nueces/Almendras",
    "Auyama"
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
    "nueces/almendras": ["nueces/almendras", "nueces", "almendras", "maní"]
}

DOMINICAN_FRUITS = [
    "Guineo", "Mango", "Piña", "Lechosa", "Chinola",
    "Limón", "Fresa", "Naranja", "Sandía", "Melón"
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
    "melón": ["melón", "melon"]
}

NUTRITIONAL_CATEGORIES = {
    "aves": ["pollo", "pavo"],
    "carnes rojas y embutidos": ["cerdo", "res", "chuleta", "longaniza", "salami dominicano"],
    "pescados y mariscos": ["pescado", "atún", "sardina", "camarones"],
    "huevos y lácteos": ["huevos", "queso de freír", "queso", "yogurt", "leche", "queso crema", "ricotta", "cottage", "yogurt griego"],
    "legumbres": ["habichuelas rojas", "habichuelas negras", "gandules", "lentejas", "garbanzos", "soya/tofu"],
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
    "especias", "hierbas", "azucar", "miel", "polvo", "cucharada", "cucharadita", "taza"
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

SUPPLEMENT_NAMES = {
    "whey_protein": "Proteína Whey",
    "creatine": "Creatina Monohidrato",
    "bcaa": "Aminoácidos BCAA",
    "glutamine": "Glutamina",
    "omega3": "Omega-3 (Aceite de Pescado)",
    "multivitamin": "Multivitamínico Completo",
    "vitamin_d": "Vitamina D3",
    "magnesium": "Magnesio (Citrato o Glicinato)",
    "pre_workout": "Pre-Entreno (Cafeína + Beta-Alanina)",
    "collagen": "Colágeno Hidrolizado",
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
    if any(r in ["mariscos", "seafood", "marisco"] for r in normalized_restrictions):
        normalized_restrictions.extend(["camaron", "camarones", "pescado", "atun"])
    if any(r in ["carne", "carnes", "meat"] for r in normalized_restrictions):
        normalized_restrictions.extend(["pollo", "cerdo", "res", "chuleta", "longaniza", "salami"])
        
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

def _get_converted_quantity(req_qty: float, req_unit: str, dispo_unit: str, base_name: str) -> float | None:
    """Convierte matemáticamente entre familias de unidades incompatibles (Masa/Vol/Unidad)."""
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

    if req_unit == 'g' and dispo_unit == 'ml' and density: return req_qty / density
    if req_unit == 'ml' and dispo_unit == 'g' and density: return req_qty * density
    if req_unit == 'g' and dispo_unit == 'unidad' and unit_weight: return req_qty / unit_weight
    if req_unit == 'unidad' and dispo_unit == 'g' and unit_weight: return req_qty * unit_weight
    if req_unit == 'ml' and dispo_unit == 'unidad' and density and unit_weight: return (req_qty * density) / unit_weight
    if req_unit == 'unidad' and dispo_unit == 'ml' and density and unit_weight: return (req_qty * unit_weight) / density
    return None

def validate_ingredients_against_pantry(generated_ingredients: list, pantry_ingredients: list, strict_quantities: bool = True, tolerance: float = 1.30) -> bool | str:
    """
    Función guardrail estricta y matemática. Comprueba:
    1. Que todos los ingredientes generados estén en la despensa.
    2. (Si strict_quantities=True) Que las CANTIDADES generadas no superen el Ledger de la despensa.
    
    En modo rotación (strict_quantities=False), solo se valida #1 (existencia), 
    ya que el LLM redistribuye las mismas macros con proporciones distintas.
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
    
    # Regex para extraer peso de contenedor embebido en el display string
    # Ejemplos: "1 Paquete (500g)" → 500g, "1 Pote (16 oz)" → 16 oz, "1 Cartón (250ml)" → 250ml
    _container_weight_re = re.compile(
        r'\((\d+(?:\.\d+)?)\s*(g|gr|kg|oz|ml|l|lb|lbs)\)',
        re.IGNORECASE
    )
    
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
            container_qty = float(container_match.group(1))
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
            if not matched_pantry_key and len(base) > 2:
                try:
                    gen_emb = get_embedding(base)
                    best_match = None
                    best_score = -1.0
                    
                    for p_key in pantry_ledger.keys():
                        if p_key not in _pantry_embeddings_cache:
                            _pantry_embeddings_cache[p_key] = get_embedding(p_key)
                        
                        p_emb = _pantry_embeddings_cache[p_key]
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
                    req_qty_in_dispo_unit = _get_converted_quantity(gen_base_qty, gen_base_unit, dispo_unit, matched_pantry_key)
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
                            req_qty_in_dispo_unit = _get_converted_quantity(fallback_g, 'g', dispo_unit, matched_pantry_key)
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
