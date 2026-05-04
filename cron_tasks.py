import logging
import os
import traceback
from datetime import datetime, timezone, timedelta
import json
import copy
import random


# [test fix] Wrapper patcheable usado por `_check_chunk_learning_ready` para
# obtener "ahora". `datetime.datetime` es un tipo built-in inmutable: no se puede
# hacer `setattr(datetime, 'now', mock)`, así que tests no pueden usar
# `@patch("cron_tasks.datetime.now")`. Esta función indirecta sí es patcheable
# vía `@patch("cron_tasks._dt_p0b_now")` y permite a tests del gate temporal
# controlar el reloj sin tocar el built-in. La firma replica `datetime.now(tz)`
# para que callers existentes no cambien.
def _dt_p0b_now(tz=None):
    return datetime.now(tz)
_tz_p0b = timezone
from db_core import execute_sql_query, execute_sql_write, connection_pool
from db_inventory import (
    deduct_consumed_meal_from_inventory,
    get_inventory_activity_since,
    get_raw_user_inventory,
    get_user_inventory_net,
    release_chunk_reservations,
    reserve_plan_ingredients,
)
from db import get_latest_meal_plan_with_id, get_user_likes, get_active_rejections, get_recent_plans
from db_facts import get_all_user_facts, get_consumed_meals_since
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from collections import Counter

from schemas import HealthProfileSchema

from constants import (
    CHUNK_MIN_FRESH_PANTRY_ITEMS,
    CHUNK_MAX_FAILURE_ATTEMPTS,
    CHUNK_LEARNING_MODE,
    CHUNK_LEARNING_READY_DELAY_HOURS,
    CHUNK_LEARNING_READY_MAX_DEFERRALS,
    CHUNK_LEARNING_READY_MIN_RATIO,
    CHUNK_TEMPORAL_GATE_MAX_RETRIES,
    CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY,
    CHUNK_TEMPORAL_GATE_PUSH_COOLDOWN_HOURS,
    CHUNK_PREV_CHUNK_PAUSE_TTL_HOURS,
    CHUNK_PANTRY_EMPTY_MAX_REMINDERS,
    CHUNK_PANTRY_EMPTY_REMINDER_HOURS,
    CHUNK_PANTRY_EMPTY_TTL_HOURS,
    CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS,
    CHUNK_REJECT_FORCED_UTC_ENQUEUE,
    CHUNK_TZ_RECOVERY_MAX_ATTEMPTS,
    CHUNK_ZERO_LOG_PROACTIVE_DETECTION,
    CHUNK_PANTRY_PROACTIVE_GUARD,
    CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES,
    CHUNK_FINAL_VALIDATION_PAUSE_TTL_HOURS,
    CHUNK_FINAL_VALIDATION_MAX_RECOVERY_ATTEMPTS,
    CHUNK_FINAL_VALIDATION_RECOVERY_GRACE_MINUTES,
    CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS,
    CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS,
    CHUNK_PANTRY_PROACTIVE_REFRESH_MAX_USERS,
    CHUNK_PANTRY_PROACTIVE_REFRESH_MINUTES,
    CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS,
    CHUNK_ZERO_LOG_RECOVERY_TTL_HOURS,
    CHUNK_PANTRY_HARD_FAIL_AGE_HOURS,
    CHUNK_PANTRY_HARD_FAIL_ON_STALE,
    CHUNK_PANTRY_QUANTITY_MODE,
    CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE,
    CHUNK_PANTRY_MAX_RETRIES,
    CHUNK_PANTRY_TOLERANCE_MIN,
    CHUNK_PANTRY_TOLERANCE_MAX,
    CHUNK_PANTRY_SNAPSHOT_TTL_HOURS,
    CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS,
    CHUNK_RETRY_BASE_MINUTES,
    CHUNK_RETRY_CRITICAL_MINUTES,
    CHUNK_RECOVERY_MIN_AGE_MINUTES,
    CHUNK_MAX_RECOVERY_ATTEMPTS,
    CHUNK_RECOVERY_BATCH_LIMIT,
    CHUNK_ZOMBIE_PARTIAL_MIN_AGE_HOURS,
    CHUNK_ZOMBIE_PARTIAL_FINALIZE_INTERVAL_MINUTES,
    CHUNK_ZOMBIE_PARTIAL_BATCH_LIMIT,
    CHUNK_ORPHAN_RESERVATION_CLEANUP_INTERVAL_MINUTES,
    CHUNK_ORPHAN_RESERVATION_BATCH_LIMIT,
    CHUNK_ORPHAN_RESERVATION_MIN_TERMINAL_AGE_HOURS,
    CHUNK_HEARTBEAT_START_FAIL_RETRY_MINUTES,
    CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS,
    CHUNK_ZERO_LOG_NUDGE_COOLDOWN_HOURS,
    CHUNK_ZERO_LOG_NUDGE_INTERVAL_MINUTES,
    CHUNK_ZERO_LOG_NUDGE_MAX_USERS,
    CHUNK_PROACTIVE_MARGIN_DAYS,
    CHUNK_SCHEDULER_INTERVAL_MINUTES,
    strip_accents,
    get_embedding,
    cosine_similarity,
)
from graph_orchestrator import run_plan_pipeline
from memory_manager import build_memory_context
from services import _save_plan_and_track_background
from agent import analyze_preferences_agent
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


# [P0-1] Campos canónicos de aprendizaje del chunk. Persistidos atómicamente en
# T1 (junto con `days`, `_merged_chunk_ids` y `plan_chunk_queue.learning_metrics`)
# para garantizar que un crash entre T1 y T2 no pierda la lección.
#
# Pre-fix: estos campos se diferían a T2 — si T2 fallaba post shopping-list,
# `plan_data.days` quedaba mergeado y `_merged_chunk_ids` estampado pero la
# lección NO se persistía y `plan_chunk_queue.learning_metrics` quedaba en
# NULL. En el retry, el path `chunk_already_merged` saltaba el merge y
# backfilleaba un STUB (porque queue.learning_metrics era NULL), perdiendo
# permanentemente las métricas reales del chunk.
#
# Post-fix: T1 commitea atómicamente days + _merged_chunk_ids + learning fields
# + plan_chunk_queue.learning_metrics dentro del mismo FOR UPDATE. Si T2 falla,
# el retry detecta `_persisted_chunk_id == week_number` y salta el backfill —
# la lección ya está persistida. T2 sólo agrega shopping_list + quality +
# `status='completed'` + `learning_persisted_at`. Ver `_chunk_worker`
# (search "[P0-1]") para el invariante completo.
#
# El nombre de la constante (DEFERRED) se preserva por compatibilidad con
# call sites externos y porque sigue siendo la lista canónica de campos de
# learning que `P0_4_T2_INCREMENTAL_KEYS` y la defense-in-depth check usan.
P0_1_DEFERRED_LEARNING_KEYS = (
    '_last_chunk_learning',
    '_recent_chunk_lessons',
    '_critical_lessons_permanent',
    '_lifetime_lessons_history',
    '_lifetime_lessons_summary',
    '_chunk_learning_stub_count',
)


# [P0-5] Allowlist explícito de keys que MATCHEAN el patrón lesson/learning pero
# NO son campos de aprendizaje persistente del worker (y por tanto no deben
# añadirse a P0_1_DEFERRED_LEARNING_KEYS). Estos son flags de transporte/modo
# (snapshot, form_data, pause_snapshot), telemetría, o keys escritas a plan_data
# vía un path atómico independiente del worker (`update_plan_data_atomic` u otra
# función dedicada).
#
# El test `test_p0_5_deferred_keys_naming_convention` escanea cron_tasks.py
# buscando literales que matcheen el patrón y exige que cada uno esté en
# P0_1_DEFERRED_LEARNING_KEYS o aquí. Sin esta disciplina, un dev futuro puede
# añadir `_meta_lessons_v2` (o similar) sin actualizar el set, persistiéndolo en
# T1 y rompiendo la atomicidad con plan_chunk_queue.learning_metrics.
#
# Reglas:
#   - Un campo de aprendizaje REAL del worker T1/T2 → P0_1_DEFERRED_LEARNING_KEYS.
#   - Un flag transitorio (no se escribe directo a meal_plans.plan_data) → aquí.
#   - Un campo escrito a plan_data via update_plan_data_atomic con su propia
#     atomicidad (independiente del worker) → aquí, con comentario justificándolo.
_P0_5_LESSON_KEY_ALLOWLIST = frozenset({
    # Telemetría / provenance:
    '_learning_provenance',
    '_learning_signal_strength',
    '_last_learning_ready_ratio',
    '_last_learning_zero_log',
    '_active_learning_signals',
    # Flags de modo / scheduling en snapshot/form_data (NO en plan_data del worker):
    '_learning_flexible_mode',
    '_learning_forced',
    '_learning_forced_reason',
    '_learning_ready_deferrals',
    '_failed_chunk_learning_disabled',  # form_data flag
    '_learning_window_starved',         # form_data flag
    # Inyección transitoria al prompt LLM (form_data/snapshot, no plan_data atómico):
    '_chunk_lessons',
    # Lecciones heredadas de plan previo: viaja en `snap` (snapshot transport) y se
    # copia a `plan_data._lifetime_lessons_*` (que SÍ están en deferred). El campo
    # `_inherited_lifetime_lessons` mismo no se persiste, sólo se consume.
    '_inherited_lifetime_lessons',
    # Telemetría P1-1 dentro de `_p11_pause` (pause_snapshot, no plan_data):
    '_p1_1_actual_lessons',
    '_p1_1_expected_lessons',
    '_p1_1_rebuilt_lessons',
    # Substring sentinel usado por el runtime defense-in-depth check
    # (`if "_learning" in k.lower()`); no es un nombre de field real.
    '_learning',
    # Persistido a plan_data vía `update_plan_data_atomic` independiente del worker
    # T1/T2 (telemetría P1-6 de chunks dead-lettered):
    '_learning_corrupted_chunks',
})


# [P0.3] Contextos canónicos para escrituras legacy de _last_chunk_learning.
# Todos los call-sites legacy DEBEN pasar uno de estos a
# `persist_legacy_learning_to_plan_data` para que la telemetría pueda agregarse
# por origen y para detectar nuevos paths que aparezcan sin clasificar.
P0_3_LEGACY_LEARNING_CONTEXTS = (
    "seed_chunk1_sync",        # routers/plans.py: seed inicial post-LLM (sync).
    "seed_chunk1_sse",         # routers/plans.py: seed inicial via SSE stream.
    "rebuild_from_queue",      # cron_tasks.py: P0-3 auto-recovery desde plan_chunk_queue.
    "synthesis_from_days",     # cron_tasks.py: P0-4 last-resort desde plan_data.days.
)


def persist_legacy_learning_to_plan_data(
    meal_plan_id: str,
    last_chunk_learning: Dict[str, Any],
    recent_chunk_lessons: List[Any] = None,
    *,
    context: str,
) -> bool:
    """[P0.3] Punto único para escribir `_last_chunk_learning` (+ opcionalmente
    `_recent_chunk_lessons`) en `meal_plans.plan_data` desde paths legacy
    (fuera del worker T2 atómico).

    Garantiza:
      - Validación de que los campos sí están en `P0_1_DEFERRED_LEARNING_KEYS`.
      - Sello CAS atómico de `_plan_modified_at` para que el FOR UPDATE del
        worker T2 detecte la mutación y re-lea plan_data fresco.
      - Telemetría centralizada por `context` ∈ `P0_3_LEGACY_LEARNING_CONTEXTS`.

    Antes de este helper, cada path duplicaba el patrón SQL `jsonb_set` y dos
    de cuatro paths NO sellaban `_plan_modified_at` — violando la invariante
    CAS y abriendo la ventana para que un worker concurrente pisara la lección.

    Args:
        meal_plan_id: id del plan al que pertenece la lección.
        last_chunk_learning: dict serializable con la lección.
        recent_chunk_lessons: lista opcional de lecciones rolling. Cuando se
            provee, se persiste en el mismo UPDATE atómico.
        context: motivo de la escritura, debe ser uno de
            `P0_3_LEGACY_LEARNING_CONTEXTS` para telemetría agregada.

    Returns:
        True si el UPDATE persistió, False en error.
    """
    if context not in P0_3_LEGACY_LEARNING_CONTEXTS:
        logger.error(
            f"[P0.3/LEGACY-PERSIST] context={context!r} no está en "
            f"P0_3_LEGACY_LEARNING_CONTEXTS={P0_3_LEGACY_LEARNING_CONTEXTS}. "
            f"Si añadiste un nuevo path legacy, regístralo allí o usa el "
            f"worker T2 atómico (`_chunk_worker`)."
        )
        return False

    if not meal_plan_id or not isinstance(last_chunk_learning, dict):
        logger.warning(
            f"[P0.3/LEGACY-PERSIST] Args inválidos: meal_plan_id={meal_plan_id!r}, "
            f"last_chunk_learning={type(last_chunk_learning).__name__}, context={context}"
        )
        return False

    payload = json.dumps(last_chunk_learning, ensure_ascii=False)

    if recent_chunk_lessons is not None:
        sql = """
            UPDATE meal_plans
            SET plan_data = jsonb_set(
                    jsonb_set(
                        jsonb_set(
                            COALESCE(plan_data, '{}'::jsonb),
                            '{_last_chunk_learning}',
                            %s::jsonb,
                            true
                        ),
                        '{_recent_chunk_lessons}',
                        %s::jsonb,
                        true
                    ),
                    '{_plan_modified_at}',
                    to_jsonb(NOW()::text),
                    true
                )
            WHERE id = %s
        """
        recent_payload = json.dumps(recent_chunk_lessons, ensure_ascii=False)
        params = (payload, recent_payload, meal_plan_id)
    else:
        sql = """
            UPDATE meal_plans
            SET plan_data = jsonb_set(
                    jsonb_set(
                        COALESCE(plan_data, '{}'::jsonb),
                        '{_last_chunk_learning}',
                        %s::jsonb,
                        true
                    ),
                    '{_plan_modified_at}',
                    to_jsonb(NOW()::text),
                    true
                )
            WHERE id = %s
        """
        params = (payload, meal_plan_id)

    try:
        execute_sql_write(sql, params)
        logger.debug(
            f"[P0.3/LEGACY-PERSIST] meal_plan={meal_plan_id} "
            f"context={context} chunk={last_chunk_learning.get('chunk')} "
            f"recent_count={len(recent_chunk_lessons) if recent_chunk_lessons is not None else None}"
        )
        return True
    except Exception as e:
        logger.warning(
            f"[P0.3/LEGACY-PERSIST] Error persistiendo learning para "
            f"meal_plan={meal_plan_id} context={context}: {e}"
        )
        return False


# [P0-4] Campos que el worker calcula ENTRE T1 (merge de days) y T2 (commit
# atómico de status='completed') y que sólo T2 persiste. Cuando T2 corre, debe
# RE-LEER plan_data con FOR UPDATE y aplicar SOLO estos campos, en lugar de
# sobrescribir el dict completo desde memoria — porque `/shift-plan` (u otro
# caller) puede haber mutado days/generation_status/grocery_start_date entre
# T1 y T2, y un overwrite ciego perdería esas modificaciones.
P0_4_T2_INCREMENTAL_KEYS = (
    # Learning (P0-1: deferidos a T2 para atomicidad con learning_metrics).
    '_last_chunk_learning',
    '_recent_chunk_lessons',
    '_critical_lessons_permanent',
    '_lifetime_lessons_history',
    '_lifetime_lessons_summary',
    '_chunk_learning_stub_count',
    # Shopping list (calculada post-merge en líneas 14194-14218).
    'aggregated_shopping_list',
    'aggregated_shopping_list_weekly',
    'aggregated_shopping_list_biweekly',
    'aggregated_shopping_list_monthly',
    # Quality flags (calculadas post-merge en líneas 14336-14367).
    'quality_warning',
    'quality_degraded_ratio',
    # [P0-5] Pantry quantity violation annotation (advisory/hybrid mode).
    # Set in T1 by the worker (cron_tasks.py:16124) and must survive T2's
    # fresh-read overlay so the UI/admin sees the chunk-level annotation.
    '_pantry_quantity_violations',
)


# [P0-2] Estados terminales del chunk en plan_chunk_queue. Si encontramos uno de
# estos pre-LLM, el chunk fue cancelado/completado por otro path (típicamente
# `save_new_meal_plan_atomic` cuando el usuario regenera el plan); abortamos sin
# llamar al LLM para no quemar tokens en un chunk ya muerto.
_P0_2_CHUNK_TERMINAL_STATES = ("cancelled", "completed", "failed")


def compute_lifetime_lesson_weight(lesson, now=None) -> float:
    """[P1-4] Devuelve el peso de una lección por edad: `decay ** weeks_old`.

    Args:
        lesson: dict con campo `timestamp` ISO 8601 (string o datetime).
            Lecciones sin timestamp se asumen recientes (peso 1.0) — caso del
            stub puro persistido en el path P0-3 cuando learning_metrics no
            estaba disponible.
        now: datetime UTC opcional para tests (default: datetime.now UTC).

    Returns:
        Peso en [0.0, 1.0]. Lecciones con peso < LIFETIME_LESSON_MIN_WEIGHT
        deben excluirse del summary por el caller.

    Convenciones:
      - timestamps en el FUTURO (clock skew, snapshot inválido) → tratamos
        como now (peso 1.0): preferimos sobreponderar a infraponderar señales
        ambiguas, ya que el impacto es solo en ranking dentro de caps.
      - timestamp no parseable → peso 1.0 (fail-open). El cutoff por días de
        LIFETIME_LESSONS_WINDOW_DAYS ya filtra entries muy viejas; esta función
        agrega ranking, no es la primera línea de defensa.
    """
    from constants import LIFETIME_LESSON_WEEKLY_DECAY
    if not isinstance(lesson, dict):
        return 0.0
    ts_raw = lesson.get("timestamp")
    if not ts_raw:
        return 1.0
    if now is None:
        now = datetime.now(timezone.utc)
    try:
        from constants import safe_fromisoformat
        ts_dt = safe_fromisoformat(ts_raw) if isinstance(ts_raw, str) else ts_raw
        if ts_dt.tzinfo is None:
            ts_dt = ts_dt.replace(tzinfo=timezone.utc)
        age_seconds = (now - ts_dt).total_seconds()
        if age_seconds <= 0:
            return 1.0
        weeks_old = age_seconds / (7.0 * 86400.0)
        return max(0.0, float(LIFETIME_LESSON_WEEKLY_DECAY) ** weeks_old)
    except Exception:
        return 1.0


# [P1-7] Provenance canónico de cada lección en `_lifetime_lessons_history`.
# Determina cuánto debe pesar la lección al recomputar `_lifetime_lessons_summary`:
# las que vienen de logs reales del usuario (high signal) NO se penalizan; las que
# vienen de proxy de inventario o síntesis desde plan_data.days (low signal) se
# multiplican por LIFETIME_LESSON_PROXY_WEIGHT_FACTOR para que los caps de top-N
# en el summary prioricen información concreta sobre la inferida.
P1_7_LEARNING_PROVENANCES = (
    "user_logs",        # learning_metrics calculado desde consumed_meals reales del usuario.
    "inventory_proxy",  # ratio basado en mutaciones de inventario (proxy débil).
    "synthesis",        # _synthesize_last_chunk_learning_from_plan_days (no logs ni proxy).
    "stub",             # metrics_unavailable: lección placeholder, sin datos.
)


def _derive_learning_provenance(lesson) -> str:
    """[P1-7] Determina la provenance de una lección según sus flags.

    Lessons creadas pre-P1-7 NO traen `_learning_provenance` explícito; lo derivamos
    de los flags existentes (`metrics_unavailable`, `low_confidence`,
    `learning_signal_strength`, marcadores `rebuilt_from_*`). Lessons creadas
    post-P1-7 SÍ lo traen y se devuelve directo.

    Returns:
        Una de P1_7_LEARNING_PROVENANCES. Default 'user_logs' si no hay señal
        contraria — preferimos sobrestimar la calidad sobre infraponderarla
        (un dato real marcado como proxy se pierde en el summary; un proxy
        marcado como real se diluye con decay/dedup).
    """
    if not isinstance(lesson, dict):
        return "stub"
    explicit = lesson.get("_learning_provenance")
    if explicit in P1_7_LEARNING_PROVENANCES:
        return str(explicit)
    if lesson.get("metrics_unavailable"):
        return "stub"
    # rebuilt_from_pipeline_failure y synthesized_from_days marcan síntesis.
    if (
        lesson.get("rebuilt_from_pipeline_failure")
        or lesson.get("synthesized_from_days")
        or lesson.get("rebuilt_from_preflight")
    ):
        return "synthesis"
    if lesson.get("low_confidence") or lesson.get("learning_signal_strength") == "weak":
        return "inventory_proxy"
    return "user_logs"


# [P1-7] Factor multiplicativo aplicado al peso de una lección con provenance no
# `user_logs`. 0.5 es agresivo pero proporcionado: una lección de 1 semana con
# provenance=inventory_proxy queda con peso 0.45 (0.9 * 0.5), competitivo contra
# una user_logs de 4 semanas (0.66). Sin esto, el ranking de
# `top_repeated_meal_names`/`top_rejection_hits` se diluye con eventos sintéticos
# que ocupan slots del cap (top 20/30) que deberían ser de datos reales.
LIFETIME_LESSON_PROXY_WEIGHT_FACTOR = 0.5


def _provenance_weight_factor(lesson) -> float:
    """[P1-7] Multiplicador de peso por provenance para summary recompute.

    user_logs → 1.0, todo lo demás → LIFETIME_LESSON_PROXY_WEIGHT_FACTOR.
    Combinado con el decay temporal de P1-4 vía multiplicación:
        final_weight = decay_weight * provenance_factor
    """
    return 1.0 if _derive_learning_provenance(lesson) == "user_logs" else LIFETIME_LESSON_PROXY_WEIGHT_FACTOR


def _validate_chunk_pre_llm(task_id, meal_plan_id, user_id):
    """[P0-2] Re-valida pre-LLM que el chunk siga vivo y el plan exista.

    Cierra la ventana TOCTOU entre el pickup (`SELECT FOR UPDATE SKIP LOCKED`
    a ~9776) y el submit del LLM (~12576). Durante esta ventana (~2500 líneas
    de prep work + reads de inventario + reads de profile) un caller externo
    como `save_new_meal_plan_atomic` puede haber cancelado el chunk o el
    usuario puede haber borrado el plan. Si seguimos, gastamos tokens en una
    generación que el merge transaccional descartará.

    Returns:
        "ok"               → proceder con el LLM call.
        "plan_missing"     → meal_plan ya no existe; el caller debe cancelar
                             el chunk y liberar reservas.
        "chunk_terminal"   → chunk en estado cancelled/completed/failed; el
                             caller debe abortar silentemente (housekeeping
                             ya lo hizo todo).
        "chunk_unknown"    → row del chunk desapareció (poco probable; row es
                             la PK). Tratamos como terminal.
        "validation_error" → error consultando DB; best-effort fallback es
                             continuar (no bloquear chunks por flaps de DB).
    """
    try:
        row = execute_sql_query(
            """
            SELECT pcq.status AS chunk_status, mp.id AS plan_exists
            FROM plan_chunk_queue pcq
            LEFT JOIN meal_plans mp ON mp.id = pcq.meal_plan_id
            WHERE pcq.id = %s
            """,
            (task_id,),
            fetch_one=True,
        )
    except Exception as e:
        logger.warning(
            f"[P0-2/PRE-LLM] Error validando chunk {task_id} (plan {meal_plan_id}): "
            f"{type(e).__name__}: {e}. Continuando best-effort."
        )
        return "validation_error"

    if not row:
        logger.warning(
            f"[P0-2/PRE-LLM] Chunk {task_id} desapareció de plan_chunk_queue "
            f"entre pickup y LLM call. Abortando."
        )
        return "chunk_unknown"

    chunk_status = row.get("chunk_status")
    plan_exists = row.get("plan_exists")

    if chunk_status in _P0_2_CHUNK_TERMINAL_STATES:
        logger.warning(
            f"[P0-2/PRE-LLM] Chunk {task_id} (plan {meal_plan_id} user {user_id}) "
            f"ya está en estado terminal {chunk_status!r} pre-LLM. Probable "
            f"cancelación por save_new_meal_plan_atomic. Abortando para no "
            f"quemar tokens."
        )
        return "chunk_terminal"

    if not plan_exists:
        logger.warning(
            f"[P0-2/PRE-LLM] meal_plan {meal_plan_id} fue borrado entre pickup "
            f"y LLM call. Cancelando chunk {task_id} y liberando reservas."
        )
        return "plan_missing"

    return "ok"


# [P0.2] El helper `_backfill_plan_anchors_oneshot` fue eliminado. El backfill
# de `_plan_start_date` / `grocery_start_date` ahora vive exclusivamente en
# `supabase/migrations/p0_3_backfill_plan_anchors.sql`. La lógica runtime
# dependía de la env var `BACKFILL_PLAN_ANCHORS_DONE` para evitar doble-corrida,
# frágil de configurar en deploy: si la var no se seteaba, los anchors se
# re-escribían en cada arranque (idempotente pero con WAL traffic creciente);
# si se seteaba antes de aplicar la migración, los planes legacy quedaban sin
# anchor y el cron caía al fallback hardcoded "8am UTC" desalineando hasta 24h
# en TZ no-UTC. Migración SQL aplicada manualmente vía Supabase es la fuente
# única de verdad.


def register_plan_chunk_scheduler(scheduler) -> None:
    """Registra el polling del worker de chunks una sola vez en el scheduler global."""
    if not scheduler:
        return

    if not scheduler.get_job("process_plan_chunk_queue"):
        scheduler.add_job(
            process_plan_chunk_queue,
            "interval",
            minutes=CHUNK_SCHEDULER_INTERVAL_MINUTES,
            id="process_plan_chunk_queue",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            f"⏰ [CHUNK SCHEDULER] Worker plan_chunk_queue registrado cada "
            f"{CHUNK_SCHEDULER_INTERVAL_MINUTES} min."
        )

    # [P0-C] Job separado: refresh proactivo de snapshots de despensa en chunks pending/stale
    # cuyo _pantry_captured_at supera la mitad del TTL. Evita que el worker llegue al
    # execute_after con un snapshot vencido si su live fetch puntual falla.
    if not scheduler.get_job("proactive_refresh_pantry_snapshots"):
        scheduler.add_job(
            _proactive_refresh_pending_pantry_snapshots,
            "interval",
            minutes=CHUNK_PANTRY_PROACTIVE_REFRESH_MINUTES,
            id="proactive_refresh_pantry_snapshots",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            f"⏰ [P0-C] Refresh proactivo de despensa registrado cada "
            f"{CHUNK_PANTRY_PROACTIVE_REFRESH_MINUTES} min "
            f"(horizonte={CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS}h, "
            f"max_users={CHUNK_PANTRY_PROACTIVE_REFRESH_MAX_USERS})."
        )

    # [P2-3] Job dedicado: limpieza de chunks huérfanos (meal_plan_id ya no existe).
    # Antes esta lógica corría dentro de `process_plan_chunk_queue` y bloqueaba el
    # hot path del worker si su query era lenta. Aislado aquí en su propio cron
    # con frecuencia configurable independiente.
    if not scheduler.get_job("cleanup_orphan_chunks"):
        scheduler.add_job(
            _cleanup_orphan_chunks,
            "interval",
            minutes=CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES,
            id="cleanup_orphan_chunks",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            f"⏰ [P2-3] Cron de cleanup de chunks huérfanos registrado cada "
            f"{CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES} min."
        )

    # [P0-1-RECOVERY] Job separado: recuperación de chunks failed en CUALQUIER plan (7d+).
    # El job_id se mantiene como `recover_failed_chunks_long_plans` para compatibilidad
    # con APScheduler (cambiar el id duplicaría el job en deploys mixtos); el alcance real
    # es el de la query SQL dentro de `_recover_failed_chunks_for_long_plans`, que ya
    # filtra `total_days_requested >= 7`.
    if not scheduler.get_job("recover_failed_chunks_long_plans"):
        scheduler.add_job(
            _recover_failed_chunks_for_long_plans,
            "interval",
            minutes=15,
            id="recover_failed_chunks_long_plans",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            "⏰ [P0-1-RECOVERY] Cron de recuperación de chunks failed registrado cada 15 min "
            f"(min_age={CHUNK_RECOVERY_MIN_AGE_MINUTES}m, max_recovery_attempts="
            f"{CHUNK_MAX_RECOVERY_ATTEMPTS}, batch_limit={CHUNK_RECOVERY_BATCH_LIMIT}, scope=plans>=7d)."
        )

    # [P1-2/ZERO-LOG-NUDGE] Cron: aviso PROACTIVO a usuarios con plan activo + 0 logs.
    # Antes el push solo se mandaba reactivamente cuando el chunk N+1 ya estaba defiriéndose;
    # ahora se manda antes de que tropiece para que el usuario tenga tiempo de registrar.
    if not scheduler.get_job("nudge_chronic_zero_log_users"):
        scheduler.add_job(
            _nudge_chronic_zero_log_users,
            "interval",
            minutes=CHUNK_ZERO_LOG_NUDGE_INTERVAL_MINUTES,
            id="nudge_chronic_zero_log_users",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            f"⏰ [P1-2/ZERO-LOG-NUDGE] Cron registrado cada {CHUNK_ZERO_LOG_NUDGE_INTERVAL_MINUTES} min "
            f"(detection_days={CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS}, "
            f"cooldown={CHUNK_ZERO_LOG_NUDGE_COOLDOWN_HOURS}h, max_users={CHUNK_ZERO_LOG_NUDGE_MAX_USERS})."
        )

    # [P1-6] Flush de buffer local de deferrals (re-intenta INSERTs que fallaron por DB blip).
    # Frecuencia configurable vía CHUNK_DEFERRALS_FLUSH_INTERVAL_MINUTES (default 5 min).
    # El cron lee deferrals_pending.jsonl, intenta re-insertar cada record y deja en el
    # archivo solo los que aún fallan (descarta los que violan NOT NULL/schema).
    if not scheduler.get_job("flush_pending_deferrals"):
        from constants import CHUNK_DEFERRALS_FLUSH_INTERVAL_MINUTES as _P16_INT
        scheduler.add_job(
            _flush_pending_deferrals,
            "interval",
            minutes=_P16_INT,
            id="flush_pending_deferrals",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(f"⏰ [P1-6] Cron _flush_pending_deferrals registrado cada {_P16_INT} min.")

    # [P1-2] Detección de deferrals crónicos por TZ (cada 6h por defecto).
    if not scheduler.get_job("detect_chronic_deferrals"):
        from constants import CHUNK_CHRONIC_DEFERRAL_CHECK_INTERVAL_MINUTES as _P12_INT
        scheduler.add_job(
            _detect_chronic_deferrals,
            "interval",
            minutes=_P12_INT,
            id="detect_chronic_deferrals",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(f"⏰ [P1-2] Cron _detect_chronic_deferrals registrado cada {_P12_INT} min.")

    # [P0-A] Alerta cuando el ratio de lecciones sintetizadas (low-confidence) supera
    # umbral. Síntoma de plan_chunk_queue.learning_metrics no persistiéndose, lo que
    # degrada silenciosamente el aprendizaje continuo.
    if not scheduler.get_job("alert_high_synthesized_lesson_ratio"):
        from constants import CHUNK_LESSON_SYNTH_ALERT_INTERVAL_MINUTES as _P0A_INT
        scheduler.add_job(
            _alert_high_synthesized_lesson_ratio,
            "interval",
            minutes=_P0A_INT,
            id="alert_high_synthesized_lesson_ratio",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            f"⏰ [P0-A] Cron _alert_high_synthesized_lesson_ratio registrado cada {_P0A_INT} min."
        )

    # [P1-2/DEAD-LETTER-ALERT] Alerta proactiva sobre chunks dead-lettered acumulados.
    # Operadores reciben señal agregada cuando _escalate_unrecoverable_chunk corre con
    # frecuencia anómala — evitando que cada caso quede invisible hasta ticket de soporte.
    if not scheduler.get_job("alert_new_dead_lettered_chunks"):
        from constants import CHUNK_DEAD_LETTER_ALERT_INTERVAL_MINUTES as _P12_DL_INT
        scheduler.add_job(
            _alert_new_dead_lettered_chunks,
            "interval",
            minutes=_P12_DL_INT,
            id="alert_new_dead_lettered_chunks",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            f"⏰ [P1-2/DEAD-LETTER-ALERT] Cron _alert_new_dead_lettered_chunks registrado cada {_P12_DL_INT} min."
        )

    # [P0-A/ZOMBIE-PARTIAL] Cron que cierra planes que se quedaron en `partial`
    # indefinidamente porque todos sus chunks restantes terminaron en estados
    # terminales sin commitear el merge final. Sin esto, el frontend mostraba
    # "generando próximos días…" eternamente y el usuario no podía pasar página.
    if not scheduler.get_job("finalize_zombie_partial_plans"):
        scheduler.add_job(
            _finalize_zombie_partial_plans,
            "interval",
            minutes=CHUNK_ZOMBIE_PARTIAL_FINALIZE_INTERVAL_MINUTES,
            id="finalize_zombie_partial_plans",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            f"⏰ [P0-A/ZOMBIE-PARTIAL] Cron _finalize_zombie_partial_plans registrado cada "
            f"{CHUNK_ZOMBIE_PARTIAL_FINALIZE_INTERVAL_MINUTES} min "
            f"(min_age={CHUNK_ZOMBIE_PARTIAL_MIN_AGE_HOURS}h, "
            f"batch_limit={CHUNK_ZOMBIE_PARTIAL_BATCH_LIMIT})."
        )

    # [P1-A/ORPHAN-RESERVATIONS] Cron de cleanup de reservas de inventario huérfanas
    # de chunks ya terminados o eliminados. Defensa en profundidad sobre P1-A: aunque
    # `release_chunk_reservations` ahora es atómico, este cron recoge legacy state y
    # casos donde el caller olvidó invocarlo.
    if not scheduler.get_job("recover_orphan_chunk_reservations"):
        scheduler.add_job(
            _recover_orphan_chunk_reservations,
            "interval",
            minutes=CHUNK_ORPHAN_RESERVATION_CLEANUP_INTERVAL_MINUTES,
            id="recover_orphan_chunk_reservations",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(
            f"⏰ [P1-A/ORPHAN-RESERVATIONS] Cron _recover_orphan_chunk_reservations registrado cada "
            f"{CHUNK_ORPHAN_RESERVATION_CLEANUP_INTERVAL_MINUTES} min "
            f"(min_terminal_age={CHUNK_ORPHAN_RESERVATION_MIN_TERMINAL_AGE_HOURS}h, "
            f"batch_limit={CHUNK_ORPHAN_RESERVATION_BATCH_LIMIT})."
        )

    # [P0-5] Sync periódico de tz_offset entre user_profiles y chunks pending/stale.
    # Cubre el caso donde el cambio de TZ del perfil ocurrió SIN pasar por
    # update_user_health_profile (e.g., migración de datos, edición admin directa) o
    # donde la sincronización inmediata falló silenciosamente.
    if not scheduler.get_job("sync_chunk_queue_tz_offsets"):
        from constants import CHUNK_TZ_SYNC_INTERVAL_MINUTES as _TZ_SYNC_INT
        scheduler.add_job(
            _sync_chunk_queue_tz_offsets,
            "interval",
            minutes=_TZ_SYNC_INT,
            id="sync_chunk_queue_tz_offsets",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info(f"⏰ [P0-5] Cron _sync_chunk_queue_tz_offsets registrado cada {_TZ_SYNC_INT} min.")

    # [P0-D] Job nocturno local: una vez por hora evaluamos qué usuarios están en su
    # ventana local de las 03:00 y refrescamos TODOS los chunks pending de planes >=15d.
    if not scheduler.get_job("nightly_refresh_long_plan_snapshots"):
        scheduler.add_job(
            _nightly_refresh_all_pending_snapshots,
            CronTrigger(minute=0, timezone=timezone.utc),
            id="nightly_refresh_long_plan_snapshots",
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        logger.info("⏰ [P0-D] Cron _nightly_refresh_all_pending_snapshots registrado cada hora (target local 03:00).")


def _pantry_refresh_horizon_hours_for_plan(total_days_requested: int | None) -> int:
    """Return the proactive pantry horizon based on plan length."""
    try:
        total_days = int(total_days_requested or 0)
    except Exception:
        total_days = 0
    if total_days >= 30:
        return CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS
    return min(CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS, 48)


def _extract_pantry_snapshot_from_inventory(inventory, top_n: int = 30) -> dict:
    """[P1-D] Convierte una lista de inventory items a snapshot compacto {name_lower: qty_total}.

    Acepta el formato heterogéneo que devuelve `get_user_inventory_net`:
      - Lista de dicts con `ingredient_name` + `quantity` (camino moderno).
      - Lista de tuplas/listas (name, qty) (camino legacy / tests).
      - Lista de strings tipo "2 uds Pollo" (path muy antiguo) — se ignoran porque
        parsear strings ad-hoc es frágil.

    Suma cantidades cuando un ingrediente aparece varias veces (e.g., dos lotes de
    pollo con fechas distintas) y retorna los `top_n` por cantidad descendente.
    Sin esto, dos lotes pequeños del mismo item se sobrescribirían y el snapshot
    subreportaría el stock real.

    Returns:
        dict {ingredient_name (lowercase, stripped): float_quantity}. Vacío si no
        hay items válidos.
    """
    if not inventory:
        return {}
    accumulator: dict = {}
    for item in inventory:
        name = None
        qty = None
        if isinstance(item, dict):
            name = item.get("ingredient_name") or item.get("name")
            qty = item.get("quantity")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            name, qty = item[0], item[1]
        if not name:
            continue
        try:
            qty_f = float(qty or 0)
        except (TypeError, ValueError):
            continue
        if qty_f <= 0:
            continue
        key = str(name).lower().strip()
        if not key:
            continue
        accumulator[key] = accumulator.get(key, 0.0) + qty_f
    if not accumulator:
        return {}
    sorted_items = sorted(accumulator.items(), key=lambda kv: -kv[1])[:top_n]
    return {k: round(v, 2) for k, v in sorted_items}


def _compute_pantry_diff_warning(
    prev_snapshot: dict,
    current_inventory,
    drop_threshold_pct: float = 30.0,
    top_n: int = 10,
) -> dict | None:
    """[P1-D] Detecta drift significativo entre el snapshot al final del chunk N y el
    inventario al inicio del chunk N+1, para inyectar un warning estructurado al LLM.

    El sistema YA hace drift validation POST-LLM con retries (línea ~12354). Pero el
    LLM nunca SUPO qué cambió antes de generar — solo recibe el inventario actual.
    Cuando un ingrediente clave bajó 80% desde que se planeó el chunk anterior, el
    LLM puede no priorizar usar lo que queda eficientemente. Este warning le da
    contexto: "estos items bajaron significativamente, prioriza usarlos antes de que
    se acaben; estos otros aumentaron, considera incorporarlos".

    Args:
        prev_snapshot: dict {ingredient_name_lowercase: quantity_float} del fin del
            chunk previo (persistido en plan_data._pantry_snapshot_per_chunk).
        current_inventory: lista de inventory items o dict ya extraído.
        drop_threshold_pct: % mínimo de cambio para reportar (default 30 = ±30%).
        top_n: máximo de items por categoría.

    Returns:
        dict con `critical_drops`, `notable_increases`, `new_items` si hay drift
        significativo. None si no hay datos suficientes o no hay cambios reportables.
    """
    if not isinstance(prev_snapshot, dict) or not prev_snapshot:
        return None

    if isinstance(current_inventory, dict):
        current_map = {str(k).lower().strip(): float(v or 0) for k, v in current_inventory.items()}
    else:
        current_map = _extract_pantry_snapshot_from_inventory(current_inventory, top_n=200)

    drops: list = []
    increases: list = []
    for name_lower, prev_qty in prev_snapshot.items():
        try:
            prev_f = float(prev_qty or 0)
        except (TypeError, ValueError):
            continue
        if prev_f <= 0:
            continue
        current_f = current_map.get(name_lower, 0.0)
        delta_pct = ((current_f - prev_f) / prev_f) * 100.0
        if delta_pct <= -drop_threshold_pct:
            drops.append({
                "ingredient": name_lower,
                "prev_qty": round(prev_f, 2),
                "current_qty": round(current_f, 2),
                "delta_pct": round(delta_pct, 1),
            })
        elif delta_pct >= drop_threshold_pct:
            increases.append({
                "ingredient": name_lower,
                "prev_qty": round(prev_f, 2),
                "current_qty": round(current_f, 2),
                "delta_pct": round(delta_pct, 1),
            })

    # Items nuevos (compras desde el último chunk): pueden ser oportunidades para
    # variedad. No se mezclan con notable_increases porque no tienen prev_qty con
    # el cual computar un %.
    new_items: list = []
    for name_lower, current_f in current_map.items():
        if name_lower not in prev_snapshot and current_f > 0:
            new_items.append({"ingredient": name_lower, "current_qty": round(current_f, 2)})

    if not drops and not increases and not new_items:
        return None

    drops.sort(key=lambda x: x["delta_pct"])
    increases.sort(key=lambda x: -x["delta_pct"])
    new_items.sort(key=lambda x: -x["current_qty"])

    return {
        "critical_drops": drops[:top_n],
        "notable_increases": increases[:top_n],
        "new_items": new_items[:top_n],
        "_drop_threshold_pct": drop_threshold_pct,
    }


def _coordinate_user_horizons(rows: list) -> dict:
    """[P1-C] Coordina el horizon de refresh entre múltiples planes activos del mismo usuario.

    Sin coordinación: un usuario con plan 7d (horizon=48h) y plan 30d (horizon=168h)
    refrescaba snapshots a cadencias distintas. Si plan 30d capturó la nevera hace 84h
    y plan 7d hace 12h, ambos pueden generar platos contemporáneos viendo neveras muy
    distintas — el del 30d planifica con stock que ya no existe.

    Con coordinación: si el usuario tiene chunks de >=2 planes distintos en el batch,
    todos sus chunks usan el horizon MÁS CORTO (refresh más frecuente = vista más
    coherente del inventario). Para usuarios con 1 plan, el horizon es el normal.

    Args:
        rows: lista de filas con keys `user_id`, `meal_plan_id`, `total_days_requested`.

    Returns:
        Dict {user_id → effective_horizon_hours}. Usuarios ausentes en `rows` no
        aparecen; el caller debe usar el helper individual `_pantry_refresh_horizon_hours_for_plan`
        como fallback cuando un user_id no esté en el dict.
    """
    user_plans: dict = {}
    user_min_horizon: dict = {}
    for row in rows or []:
        uid = row.get("user_id")
        plan_id = row.get("meal_plan_id")
        if uid is None or plan_id is None:
            continue
        user_plans.setdefault(uid, set()).add(plan_id)
        plan_horizon = _pantry_refresh_horizon_hours_for_plan(row.get("total_days_requested"))
        existing = user_min_horizon.get(uid)
        if existing is None or plan_horizon < existing:
            user_min_horizon[uid] = plan_horizon

    # Solo emitimos coordinated horizon para usuarios con >=2 planes; el resto cae al
    # fallback en el caller (que usa _pantry_refresh_horizon_hours_for_plan normal).
    coordinated: dict = {}
    for uid, plan_set in user_plans.items():
        if len(plan_set) >= 2 and uid in user_min_horizon:
            coordinated[uid] = user_min_horizon[uid]
    return coordinated


def _is_user_local_refresh_hour(now_utc: datetime, tz_offset_minutes: int | None, target_hour: int = 3) -> bool:
    try:
        offset = int(tz_offset_minutes or 0)
    except Exception:
        offset = 0
    user_now = now_utc - timedelta(minutes=offset)
    return user_now.hour == target_hour


def _nightly_refresh_all_pending_snapshots(now_utc: datetime | None = None) -> None:
    """[P0-D] Refresh all pending chunks of long plans (>=15d) at ~03:00 local user time."""
    now_utc = now_utc or datetime.now(timezone.utc)
    try:
        rows = execute_sql_query(
            """
            SELECT DISTINCT ON (q.user_id, q.meal_plan_id)
                q.id AS task_id,
                q.user_id::text AS user_id,
                q.meal_plan_id::text AS meal_plan_id,
                COALESCE(
                    (q.pipeline_snapshot->'form_data'->>'tzOffset')::int,
                    (q.pipeline_snapshot->'form_data'->>'tz_offset_minutes')::int,
                    0
                ) AS tz_offset_minutes,
                COALESCE(
                    (p.plan_data->>'total_days_requested')::int,
                    (q.pipeline_snapshot->'form_data'->>'total_days_requested')::int,
                    (q.pipeline_snapshot->'form_data'->>'totalDays')::int,
                    7
                ) AS total_days_requested
            FROM plan_chunk_queue q
            JOIN meal_plans p ON q.meal_plan_id = p.id
            WHERE q.status = 'pending'
              AND COALESCE(
                    (p.plan_data->>'total_days_requested')::int,
                    (q.pipeline_snapshot->'form_data'->>'total_days_requested')::int,
                    (q.pipeline_snapshot->'form_data'->>'totalDays')::int,
                    7
                  ) >= 15
            ORDER BY q.user_id, q.meal_plan_id, q.execute_after ASC
            LIMIT %s
            """,
            (CHUNK_PANTRY_PROACTIVE_REFRESH_MAX_USERS * 8,),
            fetch_all=True
        ) or []
        
        if not rows:
            return
            
        rows = [
            row for row in rows
            if _is_user_local_refresh_hour(
                now_utc,
                row.get("tz_offset_minutes"),
                target_hour=3,
            )
        ]
        if not rows:
            return

        refreshed_count = 0
        
        for row in rows:
            uid = row.get("user_id")
            plan_id = row.get("meal_plan_id")
            if not uid or not plan_id:
                continue
                
            live_inventory = get_user_inventory_net(uid)
            if live_inventory is not None:
                try:
                    _persist_fresh_pantry_to_chunks(row.get("task_id"), plan_id, list(live_inventory), user_id=str(uid))
                    refreshed_count += 1
                except Exception as inner_e:
                    logger.warning(f"[NIGHTLY REFRESH] Error actualizando plan {plan_id}: {inner_e}")
                    
        if refreshed_count > 0:
            logger.info(
                f"🌙 [P0-D] Nightly refresh completado: actualizados {refreshed_count} planes largos "
                f"en ventana local 03:00."
            )
            
    except Exception as e:
        logger.error(f"❌ [NIGHTLY REFRESH] Error general: {e}")

def _proactive_refresh_pending_pantry_snapshots(now_utc: datetime | None = None) -> None:
    """[P0-C] Refresca snapshots de inventario en chunks pending/stale antes de su execute_after.

    Solo targetea chunks cuyo _pantry_captured_at supera la mitad del TTL y que ejecutan
    dentro del horizonte dinámico por longitud de plan. Reusa _persist_fresh_pantry_to_chunks que ya
    propaga el live a todos los siblings vivos del mismo plan.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    refresh_threshold_hours = CHUNK_PANTRY_SNAPSHOT_TTL_HOURS / 2.0
    horizon_hours = CHUNK_PANTRY_PROACTIVE_REFRESH_HORIZON_HOURS
    max_users = CHUNK_PANTRY_PROACTIVE_REFRESH_MAX_USERS

    try:
        # Una fila por (user_id, meal_plan_id) con el chunk más urgente como ancla.
        # Filtramos por edad del snapshot dentro de SQL para no traer chunks ya frescos.
        candidates = execute_sql_query(
            """
            SELECT DISTINCT ON (user_id, meal_plan_id)
                q.id AS task_id,
                q.user_id::text AS user_id,
                q.meal_plan_id::text AS meal_plan_id,
                q.week_number,
                q.execute_after,
                COALESCE(
                    (q.pipeline_snapshot->'form_data'->>'_pantry_captured_at')::timestamptz,
                    q.created_at
                ) AS captured_at,
                COALESCE(
                    (p.plan_data->>'total_days_requested')::int,
                    (q.pipeline_snapshot->'form_data'->>'total_days_requested')::int,
                    (q.pipeline_snapshot->'form_data'->>'totalDays')::int,
                    7
                ) AS total_days_requested
            FROM plan_chunk_queue q
            LEFT JOIN meal_plans p ON q.meal_plan_id = p.id
            WHERE q.status IN ('pending', 'stale')
              AND q.execute_after <= %s
              AND COALESCE(
                  (q.pipeline_snapshot->'form_data'->>'_pantry_captured_at')::timestamptz,
                  q.created_at
              ) < %s
            ORDER BY q.user_id, q.meal_plan_id, q.execute_after ASC
            LIMIT %s
            """,
            (
                now_utc + timedelta(hours=horizon_hours),
                now_utc - timedelta(hours=refresh_threshold_hours),
                max_users * 8,
            ),
            fetch_all=True,
        ) or []
    except Exception as e:
        logger.warning(f"[P0-C/PROACTIVE] Error consultando candidatos: {e}")
        return

    # [P1-C] Coordinar horizons entre múltiples planes activos del mismo usuario.
    # Si el usuario tiene chunks de >=2 planes en el batch, usar el horizon MÁS CORTO
    # para todos. Sin esto, plan 7d (horizon=48h) y plan 30d (horizon=168h) refrescaban
    # a cadencias distintas, generando platos contemporáneos con vistas de nevera
    # desincronizadas.
    coordinated_horizons = _coordinate_user_horizons(candidates)

    def _effective_horizon_hours(row):
        uid = row.get("user_id")
        if uid in coordinated_horizons:
            return coordinated_horizons[uid]
        return _pantry_refresh_horizon_hours_for_plan(row.get("total_days_requested"))

    candidates = [
        row for row in candidates
        if row.get("execute_after") is not None
        and row.get("execute_after") <= now_utc + timedelta(hours=_effective_horizon_hours(row))
    ]
    if not candidates:
        return

    # Agrupar por user_id (una sola lectura live por usuario, sirve para todos sus planes).
    by_user: dict[str, list[dict]] = {}
    for row in candidates:
        by_user.setdefault(row["user_id"], []).append(row)
        if len(by_user) >= max_users:
            break

    refreshed_users = 0
    refreshed_plans = 0
    failed_users = 0

    for user_id, rows in by_user.items():
        try:
            live_inventory = get_user_inventory_net(user_id)
        except Exception as e:
            failed_users += 1
            logger.debug(f"[P0-C/PROACTIVE] Live fetch falló para {user_id}: {e}")
            continue

        if live_inventory is None:
            failed_users += 1
            continue

        refreshed_users += 1
        for row in rows:
            try:
                _persist_fresh_pantry_to_chunks(
                    row["task_id"], row["meal_plan_id"], list(live_inventory), user_id=str(user_id)
                )
                refreshed_plans += 1
            except Exception as e:
                logger.debug(
                    f"[P0-C/PROACTIVE] No se pudo propagar a plan {row['meal_plan_id']}: {e}"
                )

    logger.info(
        f"[P0-C/PROACTIVE] Refresh ejecutado: users_ok={refreshed_users} "
        f"users_failed={failed_users} plans_propagated={refreshed_plans} "
        f"(threshold={refresh_threshold_hours:.1f}h, max_horizon={horizon_hours}h)."
    )


def _fetch_inventory_with_backoff(user_id: str, timeouts_csv: str | None = None) -> tuple[list | None, list[int], str | None]:
    """[P0-4] Live-fetch de inventario con retry de backoff escalado.

    Cada intento usa un timeout creciente (default: 30s, 60s, 90s) ejecutándose en un
    ThreadPoolExecutor para forzar el timeout incluso si `get_user_inventory_net` no lo
    respeta internamente. Devuelve `(pantry | None, durations_ms, last_error)`.

    - `pantry`: la nevera del usuario si algún intento tuvo éxito; `None` si todos fallaron.
    - `durations_ms`: tiempos por intento, útiles para telemetría/diagnóstico de p95.
    - `last_error`: string del último error, o None si tuvo éxito.
    """
    import concurrent.futures
    import time
    from constants import CHUNK_LIVE_FETCH_BACKOFF_TIMEOUTS_SECONDS as _DEF_BACKOFF

    csv_source = (timeouts_csv or _DEF_BACKOFF or "30,60,90").strip()
    try:
        timeouts = [max(5, int(x.strip())) for x in csv_source.split(",") if x.strip()]
    except Exception:
        timeouts = [30, 60, 90]
    if not timeouts:
        timeouts = [30, 60, 90]

    durations_ms: list[int] = []
    last_error: str | None = None
    for attempt_idx, timeout_s in enumerate(timeouts, start=1):
        start_ts = time.monotonic()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_user_inventory_net, user_id)
                pantry = future.result(timeout=timeout_s)
            elapsed_ms = int((time.monotonic() - start_ts) * 1000)
            durations_ms.append(elapsed_ms)
            if pantry is not None:
                logger.info(
                    f"[P0-4/BACKOFF] Live-fetch OK para {user_id} en intento {attempt_idx}/{len(timeouts)} "
                    f"({elapsed_ms}ms, timeout={timeout_s}s)."
                )
                return pantry, durations_ms, None
            # pantry is None: tratar como fallo recuperable (sigue al próximo intento)
            last_error = "returned_none"
            logger.warning(
                f"[P0-4/BACKOFF] Live-fetch devolvió None para {user_id} en intento {attempt_idx}/{len(timeouts)} ({elapsed_ms}ms)."
            )
        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_ts) * 1000)
            durations_ms.append(elapsed_ms)
            last_error = f"{type(e).__name__}: {e}"
            logger.warning(
                f"[P0-4/BACKOFF] Live-fetch falló para {user_id} en intento {attempt_idx}/{len(timeouts)} "
                f"({elapsed_ms}ms, timeout={timeout_s}s): {last_error}"
            )
    return None, durations_ms, last_error


def _get_user_tz_live(user_id: str, fallback_minutes: int = 0) -> int:
    """Lee tz_offset_minutes vivo de user_profiles.health_profile.

    Devuelve `fallback_minutes` si el query falla, el perfil no existe o no contiene
    el campo. Usado por flujos sensibles a timezone (pantry refresh, learning gate)
    para detectar cambios de zona del usuario sin depender del snapshot del chunk.
    """
    if not user_id or user_id == "guest":
        return int(fallback_minutes)
    try:
        row = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,),
            fetch_one=True,
        )
        if not row or not row.get("health_profile"):
            return int(fallback_minutes)
        hp = row.get("health_profile") or {}
        val = hp.get("tz_offset_minutes")
        if val is None:
            val = hp.get("tzOffset")
        if val is None:
            return int(fallback_minutes)
        return int(val)
    except Exception as e:
        logger.warning(f"[TZ-LIVE] No se pudo leer tz_offset vivo para {user_id}: {e}")
        return int(fallback_minutes)


def _sync_chunk_queue_tz_offsets(target_user_id: str | None = None) -> int:
    """[P0-5] Sincroniza el tz_offset de chunks pending/stale con el user_profile vivo.

    Antes el `tz_offset_minutes` se snapshoteaba al encolar y nunca se actualizaba.
    Si el usuario viajaba y actualizaba su perfil después, los chunks ya en cola
    seguían con el offset viejo → `execute_after` calculado con la TZ equivocada
    → chunk dispara en hora errónea o el learning gate cree que el día previo
    no terminó.

    Esta función se ejecuta:
      - Periódicamente (cron horario, vía `register_plan_chunk_scheduler`).
      - Inmediatamente cuando `update_user_health_profile` detecta cambio de TZ
        (via `target_user_id` para limitar el scope).

    Detecta drift entre el snapshot y el user_profile vivo (umbral
    CHUNK_TZ_DRIFT_THRESHOLD_MINUTES) y reescribe:
      - `pipeline_snapshot.form_data.tzOffset` y `tz_offset_minutes`
      - `execute_after` ajustado por la diferencia (delta_minutos = nuevo - viejo)

    Devuelve el número de chunks actualizados.
    """
    from constants import CHUNK_TZ_DRIFT_THRESHOLD_MINUTES as _THRESHOLD
    if target_user_id and (not target_user_id or target_user_id == "guest"):
        return 0

    base_query = """
        SELECT q.id, q.user_id, q.execute_after,
               COALESCE(
                 (q.pipeline_snapshot->'form_data'->>'tz_offset_minutes')::int,
                 (q.pipeline_snapshot->'form_data'->>'tzOffset')::int,
                 0
               ) AS snapshot_tz,
               COALESCE(
                 (p.health_profile->>'tz_offset_minutes')::int,
                 (p.health_profile->>'tzOffset')::int,
                 NULL
               ) AS live_tz
        FROM plan_chunk_queue q
        JOIN user_profiles p ON p.id = q.user_id
        WHERE q.status IN ('pending', 'stale')
    """
    params: tuple
    if target_user_id:
        query = base_query + " AND q.user_id = %s LIMIT 200"
        params = (target_user_id,)
    else:
        query = base_query + " LIMIT 500"
        params = ()

    try:
        rows = execute_sql_query(query, params, fetch_all=True) or []
    except Exception as e:
        logger.warning(f"[P0-5/TZ-SYNC] SELECT falló: {e}")
        return 0

    updated = 0
    for row in rows:
        chunk_id = row.get("id")
        snapshot_tz = row.get("snapshot_tz")
        live_tz = row.get("live_tz")
        if live_tz is None:
            continue
        # [P0-4] Edge: snapshot sin tz_offset (e.g., chunk encolado por path antiguo
        # o con form_data parcial). El execute_after ya se computó con tz_offset
        # asumido como 0; no podemos reconstruir el delta exacto. Persistimos el
        # live_tz al snapshot (defensa hacia adelante) y emitimos warning para
        # auditoría — sin desplazar execute_after porque desconocemos el offset
        # original asumido. Si quedan minutos de descalce, la lógica de pantry
        # refresh (P0-2-EXT en _refresh_pantry_and_get_inventory) detectará el
        # drift al ejecutar y forzará el path correcto.
        if snapshot_tz is None:
            try:
                live_tz_int = int(live_tz)
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET pipeline_snapshot = jsonb_set(
                            jsonb_set(
                                pipeline_snapshot,
                                '{form_data,tzOffset}',
                                to_jsonb(%s::int),
                                true
                            ),
                            '{form_data,tz_offset_minutes}',
                            to_jsonb(%s::int),
                            true
                        ),
                        updated_at = NOW()
                    WHERE id = %s AND status IN ('pending', 'stale')
                    """,
                    (live_tz_int, live_tz_int, chunk_id),
                )
                logger.warning(
                    f"[P0-4/TZ-SYNC] Chunk {chunk_id} (user={row.get('user_id')}): "
                    f"snapshot_tz=NULL → live_tz={live_tz_int}m. "
                    f"Persistido al snapshot SIN shift de execute_after (delta desconocido)."
                )
            except Exception as e:
                logger.warning(f"[P0-4/TZ-SYNC] Backfill snapshot_tz NULL falló para {chunk_id}: {e}")
            continue
        try:
            snapshot_tz = int(snapshot_tz)
            live_tz = int(live_tz)
        except (TypeError, ValueError):
            continue
        drift = abs(live_tz - snapshot_tz)
        if drift < _THRESHOLD:
            continue

        # Recalcular execute_after: el snapshot original era `start_dt + delay + tz_offset_snap + 30m`.
        # Cuando el tz_offset cambia, el "ahora local" del usuario cambia en `(live - snap)` minutos.
        # Sumamos esa diferencia al execute_after para que el chunk dispare a la misma hora local
        # que originalmente se planificó. Si el usuario viajó al oeste (live < snap, p.ej. -240→-300),
        # delta es negativo y execute_after se atrasa; si viajó al este, se adelanta. Ambos
        # comportamientos preservan la intención: "ejecutar al amanecer del día N en la zona del usuario".
        delta_minutes = live_tz - snapshot_tz

        # [P0-C] Guard de overdue: si `execute_after` ya está vencido (la mañana local del
        # usuario ya pasó) y el delta es positivo (TZ al este, "mañana local" más temprana
        # de lo planificado), aplicar el shift sumaría minutos a algo ya vencido y
        # retrasaría todavía más el disparo. La intención correcta es disparar YA: el
        # usuario ya cruzó su ventana objetivo, no tiene sentido empujarla al futuro.
        # Solo aplicamos el cap a delta > 0; si delta < 0 (TZ al oeste) el shift hacia
        # atrás sobre algo vencido no cambia el comportamiento práctico (sigue vencido
        # y se procesa en el próximo tick del worker).
        current_execute_after = row.get("execute_after")
        now_utc = datetime.now(timezone.utc)
        was_overdue = False
        if current_execute_after is not None:
            try:
                if current_execute_after.tzinfo is None:
                    current_execute_after = current_execute_after.replace(tzinfo=timezone.utc)
                was_overdue = current_execute_after <= now_utc
            except Exception:
                was_overdue = False

        force_now = was_overdue and delta_minutes > 0

        try:
            if force_now:
                # Disparar inmediatamente: el chunk debió ejecutarse antes del resync.
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET pipeline_snapshot = jsonb_set(
                            jsonb_set(
                                pipeline_snapshot,
                                '{form_data,tzOffset}',
                                to_jsonb(%s::int),
                                true
                            ),
                            '{form_data,tz_offset_minutes}',
                            to_jsonb(%s::int),
                            true
                        ),
                        execute_after = NOW(),
                        updated_at = NOW()
                    WHERE id = %s AND status IN ('pending', 'stale')
                    """,
                    (live_tz, live_tz, chunk_id),
                )
                updated += 1
                logger.warning(
                    f"[P0-C/TZ-SYNC] Chunk {chunk_id} (user={row.get('user_id')}): "
                    f"tz {snapshot_tz}m → {live_tz}m (drift={drift}m). Chunk YA vencido y "
                    f"delta={delta_minutes:+d}m lo retrasaría más; "
                    f"forzando execute_after=NOW() para disparar ya."
                )
            else:
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET pipeline_snapshot = jsonb_set(
                            jsonb_set(
                                pipeline_snapshot,
                                '{form_data,tzOffset}',
                                to_jsonb(%s::int),
                                true
                            ),
                            '{form_data,tz_offset_minutes}',
                            to_jsonb(%s::int),
                            true
                        ),
                        execute_after = execute_after + make_interval(mins => %s),
                        updated_at = NOW()
                    WHERE id = %s AND status IN ('pending', 'stale')
                    """,
                    (live_tz, live_tz, delta_minutes, chunk_id),
                )
                updated += 1
                logger.info(
                    f"[P0-5/TZ-SYNC] Chunk {chunk_id} (user={row.get('user_id')}): "
                    f"tz {snapshot_tz}m → {live_tz}m (drift={drift}m, "
                    f"execute_after shift={delta_minutes}m, was_overdue={was_overdue})"
                )
        except Exception as e:
            logger.warning(
                f"[P0-5/TZ-SYNC] No se pudo actualizar chunk {chunk_id}: {e}"
            )

    if updated:
        scope = f"user={target_user_id}" if target_user_id else "global"
        logger.info(f"[P0-5/TZ-SYNC] Sync completado ({scope}): {updated} chunk(s) actualizado(s).")
    return updated


def _read_inventory_live_failure_log(user_id: str) -> list[str]:
    """[P0-2] Lee el log de timestamps ISO de fallos de live-fetch desde user_profiles.

    Devuelve lista vacía si el usuario no existe, no tiene health_profile, o no tiene
    el campo. Cualquier error lo trata como log vacío (best-effort, no propaga).
    """
    if not user_id or user_id == "guest":
        return []
    try:
        row = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,),
            fetch_one=True,
        )
        if not row:
            return []
        hp = row.get("health_profile") or {}
        log = hp.get("_inventory_live_failure_log") or []
        if not isinstance(log, list):
            return []
        return [str(ts) for ts in log if ts]
    except Exception as e:
        logger.debug(f"[P0-2/LIVE-FAILURE-LOG] No se pudo leer log para {user_id}: {e}")
        return []


def _is_inventory_live_degraded(user_id: str) -> tuple[bool, int, list[str]]:
    """[P0-2] Determina si el live-fetch del usuario está en estado degradado.

    Lee el log de fallos persistido en user_profiles.health_profile, descarta
    entradas más viejas que `CHUNK_LIVE_FETCH_DEGRADED_WINDOW_HOURS`, y compara
    contra `CHUNK_LIVE_FETCH_DEGRADED_FAILURES_THRESHOLD`.

    Devuelve `(degraded, recent_failures_count, recent_failures_iso_list)`.
    """
    from constants import (
        CHUNK_LIVE_FETCH_DEGRADED_FAILURES_THRESHOLD as _THRESHOLD,
        CHUNK_LIVE_FETCH_DEGRADED_WINDOW_HOURS as _WINDOW_H,
        safe_fromisoformat,
    )
    raw_log = _read_inventory_live_failure_log(user_id)
    if not raw_log:
        return False, 0, []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=_WINDOW_H)
    recent: list[str] = []
    for ts_iso in raw_log:
        try:
            ts = safe_fromisoformat(ts_iso)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                recent.append(ts_iso)
        except Exception:
            continue
    return (len(recent) >= _THRESHOLD, len(recent), recent)


def _record_inventory_live_failure(user_id: str, error_str: str | None = None) -> bool:
    """[P0-2] Registra un fallo de live-fetch en user_profiles.health_profile.

    Append idempotente: añade el timestamp actual al array `_inventory_live_failure_log`,
    poda entradas más viejas que la ventana, y persiste. Devuelve True si tras este
    fallo el usuario quedó en estado degradado (>= threshold dentro de la ventana).

    `error_str` se loggea pero no se persiste — el log es una lista de timestamps puros
    para mantener el JSON pequeño (cap implícito: threshold * historic chunks ≈ 30 entries).
    """
    if not user_id or user_id == "guest":
        return False
    from constants import (
        CHUNK_LIVE_FETCH_DEGRADED_WINDOW_HOURS as _WINDOW_H,
        CHUNK_LIVE_FETCH_DEGRADED_FAILURES_THRESHOLD as _THRESHOLD,
        safe_fromisoformat,
    )
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=_WINDOW_H)
    existing = _read_inventory_live_failure_log(user_id)
    pruned: list[str] = []
    for ts_iso in existing:
        try:
            ts = safe_fromisoformat(ts_iso)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                pruned.append(ts_iso)
        except Exception:
            continue
    pruned.append(now.isoformat())
    # Cap absoluto: nunca más de threshold*4 entries para evitar bloat si la ventana
    # es muy larga y hay decenas de chunks fallando.
    pruned = pruned[-max(_THRESHOLD * 4, 12):]

    try:
        execute_sql_write(
            """
            UPDATE user_profiles
            SET health_profile = jsonb_set(
                COALESCE(health_profile, '{}'::jsonb),
                '{_inventory_live_failure_log}',
                %s::jsonb,
                true
            )
            WHERE id = %s
            """,
            (json.dumps(pruned, ensure_ascii=False), user_id),
        )
    except Exception as e:
        logger.warning(
            f"[P0-2/LIVE-FAILURE-LOG] No se pudo persistir failure log para {user_id}: {e}"
        )
        # Sin persistencia no podemos detectar degradación entre chunks; no bloqueamos.
        return False

    degraded = len(pruned) >= _THRESHOLD
    if degraded:
        logger.warning(
            f"[P0-2/LIVE-DEGRADED] Usuario {user_id} en estado degradado: "
            f"{len(pruned)} fallos de live-fetch en últimas {_WINDOW_H}h "
            f"(threshold={_THRESHOLD}, last_error={error_str!r})"
        )
    else:
        logger.info(
            f"[P0-2/LIVE-FAILURE-LOG] Failure registrado para {user_id} "
            f"({len(pruned)}/{_THRESHOLD} en {_WINDOW_H}h, last_error={error_str!r})"
        )
    return degraded


def _maybe_notify_user_live_degraded(user_id: str) -> bool:
    """[P0-2] Push idempotente avisando al usuario que el live-fetch está degradado.

    Cooldown: 24h entre notificaciones (persistido en
    `user_profiles.health_profile._inventory_live_degraded_notified_at`).
    Devuelve True si se envió notificación, False si estaba en cooldown.
    """
    if not user_id or user_id == "guest":
        return False
    try:
        row = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,),
            fetch_one=True,
        )
        hp = (row or {}).get("health_profile") or {}
        last_iso = hp.get("_inventory_live_degraded_notified_at")
        if last_iso:
            try:
                from constants import safe_fromisoformat as _sfi
                last_dt = _sfi(last_iso)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - last_dt) < timedelta(hours=24):
                    return False
            except Exception:
                pass

        from constants import CHUNK_STALE_PANTRY_DEEPLINK as _DL_DEG
        _dispatch_push_notification(
            user_id=user_id,
            title="Generamos tu plan con datos parciales",
            body=(
                "Tu nevera no se está sincronizando ahora mismo. Generamos los "
                "próximos días con la última versión disponible — revísalos cuando "
                "vuelva la sincronización."
            ),
            url=_DL_DEG,
        )
        execute_sql_write(
            """
            UPDATE user_profiles
            SET health_profile = jsonb_set(
                COALESCE(health_profile, '{}'::jsonb),
                '{_inventory_live_degraded_notified_at}',
                %s::jsonb,
                true
            )
            WHERE id = %s
            """,
            (json.dumps(datetime.now(timezone.utc).isoformat()), user_id),
        )
        return True
    except Exception as e:
        logger.debug(f"[P0-2/LIVE-DEGRADED-NOTIFY] Falló para {user_id}: {e}")
        return False


# [P0-2] Mapeo de reasons internos a copy del push de "plan en modo flexible".
# El reason llega desde form_data._pantry_degraded_reason (set en cada activación
# de flexible_mode) y se traduce al usuario en lenguaje neutro: nunca menciona
# "live fetch", "snapshot", o "TTL", solo si su nevera quedó desincronizada o
# vacía. Si llega un reason desconocido, el default genérico cubre el caso.
_P02_PANTRY_DEGRADED_PUSH_COPY = {
    "live_fetch_degraded": (
        "Plan generado con datos parciales",
        "Tu nevera no se sincronizó a tiempo. Revisa los próximos días — algunos "
        "ingredientes pueden necesitar verificación.",
    ),
    "stale_snapshot_force_flex": (
        "Plan generado con tu última nevera",
        "Tu nevera lleva un tiempo sin sincronizar. Generamos el siguiente bloque "
        "con la versión disponible — confirma los ingredientes cuando puedas.",
    ),
    "stale_snapshot_auto_flex": (
        "Plan generado con tu última nevera",
        "Tu nevera lleva un tiempo sin sincronizar. Generamos el siguiente bloque "
        "con la versión disponible — confirma los ingredientes cuando puedas.",
    ),
    "zero_log_force_flex": (
        "Plan generado sin registro de comidas",
        "No registraste comidas en los días anteriores, así que generamos sin "
        "ajustar el aprendizaje. Empieza a registrar para mejorar las próximas "
        "recomendaciones.",
    ),
    "degraded_flexible_meal": (
        "Plan generado con tu nevera vacía",
        "Tu nevera está casi vacía; generamos opciones flexibles. Actualiza tu "
        "inventario para que las próximas recetas se ajusten mejor.",
    ),
    "manual_user_optin": (
        "Plan en modo flexible",
        "Estás generando con modo flexible activo. Las recetas pueden incluir "
        "ingredientes fuera de tu nevera registrada.",
    ),
}


def _maybe_notify_user_pantry_degraded(
    user_id: str,
    reason: str,
    *,
    cooldown_hours: int = 6,
) -> bool:
    """[P0-2] Push idempotente para chunks completados en flexible_mode.

    Complementa a `_maybe_notify_user_live_degraded` (cooldown 24h, específico de
    live-fetch). Esta función se dispara cuando un chunk llega a `completed`
    estando en flexible_mode por CUALQUIER reason (live degraded, snapshot stale,
    nevera vacía, zero-log, opt-in manual, etc.). El usuario debe saber que el
    plan que acaba de aterrizar en su pantalla usó datos parciales para que pueda
    auditarlo.

    Cooldown 6h (vs 24h del live-degraded) porque este push es por-chunk: en un
    plan 30d con 8 chunks repartidos en >2 semanas, 24h dejaría sin notificar
    chunks subsecuentes; 6h permite avisar sobre cada chunk degradado distinto
    sin spamear si caen consecutivos.

    Args:
        user_id: id del usuario destinatario.
        reason: motivo de la degradación (ver `_P02_PANTRY_DEGRADED_PUSH_COPY`).
        cooldown_hours: ventana de cooldown; default 6.

    Returns:
        True si el push se envió, False si estaba en cooldown / falló.
    """
    if not user_id or user_id == "guest":
        return False
    try:
        row = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,),
            fetch_one=True,
        )
        hp = (row or {}).get("health_profile") or {}
        last_iso = hp.get("_pantry_degraded_notified_at")
        if last_iso:
            try:
                from constants import safe_fromisoformat as _sfi
                last_dt = _sfi(last_iso)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - last_dt) < timedelta(hours=cooldown_hours):
                    logger.debug(
                        f"[P0-2/PANTRY-DEGRADED-NOTIFY] Cooldown activo para {user_id} "
                        f"(<{cooldown_hours}h desde último push)."
                    )
                    return False
            except Exception:
                pass

        title, body = _P02_PANTRY_DEGRADED_PUSH_COPY.get(
            reason,
            (
                "Plan generado en modo flexible",
                "Generamos los próximos días con datos parciales de tu nevera. "
                "Revisa los ingredientes y ajusta lo que necesites.",
            ),
        )
        from constants import CHUNK_STALE_PANTRY_DEEPLINK as _DL_PD
        _dispatch_push_notification(
            user_id=user_id,
            title=title,
            body=body,
            url=_DL_PD,
        )
        execute_sql_write(
            """
            UPDATE user_profiles
            SET health_profile = jsonb_set(
                COALESCE(health_profile, '{}'::jsonb),
                '{_pantry_degraded_notified_at}',
                %s::jsonb,
                true
            )
            WHERE id = %s
            """,
            (json.dumps(datetime.now(timezone.utc).isoformat()), user_id),
        )
        logger.info(
            f"[P0-2/PANTRY-DEGRADED-NOTIFY] Push enviado a {user_id} (reason={reason})."
        )
        return True
    except Exception as e:
        logger.debug(f"[P0-2/PANTRY-DEGRADED-NOTIFY] Falló para {user_id}: {e}")
        return False


def _maybe_notify_user_stale_snapshot_paused(
    user_id: str,
    snapshot_age_hours: float | None,
    *,
    cooldown_hours: int = 24,
) -> bool:
    """[P0-1] Push idempotente cuando un chunk queda en pausa por `stale_snapshot`.

    Caso original (P0-2): cuando el snapshot vence y el live-fetch también está caído,
    el chunk se pausaba con `reason="stale_snapshot"` SIN notificar al usuario, bajo el
    razonamiento "es server-side, el usuario no puede accionar nada". El recovery cron
    intentaba live-retry cada tick y, si funcionaba, desbloqueaba en silencio.

    Gap (P0-1): si el live-fetch sigue caído y/o el usuario abre la app horas después,
    se encuentra con su plan congelado sin saber por qué. Abrir la app y pulsar
    "Refrescar nevera" suele revivir el live-fetch, así que el usuario SÍ puede
    accionar — necesita saber que tiene que hacerlo. Ahora notificamos al pausar,
    con cooldown 24h (alineado con `_maybe_notify_user_live_degraded`) para
    deduplicar entre múltiples chunks pausados del mismo usuario.

    Cooldown se persiste en `user_profiles.health_profile
    ._stale_snapshot_paused_notified_at`. Falla suave: si el push o el UPDATE de
    cooldown fallan, el chunk sigue pausado correctamente — la notificación es
    best-effort.

    Args:
        user_id: id del usuario destinatario.
        snapshot_age_hours: edad del snapshot al pausar (para mensaje contextual).
        cooldown_hours: ventana de cooldown; default 24.

    Returns:
        True si se envió, False si en cooldown / falló / guest.
    """
    if not user_id or user_id == "guest":
        return False
    try:
        row = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,),
            fetch_one=True,
        )
        hp = (row or {}).get("health_profile") or {}
        last_iso = hp.get("_stale_snapshot_paused_notified_at")
        if last_iso:
            try:
                from constants import safe_fromisoformat as _sfi
                last_dt = _sfi(last_iso)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - last_dt) < timedelta(hours=cooldown_hours):
                    logger.debug(
                        f"[P0-1/STALE-PAUSE-NOTIFY] Cooldown activo para {user_id} "
                        f"(<{cooldown_hours}h desde último push)."
                    )
                    return False
            except Exception:
                pass

        from constants import CHUNK_STALE_PANTRY_DEEPLINK as _DL_SP
        _age_int = int(round(snapshot_age_hours)) if snapshot_age_hours else 0
        body = (
            f"Tu nevera lleva {_age_int}h sin sincronizar y dejamos en pausa el siguiente "
            f"bloque de tu plan. Abre la app y refresca tu nevera para continuar."
            if _age_int
            else (
                "Tu nevera no se está sincronizando y dejamos en pausa el siguiente bloque "
                "de tu plan. Abre la app y refresca tu nevera para continuar."
            )
        )
        _dispatch_push_notification(
            user_id=user_id,
            title="Tu plan está en pausa",
            body=body,
            url=_DL_SP,
        )
        execute_sql_write(
            """
            UPDATE user_profiles
            SET health_profile = jsonb_set(
                COALESCE(health_profile, '{}'::jsonb),
                '{_stale_snapshot_paused_notified_at}',
                %s::jsonb,
                true
            )
            WHERE id = %s
            """,
            (json.dumps(datetime.now(timezone.utc).isoformat()), user_id),
        )
        logger.info(
            f"[P0-1/STALE-PAUSE-NOTIFY] Push enviado a {user_id} "
            f"(snapshot_age={_age_int}h)."
        )
        return True
    except Exception as e:
        logger.debug(f"[P0-1/STALE-PAUSE-NOTIFY] Falló para {user_id}: {e}")
        return False


def _maybe_notify_user_tz_unresolved(
    user_id: str,
    *,
    cooldown_hours: int = 24,
) -> bool:
    """[P0-2] Push idempotente cuando un chunk queda pausado por TZ no resoluble.

    Caso: `_resolve_chunk_start_anchor` agotó snapshot → profile → último plan, y
    devolvió source='forced_8am_utc'. Antes el chunk se encolaba con execute_after
    a 8am UTC + delay_days; en TZs negativas eso disparaba a 3am hora local y la
    generación corría antes de que el usuario registrara comidas del día previo.

    Ahora el chunk se pausa con reason='tz_unresolved' y notificamos al usuario
    para que abra la app — al hacerlo el frontend escribe `tz_offset_minutes` en
    `user_profiles.health_profile`, lo que permite al recovery cron re-resolver
    el ancla y reanudar el chunk con execute_after correcto.

    Cooldown 24h (alineado con `_maybe_notify_user_live_degraded`) deduplica
    cuando varios chunks del mismo usuario quedan pausados por la misma causa.
    Persistencia en `user_profiles.health_profile._tz_unresolved_notified_at`.

    Returns:
        True si se envió el push; False si en cooldown / falló / guest.
    """
    if not user_id or user_id == "guest":
        return False
    try:
        row = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,),
            fetch_one=True,
        )
        hp = (row or {}).get("health_profile") or {}
        last_iso = hp.get("_tz_unresolved_notified_at")
        if last_iso:
            try:
                from constants import safe_fromisoformat as _sfi
                last_dt = _sfi(last_iso)
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc) - last_dt) < timedelta(hours=cooldown_hours):
                    logger.debug(
                        f"[P0-2/TZ-UNRESOLVED-NOTIFY] Cooldown activo para {user_id} "
                        f"(<{cooldown_hours}h desde último push)."
                    )
                    return False
            except Exception:
                pass

        _dispatch_push_notification(
            user_id=user_id,
            title="Necesitamos tu zona horaria",
            body=(
                "Dejamos en pausa los próximos días de tu plan porque no pudimos "
                "detectar tu zona horaria. Abre Mealfit y se sincronizará "
                "automáticamente para reanudar la generación."
            ),
            url="/dashboard?action_required=tz_unresolved",
        )
        execute_sql_write(
            """
            UPDATE user_profiles
            SET health_profile = jsonb_set(
                COALESCE(health_profile, '{}'::jsonb),
                '{_tz_unresolved_notified_at}',
                %s::jsonb,
                true
            )
            WHERE id = %s
            """,
            (json.dumps(datetime.now(timezone.utc).isoformat()), user_id),
        )
        logger.info(
            f"[P0-2/TZ-UNRESOLVED-NOTIFY] Push enviado a {user_id}."
        )
        return True
    except Exception as e:
        logger.debug(f"[P0-2/TZ-UNRESOLVED-NOTIFY] Falló para {user_id}: {e}")
        return False


def _record_inventory_live_success(user_id: str) -> None:
    """[P0-2] Resetea el log de fallos tras un live-fetch exitoso.

    Solo emite UPDATE si el log no estaba ya vacío, para evitar writes innecesarios
    en el caso feliz (live siempre funciona).
    """
    if not user_id or user_id == "guest":
        return
    existing = _read_inventory_live_failure_log(user_id)
    if not existing:
        return
    try:
        execute_sql_write(
            """
            UPDATE user_profiles
            SET health_profile = jsonb_set(
                COALESCE(health_profile, '{}'::jsonb),
                '{_inventory_live_failure_log}',
                '[]'::jsonb,
                true
            )
            WHERE id = %s
            """,
            (user_id,),
        )
        logger.info(
            f"[P0-2/LIVE-RECOVERED] Live-fetch recuperado para {user_id}, "
            f"reseteando failure log ({len(existing)} entries → 0)."
        )
    except Exception as e:
        logger.debug(f"[P0-2/LIVE-FAILURE-LOG] Reset falló para {user_id}: {e}")


def _refresh_chunk_pantry(
    user_id: str,
    form_data: dict,
    snapshot_form_data: dict | None = None,
    task_id: str | int | None = None,
    week_number: int | None = None,
) -> dict:
    """Refresh pantry from live inventory, falling back to the snapshot when needed.

    [P0-4] When using snapshot fallback, validates the snapshot age against
    CHUNK_PANTRY_SNAPSHOT_TTL_HOURS. If stale, attempts one extra live retry before
    accepting the stale data and marking source as 'stale_snapshot'.

    [P0-2] If live retry fails and snapshot age > TTL:
    - If age > CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS (24h): force generation in flexible mode.
    - Else: pause the chunk and request user action.
    """
    snapshot_form_data = snapshot_form_data or {}
    pantry_fallback = copy.deepcopy(snapshot_form_data.get("current_pantry_ingredients") or [])

    if not user_id or user_id == "guest":
        form_data["current_pantry_ingredients"] = pantry_fallback
        form_data["_fresh_pantry_source"] = "guest"
        return form_data

    # [P0-2-EXT] Detectar TZ drift entre snapshot y user_profile vivo ANTES del fetch.
    # Si hay drift mayor (>= CHUNK_TZ_MAJOR_DRIFT_MINUTES) significa cambio de país /
    # viaje real: el "hoy" del usuario cambió, así que validar contra un snapshot que
    # refleja consumo del día previo es peligroso. Marcamos la flag para que la caída
    # del live escale agresivamente (extended retry + pausa) en vez de aceptar snapshot.
    from constants import (
        CHUNK_TZ_DRIFT_THRESHOLD_MINUTES as _TZ_THRESHOLD,
        CHUNK_TZ_MAJOR_DRIFT_MINUTES as _TZ_MAJOR,
    )
    _snapshot_tz = int(
        snapshot_form_data.get("tzOffset")
        or snapshot_form_data.get("tz_offset_minutes")
        or form_data.get("tzOffset")
        or form_data.get("tz_offset_minutes")
        or 0
    )
    _live_tz = _get_user_tz_live(user_id, _snapshot_tz)
    _tz_drift = abs(_live_tz - _snapshot_tz)
    _major_tz_drift = _tz_drift >= _TZ_MAJOR
    if _tz_drift >= _TZ_THRESHOLD:
        # Propagar el tz vivo al form_data para que cálculos downstream (learning gate,
        # captured_at age en próximas iteraciones) lean valores coherentes.
        form_data["tzOffset"] = _live_tz
        form_data["tz_offset_minutes"] = _live_tz
        form_data["_tz_drift_at_pantry_refresh_minutes"] = _tz_drift
        if _major_tz_drift:
            form_data["_tz_major_drift_at_pantry_refresh"] = True
        logger.info(
            f"🌍 [P0-2-EXT/PANTRY-TZ] Drift detectado para {user_id}: "
            f"snapshot={_snapshot_tz}m, live={_live_tz}m, drift={_tz_drift}m, major={_major_tz_drift}."
        )

    # [P0-3] Antes: si `pantry_live` retornaba None (sin excepción), el código caía
    # silenciosamente al snapshot, marcaba `_fresh_pantry_source="live"` (mentira) y NO
    # entraba al flujo de re-validación de snapshot age. En planes 30d con chunks
    # pending durante semanas, eso significaba generar contra un snapshot de 20+ días.
    # Ahora un None se trata igual que una excepción: se desvía al flujo de fallback
    # con re-validación de age, retry y, si la nevera sigue inalcanzable, pausa o
    # escala a flexible_mode.
    try:
        pantry_live = get_user_inventory_net(user_id)
        if pantry_live is None:
            raise RuntimeError("get_user_inventory_net returned None (no exception, but no data)")
        form_data["current_pantry_ingredients"] = pantry_live
        form_data["_fresh_pantry_source"] = "live"
        # [P0-3/TELEMETRY] Aún con live OK, exponer el age del snapshot que el chunk
        # llevaba al momento del pickup. Útil para auditar atrasos del cron proactivo.
        _captured_at_for_age = snapshot_form_data.get("_pantry_captured_at")
        if _captured_at_for_age:
            try:
                from constants import safe_fromisoformat as _sfi
                _cap_dt = _sfi(_captured_at_for_age)
                if _cap_dt.tzinfo is None:
                    _cap_dt = _cap_dt.replace(tzinfo=timezone.utc)
                form_data["_pantry_snapshot_age_hours"] = round(
                    (datetime.now(timezone.utc) - _cap_dt).total_seconds() / 3600.0, 2
                )
            except Exception:
                pass
        logger.debug(f"[P0-4/PANTRY] Inventario live OK para {user_id} ({len(pantry_live)} items).")
        # [P0-2] Live OK → resetear contador de fallos sistémicos.
        _record_inventory_live_success(user_id)
        return form_data
    except Exception as e:
        logger.warning(
            f"[P0-4/PANTRY] Error refrescando inventario vivo para chunk de {user_id}: {e}. "
            "Evaluando snapshot como fallback."
        )
        # [P0-2] Registrar fallo. Si esto cruza el threshold, evaluamos escalada
        # inmediata más abajo en lugar de pasar por el preview window de 4-24h.
        _live_degraded_now = _record_inventory_live_failure(user_id, str(e))

    # [P0-2-EXT] Major TZ drift + live caído: NO degradar a snapshot. Forzar el extended-retry
    # path inmediatamente (mismo flujo que cuando el snapshot supera FORCE_GENERATE_HOURS),
    # y si también falla pausar con reason específico para que el recovery cron y el usuario
    # entiendan que fue un viaje, no un fallo de servidor.
    if _major_tz_drift:
        logger.warning(
            f"[P0-2-EXT/PANTRY-TZ] Drift mayor ({_tz_drift}m) + live caído para {user_id}. "
            f"Forzando live-retry con backoff ignorando snapshot."
        )
        pantry_tz_retry, _tz_durations_ms, _tz_last_err = _fetch_inventory_with_backoff(user_id)
        form_data["_live_fetch_backoff_durations_ms"] = _tz_durations_ms
        if pantry_tz_retry is not None:
            form_data["current_pantry_ingredients"] = pantry_tz_retry
            form_data["_fresh_pantry_source"] = "live_tz_drift_forced"
            logger.info(
                f"[P0-2-EXT/PANTRY-TZ] Live forzado por TZ drift exitoso para {user_id} "
                f"({len(pantry_tz_retry)} items, intentos={_tz_durations_ms})."
            )
            # [P0-2] Backoff retry funcionó → resetear contador.
            _record_inventory_live_success(user_id)
            return form_data
        logger.warning(
            f"[P0-2-EXT/PANTRY-TZ] Live forzado por TZ drift falló tras backoff para {user_id}: "
            f"intentos={_tz_durations_ms}, last_error={_tz_last_err}."
        )
        # [P0-2] Backoff también falló → escalar contador.
        _live_degraded_now = _record_inventory_live_failure(user_id, str(_tz_last_err)) or _live_degraded_now
        if task_id:
            from constants import CHUNK_STALE_PANTRY_DEEPLINK as _DL_TZ
            _dispatch_push_notification(
                user_id=user_id,
                title="Detectamos un cambio de zona horaria",
                body="Refresca tu nevera para continuar tu plan en tu nueva zona horaria.",
                url=_DL_TZ,
            )
            _pause_chunk_for_stale_inventory(
                task_id,
                user_id,
                week_number or 0,
                0.0,
                reason="tz_major_drift_live_unreachable",
            )
            form_data["_pantry_paused"] = True
            form_data["current_pantry_ingredients"] = pantry_fallback
            form_data["_fresh_pantry_source"] = "tz_major_drift_paused"
            form_data["_requires_pantry_review"] = True
            return form_data

    # [P0-4] El live falló. Antes de usar el snapshot, verificar su edad.
    captured_at_str = snapshot_form_data.get("_pantry_captured_at")
    snapshot_age_hours: float | None = None
    if captured_at_str:
        try:
            from constants import safe_fromisoformat
            captured_at = safe_fromisoformat(captured_at_str)
            if captured_at.tzinfo is None:
                captured_at = captured_at.replace(tzinfo=timezone.utc)
            snapshot_age_hours = (datetime.now(timezone.utc) - captured_at).total_seconds() / 3600
        except Exception:
            pass

    if snapshot_age_hours is not None and snapshot_age_hours > CHUNK_PANTRY_SNAPSHOT_TTL_HOURS:
        logger.warning(
            f"[P0-4/PANTRY] Snapshot de inventario para {user_id} tiene {snapshot_age_hours:.1f}h "
            f"(TTL={CHUNK_PANTRY_SNAPSHOT_TTL_HOURS}h). Intentando live retry adicional..."
        )
        try:
            pantry_live_retry = get_user_inventory_net(user_id)
            if pantry_live_retry is not None:
                form_data["current_pantry_ingredients"] = pantry_live_retry
                form_data["_fresh_pantry_source"] = "live"
                logger.info(
                    f"[P0-4/PANTRY] Live retry exitoso para {user_id} "
                    f"(snapshot tenía {snapshot_age_hours:.1f}h). {len(pantry_live_retry)} items."
                )
                # [P0-2] Live retry funcionó → resetear contador.
                _record_inventory_live_success(user_id)
                return form_data
        except Exception as retry_e:
            logger.warning(f"[P0-4/PANTRY] Live retry también falló para {user_id}: {retry_e}.")
            _live_degraded_now = _record_inventory_live_failure(user_id, str(retry_e)) or _live_degraded_now

        # [P0-5] Hard-fail absoluto. Cuando el snapshot supera CHUNK_PANTRY_HARD_FAIL_AGE_HOURS
        # (default 48h) y el live retry inicial ya falló, NO escalamos a flex+advisory:
        # el inventario es demasiado viejo para garantizar el contrato "solo lo que hay en la
        # nevera". Pausamos incondicionalmente. Esto aplica antes de cualquier rama
        # `_live_degraded_now → flexible_mode` aguas abajo (líneas ~1080 y ~1135), que en planes
        # de 30 días con _proactive_refresh caído podían terminar generando contra snapshots
        # de semanas. Si el switch CHUNK_PANTRY_HARD_FAIL_ON_STALE está apagado (override de
        # entorno), preservamos el comportamiento previo para tests y entornos que necesiten
        # ejercitar la rama flex.
        if (
            CHUNK_PANTRY_HARD_FAIL_ON_STALE
            and snapshot_age_hours is not None
            and snapshot_age_hours > CHUNK_PANTRY_HARD_FAIL_AGE_HOURS
        ):
            _hf_age_int = int(round(snapshot_age_hours))
            if task_id:
                logger.error(
                    f"[P0-5/HARD-FAIL] Snapshot {snapshot_age_hours:.1f}h supera "
                    f"hard-fail cap {CHUNK_PANTRY_HARD_FAIL_AGE_HOURS}h para {user_id} "
                    f"(chunk {week_number}, task {task_id}). Pausando: el inventario "
                    f"es demasiado viejo para generar de forma segura."
                )
                from constants import CHUNK_STALE_PANTRY_DEEPLINK as _DL_HF
                try:
                    _dispatch_push_notification(
                        user_id=user_id,
                        title="Refresca tu nevera para continuar tu plan",
                        body=(
                            f"No pudimos validar tu nevera y los datos guardados tienen "
                            f"{_hf_age_int}h. Abre la app para sincronizarla y seguir tu plan."
                        ),
                        url=_DL_HF,
                    )
                except Exception as _hf_notif_err:
                    logger.warning(
                        f"[P0-5/HARD-FAIL] Falló notificación push para {user_id}: {_hf_notif_err}"
                    )
                _pause_chunk_for_stale_inventory(
                    task_id, user_id, week_number or 0, snapshot_age_hours,
                    reason="snapshot_hard_fail_age_exceeded",
                )
                form_data["_pantry_paused"] = True
                form_data["current_pantry_ingredients"] = []
                form_data["_fresh_pantry_source"] = "hard_fail_paused"
                form_data["_pantry_snapshot_age_hours"] = round(snapshot_age_hours, 1)
                return form_data
            # Sin task_id (no debería ocurrir en workers, defensivo): no podemos pausar.
            # Devolvemos pantry vacío + flag para que el caller falle aguas arriba en lugar
            # de generar a ciegas contra el snapshot stale.
            logger.error(
                f"[P0-5/HARD-FAIL] Snapshot {snapshot_age_hours:.1f}h > cap "
                f"{CHUNK_PANTRY_HARD_FAIL_AGE_HOURS}h para {user_id} pero no hay task_id; "
                f"no se puede pausar. Devolviendo pantry vacío para forzar fallo upstream."
            )
            form_data["current_pantry_ingredients"] = []
            form_data["_fresh_pantry_source"] = "hard_fail_no_task"
            form_data["_pantry_paused"] = True
            form_data["_pantry_snapshot_age_hours"] = round(snapshot_age_hours, 1)
            return form_data

        # [P0-2] El live retry falló. Evaluar si pausamos o forzamos generación.
        if snapshot_age_hours > CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS:
            # [P0-4] Reemplazado el intento único con timeout fijo por retry con backoff
            # escalado (30s/60s/90s por defecto). Esto reduce pausas innecesarias por jitter
            # transitorio de la API del usuario y deja telemetría de duraciones para diagnosticar.
            logger.warning(
                f"[P0-3/PANTRY] Snapshot MUY viejo ({snapshot_age_hours:.1f}h). "
                f"Intentando live-fetch con backoff para {user_id}..."
            )
            pantry_extended_retry, _ext_durations_ms, _ext_last_err = _fetch_inventory_with_backoff(user_id)
            form_data["_live_fetch_backoff_durations_ms"] = _ext_durations_ms

            if pantry_extended_retry is not None:
                form_data["current_pantry_ingredients"] = pantry_extended_retry
                form_data["_fresh_pantry_source"] = "live"
                logger.info(
                    f"[P0-3/PANTRY] Live-fetch con backoff exitoso para {user_id}. "
                    f"{len(pantry_extended_retry)} items, intentos={_ext_durations_ms}."
                )
                # [P0-2] Backoff extendido funcionó → resetear contador.
                _record_inventory_live_success(user_id)
                return form_data
            logger.warning(
                f"[P0-3/PANTRY] Live-fetch con backoff agotado para {user_id}: "
                f"intentos={_ext_durations_ms}, last_error={_ext_last_err}."
            )
            # [P0-2] Backoff extendido falló → escalar contador.
            _live_degraded_now = _record_inventory_live_failure(user_id, str(_ext_last_err)) or _live_degraded_now
                
            # [P0-2] Si el usuario ya está en estado degradado (>= threshold de fallos
            # de live en la ventana), bypaseamos el preview window de pausa (4-24h) y
            # entramos directo en flex+advisory_only. Hacerlo aquí evita que cada chunk
            # repita el ciclo pause→retry→pause hasta CHUNK_STALE_MAX_PAUSE_HOURS antes de
            # generar nada. La validación existencial se conserva (advisory_only=True).
            if _live_degraded_now:
                logger.warning(
                    f"[P0-2/LIVE-DEGRADED-ESCALATE] Usuario {user_id} en estado degradado: "
                    f"snapshot {snapshot_age_hours:.1f}h y live caído. Saltando pausa, "
                    f"entrando directo en flex+advisory_only."
                )
                form_data["current_pantry_ingredients"] = pantry_fallback
                form_data["_fresh_pantry_source"] = "live_degraded_snapshot"
                form_data["_pantry_flexible_mode"] = True
                form_data["_pantry_advisory_only"] = True
                form_data["_inventory_live_degraded"] = True
                # [P0-2] Reason explícito para que el merge tag por-día y para que
                # el push de "chunk completed degraded" use el copy correcto.
                form_data["_pantry_degraded_reason"] = "live_fetch_degraded"
                _maybe_notify_user_live_degraded(user_id)
                return form_data

            # [P1-3] Bypass de la pausa cuando CHUNK_PANTRY_HARD_FAIL_ON_STALE=False.
            # Antes, sin importar el valor del flag, el chunk se pausaba con
            # 'stale_snapshot_live_unreachable' y esperaba hasta CHUNK_STALE_MAX_PAUSE_HOURS
            # (24h) antes de que _recover_pantry_paused_chunks lo escalara a flex+advisory.
            # Cuando el operador desactiva hard-fail explícitamente (modo "preferir
            # generar con datos parciales antes que bloquear al usuario"), esos 24h de
            # pausa son innecesarios — escalamos directo a flex+advisory_only para que
            # el chunk continúe la generación con validación existencial conservada.
            if not CHUNK_PANTRY_HARD_FAIL_ON_STALE and task_id:
                from constants import CHUNK_STALE_MAX_PAUSE_HOURS as _P13_MAX_PAUSE
                logger.warning(
                    f"[P1-3/STALE-AUTO-FLEX] Snapshot >{CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS}h "
                    f"y live caído para {user_id}, hard_fail desactivado: escalando directo a "
                    f"flex+advisory_only (sin pasar por pausa de {_P13_MAX_PAUSE}h)."
                )
                form_data["current_pantry_ingredients"] = pantry_fallback
                form_data["_fresh_pantry_source"] = "stale_snapshot_auto_flex"
                form_data["_pantry_flexible_mode"] = True
                form_data["_pantry_advisory_only"] = True
                form_data["_pantry_snapshot_age_hours"] = round(snapshot_age_hours, 1)
                # [P0-2] Reason explícito para day-tagging y push cooldown'd.
                form_data["_pantry_degraded_reason"] = "stale_snapshot_auto_flex"
                # [P0-2] Sustituimos el push directo por el helper con cooldown 6h.
                # Antes este path emitía un push por cada chunk degradado sin cooldown,
                # spameando al usuario en planes 30d con varios chunks consecutivos en
                # auto-flex.
                _maybe_notify_user_pantry_degraded(user_id, "stale_snapshot_auto_flex")
                return form_data

            # [P0-1 FIX] Snapshot MUY viejo (>CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS) y live-fetch
            # también falló. Antes, esta rama forzaba `_pantry_flexible_mode=True` *inmediatamente*
            # y generaba contra un snapshot >8h sin validación contra la nevera real, rompiendo la
            # promesa "los platos solo usan lo que hay en la nevera".
            #
            # Ahora pausamos el chunk en pending_user_action con reason="stale_snapshot_live_unreachable"
            # y notificamos al usuario para que abra la app (refrescar el snapshot). Si tras el TTL
            # de pausa el live sigue caído y el usuario no actuó, el recovery cron
            # (_recover_pantry_paused_chunks) escala a flexible_mode pero con _pantry_advisory_only=True
            # que conserva la validación existencial (¿el ingrediente existe en pantry?) aunque omita
            # la cuantitativa.
            if task_id:
                logger.warning(
                    f"[P0-1/STALE-SNAPSHOT] Snapshot >{CHUNK_STALE_SNAPSHOT_FORCE_GENERATE_HOURS}h y "
                    f"live-fetch caído para {user_id}. Pausando chunk en lugar de generar a ciegas."
                )
                from constants import CHUNK_STALE_PANTRY_DEEPLINK as _DL
                _age_int = int(round(snapshot_age_hours)) if snapshot_age_hours else 0
                _dispatch_push_notification(
                    user_id=user_id,
                    title="Refresca tu nevera para continuar tu plan",
                    body=(
                        f"Tu inventario no se sincroniza hace {_age_int}h. "
                        f"Abre la app y refresca tu nevera para que generemos los próximos días."
                    ),
                    url=_DL,
                )
                _pause_chunk_for_stale_inventory(
                    task_id,
                    user_id,
                    week_number or 0,
                    snapshot_age_hours,
                    reason="stale_snapshot_live_unreachable",
                )
                form_data["_pantry_paused"] = True
                form_data["current_pantry_ingredients"] = pantry_fallback
                form_data["_fresh_pantry_source"] = "stale_snapshot"
                form_data["_requires_pantry_review"] = True
                return form_data

        # [P0-2] Bypass del pause si el usuario está en estado degradado: en lugar de
        # ciclar pause→retry→pause cada 4h hasta MAX_PAUSE, generamos en flex+advisory_only.
        if _live_degraded_now:
            logger.warning(
                f"[P0-2/LIVE-DEGRADED-ESCALATE] Usuario {user_id} en estado degradado: "
                f"snapshot {snapshot_age_hours:.1f}h y live caído. Saltando pausa, "
                f"entrando directo en flex+advisory_only."
            )
            form_data["current_pantry_ingredients"] = pantry_fallback
            form_data["_fresh_pantry_source"] = "live_degraded_snapshot"
            form_data["_pantry_flexible_mode"] = True
            form_data["_pantry_advisory_only"] = True
            form_data["_inventory_live_degraded"] = True
            # [P0-2] Reason explícito para day-tagging.
            form_data["_pantry_degraded_reason"] = "live_fetch_degraded"
            _maybe_notify_user_live_degraded(user_id)
            return form_data

        if task_id:
            # Snapshot vencido y live inaccesible — pausar chunk para que el usuario refresque
            logger.warning(
                f"[P0-2/STALE-PAUSE] Chunk {task_id} pausado para {user_id}: "
                f"snapshot {snapshot_age_hours:.1f}h > TTL {CHUNK_PANTRY_SNAPSHOT_TTL_HOURS}h, "
                f"live también falló."
            )
            _pause_chunk_for_stale_inventory(task_id, user_id, week_number or 0, snapshot_age_hours)
            form_data["_pantry_paused"] = True
            return form_data

        # Fallback si no hay task_id (no debería ocurrir en workers)
        form_data["current_pantry_ingredients"] = pantry_fallback
        form_data["_fresh_pantry_source"] = "stale_snapshot"
        form_data["_pantry_snapshot_age_hours"] = round(snapshot_age_hours, 1)
    else:
        # Snapshot dentro del TTL (o sin timestamp) — aceptable
        form_data["current_pantry_ingredients"] = pantry_fallback
        form_data["_fresh_pantry_source"] = "snapshot"
        if snapshot_age_hours is not None:
            logger.info(
                f"[P0-4/PANTRY] Usando snapshot ({snapshot_age_hours:.1f}h < TTL {CHUNK_PANTRY_SNAPSHOT_TTL_HOURS}h) "
                f"para {user_id}. {len(pantry_fallback)} items."
            )
        else:
            logger.info(f"[P0-4/PANTRY] Usando snapshot sin timestamp para {user_id}. {len(pantry_fallback)} items.")

    return form_data


def _calculate_inventory_drift(old_inv: list, new_inv: list) -> float:
    """Calcula el porcentaje de cambio (deriva) entre dos listas de inventario.
    
    [P1-D] Usa normalización de ingredientes para comparar bases reales. 
    Retorna un float entre 0.0 y 1.0 (drift percentage).
    """
    if not old_inv and not new_inv:
        return 0.0
    
    from constants import normalize_ingredient_for_tracking
    
    def _get_bases(inv_list):
        bases = set()
        for item in (inv_list or []):
            if not item: continue
            # Intentar extraer el nombre si es un string complejo (ej: "200g Pollo")
            try:
                from shopping_calculator import _parse_quantity
                _, _, name = _parse_quantity(item)
                norm = normalize_ingredient_for_tracking(name)
                if norm: bases.add(norm)
            except Exception:
                norm = normalize_ingredient_for_tracking(item)
                if norm: bases.add(norm)
        return bases

    old_bases = _get_bases(old_inv)
    new_bases = _get_bases(new_inv)
    
    if not old_bases and not new_bases:
        return 0.0
    if not old_bases: # Todo es nuevo
        return 1.0
        
    # Diferencia simétrica: items que están en uno pero no en otro
    diff = old_bases ^ new_bases
    # Referencia: tamaño del inventario original (mínimo 1 para evitar div/0)
    drift = len(diff) / max(len(old_bases), 1)
    return min(drift, 1.0)


def _activate_flexible_mode(
    snapshot: dict,
    reason: str,
    *,
    user_id: str | None = None,
    week_num: int | str | None = None,
    meal_plan_id: str | None = None,
    advisory_only: bool = False,
    learning_flexible: bool = False,
    extra_form_data: dict | None = None,
    chunk_lessons: dict | None = None,
) -> dict:
    """[P1-3] Activación canónica de flexible_mode con log estructurado.

    Antes había 4+ callsites scattered que seteaban `_pantry_flexible_mode=True` con flags
    inconsistentes y sin telemetría unificada. Esta función centraliza la activación:
    cualquier escalación a flexible debe pasar por aquí, garantizando log con motivo y la
    semántica de `advisory_only` (validación existencial preservada).

    `reason` valores conocidos:
      - "stale_snapshot_force_flex"   : live caído + TTL excedido (snapshot_live_unreachable)
      - "zero_log_force_flex"          : usuario no logueó comidas y TTL expiró
      - "degraded_flexible_meal"       : nevera vacía persistente, fallback genérico
      - "manual_user_optin"            : (futuro) opt-in explícito del usuario

    [P1-3/MODE-HISTORY] Si `meal_plan_id` está presente, persistimos el evento en
    `meal_plans.plan_data._mode_history` (lista append-only) para que el endpoint
    que sirve el plan al frontend pueda exponer un flag y mostrar el badge
    "Plan en modo flexible — verifica tu nevera". La activación in-memory NO se
    abortaba si la persistencia falla — el chunk debe seguir generándose; la falta
    de aviso al frontend se loggea pero no bloquea.
    """
    snapshot["_degraded"] = True
    snapshot["_pantry_flexible_mode"] = True
    if learning_flexible:
        snapshot["_learning_flexible_mode"] = True
    if advisory_only:
        snapshot["_pantry_advisory_only"] = True

    fd = snapshot.get("form_data", {}) or {}
    if advisory_only:
        fd["_pantry_advisory_only"] = True
    # [P0-2] Reason canónico en form_data para day-tagging en el merge y para que
    # `_maybe_notify_user_pantry_degraded` use el copy correcto. Antes este reason
    # vivía solo en el snapshot/`_mode_history`, fuera del alcance del worker que
    # marca los días post-LLM.
    fd["_pantry_flexible_mode"] = True
    fd["_pantry_degraded_reason"] = reason
    if extra_form_data:
        fd.update(extra_form_data)
    snapshot["form_data"] = fd

    if chunk_lessons is not None:
        snapshot["_chunk_lessons"] = chunk_lessons

    snapshot["_pantry_pause_resolution"] = reason
    snapshot["_pantry_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()

    logger.warning(
        "[P1-3/FLEXIBLE-MODE] activated",
        extra={
            "user_id": user_id,
            "meal_plan_id": meal_plan_id,
            "week_num": week_num,
            "reason": reason,
            "advisory_only": advisory_only,
            "learning_flexible": learning_flexible,
        },
    )

    # [P1-3/MODE-HISTORY] Persistir evento al plan_data para visibilidad del frontend.
    if meal_plan_id:
        try:
            from db_plans import update_plan_data_atomic

            event = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "mode": "flexible",
                "reason": reason,
                "advisory_only": bool(advisory_only),
                "learning_flexible": bool(learning_flexible),
                "week_number": week_num,
            }

            def _append_mode_event(pd: dict) -> dict:
                history = pd.get("_mode_history")
                if not isinstance(history, list):
                    history = []
                history.append(event)
                # Cap defensivo a 100 eventos (planes 30d con 10 chunks rara vez
                # superan 5; este cap evita bloat si algún path bucleara).
                pd["_mode_history"] = history[-100:]
                pd["_current_mode"] = "flexible"
                pd["_current_mode_reason"] = reason
                pd["_current_mode_advisory_only"] = bool(advisory_only)
                return pd

            update_plan_data_atomic(str(meal_plan_id), _append_mode_event)
        except Exception as persist_err:
            logger.warning(
                f"[P1-3/MODE-HISTORY] No se pudo persistir el evento flexible_mode "
                f"para plan {meal_plan_id} (reason={reason}): {persist_err}. "
                f"El chunk continúa pero el frontend no verá el badge."
            )

    return snapshot


def _rolling_lessons_window_cap(total_days_requested: int | None) -> int:
    """[P1-1] Tamaño máximo del rolling window de `_recent_chunk_lessons`.

    Antes el cap era hardcoded `8 if total_days >= 15 else 4`. Para planes 7d
    (que solo tienen 2 chunks) el cap=4 sobre-asignaba; para 15d (que tienen
    4 chunks) el cap=8 también. La nueva fórmula calcula el tope exacto en
    función del número máximo de chunks históricos que un plan puede tener:

        cap = max(2, min(8, ceil(total_days / 3) - 1))

    Resultados:
      -  7d → cap=2 (chunks: 3+4 → 1 lección histórica posible, mín=2 por seguridad)
      - 15d → cap=4 (chunks: 3+4+3+... → hasta 4 lecciones históricas)
      - 30d → cap=8 (chunks: ≥10 → tope hard al 8 para evitar prompt bloat)
      -  0  → cap=4 (default seguro si total_days no está disponible)

    Reducir el cap a su valor exacto:
      1. Documenta intent (cap = N-1 chunks, no número arbitrario).
      2. Tightens 15d donde antes el cap=8 era engañoso (data-bounded a 4).
      3. Limita prompt size en 30d+ (sin cambio de comportamiento).
    """
    import math
    try:
        total = int(total_days_requested or 0)
    except (TypeError, ValueError):
        total = 0
    if total <= 0:
        return 4
    max_historical = math.ceil(total / 3) - 1
    return max(2, min(8, max_historical))


_chunk_lesson_clamp_count: dict = {"total": 0, "by_field": {}}


def _validate_lesson_schema(lesson: dict) -> tuple[bool, str | None]:
    """[P1-1 + P0-B] Valida y CLAMPA una lección reconstruida desde
    `plan_chunk_queue.learning_metrics` antes de inyectarla al prompt del LLM o
    persistirla a plan_data.

    Razón histórica: si una migración previa, un bug de round-trip JSON o una
    manipulación manual dejó un campo con NaN, inf, una string ("NaN" parseado)
    o el tipo equivocado (dict en vez de list para samples), todo el rebuild se
    persistía silenciosamente y el LLM recibía señales corruptas.
    `_safe_lessons_dict`/`_safe_lessons_list` saneaban a vacío al cargarlas,
    perdiendo la lección entera; aquí descartamos solo las inválidas y dejamos
    pasar las correctas.

    [P0-B] Además de rechazar (NaN/inf/no-numérico/bool), ahora CLAMPAMOS valores
    fuera de rango: un `repeat_pct=150` (bug histórico de cálculo) o `=-5`
    (subdesbordamiento por casteo) llegaba al prompt tal cual y el LLM podía
    "ver" porcentajes absurdos y reaccionar erráticamente. Clampamos en sitio
    (mutamos el dict) y devolvemos `(True, None)` para no perder la señal — el
    valor saneado sigue siendo útil. Cada clamp incrementa
    `_chunk_lesson_clamp_count`; un counter elevado en producción es síntoma de
    bug upstream que merece investigación.

    Devuelve (True, None) si es válida (posiblemente tras clampar); (False, reason)
    si no es recuperable, donde `reason` es un string corto para logging.
    """
    import math as _math_p11
    # [P0-B] Rangos por campo. Percentages: [0, 100]. Counters: no-negativos,
    # con tope hard a 10000 para detectar conteos absurdos (un bug que multiplique
    # violaciones por una constante gigante se delata aquí en lugar de envenenar
    # el prompt).
    _NUMERIC_FIELD_RANGES: dict = {
        "repeat_pct": (0.0, 100.0),
        "ingredient_base_repeat_pct": (0.0, 100.0),
        "rejection_violations": (0.0, 10000.0),
        "allergy_violations": (0.0, 10000.0),
        "fatigued_violations": (0.0, 10000.0),
    }
    _LIST_FIELDS = (
        "repeated_bases",
        "repeated_meal_names",
        "rejected_meals_that_reappeared",
        "allergy_hits",
    )
    if not isinstance(lesson, dict):
        return False, f"not_a_dict({type(lesson).__name__})"
    for f, (lo, hi) in _NUMERIC_FIELD_RANGES.items():
        v = lesson.get(f, 0)
        if isinstance(v, bool):
            # bool es subclase de int — rechazar para evitar True/False en métricas.
            return False, f"bool_in_numeric_{f}"
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False, f"non_numeric_{f}={type(v).__name__}"
        if not _math_p11.isfinite(fv):
            return False, f"non_finite_{f}={v!r}"
        # [P0-B] Clamp en sitio si el valor está fuera de rango.
        if fv < lo or fv > hi:
            clamped = max(lo, min(hi, fv))
            lesson[f] = clamped
            _chunk_lesson_clamp_count["total"] += 1
            _chunk_lesson_clamp_count["by_field"][f] = (
                _chunk_lesson_clamp_count["by_field"].get(f, 0) + 1
            )
            n = _chunk_lesson_clamp_count["total"]
            # Loguear el primero y cada 10 después para no spamear pero mantener
            # visibilidad. Si producción acumula muchos clamps, hay bug upstream.
            if n == 1 or n % 10 == 0:
                logger.warning(
                    f"[P0-B/CLAMP] Campo {f}={v!r} fuera de rango [{lo},{hi}] "
                    f"clampeado a {clamped} (clamp #{n}, by_field={dict(_chunk_lesson_clamp_count['by_field'])}). "
                    f"Investigar fuente del valor si el contador crece."
                )
    for f in _LIST_FIELDS:
        v = lesson.get(f, [])
        if not isinstance(v, list):
            return False, f"non_list_{f}={type(v).__name__}"
    return True, None


def _rebuild_recent_chunk_lessons_from_queue(
    meal_plan_id: str,
    up_to_week_exclusive: int,
    total_days_requested: int,
) -> list[dict]:
    """[P1-1] Reconstruye `_recent_chunk_lessons` desde `plan_chunk_queue.learning_metrics`.

    Cuando plan_data se trunca/corrompe (manual edit, migration bug, JSON roundtrip falla),
    `_recent_chunk_lessons` desaparece. Antes esto pausaba el chunk para auditoría humana;
    ahora lo intentamos auto-recuperar leyendo los `learning_metrics` que cada chunk
    completado dejó en su fila de la cola. Devuelve la ventana rolling esperada según
    `_rolling_lessons_window_cap(total_days_requested)` o lista vacía si no hay datos.
    """
    try:
        rows = execute_sql_query(
            """
            SELECT week_number, learning_metrics
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s
              AND status = 'completed'
              AND week_number < %s
              AND learning_metrics IS NOT NULL
            ORDER BY week_number ASC
            """,
            (str(meal_plan_id), int(up_to_week_exclusive)),
        ) or []
    except Exception as e:
        logger.warning(f"[P1-1/REBUILD] Fallo SELECT learning_metrics para plan {meal_plan_id}: {e}")
        return []

    rebuilt: list[dict] = []
    _p11_invalid_count = 0
    for row in rows:
        lm = row.get("learning_metrics")
        if isinstance(lm, str):
            try:
                lm = json.loads(lm)
            except Exception:
                lm = None
        if not isinstance(lm, dict):
            continue
        wn = row.get("week_number")
        is_proxy = bool(lm.get("inventory_activity_proxy_used") or lm.get("sparse_logging_proxy_used"))
        candidate = {
            "repeat_pct": lm.get("learning_repeat_pct", 0),
            "ingredient_base_repeat_pct": lm.get("ingredient_base_repeat_pct", 0),
            "rejection_violations": lm.get("rejection_violations", 0),
            "allergy_violations": lm.get("allergy_violations", 0),
            "fatigued_violations": lm.get("fatigued_violations", 0),
            "repeated_bases": lm.get("sample_repeated_bases", []),
            "repeated_meal_names": lm.get("sample_repeats", []),
            "rejected_meals_that_reappeared": lm.get("sample_rejection_hits", []),
            "allergy_hits": lm.get("sample_allergy_hits", []),
            "chunk": wn,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics_unavailable": False,
            "low_confidence": is_proxy,
            "learning_signal_strength": lm.get("learning_signal_strength", "weak" if is_proxy else "strong"),
            "rebuilt_from_queue": True,
        }
        # [P1-1] Validar antes de añadir. Una lesson con NaN o tipo equivocado en
        # samples envenena el prompt del LLM ("rejection_violations: NaN" → comportamiento
        # impredecible). Mejor descartar la lesson y loguear el motivo que persistir basura.
        _ok, _reason = _validate_lesson_schema(candidate)
        if not _ok:
            _p11_invalid_count += 1
            logger.error(
                f"[P1-1/SCHEMA-INVALID] Descartando lesson reconstruida de plan {meal_plan_id} "
                f"chunk {wn}: {_reason}. learning_metrics raw keys: {sorted(list(lm.keys()))[:10]}"
            )
            continue
        rebuilt.append(candidate)

    if _p11_invalid_count > 0:
        logger.error(
            f"[P1-1/SCHEMA-INVALID] Plan {meal_plan_id}: {_p11_invalid_count} lessons descartadas "
            f"por schema inválido. Si esto se repite consistentemente, investigar plan_chunk_queue.learning_metrics "
            f"raw para detectar bug de persistencia o migración corrupta."
        )

    window_cap = _rolling_lessons_window_cap(total_days_requested)
    return rebuilt[-window_cap:]


def _emit_plan_data_corruption_alert(
    plan_id: str | None,
    user_id: str | None,
    field_name: str,
    observed_type: str,
    expected_type: str,
) -> None:
    """[P1-6] Persiste alerta de corrupción de plan_data en system_alerts y notifica
    al usuario. Antes la corrupción se silenciaba con `[]`/`{}`: el chunk N+1 corría
    sin lecciones del N anterior y la regresión solo se notaba semanas después al
    revisar logs.

    Idempotencia:
      - `alert_key = "plan_data_corrupted:{plan_id}:{field_name}"` con UNIQUE constraint.
      - El mismo plan + mismo field genera UN row con `triggered_at` actualizado en cada
        detección posterior (vía ON CONFLICT DO UPDATE). No proliferan filas.
      - Push al usuario solo si la alerta es "fresca" (>24h desde el último push).
    """
    if not plan_id:
        # Sin plan_id no podemos deduplicar ni rutear; solo loggeamos.
        logger.warning(
            f"[P1-6/CORRUPT-LESSONS] Campo {field_name} en plan_data: tipo {observed_type}, "
            f"esperado {expected_type}. plan_id desconocido — solo log."
        )
        return

    alert_key = f"plan_data_corrupted:{plan_id}:{field_name}"

    # Dedupe: si la alerta misma se disparó en las últimas 24h NO duplicamos el push.
    # La INSERT con ON CONFLICT siempre actualiza triggered_at para tracking de severidad.
    existing_recent = None
    try:
        existing_recent = execute_sql_query(
            "SELECT triggered_at FROM system_alerts WHERE alert_key = %s "
            "AND triggered_at > NOW() - INTERVAL '24 hours' LIMIT 1",
            (alert_key,),
            fetch_one=True,
        )
    except Exception as q_err:
        logger.debug(f"[P1-6] Lookup de alerta dedupe falló: {q_err}")

    try:
        execute_sql_write(
            """
            INSERT INTO system_alerts (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
            VALUES (%s, 'plan_data_corrupted', 'warning', %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                metadata = EXCLUDED.metadata,
                affected_user_ids = EXCLUDED.affected_user_ids,
                resolved_at = NULL
            """,
            (
                alert_key,
                f"plan_data corrupto: {field_name}",
                f"Plan {plan_id}: campo {field_name} llegó como {observed_type}, "
                f"esperado {expected_type}. Chunk continúa con valor vacío; "
                f"el siguiente chunk perderá señales históricas.",
                json.dumps({
                    "field_name": field_name,
                    "observed_type": observed_type,
                    "expected_type": expected_type,
                    "meal_plan_id": plan_id,
                }),
                json.dumps([user_id] if user_id else []),
            ),
        )
    except Exception as ins_err:
        logger.warning(f"[P1-6] No se pudo persistir alerta de corrupción: {ins_err}")
        return  # Sin alerta persistida no enviamos push (puede ser problema transitorio).

    # Push al usuario solo si NO había alerta reciente (cooldown 24h).
    if user_id and not existing_recent:
        try:
            _dispatch_push_notification(
                user_id=user_id,
                title="Detectamos un problema con tu plan",
                body=(
                    "Estamos generando tu siguiente bloque, pero el aprendizaje histórico "
                    "tuvo un problema de datos. Si notas comidas repetidas, regenera tu plan."
                ),
                url="/dashboard?alert=plan_data_corrupted",
            )
        except Exception as push_err:
            logger.warning(f"[P1-6] Push de corrupción falló para {user_id}: {push_err}")


def _safe_lessons_list(
    value,
    field_name: str = "_recent_chunk_lessons",
    plan_id: str | None = None,
    user_id: str | None = None,
) -> list:
    """[P1-6] Devuelve `value` si es lista; si no, devuelve `[]` y loggea warning.

    `plan_data` puede llegar corrupto (edición manual DB, bug en JSON roundtrip,
    migración mal aplicada). Si un campo que el código asume lista llega como
    dict, str, int o None, las iteraciones siguientes crashearían el worker.
    Este helper centraliza la defensa: convierte a lista vacía Y deja rastro
    en logs para detectar la corrupción antes de que el efecto cascada llegue
    al usuario (ej. plan generado sin contexto histórico).

    [P1-6] Cuando se detecta tipo incorrecto (no es None — eso es ausencia
    legítima — sino dict/str/int donde se espera lista), se persiste alerta en
    system_alerts y se envía push al usuario para que pueda regenerar manualmente.
    """
    if isinstance(value, list):
        return value
    if value is None:
        return []
    observed = type(value).__name__
    # [P1-5] Bumped a ERROR (antes warning). El tipo-incorrecto aquí no es ausencia
    # legítima (None ya cubrió ese caso): es plan_data corrupto vivo, lo que rompe
    # la cadena de aprendizaje silenciosamente al devolver []. Logueando como ERROR
    # garantiza que aparezca en alertas de log (Sentry/CloudWatch) y permite detectar
    # racimos de corrupción antes de que afecten muchos planes. La alerta a system_alerts
    # se mantiene en _emit_plan_data_corruption_alert con dedupe por alert_key.
    logger.error(
        f"[P1-5/CORRUPT-LESSONS] Campo {field_name} en plan_data esperaba list, "
        f"recibió {observed}. plan_id={plan_id}. Tratando como []. "
        f"Si recurre en el mismo plan, investigar bug de persistencia o JSON roundtrip."
    )
    _emit_plan_data_corruption_alert(plan_id, user_id, field_name, observed, "list")
    return []


def _safe_lessons_dict(
    value,
    field_name: str = "_last_chunk_learning",
    plan_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """[P1-6] Devuelve `value` si es dict; si no, devuelve `{}` y loggea warning.

    Análogo a `_safe_lessons_list` pero para campos dict (`_last_chunk_learning`,
    `_lifetime_lessons_summary`). Si plan_data fue corrompido y el campo llega
    como list o str, evitamos el AttributeError en el `.get(...)` siguiente y
    persistimos alerta en system_alerts + push al usuario.
    """
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    observed = type(value).__name__
    # [P1-5] Bumped a ERROR (antes warning). Mismo razonamiento que en _safe_lessons_list:
    # devolver {} silenciosamente esconde un bug de persistencia que rompe la cadena de
    # aprendizaje. Logueando ERROR + system_alerts permite detección temprana.
    logger.error(
        f"[P1-5/CORRUPT-LESSONS] Campo {field_name} en plan_data esperaba dict, "
        f"recibió {observed}. plan_id={plan_id}. Tratando como {{}}. "
        f"Si recurre en el mismo plan, investigar bug de persistencia o JSON roundtrip."
    )
    _emit_plan_data_corruption_alert(plan_id, user_id, field_name, observed, "dict")
    return {}


def _is_lesson_stub(lesson) -> bool:
    """[P0-3] Detecta si un dict de `_last_chunk_learning` está vacío o es un stub puro.

    Un stub significa que el seed inicial (chunk 1) o la persistencia post-chunk falló silenciosamente,
    dejando la cadena de aprendizaje rota. Heurística:
      - No es dict, o dict vacío → stub.
      - Marcado explícitamente con `metrics_unavailable=True` → stub.
      - Sin `chunk` asignado → stub (toda lección persistida lleva el número de chunk).
      - Todos los contadores numéricos en cero Y todas las muestras vacías → stub.
    """
    if not isinstance(lesson, dict) or not lesson:
        return True
    if lesson.get("metrics_unavailable") is True:
        return True
    if lesson.get("chunk") in (None, 0):
        return True
    numeric_keys = (
        "repeat_pct",
        "ingredient_base_repeat_pct",
        "rejection_violations",
        "allergy_violations",
        "fatigued_violations",
    )
    sample_keys = (
        "repeated_bases",
        "repeated_meal_names",
        "rejected_meals_that_reappeared",
        "allergy_hits",
    )
    has_numeric_signal = any(float(lesson.get(k) or 0) > 0 for k in numeric_keys)
    has_sample_signal = any(bool(lesson.get(k)) for k in sample_keys)
    return not (has_numeric_signal or has_sample_signal)


def _read_proxy_counter(plan_data, snapshot, field: str) -> int:
    """[P1-3] Lee un contador de proxy con `plan_data` como fuente de verdad.

    Antes el contador (`_consecutive_proxy_chunks`, `_lifetime_proxy_chunks`,
    `_lifetime_total_chunks`) vivía solo en `pipeline_snapshot`. Si dos chunks
    paralelos leían snapshots desfasados o el snapshot se reescribía completo
    (no via `jsonb_set`), el cap consecutivo se podía burlar. Ahora la fuente
    canónica es `meal_plans.plan_data`, que el worker actualiza vía
    `update_plan_data_atomic` (SELECT … FOR UPDATE).

    Reglas:
      - Si `plan_data[field]` existe y es coercible a int, ése es el valor.
      - Sino, fallback al `snapshot[field]` (back-compat con tests/snapshots
        legacy y con el mirror que el worker sigue escribiendo).
      - Default 0 si ninguno existe o ambos son no-coercibles.

    Argumentos:
        plan_data: dict de `meal_plans.plan_data`. Puede ser None.
        snapshot: dict de `plan_chunk_queue.pipeline_snapshot`. Puede ser None.
        field: nombre del contador a leer.
    """
    if isinstance(plan_data, dict) and plan_data.get(field) is not None:
        try:
            return int(plan_data[field])
        except (TypeError, ValueError):
            pass
    try:
        return int((snapshot or {}).get(field, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _filter_lessons_excluding_dead_lettered(
    last_chunk_learning,
    recent_chunk_lessons,
    prior_plan_data: dict,
    current_week_number,
):
    """[P1-1] Excluye lecciones cuyo origen es un chunk dead-lettered.

    Cuando `_escalate_unrecoverable_chunk` marca un chunk como recovery_exhausted,
    el `plan_data.days` correspondiente quedó parcial/inválido (no completó la
    validación final). Si el chunk siguiente extrae `_last_chunk_learning` o
    `_recent_chunk_lessons` desde ese estado, propaga rechazos/alergias
    "fantasma" — señales calculadas sobre días que nunca llegaron a realizarse.

    Usa la lista `_recovery_exhausted_chunks` (escrita por
    `_escalate_unrecoverable_chunk`) como fuente de verdad de qué semanas son
    dead-lettered. Filtra dos canales:

      - `last_chunk_learning`: si su `chunk` figura en dead_weeks, o si la
        semana inmediatamente anterior a la actual fue dead-lettered, devuelve
        {} (forzando que el caller cae al stub o reconstruya desde DB).
      - `recent_chunk_lessons`: filtra elementos cuyo `chunk` figura en
        dead_weeks. Las demás lecciones (chunks anteriores que sí completaron)
        se conservan.

    Devuelve `(filtered_last, filtered_recent, dead_lettered_weeks)`.
    `dead_lettered_weeks` se devuelve como lista ordenada para telemetría.
    """
    if not isinstance(prior_plan_data, dict):
        return last_chunk_learning, recent_chunk_lessons, []
    raw = prior_plan_data.get("_recovery_exhausted_chunks") or []
    if not isinstance(raw, list):
        return last_chunk_learning, recent_chunk_lessons, []
    dead_weeks: set = set()
    for entry in raw:
        if isinstance(entry, dict):
            wn = entry.get("week_number")
            if isinstance(wn, (int, float)):
                dead_weeks.add(int(wn))
    if not dead_weeks:
        return last_chunk_learning, recent_chunk_lessons, []

    try:
        current_wn = int(current_week_number)
    except (TypeError, ValueError):
        current_wn = None

    filtered_last = last_chunk_learning
    if isinstance(last_chunk_learning, dict) and last_chunk_learning:
        lesson_chunk = last_chunk_learning.get("chunk")
        if isinstance(lesson_chunk, (int, float)) and int(lesson_chunk) in dead_weeks:
            filtered_last = {}
        elif current_wn is not None and (current_wn - 1) in dead_weeks:
            filtered_last = {}

    filtered_recent = recent_chunk_lessons
    if isinstance(recent_chunk_lessons, list):
        filtered_recent = []
        for lesson in recent_chunk_lessons:
            if isinstance(lesson, dict):
                lc = lesson.get("chunk")
                if isinstance(lc, (int, float)) and int(lc) in dead_weeks:
                    continue
            filtered_recent.append(lesson)

    return filtered_last, filtered_recent, sorted(dead_weeks)


def _synthesize_last_chunk_learning_from_plan_days(
    meal_plan_id: str,
    target_week: int,
    prior_plan_data: dict,
    *,
    user_id: str | None = None,
) -> dict | None:
    """[P0-4] Last-resort: sintetiza `_last_chunk_learning` desde `plan_data.days`.

    Se llama cuando `_rebuild_last_chunk_learning_from_queue` devolvió None porque
    `plan_chunk_queue.learning_metrics` quedó NULL (chunk previo crasheó pre-preflight,
    downgrade de schema, JSON corrupto). Sin esta red, chunk N arranca con dict vacío
    y puede regenerar platos idénticos al chunk N-1, rompiendo el aprendizaje continuo
    que sí se promete en planes de 7d (que no tienen rolling window de respaldo).

    Estrategia: lee los días del chunk objetivo desde `plan_data.days`, extrae nombres
    de platos consumidos (status != swapped_out/skipped/rejected) y bases de ingredientes,
    los empaqueta como `repeated_meal_names` / `repeated_bases` para que el LLM del
    chunk N+1 sepa qué evitar. Counters numéricos quedan en 0 (no hay señal real de
    repetición vs chunks anteriores) y se marca metrics_unavailable=True para que el
    prompt los interprete como "información ausente", no "0 violaciones".

    [P1-3] Schema validation: cada acceso a `meals`, `meal`, `name`, `ingredients` y
    cada `ing` se valida por tipo. Si una migración previa, una corrupción JSON o un
    bug del LLM produjo `meals=None`, `ingredients="<string>"` (no lista — iteraría
    chars), `name=42`, etc., antes generábamos lecciones inválidas que el LLM siguiente
    consumía como verdad. Ahora descartamos las entradas malformadas y emitimos
    telemetría a `chunk_lesson_telemetry` (event=synth_schema_invalid /
    synth_schema_partial_invalid) para que el cron de alerta lo detecte agregado.

    Si `user_id` se pasa, la telemetría se persiste; si no (e.g., llamadas internas
    desde `_regenerate_recent_chunk_lessons_from_plan_days` que no lo tienen en
    scope), solo logueamos. La validación schema corre igual.
    """
    from constants import normalize_ingredient_for_tracking

    days = prior_plan_data.get("days") if isinstance(prior_plan_data, dict) else None
    if not isinstance(days, list) or not days:
        return None

    target_meal_names: list[str] = []
    target_bases: list[str] = []
    saw_any_chunk_tag = False

    # [P1-3] Counters de schema-invalid por tipo de violación. Permiten distinguir
    # corrupción a nivel meal (meals no-lista, meal no-dict, name no-string) vs
    # ingredient (lista no-list, item no-dict-ni-string). Útil para diagnóstico.
    invalid_meals = 0
    invalid_ingredients = 0
    valid_meals = 0

    for d in days:
        if not isinstance(d, dict):
            continue
        d_week = d.get("week_number") or d.get("chunk_number") or d.get("chunk")
        if d_week is not None:
            saw_any_chunk_tag = True
            try:
                if int(d_week) != int(target_week):
                    continue
            except (TypeError, ValueError):
                continue
        # Si los días no traen tag de chunk (planes viejos), aceptamos todos: peor caso
        # arrastramos info de chunks aún más antiguos, pero eso es mejor que dict vacío.

        # [P1-3] Schema guard: meals debe ser lista. Si es None lo tratamos como
        # "día sin meals" (sin contar como invalid). Otros tipos (str, dict, int)
        # son corrupción real.
        meals = d.get("meals")
        if meals is None:
            continue
        if not isinstance(meals, list):
            invalid_meals += 1
            continue

        for m in meals:
            if not isinstance(m, dict):
                invalid_meals += 1
                continue
            if m.get("status") in ("swapped_out", "skipped", "rejected"):
                continue

            # [P1-3] name debe ser string non-empty. Antes `if name: ... str(name)`
            # convertía 42→"42" o {"foo":1}→"{'foo': 1}", contaminando lecciones.
            name = m.get("name")
            if not isinstance(name, str) or not name.strip():
                invalid_meals += 1
                continue
            target_meal_names.append(name.strip())
            valid_meals += 1

            # [P1-3] ingredients debe ser lista. None es válido (sin ingredientes);
            # str/dict/etc. es corrupción (un str produciría iteración por chars).
            ings = m.get("ingredients")
            if ings is None:
                continue
            if not isinstance(ings, list):
                invalid_ingredients += 1
                continue

            for ing in ings:
                if isinstance(ing, dict):
                    raw = ing.get("name") or ing.get("display_string") or ""
                    if not isinstance(raw, str):
                        invalid_ingredients += 1
                        continue
                elif isinstance(ing, str):
                    raw = ing
                else:
                    invalid_ingredients += 1
                    continue
                if not raw.strip():
                    continue
                base = normalize_ingredient_for_tracking(raw)
                if base:
                    target_bases.append(base)

    discarded_total = invalid_meals + invalid_ingredients

    # [P1-3] Caso A: nada válido. Si encima hubo discards por schema, es señal de
    # corrupción real (no plan vacío). Emitimos telemetría loud y devolvemos None
    # para que el caller no inyecte una lección con counters todos en cero (que
    # el LLM podría interpretar como "no hay repeticiones" — falso positivo).
    if not target_meal_names and not target_bases:
        if discarded_total > 0:
            logger.error(
                f"[P1-3/SYNTH-SCHEMA-INVALID] Plan {meal_plan_id} week {target_week}: "
                f"sin datos válidos para sintetizar — invalid_meals={invalid_meals}, "
                f"invalid_ingredients={invalid_ingredients}. plan_data.days corrupto."
            )
            if user_id and user_id != "guest":
                try:
                    _record_chunk_lesson_telemetry(
                        user_id=user_id,
                        meal_plan_id=meal_plan_id,
                        week_number=int(target_week),
                        event="synth_schema_invalid",
                        synthesized_count=0,
                        queue_count=0,
                        metadata={
                            "invalid_meals": invalid_meals,
                            "invalid_ingredients": invalid_ingredients,
                            "synth_failed_schema": True,
                        },
                    )
                except Exception as _tele_err:
                    logger.debug(
                        f"[P1-3/SYNTH-TELEMETRY] No se pudo persistir "
                        f"synth_schema_invalid: {_tele_err}"
                    )
        return None

    # [P1-3] Caso B: parcialmente válido. Sintetizamos con lo válido pero anotamos
    # el discard count para diagnóstico aguas abajo y emitimos telemetría partial.
    if discarded_total > 0:
        logger.warning(
            f"[P1-3/SYNTH-SCHEMA-PARTIAL] Plan {meal_plan_id} week {target_week}: "
            f"síntesis con {valid_meals} meal(s) válida(s); descartados "
            f"invalid_meals={invalid_meals} invalid_ingredients={invalid_ingredients} "
            f"por schema. Investigar el path que persistió plan_data.days corrupto."
        )
        if user_id and user_id != "guest":
            try:
                _record_chunk_lesson_telemetry(
                    user_id=user_id,
                    meal_plan_id=meal_plan_id,
                    week_number=int(target_week),
                    event="synth_schema_partial_invalid",
                    synthesized_count=valid_meals,
                    queue_count=0,
                    metadata={
                        "invalid_meals": invalid_meals,
                        "invalid_ingredients": invalid_ingredients,
                        "valid_meals": valid_meals,
                    },
                )
            except Exception as _tele_err:
                logger.debug(
                    f"[P1-3/SYNTH-TELEMETRY] No se pudo persistir "
                    f"synth_schema_partial_invalid: {_tele_err}"
                )

    base_counts = Counter(target_bases)
    name_counts = Counter(target_meal_names)

    candidate = {
        "repeat_pct": 0,
        "ingredient_base_repeat_pct": 0,
        "rejection_violations": 0,
        "allergy_violations": 0,
        "fatigued_violations": 0,
        "repeated_bases": [b for b, _ in base_counts.most_common(10)],
        "repeated_meal_names": [n for n, _ in name_counts.most_common(8)],
        "rejected_meals_that_reappeared": [],
        "allergy_hits": [],
        "chunk": int(target_week),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics_unavailable": True,
        "low_confidence": True,
        "learning_signal_strength": "weak",
        "synthesized_from_plan_days": True,
        "synthesized_meal_count": len(target_meal_names),
        "synthesized_chunk_tag_present": saw_any_chunk_tag,
    }
    # [P1-3] Anotar discards en el candidate para que dashboards puedan correlacionar
    # lecciones flacas con corrupción de schema upstream sin recurrir a logs.
    if discarded_total > 0:
        candidate["_synth_schema_discarded"] = {
            "invalid_meals": invalid_meals,
            "invalid_ingredients": invalid_ingredients,
        }

    _ok, _reason = _validate_lesson_schema(candidate)
    if not _ok:
        logger.error(
            f"[P0-4/SYNTHESIZE] Schema inválido al sintetizar para plan {meal_plan_id} "
            f"week {target_week}: {_reason}. Descartando síntesis."
        )
        return None
    return candidate


def _regenerate_recent_chunk_lessons_from_plan_days(
    meal_plan_id: str,
    plan_data: dict,
    target_week: int,
    total_days_requested: int,
    seed_lessons: list | None = None,
) -> list:
    """[P1-2] Regenera la ventana rolling `_recent_chunk_lessons` desde `plan_data.days`.

    Last-resort cuando AMBAS fuentes primarias fallaron:
      1. `plan_data._recent_chunk_lessons` truncado/vacío.
      2. `plan_chunk_queue.learning_metrics` no provee suficientes filas
         (rebuild_from_queue devolvió < expected).

    Estrategia: para cada chunk_index en [1, target_week - 1] que NO esté presente
    en `seed_lessons` (las que sí pudimos reconstruir desde la cola), llamar a
    `_synthesize_last_chunk_learning_from_plan_days` para sintetizar la lección
    desde los días persistidos en `plan_data.days`. Las lecciones sintetizadas
    quedan marcadas con `synthesized_from_plan_days=True` y `low_confidence=True`
    para que el LLM y dashboards puedan distinguirlas.

    Las lecciones de `seed_lessons` (queue rebuild) tienen prioridad sobre las
    sintetizadas — si un chunk_index ya está en seed, no lo regeneramos.

    Returns: lista combinada (seed + sintetizadas) ordenada por chunk_index,
    truncada al rolling window cap. Lista vacía si plan_data.days no tiene
    contenido utilizable.

    Args:
        meal_plan_id: id del plan (solo para logging).
        plan_data: dict completo del plan (lee plan_data["days"]).
        target_week: chunk para el cual estamos regenerando contexto previo
            (NO se sintetiza este chunk; solo los anteriores).
        total_days_requested: total días solicitados, usado para calcular
            el rolling window cap.
        seed_lessons: lecciones ya reconstruidas desde plan_chunk_queue.
    """
    if not isinstance(plan_data, dict):
        return list(seed_lessons or [])
    days = plan_data.get("days") if isinstance(plan_data.get("days"), list) else []
    if not days:
        return list(seed_lessons or [])

    seed_lessons = list(seed_lessons or [])
    seed_by_chunk: dict = {}
    for l in seed_lessons:
        if not isinstance(l, dict):
            continue
        try:
            ch = int(l.get("chunk") or 0)
        except (TypeError, ValueError):
            ch = 0
        if ch > 0:
            seed_by_chunk[ch] = l

    combined: list = []
    synthesized_count = 0
    for ch_idx in range(1, max(1, int(target_week))):
        if ch_idx in seed_by_chunk:
            combined.append(seed_by_chunk[ch_idx])
            continue
        synth = _synthesize_last_chunk_learning_from_plan_days(
            meal_plan_id, ch_idx, plan_data
        )
        if synth:
            synth["synthesized_from_plan_days"] = True
            synth["low_confidence"] = True
            synth["learning_signal_strength"] = "weak"
            combined.append(synth)
            synthesized_count += 1

    if synthesized_count > 0:
        logger.warning(
            f"[P1-2/REGEN] Plan {meal_plan_id} target_week={target_week}: "
            f"sintetizadas {synthesized_count} lecciones desde plan_data.days "
            f"(seed_from_queue={len(seed_by_chunk)}, total_combined={len(combined)})."
        )

    window_cap = _rolling_lessons_window_cap(total_days_requested)
    return combined[-window_cap:]


def _synthesize_user_history_lifetime_summary(user_id: str) -> dict | None:
    """[P1-3] Sintetiza un `_lifetime_lessons_summary` desde la historia del usuario
    cuando el plan actual no tiene lifetime heredado y la ventana rolling es débil.

    Caso típico: chunk 2 de un primer plan 7d (chunks: 3+4 días). El rolling window
    cap es 2 pero solo existe 1 chunk previo, así que `_recent_chunk_lessons` queda
    vacío. Si el usuario no tiene plan previo (P0-1 inheritance no aplica), la única
    señal histórica que recibe el LLM es `_last_chunk_learning` — sin contexto sobre
    rechazos previos del onboarding ni patrones de los últimos 6 meses.

    Esta función combina:
      - `get_active_rejections(user_id)`: platos rechazados permanentes del onboarding.
      - `get_recent_plans(user_id, days=180)`: ingredientes y nombres de los últimos
        6 meses, normalizados a bases canónicas.

    Devuelve un dict compatible con `_lifetime_lessons_summary` marcado como
    `synthesized_from_user_history=True` (low confidence pero truthful). Devuelve
    `None` si el usuario no tiene historia en absoluto (truly new user).
    """
    if not user_id or user_id == "guest":
        return None

    try:
        from db_chat import get_active_rejections as _p13_get_rej
        from db_plans import get_recent_plans as _p13_get_plans
        from constants import normalize_ingredient_for_tracking as _p13_norm
    except Exception as _imp_err:
        logger.debug(f"[P1-3/SYNTH-HISTORY] Imports fallaron: {_imp_err}")
        return None

    rejected_meals: list = []
    repeated_bases_set: set = set()
    repeated_meal_names: list = []
    seen_meal_names: set = set()

    try:
        rejections = _p13_get_rej(user_id=user_id) or []
    except Exception as _rej_err:
        logger.debug(f"[P1-3/SYNTH-HISTORY] get_active_rejections falló: {_rej_err}")
        rejections = []
    for r in rejections[:30]:
        if not isinstance(r, dict):
            continue
        name = r.get("meal_name")
        if name and name not in rejected_meals:
            rejected_meals.append(str(name))

    try:
        recent_plans = _p13_get_plans(user_id, days=180) or []
    except Exception as _plans_err:
        logger.debug(f"[P1-3/SYNTH-HISTORY] get_recent_plans falló: {_plans_err}")
        recent_plans = []
    for plan in recent_plans[:5]:
        if not isinstance(plan, dict):
            continue
        days = plan.get("days") if isinstance(plan.get("days"), list) else []
        for day in days:
            if not isinstance(day, dict):
                continue
            for meal in (day.get("meals") or [])[:6]:
                if not isinstance(meal, dict):
                    continue
                name = meal.get("name")
                if name and name not in seen_meal_names:
                    seen_meal_names.add(name)
                    repeated_meal_names.append(str(name))
                for ing in (meal.get("ingredients") or [])[:8]:
                    if not ing:
                        continue
                    base = _p13_norm(str(ing))
                    if base and len(base) > 2:
                        repeated_bases_set.add(base)

    if not rejected_meals and not repeated_meal_names and not repeated_bases_set:
        return None

    return {
        "total_rejection_violations": 0,  # No "violations" reales — son preferencias.
        "total_allergy_violations": 0,
        "top_rejection_hits": rejected_meals[:20],
        "top_repeated_bases": list(repeated_bases_set)[:20],
        "top_repeated_meal_names": repeated_meal_names[:15],
        "_lifetime_window_days": 180,
        "synthesized_from_user_history": True,
    }


def _cas_update_chunk_status(
    task_id,
    expected_attempts: int,
    new_status: str,
    expected_status: str = "processing",
    extra_set_clauses: dict | None = None,
) -> bool:
    """[P1-4] Actualiza `plan_chunk_queue.status` con guard CAS contra `attempts`.

    Antes, varios paths terminales (pantry_violation_failed, last-resort failed,
    user_lock_pending) hacían `UPDATE … WHERE id = %s` sin verificar que el chunk
    seguía siendo "nuestro". Si el zombie rescue (line 9070) había incrementado
    `attempts` y un worker B ya había re-claim el chunk, nuestra UPDATE clobbearía
    el estado del worker B (e.g., marcando 'failed' un chunk que B está procesando
    activamente).

    Este helper aplica el patrón CAS consistente:
        UPDATE plan_chunk_queue
        SET status = <new>, updated_at = NOW(), <extra...>
        WHERE id = %s
          AND attempts = %s          -- token CAS desde pickup
          AND status = %s             -- guard de transición válida

    Si el guard falla (rowcount=0), el caller sabe que fue desplazado por zombie
    rescue y debe abortar limpiamente sin clobbear el estado de otro worker.

    Args:
        task_id: id del chunk row.
        expected_attempts: el `attempts` capturado al pickup (`_pickup_attempts`).
        new_status: nuevo status target (e.g., 'failed', 'pending', 'cancelled').
        expected_status: status que esperamos encontrar (default 'processing').
        extra_set_clauses: dict de columnas adicionales a actualizar
            (e.g., {"escalated_at": "NOW()"} se agrega como SET escalated_at = NOW()).
            Los valores son SQL crudo — solo usar en este módulo, no expuestos al usuario.

    Returns:
        True si la UPDATE afectó 1 fila (operación exitosa).
        False si afectó 0 filas (CAS falló — el chunk fue desplazado o ya está en otro estado).
    """
    set_parts = ["status = %s", "updated_at = NOW()"]
    params: list = [new_status]
    if extra_set_clauses:
        for col, sql_expr in extra_set_clauses.items():
            set_parts.append(f"{col} = {sql_expr}")
    sql = (
        "UPDATE plan_chunk_queue "
        f"SET {', '.join(set_parts)} "
        "WHERE id = %s AND attempts = %s AND status = %s "
        "RETURNING id"
    )
    params.extend([task_id, int(expected_attempts), expected_status])
    try:
        result = execute_sql_write(sql, tuple(params), returning=True)
    except Exception as e:
        logger.error(
            f"[P1-4/CAS] UPDATE falló para chunk {task_id}: {type(e).__name__}: {e}. "
            f"Tratando como CAS-failed para no clobbear estado."
        )
        return False
    affected = bool(result)
    if not affected:
        logger.warning(
            f"[P1-4/CAS-DISPLACED] Chunk {task_id} no fue actualizado a status={new_status!r}: "
            f"esperaba attempts={expected_attempts} status={expected_status!r}, pero el row "
            f"cambió (probablemente zombie rescue + nuevo pickup). Aborto limpio sin clobbear."
        )
    return affected


# [P1-6] Razones canónicas registradas cuando _rebuild_last_chunk_learning_from_queue
# detecta corrupción de datos de aprendizaje. Mantenidas como constantes para que
# `/admin/metrics` y tests puedan filtrarlas sin string-matching arbitrario.
_P1_6_LEARNING_LOSS_REASONS = (
    "select_failed",       # Excepción en el SELECT contra plan_chunk_queue (DB blip).
    "json_corrupted",      # learning_metrics presente pero no parseable como JSON dict.
    "schema_invalid",      # JSON parseable pero falla _validate_lesson_schema (NaN/inf, type mismatch).
)


def _record_learning_loss(
    meal_plan_id: str,
    week_number: int,
    reason: str,
    user_id: str | None = None,
) -> None:
    """[P1-6] Registra pérdida de learning durante el rebuild del chunk previo.

    Antes el fallo del rebuild devolvía None silentemente y el chunk siguiente
    arrancaba con learning vacío sin que el sistema lo supiera. El usuario veía
    su plan generándose sin que las lecciones del chunk previo se aplicaran y
    no había forma de detectar el patrón en operaciones (ej.: tabla
    plan_chunk_queue corrupta, schema regresivo, JSON corrupto por psycopg
    encoding).

    Esta función:
      1. Persiste un evento en `plan_chunk_queue.metadata` vía la tabla
         `chunk_lesson_telemetry` para que `/admin/metrics` y crons de alerta
         puedan agregarlos cross-window.
      2. Persiste `_learning_corrupted_chunks` en `meal_plans.plan_data` (cap a
         50 entries) vía `update_plan_data_atomic` para que el frontend pueda
         renderizar un warning operacional al usuario si quisiera, y para que
         el siguiente chunk del mismo plan sepa que su predecesor tuvo problemas.

    Best-effort: cualquier excepción se suprime con log warning. La telemetría
    no debe bloquear el flujo del worker.

    Args:
        meal_plan_id: UUID del plan afectado.
        week_number: chunk cuyo rebuild falló.
        reason: ver `_P1_6_LEARNING_LOSS_REASONS` — categoría que el cron de
            alerta puede filtrar para distinguir blips transitorios (select_failed)
            de corrupción real (json_corrupted, schema_invalid).
        user_id: opcional, requerido para telemetría cross-user. Si None, la
            persistencia a plan_data sí ocurre (sólo necesita meal_plan_id) pero
            la telemetría a chunk_lesson_telemetry se skipea.
    """
    if reason not in _P1_6_LEARNING_LOSS_REASONS:
        logger.warning(
            f"[P1-6/LEARNING-LOSS] Razón desconocida {reason!r}; usando "
            f"'unknown' para telemetría. Razones válidas: {_P1_6_LEARNING_LOSS_REASONS}"
        )

    # 1. Persistir flag en plan_data atómicamente. Si la fila no existe (plan
    # cancelado entre el rebuild y este punto), update_plan_data_atomic retorna
    # {} y no hace UPDATE — perfecto para no agregar telemetría a un plan muerto.
    try:
        from db_plans import update_plan_data_atomic

        def _mutator(plan_data):
            losses = plan_data.get("_learning_corrupted_chunks", [])
            if not isinstance(losses, list):
                losses = []
            entry = {
                "chunk": int(week_number),
                "reason": str(reason),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            losses.append(entry)
            # Cap FIFO a 50 entries: planes patológicos podrían acumular cientos
            # de fallos consecutivos; sin cap el JSON crecería sin límite.
            plan_data["_learning_corrupted_chunks"] = losses[-50:]
            return plan_data

        update_plan_data_atomic(str(meal_plan_id), _mutator)
    except Exception as _persist_err:
        logger.warning(
            f"[P1-6/LEARNING-LOSS] No se pudo persistir _learning_corrupted_chunks "
            f"para plan {meal_plan_id} chunk {week_number} reason={reason}: "
            f"{type(_persist_err).__name__}: {_persist_err}"
        )

    # 2. Telemetría cross-window: chunk_lesson_telemetry. Sin user_id no podemos
    # cumplir el schema (NOT NULL) — skipeamos pero el persist a plan_data ya
    # ocurrió, así que la señal NO se pierde por completo.
    if user_id:
        try:
            _record_chunk_lesson_telemetry(
                user_id=str(user_id),
                meal_plan_id=str(meal_plan_id),
                week_number=int(week_number),
                event="learning_rebuild_failed",
                synthesized_count=0,
                queue_count=0,
                metadata={"reason": str(reason)},
            )
        except Exception as _tele_err:
            logger.debug(
                f"[P1-6/LEARNING-LOSS] Telemetría chunk_lesson_telemetry falló: {_tele_err}"
            )

    logger.error(
        f"[P1-6/LEARNING-LOSS] plan={meal_plan_id} chunk={week_number} reason={reason}. "
        f"_last_chunk_learning rebuild devolvió None; el chunk siguiente arrancará "
        f"con stub o síntesis desde plan_data.days."
    )


def _rebuild_last_chunk_learning_from_queue(
    meal_plan_id: str,
    target_week: int,
    prefer_completed: bool = False,
    user_id: str | None = None,
) -> dict | None:
    """[P0-3] Reconstruye `_last_chunk_learning` desde `plan_chunk_queue.learning_metrics`.

    Cuando el seed síncrono de chunk 1 falla o la persistencia post-chunk N-1 no escribe en
    `meal_plans.plan_data._last_chunk_learning` (timeout, lock, JSON corrupto), el chunk N
    arranca con dict vacío y todas las "lecciones" quedan en stub → cadena de aprendizaje rota.

    Esta función lee la fila completed de `plan_chunk_queue` para `target_week` (que SÍ tiene
    su `learning_metrics` columna persistida atómicamente con el commit del chunk) y construye
    el dict en el formato esperado por el worker. Devuelve None si no hay datos utilizables.

    Args:
        meal_plan_id, target_week: identifican el chunk a reconstruir.
        prefer_completed: [P1-2] Si True, descarta filas `status='failed'` aunque tengan
            learning_metrics no-NULL — el caller cae al fallback de síntesis desde
            plan_data.days, que es más truthful que las metrics de un chunk donde el
            pipeline crasheó o el commit falló a medio escribir. Default False preserva
            el comportamiento P0-2 (aceptar failed con low_confidence) para callsites
            de introspección/legacy.
    """
    try:
        # [P0-2] Aceptar también filas con status='failed' SI tienen learning_metrics no-NULL.
        # Casos cubiertos:
        #   - completed con learning_metrics full (caso ideal, alta confianza).
        #   - failed con learning_metrics["pipeline_failed"]=True (post-pipeline pero pre-commit
        #     se persistió desde el except handler) → marcamos low_confidence.
        #   - failed con learning_metrics["preflight"]=True (pipeline crasheó antes de calcular
        #     metrics; sólo tenemos contadores prior-only) → marcamos low_confidence y
        #     metrics_unavailable=True para que el LLM sepa que las violations no son fiables.
        # Orden por status: 'completed' antes que 'failed' para preferir éxito si ambos coexisten
        # (escenario raro: chunk completed con un retry posterior failed por causas externas).
        # [P1-2] prefer_completed=True restringe a 'completed' para evitar contaminación.
        if prefer_completed:
            _status_clause = "status = 'completed'"
        else:
            _status_clause = "status IN ('completed', 'failed')"
        row = execute_sql_query(
            f"""
            SELECT week_number, status, learning_metrics
            FROM plan_chunk_queue
            WHERE meal_plan_id = %s
              AND week_number = %s
              AND {_status_clause}
              AND learning_metrics IS NOT NULL
            ORDER BY CASE WHEN status = 'completed' THEN 0 ELSE 1 END, updated_at DESC
            LIMIT 1
            """,
            (str(meal_plan_id), int(target_week)),
            fetch_one=True,
        )
    except Exception as e:
        logger.warning(
            f"[P0-3/REBUILD-LAST] Fallo SELECT learning_metrics para plan {meal_plan_id} "
            f"week {target_week}: {e}"
        )
        # [P1-6] DB blip transitorio: registramos el evento para que ops detecte
        # patrones (e.g. permisos perdidos, tabla missing tras schema downgrade).
        _record_learning_loss(meal_plan_id, target_week, "select_failed", user_id=user_id)
        return None

    if not row:
        # [P1-6] No es learning loss: el chunk previo simplemente NO tiene fila
        # completed/failed con learning_metrics aún (e.g. chunk 1 todavía
        # processing, chunk previo cancelado). El caller cae a síntesis desde
        # plan_data.days, que es comportamiento NORMAL — no telemetramos.
        return None

    lm = row.get("learning_metrics")
    if isinstance(lm, str):
        try:
            lm = json.loads(lm)
        except Exception:
            lm = None
    if not isinstance(lm, dict):
        # [P1-6] El learning_metrics existía como string en DB pero no se pudo
        # parsear como JSON dict — corrupción real (psycopg encoding bug,
        # schema regresivo a TEXT, etc.). Diferente del path "select_failed":
        # aquí la query SÍ funcionó pero el dato es inválido.
        _record_learning_loss(meal_plan_id, target_week, "json_corrupted", user_id=user_id)
        return None

    is_proxy = bool(
        lm.get("inventory_activity_proxy_used") or lm.get("sparse_logging_proxy_used")
    )
    # [P0-2] Origen del learning_metrics — afecta confianza y semántica de violations.
    is_preflight = bool(lm.get("preflight"))
    pipeline_failed = bool(lm.get("pipeline_failed"))
    row_status = row.get("status") or "completed"
    low_confidence = is_proxy or is_preflight or pipeline_failed or row_status != "completed"
    if is_preflight:
        signal_strength = "preflight_only"
    elif pipeline_failed:
        signal_strength = "weak"
    else:
        signal_strength = lm.get("learning_signal_strength", "weak" if is_proxy else "strong")

    candidate = {
        "repeat_pct": lm.get("learning_repeat_pct", 0),
        "ingredient_base_repeat_pct": lm.get("ingredient_base_repeat_pct", 0),
        "rejection_violations": lm.get("rejection_violations", 0),
        "allergy_violations": lm.get("allergy_violations", 0),
        "fatigued_violations": lm.get("fatigued_violations", 0),
        "repeated_bases": lm.get("sample_repeated_bases", []),
        "repeated_meal_names": lm.get("sample_repeats", []),
        "rejected_meals_that_reappeared": lm.get("sample_rejection_hits", []),
        "allergy_hits": lm.get("sample_allergy_hits", []),
        "chunk": int(row.get("week_number") or target_week),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # [P0-2] metrics_unavailable=True cuando solo tenemos preflight (no se ejecutó
        # _calculate_learning_metrics con new_days reales). El LLM debe interpretar
        # los counters como "información ausente" en lugar de "0 violaciones".
        "metrics_unavailable": is_preflight,
        "low_confidence": low_confidence,
        "learning_signal_strength": signal_strength,
        "rebuilt_from_queue": True,
        "rebuilt_source_status": row_status,
        "rebuilt_from_preflight": is_preflight,
        "rebuilt_from_pipeline_failure": pipeline_failed,
    }
    # [P1-1] Validar schema antes de devolver. Un campo numérico NaN/inf o un sample
    # con tipo incorrecto rompería al LLM downstream. Mejor devolver None y dejar al
    # caller usar el stub que persistir basura como `_last_chunk_learning`.
    _ok, _reason = _validate_lesson_schema(candidate)
    if not _ok:
        logger.error(
            f"[P1-1/SCHEMA-INVALID] _last_chunk_learning reconstruido para plan {meal_plan_id} "
            f"chunk {target_week} descartado: {_reason}. learning_metrics raw keys: "
            f"{sorted(list(lm.keys()))[:10]}"
        )
        # [P1-6] Schema-invalid es la corrupción más severa: significa que el
        # learning_metrics persistido por el chunk previo tiene tipos/valores
        # inválidos que propagarían al prompt. Diferente de "json_corrupted":
        # aquí el JSON parseó bien pero el SHAPE del dict es incorrecto (NaN,
        # inf, list cuando esperaba str, etc.). Si este reason crece, hay un
        # bug en el writer de learning_metrics (no en el reader).
        _record_learning_loss(meal_plan_id, target_week, "schema_invalid", user_id=user_id)
        return None
    return candidate


def _prune_critical_lessons_with_priority(lessons: list, max_size: int) -> list:
    """[P0-6] Poda priorizada del store de lecciones críticas permanentes.

    Antes la poda era LRU duro (`lessons[-N:]`), lo que descartaba silenciosamente alergias
    antiguas si el usuario acumulaba 50+ lecciones. Ahora:
      - Lecciones inmortales (alergias > 0, o rejection_violations >= umbral) NUNCA se descartan.
      - Lecciones no-críticas (fatiga/repeats puros) se rotan LRU para respetar el cap.
    Si el cap se excede solo por inmortales, se devuelven todas y se loggea warning.
    """
    if not isinstance(lessons, list) or len(lessons) <= max_size:
        return lessons

    from constants import CHUNK_CRITICAL_REPEATED_REJECTION_IMMORTAL as _IMMORTAL_REJ

    def _is_immortal(lesson: dict) -> bool:
        if not isinstance(lesson, dict):
            return False
        if int(lesson.get('allergy_violations') or 0) > 0:
            return True
        if int(lesson.get('rejection_violations') or 0) >= _IMMORTAL_REJ:
            return True
        return False

    immortals = [l for l in lessons if _is_immortal(l)]
    mortals = [l for l in lessons if not _is_immortal(l)]

    # [P0-6] Hard cap de inmortales para evitar bloat infinito.
    # Antes: si len(immortals) >= max_size se devolvían TODAS, sin tope, causando memory
    # bloat + prompt overflow para usuarios con 200+ alergias/rechazos repetidos.
    # Ahora: si los inmortales superan IMMORTAL_HARD_CAP, hacemos LRU sobre los inmortales
    # MÁS VIEJOS sin re-validación reciente (last_validated_at/created_at fuera de
    # CHUNK_CRITICAL_LESSONS_REVALIDATION_DAYS).
    from constants import (
        CHUNK_CRITICAL_LESSONS_IMMORTAL_HARD_CAP as _IMMORTAL_CAP,
        CHUNK_CRITICAL_LESSONS_REVALIDATION_DAYS as _REVALIDATION_DAYS,
    )

    def _last_seen_ts(lesson: dict) -> float:
        """Timestamp más reciente disponible para una lección, en epoch seconds."""
        for fld in ("last_validated_at", "last_seen_at", "updated_at", "created_at", "timestamp"):
            val = lesson.get(fld) if isinstance(lesson, dict) else None
            if not val:
                continue
            try:
                if isinstance(val, (int, float)):
                    return float(val)
                from constants import safe_fromisoformat
                dt = safe_fromisoformat(str(val))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except Exception:
                continue
        return 0.0  # sin timestamp → tratada como muy vieja, candidata a poda primero

    if len(immortals) > _IMMORTAL_CAP:
        cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=_REVALIDATION_DAYS)).timestamp()
        recent = [l for l in immortals if _last_seen_ts(l) >= cutoff_ts]
        stale = [l for l in immortals if _last_seen_ts(l) < cutoff_ts]
        # Si las recientes ya saturan el cap, pisamos todas las stale.
        if len(recent) >= _IMMORTAL_CAP:
            evicted_stale = len(stale)
            logger.warning(
                f"[P0-6/CRITICAL-LESSONS] Hard cap de inmortales ({_IMMORTAL_CAP}) excedido. "
                f"{evicted_stale} inmortales sin re-validación >{_REVALIDATION_DAYS}d descartadas; "
                f"{len(recent)} recientes preservadas. Considera revisar el perfil del usuario."
            )
            immortals = recent
        else:
            # Conservamos todas las recientes + las stale más nuevas hasta llenar el cap.
            keep_stale = _IMMORTAL_CAP - len(recent)
            stale_sorted = sorted(stale, key=_last_seen_ts)  # más viejas primero
            evicted_stale = max(0, len(stale) - keep_stale)
            if evicted_stale > 0:
                logger.warning(
                    f"[P0-6/CRITICAL-LESSONS] Hard cap de inmortales ({_IMMORTAL_CAP}) excedido. "
                    f"Descartando {evicted_stale} inmortales más viejas sin re-validación reciente "
                    f"(>{_REVALIDATION_DAYS}d). Conservando {len(recent)} recientes + {keep_stale} stale recientes."
                )
            immortals = recent + stale_sorted[-keep_stale:] if keep_stale > 0 else recent

    if len(immortals) >= max_size:
        logger.warning(
            f"[P0-6/CRITICAL-LESSONS] Cap ({max_size}) excedido solo por lecciones inmortales "
            f"({len(immortals)} alergias/rechazos repetidos). Conservando todas las inmortales y "
            f"descartando las {len(mortals)} no-críticas. Considera subir CHUNK_CRITICAL_LESSONS_MAX."
        )
        return immortals

    keep_mortals = max_size - len(immortals)
    evicted = max(0, len(mortals) - keep_mortals)
    if evicted > 0:
        logger.info(
            f"[P0-6/CRITICAL-LESSONS] Poda priorizada: {len(immortals)} inmortales preservadas, "
            f"{evicted} no-críticas más antiguas descartadas (cap={max_size})."
        )
    return immortals + mortals[-keep_mortals:]


def _get_dominant_technique(days: list) -> str:
    """Extrae la técnica más frecuente (moda) de una lista de días.
    
    [P1-E] Útil en Smart Shuffle para que 'last_technique' no se congele 
    y refleje el contenido real siendo servido.
    """
    if not days:
        return None
    
    from collections import Counter
    techs = []
    for d in days:
        if not isinstance(d, dict): continue
        # Priorizar _technique (interno) sobre technique (UI)
        t = d.get("_technique") or d.get("technique")
        if t:
            techs.append(t)
            
    if not techs:
        return None
        
    # Retornar la técnica más común
    return Counter(techs).most_common(1)[0][0]


def _build_zero_log_push_payload(
    consecutive_zero_log_chunks: int,
    logging_preference: str = "manual",
) -> dict:
    """[P1-4] Construye el payload del push de zero-log pause según preference.

    Antes el push solo invitaba a loguear comidas y deeplinkaba a /dashboard.
    El usuario que no quería/podía loguear quedaba esperando 4-24h sin enterarse
    de la opción `auto_proxy` (PUT /api/diary/preferences/logging) ya disponible
    en el banner del frontend (vía /api/blocked-reasons).

    Ahora si `logging_preference == 'manual'`, el body menciona explícitamente
    "Continuar sin registrar" como CTA secundario y el url apunta a
    `CHUNK_ZERO_LOG_DEEPLINK` (default `/diario?banner=zero_log`) donde el
    banner muestra el toggle. Si el usuario ya optó por auto_proxy, el CTA
    no se incluye y el url vuelve a `/dashboard`.

    Args:
        consecutive_zero_log_chunks: Contador de chunks consecutivos sin logs.
            ≥3 cambia el título a "Tu plan se está generando sin tu feedback".
        logging_preference: 'manual' (default) o 'auto_proxy'.

    Returns:
        dict con keys: title, body, url. Se pasa directamente a
        _dispatch_push_notification(**payload).
    """
    from constants import CHUNK_ZERO_LOG_DEEPLINK as _DL
    offer_optout = (logging_preference == "manual")

    if int(consecutive_zero_log_chunks or 0) >= 3:
        title = "Tu plan se está generando sin tu feedback"
        body = (
            "Llevas varios bloques sin registrar comidas. "
            "Loguea en el diario para que los siguientes se ajusten a ti"
        )
        if offer_optout:
            body += (
                ", o elige 'Continuar sin registrar' en el banner para que generemos "
                "los próximos días con tu nevera actual."
            )
        else:
            body += "."
    else:
        title = "Loguea tus comidas para continuar"
        body = "Tu siguiente bloque está en pausa porque no tenemos registro de tus comidas. "
        if offer_optout:
            body += (
                "Abre el diario para loguear, o tap 'Continuar sin registrar' "
                "para que generemos los próximos días con tu nevera actual."
            )
        else:
            body += (
                "Abre el diario y loguea lo que hayas comido para que aprenda de ti."
            )

    return {
        "title": title,
        "body": body,
        "url": _DL if offer_optout else "/dashboard",
    }


def _touch_chunk_heartbeat(task_id) -> bool:
    """[P0-1] Refresca `chunk_user_locks.heartbeat_at` para el chunk indicado de forma
    síncrona desde el flujo principal del worker.

    El thread daemon ya hace updates periódicos cada `CHUNK_LOCK_HEARTBEAT_INTERVAL_SECONDS`,
    pero un thread daemon puede morir o quedarse esperando si la DB tiene un blip transitorio
    justo cuando un cycle de wait termina. Este helper se llama en puntos críticos del flujo
    principal (e.g., antes del merge transaccional largo, antes de `cursor.execute` del LLM
    pipeline) para garantizar que el heartbeat siga vivo aunque el thread esté en problemas.

    Returns True si la UPDATE fue OK, False si falló (silencioso — no debe abortar el chunk).
    """
    if task_id is None:
        return False
    try:
        execute_sql_write(
            "UPDATE chunk_user_locks SET heartbeat_at = NOW() WHERE locked_by_chunk_id = %s",
            (task_id,),
        )
        return True
    except Exception as _err:
        # Log debug — el thread daemon ya reporta fallos sistémicos, evitamos doble noise.
        logger.debug(f"[P0-1/HEARTBEAT-INLINE] Touch fallido para chunk {task_id}: {_err}")
        return False


class _PantryViolationPostMerge(Exception):
    """[P0-4] Lanzada cuando el merge de un chunk produce días con ingredientes no
    cubiertos por la nevera del usuario.

    El caller (worker outer catch) la distingue de excepciones genéricas para pausar
    el chunk con `pending_user_action` y `_pause_reason='pantry_violation_post_merge'`
    en lugar de marcarlo `failed` y reintentar — un retry no resuelve el problema
    si la nevera no cambia, solo agota retries hasta dead_letter.
    """

    def __init__(self, violations: list, pantry_size: int = 0):
        self.violations = list(violations or [])
        self.pantry_size = int(pantry_size or 0)
        super().__init__(
            f"Pantry violation post-merge: {len(self.violations)} día(s) con ingredientes no cubiertos"
        )


def _validate_merged_days_against_pantry(
    merged_days: list,
    pantry_ingredients: list,
    new_chunk_day_range: tuple | None = None,
) -> tuple:
    """[P0-4] Hard validation post-merge sobre cada día.

    Aplica `validate_ingredients_against_pantry` con `strict_quantities=True` y
    `tolerance=1.0` (sin tolerancia adicional sobre el ledger de la nevera).
    Los condimentos básicos (sal, aceite, ajo, etc.) ya quedan exentos dentro de
    `validate_ingredients_against_pantry`, así que esto endurece solo los
    ingredientes "core" (proteínas, carbohidratos, vegetales) sin penalizar
    condimentos faltantes.

    Args:
        merged_days: Lista de días post-merge.
        pantry_ingredients: Inventario de la nevera del usuario.
        new_chunk_day_range: Tupla (start_day, end_day) inclusivo. Si se pasa,
            solo valida los días dentro del rango (típicamente los del chunk
            recién mergeado). Si None, valida todos los días.

    Returns:
        (ok, violations) — `ok=True` si todos los días en rango pasan;
        `violations` es lista de dicts {day, error} (truncado a 500 chars).
    """
    from constants import validate_ingredients_against_pantry as _vip

    if not pantry_ingredients:
        return True, []

    violations = []
    for d in merged_days or []:
        if not isinstance(d, dict):
            continue
        try:
            day_num = int(d.get("day") or 0)
        except (TypeError, ValueError):
            day_num = 0

        if new_chunk_day_range is not None:
            try:
                _start, _end = int(new_chunk_day_range[0]), int(new_chunk_day_range[1])
            except (TypeError, ValueError, IndexError):
                _start, _end = 0, 0
            if not (_start <= day_num <= _end):
                continue

        ingredients = []
        for m in d.get("meals", []) or []:
            if not isinstance(m, dict):
                continue
            for ing in m.get("ingredients", []) or []:
                if isinstance(ing, str) and ing.strip():
                    ingredients.append(ing)

        if not ingredients:
            continue

        result = _vip(ingredients, pantry_ingredients, strict_quantities=True, tolerance=1.0)
        if result is not True:
            violations.append({"day": day_num, "error": str(result)[:500]})

    return (len(violations) == 0), violations


def _filter_days_by_fresh_pantry(days: list, pantry_ingredients: list, min_match_ratio: float = 0.9) -> list:
    """Keep Smart Shuffle days whose base ingredients are mostly covered by the live pantry."""
    if not days:
        return []

    pantry_ingredients = pantry_ingredients or []
    if not pantry_ingredients:
        return list(days)

    from constants import normalize_ingredient_for_tracking

    pantry_bases = set()
    for item in pantry_ingredients:
        if not item:
            continue
        normalized = normalize_ingredient_for_tracking(item)
        if normalized:
            pantry_bases.add(normalized)
        pantry_bases.add(strip_accents(str(item).lower().strip()))

    if not pantry_bases:
        return list(days)

    filtered_days = []
    for day in days:
        if not isinstance(day, dict):
            continue

        unique_bases = []
        seen = set()
        for meal in day.get("meals", []) or []:
            if not isinstance(meal, dict):
                continue
            for raw_ing in meal.get("ingredients", []) or []:
                if not raw_ing:
                    continue
                base = normalize_ingredient_for_tracking(raw_ing) or strip_accents(str(raw_ing).lower().strip())
                if base and base not in seen:
                    seen.add(base)
                    unique_bases.append(base)

        if not unique_bases:
            filtered_days.append(day)
            continue

        matched = 0
        for base in unique_bases:
            if base in pantry_bases or any(pb and len(pb) > 2 and (pb in base or base in pb) for pb in pantry_bases):
                matched += 1

        coverage = matched / max(len(unique_bases), 1)
        if coverage >= min_match_ratio:
            filtered_days.append(day)

    return filtered_days


def _count_meaningful_pantry_items(pantry_ingredients: list) -> int:
    """Counts distinct, meaningful pantry bases to detect an empty or near-empty fridge."""
    pantry_ingredients = pantry_ingredients or []
    if not pantry_ingredients:
        return 0

    from constants import normalize_ingredient_for_tracking

    ignored_terms = {
        "", "agua", "sal", "pimienta", "aceite", "vinagre", "oregano", "cilantro",
        "canela", "sazon", "condimento",
    }

    normalized_items = set()
    for item in pantry_ingredients:
        if not item:
            continue
        normalized = normalize_ingredient_for_tracking(item) or strip_accents(str(item).lower().strip())
        if normalized and normalized not in ignored_terms and len(normalized) > 2:
            normalized_items.add(normalized)

    return len(normalized_items)


def compute_pantry_degraded_summary(plan_data: dict) -> dict:
    """[P0-2] Resumen del estado de degradación de pantry para exponer al frontend.

    Inspecciona `plan_data` y agrega:
      - `_initial_chunk_pantry_degraded` (P0-1: primer chunk síncrono que agotó retries
         de validación contra la nevera).
      - `plan_data.days[i]._pantry_degraded` (P0-2: chunks 2..N generados en
         flexible_mode por live degraded / snapshot stale / nevera vacía).
      - `plan_data._current_mode == "flexible"` (señal a nivel plan persistida por
         `_activate_flexible_mode`).

    El frontend consume el dict para mostrar:
      - Banner global en el header del plan ("Plan generado con datos parciales — revisa").
      - Badge per-día en cada day card que tiene `_pantry_degraded=True`.
      - Header HTTP `X-Pantry-Degraded: true` cuando `degraded == True`.

    Returns:
        dict con shape:
            {
              "degraded": bool,
              "degraded_days": [int, ...],   # día 1..N (1-indexed según plan_data.days[i].day)
              "reasons": [str, ...],         # reasons únicos, ordenados.
              "initial_chunk_degraded": bool,
              "current_mode": str | None,    # "flexible" si plan-level activo, else None.
            }
    """
    summary = {
        "degraded": False,
        "degraded_days": [],
        "reasons": [],
        "initial_chunk_degraded": False,
        "current_mode": None,
    }
    if not isinstance(plan_data, dict):
        return summary

    initial_flag = bool(plan_data.get("_initial_chunk_pantry_degraded"))
    summary["initial_chunk_degraded"] = initial_flag

    current_mode = plan_data.get("_current_mode")
    if isinstance(current_mode, str) and current_mode:
        summary["current_mode"] = current_mode

    reason_set = set()
    days_out: list = []

    if initial_flag:
        # P0-1 marca el chunk inicial completo (días 1..PLAN_CHUNK_SIZE) como degraded.
        from constants import PLAN_CHUNK_SIZE as _PCS_FOR_SUMMARY
        for i in range(1, int(_PCS_FOR_SUMMARY) + 1):
            days_out.append(i)
        # Reason explícito si lo persistió P0-1; si no, default genérico.
        violation = plan_data.get("_initial_chunk_pantry_violation")
        if isinstance(violation, str) and violation.strip():
            reason_set.add("initial_chunk_validation_failed")
        else:
            reason_set.add("initial_chunk_validation_failed")

    for d in plan_data.get("days") or []:
        if not isinstance(d, dict):
            continue
        if not d.get("_pantry_degraded"):
            continue
        try:
            day_num = int(d.get("day") or 0)
        except (TypeError, ValueError):
            day_num = 0
        if day_num and day_num not in days_out:
            days_out.append(day_num)
        r = d.get("_pantry_degraded_reason")
        if isinstance(r, str) and r.strip():
            reason_set.add(r.strip())

    days_out.sort()
    summary["degraded_days"] = days_out
    summary["reasons"] = sorted(reason_set)
    summary["degraded"] = bool(days_out) or current_mode == "flexible" or initial_flag
    return summary


def _mark_meals_violating_pantry(result: dict, pantry_ingredients: list) -> int:
    """[P0-5] Marca per-comida `_pantry_violated=True` + `_pantry_violated_reason`
    en `result['days'][i]['meals'][j]` para cada comida cuyos ingredientes NO
    estén en pantry.

    Contexto: el flujo síncrono de chunk 1 (routers/plans.py:849) NO pausa
    cuando los retries pantry se agotan — sirve un plan degradado con el flag
    plan-level `_initial_chunk_pantry_degraded=True`. Sin granularidad, el
    frontend solo sabe que "el plan en general tiene problemas", no qué
    platos específicos son incocibles. Este helper agrega el contrato per-meal:
    cada comida que viola pantry queda marcada para que el frontend pueda
    mostrar un banner rojo en ESE plato (alineando UX con la promesa
    "platos solo con alimentos de la nevera").

    Asimetría que cierra:
      - Chunks 2+ (worker, async): pausa a pending_user_action si retries
        se agotan → contrato cumplido por refusal-to-serve.
      - Chunk 1 (sync, antes de este fix): servía con flag plan-level →
        contrato cumplido nominalmente, pero el usuario veía platos
        imposibles sin saber cuáles.
      - Chunk 1 (después): sirve con flag plan-level + flag per-meal →
        usuario ve platos imposibles MARCADOS, alineado con UX.

    Mutación in-place del dict `result`. Retorna número de comidas marcadas.

    Args:
        result: dict con `result['days'][i]['meals'][j]` (el formato del
            pipeline LLM).
        pantry_ingredients: inventario fresco del usuario (lista de strings).

    Returns:
        Número de comidas marcadas con `_pantry_violated=True`. 0 si no hay
        violaciones, no hay pantry, o `result` no tiene la estructura esperada.
    """
    if not isinstance(result, dict) or not pantry_ingredients:
        return 0
    from constants import validate_ingredients_against_pantry as _vip

    marked = 0
    for d in result.get("days") or []:
        if not isinstance(d, dict):
            continue
        for m in d.get("meals") or []:
            if not isinstance(m, dict):
                continue
            ings = [
                i for i in (m.get("ingredients") or [])
                if isinstance(i, str) and i.strip()
            ]
            if not ings:
                continue
            check = _vip(ings, pantry_ingredients, strict_quantities=False)
            if check is True:
                continue
            m["_pantry_violated"] = True
            m["_pantry_violated_reason"] = str(check)[:300]
            marked += 1
    return marked


def _validate_and_retry_initial_chunk_against_pantry(
    *,
    pipeline_data: dict,
    history: list,
    taste_profile: str,
    memory_context: str,
    background_tasks,
    pantry_ingredients: list,
    initial_result: dict,
    user_id: str | None = None,
) -> tuple[dict, dict]:
    """[P0-1] Hard guardrail post-LLM para el primer chunk del flujo síncrono /api/plans/analyze.

    Antes solo el worker (chunks 2..N) validaba que los ingredientes generados estuvieran
    en la nevera. El chunk 1 se persistía vía save_partial_plan_get_id (services.py:108)
    sin filtro, rompiendo la promesa "platos solo con alimentos de la nevera" en el primer
    plato que el usuario veía.

    Estrategia (mimica del worker, lineas ~13050-13380, sin pause path):
      1) Verifica existencia (strict_quantities=False).
      2) Si pasa y modo es 'hybrid'/'strict', verifica cantidades con tolerance del modo.
      3) Si falla, reintenta el pipeline LLM hasta CHUNK_PANTRY_MAX_RETRIES inyectando
         `_pantry_correction` en pipeline_data (mismo canal que usa el worker).
      4) Si tras reintentos persiste violación, NO bloquea — el usuario está esperando
         sincrónicamente. Devuelve el último result y un audit con degraded=True para que
         el caller persista el flag _initial_chunk_pantry_degraded en plan_data.

    Args:
        pipeline_data: dict de form_data ya preparado para el pipeline. La función mutará
            `_pantry_correction` entre reintentos (limpiado en éxito).
        history: lista de mensajes recientes (puede ser vacía).
        taste_profile: string del perfil de gustos resuelto por el agente.
        memory_context: string del contexto de memoria.
        background_tasks: objeto FastAPI BackgroundTasks (puede ser None).
        pantry_ingredients: inventario live del usuario (lista de strings).
        initial_result: resultado de la primera invocación a run_plan_pipeline.
        user_id: id del usuario (para logs).

    Returns:
        (final_result, audit) — final_result es el dict del plan a persistir; audit:
          {
            "validated_ok": bool,    # True si la última validación pasó sin violar.
            "attempts": int,         # 1..CHUNK_PANTRY_MAX_RETRIES + 1
            "degraded": bool,        # True si retries agotados con violación pendiente.
            "last_violation": str | None,  # truncado a 1000 chars.
            "mode": str,             # 'off' | 'advisory' | 'hybrid' | 'strict'
            "pantry_size": int,
          }
    """
    audit = {
        "validated_ok": False,
        "attempts": 1,
        "degraded": False,
        "last_violation": None,
        "mode": (CHUNK_PANTRY_QUANTITY_MODE or "advisory").lower(),
        "pantry_size": len(pantry_ingredients or []),
    }

    if not pantry_ingredients:
        audit["validated_ok"] = True
        return initial_result, audit
    if not isinstance(initial_result, dict):
        audit["validated_ok"] = True
        return initial_result, audit

    from constants import validate_ingredients_against_pantry as _vip

    qty_mode = audit["mode"]
    if qty_mode == "strict":
        tolerance = 1.00
    elif qty_mode in ("off", "advisory"):
        tolerance = 1.10
    else:
        tolerance = float(CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE)

    def _extract_ingredients(res: dict) -> list:
        if not isinstance(res, dict):
            return []
        out = []
        for d in res.get("days") or []:
            if not isinstance(d, dict):
                continue
            for m in d.get("meals") or []:
                if not isinstance(m, dict):
                    continue
                for ing in m.get("ingredients") or []:
                    if isinstance(ing, str) and ing.strip():
                        out.append(ing)
        return out

    max_attempts = max(0, int(CHUNK_PANTRY_MAX_RETRIES))
    current_result = initial_result

    for attempt in range(max_attempts + 1):
        audit["attempts"] = attempt + 1

        gen_ings = _extract_ingredients(current_result)
        if not gen_ings:
            # Plan vacío o pipeline falló: no es responsabilidad del guardrail.
            audit["validated_ok"] = True
            audit["last_violation"] = None
            pipeline_data.pop("_pantry_correction", None)
            return current_result, audit

        existence = _vip(gen_ings, pantry_ingredients, strict_quantities=False)
        if existence is not True:
            last_violation = str(existence)[:1000]
            audit["last_violation"] = last_violation

            if qty_mode == "off":
                # off: degradar sin reintentar (el sistema decidió no enforcar).
                logger.warning(
                    f"[P0-1/INITIAL-VALIDATION/OFF] user={user_id} violación de existencia "
                    f"(modo=off, no reintenta): {last_violation[:300]}"
                )
                audit["degraded"] = True
                # [P0-5] Marcar per-meal qué platos son incocibles para que el frontend
                # pueda renderizar warning específico en cada uno (no solo plan-level).
                audit["meals_marked_violated"] = _mark_meals_violating_pantry(
                    current_result, pantry_ingredients
                )
                return current_result, audit

            if attempt < max_attempts:
                logger.warning(
                    f"[P0-1/INITIAL-VALIDATION] user={user_id} reintento "
                    f"{attempt + 1}/{max_attempts} por ingredientes fuera de nevera: "
                    f"{last_violation[:300]}"
                )
                pipeline_data["_pantry_correction"] = last_violation
                try:
                    current_result = run_plan_pipeline(
                        pipeline_data,
                        history or [],
                        taste_profile,
                        memory_context=memory_context,
                        background_tasks=background_tasks,
                    )
                except Exception as _e:
                    logger.error(
                        f"[P0-1/INITIAL-VALIDATION] user={user_id} pipeline LLM falló en "
                        f"reintento {attempt + 1}: {_e}. Devolviendo último result válido."
                    )
                    audit["degraded"] = True
                    # [P0-5] current_result quedó con violaciones de existencia (la
                    # asignación arriba nunca se completó por el except). Marcar.
                    audit["meals_marked_violated"] = _mark_meals_violating_pantry(
                        current_result, pantry_ingredients
                    )
                    return current_result, audit
                continue

            logger.error(
                f"[P0-1/INITIAL-VALIDATION] user={user_id} retries agotados "
                f"({max_attempts}) por existencia. Última violación: {last_violation[:300]}"
            )
            audit["degraded"] = True
            # [P0-5] Retries agotados con existencia violada: marcar comidas
            # ofensoras para granularidad UX (alineación con chunks 2+ que pausan).
            audit["meals_marked_violated"] = _mark_meals_violating_pantry(
                current_result, pantry_ingredients
            )
            return current_result, audit

        # Existencia OK. En off/advisory aceptamos sin chequear cantidades.
        if qty_mode in ("off", "advisory"):
            audit["validated_ok"] = True
            audit["last_violation"] = None
            pipeline_data.pop("_pantry_correction", None)
            return current_result, audit

        # hybrid / strict: validar cantidades.
        qty_check = _vip(
            gen_ings, pantry_ingredients, strict_quantities=True, tolerance=tolerance
        )
        if qty_check is True:
            audit["validated_ok"] = True
            audit["last_violation"] = None
            pipeline_data.pop("_pantry_correction", None)
            return current_result, audit

        last_violation = str(qty_check)[:1000]
        audit["last_violation"] = last_violation

        if attempt < max_attempts:
            logger.warning(
                f"[P0-1/INITIAL-VALIDATION/{qty_mode.upper()}] user={user_id} reintento "
                f"{attempt + 1}/{max_attempts} por cantidades (tolerance={tolerance}): "
                f"{last_violation[:300]}"
            )
            pipeline_data["_pantry_correction"] = last_violation
            try:
                current_result = run_plan_pipeline(
                    pipeline_data,
                    history or [],
                    taste_profile,
                    memory_context=memory_context,
                    background_tasks=background_tasks,
                )
            except Exception as _e:
                logger.error(
                    f"[P0-1/INITIAL-VALIDATION] user={user_id} pipeline LLM falló en "
                    f"reintento de cantidades {attempt + 1}: {_e}."
                )
                audit["degraded"] = True
                return current_result, audit
            continue

        logger.error(
            f"[P0-1/INITIAL-VALIDATION/{qty_mode.upper()}] user={user_id} retries de "
            f"cantidades agotados. Última violación: {last_violation[:300]}"
        )
        audit["degraded"] = True
        return current_result, audit

    # Defensivo: en teoría inalcanzable.
    return current_result, audit


def _persist_fresh_pantry_to_chunks(
    task_id: str | int,
    meal_plan_id: str,
    fresh_inventory: list,
    user_id: str | None = None,
) -> None:
    """[P0-3] Propaga el inventario live recién leído al snapshot del chunk actual y de sus siblings.

    Sin esto, los siblings pending/stale conservan el snapshot capturado al crear el plan;
    si su live fetch falla, caen al fallback con datos de hace días. Aquí refrescamos esos
    snapshots para que el fallback sea, como mucho, tan viejo como el último chunk procesado
    con éxito y no la fecha de creación del plan.

    [P0-5] Si se pasa `user_id`, también sincroniza tz_offset_minutes del user_profile vivo
    al snapshot — así el chunk dispara/valida en la zona horaria actual del usuario aunque
    haya viajado entre la creación del plan y este punto. Sin user_id mantiene comportamiento
    legacy (solo refresca pantry).
    """
    if fresh_inventory is None:
        return
    captured_at = datetime.now(timezone.utc).isoformat()
    pantry_json = json.dumps(list(fresh_inventory), ensure_ascii=False)

    # [P0-5] Resolver user_id si no se pasó: mirar el chunk actual para no romper callers.
    _live_tz: int | None = None
    _resolved_user_id = user_id
    if _resolved_user_id is None and task_id is not None:
        try:
            _row = execute_sql_query(
                "SELECT user_id FROM plan_chunk_queue WHERE id = %s",
                (task_id,),
                fetch_one=True,
            )
            if _row and _row.get("user_id"):
                _resolved_user_id = str(_row["user_id"])
        except Exception:
            pass
    if _resolved_user_id:
        try:
            _live_tz = _get_user_tz_live(_resolved_user_id, fallback_minutes=0)
        except Exception:
            _live_tz = None

    try:
        # Chunk actual: garantizado existir y estar en processing.
        if _live_tz is not None:
            execute_sql_write(
                """
                UPDATE plan_chunk_queue
                SET pipeline_snapshot = jsonb_set(
                        jsonb_set(
                            jsonb_set(
                                jsonb_set(
                                    pipeline_snapshot,
                                    '{form_data,current_pantry_ingredients}',
                                    %s::jsonb,
                                    true
                                ),
                                '{form_data,_pantry_captured_at}',
                                to_jsonb(%s::text),
                                true
                            ),
                            '{form_data,tzOffset}',
                            to_jsonb(%s::int),
                            true
                        ),
                        '{form_data,tz_offset_minutes}',
                        to_jsonb(%s::int),
                        true
                    )
                WHERE id = %s
                """,
                (pantry_json, captured_at, _live_tz, _live_tz, task_id),
            )
            execute_sql_write(
                """
                UPDATE plan_chunk_queue
                SET pipeline_snapshot = jsonb_set(
                        jsonb_set(
                            jsonb_set(
                                jsonb_set(
                                    pipeline_snapshot,
                                    '{form_data,current_pantry_ingredients}',
                                    %s::jsonb,
                                    true
                                ),
                                '{form_data,_pantry_captured_at}',
                                to_jsonb(%s::text),
                                true
                            ),
                            '{form_data,tzOffset}',
                            to_jsonb(%s::int),
                            true
                        ),
                        '{form_data,tz_offset_minutes}',
                        to_jsonb(%s::int),
                        true
                    ),
                    updated_at = updated_at
                WHERE meal_plan_id = %s
                  AND id <> %s
                  AND status IN ('pending', 'stale')
                """,
                (pantry_json, captured_at, _live_tz, _live_tz, str(meal_plan_id), task_id),
            )
        else:
            # Sin user_id resuelto → comportamiento legacy (solo pantry).
            execute_sql_write(
                """
                UPDATE plan_chunk_queue
                SET pipeline_snapshot = jsonb_set(
                        jsonb_set(
                            pipeline_snapshot,
                            '{form_data,current_pantry_ingredients}',
                            %s::jsonb,
                            true
                        ),
                        '{form_data,_pantry_captured_at}',
                        to_jsonb(%s::text),
                        true
                    )
                WHERE id = %s
                """,
                (pantry_json, captured_at, task_id),
            )
            execute_sql_write(
                """
                UPDATE plan_chunk_queue
                SET pipeline_snapshot = jsonb_set(
                        jsonb_set(
                            pipeline_snapshot,
                            '{form_data,current_pantry_ingredients}',
                            %s::jsonb,
                            true
                        ),
                        '{form_data,_pantry_captured_at}',
                        to_jsonb(%s::text),
                        true
                    ),
                    updated_at = updated_at
                WHERE meal_plan_id = %s
                  AND id <> %s
                  AND status IN ('pending', 'stale')
                """,
                (pantry_json, captured_at, str(meal_plan_id), task_id),
            )
        logger.info(
            f"[P0-3/PANTRY-PROP] Snapshot de inventario propagado a chunk {task_id} y siblings "
            f"del plan {meal_plan_id} ({len(fresh_inventory)} items, captured_at={captured_at})."
        )
    except Exception as prop_err:
        # Best-effort: si el UPDATE falla, el chunk actual igual usa el live en memoria.
        logger.warning(
            f"[P0-3/PANTRY-PROP] No se pudo propagar inventario fresco al snapshot "
            f"(plan {meal_plan_id}, chunk {task_id}): {prop_err}"
        )


def _pause_chunk_for_stale_inventory(
    task_id: str | int,
    user_id: str,
    week_number: int,
    snapshot_age_hours: float | None,
    reason: str = "stale_snapshot",
) -> None:
    """[P0-2] Pausa el chunk cuando el inventario en vivo no se pudo leer y el snapshot está vencido.

    Antes esto se aceptaba silenciosamente y se generaba contra datos viejos, rompiendo
    la promesa "los platos solo usan lo que hay en la nevera". Ahora congelamos el chunk
    hasta que el usuario refresque su nevera (mismo mecanismo de TTL/recordatorio que
    usa _pause_chunk_for_pantry_refresh, con un motivo distinto para telemetría).
    """
    pause_snapshot = execute_sql_query(
        "SELECT pipeline_snapshot FROM plan_chunk_queue WHERE id = %s",
        (task_id,),
        fetch_one=True,
    )
    pause_snapshot = copy.deepcopy((pause_snapshot or {}).get("pipeline_snapshot") or {})
    if isinstance(pause_snapshot, str):
        pause_snapshot = json.loads(pause_snapshot)

    pause_snapshot.setdefault("_pantry_pause_started_at", datetime.now(timezone.utc).isoformat())
    pause_snapshot.setdefault("_pantry_pause_reminders", 0)
    # [P0-2] TTL específico para infraestructura caída (4h por defecto): el usuario no puede
    # accionar; conviene escalar rápido a flexible_mode si el live no se recupera.
    pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS
    pause_snapshot["_pantry_pause_reminder_hours"] = CHUNK_PANTRY_EMPTY_REMINDER_HOURS
    pause_snapshot["_pantry_pause_reason"] = reason
    if snapshot_age_hours is not None:
        pause_snapshot["_pantry_snapshot_age_hours"] = round(snapshot_age_hours, 1)

    execute_sql_write(
        """
        UPDATE plan_chunk_queue
        SET status = 'pending_user_action',
            pipeline_snapshot = %s::jsonb,
            updated_at = NOW()
        WHERE id = %s
        """,
        (json.dumps(pause_snapshot, ensure_ascii=False), task_id),
    )
    age_str = f" (snapshot {snapshot_age_hours:.1f}h)" if snapshot_age_hours is not None else ""
    logger.warning(
        f"[P0-2/STALE-PAUSE] Chunk {task_id} (Week {week_number}) pausado para {user_id}: "
        f"inventario live inaccesible y snapshot vencido{age_str}. "
        f"Recovery cron retentará live cada tick; escalará a flexible tras "
        f"{CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS}h."
    )
    # [P0-1] Notificamos al usuario al entrar en pausa por stale_snapshot puro.
    # Antes este path quedaba silencioso bajo el razonamiento "es server-side y el
    # usuario no puede accionar". En la práctica abrir la app y refrescar la nevera
    # suele revivir el live-fetch, así que el usuario SÍ puede accionar — solo
    # necesita saber que su plan está pausado. Cooldown 24h para deduplicar entre
    # múltiples chunks pausados del mismo usuario.
    #
    # Para `stale_snapshot_live_unreachable`, el caller (línea ~2027) ya emite un
    # push contextual ANTES de llamar a esta función, así que aquí lo skipeamos
    # para evitar doble-push.
    if reason == "stale_snapshot":
        try:
            _maybe_notify_user_stale_snapshot_paused(user_id, snapshot_age_hours)
        except Exception as _np_err:
            # Best-effort: la pausa ya está persistida; un fallo de push no debe
            # romper la transición de estado del chunk.
            logger.warning(
                f"[P0-1/STALE-PAUSE-NOTIFY] No se pudo notificar al usuario {user_id}: {_np_err}"
            )


def _pause_chunk_for_pantry_refresh(task_id: str | int, user_id: str, week_number: int, fresh_inventory: list, reason: str = None):
    """Pauses chunk generation when the live pantry is too empty to preserve the zero-waste promise."""
    pause_snapshot = execute_sql_query(
        "SELECT pipeline_snapshot FROM plan_chunk_queue WHERE id = %s",
        (task_id,),
        fetch_one=True,
    )
    pause_snapshot = copy.deepcopy((pause_snapshot or {}).get("pipeline_snapshot") or {})
    if isinstance(pause_snapshot, str):
        pause_snapshot = json.loads(pause_snapshot)

    pause_snapshot.setdefault("_pantry_pause_started_at", datetime.now(timezone.utc).isoformat())
    pause_snapshot.setdefault("_pantry_pause_reminders", 0)
    if reason:
        pause_snapshot["_pantry_pause_reason"] = reason
        if reason == "persistent_drift":
            pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_STALE_SNAPSHOT_PAUSE_TTL_HOURS
        else:
            pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_PANTRY_EMPTY_TTL_HOURS
    else:
        pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_PANTRY_EMPTY_TTL_HOURS
    pause_snapshot["_pantry_pause_reminder_hours"] = CHUNK_PANTRY_EMPTY_REMINDER_HOURS

    execute_sql_write(
        """
        UPDATE plan_chunk_queue
        SET status = 'pending_user_action',
            pipeline_snapshot = %s::jsonb,
            updated_at = NOW()
        WHERE id = %s
        """,
        (json.dumps(pause_snapshot, ensure_ascii=False), task_id),
    )
    logger.warning(
        f"[P1-3/PANTRY] Chunk {week_number} pausado para {user_id}: inventario fresco insuficiente "
        f"({len(fresh_inventory or [])} items brutos)."
    )

    _dispatch_push_notification(
        user_id=user_id,
        title="Actualiza tu nevera para continuar",
        body="Tu próximo chunk quedó en pausa porque tu inventario está vacío o casi vacío. Actualiza 'Mi Nevera' y reintenta.",
        url="/dashboard",
    )


def _pause_chunk_for_final_inventory_validation(
    task_id: str | int,
    user_id: str,
    week_number: int,
    reason: str,
    missing_ingredients: list | None = None,
) -> None:
    """Pause a chunk when the final live pantry validation cannot safely approve it.

    [P0-C] `missing_ingredients` (lista de strings normalizables) se persiste en el
    pause_snapshot como `_pantry_pause_missing_ingredients` para que el recovery
    cron pueda detectar cuándo el usuario las añadió a la nevera y reanudar el
    chunk de inmediato — sin esperar el TTL escalation a flexible_mode. Si no se
    proporciona (porque la causa fue live-fetch caído y no sabemos qué falta), el
    chunk seguirá la ruta TTL existente.
    """
    pause_snapshot = execute_sql_query(
        "SELECT pipeline_snapshot FROM plan_chunk_queue WHERE id = %s",
        (task_id,),
        fetch_one=True,
    )
    pause_snapshot = copy.deepcopy((pause_snapshot or {}).get("pipeline_snapshot") or {})
    if isinstance(pause_snapshot, str):
        pause_snapshot = json.loads(pause_snapshot)

    pause_snapshot.setdefault("_pantry_pause_started_at", datetime.now(timezone.utc).isoformat())
    pause_snapshot.setdefault("_pantry_pause_reminders", 0)
    pause_snapshot["_pantry_pause_reason"] = reason
    pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_FINAL_VALIDATION_PAUSE_TTL_HOURS
    pause_snapshot["_pantry_pause_reminder_hours"] = min(
        CHUNK_PANTRY_EMPTY_REMINDER_HOURS,
        CHUNK_FINAL_VALIDATION_PAUSE_TTL_HOURS,
    )
    # [P0-C] Persistir lista estructurada para que `_recover_pantry_paused_chunks`
    # pueda comparar contra la pantry actual sin parsear strings de error de _vip.
    if missing_ingredients:
        # Sanear: solo strings no vacíos, deduplicados, longitud razonable.
        _safe_missing = [
            str(x).strip() for x in missing_ingredients
            if x and isinstance(x, (str, bytes)) and str(x).strip()
        ]
        # Deduplicar preservando orden.
        _seen: set = set()
        _dedup: list = []
        for _item in _safe_missing:
            _key = _item.lower()
            if _key not in _seen:
                _seen.add(_key)
                _dedup.append(_item)
        if _dedup:
            pause_snapshot["_pantry_pause_missing_ingredients"] = _dedup[:30]
    pause_snapshot.setdefault("_pantry_pause_recovery_attempts", 0)

    execute_sql_write(
        """
        UPDATE plan_chunk_queue
        SET status = 'pending_user_action',
            pipeline_snapshot = %s::jsonb,
            updated_at = NOW()
        WHERE id = %s
        """,
        (json.dumps(pause_snapshot, ensure_ascii=False), task_id),
    )
    _missing_count = len(pause_snapshot.get("_pantry_pause_missing_ingredients") or [])
    logger.warning(
        f"[P0-2/PANTRY-FINAL] Chunk {week_number} pausado para {user_id}: "
        f"falló la validación final de inventario vivo ({reason}). TTL="
        f"{CHUNK_FINAL_VALIDATION_PAUSE_TTL_HOURS}h. "
        f"missing_count={_missing_count} (recovery cron re-evaluará pantry actual)."
    )
    _dispatch_push_notification(
        user_id=user_id,
        title="No pudimos confirmar tu nevera",
        body="Tu próximo bloque quedó en pausa porque no pudimos validar tu inventario al final. Revisa 'Mi Nevera' y vuelve a intentar.",
        url="/dashboard",
    )


# [P1-4] Telemetría de fallbacks de pantry_tolerance. Antes los fallbacks eran
# silenciosos (logger.debug) o sin telemetría agregada, así que un usuario con
# override 1.30 que sufría DB blip caía al default 1.05 sin que dashboards
# detectaran la degradación. Ahora cada fallback "inesperado" (DB error,
# valor no-numérico, clamp por valor fuera de rango) incrementa este counter
# rolling, expuesto vía /api/system/pantry-tolerance-health.
_PANTRY_TOLERANCE_FALLBACKS: list = []  # list[tuple[float_unix_ts, source, user_id]]
_PANTRY_TOLERANCE_FALLBACK_WINDOW_SECONDS = 24 * 3600
_PANTRY_TOLERANCE_FALLBACK_MAX_RECORDS = 5000


def _record_pantry_tolerance_fallback(source: str, user_id: str) -> None:
    """[P1-4] Registra un fallback no-trivial de pantry_tolerance.

    No persiste en DB (sería ruido para una métrica observacional); el endpoint
    /api/system/pantry-tolerance-health lee el ring buffer in-memory y aplica
    la ventana rolling de 24h. Si la fila excede MAX_RECORDS, se descarta el
    más antiguo para acotar memoria.
    """
    import time as _p14_time
    now = _p14_time.time()
    _PANTRY_TOLERANCE_FALLBACKS.append((now, source, str(user_id)))
    cutoff = now - _PANTRY_TOLERANCE_FALLBACK_WINDOW_SECONDS
    # Poda incremental: drop entradas viejas y exceso de cap.
    while _PANTRY_TOLERANCE_FALLBACKS and (
        _PANTRY_TOLERANCE_FALLBACKS[0][0] < cutoff
        or len(_PANTRY_TOLERANCE_FALLBACKS) > _PANTRY_TOLERANCE_FALLBACK_MAX_RECORDS
    ):
        _PANTRY_TOLERANCE_FALLBACKS.pop(0)


# [P2-1] Telemetría de fallbacks de TZ en `_enqueue_plan_chunk`. Antes cada chunk
# que caía al fallback emitía una línea WARNING; un plan de 30 días con
# `_plan_start_date` corrupto generaba ~8 líneas idénticas modulo `week_number`.
# Ahora:
#   - Primera ocurrencia de `(plan_id, reason)` en la ventana de dedupe (default
#     1h) emite WARNING accionable como antes (preserva visibilidad inmediata).
#   - Ocurrencias subsiguientes solo se registran en el ring buffer (sin WARNING),
#     evitando spam.
#   - Endpoint `/api/system/tz-fallback-health` agrega el buffer por reason/plan
#     en la ventana rolling para dashboards de operación.
_TZ_FALLBACK_EVENTS: list = []  # list[tuple[ts, user_id, plan_id, week_number, reason]]
_TZ_FALLBACK_WINDOW_SECONDS = 24 * 3600
_TZ_FALLBACK_MAX_RECORDS = 5000
_TZ_FALLBACK_DEDUPE_TTL_SECONDS = 3600  # 1h: re-emite WARNING si el problema persiste tras una hora
_TZ_FALLBACK_DEDUPE_KEYS: dict = {}  # {(plan_id, reason): last_warning_ts}


def _record_tz_fallback(
    user_id: str,
    meal_plan_id,
    week_number,
    reason: str,
    detail: str = "",
    resolution: str = "now_plus_delay",
) -> bool:
    """[P2-1] Registra un fallback de TZ; retorna True si emitió WARNING.

    Dedupe por `(plan_id, reason)` con TTL `_TZ_FALLBACK_DEDUPE_TTL_SECONDS`:
    la primera vez (o tras expirar el TTL) emite WARNING accionable; las
    repeticiones dentro de la ventana solo se acumulan en el buffer.

    Args:
        user_id, meal_plan_id, week_number: identificación del chunk afectado.
        reason: identificador estable del modo de fallo.
        detail: texto libre añadido al WARNING cuando se emite.
        resolution: cómo se resolvió el fallback. Cambia el tono del log:
            - "snapshot_recovered": el snapshot fue corregido; severidad menor.
            - "profile_today": derivado de user_profile + today; severidad menor.
            - "last_plan": derivado del último plan del usuario; severidad menor.
            - "forced_8am_utc": no se pudo recuperar TZ; chunk se programa a 8am UTC
              en lugar de NOW()+delay (mejora respecto al comportamiento previo).
            - "now_plus_delay": fallback histórico (compatibilidad).

    Returns:
        True si se emitió un WARNING (primera ocurrencia o TTL expirado),
        False si solo se acumuló en el buffer (dedupeado).
    """
    import time as _p21_time
    now = _p21_time.time()
    plan_key = str(meal_plan_id) if meal_plan_id is not None else "unknown_plan"
    dedupe_key = (plan_key, reason)

    # Append al buffer agregable (siempre, también las repetidas).
    _TZ_FALLBACK_EVENTS.append((now, str(user_id), plan_key, week_number, reason))
    cutoff = now - _TZ_FALLBACK_WINDOW_SECONDS
    while _TZ_FALLBACK_EVENTS and (
        _TZ_FALLBACK_EVENTS[0][0] < cutoff
        or len(_TZ_FALLBACK_EVENTS) > _TZ_FALLBACK_MAX_RECORDS
    ):
        _TZ_FALLBACK_EVENTS.pop(0)

    # Decisión de WARNING: primera vez o TTL de dedupe expirado.
    last_warned = _TZ_FALLBACK_DEDUPE_KEYS.get(dedupe_key)
    should_warn = last_warned is None or (now - last_warned) >= _TZ_FALLBACK_DEDUPE_TTL_SECONDS
    if should_warn:
        _TZ_FALLBACK_DEDUPE_KEYS[dedupe_key] = now
        # Limpieza ocasional del dict de dedupe para que no crezca sin límite si
        # llegan miles de planes corruptos distintos.
        if len(_TZ_FALLBACK_DEDUPE_KEYS) > _TZ_FALLBACK_MAX_RECORDS:
            stale_keys = [k for k, v in _TZ_FALLBACK_DEDUPE_KEYS.items() if (now - v) >= _TZ_FALLBACK_DEDUPE_TTL_SECONDS]
            for k in stale_keys:
                _TZ_FALLBACK_DEDUPE_KEYS.pop(k, None)
        # [P0-2] Mensaje adaptado a la resolución para no mentir cuando ya no
        # estamos cayendo a NOW()+delay.
        if resolution == "forced_8am_utc":
            _resolution_msg = "Chunk programado a 8am UTC del día N (no se respeta TZ local del usuario)."
        elif resolution in ("profile_today", "last_plan"):
            _resolution_msg = f"Anchor derivado de '{resolution}'; execute_after sí respeta TZ local."
        elif resolution == "snapshot_recovered":
            _resolution_msg = "Snapshot recuperado tras parse-fail; execute_after respeta TZ local."
        else:
            _resolution_msg = "Usando NOW()+delay; execute_after no respetará la TZ local."
        logger.warning(
            f"[P0-4/TZ-FALLBACK] chunk_tz_fallback_used "
            f"user_id={user_id} meal_plan_id={meal_plan_id} week={week_number} "
            f"reason={reason} resolution={resolution}"
            f"{(' ' + detail) if detail else ''}. "
            f"{_resolution_msg} "
            f"(Dedupe activo {_TZ_FALLBACK_DEDUPE_TTL_SECONDS}s para este (plan, reason).)"
        )
        return True
    return False


def _get_user_tz_minutes_optional(user_id: str):
    """[P0-2] Lee `tz_offset_minutes` del user_profile, devolviendo None si falta.

    A diferencia de `_get_user_tz_live`, NO infiere 0 cuando el campo está vacío:
    distinguir 'UTC=0' de 'no hay TZ' es necesario para decidir si caemos a la
    siguiente fuente del fallback chain o aceptamos el valor.
    """
    if not user_id or user_id == "guest":
        return None
    try:
        row = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,),
            fetch_one=True,
        )
        if not row or not row.get("health_profile"):
            return None
        hp = row.get("health_profile") or {}
        val = hp.get("tz_offset_minutes")
        if val is None:
            val = hp.get("tzOffset")
        if val is None:
            return None
        return int(val)
    except Exception as e:
        logger.warning(f"[P0-2/TZ-OPTIONAL] No se pudo leer tz_offset opcional para {user_id}: {e}")
        return None


def _resolve_chunk_start_anchor(
    user_id: str,
    snapshot: dict,
    meal_plan_id,
    week_number: int,
):
    """[P0-2] Resuelve el (start_dt, tz_offset_min, source) para programar un chunk.

    Cadena de fallback en 4 pasos. Sin esta cadena, cuando `_plan_start_date` faltaba
    o fallaba parseo, el chunk caía a `NOW() + delay_days` y disparaba a hora arbitraria
    del día (e.g., 3am local), rompiendo el contrato de "ejecución matutina".

    1. **snapshot**: usa `snapshot.form_data._plan_start_date` (path normal).
    2. **profile_today**: lee TZ de `user_profiles.health_profile` y usa `today` UTC
       como start_dt. El offset se aplica al combinar para llegar a midnight local.
    3. **last_plan**: si el perfil no tiene TZ, busca el último plan del usuario y
       toma su TZ + today.
    4. **forced_8am_utc**: si nada existe, devuelve `(None, 0, "forced_8am_utc")`.
       El caller usará 8am UTC del día N como execute_after — peor que la TZ local,
       pero infinitamente mejor que NOW()+delay (que dispara a las 3am).

    Returns:
        (start_dt, tz_offset_min, source). `start_dt` es None solo en source 4.
    """
    form_data = snapshot.get("form_data", {}) if isinstance(snapshot, dict) else {}
    if not isinstance(form_data, dict):
        form_data = {}

    snapshot_tz = 0
    try:
        snapshot_tz = int(
            form_data.get("tzOffset")
            or form_data.get("tz_offset_minutes")
            or 0
        )
    except (TypeError, ValueError):
        snapshot_tz = 0

    fail_reasons: list = []

    # === Source 1: snapshot._plan_start_date ===
    start_iso = form_data.get("_plan_start_date")
    if start_iso:
        try:
            from constants import safe_fromisoformat
            start_dt = safe_fromisoformat(start_iso)
            if start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=timezone.utc)
            tz_min = _get_user_tz_live(user_id, snapshot_tz)
            return (start_dt, tz_min, "snapshot")
        except Exception as e:
            fail_reasons.append(f"parse_failed:{type(e).__name__}:{str(e)[:60]}")
    else:
        fail_reasons.append("snapshot_missing")

    # === Source 2: user_profile TZ + today() ===
    profile_tz = _get_user_tz_minutes_optional(user_id)
    if profile_tz is not None:
        today_utc = datetime.now(timezone.utc)
        start_dt = datetime.combine(
            today_utc.date(), datetime.min.time()
        ).replace(tzinfo=timezone.utc)
        _record_tz_fallback(
            user_id=user_id, meal_plan_id=meal_plan_id,
            week_number=week_number, reason="anchor_via_profile_today",
            detail=f"snapshot={'/'.join(fail_reasons)} profile_tz={profile_tz}m",
            resolution="profile_today",
        )
        return (start_dt, int(profile_tz), "profile_today")
    fail_reasons.append("profile_missing")

    # === Source 3: último meal_plan del usuario ===
    try:
        row = execute_sql_query(
            "SELECT id, plan_data FROM meal_plans "
            "WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
            (user_id,),
            fetch_one=True,
        )
        if row and isinstance(row.get("plan_data"), dict):
            prior_pd = row["plan_data"]
            prior_hp = prior_pd.get("health_profile") or {}
            prior_tz_raw = prior_hp.get("tz_offset_minutes")
            if prior_tz_raw is None:
                prior_tz_raw = prior_hp.get("tzOffset")
            if prior_tz_raw is not None:
                try:
                    prior_tz = int(prior_tz_raw)
                except (TypeError, ValueError):
                    prior_tz = None
                if prior_tz is not None:
                    today_utc = datetime.now(timezone.utc)
                    start_dt = datetime.combine(
                        today_utc.date(), datetime.min.time()
                    ).replace(tzinfo=timezone.utc)
                    _record_tz_fallback(
                        user_id=user_id, meal_plan_id=meal_plan_id,
                        week_number=week_number,
                        reason="anchor_via_last_plan",
                        detail=(
                            f"snapshot={'/'.join(fail_reasons)} "
                            f"prior_plan_id={row.get('id')} prior_tz={prior_tz}m"
                        ),
                        resolution="last_plan",
                    )
                    return (start_dt, prior_tz, "last_plan")
    except Exception as e:
        fail_reasons.append(f"last_plan_query_error:{type(e).__name__}")
        logger.warning(
            f"[P0-2/LAST-PLAN] No se pudo leer último plan de {user_id}: {e}"
        )
    else:
        fail_reasons.append("last_plan_no_tz")

    # === Source 4: forced_8am_utc ===
    _record_tz_fallback(
        user_id=user_id, meal_plan_id=meal_plan_id,
        week_number=week_number, reason="no_anchor_resolvable",
        detail=f"failures={'/'.join(fail_reasons)}",
        resolution="forced_8am_utc",
    )
    return (None, 0, "forced_8am_utc")


def _resolve_pantry_tolerance(
    user_id: str,
    default: float = CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE,
) -> tuple[float, str]:
    """[P1-4] Resuelve pantry_tolerance del usuario, retornando (valor, source).

    `source` ∈ {
        'user_override': lectura exitosa con valor distinto al default;
        'user_override_default_match': valor del usuario coincide con el default;
        'default_no_override': SELECT exitoso, columna NULL → comportamiento esperado;
        'default_no_row': usuario no encontrado en user_profiles (defensivo);
        'fallback_db_error': SELECT lanzó excepción (DB blip, timeout);
        'fallback_non_numeric': valor presente pero no-parseable a float;
        'fallback_clamped': valor explícito fuera de [MIN, MAX]; se ajustó.
    }

    Solo los `fallback_*` indican degradación que dashboards deben monitorear.
    El caller `_get_pantry_tolerance_for_user` envuelve este helper, decide log
    level y registra en el counter de telemetría.
    """
    try:
        row = execute_sql_query(
            "SELECT pantry_tolerance FROM user_profiles WHERE id = %s",
            (str(user_id),),
            fetch_one=True,
        )
    except Exception as e:
        logger.warning(
            f"[P1-4/PANTRY/TOLERANCE] Fallback db_error user={user_id}: {e}. "
            f"Aplicando default={default}."
        )
        _record_pantry_tolerance_fallback("fallback_db_error", user_id)
        return float(default), "fallback_db_error"

    if not row:
        return float(default), "default_no_row"
    raw = row.get("pantry_tolerance")
    if raw is None:
        return float(default), "default_no_override"
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            f"[P1-4/PANTRY/TOLERANCE] Fallback non_numeric user={user_id} raw={raw!r}. "
            f"Aplicando default={default}."
        )
        _record_pantry_tolerance_fallback("fallback_non_numeric", user_id)
        return float(default), "fallback_non_numeric"

    # Clamp defensivo (el CHECK constraint debería atrapar primero, pero por si
    # un INSERT viejo bypassed o el constraint aún no se aplicó en este deploy).
    clamped = max(
        float(CHUNK_PANTRY_TOLERANCE_MIN),
        min(float(CHUNK_PANTRY_TOLERANCE_MAX), val),
    )
    if clamped != val:
        logger.warning(
            f"[P1-4/PANTRY/TOLERANCE] Fallback clamped user={user_id} raw={val} "
            f"out_of_range=[{CHUNK_PANTRY_TOLERANCE_MIN}, {CHUNK_PANTRY_TOLERANCE_MAX}] → {clamped}."
        )
        _record_pantry_tolerance_fallback("fallback_clamped", user_id)
        return clamped, "fallback_clamped"

    if abs(clamped - float(default)) < 1e-9:
        return clamped, "user_override_default_match"
    return clamped, "user_override"


def _get_pantry_tolerance_for_user(
    user_id: str,
    default: float = CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE,
) -> float:
    """[P1-D] Devuelve la tolerancia de cantidades de inventario configurada por el usuario.

    Lee `user_profiles.pantry_tolerance`. Si NULL o si el SELECT falla (DB blip,
    columna recién añadida en deploy híbrido), devuelve el default global. El valor
    se clampa al rango [CHUNK_PANTRY_TOLERANCE_MIN, CHUNK_PANTRY_TOLERANCE_MAX]
    para evitar que un valor fuera de bounds (que pudo entrar antes del CHECK
    constraint) llegue a `validate_ingredients_against_pantry` y altere su semántica.

    No bloqueante: cualquier excepción cae al default. La preferencia es UX, no
    safety-critical: el peor caso es que el usuario no vea su override aplicado
    durante un tick y sí lo vea en el siguiente.

    [P1-4] Telemetría: cada fallback inesperado (DB error, valor no-numérico,
    clamp) emite un log estructurado `[P1-4/PANTRY/TOLERANCE] Fallback <source> ...`
    y se registra en el ring buffer expuesto en /api/system/pantry-tolerance-health.
    Los fallbacks "esperados" (NULL = no_override) NO incrementan el counter.
    """
    value, _source = _resolve_pantry_tolerance(user_id, default=default)
    return value


def _pantry_covers_missing(missing_ingredients: list, current_pantry: list) -> tuple[bool, list]:
    """[P0-C] Devuelve (covered, still_missing) tras normalizar ambos lados a la base
    canónica usada por el resto del sistema (`normalize_ingredient_for_tracking`).

    `current_pantry` puede ser una lista de strings (`"500g pollo"`) o de dicts con
    campos `name`/`display_string` — replicamos el manejo defensivo que ya hace
    `_synthesize_last_chunk_learning_from_plan_days`. La comparación es por base
    normalizada, no por string literal: si pantry tiene "pechuga de pollo deshuesada
    500g" y missing dice "pollo", se considera cubierto.

    Retorna `(True, [])` solo si TODOS los ingredientes faltantes están cubiertos.
    Si alguno no está, devuelve `(False, [items_aún_faltantes])` para telemetría.
    """
    from constants import normalize_ingredient_for_tracking
    if not missing_ingredients:
        # Sin lista de faltantes no podemos validar cobertura; el caller decide qué hacer.
        return False, []

    pantry_bases: set = set()
    for raw in (current_pantry or []):
        if isinstance(raw, dict):
            text = raw.get("name") or raw.get("display_string") or ""
        else:
            text = str(raw)
        base = normalize_ingredient_for_tracking(text)
        if base:
            pantry_bases.add(base)

    still_missing: list = []
    for raw in missing_ingredients:
        text = str(raw or "").strip()
        if not text:
            continue
        base = normalize_ingredient_for_tracking(text)
        if base and base in pantry_bases:
            continue
        # No match por base canónica; segundo intento: substring directo (cubre casos
        # donde normalize devuelve "" o el sinónimo no está mapeado todavía).
        text_lc = text.lower()
        if any(text_lc in str(p).lower() or str(p).lower() in text_lc for p in pantry_bases):
            continue
        still_missing.append(text)

    return (len(still_missing) == 0), still_missing


def _dispatch_push_notification(user_id: str, title: str, body: str, url: str = "/dashboard") -> None:
    try:
        import threading
        from utils_push import send_push_notification

        threading.Thread(
            target=send_push_notification,
            kwargs={
                "user_id": user_id,
                "title": title,
                "body": body,
                "url": url,
            },
            daemon=True,
        ).start()
    except Exception as push_err:
        logger.warning(f"[P1-3/PANTRY] No se pudo enviar push: {push_err}")


def _reconcile_chunk_reservations(user_id: str, chunk_id: str, days: list, max_retries: int = 3) -> bool:
    """[P0-5] Reintenta las reservas faltantes de un chunk con reservation_status='partial'.

    Se llama de forma síncrona tras detectar reservas parciales. Itera hasta max_retries
    intentando reserve_plan_ingredients de nuevo. Si logra >= 50% de ingredientes parseables,
    marca reservation_status='ok' y retorna True.

    [P1-2] Cambio de contrato: ahora RETORNA bool en vez de None.
        True  → reconciliación OK (>= 50% reservado), reservation_status='ok'.
        False → reconciliación AGOTADA (CAS conflicts/Supabase lentitud persistentes).
                reservation_status permanece 'partial'. El caller DEBE liberar las
                reservas parciales aplicadas y pausar el chunk en pending_user_action
                — sin esto, el chunk se marcaría 'completed' con reservas incompletas
                y el siguiente chunk del mismo plan vería inventario sobreestimado
                tras los 5 min del bloqueo de pickup, causando overbooking.
    """
    import time
    for _attempt in range(max_retries):
        try:
            reserved = reserve_plan_ingredients(user_id, chunk_id, days)
            from shopping_calculator import _parse_quantity
            _expected = 0
            for _d in days:
                for _m in (_d or {}).get('meals', []):
                    for _i in (_m.get('ingredients') or []):
                        if _i and len(str(_i).strip()) >= 3:
                            try:
                                _q, _u, _n = _parse_quantity(str(_i))
                                if _n and _q > 0:
                                    _expected += 1
                            except Exception:
                                pass
            # [P0-5] If the chunk produced no parseable ingredients to reserve, treat
            # reservation as a no-op success. Without this, `_min_ok = max(1, 0)` forced
            # `reserved (0) >= _min_ok (1)` to be False on every retry, exhausting CAS
            # attempts and pausing chunks that legitimately had nothing to reserve
            # (e.g., emergency Edge Recipes with simplified ingredient strings, or test
            # fixtures with `{"name": "Plato N"}` and no `ingredients` key).
            if _expected == 0:
                execute_sql_write(
                    "UPDATE plan_chunk_queue SET reservation_status = 'ok', updated_at = NOW() WHERE id = %s",
                    (chunk_id,)
                )
                logger.info(
                    f"[P0-5/RECONCILE] Chunk {chunk_id} sin ingredientes parseables "
                    f"para reservar; reservation_status='ok' (no-op)."
                )
                return True
            _min_ok = max(1, int(_expected * 0.5))
            if reserved >= _min_ok:
                execute_sql_write(
                    "UPDATE plan_chunk_queue SET reservation_status = 'ok', updated_at = NOW() WHERE id = %s",
                    (chunk_id,)
                )
                logger.info(
                    f"[P0-5/RECONCILE] Reconciliación exitosa chunk {chunk_id}: "
                    f"{reserved}/{_expected} ingredientes reservados (intento {_attempt + 1})."
                )
                return True
            logger.warning(
                f"[P0-5/RECONCILE] Intento {_attempt + 1}/{max_retries} parcial para chunk {chunk_id}: "
                f"{reserved}/{_expected}."
            )
        except Exception as e:
            logger.error(f"[P0-5/RECONCILE] Error en intento {_attempt + 1} chunk {chunk_id}: {e}")
        if _attempt < max_retries - 1:
            time.sleep(2)
    logger.error(
        f"[P1-2/RECONCILE-EXHAUSTED] Reconciliación agotada para chunk {chunk_id} tras "
        f"{max_retries} intentos. reservation_status permanece 'partial'. El caller debe "
        f"liberar reservas parciales y pausar el chunk."
    )
    return False


def _should_pause_for_empty_pantry(
    fresh_inventory_source: str,
    fresh_inventory: list,
    snapshot: dict | None = None,
    form_data: dict | None = None,
) -> bool:
    """[P1-1] Devuelve True si el chunk debe pausarse por despensa insuficiente.

    Antes la condición era `source == "live" AND items < MIN`. Esto dejaba pasar
    chunks con `_fresh_pantry_source == "snapshot"` (TTL aún válido pero pantry
    vacío) o sin source poblado: el LLM generaba el plan sin restricción de
    nevera, violando la promesa "solo alimentos en la nevera".

    Ahora se pausa siempre que `items < CHUNK_MIN_FRESH_PANTRY_ITEMS`, EXCEPTO:
      - `flexible_mode` o `advisory_only`: degradaciones deliberadas que ya
        manejan TTL/escalación aparte (live_degraded_snapshot,
        stale_snapshot_auto_flex, etc.). Re-pausarlos crearía un loop.
      - `source == "guest"`: usuarios anónimos no tienen perfil para refrescar
        despensa; pausar sería un dead-end.
    """
    snapshot = snapshot or {}
    form_data = form_data or {}
    flexible_mode = bool(
        snapshot.get("_pantry_flexible_mode")
        or form_data.get("_pantry_flexible_mode")
    )
    advisory_only = bool(
        snapshot.get("_pantry_advisory_only")
        or form_data.get("_pantry_advisory_only")
    )
    if flexible_mode or advisory_only:
        return False
    if fresh_inventory_source == "guest":
        return False
    return _count_meaningful_pantry_items(fresh_inventory) < CHUNK_MIN_FRESH_PANTRY_ITEMS


def _recover_pantry_paused_chunks() -> None:
    """Revisa chunks en pending_user_action, recuerda al usuario y evita bloqueo indefinido."""
    try:
        # [P1-3] meal_plan_id incluido en el SELECT para que las activaciones de
        # flexible_mode puedan persistir el evento en plan_data._mode_history y el
        # frontend pueda mostrar el badge "Plan en modo flexible".
        paused_rows = execute_sql_query(
            """
            SELECT id, user_id, meal_plan_id, week_number, pipeline_snapshot,
                   EXTRACT(EPOCH FROM (NOW() - updated_at))::int AS paused_seconds
            FROM plan_chunk_queue
            WHERE status = 'pending_user_action'
            ORDER BY updated_at ASC
            LIMIT 50
            """,
            fetch_all=True,
        ) or []

        for row in paused_rows:
            snap = copy.deepcopy(row.get("pipeline_snapshot") or {})
            if isinstance(snap, str):
                snap = json.loads(snap)

            paused_seconds = int(row.get("paused_seconds") or 0)
            reminder_hours = int(snap.get("_pantry_pause_reminder_hours") or CHUNK_PANTRY_EMPTY_REMINDER_HOURS)
            ttl_hours = int(snap.get("_pantry_pause_ttl_hours") or CHUNK_PANTRY_EMPTY_TTL_HOURS)
            reminders_sent = int(snap.get("_pantry_pause_reminders") or 0)
            pause_reason = str(snap.get("_pantry_pause_reason") or "empty_pantry")
            user_id_str = str(row["user_id"])
            row_id = row["id"]
            week_num = row.get("week_number")
            meal_plan_id_str = str(row.get("meal_plan_id")) if row.get("meal_plan_id") else None

            # [P1-2] missing_start_date_no_anchor: el plan no tiene ancla de fecha
            # recuperable (`_plan_start_date`, `grocery_start_date`, ni `created_at`).
            # Antes este reason no tenía branch en el recovery cron y caía al fallback
            # de pantry: tras `CHUNK_PANTRY_EMPTY_TTL_HOURS` activaba `flexible_mode`
            # asumiendo problema de nevera, lo cual era incorrecto y dejaba al chunk
            # generándose con un `NOW()` fabricado o quedaba flotando sin escalar.
            # Ahora, en cada tick: re-leemos plan_data, intentamos refrescar el ancla
            # desde meal_plans.created_at; si sigue sin ancla, incrementamos el
            # contador y al exceder `CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS` escalamos a
            # dead_letter con reason='unrecoverable_missing_anchor'.
            if pause_reason == "missing_start_date_no_anchor":
                if meal_plan_id_str is None:
                    logger.warning(
                        f"[P1-2/ANCHOR-RECOVERY] Chunk {row_id} pausado por anchor missing "
                        f"pero meal_plan_id es NULL; saltando."
                    )
                    continue
                try:
                    _p12_anchor_row = execute_sql_query(
                        """
                        SELECT
                            plan_data->>'_plan_start_date' AS psd,
                            plan_data->>'grocery_start_date' AS gsd,
                            created_at,
                            COALESCE((plan_data->>'_anchor_recovery_attempts')::int, 0) AS attempts
                        FROM meal_plans WHERE id = %s
                        """,
                        (meal_plan_id_str,),
                        fetch_one=True,
                    ) or {}
                except Exception as _p12_q_err:
                    logger.warning(
                        f"[P1-2/ANCHOR-RECOVERY] Error leyendo plan {meal_plan_id_str}: {_p12_q_err}"
                    )
                    continue

                _p12_recovered = (
                    _p12_anchor_row.get("psd")
                    or _p12_anchor_row.get("gsd")
                    or _p12_anchor_row.get("created_at")
                )
                if _p12_recovered:
                    # Ancla apareció (probablemente una escritura externa cerró el gap,
                    # o el snapshot se enriqueció). Reanudamos el chunk como pending.
                    resumed_snapshot = copy.deepcopy(snap)
                    resumed_snapshot["_anchor_pause_resolution"] = "anchor_recovered"
                    resumed_snapshot["_anchor_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(resumed_snapshot, ensure_ascii=False), row_id),
                    )
                    logger.info(
                        f"[P1-2/ANCHOR-RECOVERED] Chunk {week_num} plan {meal_plan_id_str} "
                        f"reanudado: ancla detectada en plan_data."
                    )
                    continue

                _p12_prev = int(_p12_anchor_row.get("attempts") or 0)
                _p12_new = _p12_prev + 1
                try:
                    execute_sql_write(
                        "UPDATE meal_plans "
                        "SET plan_data = jsonb_set(COALESCE(plan_data, '{}'::jsonb), "
                        "'{_anchor_recovery_attempts}', to_jsonb(%s::int), true) "
                        "WHERE id = %s",
                        (_p12_new, meal_plan_id_str),
                    )
                except Exception as _p12_persist_err:
                    logger.warning(
                        f"[P1-2/ANCHOR-RECOVERY] No se pudo persistir attempts={_p12_new} "
                        f"en plan {meal_plan_id_str}: {_p12_persist_err}"
                    )

                if _p12_new >= CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS:
                    logger.error(
                        f"[P1-2/ANCHOR-ESCALATE] Chunk {week_num} plan {meal_plan_id_str} "
                        f"agotó {_p12_new} intentos sin ancla. Escalando a dead_letter."
                    )
                    try:
                        _escalate_unrecoverable_chunk(
                            task_id=str(row_id),
                            user_id=user_id_str,
                            plan_id=meal_plan_id_str,
                            week_number=int(week_num) if week_num is not None else 0,
                            recovery_attempts=_p12_new,
                            escalation_reason="unrecoverable_missing_anchor",
                        )
                    except Exception as _p12_esc_err:
                        logger.error(
                            f"[P1-2/ANCHOR-ESCALATE] Falló escalación para chunk "
                            f"{row_id}: {_p12_esc_err}"
                        )
                    continue

                logger.info(
                    f"[P1-2/ANCHOR-WAIT] Chunk {week_num} plan {meal_plan_id_str} "
                    f"sigue sin ancla (intento {_p12_new}/{CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS})."
                )
                continue

            # [P0-2 FIX] unrecoverable_corrupted_date: el plan tiene un anchor existente
            # (`_plan_start_date` o `grocery_start_date` en plan_data) pero corrupto al
            # punto de no parsear con `safe_fromisoformat` — y `created_at` tampoco
            # rescata. Mientras la corrupción persista en la fuente, el chunk no puede
            # avanzar. Pre-fix, este reason no tenía branch en el recovery cron y el
            # chunk caía al fallback genérico de `empty_pantry`, derivando en
            # `flexible_mode` tras `CHUNK_PANTRY_EMPTY_TTL_HOURS` — generando con datos
            # incorrectos en lugar de pedir intervención.
            #
            # Cada tick re-leemos los anchors candidatos y validamos que parsean. Si
            # alguno se vuelve usable (operador/usuario corrigió el dato, o un retry
            # del frontend escribió un anchor sano), reanudamos el chunk. Si tras
            # CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS sigue sin parsear, escalamos a
            # dead_letter con reason='unrecoverable_corrupted_date' para que el
            # frontend muestre el banner de regeneración.
            if pause_reason == "unrecoverable_corrupted_date":
                if meal_plan_id_str is None:
                    logger.warning(
                        f"[P0-2/CORRUPT-RECOVERY] Chunk {row_id} pausado por fecha corrupta "
                        f"pero meal_plan_id es NULL; saltando."
                    )
                    continue
                try:
                    _p02c_row = execute_sql_query(
                        """
                        SELECT
                            plan_data->>'_plan_start_date' AS psd,
                            plan_data->>'grocery_start_date' AS gsd,
                            created_at,
                            COALESCE((plan_data->>'_anchor_recovery_attempts')::int, 0) AS attempts
                        FROM meal_plans WHERE id = %s
                        """,
                        (meal_plan_id_str,),
                        fetch_one=True,
                    ) or {}
                except Exception as _p02c_q_err:
                    logger.warning(
                        f"[P0-2/CORRUPT-RECOVERY] Error leyendo plan {meal_plan_id_str}: "
                        f"{_p02c_q_err}"
                    )
                    continue

                # Probar cada candidato y quedarse con el primero que parsea.
                from constants import safe_fromisoformat as _p02c_safe_iso
                _p02c_parseable = None
                _p02c_source = None
                for _p02c_src, _p02c_raw in (
                    ("_plan_start_date", _p02c_row.get("psd")),
                    ("grocery_start_date", _p02c_row.get("gsd")),
                    ("created_at", _p02c_row.get("created_at")),
                ):
                    if _p02c_raw is None or _p02c_raw == "":
                        continue
                    try:
                        if hasattr(_p02c_raw, "date"):
                            _p02c_parseable = _p02c_raw.date().isoformat()
                        else:
                            _p02c_safe_iso(str(_p02c_raw))
                            _p02c_parseable = str(_p02c_raw)
                        _p02c_source = _p02c_src
                        break
                    except Exception:
                        continue

                if _p02c_parseable:
                    # Anchor parseable encontrado → reanudamos el chunk.
                    resumed_snapshot = copy.deepcopy(snap)
                    resumed_snapshot["_anchor_pause_resolution"] = (
                        f"corrupted_date_recovered:{_p02c_source}"
                    )
                    resumed_snapshot["_anchor_pause_resolved_at"] = (
                        datetime.now(timezone.utc).isoformat()
                    )
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(resumed_snapshot, ensure_ascii=False), row_id),
                    )
                    logger.info(
                        f"[P0-2/CORRUPT-RECOVERED] Chunk {week_num} plan {meal_plan_id_str} "
                        f"reanudado: anchor parseable encontrado en "
                        f"{_p02c_source}={_p02c_parseable!r}."
                    )
                    continue

                _p02c_prev = int(_p02c_row.get("attempts") or 0)
                _p02c_new = _p02c_prev + 1
                try:
                    execute_sql_write(
                        "UPDATE meal_plans "
                        "SET plan_data = jsonb_set(COALESCE(plan_data, '{}'::jsonb), "
                        "'{_anchor_recovery_attempts}', to_jsonb(%s::int), true) "
                        "WHERE id = %s",
                        (_p02c_new, meal_plan_id_str),
                    )
                except Exception as _p02c_persist_err:
                    logger.warning(
                        f"[P0-2/CORRUPT-RECOVERY] No se pudo persistir attempts={_p02c_new} "
                        f"en plan {meal_plan_id_str}: {_p02c_persist_err}"
                    )

                if _p02c_new >= CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS:
                    logger.error(
                        f"[P0-2/CORRUPT-ESCALATE] Chunk {week_num} plan {meal_plan_id_str} "
                        f"agotó {_p02c_new} intentos con fecha corrupta no parseable. "
                        f"Escalando a dead_letter."
                    )
                    try:
                        _escalate_unrecoverable_chunk(
                            task_id=str(row_id),
                            user_id=user_id_str,
                            plan_id=meal_plan_id_str,
                            week_number=int(week_num) if week_num is not None else 0,
                            recovery_attempts=_p02c_new,
                            escalation_reason="unrecoverable_corrupted_date",
                        )
                    except Exception as _p02c_esc_err:
                        logger.error(
                            f"[P0-2/CORRUPT-ESCALATE] Falló escalación para chunk "
                            f"{row_id}: {_p02c_esc_err}"
                        )
                    continue

                logger.info(
                    f"[P0-2/CORRUPT-WAIT] Chunk {week_num} plan {meal_plan_id_str} "
                    f"sigue con fecha corrupta (intento {_p02c_new}/{CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS})."
                )
                continue

            # [P0-2] tz_unresolved: el chunk fue insertado por _enqueue_plan_chunk con
            # status='pending_user_action' porque _resolve_chunk_start_anchor agotó la
            # cadena de fallback (snapshot → profile → último plan) y devolvió
            # 'forced_8am_utc'. Cada tick re-intentamos resolver el ancla; típicamente
            # cuando el usuario abre la app, el frontend escribe `tz_offset_minutes` en
            # `user_profiles.health_profile`, lo que hace que la fuente 'profile_today'
            # responda con un offset válido. Si tras CHUNK_TZ_RECOVERY_MAX_ATTEMPTS
            # ticks no se ha resuelto, escalamos a dead_letter (el plan necesita
            # regeneración manual).
            if pause_reason == "tz_unresolved":
                anchor_start_dt, anchor_tz_min, anchor_source = _resolve_chunk_start_anchor(
                    user_id=user_id_str,
                    snapshot=snap,
                    meal_plan_id=meal_plan_id_str,
                    week_number=int(week_num) if week_num is not None else 0,
                )
                if anchor_source != "forced_8am_utc" and anchor_start_dt is not None:
                    # TZ resuelta. Recalculamos execute_after y reanudamos.
                    resumed_snapshot = copy.deepcopy(snap)
                    if isinstance(resumed_snapshot.get("form_data"), dict):
                        resumed_snapshot["form_data"]["tzOffset"] = anchor_tz_min
                        resumed_snapshot["form_data"]["tz_offset_minutes"] = anchor_tz_min
                        resumed_snapshot["form_data"]["_chunk_anchor_source"] = anchor_source
                    resumed_snapshot["_pantry_pause_resolution"] = "tz_recovered"
                    resumed_snapshot["_pantry_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()
                    # execute_after = midnight start_dt + days_offset + tz_min + 30min.
                    # Lo replicamos del cálculo original en _enqueue_plan_chunk.
                    days_offset_row = execute_sql_query(
                        "SELECT days_offset FROM plan_chunk_queue WHERE id = %s",
                        (row_id,),
                        fetch_one=True,
                    ) or {}
                    _do = int(days_offset_row.get("days_offset") or 0)
                    start_midnight = datetime.combine(
                        anchor_start_dt.date(), datetime.min.time()
                    ).replace(tzinfo=timezone.utc)
                    fresh_target = start_midnight + timedelta(
                        days=_do, minutes=anchor_tz_min + 30
                    )
                    fresh_target = max(
                        fresh_target,
                        datetime.now(timezone.utc) + timedelta(minutes=1),
                    )
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            execute_after = %s::timestamptz,
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (
                            fresh_target.isoformat(),
                            json.dumps(resumed_snapshot, ensure_ascii=False),
                            row_id,
                        ),
                    )
                    logger.info(
                        f"[P0-2/TZ-RECOVERED] Chunk {week_num} plan {meal_plan_id_str} "
                        f"reanudado: anchor_source={anchor_source} tz={anchor_tz_min}m "
                        f"execute_after={fresh_target.isoformat()}."
                    )
                    continue

                # TZ sigue sin resolverse. Incrementar contador y, al exceder
                # MAX_ATTEMPTS, escalar a dead_letter.
                _tz_prev = int(snap.get("_tz_recovery_attempts") or 0)
                _tz_new = _tz_prev + 1
                if _tz_new >= CHUNK_TZ_RECOVERY_MAX_ATTEMPTS:
                    logger.error(
                        f"[P0-2/TZ-ESCALATE] Chunk {week_num} plan {meal_plan_id_str} "
                        f"agotó {_tz_new} intentos sin TZ resoluble. Escalando a dead_letter."
                    )
                    try:
                        _escalate_unrecoverable_chunk(
                            task_id=str(row_id),
                            user_id=user_id_str,
                            plan_id=meal_plan_id_str or "",
                            week_number=int(week_num) if week_num is not None else 0,
                            recovery_attempts=_tz_new,
                            escalation_reason="unrecoverable_tz_unresolved",
                        )
                    except Exception as _tz_esc_err:
                        logger.error(
                            f"[P0-2/TZ-ESCALATE] Falló escalación para chunk "
                            f"{row_id}: {_tz_esc_err}"
                        )
                    continue

                # Persistir contador y enviar otro recordatorio al usuario (con cooldown).
                snap["_tz_recovery_attempts"] = _tz_new
                try:
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET pipeline_snapshot = %s::jsonb, updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(snap, ensure_ascii=False), row_id),
                    )
                except Exception as _tz_persist_err:
                    logger.warning(
                        f"[P0-2/TZ-WAIT] No se pudo persistir _tz_recovery_attempts="
                        f"{_tz_new} en chunk {row_id}: {_tz_persist_err}"
                    )
                try:
                    _maybe_notify_user_tz_unresolved(user_id_str)
                except Exception as _np_err:
                    logger.debug(
                        f"[P0-2/TZ-WAIT] Notify falló para {user_id_str}: {_np_err}"
                    )
                logger.info(
                    f"[P0-2/TZ-WAIT] Chunk {week_num} plan {meal_plan_id_str} "
                    f"sigue con TZ no resoluble (intento {_tz_new}/{CHUNK_TZ_RECOVERY_MAX_ATTEMPTS})."
                )
                continue

            # [P0-2] stale_snapshot: la causa es server-side (live fetch caído).
            # En cada tick (~15 min) intentamos un live-retry. Si pasa, refrescamos
            # los snapshots y desbloqueamos al instante; el usuario no recibe push.
            # [P0-1] stale_snapshot_live_unreachable: variante con snapshot >>TTL en la que SÍ
            # notificamos al usuario al pausar (puede accionar abriendo la app), y al escalar
            # marcamos _pantry_advisory_only=True para conservar validación existencial.
            if pause_reason in ("stale_snapshot", "stale_snapshot_live_unreachable"):
                try:
                    fresh_inv = get_user_inventory_net(user_id_str)
                except Exception as live_e:
                    fresh_inv = None
                    logger.debug(f"[P0-2/STALE-RETRY] Live retry falló para {user_id_str}: {live_e}")

                if fresh_inv is not None:
                    # Live se recuperó. Persistimos al snapshot y re-encolamos sin escalar a flex.
                    try:
                        meal_plan_id_row = execute_sql_query(
                            "SELECT meal_plan_id FROM plan_chunk_queue WHERE id = %s",
                            (row_id,),
                            fetch_one=True,
                        )
                        mpid = meal_plan_id_row.get("meal_plan_id") if meal_plan_id_row else None
                        if mpid:
                            _persist_fresh_pantry_to_chunks(row_id, str(mpid), fresh_inv, user_id=user_id_str)
                    except Exception as prop_e:
                        logger.warning(f"[P0-2/STALE-RETRY] No se pudo propagar inv fresco: {prop_e}")

                    resumed_snapshot = copy.deepcopy(snap)
                    resumed_snapshot["_pantry_pause_resolution"] = "live_recovered"
                    resumed_snapshot["_pantry_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(resumed_snapshot, ensure_ascii=False), row_id),
                    )
                    logger.info(
                        f"[P0-2/STALE-RECOVERED] Chunk {week_num} reanudado: live fetch volvió "
                        f"tras {paused_seconds//60} min en stale_snapshot."
                    )
                    continue

                # [P0-1] Live sigue caído. Tres ventanas:
                #   1) [0, ttl_hours): silencio total (puede ser fallo transitorio del backend).
                #   2) [ttl_hours, MAX_PAUSE_HOURS): un push recordatorio al usuario para que
                #      abra la app y refresque (acción del usuario suele resucitar el live).
                #      No degradamos todavía — preferimos preview-only del plan a generar a ciegas.
                #   3) [MAX_PAUSE_HOURS, ∞): escalada inevitable a flexible_mode + advisory_only,
                #      con un push final que avisa de la degradación.
                from constants import CHUNK_STALE_MAX_PAUSE_HOURS as _MAX_PAUSE
                from constants import CHUNK_STALE_PANTRY_DEEPLINK as _DL_REC
                paused_hours = paused_seconds / 3600

                if paused_hours >= _MAX_PAUSE:
                    # [P1-3] Activación vía helper canónico. advisory_only=True para la
                    # variante live_unreachable (preserva validación existencial contra snapshot).
                    degraded_snapshot = _activate_flexible_mode(
                        copy.deepcopy(snap),
                        reason="stale_snapshot_force_flex",
                        user_id=user_id_str,
                        week_num=week_num,
                        meal_plan_id=meal_plan_id_str,
                        advisory_only=(pause_reason == "stale_snapshot_live_unreachable"),
                    )
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            attempts = COALESCE(attempts, 0) + 1,
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(degraded_snapshot, ensure_ascii=False), row_id),
                    )
                    # Push final notificando degradación (solo en variante live_unreachable que ya
                    # había recibido pushes; stale_snapshot pura es server-side, no spamear).
                    if pause_reason == "stale_snapshot_live_unreachable":
                        try:
                            _dispatch_push_notification(
                                user_id=user_id_str,
                                title="Generamos tu plan con datos parciales",
                                body=(
                                    f"Tu nevera lleva {int(paused_hours)}h sin sincronizar. "
                                    f"Generamos los próximos días con la última versión disponible — "
                                    f"revísalos y ajusta lo necesario."
                                ),
                                url=_DL_REC,
                            )
                        except Exception as _np_e:
                            logger.warning(f"[P0-1/STALE-FORCE-FLEX] No se pudo notificar degradación: {_np_e}")
                    logger.warning(
                        f"[P0-1/STALE-FORCE-FLEX] Chunk {week_num} expiró tras {paused_hours:.1f}h en "
                        f"{pause_reason} sin recuperar live (max={_MAX_PAUSE}h). Re-encolando en flexible_mode "
                        f"(advisory_only={pause_reason == 'stale_snapshot_live_unreachable'})."
                    )
                    continue

                # Ventana intermedia: enviar UN push recordatorio (no spam) si tocó ttl_hours
                # y aún no enviamos el reminder.
                if (
                    paused_hours >= ttl_hours
                    and pause_reason == "stale_snapshot_live_unreachable"
                    and not snap.get("_pantry_pause_max_reminder_sent")
                ):
                    try:
                        _dispatch_push_notification(
                            user_id=user_id_str,
                            title="Tu plan sigue en pausa",
                            body=(
                                f"Llevamos {int(paused_hours)}h esperando que tu nevera se actualice. "
                                f"Ábrela para continuar tu plan."
                            ),
                            url=_DL_REC,
                        )
                        snap["_pantry_pause_max_reminder_sent"] = True
                        execute_sql_write(
                            "UPDATE plan_chunk_queue SET pipeline_snapshot = %s::jsonb, updated_at = NOW() WHERE id = %s",
                            (json.dumps(snap, ensure_ascii=False), row_id),
                        )
                    except Exception as _rem_e:
                        logger.warning(f"[P0-1/STALE-REMINDER] No se pudo enviar reminder: {_rem_e}")
                # Antes del TTL: silencio. No spameamos push en pausa server-side.
                continue

            # [P0-2] learning_zero_logs: causa = usuario no logueó comidas.
            # Recovery acelerado: si detectamos ≥N mutaciones de inventario, reanudamos
            # inmediatamente con _force_variety=True (no esperamos el TTL completo).
            # Si no hay señal y expira el TTL corto (4h), forzamos flexible_mode.
            if pause_reason == "learning_zero_logs":
                # Intentar detectar actividad de inventario reciente.
                try:
                    _zl_snap_fd = snap.get("form_data", {}) or {}
                    _zl_start = snap.get("_pantry_pause_started_at") or _zl_snap_fd.get("_plan_start_date", "")
                    _zl_activity = get_inventory_activity_since(user_id_str, _zl_start)
                    _zl_mutations = int(_zl_activity.get("consumption_mutations_count") or 0) if _zl_activity else 0
                except Exception as _zl_e:
                    _zl_mutations = 0
                    logger.debug(f"[P0-2/ZERO-LOG] Error midiendo actividad inventario {user_id_str}: {_zl_e}")

                if _zl_mutations >= CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS:
                    # Señal de inventario suficiente → reanudar con variety forzada.
                    resumed_snapshot = copy.deepcopy(snap)
                    resumed_snapshot["_pantry_pause_resolution"] = "inventory_proxy_resumed"
                    resumed_snapshot["_pantry_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()
                    resumed_snapshot["_inventory_proxy_mutations_at_resume"] = _zl_mutations
                    _zl_fd = resumed_snapshot.get("form_data", {})
                    _zl_fd["_force_variety"] = True
                    _zl_fd["_inventory_activity_proxy_used"] = True
                    _zl_fd["_inventory_activity_mutations"] = _zl_mutations
                    # Bloquear técnica del chunk previo
                    _zl_prev_tech = _zl_fd.get("_last_technique")
                    if _zl_prev_tech:
                        _zl_blocked = _zl_fd.get("_blocked_techniques", [])
                        if _zl_prev_tech not in _zl_blocked:
                            _zl_blocked.append(_zl_prev_tech)
                        _zl_fd["_blocked_techniques"] = _zl_blocked
                    resumed_snapshot["form_data"] = _zl_fd
                    resumed_snapshot["_chunk_lessons"] = {"degraded": True, "reason": "inventory_proxy_no_logs"}
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(resumed_snapshot, ensure_ascii=False), row_id),
                    )
                    logger.info(
                        f"[P0-2/ZERO-LOG-RESUMED] Chunk {week_num} reanudado: "
                        f"{_zl_mutations} mutaciones de inventario detectadas tras "
                        f"{paused_seconds // 60} min en pausa. Generando con _force_variety=True."
                    )
                    continue

                # Sin señal de inventario. Si expiró el TTL corto, forzar flexible_mode.
                if paused_seconds >= ttl_hours * 3600:
                    # [P1-3] Activación canónica. learning_flexible=True porque el chunk no tiene
                    # señales de aprendizaje y debe forzar variedad sin esperar logs.
                    degraded_snapshot = _activate_flexible_mode(
                        copy.deepcopy(snap),
                        reason="zero_log_force_flex",
                        user_id=user_id_str,
                        week_num=week_num,
                        meal_plan_id=meal_plan_id_str,
                        learning_flexible=True,
                        extra_form_data={
                            "_force_variety": True,
                            "_learning_forced": True,
                            "_learning_forced_reason": "no_signal_4h",
                        },
                        chunk_lessons={"degraded": True, "reason": "no_signal_4h"},
                    )
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            attempts = COALESCE(attempts, 0) + 1,
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(degraded_snapshot, ensure_ascii=False), row_id),
                    )
                    logger.warning(
                        f"[P0-2/ZERO-LOG-FORCE-FLEX] Chunk {week_num} expiró tras {ttl_hours}h en "
                        f"zero-log sin señal. Re-encolando en flexible_mode con _force_variety."
                    )
                    _dispatch_push_notification(
                        user_id=user_id_str,
                        title="Generando tu próximo bloque",
                        body=(
                            "Estamos generando tu próximo bloque sin info de qué comiste — "
                            "márcanos lo que comes para mejorar."
                        ),
                        url="/dashboard",
                    )
                    continue

                # Antes del TTL: silencio. El reminder genérico cae al final de este loop.
                # (No hacemos continue aquí para que el bloque de reminders aplique)

            # [P0-3] prev_chunk_not_concluded: el chunk previo aún no terminó en el calendario
            # del usuario al momento del pause (TZ desalineada, _plan_start_date corrupto, o
            # CHUNK_PROACTIVE_MARGIN_DAYS demasiado optimista). Recovery:
            #   1. Re-leer plan_data + snapshot y re-evaluar el temporal_gate.
            #   2. Si el día previo ya pasó (gate ahora ready=True) → reanudar como pending.
            #   3. Si TTL agotado sin resolución → escalar a flexible_mode con _force_variety
            #      (mejor un plan en flexible que congelado indefinidamente).
            if pause_reason == "prev_chunk_not_concluded":
                if meal_plan_id_str is None:
                    logger.warning(
                        f"[P0-3/PREV-CHUNK-RECOVERY] Chunk {row_id} sin meal_plan_id; saltando."
                    )
                    continue
                # Re-leer plan_data fresco (puede haber recibido days ya consumidos)
                try:
                    _p03_plan_row = execute_sql_query(
                        "SELECT plan_data FROM meal_plans WHERE id = %s",
                        (meal_plan_id_str,), fetch_one=True,
                    )
                    _p03_plan_data = (_p03_plan_row or {}).get("plan_data") or {}
                except Exception as _p03_read_err:
                    logger.warning(
                        f"[P0-3/PREV-CHUNK-RECOVERY] No se pudo leer plan_data {meal_plan_id_str}: {_p03_read_err}"
                    )
                    _p03_plan_data = {}

                # Re-evaluar el temporal gate con el snapshot actual y plan_data fresco.
                _p03_days_offset = int(snap.get("days_offset") or 0)
                _p03_form_data = (snap or {}).get("form_data") or {}
                # `days_offset` no siempre está en el snapshot top-level; intentar leer del row.
                try:
                    _p03_chunk_row = execute_sql_query(
                        "SELECT days_offset, week_number FROM plan_chunk_queue WHERE id = %s",
                        (row_id,), fetch_one=True,
                    )
                    if _p03_chunk_row:
                        _p03_days_offset = int(_p03_chunk_row.get("days_offset") or _p03_days_offset)
                except Exception:
                    pass

                try:
                    _p03_gate = _check_chunk_learning_ready(
                        user_id=user_id_str,
                        meal_plan_id=meal_plan_id_str,
                        week_number=int(week_num) if week_num is not None else 1,
                        days_offset=_p03_days_offset,
                        plan_data=_p03_plan_data,
                        snapshot=snap,
                    )
                except Exception as _p03_gate_err:
                    logger.warning(
                        f"[P0-3/PREV-CHUNK-RECOVERY] Re-eval del gate falló para chunk "
                        f"{row_id}: {_p03_gate_err}. Esperando próximo tick."
                    )
                    continue

                _p03_gate_reason = _p03_gate.get("reason")
                _p03_now_ready = (
                    bool(_p03_gate.get("ready"))
                    and _p03_gate_reason != "prev_chunk_day_not_yet_elapsed"
                )

                if _p03_now_ready:
                    resumed_snapshot = copy.deepcopy(snap)
                    resumed_snapshot["_pantry_pause_resolution"] = "prev_chunk_concluded"
                    resumed_snapshot["_pantry_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()
                    # Limpiar flags de pause para que el worker no los lea como guardia.
                    for _k in (
                        "_pantry_pause_reason",
                        "_pantry_pause_started_at",
                        "_pantry_pause_ttl_hours",
                        "_pantry_pause_reminder_hours",
                        "_pantry_pause_reminders",
                    ):
                        resumed_snapshot.pop(_k, None)
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(resumed_snapshot, ensure_ascii=False), row_id),
                    )
                    logger.info(
                        f"[P0-3/PREV-CHUNK-RECOVERED] Chunk {week_num} plan {meal_plan_id_str} "
                        f"reanudado: gate temporal ahora ready=True (reason={_p03_gate_reason})."
                    )
                    continue

                # TTL agotado sin que el día previo haya transcurrido → escalar a flexible_mode
                # para no congelar el plan indefinidamente.
                if paused_seconds >= ttl_hours * 3600:
                    degraded_snapshot = _activate_flexible_mode(
                        copy.deepcopy(snap),
                        reason="prev_chunk_not_concluded_ttl",
                        user_id=user_id_str,
                        week_num=week_num,
                        meal_plan_id=meal_plan_id_str,
                        learning_flexible=True,
                        extra_form_data={
                            "_force_variety": True,
                            "_learning_forced": True,
                            "_learning_forced_reason": "prev_chunk_not_concluded_ttl",
                        },
                        chunk_lessons={"degraded": True, "reason": "prev_chunk_not_concluded_ttl"},
                    )
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            attempts = COALESCE(attempts, 0) + 1,
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(degraded_snapshot, ensure_ascii=False), row_id),
                    )
                    logger.warning(
                        f"[P0-3/PREV-CHUNK-FORCE-FLEX] Chunk {week_num} expiró tras {ttl_hours}h "
                        f"esperando que concluya el chunk previo. Re-encolando en flexible_mode."
                    )
                    try:
                        _dispatch_push_notification(
                            user_id=user_id_str,
                            title="Generamos tu próximo bloque",
                            body=(
                                "El bloque previo no se terminó de marcar a tiempo. "
                                "Generamos los próximos días con la mejor info disponible — "
                                "ajústalos en el diario si hace falta."
                            ),
                            url="/dashboard",
                        )
                    except Exception as _p03_push_err:
                        logger.warning(
                            f"[P0-3/PREV-CHUNK-FORCE-FLEX] No se pudo notificar: {_p03_push_err}"
                        )
                    continue

                # Antes del TTL: continuar pausado. El reminder genérico al final del loop
                # se encarga del push intermedio si aplica.

            # [P0-C] final_inventory_*: recovery proactivo cuando el usuario añade los
            # ingredientes faltantes a su nevera. Antes este caso esperaba el TTL completo
            # (2h por defecto) antes de escalar a flexible_mode incluso si el usuario los
            # añadía 5 minutos después de la pausa. Ahora cada tick del cron:
            #   1. Aplica grace period (5 min default) para no reintentar en el mismo
            #      segundo que la pausa ocurrió.
            #   2. Intenta live fetch. Si falla, fall-through al TTL existente (peor caso
            #      preservado: 2h y escalación a flexible_mode).
            #   3. Si hay `_pantry_pause_missing_ingredients` persistidos, compara contra
            #      la pantry actual usando bases canónicas; si TODOS están cubiertos,
            #      re-encola como pending. Si no hay lista (live fetch original cayó sin
            #      saber qué faltaba), confiamos en el live exitoso de ahora y resumimos.
            #   4. Cap de retries: tras MAX_RECOVERY_ATTEMPTS, dejamos al chunk seguir el
            #      TTL/escalation existente para evitar loops si el LLM re-genera siempre
            #      ingredientes que el usuario no tiene.
            _P0C_REASONS = (
                "final_inventory_unavailable",
                "final_inventory_missing",
                "flexible_live_unreachable",
                "flexible_validation_error",
            )
            if pause_reason in _P0C_REASONS:
                _p0c_grace_seconds = int(CHUNK_FINAL_VALIDATION_RECOVERY_GRACE_MINUTES) * 60
                _p0c_attempts = int(snap.get("_pantry_pause_recovery_attempts") or 0)
                if (
                    paused_seconds >= _p0c_grace_seconds
                    and _p0c_attempts < int(CHUNK_FINAL_VALIDATION_MAX_RECOVERY_ATTEMPTS)
                ):
                    try:
                        _p0c_live = get_user_inventory_net(user_id_str)
                    except Exception as _p0c_live_err:
                        _p0c_live = None
                        logger.debug(
                            f"[P0-C/RECOVERY] Live fetch falló para {user_id_str} "
                            f"chunk {week_num}: {_p0c_live_err}. Fallback a TTL existente."
                        )

                    if _p0c_live is not None:
                        _p0c_missing = snap.get("_pantry_pause_missing_ingredients") or []
                        if _p0c_missing:
                            _p0c_covered, _p0c_still = _pantry_covers_missing(_p0c_missing, _p0c_live)
                        else:
                            # Sin lista de faltantes (live original cayó); el live exitoso
                            # de ahora es señal suficiente para reintentar — el worker
                            # re-validará y, si vuelve a fallar, ahora SÍ persistirá la
                            # lista para el próximo recovery.
                            _p0c_covered, _p0c_still = True, []

                        if _p0c_covered:
                            resumed_snapshot = copy.deepcopy(snap)
                            resumed_snapshot["_pantry_pause_resolution"] = "missing_ingredients_covered"
                            resumed_snapshot["_pantry_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()
                            resumed_snapshot["_pantry_pause_recovery_attempts"] = _p0c_attempts + 1
                            execute_sql_write(
                                """
                                UPDATE plan_chunk_queue
                                SET status = 'pending',
                                    execute_after = NOW(),
                                    pipeline_snapshot = %s::jsonb,
                                    updated_at = NOW()
                                WHERE id = %s
                                """,
                                (json.dumps(resumed_snapshot, ensure_ascii=False), row_id),
                            )
                            logger.info(
                                f"[P0-C/RECOVERED] Chunk {week_num} reanudado tras "
                                f"{paused_seconds//60} min en {pause_reason}: pantry actual "
                                f"cubre {len(_p0c_missing)} item(s) faltantes "
                                f"(attempt={_p0c_attempts + 1}/{int(CHUNK_FINAL_VALIDATION_MAX_RECOVERY_ATTEMPTS)})."
                            )
                            continue
                        else:
                            # Algunos siguen faltando: solo bumpeamos el contador para
                            # dejar telemetría visible y caemos al TTL existente.
                            logger.debug(
                                f"[P0-C/RECOVERY] Chunk {week_num} aún incompleto: "
                                f"{len(_p0c_still)} item(s) sin cubrir. Esperando próximo tick."
                            )
                elif _p0c_attempts >= int(CHUNK_FINAL_VALIDATION_MAX_RECOVERY_ATTEMPTS):
                    logger.warning(
                        f"[P0-C/RECOVERY] Chunk {week_num} agotó "
                        f"{int(CHUNK_FINAL_VALIDATION_MAX_RECOVERY_ATTEMPTS)} reintentos de "
                        f"recovery. Cayendo al TTL existente para escalar a flexible_mode."
                    )
                # Fall-through al bloque de reminders/TTL existente abajo (no continue):
                # si el usuario nunca añade los ingredientes, el chunk seguirá la ruta
                # original de escalación a flexible_mode tras el TTL.

            # [P0-4] empty_pantry_proactive: el chunk fue insertado por
            # `_enqueue_plan_chunk` con esta reason cuando el live-probe al enqueue
            # detectó items < CHUNK_MIN_FRESH_PANTRY_ITEMS. Cada tick re-probamos:
            # si el usuario restockeó, reanudamos inmediatamente; si no, caemos al
            # TTL existente que escala a flexible_mode (preserva comportamiento del
            # `empty_pantry` reactivo). NO modificamos la rama de `empty_pantry`
            # legacy (worker reactivo) para evitar cambiar comportamiento existente.
            if pause_reason == "empty_pantry_proactive":
                try:
                    _p04_live = get_user_inventory_net(user_id_str)
                except Exception as _p04_live_err:
                    _p04_live = None
                    logger.debug(
                        f"[P0-4/RECOVERY] Live fetch falló para {user_id_str} "
                        f"chunk {week_num}: {_p04_live_err}. Esperando próximo tick."
                    )
                if _p04_live is not None and _count_meaningful_pantry_items(_p04_live) >= CHUNK_MIN_FRESH_PANTRY_ITEMS:
                    resumed_snapshot = copy.deepcopy(snap)
                    resumed_snapshot["_pantry_pause_resolution"] = "pantry_restocked"
                    resumed_snapshot["_pantry_pause_resolved_at"] = datetime.now(timezone.utc).isoformat()
                    if isinstance(resumed_snapshot.get("form_data"), dict):
                        resumed_snapshot["form_data"]["current_pantry_ingredients"] = _p04_live
                        resumed_snapshot["form_data"]["_pantry_captured_at"] = datetime.now(timezone.utc).isoformat()
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            execute_after = NOW(),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(resumed_snapshot, ensure_ascii=False), row_id),
                    )
                    logger.info(
                        f"[P0-4/RECOVERED] Chunk {week_num} plan {meal_plan_id_str} "
                        f"reanudado tras {paused_seconds//60} min: pantry restockeado "
                        f"({_count_meaningful_pantry_items(_p04_live)} items >= "
                        f"{CHUNK_MIN_FRESH_PANTRY_ITEMS})."
                    )
                    continue
                # Si live falló o sigue por debajo del mínimo, caemos al TTL/reminders
                # existente. La escalación a flexible_mode tras `ttl_hours` preserva
                # el comportamiento histórico de empty_pantry (no cambio de UX final).

            # empty_pantry / otros: mantenemos el comportamiento original (12h TTL + push).
            if paused_seconds >= ttl_hours * 3600:
                # [P1-3] Activación canónica vía helper.
                degraded_snapshot = _activate_flexible_mode(
                    copy.deepcopy(snap),
                    reason="degraded_flexible_meal",
                    user_id=user_id_str,
                    week_num=week_num,
                    meal_plan_id=meal_plan_id_str,
                )

                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET status = 'pending',
                        attempts = COALESCE(attempts, 0) + 1,
                        execute_after = NOW(),
                        pipeline_snapshot = %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (json.dumps(degraded_snapshot, ensure_ascii=False), row_id),
                )
                logger.warning(
                    f"[P1-3/PANTRY] Chunk {week_num} expiró en pending_user_action. "
                    f"Re-encolando en modo flexible."
                )
                _dispatch_push_notification(
                    user_id=user_id_str,
                    title="Seguimos con un plato flexible",
                    body="Tu chunk seguía en pausa por nevera vacía. Lo reintentaremos con un plato flexible para no bloquear tu plan.",
                    url="/dashboard",
                )
                continue

            next_reminder_threshold = reminder_hours * 3600 * (reminders_sent + 1)
            if reminders_sent < CHUNK_PANTRY_EMPTY_MAX_REMINDERS and paused_seconds >= next_reminder_threshold:
                reminder_snapshot = copy.deepcopy(snap)
                reminder_snapshot["_pantry_pause_reminders"] = reminders_sent + 1
                reminder_snapshot["_pantry_pause_last_reminder_at"] = datetime.now(timezone.utc).isoformat()
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET pipeline_snapshot = %s::jsonb,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (json.dumps(reminder_snapshot, ensure_ascii=False), row_id),
                )
                _dispatch_push_notification(
                    user_id=user_id_str,
                    title="Tu chunk sigue esperando tu nevera",
                    body="Actualiza 'Mi Nevera' para continuar con el siguiente bloque del plan. Si no, usaremos una opción flexible más adelante.",
                    url="/dashboard",
                )
    except Exception as e:
        logger.warning(f"[P1-3/PANTRY] Error recuperando chunks pausados por pantry vacío: {e}")


def _recover_failed_chunks_for_long_plans() -> None:
    """
    [P0-1-RECOVERY] Cron task que detecta chunks failed en CUALQUIER plan (>=7d) con
    días remanentes y los re-encola con chunk_kind="catchup". Garantiza que el plan
    se complete aunque el usuario no abra la app.

    Antes:
      - Umbral 15d+: planes de 7d quedaban sin recuperación (un chunk 2 fallido
        dejaba al usuario con 3/7 días generados para siempre).
      - Sin min_age: chunks que acababan de fallar en el mismo tick eran reactivados,
        cortocircuitando el backoff exponencial.
      - Sin tope de recoveries: un chunk irrecuperable loopeaba cada 15 min sin que
        el usuario fuera notificado nunca.

    Ahora:
      1. Solo recoge chunks con `updated_at` más viejo que CHUNK_RECOVERY_MIN_AGE_MINUTES
         (deja respirar al backoff normal).
      2. Lee `learning_metrics->>'recovery_attempts'` como contador persistente.
      3. Si recovery_attempts < CHUNK_MAX_RECOVERY_ATTEMPTS → incrementa contador y
         re-encola con chunk_kind='catchup'.
      4. Si recovery_attempts >= CHUNK_MAX_RECOVERY_ATTEMPTS → escala: marca el chunk
         como dead_letter permanente, anota en plan_data._recovery_exhausted, manda
         push al usuario para que regenere manualmente.
    """
    try:
        # [P0-1-RECOVERY/A] Plan-still-active: COALESCE con meal_plans.created_at como
        # fallback cuando plan_data->>'grocery_start_date' es NULL. En producción ~53% de
        # los meal_plans no tienen grocery_start_date persistido (bug de creación que se
        # cierra en P0-1-C/Backfill); sin este COALESCE el cron descartaba todos esos
        # planes y los chunks failed quedaban huérfanos de recovery. Margen de +7 días
        # cubre catchup chunks que se generan tras la fecha objetivo nominal.
        failed_candidates = execute_sql_query("""
            SELECT
                q.id, q.user_id, q.meal_plan_id, q.week_number, q.days_offset, q.days_count,
                q.pipeline_snapshot,
                COALESCE(q.learning_metrics, '{}'::jsonb) AS learning_metrics,
                (p.plan_data->>'total_days_requested')::int as total_days,
                COALESCE(
                    (p.plan_data->>'grocery_start_date')::timestamptz,
                    p.created_at
                )::text as plan_start_effective
            FROM plan_chunk_queue q
            JOIN meal_plans p ON q.meal_plan_id = p.id
            WHERE q.status = 'failed'
              AND (p.plan_data->>'total_days_requested')::int >= 7
              AND (
                  COALESCE(
                      (p.plan_data->>'grocery_start_date')::timestamptz,
                      p.created_at
                  ) + (((p.plan_data->>'total_days_requested')::int + 7) * interval '1 day')
              ) > NOW()
              AND q.updated_at < (NOW() - make_interval(mins => %s))
            ORDER BY q.updated_at ASC
            LIMIT %s
        """, (CHUNK_RECOVERY_MIN_AGE_MINUTES, CHUNK_RECOVERY_BATCH_LIMIT), fetch_all=True) or []

        if not failed_candidates:
            # Aun sin candidatos seguimos detectando chunks atrasados.
            _detect_and_escalate_stuck_chunks()
            return

        for row in failed_candidates:
            task_id = row['id']
            user_id = row['user_id']
            plan_id = row['meal_plan_id']
            week_number = row['week_number']
            days_offset = row['days_offset']
            days_count = row['days_count']
            snapshot = row['pipeline_snapshot']
            if isinstance(snapshot, str):
                snapshot = json.loads(snapshot)

            lm = row.get('learning_metrics') or {}
            if isinstance(lm, str):
                try:
                    lm = json.loads(lm)
                except Exception:
                    lm = {}
            try:
                prior_recovery_attempts = int(lm.get('recovery_attempts') or 0)
            except (TypeError, ValueError):
                prior_recovery_attempts = 0

            if prior_recovery_attempts >= CHUNK_MAX_RECOVERY_ATTEMPTS:
                _escalate_unrecoverable_chunk(
                    task_id=task_id,
                    user_id=str(user_id),
                    plan_id=str(plan_id),
                    week_number=week_number,
                    recovery_attempts=prior_recovery_attempts,
                )
                continue

            # Incrementar el contador en learning_metrics ANTES de re-encolar.
            # _enqueue_plan_chunk hace UPSERT que NO toca learning_metrics, así que
            # el contador persiste a través del reset attempts=0.
            new_recovery_attempts = prior_recovery_attempts + 1
            try:
                execute_sql_write("""
                    UPDATE plan_chunk_queue
                    SET learning_metrics = COALESCE(learning_metrics, '{}'::jsonb)
                        || jsonb_build_object(
                            'recovery_attempts', %s::int,
                            'last_recovery_at', NOW()::text
                        )
                    WHERE id = %s
                """, (new_recovery_attempts, task_id))
            except Exception as bump_err:
                logger.warning(
                    f"[P0-1-RECOVERY] No se pudo persistir recovery_attempts={new_recovery_attempts} "
                    f"en chunk {task_id}: {bump_err}. Continuando con re-encolado."
                )

            logger.info(
                f"🔄 [P0-1-RECOVERY] Recuperando chunk failed (id={task_id}) "
                f"week={week_number} plan={plan_id} total_days={row.get('total_days')} "
                f"recovery_attempt={new_recovery_attempts}/{CHUNK_MAX_RECOVERY_ATTEMPTS}. "
                f"Re-encolando como catchup..."
            )

            _enqueue_plan_chunk(
                user_id=str(user_id),
                meal_plan_id=str(plan_id),
                week_number=week_number,
                days_offset=days_offset,
                days_count=days_count,
                pipeline_snapshot=snapshot,
                chunk_kind="catchup"
            )

        # [P1-3] Detección y escalado proactivo de chunks atrasados (lag > 24h)
        _detect_and_escalate_stuck_chunks()

    except Exception as e:
        logger.error(f"❌ [P0-1-RECOVERY] Error en cron de recuperación: {e}")


def _escalate_unrecoverable_chunk(
    task_id: str,
    user_id: str,
    plan_id: str,
    week_number: int,
    recovery_attempts: int,
    escalation_reason: str = "recovery_exhausted",
) -> None:
    """[P0-1-RECOVERY] Marca un chunk como dead_letter permanente tras agotar recoveries.

    Acciones:
      1. UPDATE plan_chunk_queue → status sigue 'failed' pero `dead_lettered_at`,
         `dead_letter_reason=<escalation_reason>`, anota en learning_metrics.
      2. UPDATE meal_plans.plan_data._recovery_exhausted_chunks (lista de week_numbers
         escalados) para que el frontend pueda mostrar banner "regenera tu plan".
      3. Dispara push notification al usuario explicando la situación.

    Argumento `escalation_reason` controla el copy del push y la URL del deeplink:
      - "recovery_exhausted" (default): chunk falló >=CHUNK_MAX_RECOVERY_ATTEMPTS veces
        en pipeline. Mensaje: "regenera tu plan con tu nevera actual".
      - "unrecoverable_missing_anchor" [P1-2]: el plan no tiene fecha de inicio
        recuperable (ni `_plan_start_date` en snapshot, ni `grocery_start_date` en
        plan_data, ni `created_at` en meal_plans). Mensaje pide regenerar el plan
        desde cero porque sin ancla no podemos calcular calendario de chunks.
      - "unrecoverable_corrupted_date" [P0-2 FIX]: el plan tiene anchors pero
        corruptos (no parseables con `safe_fromisoformat`). Mensaje pide regenerar
        porque la fuente del dato quedó dañada y el recovery cron agotó intentos.

    No re-encola el chunk: la única salida es regeneración manual del plan por parte
    del usuario o intervención de soporte.
    """
    try:
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET dead_lettered_at = COALESCE(dead_lettered_at, NOW()),
                dead_letter_reason = COALESCE(dead_letter_reason, %s),
                learning_metrics = COALESCE(learning_metrics, '{}'::jsonb)
                    || jsonb_build_object(
                        'recovery_attempts', %s::int,
                        'recovery_exhausted_at', NOW()::text,
                        'escalation_reason', %s::text
                    ),
                updated_at = NOW()
            WHERE id = %s
        """, (escalation_reason, recovery_attempts, escalation_reason, task_id))
    except Exception as upd_err:
        logger.error(
            f"[P0-1-RECOVERY] Falló UPDATE dead_letter en chunk {task_id}: {upd_err}"
        )

    try:
        execute_sql_write("""
            UPDATE meal_plans
            SET plan_data = COALESCE(plan_data, '{}'::jsonb)
                || jsonb_build_object(
                    '_recovery_exhausted_chunks',
                    COALESCE(plan_data->'_recovery_exhausted_chunks', '[]'::jsonb)
                        || jsonb_build_array(jsonb_build_object(
                            'week_number', %s::int,
                            'escalated_at', NOW()::text,
                            'recovery_attempts', %s::int,
                            'escalation_reason', %s::text
                        ))
                )
                || jsonb_build_object(
                    '_user_action_required',
                    jsonb_build_object(
                        'reason', %s::text,
                        'week_number', %s::int,
                        'requested_at', NOW()::text
                    )
                )
            WHERE id = %s
        """, (
            week_number, recovery_attempts, escalation_reason,
            escalation_reason, week_number,
            plan_id,
        ))
    except Exception as plan_err:
        logger.warning(
            f"[P0-1-RECOVERY] No se pudo anotar _recovery_exhausted_chunks/_user_action_required "
            f"en plan {plan_id}: {plan_err}"
        )

    logger.error(
        f"❌ [P0-1-RECOVERY/ESCALATED] Chunk task_id={task_id} (plan={plan_id} week={week_number}) "
        f"escalation_reason={escalation_reason} recovery_attempts={recovery_attempts}. "
        f"Dead-lettered permanentemente. Se notifica al usuario {user_id}."
    )
    # [P1-1] Telemetría dedicada para correlacionar escalación → skip de lecciones
    # fantasma en chunks posteriores. Cualquier chunk N+k que lea este plan_data debe
    # excluir week_number de su agregado de lecciones (gestionado por
    # `_filter_lessons_excluding_dead_lettered`).
    logger.warning(
        f"[P1-1/DEAD-LETTERED] plan={plan_id} week={week_number} reason={escalation_reason}; "
        f"chunks posteriores excluirán sus lecciones del prompt."
    )

    # [P1-2] Copy y deeplink dependen del escalation_reason. Para missing_anchor el
    # plan está estructuralmente roto: regenerar es la única acción del usuario.
    if escalation_reason == "unrecoverable_missing_anchor":
        push_title = "Tu plan necesita regenerarse"
        push_body = (
            "Detectamos un problema técnico con tu plan que impide continuar "
            "generando los próximos días. Tócalo para regenerarlo con tu nevera actual."
        )
        push_url = "/dashboard?action_required=missing_anchor"
    elif escalation_reason == "unrecoverable_corrupted_date":
        # [P0-2 FIX] Anchor existente pero corrupto al punto de no parsear. Mismo
        # CTA que missing_anchor (regenerar) pero deeplink distinto para que el
        # frontend pueda mostrar copy específico ("la fecha de inicio quedó dañada
        # tras una migración") si lo necesita en el futuro.
        push_title = "Tu plan necesita regenerarse"
        push_body = (
            "Detectamos datos inválidos en la fecha de inicio de tu plan. "
            "Tócalo para regenerarlo con tu nevera actual."
        )
        push_url = "/dashboard?action_required=corrupted_date"
    else:
        push_title = "Tu plan necesita atención"
        push_body = (
            "No pudimos completar parte de tu plan automáticamente. "
            "Abre Mealfit y regenera tu plan para que volvamos a generarlo con tu nevera actual."
        )
        push_url = "/dashboard?recovery_exhausted=1"

    try:
        _dispatch_push_notification(
            user_id=user_id,
            title=push_title,
            body=push_body,
            url=push_url,
        )
    except Exception as push_err:
        logger.warning(
            f"[P0-1-RECOVERY] No se pudo despachar push de escalación a {user_id}: {push_err}"
        )


def _finalize_zombie_partial_plans() -> int:
    """[P0-A] Cierra planes que se quedaron en `generation_status='partial'` indefinidamente.

    Se considera "zombie" un plan que cumple TODAS estas condiciones:
      1. `generation_status` ∈ ('partial', 'generating_next').
      2. `created_at < NOW() - CHUNK_ZOMBIE_PARTIAL_MIN_AGE_HOURS` (margen para recovery normal).
      3. NO tiene chunks vivos en plan_chunk_queue. "Vivo" = estado que aún puede commitear:
         pending, processing, stale, pending_user_action. También se considera vivo un
         `failed` que NO ha sido dead-lettered (puede ser rescatado por
         `_recover_failed_chunks_for_long_plans`). Si solo quedan estados terminales
         (completed, cancelled, failed-with-dead_lettered_at), el plan no avanzará más.

    Acción según los días materializados en plan_data.days:
      - len(days) > 0 → marcar `generation_status='complete_partial'` (mismo terminal
        que usa el path de degraded zombie en línea 13744). El usuario conserva los
        días que sí se generaron y el frontend deja de mostrar el spinner.
      - len(days) == 0 → marcar `generation_status='failed'`. Esto solo ocurre en el
        caso patológico donde ni el chunk inicial sincrónico se persistió.

    En ambos casos sellamos `_partial_finalized_at` y `_partial_finalized_reason` en
    plan_data para auditoría y para que tests puedan diferenciarlo del path normal.

    Returns:
        Número de planes finalizados (útil para tests y telemetría).
    """
    try:
        candidates = execute_sql_query(
            """
            SELECT
                mp.id::text AS plan_id,
                mp.user_id::text AS user_id,
                COALESCE(jsonb_array_length(mp.plan_data->'days'), 0) AS days_count,
                COALESCE((mp.plan_data->>'total_days_requested')::int, 7) AS total_days_requested
            FROM meal_plans mp
            WHERE mp.plan_data->>'generation_status' IN ('partial', 'generating_next')
              AND mp.created_at < NOW() - make_interval(hours => %s)
              AND NOT EXISTS (
                  SELECT 1 FROM plan_chunk_queue q
                  WHERE q.meal_plan_id = mp.id
                    AND (
                        q.status IN ('pending', 'processing', 'stale', 'pending_user_action')
                        OR (q.status = 'failed' AND q.dead_lettered_at IS NULL)
                    )
              )
            ORDER BY mp.created_at ASC
            LIMIT %s
            """,
            (CHUNK_ZOMBIE_PARTIAL_MIN_AGE_HOURS, CHUNK_ZOMBIE_PARTIAL_BATCH_LIMIT),
            fetch_all=True,
        ) or []
    except Exception as e:
        logger.error(f"[P0-A/ZOMBIE-PARTIAL] SELECT de candidatos falló: {e}")
        return 0

    if not candidates:
        return 0

    finalized = 0
    for row in candidates:
        plan_id = row.get("plan_id")
        user_id = row.get("user_id")
        days_count = int(row.get("days_count") or 0)
        total_days_requested = int(row.get("total_days_requested") or 0)

        if days_count > 0:
            new_status = "complete_partial"
            reason = "all_chunks_terminated_partial_data"
        else:
            new_status = "failed"
            reason = "all_chunks_terminated_no_days"

        try:
            execute_sql_write(
                """
                UPDATE meal_plans
                SET plan_data = COALESCE(plan_data, '{}'::jsonb)
                    || jsonb_build_object(
                        'generation_status', %s::text,
                        '_partial_finalized_at', NOW()::text,
                        '_partial_finalized_reason', %s::text
                    )
                WHERE id = %s
                  AND plan_data->>'generation_status' IN ('partial', 'generating_next')
                """,
                (new_status, reason, plan_id),
            )
            finalized += 1
            logger.warning(
                f"[P0-A/ZOMBIE-PARTIAL] Plan {plan_id} (user={user_id}) finalizado: "
                f"{days_count}/{total_days_requested} días → status='{new_status}' "
                f"(reason={reason})."
            )
        except Exception as upd_err:
            logger.error(
                f"[P0-A/ZOMBIE-PARTIAL] UPDATE falló para plan {plan_id}: {upd_err}"
            )

    if finalized:
        logger.info(
            f"[P0-A/ZOMBIE-PARTIAL] Finalizados {finalized} planes zombie en este tick "
            f"(min_age={CHUNK_ZOMBIE_PARTIAL_MIN_AGE_HOURS}h, "
            f"batch_limit={CHUNK_ZOMBIE_PARTIAL_BATCH_LIMIT})."
        )
    return finalized


def _recover_orphan_chunk_reservations() -> int:
    """[P1-A] Cleanup: libera reservas de inventario huérfanas de chunks ya terminados.

    `release_chunk_reservations` ahora es transaccional (P1-A all-or-nothing), pero
    pueden quedar:
      - Reservas legacy de incidentes anteriores (pre-fix) donde la liberación fue parcial.
      - Casos donde un caller olvidó invocar `release_chunk_reservations` antes de
        marcar el chunk como cancelado/completado.
      - Chunks eliminados de `plan_chunk_queue` por GC sin que sus reservas se hayan
        limpiado (defensa en profundidad).

    Lógica:
      1. SELECT filas de `user_inventory` con `reserved_quantity > 0` y `reservation_details`
         no vacío.
      2. Extrae los `chunk_id` referenciados (keys con prefijo `chunk:<id>:`).
      3. Para cada `chunk_id`: si NO existe en `plan_chunk_queue`, o existe en estado
         terminal (`completed`/`cancelled`) con edad >= `CHUNK_ORPHAN_RESERVATION_MIN_TERMINAL_AGE_HOURS`,
         invoca `release_chunk_reservations(user_id, chunk_id)` (que es atómico).

    Returns:
        Total de keys de reserva liberadas (suma de todos los chunks limpiados).
    """
    try:
        rows = execute_sql_query(
            """
            SELECT id, user_id::text AS user_id, reservation_details
            FROM user_inventory
            WHERE reserved_quantity > 0
              AND reservation_details IS NOT NULL
              AND reservation_details::text <> '{}'
            LIMIT %s
            """,
            (CHUNK_ORPHAN_RESERVATION_BATCH_LIMIT,),
            fetch_all=True,
        ) or []
    except Exception as e:
        logger.error(f"[P1-A/CLEANUP] SELECT inicial falló: {e}")
        return 0

    if not rows:
        return 0

    # Map chunk_id → set of (user_id) que lo referencia. Usamos set para deduplicar
    # cuando varias filas del mismo usuario referencian el mismo chunk.
    chunk_user_pairs: set = set()
    for row in rows:
        details_raw = row.get("reservation_details")
        # _normalize_reservation_details está en db_inventory pero no la importamos
        # aquí porque el módulo ya importa release_chunk_reservations al top. Hacemos
        # el parse inline para no inflar imports.
        if isinstance(details_raw, str):
            try:
                details = json.loads(details_raw)
            except Exception:
                continue
        elif isinstance(details_raw, dict):
            details = details_raw
        else:
            continue
        for key in details.keys():
            if not isinstance(key, str) or not key.startswith("chunk:"):
                continue
            parts = key.split(":")
            # Formato: chunk:<chunk_id>:meal:<meal_token>
            if len(parts) < 4:
                continue
            chunk_id = parts[1]
            user_id = row.get("user_id")
            if chunk_id and user_id:
                chunk_user_pairs.add((str(user_id), str(chunk_id)))

    if not chunk_user_pairs:
        return 0

    chunk_ids = list({chunk_id for _uid, chunk_id in chunk_user_pairs})

    # Verificar el estado de cada chunk en plan_chunk_queue. Un chunk es "terminable":
    #   - existe y está en ('completed', 'cancelled') con edad >= MIN_TERMINAL_AGE_HOURS
    #   - O ya no existe (DELETE-y-cleanup huérfano).
    try:
        existing_rows = execute_sql_query(
            """
            SELECT id::text AS id, status,
                   (updated_at < NOW() - make_interval(hours => %s)) AS is_old_enough
            FROM plan_chunk_queue
            WHERE id::text = ANY(%s)
            """,
            (CHUNK_ORPHAN_RESERVATION_MIN_TERMINAL_AGE_HOURS, chunk_ids),
            fetch_all=True,
        ) or []
    except Exception as e:
        logger.error(f"[P1-A/CLEANUP] SELECT plan_chunk_queue falló: {e}")
        return 0

    existing_by_id = {r["id"]: r for r in existing_rows}
    terminal_statuses = {"completed", "cancelled"}
    cleanable_chunk_ids: set = set()
    for chunk_id in chunk_ids:
        info = existing_by_id.get(chunk_id)
        if info is None:
            # Chunk borrado de plan_chunk_queue → reserva huérfana definitiva.
            cleanable_chunk_ids.add(chunk_id)
        elif info.get("status") in terminal_statuses and info.get("is_old_enough"):
            cleanable_chunk_ids.add(chunk_id)
        # Resto (pending/processing/stale o terminal-pero-reciente): NO tocar.

    if not cleanable_chunk_ids:
        return 0

    from db_inventory import release_chunk_reservations as _release
    released_total = 0
    cleaned_chunks = 0
    for user_id, chunk_id in chunk_user_pairs:
        if chunk_id not in cleanable_chunk_ids:
            continue
        try:
            n = _release(user_id, chunk_id)
            released_total += int(n or 0)
            if n:
                cleaned_chunks += 1
        except Exception as e:
            logger.warning(
                f"[P1-A/CLEANUP] release_chunk_reservations falló para "
                f"chunk={chunk_id} user={user_id}: {e}"
            )

    if released_total:
        logger.info(
            f"[P1-A/CLEANUP] Liberadas {released_total} reservas huérfanas en "
            f"{cleaned_chunks} chunk(s) terminales/eliminados."
        )
    return released_total


def _nudge_chronic_zero_log_users() -> int:
    """[P1-2/ZERO-LOG-NUDGE] Aviso PROACTIVO al usuario antes de que el chunk N+1
    falle por zero-log.

    Antes el push solo se mandaba cuando el chunk N+1 ya estaba defiriéndose (path en
    L6713-L6739) o pausado (L6741+). El usuario descubría el problema cuando abría la
    app y no veía plan nuevo. Ahora un cron busca usuarios con plan activo + cero
    `consumed_meals` en los últimos `CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS` y les manda
    un nudge antes de que el siguiente bloque tropiece.

    Cooldown: `CHUNK_ZERO_LOG_NUDGE_COOLDOWN_HOURS` (24h por defecto). Persiste
    `last_zero_log_nudge_at` en `user_profiles.health_profile` (NO crea tabla nueva
    para no proliferar artefactos).

    Filtros:
      - meal_plans.created_at en los últimos 30 días (descarta planes archivados).
      - generation_status indica plan activo (no failed ni expired).
      - 0 rows en consumed_meals durante la ventana de detección.
      - Cooldown respetado.

    Returns:
        Número de usuarios a los que se envió push.
    """
    try:
        candidates = execute_sql_query(
            """
            SELECT DISTINCT mp.user_id::text AS user_id,
                   COALESCE(p.health_profile->>'last_zero_log_nudge_at', '') AS last_nudge_at
            FROM meal_plans mp
            JOIN user_profiles p ON p.id = mp.user_id
            WHERE mp.plan_data->>'generation_status' IN
                ('complete', 'complete_partial', 'partial', 'generating_next')
              AND mp.created_at > NOW() - INTERVAL '30 days'
              AND NOT EXISTS (
                  SELECT 1 FROM consumed_meals cm
                  WHERE cm.user_id = mp.user_id
                    AND cm.consumed_at > NOW() - make_interval(days => %s)
              )
              AND (
                  p.health_profile->>'last_zero_log_nudge_at' IS NULL
                  OR (p.health_profile->>'last_zero_log_nudge_at')::timestamptz
                     < NOW() - make_interval(hours => %s)
              )
            LIMIT %s
            """,
            (
                CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS,
                CHUNK_ZERO_LOG_NUDGE_COOLDOWN_HOURS,
                CHUNK_ZERO_LOG_NUDGE_MAX_USERS,
            ),
            fetch_all=True,
        ) or []
    except Exception as e:
        logger.warning(f"[P1-2/ZERO-LOG-NUDGE] SELECT falló: {e}")
        return 0

    if not candidates:
        return 0

    sent = 0
    now_iso = datetime.now(timezone.utc).isoformat()
    for row in candidates:
        uid = row.get("user_id")
        if not uid:
            continue
        try:
            _dispatch_push_notification(
                user_id=uid,
                title="Loguea tus comidas para tu próximo bloque",
                body=(
                    f"Llevas {CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS} días sin registrar comidas. "
                    "Si no logueas pronto, el siguiente bloque de tu plan se pausará. "
                    "Abre el diario y registra lo que comiste."
                ),
                url="/dashboard?nudge=zero_log",
            )
            # Persistir last_zero_log_nudge_at en health_profile vía jsonb_set para
            # respetar el cooldown sin sobrescribir otras keys.
            try:
                execute_sql_write(
                    """
                    UPDATE user_profiles
                    SET health_profile = jsonb_set(
                            COALESCE(health_profile, '{}'::jsonb),
                            '{last_zero_log_nudge_at}',
                            to_jsonb(%s::text),
                            true
                        )
                    WHERE id = %s
                    """,
                    (now_iso, uid),
                )
            except Exception as persist_err:
                logger.warning(
                    f"[P1-2/ZERO-LOG-NUDGE] Push enviado a {uid} pero falló persistir "
                    f"last_zero_log_nudge_at: {persist_err}. Riesgo de re-nudge dentro del cooldown."
                )
            sent += 1
        except Exception as push_err:
            logger.warning(f"[P1-2/ZERO-LOG-NUDGE] Push falló para {uid}: {push_err}")

    if sent:
        logger.info(
            f"[P1-2/ZERO-LOG-NUDGE] Enviados {sent} nudges proactivos "
            f"(detection_days={CHUNK_ZERO_LOG_NUDGE_DETECTION_DAYS}, "
            f"cooldown={CHUNK_ZERO_LOG_NUDGE_COOLDOWN_HOURS}h)."
        )
    return sent


def _compute_chunk_retry_delay_minutes(next_attempt: int, is_critical: bool = False) -> int:
    """Backoff exponencial acotado para retries automáticos del worker."""
    if is_critical:
        return CHUNK_RETRY_CRITICAL_MINUTES
    exponent = max(0, next_attempt - 1)
    return min(CHUNK_RETRY_BASE_MINUTES * (2 ** exponent), 12 * 60)





def calculate_meal_success_scores(user_id: str, days_back: int = 14):
    """Calcula que platos del historial tuvieron mayor adherencia."""
    days_back_iso = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    
    plans = get_recent_plans(user_id, days=days_back)
    consumed = get_consumed_meals_since(user_id, since_iso_date=days_back_iso)
    
    def normalize(text):
        if not text: return ""
        return strip_accents(text.lower()).strip()
        
    consumed_names_list = [normalize(m.get('meal_name', '')) for m in consumed]
    consumed_names = set(consumed_names_list)
    
    consumed_counts = {}
    for name in consumed_names_list:
        if name:
            consumed_counts[name] = consumed_counts.get(name, 0) + 1
            
    frequent_meals = [m[0] for m in sorted(consumed_counts.items(), key=lambda x: x[1], reverse=True)]
    
    scores = {}
    for plan in plans:
        if not isinstance(plan, dict): continue
        for day in plan.get('days', []):
            if not isinstance(day, dict): continue
            for meal in day.get('meals', []):
                if not isinstance(meal, dict): continue
                name = normalize(meal.get('name', ''))
                if name:
                    scores[name] = {
                        'was_eaten': name in consumed_names,
                        'meal_type': meal.get('meal'),
                        'technique': meal.get('technique', ''),
                    }
                
    successful_techniques = [s['technique'] for s in scores.values() if s['was_eaten'] and s['technique']]
    abandoned_techniques = [s['technique'] for s in scores.values() if not s['was_eaten'] and s['technique']]
    
    return successful_techniques, abandoned_techniques, frequent_meals

def calculate_ingredient_fatigue(user_id: str, days_back: int = 14, tuning_metrics: dict = None):
    """
    Mejora 3 y GAP 5: Calcula la monotonia de ingredientes y categorias nutricionales
    en los ultimos dias usando decaimiento temporal y NLP.
    Retorna ingredientes y categorias con alta frecuencia de aparicion (fatiga cruzada).

    [P1-4] El factor de decay y los thresholds son constantes nombradas en constants.py
    (con override por env var) en lugar de literales hardcoded. `tuning_metrics.fatigue_decay`
    sigue siendo respetado para A/B testing por usuario; cae a la constante global si no está.
    """
    from constants import (
        normalize_ingredient_for_tracking,
        get_nutritional_category,
        INGREDIENT_FATIGUE_DECAY_FACTOR,
        INGREDIENT_FATIGUE_INDIVIDUAL_THRESHOLD,
        INGREDIENT_FATIGUE_INDIVIDUAL_RATIO,
        INGREDIENT_FATIGUE_CATEGORY_THRESHOLD,
        INGREDIENT_FATIGUE_CATEGORY_RATIO,
    )
    from collections import defaultdict
    import ast

    now = datetime.now(timezone.utc)
    days_back_iso = (now - timedelta(days=days_back)).isoformat()
    consumed = get_consumed_meals_since(user_id, since_iso_date=days_back_iso)

    if not consumed:
        return {"score": 0.0, "fatigued_ingredients": []}

    ingredient_counts = defaultdict(float)
    category_counts = defaultdict(float)
    total_weight = 0.0

    # [P1-4] Permitir override por usuario vía tuning_metrics (A/B testing); fallback a constante global.
    base_decay = float(
        (tuning_metrics or {}).get("fatigue_decay", INGREDIENT_FATIGUE_DECAY_FACTOR)
    )

    for meal in consumed:
        # 1. Temporal Decay
        created_at_str = meal.get('created_at')
        days_ago = 0
        if created_at_str:
            try:
                if created_at_str.endswith('Z'):
                    created_at_str = created_at_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(created_at_str)
                days_ago = max(0, (now - dt).days)
            except Exception:
                pass

        decay_weight = base_decay ** days_ago
        total_weight += decay_weight

        # 2. Extraer ingredientes
        ingredients_raw = meal.get('ingredients')
        ing_list = []
        if isinstance(ingredients_raw, list):
            ing_list = ingredients_raw
        elif isinstance(ingredients_raw, str):
            try:
                ing_list = ast.literal_eval(ingredients_raw)
                if not isinstance(ing_list, list):
                    ing_list = [ingredients_raw]
            except Exception:
                ing_list = [i.strip() for i in ingredients_raw.split(',')]

        # 3. Normalizacion Avanzada (Constants.py NLP)
        for ing in ing_list:
            if not isinstance(ing, str) or not ing.strip():
                continue
            normalized = normalize_ingredient_for_tracking(ing)
            # Solo guardamos cosas que lograron normalizarse a bases reales (evitar ruido como 'agua')
            if normalized:
                ingredient_counts[normalized] += decay_weight
                category = get_nutritional_category(normalized)
                if category:
                    category_counts[category] += decay_weight

    fatigued_items = []
    # [P1-4] Thresholds ahora desde constants.py (env-tunable).
    for ing, weight in ingredient_counts.items():
        if (
            weight >= INGREDIENT_FATIGUE_INDIVIDUAL_THRESHOLD
            or (total_weight > 0 and (weight / total_weight) > INGREDIENT_FATIGUE_INDIVIDUAL_RATIO)
        ):
            fatigued_items.append(ing)

    for cat, weight in category_counts.items():
        if (
            weight >= INGREDIENT_FATIGUE_CATEGORY_THRESHOLD
            or (total_weight > 0 and (weight / total_weight) > INGREDIENT_FATIGUE_CATEGORY_RATIO)
        ):
            fatigued_items.append(f"[CATEGORÃA] {cat}")
            
    # Set completo de ingredientes normalizados consumidos — usado para auto-tune cross-ciclo
    consumed_ingredient_set = set(ingredient_counts.keys())

    fatigue_score = min(1.0, len(fatigued_items) * 0.2)

    return {
        "score": round(fatigue_score, 2),
        "fatigued_ingredients": fatigued_items,
        "consumed_ingredient_set": consumed_ingredient_set,
    }


def calculate_day_of_week_adherence(user_id: str, days_back: int = 30):
    """
    Calcula un perfil de adherencia predictivo usando EMA (Exponential Moving Average)
    para cada dia de la semana. Detecta patrones emergentes de abandono.
    """
    days_back_iso = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
    consumed = get_consumed_meals_since(user_id, since_iso_date=days_back_iso)
    
    # Agrupar comidas por fecha especifica (YYYY-MM-DD)
    date_counts = {}
    for meal in consumed:
        created_at_str = meal.get('created_at')
        if created_at_str:
            try:
                if created_at_str.endswith('Z'):
                    created_at_str = created_at_str[:-1] + '+00:00'
                dt = datetime.fromisoformat(created_at_str)
                d_str = dt.date().isoformat()
                date_counts[d_str] = date_counts.get(d_str, 0) + 1
            except Exception:
                pass
                
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Si no hay datos, asumimos 100% de adherencia temporal
    if not date_counts:
        return {day_names[i]: 1.0 for i in range(7)}
        
    # Iterar explicitamente desde el primer dia registrado para incluir dias en 0
    first_tracking_date = min(date_counts.keys())
    first_date_dt = datetime.fromisoformat(first_tracking_date).date()
    now = datetime.now(timezone.utc).date()
    
    date_list = []
    current_d = first_date_dt
    while current_d <= now:
        date_list.append(current_d)
        current_d += timedelta(days=1)
        
    weekday_history = {i: [] for i in range(7)}
    for d in date_list:
        d_str = d.isoformat()
        count = date_counts.get(d_str, 0)
        weekday_history[d.weekday()].append(count)
        
    global_max_day_count = max(date_counts.values()) if date_counts else 3
    if global_max_day_count == 0:
        global_max_day_count = 3
        
    alpha = 0.4 # Factor EMA: 40% peso al evento mas reciente
    day_ema = {i: 0.0 for i in range(7)}
    
    for i in range(7):
        history = weekday_history[i] 
        if not history:
            day_ema[i] = 1.0 # No hay dias registrados para este weekday
            continue
            
        ema = None
        for count in history:
            ratio = min(count / global_max_day_count, 1.0)
            if ema is None:
                ema = ratio
            else:
                ema = (alpha * ratio) + ((1 - alpha) * ema)
                
        day_ema[i] = ema if ema is not None else 1.0
        
    return {day_names[k]: round(v, 2) for k, v in day_ema.items()}


def calculate_meal_level_adherence(user_id: str, plan_days: list, consumed_records: list, household_size: int = 1):
    """Calcula adherencia por tipo de comida (desayuno, almuerzo, cena, merienda)."""
    meal_type_stats = {}
    
    # Asumimos que el primer dia del plan representa la estructura diaria esperada
    days_to_check = [plan_days[0]] if plan_days else []
    
    for day in days_to_check:
        for meal in day.get('meals', []):
            mt = meal.get('meal', 'otro').lower()
            if mt not in meal_type_stats:
                meal_type_stats[mt] = {'planned_per_day': 0, 'eaten': 0}
            meal_type_stats[mt]['planned_per_day'] += 1
            
    # Contar comidas ingeridas
    unique_dates = set()
    for record in consumed_records:
        mt = record.get('meal_type', 'otro').lower()
        if mt in meal_type_stats:
            meal_type_stats[mt]['eaten'] += 1
        if 'created_at' in record:
            unique_dates.add(str(record['created_at'])[:10])
            
    days_passed = max(1, len(unique_dates))
            
    return {
        mt: min(1.0, round((s['eaten'] / household_size) / max(s['planned_per_day'] * days_passed, 1), 2))
        for mt, s in meal_type_stats.items()
    }


def calculate_plan_quality_score(user_id: str, plan_data: dict, consumed_records: list, household_size: int = 1) -> float:
    """
    Mejora 4: Evalua retrospectivamente la calidad del plan midiendo satisfaccion real y retencion.
    """
    total_meals = sum(len(d.get('meals', [])) for d in plan_data.get('days', []))
    eaten_meals = len(consumed_records) / household_size
    
    # 1. Adherencia bruta
    adherence = eaten_meals / max(total_meals, 1)
    
    # 2. Diversidad: ¿cuantos ingredientes distintos se comieron?
    eaten_ingredients = set()
    
    def normalize(text):
        if not text: return ""
        return strip_accents(text.lower()).strip()
        
    for r in consumed_records:
        ing_list = r.get('ingredients', [])
        if isinstance(ing_list, list):
            for ing in ing_list:
                if isinstance(ing, dict) and "name" in ing:
                    eaten_ingredients.add(normalize(ing["name"]))
                elif isinstance(ing, dict) and "display_string" in ing:
                    eaten_ingredients.add(normalize(ing["display_string"]))
                elif isinstance(ing, str):
                    eaten_ingredients.add(normalize(ing))
                    
    diversity_score = min(1.0, len(eaten_ingredients) / 5)
    
    # 3. Satisfaccion explicita (Likes vs Rejections)
    likes = get_user_likes(user_id)
    rejections = get_active_rejections(user_id)
    
    # Calculamos un ratio de satisfaccion neto normalizado
    # Partimos de un 0.5 (neutral). Likes suman, rechazos restan.
    net_satisfaction = len(likes) - len(rejections)
    satisfaction_score = max(0.0, min(1.0, 0.5 + (net_satisfaction / max(total_meals, 1) * 0.5)))
    rejection_penalty = len(rejections) * 0.05
    
    # 4. Retention Signal (¿Sigue activo el usuario?)
    from db_facts import get_consumed_meals_since
    from datetime import datetime, timezone, timedelta
    
    now = datetime.now(timezone.utc)
    # Verificamos si ha registrado comidas en las ultimas 48 horas como seÃ±al de retencion
    recently_consumed = get_consumed_meals_since(user_id, since_iso_date=(now - timedelta(days=2)).isoformat())
    retention_score = 1.0 if recently_consumed else 0.3
    
    # Weighted score final
    quality = round(
        (adherence * 0.35 + 
         diversity_score * 0.20 + 
         satisfaction_score * 0.25 + 
         retention_score * 0.20) - rejection_penalty, 
        2
    )
    
    return max(0.0, min(1.0, quality))


def get_similar_user_patterns(user_id: str, health_profile: dict):
    """Para usuarios sin historial, busca que funciono para perfiles similares (Mejora 4)."""
    goal = health_profile.get('mainGoal')
    activity = health_profile.get('activityLevel')
    diet_types = health_profile.get('dietTypes', [])
    country = health_profile.get('country')
    
    if not goal or not activity:
        return []
        
    try:
        # 1. Recuperar rechazos activos para post-filtrado
        rejections = get_active_rejections(user_id)
        
        query = """
            SELECT meal_name, COUNT(*) as popularity
            FROM consumed_meals cm
            JOIN user_profiles up ON cm.user_id = up.id
            WHERE up.health_profile->>'mainGoal' = %s
            AND up.health_profile->>'activityLevel' = %s
            AND cm.created_at > NOW() - INTERVAL '30 days'
        """
        params = [goal, activity]
        
        # 2. Segmentacion de Dieta
        if diet_types and len(diet_types) > 0:
            import json
            diet = diet_types[0]
            query += " AND up.health_profile->'dietTypes' @> %s::jsonb"
            params.append(json.dumps([diet]))
            
        # 3. Segmentacion Cultural
        if country:
            query += " AND up.health_profile->>'country' = %s"
            params.append(country)
            
        query += " GROUP BY meal_name ORDER BY popularity DESC LIMIT 20"
        
        popular_meals = execute_sql_query(query, tuple(params), fetch_all=True)
        
        if not popular_meals:
            return []
            
        # 4. Post-filtro de Calidad (Excluir platos rechazados por el usuario)
        if rejections:
            def normalize(text):
                if not text: return ""
                return strip_accents(str(text).lower()).strip()
                
            normalized_rejections = [normalize(r.get('ingredient', '')) for r in rejections]
            
            filtered_meals = []
            for pm in popular_meals:
                meal_name_norm = normalize(pm.get('meal_name', ''))
                is_rejected = any(rej in meal_name_norm for rej in normalized_rejections if rej)
                
                if not is_rejected:
                    filtered_meals.append(pm)
            
            return filtered_meals[:10]
            
        return popular_meals[:10]
        
    except Exception as e:
        if "relation" not in str(e).lower():
            logger.warning(f" [COLD-START] Error en query similar users: {e}")
        return []
        

def _build_facts_memory_context(user_id: str) -> str:
    """
    Construye un string de contexto de memoria a partir de los hechos (facts)
    aprendidos por la IA sobre el usuario. Esto permite que la rotacion nocturna
    sepa cosas como "le gusta el pollo", "es intolerante a la lactosa", etc.
    """
    try:
        facts = get_all_user_facts(user_id)
        if not facts:
            return ""
        
        # Priorizar por categoria para que alergias/condiciones medicas aparezcan primero
        CATEGORY_ORDER = {
            "alergia": 0, "condicion_medica": 1, "rechazo": 2,
            "dieta": 3, "objetivo": 4, "preferencia": 5, "sintoma_temporal": 6
        }
        
        def sort_key(f):
            meta = f.get("metadata", {})
            cat = meta.get("category", "") if isinstance(meta, dict) else ""
            return CATEGORY_ORDER.get(cat, 7)
        
        facts_sorted = sorted(facts, key=sort_key)
        
        # Construir el contexto legible (maximo 15 facts para no saturar el prompt)
        fact_lines = []
        for f in facts_sorted[:15]:
            fact_text = f.get("fact", "")
            meta = f.get("metadata", {})
            cat = meta.get("category", "general") if isinstance(meta, dict) else "general"
            if fact_text:
                fact_lines.append(f"â€¢ [{cat.upper()}] {fact_text}")
        
        if not fact_lines:
            return ""
        
        return (
            "\n\n--- MEMORIA DEL CEREBRO IA (HECHOS APRENDIDOS SOBRE ESTE USUARIO) ---\n"
            "DEBES respetar OBLIGATORIAMENTE esta informacion al generar el plan:\n"
            + "\n".join(fact_lines)
            + "\n--------------------------------------------------------------------"
        )
    except Exception as e:
        logger.warning(f" [CRON] Error building facts memory context for {user_id}: {e}")
        return ""


def _persist_nightly_learning_signals(user_id: str, health_profile: dict, days: list, consumed_records: list):
    """Persiste señales de aprendizaje en health_profile durante planes largos de forma segura.
    
    [P1-5 FIX] Previene condiciones de carrera cuando múltiples chunks de diferentes
    planes activos del mismo usuario escriben simultáneamente. Utiliza un SELECT FOR UPDATE
    atómico para un read-modify-write estricto, dejando el LLM fuera del lock para evitar deadlocks.
    """
    from db_core import execute_sql_query, connection_pool
    import json
    from datetime import datetime, timezone, timedelta
    import psycopg
    from psycopg.rows import dict_row

    # 1. Ejecutar procesos pesados y propensos a red FUERA del lock
    run_retro = False
    last_retro_str = health_profile.get("last_llm_retrospective_date")
    if not last_retro_str:
        run_retro = True
    else:
        try:
            if last_retro_str.endswith('Z'):
                last_retro_str = last_retro_str[:-1] + '+00:00'
            last_retro_date = datetime.fromisoformat(last_retro_str)
            if last_retro_date.tzinfo is None:
                last_retro_date = last_retro_date.replace(tzinfo=timezone.utc)
            if (datetime.now(timezone.utc) - last_retro_date).days >= 7:
                run_retro = True
        except Exception as dt_err:
            logger.warning(f" [LLM-as-Judge] Error parseando last_retro_date: {dt_err}")
            run_retro = True

    llm_retro = None
    liked_flavor_profiles = None
    if run_retro:
        try:
            from ai_helpers import generate_llm_retrospective, extract_liked_flavor_profiles
            from db import get_user_likes, get_active_rejections
            
            recent_likes = get_user_likes(user_id)
            recent_rejections = get_active_rejections(user_id=user_id, session_id=None)
            
            llm_retro = generate_llm_retrospective(
                user_id=user_id, plan_data={'days': days},
                consumed_records=consumed_records, recent_likes=recent_likes,
                recent_rejections=recent_rejections
            )
            liked_flavor_profiles = extract_liked_flavor_profiles(recent_likes)
        except Exception as e:
            logger.error(f" [LLM-as-Judge] Error en el flujo offline: {e}")

    # 2. Bloque transaccional estricto (Read-Modify-Write atómico)
    if not connection_pool:
        logger.error(" [NIGHTLY LEARN] Error: connection_pool no está disponible.")
        return

    try:
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cursor:
                    # [P1-5] Lock timeout para evitar bloqueos perpetuos cuando múltiples
                    # planes activos del mismo usuario compiten por el mismo row.
                    # Si tras 10s no podemos adquirir el lock, aborta esta señal en vez de
                    # bloquear el chunk completo (la próxima generación re-intentará).
                    from constants import CHUNK_LEARNING_LOCK_TIMEOUT_MS as _P15_LOCK_MS
                    try:
                        cursor.execute(f"SET LOCAL lock_timeout = '{int(_P15_LOCK_MS)}ms'")
                    except Exception as _p15_set_err:
                        logger.debug(f"[P1-5] No se pudo setear lock_timeout: {_p15_set_err}")
                    try:
                        cursor.execute("SELECT health_profile FROM user_profiles WHERE id = %s FOR UPDATE", (user_id,))
                    except Exception as _p15_lock_err:
                        # psycopg.errors.LockNotAvailable o similar.
                        if "lock" in str(_p15_lock_err).lower() or "timeout" in str(_p15_lock_err).lower():
                            logger.warning(
                                f"[P1-5/LOCK-TIMEOUT] No se adquirió lock en {_P15_LOCK_MS}ms para user {user_id}; "
                                f"saltando esta señal nightly (otro plan tiene el lock). El próximo chunk reintentará."
                            )
                            return
                        raise
                    row = cursor.fetchone()
                    if not row:
                        return

                    # [P0-5] `row['health_profile']` raised KeyError when the cursor
                    # returned a dict-like row missing the column (e.g. test stubs, or a
                    # corrupted profile). The whole nightly persistence transaction was
                    # aborted by the outer except — silently dropping adherence/quality
                    # signals for the user. Defaulting to {} keeps the RMW going; if the
                    # profile was truly empty, downstream writes still produce a valid
                    # {} → updated dict. Telemetry note: log when we observe a missing
                    # column so DB drift is visible.
                    if 'health_profile' not in row:
                        logger.warning(
                            f" [NIGHTLY LEARN] row sin columna 'health_profile' para user {user_id}; "
                            f"asumiendo {{}}. Posible drift de schema."
                        )
                    fresh_profile = row.get('health_profile') or {}
                    
                    try:
                        validated_profile = HealthProfileSchema(**fresh_profile)
                    except Exception:
                        validated_profile = HealthProfileSchema()

                    # Adherencia
                    expected_meals_count = len(days[0].get('meals', [])) if days else 0
                    if expected_meals_count > 0:
                        current_cycle_adherence = calculate_meal_level_adherence(user_id, days, consumed_records)
                        is_weekend = (datetime.now(timezone.utc) - timedelta(hours=12)).weekday() >= 5
                        profile_key = 'meal_adherence_weekend' if is_weekend else 'meal_adherence_weekday'
                        
                        historical_adherence = getattr(validated_profile, profile_key, {})
                        alpha = 0.3
                        smoothed_adherence = {}
                        all_meal_types = set(list(current_cycle_adherence.keys()) + list(historical_adherence.keys()))
                        
                        for mt in all_meal_types:
                            current_val = current_cycle_adherence.get(mt, 1.0)
                            hist_val = historical_adherence.get(mt, 1.0)
                            smoothed_adherence[mt] = round((alpha * current_val) + ((1 - alpha) * hist_val), 2)
                            
                        avg_adherence = round(sum(smoothed_adherence.values()) / max(1, len(smoothed_adherence)), 2)
                        
                        adherence_history = fresh_profile.get('adherence_history_rotations', [])
                        if not isinstance(adherence_history, list): adherence_history = []
                        adherence_history.append(avg_adherence)
                        
                        fresh_profile[profile_key] = smoothed_adherence
                        fresh_profile['adherence_history_rotations'] = adherence_history[-5:]

                    # Quality Score
                    quality_score = calculate_plan_quality_score(user_id, {'days': days}, consumed_records)
                    quality_history = getattr(validated_profile, 'quality_history_rotations', [])
                    if not isinstance(quality_history, list): quality_history = []
                    quality_history.append(quality_score)
                    
                    fresh_profile['last_plan_quality'] = quality_score
                    fresh_profile['quality_history_rotations'] = quality_history[-5:]

                    # Snapshot de peso
                    cursor.execute("SELECT weight, unit, created_at::date as date FROM weight_log WHERE user_id = %s ORDER BY created_at DESC LIMIT 1", (user_id,))
                    weight_row = cursor.fetchone()
                    if weight_row:
                        fresh_profile['latest_weight_snapshot'] = {
                            "date": str(weight_row['date']), 
                            "weight": weight_row['weight'], 
                            "unit": weight_row.get('unit', 'lb')
                        }

                    # Retro LLM
                    if run_retro and (llm_retro or liked_flavor_profiles):
                        if llm_retro:
                            fresh_profile['llm_retrospective'] = llm_retro
                            fresh_profile['last_llm_retrospective_date'] = datetime.now(timezone.utc).isoformat()
                        if liked_flavor_profiles:
                            fresh_profile['liked_flavor_profiles'] = liked_flavor_profiles

                    # Actualización final unificada
                    cursor.execute(
                        "UPDATE user_profiles SET health_profile = %s::jsonb WHERE id = %s",
                        (json.dumps(fresh_profile), user_id)
                    )
                    logger.info(f" [NIGHTLY LEARN] Señales de aprendizaje guardadas atómicamente para {user_id}")
    except Exception as e:
        logger.error(f" [NIGHTLY LEARN] Error en la persistencia transaccional: {e}")



def _refill_emergency_backup_plan(user_id: str, pipeline_data: dict, taste_profile: str, memory_context: str):
    """Genera asincronamente un plan de respaldo si el usuario no tiene suficientes dias en cache."""
    logger.info(f" [CRON:REFILL] Checking emergency backup cache for user {user_id}...")
    try:
        from db_core import execute_sql_query, execute_sql_write
        user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
        if not user_res:
            return
            
        health_profile = user_res.get("health_profile", {})
        try:
            validated_profile = HealthProfileSchema(**health_profile)
        except Exception as e:
            logger.warning(f" [CRON:REFILL] health_profile malformado para {user_id}. Usando defaults: {e}")
            validated_profile = HealthProfileSchema()
            
        backup_plan = validated_profile.emergency_backup_plan
        
        if isinstance(backup_plan, list) and len(backup_plan) >= 3:
            logger.info(f" [CRON:REFILL] Cache full for user {user_id} ({len(backup_plan)} days). Skipping generation.")
            return
            
        logger.info(f" [CRON:REFILL] Cache low/empty for user {user_id}. Generating 3-day emergency backup...")
        
        # Override to ensure it's diverse and basic just in case
        pipeline_data["_is_emergency_generation"] = True
        
        emergency_memory = memory_context + "\n[INSTRUCCION CRITICA: Este es un plan de EMERGENCIA de respaldo. Asegurate de que las comidas sean seguras, sencillas de preparar, y muy amigables con las restricciones del usuario.]"
        
        succ_techs = pipeline_data.get('successful_techniques', [])
        freq_meals = pipeline_data.get('frequent_meals', [])
        
        if succ_techs or freq_meals:
            emergency_memory += "\n\n[SEMILLA DE ADHERENCIA: Utiliza OBLIGATORIAMENTE las siguientes tecnicas y platos recurrentes porque el usuario tiene 90%+ de probabilidad de adherirse a ellos.]"
            if succ_techs:
                emergency_memory += f"\n- Tecnicas de alta adherencia: {', '.join(succ_techs[:5])}"
            if freq_meals:
                emergency_memory += f"\n- Platos recurrentes: {', '.join(freq_meals[:5])}"
        
        from graph_orchestrator import run_plan_pipeline
        # [P1-4] Antes este call pasaba 6 args posicionales: (..., None, None) donde
        # el 6° era `previous_ai_error` (ya eliminado del signature). Ahora son 5
        # explícitos vía kwargs para inmunizar contra futuros cambios de orden.
        result = run_plan_pipeline(
            pipeline_data, [], taste_profile, emergency_memory,
            progress_callback=None, background_tasks=None,
        )
        
        if 'error' not in result and 'days' in result and isinstance(result['days'], list) and len(result['days']) > 0:
            execute_sql_write(
                "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{emergency_backup_plan}', %s::jsonb) WHERE id = %s",
                (json.dumps(result['days']), user_id)
            )
            logger.info(f"âœ… ðŸ›¡ï¸ [CRON:REFILL] Successfully generated and saved {len(result['days'])} backup days for user {user_id}.")
        else:
            logger.warning(f" [CRON:REFILL] Failed to generate emergency backup for {user_id}: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f" [CRON:REFILL] Exception during emergency backup generation for {user_id}: {e}")


# [GAP 6] Sembrado inmediato del emergency_backup_plan sin llamar al LLM.
# Llamado desde routers/plans.py justo despues de persistir el chunk 1, asi:
#   - Si chunk 2+ cae en Smart Shuffle antes de la primera rotacion nocturna,
#     el pool de respaldo no esta vacio.
#   - Idempotente: solo siembra si el backup actual tiene <3 dias.
def _seed_emergency_backup_if_empty(user_id: str, fresh_days):
    try:
        if not user_id or not isinstance(fresh_days, list) or len(fresh_days) == 0:
            return
        user_res = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,), fetch_one=True
        )
        if not user_res:
            return
        hp = user_res.get("health_profile") or {}
        current = hp.get("emergency_backup_plan") or []
        if isinstance(current, list) and len(current) >= 3:
            return  # Ya hay backup suficiente, no pisar

        import copy, json as _json
        seed = []
        for d in fresh_days[:3]:
            if not isinstance(d, dict) or not d.get("meals"):
                continue
            d_copy = copy.deepcopy(d)
            d_copy.pop("_is_degraded_shuffle", None)
            d_copy.pop("_mutated", None)
            seed.append(d_copy)
        if not seed:
            return

        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(COALESCE(health_profile, '{}'::jsonb), '{emergency_backup_plan}', %s::jsonb) WHERE id = %s",
            (_json.dumps(seed, ensure_ascii=False), user_id)
        )
        logger.info(f"[GAP6:SEED] emergency_backup_plan sembrado con {len(seed)} dias para user {user_id}.")
    except Exception as e:
        logger.warning(f"[GAP6:SEED] No se pudo sembrar emergency_backup_plan para {user_id}: {e}")


def _inject_advanced_learning_signals(user_id: str, pipeline_data: dict, health_profile: dict, days: list, consumed_records: list, days_since_last_chunk: int = 0):
    """Extrae las señales avanzadas de aprendizaje y las inyecta en pipeline_data."""
    from db_core import execute_sql_query, execute_sql_write
    # [P0-1-RECOVERY/WORKER-FIX] datetime/timezone/timedelta vienen de globals del módulo.

    # [P1-4] La ventana de señales se basa en la distancia temporal real al chunk anterior
    # (days_since_last_chunk = days_offset del chunk actual), no en chunk_kind.
    # Ejemplo: chunk 2 de plan 30d tiene offset=3 → ventana de 4 días, igual que rolling refill.
    # Chunk 8 de plan 30d tiene offset=21 → ventana de 14 días (capped).
    # El lower bound (PLAN_CHUNK_SIZE+1) evita ventanas de 0-1 días en el primer chunk.
    from constants import PLAN_CHUNK_SIZE
    _days_elapsed = max(PLAN_CHUNK_SIZE + 1, int(days_since_last_chunk or 0))
    _fatigue_days_back = min(_days_elapsed, 14)
    _success_days_back = min(_days_elapsed, 14)
    _adherence_days_back = min(_days_elapsed, 30)
    logger.info(f"[P1-4] Ventana de aprendizaje por distancia temporal: days_elapsed={_days_elapsed} → fatigue/success={_fatigue_days_back}d adherencia={_adherence_days_back}d")

    # MEJORA 5: Ingredient Fatigue Detection
    fatigue_data = None
    try:
        tuning_metrics = health_profile.get("tuning_metrics", {})
        fatigue_data = calculate_ingredient_fatigue(user_id, days_back=_fatigue_days_back, tuning_metrics=tuning_metrics)
        if fatigue_data and fatigue_data.get('fatigued_ingredients'):
            pipeline_data['fatigued_ingredients'] = fatigue_data['fatigued_ingredients']
            logger.info(f"[CRON] Ingredient Fatigue detected for {user_id}: {fatigue_data['fatigued_ingredients']}")
    except Exception as e:
        logger.error(f"[CRON] Error calculating ingredient fatigue: {e}")

    # MEJORA 4: Auto-Tuning del fatigue_decay mediante tasa de falsos positivos cross-ciclo
    # Compara ingredientes "fatigados" en el ciclo anterior vs lo que el usuario comio en este ciclo.
    # Alto solapamiento = falso positivo = el decay era demasiado lento -> olvidar mas rapido.
    # [P1-4] Todos los thresholds, clamps y steps vienen de constants.py (env-tunable)
    # en lugar de literales hardcoded. Antes el fallback `0.9` ignoraba cambios en
    # INGREDIENT_FATIGUE_DECAY_FACTOR para usuarios sin tuning_metrics persistido.
    from constants import (
        INGREDIENT_FATIGUE_DECAY_FACTOR as _FAT_DEFAULT,
        INGREDIENT_FATIGUE_DECAY_LOWER_CLAMP as _FAT_LOW_CLAMP,
        INGREDIENT_FATIGUE_DECAY_UPPER_CLAMP as _FAT_HIGH_CLAMP,
        INGREDIENT_FATIGUE_FP_HIGH_THRESHOLD as _FAT_FP_HIGH,
        INGREDIENT_FATIGUE_FP_LOW_THRESHOLD as _FAT_FP_LOW,
        INGREDIENT_FATIGUE_DECAY_STEP_DOWN as _FAT_STEP_DOWN,
        INGREDIENT_FATIGUE_DECAY_STEP_UP as _FAT_STEP_UP,
    )
    try:
        tuning_metrics = health_profile.get("tuning_metrics", {})
        fatigue_decay = float(tuning_metrics.get("fatigue_decay", _FAT_DEFAULT))
        last_fatigued = health_profile.get("last_fatigued_ingredients", [])
        current_consumed = fatigue_data.get("consumed_ingredient_set", set()) if fatigue_data else set()

        if last_fatigued and current_consumed:
            fp_count = sum(1 for f in last_fatigued if f in current_consumed)
            fp_rate = round(fp_count / len(last_fatigued), 3)

            fp_history = tuning_metrics.get("fatigue_fp_history", [])
            fp_history.append(fp_rate)
            fp_history = fp_history[-3:]

            if len(fp_history) >= 2:
                mean_fp = sum(fp_history) / len(fp_history)
                if mean_fp > _FAT_FP_HIGH and fatigue_decay > _FAT_LOW_CLAMP:
                    fatigue_decay = round(fatigue_decay - _FAT_STEP_DOWN, 2)
                    logger.info(f"[FATIGUE-TUNE] Alta tasa FP ({mean_fp:.2f}): fatigue_decay -> {fatigue_decay} (olvidar mas rapido).")
                elif mean_fp < _FAT_FP_LOW and fatigue_decay < _FAT_HIGH_CLAMP:
                    fatigue_decay = round(fatigue_decay + _FAT_STEP_UP, 2)
                    logger.info(f"[FATIGUE-TUNE] Baja tasa FP ({mean_fp:.2f}): fatigue_decay -> {fatigue_decay} (memoria mas larga).")

            tuning_metrics["fatigue_fp_history"] = fp_history
            tuning_metrics["fatigue_decay"] = fatigue_decay
            execute_sql_write(
                "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{tuning_metrics}', %s::jsonb) WHERE id = %s",
                (json.dumps(tuning_metrics), user_id)
            )
            health_profile["tuning_metrics"] = tuning_metrics
            logger.info(f"[FATIGUE-TUNE] decay={fatigue_decay}, fp_rate={fp_rate}, history={fp_history}")

        # Guardar fatigued de ESTE ciclo (solo ingredientes individuales) para el proximo ciclo
        current_fatigued_pure = [
            f for f in (fatigue_data.get("fatigued_ingredients") or [])
            if fatigue_data and not f.startswith("[CATEGOR")
        ]
        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{last_fatigued_ingredients}', %s::jsonb) WHERE id = %s",
            (json.dumps(current_fatigued_pure), user_id)
        )
        health_profile["last_fatigued_ingredients"] = current_fatigued_pure
    except Exception as e:
        logger.error(f"[FATIGUE-TUNE] Error en auto-tuning de fatigue_decay: {e}")

    # MEJORA 2: Feedback Loop Cerrado Granular (Self-Improving) con Decay
    try:
        household_size = max(1, int(health_profile.get('householdSize', 1)))
        expected_meals_count = len(days[0].get('meals', [])) if days else 0
        if not consumed_records or len(consumed_records) < 3:
            logger.info(f" [FEEDBACK LOOP] Insuficientes datos reales ({len(consumed_records) if consumed_records else 0} < 3). Omitiendo _meal_level_adherence para evitar falsos positivos EMA.")
        elif expected_meals_count > 0:
            # 1. Calcular adherencia granular del ciclo actual
            current_cycle_adherence = calculate_meal_level_adherence(user_id, days, consumed_records, household_size)
            
            # 2. Determinar si el ciclo evaluado fue fin de semana o dia de semana
            # [P0-1-RECOVERY/WORKER-FIX] datetime/timezone/timedelta vienen de globals del módulo.
            # Restamos 12 horas para que si corre a las 3 AM, cuente como el dia anterior
            is_weekend = (datetime.now(timezone.utc) - timedelta(hours=12)).weekday() >= 5
            profile_key = 'meal_adherence_weekend' if is_weekend else 'meal_adherence_weekday'
            
            # 3. Obtener el historial de adherencia (por defecto 1.0 para todas las comidas)
            historical_adherence = health_profile.get(profile_key, {})
            if not isinstance(historical_adherence, dict):
                historical_adherence = {}
            
            # 4. Multi-Horizon EMA: corto plazo (responsivo) + largo plazo (estratégico)
            tuning_metrics = health_profile.get("tuning_metrics", {})

            # Alpha corto: auto-tunable (0.3–0.8) — ventana ~3–7 días, detecta caídas rápidas
            alpha_short = tuning_metrics.get("ema_alpha", 0.3)
            # Alpha largo: fijo 0.15 — ventana ~30 días, captura tendencias estacionales
            ALPHA_LONG = 0.15

            all_meal_types = set(list(current_cycle_adherence.keys()) + list(historical_adherence.keys()))

            # Historial del EMA largo (persiste en clave separada)
            long_profile_key = profile_key + '_long'
            hist_long = health_profile.get(long_profile_key, {})
            if not isinstance(hist_long, dict):
                hist_long = {}

            short_ema = {}
            long_ema = {}
            divergences = []
            for mt in all_meal_types:
                current_val = current_cycle_adherence.get(mt, 1.0)
                hist_short_val = historical_adherence.get(mt, 1.0)
                hist_long_val = hist_long.get(mt, 1.0)
                short_ema[mt] = round((alpha_short * current_val) + ((1 - alpha_short) * hist_short_val), 2)
                long_ema[mt] = round((ALPHA_LONG * current_val) + ((1 - ALPHA_LONG) * hist_long_val), 2)
                divergences.append(abs(current_val - hist_short_val))

            # AUTO-TUNE del alpha corto (sin tocar el largo que es fijo)
            if divergences:
                avg_div = sum(divergences) / len(divergences)
                div_history = tuning_metrics.get("ema_divergence_history", [])
                div_history.append(avg_div)
                div_history = div_history[-3:]
                if len(div_history) >= 3:
                    mean_div = sum(div_history) / 3
                    if mean_div > 0.4 and alpha_short < 0.8:
                        alpha_short = round(alpha_short + 0.05, 2)
                        logger.info(f" [AUTO-TUNE] Alta divergencia EMA ({mean_div:.2f}). Aumentando alpha_short a {alpha_short}.")
                    elif mean_div < 0.1 and alpha_short > 0.1:
                        alpha_short = round(alpha_short - 0.05, 2)
                        logger.info(f" [AUTO-TUNE] Baja divergencia EMA ({mean_div:.2f}). Reduciendo alpha_short a {alpha_short}.")
                tuning_metrics["ema_divergence_history"] = div_history
                tuning_metrics["ema_alpha"] = alpha_short
                execute_sql_write("UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{tuning_metrics}', %s::jsonb) WHERE id = %s", (json.dumps(tuning_metrics), user_id))

            # 5. Determinar hint contextual con ambos horizontes
            avg_short = sum(short_ema.values()) / len(short_ema) if short_ema else 1.0
            avg_long = sum(long_ema.values()) / len(long_ema) if long_ema else 1.0

            if avg_short < 0.3 and avg_long > 0.6:
                ema_hint = "temporary_dip"   # caída reciente, históricamente buen usuario → comfort food
            elif avg_short < 0.3 and avg_long < 0.4:
                ema_hint = "drastic_change"  # consistentemente bajo → intervención drástica
            elif avg_short > 0.6 and avg_long < 0.4:
                ema_hint = "improving"       # recuperación reciente → reforzar progreso
            else:
                ema_hint = "stable"

            logger.info(f" [DUAL-EMA] short={avg_short:.2f} long={avg_long:.2f} hint='{ema_hint}' ({profile_key})")

            # Inyectar ambas señales al pipeline
            pipeline_data['_meal_level_adherence'] = short_ema          # urgente: alarmas inmediatas
            pipeline_data['_meal_level_adherence_long'] = long_ema       # estratégico: tendencia
            pipeline_data['_adherence_ema_hint'] = ema_hint

            # Calcular adherencia general para el hint binario existente (compatibilidad)
            adherence_score = (len(consumed_records) / household_size) / expected_meals_count
            if adherence_score < 0.3:
                pipeline_data['_adherence_hint'] = 'low'
                logger.info(f" [FEEDBACK LOOP] Baja adherencia detectada (Score: {adherence_score:.2f}). Se pedirá simplificar el plan.")
            elif adherence_score > 0.8:
                pipeline_data['_adherence_hint'] = 'high'
                logger.info(f" [FEEDBACK LOOP] Alta adherencia detectada (Score: {adherence_score:.2f}). Se permitirá mayor variedad/creatividad.")

            # 6. Guardar ambos EMA en health_profile
            execute_sql_write(
                f"UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{{{profile_key}}}', %s::jsonb) WHERE id = %s",
                (json.dumps(short_ema), user_id)
            )
            execute_sql_write(
                f"UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{{{long_profile_key}}}', %s::jsonb) WHERE id = %s",
                (json.dumps(long_ema), user_id)
            )
    except Exception as e:
        logger.warning(f" [FEEDBACK LOOP] Error calculando adherencia: {e}")

    # GAP 2: Causal Loop - Extraer razones de abandono
    try:
        causal_reasons_raw = execute_sql_query(
            """SELECT meal_type, reason
               FROM abandoned_meal_reasons
               WHERE user_id = %s
                 AND created_at >= NOW() - INTERVAL '14 days'
                 AND NOT (
                     reason IN ('swap:cravings', 'swap:weekend', 'swap:variety')
                     AND created_at < NOW() - INTERVAL '48 hours'
                 )""",
            (user_id,),
            fetch_all=True,
        )
        if causal_reasons_raw:
            abandoned_reasons = {}
            for row in causal_reasons_raw:
                mt = row['meal_type']
                r = row['reason']
                if mt not in abandoned_reasons:
                    abandoned_reasons[mt] = []
                abandoned_reasons[mt].append(r)
            
            # Obtener la razon mas comun por comida
            most_common_reasons = {}
            for mt, reasons in abandoned_reasons.items():
                from collections import Counter
                most_common_reasons[mt] = Counter(reasons).most_common(1)[0][0]
            
            pipeline_data['_abandoned_reasons'] = most_common_reasons
            logger.info(f" [CAUSAL LOOP] Razones de abandono extraidas: {most_common_reasons}")
    except Exception as e:
        logger.warning(f" [CAUSAL LOOP] Error extrayendo razones de abandono: {e}")

    # GAP 4: Emotional State - Ajustar plan al estado animico reciente
    try:
        recent_sentiments = execute_sql_query(
            "SELECT response_sentiment FROM nudge_outcomes WHERE user_id = %s AND response_sentiment IS NOT NULL ORDER BY sent_at DESC LIMIT 3",
            (user_id,),
            fetch_all=True,
        )
        if recent_sentiments:
            sentiments = [row['response_sentiment'] for row in recent_sentiments]
            from collections import Counter
            dominant_sentiment = Counter(sentiments).most_common(1)[0][0]
            
            needs_comfort_sentiments = ['frustration', 'sadness', 'guilt', 'annoyed']
            ready_challenge_sentiments = ['motivation', 'positive', 'curiosity']
            
            if dominant_sentiment in needs_comfort_sentiments:
                pipeline_data['_emotional_state'] = 'needs_comfort'
                logger.info(f" [EMOTIONAL STATE] Usuario {user_id} necesita confort (Sentimiento: {dominant_sentiment})")
            elif dominant_sentiment in ready_challenge_sentiments:
                pipeline_data['_emotional_state'] = 'ready_for_challenge'
                logger.info(f"[EMOTIONAL STATE] Usuario {user_id} listo para reto (Sentimiento: {dominant_sentiment})")
    except Exception as e:
        logger.warning(f" [EMOTIONAL STATE] Error extrayendo sentimiento: {e}")

    # MEJORA 4: Auto-Evaluacion (Quality Score) y Consecuencias Adaptativas
    try:
        household_size = max(1, int(health_profile.get('householdSize', 1)))
        quality_score = calculate_plan_quality_score(user_id, {'days': days}, consumed_records, household_size)
        pipeline_data['_previous_plan_quality'] = quality_score
        logger.info(f" [SELF-EVALUATION] Calidad del Plan Anterior para {user_id}: {quality_score:.2f}")
        
        # --- MEJORA 2: ATTRIBUTION TRACKER (CLOSED-LOOP) ---
        try:
            from db_plans import get_latest_meal_plan
            prev_plan = get_latest_meal_plan(user_id)
            if prev_plan:
                # --- GAP 2: JUDGE FEEDBACK LOOP ---
                adv_winner = prev_plan.get("_adversarial_winner")
                if adv_winner:
                    # [P0-1-RECOVERY/WORKER-FIX] `json` ya es global en este módulo;
                    # importarlo localmente lo promovía a variable local de
                    # _inject_advanced_learning_signals y rompía cualquier uso anterior
                    # (e.g., json.loads en L3295) con UnboundLocalError.
                    judge_calib = health_profile.get("judge_calibration", {"hits": 0, "total": 0, "score": 1.0})
                    # Si el quality_score > 0.6 o supera el promedio histórico, el juez tomó una buena decisión.
                    is_hit = 1 if quality_score > 0.6 else 0
                    
                    judge_calib["total"] += 1
                    judge_calib["hits"] += is_hit
                    judge_calib["score"] = round(judge_calib["hits"] / max(1, judge_calib["total"]), 2)
                    
                    health_profile["judge_calibration"] = judge_calib
                    execute_sql_write(
                        "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{judge_calibration}', %s::jsonb) WHERE id = %s",
                        (json.dumps(judge_calib), user_id)
                    )
                    logger.info(f" [JUDGE FEEDBACK] Calibración actualizada: {judge_calib['score']} (Hits: {judge_calib['hits']}/{judge_calib['total']})")

            if prev_plan and "_active_learning_signals" in prev_plan:
                signals_snapshot = prev_plan["_active_learning_signals"]
                if signals_snapshot:
                    attribution_tracker = health_profile.get("attribution_tracker", {})
                    # [P0-1-RECOVERY/WORKER-FIX] datetime/timezone/timedelta/json vienen de globals.
                    for signal_key, signal_value in signals_snapshot.items():
                        tracker_key = f"{signal_key}:{signal_value}" if isinstance(signal_value, str) else str(signal_key)
                        stats = attribution_tracker.get(tracker_key, {"avg_score": 0.0, "count": 0})
                        
                        # --- GAP 1: Signal Decay Temporal ---
                        last_updated = stats.get("last_updated")
                        if last_updated:
                            try:
                                # Manejo seguro de Z y parseo de fecha
                                last_dt_str = str(last_updated)
                                if last_dt_str.endswith('Z'):
                                    last_dt_str = last_dt_str[:-1] + '+00:00'
                                last_dt = datetime.fromisoformat(last_dt_str)
                                if last_dt.tzinfo is None:
                                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                                
                                age_days = (datetime.now(timezone.utc) - last_dt).days
                                if age_days > 60:
                                    # Reset parcial: reducimos count a la mitad para dar peso a las nuevas observaciones
                                    # y olvidamos parcialmente el rendimiento histórico lejano.
                                    stats["count"] = max(1, stats["count"] // 2)
                            except Exception as e:
                                logger.warning(f" [ATTRIBUTION DECAY] Error parseando last_updated: {e}")
                        
                        new_count = stats["count"] + 1
                        new_avg = ((stats["avg_score"] * stats["count"]) + quality_score) / new_count
                        
                        attribution_tracker[tracker_key] = {
                            "avg_score": round(new_avg, 3),
                            "count": new_count,
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        }
                    
                    execute_sql_write(
                        "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{attribution_tracker}', %s::jsonb) WHERE id = %s",
                        (json.dumps(attribution_tracker), user_id)
                    )
                    health_profile["attribution_tracker"] = attribution_tracker  # Actualizar en memoria también
                    logger.info(f" [ATTRIBUTION] Quality Score {quality_score:.2f} atribuido a señales: {list(signals_snapshot.keys())}")
        except Exception as e:
            logger.warning(f" [ATTRIBUTION] Error procesando attribution tracker: {e}")
        
        # [A/B TESTING] Resolver experimento previo
        try:
            active_exp_id = health_profile.get("active_experiment_id")
            if active_exp_id:
                execute_sql_write("UPDATE learning_experiments SET outcome_quality_score = %s WHERE id = %s", (quality_score, active_exp_id))
                execute_sql_write("UPDATE user_profiles SET health_profile = health_profile - 'active_experiment_id' WHERE id = %s", (user_id,))
                logger.info(f" [A/B TESTING] Experimento {active_exp_id} resuelto con Quality Score: {quality_score:.2f}")
        except Exception as e:
            logger.warning(f" [A/B TESTING] Error resolviendo experimento: {e}")
        
        # MEJORA 3: Counterfactual Attribution — evaluar señales podadas en ciclos anteriores
        try:
            counterfactual_pending = health_profile.get("counterfactual_pending", {})
            if counterfactual_pending:
                attribution_tracker_cf = health_profile.get("attribution_tracker", {})
                reinstated = []
                confirmed_pruned = []

                for tracker_key, cf_data in list(counterfactual_pending.items()):
                    original_avg = cf_data.get("original_avg_score", 0.0)
                    pruned_at_str = cf_data.get("pruned_at", "")

                    # TTL: si fue podada hace >30 días, confirmar sin evaluar
                    try:
                        pruned_dt = datetime.fromisoformat(pruned_at_str.replace("Z", "+00:00"))
                        age_days = (datetime.now(timezone.utc) - pruned_dt).days
                    except Exception:
                        age_days = 999

                    if age_days > 30:
                        confirmed_pruned.append(tracker_key)
                        continue

                    # quality_score de ESTE ciclo = plan generado SIN la señal = contrafactual
                    delta = quality_score - original_avg
                    if delta < -0.15:
                        # El plan empeoró SIN la señal → era útil → re-instaurar
                        if tracker_key in attribution_tracker_cf:
                            # Subir avg_score por encima del umbral de poda (0.4) sin borrar historial
                            attribution_tracker_cf[tracker_key]["avg_score"] = max(
                                attribution_tracker_cf[tracker_key].get("avg_score", 0.0),
                                0.50
                            )
                            attribution_tracker_cf[tracker_key]["last_updated"] = datetime.now(timezone.utc).isoformat()
                        reinstated.append(f"{tracker_key} (delta={delta:+.2f}, orig={original_avg:.2f})")
                    else:
                        # El plan se mantuvo o mejoró → el pruning fue correcto
                        confirmed_pruned.append(tracker_key)

                # Limpiar entradas ya evaluadas
                new_pending = {k: v for k, v in counterfactual_pending.items()
                               if k not in reinstated and k not in confirmed_pruned}

                if reinstated:
                    logger.info(f" [COUNTERFACTUAL] Señales re-instauradas (el plan empeoró sin ellas): {', '.join(reinstated)}")
                    execute_sql_write(
                        "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{attribution_tracker}', %s::jsonb) WHERE id = %s",
                        (json.dumps(attribution_tracker_cf), user_id)
                    )
                    health_profile["attribution_tracker"] = attribution_tracker_cf
                if confirmed_pruned:
                    logger.info(f" [COUNTERFACTUAL] Podas confirmadas (plan no empeoró sin señal): {', '.join(confirmed_pruned)}")

                execute_sql_write(
                    "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{counterfactual_pending}', %s::jsonb) WHERE id = %s",
                    (json.dumps(new_pending), user_id)
                )
                health_profile["counterfactual_pending"] = new_pending
        except Exception as e:
            logger.warning(f" [COUNTERFACTUAL] Error en evaluación contrafactual: {e}")

        # [GAP 8] Diferenciar fuente de chunks (para evitar saturar con shifts diarios)
        quality_history_chunks = health_profile.get('quality_history_chunks', [])
        if not isinstance(quality_history_chunks, list):
            quality_history_chunks = []
            
        quality_history_chunks.append(quality_score)
        quality_history_chunks = quality_history_chunks[-5:] # Mantener los últimos 5 ciclos de chunks
        
        pipeline_data['quality_history_chunks'] = quality_history_chunks
        
        # Analizar tendencias para inyectar un hint al LLM
        if len(quality_history_chunks) >= 3:
            last_3 = quality_history_chunks[-3:]
            if all(score < 0.3 for score in last_3):
                pipeline_data['_quality_hint'] = 'drastic_change'
                
                # [A/B TESTING] Iniciar nuevo experimento Epsilon-Greedy
                import random
                try:
                    strategies = ["ethnic_rotation", "texture_swap", "protein_shock"]
                    chosen_strategy = random.choice(strategies)
                    
                    past_exps = execute_sql_query(
                        "SELECT strategy_applied, AVG(outcome_quality_score) as avg_score FROM learning_experiments WHERE user_id = %s AND outcome_quality_score IS NOT NULL GROUP BY strategy_applied ORDER BY avg_score DESC",
                        (user_id,), fetch_all=True
                    )
                    
                    # --- GAP 3: EPSILON DECAY PARA A/B TESTING ---
                    # En lugar de un 20% estático, reducimos la exploración conforme acumulamos experimentos.
                    total_exps_query = execute_sql_query("SELECT COUNT(*) as c FROM learning_experiments WHERE user_id = %s", (user_id,), fetch_all=True)
                    total_experiments = total_exps_query[0]["c"] if total_exps_query else 0
                    epsilon = max(0.05, 0.3 * (0.95 ** total_experiments))
                    
                    if past_exps and random.random() > epsilon:
                        best_strategy = past_exps[0]["strategy_applied"]
                        if past_exps[0].get("avg_score", 0) > 0.5:
                            chosen_strategy = best_strategy
                            
                    pipeline_data['_drastic_change_strategy'] = chosen_strategy
                    
                    res = execute_sql_query(
                        "INSERT INTO learning_experiments (user_id, strategy_applied) VALUES (%s, %s) RETURNING id",
                        (user_id, chosen_strategy), fetch_one=True
                    )
                    if res and res.get("id"):
                        exp_id = res["id"]
                        execute_sql_write("UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{active_experiment_id}', %s::jsonb) WHERE id = %s", (json.dumps(exp_id), user_id))
                        logger.info(f" [A/B TESTING] Iniciado experimento {exp_id} con estrategia: {chosen_strategy}")
                        
                except Exception as e:
                    logger.warning(f" [A/B TESTING] Error orquestando experimento: {e}")
                
                logger.warning(f" [FEEDBACK LOOP] Quality Score muy bajo por 3 ciclos consecutivos. Se activará un CAMBIO RADICAL (Estrategia: {pipeline_data.get('_drastic_change_strategy', 'default')}).")
            elif all(score > 0.8 for score in last_3):
                pipeline_data['_quality_hint'] = 'increase_complexity'
                logger.info(f" [FEEDBACK LOOP] Quality Score muy alto por 3 ciclos consecutivos. Se permitirá MAYOR COMPLEJIDAD.")
        
        # Detectar Plateau Silencioso (GAP 6 / 8)
        if len(quality_history_chunks) >= 4 and not pipeline_data.get('_quality_hint'):
            mean_q = sum(quality_history_chunks) / len(quality_history_chunks)
            variance = sum((q - mean_q)**2 for q in quality_history_chunks) / len(quality_history_chunks)
            if variance < 0.01 and mean_q < 0.6:
                pipeline_data['_quality_hint'] = 'break_plateau'
                logger.warning(f" [FEEDBACK LOOP] Plateau Silencioso detectado. Quality score estancado en {mean_q:.2f}. Se forzará una ruptura de monotonía.")
                
        # Detectar Plateau de Adherencia (Mejora 2)
        adherence_history = health_profile.get('adherence_history_rotations', [])
        if len(adherence_history) >= 3 and not pipeline_data.get('_quality_hint'):
            last_3_adherence = adherence_history[-3:]
            # Check for consistent drop: e.g. 0.8 -> 0.7 -> 0.6
            is_dropping = all(last_3_adherence[i] < last_3_adherence[i-1] for i in range(1, 3))
            if is_dropping and last_3_adherence[-1] < 0.65:
                pipeline_data['_quality_hint'] = 'simplify_urgently'
                logger.warning(f" [FEEDBACK LOOP] Plateau de Adherencia detectado. Cayendo consistentemente a {last_3_adherence[-1]:.2f}. Se forzará a simplificar el plan.")
        
        # Guardamos el score y el historial en el health_profile
        execute_sql_write(
            "UPDATE user_profiles SET health_profile = jsonb_set(jsonb_set(health_profile, '{last_plan_quality}', %s::jsonb), '{quality_history_chunks}', %s::jsonb) WHERE id = %s",
            (json.dumps(quality_score), json.dumps(quality_history_chunks), user_id)
        )
    except Exception as e:
        logger.warning(f" [SELF-EVALUATION] Error calculando Quality Score: {e}")


    # MEJORA 2: Historial de Peso para Metabolismo Evolutivo
    try:
        weight_log = execute_sql_query(
            "SELECT weight, unit, created_at::date as date FROM weight_log WHERE user_id = %s ORDER BY created_at",
            (user_id,), fetch_all=True
        )
        if weight_log:
            pipeline_data['weight_history'] = [
                {"date": str(w['date']), "weight": w['weight'], "unit": w.get('unit', 'lb')}
                for w in weight_log
            ]
            logger.info(f" [METABOLISMO EVOLUTIVO] {len(weight_log)} registros de peso cargados para el usuario {user_id}")
    except Exception as e:
        logger.error(f" [METABOLISMO EVOLUTIVO] Error cargando historial de peso: {e}")

    # MEJORA 3: Aprendizaje de Patrones de Ã‰xito y Temporalidad
    try:
        # [GAP 7] Ventana dinámica de aprendizaje basada en el offset del plan
        days_offset = int(pipeline_data.get('_days_offset', 0))
        dynamic_days_back = max(14, days_offset + 7)
        
        succ_techs, aban_techs, freq_meals = calculate_meal_success_scores(user_id, days_back=dynamic_days_back)
        day_adherence = calculate_day_of_week_adherence(user_id, days_back=max(30, dynamic_days_back))
        
        pipeline_data['successful_techniques'] = list(set(succ_techs))
        pipeline_data['abandoned_techniques'] = list(set(aban_techs))
        pipeline_data['frequent_meals'] = freq_meals
        pipeline_data['day_of_week_adherence'] = day_adherence
        
        logger.info(f" [PATRONES DE Ã‰XITO] {len(succ_techs)} exitos, {len(aban_techs)} abandonos extraidos para {user_id}")
        logger.info(f" [TEMPORALIDAD] Perfil de dias: {day_adherence}")
    except Exception as e:
        logger.error(f" [PATRONES DE Ã‰XITO] Error calculando scores o temporalidad: {e}")

    # MEJORA 5 y 2: Sincronizacion Nudge <-> Rotacion con Efectividad Real
    try:
        nudge_data = execute_sql_query(
            "SELECT nudge_type, responded, meal_logged, response_sentiment FROM nudge_outcomes "
            "WHERE user_id = %s AND sent_at > NOW() - INTERVAL '7 days'",
            (user_id,), fetch_all=True
        )
        if nudge_data:
            ignored_meal_types = []
            frustrated_meal_types = []
            for n in nudge_data:
                if not n.get('responded') and not n.get('meal_logged'):
                    ignored_meal_types.append(n.get('nudge_type'))
                sentiment = n.get('response_sentiment')
                if sentiment in ['annoyed', 'frustration', 'sadness', 'guilt']:
                    frustrated_meal_types.append(n.get('nudge_type'))
                    
            if ignored_meal_types:
                pipeline_data['_ignored_meal_types'] = list(set(ignored_meal_types))
            if frustrated_meal_types:
                pipeline_data['_frustrated_meal_types'] = list(set(frustrated_meal_types))
                
        # Filtrar nudge_type por efectividad REAL
        effective_nudge_query = """
            SELECT nudge_type, 
            AVG(CASE WHEN meal_logged THEN 1.0 ELSE 0.0 END) as conversion_rate
            FROM nudge_outcomes 
            WHERE user_id = %s
            GROUP BY nudge_type
        """
        try:
            effective_nudges = execute_sql_query(effective_nudge_query, (user_id,), fetch_all=True)
            if effective_nudges:
                pipeline_data['_nudge_conversion_rates'] = {
                    n['nudge_type']: float(n['conversion_rate']) for n in effective_nudges if n['conversion_rate'] is not None
                }
                logger.info(f" [NUDGE SYNC] Tasas de conversion reales agregadas al contexto.")
        except Exception as eff_err:
            pass

        # MEJORA Gap F: Tonos de comunicacion exitosos
        try:
            successful_styles = execute_sql_query(
                "SELECT nudge_style, COUNT(*) as successes FROM nudge_outcomes WHERE user_id = %s AND nudge_style IS NOT NULL AND (meal_logged = true OR response_sentiment IN ('motivation', 'positive', 'happy', 'excited')) GROUP BY nudge_style ORDER BY successes DESC LIMIT 2",
                (user_id,), fetch_all=True
            )
            if successful_styles:
                styles_list = [row['nudge_style'] for row in successful_styles]
                pipeline_data['_successful_tone_strategies'] = styles_list
                logger.info(f"  [TONE SYNC] Estilos de comunicacion exitosos inyectados: {styles_list}")
        except Exception as style_err:
            pass
            
    except Exception as e:
        if "relation" not in str(e).lower() and "column" not in str(e).lower():
            logger.warning(f" [NUDGE SYNC] Error consultando nudge_outcomes: {e}")

    # MEJORA 7: Cold-Start Intelligence (Collaborative Filtering Ligero)
    try:
        if not consumed_records or len(consumed_records) < 3:
            popular_meals_data = get_similar_user_patterns(user_id, health_profile)
            if popular_meals_data:
                popular_names = [p['meal_name'] for p in popular_meals_data if p.get('meal_name')]
                pipeline_data['_cold_start_recommendations'] = popular_names
                logger.info(f" [COLD-START] Inyectando {len(popular_names)} platos populares de usuarios similares para {user_id}")
    except Exception as e:
        logger.warning(f" [COLD-START] Error procesando recomendaciones de inicio en frio: {e}")

    # MEJORA 8: Explicit Likes (❤️)
    try:
        likes_data = execute_sql_query(
            "SELECT meal_name FROM meal_likes WHERE user_id = %s",
            (user_id,), fetch_all=True
        )
        if likes_data:
            liked_meals = [row['meal_name'] for row in likes_data if row.get('meal_name')]
            if liked_meals:
                # Deduplicate and keep the most recent ones (assuming later inserts are at the end)
                liked_meals = list(dict.fromkeys(liked_meals))[-20:]
                pipeline_data['_liked_meals'] = liked_meals
                logger.info(f" [LIKES SYNC] Inyectando {len(liked_meals)} platos likeados al contexto para {user_id}.")
                
        liked_flavor_profiles = health_profile.get('liked_flavor_profiles', [])
        if liked_flavor_profiles:
            pipeline_data['_liked_flavor_profiles'] = liked_flavor_profiles
            logger.info(f" [LIKES SYNC] Inyectando {len(liked_flavor_profiles)} perfiles de sabor al contexto.")
    except Exception as e:
        logger.warning(f" [LIKES SYNC] Error consultando meal_likes: {e}")

    # MEJORA 2: Attribution Pruning (Closed-Loop)
    try:
        attribution_tracker = health_profile.get("attribution_tracker", {})
        if attribution_tracker:
            pruned_signals = []
            
            # Map the signals we care about pruning
            signals_to_check = {
                "quality_hint": "_quality_hint",
                "adherence_hint": "_adherence_hint",
                "emotional_state": "_emotional_state",
                "drastic_strategy": "_drastic_change_strategy",
                "cold_start": "_cold_start_recs"
            }
            
            for base_key, pipeline_key in signals_to_check.items():
                signal_value = pipeline_data.get(pipeline_key)
                if signal_value:
                    tracker_key = f"{base_key}:{signal_value}" if isinstance(signal_value, str) else str(base_key)
                    stats = attribution_tracker.get(tracker_key)
                    
                    if stats:
                        # PRUNING THRESHOLDS: attempted at least 2 times, avg score < 0.4
                        if stats.get("count", 0) >= 2 and stats.get("avg_score", 1.0) < 0.4:
                            # Prune it!
                            del pipeline_data[pipeline_key]
                            pruned_signals.append(f"{pipeline_key}={signal_value} (score: {stats.get('avg_score')})")
                            # MEJORA 3: Registrar en counterfactual_pending para medir impacto real en el siguiente ciclo
                            cf_pending = health_profile.get("counterfactual_pending", {})
                            cf_pending[tracker_key] = {
                                "pruned_at": datetime.now(timezone.utc).isoformat(),
                                "original_avg_score": round(stats.get("avg_score", 0.0), 3),
                                "pipeline_key": pipeline_key,
                                "signal_value": str(signal_value),
                            }
                            health_profile["counterfactual_pending"] = cf_pending
                            
            if pruned_signals:
                logger.warning(f" [ATTRIBUTION PRUNING] Señales podadas por bajo rendimiento histórico: {', '.join(pruned_signals)}")
                # Persistir counterfactual_pending actualizado (acumulado en el loop de pruning)
                execute_sql_write(
                    "UPDATE user_profiles SET health_profile = jsonb_set(health_profile, '{counterfactual_pending}', %s::jsonb) WHERE id = %s",
                    (json.dumps(health_profile.get("counterfactual_pending", {})), user_id)
                )
    except Exception as e:
        logger.warning(f" [ATTRIBUTION PRUNING] Error procesando la poda: {e}")

    # P1: AUTO-ACTIVACIÓN AUTÓNOMA del Adversarial Self-Play (Cron Path)
    # Decide si activar el adversarial self-play basándose en la salud del pipeline.
    try:
        if not pipeline_data.get('_use_adversarial_play'):
            _auto_reasons = []

            # Condición 1: Quality Score bajo sostenido (< 0.5 por 2+ ciclos)
            qh = health_profile.get("quality_history_chunks", [])
            if isinstance(qh, list) and len(qh) >= 2 and all(s < 0.5 for s in qh[-2:]):
                _auto_reasons.append("quality_low_sustained")

            # Condición 2: Alta varianza en Attribution Tracker
            attr_tracker = health_profile.get("attribution_tracker", {})
            if len(attr_tracker) >= 3:
                scores = [v.get("avg_score", 0.5) for v in attr_tracker.values() if isinstance(v, dict)]
                if scores:
                    mean_s = sum(scores) / len(scores)
                    variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
                    if variance > 0.06:
                        _auto_reasons.append(f"attribution_high_variance ({variance:.3f})")

            # Condición 3: Historial de rechazos médicos frecuentes
            rejection_patterns = health_profile.get("rejection_patterns", [])
            if isinstance(rejection_patterns, list) and len(rejection_patterns) >= 5:
                _auto_reasons.append(f"frequent_rejections ({len(rejection_patterns)})")

            if _auto_reasons:
                pipeline_data['_use_adversarial_play'] = True
                logger.info(f" [ADVERSARIAL AUTO-ACTIVATE] Activado para {user_id}: {', '.join(_auto_reasons)}")
    except Exception as e:
        logger.warning(f" [ADVERSARIAL AUTO-ACTIVATE] Error evaluando condiciones: {e}")

    return pipeline_data


# [P1-7] Cache TTL para inject_learning_signals_from_profile.
# Antes, cuando el SSE fallaba (LLM circuit breaker, timeout) y el frontend
# caía al endpoint sync, esta función volvía a ejecutar ~7-8 SELECTs a Postgres
# (~50-300ms) que ya habían corrido segundos antes. No corrompe datos
# (read-only + setdefault), pero desperdicia latencia + un slot de connection
# pool en el path donde el usuario ya está esperando un fallback. 5 minutos
# cubre el escenario típico SSE→sync (ocurre en <1 minuto por timeout).
#
# Multi-worker: cache in-memory por proceso. Si el segundo request cae en
# OTRO worker (Gunicorn round-robin), no hay hit y se re-consulta — mismo
# comportamiento que hoy. Para distribución cross-worker se podría migrar a
# Redis con la misma API; out of scope para P1-7.
_SIGNAL_CACHE: dict = {}  # user_id → (monotonic_cached_at, injected_signals_dict)
_SIGNAL_CACHE_TTL_S = 300
_SIGNAL_CACHE_LOCK = None  # lazy init en primer uso para evitar early threading import


def _signal_cache_lock():
    global _SIGNAL_CACHE_LOCK
    if _SIGNAL_CACHE_LOCK is None:
        import threading as _t
        _SIGNAL_CACHE_LOCK = _t.Lock()
    return _SIGNAL_CACHE_LOCK


def _signal_cache_get(user_id: str):
    """Retorna el dict de señales cacheado si está fresco, o `None`."""
    import time as _t
    lock = _signal_cache_lock()
    with lock:
        entry = _SIGNAL_CACHE.get(user_id)
    if not entry:
        return None
    cached_at, signals = entry
    if _t.monotonic() - cached_at >= _SIGNAL_CACHE_TTL_S:
        return None
    return signals


def _signal_cache_set(user_id: str, signals: dict) -> None:
    """Cachea las señales y hace GC oportunista de entries stale (>2× TTL)."""
    import time as _t
    now = _t.monotonic()
    lock = _signal_cache_lock()
    with lock:
        _SIGNAL_CACHE[user_id] = (now, signals)
        # GC oportunista: previene leak si el proceso vive días con muchos users.
        stale = [
            uid for uid, (ts, _) in _SIGNAL_CACHE.items()
            if now - ts > _SIGNAL_CACHE_TTL_S * 2
        ]
        for uid in stale:
            _SIGNAL_CACHE.pop(uid, None)


def inject_learning_signals_from_profile(user_id: str, pipeline_data: dict) -> dict:
    """Inyecta señales de aprendizaje para generaciones manuales (API path).

    Equivalente ligero de _inject_advanced_learning_signals (cron path).
    Lee señales persistidas del health_profile + queries ligeros en vivo.
    Solo escribe keys que NO estén ya presentes (no sobreescribe).

    [P1-7] Cacheado in-memory por 5 minutos para mitigar doble ejecución
    cuando el frontend cae del SSE al endpoint sync (segundo request reusa
    señales del primero sin re-consultar DB).
    """
    from db_core import execute_sql_query
    from datetime import datetime, timezone, timedelta

    # [P1-7] Cache hit → fusionar señales cacheadas vía setdefault y salir
    # sin tocar DB. La fusión usa setdefault para preservar valores que el
    # caller ya pudo haber inyectado en `pipeline_data` (e.g., overrides
    # explícitos pasados por la request).
    if user_id:
        _cached = _signal_cache_get(user_id)
        if _cached is not None:
            for _k, _v in _cached.items():
                pipeline_data.setdefault(_k, _v)
            logger.info(
                f" [SIGNAL INJECT/CACHE-HIT] {len(_cached)} señales servidas desde "
                f"cache para user={user_id} (TTL {_SIGNAL_CACHE_TTL_S}s, sin DB hits)."
            )
            return pipeline_data

    # [P1-7] Snapshot de keys ANTES de la lógica original — al final calculamos
    # el delta para cachear solo lo que realmente inyectamos en este call.
    _keys_before = set(pipeline_data.keys())

    try:
        profile_row = execute_sql_query(
            "SELECT health_profile FROM user_profiles WHERE id = %s",
            (user_id,), fetch_one=True
        )
        if not profile_row or not profile_row.get("health_profile"):
            return pipeline_data
        hp = profile_row["health_profile"]
    except Exception as e:
        logger.warning(f" [SIGNAL INJECT] Error reading health_profile for {user_id}: {e}")
        return pipeline_data

    # ── 1. Señales persistidas del health_profile ──

    # EMA Adherence (weekday/weekend)
    is_weekend = datetime.now(timezone.utc).weekday() >= 5
    profile_key = 'meal_adherence_weekend' if is_weekend else 'meal_adherence_weekday'
    meal_adherence = hp.get(profile_key, {})
    if meal_adherence and not pipeline_data.get('_meal_level_adherence'):
        pipeline_data['_meal_level_adherence'] = meal_adherence
        avg = sum(meal_adherence.values()) / max(len(meal_adherence), 1)
        if avg < 0.3:
            pipeline_data.setdefault('_adherence_hint', 'low')
        elif avg > 0.8:
            pipeline_data.setdefault('_adherence_hint', 'high')

    # Quality Score & adaptive hint
    quality_score = hp.get('last_plan_quality')
    if quality_score is not None:
        pipeline_data.setdefault('_previous_plan_quality', quality_score)

    qh_chunks = hp.get('quality_history_chunks', [])
    if isinstance(qh_chunks, list) and qh_chunks and not pipeline_data.get('_quality_hint'):
        if len(qh_chunks) >= 3:
            last_3 = qh_chunks[-3:]
            if all(s < 0.3 for s in last_3):
                pipeline_data['_quality_hint'] = 'drastic_change'
                import random
                pipeline_data['_drastic_change_strategy'] = random.choice(
                    ["ethnic_rotation", "texture_swap", "protein_shock"]
                )
            elif all(s > 0.8 for s in last_3):
                pipeline_data['_quality_hint'] = 'increase_complexity'
        if len(qh_chunks) >= 4 and not pipeline_data.get('_quality_hint'):
            mean_q = sum(qh_chunks) / len(qh_chunks)
            var = sum((q - mean_q) ** 2 for q in qh_chunks) / len(qh_chunks)
            if var < 0.01 and mean_q < 0.6:
                pipeline_data['_quality_hint'] = 'break_plateau'
        # Adherence plateau check
        adh_hist = hp.get('adherence_history_rotations', [])
        if isinstance(adh_hist, list) and len(adh_hist) >= 3 and not pipeline_data.get('_quality_hint'):
            last_3a = adh_hist[-3:]
            is_dropping = all(last_3a[i] < last_3a[i - 1] for i in range(1, 3))
            if is_dropping and last_3a[-1] < 0.65:
                pipeline_data['_quality_hint'] = 'simplify_urgently'

    # LLM Retrospective
    if hp.get('llm_retrospective'):
        pipeline_data.setdefault('_llm_retrospective', hp['llm_retrospective'])

    # Liked Flavor Profiles
    if hp.get('liked_flavor_profiles'):
        pipeline_data.setdefault('_liked_flavor_profiles', hp['liked_flavor_profiles'])

    # Frequent meals (cold-start seed)
    if hp.get('previous_plan_frequent_meals'):
        pipeline_data.setdefault('frequent_meals', hp['previous_plan_frequent_meals'])

    # ── 2. Queries ligeros en vivo ──

    # Weight History
    try:
        if not pipeline_data.get('weight_history'):
            wl = execute_sql_query(
                "SELECT weight, unit, created_at::date as date FROM weight_log WHERE user_id = %s ORDER BY created_at",
                (user_id,), fetch_all=True
            )
            if wl:
                pipeline_data['weight_history'] = [
                    {"date": str(w['date']), "weight": w['weight'], "unit": w.get('unit', 'lb')} for w in wl
                ]
    except Exception:
        pass

    # Emotional State
    try:
        if not pipeline_data.get('_emotional_state'):
            rows = execute_sql_query(
                "SELECT response_sentiment FROM nudge_outcomes WHERE user_id = %s AND response_sentiment IS NOT NULL ORDER BY sent_at DESC LIMIT 3",
                (user_id,), fetch_all=True
            )
            if rows:
                from collections import Counter
                dom = Counter([r['response_sentiment'] for r in rows]).most_common(1)[0][0]
                if dom in ('frustration', 'sadness', 'guilt', 'annoyed'):
                    pipeline_data['_emotional_state'] = 'needs_comfort'
                elif dom in ('motivation', 'positive', 'curiosity'):
                    pipeline_data['_emotional_state'] = 'ready_for_challenge'
    except Exception:
        pass

    # Abandoned Reasons
    try:
        if not pipeline_data.get('_abandoned_reasons'):
            causal = execute_sql_query(
                """SELECT meal_type, reason 
                   FROM abandoned_meal_reasons 
                   WHERE user_id = %s 
                     AND created_at >= NOW() - INTERVAL '14 days'
                     AND NOT (
                         reason IN ('swap:cravings', 'swap:weekend', 'swap:variety') 
                         AND created_at < NOW() - INTERVAL '48 hours'
                     )""",
                (user_id,), fetch_all=True
            )
            if causal:
                from collections import Counter
                by_meal = {}
                for row in causal:
                    by_meal.setdefault(row['meal_type'], []).append(row['reason'])
                pipeline_data['_abandoned_reasons'] = {
                    mt: Counter(reasons).most_common(1)[0][0] for mt, reasons in by_meal.items()
                }
    except Exception:
        pass

    # Ingredient Fatigue
    try:
        if not pipeline_data.get('fatigued_ingredients'):
            fatigue = calculate_ingredient_fatigue(user_id, days_back=14, tuning_metrics=hp.get("tuning_metrics", {}))
            if fatigue and fatigue.get('fatigued_ingredients'):
                pipeline_data['fatigued_ingredients'] = fatigue['fatigued_ingredients']
    except Exception:
        pass

    # Successful/Abandoned Techniques + Day-of-Week Adherence
    # [P1-1 FIX] Usar ventana corta para rolling refills para evitar dilución EMA
    _is_rolling = pipeline_data.get('_is_rolling_refill', False)
    if _is_rolling:
        from constants import PLAN_CHUNK_SIZE
        _success_db = PLAN_CHUNK_SIZE + 1
        _adherence_db = PLAN_CHUNK_SIZE + 1
    else:
        _success_db = 14
        _adherence_db = 30
    try:
        if not pipeline_data.get('successful_techniques'):
            st, at, fm = calculate_meal_success_scores(user_id, days_back=_success_db)
            pipeline_data['successful_techniques'] = list(set(st))
            pipeline_data['abandoned_techniques'] = list(set(at))
            if fm:
                pipeline_data.setdefault('frequent_meals', fm)
        if not pipeline_data.get('day_of_week_adherence'):
            pipeline_data['day_of_week_adherence'] = calculate_day_of_week_adherence(user_id, days_back=_adherence_db)
    except Exception:
        pass

    # Nudge data (ignored, frustrated, conversion rates, tone strategies)
    try:
        nudge_rows = execute_sql_query(
            "SELECT nudge_type, nudge_style, responded, meal_logged, response_sentiment "
            "FROM nudge_outcomes WHERE user_id = %s AND sent_at > NOW() - INTERVAL '7 days'",
            (user_id,), fetch_all=True
        )
        if nudge_rows:
            if not pipeline_data.get('_ignored_meal_types'):
                ignored = list({n['nudge_type'] for n in nudge_rows if not n.get('responded') and not n.get('meal_logged')})
                if ignored:
                    pipeline_data['_ignored_meal_types'] = ignored
            if not pipeline_data.get('_frustrated_meal_types'):
                frust = list({n['nudge_type'] for n in nudge_rows if n.get('response_sentiment') in ('annoyed', 'frustration', 'sadness', 'guilt')})
                if frust:
                    pipeline_data['_frustrated_meal_types'] = frust
            if not pipeline_data.get('_nudge_conversion_rates'):
                from collections import defaultdict
                totals = defaultdict(lambda: [0, 0])
                for n in nudge_rows:
                    nt = n['nudge_type']
                    totals[nt][1] += 1
                    if n.get('meal_logged'):
                        totals[nt][0] += 1
                rates = {nt: round(v[0] / v[1], 2) for nt, v in totals.items() if v[1] > 0}
                if rates:
                    pipeline_data['_nudge_conversion_rates'] = rates
            if not pipeline_data.get('_successful_tone_strategies'):
                from collections import Counter
                success_styles = [n['nudge_style'] for n in nudge_rows
                                  if n.get('nudge_style') and (n.get('meal_logged') or n.get('response_sentiment') in ('motivation', 'positive', 'happy', 'excited'))]
                if success_styles:
                    pipeline_data['_successful_tone_strategies'] = [s for s, _ in Counter(success_styles).most_common(2)]
    except Exception:
        pass

    # Explicit Likes
    try:
        if not pipeline_data.get('_liked_meals'):
            likes_rows = execute_sql_query(
                "SELECT meal_name FROM meal_likes WHERE user_id = %s", (user_id,), fetch_all=True
            )
            if likes_rows:
                names = list(dict.fromkeys([r['meal_name'] for r in likes_rows if r.get('meal_name')]))[-20:]
                if names:
                    pipeline_data['_liked_meals'] = names
    except Exception:
        pass

    # Cold-Start Recommendations
    try:
        if not pipeline_data.get('_cold_start_recommendations'):
            from db_facts import get_consumed_meals_since
            recent = get_consumed_meals_since(user_id, since_iso_date=(datetime.now(timezone.utc) - timedelta(days=14)).isoformat())
            if not recent or len(recent) < 3:
                popular = get_similar_user_patterns(user_id, hp)
                if popular:
                    pipeline_data['_cold_start_recommendations'] = [p['meal_name'] for p in popular if p.get('meal_name')]
    except Exception:
        pass

    # P1: AUTO-ACTIVACIÓN AUTÓNOMA del Adversarial Self-Play (API Path)
    try:
        if not pipeline_data.get('_use_adversarial_play'):
            _auto_reasons = []

            qh = hp.get("quality_history_chunks", [])
            if isinstance(qh, list) and len(qh) >= 2 and all(s < 0.5 for s in qh[-2:]):
                _auto_reasons.append("quality_low_sustained")

            attr_tracker = hp.get("attribution_tracker", {})
            if len(attr_tracker) >= 3:
                scores = [v.get("avg_score", 0.5) for v in attr_tracker.values() if isinstance(v, dict)]
                if scores:
                    mean_s = sum(scores) / len(scores)
                    variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
                    if variance > 0.06:
                        _auto_reasons.append(f"attribution_high_variance ({variance:.3f})")

            rejection_patterns = hp.get("rejection_patterns", [])
            if isinstance(rejection_patterns, list) and len(rejection_patterns) >= 5:
                _auto_reasons.append(f"frequent_rejections ({len(rejection_patterns)})")

            if _auto_reasons:
                pipeline_data['_use_adversarial_play'] = True
                logger.info(f" [ADVERSARIAL AUTO-ACTIVATE] API path activado para {user_id}: {', '.join(_auto_reasons)}")
    except Exception as e:
        logger.warning(f" [ADVERSARIAL AUTO-ACTIVATE] Error en API path: {e}")

    injected_keys = [k for k in pipeline_data if k.startswith('_') or k in ('fatigued_ingredients', 'weight_history', 'successful_techniques', 'day_of_week_adherence', 'frequent_meals')]
    logger.info(f" [SIGNAL INJECT] {len(injected_keys)} señales inyectadas para generación manual: user={user_id}")

    # [P1-7] Cachear el delta (keys que esta llamada agregó) para que un
    # request subsiguiente del mismo user dentro de TTL las reuse sin re-DB.
    # Solo capturamos las keys NUEVAS — si el caller ya traía un valor
    # explícito en `pipeline_data`, lo preservamos via setdefault y NO lo
    # sobrescribimos en el cache (no es nuestro para guardarlo).
    if user_id:
        _new_keys = set(pipeline_data.keys()) - _keys_before
        if _new_keys:
            _signal_cache_set(
                user_id,
                {k: pipeline_data[k] for k in _new_keys},
            )

    return pipeline_data


# ============================================================
# BACKGROUND CHUNKING â€" Generacion de Semanas 2-4 en background
# ============================================================

def _compute_chunk_delay_days(
    days_offset: int,
    days_count: int,
    week_number: int,
    pipeline_snapshot: dict,
    chunk_kind: str | None = None,
    *,
    for_failed_retry: bool = False,
):
    """Calcula el delay del chunk según la política activa (`CHUNK_LEARNING_MODE`).

    Retorna `(delay_days, mode, days_offset_int, days_count_int)`. `delay_days`
    se suma a la fecha de inicio del plan para obtener el `execute_after` del
    chunk; un delay menor que `days_offset` significa que el chunk se generará
    ANTES de necesitarse (adelanto).

    Modos (ver docstring de `CHUNK_LEARNING_MODE` en constants.py para tradeoffs):

      - "strict": delay_days = days_offset (el chunk N+1 se programa para
        ejecutarse al iniciar su ventana). Excepción: chunks `initial_plan` con
        offset>0 reciben `CHUNK_PROACTIVE_MARGIN_DAYS` de adelanto (default 0).
        Reintentos de chunks failed reciben mínimo 1 día de adelanto.

      - "safety_margin": delay_days = max(0, days_offset - ceil(days_count/2)).
        Chunks de 3 días → 2 días de adelanto. Para planes >=15d el chunk final
        recibe adelanto adicional (delay = days_offset - 3) para asegurar que
        el usuario nunca se quede sin plan al final.

    `min(delay_days, 180)` clampa el horizonte máximo para evitar overflows en
    planes hipotéticos largos.
    """
    import math

    days_offset_int = max(0, int(days_offset))
    days_count_int = max(1, int(days_count))
    mode = (CHUNK_LEARNING_MODE or "strict").strip().lower()
    normalized_chunk_kind = (
        chunk_kind
        or ("rolling_refill" if (pipeline_snapshot or {}).get("_is_rolling_refill") else "initial_plan")
    ).strip().lower()

    if mode == "strict":
        proactive_margin_days = 0
        if normalized_chunk_kind == "initial_plan" and days_offset_int > 0:
            proactive_margin_days = CHUNK_PROACTIVE_MARGIN_DAYS
        if for_failed_retry:
            proactive_margin_days = max(proactive_margin_days, 1)
        delay_days = max(0, days_offset_int - proactive_margin_days)
    else:
        delay_days = max(0, days_offset_int - math.ceil(days_count_int / 2))

        try:
            total_days = int((pipeline_snapshot or {}).get("totalDays") or 0)
            if total_days >= 15:
                from constants import PLAN_CHUNK_SIZE
                total_weeks = math.ceil(total_days / PLAN_CHUNK_SIZE)
                if week_number >= total_weeks - 1:
                    delay_days = max(0, days_offset_int - 3)
                    logger.info(f" [GAP B] Chunk final {week_number}/{total_weeks} adelantado: delay={delay_days}d (mode={mode})")
        except Exception as e:
            logger.debug(f" [GAP B] No se pudo aplicar adelanto de chunk final: {e}")

    return min(delay_days, 180), mode, days_offset_int, days_count_int


def _compute_expected_preemption_seconds(days_offset: int, delay_days: int) -> int:
    """Cuánto del lag es esperado por adelantar intencionalmente el execute_after."""
    try:
        offset_days = max(0, int(days_offset))
        scheduled_days = max(0, int(delay_days))
        return max(0, offset_days - scheduled_days) * 86400
    except Exception:
        return 0


def _detect_proactive_zero_log_at_boundary(
    user_id: str,
    meal_plan_id: str,
    lookback_days: int,
) -> dict | None:
    """[P0-3] Detecta zero-log en la ventana del chunk previo (`lookback_days` hacia atrás).

    Antes el zero-log se descubría REACTIVAMENTE por el worker tras 1-2 deferrals
    (~2-4h de retraso silencioso). Esta función prueba PROACTIVAMENTE al momento
    de enqueueing chunk N+1: si en los últimos `lookback_days` días no hubo NI
    logs explícitos en `consumed_meals` NI mutaciones por consumo en
    `user_inventory`, la señal es: el usuario no consumió nada del chunk previo.

    Conservadora: si CUALQUIER query falla (DB blip, schema desconocido), retorna
    None — no queremos falsos positivos que pausen chunks de usuarios que sí están
    logueando. Solo retornamos detección cuando AMBAS fuentes confirman 0.

    Args:
        user_id: usuario destino del plan.
        meal_plan_id: plan padre del chunk (no se usa en queries actuales pero queda
            disponible para extensiones futuras como filtrar logs solo del plan).
        lookback_days: ventana hacia atrás desde NOW() en días. Típicamente
            `days_count` del chunk N+1 (3-4 días).

    Returns:
        dict con la evidencia si zero-log detectado, None si hay actividad o si
        alguna query falló (para ser conservadores con falsos positivos).
    """
    if not user_id or user_id == "guest" or lookback_days <= 0:
        return None
    since_dt = datetime.now(timezone.utc) - timedelta(days=int(lookback_days))
    since_iso = since_dt.isoformat()

    # 1. Logs explícitos en consumed_meals.
    log_count: int | None = None
    try:
        log_row = execute_sql_query(
            """
            SELECT COUNT(*) AS log_count
            FROM consumed_meals
            WHERE user_id = %s AND consumed_at >= %s
            """,
            (user_id, since_iso),
            fetch_one=True,
        )
        log_count = int((log_row or {}).get("log_count") or 0)
    except Exception as e:
        logger.debug(f"[P0-3/ZERO-LOG-PROBE] consumed_meals query falló para {user_id}: {e}")
        return None  # conservador: no marcar zero-log si no podemos confirmar

    # 2. Mutaciones de inventario por consumo.
    mutations: int | None = None
    try:
        from db_inventory import get_inventory_activity_since
        activity = get_inventory_activity_since(user_id, since_iso)
        mutations = int((activity or {}).get("consumption_mutations_count") or 0)
    except Exception as e:
        logger.debug(f"[P0-3/ZERO-LOG-PROBE] get_inventory_activity_since falló para {user_id}: {e}")
        return None

    if log_count == 0 and mutations == 0:
        return {
            "log_count": 0,
            "consumption_mutations_count": 0,
            "probed_since": since_iso,
            "lookback_days": int(lookback_days),
        }
    return None


def _enqueue_plan_chunk(
    user_id: str,
    meal_plan_id: str,
    week_number: int,
    days_offset: int,
    days_count: int,
    pipeline_snapshot: dict,
    chunk_kind: str = None,
):
    """Inserta un job en plan_chunk_queue para generar un chunk en background.

    [P0-1] La idempotencia ahora se resuelve en una única sentencia SQL atómica
    (INSERT … ON CONFLICT DO UPDATE WHERE status='failed' RETURNING xmax). Antes,
    el patrón era SELECT-luego-UPDATE/INSERT con ~150 líneas de cómputo en medio,
    lo que abría una ventana TOCTOU: dos workers (catchup sweep + retry manual,
    p. ej.) podían leer el mismo estado, decidir UPDATE, y solo uno aplicaba —
    sin que el segundo detectara la pérdida. En planes 15d/30d esto causaba
    huecos permanentes (`[chunk1✓, chunk2✓, chunk3✓, chunk4=failed, chunk5=…]`).
    Ahora la transacción del UPSERT en Postgres es atómica vía el unique index
    parcial `ux_plan_chunk_queue_live_week`, y el RETURNING distingue insert vs
    update vs skip-active para telemetría.
    """
    import json
    from datetime import timedelta
    # [P0-4] Estampar cuándo fue capturado el snapshot del inventario.
    pipeline_snapshot = copy.deepcopy(pipeline_snapshot) if pipeline_snapshot else {}
    if isinstance(pipeline_snapshot.get("form_data"), dict):
        pipeline_snapshot["form_data"]["_pantry_captured_at"] = datetime.now(timezone.utc).isoformat()
        # [P1-5] Snapshot del modo de validación de cantidades + tolerance al
        # encolar. Antes el worker leía `CHUNK_PANTRY_QUANTITY_MODE` y
        # `CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE` directamente del módulo en
        # cada ejecución, así que un cambio mid-plan (admin sube tolerance,
        # cambio de mode hybrid→strict) afectaba chunks ya encolados con una
        # expectativa distinta — rompiendo planes 15d/30d que asumían mode
        # consistente entre chunks. Ahora capturamos el valor global vigente
        # al enqueue y el worker lo prefiere sobre la constante. La lectura
        # de `chunk_health_profile._pantry_quantity_mode` (per-user opt-in)
        # mantiene precedencia sobre el snapshot cuando exista — un usuario
        # que pidió strict explícitamente debe seguir en strict aunque el
        # snapshot contenga hybrid.
        if "_pantry_quantity_mode" not in pipeline_snapshot["form_data"]:
            pipeline_snapshot["form_data"]["_pantry_quantity_mode"] = (
                CHUNK_PANTRY_QUANTITY_MODE or "advisory"
            )
        if "_pantry_quantity_hybrid_tolerance" not in pipeline_snapshot["form_data"]:
            pipeline_snapshot["form_data"]["_pantry_quantity_hybrid_tolerance"] = float(
                CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE
            )

    normalized_chunk_kind = chunk_kind or ("rolling_refill" if pipeline_snapshot.get("_is_rolling_refill") else "initial_plan")

    # Delay para INSERT (chunk fresh) vs UPDATE (retry de un chunk failed).
    # Calculamos AMBOS porque hasta que la UPSERT no devuelva `xmax` no sabemos
    # cuál se aplicó: hacerlo después abriría la TOCTOU que estamos cerrando.
    fresh_delay_days, chunk_mode, days_offset_int, days_count_int = _compute_chunk_delay_days(
        days_offset,
        days_count,
        week_number,
        pipeline_snapshot or {},
        normalized_chunk_kind,
    )
    retry_delay_days, _, _, _ = _compute_chunk_delay_days(
        days_offset_int,
        days_count_int,
        week_number,
        pipeline_snapshot or {},
        for_failed_retry=True,
    )
    fresh_preemption_seconds = _compute_expected_preemption_seconds(days_offset_int, fresh_delay_days)
    retry_preemption_seconds = _compute_expected_preemption_seconds(days_offset_int, retry_delay_days)

    # [P0-2] Cadena de fallback completa: snapshot → user_profile+today → último plan
    # → 8am UTC. El bloque previo solo manejaba snapshot y caía a NOW()+delay (hora
    # arbitraria del día) cuando faltaba `_plan_start_date`. Ahora el resolver
    # devuelve siempre un anchor accionable o, en peor caso, fuerza 8am UTC.
    anchor_start_dt, anchor_tz_min, anchor_source = _resolve_chunk_start_anchor(
        user_id=user_id,
        snapshot=pipeline_snapshot,
        meal_plan_id=meal_plan_id,
        week_number=week_number,
    )

    # Propagar TZ resuelta al snapshot para que el resto del pipeline (worker,
    # validators) lea el mismo offset que se usó para programar.
    if isinstance(pipeline_snapshot.get("form_data"), dict):
        _snapshot_tz_before = pipeline_snapshot["form_data"].get("tz_offset_minutes")
        if _snapshot_tz_before is None:
            _snapshot_tz_before = pipeline_snapshot["form_data"].get("tzOffset")
        if _snapshot_tz_before != anchor_tz_min:
            pipeline_snapshot["form_data"]["tzOffset"] = anchor_tz_min
            pipeline_snapshot["form_data"]["tz_offset_minutes"] = anchor_tz_min
            logger.info(
                f"[P0-2/ANCHOR-TZ] chunk {week_number} de {user_id}: "
                f"snapshot_tz={_snapshot_tz_before} → anchor_tz={anchor_tz_min}m "
                f"(source={anchor_source}). Propagado al pipeline_snapshot."
            )
        pipeline_snapshot["form_data"]["_chunk_anchor_source"] = anchor_source

    fresh_execute_dt: datetime | None = None
    retry_execute_dt: datetime | None = None

    if anchor_start_dt is not None:
        start_dt_midnight_utc = datetime.combine(
            anchor_start_dt.date(), datetime.min.time()
        ).replace(tzinfo=timezone.utc)
        fresh_target = start_dt_midnight_utc + timedelta(
            days=fresh_delay_days, minutes=anchor_tz_min + 30
        )
        retry_target = anchor_start_dt + timedelta(days=retry_delay_days, hours=-3)

        execute_dt_min = datetime.now(timezone.utc) + timedelta(minutes=1)
        fresh_execute_dt = max(fresh_target, execute_dt_min)
        retry_execute_dt = max(retry_target, execute_dt_min)
    else:
        # [P0-2] Source 4: 8am UTC del día delay_days. Mejora estricta sobre
        # NOW()+delay (que disparaba a las 3am cuando el cron levantaba el chunk).
        _now = datetime.now(timezone.utc)
        fresh_target_date = (_now + timedelta(days=fresh_delay_days)).date()
        retry_target_date = (_now + timedelta(days=retry_delay_days)).date()
        fresh_execute_dt = datetime.combine(
            fresh_target_date, datetime.min.time()
        ).replace(tzinfo=timezone.utc) + timedelta(hours=8)
        retry_execute_dt = datetime.combine(
            retry_target_date, datetime.min.time()
        ).replace(tzinfo=timezone.utc) + timedelta(hours=8)
        execute_dt_min = _now + timedelta(minutes=1)
        fresh_execute_dt = max(fresh_execute_dt, execute_dt_min)
        retry_execute_dt = max(retry_execute_dt, execute_dt_min)

    # [P0-1] UPSERT atómico. Tres outcomes posibles:
    #   1. INSERT — no había fila → la creamos. RETURNING (xmax = 0) AS inserted = TRUE.
    #   2. UPDATE — había fila con status='failed' → la reactivamos. inserted = FALSE.
    #   3. NADA — había fila con status IN ('pending','processing','stale') → ON CONFLICT
    #      dispara DO UPDATE pero la cláusula WHERE status='failed' filtra y la sentencia
    #      no devuelve fila. Esto es la rama "skip" (chunk ya activo).
    # El UPSERT es atómico vía el unique index parcial ux_plan_chunk_queue_live_week,
    # cerrando la ventana TOCTOU del patrón check-then-write previo.
    snapshot_json = json.dumps(pipeline_snapshot, ensure_ascii=False)
    upsert_sql = """
        INSERT INTO plan_chunk_queue
            (user_id, meal_plan_id, week_number, chunk_kind,
             days_offset, days_count, pipeline_snapshot,
             execute_after, expected_preemption_seconds, status, attempts)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::timestamptz, %s, 'pending', 0)
        ON CONFLICT (meal_plan_id, week_number)
        WHERE status IN ('pending', 'processing', 'stale', 'failed')
        DO UPDATE SET
            status = 'pending',
            attempts = 0,
            chunk_kind = EXCLUDED.chunk_kind,
            pipeline_snapshot = EXCLUDED.pipeline_snapshot,
            execute_after = %s::timestamptz,
            expected_preemption_seconds = %s,
            updated_at = NOW(),
            days_offset = EXCLUDED.days_offset,
            days_count = EXCLUDED.days_count
        WHERE plan_chunk_queue.status = 'failed'
        RETURNING id, status, (xmax = 0) AS inserted
    """
    upsert_params = (
        user_id,
        str(meal_plan_id),
        week_number,
        normalized_chunk_kind,
        days_offset_int,
        days_count_int,
        snapshot_json,
        fresh_execute_dt.isoformat(),
        fresh_preemption_seconds,
        retry_execute_dt.isoformat(),
        retry_preemption_seconds,
    )

    try:
        result = execute_sql_query(upsert_sql, upsert_params, fetch_one=True)
    except Exception as e:
        logger.error(
            f"❌ [P0-1] UPSERT atómico falló para plan {meal_plan_id} week {week_number}: {e}"
        )
        raise

    if not result:
        # Skip-active: ya existe una fila viva (pending/processing/stale).
        logger.info(
            f"[P0-1/IDEMPOTENT] Chunk {week_number} para plan {meal_plan_id} ya activo "
            f"(skip enqueue, race resuelta atómicamente por el DB)."
        )
        return

    if result.get("inserted"):
        logger.info(
            f" [CHUNK] Chunk {week_number} encolado para plan {meal_plan_id} "
            f"(días {days_offset_int+1}–{days_offset_int+days_count_int}, "
            f"kind={normalized_chunk_kind}, mode={chunk_mode}) "
            f"ejecutará a las {fresh_execute_dt.isoformat()}"
        )
    else:
        logger.warning(
            f"⚠️ [P0-1] Reactivado chunk failed (id={result.get('id')}) para plan {meal_plan_id} "
            f"week {week_number} en modo {chunk_mode} con retry_delay={retry_delay_days}d "
            f"(margen {days_offset_int - retry_delay_days}d, "
            f"ejecutará a las {retry_execute_dt.isoformat()})"
        )

    # [P0-2] Si la cadena de fallback agotó todas las fuentes y caímos a 8am UTC,
    # el chunk podría dispararse a las 3am hora local en TZs negativas (Bogotá UTC-5,
    # PST UTC-8). Antes de que el worker lo levante, lo flipeamos a
    # `pending_user_action` con reason='tz_unresolved'. El recovery cron retentará
    # `_resolve_chunk_start_anchor` cada tick (~15 min) y reanudará en cuanto el
    # usuario abra la app y se persista `tz_offset_minutes` en su perfil.
    #
    # El flag `CHUNK_REJECT_FORCED_UTC_ENQUEUE` permite a operadores deshabilitar
    # este gate (e.g., para cargar bulk planes en pruebas) y volver al
    # comportamiento legacy de aceptar el 8am UTC.
    if anchor_source == "forced_8am_utc" and CHUNK_REJECT_FORCED_UTC_ENQUEUE:
        chunk_id = result.get("id")
        try:
            tz_pause_snapshot = copy.deepcopy(pipeline_snapshot)
            tz_pause_snapshot["_pantry_pause_reason"] = "tz_unresolved"
            tz_pause_snapshot["_pantry_pause_started_at"] = datetime.now(timezone.utc).isoformat()
            tz_pause_snapshot["_pantry_pause_reminders"] = 0
            tz_pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_PANTRY_EMPTY_TTL_HOURS
            tz_pause_snapshot["_pantry_pause_reminder_hours"] = CHUNK_PANTRY_EMPTY_REMINDER_HOURS
            tz_pause_snapshot["_tz_recovery_attempts"] = 0
            execute_sql_write(
                """
                UPDATE plan_chunk_queue
                SET status = 'pending_user_action',
                    pipeline_snapshot = %s::jsonb,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (json.dumps(tz_pause_snapshot, ensure_ascii=False), chunk_id),
            )
            logger.warning(
                f"[P0-2/TZ-UNRESOLVED] Chunk {week_number} para plan {meal_plan_id} "
                f"flipped a pending_user_action: anchor_source=forced_8am_utc evita "
                f"disparo a hora local incorrecta. Recovery cron retentará "
                f"_resolve_chunk_start_anchor cada tick."
            )
            try:
                _maybe_notify_user_tz_unresolved(user_id)
            except Exception as _np_err:
                logger.warning(
                    f"[P0-2/TZ-UNRESOLVED] No se pudo notificar al usuario {user_id}: {_np_err}"
                )
        except Exception as _flip_err:
            # Si el flip falla, el chunk queda en 'pending' con execute_after a 8am UTC
            # — comportamiento legacy. Mejor que quedarse atrapado sin estado consistente.
            logger.error(
                f"[P0-2/TZ-UNRESOLVED] Falló flip a pending_user_action para chunk "
                f"{chunk_id} (plan {meal_plan_id}): {_flip_err}. Cae a comportamiento legacy."
            )
        # El P0-3/P0-4 no aplican si ya estamos pausando por TZ — ya el chunk
        # está en pending_user_action, no necesita más guards.
        return

    # [P0-4] Guard proactivo de nevera. Antes el worker descubría reactivamente
    # que el inventario vivo estaba por debajo de `CHUNK_MIN_FRESH_PANTRY_ITEMS`
    # al levantar el chunk para generar (línea ~13193); el plan quedaba en
    # estado "fantasma" en la UI hasta entonces. Aquí probamos `get_user_inventory_net`
    # al ENQUEUE y, si los items vivos no alcanzan el mínimo, flipeamos directo a
    # `pending_user_action` con reason='empty_pantry_proactive'. El recovery cron
    # existente reanuda en cuanto detecta items suficientes.
    #
    # Conservadores: si la lectura live falla (DB blip / Supabase down), NO
    # pausamos — preferimos dejar el chunk en pending y que el worker, con su
    # backoff y snapshot fallback, decida más tarde. Pausar agresivamente con
    # info parcial sería peor UX que aceptar un retraso.
    #
    # Solo aplicamos al primer INSERT de un chunk no-inicial.
    chunk_already_handled = False
    if (
        CHUNK_PANTRY_PROACTIVE_GUARD
        and result.get("inserted")
        and week_number > 1
        and normalized_chunk_kind != "initial_plan"
    ):
        chunk_id_p04 = result.get("id")
        try:
            from db_inventory import get_user_inventory_net as _p04_get_inv
            live_inv_p04 = _p04_get_inv(user_id)
        except Exception as _p04_inv_err:
            live_inv_p04 = None
            logger.debug(
                f"[P0-4/PANTRY-PROBE] get_user_inventory_net falló para "
                f"chunk {chunk_id_p04} user {user_id}: {_p04_inv_err}. "
                f"Skipeando guard proactivo (worker decidirá con su fallback)."
            )
        if live_inv_p04 is not None:
            meaningful_p04 = _count_meaningful_pantry_items(live_inv_p04)
            if meaningful_p04 < CHUNK_MIN_FRESH_PANTRY_ITEMS:
                try:
                    _pause_chunk_for_pantry_refresh(
                        chunk_id_p04,
                        user_id,
                        week_number,
                        live_inv_p04,
                        reason="empty_pantry_proactive",
                    )
                    logger.warning(
                        f"[P0-4/PANTRY-PROACTIVE] Chunk {week_number} plan {meal_plan_id} "
                        f"flipped a pending_user_action al enqueue: items_meaningful="
                        f"{meaningful_p04} < min={CHUNK_MIN_FRESH_PANTRY_ITEMS} (raw={len(live_inv_p04 or [])}). "
                        f"El usuario verá pausa inmediata en lugar de plan fantasma."
                    )
                    chunk_already_handled = True
                except Exception as _p04_pause_err:
                    # Si la pausa falla, el chunk queda en 'pending' — el worker
                    # lo manejará reactivamente como antes. Mejor degradación
                    # silenciosa que romper enqueue.
                    logger.error(
                        f"[P0-4/PANTRY-PROACTIVE] Falló pausa para chunk "
                        f"{chunk_id_p04} (plan {meal_plan_id}): {_p04_pause_err}. "
                        f"Cae a comportamiento legacy."
                    )
    if chunk_already_handled:
        return

    # [P0-3] Zero-log proactivo. Probamos al ENQUEUE si el chunk previo tuvo señal real.
    # Sin esto, chunk N+1 entraría en el ciclo de deferrals del worker (2-4h de retraso
    # silencioso) antes de pausarse. Aquí marcamos directo `_learning_ready_deferrals=MAX`
    # para que el worker, al picking, salte la cola de deferrals y pause inmediatamente
    # con `learning_zero_logs` (push al usuario explicando que necesitamos sus logs para
    # generar variedad real en el siguiente bloque).
    #
    # Solo aplicamos al primer INSERT de un chunk no-inicial para evitar:
    #   - Probar para chunks `initial_plan` encolados upfront (no hay ventana previa).
    #   - Probar en reactivaciones de chunks `failed` (ya tuvieron una pasada).
    if (
        CHUNK_ZERO_LOG_PROACTIVE_DETECTION
        and result.get("inserted")
        and week_number > 1
        and normalized_chunk_kind != "initial_plan"
        and days_count_int > 0
    ):
        chunk_id_zl = result.get("id")
        try:
            zl_signal = _detect_proactive_zero_log_at_boundary(
                user_id=user_id,
                meal_plan_id=str(meal_plan_id),
                lookback_days=days_count_int,
            )
        except Exception as _zl_probe_err:
            zl_signal = None
            logger.debug(
                f"[P0-3/ZERO-LOG-PROBE] Probe falló para chunk {chunk_id_zl}: {_zl_probe_err}"
            )
        if zl_signal:
            try:
                from constants import CHUNK_LEARNING_READY_MAX_DEFERRALS as _ZL_MAX_DEF
                zl_pre_snapshot = copy.deepcopy(pipeline_snapshot)
                zl_pre_snapshot["_zero_log_proactive_detected"] = True
                zl_pre_snapshot["_zero_log_proactive_signal"] = zl_signal
                # Pre-saturar el contador de deferrals para que el worker, al picking,
                # caiga directo en la rama "max alcanzado → pausar" en lugar de diferir.
                zl_pre_snapshot["_learning_ready_deferrals"] = int(_ZL_MAX_DEF)
                if isinstance(zl_pre_snapshot.get("form_data"), dict):
                    zl_pre_snapshot["form_data"]["_force_variety"] = True
                    zl_pre_snapshot["form_data"]["_zero_log_proactive_detected"] = True
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET pipeline_snapshot = %s::jsonb, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (json.dumps(zl_pre_snapshot, ensure_ascii=False), chunk_id_zl),
                )
                logger.warning(
                    f"[P0-3/ZERO-LOG-PROACTIVE] Chunk {week_number} plan {meal_plan_id} "
                    f"marcado proactivamente como zero-log: lookback={days_count_int}d, "
                    f"logs=0, mutations=0. Worker saltará deferrals y pausará inmediatamente."
                )
            except Exception as _zl_persist_err:
                logger.warning(
                    f"[P0-3/ZERO-LOG-PROACTIVE] No se pudo persistir flag para chunk "
                    f"{chunk_id_zl}: {_zl_persist_err}. Cae al deferral cycle del worker."
                )


def _process_pending_shopping_lists():
    """[GAP F FIX] Recalcula shopping lists asincronamente para planes que fallaron su generacion sincrona."""
    try:
        from shopping_calculator import get_shopping_list_delta
        import json
        
        # Buscar planes con status 'partial_no_shopping'
        plans = execute_sql_query(
            """
            SELECT id, user_id, plan_data
            FROM meal_plans
            WHERE plan_data->>'generation_status' = 'partial_no_shopping'
            """,
            fetch_all=True,
        )
        
        if not plans:
            return
            
        logger.info(f" [GAP F] Procesando shopping lists pendientes para {len(plans)} planes...")
        
        for p in plans:
            meal_plan_id = p.get('id', 'unknown')
            try:
                # [P0-5] Skip rows missing user_id instead of KeyError-ing the whole loop.
                # Originally `p['user_id']` raised KeyError on rows with NULL user_id (or on
                # test fixtures that don't include the column), bubbling up to the outer
                # except and logging "Error recuperando shopping list ... 'user_id'" without
                # signaling WHICH plans were skipped. Now we log the skip explicitly and
                # continue with the next plan, so one bad row doesn't blackhole the batch.
                user_id = p.get('user_id')
                if not user_id:
                    logger.warning(
                        f" [GAP F/SKIP] Plan {meal_plan_id} sin user_id en row; saltando."
                    )
                    continue
                plan_data = p.get('plan_data') or {}
                
                # Fetch form_data for household and groceryDuration
                snap = execute_sql_query("SELECT pipeline_snapshot FROM plan_chunk_queue WHERE meal_plan_id = %s ORDER BY created_at DESC LIMIT 1", (meal_plan_id,), fetch_one=True)
                if snap and snap.get('pipeline_snapshot'):
                    snapshot = snap['pipeline_snapshot']
                    if isinstance(snapshot, str): snapshot = json.loads(snapshot)
                    form_data = snapshot.get("form_data", {})
                else:
                    form_data = {}
                    
                household = form_data.get("householdSize", 1)
                
                aggr_7 = get_shopping_list_delta(user_id, plan_data, is_new_plan=False, structured=True, multiplier=1.0 * household)
                aggr_15 = get_shopping_list_delta(user_id, plan_data, is_new_plan=False, structured=True, multiplier=2.0 * household)
                aggr_30 = get_shopping_list_delta(user_id, plan_data, is_new_plan=False, structured=True, multiplier=4.0 * household)
                
                grocery_duration = form_data.get("groceryDuration", "weekly")
                if grocery_duration == "biweekly":
                    aggr_active = aggr_15
                elif grocery_duration == "monthly":
                    aggr_active = aggr_30
                else:
                    aggr_active = aggr_7
                    
                total_generated = plan_data.get('total_days_generated', 0)
                total_requested = plan_data.get('total_days_requested', 7)
                new_status = "complete" if total_generated >= int(total_requested) else "partial"
                
                execute_sql_write("""
                    UPDATE meal_plans 
                    SET plan_data = jsonb_set(
                        jsonb_set(
                            jsonb_set(
                                jsonb_set(
                                    jsonb_set(plan_data, '{aggregated_shopping_list_weekly}', %s::jsonb),
                                    '{aggregated_shopping_list_biweekly}', %s::jsonb
                                ),
                                '{aggregated_shopping_list_monthly}', %s::jsonb
                            ),
                            '{aggregated_shopping_list}', %s::jsonb
                        ),
                        '{generation_status}', %s::jsonb
                    )
                    WHERE id = %s
                """, (
                    json.dumps(aggr_7, ensure_ascii=False),
                    json.dumps(aggr_15, ensure_ascii=False),
                    json.dumps(aggr_30, ensure_ascii=False),
                    json.dumps(aggr_active, ensure_ascii=False),
                    json.dumps(new_status),
                    meal_plan_id
                ))
                logger.info(f" [GAP F] Shopping list recuperada para plan {meal_plan_id}.")
            except Exception as e:
                logger.error(f" [GAP F] Error recuperando shopping list para plan {meal_plan_id}: {e}")
    except Exception as e:
        logger.error(f" [GAP F] Error general procesando pending shopping lists: {e}")

def _record_chunk_metric(
    chunk_id: str,
    meal_plan_id: str,
    user_id: str,
    week_number: int,
    days_count: int,
    duration_ms: int,
    quality_tier: str,
    was_degraded: bool,
    retries: int,
    lag_seconds: int,
    learning_metrics: dict = None,
    error_message: str = None,
    is_rolling_refill: bool = False,
    pantry_snapshot_age_hours: float = None,
):
    """[GAP G] Inserta una fila en plan_chunk_metrics para análisis histórico.

    [P0-3] `pantry_snapshot_age_hours` permite ver en producción la distribución
    real de edades de snapshot al momento del pickup. Sin este dato era imposible
    saber si los chunks de planes 30d estaban ejecutando con snapshots de 20+ días
    (escenario donde el LLM genera platos con ingredientes que ya no existen).
    """
    try:
        execute_sql_write("ALTER TABLE plan_chunk_metrics ADD COLUMN IF NOT EXISTS is_rolling_refill BOOLEAN DEFAULT false")
    except Exception:
        pass

    try:
        repeat_pct = None
        rej_viol = 0
        alg_viol = 0
        if learning_metrics:
            repeat_pct = learning_metrics.get("learning_repeat_pct")
            rej_viol = int(learning_metrics.get("rejection_violations") or 0)
            alg_viol = int(learning_metrics.get("allergy_violations") or 0)
            # Si el caller no nos pasó age explícito, intentar leerlo de learning_metrics
            # (pendiente de inyección upstream). Aceptar None silenciosamente.
            if pantry_snapshot_age_hours is None:
                _lm_age = learning_metrics.get("pantry_snapshot_age_hours_at_pickup")
                if _lm_age is not None:
                    try:
                        pantry_snapshot_age_hours = float(_lm_age)
                    except (TypeError, ValueError):
                        pantry_snapshot_age_hours = None

        execute_sql_write(
            """
            INSERT INTO plan_chunk_metrics
                (chunk_id, meal_plan_id, user_id, week_number, days_count,
                 duration_ms, quality_tier, was_degraded, retries, lag_seconds,
                 learning_repeat_pct, rejection_violations, allergy_violations, error_message, is_rolling_refill,
                 pantry_snapshot_age_hours)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                chunk_id, meal_plan_id, user_id, week_number, days_count,
                duration_ms, quality_tier, was_degraded, retries, lag_seconds,
                repeat_pct, rej_viol, alg_viol, error_message, is_rolling_refill,
                pantry_snapshot_age_hours,
            ),
        )
    except Exception as e:
        # No bloquear al worker por fallas de observabilidad
        logger.warning(f"[GAP G] Error insertando métrica de chunk: {e}")


# [P1-3] Contador en memoria de fallos de telemetría de deferrals. Se incrementa
# cuando la INSERT a chunk_deferrals lanza excepción y se loggea cada N fallos para
# detectar problemas sistémicos (tabla missing, permisos, etc.) sin spamear logs.
_chunk_deferral_telemetry_failures: dict = {"count": 0, "last_error": None}

# [P1-6] Lock thread-safe para acceso al buffer local de deferrals.
# El append (escritura) y el flush (lectura+reescritura) se serializan vía este lock
# para evitar corrupción del archivo bajo concurrencia entre worker threads y el cron.
import threading as _p16_threading
_p16_buffer_lock = _p16_threading.Lock()

import uuid as _p16_uuid


def _is_valid_uuid(value) -> bool:
    """True si `value` es parseable como UUID (cubre str y uuid.UUID)."""
    if value is None:
        return False
    try:
        _p16_uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _append_deferral_to_buffer(record: dict) -> bool:
    """[P1-6] Escribe un deferral fallido al buffer local jsonl para retry posterior.

    Cuando `_record_chunk_deferral` no puede insertar en `chunk_deferrals` por DB
    caída/blip, el record queda en este archivo (una línea JSON por record). El
    cron `_flush_pending_deferrals` lo lee periódicamente, intenta persistir cada
    record, y borra solo los que tuvieron éxito.

    Cap a `CHUNK_DEFERRALS_BUFFER_MAX_RECORDS`: si el archivo ya tiene >= N líneas,
    descartamos las MÁS VIEJAS antes de añadir la nueva (FIFO). Mejor perder
    telemetría antigua que crecer sin límite durante outages prolongados.

    Returns:
        True si la persistencia local fue OK; False si el filesystem falló.
    """
    try:
        from constants import (
            CHUNK_DEFERRALS_BUFFER_PATH,
            CHUNK_DEFERRALS_BUFFER_MAX_RECORDS,
        )
        path = CHUNK_DEFERRALS_BUFFER_PATH
        with _p16_buffer_lock:
            # FIFO cap: si excedemos, descartamos las líneas más viejas.
            existing = []
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        existing = [ln for ln in f.readlines() if ln.strip()]
            except Exception:
                existing = []  # archivo corrupto: arrancar de cero

            existing.append(json.dumps(record, ensure_ascii=False))
            if len(existing) > CHUNK_DEFERRALS_BUFFER_MAX_RECORDS:
                _dropped = len(existing) - CHUNK_DEFERRALS_BUFFER_MAX_RECORDS
                existing = existing[-CHUNK_DEFERRALS_BUFFER_MAX_RECORDS:]
                logger.warning(
                    f"[P1-6/BUFFER] Cap excedido ({CHUNK_DEFERRALS_BUFFER_MAX_RECORDS}); "
                    f"descartadas {_dropped} líneas más viejas (FIFO)."
                )

            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(existing) + "\n")
        return True
    except Exception as e:
        logger.error(f"[P1-6/BUFFER] No se pudo persistir buffer local: {e}")
        return False


def _flush_pending_deferrals() -> dict:
    """[P1-6] Cron: re-intenta INSERT de deferrals buffered cuando DB se recupera.

    Lee `CHUNK_DEFERRALS_BUFFER_PATH`, intenta persistir cada record en
    `chunk_deferrals` preservando el `created_at` original (timestamp del
    momento del deferral, no del flush). Records que fallan con violación de
    NOT NULL (e.g. meal_plan_id None) se descartan permanentemente — un retry
    no los recuperará. Records con otros errores quedan en el archivo para el
    siguiente intento.

    Returns:
        dict con contadores: {flushed, remaining, discarded_invalid}
    """
    from constants import CHUNK_DEFERRALS_BUFFER_PATH
    path = CHUNK_DEFERRALS_BUFFER_PATH
    stats = {"flushed": 0, "remaining": 0, "discarded_invalid": 0}

    if not os.path.exists(path):
        return stats

    with _p16_buffer_lock:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [ln for ln in f.readlines() if ln.strip()]
        except Exception as e:
            logger.error(f"[P1-6/FLUSH] Error leyendo buffer: {e}")
            return stats

        remaining_records: list = []
        for line in lines:
            try:
                rec = json.loads(line)
            except Exception:
                stats["discarded_invalid"] += 1
                continue

            # Validación previa de schema: descarte silencioso para casos que
            # sabemos que el INSERT rechazará siempre (no se recuperan en retry):
            #   - meal_plan_id es NOT NULL en la tabla.
            #   - user_id y meal_plan_id deben ser UUIDs parseables.
            # Sin este pre-check, cada record basura generaba un warning por tick
            # del cron (spam masivo cuando el buffer acumulaba records de tests).
            if (
                rec.get("meal_plan_id") is None
                or not _is_valid_uuid(rec.get("user_id"))
                or not _is_valid_uuid(rec.get("meal_plan_id"))
            ):
                stats["discarded_invalid"] += 1
                continue

            try:
                execute_sql_write(
                    "INSERT INTO chunk_deferrals "
                    "(user_id, meal_plan_id, week_number, reason, days_until_prev_end, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, COALESCE(%s::timestamptz, NOW()))",
                    (
                        rec.get("user_id"),
                        rec.get("meal_plan_id"),
                        int(rec.get("week_number") or 0),
                        str(rec.get("reason") or ""),
                        (
                            int(rec["days_until_prev_end"])
                            if rec.get("days_until_prev_end") is not None
                            else None
                        ),
                        rec.get("buffered_at"),
                    ),
                )
                stats["flushed"] += 1
            except Exception as _flush_err:
                # NOT NULL u otra violación de schema → descartar (no se recupera).
                # Errores de conexión/timeout → mantener para retry.
                _err_msg = str(_flush_err).lower()
                if "violates not-null" in _err_msg or "invalid input syntax" in _err_msg:
                    stats["discarded_invalid"] += 1
                else:
                    remaining_records.append(line.rstrip())

        if remaining_records:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(remaining_records) + "\n")
            except Exception as e:
                logger.error(f"[P1-6/FLUSH] Error reescribiendo buffer: {e}")
        else:
            try:
                os.remove(path)
            except Exception:
                pass

        stats["remaining"] = len(remaining_records)

    if stats["flushed"] > 0 or stats["remaining"] > 0 or stats["discarded_invalid"] > 0:
        logger.info(
            f"[P1-6/FLUSH] Buffer deferrals: flushed={stats['flushed']}, "
            f"remaining={stats['remaining']}, discarded={stats['discarded_invalid']}"
        )
    return stats


# [P0-A] Contador en memoria de fallos de telemetría de lecciones sintetizadas. Mismo
# patrón que `_chunk_deferral_telemetry_failures`: detecta tabla missing/permisos sin
# spamear logs. Se resetea al primer éxito tras una racha de fallos.
_chunk_lesson_telemetry_failures: dict = {"count": 0, "last_error": None}

# [P1-B] Contador de fallos al arrancar el thread daemon de heartbeat. Picos sostenidos
# indican presión de threads del proceso (límite del SO, fugas de threads no-daemon).
# Lo expone process_chunk_task vía métricas para que SRE pueda alertar; reset implícito
# en restart del proceso (no se persiste — es señal de salud actual, no histórica).
_chunk_heartbeat_start_failures: dict = {
    "count": 0,
    "last_failure_at": None,
    "last_chunk_id": None,
}


def _handle_heartbeat_start_failure(task_id, user_id) -> None:
    """[P1-B] Aborta de forma segura un chunk cuyo heartbeat thread no arrancó.

    Antes el código solo loguaba ERROR y continuaba al LLM call: si el thread daemon
    no arrancó (límite de threads del SO, OOM transient), el chunk quedaba sin
    `heartbeat_at` refrescándose y el zombie rescue lo mataba tras CHUNK_LOCK_STALE_MINUTES
    en pleno LLM call → tokens perdidos + reintento desde cero. Ahora abortamos antes,
    diferimos a NOW + CHUNK_HEARTBEAT_START_FAIL_RETRY_MINUTES, y liberamos lock +
    reservas para que un tick posterior (con menos presión de threads) lo recoja.

    Acciones (todas best-effort, ningún error aquí debe propagar al worker):
      1. Incrementa `_chunk_heartbeat_start_failures` (métrica in-memory).
      2. Loguea ERROR con count acumulado.
      3. Libera reservas de inventario (atómico vía P1-A).
      4. UPDATE plan_chunk_queue: status='pending', attempts++, execute_after=NOW()+RETRY.
      5. DELETE del chunk_user_locks (otro worker puede recogerlo tras el delay).
    """
    global _chunk_heartbeat_start_failures
    _chunk_heartbeat_start_failures["count"] += 1
    _chunk_heartbeat_start_failures["last_failure_at"] = datetime.now(timezone.utc)
    _chunk_heartbeat_start_failures["last_chunk_id"] = str(task_id)

    logger.error(
        f"[P1-B/HEARTBEAT-START-FAIL] Thread NO arrancó para chunk {task_id} "
        f"(total_failures_in_process={_chunk_heartbeat_start_failures['count']}). "
        f"Abortando antes del LLM call y difiriendo "
        f"{CHUNK_HEARTBEAT_START_FAIL_RETRY_MINUTES}min."
    )

    # 3. Liberar reservas de inventario (atómico vía P1-A) — el chunk no va a usarlas
    # en este intento.
    try:
        release_chunk_reservations(user_id, str(task_id))
    except Exception as _rel_err:
        logger.warning(
            f"[P1-B/HEARTBEAT-START-FAIL] release_chunk_reservations falló "
            f"para chunk {task_id}: {_rel_err}. Continuando con el defer."
        )

    # 4. Devolver el chunk a pending con backoff. Bumpear attempts para que
    # CHUNK_MAX_FAILURE_ATTEMPTS proteja contra loops infinitos si el problema de
    # threads es persistente (en cuyo caso el chunk eventualmente cae a 'failed' y
    # _recover_failed_chunks_for_long_plans lo gestiona).
    try:
        execute_sql_write(
            """
            UPDATE plan_chunk_queue
            SET status = 'pending',
                attempts = COALESCE(attempts, 0) + 1,
                execute_after = NOW() + make_interval(mins => %s),
                updated_at = NOW()
            WHERE id = %s
            """,
            (CHUNK_HEARTBEAT_START_FAIL_RETRY_MINUTES, task_id),
        )
    except Exception as _defer_err:
        logger.error(
            f"[P1-B/HEARTBEAT-START-FAIL] No se pudo diferir el chunk {task_id}: "
            f"{_defer_err}. Cae al defaults del worker."
        )

    # 5. Liberar el lock que ya adquirimos para que otro worker pueda picar el chunk
    # cuando expire el delay.
    try:
        execute_sql_write(
            "DELETE FROM chunk_user_locks WHERE locked_by_chunk_id = %s",
            (task_id,),
        )
    except Exception as _lock_err:
        logger.warning(
            f"[P1-B/HEARTBEAT-START-FAIL] No se pudo liberar lock para "
            f"chunk {task_id}: {_lock_err}. Housekeeping lo recogerá tras stale."
        )


def _record_chunk_lesson_telemetry(
    user_id: str,
    meal_plan_id: str,
    week_number: int,
    event: str,
    synthesized_count: int = 0,
    queue_count: int = 0,
    metadata: dict | None = None,
) -> bool:
    """[P0-A] Persiste un evento de resolución de lecciones por chunk.

    `event` es categórico:
      - 'lesson_synthesized_low_confidence': el sistema cayó a
        `_synthesize_last_chunk_learning_from_plan_days` porque
        `plan_chunk_queue.learning_metrics` del chunk previo está NULL. La lección
        que recibe el LLM tiene counters en cero y `low_confidence=True`.
      - 'recent_lessons_partial_synthesis': la ventana rolling
        `_recent_chunk_lessons` se regeneró desde plan_data.days; al menos una de
        las entradas combinadas es sintetizada (low-confidence).

    `synthesized_count` y `queue_count` permiten cuantificar la mezcla por evento
    sin tener que inspeccionar metadata. Útil para el cron de alerta que solo
    cuenta filas y agrega ratios.

    No bloquea al worker: si la INSERT falla devolvemos False y el caller continúa.
    La telemetría es best-effort; el aprendizaje en sí ya está mejor servido por
    los defensores P0-3/P0-4/P1-1/P1-2 aguas arriba.
    """
    global _chunk_lesson_telemetry_failures
    try:
        execute_sql_write(
            "INSERT INTO chunk_lesson_telemetry "
            "(user_id, meal_plan_id, week_number, event, "
            " synthesized_count, queue_count, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)",
            (
                str(user_id),
                str(meal_plan_id),
                int(week_number),
                str(event),
                int(synthesized_count),
                int(queue_count),
                json.dumps(metadata or {}, ensure_ascii=False),
            ),
        )
        if _chunk_lesson_telemetry_failures["count"] > 0:
            logger.info(
                f"[P0-A/LESSON-TELEMETRY] Recuperado tras "
                f"{_chunk_lesson_telemetry_failures['count']} fallo(s). "
                f"Último error: {_chunk_lesson_telemetry_failures['last_error']!r}"
            )
            _chunk_lesson_telemetry_failures = {"count": 0, "last_error": None}
        return True
    except Exception as e:
        _chunk_lesson_telemetry_failures["count"] += 1
        _chunk_lesson_telemetry_failures["last_error"] = repr(e)
        n = _chunk_lesson_telemetry_failures["count"]
        if n == 1 or n % 10 == 0:
            logger.error(
                f"[P0-A/LESSON-TELEMETRY] INSERT chunk_lesson_telemetry falló (#{n}) — "
                f"posible tabla missing o permisos. user={user_id} plan={meal_plan_id} "
                f"week={week_number} event={event!r} error={e!r}"
            )
        return False


def _record_chunk_deferral(
    user_id: str,
    meal_plan_id: str | None,
    week_number: int,
    reason: str,
    days_until_prev_end: int | None = None,
) -> bool:
    """[P1-3] Persiste una fila en `chunk_deferrals` con manejo robusto de errores.

    Antes la INSERT estaba inline con `try/except: logger.debug(...) pass` — si la
    tabla no existía o había un permiso roto, los deferrals se perdían en silencio
    y `_detect_chronic_deferrals` no podía detectar usuarios con TZ desalineada.

    Ahora el helper:
      1. Promueve fallos repetidos de `debug` → `error` con contexto estructurado.
      2. Mantiene un contador in-memory para detectar degradación sistémica
         (e.g., 50 fallos seguidos = tabla rota o permisos perdidos).
      3. Resetea el contador al primer éxito (auto-recovery sin restart).

    No bloquea al worker: si la INSERT falla devolvemos False y el caller continúa
    con el deferral del chunk normalmente — la telemetría es best-effort, no path
    crítico.
    """
    global _chunk_deferral_telemetry_failures

    # Hard guard: si user_id o meal_plan_id no son UUID válidos, no intentes el
    # INSERT (siempre fallará por type uuid) y NO escribas al buffer (donde se
    # acumularía como basura permanente). Esto bloquea contaminación desde tests
    # que pasan strings tipo "user-x" / "test_user_race" sin mockear esta función.
    if not _is_valid_uuid(user_id) or (meal_plan_id is not None and not _is_valid_uuid(meal_plan_id)):
        logger.debug(
            f"[P1-3/DEFERRAL-TELEMETRY] Skip por UUID inválido: "
            f"user={user_id!r} plan={meal_plan_id!r} (probablemente test o datos legacy)"
        )
        return False

    try:
        execute_sql_write(
            "INSERT INTO chunk_deferrals "
            "(user_id, meal_plan_id, week_number, reason, days_until_prev_end) "
            "VALUES (%s, %s, %s, %s, %s)",
            (
                user_id,
                meal_plan_id,
                int(week_number),
                str(reason),
                int(days_until_prev_end) if days_until_prev_end is not None else None,
            ),
        )
        # Reset al primer éxito tras una racha de fallos.
        if _chunk_deferral_telemetry_failures["count"] > 0:
            logger.info(
                f"[P1-3/DEFERRAL-TELEMETRY] Recuperado tras "
                f"{_chunk_deferral_telemetry_failures['count']} fallo(s). "
                f"Último error: {_chunk_deferral_telemetry_failures['last_error']!r}"
            )
            _chunk_deferral_telemetry_failures = {"count": 0, "last_error": None}
        return True
    except Exception as e:
        _chunk_deferral_telemetry_failures["count"] += 1
        _chunk_deferral_telemetry_failures["last_error"] = repr(e)
        n = _chunk_deferral_telemetry_failures["count"]
        # Primer fallo y cada 10 después: log error con contexto. Entre medias: warning
        # silencioso con contador para no spamear pero mantener visibilidad.
        if n == 1 or n % 10 == 0:
            logger.error(
                f"[P1-3/DEFERRAL-TELEMETRY] INSERT chunk_deferrals falló (#{n}) — "
                f"posible tabla missing o permisos. user={user_id} plan={meal_plan_id} "
                f"week={week_number} reason={reason!r} error={e!r}"
            )
        else:
            logger.warning(
                f"[P1-3/DEFERRAL-TELEMETRY] INSERT failure #{n}: {e!r}"
            )
        # [P1-6] Persistir el record al buffer local jsonl para retry posterior.
        # Cron `_flush_pending_deferrals` lo intentará re-insertar cuando DB se recupere.
        # Si meal_plan_id es None, NO se buffea (la columna es NOT NULL en el schema):
        # el INSERT siempre fallará permanentemente, así que mejor descartar inline en
        # vez de almacenar basura que el flush descartará después.
        if meal_plan_id is not None:
            _append_deferral_to_buffer({
                "user_id": str(user_id),
                "meal_plan_id": str(meal_plan_id),
                "week_number": int(week_number),
                "reason": str(reason),
                "days_until_prev_end": (
                    int(days_until_prev_end) if days_until_prev_end is not None else None
                ),
                "buffered_at": datetime.now(timezone.utc).isoformat(),
            })
        return False


def _detect_chronic_deferrals() -> None:
    """[P1-2] Detecta usuarios con patrón crónico de deferrals temporales y los notifica.

    `chunk_deferrals` se llena cada vez que el temporal_gate rechaza un chunk porque su
    día previo no concluyó. La causa habitual es desalineación de TZ (viaje silencioso,
    usuario fijó el TZ mal en su perfil). Si un mismo usuario acumula >= 5 deferrals
    en 48h sobre el mismo (meal_plan_id, week_number), enviamos una notificación
    proactiva sugeriendo revisar su zona horaria. Dedupe a 24h vía `system_alerts`
    (alert_key único `chronic_deferrals:{user_id}`).
    """
    from constants import (
        CHUNK_CHRONIC_DEFERRAL_MIN_COUNT,
        CHUNK_CHRONIC_DEFERRAL_WINDOW_HOURS,
        CHUNK_CHRONIC_DEFERRAL_NOTIFY_COOLDOWN_HOURS,
        CHUNK_STALE_PANTRY_DEEPLINK,
    )
    _ensure_quality_alert_schema()

    try:
        rows = execute_sql_query(
            """
            SELECT user_id::text AS user_id,
                   meal_plan_id::text AS meal_plan_id,
                   week_number,
                   COUNT(*)::int AS deferral_count,
                   MAX(created_at) AS last_at
            FROM chunk_deferrals
            WHERE created_at > NOW() - make_interval(hours => %s)
              AND reason = 'temporal_gate'
            GROUP BY user_id, meal_plan_id, week_number
            HAVING COUNT(*) >= %s
            """,
            (int(CHUNK_CHRONIC_DEFERRAL_WINDOW_HOURS), int(CHUNK_CHRONIC_DEFERRAL_MIN_COUNT)),
            fetch_all=True,
        ) or []
    except Exception as e:
        logger.warning(f"[P1-2/CHRONIC] No se pudo consultar chunk_deferrals: {e}")
        return

    if not rows:
        return

    notified = 0
    for row in rows:
        user_id = row.get("user_id")
        if not user_id:
            continue
        alert_key = f"chronic_deferrals:{user_id}"
        # Dedupe vía system_alerts (UNIQUE constraint en alert_key + cooldown explícito).
        try:
            existing = execute_sql_query(
                """
                SELECT triggered_at FROM system_alerts
                WHERE alert_key = %s
                  AND triggered_at > NOW() - make_interval(hours => %s)
                LIMIT 1
                """,
                (alert_key, int(CHUNK_CHRONIC_DEFERRAL_NOTIFY_COOLDOWN_HOURS)),
                fetch_one=True,
            )
            if existing:
                continue
        except Exception:
            pass

        try:
            _dispatch_push_notification(
                user_id=user_id,
                title="Tu plan parece atrasado",
                body=(
                    f"Detectamos {row['deferral_count']} reintentos en las últimas "
                    f"{int(CHUNK_CHRONIC_DEFERRAL_WINDOW_HOURS)}h. Verifica que tu zona horaria "
                    f"esté correcta en tu perfil."
                ),
                url=CHUNK_STALE_PANTRY_DEEPLINK,
            )
            execute_sql_write(
                """
                INSERT INTO system_alerts (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
                VALUES (%s, 'chronic_deferrals', 'warning', %s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (alert_key) DO UPDATE
                SET triggered_at = NOW(),
                    metadata = EXCLUDED.metadata,
                    affected_user_ids = EXCLUDED.affected_user_ids,
                    resolved_at = NULL
                """,
                (
                    alert_key,
                    "Deferrals crónicos detectados",
                    f"User {user_id} acumuló {row['deferral_count']} deferrals en {CHUNK_CHRONIC_DEFERRAL_WINDOW_HOURS}h.",
                    json.dumps({
                        "deferral_count": row["deferral_count"],
                        "meal_plan_id": row["meal_plan_id"],
                        "week_number": row["week_number"],
                        "window_hours": CHUNK_CHRONIC_DEFERRAL_WINDOW_HOURS,
                    }),
                    json.dumps([user_id]),
                ),
            )
            notified += 1
        except Exception as e:
            logger.warning(f"[P1-2/CHRONIC] No se pudo notificar deferrals crónicos a {user_id}: {e}")

    if notified > 0:
        logger.info(f"[P1-2/CHRONIC] Notificados {notified}/{len(rows)} usuarios con deferrals crónicos.")


def _alert_high_synthesized_lesson_ratio() -> None:
    """[P0-A] Detecta degradación silenciosa del aprendizaje continuo.

    Compara el número de eventos `lesson_synthesized_low_confidence` y
    `recent_lessons_partial_synthesis` en `chunk_lesson_telemetry` contra el total
    de chunks procesados (`plan_chunk_queue` con status `completed`/`failed`) en
    la ventana CHUNK_LESSON_SYNTH_WINDOW_HOURS. Si el ratio supera
    CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD (con un mínimo de muestras para
    evitar falsos positivos), inserta una alerta deduplicada en `system_alerts`.

    Síntoma típico cuando dispara: `plan_chunk_queue.learning_metrics` no se está
    persistiendo (bug en el commit del chunk previo, schema downgrade, JSON
    corrupto). Los chunks siguen completándose pero el LLM "aprende" desde
    señales degradadas — los platos del chunk N+1 no responden realmente a las
    repeticiones del N, rompiendo silenciosamente la promesa del aprendizaje
    continuo prometido a usuarios con planes 7d/15d/30d.
    """
    from constants import (
        CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD,
        CHUNK_LESSON_SYNTH_MIN_SAMPLES,
        CHUNK_LESSON_SYNTH_WINDOW_HOURS,
        CHUNK_LESSON_SYNTH_ALERT_COOLDOWN_HOURS,
    )
    _ensure_quality_alert_schema()

    try:
        stats = execute_sql_query(
            """
            SELECT
                (SELECT COUNT(*)::int FROM chunk_lesson_telemetry
                 WHERE event IN ('lesson_synthesized_low_confidence',
                                 'recent_lessons_partial_synthesis')
                   AND created_at > NOW() - make_interval(hours => %s)) AS synthesized_events,
                (SELECT COUNT(*)::int FROM plan_chunk_queue
                 WHERE status IN ('completed', 'failed')
                   AND updated_at > NOW() - make_interval(hours => %s)) AS total_chunks
            """,
            (
                int(CHUNK_LESSON_SYNTH_WINDOW_HOURS),
                int(CHUNK_LESSON_SYNTH_WINDOW_HOURS),
            ),
            fetch_one=True,
        ) or {}
    except Exception as e:
        logger.warning(f"[P0-A/SYNTH-ALERT] No se pudo consultar chunk_lesson_telemetry: {e}")
        return

    synthesized = int(stats.get("synthesized_events") or 0)
    total = int(stats.get("total_chunks") or 0)

    if total < int(CHUNK_LESSON_SYNTH_MIN_SAMPLES):
        # Tráfico insuficiente para evaluar; salimos en silencio.
        return

    ratio = (synthesized / total) if total > 0 else 0.0
    if ratio < float(CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD):
        return

    alert_key = "chunk_lesson_synth_ratio_high"
    try:
        existing = execute_sql_query(
            """
            SELECT triggered_at FROM system_alerts
            WHERE alert_key = %s
              AND triggered_at > NOW() - make_interval(hours => %s)
            LIMIT 1
            """,
            (alert_key, int(CHUNK_LESSON_SYNTH_ALERT_COOLDOWN_HOURS)),
            fetch_one=True,
        )
        if existing:
            return
    except Exception:
        pass

    try:
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
            VALUES (%s, 'chunk_lesson_synth_ratio_high', 'warning', %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                metadata = EXCLUDED.metadata,
                affected_user_ids = EXCLUDED.affected_user_ids,
                resolved_at = NULL
            """,
            (
                alert_key,
                "Aprendizaje continuo degradado: lecciones sintetizadas",
                (
                    f"En las últimas {int(CHUNK_LESSON_SYNTH_WINDOW_HOURS)}h, "
                    f"{synthesized}/{total} chunks ({ratio:.1%}) usaron lecciones "
                    f"sintetizadas low-confidence. Umbral: "
                    f"{float(CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD):.0%}. "
                    f"Posible causa: plan_chunk_queue.learning_metrics no se está "
                    f"persistiendo correctamente."
                ),
                json.dumps({
                    "synthesized_events": synthesized,
                    "total_chunks": total,
                    "ratio": round(ratio, 4),
                    "threshold": float(CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD),
                    "window_hours": int(CHUNK_LESSON_SYNTH_WINDOW_HOURS),
                }),
                json.dumps([]),
            ),
        )
        logger.warning(
            f"[P0-A/SYNTH-ALERT] Alerta disparada: synth={synthesized} total={total} "
            f"ratio={ratio:.1%} threshold={float(CHUNK_LESSON_SYNTH_RATIO_ALERT_THRESHOLD):.0%}"
        )
    except Exception as e:
        logger.error(f"[P0-A/SYNTH-ALERT] No se pudo persistir la alerta: {e}")


def _per_user_synthesis_ratio_exceeded(user_id: str) -> dict:
    """[P0-B] Versión per-usuario de la métrica que evalúa _alert_high_synthesized_lesson_ratio.

    Cuenta los eventos de síntesis low-confidence de UN usuario contra sus chunks
    completados/fallidos en CHUNK_SYNTH_PER_USER_WINDOW_HOURS. Devuelve dict con:
      - synth: count de eventos sintetizados (lesson_synthesized_low_confidence
        + recent_lessons_partial_synthesis).
      - total: count de chunks procesados (completed + failed) del mismo usuario.
      - ratio: synth/total (0.0 si total==0).
      - exceeded: bool — True si samples >= MIN_SAMPLES y ratio >= THRESHOLD.

    En caso de error de DB devuelve `exceeded=False` (fail-open): preferimos
    un chunk con learning low-confidence ocasional que bloquear al usuario por
    un blip de la query.
    """
    from constants import (
        CHUNK_SYNTH_PER_USER_RATIO_THRESHOLD,
        CHUNK_SYNTH_PER_USER_MIN_SAMPLES,
        CHUNK_SYNTH_PER_USER_WINDOW_HOURS,
    )
    fallback = {"synth": 0, "total": 0, "ratio": 0.0, "exceeded": False}
    try:
        row = execute_sql_query(
            """
            SELECT
                (SELECT COUNT(*)::int FROM chunk_lesson_telemetry
                 WHERE user_id = %s
                   AND event IN ('lesson_synthesized_low_confidence',
                                 'recent_lessons_partial_synthesis')
                   AND created_at > NOW() - make_interval(hours => %s)) AS synth,
                (SELECT COUNT(*)::int FROM plan_chunk_queue
                 WHERE user_id = %s
                   AND status IN ('completed', 'failed')
                   AND updated_at > NOW() - make_interval(hours => %s)) AS total
            """,
            (
                user_id,
                int(CHUNK_SYNTH_PER_USER_WINDOW_HOURS),
                user_id,
                int(CHUNK_SYNTH_PER_USER_WINDOW_HOURS),
            ),
            fetch_one=True,
        ) or {}
    except Exception as exc:
        logger.warning(
            f"[P0-B/PER-USER-SYNTH] Query falló para user {user_id}: "
            f"{type(exc).__name__}: {exc}. Fail-open."
        )
        return fallback

    synth = int(row.get("synth") or 0)
    total = int(row.get("total") or 0)
    ratio = (synth / total) if total > 0 else 0.0
    exceeded = (
        total >= int(CHUNK_SYNTH_PER_USER_MIN_SAMPLES)
        and ratio >= float(CHUNK_SYNTH_PER_USER_RATIO_THRESHOLD)
    )
    return {"synth": synth, "total": total, "ratio": ratio, "exceeded": exceeded}


def _pause_chunk_for_synthesis_overload(
    *,
    task_id: str,
    snap: dict,
    user_id: str,
    meal_plan_id: str,
    week_number: int,
    ratio_info: dict,
    source: str,
) -> bool:
    """[P0-B] Pausa el chunk actual cuando el ratio de síntesis per-usuario está
    por encima del umbral.

    Comportamiento:
      1. Cooldown check: si ya hay una pausa P0-B para este (plan, week) en la
         última CHUNK_SYNTH_PER_USER_PAUSE_COOLDOWN_HOURS, NO re-pausar (el
         operador ya intervino y forzó override; respetar su decisión).
      2. UPDATE plan_chunk_queue.status = 'pending_user_action' con
         pipeline_snapshot._pause_reason = 'synthesis_ratio_exceeded' + metadata.
      3. INSERT system_alerts dedup por (user, plan, week) para SRE visibility
         con affected_user_ids=[user_id]; complementa al alert agregado.
      4. Push notification al usuario.

    Args:
        source: identificador legible del callsite ("last_chunk_learning_synth"
                o "recent_lessons_regen") — se almacena en metadata para
                debugging.

    Returns:
        True si pausó (caller debe `return` inmediatamente).
        False si cooldown activo o pausa misma falló (caller continúa con
        learning low-confidence, prefiriendo señal degradada a deadlock).
    """
    from constants import CHUNK_SYNTH_PER_USER_PAUSE_COOLDOWN_HOURS

    # 1. Cooldown: ya pausamos este chunk recientemente.
    try:
        existing = execute_sql_query(
            """
            SELECT updated_at FROM plan_chunk_queue
            WHERE id = %s
              AND status = 'pending_user_action'
              AND pipeline_snapshot->>'_pause_reason' = 'synthesis_ratio_exceeded'
              AND updated_at > NOW() - make_interval(hours => %s)
            LIMIT 1
            """,
            (task_id, int(CHUNK_SYNTH_PER_USER_PAUSE_COOLDOWN_HOURS)),
            fetch_one=True,
        )
        if existing:
            logger.info(
                f"[P0-B/COOLDOWN] Chunk {task_id} ya está pausado por "
                f"synthesis_ratio_exceeded dentro de cooldown "
                f"({int(CHUNK_SYNTH_PER_USER_PAUSE_COOLDOWN_HOURS)}h). "
                f"Continuando sin re-pausar."
            )
            return False
    except Exception as _cooldown_err:
        logger.debug(
            f"[P0-B/COOLDOWN] Cooldown query falló (best-effort): {_cooldown_err}"
        )

    # 2. Persistir pausa en plan_chunk_queue.
    try:
        _pause_snap = copy.deepcopy(snap) if isinstance(snap, dict) else {}
        _pause_snap["_pause_reason"] = "synthesis_ratio_exceeded"
        _pause_snap["_p0b_synth_count"] = int(ratio_info.get("synth") or 0)
        _pause_snap["_p0b_total_count"] = int(ratio_info.get("total") or 0)
        _pause_snap["_p0b_ratio"] = float(ratio_info.get("ratio") or 0.0)
        _pause_snap["_p0b_source"] = source
        _pause_snap["_p0b_paused_at"] = datetime.now(timezone.utc).isoformat()
        execute_sql_write(
            """
            UPDATE plan_chunk_queue
            SET status = 'pending_user_action',
                pipeline_snapshot = %s::jsonb,
                updated_at = NOW()
            WHERE id = %s
            """,
            (json.dumps(_pause_snap, ensure_ascii=False), task_id),
        )
    except Exception as _pause_err:
        logger.error(
            f"[P0-B/PAUSE-FAIL] No se pudo pausar chunk {task_id} para user "
            f"{user_id}: {type(_pause_err).__name__}: {_pause_err}. "
            f"Continuando con learning low-confidence en lugar de deadlock."
        )
        return False

    # 3. system_alerts dedup por (user, plan, week) — complementa el alert agregado.
    try:
        _ensure_quality_alert_schema()
        _alert_key = f"chunk_synthesis_overload:{user_id}:{meal_plan_id}:{int(week_number)}"
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
            VALUES (%s, 'chunk_synthesis_overload_per_user', 'warning', %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                metadata = EXCLUDED.metadata,
                resolved_at = NULL
            """,
            (
                _alert_key,
                "Plan de usuario pausado: aprendizaje degradado",
                (
                    f"Usuario {user_id} plan {meal_plan_id} chunk {int(week_number)}: "
                    f"{int(ratio_info.get('synth') or 0)}/{int(ratio_info.get('total') or 0)} "
                    f"chunks recientes ({float(ratio_info.get('ratio') or 0.0):.0%}) "
                    f"usaron lecciones sintetizadas low-confidence. Pausado para "
                    f"intervención manual; investigar plan_chunk_queue.learning_metrics "
                    f"de chunks anteriores del mismo plan."
                ),
                json.dumps({
                    "synth": int(ratio_info.get("synth") or 0),
                    "total": int(ratio_info.get("total") or 0),
                    "ratio": round(float(ratio_info.get("ratio") or 0.0), 4),
                    "source": source,
                    "task_id": task_id,
                }),
                json.dumps([user_id]),
            ),
        )
    except Exception as _alert_err:
        logger.warning(
            f"[P0-B/ALERT] No se pudo persistir alert per-user: {_alert_err}"
        )

    # 4. Push notification al usuario.
    try:
        import threading as _p0b_threading
        from utils_push import send_push_notification as _p0b_push
        _p0b_threading.Thread(
            target=_p0b_push,
            kwargs={
                "user_id": user_id,
                "title": "Tu plan necesita una revisión",
                "body": (
                    "Detectamos que tu historial reciente no tiene suficiente "
                    "información para generar el siguiente bloque. Ábrelo para "
                    "que lo revisemos juntos."
                ),
                "url": "/dashboard",
            },
            daemon=True,
        ).start()
    except Exception as _push_err:
        logger.warning(
            f"[P0-B/PUSH] No se pudo enviar push notification: {_push_err}"
        )

    logger.error(
        f"[P0-B/PAUSED] Chunk {task_id} (user {user_id} plan {meal_plan_id} "
        f"week {int(week_number)}) pausado por synthesis_ratio_exceeded: "
        f"synth={int(ratio_info.get('synth') or 0)}/"
        f"total={int(ratio_info.get('total') or 0)} "
        f"ratio={float(ratio_info.get('ratio') or 0.0):.1%} source={source}"
    )
    return True


def _alert_new_dead_lettered_chunks() -> None:
    """[P1-2] Alerta proactiva sobre chunks dead-lettered acumulándose.

    `_escalate_unrecoverable_chunk` (cron_tasks.py:5631) marca chunks como
    `dead_lettered_at` permanentes cuando agotan los reintentos de recovery o
    pierden el ancla temporal del plan. El usuario afectado recibe un push,
    pero hasta ahora operadores no tenían visibilidad agregada: cada incidente
    quedaba enterrado en logs y solo emergía vía ticket de soporte.

    Este cron corre cada `CHUNK_DEAD_LETTER_ALERT_INTERVAL_MINUTES` (default 60 min),
    cuenta chunks con `dead_lettered_at > NOW() - WINDOW_HOURS`, agrega por
    `dead_letter_reason`, y persiste una alerta deduplicada en `system_alerts`
    cuando el total ≥ `CHUNK_DEAD_LETTER_ALERT_MIN_COUNT`. Cooldown
    `CHUNK_DEAD_LETTER_ALERT_COOLDOWN_HOURS` evita repetir mientras los mismos
    chunks siguen ahí; si la situación persiste tras el cooldown se re-alerta.

    Inspección concreta vía endpoint `GET /api/plans/admin/chunks/dead-lettered`
    (Bearer CRON_SECRET).
    """
    from constants import (
        CHUNK_DEAD_LETTER_ALERT_WINDOW_HOURS,
        CHUNK_DEAD_LETTER_ALERT_COOLDOWN_HOURS,
        CHUNK_DEAD_LETTER_ALERT_MIN_COUNT,
    )
    _ensure_quality_alert_schema()

    window_hours = int(CHUNK_DEAD_LETTER_ALERT_WINDOW_HOURS)
    try:
        totals = execute_sql_query(
            """
            SELECT
                COUNT(*)::int AS total,
                COUNT(DISTINCT user_id)::int AS affected_users,
                COUNT(DISTINCT meal_plan_id)::int AS affected_plans
            FROM plan_chunk_queue
            WHERE dead_lettered_at > NOW() - make_interval(hours => %s)
            """,
            (window_hours,),
            fetch_one=True,
        ) or {}
    except Exception as e:
        logger.warning(
            f"[P1-2/DEAD-LETTER-ALERT] No se pudo consultar totals "
            f"plan_chunk_queue: {e}"
        )
        return

    total = int(totals.get("total") or 0)
    if total < int(CHUNK_DEAD_LETTER_ALERT_MIN_COUNT):
        return

    affected_users = int(totals.get("affected_users") or 0)
    affected_plans = int(totals.get("affected_plans") or 0)

    by_reason: dict = {}
    try:
        reason_rows = execute_sql_query(
            """
            SELECT COALESCE(dead_letter_reason, 'unknown') AS reason,
                   COUNT(*)::int AS cnt
            FROM plan_chunk_queue
            WHERE dead_lettered_at > NOW() - make_interval(hours => %s)
            GROUP BY COALESCE(dead_letter_reason, 'unknown')
            ORDER BY cnt DESC
            """,
            (window_hours,),
            fetch_all=True,
        ) or []
        by_reason = {str(r.get("reason") or "unknown"): int(r.get("cnt") or 0) for r in reason_rows}
    except Exception as _reason_err:
        logger.debug(f"[P1-2/DEAD-LETTER-ALERT] reason breakdown falló: {_reason_err}")

    alert_key = "dead_lettered_chunks_recent"

    # Cooldown: si ya alertamos en la ventana, no repetir.
    try:
        existing = execute_sql_query(
            """
            SELECT triggered_at FROM system_alerts
            WHERE alert_key = %s
              AND triggered_at > NOW() - make_interval(hours => %s)
            LIMIT 1
            """,
            (alert_key, int(CHUNK_DEAD_LETTER_ALERT_COOLDOWN_HOURS)),
            fetch_one=True,
        )
        if existing:
            return
    except Exception:
        # Cooldown lookup falló (e.g., system_alerts no existe) — seguimos al
        # INSERT que crea la tabla vía _ensure_quality_alert_schema y persiste igual.
        pass

    affected_user_ids: list = []
    try:
        uid_rows = execute_sql_query(
            """
            SELECT DISTINCT user_id::text AS user_id
            FROM plan_chunk_queue
            WHERE dead_lettered_at > NOW() - make_interval(hours => %s)
              AND user_id IS NOT NULL
            LIMIT 200
            """,
            (window_hours,),
            fetch_all=True,
        ) or []
        affected_user_ids = [r["user_id"] for r in uid_rows if r.get("user_id")]
    except Exception:
        pass

    metadata = {
        "window_hours": window_hours,
        "total_dead_lettered": total,
        "affected_users": affected_users,
        "affected_plans": affected_plans,
        "by_reason": by_reason,
    }
    reasons_str = ", ".join(f"{k}={v}" for k, v in by_reason.items()) or "n/a"
    message = (
        f"En las últimas {window_hours}h se dead-letteraron {total} chunk(s) "
        f"de {affected_plans} plan(es) ({affected_users} usuario(s)). "
        f"Reasons: {reasons_str}. "
        f"Inspeccionar vía GET /api/plans/admin/chunks/dead-lettered "
        f"(Authorization: Bearer CRON_SECRET)."
    )

    try:
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids, triggered_at, resolved_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW(), NULL)
            ON CONFLICT (alert_key) DO UPDATE
            SET severity = EXCLUDED.severity,
                title = EXCLUDED.title,
                message = EXCLUDED.message,
                metadata = EXCLUDED.metadata,
                affected_user_ids = EXCLUDED.affected_user_ids,
                triggered_at = NOW(),
                resolved_at = NULL
            """,
            (
                alert_key,
                "dead_lettered_chunks_recent",
                "warning",
                "Chunks dead-lettered acumulándose",
                message,
                json.dumps(metadata, ensure_ascii=False),
                json.dumps(affected_user_ids),
            ),
        )
        logger.warning(f"[P1-2/DEAD-LETTER-ALERT] Alerta disparada: {message}")
    except Exception as e:
        logger.error(f"[P1-2/DEAD-LETTER-ALERT] No se pudo persistir la alerta: {e}")


def _ensure_quality_alert_schema():
    """Crea el esquema mínimo para alertas persistentes si aún no existe."""
    try:
        execute_sql_write(
            """
            CREATE TABLE IF NOT EXISTS system_alerts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                alert_key TEXT NOT NULL UNIQUE,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'warning',
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                affected_user_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
                triggered_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                resolved_at TIMESTAMP WITH TIME ZONE NULL
            )
            """
        )
        execute_sql_write(
            """
            ALTER TABLE user_profiles
            ADD COLUMN IF NOT EXISTS quality_alert_at TIMESTAMP WITH TIME ZONE
            """
        )
    except Exception as e:
        logger.warning(f"[GAP G] Error asegurando esquema de quality alerts: {e}")


def _persist_quality_degradation_alert(is_refill: bool, ratio: float, degraded: int, total: int):
    """Persiste una alerta deduplicada y marca perfiles afectados para mostrar banner en UI."""
    _ensure_quality_alert_schema()

    tipo = "refill" if is_refill else "initial"
    label = "Refill" if is_refill else "Inicial"
    alert_key = f"degraded_rate_high:{tipo}"

    try:
        degraded_rows = execute_sql_query(
            """
            SELECT DISTINCT user_id
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - INTERVAL '24 hours'
              AND COALESCE(is_rolling_refill, false) = %s
              AND (was_degraded OR quality_tier IN ('shuffle', 'edge', 'emergency'))
              AND user_id IS NOT NULL
            """,
            (is_refill,),
            fetch_all=True,
        ) or []
        affected_user_ids = [str(row.get("user_id")) for row in degraded_rows if row.get("user_id")]

        metadata = {
            "window_hours": 24,
            "is_rolling_refill": is_refill,
            "ratio": round(ratio, 4),
            "degraded_chunks": degraded,
            "total_chunks": total,
            "threshold": 0.15,
            "affected_users": len(affected_user_ids),
        }
        message = (
            f"Degraded rate 24h ({label}) = {ratio:.1%} "
            f"({degraded}/{total} chunks). LLM posiblemente inestable."
        )

        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids, triggered_at, resolved_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW(), NULL)
            ON CONFLICT (alert_key)
            DO UPDATE SET
                severity = EXCLUDED.severity,
                title = EXCLUDED.title,
                message = EXCLUDED.message,
                metadata = EXCLUDED.metadata,
                affected_user_ids = EXCLUDED.affected_user_ids,
                triggered_at = NOW(),
                resolved_at = NULL
            """,
            (
                alert_key,
                "degraded_rate_high",
                "critical",
                f"Calidad degradada alta ({label})",
                message,
                json.dumps(metadata, ensure_ascii=False),
                json.dumps(affected_user_ids),
            ),
        )

        if affected_user_ids:
            execute_sql_write(
                """
                UPDATE user_profiles
                SET quality_alert_at = NOW()
                WHERE id = ANY(%s::uuid[])
                """,
                (affected_user_ids,),
            )
    except Exception as e:
        logger.warning(f"[GAP G] Error persistiendo quality degradation alert: {e}")


def _alert_if_degraded_rate_high():
    """[GAP G] Escala degraded-rate alto a persistencia en DB y flag visible por la UI."""
    try:
        rows = execute_sql_query(
            """
            SELECT
                is_rolling_refill,
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE was_degraded OR quality_tier IN ('shuffle', 'edge', 'emergency')) AS degraded
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - INTERVAL '24 hours'
            GROUP BY is_rolling_refill
            """,
            fetch_all=True,
        )
        for row in rows:
            total = int(row.get("total") or 0)
            degraded = int(row.get("degraded") or 0)
            is_refill = row.get("is_rolling_refill", False)
            if total >= 10:  # umbral mínimo de muestra
                ratio = degraded / total
                if ratio > 0.15:
                    tipo = "Refill" if is_refill else "Inicial"
                    logger.error(
                        f"[GAP G/ALERT] Degraded rate 24h ({tipo}) = {ratio:.1%} ({degraded}/{total} chunks). "
                        f"LLM possiblemente inestable. Investigar."
                    )
                    _persist_quality_degradation_alert(
                        is_refill=is_refill,
                        ratio=ratio,
                        degraded=degraded,
                        total=total,
                    )
    except Exception as e:
        logger.warning(f"[GAP G] Error en alert de degraded rate: {e}")


def _normalize_meal_name(text: str) -> str:
    if not text:
        return ""
    return strip_accents(str(text).lower()).strip()


def _calculate_chunk_consumption_ratio(previous_chunk_days: list, consumed_records: list, consumption_mutations_count: int = 0) -> dict:
    """Calcula cuánto del chunk previo fue realmente consumido usando nombres de platos.

    [P0-3] Proxy implícito extendido a logging esparso:
    Si el usuario logea 0 comidas → se asume 100 % consumido (comportamiento original).
    Si logea MUY pocas (< 25 % del total planeado), también se usa el proxy porque
    ese nivel de logging no es señal representativa de baja adherencia.
    
    [P0-B] Matching flexible (exact, substring bidireccional, embedding top-3)
    para evitar castigar al usuario por variaciones menores al registrar manualmente.
    """
    _SPARSE_LOGGING_THRESHOLD = 0.25

    planned_pool = {}
    for day in previous_chunk_days or []:
        if not isinstance(day, dict):
            continue
        for meal in day.get("meals", []) or []:
            if not isinstance(meal, dict):
                continue
            meal_name = _normalize_meal_name(meal.get("name"))
            if meal_name and meal.get("status") not in ["swapped_out", "skipped", "rejected"]:
                planned_pool[meal_name] = planned_pool.get(meal_name, 0) + 1

    consumed_list = []
    for record in consumed_records or []:
        meal_name = _normalize_meal_name(record.get("meal_name") or record.get("name"))
        if meal_name:
            consumed_list.append(meal_name)

    planned_total = sum(planned_pool.values())
    explicit_logged = len(consumed_list)
    
    match_stats = {"exact": 0, "substring": 0, "embedding": 0, "unmatched": 0}
    explicit_matched = 0

    def _word_overlap(a: str, b: str) -> int:
        return len(set(a.split()) & set(b.split()))

    for c_name in consumed_list:
        # 1. Exact Match
        if planned_pool.get(c_name, 0) > 0:
            planned_pool[c_name] -= 1
            match_stats["exact"] += 1
            explicit_matched += 1
            continue

        # 2. Substring Match Bidireccional
        matched_p = None
        for p_name, count in planned_pool.items():
            if count > 0 and len(c_name) > 3 and len(p_name) > 3:
                if c_name in p_name or p_name in c_name:
                    matched_p = p_name
                    break
        
        if matched_p:
            planned_pool[matched_p] -= 1
            match_stats["substring"] += 1
            explicit_matched += 1
            continue

        # 3. Embedding Match Cosine >= 0.85 (Top 3 candidatos)
        available_planned = [p for p, count in planned_pool.items() if count > 0]
        if available_planned:
            candidates = sorted(available_planned, key=lambda p: _word_overlap(c_name, p), reverse=True)[:3]
            try:
                c_emb = get_embedding(c_name)
                best_score = -1.0
                best_cand = None
                
                for cand in candidates:
                    cand_emb = get_embedding(cand)
                    score = cosine_similarity(c_emb, cand_emb)
                    if score > best_score:
                        best_score = score
                        best_cand = cand
                
                if best_score >= 0.85 and best_cand:
                    planned_pool[best_cand] -= 1
                    match_stats["embedding"] += 1
                    explicit_matched += 1
                    continue
            except Exception as e:
                logger.warning(f"[P0-B/MATCH] Error en embedding para '{c_name}': {e}")
                
        # Unmatched
        match_stats["unmatched"] += 1

    logger.info(f"[P0-B/MATCH] Auditoría de Adherencia: exact={match_stats['exact']} substr={match_stats['substring']} emb={match_stats['embedding']} unmatched={match_stats['unmatched']}")

    # [P0-3] Proxy implícito: sin logs O logging esparso (< 25 % de lo planeado).
    sparse_logging = (
        planned_total > 0
        and explicit_logged > 0
        and explicit_logged < max(2, planned_total * _SPARSE_LOGGING_THRESHOLD)
    )
    # [P0-1] Distinguir zero-log de sparse-log. Antes ambos colapsaban en used_implicit_proxy
    # con ratio=1.0; eso permitía que un chunk generara sin NINGUNA señal real de adherencia
    # (el caso más frágil del aprendizaje continuo). Ahora exponemos zero_log_proxy aparte
    # para que el caller pueda pausar / forzar variedad / notificar de forma diferenciada.
    zero_log_proxy = planned_total > 0 and explicit_logged == 0
    use_implicit_proxy = zero_log_proxy or sparse_logging

    # [P1-8] Si el usuario tiene zero logs Y la nevera no se tocó
    # (consumption_mutations_count=0), no hay NINGUNA evidencia de que se haya cocinado
    # del plan — pudo haber comido takeout 100%. Antes calculábamos ratio=0.5 base
    # (la fórmula `0.5 + mutations/max(planned, 6)` arrancaba en 0.5 con 0 mutations),
    # lo cual envenenaba `learning_metrics` con "50% adherencia asumida" y dejaba que el
    # LLM construyera el siguiente chunk sobre datos falsos. Ahora retornamos ratio=0.0
    # cuando no hay logs ni actividad de inventario: refleja la realidad ("no evidencia"),
    # y el caller (gate y dashboards) puede tratar el chunk como "sin señal".
    # Para sparse-log (algún log + 0 mutations) preservamos la fórmula original porque el
    # log explícito mismo es señal mínima y la mutación cero podría ser ruido.
    _zero_log_no_mutations = (
        zero_log_proxy and int(consumption_mutations_count or 0) == 0
    )

    if use_implicit_proxy:
        matched = planned_total
        if _zero_log_no_mutations:
            ratio = 0.0
            matched = 0
        else:
            ratio = min(0.5 + (consumption_mutations_count / max(planned_total, 6)), 0.85)
    else:
        matched = explicit_matched
        ratio = matched / planned_total if planned_total else 1.0

    return {
        "ratio": round(ratio, 4),
        "matched_meals": matched,
        "planned_meals": planned_total,
        "explicit_matched_meals": explicit_matched,
        "explicit_logged_meals": explicit_logged,
        "used_implicit_proxy": use_implicit_proxy,
        "sparse_logging_proxy": sparse_logging,
        "zero_log_proxy": zero_log_proxy,
        "zero_log_no_mutations": _zero_log_no_mutations,
        "match_breakdown": match_stats,
    }


def _compute_prev_chunk_meal_breakdown(
    plan_days: list,
    prev_offset: int,
    prev_count: int,
    consumed_records: list,
    prev_chunk_number: int,
) -> dict | None:
    """[P0-5] Devuelve qué platos del chunk previo INMEDIATO consumió/saltó el usuario.

    A diferencia de `_meal_level_adherence` (EMA por TIPO de comida acumulada en todos
    los chunks), este breakdown es chunk-específico y por NOMBRE de plato — el insumo
    directo para que el LLM refuerce variantes de lo aceptado y evite repetir lo saltado.

    Devuelve None si no hay datos planeados (chunk_number<2, sin prior_days, etc.) para
    que el caller pueda omitir la inyección sin condicionales.
    """
    if not plan_days or prev_count <= 0 or prev_chunk_number < 1:
        return None

    prev_start_day = prev_offset + 1
    prev_end_day = prev_offset + prev_count

    planned_names: list[str] = []
    seen: set = set()
    for day in plan_days:
        if not isinstance(day, dict):
            continue
        day_num = int(day.get("day") or 0)
        if not (prev_start_day <= day_num <= prev_end_day):
            continue
        for meal in day.get("meals") or []:
            if not isinstance(meal, dict):
                continue
            if meal.get("status") in ("swapped_out", "skipped", "rejected"):
                continue
            raw_name = meal.get("name")
            if not raw_name:
                continue
            norm = _normalize_meal_name(raw_name)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            planned_names.append(raw_name)

    if not planned_names:
        return None

    consumed_norm = set()
    for record in consumed_records or []:
        if not isinstance(record, dict):
            continue
        norm = _normalize_meal_name(record.get("meal_name") or record.get("name"))
        if norm:
            consumed_norm.add(norm)

    consumed: list[str] = []
    skipped: list[str] = []
    for raw_name in planned_names:
        if _normalize_meal_name(raw_name) in consumed_norm:
            consumed.append(raw_name)
        else:
            skipped.append(raw_name)

    if not consumed and not skipped:
        return None

    return {
        "chunk_number": int(prev_chunk_number),
        "prev_start_day": prev_start_day,
        "prev_end_day": prev_end_day,
        "consumed_meals": consumed[:8],
        "skipped_meals": skipped[:8],
        "consumed_count": len(consumed),
        "skipped_count": len(skipped),
    }


def _resolve_previous_chunk_window(meal_plan_id: str, week_number: int, days_offset: int, total_days_requested: int = None) -> tuple[int, int]:
    """Encuentra el offset/count del chunk anterior para gating por consumo real."""
    if week_number <= 2:
        return 0, max(0, int(days_offset))

    previous_chunk = execute_sql_query(
        """
        SELECT days_offset, days_count
        FROM plan_chunk_queue
        WHERE meal_plan_id = %s AND week_number = %s
          AND status IN ('completed', 'processing')
        ORDER BY status = 'completed' DESC, updated_at DESC NULLS LAST
        LIMIT 1
        """,
        (str(meal_plan_id), int(week_number) - 1),
        fetch_one=True
    )
    if previous_chunk:
        prev_offset = max(0, int(previous_chunk.get("days_offset") or 0))
        prev_count = max(1, int(previous_chunk.get("days_count") or 1))
        return prev_offset, prev_count

    # [P0-5] Fallback: recalcular desde split_with_absorb si sabemos el total.
    # Evita asumir PLAN_CHUNK_SIZE fijo para planes de 15/30 días con chunks de 3 o 4.
    if total_days_requested and int(total_days_requested) > 0:
        from constants import split_with_absorb as _split
        chunks = _split(int(total_days_requested))
        prev_idx = int(week_number) - 2  # 0-indexed del chunk anterior
        if 0 <= prev_idx < len(chunks):
            prev_offset = sum(chunks[:prev_idx])
            prev_count = chunks[prev_idx]
            return prev_offset, prev_count

    from constants import PLAN_CHUNK_SIZE
    prev_count = min(max(1, PLAN_CHUNK_SIZE), max(1, int(days_offset)))
    prev_offset = max(0, int(days_offset) - prev_count)
    return prev_offset, prev_count


def _check_chunk_learning_ready(user_id: str, meal_plan_id: str, week_number: int, days_offset: int, plan_data: dict, snapshot: dict) -> dict:
    """Verifica si el chunk previo fue suficientemente consumido para habilitar el siguiente."""
    if int(week_number) <= 1 or int(days_offset) <= 0:
        return {"ready": True, "reason": "first_chunk"}

    form_data = (snapshot or {}).get("form_data", {}) or {}
    plan_start_date_str = form_data.get("_plan_start_date")
    _start_date_fallback_source: str | None = None

    # [P0-A] Validar antes del truthy check. Strings corruptos pero no vacíos
    # (gibberish, fechas inválidas tipo "2025-13-45", años absurdos como 2099)
    # antes pasaban este filtro y reventaban en `safe_fromisoformat` línea ~10499
    # — provocando ValueError sin atrapar que mataba el worker, o un parse
    # exitoso a una fecha imposible que congelaba el temporal gate hasta agotar
    # CHUNK_TEMPORAL_GATE_MAX_RETRIES. Ahora corruptos disparan la cascada de
    # recovery (grocery_start_date → created_at → no_anchor) igual que los
    # ausentes, y registramos telemetría dedicada para detectar el patrón.
    _corruption_reason: str | None = None
    if plan_start_date_str is not None:
        from constants import validate_plan_start_date as _p0a_validate
        _validated_dt, _corruption_reason = _p0a_validate(plan_start_date_str)
        if _corruption_reason and _corruption_reason != "empty":
            logger.error(
                f"[P0-A/CORRUPT-DATE] Plan {meal_plan_id} chunk {week_number}: "
                f"_plan_start_date={plan_start_date_str!r} inválido "
                f"({_corruption_reason}). Cayendo a cascada de recovery."
            )
            try:
                _record_chunk_deferral(
                    user_id=user_id,
                    meal_plan_id=meal_plan_id,
                    week_number=int(week_number),
                    reason=f"corrupted_plan_start_date:{_corruption_reason}",
                    days_until_prev_end=None,
                )
            except Exception as _p0a_tele_err:
                logger.warning(
                    f"[P0-A] Telemetría corrupted_plan_start_date falló: {_p0a_tele_err}"
                )
            plan_start_date_str = None  # forzar entrada a la cascada

    if not plan_start_date_str:
        # [P0-2 v2] Cascada estricta: grocery_start_date -> created_at -> BLOQUEAR.
        # El fallback a NOW() (versión previa) corrompía el temporal gate: el chunk
        # creía que el plan empezaba "hoy" y nunca permitía que chunk N+1 disparara.
        # Ahora si ningún ancla existe, devolvemos ready=False con reason
        # 'missing_start_date_no_anchor' para que el caller pause el chunk
        # explícitamente y notifique al usuario en lugar de avanzar con datos basura.
        # [P0-2] Usamos el binding de módulo (línea 19) en lugar de re-importar
        # localmente, así los tests pueden patchear `cron_tasks.execute_sql_query`.
        fallback_row = execute_sql_query(
            "SELECT plan_data->>'grocery_start_date' AS gsd, created_at FROM meal_plans WHERE id = %s",
            (meal_plan_id,), fetch_one=True
        )
        gsd = fallback_row.get("gsd") if fallback_row else None
        created_at = fallback_row.get("created_at") if fallback_row else None
        if gsd:
            plan_start_date_str = gsd
            _start_date_fallback_source = "grocery_start_date"
            logger.warning(f"[P0-2] _plan_start_date ausente; recuperado desde grocery_start_date para chunk {week_number}.")
        elif created_at:
            plan_start_date_str = created_at.date().isoformat() if hasattr(created_at, "date") else str(created_at)[:10]
            _start_date_fallback_source = "created_at"
            logger.warning(f"[P0-2] _plan_start_date y grocery_start_date ausentes; usando meal_plans.created_at={plan_start_date_str} para chunk {week_number}.")
            try:
                execute_sql_write(
                    "UPDATE meal_plans SET plan_data = jsonb_set(COALESCE(plan_data, '{}'::jsonb), '{grocery_start_date}', to_jsonb(%s::text), true) WHERE id = %s AND (plan_data->>'grocery_start_date') IS NULL",
                    (plan_start_date_str, meal_plan_id),
                )
            except Exception as _persist_err:
                logger.warning(f"[P0-2] No se pudo persistir grocery_start_date desde created_at: {_persist_err}")
        else:
            # [P0-2 v2] Sin ancla recuperable: bloquear en lugar de fabricar fecha con NOW().
            # Telemetría: insertar en chunk_deferrals con reason='start_date_fallback:no_anchor'
            # para que _detect_chronic_deferrals y dashboards de operación detecten el patrón.
            logger.error(
                f"[P0-2/NO-ANCHOR] _plan_start_date, grocery_start_date y created_at ausentes "
                f"para meal_plan {meal_plan_id} chunk {week_number}. Bloqueando chunk para "
                f"intervención manual; NO se fabricará una fecha con NOW()."
            )
            try:
                _record_chunk_deferral(
                    user_id=user_id,
                    meal_plan_id=meal_plan_id,
                    week_number=int(week_number),
                    reason="start_date_fallback:no_anchor",
                    days_until_prev_end=None,
                )
            except Exception as _tele_err:
                logger.warning(f"[P0-2] Telemetría no_anchor falló: {_tele_err}")
            return {
                "ready": False,
                "reason": "missing_start_date_no_anchor",
                "_fallback_source": "no_anchor",
            }

        # [P1-3 FIX] Persistir el _plan_start_date recuperado dentro del snapshot del chunk queue
        # para que las próximas evaluaciones del gate (este chunk si se difiere, y los chunks
        # siblings pending/stale del mismo plan) lo lean directo del snapshot sin reintentar
        # el fallback SQL. Antes, cada evaluación repetía la query a meal_plans aunque el valor
        # ya hubiera sido recuperado.
        # [P0-A] Si la entrada original era corrupta (no `empty`), pisamos el valor existente
        # en lugar de saltar cuando el snapshot ya tiene algo: ese "algo" es justamente la
        # basura que detectamos. Por la misma razón, sembramos el valor recuperado en
        # meal_plans.plan_data._plan_start_date — de otro modo, cualquier rebuild futuro del
        # snapshot (renovaciones, recovery paths) reintroduciría la corrupción.
        if plan_start_date_str:
            _p0a_overwrite_corrupt = bool(_corruption_reason) and _corruption_reason != "empty"
            try:
                form_data["_plan_start_date"] = plan_start_date_str
                if _p0a_overwrite_corrupt:
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET pipeline_snapshot = jsonb_set(
                                COALESCE(pipeline_snapshot, '{}'::jsonb),
                                '{form_data,_plan_start_date}',
                                to_jsonb(%s::text),
                                true
                            ),
                            updated_at = NOW()
                        WHERE meal_plan_id = %s
                          AND status IN ('pending', 'processing', 'stale')
                        """,
                        (plan_start_date_str, meal_plan_id),
                    )
                else:
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET pipeline_snapshot = jsonb_set(
                                COALESCE(pipeline_snapshot, '{}'::jsonb),
                                '{form_data,_plan_start_date}',
                                to_jsonb(%s::text),
                                true
                            ),
                            updated_at = NOW()
                        WHERE meal_plan_id = %s
                          AND status IN ('pending', 'processing', 'stale')
                          AND (pipeline_snapshot->'form_data'->>'_plan_start_date') IS NULL
                        """,
                        (plan_start_date_str, meal_plan_id),
                    )
            except Exception as _snap_persist_err:
                logger.warning(f"[P1-3] No se pudo persistir _plan_start_date al snapshot del chunk queue: {_snap_persist_err}")

            # [P0-A] Curar la corrupción en la fuente: meal_plans.plan_data._plan_start_date.
            # Sin esto, el campo permanece corrupto en plan_data y cualquier path que lo lea
            # directamente (renovación de plan, retro-actividad, exportes) reproduciría el bug.
            if _p0a_overwrite_corrupt:
                try:
                    execute_sql_write(
                        """
                        UPDATE meal_plans
                        SET plan_data = jsonb_set(
                                COALESCE(plan_data, '{}'::jsonb),
                                '{_plan_start_date}',
                                to_jsonb(%s::text),
                                true
                            )
                        WHERE id = %s
                        """,
                        (plan_start_date_str, meal_plan_id),
                    )
                except Exception as _p0a_heal_err:
                    logger.warning(
                        f"[P0-A] No se pudo curar _plan_start_date en meal_plans "
                        f"(plan {meal_plan_id}): {_p0a_heal_err}"
                    )

        # [P0-2] Telemetría de fallback exitoso (gsd o created_at). Registrar una sola vez
        # por chunk para alertar si >5% de chunks dependen de fallback (síntoma de
        # corrupción histórica del campo _plan_start_date al construir el snapshot).
        if _start_date_fallback_source and not form_data.get("_plan_start_date_fallback_logged"):
            try:
                _record_chunk_deferral(
                    user_id=user_id,
                    meal_plan_id=meal_plan_id,
                    week_number=int(week_number),
                    reason=f"start_date_fallback:{_start_date_fallback_source}",
                    days_until_prev_end=None,
                )
                form_data["_plan_start_date_fallback_logged"] = True
            except Exception as _tele_err:
                logger.warning(f"[P0-2] Telemetría fallback {_start_date_fallback_source} falló: {_tele_err}")

    total_days_requested = (snapshot or {}).get("totalDays") or form_data.get("totalDays")
    prev_offset, prev_count = _resolve_previous_chunk_window(meal_plan_id, week_number, days_offset, total_days_requested)
    prev_start_day = prev_offset + 1
    prev_end_day = prev_offset + prev_count
    previous_chunk_days = [
        d for d in (plan_data.get("days", []) or [])
        if isinstance(d, dict) and prev_start_day <= int(d.get("day") or 0) <= prev_end_day
    ]
    if not previous_chunk_days:
        # [P0-3/QUEUE-CHECK] Antes de fail-open, verificar plan_chunk_queue.
        # El path original ("plan_data.days no tiene los días del chunk previo →
        # ready=True") era demasiado permisivo: si chunk N sigue en estado activo
        # ('pending', 'processing', 'stale'), todavía no commiteó days a plan_data,
        # y devolver ready=True dejaría a chunk N+1 dispararse con learning vacío
        # — exactamente la race que P0-1 cierra a nivel de transacción pero que
        # también debe cerrarse aquí para defensa en profundidad. La pickup query
        # (process_plan_chunk_queue ~9776) ya filtra por NOT EXISTS de chunks
        # processing del mismo plan, pero el gate también se invoca desde paths
        # de recovery (_recover_pantry_paused_chunks ~4455) que no pasan por la
        # pickup query — esos sí pueden race. Este check los cubre.
        #
        # Si NO hay chunks anteriores activos en queue, mantenemos el fail-open
        # original (escenarios legítimos: chunk N falló permanentemente, plan
        # corrupto post-recovery, etc.) para no bloquear el plan en deadlock.
        try:
            _p03q_row = execute_sql_query(
                """
                SELECT COUNT(*) AS n
                FROM plan_chunk_queue
                WHERE meal_plan_id = %s
                  AND week_number < %s
                  AND status IN ('pending', 'processing', 'stale')
                """,
                (meal_plan_id, int(week_number)),
                fetch_one=True,
            )
            _p03q_active = int((_p03q_row or {}).get("n") or 0)
        except Exception as _p03q_err:
            logger.warning(
                f"[P0-3/GATE-QUEUE] Error consultando estado de chunks anteriores "
                f"para plan {meal_plan_id} week {week_number}: "
                f"{type(_p03q_err).__name__}: {_p03q_err}. Best-effort: continuar "
                f"con fallback fail-open original."
            )
            _p03q_active = 0
        if _p03q_active > 0:
            logger.warning(
                f"[P0-3/GATE-QUEUE] Chunk {week_number} plan {meal_plan_id}: "
                f"plan_data.days no tiene los días del chunk previo Y "
                f"{_p03q_active} chunk(s) anteriores siguen activos en queue "
                f"(pending/processing/stale). Deferring para evitar dispatch con "
                f"learning vacío."
            )
            try:
                _record_chunk_deferral(
                    user_id=user_id,
                    meal_plan_id=meal_plan_id,
                    week_number=int(week_number),
                    reason="prev_chunk_still_active_in_queue",
                    days_until_prev_end=None,
                )
            except Exception as _tele_err:
                logger.debug(
                    f"[P0-3/GATE-QUEUE] Telemetría chunk_deferrals falló: {_tele_err}"
                )
            return {
                "ready": False,
                "reason": "prev_chunk_still_active_in_queue",
                "_active_prior_chunks": _p03q_active,
            }

        # [P0-3 FIX] Cerrar el race entre lectura upstream de plan_data y commit
        # de T1/T2 del chunk previo.
        #
        # Escenario: el caller (`_chunk_worker`, `_recover_pantry_paused_chunks`)
        # leyó plan_data al tiempo T. Entre T y este punto del gate, chunk N pudo:
        #   1. Commitear T1 (días + learning a plan_data) — ahora con P0-1 atómico.
        #   2. Commitear T2 (status='completed' en plan_chunk_queue).
        # Bajo READ COMMITTED, las dos lecturas (plan_data upstream y queue check
        # arriba) viven en transacciones distintas y ven snapshots distintos. Si
        # T1+T2 commitean entre ambas, `previous_chunk_days` queda vacío (de la
        # plan_data stale), pero el queue check ve 0 active (post-T2). El path
        # legacy (`return ready=True, reason='missing_previous_chunk_days'`) deja
        # a chunk N+1 dispatchear con `previous_chunk_days=[]` y, por tanto, sin
        # las lecciones reales de N — exactamente el bug que P0-3 cierra.
        #
        # La pickup query (`process_plan_chunk_queue ~12490`) ya serializa por
        # meal_plan_id (NOT EXISTS sobre status='processing'), pero los call sites
        # de recovery (`_recover_pantry_paused_chunks ~6037` y futuros) NO pasan
        # por el pickup; allí la race es real.
        #
        # Fix: si el queue check confirma 0 chunks activos (estado consistente al
        # momento del check), la plan_data autoritativa también debe estar al día
        # con N. Re-leemos fresca aquí y recomputamos `previous_chunk_days`. Si
        # ahora aparecen días, el snapshot upstream era stale; continuamos con
        # datos frescos. Si siguen ausentes, el fail-open original aplica
        # (chunk previo falló permanentemente, plan corrupto post-recovery, etc.).
        try:
            _p03f_row = execute_sql_query(
                "SELECT plan_data FROM meal_plans WHERE id = %s",
                (meal_plan_id,),
                fetch_one=True,
            )
            _p03f_pd = (_p03f_row or {}).get("plan_data") or {}
            if isinstance(_p03f_pd, str):
                try:
                    _p03f_pd = json.loads(_p03f_pd)
                except Exception:
                    _p03f_pd = {}
            _p03f_days = [
                d for d in (_p03f_pd.get("days", []) or [])
                if isinstance(d, dict) and prev_start_day <= int(d.get("day") or 0) <= prev_end_day
            ]
        except Exception as _p03f_err:
            logger.warning(
                f"[P0-3/STALE-RECOVERY] Error re-leyendo plan_data fresca para "
                f"plan {meal_plan_id} chunk {week_number}: "
                f"{type(_p03f_err).__name__}: {_p03f_err}. Fail-open con snapshot stale."
            )
            return {"ready": True, "reason": "missing_previous_chunk_days"}

        if _p03f_days:
            # Race confirmada: el snapshot upstream perdió el merge de N. Continuamos
            # con datos frescos para que el resto del gate (temporal check, learning
            # ratio) opere sobre el estado real.
            logger.info(
                f"[P0-3/STALE-RECOVERY] chunk {week_number} plan {meal_plan_id}: "
                f"plan_data upstream era stale (faltaban días {prev_start_day}-{prev_end_day} "
                f"del chunk previo). Recuperados {len(_p03f_days)} días tras re-lectura "
                f"fresca. Continuando con datos actualizados."
            )
            try:
                _record_chunk_deferral(
                    user_id=user_id,
                    meal_plan_id=meal_plan_id,
                    week_number=int(week_number),
                    reason="stale_plan_data_recovered",
                    days_until_prev_end=None,
                )
            except Exception as _p03f_tele_err:
                logger.debug(
                    f"[P0-3/STALE-RECOVERY] Telemetría stale_plan_data_recovered falló: "
                    f"{_p03f_tele_err}"
                )
            previous_chunk_days = _p03f_days
            # Continuar con la evaluación normal del gate (cae al bloque temporal abajo).
        else:
            return {"ready": True, "reason": "missing_previous_chunk_days"}

    from constants import safe_fromisoformat, CHUNK_PROACTIVE_MARGIN_DAYS as _proactive_margin
    # [P0-A] Defensa en profundidad: incluso después del cascade, el valor recuperado
    # (grocery_start_date / created_at) podría ser corrupto si plan_data fue dañado
    # globalmente. Antes este parse era no-protegido y un ValueError aquí mataba al
    # worker en bucle de retry hasta dead-letter. Ahora pausamos el chunk con un
    # reason específico para que operadores puedan investigar el plan en lugar de
    # ver al usuario con su plan colgado.
    try:
        plan_start_dt = safe_fromisoformat(plan_start_date_str)
    except (ValueError, TypeError) as _p0a_parse_err:
        logger.error(
            f"[P0-A/UNRECOVERABLE] Plan {meal_plan_id} chunk {week_number}: "
            f"valor recuperado {plan_start_date_str!r} (fuente="
            f"{_start_date_fallback_source or 'snapshot'}) no parsea: "
            f"{type(_p0a_parse_err).__name__}: {_p0a_parse_err}. "
            f"Bloqueando chunk para intervención manual."
        )
        try:
            _record_chunk_deferral(
                user_id=user_id,
                meal_plan_id=meal_plan_id,
                week_number=int(week_number),
                reason="unrecoverable_corrupted_date",
                days_until_prev_end=None,
            )
        except Exception as _p0a_tele_err:
            logger.warning(
                f"[P0-A] Telemetría unrecoverable_corrupted_date falló: {_p0a_tele_err}"
            )
        return {
            "ready": False,
            "reason": "unrecoverable_corrupted_date",
            "_fallback_source": _start_date_fallback_source or "snapshot",
            "_corruption_detail": f"{type(_p0a_parse_err).__name__}",
        }

    # [P0-1 FIX] Temporal gate: el chunk N+1 sólo puede aprender de N una vez el último día
    # de N haya transcurrido (margin=0, default). Antes la comparación era `>` lo que permitía
    # disparar el gate el MISMO día en que terminaba el chunk previo, antes de que el usuario
    # hubiera tenido cena/última comida — produciendo "aprendizaje" con un día incompleto.
    # Con `>=`, margin=0 bloquea cuando today<=prev_end y sólo pasa cuando today>prev_end.
    # Para usuarios que reporten gaps al despertar, subir CHUNK_PROACTIVE_MARGIN_DAYS a 1
    # restaura la posibilidad de disparar el chunk el mismo día que termina el previo.
    # [test fix] Usamos los aliases a nivel de módulo (_dt_p0b, _tz_p0b definidos arriba)
    # en lugar de re-importarlos localmente. Esto permite que tests
    # (`@patch("cron_tasks._dt_p0b.now")`) controlen el reloj — un re-import local
    # creaba un binding nuevo en el frame de la función, invisible al patcher.

    # [P0-4] User timezone alignment
    _tz_offset_snapshot = int(form_data.get("tzOffset") or form_data.get("tz_offset_minutes") or 0)
    
    # [P0-delta] Leer tz_offset vivo del user_profile para detectar viajes / cambios de zona horaria
    _tz_offset_live = _tz_offset_snapshot
    try:
        # [P0-2] execute_sql_query y execute_sql_write usan el binding del módulo
        # (línea 19) en lugar de re-importarse localmente. El re-import creaba un
        # frame-local que rompía `@patch("cron_tasks.execute_sql_query")` en tests
        # y además causaba UnboundLocalError si una rama anterior nunca lo asignaba.
        from constants import (
            CHUNK_TZ_DRIFT_THRESHOLD_MINUTES,
            CHUNK_TZ_FORCED_RESYNC_DEFERRALS,
        )
        _profile_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
        if _profile_res and _profile_res.get("health_profile"):
            _hp = _profile_res.get("health_profile")
            _live_val = _hp.get("tz_offset_minutes") if "tz_offset_minutes" in _hp else _hp.get("tzOffset")
            if _live_val is not None:
                _tz_offset_live = int(_live_val)

        # [P0-5] Resync por dos caminos:
        #   (a) drift >= 15 min (cubre DST y viajes a TZs adyacentes incl. +30m/+45m).
        #   (b) red de seguridad: si el chunk lleva >= 3 deferrals por temporal_gate, refrescar
        #       igual aunque drift sea menor — el snapshot pudo nacer con un TZ ya incorrecto.
        _tz_drift = abs(_tz_offset_live - _tz_offset_snapshot)
        _temporal_deferrals = int((snapshot or {}).get("_learning_ready_deferrals") or 0)
        _drift_triggers = _tz_drift >= CHUNK_TZ_DRIFT_THRESHOLD_MINUTES
        _safety_triggers = _temporal_deferrals >= CHUNK_TZ_FORCED_RESYNC_DEFERRALS and _tz_drift > 0
        if _drift_triggers or _safety_triggers:
            _trigger_reason = "drift_threshold" if _drift_triggers else "deferral_safety_net"
            logger.info(
                f"🌍 [P0-5/TZ-DRIFT] Usuario {user_id}: snapshot={_tz_offset_snapshot}m, live={_tz_offset_live}m, "
                f"drift={_tz_drift}m, deferrals={_temporal_deferrals}, trigger={_trigger_reason}. "
                f"Iniciando resync atómico."
            )
            # [P0-5 v2] Atomicidad del resync. Antes:
            #   1. SELECT health_profile (sin lock).
            #   2. Calcular drift contra _tz_offset_snapshot del form_data en frame.
            #   3. UPDATE bulk de todos los chunks pendientes del meal_plan.
            #   4. (Opcional) Push notification + UPDATE flag _tz_drift_notified.
            # Cuando dos workers del MISMO meal_plan evalúan el gate simultáneamente,
            # ambos podían ver el snapshot stale, ambos disparaban push notification y
            # ambos hacían UPDATEs redundantes (idempotentes en valor pero generaban
            # bloat y duplicaban la notificación al usuario).
            # Ahora: pg_advisory_xact_lock serializa los workers a nivel meal_plan; el
            # primer worker hace el resync, el segundo entra al lock, RE-LEE el snapshot
            # del chunk (ya actualizado), detecta que el drift es 0 contra el live y se
            # salta el resync evitando doble push.
            from db_core import connection_pool as _p05_pool
            _p05_done_inline = False
            if _p05_pool:
                try:
                    from psycopg.rows import dict_row as _p05_dict_row
                    with _p05_pool.connection() as _p05_conn:
                        with _p05_conn.transaction():
                            with _p05_conn.cursor(row_factory=_p05_dict_row) as _p05_cur:
                                # [P1-5] Lock advisory por meal_plan: solo un worker del mismo
                                # plan puede ejecutar el resync a la vez. Migrado al helper
                                # canónico (`acquire_meal_plan_advisory_lock`) para unificar
                                # la hash function y namespace de keys con catchup, en vez de
                                # invocar `pg_advisory_xact_lock(hashtextextended(...))` directo.
                                from db_plans import acquire_meal_plan_advisory_lock as _p05_acquire_lock
                                _p05_acquire_lock(_p05_cur, meal_plan_id, purpose="tz_resync")
                                # Re-leer pipeline_snapshot del chunk actual DENTRO del lock
                                # para ver si otro worker ya hizo el resync.
                                _p05_cur.execute(
                                    """
                                    SELECT pipeline_snapshot
                                    FROM plan_chunk_queue
                                    WHERE meal_plan_id = %s AND week_number = %s
                                    ORDER BY created_at DESC
                                    LIMIT 1
                                    """,
                                    (meal_plan_id, week_number),
                                )
                                _p05_row = _p05_cur.fetchone()
                                _p05_fresh_snap = (_p05_row or {}).get("pipeline_snapshot") or {}
                                if isinstance(_p05_fresh_snap, str):
                                    try:
                                        _p05_fresh_snap = json.loads(_p05_fresh_snap)
                                    except Exception:
                                        _p05_fresh_snap = {}
                                _p05_fresh_form = (_p05_fresh_snap or {}).get("form_data") or {}
                                _p05_fresh_tz = int(
                                    _p05_fresh_form.get("tzOffset")
                                    or _p05_fresh_form.get("tz_offset_minutes")
                                    or 0
                                )
                                _p05_fresh_drift = abs(_tz_offset_live - _p05_fresh_tz)
                                _p05_fresh_drift_triggers = _p05_fresh_drift >= CHUNK_TZ_DRIFT_THRESHOLD_MINUTES
                                _p05_fresh_safety_triggers = (
                                    _temporal_deferrals >= CHUNK_TZ_FORCED_RESYNC_DEFERRALS
                                    and _p05_fresh_drift > 0
                                )
                                _p05_already_notified = bool(_p05_fresh_snap.get("_tz_drift_notified"))

                                if _p05_fresh_drift_triggers or _p05_fresh_safety_triggers:
                                    _p05_cur.execute(
                                        """
                                        UPDATE plan_chunk_queue
                                        SET pipeline_snapshot = jsonb_set(
                                                jsonb_set(pipeline_snapshot, '{form_data,tzOffset}', %s::jsonb, true),
                                                '{form_data,tz_offset_minutes}', %s::jsonb, true
                                            ),
                                            updated_at = NOW()
                                        WHERE meal_plan_id = %s AND status IN ('pending', 'processing', 'stale')
                                        """,
                                        (str(_tz_offset_live), str(_tz_offset_live), meal_plan_id),
                                    )
                                    if _p05_fresh_drift_triggers and not _p05_already_notified:
                                        try:
                                            _dispatch_push_notification(
                                                user_id=user_id,
                                                title="Detectamos un cambio de zona horaria",
                                                body="Ajustamos tu plan para que coincida con tu hora local actual.",
                                                url="/dashboard",
                                            )
                                            _p05_cur.execute(
                                                """
                                                UPDATE plan_chunk_queue
                                                SET pipeline_snapshot = jsonb_set(
                                                        COALESCE(pipeline_snapshot, '{}'::jsonb),
                                                        '{_tz_drift_notified}',
                                                        'true'::jsonb,
                                                        true
                                                    )
                                                WHERE meal_plan_id = %s AND status IN ('pending', 'processing', 'stale')
                                                """,
                                                (meal_plan_id,),
                                            )
                                        except Exception as _p05_notify_err:
                                            logger.warning(
                                                f"[P0-5/TZ-DRIFT] No se pudo notificar al usuario: {_p05_notify_err}"
                                            )
                                else:
                                    logger.info(
                                        f"[P0-5/TZ-DRIFT] Resync no necesario tras lock: "
                                        f"otro worker ya alineó snapshot (fresh_tz={_p05_fresh_tz}, "
                                        f"live={_tz_offset_live}, drift={_p05_fresh_drift}m)."
                                    )
                    _p05_done_inline = True
                except Exception as _p05_err:
                    # Pool/transacción falló — caer al UPDATE no-atómico legacy abajo
                    # como degradación graceful (mejor un resync no-serializado que ningún
                    # resync, que dejaría al chunk disparando en TZ vieja).
                    logger.warning(
                        f"[P0-5/TZ-DRIFT] Resync atómico falló, cayendo a UPDATE legacy: "
                        f"{type(_p05_err).__name__}: {_p05_err}"
                    )

            if not _p05_done_inline:
                # Fallback no-atómico (vulnerable a race) — solo si la transacción atómica
                # no pudo correr (e.g., pool no disponible).
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET pipeline_snapshot = jsonb_set(
                            jsonb_set(pipeline_snapshot, '{form_data,tzOffset}', %s::jsonb, true),
                            '{form_data,tz_offset_minutes}', %s::jsonb, true
                        ),
                        updated_at = NOW()
                    WHERE meal_plan_id = %s AND status IN ('pending', 'processing', 'stale')
                    """,
                    (str(_tz_offset_live), str(_tz_offset_live), meal_plan_id)
                )
                if _drift_triggers and not (snapshot or {}).get("_tz_drift_notified"):
                    try:
                        _dispatch_push_notification(
                            user_id=user_id,
                            title="Detectamos un cambio de zona horaria",
                            body="Ajustamos tu plan para que coincida con tu hora local actual.",
                            url="/dashboard",
                        )
                        execute_sql_write(
                            """
                            UPDATE plan_chunk_queue
                            SET pipeline_snapshot = jsonb_set(
                                    COALESCE(pipeline_snapshot, '{}'::jsonb),
                                    '{_tz_drift_notified}',
                                    'true'::jsonb,
                                    true
                                )
                            WHERE meal_plan_id = %s AND status IN ('pending', 'processing', 'stale')
                            """,
                            (meal_plan_id,),
                        )
                    except Exception as _notify_err:
                        logger.warning(f"[P0-5/TZ-DRIFT] No se pudo notificar al usuario: {_notify_err}")

            # Mantener form_data en frame coherente con el live (independiente de la rama
            # tomada). El frame local termina aquí; el snapshot ya quedó persistido.
            form_data["tzOffset"] = _tz_offset_live
            form_data["tz_offset_minutes"] = _tz_offset_live
    except Exception as e:
        logger.warning(f"[P0-delta] Error sincronizando TZ drift para {user_id}: {e}")

    _tz_offset_min = _tz_offset_live
    _user_now = _dt_p0b_now(_tz_p0b.utc) - timedelta(minutes=_tz_offset_min)
    _today_user = _user_now.date()
    
    _shift_days_accumulated = int(plan_data.get("_shift_days_accumulated", 0))

    # [P1-7] Override de _prev_end_date para rolling refill cuando el chunk previo
    # pertenece al plan ORIGINAL (no al refill). Sin esto, el cálculo legacy
    #   _prev_end_date = new_plan_start + (prev_offset + prev_count - 1)
    # combinaba `new_plan_start_iso` (ancla del refill, ~today) con `prev_offset/count`
    # del chunk_kind='initial_plan' (offsets relativos al plan_start ORIGINAL), dando
    # fechas futuras erróneas (e.g. plan creado 2026-04-25, refill 2026-05-01, chunk_3
    # original con offset=6, count=3 → _prev_end_date = 2026-05-09 vs lo correcto
    # 2026-04-30). Resultado: el primer chunk del refill se difería 8+ días después de
    # que el plan original ya había terminado.
    #
    # Estrategia: si el snapshot lleva `_is_continuation=True` y `_continuation_anchor_iso`
    # (set por _enqueue durante el rolling refill), Y el chunk previo en plan_chunk_queue
    # NO es del refill, usamos `anchor_iso - 1 día` como _prev_end_date (el último día
    # del plan original terminó el día antes del primer día del refill). Para chunks
    # subsiguientes del refill (cuyo prev SÍ es del refill), el cálculo legacy es
    # correcto porque el chunk previo se ancla al new_plan_start_iso.
    _prev_end_date = (plan_start_dt + timedelta(days=prev_end_day - 1 + _shift_days_accumulated)).date()

    _is_continuation = bool(form_data.get("_is_continuation"))
    _anchor_iso = form_data.get("_continuation_anchor_iso")
    if _is_continuation and _anchor_iso and int(week_number) >= 2:
        try:
            _prev_chunk_kind_row = execute_sql_query(
                """
                SELECT chunk_kind FROM plan_chunk_queue
                WHERE meal_plan_id = %s AND week_number = %s
                  AND status IN ('completed', 'processing')
                ORDER BY status = 'completed' DESC, updated_at DESC NULLS LAST
                LIMIT 1
                """,
                (str(meal_plan_id), int(week_number) - 1),
                fetch_one=True,
            )
            _prev_kind = (_prev_chunk_kind_row or {}).get("chunk_kind")
            _prev_is_original = (_prev_kind != "rolling_refill") if _prev_kind else True
            if _prev_is_original:
                _anchor_dt = safe_fromisoformat(_anchor_iso)
                if _anchor_dt is not None:
                    _override_end = (_anchor_dt - timedelta(days=1)).date()
                    if _override_end != _prev_end_date:
                        logger.info(
                            f"[P1-7/CONT-ANCHOR] Plan {meal_plan_id} chunk {week_number}: "
                            f"override _prev_end_date legacy={_prev_end_date.isoformat()} "
                            f"→ anchor={_override_end.isoformat()} (prev_kind={_prev_kind!r})."
                        )
                    _prev_end_date = _override_end
        except Exception as _p17_err:
            logger.warning(
                f"[P1-7/CONT-ANCHOR] Error resolviendo anchor para plan {meal_plan_id} "
                f"chunk {week_number}: {_p17_err}. Cayendo a cálculo legacy."
            )

    _days_until_prev_end = (_prev_end_date - _today_user).days  # >=0 = último día aún no concluyó

    # [P0-gamma] Calcular diversidad de despensa
    from constants import normalize_ingredient_for_tracking, get_nutritional_category
    current_pantry = form_data.get("current_pantry_ingredients") or form_data.get("current_shopping_list", [])
    unique_protein_bases = set()
    for item in current_pantry:
        if not item or not isinstance(item, str):
            continue
        base = normalize_ingredient_for_tracking(item)
        if base and get_nutritional_category(base) == "protein":
            unique_protein_bases.add(base)
            
    chunk_size = max(1, prev_count)
    pantry_diversity_score = len(unique_protein_bases) / chunk_size
    
    if pantry_diversity_score < 1.5:
        form_data["_pantry_diversity_warning"] = True
        
    if _days_until_prev_end == 1 and pantry_diversity_score < 1.0 and not form_data.get("_pantry_warning_sent"):
        form_data["_pantry_warning_sent"] = True
        try:
            # [P0-2] execute_sql_write usa el binding del módulo (línea 19). Re-importarlo
            # localmente declaraba la variable como frame-local y rompía las ramas previas
            # de la misma función con UnboundLocalError.
            # [P0-4] La columna real en plan_chunk_queue es `pipeline_snapshot` (ver app.py:128-136).
            # Antes este UPDATE usaba `snapshot`, lo cual lanzaba UndefinedColumn que el except de
            # abajo enmascaraba como warning. Resultado: el flag _pantry_warning_sent jamás se
            # persistía y la notificación "Actualiza tu nevera" se reenviaba en cada evaluación
            # del gate. Ahora persiste correctamente y, si vuelve a fallar, lo reportamos como
            # ERROR para que regresiones de schema no vuelvan a ocultarse.
            execute_sql_write(
                "UPDATE plan_chunk_queue "
                "SET pipeline_snapshot = jsonb_set(pipeline_snapshot, '{form_data,_pantry_warning_sent}', 'true'::jsonb) "
                "WHERE meal_plan_id = %s AND week_number = %s",
                (meal_plan_id, week_number)
            )
            from services import create_notification
            create_notification(
                user_id=user_id,
                title="Actualiza tu nevera",
                message="Tu nevera necesita reposición para que el plan siga variado",
                notification_type="warning"
            )
            logger.info(f"[P0-gamma] Notificación de diversidad de nevera enviada a {user_id}")
        except Exception as e:
            logger.error(
                f"[P0-gamma] Error enviando notif de diversidad para plan {meal_plan_id} "
                f"week {week_number}: {type(e).__name__}: {e}"
            )

    # [P0-1] Cambio de `>` a `>=`. Con margin=0 (default) el operador anterior dejaba
    # pasar el chunk el MISMO día en que terminaba el chunk previo (days_until_prev_end==0),
    # rompiendo el contrato "aprender del chunk previo una vez concluido": el LLM corría
    # antes de que el usuario tuviera cena/última comida del día final, perdiendo señales
    # de adherencia. Ahora pasa solo cuando today > prev_end. La intención ya estaba
    # documentada en el bloque [P0-1 FIX] de arriba (línea 5594) y en el comentario de
    # _days_until_prev_end (línea 5686), pero el operador nunca se había actualizado.
    # Para usuarios que prefieran disparar proactivamente antes del fin, subir
    # CHUNK_PROACTIVE_MARGIN_DAYS a 1 o 2 ajusta la ventana sin tocar este código.
    if _days_until_prev_end >= _proactive_margin:
        _prev_start_iso_log = (plan_start_dt + timedelta(days=prev_offset)).isoformat()

        # [P1-C] Cap de reintentos del temporal gate. Sin este cap, un chunk con
        # `_plan_start_date` lejano-en-el-futuro (típico de TZ desalineada o
        # snapshot corrupto) se diferia indefinidamente: cada tick del cron repetía
        # la misma evaluación, llamaba a `_record_chunk_deferral` y volvía con
        # `ready=False`. El usuario veía su plan trabado para siempre y solo
        # `_detect_chronic_deferrals` notificaba (pero la notificación NO desbloquea
        # nada — depende de que el usuario corrija). Tras CHUNK_TEMPORAL_GATE_MAX_RETRIES
        # forzamos `ready=True` con telemetría especial; el chunk avanza usando la
        # fecha best-effort que tenemos. Mejor un plan generado tarde que un plan colgado.
        _p1c_retries = int((snapshot or {}).get("_temporal_gate_retries") or 0)
        _p1c_next_retries = _p1c_retries + 1
        if _p1c_next_retries > int(CHUNK_TEMPORAL_GATE_MAX_RETRIES):
            logger.error(
                f"[P1-C/MAX-RETRIES] Plan {meal_plan_id} chunk {week_number} "
                f"alcanzó {_p1c_next_retries} reintentos del temporal_gate "
                f"(cap={int(CHUNK_TEMPORAL_GATE_MAX_RETRIES)}). Forzando ready=True "
                f"para evitar atasco indefinido. days_until_prev_end={_days_until_prev_end}, "
                f"prev_end_date={_prev_end_date.isoformat()}. "
                f"Investigar plan_start_date / TZ del usuario {user_id}."
            )
            try:
                _record_chunk_deferral(
                    user_id=user_id,
                    meal_plan_id=meal_plan_id,
                    week_number=int(week_number),
                    reason="temporal_gate_max_retries_exceeded",
                    days_until_prev_end=int(_days_until_prev_end),
                )
            except Exception as _p1c_tele_err:
                logger.warning(f"[P1-C/MAX-RETRIES] Telemetría falló: {_p1c_tele_err}")
            # Reset el contador a 0 para que evaluaciones futuras (post-override)
            # no vean un valor que ya superó el cap (evita re-disparar inmediatamente).
            # `jsonb_set` anidado para escribir dos claves en una sola UPDATE.
            try:
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET pipeline_snapshot = jsonb_set(
                            jsonb_set(
                                COALESCE(pipeline_snapshot, '{}'::jsonb),
                                '{_temporal_gate_retries}',
                                '0'::jsonb,
                                true
                            ),
                            '{_temporal_gate_max_retries_overridden_at}',
                            to_jsonb(%s::text),
                            true
                        ),
                        updated_at = NOW()
                    WHERE meal_plan_id = %s
                      AND week_number = %s
                      AND status IN ('pending', 'processing', 'stale')
                    """,
                    (datetime.now(timezone.utc).isoformat(), meal_plan_id, int(week_number)),
                )
            except Exception as _p1c_persist_err:
                logger.warning(
                    f"[P1-C/MAX-RETRIES] No se pudo resetear contador en snapshot: {_p1c_persist_err}"
                )
            return {
                "ready": True,
                "reason": "temporal_gate_max_retries_exceeded",
                "ratio": None,
                "previous_chunk_start_day": prev_start_day,
                "previous_chunk_end_day": prev_end_day,
                "previous_chunk_start_iso": _prev_start_iso_log,
                "prev_end_date": _prev_end_date.isoformat(),
                "days_until_prev_end": _days_until_prev_end,
                "temporal_gate_retries": _p1c_next_retries,
                "forced_override": True,
            }

        # [P1-2/P1-3] Telemetría con manejo de fallos visible. El helper centraliza
        # el INSERT, resetea el contador de fallos al primer éxito tras una racha,
        # y promueve `logger.debug` → `logger.error` cuando la tabla está caída
        # (antes los deferrals se perdían silenciosamente y _detect_chronic_deferrals
        # no podía detectar usuarios con TZ desalineada). No bloqueante: un fallo
        # aquí no impide el deferral del chunk en sí.
        _record_chunk_deferral(
            user_id=user_id,
            meal_plan_id=meal_plan_id,
            week_number=int(week_number),
            reason="temporal_gate",
            days_until_prev_end=int(_days_until_prev_end),
        )

        # [P1-3] Push notification proactiva al N-th deferral consecutivo del mismo
        # chunk. Antes el único canal de aviso al usuario era `_detect_chronic_deferrals`
        # con umbral 5/48h y cron de 6h — un chunk con TZ desalineada podía agotar
        # CHUNK_TEMPORAL_GATE_MAX_RETRIES (ahora 5) y forzar ready=True ANTES de que
        # el detector cross-window lo notificara. Resultado: el usuario veía su plan
        # generándose con datos posiblemente incorrectos sin oportunidad de corregir
        # su TZ a tiempo. Ahora: en el N-th deferral (default 3 ≈ ~3 min), enviamos
        # push directo. Dedupe por (user_id, meal_plan_id, week_number) vía
        # system_alerts con cooldown configurable para evitar spam si el usuario
        # ignora la primera notificación.
        if _p1c_next_retries == int(CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY):
            _p13_alert_key = f"temporal_gate_proactive:{user_id}:{meal_plan_id}:{int(week_number)}"
            try:
                _ensure_quality_alert_schema()
                _p13_existing = execute_sql_query(
                    """
                    SELECT triggered_at FROM system_alerts
                    WHERE alert_key = %s
                      AND triggered_at > NOW() - make_interval(hours => %s)
                    LIMIT 1
                    """,
                    (_p13_alert_key, int(CHUNK_TEMPORAL_GATE_PUSH_COOLDOWN_HOURS)),
                    fetch_one=True,
                )
            except Exception as _p13_dedup_err:
                logger.debug(
                    f"[P1-3/PROACTIVE-PUSH] Dedupe lookup falló (continuamos best-effort): "
                    f"{_p13_dedup_err}"
                )
                _p13_existing = None

            if not _p13_existing:
                try:
                    _dispatch_push_notification(
                        user_id=user_id,
                        title="Tu plan está esperando",
                        body=(
                            "Tu próximo bloque parece atrasado. Verifica que la "
                            "zona horaria de tu perfil sea correcta para que "
                            "podamos generarlo."
                        ),
                        url="/dashboard",
                    )
                    # Persistir alerta para dedupe en próximas evaluaciones del gate.
                    execute_sql_write(
                        """
                        INSERT INTO system_alerts
                            (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
                        VALUES (%s, 'temporal_gate_proactive', 'info', %s, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (alert_key) DO UPDATE
                        SET triggered_at = NOW(),
                            metadata = EXCLUDED.metadata,
                            affected_user_ids = EXCLUDED.affected_user_ids,
                            resolved_at = NULL
                        """,
                        (
                            _p13_alert_key,
                            "Aviso proactivo de deferrals consecutivos",
                            (
                                f"Plan {meal_plan_id} chunk {week_number} acumuló "
                                f"{_p1c_next_retries} deferrals consecutivos del temporal gate. "
                                f"Push enviado para que el usuario revise TZ."
                            ),
                            json.dumps({
                                "consecutive_retries": _p1c_next_retries,
                                "max_retries_cap": int(CHUNK_TEMPORAL_GATE_MAX_RETRIES),
                                "meal_plan_id": str(meal_plan_id),
                                "week_number": int(week_number),
                                "days_until_prev_end": int(_days_until_prev_end),
                            }),
                            json.dumps([user_id]),
                        ),
                    )
                    logger.info(
                        f"[P1-3/PROACTIVE-PUSH] Push enviado a {user_id} para chunk "
                        f"{week_number} plan {meal_plan_id} tras {_p1c_next_retries} deferrals."
                    )
                except Exception as _p13_push_err:
                    logger.warning(
                        f"[P1-3/PROACTIVE-PUSH] No se pudo enviar push proactivo "
                        f"para {user_id} chunk {week_number}: {_p13_push_err}. "
                        f"`_detect_chronic_deferrals` cubrirá el gap."
                    )

        # [P1-C → P1-4] Persistir el contador incrementado Y bumpear `execute_after`
        # con backoff exponencial. Antes el contador se persistía pero el chunk
        # quedaba con su `execute_after` original — el scheduler lo re-pickeaba en el
        # próximo tick (~1 min) y la "5 retries × 1 min ≈ 5 min de gracia" del
        # comentario era demasiado corto: TZ drifts >5 min escalaban el chunk a
        # `pending_user_action` antes de que el día previo realmente terminara en
        # el calendario local del usuario. Ahora cada deferral se backoff'ea con
        # `min(2^retry, BACKOFF_CAP_MINUTES)` minutos:
        #   retry 1 →   1 min
        #   retry 2 →   2 min
        #   retry 3 →   4 min
        #   retry 4 →   8 min
        #   retry 5 →  16 min
        #   retry ≥6 →  30 min (capped)
        # Combinado con MAX_RETRIES=8, da ~2h de gracia total antes del forced
        # override (vs ~5 min antes). El push proactivo en retry=5 (PUSH_AT_RETRY)
        # llega a ~31 min — ventana razonable para que el usuario revise su TZ.
        # WHERE incluye week_number y status IN (pending, processing, stale) para
        # no tocar chunks que ya fueron paused o cancelled por otra ruta.
        from constants import CHUNK_TEMPORAL_GATE_BACKOFF_CAP_MINUTES
        _p14_backoff_min = min(
            int(2 ** max(0, _p1c_next_retries - 1)),
            int(CHUNK_TEMPORAL_GATE_BACKOFF_CAP_MINUTES),
        )
        try:
            execute_sql_write(
                """
                UPDATE plan_chunk_queue
                SET pipeline_snapshot = jsonb_set(
                        COALESCE(pipeline_snapshot, '{}'::jsonb),
                        '{_temporal_gate_retries}',
                        to_jsonb(%s::int),
                        true
                    ),
                    execute_after = NOW() + make_interval(mins => %s),
                    updated_at = NOW()
                WHERE meal_plan_id = %s
                  AND week_number = %s
                  AND status IN ('pending', 'processing', 'stale')
                """,
                (_p1c_next_retries, _p14_backoff_min, meal_plan_id, int(week_number)),
            )
        except Exception as _p1c_inc_err:
            # Best-effort: si el UPDATE falla, el counter no avanza y eventualmente
            # `_detect_chronic_deferrals` alertará. No bloqueamos el deferral.
            logger.debug(
                f"[P1-C/P1-4] No se pudo persistir _temporal_gate_retries={_p1c_next_retries} "
                f"+ backoff={_p14_backoff_min}min: {_p1c_inc_err}"
            )

        return {
            "ready": False,
            "reason": "prev_chunk_day_not_yet_elapsed",
            "ratio": None,
            "previous_chunk_start_day": prev_start_day,
            "previous_chunk_end_day": prev_end_day,
            "previous_chunk_start_iso": _prev_start_iso_log,
            "prev_end_date": _prev_end_date.isoformat(),
            "days_until_prev_end": _days_until_prev_end,
            "temporal_gate_retries": _p1c_next_retries,
        }

    prev_start_iso = (plan_start_dt + timedelta(days=prev_offset)).isoformat()
    consumed_records = get_consumed_meals_since(user_id, prev_start_iso) or []
    
    # [P0-D] Evaluar actividad de inventario antes del cálculo para el proxy honesto
    activity = {}
    try:
        activity = get_inventory_activity_since(user_id, prev_start_iso)
    except Exception as e:
        logger.debug(f"[P0-NEW2] No se pudo medir actividad de inventario para {user_id}: {e}")
        
    consumption_mutations_count = int(activity.get("consumption_mutations_count") or 0)
    inventory_mutations = int(activity.get("mutations_count") or 0)

    ratio_info = _calculate_chunk_consumption_ratio(previous_chunk_days, consumed_records, consumption_mutations_count)
    ratio = ratio_info["ratio"]

    # [P0-1] zero_log_proxy=True significa que NO hubo logs reales del chunk previo.
    # El gate "técnicamente" pasa con ratio=1.0 por el proxy, pero no hay señal de aprendizaje.
    # Marcamos ready=False en este caso aunque el ratio sea 1.0, para que el worker pueda
    # decidir entre deferral/pausa/forzado con criterio explícito.
    is_zero_log = ratio_info.get("zero_log_proxy", False)
    ratio_ready = ratio >= CHUNK_LEARNING_READY_MIN_RATIO

    # [P0-D] Si no hay logs explícitos, mirar mutaciones del inventario como proxy de actividad.
    # Si el usuario tocó su nevera ≥ N veces desde el inicio del chunk previo (auto-deducción
    # al consumir, edición manual de cantidades, agregar items), asumimos que está siguiendo
    # el plan aunque no loguee. Mantenemos zero_log_proxy=True para que el worker sepa que
    # el aprendizaje será débil, pero ya no bloqueamos la generación silenciosamente.
    #
    # [P1-3] Fuente de verdad: `meal_plans.plan_data._consecutive_proxy_chunks`. Ver
    # `_read_proxy_counter` para precedencia y back-compat. Antes el contador vivía
    # solo en `pipeline_snapshot`; si el snapshot se reescribía completo (no via
    # jsonb_set) o si chunks paralelos leían snapshots desfasados, el cap consecutivo
    # se podía burlar.
    _consecutive_proxy_chunks = _read_proxy_counter(plan_data, snapshot, "_consecutive_proxy_chunks")
    # [P0-3] Contadores acumulativos que NO se resetean ante chunks fuertes — defensa contra
    # el patrón alternado [proxy, strong, proxy, strong, ...] que antes burlaba el cap consecutivo.
    _lifetime_proxy_chunks = _read_proxy_counter(plan_data, snapshot, "_lifetime_proxy_chunks")
    _lifetime_total_chunks = _read_proxy_counter(plan_data, snapshot, "_lifetime_total_chunks")

    inventory_proxy_used = False
    _signal_too_weak = is_zero_log or ratio_info.get("sparse_logging_proxy")
    if _signal_too_weak:
        if consumption_mutations_count >= CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS:
            from constants import (
                CHUNK_MAX_CONSECUTIVE_PROXY_LEARNING,
                CHUNK_MAX_LIFETIME_PROXY_RATIO,
                CHUNK_LIFETIME_PROXY_MIN_TOTAL,
            )
            # [P1-4] Respetar la preferencia del usuario antes de pausar por falta de logs.
            # Si user.logging_preference == 'auto_proxy', el usuario explícitamente optó por
            # confiar en el plan sin loguear comidas. En ese modo continuamos con señal débil
            # (inventory_proxy) en vez de pausar el chunk en pending_user_action.
            _p14_pref = "manual"
            try:
                from db_core import execute_sql_query as _p14_q
                _p14_row = _p14_q(
                    "SELECT logging_preference FROM user_profiles WHERE id = %s",
                    (user_id,),
                    fetch_one=True,
                )
                if _p14_row and _p14_row.get("logging_preference"):
                    _p14_pref = str(_p14_row["logging_preference"])
            except Exception as _p14_err:
                logger.debug(f"[P1-4] No se pudo leer logging_preference para {user_id}: {_p14_err}")

            if _consecutive_proxy_chunks >= CHUNK_MAX_CONSECUTIVE_PROXY_LEARNING:
                if _p14_pref != "auto_proxy":
                    return {"ready": False, "reason": "learning_proxy_exhausted"}
                logger.info(
                    f"[P1-4] {user_id} en auto_proxy: omitiendo pausa learning_proxy_exhausted "
                    f"(consec={_consecutive_proxy_chunks})."
                )
            # [P0-3] Pausa por logging crónico ausente: aunque los chunks consecutivos no
            # acumulen (por intercalar uno fuerte), si la mayoría histórica del plan vino de
            # proxy de inventario, el aprendizaje real es marginal y forzamos pausa.
            if _lifetime_total_chunks >= CHUNK_LIFETIME_PROXY_MIN_TOTAL:
                _lifetime_ratio = _lifetime_proxy_chunks / max(1, _lifetime_total_chunks)
                if _lifetime_ratio >= CHUNK_MAX_LIFETIME_PROXY_RATIO:
                    if _p14_pref != "auto_proxy":
                        return {"ready": False, "reason": "chronic_zero_logging"}
                    logger.info(
                        f"[P1-4] {user_id} en auto_proxy: omitiendo pausa chronic_zero_logging "
                        f"(lifetime_ratio={_lifetime_ratio:.0%})."
                    )
            inventory_proxy_used = True

    learning_signal_strength = "none"
    if inventory_proxy_used or _signal_too_weak:
        learning_signal_strength = "weak"
    elif ratio_ready:
        learning_signal_strength = "strong"

    return {
        "ready": (ratio_ready and not _signal_too_weak) or inventory_proxy_used,
        "ratio": ratio,
        "matched_meals": ratio_info["matched_meals"],
        "planned_meals": ratio_info["planned_meals"],
        "explicit_matched_meals": ratio_info.get("explicit_matched_meals", 0),
        "explicit_logged_meals": ratio_info.get("explicit_logged_meals", 0),
        "used_implicit_proxy": ratio_info.get("used_implicit_proxy", False),
        "sparse_logging_proxy": ratio_info.get("sparse_logging_proxy", False),
        "zero_log_proxy": is_zero_log,
        # [P1-8] True si zero-log Y zero mutations: bandera explícita para que el caller,
        # los dashboards y el rebuild de lecciones puedan distinguir "no hay evidencia
        # de adherencia" vs "ratio bajo medido". Cuando es True, el ratio retornado es 0.0
        # (no la fórmula 0.5 + mutations/total que asumía 50% por defecto).
        "zero_log_no_mutations": ratio_info.get("zero_log_no_mutations", False),
        "inventory_proxy_used": inventory_proxy_used,
        "inventory_mutations": inventory_mutations,
        "learning_signal_strength": learning_signal_strength,
        "previous_chunk_start_day": prev_start_day,
        "previous_chunk_end_day": prev_end_day,
        "previous_chunk_start_iso": prev_start_iso,
    }


def _calculate_learning_metrics(new_days: list, prior_meals: list, prior_days: list, rejected_names: list, allergy_keywords: list, fatigued_ingredients: list) -> dict:
    """[GAP F] Calcula métricas para validar que el aprendizaje continuo funciona.

    Devuelve:
      - learning_repeat_pct: % de prior_meals que reaparecen en new_days (idealmente 0-10%)
      - rejection_violations: # de nombres rechazados que reaparecen (debe ser 0)
      - allergy_violations: # de keywords de alergia en ingredientes de new_days (debe ser 0)
      - fatigued_violations: # de ingredientes fatigados que reaparecen (informativo)
      - sample_repeats: primeros 5 nombres que repitieron (para debug)
    """
    from constants import get_nutritional_category, normalize_ingredient_for_tracking

    def _norm(s: str) -> str:
        if not s:
            return ""
        try:
            return strip_accents(str(s).lower()).strip()
        except Exception:
            return str(s).lower().strip()

    def _normalize_fatigue_marker(value: str) -> tuple[str, str]:
        normalized = _norm(value)
        if not normalized:
            return "ingredient", ""
        if normalized.startswith("[") and "]" in normalized:
            prefix, remainder = normalized.split("]", 1)
            prefix_alpha = "".join(ch for ch in prefix if ch.isalpha())
            if prefix_alpha.startswith("categoria") or prefix_alpha.startswith("category"):
                return "category", remainder.strip()
        return "ingredient", normalized

    def _extract_bases_from_meal(meal: dict) -> list:
        bases = []
        if not isinstance(meal, dict):
            return bases
        for ing in meal.get("ingredients", []) or []:
            raw_ing = ""
            if isinstance(ing, dict):
                raw_ing = ing.get("name") or ing.get("display_string") or ""
            else:
                raw_ing = str(ing)
            base = normalize_ingredient_for_tracking(raw_ing)
            if base:
                bases.append(base)
        return bases

    prior_set = {_norm(m) for m in (prior_meals or []) if m}
    rejected_set = {_norm(m) for m in (rejected_names or []) if m}
    allergy_set = {_norm(k) for k in (allergy_keywords or []) if k and len(k) > 2}
    fatigued_ingredient_set = set()
    fatigued_category_set = set()
    for fatigue_marker in fatigued_ingredients or []:
        marker_kind, marker_value = _normalize_fatigue_marker(fatigue_marker)
        if not marker_value:
            continue
        if marker_kind == "category":
            fatigued_category_set.add(marker_value)
        else:
            fatigued_ingredient_set.add(marker_value)

    new_meal_names = []
    new_ingredients_blob = []
    new_ingredient_categories = set()
    prior_meal_bases = set()
    new_meal_bases = []
    repeated_base_samples = []
    for d in (new_days or []):
        if not isinstance(d, dict):
            continue
        for m in d.get("meals", []) or []:
            if not isinstance(m, dict):
                continue
            name = _norm(m.get("name", ""))
            if name:
                new_meal_names.append(name)
            meal_bases = _extract_bases_from_meal(m)
            if meal_bases:
                new_meal_bases.append(sorted(set(meal_bases)))
            for ing in m.get("ingredients", []) or []:
                raw_ing = ""
                if isinstance(ing, dict):
                    raw_ing = ing.get("name") or ing.get("display_string") or ""
                else:
                    raw_ing = str(ing)
                normalized_ing = _norm(raw_ing)
                if normalized_ing:
                    new_ingredients_blob.append(normalized_ing)
                normalized_base = normalize_ingredient_for_tracking(raw_ing)
                normalized_category = _norm(get_nutritional_category(normalized_base)) if normalized_base else ""
                if normalized_category:
                    new_ingredient_categories.add(normalized_category)

    for d in (prior_days or []):
        if not isinstance(d, dict):
            continue
        for m in d.get("meals", []) or []:
            prior_meal_bases.update(_extract_bases_from_meal(m))

    repeats = [n for n in new_meal_names if n in prior_set]
    rejection_hits = [n for n in new_meal_names if n in rejected_set]

    # [P1-6] Antes el matching de alergias y fatiga era substring (`k in ing`), lo que
    # producía falsos positivos como "ajo" matching "ajonjolí" o "uva" matching "guayaba".
    # Ahora tokenizamos cada ingrediente con \w+ (después de _norm que ya quitó acentos)
    # y hacemos set membership exacto. Para keywords multi-palabra ("frutos secos") un
    # match requiere que TODOS los tokens del keyword estén presentes en los tokens del
    # ingrediente — preserva la semántica AND original sin el problema de partial match.
    # Limitación conocida: "huevo" no matchea "huevos" (plural). Si esto causa falsos
    # negativos en producción, el fix correcto es normalizar ambos lados vía
    # normalize_ingredient_for_tracking aguas arriba (ya lo hace _extract_bases_from_meal).
    import re as _re_p16
    def _token_set(text: str) -> set:
        return set(_re_p16.findall(r"\w+", str(text or "").lower()))

    new_ingredient_token_sets = [_token_set(ing) for ing in new_ingredients_blob]

    def _kw_matches_any_ingredient(kw: str) -> bool:
        kw_tokens = _re_p16.findall(r"\w+", str(kw or "").lower())
        if not kw_tokens:
            return False
        return any(
            all(t in ing_tokens for t in kw_tokens)
            for ing_tokens in new_ingredient_token_sets
        )

    allergy_hits = [k for k in allergy_set if _kw_matches_any_ingredient(k)]
    fatigued_hits = [k for k in fatigued_ingredient_set if _kw_matches_any_ingredient(k)]
    fatigued_category_hits = [k for k in fatigued_category_set if k in new_ingredient_categories]
    all_fatigued_hits = fatigued_hits + fatigued_category_hits

    repeated_base_meals = 0
    for meal_name, bases in zip(new_meal_names, new_meal_bases):
        repeated_bases = [b for b in bases if b in prior_meal_bases]
        if repeated_bases:
            repeated_base_meals += 1
            if len(repeated_base_samples) < 5:
                repeated_base_samples.append({"meal": meal_name, "bases": repeated_bases[:3]})

    total_new = len(new_meal_names)
    repeat_pct = round((len(repeats) / total_new) * 100.0, 2) if total_new else 0.0
    ingredient_base_repeat_pct = round((repeated_base_meals / total_new) * 100.0, 2) if total_new else 0.0

    return {
        "total_new_meals": total_new,
        "learning_repeat_pct": repeat_pct,
        "ingredient_base_repeat_pct": ingredient_base_repeat_pct,
        "rejection_violations": len(rejection_hits),
        "allergy_violations": len(allergy_hits),
        "fatigued_violations": len(all_fatigued_hits),
        "sample_repeats": repeats[:5],
        "sample_repeated_bases": repeated_base_samples,
        "sample_rejection_hits": rejection_hits[:5],
        "sample_allergy_hits": allergy_hits[:5],
        "prior_meals_count": len(prior_set),
        "prior_meal_bases_count": len(prior_meal_bases),
        "rejected_count": len(rejected_set),
        "allergy_keywords_count": len(allergy_set),
    }


def _build_filtered_edge_recipe_day(
    allergies: list | tuple | None,
    dislikes: list | tuple | None,
    diet: str = "",
    pantry_items: list | None = None,
) -> dict | None:
    """Construye un Edge Recipe usando solo ingredientes permitidos para el usuario.

    [P0-C FIX] Si se pasa pantry_items, intersecta cada categoría del catálogo con los
    ingredientes disponibles en la nevera. Si la intersección está vacía para una categoría,
    usa el catálogo completo filtrado como fallback para no bloquear la generación.
    """
    from constants import _get_fast_filtered_catalogs, normalize_ingredient_for_tracking

    allergies = tuple(a for a in (allergies or []) if isinstance(a, str) and a.strip())
    dislikes = tuple(d for d in (dislikes or []) if isinstance(d, str) and d.strip())
    filtered_proteins, filtered_carbs, filtered_veggies, _ = _get_fast_filtered_catalogs(
        allergies,
        dislikes,
        (diet or "").strip().lower(),
    )

    if not filtered_proteins or not filtered_carbs or not filtered_veggies:
        return None

    # [P0-C FIX] Preferir ingredientes disponibles en la nevera cuando se proporciona.
    if pantry_items:
        pantry_bases: set[str] = set()
        for item in pantry_items:
            if not item:
                continue
            normalized = normalize_ingredient_for_tracking(str(item))
            if normalized:
                pantry_bases.add(normalized)
            pantry_bases.add(strip_accents(str(item).lower().strip()))

        def _pantry_intersect(catalog: list) -> list:
            """Keep only catalog items whose normalized name matches a pantry base."""
            matched = [
                c for c in catalog
                if any(
                    (pb and len(pb) > 2 and (pb in strip_accents(c.lower()) or strip_accents(c.lower()) in pb))
                    for pb in pantry_bases
                )
            ]
            return matched if matched else catalog  # fallback: use full filtered catalog

        pantry_proteins = _pantry_intersect(filtered_proteins)
        pantry_carbs = _pantry_intersect(filtered_carbs)
        pantry_veggies = _pantry_intersect(filtered_veggies)

        _any_narrowed = (
            len(pantry_proteins) < len(filtered_proteins)
            or len(pantry_carbs) < len(filtered_carbs)
            or len(pantry_veggies) < len(filtered_veggies)
        )
        if not _any_narrowed:
            logger.debug("[P0-C] Edge Recipe: ninguna categoría intersectó con la nevera, usando catálogo completo.")
    else:
        pantry_proteins, pantry_carbs, pantry_veggies = filtered_proteins, filtered_carbs, filtered_veggies

    def _cap_ingredient(ingredient: str, default_g: int) -> str:
        if not pantry_items:
            return f"{default_g}g {ingredient}"
        from constants import normalize_ingredient_for_tracking
        norm = normalize_ingredient_for_tracking(ingredient)
        if not norm:
            return f"{default_g}g {ingredient}"
            
        total_g = 0.0
        try:
            from shopping_calculator import _parse_quantity
            from constants import _to_base_unit
            for p in pantry_items:
                p_norm = normalize_ingredient_for_tracking(str(p))
                if p_norm == norm:
                    q, u, _ = _parse_quantity(str(p))
                    bq, bu = _to_base_unit(q, u)
                    if bu == 'g' or bu == 'ml':
                        total_g += bq
        except Exception as e:
            logger.warning(f"[P0-NEW1] Error parsing quantity in Edge Recipe builder: {e}")

        if total_g > 0:
            capped_g = min(default_g, int(total_g))
            return f"{capped_g}g {ingredient}"
        return f"{default_g}g {ingredient}"

    breakfast_protein = _cap_ingredient(random.choice(pantry_proteins), 150)
    breakfast_carb = _cap_ingredient(random.choice(pantry_carbs), 100)
    lunch_protein = _cap_ingredient(random.choice(pantry_proteins), 200)
    lunch_carb = _cap_ingredient(random.choice(pantry_carbs), 150)
    lunch_veggie = _cap_ingredient(random.choice(pantry_veggies), 100)
    dinner_protein = _cap_ingredient(random.choice(pantry_proteins), 150)
    dinner_veggie = _cap_ingredient(random.choice(pantry_veggies), 150)

    return {
        "day": 0,
        "day_name": "",
        "meals": [
            {
                "name": f"Desayuno: {breakfast_protein} con {breakfast_carb}",
                "type": "Desayuno",
                "description": "Desayuno tradicional (Edge Recipe)",
                "ingredients": [breakfast_protein, breakfast_carb],
                "macros": {"calories": 400, "protein": 20, "carbs": 35, "fat": 15},
                "instructions": ["Preparar ingredientes según método tradicional"],
            },
            {
                "name": f"Almuerzo: {lunch_protein} con {lunch_carb} y {lunch_veggie}",
                "type": "Almuerzo",
                "description": "Almuerzo tradicional (Edge Recipe)",
                "ingredients": [lunch_protein, lunch_carb, lunch_veggie],
                "macros": {"calories": 500, "protein": 35, "carbs": 60, "fat": 10},
                "instructions": ["Cocinar a la plancha o al vapor"],
            },
            {
                "name": f"Cena: {dinner_protein} con {dinner_veggie}",
                "type": "Cena",
                "description": "Cena ligera (Edge Recipe)",
                "ingredients": [dinner_protein, dinner_veggie],
                "macros": {"calories": 350, "protein": 25, "carbs": 20, "fat": 15},
                "instructions": ["Saltear ingredientes juntos"],
            },
        ],
    }


def _detect_and_escalate_stuck_chunks():
    """[P1-3] Detecta chunks atrasados (lag > 24h sin pickup) y los escala.

    Acciones:
      - Loguea métricas [P1-3/STUCK-DETECT].
      - Si lag > 24h: marca escalated_at, baja execute_after a NOW() para que el
        worker los tome en el siguiente tick.
      - Idempotencia: si ya escalado en los últimos 30 min, skip.
      - Notificación: si lag > 4h (siempre cierto si lag > 24h), envía push suave
        para avisar que el plan está demorado (solo una vez por ventana de escalado).
      - Si lag > 72h y attempts >= 3: marca como 'failed' y envía push de rescate fallido.
    """
    try:
        # 1. Detectar stuck (>24h sin pickup) que aún NO fueron escalados o lo fueron hace > 30 min
        stuck_rows = execute_sql_query(
            """
            SELECT id, user_id, meal_plan_id, week_number, attempts,
                   EXTRACT(EPOCH FROM (NOW() - execute_after))::int AS lag_seconds,
                   GREATEST(
                       0,
                       EXTRACT(EPOCH FROM (NOW() - execute_after))::int - COALESCE(expected_preemption_seconds, 0)
                   )::int AS effective_lag_seconds,
                   escalated_at
            FROM plan_chunk_queue
            WHERE status IN ('pending', 'stale')
              AND GREATEST(
                    0,
                    EXTRACT(EPOCH FROM (NOW() - execute_after))::int - COALESCE(expected_preemption_seconds, 0)
                  ) > 86400
              AND (escalated_at IS NULL OR escalated_at < NOW() - INTERVAL '30 minutes')
            ORDER BY execute_after ASC
            LIMIT 50
            """,
            fetch_all=True,
        ) or []

        if stuck_rows:
            logger.warning(f" [P1-3/STUCK-DETECT] {len(stuck_rows)} chunks stuck (effective_lag>24h) detectados. Escalando...")
            
            # Notificación push proactiva (deduplicada por usuario)
            notified_users = set()
            
            for r in stuck_rows:
                lag_s = r.get('effective_lag_seconds') or 0
                lag_h = round(lag_s / 3600.0, 1)
                effective_lag_h = round((r.get('effective_lag_seconds') or 0) / 3600.0, 1)
                
                _already_escalated = r.get('escalated_at') is not None
                _tag = "[RE-ESCALANDO]" if _already_escalated else "[ESCALANDO]"
                
                logger.warning(
                    f"   ↳ {_tag} chunk {r['id']} plan={r['meal_plan_id']} week={r['week_number']} "
                    f"lag={lag_h}h effective_lag={effective_lag_h}h attempts={r.get('attempts', 0)}"
                )

                # Push guard: solo emitir si lag > 4h y es la primera escalación o ha pasado tiempo suficiente
                # (la idempotencia de 30 min ya filtra el grueso del spam)
                if lag_s > 4 * 3600:
                    uid = str(r['user_id'])
                    if uid not in notified_users:
                        notified_users.add(uid)
                        try:
                            import threading
                            from utils_push import send_push_notification
                            threading.Thread(
                                target=send_push_notification,
                                kwargs={
                                    "user_id": uid,
                                    "title": "⚡ Optimizando tu plan",
                                    "body": "Estamos terminando de ajustar los últimos detalles de tu plan. Estará listo en breve.",
                                    "url": "/dashboard"
                                },
                                daemon=True
                            ).start()
                        except Exception as push_err:
                            logger.warning(f" [P1-3/STUCK-DETECT] Error enviando push proactivo a {uid}: {push_err}")

            execute_sql_write("""
                UPDATE plan_chunk_queue
                SET escalated_at = NOW(),
                    execute_after = NOW(),
                    updated_at = NOW()
                WHERE status IN ('pending', 'stale')
                  AND GREATEST(
                        0,
                        EXTRACT(EPOCH FROM (NOW() - execute_after))::int - COALESCE(expected_preemption_seconds, 0)
                      ) > 86400
                  AND (escalated_at IS NULL OR escalated_at < NOW() - INTERVAL '30 minutes')
            """)

        # 2. Detectar stuck terminal (>72h y ya intentado 3+ veces) → fail + notify
        terminal = execute_sql_query(
            """
            SELECT id, user_id, meal_plan_id, week_number
            FROM plan_chunk_queue
            WHERE status IN ('pending', 'stale')
              AND escalated_at < NOW() - INTERVAL '72 hours'
              AND COALESCE(attempts, 0) >= 3
            LIMIT 50
            """,
            fetch_all=True,
        ) or []

        if terminal:
            ids = [str(r['id']) for r in terminal]
            execute_sql_write(
                "UPDATE plan_chunk_queue SET status = 'failed', updated_at = NOW() WHERE id = ANY(%s::uuid[])",
                (ids,)
            )
            logger.error(f" [P1-3/STUCK-DETECT] {len(terminal)} chunks marcados como 'failed' tras 72h sin recuperación.")

            # Push de notificación por usuario afectado (deduplicado)
            notified_users = set()
            for r in terminal:
                uid = str(r['user_id'])
                if uid in notified_users:
                    continue
                notified_users.add(uid)
                try:
                    import threading
                    from utils_push import send_push_notification
                    threading.Thread(
                        target=send_push_notification,
                        kwargs={
                            "user_id": uid,
                            "title": "⚠️ Tu plan necesita atención",
                            "body": "No pudimos generar parte de tu plan a largo plazo. Entra a la app para regenerarlo.",
                            "url": "/dashboard"
                        },
                        daemon=True
                    ).start()
                except Exception as push_err:
                    logger.warning(f" [P1-3/STUCK-DETECT] Error enviando push de fallo terminal a {uid}: {push_err}")
    except Exception as e:
        logger.error(f" [P1-3/STUCK-DETECT] Error en _detect_and_escalate_stuck_chunks: {e}")


def _count_pending_chunks_due() -> int:
    """[P1-5] Cuenta chunks listos para ejecutar AHORA (status=pending/stale, execute_after <= NOW()).

    Útil para medir backlog antes/después de una corrida del worker. Si el delta es positivo
    (pending_after > pending_before) la cola está creciendo más rápido de lo que el worker la
    drena → backlog acumulándose; con duración cercana al intervalo es señal de overlap.
    """
    try:
        row = execute_sql_query(
            """
            SELECT COUNT(*)::int AS cnt
            FROM plan_chunk_queue
            WHERE status IN ('pending', 'stale') AND execute_after <= NOW()
            """,
            fetch_one=True,
        )
        return int((row or {}).get("cnt") or 0)
    except Exception as e:
        logger.debug(f"[P1-5/WORKER-METRICS] No se pudo contar pending: {e}")
        return -1


# [P1-5] Estado in-memory de la última corrida del worker. Usado por el endpoint
# admin /chunk-queue-health para devolver visibilidad sin tocar DB. NO persistente:
# se reinicia al reiniciar el proceso. Para historial usar plan_chunk_metrics.
_LAST_WORKER_RUN: dict = {
    "started_at": None,
    "duration_seconds": None,
    "pending_before": None,
    "pending_after": None,
    "overlap_warning": False,
    "interval_seconds": None,
}


def _emit_worker_run_metric(duration_s: float, pending_before: int, pending_after: int) -> None:
    """[P1-5] Log estructurado + warning de overlap del worker.

    Si la corrida tarda más del 80% del intervalo del scheduler, emite un WARNING
    `worker_overlapping`. Esto detecta backlog antes de que el siguiente tick coalesa
    sobre uno aún corriendo (APScheduler tiene `coalesce=True` para no encolar
    múltiples instancias, pero a costa de saltarse ticks — degrada la cadencia
    real del worker).
    """
    interval_s = CHUNK_SCHEDULER_INTERVAL_MINUTES * 60
    overlap_threshold = interval_s * 0.8
    delta = (pending_after - pending_before) if (pending_before >= 0 and pending_after >= 0) else None

    _LAST_WORKER_RUN["started_at"] = datetime.now(timezone.utc).isoformat()
    _LAST_WORKER_RUN["duration_seconds"] = round(duration_s, 3)
    _LAST_WORKER_RUN["pending_before"] = pending_before
    _LAST_WORKER_RUN["pending_after"] = pending_after
    _LAST_WORKER_RUN["overlap_warning"] = duration_s > overlap_threshold
    _LAST_WORKER_RUN["interval_seconds"] = interval_s

    if duration_s > overlap_threshold:
        logger.warning(
            f"[P1-5/WORKER-OVERLAP] worker_overlapping "
            f"duration_s={duration_s:.2f} interval_s={interval_s} "
            f"pending_before={pending_before} pending_after={pending_after} delta={delta}"
        )
    else:
        logger.info(
            f"[P1-5/WORKER-METRICS] chunk_queue_run_complete "
            f"duration_s={duration_s:.2f} interval_s={interval_s} "
            f"pending_before={pending_before} pending_after={pending_after} delta={delta}"
        )


def _with_worker_metrics(fn):
    """[P1-5] Decorador que mide duración y backlog alrededor de cada corrida del worker."""
    def wrapper(target_plan_id=None):
        # Si target_plan_id está set, es ejecución manual (admin) — no medimos backlog
        # para no contaminar las métricas del cron real.
        if target_plan_id is not None:
            return fn(target_plan_id)

        import time as _t
        _start_ts = _t.monotonic()
        _pending_before = _count_pending_chunks_due()
        try:
            return fn(target_plan_id)
        finally:
            try:
                _duration = _t.monotonic() - _start_ts
                _pending_after = _count_pending_chunks_due()
                _emit_worker_run_metric(_duration, _pending_before, _pending_after)
            except Exception as metric_err:
                logger.debug(f"[P1-5/WORKER-METRICS] Error emitiendo métrica: {metric_err}")
    wrapper.__wrapped__ = fn  # acceso al impl para tests
    wrapper.__doc__ = fn.__doc__
    wrapper.__name__ = fn.__name__
    return wrapper


def _cleanup_orphan_chunks() -> int:
    """[P2-3] Cancela chunks vivos cuyo `meal_plan_id` ya no existe en `meal_plans`.

    Antes esta lógica vivía embebida en `process_plan_chunk_queue` (corre cada
    `CHUNK_SCHEDULER_INTERVAL_MINUTES`, default 1 min) y se ejecutaba al inicio
    de cada tick. Si la query de orphan-detection era lenta o fallaba, bloqueaba
    el hot path del worker. Ahora corre en su propio cron job
    (`cleanup_orphan_chunks`, cada `CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES`,
    default 5 min).

    Pasos:
      1. SELECT chunks vivos (pending/stale/processing) con meal_plan_id NOT IN
         meal_plans (plan eliminado o nunca existió por inserción fallida).
      2. Liberar reservas de inventario de cada chunk para evitar phantom
         reserved_quantity en `user_inventory`.
      3. UPDATE atómico marcando los chunks como 'cancelled'.

    Returns: número de chunks cancelados (0 si no había huérfanos o si hubo
    excepción). El error se loguea pero no se propaga, ya que es un job de
    mantenimiento que no debe romper el scheduler global.
    """
    try:
        orphan_chunks = execute_sql_query(
            """
            SELECT id, user_id FROM plan_chunk_queue
            WHERE status IN ('pending', 'stale', 'processing')
            AND meal_plan_id::text NOT IN (SELECT id::text FROM meal_plans)
            """,
            fetch_all=True,
        )
        orphan_list = list(orphan_chunks or [])
        for oc in orphan_list:
            try:
                release_chunk_reservations(str(oc["user_id"]), str(oc["id"]))
            except Exception as rel_err:
                logger.warning(
                    f"[P2-3] No se pudo liberar reservas de chunk huérfano {oc.get('id')}: {rel_err}"
                )
        if orphan_list:
            execute_sql_write("""
                UPDATE plan_chunk_queue SET status = 'cancelled', updated_at = NOW()
                WHERE status IN ('pending', 'stale', 'processing')
                AND meal_plan_id::text NOT IN (SELECT id::text FROM meal_plans)
            """)
            logger.info(
                f"[P2-3/ORPHAN-CLEANUP] Cancelados {len(orphan_list)} chunks huérfanos."
            )
        return len(orphan_list)
    except Exception as e:
        logger.warning(f"[P2-3/ORPHAN-CLEANUP] Error limpiando chunks huérfanos: {e}")
        return 0


@_with_worker_metrics
def process_plan_chunk_queue(target_plan_id=None):
    """Worker que genera las semanas 2-N de planes de largo plazo. Corre cada minuto vía APScheduler.

    [P1-8] Esta función es el corazón del state machine de `plan_chunk_queue`. La doc
    siguiente fija las transiciones canónicas para que operadores y futuros refactors
    no tengan que reverse-engineer-las del código.

    --------------------------------------------------------------------------
    ESTADOS Y TRANSICIONES de `plan_chunk_queue.status`
    --------------------------------------------------------------------------

    Estados (7 + sub-estados de pending_user_action):

      ┌─────────────────────────────────────────────────────────────────┐
      │ pending          → ejecutará en `execute_after`. Cola activa.   │
      │ processing       → worker generándolo en este momento.          │
      │ stale            → idéntico a pending pero re-encolado tras     │
      │                    `_invalidate_stale_chunks` (cambio crítico   │
      │                    de perfil que invalida el snapshot).         │
      │ pending_user_action → pausado a la espera de algo del usuario   │
      │                    (ver "Pause reasons" abajo). NO terminal.    │
      │ completed        → ✅ TERMINAL. Días mergeados al plan; shopping │
      │                    list recalculada; pipeline_snapshot purgado. │
      │ failed           → ❌ TERMINAL (transitorio: catchup sweep      │
      │                    `_recover_failed_chunks_for_long_plans` lo   │
      │                    re-encola si plan aún vigente).              │
      │ cancelled        → 🗑 TERMINAL DURO. Plan deleted o conflicto   │
      │                    irreparable. Purgado tras 48h.               │
      └─────────────────────────────────────────────────────────────────┘

    Transiciones (caller → status_destino):

       (nuevo)             ─┐
       failed (catchup)    ─┴─► pending      via _enqueue_plan_chunk
                                               (UPSERT atomic; P0-1)

       pending  ──► processing                via SELECT FOR UPDATE SKIP LOCKED
                                               (filtros: meal_plan_id, user_id,
                                                reservation_status; P0-4)

       processing ──► completed               via merge + shopping list OK

       processing ──► pending_user_action     via _pause_chunk_for_*
                                               (pantry vacío, stale snapshot,
                                                zero logs, missing lessons)

       processing ──► pending                 con backoff exponencial
                                               (P1-5: 2→4→8→16→32 min)

       processing ──► failed                  tras attempts >= 5
                                               (CHUNK_MAX_FAILURE_ATTEMPTS)

       processing ──► pending                 zombie rescue: heartbeat>10min
                                               sin updated_at touch

       pending_user_action ──► pending        via _recover_pantry_paused_chunks
                                               (live recovered / inventory
                                                proxy detected / TTL escalado
                                                a flexible_mode)

       pending|stale|processing ──► cancelled orphan cleanup (plan deleted)

       cancelled (>48h) ──► (DELETE)          GC nocturno

    --------------------------------------------------------------------------
    PAUSE REASONS (snap._pantry_pause_reason cuando status='pending_user_action')
    --------------------------------------------------------------------------

      reason                              | TTL pausa     | Resolución típica
      ------------------------------------|---------------|------------------------------
      empty_pantry                        | 12h           | usuario sube items o flex_mode
      stale_snapshot                      |  4h           | live recupera o flex_mode
      stale_snapshot_live_unreachable     | 24h*          | flex+advisory_only (P0-2)
      tz_major_drift_live_unreachable     |  4h           | usuario refresca nevera
      learning_zero_logs                  |  6h           | inventory proxy o force_flex
      missing_prior_lessons               | indefinido    | revisión humana (P1-1 falló)
      inventory_live_degraded             | (bypass)      | flex+advisory_only directo (P0-2)

      * Bypass automático: si `_is_inventory_live_degraded()` detecta caída sistémica,
        saltamos las ventanas y vamos directo a flex+advisory_only sin esperar TTL.

    --------------------------------------------------------------------------
    JOBS DE HOUSEKEEPING ejecutados ANTES de la lógica principal cada tick:
    --------------------------------------------------------------------------
      1. _process_pending_shopping_lists()    — recupera shopping list de planes
                                                 que completaron sin lista (GAP F).
      2. _recover_pantry_paused_chunks()      — desbloquea pending_user_action si
                                                 condición se resolvió (live OK,
                                                 inventory proxy, TTL → flex_mode).
      3. Cleanup chunks huérfanos             — meal_plan_id no existe → cancelled.
      4. GC pipeline_snapshot                 — vacía blob >10MB en cancelled/
                                                 completed-llm para liberar memoria.
      5. Purga de cancelled >48h              — DELETE permanente.
      6. Liberación de chunk_user_locks       — heartbeat >LOCK_STALE_MINUTES.
      7. Zombie rescue                        — processing con updated_at >10min
                                                 → pending (o failed si attempts>=5).

    --------------------------------------------------------------------------
    GUARANTEES de idempotencia y consistencia:
    --------------------------------------------------------------------------
      - Unique partial index (meal_plan_id, week_number) WHERE status IN ('pending',
        'processing', 'stale', 'failed') garantiza que no haya duplicados vivos.
      - Re-enqueue tras failed es atómico vía UPSERT con CAS sobre `status='failed'`
        (P0-1). Si dos sweeps concurrentes intentan reactivar, solo uno gana.
      - Merge de days al plan_data es idempotente vía `_merged_chunk_ids` marker:
        re-procesar un chunk completed no duplica días.
      - CAS-with-retry en `reserve_plan_ingredients` (P0-4) protege contra lost-
        update entre reservas concurrentes en la misma user_inventory row.

    --------------------------------------------------------------------------
    Args:
      target_plan_id: si se pasa, procesa SOLO chunks de ese plan_id (uso manual
                      vía /admin endpoints). Sin él, escanea hasta 3 chunks de
                      distintos (user_id, meal_plan_id) por tick.
    """
    # [P0-1-RECOVERY/WORKER-FIX] `json` ya es global del módulo.

    _process_pending_shopping_lists()
    _recover_pantry_paused_chunks()

    # [P2-3] El cleanup de chunks huérfanos se extrajo a `_cleanup_orphan_chunks`
    # como cron job dedicado (`cleanup_orphan_chunks`, cada
    # CHUNK_ORPHAN_CLEANUP_INTERVAL_MINUTES). Antes corría inline aquí en cada
    # tick del worker (1 min) y un timeout en su query bloqueaba el hot path del
    # processor. Ahora son crons independientes.

    # [GAP 7 FIX: Garbage collection eager de pipeline_snapshots]
    # Libera memoria masiva (~10MB por chunk) inmediatamente luego de que un chunk termina o se cancela.
    # [GAP C] Preservar snapshots de chunks degradados por 48h para permitir /regen-degraded.
    try:
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET pipeline_snapshot = '{}'::jsonb, updated_at = NOW()
            WHERE pipeline_snapshot::text != '{}'
            AND (
                status = 'cancelled'
                OR (status = 'completed' AND quality_tier = 'llm')
                OR (status = 'completed' AND quality_tier IN ('shuffle', 'edge', 'emergency')
                    AND updated_at < NOW() - INTERVAL '48 hours')
            )
        """)
    except Exception as e:
        logger.warning(f" [CHUNK] Error limpiando snapshots pesados: {e}")

    # [GAP 11 FIX: Purga definitiva de chunks cancelados > 48h]
    try:
        execute_sql_write("""
            DELETE FROM plan_chunk_queue
            WHERE status = 'cancelled'
            AND updated_at < NOW() - INTERVAL '48 hours'
        """)
    except Exception as e:
        logger.warning(f" [CHUNK] Error purgando chunks cancelados: {e}")

    # [P0-4] Housekeeping de Locks Zombies — ahora basado en heartbeat_at en vez de locked_at.
    # El worker actualiza heartbeat_at cada CHUNK_LOCK_HEARTBEAT_INTERVAL_SECONDS mientras procesa.
    # Si el heartbeat queda obsoleto > CHUNK_LOCK_STALE_MINUTES, asumimos worker crasheado
    # y liberamos. Esto distingue jobs largos vivos (heartbeat fresco) de huérfanos.
    try:
        from constants import CHUNK_LOCK_STALE_MINUTES as _LOCK_STALE_MIN
        execute_sql_write(
            """
            DELETE FROM chunk_user_locks
            WHERE heartbeat_at < NOW() - make_interval(mins => %s)
            """,
            (_LOCK_STALE_MIN,)
        )
    except Exception as e:
        logger.warning(f" [CHUNK] Error limpiando locks zombies: {e}")

    # Rescate de zombies: chunks procesando por más de 10 minutos
    try:
        execute_sql_write("""
            UPDATE plan_chunk_queue
            SET attempts = COALESCE(attempts, 0) + 1,
                status = CASE WHEN COALESCE(attempts, 0) + 1 >= 5 THEN 'failed' ELSE 'pending' END,
                execute_after = NOW() + make_interval(mins => 5),
                updated_at = NOW()
            WHERE status = 'processing'
            AND updated_at < NOW() - INTERVAL '10 minutes'
        """)
    except Exception as e:
        logger.warning(f" [CHUNK] Error rescatando zombies: {e}")

    # [P0-5/PRE-PICKUP] TZ drift sync ANTES del SELECT FOR UPDATE de pickup.
    #
    # Antes: el TZ sync vivía SOLO en (a) un cron separado cada 15 min, (b) el
    # gate `_check_chunk_learning_ready` AFTER pickup. El cron 15min dejaba
    # ventana TOCTOU de hasta 14 min entre cambio de TZ del usuario y un nuevo
    # pickup; el gate-AFTER-pickup detectaba drift pero el chunk ya tenía
    # status='processing' con el snapshot stale en memoria del worker (resync
    # del gate updateaba la DB pero no el dict ya cargado). Resultado: chunks
    # disparaban con TZ vieja, gate de adherencia mal-calculado, learning ruido.
    #
    # Ahora: cada tick (1 min), corremos `_sync_chunk_queue_tz_offsets()` ANTES
    # de la pickup query. Cualquier chunk con drift >= CHUNK_TZ_DRIFT_THRESHOLD_
    # MINUTES (15 min default) se actualiza en DB pre-pickup; el subsiguiente
    # SELECT FOR UPDATE lo promociona a 'processing' CON la TZ fresca leída del
    # snapshot ya actualizado. Cierra la ventana TOCTOU de pickup.
    #
    # El cron dedicado de 15min sigue activo como red de seguridad para casos
    # donde process_plan_chunk_queue esté pausado (apscheduler issue, deploy).
    # El gate-resync sigue activo como última defensa para chunks que YA están
    # processing y para casos donde el drift cambia entre pickup y el LLM call.
    #
    # Costo: 1 SELECT (LIMIT 500, JOIN user_profiles) + N UPDATEs (N=chunks con
    # drift, típicamente 0). Acotado por LIMIT y selectivo (skip si drift <
    # threshold). Best-effort: si falla, logueamos warning y dejamos los crons
    # de respaldo encargarse.
    try:
        _p05_synced = _sync_chunk_queue_tz_offsets(target_user_id=None)
        if _p05_synced > 0:
            logger.info(
                f"⏱️  [P0-5/PRE-PICKUP] TZ sync ANTES del pickup actualizó "
                f"{_p05_synced} chunk(s) con drift."
            )
    except Exception as _p05_pre_err:
        logger.warning(
            f"[P0-5/PRE-PICKUP] TZ sync pre-pickup falló (best-effort, los crons "
            f"de respaldo cubren): {type(_p05_pre_err).__name__}: {_p05_pre_err}"
        )

    # [GAP B FIX: Serializar chunks por meal_plan_id y procesar en orden secuencial]
    # [GAP A] Capturamos lag_seconds_at_pickup en el mismo UPDATE para tener métrica de SLA.
    if target_plan_id:
        query = """
            UPDATE plan_chunk_queue
            SET status = 'processing',
                updated_at = NOW(),
                lag_seconds_at_pickup = EXTRACT(EPOCH FROM (NOW() - execute_after))::int,
                effective_lag_seconds_at_pickup = GREATEST(
                    0,
                    EXTRACT(EPOCH FROM (NOW() - execute_after))::int - COALESCE(expected_preemption_seconds, 0)
                )::int
            WHERE id IN (
                SELECT q1.id FROM plan_chunk_queue q1
                WHERE q1.status IN ('pending', 'stale')
                AND q1.meal_plan_id = %s
                -- [P0-4] Defense-in-depth: aún cuando llegamos por target_plan_id,
                -- bloquear pickup si el MISMO usuario tiene OTRO chunk processing
                -- (e.g. usuario con 2 planes activos). La rama no-target ya hace
                -- esto; sin replicarlo aquí, dos planes del mismo user podían
                -- correr en paralelo y competir por la misma user_inventory row,
                -- abriendo la ventana que la CAS de _apply_reservation_delta
                -- mitiga pero no elimina (cada conflicto cuesta retries).
                AND q1.user_id NOT IN (
                    SELECT user_id FROM plan_chunk_queue WHERE status = 'processing'
                )
                -- [P0-4 FIX] Serializar pickups concurrentes por user_id A NIVEL SQL.
                -- El filtro `user_id NOT IN ('processing')` arriba evalúa la subquery
                -- contra el snapshot pre-UPDATE: dos pickups paralelos (cron threads
                -- distintos, mismo tick) no se ven entre sí y ambos pueden seleccionar
                -- chunks distintos del mismo usuario. La defensa downstream
                -- (chunk_user_locks INSERT en `_chunk_worker` ~L12716) detecta el
                -- conflicto y demote uno a 'pending', pero el bounce gasta ciclos:
                -- transición processing→pending→processing, log noise, métricas
                -- ruidosas, lag visible al usuario.
                --
                -- pg_try_advisory_xact_lock es transaccional: el lock se libera al
                -- COMMIT del UPDATE. Para una TX paralela tratando de tomar el mismo
                -- user_id, retorna FALSE → la fila se excluye limpiamente del WHERE.
                -- Una vez la primera TX commitea, la fila ya está en 'processing' y
                -- el filtro `user_id NOT IN` excluye al usuario en pickups
                -- subsecuentes (sin dependencia del lock).
                --
                -- Namespace 'chunk_pickup_user': distinto de los locks de
                -- `acquire_meal_plan_advisory_lock` (que usa 'meal_plan:...') para no
                -- colisionar entre paths ortogonales.
                --
                -- chunk_user_locks (DB table) sigue siendo defensa-en-profundidad
                -- contra paths que bypassean el pickup (recovery, sync API, etc.).
                AND pg_try_advisory_xact_lock(
                    hashtextextended('chunk_pickup_user:' || q1.user_id::text, 0)
                )
                AND q1.id = (
                    SELECT q2.id FROM plan_chunk_queue q2
                    WHERE q2.meal_plan_id = q1.meal_plan_id
                    AND q2.status IN ('pending', 'stale')
                    ORDER BY q2.week_number ASC
                    LIMIT 1
                )
                AND NOT EXISTS (
                    SELECT 1 FROM plan_chunk_queue q3
                    WHERE q3.meal_plan_id = q1.meal_plan_id AND q3.status = 'processing'
                )
                ORDER BY q1.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, user_id, meal_plan_id, week_number, chunk_kind, days_offset, days_count,
                      pipeline_snapshot, lag_seconds_at_pickup, effective_lag_seconds_at_pickup,
                      expected_preemption_seconds, escalated_at, attempts;
        """
        params = (target_plan_id,)
    else:
        query = """
            UPDATE plan_chunk_queue
            SET status = 'processing',
                updated_at = NOW(),
                lag_seconds_at_pickup = EXTRACT(EPOCH FROM (NOW() - execute_after))::int,
                effective_lag_seconds_at_pickup = GREATEST(
                    0,
                    EXTRACT(EPOCH FROM (NOW() - execute_after))::int - COALESCE(expected_preemption_seconds, 0)
                )::int
            WHERE id IN (
                SELECT q1.id FROM plan_chunk_queue q1
                WHERE q1.status IN ('pending', 'stale')
                AND q1.execute_after <= NOW()
                AND q1.meal_plan_id NOT IN (
                    SELECT meal_plan_id FROM plan_chunk_queue WHERE status = 'processing'
                )
                -- [P0-4] Serializar a 1 chunk por USUARIO por tick. Sin esto, si el usuario
                -- tiene 2 planes activos (migración, invitado, etc.), ambos chunks correrían
                -- en paralelo y competirían por get_user_inventory_net() sin lock, causando
                -- race conditions en el ledger de inventario. LIMIT 3 = 3 usuarios distintos.
                AND q1.user_id NOT IN (
                    SELECT user_id FROM plan_chunk_queue WHERE status = 'processing'
                )
                -- [P0-4 FIX] Ver comentario detallado en la rama target_plan_id arriba.
                -- pg_try_advisory_xact_lock per user_id cierra la race SELECT→UPDATE
                -- entre pickups paralelos (cron threads distintos, mismo tick), evitando
                -- el bounce processing→pending→processing que el filtro `user_id NOT IN`
                -- por sí solo no previene (la subquery ve snapshot pre-UPDATE).
                AND pg_try_advisory_xact_lock(
                    hashtextextended('chunk_pickup_user:' || q1.user_id::text, 0)
                )
                AND q1.meal_plan_id NOT IN (
                    SELECT meal_plan_id FROM plan_chunk_queue
                    WHERE reservation_status = 'partial'
                    AND updated_at > NOW() - INTERVAL '5 minutes'
                )
                AND q1.id = (
                    SELECT q2.id FROM plan_chunk_queue q2
                    WHERE q2.meal_plan_id = q1.meal_plan_id
                    AND q2.status IN ('pending', 'stale')
                    ORDER BY q2.week_number ASC
                    LIMIT 1
                )
                AND NOT EXISTS (
                    SELECT 1 FROM plan_chunk_queue q3
                    WHERE q3.meal_plan_id = q1.meal_plan_id AND q3.status = 'processing'
                )
                ORDER BY q1.created_at ASC
                FOR UPDATE SKIP LOCKED
                LIMIT 3
            )
            RETURNING id, user_id, meal_plan_id, week_number, chunk_kind, days_offset, days_count,
                      pipeline_snapshot, lag_seconds_at_pickup, effective_lag_seconds_at_pickup,
                      expected_preemption_seconds, escalated_at, attempts;
        """
        params = ()

    try:
        tasks = execute_sql_write(query, params, returning=True)
    except Exception as e:
        if "relation" not in str(e).lower():
            logger.error(f" [CHUNK] Error obteniendo tasks de plan_chunk_queue: {e}")
        return

    if not tasks:
        return

    # [GAP A] Resumen de SLA por batch: cuántos chunks tomados y con qué lag
    try:
        lags = [int(t.get("effective_lag_seconds_at_pickup") or 0) for t in tasks]
        if lags:
            max_lag_h = max(lags) / 3600.0
            avg_lag_h = (sum(lags) / len(lags)) / 3600.0
            escalated_count = sum(1 for t in tasks if t.get("escalated_at") is not None)
            if max_lag_h >= 1.0 or escalated_count > 0:
                logger.warning(
                    f"📊 [GAP A/SLA] Pickup batch: n={len(tasks)} avg_effective_lag={avg_lag_h:.1f}h "
                    f"max_effective_lag={max_lag_h:.1f}h escalated={escalated_count}"
                )
    except Exception:
        pass

    logger.info(f" [CHUNK] Procesando {len(tasks)} chunks de planes en background.")

    def _chunk_worker(task):
        task_id = task["id"]
        user_id = str(task["user_id"])
        meal_plan_id = str(task["meal_plan_id"])
        week_number = task["week_number"]
        days_offset = task["days_offset"]
        days_count = task["days_count"]
        lag_seconds = int(task.get("effective_lag_seconds_at_pickup") or 0)
        # [P0-6] CAS token: attempts al momento del pickup. El zombie rescue siempre lo incrementa,
        # lo que nos permite detectar si fuimos desplazados aun cuando el nuevo worker ya tomó el chunk.
        _pickup_attempts = int(task.get("attempts") or 0)
        # [GAP G] Métricas de observabilidad del chunk
        import time as _t
        chunk_start_ts = _t.time()
        # [GAP F] Defaults para que existan si no entramos al path LLM
        prior_meals = []
        rejected_meal_names = []
        _fatigued_ingredients = []
        _allergy_keywords = []
        learning_metrics = None

        snap = task["pipeline_snapshot"]
        if isinstance(snap, str):
            snap = json.loads(snap)

        chunk_kind = task.get("chunk_kind") or ("rolling_refill" if snap.get("_is_rolling_refill", False) else "initial_plan")
        is_rolling_refill = chunk_kind == "rolling_refill"
        form_data = copy.deepcopy(snap.get("form_data", {}))
        snapshot_form_data = snap.get("form_data", {}) or {}

        # [P0-5] Default-init names that are only conditionally bound deeper in the
        # function. Python's lexical scoping treats any later assignment as creating a
        # local, so a downstream read (e.g. the stub-lesson reconstruction at L16085,
        # which references `current_allergies` regardless of which generation path
        # ran) raised UnboundLocalError when the conditional path was skipped — for
        # instance when the LLM branch handled allergies via L14517 only if
        # `alergias_facts` was non-empty, and the Smart-Shuffle binding at L13485
        # was skipped because `is_degraded=False`.
        current_allergies: list = []

        # [P0-4] Race condition fix: Acquire user-level lock con heartbeat para distinguir
        # workers vivos de crasheados. INSERT con heartbeat_at = NOW(); un thread daemon refresca
        # heartbeat_at cada CHUNK_LOCK_HEARTBEAT_INTERVAL_SECONDS. El housekeeping considera
        # zombie cualquier lock con heartbeat_at más viejo que CHUNK_LOCK_STALE_MINUTES.
        _heartbeat_stop_event = None
        _heartbeat_thread = None
        try:
            user_lock = execute_sql_write("""
                INSERT INTO chunk_user_locks (user_id, locked_at, locked_by_chunk_id, heartbeat_at)
                VALUES (%s, NOW(), %s, NOW())
                ON CONFLICT (user_id) DO NOTHING
                RETURNING user_id;
            """, (user_id, task_id), returning=True)

            if not user_lock:
                logger.warning(f" [CHUNK] Usuario {user_id} lockeado. Deferring chunk {task_id}.")
                # [P1-4] CAS guard: si zombie rescue + nuevo pickup ya ocurrió, NO clobbear
                # el processing del worker B. Si CAS falla, el chunk ya está en otro estado.
                _cas_update_chunk_status(task_id, _pickup_attempts, "pending")
                return

            # Arranque del heartbeat. threading.Event permite cierre limpio en el `finally`.
            import threading as _threading
            from constants import (
                CHUNK_LOCK_HEARTBEAT_INTERVAL_SECONDS as _HB_INTERVAL,
                CHUNK_LOCK_STALE_MINUTES as _HB_STALE_MIN,
            )
            _heartbeat_stop_event = _threading.Event()

            # [P0-1] State compartido por flujo principal y thread:
            #   - last_heartbeat_at: timestamp UTC del último update OK (None si nunca tuvo éxito).
            #   - consecutive_failures: contador de fallos seguidos en el thread.
            # El flujo principal lee `last_heartbeat_at` para detectar threads colgados y
            # forzar un heartbeat inline antes de cruzar puntos críticos (merge/transacciones largas).
            _heartbeat_state = {
                "last_heartbeat_at": datetime.now(timezone.utc),  # INSERT puso heartbeat_at = NOW()
                "consecutive_failures": 0,
                "lock_chunk_id": task_id,
            }

            def _heartbeat_loop(stop_event, state):
                """[P0-1] Thread daemon que refresca heartbeat_at cada _HB_INTERVAL.

                Cambios vs. versión anterior:
                  1. Primer UPDATE inmediato (antes del primer `wait`) para no depender solo
                     del INSERT inicial — si el thread arranca con latencia >180s respecto del
                     INSERT, el zombie rescue procedería sin un solo refresh.
                  2. Counter de fallos consecutivos con logging escalado: el primer fallo y
                     cada N=3 después se loggean como ERROR (no debug) para que problemas
                     sistémicos (DB caída, schema roto) sean visibles. Reset al primer éxito.
                  3. try/except OUTER envolviendo todo el body — un crash del thread por
                     bug en logger/constants/etc no puede colgar el heartbeat sin trazas.
                """
                lock_chunk_id = state["lock_chunk_id"]

                def _do_update():
                    try:
                        execute_sql_write(
                            "UPDATE chunk_user_locks SET heartbeat_at = NOW() WHERE locked_by_chunk_id = %s",
                            (lock_chunk_id,)
                        )
                        state["last_heartbeat_at"] = datetime.now(timezone.utc)
                        if state["consecutive_failures"] > 0:
                            logger.info(
                                f"[P0-1/HEARTBEAT] Recovered tras {state['consecutive_failures']} "
                                f"fallo(s) consecutivos para chunk {lock_chunk_id}."
                            )
                            state["consecutive_failures"] = 0
                    except Exception as _hb_err:
                        state["consecutive_failures"] += 1
                        n = state["consecutive_failures"]
                        # _HB_STALE_MIN minutos / _HB_INTERVAL segundos = max ciclos antes de zombie.
                        max_cycles_before_stale = max(1, int((_HB_STALE_MIN * 60) // max(_HB_INTERVAL, 1)))
                        if n == 1 or n % 3 == 0 or n >= max_cycles_before_stale:
                            logger.error(
                                f"[P0-1/HEARTBEAT] Update FALLÓ #{n} para chunk {lock_chunk_id} "
                                f"(stale_threshold={_HB_STALE_MIN}min, interval={_HB_INTERVAL}s, "
                                f"max_cycles={max_cycles_before_stale}): "
                                f"{type(_hb_err).__name__}: {_hb_err}"
                            )
                        else:
                            logger.debug(
                                f"[P0-1/HEARTBEAT] Update fallido (#{n}) para chunk {lock_chunk_id}: {_hb_err}"
                            )

                try:
                    # [P0-1] Update inmediato al arrancar — cubre la ventana entre el INSERT
                    # del lock (heartbeat_at = NOW()) y el primer cycle del wait, que puede
                    # ser de hasta _HB_INTERVAL segundos.
                    _do_update()
                    while not stop_event.wait(_HB_INTERVAL):
                        _do_update()
                except Exception as _outer_err:
                    # Red de seguridad: si algo no-DB rompe el thread, dejamos rastro.
                    logger.error(
                        f"[P0-1/HEARTBEAT] Thread daemon murió inesperadamente para chunk "
                        f"{lock_chunk_id}: {type(_outer_err).__name__}: {_outer_err}"
                    )

            _heartbeat_thread = _threading.Thread(
                target=_heartbeat_loop,
                args=(_heartbeat_stop_event, _heartbeat_state),
                daemon=True,
                name=f"chunk-heartbeat-{task_id}",
            )
            _heartbeat_thread.start()
            # [P1-B] Sanity check + ABORT: antes solo logueábamos ERROR y continuábamos,
            # dejando al chunk vulnerable a zombie rescue tras _HB_STALE_MIN minutos
            # (en pleno LLM call → tokens perdidos + reintento desde cero). Ahora si el
            # thread no arrancó (límite de threads del SO, OOM transient), `_handle_heartbeat_start_failure`
            # libera lock + reservas y defiere el chunk para que un tick posterior lo
            # recoja con menos presión de threads.
            if not _heartbeat_thread.is_alive():
                _handle_heartbeat_start_failure(task_id, user_id)
                return
        except Exception as lock_err:
            logger.warning(f" [CHUNK] No se pudo adquirir lock para {user_id}: {lock_err}")

        try:
            # [GAP 3 FIX: GUARD validar plan activo y no-fallido]
            active_plan = execute_sql_query(
                "SELECT id, plan_data->>'generation_status' as status FROM meal_plans WHERE id = %s",
                (meal_plan_id,), fetch_one=True
            )
            if not active_plan:
                logger.info(f" [CHUNK] Plan {meal_plan_id} no existe. Cancelando chunk {week_number}.")
                release_chunk_reservations(user_id, str(task_id))
                execute_sql_write("UPDATE plan_chunk_queue SET status = 'cancelled' WHERE id = %s", (task_id,))
                return
                
            if active_plan.get('status') == 'failed':
                logger.info(f" [CHUNK] Plan {meal_plan_id} esta fallido. Cancelando chunk {week_number}.")
                release_chunk_reservations(user_id, str(task_id))
                execute_sql_write("UPDATE plan_chunk_queue SET status = 'cancelled' WHERE id = %s", (task_id,))
                return

            # [P0-3] Race guard: si el chunk N-1 todavía está pending o processing, diferir
            # este chunk N por 5 min. El user_lock (línea 6751) previene ejecución concurrente
            # de dos chunks del MISMO usuario en el mismo proceso, pero no cubre tres casos:
            #   1) Acquisition del user_lock falló por DB transitorio → el except de línea 6786
            #      cae al try principal sin lock → dos chunks consecutivos podrían correr en
            #      paralelo en workers distintos.
            #   2) Chunk N-1 commiteó plan_data pero el worker crasheó antes de marcar
            #      status='completed' en plan_chunk_queue → quedó en 'processing' colgado;
            #      housekeeping eventualmente lo resuelve, pero hasta entonces es ambiguo.
            #   3) Tabla chunk_user_locks no existe / fue dropeada → todo corre sin
            #      serialización.
            # El gate explícito a nivel de chunk hace el contrato visible: chunk N solo
            # arranca cuando chunk N-1 está en estado terminal ('completed', 'failed',
            # 'cancelled'). Con P0-2 la cadena de aprendizaje sobrevive si N-1 acabó en
            # 'failed' (preflight learning_metrics + auto-recovery), así que NO bloqueamos
            # en 'failed' — solo en estados activos. Best-effort: si la query falla,
            # continuar con el flujo previo en lugar de bloquear el chunk.
            if int(week_number) >= 2:
                try:
                    _prev_week_p03 = int(week_number) - 1
                    _prev_chunk_row = execute_sql_query(
                        "SELECT status FROM plan_chunk_queue "
                        "WHERE meal_plan_id = %s AND week_number = %s "
                        "ORDER BY created_at DESC LIMIT 1",
                        (meal_plan_id, _prev_week_p03), fetch_one=True
                    )
                    _prev_status = (_prev_chunk_row or {}).get("status")
                    if _prev_status in ("pending", "processing"):
                        logger.warning(
                            f"[P0-3/RACE-GUARD] Chunk {week_number} plan {meal_plan_id} "
                            f"difiere 5min: chunk previo {_prev_week_p03} aún en estado "
                            f"'{_prev_status}'. Evita leer plan_data antes de que "
                            f"N-1 commitee learning."
                        )
                        execute_sql_write(
                            "UPDATE plan_chunk_queue "
                            "SET status = 'pending', "
                            "    execute_after = NOW() + INTERVAL '5 minutes', "
                            "    updated_at = NOW() "
                            "WHERE id = %s",
                            (task_id,)
                        )
                        # Telemetría: usar el helper existente para que
                        # _detect_chronic_deferrals pueda detectar planes con N-1 colgado
                        # crónicamente (síntoma de housekeeping fallido).
                        try:
                            _record_chunk_deferral(
                                user_id=user_id,
                                meal_plan_id=meal_plan_id,
                                week_number=int(week_number),
                                reason="prev_chunk_in_flight",
                                days_until_prev_end=None,
                            )
                        except Exception as _tele_err:
                            logger.debug(
                                f"[P0-3/RACE-GUARD] Telemetría chunk_deferrals falló: {_tele_err}"
                            )
                        release_chunk_reservations(user_id, str(task_id))
                        return
                except Exception as _race_err:
                    # Best-effort: el race guard no debe bloquear chunks si la query falla.
                    # El user_lock + auto-recovery P0-2 son la red de seguridad subyacente.
                    logger.warning(
                        f"[P0-3/RACE-GUARD] Error consultando estado de chunk previo "
                        f"para plan {meal_plan_id} chunk {week_number}: "
                        f"{type(_race_err).__name__}: {_race_err}. Continuando."
                    )

            plan_row_prior = execute_sql_query(
                "SELECT plan_data FROM meal_plans WHERE id = %s",
                (meal_plan_id,), fetch_one=True
            )
            prior_plan_data = plan_row_prior.get("plan_data", {}) if plan_row_prior else {}
            # [P0-2] Sello CAS: capturar el timestamp ANTES del LLM call para poder
            # compararlo dentro del bloque FOR UPDATE y detectar modificaciones externas.
            pre_read_modified_at = prior_plan_data.get('_plan_modified_at') if prior_plan_data else None

            # [P0-3] Auto-recovery de `_last_chunk_learning` cuando el seed síncrono falló o
            # la persistencia post-chunk N-1 no escribió en plan_data (timeout, lock, JSON
            # corrupto). Sin esto, el chunk N consume un dict vacío y todas las "lecciones"
            # quedan en stub → cadena de aprendizaje rota silenciosamente. Reconstruimos
            # desde plan_chunk_queue.learning_metrics del chunk previo (la columna SÍ se
            # persiste atómicamente con el commit del chunk).
            if int(week_number) >= 2 and isinstance(prior_plan_data, dict):
                _p03_existing = prior_plan_data.get("_last_chunk_learning")
                _p03_target_week = int(week_number) - 1
                _p03_needs_rebuild = False
                if _is_lesson_stub(_p03_existing):
                    _p03_needs_rebuild = True
                    _p03_reason = "stub_or_empty"
                elif isinstance(_p03_existing, dict) and _p03_existing.get("chunk") != _p03_target_week:
                    # Lección persistida pero corresponde a otro chunk (desincronización).
                    _p03_needs_rebuild = True
                    _p03_reason = (
                        f"chunk_mismatch(persisted={_p03_existing.get('chunk')},"
                        f"expected={_p03_target_week})"
                    )
                else:
                    _p03_reason = None

                if _p03_needs_rebuild:
                    # [P1-2] prefer_completed=True: si el único registro de aprendizaje
                    # del chunk previo viene de un row 'failed' (pipeline crash o commit
                    # roto), preferimos sintetizar desde plan_data.days. Razón: las
                    # learning_metrics de un chunk failed pueden ser parciales o
                    # inconsistentes (commit a medio escribir) y se propagan como ruido
                    # a chunks posteriores. La síntesis lee los days realmente
                    # persistidos y deriva lecciones truthful (low_confidence pero
                    # correctas). El comportamiento legacy P0-2 sigue disponible vía
                    # prefer_completed=False para introspección manual.
                    _p03_rebuilt = _rebuild_last_chunk_learning_from_queue(
                        meal_plan_id, _p03_target_week, prefer_completed=True,
                        user_id=user_id,  # [P1-6] habilita telemetría learning loss
                    )
                    # [P1-2] Telemetría: si el rebuild devolvió None y existe un row
                    # 'failed' con learning_metrics no-NULL, registramos que descartamos
                    # un chunk fallido para no contaminar. Útil para alertar si el ratio
                    # de chunks failed-with-metrics crece (síntoma de commits inestables).
                    if _p03_rebuilt is None:
                        try:
                            _p12_failed_row = execute_sql_query(
                                """
                                SELECT id FROM plan_chunk_queue
                                WHERE meal_plan_id = %s AND week_number = %s
                                  AND status = 'failed'
                                  AND learning_metrics IS NOT NULL
                                LIMIT 1
                                """,
                                (str(meal_plan_id), int(_p03_target_week)),
                                fetch_one=True,
                            )
                            if _p12_failed_row:
                                _record_chunk_lesson_telemetry(
                                    user_id=user_id,
                                    meal_plan_id=meal_plan_id,
                                    week_number=int(week_number),
                                    event="failed_chunk_skipped_for_learning",
                                    synthesized_count=0,
                                    queue_count=0,
                                    metadata={
                                        "prev_week": int(_p03_target_week),
                                        "reason": "prefer_completed_strict",
                                    },
                                )
                                logger.info(
                                    f"[P1-2/SKIP-FAILED] chunk {week_number} plan {meal_plan_id}: "
                                    f"chunk previo (week {_p03_target_week}) está en status='failed'. "
                                    f"Descartando learning_metrics potencialmente incompletos; "
                                    f"caerá al fallback de síntesis desde plan_data.days."
                                )
                        except Exception as _p12_tele_err:
                            logger.debug(
                                f"[P1-2/SKIP-FAILED] Telemetría falló: {_p12_tele_err}"
                            )
                    if _p03_rebuilt:
                        # [P0.3] Centralizado vía `persist_legacy_learning_to_plan_data`.
                        # El helper aplica el sello CAS de `_plan_modified_at` y emite
                        # telemetría agregada por context — antes había duplicación SQL
                        # entre éste y el path de síntesis P0-4 más abajo.
                        _p03_persisted = persist_legacy_learning_to_plan_data(
                            meal_plan_id, _p03_rebuilt,
                            context="rebuild_from_queue",
                        )
                        # Siempre actualizamos in-memory: aunque la persistencia falle,
                        # mejor inyectar lecciones reales al chunk actual que arrancar con stub.
                        prior_plan_data["_last_chunk_learning"] = _p03_rebuilt
                        if _p03_persisted:
                            logger.warning(
                                f"[P0-3/AUTO-RECOVERED] _last_chunk_learning reconstruido para "
                                f"plan {meal_plan_id} chunk {week_number} desde plan_chunk_queue "
                                f"(prev_week={_p03_target_week}, reason={_p03_reason}, "
                                f"repeat_pct={_p03_rebuilt.get('repeat_pct')}%, "
                                f"base_repeat={_p03_rebuilt.get('ingredient_base_repeat_pct')}%, "
                                f"rej_viol={_p03_rebuilt.get('rejection_violations')})"
                            )
                        else:
                            logger.error(
                                f"[P0-3/AUTO-RECOVERY] Reconstrucción exitosa pero falló "
                                f"persistir en plan_data para plan {meal_plan_id}. "
                                f"Continuando con copia in-memory."
                            )
                    else:
                        # [P0-4] Last-resort antes de rendirse: sintetizar lección desde
                        # plan_data.days del chunk previo. Aplica especialmente a planes
                        # de 7d donde no hay _recent_chunk_lessons rolling window que
                        # pueda servir de red de seguridad. Si los días del chunk N-1
                        # ya están persistidos en meal_plans.plan_data.days, podemos
                        # extraer nombres+bases y al menos darle al LLM "no repitas esto".
                        _p04_synth = _synthesize_last_chunk_learning_from_plan_days(
                            meal_plan_id, _p03_target_week, prior_plan_data,
                            user_id=user_id,  # [P1-3] habilita telemetría schema_invalid
                        )
                        if _p04_synth:
                            # [P0-A] Telemetría: este chunk arrancó con `_last_chunk_learning`
                            # sintetizado (low-confidence) en vez de queue-based. Si demasiados
                            # chunks caen aquí, el cron `_alert_high_synthesized_lesson_ratio`
                            # disparará alerta — síntoma de que `plan_chunk_queue.learning_metrics`
                            # no se está persistiendo correctamente.
                            _record_chunk_lesson_telemetry(
                                user_id=user_id,
                                meal_plan_id=meal_plan_id,
                                week_number=int(week_number),
                                event="lesson_synthesized_low_confidence",
                                synthesized_count=1,
                                queue_count=0,
                                metadata={
                                    "prev_week": int(_p03_target_week),
                                    "synthesized_meal_count": int(_p04_synth.get("synthesized_meal_count") or 0),
                                    "repeated_bases_count": len(_p04_synth.get("repeated_bases") or []),
                                    "chunk_tag_present": bool(_p04_synth.get("synthesized_chunk_tag_present")),
                                    "rebuild_reason": _p03_reason,
                                },
                            )
                            # [P0.3] Centralizado vía `persist_legacy_learning_to_plan_data`.
                            _p04_persisted = persist_legacy_learning_to_plan_data(
                                meal_plan_id, _p04_synth,
                                context="synthesis_from_days",
                            )
                            # In-memory unconditional: aunque la persistencia falle, dejamos
                            # la síntesis in-memory para que este chunk arranque con señal real.
                            prior_plan_data["_last_chunk_learning"] = _p04_synth
                            if _p04_persisted:
                                logger.warning(
                                    f"[P0-4/SYNTHESIZED] _last_chunk_learning sintetizado desde "
                                    f"plan_data.days para plan {meal_plan_id} chunk {week_number} "
                                    f"(prev_week={_p03_target_week}, "
                                    f"meals={_p04_synth['synthesized_meal_count']}, "
                                    f"bases={len(_p04_synth['repeated_bases'])}, "
                                    f"chunk_tagged={_p04_synth['synthesized_chunk_tag_present']}). "
                                    f"Last-resort tras fallar plan_chunk_queue.learning_metrics."
                                )
                            else:
                                logger.error(
                                    f"[P0-4/SYNTHESIZE] Síntesis exitosa pero falló persistir "
                                    f"para plan {meal_plan_id}. Continuando con copia in-memory."
                                )

                            # [P0-B] Circuit breaker per-usuario. Este chunk acaba de
                            # caer en synthesis_from_days (low-confidence) — si el usuario
                            # ya viene con ratio alto, pausamos antes de despachar al LLM
                            # con señal pobre. Sin esto, chunks N+1, N+2... seguirían
                            # generándose con learning degradado hasta que SRE detecte
                            # la alerta agregada (que tiene cooldown 24h y umbral global).
                            _p0b_ratio = _per_user_synthesis_ratio_exceeded(user_id)
                            if _p0b_ratio.get("exceeded"):
                                if _pause_chunk_for_synthesis_overload(
                                    task_id=task_id,
                                    snap=snap,
                                    user_id=user_id,
                                    meal_plan_id=meal_plan_id,
                                    week_number=int(week_number),
                                    ratio_info=_p0b_ratio,
                                    source="last_chunk_learning_synth",
                                ):
                                    return
                        else:
                            # No hay learning_metrics en la cola NI días utilizables en
                            # plan_data. Esto puede pasar si el chunk previo aún no completó
                            # o si plan_data fue truncado. No es bloqueante: los demás
                            # defensores (recent_chunk_lessons, P0-2 backfill,
                            # critical_lessons_permanent) seguirán protegiendo el aprendizaje.
                            logger.warning(
                                f"[P0-3/AUTO-RECOVERY] No se pudo reconstruir _last_chunk_learning "
                                f"para plan {meal_plan_id} chunk {week_number} (prev_week={_p03_target_week}, "
                                f"reason={_p03_reason}): plan_chunk_queue.learning_metrics no disponible "
                                f"y plan_data.days tampoco tuvo señal sintetizable. "
                                f"Continuando con dict existente."
                            )

            # [P1-1] Anti-corrupción de plan_data en planes largos.
            # Si llegamos a un chunk N>3 dentro de un plan ≥15d, _recent_chunk_lessons DEBE
            # contener al menos min(N-1, 8) entradas (la ventana rolling se rellena en cada chunk
            # previo; ver el bloque que setea _recent_chunk_lessons tras cada generación). Si
            # falta o está corta, plan_data fue truncado/corrompido y generar el chunk perdería
            # 5+ chunks de aprendizaje silenciosamente. Pausamos para auditoría humana.
            try:
                _p11_total_days = int(
                    (prior_plan_data or {}).get("total_days_requested")
                    or (snap or {}).get("form_data", {}).get("totalDays")
                    or 0
                )
            except (TypeError, ValueError):
                _p11_total_days = 0
            if int(week_number) > 3 and _p11_total_days >= 15:
                _p11_window_cap = _rolling_lessons_window_cap(_p11_total_days)
                _p11_expected = min(int(week_number) - 1, _p11_window_cap)
                _p11_recent = (prior_plan_data or {}).get("_recent_chunk_lessons")
                _p11_actual = len(_p11_recent) if isinstance(_p11_recent, list) else 0
                if _p11_actual < _p11_expected:
                    # [P1-1 AUTO-RECOVERY] Antes pausábamos directo. Ahora intentamos
                    # reconstruir _recent_chunk_lessons desde plan_chunk_queue.learning_metrics
                    # de los chunks completados. Si lo logramos, persistimos a plan_data y
                    # continuamos. Solo pausamos si la reconstrucción tampoco alcanza el mínimo.
                    logger.warning(
                        f"[P1-1/MISSING-LESSONS] Chunk {week_number} de plan {meal_plan_id} "
                        f"(total_days={_p11_total_days}) tiene {_p11_actual} lecciones; "
                        f"esperadas >= {_p11_expected}. Intentando auto-recovery desde plan_chunk_queue."
                    )
                    _p11_rebuilt = _rebuild_recent_chunk_lessons_from_queue(
                        meal_plan_id, int(week_number), _p11_total_days
                    )
                    if len(_p11_rebuilt) >= _p11_expected:
                        try:
                            # [P0-B] Sellar `_plan_modified_at` para que el CAS pueda
                            # detectar este UPDATE intermedio (ver nota P0-3 rebuild arriba).
                            execute_sql_write(
                                """
                                UPDATE meal_plans
                                SET plan_data = jsonb_set(
                                        jsonb_set(
                                            COALESCE(plan_data, '{}'::jsonb),
                                            '{_recent_chunk_lessons}',
                                            %s::jsonb,
                                            true
                                        ),
                                        '{_plan_modified_at}',
                                        to_jsonb(NOW()::text),
                                        true
                                    )
                                WHERE id = %s
                                """,
                                (json.dumps(_p11_rebuilt, ensure_ascii=False), meal_plan_id),
                            )
                            logger.info(
                                f"[P1-1/AUTO-RECOVERED] Reconstruidas {len(_p11_rebuilt)} lecciones "
                                f"desde plan_chunk_queue para plan {meal_plan_id}. Continuando chunk {week_number}."
                            )
                            # Refrescar la copia in-memory para que el resto del worker la use.
                            if not isinstance(prior_plan_data, dict):
                                prior_plan_data = {}
                            prior_plan_data["_recent_chunk_lessons"] = _p11_rebuilt
                        except Exception as _p11_persist_err:
                            logger.error(
                                f"[P1-1/AUTO-RECOVERY] Falló persistir lecciones reconstruidas: "
                                f"{_p11_persist_err}. Cayendo a pausa para auditoría."
                            )
                            _p11_rebuilt = []  # forzar caída a la rama de pausa abajo

                    if len(_p11_rebuilt) < _p11_expected:
                        # [P1-2] Capa intermedia ANTES de pausar: regenerar lecciones
                        # faltantes desde plan_data.days (la fuente de verdad de los meals
                        # consumidos por chunk). Útil cuando plan_chunk_queue.learning_metrics
                        # también está caído (e.g., chunks crashearon pre-preflight, schema
                        # downgrade, JSON corrupto). Combina las lecciones que SÍ vinieron
                        # de la cola con las sintetizadas desde plan_data.days; las
                        # sintetizadas quedan marcadas low_confidence=True para que el LLM
                        # las pondere menos pero al menos tenga señal.
                        _p12_combined = _regenerate_recent_chunk_lessons_from_plan_days(
                            meal_plan_id=meal_plan_id,
                            plan_data=prior_plan_data if isinstance(prior_plan_data, dict) else {},
                            target_week=int(week_number),
                            total_days_requested=_p11_total_days,
                            seed_lessons=_p11_rebuilt,
                        )
                        # [P0-A] Telemetría: contar entradas sintetizadas vs queue-based en la
                        # ventana rolling resultante. Una entrada con `synthesized_from_plan_days=True`
                        # significa que ese chunk previo no tenía learning_metrics en la cola y
                        # tuvimos que reconstruir desde plan_data.days. Solo emitimos si hubo al
                        # menos una síntesis — ratios queue-only no necesitan telemetría
                        # (caso normal).
                        try:
                            _p0a_synth = sum(
                                1 for _l in _p12_combined
                                if isinstance(_l, dict) and _l.get("synthesized_from_plan_days")
                            )
                            _p0a_queue = len(_p12_combined) - _p0a_synth
                            if _p0a_synth > 0:
                                _record_chunk_lesson_telemetry(
                                    user_id=user_id,
                                    meal_plan_id=meal_plan_id,
                                    week_number=int(week_number),
                                    event="recent_lessons_partial_synthesis",
                                    synthesized_count=_p0a_synth,
                                    queue_count=_p0a_queue,
                                    metadata={
                                        "total_days_requested": int(_p11_total_days),
                                        "expected_lessons": int(_p11_expected),
                                        "actual_combined": int(len(_p12_combined)),
                                        "seed_from_queue": int(len(_p11_rebuilt)),
                                    },
                                )
                        except Exception as _p0a_tele_err:
                            logger.debug(
                                f"[P0-A/LESSON-TELEMETRY] No se pudo emitir evento de "
                                f"recent_lessons_partial_synthesis: {_p0a_tele_err!r}"
                            )

                        # [P0-B] Circuit breaker per-usuario tras regen. Idéntico al
                        # check del site `_synthesize_last_chunk_learning_from_plan_days`:
                        # si la ventana rolling acaba de regenerarse parcialmente desde
                        # plan_data.days y este usuario lleva ratio alto, pausar antes
                        # de persistir + dispatch. Sin esto, el LLM corre con lecciones
                        # marcadas low_confidence y propaga la degradación al siguiente
                        # chunk (cuyo learning vendrá del output de éste).
                        if _p0a_synth > 0:
                            _p0b_ratio_regen = _per_user_synthesis_ratio_exceeded(user_id)
                            if _p0b_ratio_regen.get("exceeded"):
                                if _pause_chunk_for_synthesis_overload(
                                    task_id=task_id,
                                    snap=snap,
                                    user_id=user_id,
                                    meal_plan_id=meal_plan_id,
                                    week_number=int(week_number),
                                    ratio_info=_p0b_ratio_regen,
                                    source="recent_lessons_regen",
                                ):
                                    return

                        if len(_p12_combined) >= _p11_expected:
                            try:
                                # [P0-B] Sellar `_plan_modified_at` (ver nota P0-3 rebuild arriba).
                                execute_sql_write(
                                    """
                                    UPDATE meal_plans
                                    SET plan_data = jsonb_set(
                                            jsonb_set(
                                                COALESCE(plan_data, '{}'::jsonb),
                                                '{_recent_chunk_lessons}',
                                                %s::jsonb,
                                                true
                                            ),
                                            '{_plan_modified_at}',
                                            to_jsonb(NOW()::text),
                                            true
                                        )
                                    WHERE id = %s
                                    """,
                                    (json.dumps(_p12_combined, ensure_ascii=False), meal_plan_id),
                                )
                                logger.info(
                                    f"[P1-2/REGEN-OK] Reconstruidas {len(_p12_combined)} lecciones "
                                    f"combinando queue+plan_days para plan {meal_plan_id} chunk "
                                    f"{week_number}. Continuando sin pausa."
                                )
                                if not isinstance(prior_plan_data, dict):
                                    prior_plan_data = {}
                                prior_plan_data["_recent_chunk_lessons"] = _p12_combined
                                _p11_rebuilt = _p12_combined  # actualizar para que el if siguiente vea ya OK
                            except Exception as _p12_persist_err:
                                # Persistencia falló: dejamos las lecciones in-memory para este
                                # chunk; futuros chunks intentarán de nuevo. Mejor inyectar señal
                                # al LLM aunque no quede sellada que pausar.
                                logger.error(
                                    f"[P1-2/REGEN] Síntesis combinada exitosa pero falló persistir "
                                    f"para plan {meal_plan_id}: {_p12_persist_err}. Continuando "
                                    f"con copia in-memory."
                                )
                                if not isinstance(prior_plan_data, dict):
                                    prior_plan_data = {}
                                prior_plan_data["_recent_chunk_lessons"] = _p12_combined
                                _p11_rebuilt = _p12_combined

                    if len(_p11_rebuilt) < _p11_expected:
                        logger.error(
                            f"[P1-1/MISSING-LESSONS] Auto-recovery insuficiente: reconstruidas "
                            f"{len(_p11_rebuilt)}, requeridas >= {_p11_expected}. Pausando para auditoría."
                        )
                        # [P0-3] Envolver TODO el bloque de pausa en try/except. Si la pausa
                        # misma falla (DB blip durante UPDATE, snap no serializable, etc.) la
                        # excepción antes se propagaba al outer catch y el chunk caía al flujo
                        # standard de 'failed + retry'. Eso significaba reintentar contra el
                        # mismo plan_data corrupto hasta agotar attempts en dead_letter — sin
                        # señal clara al operador y sin protección contra reentrar al pipeline
                        # LLM en variantes futuras del flujo.
                        # Ahora: si la pausa falla, hard-fail explícito con
                        # dead_letter_reason='guard_pause_failed:missing_prior_lessons' para
                        # que el operador investigue. NO retry — el problema no se arregla solo.
                        try:
                            _p11_pause = copy.deepcopy(snap) if isinstance(snap, dict) else {}
                            _p11_pause["_pause_reason"] = "missing_prior_lessons"
                            _p11_pause["_p1_1_expected_lessons"] = _p11_expected
                            _p11_pause["_p1_1_actual_lessons"] = _p11_actual
                            _p11_pause["_p1_1_rebuilt_lessons"] = len(_p11_rebuilt)
                            _p11_pause["_p1_1_paused_at"] = datetime.now(timezone.utc).isoformat()
                            execute_sql_write(
                                """
                                UPDATE plan_chunk_queue
                                SET status = 'pending_user_action',
                                    pipeline_snapshot = %s::jsonb,
                                    updated_at = NOW()
                                WHERE id = %s
                                """,
                                (json.dumps(_p11_pause, ensure_ascii=False), task_id),
                            )
                            try:
                                import threading as _p11_threading
                                from utils_push import send_push_notification as _p11_push
                                _p11_threading.Thread(
                                    target=_p11_push,
                                    kwargs={
                                        "user_id": user_id,
                                        "title": "Tu plan necesita una revisión",
                                        "body": (
                                            "Detectamos un problema con el historial de tu plan. "
                                            "Ábrelo para que lo revisemos juntos."
                                        ),
                                        "url": "/dashboard",
                                    },
                                    daemon=True,
                                ).start()
                            except Exception as _p11_push_err:
                                logger.warning(
                                    f"[P1-1] No se pudo enviar push notification: {_p11_push_err}"
                                )
                            return
                        except Exception as _p11_pause_err:
                            # [P0-3/GUARD-FAIL] La pausa misma falló. Hard-fail el chunk con
                            # dead_letter_reason explícito para evitar retry inútil.
                            logger.critical(
                                f"[P0-3/GUARD-FAIL] Falló pausar chunk {task_id} por "
                                f"missing_prior_lessons: {type(_p11_pause_err).__name__}: "
                                f"{_p11_pause_err}. Hard-failing chunk para evitar retry "
                                f"contra plan_data corrupto."
                            )
                            try:
                                execute_sql_write(
                                    """
                                    UPDATE plan_chunk_queue
                                    SET status = 'failed',
                                        dead_lettered_at = COALESCE(dead_lettered_at, NOW()),
                                        dead_letter_reason = %s,
                                        updated_at = NOW()
                                    WHERE id = %s
                                    """,
                                    (
                                        f"guard_pause_failed:missing_prior_lessons:"
                                        f"{type(_p11_pause_err).__name__}",
                                        task_id,
                                    ),
                                )
                            except Exception as _p11_hard_fail_err:
                                logger.error(
                                    f"[P0-3/GUARD-FAIL] Hard-fail también falló para chunk "
                                    f"{task_id}: {type(_p11_hard_fail_err).__name__}: "
                                    f"{_p11_hard_fail_err}. El chunk caerá al outer catch retry."
                                )
                            # Telemetría: registrar métrica de fallo del guard para que
                            # operadores detecten patrón sistémico (DB problemas, schema roto).
                            try:
                                duration_ms = int((_t.time() - chunk_start_ts) * 1000)
                                _record_chunk_metric(
                                    chunk_id=task_id,
                                    meal_plan_id=meal_plan_id,
                                    user_id=user_id,
                                    week_number=week_number,
                                    days_count=days_count,
                                    duration_ms=duration_ms,
                                    quality_tier="failed",
                                    was_degraded=True,
                                    retries=int(_pickup_attempts or 0),
                                    lag_seconds=lag_seconds,
                                    learning_metrics=None,
                                    error_message=(
                                        f"guard_pause_failed:missing_prior_lessons:"
                                        f"{type(_p11_pause_err).__name__}"
                                    )[:500],
                                    is_rolling_refill=is_rolling_refill,
                                )
                            except Exception as _p11_metric_err:
                                logger.warning(
                                    f"[P0-3/GUARD-FAIL] No se pudo registrar métrica: "
                                    f"{_p11_metric_err}"
                                )
                            return

            learning_ready = _check_chunk_learning_ready(
                user_id=user_id,
                meal_plan_id=meal_plan_id,
                week_number=week_number,
                days_offset=days_offset,
                plan_data=prior_plan_data,
                snapshot=snap,
            )
            learning_ready_ratio = learning_ready.get("ratio")
            learning_ready_deferrals = int(snap.get("_learning_ready_deferrals") or 0)
            if learning_ready_ratio is not None:
                _proxy_flags = []
                if learning_ready.get("used_implicit_proxy"):
                    _proxy_flags.append("sparse_proxy" if learning_ready.get("sparse_logging_proxy") else "zero_log_proxy")
                if learning_ready.get("inventory_proxy_used"):
                    _proxy_flags.append(f"inv_proxy({learning_ready.get('inventory_mutations', 0)}m)")
                _proxy_tag = f" [{','.join(_proxy_flags)}]" if _proxy_flags else ""
                logger.info(
                    f"[CHUNK/LEARNING-READY] plan={meal_plan_id} chunk={week_number} "
                    f"ratio={learning_ready_ratio:.0%} matched={learning_ready.get('matched_meals', 0)}/"
                    f"{learning_ready.get('planned_meals', 0)} "
                    f"explicit_logged={learning_ready.get('explicit_logged_meals', 0)}{_proxy_tag} "
                    f"window=days {learning_ready.get('previous_chunk_start_day')}–{learning_ready.get('previous_chunk_end_day')} "
                    f"deferrals={learning_ready_deferrals}/{CHUNK_LEARNING_READY_MAX_DEFERRALS}"
                )

            _learning_signal_strength = learning_ready.get("learning_signal_strength")
            if _learning_signal_strength and _learning_signal_strength != "none":
                form_data["_learning_signal_strength"] = _learning_signal_strength

            # [P0-D] Si el inventory proxy aprobó el chunk pese a zero-log, marcamos para que
            # learning_metrics y _last_chunk_learning lo registren (telemetría + lección).
            if learning_ready.get("inventory_proxy_used"):
                form_data["_inventory_activity_proxy_used"] = True
                form_data["_inventory_activity_mutations"] = int(learning_ready.get("inventory_mutations") or 0)
                # [P0-2] Proxy aprobó sin logs reales → forzar variedad y bloquear
                # la técnica culinaria del chunk previo para anti-aburrimiento.
                form_data["_force_variety"] = True
                _prev_technique = prior_plan_data.get("last_technique")
                if _prev_technique:
                    _blocked = form_data.get("_blocked_techniques", [])
                    if _prev_technique not in _blocked:
                        _blocked.append(_prev_technique)
                    form_data["_blocked_techniques"] = _blocked
                    logger.info(
                        f"[P0-2/INV-PROXY] Chunk {week_number} plan {meal_plan_id}: "
                        f"inventory proxy aprobó con {learning_ready.get('inventory_mutations', 0)} mutaciones. "
                        f"_force_variety=True, técnica bloqueada={_prev_technique}"
                    )
                # [P0-6] Propaga contador consecutivo + lifetime acumulativo a chunks futuros.
                # [P1-3] Fuente de verdad migrada a `meal_plans.plan_data` vía
                # `update_plan_data_atomic` (SELECT … FOR UPDATE). Esto cierra la race
                # condition donde dos chunks paralelos leían un `pipeline_snapshot`
                # desfasado y cada uno incrementaba sobre la misma base, perdiendo un +1.
                # El UPDATE de `pipeline_snapshot` se mantiene como espejo para callers
                # legacy que aún leen del snapshot; el gate prefiere `plan_data` (ver
                # `_p13_read_counter`).
                _new_proxy_count = int(prior_plan_data.get("_consecutive_proxy_chunks") or 0) + 1
                _new_lifetime_proxy = int(prior_plan_data.get("_lifetime_proxy_chunks") or 0) + 1
                _new_lifetime_total = int(prior_plan_data.get("_lifetime_total_chunks") or 0) + 1
                try:
                    from db_plans import update_plan_data_atomic as _p13_atomic

                    def _p13_proxy_mutator(pd: dict) -> dict:
                        pd["_consecutive_proxy_chunks"] = int(pd.get("_consecutive_proxy_chunks") or 0) + 1
                        pd["_lifetime_proxy_chunks"] = int(pd.get("_lifetime_proxy_chunks") or 0) + 1
                        pd["_lifetime_total_chunks"] = int(pd.get("_lifetime_total_chunks") or 0) + 1
                        return pd

                    _p13_updated = _p13_atomic(meal_plan_id, _p13_proxy_mutator)
                    if isinstance(_p13_updated, dict):
                        _new_proxy_count = int(_p13_updated.get("_consecutive_proxy_chunks") or _new_proxy_count)
                        _new_lifetime_proxy = int(_p13_updated.get("_lifetime_proxy_chunks") or _new_lifetime_proxy)
                        _new_lifetime_total = int(_p13_updated.get("_lifetime_total_chunks") or _new_lifetime_total)
                except Exception as _p13_err:
                    logger.warning(
                        f"[P1-3] update_plan_data_atomic falló para plan {meal_plan_id} "
                        f"(proxy_used branch): {_p13_err}. Continuando con escritura solo a "
                        f"pipeline_snapshot."
                    )
                execute_sql_write("""
                    UPDATE plan_chunk_queue
                    SET pipeline_snapshot = jsonb_set(
                        jsonb_set(
                            jsonb_set(
                                COALESCE(pipeline_snapshot, '{}'::jsonb),
                                '{_consecutive_proxy_chunks}',
                                %s::jsonb
                            ),
                            '{_lifetime_proxy_chunks}',
                            %s::jsonb
                        ),
                        '{_lifetime_total_chunks}',
                        %s::jsonb
                    )
                    WHERE meal_plan_id = %s AND status IN ('pending', 'stale') AND week_number > %s
                """, (str(_new_proxy_count), str(_new_lifetime_proxy), str(_new_lifetime_total),
                      meal_plan_id, week_number))
            elif _learning_signal_strength == "strong":
                # [P0-3] Antes: reset duro vía `pipeline_snapshot - '_consecutive_proxy_chunks'`.
                # Ahora: decremento (max(0, n-1)) + incremento del total lifetime SIN tocar el
                # contador lifetime de proxy. Esto cierra el bypass [proxy, strong, proxy, strong]
                # porque cada chunk fuerte solo "perdona" uno consecutivo en vez de borrarlos todos.
                # [P1-3] Fuente de verdad: `meal_plans.plan_data` (ver helper proxy_used arriba).
                _curr_consec = int(prior_plan_data.get("_consecutive_proxy_chunks") or 0)
                _new_consec = max(0, _curr_consec - 1)
                _new_lifetime_total = int(prior_plan_data.get("_lifetime_total_chunks") or 0) + 1
                try:
                    from db_plans import update_plan_data_atomic as _p13_atomic

                    def _p13_strong_mutator(pd: dict) -> dict:
                        pd["_consecutive_proxy_chunks"] = max(0, int(pd.get("_consecutive_proxy_chunks") or 0) - 1)
                        pd["_lifetime_total_chunks"] = int(pd.get("_lifetime_total_chunks") or 0) + 1
                        return pd

                    _p13_updated = _p13_atomic(meal_plan_id, _p13_strong_mutator)
                    if isinstance(_p13_updated, dict):
                        _new_consec = int(_p13_updated.get("_consecutive_proxy_chunks") or _new_consec)
                        _new_lifetime_total = int(_p13_updated.get("_lifetime_total_chunks") or _new_lifetime_total)
                except Exception as _p13_err:
                    logger.warning(
                        f"[P1-3] update_plan_data_atomic falló para plan {meal_plan_id} "
                        f"(strong branch): {_p13_err}. Continuando con escritura solo a "
                        f"pipeline_snapshot."
                    )
                execute_sql_write("""
                    UPDATE plan_chunk_queue
                    SET pipeline_snapshot = jsonb_set(
                        jsonb_set(
                            COALESCE(pipeline_snapshot, '{}'::jsonb),
                            '{_consecutive_proxy_chunks}',
                            %s::jsonb
                        ),
                        '{_lifetime_total_chunks}',
                        %s::jsonb
                    )
                    WHERE meal_plan_id = %s AND status IN ('pending', 'stale') AND week_number > %s
                """, (str(_new_consec), str(_new_lifetime_total), meal_plan_id, week_number))
            # [P0-1] El gate ahora distingue tres casos al fallar:
            #   (a) zero-log: el usuario no logueó NADA del chunk previo → no hay aprendizaje real.
            #   (b) sparse-log o ratio bajo: hay alguna señal pero por debajo del umbral.
            #   (c) prior_chunk_not_elapsed: el chunk previo aún no terminó en el calendario.
            # flexible_mode (heredado de pausa por nevera vacía o de un opt-in del usuario) bypasea el gate.
            _learning_flexible_mode = bool(snap.get("_pantry_flexible_mode") or snap.get("_learning_flexible_mode"))
            _is_zero_log = bool(learning_ready.get("zero_log_proxy"))
            _is_sparse_log = bool(learning_ready.get("sparse_logging_proxy"))
            _signal_too_weak = _is_zero_log or _is_sparse_log

            if not learning_ready.get("ready", True) and not _learning_flexible_mode:
                _defer_reason = learning_ready.get("reason")
                # [P0-2 v2] Sin ancla de fecha (form_data._plan_start_date, grocery_start_date,
                # ni meal_plans.created_at recuperable): pausar para intervención manual en
                # lugar de avanzar con un NOW() fabricado que rompería el temporal gate.
                if _defer_reason == "missing_start_date_no_anchor":
                    # [P1-2] Antes el chunk se quedaba en `pending_user_action` indefinidamente:
                    # `_recover_pantry_paused_chunks` no tenía branch para este reason, así que
                    # el plan quedaba congelado sin escalación visible al usuario. Ahora
                    # contamos los intentos consecutivos en `plan_data._anchor_recovery_attempts`
                    # y, al exceder `CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS`, escalamos vía
                    # `_escalate_unrecoverable_chunk(reason='unrecoverable_missing_anchor')`.
                    _p12_attempts_row = None
                    try:
                        _p12_attempts_row = execute_sql_query(
                            "SELECT COALESCE((plan_data->>'_anchor_recovery_attempts')::int, 0) AS attempts "
                            "FROM meal_plans WHERE id = %s",
                            (meal_plan_id,),
                            fetch_one=True,
                        )
                    except Exception as _p12_read_err:
                        logger.warning(
                            f"[P1-2] No se pudo leer _anchor_recovery_attempts para plan "
                            f"{meal_plan_id}: {_p12_read_err}. Asumiendo 0."
                        )
                    _p12_prev_attempts = int((_p12_attempts_row or {}).get("attempts") or 0)
                    _p12_new_attempts = _p12_prev_attempts + 1
                    try:
                        execute_sql_write(
                            "UPDATE meal_plans "
                            "SET plan_data = jsonb_set(COALESCE(plan_data, '{}'::jsonb), "
                            "'{_anchor_recovery_attempts}', to_jsonb(%s::int), true) "
                            "WHERE id = %s",
                            (_p12_new_attempts, meal_plan_id),
                        )
                    except Exception as _p12_persist_err:
                        logger.warning(
                            f"[P1-2] No se pudo persistir _anchor_recovery_attempts={_p12_new_attempts} "
                            f"en plan {meal_plan_id}: {_p12_persist_err}"
                        )

                    if _p12_new_attempts >= CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS:
                        logger.error(
                            f"[P1-2/ANCHOR-ESCALATE] Chunk {week_number} plan {meal_plan_id} "
                            f"agotó {_p12_new_attempts} intentos sin ancla recuperable. "
                            f"Escalando a dead_letter (unrecoverable_missing_anchor)."
                        )
                        try:
                            _escalate_unrecoverable_chunk(
                                task_id=task_id,
                                user_id=user_id,
                                plan_id=meal_plan_id,
                                week_number=int(week_number),
                                recovery_attempts=_p12_new_attempts,
                                escalation_reason="unrecoverable_missing_anchor",
                            )
                        except Exception as _p12_esc_err:
                            logger.error(
                                f"[P1-2/ANCHOR-ESCALATE] Falló escalación para chunk "
                                f"{task_id}: {_p12_esc_err}"
                            )
                        return

                    logger.error(
                        f"[P0-2/NO-ANCHOR] Pausando chunk {week_number} plan {meal_plan_id} "
                        f"(intento {_p12_new_attempts}/{CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS}): "
                        f"sin ancla de fecha recuperable. Requiere intervención manual."
                    )
                    pause_snapshot = copy.deepcopy(snap) if isinstance(snap, dict) else {}
                    pause_snapshot["_pause_reason"] = "missing_start_date_no_anchor"
                    pause_snapshot["_p0_2_paused_at"] = datetime.now(timezone.utc).isoformat()
                    pause_snapshot["_anchor_recovery_attempts"] = _p12_new_attempts
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending_user_action',
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(pause_snapshot, ensure_ascii=False), task_id),
                    )
                    try:
                        import threading as _p02_threading
                        from utils_push import send_push_notification as _p02_push
                        _p02_threading.Thread(
                            target=_p02_push,
                            kwargs={
                                "user_id": user_id,
                                "title": "Tu plan necesita una revisión",
                                "body": (
                                    "Detectamos un problema con la fecha de inicio de tu plan. "
                                    "Ábrelo para que lo revisemos juntos."
                                ),
                                "url": "/dashboard",
                            },
                            daemon=True,
                        ).start()
                    except Exception as _p02_push_err:
                        logger.warning(
                            f"[P0-2] No se pudo enviar push notification: {_p02_push_err}"
                        )
                    return

                # [P0-2 FIX] unrecoverable_corrupted_date: la cascada del gate recuperó
                # un valor (snapshot/grocery_start_date/created_at) pero `safe_fromisoformat`
                # falló — datos corruptos en la fuente final. Antes este reason caía al
                # bloque de deferrals genérico (línea ~13473) y, tras agotar
                # CHUNK_LEARNING_READY_MAX_DEFERRALS, terminaba en _force_variety o pausa
                # genérica sin escalar. El plan quedaba colgado sin auto-recovery — el
                # usuario debía regenerarlo manualmente sin saber por qué.
                #
                # Espejo de la rama `missing_start_date_no_anchor` arriba: contamos en
                # `plan_data._anchor_recovery_attempts` (compartido con el caso
                # missing_anchor; ambos son "ancla de fecha inutilizable") y al exceder
                # CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS escalamos a dead_letter con
                # reason='unrecoverable_corrupted_date'. El recovery cron
                # (`_recover_pantry_paused_chunks` ~L5512) intenta re-parsear los anchors
                # en cada tick y reanuda si alguno se vuelve válido.
                if _defer_reason == "unrecoverable_corrupted_date":
                    _p02c_attempts_row = None
                    try:
                        _p02c_attempts_row = execute_sql_query(
                            "SELECT COALESCE((plan_data->>'_anchor_recovery_attempts')::int, 0) AS attempts "
                            "FROM meal_plans WHERE id = %s",
                            (meal_plan_id,),
                            fetch_one=True,
                        )
                    except Exception as _p02c_read_err:
                        logger.warning(
                            f"[P0-2/CORRUPT] No se pudo leer _anchor_recovery_attempts para plan "
                            f"{meal_plan_id}: {_p02c_read_err}. Asumiendo 0."
                        )
                    _p02c_prev_attempts = int((_p02c_attempts_row or {}).get("attempts") or 0)
                    _p02c_new_attempts = _p02c_prev_attempts + 1
                    try:
                        execute_sql_write(
                            "UPDATE meal_plans "
                            "SET plan_data = jsonb_set(COALESCE(plan_data, '{}'::jsonb), "
                            "'{_anchor_recovery_attempts}', to_jsonb(%s::int), true) "
                            "WHERE id = %s",
                            (_p02c_new_attempts, meal_plan_id),
                        )
                    except Exception as _p02c_persist_err:
                        logger.warning(
                            f"[P0-2/CORRUPT] No se pudo persistir _anchor_recovery_attempts={_p02c_new_attempts} "
                            f"en plan {meal_plan_id}: {_p02c_persist_err}"
                        )

                    if _p02c_new_attempts >= CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS:
                        logger.error(
                            f"[P0-2/CORRUPT-ESCALATE] Chunk {week_number} plan {meal_plan_id} "
                            f"agotó {_p02c_new_attempts} intentos con fecha de inicio corrupta. "
                            f"Escalando a dead_letter (unrecoverable_corrupted_date)."
                        )
                        try:
                            _escalate_unrecoverable_chunk(
                                task_id=task_id,
                                user_id=user_id,
                                plan_id=meal_plan_id,
                                week_number=int(week_number),
                                recovery_attempts=_p02c_new_attempts,
                                escalation_reason="unrecoverable_corrupted_date",
                            )
                        except Exception as _p02c_esc_err:
                            logger.error(
                                f"[P0-2/CORRUPT-ESCALATE] Falló escalación para chunk "
                                f"{task_id}: {_p02c_esc_err}"
                            )
                        return

                    logger.error(
                        f"[P0-2/CORRUPT] Pausando chunk {week_number} plan {meal_plan_id} "
                        f"(intento {_p02c_new_attempts}/{CHUNK_ANCHOR_RECOVERY_MAX_ATTEMPTS}): "
                        f"fecha de inicio corrupta sin anchor parseable. "
                        f"detalle={learning_ready.get('_corruption_detail') or 'unknown'}, "
                        f"fuente={learning_ready.get('_fallback_source') or 'snapshot'}."
                    )
                    pause_snapshot = copy.deepcopy(snap) if isinstance(snap, dict) else {}
                    pause_snapshot["_pause_reason"] = "unrecoverable_corrupted_date"
                    pause_snapshot["_pantry_pause_reason"] = "unrecoverable_corrupted_date"
                    pause_snapshot["_p0_2_paused_at"] = datetime.now(timezone.utc).isoformat()
                    pause_snapshot["_anchor_recovery_attempts"] = _p02c_new_attempts
                    pause_snapshot["_corruption_detail"] = learning_ready.get("_corruption_detail")
                    pause_snapshot["_corruption_fallback_source"] = learning_ready.get("_fallback_source")
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending_user_action',
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(pause_snapshot, ensure_ascii=False), task_id),
                    )
                    try:
                        import threading as _p02c_threading
                        from utils_push import send_push_notification as _p02c_push
                        _p02c_threading.Thread(
                            target=_p02c_push,
                            kwargs={
                                "user_id": user_id,
                                "title": "Tu plan necesita una revisión",
                                "body": (
                                    "Detectamos un problema con la fecha de inicio de tu plan. "
                                    "Ábrelo para que lo revisemos juntos."
                                ),
                                "url": "/dashboard",
                            },
                            daemon=True,
                        ).start()
                    except Exception as _p02c_push_err:
                        logger.warning(
                            f"[P0-2/CORRUPT] No se pudo enviar push notification: {_p02c_push_err}"
                        )
                    return

                if _defer_reason == "learning_proxy_exhausted":
                    logger.warning(f"[P0-6/PROXY-EXHAUSTED] Chunk {week_number} pausado. Demasiados chunks consecutivos con inventory_proxy.")
                    pause_snapshot = copy.deepcopy(snap)
                    pause_snapshot["_pause_reason"] = "learning_proxy_exhausted"
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending_user_action',
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(pause_snapshot, ensure_ascii=False), task_id)
                    )
                    try:
                        import threading
                        from utils_push import send_push_notification
                        threading.Thread(
                            target=send_push_notification,
                            kwargs={
                                "user_id": user_id,
                                "title": "Tu plan necesita tu feedback",
                                "body": "Necesitamos que registres tus comidas para seguir personalizando tu menú.",
                                "url": "/dashboard"
                            }
                        ).start()
                    except Exception as e:
                        logger.warning(f"Error mandando push proxy exhausted: {e}")
                    return

                if learning_ready_deferrals < CHUNK_LEARNING_READY_MAX_DEFERRALS:
                    deferred_snapshot = copy.deepcopy(snap)
                    deferred_snapshot["_learning_ready_deferrals"] = learning_ready_deferrals + 1
                    deferred_snapshot["_last_learning_ready_ratio"] = learning_ready_ratio
                    deferred_snapshot["_last_learning_zero_log"] = _is_zero_log
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending',
                            execute_after = NOW() + make_interval(hours => %s),
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (
                            CHUNK_LEARNING_READY_DELAY_HOURS,
                            json.dumps(deferred_snapshot, ensure_ascii=False),
                            task_id,
                        )
                    )
                    _defer_reason = learning_ready.get("reason")
                    # [P0-2] La rama 'missing_plan_start_date_unrecoverable' ya no es alcanzable:
                    # _check_chunk_learning_ready ahora siempre resuelve una fecha vía cascada
                    # (form_data._plan_start_date -> grocery_start_date -> created_at -> NOW()).
                    #
                    # [P1] Log fix: antes mostraba "ratio < umbral" siempre, lo que mentía cuando
                    # la causa real era zero_log o sparse_log. El gate en _check_chunk_learning_ready
                    # (línea 6148) es:
                    #   ready = (ratio_ready and not _signal_too_weak) or inventory_proxy_used
                    # Cuando _signal_too_weak=True, ready=False aunque ratio>=umbral (líneas 6076-6079
                    # explican que sin logs reales el ratio es engañoso). El mensaje original
                    # "ratio 50% < 50%" era inválido cuando ratio==umbral pero el deferral disparó
                    # por otra rama. Ahora explicamos la causa real.
                    if _is_zero_log:
                        _inv_muts = learning_ready.get("inventory_mutations", 0)
                        _defer_explanation = (
                            f"sin logs explícitos del chunk previo y solo {_inv_muts} mutaciones de inventario "
                            f"(mín {CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS} para proxy)"
                        )
                    elif _is_sparse_log:
                        _explicit = learning_ready.get('explicit_logged_meals', 0)
                        _planned = learning_ready.get('planned_meals', 0)
                        _defer_explanation = (
                            f"sparse log ({_explicit} de {_planned} comidas planificadas registradas)"
                        )
                    else:
                        _defer_explanation = (
                            f"ratio {(learning_ready_ratio or 0):.0%} < umbral "
                            f"{CHUNK_LEARNING_READY_MIN_RATIO:.0%}"
                        )
                    logger.warning(
                        f"[CHUNK/LEARNING-READY] chunk {week_number} pospuesto {CHUNK_LEARNING_READY_DELAY_HOURS}h: "
                        f"{_defer_explanation}. "
                        f"(zero_log={_is_zero_log}, sparse={_is_sparse_log}, ratio={(learning_ready_ratio or 0):.0%}). "
                        f"Re-encolado {learning_ready_deferrals + 1}/{CHUNK_LEARNING_READY_MAX_DEFERRALS}."
                    )
                    # [P0-1] Push solo en el PRIMER deferral para no spamear. Mensaje cambia según motivo.
                    if learning_ready_deferrals in (0, CHUNK_LEARNING_READY_MAX_DEFERRALS - 1):
                        if _is_zero_log:
                            _push_title = "Loguea tus comidas para tu próximo bloque"
                            _push_body = (
                                "No hemos visto qué comiste de tu plan actual. "
                                "Loguea tus comidas en el diario para que el siguiente bloque aprenda de ti."
                            )
                        elif _is_sparse_log:
                            _explicit = learning_ready.get('explicit_logged_meals', 0)
                            _planned = learning_ready.get('planned_meals', 0)
                            _push_title = "Loguea más comidas para que el plan aprenda"
                            _push_body = (
                                f"Solo registramos {_explicit} de {_planned} comidas — necesitamos al menos un 25 % para ajustar el siguiente bloque."
                            )
                        else:
                            _push_title = "Tu próximo bloque espera más feedback"
                            _push_body = (
                                "Loguea las comidas que hiciste estos días — el siguiente bloque del plan "
                                "se ajusta a partir de eso."
                            )
                        _dispatch_push_notification(
                            user_id=user_id,
                            title=_push_title,
                            body=_push_body,
                            url="/dashboard",
                        )
                    return
            if not learning_ready.get("ready", True) and not _learning_flexible_mode:
                # [P0-1] Deferrals agotados. Si el motivo es zero-log, NO forzamos la generación
                # silenciosamente: pausamos en pending_user_action. El recovery ya existente
                # (_recover_pantry_paused_chunks) cae a flexible_mode tras CHUNK_PANTRY_EMPTY_TTL_HOURS,
                # con lo cual el chunk se acaba generando aunque el usuario nunca responda — pero
                # como camino explícito, no como bypass silencioso.
                if _is_zero_log:
                    pause_snapshot = copy.deepcopy(snap)
                    pause_snapshot.setdefault("_pantry_pause_started_at", datetime.now(timezone.utc).isoformat())
                    pause_snapshot.setdefault("_pantry_pause_reminders", 0)
                    # [P0-2] Usar TTL corto de 4h en vez del genérico de 12h para zero-log.
                    pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_ZERO_LOG_RECOVERY_TTL_HOURS
                    pause_snapshot["_pantry_pause_reminder_hours"] = min(CHUNK_PANTRY_EMPTY_REMINDER_HOURS, CHUNK_ZERO_LOG_RECOVERY_TTL_HOURS - 1)
                    pause_snapshot["_pantry_pause_reason"] = "learning_zero_logs"
                    pause_snapshot["_last_learning_ready_ratio"] = learning_ready_ratio
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending_user_action',
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(pause_snapshot, ensure_ascii=False), task_id),
                    )
                    logger.warning(
                        f"[P0-2/LEARNING-PAUSED] chunk {week_number} plan {meal_plan_id} pausado: "
                        f"zero-log tras {CHUNK_LEARNING_READY_MAX_DEFERRALS} deferrals. "
                        f"TTL={CHUNK_ZERO_LOG_RECOVERY_TTL_HOURS}h (no {CHUNK_PANTRY_EMPTY_TTL_HOURS}h). "
                        f"Esperando logs, inventario ≥{CHUNK_LEARNING_INVENTORY_PROXY_MIN_MUTATIONS} mutaciones, o expiración."
                    )
                    # [P0-2] Read-Modify-Write atómico contra meal_plans.plan_data.
                    # Antes: leíamos prior_plan_data al inicio del worker, incrementábamos
                    # _consecutive_zero_log_chunks en memoria, y sobrescribíamos toda la
                    # plan_data con json.dumps(prior_plan_data). Si dos chunks corrían a la
                    # vez (15d/30d), uno podía sobrescribir el contador del otro y el
                    # gating de "≥3 zero-log → degradar" nunca disparaba.
                    # [BONUS-FIX] Antes intentaba SET generation_status=%s — `generation_status`
                    # NO es columna en meal_plans (solo vive dentro de plan_data). La query
                    # fallaba con `column does not exist` y el degradado nunca se persistía.
                    from db_plans import update_plan_data_atomic

                    def _bump_zero_log(pd: dict) -> dict:
                        n = int(pd.get("_consecutive_zero_log_chunks", 0) or 0) + 1
                        pd["_consecutive_zero_log_chunks"] = n
                        if n >= 3:
                            pd["generation_status"] = "degraded_pending_engagement"
                        return pd

                    try:
                        new_pd = update_plan_data_atomic(meal_plan_id, _bump_zero_log)
                    except Exception as _p02_err:
                        logger.warning(
                            f"[P0-2/ZERO-LOG] No se pudo actualizar plan_data atómicamente para "
                            f"plan {meal_plan_id}: {_p02_err}. Saltando incremento; otro chunk reintentará."
                        )
                        new_pd = prior_plan_data  # fallback in-memory, no se persiste

                    _consecutive_zero_log_chunks = int(new_pd.get("_consecutive_zero_log_chunks", 0) or 0)
                    # Reflejar el contador autoritativo en la copia in-memory que el resto
                    # del worker usa para construir prompts / lecciones de este chunk.
                    prior_plan_data["_consecutive_zero_log_chunks"] = _consecutive_zero_log_chunks
                    if new_pd.get("generation_status") == "degraded_pending_engagement":
                        prior_plan_data["generation_status"] = "degraded_pending_engagement"

                    # [P1-4] Antes la push solo invitaba a loguear; el usuario que no quería/podía
                    # loguear quedaba esperando 4-24h sin saber que existía la opción "auto_proxy"
                    # (ya expuesta en el banner del frontend vía /api/blocked-reasons). Ahora el
                    # body del push menciona explícitamente la alternativa "Continuar sin registrar"
                    # y deeplinka al diario donde el banner muestra el toggle. El CTA solo aparece
                    # si el usuario está en logging_preference='manual' — si ya está en auto_proxy
                    # no tendría sentido ofrecérselo. Lógica completa en _build_zero_log_push_payload.
                    _p14_logging_pref = "manual"
                    try:
                        _p14_pref_row = execute_sql_query(
                            "SELECT logging_preference FROM user_profiles WHERE id = %s",
                            (user_id,),
                            fetch_one=True,
                        )
                        if _p14_pref_row and _p14_pref_row.get("logging_preference"):
                            _p14_logging_pref = str(_p14_pref_row["logging_preference"])
                    except Exception as _p14_pref_err:
                        logger.debug(f"[P1-4] No se pudo leer logging_preference: {_p14_pref_err}")

                    if _consecutive_zero_log_chunks >= 3:
                        logger.error(f"[P0-4-NEW] Plan {meal_plan_id} degradado por falta de engagement (≥3 zero-log).")
                    _p14_payload = _build_zero_log_push_payload(
                        consecutive_zero_log_chunks=_consecutive_zero_log_chunks,
                        logging_preference=_p14_logging_pref,
                    )
                    _dispatch_push_notification(
                        user_id=user_id,
                        title=_p14_payload["title"],
                        body=_p14_payload["body"],
                        url=_p14_payload["url"],
                    )

                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET execute_after = NOW() + interval '24 hours', updated_at = NOW()
                        WHERE meal_plan_id = %s AND status = 'pending'
                          AND week_number > %s
                          AND execute_after <= NOW() + interval '24 hours'
                        """,
                        (meal_plan_id, week_number)
                    )
                    return

                # [P0-3] Si el motivo del defer fue el temporal_gate (chunk N+1 no debe
                # dispararse hasta que el último día de N haya transcurrido), preferimos
                # pausar a pending_user_action en lugar de forzar la generación. Sin esto,
                # un plan con TZ desalineada generaba chunks sobre días aún en consumo —
                # exactamente lo que P0-3 quiere prevenir. La pausa permite que:
                #   1. _recover_pantry_paused_chunks re-evalúe el gate cuando el día ya
                #      haya pasado y reanude el chunk con la fecha real.
                #   2. Si tras CHUNK_PREV_CHUNK_PAUSE_TTL_HOURS adicionales aún no se
                #      resuelve, el recovery escale a flexible_mode (no congela el plan).
                #
                # [P1-4] Postergamos la pausa hasta que el contador `_temporal_gate_retries`
                # supere CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY. En las primeras N evaluaciones,
                # el gate ya bumpeó `execute_after` con backoff exponencial (1, 2, 4, 8, 16
                # min...) y dejó el chunk en su status anterior — el scheduler lo re-pickeará
                # cuando el backoff venza, sin pausar al usuario. Solo cuando el drift de TZ
                # se confirma (≥5 deferrals con backoff ≈ 31 min de espera) escalamos a
                # `pending_user_action` con push notification. Antes pausábamos al primer
                # deferral, que con TZ drift moderado era prematuro y empujaba al usuario
                # a flexible_mode innecesariamente.
                _temporal_gate_reason = learning_ready.get("reason") == "prev_chunk_day_not_yet_elapsed"
                _temporal_gate_retries = int(learning_ready.get("temporal_gate_retries") or 0)
                if _temporal_gate_reason and _temporal_gate_retries < int(CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY):
                    logger.info(
                        f"[P1-4/TEMPORAL-GATE-BACKOFF] chunk {week_number} plan {meal_plan_id} "
                        f"deferred con backoff (retry={_temporal_gate_retries}/{int(CHUNK_TEMPORAL_GATE_MAX_RETRIES)}, "
                        f"push_at={int(CHUNK_TEMPORAL_GATE_PUSH_AT_RETRY)}). "
                        f"execute_after ya bumpeado por el gate — no pausamos aún."
                    )
                    return
                if _temporal_gate_reason:
                    pause_snapshot = copy.deepcopy(snap)
                    pause_snapshot.setdefault(
                        "_pantry_pause_started_at",
                        datetime.now(timezone.utc).isoformat(),
                    )
                    pause_snapshot.setdefault("_pantry_pause_reminders", 0)
                    pause_snapshot["_pantry_pause_reason"] = "prev_chunk_not_concluded"
                    pause_snapshot["_pantry_pause_ttl_hours"] = CHUNK_PREV_CHUNK_PAUSE_TTL_HOURS
                    pause_snapshot["_pantry_pause_reminder_hours"] = max(
                        1, CHUNK_PREV_CHUNK_PAUSE_TTL_HOURS // 2
                    )
                    pause_snapshot["_prev_chunk_pause_meta"] = {
                        "previous_chunk_start_day": learning_ready.get("previous_chunk_start_day"),
                        "previous_chunk_end_day": learning_ready.get("previous_chunk_end_day"),
                        "prev_end_date": learning_ready.get("prev_end_date"),
                        "days_until_prev_end": learning_ready.get("days_until_prev_end"),
                    }
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending_user_action',
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(pause_snapshot, ensure_ascii=False), task_id),
                    )
                    logger.warning(
                        f"[P0-3/PREV-CHUNK-PAUSE] chunk {week_number} plan {meal_plan_id} "
                        f"pausado: chunk previo aún no concluyó tras "
                        f"{CHUNK_LEARNING_READY_MAX_DEFERRALS} deferrals × "
                        f"{CHUNK_LEARNING_READY_DELAY_HOURS}h. "
                        f"prev_end={learning_ready.get('prev_end_date')} "
                        f"days_until_prev_end={learning_ready.get('days_until_prev_end')}. "
                        f"TTL={CHUNK_PREV_CHUNK_PAUSE_TTL_HOURS}h hasta escalar a flexible_mode."
                    )
                    try:
                        _dispatch_push_notification(
                            user_id=user_id,
                            title="Tu próximo bloque está esperando",
                            body=(
                                "Estamos esperando que termines los días anteriores de tu plan "
                                "para generar los siguientes. Loguea tus comidas o tócalas en el diario."
                            ),
                            url="/dashboard",
                        )
                    except Exception as _p03_push_err:
                        logger.warning(
                            f"[P0-3/PREV-CHUNK-PAUSE] No se pudo enviar push: {_p03_push_err}"
                        )
                    return

                # Sparse-log o ratio explícito bajo: generamos pero marcamos el chunk como
                # forzado para compensar con _force_variety y dejar trazado en plan_data.
                logger.warning(
                    f"[P0-1/LEARNING-FORCED] chunk {week_number} sigue bajo ({(learning_ready_ratio or 0):.0%}) "
                    f"tras {CHUNK_LEARNING_READY_MAX_DEFERRALS} deferrals. Generando con _force_variety=True."
                )
                form_data["_force_variety"] = True
                form_data["_learning_forced"] = True
                if learning_ready.get("sparse_logging_proxy"):
                    form_data["_sparse_logging_proxy"] = True
                    if week_number not in prior_plan_data.get("_sparse_forced_chunks", []):
                        _logged_qty = int(learning_ready.get("logged_meals", 0))
                        _expected_logs = int(learning_ready.get("total_meals", 0))
                        _dispatch_push_notification(
                            user_id=user_id,
                            title="Tu plan se generó con poca info",
                            body=f"Solo registramos {_logged_qty} de {_expected_logs} comidas. Loguea más para que el siguiente bloque ajuste mejor.",
                            url="/dashboard",
                        )
                form_data["_learning_forced_reason"] = (
                    "sparse_log" if learning_ready.get("sparse_logging_proxy") else "low_ratio"
                )
                snap["_learning_forced"] = True
            elif not learning_ready.get("ready", True) and _learning_flexible_mode:
                # flexible_mode bypassea el gate (usuario optó por seguir aunque sin aprendizaje).
                # Igualmente marcamos para que la lección y el prompt lo sepan.
                logger.warning(
                    f"[P0-1/LEARNING-FLEXIBLE] chunk {week_number} ratio={learning_ready_ratio:.0%}, "
                    f"flexible_mode activo. Generando con _force_variety=True como compensación."
                )
                form_data["_force_variety"] = True
                form_data["_learning_forced"] = True
                form_data["_learning_forced_reason"] = "flexible_mode_bypass"
                snap["_learning_forced"] = True

            form_data = _refresh_chunk_pantry(user_id, form_data, snapshot_form_data, task_id=task_id, week_number=week_number)
            if form_data.get("_pantry_paused"):
                return
            
            fresh_inventory = form_data.get("current_pantry_ingredients", [])
            fresh_inventory_source = form_data.get("_fresh_pantry_source")

            # [P0-3] Si tenemos lectura live, propagamos al snapshot del chunk actual y de los siblings vivos
            # para que su fallback futuro sea reciente (antes era la foto del momento de creación del plan).
            if fresh_inventory_source == "live":
                _persist_fresh_pantry_to_chunks(task_id, meal_plan_id, fresh_inventory, user_id=user_id)

            if _should_pause_for_empty_pantry(fresh_inventory_source, fresh_inventory, snap, form_data):
                # [P1-1] Source distinto a "live" se pausaba antes silenciosamente. Logueamos
                # explícitamente la fuente para detectar si la mayoría de pausas vienen de
                # snapshots vacíos (síntoma de un frontend que no envía despensa al crear plan).
                _meaningful = _count_meaningful_pantry_items(fresh_inventory)
                logger.warning(
                    f"[P1-1/PANTRY-EMPTY] Chunk {week_number} plan {meal_plan_id} pausado: "
                    f"items_meaningful={_meaningful} < min={CHUNK_MIN_FRESH_PANTRY_ITEMS}, "
                    f"source={fresh_inventory_source!r}, raw_count={len(fresh_inventory or [])}."
                )
                _pause_chunk_for_pantry_refresh(task_id, user_id, week_number, fresh_inventory)
                return

            is_degraded = snap.get("_degraded", False)
            result = {}
            
            while True:
                if is_degraded:
                    # [GAP 6 FIX: Probe LLM para auto-recovery]
                    try:
                        from langchain_google_genai import ChatGoogleGenerativeAI
                        import os
                        # [P0-1-RECOVERY/WORKER-FIX] No importar datetime/timezone aquí: el módulo
                        # ya los tiene globales (cron_tasks.py:3). Importarlos localmente
                        # promovía a `datetime` y `timezone` a variables locales de toda la función
                        # `_chunk_worker`, causando UnboundLocalError en el primer uso (L6651,
                        # rama zero_log) cuando esa rama corría antes que el bloque is_degraded.
                        # Era exactamente la causa de chunk a1a6025e-... que falló 5 veces con
                        # `cannot access local variable 'datetime'`.

                        # Evitar flapping: revisar si hicimos downgrade hace menos de 10 minutos
                        user_res_flap = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
                        hp_flap = user_res_flap.get("health_profile", {}) if user_res_flap else {}
                        last_downgrade = hp_flap.get('_last_downgrade_time')
                        can_probe = True
                    
                        if last_downgrade:
                            from constants import safe_fromisoformat
                            ld_dt = safe_fromisoformat(last_downgrade)
                            if (datetime.now(timezone.utc) - ld_dt).total_seconds() < 600:
                                can_probe = False
                                logger.info(f" [GAP 6] Downgrade reciente ({last_downgrade}), saltando probe para evitar flapping.")

                        if can_probe:
                            logger.info(f" [GAP 6] Iniciando Probe LLM para auto-recovery del chunk {week_number}...")
                            probe_llm = ChatGoogleGenerativeAI(
                                model="gemini-3.1-flash-lite-preview",
                                temperature=0.0,
                                google_api_key=os.environ.get("GEMINI_API_KEY"),
                                max_retries=0
                            )
                            import concurrent.futures as _cf
                            def _do_probe():
                                return probe_llm.invoke("ping")
                        
                            with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                                ex.submit(_do_probe).result(timeout=10)
                            
                            logger.info(f" [GAP 6] LLM Probe exitoso. Sistema estabilizado, restaurando a modo AI.")
                            is_degraded = False
                            snap.pop('_degraded', None)
                        
                            # Actualizar en BD para el actual y todos los futuros chunks
                            execute_sql_write("UPDATE plan_chunk_queue SET pipeline_snapshot = pipeline_snapshot - '_degraded' WHERE meal_plan_id = %s", (meal_plan_id,))
                        
                            # Limpiar historial de downgrade
                            execute_sql_write("UPDATE user_profiles SET health_profile = health_profile - '_last_downgrade_time' WHERE id = %s", (user_id,))
                        
                    except Exception as probe_e:
                        logger.warning(f" [CHUNK DEGRADED] Probe LLM falló o no pudo ejecutar ({probe_e}). Modo Smart Shuffle activo.")
            
                learning_metrics = None  # [P1-5] Inicializar para ambos paths (degraded/LLM)
                if is_degraded:
                    logger.warning(f" [CHUNK DEGRADED] Generando chunk {week_number} en modo degraded (Smart Shuffle) para plan {meal_plan_id}...")
                    form_data = _refresh_chunk_pantry(user_id, form_data, snapshot_form_data, task_id=task_id, week_number=week_number)
                    if form_data.get("_pantry_paused"):
                        return
                    prior_days = prior_plan_data.get("days", [])
                
                    if not prior_days:
                        raise Exception("No prior days available for Smart Shuffle")
                
                
                    new_days = []
                    safe_pool = [d for d in prior_days if isinstance(d.get("meals"), list) and len(d["meals"]) > 0]
                    if not safe_pool:
                        safe_pool = prior_days.copy()
                    else:
                        safe_pool = safe_pool.copy()
                        
                    # [GAP C FIX: Filtrar prior_days contra alergias y rechazos actuales]
                    user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
                    health_profile = user_res.get("health_profile", {}) if user_res else {}
                    current_allergies = health_profile.get("allergies", [])
                    current_dislikes = health_profile.get("dislikes", [])
                    current_diet = (
                        health_profile.get("dietType")
                        or ((health_profile.get("dietTypes") or [None])[0])
                        or ((snap.get("form_data", {}) or {}).get("dietType"))
                        or ""
                    )
                
                    if isinstance(current_allergies, str): current_allergies = [current_allergies] if current_allergies.strip() else []
                    if isinstance(current_dislikes, str): current_dislikes = [current_dislikes] if current_dislikes.strip() else []

                    blocklist = current_allergies + current_dislikes

                    # [P1-6 FIX] Construir edge recipes con catálogos ya filtrados por alergias/dislikes/dieta.
                    # [P0-C FIX] Pasar pantry para que solo elija ingredientes disponibles en la nevera.
                    from constants import PLAN_CHUNK_SIZE as _PCS
                    _fresh_pantry_for_edge = form_data.get("current_pantry_ingredients", [])
                    if len(safe_pool) < _PCS:
                        for _ in range(_PCS - len(safe_pool)):
                            edge_day = _build_filtered_edge_recipe_day(
                                current_allergies,
                                current_dislikes,
                                current_diet,
                                pantry_items=_fresh_pantry_for_edge,
                            )
                            if not edge_day:
                                logger.warning(
                                    f"[P1-6] No se pudo construir Edge Recipe seguro para {user_id} "
                                    f"con restricciones actuales. Se omite expansión del pool."
                                )
                                break
                            edge_day["day_name"] = "Edge Recipe"
                            safe_pool.append(edge_day)
                
                    if blocklist:
                        def _is_blocked(day):
                            for meal in day.get("meals", []):
                                txt = (meal.get("name", "") + " " + " ".join(meal.get("ingredients", []))).lower()
                                for alg in blocklist:
                                    if alg.strip() and alg.strip().lower() in txt:
                                        return True
                            return False
                        
                        filtered_pool = [d for d in safe_pool if not _is_blocked(d)]
                        if filtered_pool:
                            safe_pool = filtered_pool
                        else:
                            # [P1-3] Si TODOS los prior days violan las restricciones actuales,
                            # antes el código seguía con safe_pool=[] y el shuffle caía a backup_days
                            # / Edge Recipes / repetición forzada — pero esos fallbacks no están
                            # filtrados por blocklist y podían contener alérgenos del usuario. Es
                            # un riesgo de seguridad (no solo UX): generar con un alérgeno aunque
                            # sea por degradación rompe el contrato más estricto. Pausamos el chunk
                            # en pending_user_action y avisamos al usuario para que regenere su plan
                            # con sus restricciones actuales (o las revise si fue un cambio reciente).
                            logger.error(
                                f"[P1-3/RESTRICTIONS-BLOCK] Todos los prior days de plan {meal_plan_id} "
                                f"chunk {week_number} violan restricciones actuales {blocklist}. "
                                f"Pausando — generar con safe_pool=[] arrastraría alérgenos vía fallbacks."
                            )
                            try:
                                _p13_pause_snap = execute_sql_query(
                                    "SELECT pipeline_snapshot FROM plan_chunk_queue WHERE id = %s",
                                    (task_id,), fetch_one=True
                                )
                                _p13_snap = copy.deepcopy(
                                    (_p13_pause_snap or {}).get("pipeline_snapshot") or {}
                                )
                                if isinstance(_p13_snap, str):
                                    _p13_snap = json.loads(_p13_snap)
                                _p13_snap["_pause_reason"] = "all_prior_days_blocked_by_restrictions"
                                _p13_snap["_p1_3_blocklist_at_pause"] = list(blocklist)[:20]
                                _p13_snap["_p1_3_paused_at"] = datetime.now(timezone.utc).isoformat()
                                execute_sql_write(
                                    """
                                    UPDATE plan_chunk_queue
                                    SET status = 'pending_user_action',
                                        pipeline_snapshot = %s::jsonb,
                                        updated_at = NOW()
                                    WHERE id = %s
                                    """,
                                    (json.dumps(_p13_snap, ensure_ascii=False), task_id),
                                )
                                _dispatch_push_notification(
                                    user_id=user_id,
                                    title="Tu plan necesita actualizarse",
                                    body=(
                                        "Tus restricciones actuales no permiten reusar los días "
                                        "previos del plan. Regenera el plan para que se adapte a "
                                        "tus alergias y preferencias actuales."
                                    ),
                                    url="/dashboard",
                                )
                            except Exception as _p13_pause_err:
                                logger.error(
                                    f"[P1-3/RESTRICTIONS-BLOCK] Falló pausa segura del chunk: "
                                    f"{type(_p13_pause_err).__name__}: {_p13_pause_err}. "
                                    f"Releasing reservations y returning para no continuar con fallbacks."
                                )
                            release_chunk_reservations(user_id, str(task_id))
                            return

                    # [P0-1] Filtrar por bases de ingredientes con fatiga aprendida.
                    # Lee _last_chunk_learning, _recent_chunk_lessons y _lifetime_lessons_summary
                    # del plan anterior para excluir días donde TODAS las comidas usan bases
                    # que el sistema identificó como repetitivas. Si el pool quedaría vacío,
                    # conserva el original (degradación sin bloqueo).
                    _learned_bases_to_avoid: set = set()
                    # [P1-6] Helpers tipo-seguros: si plan_data fue corrompido (edición DB, JSON
                    # roundtrip, migración mal aplicada), `or {}` / `or []` no protegen contra
                    # tipos truthy incorrectos (ej. dict en lugar de list → iter da keys; lista
                    # en lugar de dict → .get() crashea). Los helpers centralizan la defensa.
                    _p01_last = _safe_lessons_dict(
                        prior_plan_data.get("_last_chunk_learning"),
                        field_name="_last_chunk_learning", plan_id=meal_plan_id,
                        user_id=user_id,
                    )
                    _p01_recent = _safe_lessons_list(
                        prior_plan_data.get("_recent_chunk_lessons"),
                        field_name="_recent_chunk_lessons", plan_id=meal_plan_id,
                        user_id=user_id,
                    )
                    # [P1-1] Excluir bases aprendidas desde chunks dead-lettered:
                    # esos chunks tienen plan_data parcial y sus "repeated_bases" son
                    # un artefacto de la ejecución abortada, no señal real del usuario.
                    _p01_last, _p01_recent, _p01_dead_weeks = (
                        _filter_lessons_excluding_dead_lettered(
                            _p01_last, _p01_recent, prior_plan_data, week_number,
                        )
                    )
                    if _p01_dead_weeks:
                        logger.info(
                            f"[P1-1/SHUFFLE] plan={meal_plan_id} chunk={week_number} "
                            f"omitiendo bases aprendidas de chunks dead-lettered={_p01_dead_weeks} "
                            f"al filtrar pool de days candidatos."
                        )
                    _p01_lifetime = _safe_lessons_dict(
                        prior_plan_data.get("_lifetime_lessons_summary"),
                        field_name="_lifetime_lessons_summary", plan_id=meal_plan_id,
                        user_id=user_id,
                    )

                    for _p01_rb in (_p01_last.get("repeated_bases") or []):
                        if isinstance(_p01_rb, dict):
                            for _b in (_p01_rb.get("bases") or []):
                                if _b:
                                    _learned_bases_to_avoid.add(str(_b).lower())
                        elif isinstance(_p01_rb, str) and _p01_rb:
                            _learned_bases_to_avoid.add(_p01_rb.lower())

                    for _p01_lesson in _p01_recent:
                        if not isinstance(_p01_lesson, dict):
                            continue
                        for _p01_rb in (_p01_lesson.get("repeated_bases") or []):
                            if isinstance(_p01_rb, dict):
                                for _b in (_p01_rb.get("bases") or []):
                                    if _b:
                                        _learned_bases_to_avoid.add(str(_b).lower())
                            elif isinstance(_p01_rb, str) and _p01_rb:
                                _learned_bases_to_avoid.add(_p01_rb.lower())

                    for _b in (_p01_lifetime.get("top_repeated_bases") or []):
                        if _b:
                            _learned_bases_to_avoid.add(str(_b).lower())

                    # Cap: evitar sobre-restringir el pool
                    _learned_bases_to_avoid = set(list(_learned_bases_to_avoid)[:8])
                    _shuffle_learning_applied = False

                    if _learned_bases_to_avoid:
                        def _day_is_high_fatigue(day: dict) -> bool:
                            meals = day.get("meals", [])
                            if not meals:
                                return False
                            for meal in meals:
                                meal_text = (
                                    meal.get("name", "") + " " +
                                    " ".join(meal.get("ingredients", []))
                                ).lower()
                                if not any(base in meal_text for base in _learned_bases_to_avoid):
                                    return False
                            return True

                        _low_fatigue_pool = [d for d in safe_pool if not _day_is_high_fatigue(d)]
                        if _low_fatigue_pool:
                            _excluded = len(safe_pool) - len(_low_fatigue_pool)
                            if _excluded > 0:
                                logger.info(
                                    f"[P0-1/SHUFFLE-LEARNING] Chunk {week_number} plan {meal_plan_id}: "
                                    f"excluidos {_excluded} días con bases fatigadas {_learned_bases_to_avoid}. "
                                    f"Pool: {len(safe_pool)} → {len(_low_fatigue_pool)}."
                                )
                            safe_pool = _low_fatigue_pool
                            _shuffle_learning_applied = True
                        else:
                            logger.warning(
                                f"[P0-1/SHUFFLE-LEARNING] Chunk {week_number} plan {meal_plan_id}: "
                                f"todos los días tienen bases fatigadas {_learned_bases_to_avoid}. "
                                f"Conservando pool original (shuffle_with_learning=fallback)."
                            )

                    pantry_filtered_pool = _filter_days_by_fresh_pantry(
                        safe_pool,
                        form_data.get("current_pantry_ingredients", []),
                    )
                    if pantry_filtered_pool:
                        if len(pantry_filtered_pool) < len(safe_pool):
                            logger.info(
                                f"[P1-2/PANTRY] Smart Shuffle filtró {len(safe_pool) - len(pantry_filtered_pool)} "
                                f"días por baja cobertura de inventario fresco."
                            )
                        safe_pool = pantry_filtered_pool
                    elif safe_pool:
                        _current_pantry = form_data.get("current_pantry_ingredients", [])
                        # [P0-4] Module-level import at cron_tasks.py:38 already provides this.
                        # Re-importing locally would shadow the module-level binding for the
                        # entire `_chunk_worker` scope, raising UnboundLocalError at L13359
                        # (the [P1-1/PANTRY-EMPTY] log path that runs before this branch).
                        if _count_meaningful_pantry_items(_current_pantry) >= CHUNK_MIN_FRESH_PANTRY_ITEMS:
                            logger.warning(
                                f"[P0-2/PANTRY] Smart Shuffle sin cobertura para inventario fresco ({len(_current_pantry)} items). "
                                f"Pausando chunk en lugar de ignorar despensa."
                            )
                            _pause_chunk_for_pantry_refresh(task_id, user_id, week_number, _current_pantry, reason="degraded_no_pantry_coverage")
                            return

                        logger.warning(
                            f"[P1-2/PANTRY] Ningún día del Smart Shuffle quedó mayoritariamente cubierto por "
                            f"el inventario fresco de {user_id}. Se conservan fallbacks previos."
                        )
                    
                    # [GAP 4 FIX: Variedad en Smart Shuffle]
                    backup_plan = execute_sql_query(
                        "SELECT health_profile->'emergency_backup_plan' as backup FROM user_profiles WHERE id = %s",
                        (user_id,), fetch_one=True
                    )
                    backup_days = backup_plan.get('backup', []) if backup_plan else []
                    used_meal_names = set()
                
                    last_chosen_hash = None
                    fallback_failed = False

                    for _shuffle_idx in range(days_count):
                        available_days = [d for d in safe_pool if str([m.get('name') for m in d.get('meals', [])]) != last_chosen_hash]
                    
                        if not available_days:
                            if backup_days:
                                available_days = [d for d in backup_days if str([m.get('name') for m in d.get('meals', [])]) != last_chosen_hash]

                        # [GAP 6] Ultimo recurso: si la restriccion de no-repetir-consecutivo no puede
                        # satisfacerse, inyectar una "Edge Recipe" del catalogo global antes de repetir.
                        is_emergency_repeat = False
                        is_edge_recipe = False
                    
                        if not available_days:
                            try:
                                edge_day = _build_filtered_edge_recipe_day(
                                    current_allergies,
                                    current_dislikes,
                                    current_diet,
                                    pantry_items=_fresh_pantry_for_edge,  # [P0-C FIX]
                                )
                                # Validar que no estemos repitiendo el hash por accidente
                                if edge_day and str([m.get('name') for m in edge_day.get('meals', [])]) != last_chosen_hash:
                                    logger.info(f"[GAP6/CHUNK] Smart Shuffle sin variedad para {user_id}: inyectando Edge Recipe filtrado.")
                                    available_days = [edge_day]
                                    is_edge_recipe = True
                            except Exception as e:
                                logger.error(f"[GAP6] Error generando Edge Recipe: {e}")

                        if not available_days:
                            # Si Edge Recipe fallo, caer en el ultimo recurso de repeticion
                            repeat_pool = safe_pool or backup_days
                            if repeat_pool:
                                logger.warning(
                                    f"[GAP6/CHUNK] Smart Shuffle sin variedad para {user_id}: "
                                    f"permitiendo repeticion consecutiva como ultimo recurso."
                                )
                                available_days = repeat_pool
                                is_emergency_repeat = True
                            else:
                                logger.error(f"[CHUNK] Smart Shuffle fallo para {user_id}: pool vacio, no hay dias a repetir.")
                                fallback_failed = True
                                break
                        shuffled_day = copy.deepcopy(random.choice(available_days))
                    
                        # [P0-NEW1] Validate quantities against pantry for ALL day types in degraded mode.
                        # [P0-4] Do NOT re-import CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE here:
                        # binding it locally at this Smart-Shuffle-only line shadowed the
                        # module-level import (cron_tasks.py:69) for the entire `_chunk_worker`
                        # scope, raising UnboundLocalError at L14868 when the LLM branch (which
                        # never executes this line) tried to read it. Keep `validate_ingredients_against_pantry`
                        # as a local import: it has no module-level alias and the LLM branch
                        # has its own local imports of the same name in scope before reading it.
                        from constants import validate_ingredients_against_pantry
                        _pantry_snap = form_data.get("current_pantry_ingredients", [])
                        _qty_validated = False
                        _shuffle_qty_attempts = 0
                        _max_shuffle_qty_attempts = 3
                        _fell_to_edge = is_edge_recipe
                    
                        # [P1-D] Resolver tolerance per-usuario una vez por loop
                        # de validación. Override en `user_profiles.pantry_tolerance` o
                        # default global si NULL/no-leíble. Cualquier excepción cae al
                        # default global vía el helper.
                        _p1d_tolerance = _get_pantry_tolerance_for_user(user_id)
                        while not _qty_validated and _shuffle_qty_attempts < _max_shuffle_qty_attempts:
                            _shuffled_ing = [
                                ing for m in shuffled_day.get('meals', [])
                                for ing in m.get('ingredients', []) if isinstance(ing, str) and ing.strip()
                            ]
                            _qty_check = validate_ingredients_against_pantry(
                                _shuffled_ing,
                                _pantry_snap,
                                strict_quantities=True,
                                tolerance=_p1d_tolerance,
                            )
                            if _qty_check is True:
                                _qty_validated = True
                                break
                        
                            _shuffle_qty_attempts += 1
                            logger.debug(f" [SHUFFLE-QTY] Candidato falló validación de cantidad: {_qty_check}")
                        
                            # Remove the failing candidate and try another
                            available_days = [d for d in available_days if d is not shuffled_day]
                            if not available_days:
                                break
                            shuffled_day = copy.deepcopy(random.choice(available_days))
                            _fell_to_edge = False # Reset flag if we pick a new day
                            is_emergency_repeat = False # Also reset this since it's a new day
                            is_edge_recipe = False

                        # Fallback to quantity-aware Edge Recipe if all 3 attempts fail
                        if not _qty_validated:
                            logger.warning(f"[P0-NEW1] Smart Shuffle falló {_shuffle_qty_attempts} intentos de cantidades, intentando Edge Recipe.")
                            try:
                                edge_day = _build_filtered_edge_recipe_day(
                                    current_allergies,
                                    current_dislikes,
                                    current_diet,
                                    pantry_items=_pantry_snap,
                                )
                                if edge_day:
                                    _edge_ing = [
                                        ing for m in edge_day.get('meals', [])
                                        for ing in m.get('ingredients', []) if isinstance(ing, str) and ing.strip()
                                    ]
                                    # [P1-D] Reusar tolerance per-usuario resuelto arriba.
                                    _edge_qty_check = validate_ingredients_against_pantry(
                                        _edge_ing, _pantry_snap, strict_quantities=True, tolerance=_p1d_tolerance
                                    )
                                    if _edge_qty_check is True:
                                        shuffled_day = edge_day
                                        is_edge_recipe = True
                                        _fell_to_edge = True
                                        _qty_validated = True
                                    else:
                                        logger.warning(f"[P0-NEW1] Edge Recipe también falló cantidades: {_edge_qty_check}")
                            except Exception as e:
                                logger.error(f"[P0-NEW1] Error construyendo Edge Recipe con cantidades: {e}")

                        # If still unfeasible, pause the chunk
                        if not _qty_validated:
                            logger.warning(
                                f"[P0-NEW1/SHUFFLE-QTY] plan={meal_plan_id} chunk={week_number} "
                                f"candidate_attempts={_shuffle_qty_attempts} fallback=pause "
                                f"reason=degraded_quantities_unfeasible"
                            )
                            _pause_chunk_for_pantry_refresh(task_id, user_id, week_number, _pantry_snap, reason="degraded_quantities_unfeasible")
                            return
                    
                        logger.info(
                            f"[P0-NEW1/SHUFFLE-QTY] plan={meal_plan_id} chunk={week_number} "
                            f"day_idx={_shuffle_idx} candidate_attempts={_shuffle_qty_attempts} "
                            f"fallback={'edge' if _fell_to_edge else 'none'}"
                        )

                        last_chosen_hash = str([m.get('name') for m in shuffled_day.get('meals', [])])
                    
                        meals = shuffled_day.get('meals', [])
                        for m_idx in range(len(meals)):
                            meal_name = meals[m_idx].get('name', '')
                            if meal_name in used_meal_names and backup_days:
                                # Swap con un meal del backup pool
                                backup_meals = [m for bd in backup_days for m in bd.get('meals', []) if m.get('name') not in used_meal_names]
                                if backup_meals:
                                    meals[m_idx] = copy.deepcopy(random.choice(backup_meals))
                                    shuffled_day['_mutated'] = True
                            used_meal_names.add(meals[m_idx].get('name', ''))
                    
                        shuffled_day['_is_degraded_shuffle'] = True
                        # [GAP C] Marcar el tier de calidad del día generado
                        # [P1-3 FIX] Añadir _degraded_fallback_level por día para
                        # distinguir "fallback funcionó" vs "fallback degradado".
                        if is_edge_recipe:
                            shuffled_day['_is_edge_recipe'] = True
                            shuffled_day['quality_tier'] = 'edge'
                            shuffled_day['_degraded_fallback_level'] = 'degraded'
                        elif is_emergency_repeat:
                            shuffled_day['_is_emergency_repeat'] = True
                            shuffled_day['quality_tier'] = 'emergency'
                            shuffled_day['_degraded_fallback_level'] = 'degraded'
                        else:
                            shuffled_day['quality_tier'] = 'shuffle'
                            shuffled_day['_degraded_fallback_level'] = 'functional'
                        # [GAP 3] Renumerar al dia absoluto que corresponde a este chunk.
                        # Sin esto, el dia shuffled conserva su 'day' original (ej. 1) y al mergear
                        # sobrescribiria el dia 1 del plan en vez de agregar dia 4/5/6.
                        shuffled_day['day'] = days_offset + _shuffle_idx + 1
                        # [GAP 3 FIX]: Calcular day_name exacto usando la fecha de inicio del plan
                        start_date_str = snap.get("form_data", {}).get("_plan_start_date")
                        if start_date_str:
                            from constants import safe_fromisoformat
                            # [P0-1-RECOVERY/WORKER-FIX] timedelta es global del módulo.
                            try:
                                dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                                start_dt = safe_fromisoformat(start_date_str)
                                target_date = start_dt + timedelta(days=shuffled_day['day'] - 1)
                                shuffled_day['day_name'] = dias_es[target_date.weekday()]
                            except Exception:
                                shuffled_day['day_name'] = ""
                        else:
                            shuffled_day['day_name'] = ""
                        new_days.append(shuffled_day)
                    
                    if fallback_failed:
                        # [P1-4] CAS guard contra zombie rescue + nuevo pickup.
                        _cas_update_chunk_status(task_id, _pickup_attempts, "failed")
                        return

                    # [P1-5] Validación de meal_count por día: rechazar días con `meals` vacío o
                    # con menos comidas que el día más rico del pool (heurística: usar el max de
                    # safe_pool/backup_days como expected). Smart Shuffle puede producir días
                    # vacíos cuando todos los candidatos son edge recipes con catálogo restringido,
                    # pantry items que no intersectan con el catálogo, o pools degradados que se
                    # quedaron sin contenido. Antes el día vacío se persistía silenciosamente y
                    # el usuario veía un día sin comidas en su plan.
                    #
                    # Estrategia:
                    #   1. Calcular expected = max(meals_count) en safe_pool/backup_days, o 3.
                    #   2. Detectar días con meals_count < expected (subset estricto).
                    #   3. Para cada día insuficiente, intentar reemplazarlo con un edge_day sin
                    #      restricción de pantry (pantry_items=None) — más permisivo que el flujo
                    #      original que limitaba al inventario disponible.
                    #   4. Si el reemplazo tampoco alcanza, pausar el chunk con
                    #      _pause_reason='shuffle_empty_day' en lugar de persistir días vacíos.
                    def _p15_meal_count(d):
                        if not isinstance(d, dict):
                            return 0
                        return len([
                            m for m in (d.get("meals") or [])
                            if isinstance(m, dict) and m.get("name")
                        ])

                    _p15_pool_for_expected = (safe_pool or []) + (backup_days or [])
                    _p15_pool_counts = [_p15_meal_count(d) for d in _p15_pool_for_expected[:10]]
                    _p15_pool_counts = [c for c in _p15_pool_counts if c > 0]
                    _p15_expected_meals = max(_p15_pool_counts) if _p15_pool_counts else 3

                    _p15_insufficient = [
                        i for i, d in enumerate(new_days)
                        if _p15_meal_count(d) < _p15_expected_meals
                    ]
                    if _p15_insufficient:
                        logger.warning(
                            f"[P1-5/EMPTY-DAY] Smart Shuffle produjo {len(_p15_insufficient)}/{len(new_days)} "
                            f"días con menos de {_p15_expected_meals} comidas para plan {meal_plan_id} "
                            f"chunk {week_number}. Intentando rescatar con edge_day sin restricción de pantry."
                        )
                        for _p15_idx in _p15_insufficient:
                            try:
                                _p15_edge = _build_filtered_edge_recipe_day(
                                    current_allergies,
                                    current_dislikes,
                                    current_diet,
                                    pantry_items=None,  # sin restricción → catálogo completo
                                )
                            except Exception as _p15_edge_err:
                                logger.error(
                                    f"[P1-5/EMPTY-DAY] Error generando edge_day rescue: {_p15_edge_err}"
                                )
                                _p15_edge = None
                            if _p15_edge and _p15_meal_count(_p15_edge) >= _p15_expected_meals:
                                # Preservar metadata del slot original.
                                _p15_orig_day = new_days[_p15_idx].get("day")
                                _p15_orig_day_name = new_days[_p15_idx].get("day_name", "")
                                _p15_edge["day"] = _p15_orig_day
                                _p15_edge["day_name"] = _p15_orig_day_name
                                _p15_edge["_is_degraded_shuffle"] = True
                                _p15_edge["_is_edge_recipe"] = True
                                _p15_edge["_p1_5_rescued"] = True
                                _p15_edge["quality_tier"] = "edge"
                                _p15_edge["_degraded_fallback_level"] = "degraded"
                                new_days[_p15_idx] = _p15_edge
                                logger.info(
                                    f"[P1-5/EMPTY-DAY] Rescatado day_idx={_p15_idx} (day={_p15_orig_day}) "
                                    f"con edge_day flex, meals_count={_p15_meal_count(_p15_edge)}."
                                )

                        # Re-validar tras intentar rescate. Si algún día sigue insuficiente, pausar.
                        _p15_still_insufficient = [
                            i for i, d in enumerate(new_days)
                            if _p15_meal_count(d) < _p15_expected_meals
                        ]
                        if _p15_still_insufficient:
                            _p15_sample = new_days[_p15_still_insufficient[0]]
                            logger.error(
                                f"[P1-5/EMPTY-DAY] {len(_p15_still_insufficient)} días siguen insuficientes "
                                f"tras rescue para plan {meal_plan_id} chunk {week_number}. "
                                f"Pausando con reason='shuffle_empty_day' para evitar persistir días vacíos. "
                                f"Sample: day={_p15_sample.get('day')} meals={_p15_meal_count(_p15_sample)}/"
                                f"{_p15_expected_meals}"
                            )
                            try:
                                _pause_chunk_for_pantry_refresh(
                                    task_id, user_id, week_number,
                                    form_data.get("current_pantry_ingredients", []),
                                    reason="shuffle_empty_day",
                                )
                            except Exception as _p15_pause_err:
                                logger.error(
                                    f"[P1-5/EMPTY-DAY] Pausa también falló: {_p15_pause_err}. "
                                    f"Marcando chunk failed."
                                )
                                try:
                                    execute_sql_write(
                                        "UPDATE plan_chunk_queue SET status = 'failed', "
                                        "dead_lettered_at = NOW(), "
                                        "dead_letter_reason = 'shuffle_empty_day_pause_failed' "
                                        "WHERE id = %s",
                                        (task_id,),
                                    )
                                except Exception:
                                    pass
                            return

                    # [P0-5][P0-4] Validar que el Smart Shuffle respete el pantry en estricto antes de mergear.
                    # Antes usaba strict_quantities=False (solo existencia), lo que dejaba pasar días
                    # con cantidades que excedían la nevera (e.g. 500g de pollo cuando solo hay 200g).
                    # Ahora endurecido a strict_quantities=True con tolerance=1.0 para alinear con el
                    # comportamiento del path LLM y la hard validation post-merge.
                    from constants import validate_ingredients_against_pantry
                    _shuffle_validation_failed_count = 0
                    for _nd in new_days:
                        _day_ingredients = []
                        for _m in _nd.get('meals', []):
                            _day_ingredients.extend([ing for ing in _m.get('ingredients', []) if isinstance(ing, str) and ing.strip()])

                        _val_res = validate_ingredients_against_pantry(
                            _day_ingredients, form_data.get("current_pantry_ingredients", []),
                            strict_quantities=True, tolerance=1.0,
                        )
                        if _val_res is True:
                            _val_res = validate_ingredients_against_pantry(
                                _day_ingredients, _pantry_snap,
                                strict_quantities=True, tolerance=1.0,
                            )
                        if _val_res is not True:
                            _nd["_shuffle_validation_failed"] = True
                            _shuffle_validation_failed_count += 1

                    if _shuffle_validation_failed_count > 0:
                        logger.warning(
                            f"[P0-5/SHUFFLE] {_shuffle_validation_failed_count} días fallaron "
                            f"validación estricta de inventario post-shuffle (strict_quantities=True). "
                            f"Abortando shuffle y cayendo a IA."
                        )
                        # Caer a IA desactivando is_degraded y reseteando new_days
                        is_degraded = False
                        new_days = []
                        continue

                    # [P0-A FIX] Calcular learning_metrics en Smart Shuffle para no romper la cadena
                    # de aprendizaje continuo. Sin esto, _last_chunk_learning y _recent_chunk_lessons
                    # nunca se actualizan cuando el chunk cae en modo degraded, rompiendo el contexto
                    # de aprendizaje de todos los chunks LLM que vengan después.
                    try:
                        prior_days_for_metrics = prior_plan_data.get("days", [])
                        prior_meals_for_metrics = [
                            m.get("name") for d in prior_days_for_metrics
                            for m in (d.get("meals") or [])
                            if m.get("name") and m.get("status") not in ["swapped_out", "skipped", "rejected"]
                        ]
                        rechazos_shuffle = get_active_rejections(user_id=user_id)
                        rejected_names_for_metrics = [r["meal_name"] for r in rechazos_shuffle] if rechazos_shuffle else []
                        learning_metrics = _calculate_learning_metrics(
                            new_days=new_days,
                            prior_meals=prior_meals_for_metrics,
                            prior_days=prior_days_for_metrics,
                            rejected_names=rejected_names_for_metrics,
                            allergy_keywords=[],
                            fatigued_ingredients=[],
                        )
                        # [P0-1] Anotar si el filtro de aprendizaje se aplicó efectivamente.
                        learning_metrics["shuffle_learning_applied"] = _shuffle_learning_applied
                        learning_metrics["shuffle_source"] = (
                            "shuffle_with_learning" if _shuffle_learning_applied else "shuffle_naive"
                        )
                        logger.info(
                            f"[P0-A/MEASURE] source={learning_metrics['shuffle_source']} "
                            f"plan={meal_plan_id} chunk={week_number} "
                            f"repeat_pct={learning_metrics.get('learning_repeat_pct', 'N/A')}% "
                            f"ingredient_base_repeat_pct={learning_metrics.get('ingredient_base_repeat_pct', 'N/A')}% "
                            f"rejection_violations={learning_metrics.get('rejection_violations', 'N/A')}"
                        )
                    except Exception as _p0a_e:
                        logger.warning(f"[P0-A] Error calculando learning_metrics en Smart Shuffle: {_p0a_e}")
                        learning_metrics = None
                else:
                    # [FIX CRÃTICO 3 â€" Anti-repeticion cross-semanas]:
                    # NO usar snap["previous_meals"] (solo tiene semana 1). Releer TODOS los platos
                    # ya generados del plan actual en DB. Esto cubre el caso donde los chunks
                    # de semanas 2, 3, 4 se procesan en orden o se solapan.
                    if not plan_row_prior:
                        raise Exception(f"Plan {meal_plan_id} no encontrado al leer contexto")
                    prior_days = prior_plan_data.get("days", []) or []
                    prior_meals = [
                        m.get("name") for d in prior_days
                        for m in (d.get("meals") or []) 
                        if m.get("name") and m.get("status") not in ["swapped_out", "skipped", "rejected"]
                    ]
                    # [P1-6] Helpers tipo-seguros para defensa contra plan_data corrupto.
                    last_chunk_learning = _safe_lessons_dict(
                        prior_plan_data.get("_last_chunk_learning"),
                        field_name="_last_chunk_learning", plan_id=meal_plan_id,
                        user_id=user_id,
                    )
                    # [P0-4] Ventana rolling de lecciones: chunk N+1 hereda solo el último;
                    # con _recent_chunk_lessons, chunk N+k hereda hasta 4 chunks anteriores.
                    recent_chunk_lessons = _safe_lessons_list(
                        prior_plan_data.get("_recent_chunk_lessons"),
                        field_name="_recent_chunk_lessons", plan_id=meal_plan_id,
                        user_id=user_id,
                    )

                    # [P1-1] Excluir lecciones de chunks dead-lettered (recovery_exhausted).
                    # Esos chunks no completaron validación final; sus lecciones serían
                    # "fantasmas" y propagarlas amplifica el error original. Si el inmediato
                    # predecesor estaba dead-lettered, perdemos `_last_chunk_learning` y
                    # forzamos variedad para que el LLM no caiga en repetición ciega.
                    last_chunk_learning, recent_chunk_lessons, _p11_dead_weeks = (
                        _filter_lessons_excluding_dead_lettered(
                            last_chunk_learning,
                            recent_chunk_lessons,
                            prior_plan_data,
                            week_number,
                        )
                    )
                    if _p11_dead_weeks:
                        form_data["_failed_chunk_learning_disabled"] = True
                        form_data["_failed_chunk_predecessor_weeks"] = _p11_dead_weeks
                        form_data["_force_variety"] = True
                        logger.warning(
                            f"[P1-1/SKIP-FANTASMA] plan={meal_plan_id} chunk={week_number} "
                            f"omitiendo lecciones de chunks dead-lettered={_p11_dead_weeks}. "
                            f"last_chunk_learning_kept={bool(last_chunk_learning)} "
                            f"recent_chunk_lessons_kept={len(recent_chunk_lessons)}"
                        )

                    # Construir form_data para el pipeline con el offset correcto
                    # form_data fue inicializado arriba
                    form_data["_days_offset"] = days_offset
                    form_data["_days_to_generate"] = days_count
                    # Platos de TODOS los dias previos (no solo semana 1) para anti-repeticion real
                    form_data["_chunk_prior_meals"] = prior_meals
                    form_data["previous_meals"] = prior_meals

                    # [P0-4] _force_variety si CUALQUIER lección de la ventana superó el umbral.
                    _any_high_repeat = float(last_chunk_learning.get("ingredient_base_repeat_pct") or 0) > 60.0
                    if not _any_high_repeat:
                        for _lesson in recent_chunk_lessons:
                            if float((_lesson or {}).get("ingredient_base_repeat_pct") or 0) > 60.0:
                                _any_high_repeat = True
                                break
                    if _any_high_repeat:
                        # [P0-gamma] Si la nevera está agotada, forzar técnica en vez de ingredientes
                        if form_data.get("_pantry_diversity_warning"):
                            form_data["_force_technique_variety"] = True
                            logger.warning(
                                f"[P0-gamma] _force_variety downgradeado a _force_technique_variety por baja diversidad en nevera."
                            )
                        else:
                            form_data["_force_variety"] = True
                            logger.warning(
                                f"[P0-4/FORCE-VARIETY] Activando _force_variety para chunk {week_number} "
                                f"por ingredient_base_repeat_pct alta en ventana rolling "
                                f"(último={last_chunk_learning.get('ingredient_base_repeat_pct')}%)"
                            )

                    # [P0-4 / P1-5 / P1-A] Propagar lecciones agregadas de la ventana rolling al LLM.
                    # Agrega: repeated_bases (unión), violaciones (suma), repeat_pct (máx).
                    # [P1-A] Ahora incluye también _lifetime_lessons_summary si existe.
                    # [P0-7] Mergear _critical_lessons_permanent (señales que sobreviven al rolling window).
                    _critical_permanent = prior_plan_data.get("_critical_lessons_permanent", [])
                    if not isinstance(_critical_permanent, list):
                        _critical_permanent = []
                    # Dedup: no incluir critical lessons que ya estén en recent_chunk_lessons por chunk number
                    _recent_chunk_nums = set()
                    for _rcl in recent_chunk_lessons:
                        if isinstance(_rcl, dict) and _rcl.get("chunk") is not None:
                            _recent_chunk_nums.add(_rcl["chunk"])
                    if last_chunk_learning and last_chunk_learning.get("chunk") is not None:
                        _recent_chunk_nums.add(last_chunk_learning["chunk"])
                    _critical_extras = [
                        cl for cl in _critical_permanent
                        if isinstance(cl, dict) and cl.get("chunk") not in _recent_chunk_nums
                    ]
                    if _critical_extras:
                        logger.info(
                            f"[P0-7/INJECT] Chunk {week_number}: inyectando {len(_critical_extras)} lecciones "
                            f"críticas permanentes de chunks {[c.get('chunk') for c in _critical_extras]} "
                            f"que ya salieron del rolling window."
                        )

                    _all_lessons = ([last_chunk_learning] if last_chunk_learning else []) + recent_chunk_lessons + _critical_extras
                    _all_lessons_filtered = [l for l in _all_lessons if not l.get("metrics_unavailable")]
                    _high_conf = [l for l in _all_lessons_filtered if not l.get("low_confidence")]
                    if _high_conf:
                        _all_lessons_filtered = _high_conf
                    if _all_lessons and not _all_lessons_filtered:
                        form_data["_learning_window_starved"] = True
                        form_data["_force_variety"] = True
                    _all_lessons = _all_lessons_filtered
                
                    # [P1-6] Tipo-seguro: si _lifetime_lessons_summary llega corrompido como list
                    # o str, el helper devuelve {} y el fallback al snapshot heredado entra en juego.
                    # También protegemos el chain `_inherited_lifetime_lessons.summary` por si
                    # `_inherited_lifetime_lessons` no es dict (cortocircuita el AttributeError).
                    _inherited_blob = _safe_lessons_dict(
                        snap.get("_inherited_lifetime_lessons"),
                        field_name="_inherited_lifetime_lessons", plan_id=meal_plan_id,
                        user_id=user_id,
                    )
                    lifetime_summary = (
                        _safe_lessons_dict(
                            prior_plan_data.get("_lifetime_lessons_summary"),
                            field_name="_lifetime_lessons_summary", plan_id=meal_plan_id,
                            user_id=user_id,
                        )
                        or _safe_lessons_dict(
                            _inherited_blob.get("summary"),
                            field_name="_inherited_lifetime_lessons.summary", plan_id=meal_plan_id,
                            user_id=user_id,
                        )
                    )

                    # [P1-3] Bootstrap de aprendizaje para chunks tempranos con ventana
                    # débil. Caso típico: chunk 2 de primer plan 7d con usuario que tiene
                    # rechazos del onboarding pero sin _lifetime_lessons heredado (P0-1
                    # solo aplica si hay plan previo). Sin esto, el LLM solo recibe
                    # _last_chunk_learning (1 chunk) — pierde contexto de rechazos
                    # explícitos y patrones históricos de los últimos 6 meses.
                    if (
                        not lifetime_summary
                        and len(recent_chunk_lessons) == 0
                        and int(week_number) >= 2
                    ):
                        _p13_synth = _synthesize_user_history_lifetime_summary(user_id)
                        if _p13_synth:
                            lifetime_summary = _p13_synth
                            logger.info(
                                f"[P1-3/SYNTH-HISTORY] chunk {week_number} plan {meal_plan_id}: "
                                f"_lifetime_lessons_summary sintetizado desde user history "
                                f"(rejections={len(_p13_synth.get('top_rejection_hits') or [])}, "
                                f"bases={len(_p13_synth.get('top_repeated_bases') or [])}, "
                                f"meal_names={len(_p13_synth.get('top_repeated_meal_names') or [])}). "
                                f"Trigger: ventana_rolling=0 y sin lifetime heredado."
                            )

                    if _all_lessons or lifetime_summary:
                        _agg_repeated_bases: list = []
                        _agg_repeated_bases_seen: set = set()
                        _agg_repeated_meals: list = []
                        _agg_repeated_meals_seen: set = set()
                        _agg_rejected_meals: list = []
                        _agg_rejected_meals_seen: set = set()
                        _agg_allergy_hits: list = []
                        _agg_allergy_hits_seen: set = set()
                        _agg_rej_viol = 0
                        _agg_alg_viol = 0
                        _agg_max_repeat_pct = 0.0
                        _agg_max_base_repeat_pct = 0.0
                        _lesson_chunks: list = []

                        for _lesson in _all_lessons:
                            if not isinstance(_lesson, dict):
                                continue
                            _lesson_chunks.append(_lesson.get("chunk"))
                            _agg_rej_viol += int(_lesson.get("rejection_violations") or 0)
                            _agg_alg_viol += int(_lesson.get("allergy_violations") or 0)
                            _agg_max_repeat_pct = max(_agg_max_repeat_pct, float(_lesson.get("repeat_pct") or 0))
                            _agg_max_base_repeat_pct = max(_agg_max_base_repeat_pct, float(_lesson.get("ingredient_base_repeat_pct") or 0))
                            for _rb in (_lesson.get("repeated_bases") or []):
                                _key = str(_rb)
                                if _key not in _agg_repeated_bases_seen:
                                    _agg_repeated_bases_seen.add(_key)
                                    _agg_repeated_bases.append(_rb)
                            for _rm in (_lesson.get("repeated_meal_names") or []):
                                if _rm not in _agg_repeated_meals_seen:
                                    _agg_repeated_meals_seen.add(_rm)
                                    _agg_repeated_meals.append(_rm)
                            for _rj in (_lesson.get("rejected_meals_that_reappeared") or []):
                                if _rj not in _agg_rejected_meals_seen:
                                    _agg_rejected_meals_seen.add(_rj)
                                    _agg_rejected_meals.append(_rj)
                            for _ah in (_lesson.get("allergy_hits") or []):
                                if _ah not in _agg_allergy_hits_seen:
                                    _agg_allergy_hits_seen.add(_ah)
                                    _agg_allergy_hits.append(_ah)

                        # [P1-A] Inyectar datos del lifetime_summary si son más severos o adicionales
                        if lifetime_summary:
                            _agg_rej_viol = max(_agg_rej_viol, int(lifetime_summary.get("total_rejection_violations") or 0))
                            _agg_alg_viol = max(_agg_alg_viol, int(lifetime_summary.get("total_allergy_violations") or 0))

                            # Agregar ingredientes rechazados históricos que no estén en la ventana actual
                            for _hist_rj in (lifetime_summary.get("top_rejection_hits") or []):
                                if _hist_rj not in _agg_rejected_meals_seen:
                                    _agg_rejected_meals_seen.add(_hist_rj)
                                    _agg_rejected_meals.append(_hist_rj)

                            # Agregar bases históricas críticas
                            for _hist_rb in (lifetime_summary.get("top_repeated_bases") or []):
                                _key = str(_hist_rb)
                                if _key not in _agg_repeated_bases_seen:
                                    _agg_repeated_bases_seen.add(_key)
                                    _agg_repeated_bases.append(_hist_rb)

                            # [P1-5] Inyectar meals que aparecen en >=2 chunks (permanent blocklist)
                            # PRIMERO — tienen prioridad por encima de single-chunk repeats.
                            # Sin esto, plan 30d perdía meals repetidos del chunk 1 cuando salían
                            # del rolling window (cap 8) y el LLM podía regenerarlos en chunk 9+.
                            for _hist_pb in (lifetime_summary.get("permanent_meal_blocklist") or []):
                                if _hist_pb and _hist_pb not in _agg_repeated_meals_seen:
                                    _agg_repeated_meals_seen.add(_hist_pb)
                                    _agg_repeated_meals.append(_hist_pb)
                            # Y meals repetidos en algún chunk pasado (single-chunk repeats).
                            for _hist_rm in (lifetime_summary.get("top_repeated_meal_names") or []):
                                if _hist_rm and _hist_rm not in _agg_repeated_meals_seen:
                                    _agg_repeated_meals_seen.add(_hist_rm)
                                    _agg_repeated_meals.append(_hist_rm)

                        _weak_signal_present = any(
                            isinstance(_lesson, dict) and _lesson.get("learning_signal_strength") == "weak"
                            for _lesson in _all_lessons
                        )
                        _has_actionable = (
                            _agg_repeated_bases
                            or _agg_rej_viol > 0
                            or _agg_alg_viol > 0
                            or lifetime_summary
                            or _weak_signal_present
                        )
                        # [P1-B] Compresión de lecciones expulsadas del rolling window.
                        # Los chunks que ya no caben en `_recent_chunk_lessons` (cap 8) ni en
                        # `_critical_lessons_permanent` ni en `last_chunk_learning` no aparecen
                        # en el prompt en detalle. Existen en `_lifetime_lessons_history` y se
                        # agregan a `_lifetime_lessons_summary`, pero el LLM no recibe señal
                        # explícita de cuántos chunks están "ocultos" tras la agregación.
                        # Sin esto, el modelo puede subestimar el peso histórico de las
                        # repeticiones reportadas. Computamos:
                        #   - count: número de chunks en lifetime_history NO mostrados en detalle.
                        #   - max_repeat_pct / avg_repeat_pct sobre esos chunks (telemetría
                        #     compacta para que el prompt builder pueda renderizar contexto).
                        _p1b_history_raw = prior_plan_data.get("_lifetime_lessons_history") or []
                        if not isinstance(_p1b_history_raw, list):
                            _p1b_history_raw = []
                        _p1b_shown_chunks: set = set()
                        for _l in _all_lessons:
                            if isinstance(_l, dict) and _l.get("chunk") is not None:
                                _p1b_shown_chunks.add(_l.get("chunk"))
                        _p1b_compressed_lessons = [
                            _l for _l in _p1b_history_raw
                            if isinstance(_l, dict)
                            and _l.get("chunk") is not None
                            and _l.get("chunk") not in _p1b_shown_chunks
                        ]
                        _p1b_compressed_count = len(_p1b_compressed_lessons)
                        if _p1b_compressed_count > 0:
                            _p1b_repeat_pcts = [
                                float(_l.get("repeat_pct") or 0)
                                for _l in _p1b_compressed_lessons
                            ]
                            _p1b_max_repeat_pct = max(_p1b_repeat_pcts) if _p1b_repeat_pcts else 0.0
                            _p1b_avg_repeat_pct = (
                                sum(_p1b_repeat_pcts) / len(_p1b_repeat_pcts)
                                if _p1b_repeat_pcts else 0.0
                            )
                            _p1b_max_violations = max(
                                int(_l.get("rejection_violations") or 0)
                                + int(_l.get("allergy_violations") or 0)
                                + int(_l.get("fatigued_violations") or 0)
                                for _l in _p1b_compressed_lessons
                            )
                        else:
                            _p1b_max_repeat_pct = 0.0
                            _p1b_avg_repeat_pct = 0.0
                            _p1b_max_violations = 0

                        if _has_actionable:
                            # [P1-5] Cap dinámico para repeated_meal_names: total_chunks * 3 con
                            # tope de 30. Plan 30d (~10 chunks) → cap 30; plan 7d (2 chunks) → cap 10.
                            # Antes era hardcoded a 10, lo que truncaba la mitad de los repeats en 30d
                            # cuando había muchos meals que reaparecieron a través de los chunks.
                            try:
                                _p15_total_days = int(prior_plan_data.get("total_days_requested") or 0)
                            except (TypeError, ValueError):
                                _p15_total_days = 0
                            _p15_max_chunks = max(2, (_p15_total_days + 2) // 3)  # ceil(total/3)
                            _p15_meal_cap = min(30, max(10, _p15_max_chunks * 3))
                            form_data["_chunk_lessons"] = {
                                "chunk_numbers": _lesson_chunks,
                                "ingredient_base_repeat_pct": _agg_max_base_repeat_pct,
                                "repeated_bases": _agg_repeated_bases[:12],  # Aumentado un poco para P1-A
                                "repeat_pct": _agg_max_repeat_pct,
                                "repeated_meal_names": _agg_repeated_meals[:_p15_meal_cap],
                                # [P1-5] Permanent blocklist como campo distinto (high-priority).
                                # El prompt builder puede tratarlo como "NUNCA generar" vs el más
                                # blando "evitar repetir" de repeated_meal_names.
                                "permanent_meal_blocklist": (
                                    (lifetime_summary or {}).get("permanent_meal_blocklist") or []
                                )[:50],
                                # [P1-5] Lifetime fields EXPLÍCITOS para que el prompt builder pueda
                                # emitir bullets independientes de las métricas recientes. Antes el
                                # builder gateaba `rejected_meals_that_reappeared` en `rej_viol > 0`
                                # y `repeated_meal_names` en `repeat_pct > 15.0` — métricas calculadas
                                # SOLO sobre el rolling window. Si la ventana reciente estaba limpia
                                # pero lifetime acumulaba "pollo rechazado hace 5 chunks", el LLM
                                # nunca veía esa señal porque el bullet entero quedaba suprimido
                                # por el gate. Ahora exponemos `lifetime_top_rejection_hits` y
                                # `lifetime_top_repeated_meal_names` como campos separados, y el
                                # builder emite bullets dedicados con wording "ACUMULADO HISTÓRICO"
                                # incluso cuando recent está limpio.
                                "lifetime_top_rejection_hits": (
                                    (lifetime_summary or {}).get("top_rejection_hits") or []
                                )[:20],
                                "lifetime_top_repeated_meal_names": (
                                    (lifetime_summary or {}).get("top_repeated_meal_names") or []
                                )[:20],
                                "lifetime_top_repeated_bases": (
                                    (lifetime_summary or {}).get("top_repeated_bases") or []
                                )[:10],
                                "rejection_violations": _agg_rej_viol,
                                "rejected_meals_that_reappeared": _agg_rejected_meals[:12],
                                "allergy_violations": _agg_alg_viol,
                                "allergy_hits": _agg_allergy_hits[:10],
                                "is_lifetime_aggregated": bool(lifetime_summary),
                                "_lifetime_window_days": lifetime_summary.get("_lifetime_window_days") if lifetime_summary else None,
                                # [P1-B] Awareness de chunks históricos comprimidos: el LLM
                                # debe saber cuántos chunks pasados influyen en estas métricas
                                # aunque no aparezcan en detalle.
                                "compressed_history_chunks_count": _p1b_compressed_count,
                                "compressed_history_max_repeat_pct": round(_p1b_max_repeat_pct, 1),
                                "compressed_history_avg_repeat_pct": round(_p1b_avg_repeat_pct, 1),
                                "compressed_history_max_violations": _p1b_max_violations,
                            }

                            _lesson_signal_strength = None
                            if _weak_signal_present:
                                _lesson_signal_strength = "weak"
                            elif _all_lessons:
                                _lesson_signal_strength = "strong"
                            if _lesson_signal_strength:
                                form_data["_chunk_lessons"]["learning_signal_strength"] = _lesson_signal_strength
                                form_data["_chunk_lessons"]["weak_signal"] = _lesson_signal_strength == "weak"

                            if form_data.get("_pantry_diversity_warning"):
                                form_data["_chunk_lessons"]["pantry_diversity_warning"] = True

                            logger.info(
                                f"[P0-4] Lecciones agregadas chunks {_lesson_chunks} → chunk {week_number}: "
                                f"base_repeat_max={_agg_max_base_repeat_pct:.1f}% "
                                f"rej_viol={_agg_rej_viol} alg_viol={_agg_alg_viol} "
                                f"bases_únicas={len(_agg_repeated_bases)}"
                            )

                    # [GAP 1 FIX] / [P1-2 FIX]: Propagar la ultima tecnica del chunk anterior
                    # Leer desde la raiz (donde sobrevive a shifts) o fallback al _skeleton original
                    last_tech = prior_plan_data.get("last_technique")
                    if last_tech:
                        form_data["_last_technique"] = last_tech
                    else:
                        prior_skeleton = prior_plan_data.get("_skeleton", {})
                        prior_techniques = prior_skeleton.get("_selected_techniques", [])
                        if prior_techniques:
                            form_data["_last_technique"] = prior_techniques[-1]
                        
                    # [P0-4] Forzar rotación de proteínas si la señal de inventario es débil
                    if form_data.get("_inventory_activity_proxy_used"):
                        form_data["_force_variety"] = True
                        _banned_proteins = set()
                        for _d in prior_days:
                            for _p in _d.get("protein_pool", []):
                                _banned_proteins.add(_p)
                    
                        if "_chunk_lessons" not in form_data:
                            form_data["_chunk_lessons"] = {}
                        form_data["_chunk_lessons"]["learning_signal_strength"] = "weak"
                        form_data["_chunk_lessons"]["weak_signal"] = True
                        form_data["_chunk_lessons"]["banned_proteins"] = list(_banned_proteins)
                        logger.info(f"[P0-4/WEAK-SIGNAL] Chunk {week_number} fuerza variedad y banea proteínas: {list(_banned_proteins)}")
    
                    # [APRENDIZAJE JIT]: Recalcular perfil y memoria con datos frescos
                    logger.info(f" [CHUNK] Recalculando perfil y memoria Just-in-Time para usuario {user_id}...")
                    session_id = form_data.get("session_id") or snap.get("form_data", {}).get("session_id")
                    history = []

                    likes_actualizados = get_user_likes(user_id)
                    rechazos_nuevos = get_active_rejections(user_id=user_id)
                    rejected_meal_names = [r["meal_name"] for r in rechazos_nuevos] if rechazos_nuevos else []

                    if session_id:
                        try:
                            memory = build_memory_context(session_id)
                            history = memory.get("recent_messages", []) or []
                            memory_context = memory.get("full_context_str", "") or ""
                        except Exception as mem_e:
                            logger.warning(f"[P1-4] Error reconstruyendo memory_context desde session_id={session_id}: {mem_e}")
                            memory_context = _build_facts_memory_context(user_id)
                    else:
                        memory_context = _build_facts_memory_context(user_id)

                    # Re-evaluar gustos usando también la conversación reciente del plan actual
                    taste_profile = analyze_preferences_agent(likes_actualizados, history, active_rejections=rejected_meal_names)

                    # [GAP 1 - SEGURIDAD ALIMENTARIA]: Inyectar alergias aprendidas al snapshot
                    # Esto asegura que el self_critique_node evalue estrictamente las alergias nuevas.
                    from db_facts import get_user_facts_by_metadata
                    alergias_facts = get_user_facts_by_metadata(user_id, 'category', 'alergia')
                    if alergias_facts:
                        current_allergies = form_data.get('allergies', [])
                        if isinstance(current_allergies, str):
                            current_allergies = [current_allergies] if current_allergies.strip() else []
                    
                        for f in alergias_facts:
                            fact_text = f.get('fact', '')
                            if fact_text and fact_text not in current_allergies:
                                current_allergies.append(fact_text)
                    
                        form_data['allergies'] = current_allergies

                    # [GAP 1 FIX]: Inyectar seÃ±ales avanzadas (Quality Score, EMA, etc) en el chunk worker
                    from db_facts import get_consumed_meals_since
                    # [P0-1-RECOVERY/WORKER-FIX] No re-importar datetime/timezone/timedelta:
                    # ya están globales en el módulo. Importarlos aquí promovía a las tres
                    # a variables locales de _chunk_worker → UnboundLocalError en cualquier
                    # uso anterior (L6651 zero_log). Bug que causó el chunk a1a6025e-...
                    # [GAP 1 de 30 DÃAS FIX]: En vez de mirar solo 7 dias atras (lo que arruina el math para la semana 4),
                    # miramos desde que inicio el plan actual para calcular la adherencia exacta sobre todos los 'prior_days'.
                    # [GAP 3 FIX: _plan_start_date vs plan_start_date Inconsistencia]
                    # Usar la clave correcta con underscore (que es como plans.py lo guarda)
                    plan_start_date_str = snap.get("form_data", {}).get("_plan_start_date")
                    now_utc = datetime.now(timezone.utc)
                    if plan_start_date_str:
                        # [P1-4] Ventana basada en distancia temporal real (days_offset) en vez de chunk_kind.
                        # chunk 2 de 30d (offset=3) → 4 días; chunk 8 de 30d (offset=21) → 14 días (capped).
                        from constants import PLAN_CHUNK_SIZE, safe_fromisoformat
                        try:
                            start_dt = safe_fromisoformat(plan_start_date_str)
                            if start_dt.tzinfo is None:
                                start_dt = start_dt.replace(tzinfo=timezone.utc)
                            _consumed_window = min(max(PLAN_CHUNK_SIZE + 1, int(days_offset or 0)), 14)
                            window_start = now_utc - timedelta(days=_consumed_window)
                            since_time = max(start_dt, window_start).isoformat()
                        except Exception as e:
                            logger.warning(f"[P1-4] Error parseando plan_start_date: {e}")
                            since_time = plan_start_date_str
                    else:
                        since_time = (now_utc - timedelta(days=max(7, days_offset))).isoformat()
                    
                    chunk_consumed_records = get_consumed_meals_since(user_id, since_time)

                    # [P0-5] Inyectar adherencia chunk-específica del previo INMEDIATO al prompt.
                    # Diferencia con _meal_level_adherence (EMA por tipo, todo el historial):
                    # esto lista NOMBRES de platos que el usuario consumió/saltó del chunk N-1.
                    # El builder lo traduce a "refuerza variantes de X / evita repetir Y".
                    if int(week_number) >= 2 and prior_days:
                        try:
                            _total_days_for_split = (
                                (snap.get("form_data", {}) or {}).get("totalDays")
                                or (snap.get("form_data", {}) or {}).get("total_days_requested")
                                or snap.get("totalDays")
                            )
                            _prev_offset, _prev_count = _resolve_previous_chunk_window(
                                meal_plan_id, int(week_number), int(days_offset or 0), _total_days_for_split
                            )
                            _breakdown = _compute_prev_chunk_meal_breakdown(
                                plan_days=prior_days,
                                prev_offset=_prev_offset,
                                prev_count=_prev_count,
                                consumed_records=chunk_consumed_records or [],
                                prev_chunk_number=int(week_number) - 1,
                            )
                            if _breakdown:
                                form_data["_prev_chunk_adherence"] = _breakdown
                                logger.info(
                                    f"[P0-5/PREV-ADHERENCE] chunk={week_number} prev={_breakdown['chunk_number']} "
                                    f"consumed={_breakdown['consumed_count']} skipped={_breakdown['skipped_count']} "
                                    f"window=days {_breakdown['prev_start_day']}-{_breakdown['prev_end_day']}"
                                )
                        except Exception as _adh_e:
                            logger.warning(f"[P0-5/PREV-ADHERENCE] No se pudo calcular breakdown: {_adh_e}")

                    user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
                    chunk_health_profile = user_res.get("health_profile", {}) if user_res else {}
                
                    # [GAP 3 DE 30 DÃAS FIX - Descongelar form_data Snapshot]: 
                    # Inyectar perfil en vivo para que los chunks asincronicos usen el objetivo (goal), peso, alergias y budget actualizados.
                    # Se protegen variables internas de generacion como _days_offset.
                    if chunk_health_profile:
                        # [GAP 5] Blindar ambas variantes de plan_start_date para que el merge
                        # de health_profile NO pise la fecha canonica guardada en form_data.
                        _protected_keys = {
                            '_plan_start_date', 'plan_start_date',
                            'generation_mode', 'session_id', 'user_id', 'total_days_requested',
                        }
                        for k, v in chunk_health_profile.items():
                            if not k.startswith('_') and k not in _protected_keys:
                                form_data[k] = v

                    form_data = _refresh_chunk_pantry(user_id, form_data, snapshot_form_data, task_id=task_id, week_number=week_number)
                    if form_data.get("_pantry_paused"):
                        return
                 
                    form_data = _inject_advanced_learning_signals(user_id, form_data, chunk_health_profile, prior_days, chunk_consumed_records, days_since_last_chunk=days_offset)

                    # [GAP F] Capturar triggers de aprendizaje para medir violaciones post-generación
                    tuning_metrics = chunk_health_profile.get("tuning_metrics", {})
                    _fatigue_data = calculate_ingredient_fatigue(user_id, tuning_metrics=tuning_metrics)
                    _fatigued_ingredients = list(_fatigue_data.get('fatigued_ingredients', []) or [])
                    _allergy_keywords = []
                    try:
                        for f in (alergias_facts or []):
                            fact_text = f.get('fact', '').lower()
                            stopw = {"alergia", "alergico", "intolerante", "intolerancia", "condicion", "medica", "tiene", "sufre", "para"}
                            _allergy_keywords.extend([w for w in fact_text.split() if len(w) > 3 and w not in stopw])
                    except Exception:
                        pass

                    # Log estructurado: señales que entran al prompt del LLM
                    logger.info(
                        f"[GAP F/SIGNALS] plan={meal_plan_id} chunk={week_number} "
                        f"prior_meals={len(prior_meals)} rejections={len(rejected_meal_names)} "
                        f"fatigued={len(_fatigued_ingredients)} allergy_kw={len(_allergy_keywords)}"
                    )

                    logger.info(f" [CHUNK] Generando chunk {week_number} para plan {meal_plan_id} "
                                f"(dias {days_offset+1}-{days_offset+days_count}, {len(prior_meals)} platos previos)...")

                    # [P0-2] Preflight: persistir contadores prior-only en plan_chunk_queue.learning_metrics
                    # ANTES de invocar el pipeline. Si el LLM crashea / hace timeout / el worker se mata
                    # mid-flight, plan_chunk_queue.learning_metrics queda con counts útiles del chunk
                    # previo en vez de NULL — y _rebuild_last_chunk_learning_from_queue puede reconstruir
                    # `_last_chunk_learning` para el chunk siguiente sin caer al stub. Reusamos
                    # _calculate_learning_metrics con new_days=[] (todas las violations=0) para no
                    # duplicar lógica de extracción de bases. La cláusula `learning_metrics IS NULL`
                    # impide pisar resultados de un éxito previo (caso retry tras success parcial).
                    try:
                        _preflight_lm = _calculate_learning_metrics(
                            new_days=[],
                            prior_meals=prior_meals,
                            prior_days=prior_days,
                            rejected_names=rejected_meal_names,
                            allergy_keywords=_allergy_keywords,
                            fatigued_ingredients=_fatigued_ingredients,
                        )
                        _preflight_lm["preflight"] = True
                        _preflight_lm["preflight_at"] = datetime.now(timezone.utc).isoformat()
                        from db_core import execute_sql_write as _exec_write_pf
                        _exec_write_pf(
                            "UPDATE plan_chunk_queue SET learning_metrics = %s::jsonb "
                            "WHERE id = %s AND learning_metrics IS NULL",
                            (json.dumps(_preflight_lm, ensure_ascii=False), task_id)
                        )
                        logger.debug(
                            f"[P0-2/PREFLIGHT] plan={meal_plan_id} chunk={week_number} "
                            f"prior_meals={_preflight_lm.get('prior_meals_count')} "
                            f"prior_bases={_preflight_lm.get('prior_meal_bases_count')} "
                            f"rej={_preflight_lm.get('rejected_count')} "
                            f"alg={_preflight_lm.get('allergy_keywords_count')}"
                        )
                    except Exception as _pf_err:
                        # Best-effort: un fallo aquí no debe bloquear la generación. La cadena
                        # de aprendizaje cae al comportamiento previo (stub si el chunk falla).
                        logger.warning(
                            f"[P0-2/PREFLIGHT] Error persistiendo preflight learning_metrics "
                            f"para plan {meal_plan_id} chunk {week_number}: "
                            f"{type(_pf_err).__name__}: {_pf_err}"
                        )

                    # [P1-D] Inyectar warning estructurado al LLM si el inventario cambió
                    # significativamente desde el final del chunk previo. El sistema ya hace
                    # drift validation POST-LLM con retries (L12493), pero ese drift solo
                    # detecta divergencia entre el snapshot pre-LLM y el live post-LLM. Con
                    # P1-D el LLM tiene contexto ANTES de generar: sabe qué bajó (priorizar
                    # usar lo que queda) y qué aumentó (incorporar variedad). Best-effort:
                    # si falta snapshot del chunk previo o el inventario no es parseable, skip.
                    if int(week_number) >= 2 and isinstance(prior_plan_data, dict):
                        try:
                            _p1d_per_chunk = prior_plan_data.get("_pantry_snapshot_per_chunk") or {}
                            _p1d_prev_snap = _p1d_per_chunk.get(str(int(week_number) - 1))
                            if _p1d_prev_snap:
                                _p1d_warning = _compute_pantry_diff_warning(
                                    _p1d_prev_snap,
                                    form_data.get("current_pantry_ingredients") or [],
                                )
                                if _p1d_warning:
                                    form_data["_pantry_drift_warning"] = _p1d_warning
                                    logger.info(
                                        f"[P1-D/WARNING] plan={meal_plan_id} chunk={week_number}: "
                                        f"drift detectado vs chunk previo "
                                        f"(drops={len(_p1d_warning.get('critical_drops') or [])}, "
                                        f"increases={len(_p1d_warning.get('notable_increases') or [])}, "
                                        f"new_items={len(_p1d_warning.get('new_items') or [])}). "
                                        f"Inyectado a form_data._pantry_drift_warning."
                                    )
                        except Exception as _p1d_err:
                            logger.debug(
                                f"[P1-D/WARNING] No se pudo computar pantry_diff_warning para "
                                f"plan {meal_plan_id} chunk {week_number}: {_p1d_err}"
                            )

                    # [P0-3] Retry loop: genera el chunk y valida que solo use ingredientes de la nevera.
                    # Hasta PANTRY_MAX_RETRIES reintentos con feedback explícito al LLM.
                    # Si agota reintentos, PAUSA el chunk en pending_user_action (P1-1):
                    # marcar 'failed' provocaría un loop con _recover_failed_chunks_for_long_plans
                    # quemando tokens contra la misma nevera. La constante vive en
                    # `constants.CHUNK_PANTRY_MAX_RETRIES` (override por env var).
                    _PANTRY_MAX_RETRIES = CHUNK_PANTRY_MAX_RETRIES
                    _pantry_ok = False
                    result = {}
                    new_days = []
                    def _finalize_live_pantry_validation(_correction_message: str) -> str:
                        """Return 'ok', 'retry', or 'paused' after final live pantry validation."""
                        try:
                            fresh_live_inv = get_user_inventory_net(user_id)
                        except Exception as drift_e:
                            logger.warning(
                                f"[P0-2/PANTRY-FINAL] plan={meal_plan_id} chunk={week_number}: "
                                f"falló get_user_inventory_net durante validación final: {drift_e}"
                            )
                            _pause_chunk_for_final_inventory_validation(
                                task_id,
                                user_id,
                                week_number,
                                reason="final_inventory_unavailable",
                            )
                            return "paused"

                        if fresh_live_inv is None:
                            logger.warning(
                                f"[P0-2/PANTRY-FINAL] plan={meal_plan_id} chunk={week_number}: "
                                "inventario vivo ausente durante validación final."
                            )
                            _pause_chunk_for_final_inventory_validation(
                                task_id,
                                user_id,
                                week_number,
                                reason="final_inventory_missing",
                            )
                            return "paused"

                        old_inv = form_data.get("current_pantry_ingredients", [])
                        drift_pct = _calculate_inventory_drift(old_inv, fresh_live_inv)
                        
                        from constants import validate_ingredients_against_pantry as _vip
                        _all_gen_ing = [ing for d in new_days for m in d.get("meals", []) for ing in m.get("ingredients", []) if isinstance(ing, str)]
                        _live_check = _vip(_all_gen_ing, fresh_live_inv, strict_quantities=True, tolerance=1.0)
                        
                        logger.info(f"[P0-2/LIVE-VALIDATION] plan={meal_plan_id} chunk={week_number} drift={drift_pct*100:.1f}% live_check_failed={not _live_check}")
                        
                        if drift_pct > 0.05 or _live_check is not True:
                            _dr = form_data.get("_drift_retries", 0)
                            if _dr < 2:
                                form_data["_drift_retries"] = _dr + 1
                                logger.warning(
                                    f"[P1-D/DRIFT] plan={meal_plan_id} chunk={week_number}: "
                                    f"Inventario cambió {drift_pct*100:.1f}% o falló live_check. "
                                    f"Activando reintento {form_data['_drift_retries']} con datos frescos."
                                )
                                form_data["current_pantry_ingredients"] = fresh_live_inv
                                form_data["_pantry_correction"] = _correction_message
                                return "retry"

                            logger.error(
                                f"[P0-3/DRIFT] plan={meal_plan_id} chunk={week_number}: "
                                "Deriva persistente o invalidación > 5%. Pausando chunk."
                            )
                            _pause_chunk_for_pantry_refresh(
                                task_id,
                                user_id,
                                week_number,
                                fresh_live_inv,
                                reason="persistent_drift",
                            )
                            _dispatch_push_notification(
                                user_id=user_id,
                                title="Confirma tu inventario",
                                body="Tu nevera cambió mucho durante la generación. Confirma su contenido para continuar.",
                                url="/dashboard"
                            )
                            return "paused"
                        return "ok"

                    for _pantry_attempt in range(_PANTRY_MAX_RETRIES + 1):
                        # [P0-2/PRE-LLM] Re-validar pre-LLM que el chunk siga vivo y el plan
                        # exista. Cierra la ventana TOCTOU entre el pickup (FOR UPDATE SKIP
                        # LOCKED ~9776) y el submit del LLM aquí: durante ~2500 líneas de prep
                        # work, otro path como `save_new_meal_plan_atomic` puede haber
                        # cancelado este chunk en plan_chunk_queue, o el usuario puede haber
                        # borrado el meal_plan. Sin este guard, gastamos tokens en un chunk
                        # ya muerto cuyo merge será descartado por el TOCTOU guard interno
                        # (líneas ~13100). La validación corre en cada iteración del retry
                        # loop porque el LLM puede tardar 5-30s y la cancelación puede llegar
                        # entre intentos.
                        _p02_state = _validate_chunk_pre_llm(task_id, meal_plan_id, user_id)
                        if _p02_state == "plan_missing":
                            try:
                                release_chunk_reservations(user_id, str(task_id))
                            except Exception as _p02_rel_err:
                                logger.warning(
                                    f"[P0-2/PRE-LLM] Error liberando reservas para chunk "
                                    f"{task_id}: {_p02_rel_err}"
                                )
                            try:
                                execute_sql_write(
                                    "UPDATE plan_chunk_queue SET status = 'cancelled', "
                                    "updated_at = NOW() WHERE id = %s",
                                    (task_id,),
                                )
                            except Exception as _p02_cnc_err:
                                logger.warning(
                                    f"[P0-2/PRE-LLM] Error cancelando chunk {task_id}: "
                                    f"{_p02_cnc_err}"
                                )
                            return
                        if _p02_state in ("chunk_terminal", "chunk_unknown"):
                            # save_new_meal_plan_atomic ya hizo release_chunk_reservations +
                            # UPDATE status='cancelled' atómicamente al regenerar el plan.
                            # Llamamos release otra vez por defensa (idempotente vía CAS).
                            try:
                                release_chunk_reservations(user_id, str(task_id))
                            except Exception as _p02_rel_err:
                                logger.warning(
                                    f"[P0-2/PRE-LLM] Error liberando reservas para chunk "
                                    f"{task_id} en estado terminal: {_p02_rel_err}"
                                )
                            return
                        # validation_error: best-effort, dejamos pasar para no bloquear
                        # chunks por flaps transientes de DB (el TOCTOU guard interno cubre).

                        import concurrent.futures as _cf
                        _exec = _cf.ThreadPoolExecutor(max_workers=1)
                        _fut = _exec.submit(run_plan_pipeline, form_data, [], taste_profile, memory_context, None, None)
                        try:

                            from constants import CHUNK_PIPELINE_TIMEOUT_SECONDS
                            import time
                            _current_timeout = CHUNK_PIPELINE_TIMEOUT_SECONDS
                            if _pantry_attempt > 0 or _pickup_attempts > 1:
                                _current_timeout = int(_current_timeout * 1.5)
                        
                            _t0 = time.time()
                            result = _fut.result(timeout=_current_timeout)
                        except _cf.TimeoutError:

                            _elapsed_ms = int((time.time() - _t0) * 1000)
                            logger.error(
                                f"[P0-1/TIMEOUT] plan={meal_plan_id} chunk={week_number} "
                                f"attempt={_pickup_attempts} elapsed={_elapsed_ms}ms "
                                f"(timeout={_current_timeout}s)"
                            )
                            if _pickup_attempts >= 3:
                                logger.error(f"[P0-1/TIMEOUT] Degradando a Smart Shuffle en intento {_pickup_attempts}.")
                                is_degraded = True
                                form_data["_pantry_degraded_reason"] = f"gemini_timeout_{_current_timeout}s_attempt_{_pickup_attempts}"
                                result = {"_force_degraded_retry": True}
                                break
                            raise Exception(f"Chunk pipeline timed out after {_current_timeout}s")
                        finally:
                            _exec.shutdown(wait=False, cancel_futures=True)

                        new_days = result.get("days", [])

                        # Manejo de timeout que pidió Smart Shuffle: salir del for
                        # y dejar que el while True externo arranque la rama is_degraded.
                        if result.get("_force_degraded_retry"):
                            break

                        if not new_days or "error" in result:
                            raise Exception(result.get("error", "No days generated"))

                        # [P0-3] Validación post-generación: todos los ingredientes deben estar en la nevera.
                        # strict_quantities=False: solo validamos existencia, no cantidades exactas.
                        _pantry_snapshot = form_data.get("current_pantry_ingredients", [])
                        if _pantry_snapshot:
                            _all_gen_ing = [
                                ing for d in new_days
                                for m in (d.get("meals") or [])
                                for ing in (m.get("ingredients") or [])
                                if isinstance(ing, str) and ing.strip()
                            ]
                            from constants import validate_ingredients_against_pantry as _vip
                            _val_result = _vip(_all_gen_ing, _pantry_snapshot, strict_quantities=False)
                            if _val_result is True:
                                # [P0-B] Existencia OK. Validamos cantidades según el modo configurado.
                                #   off      → aceptar tal cual.
                                #   advisory → loguear violaciones, anotarlas en form_data y aceptar.
                                #   hybrid   → reintentar con feedback; si retries se agotan, anotar y aceptar.
                                #   strict   → reintentar con feedback al LLM, fallar tras agotar retries.
                                #
                                # [P1-5] Precedencia del modo (mayor a menor):
                                #   1. `chunk_health_profile._pantry_quantity_mode` — opt-in
                                #      explícito del usuario en su perfil. Win siempre.
                                #   2. `form_data._pantry_quantity_mode` — snapshot capturado
                                #      en `_enqueue_plan_chunk` con el global vigente al
                                #      momento del enqueue. Cubre el caso donde admin cambió
                                #      el global (e.g., hybrid → strict) entre el enqueue del
                                #      chunk N y su ejecución, y queremos que chunks
                                #      encolados con expectativa hybrid se mantengan así.
                                #   3. `CHUNK_PANTRY_QUANTITY_MODE` — constante global vigente.
                                #      Última red para chunks legacy encolados antes de P1-5.
                                _qty_mode = (
                                    chunk_health_profile.get("_pantry_quantity_mode")
                                    or form_data.get("_pantry_quantity_mode")
                                    or CHUNK_PANTRY_QUANTITY_MODE
                                ).lower()
                                # [P1-5] Tolerance default sigue la misma cascada: snapshot
                                # primero (consistencia con el modo), constante después.
                                _tolerance = float(
                                    form_data.get("_pantry_quantity_hybrid_tolerance")
                                    or CHUNK_PANTRY_QUANTITY_HYBRID_TOLERANCE
                                )
                                if _qty_mode == "strict":
                                    _tolerance = 1.00
                                elif _qty_mode in ("off", "advisory"):
                                    _tolerance = 1.10 # Legacy/Advisory (10% overflow)

                                if _qty_mode == "off":
                                    _final_validation = _finalize_live_pantry_validation(
                                        "Tu generación anterior usó ingredientes que el usuario acaba de eliminar "
                                        "o no incluyó nuevos disponibles. Por favor, regenera el plan usando este "
                                        "inventario actualizado."
                                    )
                                    if _final_validation == "retry":
                                        continue
                                    if _final_validation == "paused":
                                        return
                                    _pantry_ok = True
                                    form_data.pop("_pantry_correction", None)
                                    break

                                _qty_result = _vip(_all_gen_ing, _pantry_snapshot, strict_quantities=True, tolerance=_tolerance)
                                if _qty_result is True:
                                    _final_validation = _finalize_live_pantry_validation(
                                        "Inventario actualizado durante generación. Por favor, ajusta el plan."
                                    )
                                    if _final_validation == "retry":
                                        continue
                                    if _final_validation == "paused":
                                        return
                                    _pantry_ok = True
                                    form_data.pop("_pantry_correction", None)
                                    form_data.pop("_pantry_quantity_violations", None)
                                    break

                                _qty_feedback = str(_qty_result)
                                # Solo nos interesa la línea de over-limit; las "inexistentes" ya pasaron
                                # la primera validación (matchearon vía similitud/fallback). Recortamos para logs.
                                _qty_summary = _qty_feedback.split("\n", 2)[0][:300]

                                if _qty_mode in ("strict", "hybrid") and _pantry_attempt < _PANTRY_MAX_RETRIES:
                                    logger.warning(
                                        f"[P1-1/QTY-{_qty_mode.upper()}] Cantidad fuera de límite en plan={meal_plan_id} "
                                        f"chunk={week_number} intento={_pantry_attempt + 1}/{_PANTRY_MAX_RETRIES} "
                                        f"(tolerance={_tolerance}): {_qty_summary}"
                                    )
                                    form_data["_pantry_correction"] = _qty_feedback
                                    continue  # retry con feedback de cantidades

                                if _qty_mode == "strict":
                                    # [P0-3 FIX] Reintentos agotados en modo estricto.
                                    # En vez de fallar silenciosamente (raise → failed → recovery),
                                    # pausamos en pending_user_action con push para que el usuario
                                    # actualice su nevera. El recovery cron existente lo retomará.
                                    logger.error(
                                        f"[P0-3/QTY-STRICT] PAUSA POR CANTIDAD | plan={meal_plan_id} "
                                        f"chunk={week_number}: {_qty_summary}"
                                    )
                                    # Extraer el ingrediente principal del feedback para el push
                                    _qty_ingredient = "ingredientes"
                                    try:
                                        import re as _re_qty
                                        _ing_match = _re_qty.search(r'excede.*?de\s+(.+?)(?:\s*\(|$)', _qty_feedback)
                                        if _ing_match:
                                            _qty_ingredient = _ing_match.group(1).strip()[:40]
                                    except Exception:
                                        pass
                                    _pause_chunk_for_pantry_refresh(
                                        task_id, user_id, week_number,
                                        fresh_inventory=_pantry_snapshot,
                                        reason="quantity_unfeasible",
                                    )
                                    _dispatch_push_notification(
                                        user_id=user_id,
                                        title="Tu plan necesita más ingredientes",
                                        body=f"Tu próximo bloque necesita más {_qty_ingredient}. Actualiza tu nevera o registra una compra.",
                                        url="/dashboard",
                                    )
                                    return  # chunk queda en pending_user_action

                                # Advisory / hybrid-retries-agotados: anotamos y seguimos.
                                _delta_pct_str = f"{(_tolerance - 1.0) * 100:.0f}%"
                                logger.warning(
                                    f"[P1-1/QTY-{_qty_mode.upper()}] plan={meal_plan_id} chunk={week_number} "
                                    f"acepta sobreuso de despensa tras agotar reintentos (modo={_qty_mode}, "
                                    f"intento={_pantry_attempt + 1}, delta_pct={_delta_pct_str}): {_qty_summary}"
                                )
                                # [P0-5] Annotate BEFORE the final-live validation. Otherwise an advisory
                                # chunk whose pantry violation is real (live_check returns the same
                                # over-limit string we just decided to accept) gets bounced back to
                                # `_drift_retries` — re-running the LLM up to 2 extra times even though
                                # the policy is "accept and continue". Pre-annotating means the loop
                                # exits cleanly when finalize says "ok" and still exposes the violation
                                # via `_pantry_quantity_violations` even when the drift fallback paused
                                # the chunk for genuinely-different drift signals.
                                form_data["_pantry_quantity_violations"] = _qty_feedback
                                # Advisory's contract is "single LLM call, accept the result" — skip the
                                # drift second-opinion entirely. Hybrid (after exhausting retries) keeps
                                # the finalize call so a fresh inventory snapshot can recover when the
                                # drift was a real env change rather than a deliberate overshoot.
                                if _qty_mode == "advisory":
                                    _pantry_ok = True
                                    form_data.pop("_pantry_correction", None)
                                    break
                                _final_validation = _finalize_live_pantry_validation("Inventario actualizado.")
                                if _final_validation == "retry":
                                    continue
                                if _final_validation == "paused":
                                    return
                                _pantry_ok = True
                                form_data.pop("_pantry_correction", None)
                                break
                            elif _pantry_attempt < _PANTRY_MAX_RETRIES:
                                _feedback_str = str(_val_result)
                                logger.warning(
                                    f"[P0-3] Violación de despensa en plan={meal_plan_id} chunk={week_number} "
                                    f"intento={_pantry_attempt + 1}/{_PANTRY_MAX_RETRIES}: "
                                    f"{_feedback_str[:300]}"
                                )
                                form_data["_pantry_correction"] = _feedback_str
                            else:
                                # [P1-1] Pausar como pending_user_action en vez de marcar 'failed'.
                                # Antes: marcábamos 'failed' → _recover_failed_chunks_for_long_plans
                                # re-encolaba el chunk con backoff exponencial (2→4→8→16→32 min) y
                                # el LLM repetía las mismas alucinaciones contra la misma nevera —
                                # quemando tokens sin convergencia hasta CHUNK_MAX_RECOVERY_ATTEMPTS.
                                # Ahora: pausa en pending_user_action con reason='pantry_violation_after_retries'.
                                # El `_recover_pantry_paused_chunks` cron sólo lo reanuda cuando la
                                # nevera del usuario CAMBIA — i.e., cuando hay esperanza de que el
                                # LLM converja. Simétrico al path de quantity violations strict
                                # (líneas 12947-12958) que ya usaba esta estrategia.
                                logger.error(
                                    f"[P1-1] Violación persistente tras {_PANTRY_MAX_RETRIES} reintentos "
                                    f"en plan={meal_plan_id} chunk={week_number}. Pausando chunk en "
                                    f"pending_user_action:pantry_violation_after_retries."
                                )
                                # Persistir el feedback final para diagnóstico operacional y
                                # para que el frontend pueda mostrar al usuario qué ingredientes
                                # el LLM no pudo resolver tras varios intentos.
                                form_data["_pantry_correction"] = str(_val_result)[:1000]
                                _pause_chunk_for_pantry_refresh(
                                    task_id,
                                    user_id,
                                    week_number,
                                    fresh_inventory=_pantry_snapshot,
                                    reason="pantry_violation_after_retries",
                                )
                                _dispatch_push_notification(
                                    user_id=user_id,
                                    title="Tu plan necesita revisión de ingredientes",
                                    body=(
                                        "No pudimos generar los próximos días con los "
                                        "ingredientes que tienes. Actualiza tu nevera para continuar."
                                    ),
                                    url="/dashboard",
                                )
                                return
                        else:
                            _final_validation = _finalize_live_pantry_validation("Inventario actualizado.")
                            if _final_validation == "retry":
                                continue
                            if _final_validation == "paused":
                                return
                            _pantry_ok = True
                            break

                    # [POST-FOR] Si el for termino sin _pantry_ok=True por timeout
                    # repetido, saltar a la rama is_degraded del while True externo.
                    if result.get("_force_degraded_retry"):
                        continue

                    # [P0-2] Segunda validación post-merge si flexible_mode o stale_snapshot
                    _is_flex = bool(form_data.get("_pantry_flexible_mode") or form_data.get("_fresh_pantry_source") == "stale_snapshot")
                    if _is_flex:
                        form_data["_strict_post_gen_required"] = True
                        try:
                            import time
                            from db_inventory import get_user_inventory_net as _live_inventory_fetch
                            live_inv = _live_inventory_fetch(user_id)
                            if live_inv is None:
                                time.sleep(5)
                                live_inv = _live_inventory_fetch(user_id)
                                
                            if live_inv is None:
                                logger.warning(f"[P0-1/FLEXIBLE] Chunk {week_number} pausado para {user_id}: live_inv inalcanzable tras 2 intentos en flexible_mode.")
                                _pause_chunk_for_final_inventory_validation(task_id, user_id, week_number, reason="flexible_live_unreachable")
                                return False
                        
                            if live_inv is not None:
                                _all_gen_ing = [
                                    ing for d in new_days
                                    for m in (d.get("meals") or [])
                                    for ing in (m.get("ingredients") or [])
                                    if isinstance(ing, str) and ing.strip()
                                ]
                                from constants import validate_ingredients_against_pantry as _vip
                                _safe = _vip(_all_gen_ing, live_inv, strict_quantities=True)
                                if _safe is not True:
                                    logger.warning(f"[P0-2] Chunk {week_number} generado con flexible_mode falló validación vs live inventory. Pausando chunk.")
                                    
                                    # Extraer ingredientes faltantes del output de _vip
                                    import re
                                    missing_list = []
                                    if isinstance(_safe, str):
                                        # Parsear "INEXISTENTES en inventario: pollo, arroz."
                                        match_unauth = re.search(r"INEXISTENTES en inventario: (.*?)\.", _safe)
                                        if match_unauth:
                                            missing_list.extend([x.strip() for x in match_unauth.group(1).split(",")])
                                            
                                        # Parsear "CANTIDADES (Tu inventario restringe esto matemáticamente): [pollo] (Pediste 500g, límite: 200g), [arroz]..."
                                        match_limit = re.search(r"matemáticamente\): (.*?)\.", _safe)
                                        if match_limit:
                                            # Items are like "[pollo] (Pediste...)"
                                            items = re.findall(r"\[(.*?)\]", match_limit.group(1))
                                            missing_list.extend(items)
                                            
                                        # Limpiar duplicados y vacíos
                                        missing_list = list(set([x for x in missing_list if x]))

                                    if missing_list:
                                        # Setear a nivel de plan para que shopping_calculator lo vea
                                        form_data["_pantry_supplement_required"] = missing_list
                                        
                                        # Push notification de compras urgentes
                                        _dispatch_push_notification(
                                            user_id=user_id,
                                            title="Tu plan tiene compras urgentes",
                                            body=f"Generamos los días {days_offset+1}-{days_offset+days_count} pero te faltan ingredientes. Revisa tu lista de compras.",
                                            url="/shopping-list"
                                        )

                                    for d in new_days:
                                        if isinstance(d, dict):
                                            d['quality_tier'] = 'emergency_pantry_unsafe'
                                            for m in (d.get("meals") or []):
                                                if isinstance(m, dict):
                                                    m['_pantry_unsafe_after_flexible'] = True
                                                    if missing_list:
                                                        m['_missing_ingredients'] = missing_list

                                    # [P0-C] Pasar missing_list al pause helper para que el recovery
                                    # cron pueda detectar cuándo el usuario las añadió a la nevera y
                                    # reanudar el chunk sin esperar el TTL escalation a flexible_mode.
                                    _pause_chunk_for_final_inventory_validation(
                                        task_id, user_id, week_number,
                                        reason="flexible_live_unreachable",
                                        missing_ingredients=missing_list or None,
                                    )
                                    return False
                        except Exception as e:
                            # [P0-1/FAIL-CLOSED] Antes solo logueábamos warning y dejábamos que el
                            # chunk continuara al merge sin la segunda validación. En flexible_mode
                            # (donde ya relajamos el snapshot por stale/zero-log/drift), un fallo
                            # aquí permitía shipear platos con ingredientes fantasma sin que la red
                            # de seguridad disparara. Ahora pausamos el chunk: el recovery cron lo
                            # retomará cuando el live inventory se estabilice. Si la pausa misma
                            # falla, marcamos failed como último recurso — peor para el throughput
                            # que un fail-open silencioso que rompe la promesa de zero-waste.
                            logger.error(
                                f"[P0-1/FAIL-CLOSED] Excepción en doble validación flexible para "
                                f"plan={meal_plan_id} chunk={week_number}: "
                                f"{type(e).__name__}: {e}. Pausando chunk en vez de shipear sin re-validación."
                            )
                            try:
                                _pause_chunk_for_final_inventory_validation(
                                    task_id, user_id, week_number,
                                    reason="flexible_validation_error",
                                )
                            except Exception as _pause_err:
                                logger.error(
                                    f"[P0-1/FAIL-CLOSED] Pausa también falló: "
                                    f"{type(_pause_err).__name__}: {_pause_err}. "
                                    f"Marcando chunk failed como último recurso."
                                )
                                # [P1-4] CAS guard: si zombie rescue + nuevo pickup ya
                                # ocurrió, no clobbear el processing del worker B.
                                _cas_update_chunk_status(task_id, _pickup_attempts, "failed")
                            return False

                    # [P0-1] Marcar requires_pantry_review si venimos de un force-generate
                    if form_data.get("_requires_pantry_review"):
                        for d in new_days:
                            if isinstance(d, dict):
                                for m in (d.get("meals") or []):
                                    if isinstance(m, dict):
                                        m.setdefault("meta", {})["requires_pantry_review"] = True

                    # [GAP C] Marcar días generados por LLM con su tier de calidad
                    for d in new_days:
                        if isinstance(d, dict) and 'quality_tier' not in d:
                            d['quality_tier'] = 'llm'

                    # [GAP F] Métricas de aprendizaje continuo (solo en path LLM — Smart Shuffle no aplica)
                    try:
                        learning_metrics = _calculate_learning_metrics(
                            new_days=new_days,
                            prior_meals=prior_meals,
                            prior_days=prior_days,
                            rejected_names=rejected_meal_names,
                            allergy_keywords=_allergy_keywords,
                            fatigued_ingredients=_fatigued_ingredients,
                        )
                        repeat_pct = learning_metrics["learning_repeat_pct"]
                        ingredient_base_repeat_pct = learning_metrics["ingredient_base_repeat_pct"]
                        rej_viol = learning_metrics["rejection_violations"]
                        alg_viol = learning_metrics["allergy_violations"]

                        # Log estructurado por chunk (observable con grep [GAP F/MEASURE])
                        logger.info(
                            f"[GAP F/MEASURE] plan={meal_plan_id} chunk={week_number} "
                            f"repeat_pct={repeat_pct}% ingredient_base_repeat_pct={ingredient_base_repeat_pct}% "
                            f"rejection_violations={rej_viol} allergy_violations={alg_viol} "
                            f"fatigued_hits={learning_metrics['fatigued_violations']}"
                        )

                        # Alertas: rechazos y alergias NUNCA deberían reaparecer
                        if rej_viol > 0:
                            logger.error(
                                f"[GAP F/VIOLATION] Chunk {week_number} plan {meal_plan_id}: "
                                f"{rej_viol} meals rechazados reaparecieron: {learning_metrics['sample_rejection_hits']}"
                            )
                        if alg_viol > 0:
                            logger.error(
                                f"[GAP F/VIOLATION] Chunk {week_number} plan {meal_plan_id}: "
                                f"{alg_viol} alergias violadas en ingredientes: {learning_metrics['sample_allergy_hits']}"
                            )
                        # Umbral de repetición: si >20%, warning (el LLM está copiando demasiado)
                        if repeat_pct > 20.0 and learning_metrics["prior_meals_count"] > 0:
                            logger.warning(
                                f"[GAP F/HIGH-REPEAT] Chunk {week_number} plan {meal_plan_id}: "
                                f"{repeat_pct}% meals repetidos de chunks previos: {learning_metrics['sample_repeats']}"
                            )
                        if ingredient_base_repeat_pct > 60.0 and learning_metrics["prior_meal_bases_count"] > 0:
                            logger.warning(
                                f"[P0-3/HIGH-BASE-REPEAT] Chunk {week_number} plan {meal_plan_id}: "
                                f"{ingredient_base_repeat_pct}% repite ingrediente base: {learning_metrics['sample_repeated_bases']}"
                            )

                        # [P0-B] Anotar violaciones de cantidad detectadas en advisory mode para que
                        # queden en plan_chunk_queue.learning_metrics y en _last_chunk_learning.
                        _qty_violations_str = form_data.get("_pantry_quantity_violations")
                        if _qty_violations_str:
                            learning_metrics["pantry_quantity_violations"] = 1
                            learning_metrics["sample_pantry_quantity_violations"] = _qty_violations_str[:500]
                        else:
                            learning_metrics.setdefault("pantry_quantity_violations", 0)

                        # [P0-D] Marcar si el chunk avanzó por inventory_proxy en lugar de logs reales.
                        # Permite filtrar en métricas: chunks con inventory_proxy_used=true son de
                        # "aprendizaje débil" (señal indirecta) y no deberían contar como adherencia plena.
                        if form_data.get("_inventory_activity_proxy_used"):
                            learning_metrics["inventory_activity_proxy_used"] = True
                            learning_metrics["inventory_activity_mutations"] = int(
                                form_data.get("_inventory_activity_mutations") or 0
                            )
                        else:
                            learning_metrics.setdefault("inventory_activity_proxy_used", False)
                        
                        # [P0-NEW2] Auditar sparse_logging_proxy
                        if form_data.get("_sparse_logging_proxy"):
                            learning_metrics["sparse_logging_proxy_used"] = True

                        if form_data.get("_learning_signal_strength"):
                            learning_metrics["learning_signal_strength"] = form_data["_learning_signal_strength"]
                        
                        # [P0-3] Auditar si fue degradado por snapshot stale
                        if form_data.get("_pantry_degraded_reason"):
                            learning_metrics["pantry_degraded_reason"] = form_data["_pantry_degraded_reason"]

                        # [P0-3/TELEMETRY] Capturar la edad del snapshot en el momento del pickup,
                        # incluso si el live succeeded (en cuyo caso el age es informativo, no
                        # determinante). Permite construir histogramas por longitud de plan y
                        # detectar regresiones del cron proactivo (_proactive_refresh_pending_pantry_snapshots)
                        # desde plan_chunk_metrics sin tocar plan_chunk_queue.
                        try:
                            _snap_age = form_data.get("_pantry_snapshot_age_hours")
                            if _snap_age is None:
                                _captured_at_str = (snap.get("form_data") or {}).get("_pantry_captured_at")
                                if _captured_at_str:
                                    from constants import safe_fromisoformat
                                    _captured_dt = safe_fromisoformat(_captured_at_str)
                                    if _captured_dt.tzinfo is None:
                                        _captured_dt = _captured_dt.replace(tzinfo=timezone.utc)
                                    _snap_age = round(
                                        (datetime.now(timezone.utc) - _captured_dt).total_seconds() / 3600.0, 2
                                    )
                            if _snap_age is not None:
                                learning_metrics["pantry_snapshot_age_hours_at_pickup"] = float(_snap_age)
                                if float(_snap_age) > CHUNK_PANTRY_SNAPSHOT_TTL_HOURS:
                                    logger.warning(
                                        f"[P0-3/SNAPSHOT-AGE] chunk {week_number} plan {meal_plan_id} ejecutó "
                                        f"con snapshot de {_snap_age}h (> TTL {CHUNK_PANTRY_SNAPSHOT_TTL_HOURS}h). "
                                        f"Source={form_data.get('_fresh_pantry_source', 'unknown')}."
                                    )
                        except Exception as _age_err:
                            logger.debug(f"[P0-3/TELEMETRY] No se pudo calcular snapshot age: {_age_err}")

                        if learning_metrics.get("inventory_activity_proxy_used") or learning_metrics.get("sparse_logging_proxy_used") or learning_metrics.get("pantry_degraded_reason"):
                            learning_metrics["learning_confidence"] = "low"
                    except Exception as lm_e:
                        logger.warning(f"[GAP F] Error calculando learning_metrics: {lm_e}")
                        learning_metrics = None

                break
            # [FIX CRÃTICO 1 â€" Truncar si el pipeline devuelve mas dias de los pedidos]
            # Como medida de seguridad extra, si devuelve mas dias de los esperados, truncamos.
            if len(new_days) > days_count:
                new_days = new_days[:days_count]

            # [GAP 3 - VALIDACION PRE-MERGE: numeracion de dias del chunk]
            # Antes de mergear, verificar que los nuevos dias tienen EXACTAMENTE los
            # numeros absolutos esperados: {days_offset+1 .. days_offset+days_count}.
            # Si el pipeline (o Smart Shuffle) olvida setear 'day' o devuelve numeros incorrectos,
            # el merge sobrescribiria dias previos o dejaria huecos silenciosamente. Mejor fallar
            # aqui y que el outer catch re-encole el chunk con backoff.
            expected_day_nums = set(range(days_offset + 1, days_offset + days_count + 1))
            actual_day_nums = set()
            missing_day_field = 0
            for _nd in new_days:
                if not isinstance(_nd, dict):
                    continue
                _dn = _nd.get('day')
                if _dn is None:
                    missing_day_field += 1
                else:
                    actual_day_nums.add(_dn)

            if missing_day_field > 0 or actual_day_nums != expected_day_nums:
                missing = sorted(expected_day_nums - actual_day_nums)
                extra = sorted(actual_day_nums - expected_day_nums)
                
                raise Exception(
                    f"[GAP3] Chunk {week_number} numeracion invalida. "
                    f"Esperado {sorted(list(expected_day_nums))}, recibido {sorted(list(actual_day_nums))}. "
                    f"Faltan: {missing}, Extra: {extra}, sin_campo_day: {missing_day_field}"
                )

            # [P0-3 FIX] Guarda pre-merge: verificar que este chunk no fue cancelado
            # por una regeneración concurrente del usuario Y que el meal_plan_id sigue
            # siendo el plan activo. Si el usuario regeneró, services.py marca los chunks
            # como 'cancelled', pero hay una ventana donde el worker ya pasó el SELECT
            # inicial y está en medio de la generación LLM.
            try:
                _chunk_status_row = execute_sql_query(
                    "SELECT status FROM plan_chunk_queue WHERE id = %s",
                    (task_id,), fetch_one=True
                )
                if _chunk_status_row and _chunk_status_row.get('status') == 'cancelled':
                    logger.warning(
                        f"[P0-3] Chunk {task_id} fue cancelado durante generación (plan regenerado). "
                        f"Abortando merge para evitar contaminación cruzada."
                    )
                    return  # Salir sin hacer merge ni marcar como completed

                _latest_plan_row = execute_sql_query(
                    "SELECT id FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
                    (user_id,), fetch_one=True
                )
                # [P1] Type-safe compare: psycopg3 devuelve uuid.UUID para columnas UUID,
                # mientras que meal_plan_id arrastrado desde plan_chunk_queue puede ser str
                # (depende del path que lo insertó). UUID('x') != 'x' es True aunque
                # representen lo mismo, así que cada chunk legítimo era marcado 'cancelled'
                # tras generarse — el log mostraba "Plan {x} ya no es el activo. Plan activo: {x}"
                # con el mismo UUID a ambos lados. Normalizamos a str para que la comparación
                # mida valores y no tipos.
                if _latest_plan_row and str(_latest_plan_row.get('id')) != str(meal_plan_id):
                    logger.warning(
                        f"[P0-3] Plan {meal_plan_id} ya no es el plan activo del usuario {user_id}. "
                        f"Plan activo: {_latest_plan_row.get('id')}. Abortando merge de chunk huérfano."
                    )
                    # Marcar como cancelled para que no se reintente
                    try:
                        execute_sql_write(
                            "UPDATE plan_chunk_queue SET status = 'cancelled', updated_at = NOW() WHERE id = %s",
                            (task_id,)
                        )
                    except Exception:
                        pass
                    return
            except Exception as _p03_err:
                logger.warning(f"[P0-3] Error en guarda pre-merge (continuando): {_p03_err}")

            # [GAP 2 - Race condition en merge atomico de chunks (P0)]
            # Usar FOR UPDATE real en la fila antes de mergear JSONB en Python para evitar 
            # solapamiento o perdida de dias cuando multiples chunks terminan simultaneamente.
            from db_core import connection_pool
            from psycopg.rows import dict_row

            update_result = None
            _stale_abort = False  # [P0-2] inicializado antes del bloque para ser accesible post-transacción
            _toctou_abort = False  # [P0-1] True si el abort fue por chunk cancelado (no re-encolar)
            if not connection_pool:
                raise Exception("db connection_pool is not available for atomic merge.")

            # [P0-1] Refresh inline del heartbeat antes del bloque transaccional largo.
            # Si el thread daemon estuviera atrasado o muerto, esta línea garantiza que
            # `heartbeat_at` esté fresco al entrar al merge — el zombie rescue no podrá
            # marcarnos como muertos durante el FOR UPDATE.
            _touch_chunk_heartbeat(task_id)

            with connection_pool.connection() as conn:
                with conn.transaction():
                    with conn.cursor(row_factory=dict_row) as cursor:
                        # [P0-4] Advisory lock por meal_plan ANTES del FOR UPDATE para
                        # serializar contra `/shift-plan` y otros call sites que ya usan
                        # purpose='general' (ver db_plans.acquire_meal_plan_advisory_lock).
                        # FOR UPDATE solo serializa transacciones que también hacen FOR UPDATE
                        # sobre la misma fila; un caller que modifica plan_data vía UPDATE
                        # directo (sin lock previo) podría intercalarse. El advisory lock hace
                        # la intención explícita y serializa contra cualquier caller que use
                        # el mismo helper, incluso si su path no incluye FOR UPDATE.
                        from db_plans import acquire_meal_plan_advisory_lock as _p04_acquire_lock
                        _p04_acquire_lock(cursor, meal_plan_id, purpose="general")

                        # 1. Bloquear la fila de forma exclusiva
                        cursor.execute("SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE", (meal_plan_id,))
                        row = cursor.fetchone()
                        if not row:
                            raise Exception(f"Meal plan {meal_plan_id} not found during atomic merge")

                        plan_data = row['plan_data'] or {}
                        existing_days = plan_data.get('days', [])
                        prior_count_raw = len(existing_days)  # Conteo bruto antes de dedup

                        # [P0-1/TOCTOU FIX] Verificar estado del chunk DENTRO de la transacción,
                        # con lock en plan_chunk_queue para eliminar la ventana TOCTOU entre
                        # el guard externo (líneas ~3606-3616) y la adquisición del FOR UPDATE.
                        cursor.execute(
                            "SELECT status, attempts FROM plan_chunk_queue WHERE id = %s FOR UPDATE",
                            (task_id,)
                        )
                        chunk_status_row = cursor.fetchone()
                        _current_status = chunk_status_row.get('status') if chunk_status_row else None
                        _current_attempts = int(chunk_status_row.get('attempts') or 0) if chunk_status_row else -1

                        if _current_status == 'cancelled':
                            logger.warning(
                                f"[P0-1/TOCTOU] Chunk {task_id} fue cancelado entre el guard externo "
                                f"y el FOR UPDATE. Abortando merge para evitar contaminación cruzada."
                            )
                            _stale_abort = True
                            _toctou_abort = True
                        elif _current_attempts != _pickup_attempts:
                            # [P0-6] El zombie rescue incrementó attempts → fuimos desplazados.
                            # El chunk ya está en cola (pending/processing) para otro worker.
                            # No re-encolar: _toctou_abort=True.
                            logger.warning(
                                f"[P0-6/ZOMBIE-DISPLACED] Chunk {task_id} desplazado por zombie rescue "
                                f"(attempts pickup={_pickup_attempts}, current={_current_attempts}). "
                                f"Abortando merge duplicado."
                            )
                            _stale_abort = True
                            _toctou_abort = True
                        elif _current_status != 'processing':
                            # [P0-6] Estado inesperado: no es processing ni cancelled.
                            # Puede ocurrir si el zombie reset a pending pero nadie lo recogió aún.
                            logger.warning(
                                f"[P0-6/UNEXPECTED-STATUS] Chunk {task_id} en estado inesperado "
                                f"durante merge: {_current_status!r}. Abortando."
                            )
                            _stale_abort = True
                            _toctou_abort = True

                        # [P0-2/CAS] Detectar si shift-plan modificó el plan durante el LLM call.
                        # Si el timestamp cambió Y estamos en modo degradado (Smart Shuffle usa
                        # prior_days del pre-read), los días del shuffle pueden ser obsoletos.
                        # Solución: abortar y re-encolar para que el próximo intento use datos frescos.
                        _locked_modified_at = plan_data.get('_plan_modified_at')
                        _locked_total_requested = int(plan_data.get('total_days_requested', 0))
                        _pre_read_total_requested = int(prior_plan_data.get('total_days_requested', 0)) if prior_plan_data else 0
                        _cas_stale = (
                            pre_read_modified_at is not None
                            and _locked_modified_at != pre_read_modified_at
                        )
                        if _cas_stale and not _stale_abort:
                            logger.warning(
                                f"[P0-2/CAS] Plan {meal_plan_id} chunk {week_number}: "
                                f"detectado cambio externo durante el LLM call "
                                f"(pre_read={pre_read_modified_at!r}, locked={_locked_modified_at!r}). "
                                f"is_degraded={is_degraded}."
                            )
                            if is_degraded:
                                # El Smart Shuffle usó prior_days obsoletos — re-encolar.
                                # La transacción no tiene escrituras así que hace rollback silencioso.
                                _stale_abort = True
                            elif _locked_total_requested != _pre_read_total_requested and _pre_read_total_requested > 0:
                                # total_days_requested cambió → regeneración completa del plan, no solo shift.
                                # Los días del LLM pueden ser válidos pero el plan ya no los espera.
                                # Abortar para que el cron re-evalúe si este chunk sigue siendo necesario.
                                logger.warning(
                                    f"[P0-1/CAS] Plan {meal_plan_id} chunk {week_number}: "
                                    f"total_days_requested cambió ({_pre_read_total_requested} → {_locked_total_requested}). "
                                    f"Posible regeneración completa. Abortando merge."
                                )
                                _stale_abort = True
                            else:
                                # En modo LLM los días son generados fresh por el modelo y
                                # total_requested no cambió (solo fue un shift). Continuar merge.
                                logger.info(
                                    f"[P0-2/CAS] Modo LLM: continuando merge (cambio fue shift, "
                                    f"total_requested={_locked_total_requested} sin cambio)."
                                )
                                _stale_abort = False
                        elif not _stale_abort:
                            _stale_abort = False

                        # [P1-1] PRE-CHECK DE DUPLICADOS POR CONTENIDO.
                        # El idempotency marker `_merged_chunk_ids` puede perderse si la transacción
                        # 2 (UPDATE final con plan_data completo) falla pero la transacción 1 había
                        # commiteado los días (poco probable con el fix actual, pero defensivo).
                        # Si los últimos N días en `existing_days` tienen las MISMAS signatures de
                        # meals que `new_days`, asumir que el chunk ya fue mergeado en un intento
                        # previo cuyo marker se perdió. Esto previene duplicación silenciosa.
                        def _p11_meal_signature(day):
                            if not isinstance(day, dict):
                                return ()
                            return tuple(sorted(
                                (str(m.get('name') or ''), str(m.get('type') or ''))
                                for m in (day.get('meals') or []) if isinstance(m, dict)
                            ))

                        _p11_already_in_storage = False
                        _p11_new_count = len([d for d in new_days if isinstance(d, dict)])
                        if (
                            _p11_new_count > 0
                            and len(existing_days) >= _p11_new_count
                            and str(task_id) not in [str(x) for x in plan_data.get('_merged_chunk_ids', [])]
                        ):
                            _p11_last_existing = existing_days[-_p11_new_count:]
                            _p11_existing_sigs = [
                                _p11_meal_signature(d)
                                for d in _p11_last_existing
                                if isinstance(d, dict) and _p11_meal_signature(d) != ()
                            ]
                            _p11_new_sigs = [
                                _p11_meal_signature(d)
                                for d in new_days
                                if isinstance(d, dict) and _p11_meal_signature(d) != ()
                            ]
                            if (
                                _p11_existing_sigs
                                and _p11_new_sigs
                                and len(_p11_existing_sigs) == len(_p11_new_sigs)
                                and _p11_existing_sigs == _p11_new_sigs
                            ):
                                _p11_already_in_storage = True
                                logger.warning(
                                    f"[P1-1/PRE-CHECK] Chunk {task_id} (week {week_number}) parece "
                                    f"ya mergeado: las signatures de los últimos {_p11_new_count} "
                                    f"días en storage coinciden con new_days, pero el marker "
                                    f"_merged_chunk_ids está ausente. Re-añadiendo marker y "
                                    f"saltando merge para evitar duplicación silenciosa."
                                )
                                # Re-añadir marker antes del bloque idempotency normal.
                                _p11_ids = plan_data.get('_merged_chunk_ids', [])
                                if not isinstance(_p11_ids, list):
                                    _p11_ids = []
                                _p11_ids.append(str(task_id))
                                plan_data['_merged_chunk_ids'] = _p11_ids

                        # 2. Mergear asegurando continuidad 1..N (P0-1 FIX)
                        # Debido a /shift-plan, el days_offset original del chunk puede estar obsoleto.
                        # Re-renumeramos todo a partir de 1 para empalmar perfectamente sin huecos ni colisiones.
                        # [GAP E] Detectar y loguear duplicados previos.
                        days_dict = {}
                        existing_day_nums_seen = set()
                        duplicates_in_existing = []

                        idx = 1
                        for d in existing_days:
                            if isinstance(d, dict):
                                original_day = d.get('day', idx)
                                if original_day in existing_day_nums_seen:
                                    duplicates_in_existing.append(original_day)
                                existing_day_nums_seen.add(original_day)

                                d['day'] = idx
                                days_dict[idx] = d
                                idx += 1

                        if duplicates_in_existing:
                            logger.error(
                                f"[GAP E] Plan {meal_plan_id} tenía días duplicados en storage "
                                f"({duplicates_in_existing}). Deduplicando (última ocurrencia gana)."
                            )

                        # [P0-5 FIX] prior_count debe reflejar los días ÚNICOS post-deduplicación,
                        # no el len(existing_days) bruto. Si había duplicados, el conteo bruto
                        # es mayor que los días reales, causando falso-negativo en la validación.
                        prior_count = idx - 1  # idx se incrementó por cada día único procesado

                        sorted_new_days = sorted([d for d in new_days if isinstance(d, dict)], key=lambda x: x.get('day', 0))

                        # [P0-2] Day-level tagging: si el chunk se generó en flexible_mode,
                        # marcar cada día NUEVO con `_pantry_degraded=True` y el reason que
                        # detonó el modo. Sin esto, el frontend no puede distinguir días
                        # confiables (LLM con nevera live) de días que pueden tener
                        # ingredientes fuera de la nevera registrada (LLM con snapshot stale,
                        # nevera vacía, live caído, etc). El plan-level _current_mode ya
                        # existía via _activate_flexible_mode, pero vive a nivel plan; el
                        # banner per-día requiere el flag aquí.
                        _p02_flexible = bool(form_data.get("_pantry_flexible_mode"))
                        _p02_advisory_only = bool(form_data.get("_pantry_advisory_only"))
                        _p02_reason = (
                            form_data.get("_pantry_degraded_reason")
                            or form_data.get("_fresh_pantry_source")
                            or "unknown"
                        )

                        for d in sorted_new_days:
                            d['day'] = idx
                            if _p02_flexible:
                                d['_pantry_degraded'] = True
                                d['_pantry_degraded_reason'] = _p02_reason
                                if _p02_advisory_only:
                                    d['_pantry_advisory_only'] = True
                            days_dict[idx] = d
                            idx += 1

                        # Ordenar los dias para mantener coherencia en el array JSON
                        merged_days = [days_dict[k] for k in sorted(days_dict.keys())]

                        # [GAP 3 - VALIDACION POST-MERGE: continuidad 1..N]
                        # El set de days debe formar una secuencia contigua desde 1 hasta len(merged_days).
                        # Si hay hueco (ej. [1,2,3,5,6] sin dia 4) raise: la transaccion hace ROLLBACK
                        # automatico y el outer catch re-encola el chunk con backoff exponencial.
                        sorted_keys = sorted(days_dict.keys())
                        expected_keys = list(range(1, len(sorted_keys) + 1))
                        if sorted_keys != expected_keys:
                            gaps = sorted(set(expected_keys) - set(sorted_keys))
                            raise Exception(
                                f"[GAP3] Continuidad rota post-merge para plan {meal_plan_id} "
                                f"(chunk {week_number}). Keys={sorted_keys}, esperado={expected_keys}, "
                                f"huecos={gaps}. Abortando merge para preservar integridad del plan."
                            )

                        # [P0-4] HARD VALIDATION POST-MERGE: validar que cada día del chunk
                        # recién mergeado cumpla con la nevera bajo strict_quantities=True
                        # y tolerance=1.0 (sin holgura sobre el ledger). Si alguno falla,
                        # raise _PantryViolationPostMerge para que la transacción haga
                        # ROLLBACK automático y el outer catch pause el chunk con
                        # 'pending_user_action' en lugar de retry — un retry no resuelve el
                        # problema si la nevera no cambia.
                        #
                        # Solo validamos los días NUEVOS del chunk actual; días pre-existentes
                        # no se re-validan para no romper planes históricos cuya nevera ya cambió.
                        _p04_pantry = form_data.get("current_pantry_ingredients", []) or []
                        # [P0-5] Skip the post-merge hard pantry guard when the chunk already
                        # accepted a quantity violation under advisory/hybrid policy. Otherwise
                        # `_validate_merged_days_against_pantry` re-runs the same `strict_quantities=True`
                        # check that produced the original violation and raises `_PantryViolationPostMerge`,
                        # rolling back the merge — so the deliberately-accepted overshoot becomes
                        # a chunk pause instead of an annotation. The advisory annotation is the
                        # contract; the post-merge guard should only catch UNEXPECTED new violations.
                        _p04_advisory_skip = bool(form_data.get("_pantry_quantity_violations"))
                        if _p04_pantry and not _p04_advisory_skip:
                            _p04_new_start = int(days_offset) + 1
                            _p04_new_end = int(days_offset) + int(days_count)
                            _p04_ok, _p04_violations = _validate_merged_days_against_pantry(
                                merged_days,
                                _p04_pantry,
                                new_chunk_day_range=(_p04_new_start, _p04_new_end),
                            )
                            if not _p04_ok:
                                _sample = _p04_violations[0] if _p04_violations else {}
                                logger.error(
                                    f"[P0-4/POST-MERGE] Plan {meal_plan_id} chunk {week_number}: "
                                    f"{len(_p04_violations)}/{int(days_count)} día(s) violan pantry tras merge. "
                                    f"Pausando para intervención manual. Sample day={_sample.get('day')} "
                                    f"err={_sample.get('error', '')[:120]!r}"
                                )
                                raise _PantryViolationPostMerge(
                                    _p04_violations, pantry_size=len(_p04_pantry)
                                )

                        # 3. Recalcular contadores absolutos de forma segura
                        fallback_total = snap.get("totalDays", 7)
                        total_requested = int(plan_data.get('total_days_requested', fallback_total))

                        # [GAP 3] Limpieza de días huérfanos al regenerar en background
                        if len(merged_days) > total_requested:
                            logger.warning(f" [GAP 3] Recortando días huérfanos en chunk {week_number}. De {len(merged_days)} a {total_requested}")
                            merged_days = merged_days[:total_requested]

                        # [GAP E / P0.1 FIX] Invariante en memoria: el conteo final debe
                        # reflejar exactamente los días pre-existentes (deduplicados) más los
                        # nuevos del chunk actual.
                        #
                        # Antes la rama multi-chunk usaba `SELECT SUM(days_count) WHERE
                        # status='completed'` contra plan_chunk_queue para reconstruir el
                        # esperado. Ese conteo se desincroniza cuando un chunk se re-procesa
                        # tras fallo intermitente (timeout LLM, DB blip, race CAS): sus días
                        # YA viven en plan_data desde un merge previo committed, pero el row
                        # del queue sigue en 'processing'/'failed'. Resultado: prior_completed≈0,
                        # expected_total≈PLAN_CHUNK_SIZE+days_count, y planes 15d/30d quedaban
                        # atrapados en loop de re-fail con merged_days mucho mayor.
                        #
                        # prior_count (post-dedup) + days_count == len(merged_days) es la
                        # invariante natural del merge: days_dict se llena con prior_count
                        # entradas únicas y luego con days_count nuevas. Excepción: el guard
                        # [GAP 3] previo recorta a total_requested cuando hay días huérfanos.
                        is_rolling_refill = snap.get('_is_rolling_refill', False)
                        if prior_count != prior_count_raw:
                            logger.warning(
                                f"[P0-5] prior_count ajustado por dedup en plan {meal_plan_id}: "
                                f"raw={prior_count_raw}, post-dedup={prior_count}."
                            )

                        expected_merge_count = prior_count + days_count
                        was_trimmed_to_total = (
                            len(merged_days) == total_requested
                            and expected_merge_count > total_requested
                        )
                        if not was_trimmed_to_total and len(merged_days) != expected_merge_count:
                            mode_label = 'rolling_refill' if is_rolling_refill else 'multi_chunk'
                            logger.error(
                                f"[GAP E] Plan {meal_plan_id} chunk {week_number} ({mode_label}): "
                                f"len(merged_days)={len(merged_days)} pero esperado {expected_merge_count} "
                                f"(prior_count={prior_count}, days_count={days_count}). "
                                f"Posible corrupción. Abortando merge."
                            )
                            raise Exception(
                                f"[GAP E] Conteo inconsistente en merge: got {len(merged_days)}, "
                                f"expected {expected_merge_count}"
                            )

                        # [P0-4 FIX] Idempotencia: verificar si este chunk ya fue mergeado previamente.
                        # Si el merge se commiteó pero la shopping list falló y el chunk fue re-encolado,
                        # los días ya están en plan_data. Re-mergearlos duplicaría silenciosamente.
                        merged_chunk_ids = plan_data.get('_merged_chunk_ids', [])
                        chunk_already_merged = str(task_id) in [str(x) for x in merged_chunk_ids]

                        if chunk_already_merged:
                            # Los días ya están en el plan — solo necesitamos reintentar la shopping list
                            logger.info(
                                f"[P0-4] Chunk {task_id} ya fue mergeado previamente en plan {meal_plan_id}. "
                                f"Saltando merge, solo recalculando shopping list."
                            )
                            new_total = len(existing_days)
                            new_status = plan_data.get('generation_status', 'partial')
                            merged_days = existing_days

                            # [P0-2] Backfill defensivo: si el chunk fue mergeado en una versión pre-P0-2
                            # (o si la transacción anterior se commit-eó pero la lección no se escribió por
                            # un bug), _last_chunk_learning puede no corresponder a ESTE chunk. Mientras
                            # estamos aún dentro del FOR UPDATE, reconstruimos la lección desde
                            # plan_chunk_queue.learning_metrics si existe, o un stub si no, y reescribimos
                            # plan_data atómicamente. Esto cierra la única ventana donde un chunk podía
                            # pasar a 'completed' sin dejar lección.
                            _persisted_lesson = plan_data.get('_last_chunk_learning') or {}
                            _persisted_chunk_id = _persisted_lesson.get('chunk') if isinstance(_persisted_lesson, dict) else None
                            if _persisted_chunk_id != week_number:
                                cursor.execute(
                                    "SELECT learning_metrics FROM plan_chunk_queue WHERE id = %s",
                                    (task_id,),
                                )
                                _row_lm = cursor.fetchone()
                                _queue_lm = (_row_lm or {}).get('learning_metrics') if _row_lm else None
                                if isinstance(_queue_lm, str):
                                    try:
                                        _queue_lm = json.loads(_queue_lm)
                                    except Exception:
                                        _queue_lm = None

                                from datetime import datetime as _dt_bf, timezone as _tz_bf
                                if _queue_lm:
                                    _backfill_lesson = {
                                        'repeat_pct': _queue_lm.get('learning_repeat_pct', 0),
                                        'ingredient_base_repeat_pct': _queue_lm.get('ingredient_base_repeat_pct', 0),
                                        'rejection_violations': _queue_lm.get('rejection_violations', 0),
                                        'allergy_violations': _queue_lm.get('allergy_violations', 0),
                                        'fatigued_violations': _queue_lm.get('fatigued_violations', 0),
                                        'repeated_bases': _queue_lm.get('sample_repeated_bases', []),
                                        'repeated_meal_names': _queue_lm.get('sample_repeats', []),
                                        'rejected_meals_that_reappeared': _queue_lm.get('sample_rejection_hits', []),
                                        'allergy_hits': _queue_lm.get('sample_allergy_hits', []),
                                    'chunk': week_number,
                                    'timestamp': _dt_bf.now(_tz_bf.utc).isoformat(),
                                    'metrics_unavailable': False,
                                    'low_confidence': bool(_queue_lm.get('inventory_activity_proxy_used') or _queue_lm.get('sparse_logging_proxy_used')),
                                    'learning_signal_strength': _queue_lm.get(
                                        'learning_signal_strength',
                                        'weak' if (_queue_lm.get('inventory_activity_proxy_used') or _queue_lm.get('sparse_logging_proxy_used')) else 'strong'
                                    ),
                                    'backfilled': True,
                                }
                                else:
                                    _backfill_lesson = {
                                        'repeat_pct': 0,
                                        'ingredient_base_repeat_pct': 0,
                                        'rejection_violations': 0,
                                        'allergy_violations': 0,
                                        'fatigued_violations': 0,
                                        'repeated_bases': [],
                                        'repeated_meal_names': [],
                                        'rejected_meals_that_reappeared': [],
                                        'allergy_hits': [],
                                        'chunk': week_number,
                                        'timestamp': _dt_bf.now(_tz_bf.utc).isoformat(),
                                        'metrics_unavailable': True,
                                        'low_confidence': True,
                                        'backfilled': True,
                                    }
                                logger.warning(
                                    f"[P0-2/BACKFILL] Chunk {task_id} (week {week_number}) ya mergeado pero "
                                    f"sin lección coherente en plan_data (encontrada chunk={_persisted_chunk_id}). "
                                    f"Reconstruyendo desde plan_chunk_queue.learning_metrics "
                                    f"(disponible={_queue_lm is not None})."
                                )
                                plan_data['_last_chunk_learning'] = _backfill_lesson
                                _recent_bf = plan_data.get('_recent_chunk_lessons', [])
                                if not isinstance(_recent_bf, list):
                                    _recent_bf = []
                                # No duplicar si ya hay una entrada para este week_number
                                _recent_bf = [l for l in _recent_bf if not (isinstance(l, dict) and l.get('chunk') == week_number)]
                                _recent_bf.append(_backfill_lesson)
                                
                                _total_req_bf = int(plan_data.get('total_days_requested', 7))
                                _win_size_bf = _rolling_lessons_window_cap(_total_req_bf)
                                plan_data['_recent_chunk_lessons'] = _recent_bf[-_win_size_bf:]

                                # [P0-7] Backfill: also update critical permanent if applicable
                                _is_crit_bf = (
                                    int(_backfill_lesson.get('rejection_violations') or 0) > 0
                                    or int(_backfill_lesson.get('allergy_violations') or 0) > 0
                                    or float(_backfill_lesson.get('ingredient_base_repeat_pct') or 0) >= 85.0
                                )
                                if _is_crit_bf:
                                    _crit_bf = plan_data.get('_critical_lessons_permanent', [])
                                    if not isinstance(_crit_bf, list):
                                        _crit_bf = []
                                    _crit_bf = [l for l in _crit_bf if not (isinstance(l, dict) and l.get('chunk') == week_number)]
                                    _backfill_lesson['_critical'] = True
                                    _crit_bf.append(_backfill_lesson)
                                    from constants import CHUNK_CRITICAL_LESSONS_MAX
                                    plan_data['_critical_lessons_permanent'] = _prune_critical_lessons_with_priority(
                                        _crit_bf, CHUNK_CRITICAL_LESSONS_MAX
                                    )

                                # [P0-A/BACKFILL] Mantener `_lifetime_lessons_history` y
                                # `_lifetime_lessons_summary` simétricos con el path normal
                                # (~línea 13025). Sin esto, los chunks que entran por backfill
                                # (lección original perdida en transacción previa) NO contribuyen
                                # al historial cross-plan y el plan siguiente hereda lecciones
                                # incompletas. Variables prefijadas con _bf_ para no colisionar
                                # con el path normal en el `else` siguiente.
                                from constants import LIFETIME_LESSONS_WINDOW_DAYS
                                _bf_history = plan_data.get('_lifetime_lessons_history', [])
                                if not isinstance(_bf_history, list):
                                    _bf_history = []
                                if not _bf_history and snap.get('_inherited_lifetime_lessons'):
                                    _bf_history = snap['_inherited_lifetime_lessons'].get('history', [])
                                    if not plan_data.get('_lifetime_lessons_summary'):
                                        plan_data['_lifetime_lessons_summary'] = snap['_inherited_lifetime_lessons'].get('summary', {})
                                # Evitar duplicados: si ya hay una entrada para este week_number
                                # (p. ej. backfill que reemplaza un stub previo), la sustituimos.
                                _bf_history = [
                                    l for l in _bf_history
                                    if not (isinstance(l, dict) and l.get('chunk') == week_number)
                                ]
                                _bf_history.append(_backfill_lesson)
                                _bf_cutoff = (datetime.now(timezone.utc) - timedelta(days=LIFETIME_LESSONS_WINDOW_DAYS)).isoformat()
                                _bf_history = [
                                    l for l in _bf_history
                                    if isinstance(l, dict) and (l.get('timestamp') or "") >= _bf_cutoff
                                ]
                                plan_data['_lifetime_lessons_history'] = _bf_history

                                # Recomputar `_lifetime_lessons_summary` desde el historial
                                # filtrado, idéntico schema al path normal: top_rejection_hits,
                                # top_repeated_bases, top_repeated_meal_names y
                                # permanent_meal_blocklist (meals que aparecen en >=2 chunks).
                                _bf_lifetime = {
                                    "total_rejection_violations": sum(int(l.get("rejection_violations") or 0) for l in _bf_history),
                                    "total_allergy_violations": sum(int(l.get("allergy_violations") or 0) for l in _bf_history),
                                    "top_rejection_hits": [],
                                    "top_repeated_bases": [],
                                    "top_repeated_meal_names": [],
                                    "permanent_meal_blocklist": [],
                                    "_lifetime_window_days": LIFETIME_LESSONS_WINDOW_DAYS,
                                }
                                _bf_rej_set = set()
                                _bf_base_set = set()
                                _bf_meal_chunk_counts: dict = {}
                                for _bf_l in _bf_history:
                                    _bf_ch = _bf_l.get("chunk")
                                    for _bf_rj in (_bf_l.get("rejected_meals_that_reappeared") or []):
                                        _bf_rej_set.add(_bf_rj)
                                    for _bf_rb_entry in (_bf_l.get("repeated_bases") or []):
                                        if isinstance(_bf_rb_entry, dict):
                                            for _bf_b in (_bf_rb_entry.get("bases") or []):
                                                _bf_base_set.add(_bf_b)
                                    for _bf_rm in (_bf_l.get("repeated_meal_names") or []):
                                        if not _bf_rm:
                                            continue
                                        _bf_meal_chunk_counts.setdefault(str(_bf_rm), set()).add(_bf_ch)
                                _bf_lifetime["top_rejection_hits"] = list(_bf_rej_set)[:20]
                                _bf_lifetime["top_repeated_bases"] = list(_bf_base_set)[:20]
                                _bf_all_repeated_meals = sorted(
                                    _bf_meal_chunk_counts.keys(),
                                    key=lambda m: -len(_bf_meal_chunk_counts[m]),
                                )
                                _bf_lifetime["top_repeated_meal_names"] = _bf_all_repeated_meals[:30]
                                _bf_lifetime["permanent_meal_blocklist"] = [
                                    m for m in _bf_all_repeated_meals
                                    if len(_bf_meal_chunk_counts[m]) >= 2
                                ][:50]
                                plan_data['_lifetime_lessons_summary'] = _bf_lifetime

                                logger.info(
                                    f"[P0-A/BACKFILL] Chunk {week_number} plan {meal_plan_id}: "
                                    f"_lifetime_lessons_history actualizado vía backfill "
                                    f"(history_len={len(_bf_history)}, "
                                    f"top_blocklist={len(_bf_lifetime['permanent_meal_blocklist'])})."
                                )

                                # [P0-B] Path chunk_already_merged: estamos reescribiendo
                                # plan_data completo dentro del FOR UPDATE pero el sellado
                                # de `_plan_modified_at` en línea 13312 está en la rama
                                # `else: # Merge normal`, no aquí. Sin sellar manualmente,
                                # un chunk concurrente que leyó plan_data ANTES de este
                                # UPDATE no detectaría el cambio (mismo timestamp). Sellamos
                                # en memoria justo antes del UPDATE para mantener la
                                # invariante "todo UPDATE a plan_data refresca el sello CAS".
                                from datetime import datetime as _dt_p0b, timezone as _tz_p0b_local
                                plan_data['_plan_modified_at'] = _dt_p0b.now(_tz_p0b_local.utc).isoformat()
                                # Re-escribir plan_data dentro del mismo FOR UPDATE
                                cursor.execute(
                                    "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                                    (json.dumps(plan_data, ensure_ascii=False), meal_plan_id),
                                )
                                # Sellar también en la cola que la lección ya está garantizada
                                cursor.execute(
                                    """
                                    UPDATE plan_chunk_queue
                                    SET learning_persisted_at = COALESCE(learning_persisted_at, NOW())
                                    WHERE id = %s
                                    """,
                                    (task_id,),
                                )

                            update_result = [{
                                "new_total": new_total,
                                "new_status": new_status,
                                "full_plan_data": plan_data,
                                "_skip_merge_write": True,  # señal para no re-escribir plan_data
                            }]
                        else:
                            # Merge normal: primera vez que este chunk se integra

                            plan_data['days'] = merged_days
                            new_total = len(merged_days)
                            plan_data['total_days_generated'] = new_total

                            if form_data.get("_sparse_logging_proxy"):
                                _sfc = plan_data.get("_sparse_forced_chunks", [])
                                if week_number not in _sfc:
                                    _sfc.append(week_number)
                                    plan_data["_sparse_forced_chunks"] = _sfc

                            # [P1-2 FIX] / [P1-E FIX] Guardar última técnica del chunk generado para uso futuro.
                            # Fallback robusto: si _skeleton no tiene técnicas (modo degradado/emergency),
                            # intentar extraerla de los días generados o del form_data del pipeline.
                            chunk_skeleton = result.get("_skeleton", {})
                            chunk_techniques = chunk_skeleton.get("_selected_techniques", [])
                            if chunk_techniques:
                                plan_data['last_technique'] = chunk_techniques[-1]
                            else:
                                # [P1-E] Fallback 1: Derivar técnica dominante de los días finales (Smart Shuffle friendly)
                                _dominant_tech = _get_dominant_technique(new_days)
                                if _dominant_tech:
                                    plan_data['last_technique'] = _dominant_tech
                                    logger.info(f"[P1-E] last_technique (dominante) extraída de días finales: {_dominant_tech}")
                                else:
                                    # Fallback 2: buscar en el último día individualmente (legacy check).
                                    # [P0-5] Renamed the loop var from `_t` to `_tech`: the
                                    # outer scope of `_chunk_worker` already binds `_t = time`
                                    # at L12033 (`import time as _t`), used downstream at
                                    # L16768 (`_t.time() - chunk_start_ts`). Rebinding `_t` to
                                    # a string here clobbered the time module for the rest of
                                    # the function, so the success-path metric write raised
                                    # `'NoneType' object has no attribute 'time'` and chunk
                                    # observability was permanently broken on the LLM happy path.
                                    _fallback_tech = None
                                    for _nd in reversed(new_days):
                                        if isinstance(_nd, dict):
                                            _tech = _nd.get('_technique') or _nd.get('technique')
                                            if _tech:
                                                _fallback_tech = _tech
                                                break
                                    if _fallback_tech:
                                        plan_data['last_technique'] = _fallback_tech
                                        logger.info(f"[P1-2] last_technique extraída de último día: {_fallback_tech}")
                                    elif form_data.get("_last_technique"):
                                        # Fallback 3: mantener la técnica del chunk anterior si este no produjo ninguna
                                        plan_data['last_technique'] = form_data["_last_technique"]
                                        logger.info(f"[P1-2] last_technique preservada del chunk anterior: {form_data['_last_technique']}")

                            # Rolling refills siempre marcan 'complete': la ventana de 3 días está llena.
                            # Planes normales usan total_requested para determinar si faltan más chunks.
                            if is_rolling_refill:
                                new_status = "complete"
                            else:
                                new_status = "complete" if new_total >= total_requested else "partial"
                            plan_data['generation_status'] = new_status

                            # [P0-4] Estampar idempotency marker ANTES del commit
                            if '_merged_chunk_ids' not in plan_data:
                                plan_data['_merged_chunk_ids'] = []
                            plan_data['_merged_chunk_ids'].append(str(task_id))

                            # [P1-5 / P0-4 / P0-2] Persistir métricas de aprendizaje en plan_data con muestras
                            # concretas para que el chunk siguiente pueda construir LECCIONES accionables.
                            # _last_chunk_learning: lección del chunk inmediatamente anterior (compat).
                            # _recent_chunk_lessons: ventana rolling de las últimas 4 lecciones (P0-4).
                            #
                            # [P0-2] Siempre escribimos una lección, incluso si _calculate_learning_metrics
                            # falló o el path es Smart Shuffle sin métricas. Si no hay datos, persistimos
                            # un stub con metrics_unavailable=True para que la ventana rolling no quede
                            # con huecos silenciosos (chunk N+1 necesita saber que chunk N ocurrió,
                            # aunque no se pudieron medir las violaciones).
                            # [P0-1-RECOVERY/WORKER-FIX] datetime/timezone son globals del módulo.
                            if learning_metrics:
                                _new_lesson = {
                                    'repeat_pct': learning_metrics.get('learning_repeat_pct', 0),
                                    'ingredient_base_repeat_pct': learning_metrics.get('ingredient_base_repeat_pct', 0),
                                    'rejection_violations': learning_metrics.get('rejection_violations', 0),
                                    'allergy_violations': learning_metrics.get('allergy_violations', 0),
                                    'fatigued_violations': learning_metrics.get('fatigued_violations', 0),
                                    'repeated_bases': learning_metrics.get('sample_repeated_bases', []),
                                    'repeated_meal_names': learning_metrics.get('sample_repeats', []),
                                    'rejected_meals_that_reappeared': learning_metrics.get('sample_rejection_hits', []),
                                    'allergy_hits': learning_metrics.get('sample_allergy_hits', []),
                                    'chunk': week_number,
                                    'timestamp': datetime.now(timezone.utc).isoformat(),
                                    'metrics_unavailable': False,
                                    'low_confidence': bool(learning_metrics.get('inventory_activity_proxy_used') or learning_metrics.get('sparse_logging_proxy_used')),
                                    'learning_signal_strength': learning_metrics.get(
                                        'learning_signal_strength',
                                        'weak' if (learning_metrics.get('inventory_activity_proxy_used') or learning_metrics.get('sparse_logging_proxy_used')) else 'strong'
                                    ),
                                }
                            else:
                                _prior_days_for_stub = prior_plan_data.get("days", [])
                                _prior_meals_for_stub = [
                                    m.get("name") for d in _prior_days_for_stub
                                    for m in (d.get("meals") or [])
                                    if m.get("name") and m.get("status") not in ["swapped_out", "skipped", "rejected"]
                                ]
                                _new_lesson = None
                                if _prior_meals_for_stub and new_days:
                                    try:
                                        _stub_rechazos = get_active_rejections(user_id=user_id)
                                        _stub_rejected_names = [r["meal_name"] for r in _stub_rechazos] if _stub_rechazos else []
                                        _stub_metrics = _calculate_learning_metrics(
                                            new_days=new_days,
                                            prior_meals=_prior_meals_for_stub,
                                            prior_days=_prior_days_for_stub,
                                            rejected_names=_stub_rejected_names,
                                            allergy_keywords=current_allergies,
                                            fatigued_ingredients=list(_learned_bases_to_avoid)
                                        )
                                        _new_lesson = {
                                            'repeat_pct': _stub_metrics.get('learning_repeat_pct', 0),
                                            'ingredient_base_repeat_pct': _stub_metrics.get('ingredient_base_repeat_pct', 0),
                                            'rejection_violations': _stub_metrics.get('rejection_violations', 0),
                                            'allergy_violations': _stub_metrics.get('allergy_violations', 0),
                                            'fatigued_violations': _stub_metrics.get('fatigued_violations', 0),
                                            'repeated_bases': _stub_metrics.get('sample_repeated_bases', []),
                                            'repeated_meal_names': _stub_metrics.get('sample_repeats', []),
                                            'rejected_meals_that_reappeared': _stub_metrics.get('sample_rejection_hits', []),
                                            'allergy_hits': _stub_metrics.get('sample_allergy_hits', []),
                                            'chunk': week_number,
                                            'timestamp': datetime.now(timezone.utc).isoformat(),
                                            'metrics_unavailable': False,
                                            'low_confidence': True,
                                            'learning_signal_strength': 'weak',
                                        }
                                        logger.info(f"[P0-3/STUB-LESSON] Chunk {week_number} plan {meal_plan_id}: "
                                                    f"Métricas reconstruidas a partir de prior_meals.")
                                    except Exception as e:
                                        logger.error(f"Error reconstruyendo stub metrics: {e}")
                                
                                if not _new_lesson:
                                    plan_data['_chunk_learning_stub_count'] = plan_data.get('_chunk_learning_stub_count', 0) + 1
                                    if plan_data['_chunk_learning_stub_count'] >= 2:
                                        logger.error(f"[P0-3/STUB-ALERT] El plan {meal_plan_id} acumula {plan_data['_chunk_learning_stub_count']} stubs puros. Posible problema sistémico.")
                                    _new_lesson = {
                                        'repeat_pct': 0,
                                        'ingredient_base_repeat_pct': 0,
                                        'rejection_violations': 0,
                                        'allergy_violations': 0,
                                        'fatigued_violations': 0,
                                        'repeated_bases': [],
                                        'repeated_meal_names': [],
                                        'rejected_meals_that_reappeared': [],
                                        'allergy_hits': [],
                                        'chunk': week_number,
                                        'timestamp': datetime.now(timezone.utc).isoformat(),
                                        'metrics_unavailable': True,
                                        'chunk_learning_stub_count': plan_data['_chunk_learning_stub_count'],
                                        'low_confidence': True,
                                        'learning_signal_strength': 'weak',
                                    }
                                    logger.warning(
                                        f"[P0-3/STUB-LESSON] Chunk {week_number} plan {meal_plan_id}: "
                                        f"learning_metrics no disponible y no se pudo reconstruir. Persistiendo lección stub pura para "
                                        f"mantener la cadena de aprendizaje sin huecos."
                                    )
                            plan_data['_last_chunk_learning'] = _new_lesson
                            # [P0-5] Propagate the advisory/hybrid quantity-violation annotation
                            # captured in form_data (cron_tasks.py:15020) to plan_data so the
                            # UI/admin can surface "this chunk overshot pantry by X%" without
                            # digging into plan_chunk_queue.learning_metrics. Cleared if the
                            # current chunk did not violate, so the field never carries stale
                            # data from a prior chunk.
                            _qty_violation_str = form_data.get("_pantry_quantity_violations")
                            if _qty_violation_str:
                                plan_data['_pantry_quantity_violations'] = str(_qty_violation_str)[:1000]
                            else:
                                plan_data.pop('_pantry_quantity_violations', None)
                            # [P0-4] Append a la ventana rolling (P1-A: 8 para planes largos, else 4).
                            _recent = plan_data.get('_recent_chunk_lessons', [])
                            if not isinstance(_recent, list):
                                _recent = []
                            _recent.append(_new_lesson)
                            
                            _total_req = int(plan_data.get('total_days_requested', 7))
                            _win_size = _rolling_lessons_window_cap(_total_req)
                            plan_data['_recent_chunk_lessons'] = _recent[-_win_size:]

                            # [P0-7] Extraer lecciones críticas permanentes que sobreviven al rolling window.
                            # Criterios: rejection_violations > 0, allergy_violations > 0,
                            # ingredient_base_repeat_pct >= 85, o confidence >= 0.85 con señal fuerte.
                            from constants import CHUNK_CRITICAL_LESSONS_MAX
                            _critical_permanent = plan_data.get('_critical_lessons_permanent', [])
                            if not isinstance(_critical_permanent, list):
                                _critical_permanent = []
                            _is_critical = (
                                int(_new_lesson.get('rejection_violations') or 0) > 0
                                or int(_new_lesson.get('allergy_violations') or 0) > 0
                                or float(_new_lesson.get('ingredient_base_repeat_pct') or 0) >= 85.0
                                or (
                                    _new_lesson.get('learning_signal_strength') == 'strong'
                                    and not _new_lesson.get('low_confidence')
                                    and not _new_lesson.get('metrics_unavailable')
                                )
                            )
                            if _is_critical:
                                _new_lesson['_critical'] = True
                                _critical_permanent.append(_new_lesson)
                                _critical_permanent = _prune_critical_lessons_with_priority(
                                    _critical_permanent, CHUNK_CRITICAL_LESSONS_MAX
                                )
                                plan_data['_critical_lessons_permanent'] = _critical_permanent
                                logger.info(
                                    f"[P0-7/CRITICAL] Chunk {week_number}: lección crítica preservada permanentemente. "
                                    f"rej_viol={_new_lesson.get('rejection_violations')}, "
                                    f"alg_viol={_new_lesson.get('allergy_violations')}, "
                                    f"base_repeat={_new_lesson.get('ingredient_base_repeat_pct')}%. "
                                    f"Total permanentes: {len(_critical_permanent)}"
                                )

                            # [P1-5] Mantener historial para ventana de 60 días
                            from constants import LIFETIME_LESSONS_WINDOW_DAYS
                            # [P0-1-RECOVERY/WORKER-FIX] timedelta es global del módulo.

                            _history = plan_data.get('_lifetime_lessons_history', [])
                            if not isinstance(_history, list):
                                _history = []

                            if not _history and snap.get('_inherited_lifetime_lessons'):
                                _history = snap['_inherited_lifetime_lessons'].get('history', [])
                                if not plan_data.get('_lifetime_lessons_summary'):
                                    plan_data['_lifetime_lessons_summary'] = snap['_inherited_lifetime_lessons'].get('summary', {})

                            # [P0-5] Capture prior summary totals BEFORE we overwrite the
                            # summary below. If a prior summary exists but `_history` is
                            # empty after the 60-day cutoff (or because this plan was
                            # restored without history), the recompute below would reset
                            # totals to just the current chunk — silently losing the
                            # lifetime accumulation. We fold the prior totals into the
                            # new summary so accumulation survives a history-window roll.
                            _prior_summary_for_carryover = plan_data.get('_lifetime_lessons_summary') or {}
                            _history_was_empty_pre_append = not [l for l in _history if isinstance(l, dict)]

                            _history.append(_new_lesson)

                            _cutoff = (datetime.now(timezone.utc) - timedelta(days=LIFETIME_LESSONS_WINDOW_DAYS)).isoformat()
                            _history = [l for l in _history if isinstance(l, dict) and (l.get('timestamp') or "") >= _cutoff]
                            plan_data['_lifetime_lessons_history'] = _history

                            # [P1-A/P1-5] Recalcular _lifetime_lessons_summary desde el historial filtrado.
                            # [P1-5] Schema extendido con `top_repeated_meal_names` y
                            # `permanent_meal_blocklist`. Antes solo bases y rejections persistían
                            # cross-chunk; los meal_names repetidos se perdían cuando salían del
                            # rolling window (cap 8 en plan 30d con 10+ chunks). Ahora:
                            #   - top_repeated_meal_names: unión de meal_names repetidos en cualquier chunk del history.
                            #   - permanent_meal_blocklist: meals que aparecen en >=2 chunks (señal fuerte de
                            #     repetición sistémica → nunca regenerarse, distinto de "repetido en 1 chunk
                            #     que puede ser un patrón aceptable").
                            _lifetime = {
                                "total_rejection_violations": sum(int(l.get("rejection_violations") or 0) for l in _history),
                                "total_allergy_violations": sum(int(l.get("allergy_violations") or 0) for l in _history),
                                "top_rejection_hits": [],
                                "top_repeated_bases": [],
                                "top_repeated_meal_names": [],
                                "permanent_meal_blocklist": [],
                                "_lifetime_window_days": LIFETIME_LESSONS_WINDOW_DAYS
                            }
                            # [P0-5] Carry over the prior summary's totals when the history was
                            # empty before appending the current chunk — i.e., the prior totals
                            # represent lessons no longer present in `_history` (60-day window
                            # eviction or restored-plan-without-history). Without this, total
                            # rejection/allergy counts silently reset every time history rolls.
                            if _history_was_empty_pre_append and _prior_summary_for_carryover:
                                _lifetime["total_rejection_violations"] += int(
                                    _prior_summary_for_carryover.get("total_rejection_violations") or 0
                                )
                                _lifetime["total_allergy_violations"] += int(
                                    _prior_summary_for_carryover.get("total_allergy_violations") or 0
                                )

                            # [P1-4] Aggregación con DECAY TEMPORAL.
                            # Antes los sets/counts trataban todas las lecciones por igual: una
                            # rejection del chunk 1 (8 semanas atrás) competía con una del chunk
                            # 9 (1 semana atrás) por el cap de 20. Si el cap se llenaba con
                            # ítems viejos, los recientes no llegaban al prompt.
                            # Ahora ponderamos por `LIFETIME_LESSON_WEEKLY_DECAY ** weeks_old` y
                            # rankeamos por peso descendente. Items con peso < LIFETIME_LESSON_
                            # MIN_WEIGHT se descartan (forward-compat para ventanas amplias).
                            from constants import (
                                LIFETIME_LESSON_MIN_WEIGHT as _P14_MIN_W,
                            )
                            _p14_now = datetime.now(timezone.utc)
                            _rej_weights: dict = {}
                            _base_weights: dict = {}
                            _meal_chunk_counts: dict = {}  # meal_name → set(chunk) (heurística permanent_blocklist)
                            _meal_weights: dict = {}       # meal_name → suma de pesos (ranking)
                            # [P1-7] Tracking de provenance para alertar si el ratio proxy/synthesis
                            # del lifetime crece. _proxy_count = lessons con provenance != 'user_logs'.
                            _p17_user_logs_count = 0
                            _p17_proxy_count = 0
                            for _l in _history:
                                _w = compute_lifetime_lesson_weight(_l, now=_p14_now)
                                if _w < float(_P14_MIN_W):
                                    continue  # lesson too old to influence ranking
                                # [P1-7] Aplicar factor por provenance: lessons de logs reales pesan
                                # 1.0; las que vienen de proxy de inventario, síntesis o stub se
                                # multiplican por LIFETIME_LESSON_PROXY_WEIGHT_FACTOR (0.5). Sin esto
                                # un usuario que no loguea tendría todo su lifetime dominado por
                                # señales de baja confianza (proxy/synthesis), bloqueando proteínas
                                # o platos que en realidad nunca tuvo problema con — solo no los
                                # logueó. Multiplicación porque combina con decay temporal:
                                # final = decay_weight * provenance_factor.
                                _prov = _derive_learning_provenance(_l)
                                if _prov == "user_logs":
                                    _p17_user_logs_count += 1
                                else:
                                    _p17_proxy_count += 1
                                _w *= 1.0 if _prov == "user_logs" else LIFETIME_LESSON_PROXY_WEIGHT_FACTOR
                                _ch = _l.get("chunk")
                                for _rj in (_l.get("rejected_meals_that_reappeared") or []):
                                    _key = str(_rj)
                                    _rej_weights[_key] = _rej_weights.get(_key, 0.0) + _w
                                for _rb_entry in (_l.get("repeated_bases") or []):
                                    if isinstance(_rb_entry, dict):
                                        for _b in (_rb_entry.get("bases") or []):
                                            _bk = str(_b)
                                            _base_weights[_bk] = _base_weights.get(_bk, 0.0) + _w
                                # [P1-5] Track meal_name → set(chunk) para detectar repetición real
                                # cross-chunk. Si el mismo meal_name apareció en chunks 1 y 4, va al
                                # permanent_blocklist; si solo en 1, va a top_repeated_meal_names.
                                # [P1-4] _meal_weights paralelo para rankear por recencia dentro de
                                # los caps; el conteo distinto de chunks sigue determinando si va
                                # al permanent_blocklist (heurística "≥ 2 chunks distintos").
                                for _rm in (_l.get("repeated_meal_names") or []):
                                    if not _rm:
                                        continue
                                    _mk = str(_rm)
                                    _meal_chunk_counts.setdefault(_mk, set()).add(_ch)
                                    _meal_weights[_mk] = _meal_weights.get(_mk, 0.0) + _w

                            # [P1-4] Sort por peso descendente (recencia primero), luego truncar.
                            _lifetime["top_rejection_hits"] = [
                                k for k, _ in sorted(
                                    _rej_weights.items(), key=lambda kv: -kv[1]
                                )
                            ][:20]
                            _lifetime["top_repeated_bases"] = [
                                k for k, _ in sorted(
                                    _base_weights.items(), key=lambda kv: -kv[1]
                                )
                            ][:20]
                            # [P1-5/P1-4] Cap dinámico: 30 default. Cubre plan 30d con ~10 chunks ×
                            # 3 platos repetidos por chunk = 30 entradas máximas razonables.
                            # Tie-break: peso (recencia) primero; #chunks distintos segundo.
                            _all_repeated_meals = sorted(
                                _meal_weights.keys(),
                                key=lambda m: (-_meal_weights[m], -len(_meal_chunk_counts[m])),
                            )
                            _lifetime["top_repeated_meal_names"] = _all_repeated_meals[:30]
                            # [P1-5] permanent_blocklist: meals con >=2 chunks distintos.
                            # Razón: 1 chunk puede ser ruido (LLM regeneró por casualidad); 2+ es patrón.
                            # [P1-4] Sort por peso descendente para que los items más recientes
                            # ocupen los primeros slots cuando el cap de 50 se llene.
                            _lifetime["permanent_meal_blocklist"] = sorted(
                                [m for m in _all_repeated_meals if len(_meal_chunk_counts[m]) >= 2],
                                key=lambda m: -_meal_weights[m],
                            )[:50]  # Cap de seguridad para no inflar el prompt indefinidamente.
                            # [P1-7] Persistir el ratio proxy/total para que /admin/metrics y
                            # crons de alerta puedan agregar cross-plan sin reescanear el
                            # historial completo. Default 0 cuando no hay lessons (chunk 1).
                            _p17_total_lessons = _p17_user_logs_count + _p17_proxy_count
                            _lifetime["_lifetime_proxy_ratio"] = (
                                round(_p17_proxy_count / _p17_total_lessons, 3)
                                if _p17_total_lessons > 0 else 0.0
                            )
                            _lifetime["_lifetime_user_logs_count"] = _p17_user_logs_count
                            _lifetime["_lifetime_proxy_count"] = _p17_proxy_count
                            # [P1-7] Si el ratio supera CHUNK_MAX_LIFETIME_PROXY_RATIO, emitimos
                            # evento de telemetría ('lifetime_proxy_ratio_exceeded') para que
                            # /admin/metrics pueda alertar. NO bloqueamos aquí — el gate ya tiene
                            # su propia pausa (chronic_zero_logging en línea ~9408). Esto es
                            # observabilidad complementaria.
                            try:
                                from constants import (
                                    CHUNK_MAX_LIFETIME_PROXY_RATIO as _P17_RATIO_THRESHOLD,
                                    CHUNK_LIFETIME_PROXY_MIN_TOTAL as _P17_MIN_TOTAL,
                                )
                                if (
                                    _p17_total_lessons >= int(_P17_MIN_TOTAL)
                                    and _lifetime["_lifetime_proxy_ratio"] >= float(_P17_RATIO_THRESHOLD)
                                ):
                                    _record_chunk_lesson_telemetry(
                                        user_id=user_id,
                                        meal_plan_id=meal_plan_id,
                                        week_number=int(week_number),
                                        event="lifetime_proxy_ratio_exceeded",
                                        synthesized_count=_p17_proxy_count,
                                        queue_count=_p17_total_lessons,
                                        metadata={
                                            "ratio": _lifetime["_lifetime_proxy_ratio"],
                                            "threshold": float(_P17_RATIO_THRESHOLD),
                                            "min_total": int(_P17_MIN_TOTAL),
                                        },
                                    )
                            except Exception as _p17_alert_err:
                                logger.debug(
                                    f"[P1-7] Telemetría lifetime_proxy_ratio_exceeded falló: {_p17_alert_err}"
                                )
                            plan_data['_lifetime_lessons_summary'] = _lifetime


                            # [P0-1 FIX] learning_metrics se persiste DENTRO de T1 (en el
                            # mismo FOR UPDATE que escribe days + learning fields a plan_data).
                            # Esto garantiza que plan_data._last_chunk_learning y
                            # plan_chunk_queue.learning_metrics son consistentes entre sí
                            # incluso si T2 (status='completed' + shopping list) falla por
                            # crash o DB blip. T2 sólo agrega `learning_persisted_at` para
                            # marcar el cierre observacional.
                            _lm_for_queue = json.dumps(learning_metrics, ensure_ascii=False) if learning_metrics else None

                            # [P0-2] Si el CAS detectó datos obsoletos (degraded mode),
                            # NO escribir en BD. El re-encole ocurre post-transacción.
                            if not _stale_abort:
                                # Sellar nuevo timestamp CAS en memoria. Se escribirá al final.
                                from datetime import datetime as _dt, timezone as _tz
                                plan_data['_plan_modified_at'] = _dt.now(_tz.utc).isoformat()

                                # [P1-D] Persistir snapshot de pantry POST-completion del chunk
                                # actual para que el chunk N+1 pueda detectar drift y avisar al LLM.
                                # `fresh_live_inv` queda atrapado dentro de la closure
                                # `_finalize_live_pantry_validation` (~línea 13474) y nunca está visible
                                # en este scope, así que usamos directamente el snapshot pre-LLM que vive
                                # en `form_data["current_pantry_ingredients"]` — refleja el inventario que
                                # el chunk efectivamente consumió. Si tampoco está, `_extract_pantry_snapshot_from_inventory`
                                # devuelve un dict vacío y el bloque hace skip.
                                try:
                                    _p1d_inv_source = form_data.get("current_pantry_ingredients")
                                    _p1d_snap = _extract_pantry_snapshot_from_inventory(_p1d_inv_source)
                                    if _p1d_snap:
                                        _p1d_per_chunk = plan_data.get('_pantry_snapshot_per_chunk', {})
                                        if not isinstance(_p1d_per_chunk, dict):
                                            _p1d_per_chunk = {}
                                        _p1d_per_chunk[str(week_number)] = _p1d_snap
                                        # Cap: solo mantenemos los últimos 6 chunks (cubre planes
                                        # 30d con 8 chunks dejando margen para los más recientes,
                                        # que son los relevantes para detectar drift incremental).
                                        if len(_p1d_per_chunk) > 6:
                                            _kept = sorted(
                                                _p1d_per_chunk.items(),
                                                key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0,
                                            )[-6:]
                                            _p1d_per_chunk = dict(_kept)
                                        plan_data['_pantry_snapshot_per_chunk'] = _p1d_per_chunk
                                except Exception as _p1d_err:
                                    logger.debug(
                                        f"[P1-D/PERSIST] No se pudo persistir snapshot del chunk "
                                        f"{week_number} para plan {meal_plan_id}: {_p1d_err}"
                                    )

                                # [P1-1] Persistir plan_data (con days mergeados Y
                                # _merged_chunk_ids) DENTRO de la transacción 1 — en el mismo
                                # FOR UPDATE que protege contra concurrencia. Antes los
                                # cambios solo quedaban en `update_result` y se persistían
                                # en una transacción 2 separada al final del worker. Si la
                                # transacción 2 fallaba (DB blip durante shopping list), los
                                # nuevos days y el marker _merged_chunk_ids NO se persistían
                                # — y un retry posterior reentraba al merge sin detectar la
                                # mergeación previa (pre-check P1-1 lo detectaría por
                                # signatures, pero esta capa garantiza el invariante a nivel
                                # transaccional). La transacción 2 sigue persistiendo
                                # plan_data completo (con shopping_list, lessons), idempotente.
                                #
                                # [P0-1 FIX] Atomicidad de learning ↔ chunk_queue.learning_metrics:
                                # T1 ahora persiste TODOS los campos canónicos de aprendizaje
                                # (P0_1_DEFERRED_LEARNING_KEYS: _last_chunk_learning,
                                # _recent_chunk_lessons, _critical_lessons_permanent,
                                # _lifetime_lessons_history, _lifetime_lessons_summary,
                                # _chunk_learning_stub_count) JUNTO con `days` y
                                # `_merged_chunk_ids` en el mismo FOR UPDATE, y además
                                # estampa `learning_metrics` en plan_chunk_queue dentro de
                                # la misma transacción.
                                #
                                # Pre-fix: T1 strippeaba estos campos para "diferirlos" a
                                # T2. Pero si el worker crasheaba entre T1 y T2 (durante
                                # shopping list ~segundos), `_merged_chunk_ids` quedaba
                                # commiteado SIN lección. En el retry, el path
                                # `chunk_already_merged` saltaba el merge y leía
                                # queue.learning_metrics=NULL → backfill caía a STUB y la
                                # lección real se perdía permanentemente.
                                #
                                # Post-fix: si T2 falla, el retry detecta
                                # `_persisted_chunk_id == week_number` (línea ~16156),
                                # salta el backfill y reintenta sólo shopping_list + T2
                                # final. La lección está garantizada por T1.
                                #
                                # T2 sigue overlay-eando learning vía P0_4_T2_INCREMENTAL_KEYS
                                # — idempotente (mismo valor que T1 escribió, ya que entre
                                # T1 y T2 nadie muta learning fields).
                                _t1_persist_view = dict(plan_data)
                                # [P0-5] Defense-in-depth runtime check. Si un dev futuro añade
                                # un campo `_xxx_lesson*` o `_xxx_learning*` a plan_data SIN
                                # registrarlo en P0_1_DEFERRED_LEARNING_KEYS ni en
                                # _P0_5_LESSON_KEY_ALLOWLIST, queremos cazarlo en monitoring
                                # antes de que cause confusión downstream (e.g. T2's overlay
                                # podría dejar de aplicarlo, o el rebuilder no lo reconocería
                                # como learning). Loguea ERROR sin crashear: preferimos
                                # atomicidad ligeramente rota a un chunk que falla. El test
                                # de naming convention en CI debería atrapar estos casos.
                                _p05_unknown = [
                                    k for k in _t1_persist_view
                                    if (
                                        ("lesson" in k.lower() or "_learning" in k.lower())
                                        and k not in P0_1_DEFERRED_LEARNING_KEYS
                                        and k not in _P0_5_LESSON_KEY_ALLOWLIST
                                    )
                                ]
                                if _p05_unknown:
                                    logger.error(
                                        f"[P0-5/UNKNOWN-LESSON-KEY] Chunk {task_id} plan {meal_plan_id}: "
                                        f"plan_data contiene key(s) con patrón lesson/learning NO declarado(s) "
                                        f"en P0_1_DEFERRED_LEARNING_KEYS ni en _P0_5_LESSON_KEY_ALLOWLIST: "
                                        f"{_p05_unknown}. Clasifícalas en cron_tasks.py para que el invariante "
                                        f"P0-1 las cubra (overlay T2, naming convention test, etc)."
                                    )
                                cursor.execute(
                                    "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                                    (json.dumps(_t1_persist_view, ensure_ascii=False), meal_plan_id),
                                )

                                # [P0-1 FIX] Persistir learning_metrics en plan_chunk_queue
                                # DENTRO de la misma transacción T1 que commiteó learning a
                                # plan_data. Esto cierra la ventana donde un crash entre T1
                                # y T2 dejaba plan_data._last_chunk_learning persistido pero
                                # plan_chunk_queue.learning_metrics=NULL — race que el
                                # rebuilder (_rebuild_last_chunk_learning_from_queue) y el
                                # path `chunk_already_merged` no podían recuperar sin caer a
                                # stub. NO seteamos learning_persisted_at aquí: ese campo
                                # marca el cierre observacional del chunk y lo escribe T2
                                # junto con status='completed' (un retry post-T1 puede
                                # ejecutar T2 múltiples veces hasta que la shopping list
                                # converja, y queremos la marca del cierre exitoso).
                                if _lm_for_queue is not None:
                                    cursor.execute(
                                        """
                                        UPDATE plan_chunk_queue
                                        SET learning_metrics = %s::jsonb,
                                            updated_at = NOW()
                                        WHERE id = %s
                                        """,
                                        (_lm_for_queue, task_id),
                                    )

                                update_result = [{
                                    "new_total": new_total,
                                    "new_status": new_status,
                                    "full_plan_data": plan_data
                                }]
            # [P0-1/TOCTOU] Chunk cancelado durante la ventana TOCTOU: no re-encolar, ya está cancelled.
            if _toctou_abort:
                logger.warning(
                    f"[P0-1/TOCTOU] Chunk {task_id} (plan {meal_plan_id}) abortado por cancelación "
                    f"detectada dentro del FOR UPDATE. No se re-encola."
                )
                return

            # [P0-2/CAS] Si se detectaron datos obsoletos (degraded o regeneración completa), re-encolar.
            if _stale_abort:
                logger.warning(
                    f"[P0-2/CAS] Re-encolando chunk {week_number} plan {meal_plan_id}: "
                    f"datos obsoletos detectados (degraded={is_degraded}). El próximo intento leerá datos frescos."
                )
                execute_sql_write(
                    """
                    UPDATE plan_chunk_queue
                    SET status = 'pending',
                        attempts = COALESCE(attempts, 0) + 1,
                        execute_after = NOW() + make_interval(secs => 10),
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (task_id,)
                )
                return

            if not update_result:
                raise Exception(f"Plan {meal_plan_id} no encontrado en UPDATE atomico")

            new_total = update_result[0].get("new_total", 0)
            new_status = update_result[0].get("new_status", "partial")
            full_plan_data = update_result[0].get("full_plan_data", {})

            # [GAP 2 FIX]: Recalcular lista de compras CON RETRY + ROLLBACK del merge si falla
            # Antes: solo logger.warning si fallaba -> plan quedaba con dias nuevos + shopping list vieja.
            # Ahora: retry 3x con backoff. Si falla -> marca chunk con flag para reintentar solo shopping.
            shopping_list_ok = False
            last_shop_error = None
            _SHOP_MAX_RETRIES = 3

            for _shop_attempt in range(1, _SHOP_MAX_RETRIES + 1):
                try:
                    from shopping_calculator import get_shopping_list_delta
                    household = form_data.get("householdSize", 1)
                    
                    # full_plan_data ya tiene el array de 'days' fusionado y actualizado
                    aggr_7 = get_shopping_list_delta(user_id, full_plan_data, is_new_plan=False, structured=True, multiplier=1.0 * household)
                    aggr_15 = get_shopping_list_delta(user_id, full_plan_data, is_new_plan=False, structured=True, multiplier=2.0 * household)
                    aggr_30 = get_shopping_list_delta(user_id, full_plan_data, is_new_plan=False, structured=True, multiplier=4.0 * household)
                    
                    grocery_duration = form_data.get("groceryDuration", "weekly")
                    if grocery_duration == "biweekly":
                        aggr_active = aggr_15
                    elif grocery_duration == "monthly":
                        aggr_active = aggr_30
                    else:
                        aggr_active = aggr_7
                        
                    full_plan_data['aggregated_shopping_list_weekly'] = aggr_7
                    full_plan_data['aggregated_shopping_list_biweekly'] = aggr_15
                    full_plan_data['aggregated_shopping_list_monthly'] = aggr_30
                    full_plan_data['aggregated_shopping_list'] = aggr_active
                    shopping_list_ok = True
                    logger.info(f"[CHUNK/GAP2] Shopping list consolidada recalculada para {new_total} dias (intento {_shop_attempt}).")
                    break  # Exito, salir del retry loop

                except Exception as shop_e:
                    last_shop_error = shop_e
                    if _shop_attempt < _SHOP_MAX_RETRIES:
                        backoff_secs = 2 ** _shop_attempt  # 2s, 4s
                        logger.warning(f"[CHUNK/GAP2] Shopping list intento {_shop_attempt}/{_SHOP_MAX_RETRIES} fallo: {shop_e}. "
                                       f"Reintentando en {backoff_secs}s...")
                        import time as _time
                        _time.sleep(backoff_secs)
                    else:
                        logger.error(f"[CHUNK/GAP2] Shopping list fallo {_SHOP_MAX_RETRIES} veces. Ultimo error: {shop_e}")

            # [P0-4 FIX] Si la shopping list falló, re-encolar chunk para reintentar.
            # Gracias al idempotency marker (_merged_chunk_ids), el reintento saltará el merge
            # y solo recalculará la shopping list, sin duplicar días.
            if not shopping_list_ok:
                logger.error(f" [CHUNK/P0-4] Shopping list falló para plan {meal_plan_id}. "
                             f"Re-encolando chunk (merge idempotente protege contra duplicación).")
                try:
                    # [P1-5] Antes pasábamos `1` hardcoded → cada retry de shopping list usaba
                    # el mismo delay (2min) sin importar si era el 1er o 5to fallo. Si el servicio
                    # de shopping (cálculo de listas) se cae sostenidamente, esto causaba un loop
                    # de retries cada 2 min hasta marcar el chunk como `failed`. Ahora pasamos
                    # `_pickup_attempts + 1` para que el backoff exponencial real se aplique
                    # (2 → 4 → 8 → 16 → 32 min, mismo comportamiento que el path de fallo LLM).
                    shopping_next_attempt = max(1, int(_pickup_attempts) + 1)
                    shopping_retry_minutes = _compute_chunk_retry_delay_minutes(shopping_next_attempt)
                    execute_sql_write("""
                        UPDATE plan_chunk_queue 
                        SET attempts = COALESCE(attempts, 0) + 1,
                            status = CASE
                                WHEN COALESCE(attempts, 0) + 1 >= %s THEN 'failed'
                                ELSE 'pending'
                            END,
                            execute_after = NOW() + make_interval(mins => %s),
                            dead_lettered_at = CASE
                                WHEN COALESCE(attempts, 0) + 1 >= %s THEN NOW()
                                ELSE dead_lettered_at
                            END,
                            dead_letter_reason = CASE
                                WHEN COALESCE(attempts, 0) + 1 >= %s THEN %s
                                ELSE dead_letter_reason
                            END,
                            updated_at = NOW()
                        WHERE id = %s
                    """, (
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        shopping_retry_minutes,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        "shopping_list_retry_exhausted",
                        task_id,
                    ))
                except Exception as status_err:
                    logger.error(f" [CHUNK/P0-4] Error regresando chunk a pending: {status_err}")
                
                return  # Abortamos para no marcarlo como completed ni sobreescribir status

            try:
                from shopping_calculator import _parse_quantity
                _expected_ingredients = 0
                for _rd in new_days:
                    for _rm in (_rd or {}).get('meals', []):
                        for _ri in (_rm.get('ingredients') or []):
                            if _ri and len(str(_ri).strip()) >= 3:
                                try:
                                    _rq, _ru, _rn = _parse_quantity(str(_ri))
                                    if _rn and _rq > 0:
                                        _expected_ingredients += 1
                                except Exception:
                                    pass
                reserved_items = reserve_plan_ingredients(user_id, str(task_id), new_days)
                # [P0-5] Mirror the reconcile fix: skip when no parseable ingredients exist.
                _expected_min = max(1, int(_expected_ingredients * 0.5)) if _expected_ingredients > 0 else 0
                if _expected_ingredients == 0 or reserved_items >= _expected_min:
                    logger.info(
                        f"[P1-2] Reservas de inventario aplicadas para chunk {week_number} "
                        f"plan {meal_plan_id}: {reserved_items}/{_expected_ingredients} ingredientes."
                    )
                    try:
                        execute_sql_write(
                            "UPDATE plan_chunk_queue SET reservation_status = 'ok' WHERE id = %s",
                            (task_id,)
                        )
                    except Exception:
                        pass
                else:
                    logger.warning(
                        f"[P0-5/PARTIAL] Reservas parciales chunk {week_number} plan {meal_plan_id}: "
                        f"{reserved_items}/{_expected_ingredients} (min={_expected_min}). "
                        f"Marcando reservation_status='partial'."
                    )
                    try:
                        execute_sql_write(
                            "UPDATE plan_chunk_queue SET reservation_status = 'partial', updated_at = NOW() WHERE id = %s",
                            (task_id,)
                        )
                    except Exception as _partial_err:
                        logger.error(f"[P0-5] Error marcando partial: {_partial_err}")
                    # Agendar reconciliación
                    _p12_reconciled = _reconcile_chunk_reservations(user_id, str(task_id), new_days)
                    if not _p12_reconciled:
                        # [P1-2] Reconciliación agotada (CAS conflicts persistentes contra
                        # user_inventory). Antes el worker continuaba a T2 y marcaba
                        # status='completed' con reservation_status='partial'. El siguiente
                        # chunk del mismo plan, tras los 5 min del bloqueo de pickup
                        # (process_plan_chunk_queue ~9844-9847), pickea inventario
                        # sobreestimado y comete overbooking.
                        # Ahora: liberamos las reservas parciales aplicadas (si las hay) y
                        # pausamos en pending_user_action:inventory_reconciliation_failed.
                        # El recovery cron `_recover_pantry_paused_chunks` reanuda cuando la
                        # nevera del usuario cambia (señal de que CAS contention puede
                        # haberse aliviado y la reconciliación tiene chance de converger).
                        logger.error(
                            f"[P1-2/RECONCILE-EXHAUSTED] Pausando chunk {week_number} plan {meal_plan_id}: "
                            f"liberando reservas parciales y pasando a pending_user_action."
                        )
                        try:
                            release_chunk_reservations(user_id, str(task_id))
                        except Exception as _p12_rel_err:
                            logger.error(
                                f"[P1-2/RECONCILE-EXHAUSTED] Error liberando reservas parciales "
                                f"para chunk {task_id}: {_p12_rel_err}. _recover_orphan_chunk_"
                                f"reservations cubrirá el cleanup en el próximo ciclo."
                            )
                        _pause_chunk_for_pantry_refresh(
                            task_id,
                            user_id,
                            week_number,
                            fresh_inventory=form_data.get("current_pantry_ingredients", []),
                            reason="inventory_reconciliation_failed",
                        )
                        _dispatch_push_notification(
                            user_id=user_id,
                            title="Tu plan necesita revisión de nevera",
                            body=(
                                "Tuvimos un problema actualizando tu inventario. "
                                "Refresca tu nevera para que retomemos la generación."
                            ),
                            url="/dashboard",
                        )
                        return
            except Exception as reserve_err:
                logger.warning(f"[P0-5] Reservas fallidas para chunk {task_id}: {reserve_err}")
                try:
                    execute_sql_write(
                        "UPDATE plan_chunk_queue SET reservation_status = 'partial', updated_at = NOW() WHERE id = %s",
                        (task_id,)
                    )
                except Exception:
                    pass
                _p12_reconciled_2 = _reconcile_chunk_reservations(user_id, str(task_id), new_days)
                if not _p12_reconciled_2:
                    # [P1-2] Mismo escenario que el branch anterior pero entrando desde el
                    # except (reserve_plan_ingredients lanzó excepción en lugar de devolver
                    # parcial). Con CAS exhausted no podemos garantizar que el chunk no
                    # cause overbooking, así que pausamos.
                    logger.error(
                        f"[P1-2/RECONCILE-EXHAUSTED] Pausando chunk {week_number} plan {meal_plan_id} "
                        f"(reserva inicial lanzó excepción + reconcile agotó retries)."
                    )
                    try:
                        release_chunk_reservations(user_id, str(task_id))
                    except Exception as _p12_rel_err2:
                        logger.error(
                            f"[P1-2/RECONCILE-EXHAUSTED] Error liberando reservas parciales "
                            f"(excepción path) para chunk {task_id}: {_p12_rel_err2}."
                        )
                    _pause_chunk_for_pantry_refresh(
                        task_id,
                        user_id,
                        week_number,
                        fresh_inventory=form_data.get("current_pantry_ingredients", []),
                        reason="inventory_reconciliation_failed",
                    )
                    _dispatch_push_notification(
                        user_id=user_id,
                        title="Tu plan necesita revisión de nevera",
                        body=(
                            "Tuvimos un problema actualizando tu inventario. "
                            "Refresca tu nevera para que retomemos la generación."
                        ),
                        url="/dashboard",
                    )
                    return

            # [GAP C] Determinar tier dominante del chunk (peor tier de los días generados)
            chunk_tier = 'llm'
            try:
                tier_priority = {'emergency': 4, 'edge': 3, 'shuffle': 2, 'llm': 1}
                worst = 0
                for d in new_days:
                    if isinstance(d, dict):
                        t = d.get('quality_tier', 'llm')
                        if tier_priority.get(t, 0) > worst:
                            worst = tier_priority[t]
                            chunk_tier = t
            except Exception:
                pass

            # [GAP C] Recalcular quality_warning del plan en memoria (asumiendo que este chunk es completed)
            try:
                tier_stats = execute_sql_query("""
                    SELECT
                        COUNT(*) FILTER (WHERE status = 'completed') AS completed_total,
                        COUNT(*) FILTER (WHERE status = 'completed' AND quality_tier IN ('shuffle', 'edge', 'emergency')) AS degraded
                    FROM plan_chunk_queue
                    WHERE meal_plan_id = %s AND id != %s
                """, (meal_plan_id, task_id), fetch_one=True)

                completed_total = int((tier_stats.get('completed_total') if tier_stats else 0) or 0) + 1
                degraded = int((tier_stats.get('degraded') if tier_stats else 0) or 0)
                if chunk_tier in ('shuffle', 'edge', 'emergency'):
                    degraded += 1
                    
                if completed_total > 0:
                    degraded_ratio = degraded / completed_total
                    quality_warning = degraded_ratio > 0.30

                    full_plan_data['quality_warning'] = quality_warning
                    full_plan_data['quality_degraded_ratio'] = round(degraded_ratio, 3)
                    
                    if quality_warning:
                        logger.warning(f"[GAP C] Plan {meal_plan_id} marcado con quality_warning=True ({degraded}/{completed_total} chunks degradados, {degraded_ratio:.0%}).")
            except Exception as q_err:
                logger.warning(f"[GAP C] Error calculando quality_warning para plan {meal_plan_id}: {q_err}")

            # [P0-1 FIX] Transacción atómica final: persistir lección, shopping list, días y status='completed' en un solo commit.
            # [P0-4 FIX] Antes T2 hacía `UPDATE plan_data = %s::jsonb` con el dict
            # completo del worker en memoria. Si `/shift-plan` corría entre T1 y T2
            # (ventana inevitable porque el shopping list calc dura segundos), su
            # modificación de days/grocery_start_date/generation_status se perdía.
            # Ahora T2 re-lee plan_data bajo FOR UPDATE + advisory lock 'general',
            # y aplica SOLO los campos incrementales que sólo T2 sabe (learning +
            # shopping_list + quality). Los demás campos del fresh plan_data se
            # preservan, así si /shift-plan modificó days, los cambios sobreviven.
            lm_json = json.dumps(learning_metrics, ensure_ascii=False) if learning_metrics else None
            try:
                from db_core import connection_pool
                if not connection_pool:
                    raise Exception("db connection_pool is not available for atomic merge.")
                from psycopg.rows import dict_row as _p04_dict_row
                with connection_pool.connection() as conn:
                    with conn.transaction():
                        with conn.cursor(row_factory=_p04_dict_row) as cursor:
                            # [P0-4] Advisory lock + FOR UPDATE para serializar contra
                            # /shift-plan y otros workers. Mismo purpose='general' que T1
                            # y /shift-plan: cualquier caller que tome el lock se serializa.
                            from db_plans import acquire_meal_plan_advisory_lock as _p04_acquire_lock
                            _p04_acquire_lock(cursor, meal_plan_id, purpose="general")
                            cursor.execute(
                                "SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE",
                                (meal_plan_id,)
                            )
                            _p04_row = cursor.fetchone()
                            if not _p04_row:
                                raise Exception(
                                    f"Meal plan {meal_plan_id} desapareció antes de T2 "
                                    f"(probable cancelación por save_new_meal_plan_atomic)."
                                )
                            _p04_fresh_plan_data = _p04_row.get('plan_data') or {}
                            if not isinstance(_p04_fresh_plan_data, dict):
                                _p04_fresh_plan_data = {}
                            # Aplicar SOLO los campos incrementales del worker sobre el
                            # fresh plan_data. Days, grocery_start_date, generation_status,
                            # _pantry_snapshot_per_chunk, _plan_modified_at, etc. quedan
                            # intactos del estado en DB (que puede haber sido modificado
                            # por /shift-plan entre T1 y T2 — y queremos preservarlo).
                            for _p04_k in P0_4_T2_INCREMENTAL_KEYS:
                                if _p04_k in full_plan_data:
                                    _p04_fresh_plan_data[_p04_k] = full_plan_data[_p04_k]
                            cursor.execute(
                                "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                                (json.dumps(_p04_fresh_plan_data, ensure_ascii=False), meal_plan_id)
                            )
                            # [P1] Bug de indentación: este execute estaba fuera del
                            # `with conn.cursor()` block (un nivel a la izquierda),
                            # así que se ejecutaba sobre un cursor ya cerrado y
                            # lanzaba "the cursor is closed". Resultado: meal_plans
                            # SÍ se actualizaba pero plan_chunk_queue.status nunca
                            # llegaba a 'completed' — los chunks quedaban en
                            # 'processing' hasta que el cron zombie los recogía.
                            cursor.execute(
                                """
                                UPDATE plan_chunk_queue
                                SET status = 'completed',
                                    quality_tier = %s,
                                    learning_metrics = %s::jsonb,
                                    learning_persisted_at = NOW(),
                                    updated_at = NOW()
                                WHERE id = %s
                                """,
                                (chunk_tier, lm_json, task_id)
                            )
            except Exception as atomic_err:
                logger.error(f"[P0-1] Error en transacción final de completado para plan {meal_plan_id}: {atomic_err}")
                return

            logger.info(f"[CHUNK] Chunk {week_number} completado para plan {meal_plan_id} "
                        f"(+{len(new_days)} dias, total={new_total}, status={new_status}, tier={chunk_tier})")

            # [P0-2] Si el chunk se completó en flexible_mode, notificar al usuario con
            # cooldown 6h. Llamamos POST-commit (después del UPDATE plan_chunk_queue) para
            # garantizar que el chunk realmente quedó persistido antes de avisar; un fallo
            # de la transacción anterior ya habría retornado en `atomic_err` arriba. Best-
            # effort: si el push falla, el chunk sigue siendo válido y el flag por-día queda
            # persistido en plan_data.days[i] para que el frontend muestre el banner.
            try:
                if form_data.get("_pantry_flexible_mode"):
                    _p02_reason_done = (
                        form_data.get("_pantry_degraded_reason")
                        or form_data.get("_fresh_pantry_source")
                        or "unknown"
                    )
                    _maybe_notify_user_pantry_degraded(user_id, _p02_reason_done)
            except Exception as _p02_notify_err:
                logger.debug(
                    f"[P0-2/PANTRY-DEGRADED-NOTIFY] post-completion push falló para chunk "
                    f"{week_number} plan {meal_plan_id}: {_p02_notify_err}"
                )

            # [GAP G] Registrar métrica de observabilidad
            try:
                duration_ms = int((_t.time() - chunk_start_ts) * 1000)
                # Obtener retries actuales del chunk
                _attempts_row = execute_sql_query(
                    "SELECT attempts FROM plan_chunk_queue WHERE id = %s",
                    (task_id,), fetch_one=True
                )
                _retries = int(_attempts_row.get("attempts") or 0) if _attempts_row else 0
                _record_chunk_metric(
                    chunk_id=task_id,
                    meal_plan_id=meal_plan_id,
                    user_id=user_id,
                    week_number=week_number,
                    days_count=days_count,
                    duration_ms=duration_ms,
                    quality_tier=chunk_tier,
                    was_degraded=chunk_tier != 'llm',
                    retries=_retries,
                    lag_seconds=lag_seconds,
                    learning_metrics=learning_metrics,
                    error_message=None,
                    is_rolling_refill=is_rolling_refill,
                )
                _alert_if_degraded_rate_high()
            except Exception as _mt_e:
                logger.warning(f"[GAP G] Error en registro métrica chunk exitoso: {_mt_e}")
                        
            # [GAP D FIX: Persistir señales de aprendizaje inter-chunk]
            # Esto permite que el chunk N+1 se beneficie de la adherencia recalculada por el chunk N,
            # manteniendo el aprendizaje verdaderamente continuo dentro del mismo día.
            try:
                user_res = execute_sql_query("SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True)
                current_health_profile = user_res.get("health_profile", {}) if user_res else {}

                from db_facts import get_consumed_meals_since
                # [P0-1-RECOVERY/WORKER-FIX] datetime/timezone/timedelta son globals del módulo.

                plan_start_date_str = snap.get("form_data", {}).get("_plan_start_date")
                if plan_start_date_str:
                    since_time = plan_start_date_str
                else:
                    since_time = (datetime.now(timezone.utc) - timedelta(days=max(7, days_offset))).isoformat()
                    
                chunk_consumed_records = get_consumed_meals_since(user_id, since_time)
                
                _persist_nightly_learning_signals(
                    user_id,
                    current_health_profile,
                    full_plan_data.get('days', []),
                    chunk_consumed_records
                )
                logger.info(f" [CHUNK] Señales de aprendizaje persistidas tras chunk {week_number} para plan {meal_plan_id}")
            except Exception as persist_err:
                logger.warning(f" [CHUNK] Error persistiendo señales de aprendizaje en chunk {week_number}: {persist_err}")

            # [P0.1/PANTRY-SYNC] Reconciliar user_inventory contra consumed_meals
            # del chunk recién cerrado. Garantiza que el chunk N+1 lea una nevera
            # en sync con lo que el usuario realmente registró durante el chunk N.
            # Idempotente: filas ya descontadas (inventory_synced_at != NULL) se
            # omiten gracias al filtro WHERE de sync_inventory_after_chunk_completion.
            # Best-effort: si falla, se loggea y el chunk no se bloquea — el cron
            # cierra igual y la divergencia se ve en logs para diagnóstico.
            try:
                _p01_plan_start_str = snap.get("form_data", {}).get("_plan_start_date")
                if _p01_plan_start_str:
                    from constants import safe_fromisoformat as _p01_safe_iso
                    from db_inventory import sync_inventory_after_chunk_completion as _p01_sync
                    _p01_start_dt = _p01_safe_iso(_p01_plan_start_str)
                    if _p01_start_dt.tzinfo is None:
                        _p01_start_dt = _p01_start_dt.replace(tzinfo=timezone.utc)
                    _p01_window_start = _p01_start_dt + timedelta(days=int(days_offset or 0))
                    _p01_window_end = _p01_window_start + timedelta(days=int(days_count or 0))
                    _p01_stats = _p01_sync(
                        user_id,
                        _p01_window_start.isoformat(),
                        _p01_window_end.isoformat(),
                    )
                    logger.info(
                        f"[P0.1/PANTRY-SYNC] chunk={week_number} plan={meal_plan_id} "
                        f"reconciled={_p01_stats.get('reconciled_count', 0)} "
                        f"items_deducted={_p01_stats.get('items_deducted', 0)}"
                    )
                else:
                    logger.debug(
                        f"[P0.1/PANTRY-SYNC] Saltado para chunk {week_number} plan "
                        f"{meal_plan_id}: _plan_start_date ausente en snapshot."
                    )
            except Exception as _p01_err:
                logger.warning(
                    f"[P0.1/PANTRY-SYNC] No se pudo sincronizar inventario tras "
                    f"chunk {week_number} plan {meal_plan_id}: {_p01_err}"
                )

        except Exception as e:
            import traceback; tb_str = traceback.format_exc(); logger.error(f" [CHUNK] Error procesando chunk {week_number} para plan {meal_plan_id}: {e}\n{tb_str}")

            # [P0-4] Pantry violation post-merge: la transacción ya hizo ROLLBACK por el raise
            # dentro del bloque atomic merge, así que merged_days NO quedaron escritos a
            # meal_plans. Pausamos el chunk con 'pending_user_action' en lugar de marcarlo
            # 'failed' para retry — un retry contra la misma nevera produciría el mismo
            # resultado y solo agotaría intentos hasta dead_letter. El usuario debe actualizar
            # la nevera (o nosotros via push) y reanudar manualmente.
            if isinstance(e, _PantryViolationPostMerge):
                try:
                    _p04_pause_snap = copy.deepcopy(snap) if isinstance(snap, dict) else {}
                    _p04_pause_snap["_pause_reason"] = "pantry_violation_post_merge"
                    _p04_pause_snap["_p0_4_paused_at"] = datetime.now(timezone.utc).isoformat()
                    _p04_pause_snap["_p0_4_violations"] = e.violations[:10]
                    _p04_pause_snap["_p0_4_pantry_size"] = e.pantry_size
                    execute_sql_write(
                        """
                        UPDATE plan_chunk_queue
                        SET status = 'pending_user_action',
                            pipeline_snapshot = %s::jsonb,
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (json.dumps(_p04_pause_snap, ensure_ascii=False), task_id),
                    )
                    logger.warning(
                        f"[P0-4/POST-MERGE] Chunk {week_number} plan {meal_plan_id} pausado en "
                        f"pending_user_action: {len(e.violations)} día(s) violan pantry."
                    )
                    try:
                        import threading as _p04_threading
                        from utils_push import send_push_notification as _p04_push
                        _p04_threading.Thread(
                            target=_p04_push,
                            kwargs={
                                "user_id": user_id,
                                "title": "Tu plan necesita una revisión",
                                "body": (
                                    "Detectamos ingredientes que ya no están en tu nevera. "
                                    "Actualízala para que generemos los días siguientes."
                                ),
                                "url": "/dashboard",
                            },
                            daemon=True,
                        ).start()
                    except Exception as _push_err:
                        logger.warning(
                            f"[P0-4] No se pudo enviar push notification: {_push_err}"
                        )
                except Exception as _pause_err:
                    logger.error(
                        f"[P0-4] Falló pausar chunk {task_id} tras violación pantry: "
                        f"{_pause_err}. El chunk caerá al flujo failed/retry estándar."
                    )
                else:
                    # Pausa exitosa: registrar métrica y salir sin pasar al flujo de retry.
                    try:
                        duration_ms = int((_t.time() - chunk_start_ts) * 1000)
                        _record_chunk_metric(
                            chunk_id=task_id,
                            meal_plan_id=meal_plan_id,
                            user_id=user_id,
                            week_number=week_number,
                            days_count=days_count,
                            duration_ms=duration_ms,
                            quality_tier="paused",
                            was_degraded=True,
                            retries=int(_pickup_attempts or 0),
                            lag_seconds=lag_seconds,
                            learning_metrics=None,
                            error_message=f"pantry_violation_post_merge:{len(e.violations)}",
                            is_rolling_refill=is_rolling_refill,
                        )
                    except Exception as _mt_err:
                        logger.warning(f"[P0-4] No se pudo registrar métrica de pausa: {_mt_err}")
                    return

            # [P0-2] Persistir learning_metrics a plan_chunk_queue.learning_metrics si llegamos
            # a calcularlo antes del crash (post-pipeline pero pre-commit). Sin esto el
            # éxito al persistir se daba SOLO en el path de éxito (línea ~9575) y un crash
            # entre el cálculo y el commit dejaba la columna NULL — rompiendo
            # _rebuild_last_chunk_learning_from_queue para el chunk siguiente. Marcamos
            # pipeline_failed=True para que el rebuild sepa que la confianza es baja
            # (las violations pueden estar incompletas si el pipeline produjo un plan parcial).
            # Si learning_metrics es None, el preflight ya escrito en plan_chunk_queue se
            # mantiene tal cual (el UPDATE de éxito nunca corrió, así que no hay nada que pisar).
            if isinstance(learning_metrics, dict):
                try:
                    learning_metrics["pipeline_failed"] = True
                    learning_metrics["learning_confidence"] = "low"
                    execute_sql_write(
                        "UPDATE plan_chunk_queue SET learning_metrics = %s::jsonb "
                        "WHERE id = %s",
                        (json.dumps(learning_metrics, ensure_ascii=False), task_id)
                    )
                    logger.info(
                        f"[P0-2/FAILURE-PERSIST] learning_metrics persistido en plan_chunk_queue "
                        f"para plan {meal_plan_id} chunk {week_number} con pipeline_failed=True."
                    )
                except Exception as _lm_persist_err:
                    logger.warning(
                        f"[P0-2/FAILURE-PERSIST] No se pudo persistir learning_metrics tras fallo "
                        f"de pipeline para plan {meal_plan_id} chunk {week_number}: "
                        f"{type(_lm_persist_err).__name__}: {_lm_persist_err}"
                    )

            # [GAP G] Registrar métrica de fallo (truncar mensaje a 1KB para no explotar tabla)
            try:
                duration_ms = int((_t.time() - chunk_start_ts) * 1000)
                _attempts_row = execute_sql_query(
                    "SELECT attempts FROM plan_chunk_queue WHERE id = %s",
                    (task_id,), fetch_one=True
                )
                _retries = int(_attempts_row.get("attempts") or 0) if _attempts_row else 0
                err_msg = str(e)[:1000]
                _record_chunk_metric(
                    chunk_id=task_id,
                    meal_plan_id=meal_plan_id,
                    user_id=user_id,
                    week_number=week_number,
                    days_count=days_count,
                    duration_ms=duration_ms,
                    quality_tier='error',
                    was_degraded=True,
                    retries=_retries,
                    lag_seconds=lag_seconds,
                    learning_metrics=learning_metrics,
                    error_message=err_msg,
                    is_rolling_refill=is_rolling_refill,
                )
            except Exception as _mt_e:
                logger.warning(f"[GAP G] Error en registro métrica chunk fallido: {_mt_e}")
            try:
                # [GAP B] Reintento agresivo (30 min fijo) si el chunk ya está atrasado >24h o fue escalado.
                # Si no, mantener backoff exponencial original (2^n * 2 - 1 min: 2, 8, 32, 128, 512).
                # Esto evita esperar horas en chunks críticos cuando el plan se está consumiendo.
                is_critical = lag_seconds > 86400 or task.get("escalated_at") is not None
                next_attempt = _retries + 1
                retry_delay_minutes = _compute_chunk_retry_delay_minutes(next_attempt, is_critical=is_critical)
                if is_critical:
                    res = execute_sql_write("""
                        UPDATE plan_chunk_queue
                        SET attempts = COALESCE(attempts, 0) + 1,
                            status = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN 'failed' ELSE 'pending' END,
                            execute_after = NOW() + make_interval(mins => %s),
                            dead_lettered_at = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN NOW() ELSE dead_lettered_at END,
                            dead_letter_reason = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN %s ELSE dead_letter_reason END,
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING status
                    """, (
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        retry_delay_minutes,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        str(e)[:240],
                        task_id,
                    ), returning=True)
                    logger.warning(f"[GAP B] Chunk {week_number} crítico (lag={lag_seconds//3600}h): retry en {retry_delay_minutes}min.")
                else:
                    res = execute_sql_write("""
                        UPDATE plan_chunk_queue
                        SET attempts = COALESCE(attempts, 0) + 1,
                            status = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN 'failed' ELSE 'pending' END,
                            execute_after = NOW() + make_interval(mins => %s),
                            dead_lettered_at = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN NOW() ELSE dead_lettered_at END,
                            dead_letter_reason = CASE WHEN COALESCE(attempts, 0) + 1 >= %s THEN %s ELSE dead_letter_reason END,
                            updated_at = NOW()
                        WHERE id = %s
                        RETURNING status
                    """, (
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        retry_delay_minutes,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        CHUNK_MAX_FAILURE_ATTEMPTS,
                        str(e)[:240],
                        task_id,
                    ), returning=True)
                
                # [GAP 4 DE 30 DÃAS FIX / GAP 2 IMPLEMENTATION]: Manejo de Zombies y Fallbacks
                if res and res[0].get('status') == 'failed':
                    snap = task.get("pipeline_snapshot", {})
                    if isinstance(snap, str):
                        snap = json.loads(snap)
                    is_degraded = snap.get("_degraded", False)
                    
                    if is_degraded:
                        logger.error(f" [CHUNK ZOMBIE FATAL] Chunk {week_number} (Degraded Mode) fallo 5 veces. Abortando plan permanentemente.")
                        
                        # 1. Quitar el status 'partial' para liberar el frontend
                        execute_sql_write("""
                            UPDATE meal_plans 
                            SET plan_data = jsonb_set(plan_data, '{generation_status}', '"failed"') 
                            WHERE id = %s
                        """, (meal_plan_id,))
                        
                        # 2. Cancelar los chunks futuros
                        execute_sql_write("""
                            UPDATE plan_chunk_queue 
                            SET status = 'cancelled', updated_at = NOW() 
                            WHERE meal_plan_id = %s AND status = 'pending'
                        """, (meal_plan_id,))
                        
                        # 3. Notificar al usuario del fallo
                        try:
                            import threading
                            from utils_push import send_push_notification
                            threading.Thread(
                                target=send_push_notification,
                                kwargs={
                                    "user_id": user_id,
                                    "title": "âš ï¸ Error extendiendo tu plan",
                                    "body": "Hubo un problema generando tus proximas semanas. Tus dias actuales estan intactos. Intenta generar un nuevo plan pronto.",
                                    "url": "/dashboard"
                                }
                            ).start()
                        except Exception as push_err:
                            logger.warning(f" [CHUNK ZOMBIE] Fallo el push de error: {push_err}")
                    else:
                        logger.warning(f" [CHUNK ZOMBIE] Chunk {week_number} fallo 5 veces en modo IA. Activando Degraded Mode (Smart Shuffle).")
                        
                        # 1. Marcar el plan como 'complete_partial' (valido pero faltan dias)
                        execute_sql_write("""
                            UPDATE meal_plans 
                            SET plan_data = jsonb_set(plan_data, '{generation_status}', '"complete_partial"') 
                            WHERE id = %s
                        """, (meal_plan_id,))
                        
                        # 2. Rescatar este chunk y los futuros en degraded mode
                        execute_sql_write("""
                            UPDATE plan_chunk_queue 
                            SET status = 'pending', 
                                attempts = 0,
                                pipeline_snapshot = jsonb_set(pipeline_snapshot, '{_degraded}', 'true'::jsonb),
                                updated_at = NOW() 
                            WHERE meal_plan_id = %s AND status IN ('pending', 'failed')
                        """, (meal_plan_id,))
                        
                        # [GAP 6] Guardar timestamp del downgrade para evitar flapping
                        # [P0-1-RECOVERY/WORKER-FIX] datetime/timezone son globals del módulo.
                        execute_sql_write(
                            """
                            UPDATE user_profiles 
                            SET health_profile = jsonb_set(
                                COALESCE(health_profile, '{}'::jsonb), 
                                '{_last_downgrade_time}', 
                                %s::jsonb
                            ) WHERE id = %s
                            """,
                            (f'"{datetime.now(timezone.utc).isoformat()}"', str(user_id))
                        )
            except Exception as inner_e:
                logger.error(f" [CHUNK ZOMBIE] Error critico procesando fallback: {inner_e}")
        finally:
            # [P0-4] Detener heartbeat antes de liberar lock para no resucitarlo entre el DELETE
            # y el cierre del thread. El thread es daemon así que no impide el shutdown del proceso.
            try:
                if _heartbeat_stop_event is not None:
                    _heartbeat_stop_event.set()
                if _heartbeat_thread is not None and _heartbeat_thread.is_alive():
                    _heartbeat_thread.join(timeout=2)
            except Exception as _stop_err:
                logger.debug(f"[P0-4/HEARTBEAT] Error parando heartbeat thread: {_stop_err}")
            # Liberar lock en exit paths
            try:
                execute_sql_write("DELETE FROM chunk_user_locks WHERE locked_by_chunk_id = %s", (task_id,))
            except Exception as e:
                logger.error(f" [CHUNK] Error liberando lock para {task_id}: {e}")

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(_chunk_worker, tasks)

def trigger_incremental_learning(user_id: str):
    """Hook asíncrono para [GAP 4] que calcula la adherencia y el Quality Score 
    inmediatamente después de que el usuario loguea una comida, en lugar de 
    esperar 18+ horas hasta el nightly rotation.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        from db_core import execute_sql_query
        from db_profiles import get_user_profile
        from db_facts import get_consumed_meals_since
        from datetime import datetime, timezone, timedelta
        
        # 1. Obtener el plan activo
        plan_res = execute_sql_query(
            "SELECT plan_data FROM meal_plans WHERE user_id = %s AND status = 'active' ORDER BY created_at DESC LIMIT 1",
            (user_id,)
        )
        if not plan_res:
            return
            
        plan_data = plan_res[0].get('plan_data', {})
        days = plan_data.get('days', [])
        if not days:
            return
            
        # 2. Determinar start_date.
        plan_start_date_str = plan_data.get("_plan_start_date")
        if not plan_start_date_str:
            plan_start_date_str = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            
        # 3. Obtener comidas consumidas en el marco del plan
        consumed_records = get_consumed_meals_since(user_id, plan_start_date_str)
        
        # 4. Obtener perfil de salud actual
        profile = get_user_profile(user_id)
        if not profile:
            return
        health_profile = profile.get("health_profile", {})
        
        # 5. Modificar temporalmente el alpha para no decaer bruscamente la historia intradía
        # y delegar a la función principal de aprendizaje nocturno.
        # En una arquitectura estricta pasaríamos alpha como parámetro, pero para 
        # mantener la interfaz limpia aprovechamos la robustez del EMA suavizado actual.
        _persist_nightly_learning_signals(user_id, health_profile, days, consumed_records)
        
        logger.info(f"?? [GAP 4] Aprendizaje incremental persistido con éxito para {user_id}")
    except Exception as e:
        logger.error(f"?? [GAP 4] Error procesando aprendizaje incremental para {user_id}: {e}")


# ─── P0-2: ROLLING REFILL EN BACKGROUND (usuarios que no abren la app) ───────

def _background_shift_plan_for_user(user_id: str, tz_offset: int = 0) -> bool:
    """
    Replica la lógica de api_shift_plan() sin autenticación HTTP.
    Llamada por el cron P0-2 para usuarios inactivos.
    Retorna True si se encolaron chunks nuevos.
    """
    import copy
    import json as _json_bg
    from datetime import datetime, timezone, timedelta
    from constants import PLAN_CHUNK_SIZE, split_with_absorb, safe_fromisoformat
    from db_core import connection_pool
    from psycopg.rows import dict_row

    try:
        # chunk_size_for_next_slot se importa aquí para evitar import circular en módulo top-level
        from routers.plans import chunk_size_for_next_slot

        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cursor:
                    cursor.execute(
                        "SELECT id, plan_data FROM meal_plans "
                        "WHERE user_id = %s ORDER BY created_at DESC LIMIT 1 FOR UPDATE",
                        (user_id,),
                    )
                    plan_record = cursor.fetchone()
                    if not plan_record:
                        return False

                    plan_id = plan_record["id"]

                    # [P0-2/BG-LOCK] Advisory lock 'general' por meal_plan: este path
                    # background es funcionalmente un duplicado de /shift-plan HTTP
                    # (api_shift_plan en routers/plans.py:257) y debe seguir el mismo
                    # invariante: todo escritor de plan_data adquiere purpose='general'
                    # antes de mutar. El FOR UPDATE de arriba serializa el row lock,
                    # pero sin el advisory lock un futuro refactor que reemplace
                    # FOR UPDATE por SELECT plano (e.g. para performance) reabriría
                    # silenciosamente la race contra el worker T2, que SÍ adquiere
                    # purpose='general'. Belt + suspenders.
                    from db_plans import acquire_meal_plan_advisory_lock as _p02_acquire_lock
                    _p02_acquire_lock(cursor, plan_id, purpose="general")

                    plan_data = plan_record.get("plan_data", {})
                    days = plan_data.get("days", [])
                    if not days:
                        return False

                    total_planned_days = max(3, int(plan_data.get("total_days_requested", len(days))))

                    today = datetime.now(timezone.utc)
                    if tz_offset:
                        today -= timedelta(minutes=int(tz_offset))

                    start_date_str = plan_data.get("grocery_start_date")
                    if not start_date_str:
                        return False

                    try:
                        start_dt = safe_fromisoformat(start_date_str)
                        if start_dt.tzinfo is None:
                            start_dt = start_dt.replace(tzinfo=timezone.utc)
                        else:
                            start_dt = start_dt.astimezone(timezone.utc)
                        start_dt = start_dt - timedelta(minutes=int(tz_offset))
                        days_since_creation = (today.date() - start_dt.date()).days
                    except Exception as e:
                        logger.warning(f"[BG-REFILL] Error parseando fecha plan {plan_id}: {e}")
                        return False

                    days_remaining_in_plan = max(0, total_planned_days - days_since_creation)
                    is_expired_renewable = total_planned_days in (7, 15, 30) and days_remaining_in_plan == 0
                    # Solo ignorar planes expirados que no son renovables (7d/15d/30d se renuevan)
                    if days_remaining_in_plan == 0 and not is_expired_renewable:
                        return False

                    # [P0-4 FIX] Antes bloqueaba TODO refill mientras un plan de 7d siguiera vivo,
                    # incluso si su encolado síncrono inicial había fallado dejando huecos.
                    # Ahora solo bloqueamos si NO existe gap huérfano (plan completo en su día actual).
                    # La detección de gap se hace tras los guards de chunks-en-vuelo más abajo.
                    disable_rolling_refill_for_active_7d = (
                        total_planned_days == 7 and days_remaining_in_plan > 0
                    )

                    # Si ya hay chunks en camino, no duplicar
                    is_partial = plan_data.get("generation_status") in ("partial", "generating_next")
                    if is_partial:
                        return False

                    cursor.execute(
                        "SELECT COUNT(*) AS cnt FROM plan_chunk_queue "
                        "WHERE meal_plan_id = %s AND status IN ('pending', 'processing', 'stale')",
                        (plan_id,),
                    )
                    if ((cursor.fetchone() or {}).get("cnt") or 0) > 0:
                        return False

                    # [P0-4 FIX] Llegamos aquí con 0 chunks vivos. Si el plan de 7d aún tiene días
                    # por delante pero len(days) < total_planned_days, hay gap huérfano real:
                    # destrabar el refill para recuperarlo.
                    if (
                        disable_rolling_refill_for_active_7d
                        and len(days) < total_planned_days
                    ):
                        logger.warning(
                            f"[P0-4] Plan de 7 días {plan_id}: gap huérfano detectado "
                            f"(visibles={len(days)}/{total_planned_days}, sin chunks vivos). "
                            f"Habilitando rolling refill de recuperación."
                        )
                        disable_rolling_refill_for_active_7d = False

                    window_size = chunk_size_for_next_slot(
                        max(0, days_since_creation), total_planned_days, PLAN_CHUNK_SIZE
                    )
                    window_needed = min(window_size, days_remaining_in_plan)

                    needs_shift = days_since_creation > 0
                    needs_fill = len(days) < window_needed

                    if not needs_shift and not needs_fill and not is_expired_renewable:
                        return False

                    shifted_data = copy.deepcopy(plan_data)
                    shifted_days = shifted_data.get("days", [])

                    if needs_shift:
                        shift_amount = min(days_since_creation, len(shifted_days))
                        shifted_days = shifted_days[shift_amount:]

                    dias_es = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
                    for i, day_obj in enumerate(shifted_days):
                        target_date = today + timedelta(days=i)
                        day_obj["day_name"] = dias_es[target_date.weekday()]
                        day_obj["day"] = i + 1

                    modified = needs_shift
                    needs_fill_after_shift = (
                        len(shifted_days) < window_needed
                        and days_remaining_in_plan > 0
                        and not disable_rolling_refill_for_active_7d
                    )

                    chunks_enqueued = 0
                    if needs_fill_after_shift:
                        cursor.execute(
                            "SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,)
                        )
                        profile_row = cursor.fetchone()
                        hp = (profile_row or {}).get("health_profile", {}) or {}
                        if not hp:
                            logger.warning(f"[BG-REFILL] Sin health_profile para user {user_id}.")
                        else:
                            previous_meals = [
                                m.get("name", "")
                                for d in shifted_days
                                for m in d.get("meals", [])
                                if m.get("name")
                            ]
                            days_offset = len(shifted_days)
                            total_missing = days_remaining_in_plan - days_offset
                            catchup_chunks = (
                                split_with_absorb(total_missing, PLAN_CHUNK_SIZE)
                                if total_missing > 0
                                else []
                            )

                            # [P0-1 FIX] Propagar _plan_start_date vigente (post-shift) al snapshot
                            # para que _check_chunk_learning_ready pueda calcular ventanas de adherencia.
                            # Sin esto, el gate retornaba ready=True con reason=missing_plan_start_date
                            # y el aprendizaje continuo se desactivaba para todo rolling refill.
                            new_plan_start_iso = (
                                (start_dt + timedelta(days=days_since_creation)).isoformat()
                                if needs_shift else start_date_str
                            )

                            cursor.execute(
                                "SELECT COALESCE(MAX(week_number), 1) AS max_week "
                                "FROM plan_chunk_queue WHERE meal_plan_id = %s AND status <> 'cancelled'",
                                (plan_id,),
                            )
                            next_week = int((cursor.fetchone() or {}).get("max_week") or 1) + 1
                            current_offset = days_offset
                            _hist = plan_data.get("_lifetime_lessons_history", [])
                            _summ = plan_data.get("_lifetime_lessons_summary", {})
                            inherited = {"history": _hist, "summary": _summ} if (_hist or _summ) else None
                            is_first_catchup = True

                            for chunk_count in catchup_chunks:
                                cursor.execute(
                                    "SELECT id FROM plan_chunk_queue "
                                    "WHERE meal_plan_id = %s AND week_number = %s "
                                    "AND status IN ('pending','processing','stale','failed') LIMIT 1",
                                    (plan_id, next_week),
                                )
                                if cursor.fetchone():
                                    next_week += 1
                                    current_offset += chunk_count
                                    continue

                                snapshot = {
                                    "form_data": {
                                        **hp,
                                        "user_id": user_id,
                                        "totalDays": chunk_count,
                                        "_plan_start_date": new_plan_start_iso,
                                        # [P1-7] Marker para que el temporal gate sepa que
                                        # este chunk continúa un plan previo. anchor_iso es
                                        # la fecha del primer día del refill; el chunk previo
                                        # del plan original terminó el día anterior.
                                        "_is_continuation": True,
                                        "_continuation_anchor_iso": new_plan_start_iso,
                                    },
                                    "taste_profile": "",
                                    "memory_context": "",
                                    "previous_meals": previous_meals,
                                    "totalDays": chunk_count,
                                    "_is_rolling_refill": True,
                                    "_triggered_by": "background_cron_p0_2",
                                }
                                if is_first_catchup and inherited:
                                    snapshot["_inherited_lifetime_lessons"] = inherited
                                    is_first_catchup = False
                                _enqueue_plan_chunk(
                                    user_id,
                                    plan_id,
                                    next_week,
                                    current_offset,
                                    chunk_count,
                                    snapshot,
                                    chunk_kind="rolling_refill",
                                )
                                chunks_enqueued += 1
                                logger.info(
                                    f"🤖 [BG-REFILL] Chunk encolado user={user_id} "
                                    f"week={next_week} offset={current_offset} count={chunk_count}"
                                )
                                next_week += 1
                                current_offset += chunk_count

                            if chunks_enqueued > 0:
                                shifted_data["generation_status"] = "generating_next"
                                modified = True

                    # [P0-1] Plan renovable expirado (7d/15d/30d): auto-renovar con señales de aprendizaje frescas
                    elif is_expired_renewable:
                        cursor.execute(
                            "SELECT health_profile FROM user_profiles WHERE id = %s", (user_id,)
                        )
                        profile_row = cursor.fetchone()
                        hp = (profile_row or {}).get("health_profile", {}) or {}
                        if not hp:
                            logger.warning(f"[BG-REFILL P0-1] Sin health_profile para user {user_id}.")
                        else:
                            # [P0-1 FIX] Verificar nevera antes de encolar a ciegas
                            # (replica la validación de routers/plans.py:287-298)
                            from db_inventory import get_user_inventory_net as _bg_get_inv
                            try:
                                live_inv = _bg_get_inv(user_id)
                            except Exception as _inv_err:
                                logger.warning(f"[BG-REFILL P0-1] Error obteniendo inventario para {user_id}: {_inv_err}")
                                live_inv = None

                            if live_inv is None or len(live_inv) < CHUNK_MIN_FRESH_PANTRY_ITEMS:
                                # Nevera vacía / inaccesible: pausar renovación
                                shifted_data["generation_status"] = "expired_pending_pantry"
                                shifted_data["pending_user_action"] = {
                                    "type": "pantry_required",
                                    "message": "Actualiza tu nevera para renovar tu plan"
                                }
                                # [P0-6] Cambio estructural (generation_status +
                                # pending_user_action) — sellar `_plan_modified_at`
                                # para que el CAS de cualquier chunk concurrente
                                # detecte el cambio. El path success más abajo
                                # (~16828) ya sella; este path de pausa-por-pantry
                                # estaba quedando ciego al CAS por omisión.
                                shifted_data["_plan_modified_at"] = datetime.now(timezone.utc).isoformat()
                                cursor.execute(
                                    "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                                    (_json_bg.dumps(shifted_data, ensure_ascii=False), plan_id)
                                )
                                logger.warning(
                                    f"[BG-REFILL P0-2] Plan {plan_id} ({total_planned_days}d) "
                                    f"pendiente de nevera para user {user_id}."
                                )
                                try:
                                    import threading as _thr_bg
                                    from utils_push import send_push_notification
                                    _thr_bg.Thread(
                                        target=send_push_notification,
                                        kwargs={
                                            "user_id": user_id,
                                            "title": "Renovación pausada",
                                            "body": "Actualiza tu nevera para renovar tu plan.",
                                            "url": "/dashboard"
                                        },
                                        daemon=True
                                    ).start()
                                except Exception as _push_err:
                                    logger.error(f"Error push renewal: {_push_err}")
                                return False
                            else:
                                cursor.execute(
                                    "SELECT COALESCE(MAX(week_number), 0) AS max_week FROM plan_chunk_queue "
                                    "WHERE meal_plan_id = %s AND status <> 'cancelled'",
                                    (plan_id,),
                                )
                                next_week = int((cursor.fetchone() or {}).get("max_week") or 0) + 1
                                previous_meals = [
                                    m.get("name", "")
                                    for d in shifted_days
                                    for m in d.get("meals", [])
                                    if m.get("name")
                                ]
                                current_offset = 0
                                # [P0-1 FIX] El plan renovable se renueva con start = today.
                                renewal_plan_start_iso = today.isoformat()
                                is_first_chunk = True
                                for chunk_count in split_with_absorb(total_planned_days, PLAN_CHUNK_SIZE):
                                    snapshot = {
                                        "form_data": {
                                            **hp,
                                            "user_id": user_id,
                                            "totalDays": chunk_count,
                                            "_plan_start_date": renewal_plan_start_iso,
                                            "current_pantry_ingredients": live_inv or [],
                                            "_pantry_captured_at": today.isoformat(),
                                            # [P1-7] Continuation marker: la renovación
                                            # semanal hereda contexto del plan previo. anchor_iso
                                            # = fecha del primer día del nuevo periodo; el plan
                                            # previo terminó el día anterior.
                                            "_is_continuation": True,
                                            "_continuation_anchor_iso": renewal_plan_start_iso,
                                        },
                                        "taste_profile": "",
                                        "memory_context": "",
                                        "previous_meals": previous_meals,
                                        "totalDays": chunk_count,
                                        "_is_rolling_refill": True,
                                        "_is_weekly_renewal": True,
                                        "_triggered_by": "background_cron_p0_1",
                                    }
                                    # Propagar lecciones históricas al primer chunk de renovación
                                    if is_first_chunk:
                                        _history = plan_data.get("_lifetime_lessons_history")
                                        _summary = plan_data.get("_lifetime_lessons_summary")
                                        if _history or _summary:
                                            snapshot["_inherited_lifetime_lessons"] = {
                                                "history": _history or [],
                                                "summary": _summary or {}
                                            }
                                        is_first_chunk = False
                                    _enqueue_plan_chunk(
                                        user_id, plan_id, next_week, current_offset,
                                        chunk_count, snapshot, chunk_kind="rolling_refill",
                                    )
                                    chunks_enqueued += 1
                                    logger.info(
                                        f"[BG-REFILL P0-1] Chunk renovacion user={user_id} "
                                        f"week={next_week} offset={current_offset} count={chunk_count}"
                                    )
                                    next_week += 1
                                    current_offset += chunk_count
                                shifted_days = []
                                shifted_data["grocery_start_date"] = today.isoformat()
                                shifted_data["generation_status"] = "generating_next"
                                modified = True
                                logger.info(
                                    f"[BG-REFILL P0-1] Plan {total_planned_days}d {plan_id} "
                                    f"renovado para user {user_id} ({chunks_enqueued} chunks)."
                                )

                    if modified:
                        shifted_data["days"] = shifted_days
                        new_plan_start_iso = None
                        if needs_shift and not is_expired_renewable:
                            new_start = start_dt + timedelta(days=days_since_creation)
                            new_plan_start_iso = new_start.isoformat()
                            shifted_data["grocery_start_date"] = new_plan_start_iso
                        elif is_expired_renewable:
                            new_plan_start_iso = today.isoformat()
                        shifted_data["_plan_modified_at"] = datetime.now(timezone.utc).isoformat()
                        cursor.execute(
                            "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                            (_json_bg.dumps(shifted_data, ensure_ascii=False), plan_id),
                        )

                        # [P0-5 FIX] Sincronizar _plan_start_date en los snapshots de chunks
                        # vivos (ver justificación en routers/plans.py).
                        if new_plan_start_iso:
                            cursor.execute(
                                """
                                UPDATE plan_chunk_queue
                                SET pipeline_snapshot = jsonb_set(
                                        pipeline_snapshot,
                                        '{form_data,_plan_start_date}',
                                        %s::jsonb,
                                        true
                                    ),
                                    updated_at = NOW()
                                WHERE meal_plan_id = %s
                                  AND status IN ('pending', 'processing', 'stale', 'failed', 'pending_user_action')
                                """,
                                (_json_bg.dumps(new_plan_start_iso), plan_id),
                            )

                    return chunks_enqueued > 0

    except Exception as e:
        logger.error(f"❌ [BG-REFILL] Error procesando user {user_id}: {e}")
        return False


def trigger_background_rolling_refill() -> None:
    """
    [P0-2] Cron diario: dispara shift-plan para usuarios con planes activos que llevan
    >= 3 días sin abrir la app. Garantiza que los chunks de rolling refill se generen
    a tiempo aunque el usuario no haya entrado, preservando la promesa temporal del sistema.
    """
    try:
        from db_core import connection_pool
        if not connection_pool:
            return

        # Usuarios cuyo plan más reciente no ha sido actuado sobre en los últimos 3 días
        # (sin sesión activa) pero aún tienen días por vivir en su plan.
        query = """
            SELECT DISTINCT ON (mp.user_id) mp.user_id
            FROM meal_plans mp
            WHERE mp.user_id NOT IN (
                SELECT DISTINCT user_id
                FROM agent_sessions
                WHERE user_id IS NOT NULL
                  AND user_id::text != 'guest'
                  AND created_at >= NOW() - INTERVAL '3 days'
            )
            AND mp.id = (
                SELECT id FROM meal_plans mp2
                WHERE mp2.user_id = mp.user_id
                ORDER BY mp2.created_at DESC
                LIMIT 1
            )
            ORDER BY mp.user_id, mp.created_at DESC
        """
        inactive_users = execute_sql_query(query, fetch_all=True)

        if not inactive_users:
            logger.info("✅ [BG-REFILL] No hay usuarios inactivos con planes que refrescar.")
            return

        logger.info(f"⏰ [BG-REFILL] Procesando {len(inactive_users)} usuario(s) inactivos (P0-2)...")
        refilled = 0
        for row in inactive_users:
            uid = str(row.get("user_id") or "")
            if not uid:
                continue
            if _background_shift_plan_for_user(uid):
                refilled += 1

        logger.info(
            f"✅ [BG-REFILL] {refilled}/{len(inactive_users)} usuario(s) con chunks encolados."
        )
    except Exception as e:
        logger.error(f"❌ [BG-REFILL] Error en trigger_background_rolling_refill: {e}")
