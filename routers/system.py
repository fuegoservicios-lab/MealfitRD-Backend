from fastapi import APIRouter, HTTPException, Depends
import logging
from db_core import execute_sql_query
import json

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/system",
    tags=["system"]
)

# Puedes añadir una dependencia de autenticación de admin aquí después si es necesario
# async def verify_admin(token: str = Header(...)): ...

@router.get("/health")
def get_system_health():
    """
    Meta-Dashboard (Gap 5): Retorna el "Health Status" de la Inteligencia Autónoma.
    Calcula al vuelo las métricas de:
    - Quality Score global
    - Efectividad de Nudges
    - Distribución de abandono causal (Gap 2)
    - Distribución emocional (Gap 4)
    """
    metrics = {
        "nudge_effectiveness": {},
        "abandonment_reasons": {},
        "emotional_distribution": {},
        "average_quality_score": 0.0,
        "users_evaluated": 0
    }
    
    try:
        # 1. Nudge Response Rate Global
        nudge_stats = execute_sql_query(
            "SELECT COUNT(*) as total, SUM(CASE WHEN responded THEN 1 ELSE 0 END) as responded_count FROM nudge_outcomes",
            fetch_one=True
        )
        if nudge_stats and nudge_stats.get("total", 0) > 0:
            total = nudge_stats["total"]
            responded = nudge_stats["responded_count"] or 0
            metrics["nudge_effectiveness"] = {
                "total_sent": total,
                "total_responded": responded,
                "response_rate_percent": round((responded / total) * 100, 2)
            }
            
        # 2. Abandonment Reasons (Gap 2)
        reasons = execute_sql_query(
            "SELECT reason, COUNT(*) as count FROM abandoned_meal_reasons GROUP BY reason ORDER BY count DESC",
            fetch_all=True
        )
        if reasons:
            metrics["abandonment_reasons"] = {row['reason']: row['count'] for row in reasons}
            
        # 3. Emotional State Distribution (Gap 4)
        emotions = execute_sql_query(
            "SELECT response_sentiment, COUNT(*) as count FROM nudge_outcomes WHERE response_sentiment IS NOT NULL GROUP BY response_sentiment ORDER BY count DESC",
            fetch_all=True
        )
        if emotions:
            metrics["emotional_distribution"] = {row['response_sentiment']: row['count'] for row in emotions}
            
        # 4. Average Quality Score de todos los perfiles
        profiles = execute_sql_query(
            "SELECT health_profile->>'quality_history' as qh FROM user_profiles WHERE health_profile->>'quality_history' IS NOT NULL",
            fetch_all=True
        )
        if profiles:
            total_score = 0.0
            count = 0
            for p in profiles:
                try:
                    qh_str = p.get('qh')
                    if qh_str:
                        history = json.loads(qh_str)
                        if history and isinstance(history, list) and len(history) > 0:
                            # Tomamos el score más reciente
                            total_score += float(history[-1])
                            count += 1
                except Exception:
                    continue
            
            if count > 0:
                metrics["average_quality_score"] = round(total_score / count, 2)
                metrics["users_evaluated"] = count

        return {
            "success": True,
            "status": "healthy",
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Error calculando system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunk-queue-health")
def get_chunk_queue_health():
    """[P1-5] Visibilidad del worker de chunks: backlog, antigüedad, último run y
    tasa de fallos en las últimas 24h.

    Sin este endpoint la única forma de saber si el worker está atrasado era leer
    los logs (donde [P1-5/WORKER-OVERLAP] se emite). Aquí lo exponemos como API
    para dashboards de operación o alertas externas.

    Devuelve:
      - pending_due: chunks listos para ejecutar AHORA (status pending/stale, execute_after <= NOW).
      - oldest_pending_age_seconds: antigüedad del chunk pending más viejo.
      - last_worker_run: snapshot in-memory de la última corrida (duración, backlog, overlap_warning).
      - failures_24h: chunks con status='failed' en las últimas 24h (counter).
      - dead_lettered_24h: chunks con dead_lettered_at en las últimas 24h.
      - degraded_rate_24h: % de chunks que terminaron en quality_tier != 'llm' en plan_chunk_metrics.
    """
    try:
        # 1. Backlog actual
        pending_row = execute_sql_query(
            """
            SELECT
                COUNT(*) FILTER (WHERE status IN ('pending', 'stale') AND execute_after <= NOW())::int AS pending_due,
                COUNT(*) FILTER (WHERE status IN ('pending', 'stale'))::int AS pending_total,
                COALESCE(EXTRACT(EPOCH FROM (NOW() - MIN(execute_after) FILTER (
                    WHERE status IN ('pending', 'stale') AND execute_after <= NOW()
                )))::int, 0) AS oldest_pending_age_seconds
            FROM plan_chunk_queue
            """,
            fetch_one=True,
        ) or {}

        # 2. Tasa de fallos / dead-letter en 24h
        fail_row = execute_sql_query(
            """
            SELECT
                COUNT(*) FILTER (WHERE status = 'failed' AND updated_at > NOW() - INTERVAL '24 hours')::int AS failures_24h,
                COUNT(*) FILTER (WHERE dead_lettered_at > NOW() - INTERVAL '24 hours')::int AS dead_lettered_24h,
                COUNT(*) FILTER (WHERE updated_at > NOW() - INTERVAL '24 hours')::int AS chunks_touched_24h
            FROM plan_chunk_queue
            """,
            fetch_one=True,
        ) or {}

        # 3. Degraded rate desde plan_chunk_metrics
        degraded_row = execute_sql_query(
            """
            SELECT
                COUNT(*)::int AS total,
                COUNT(*) FILTER (WHERE was_degraded IS TRUE)::int AS degraded
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - INTERVAL '24 hours'
            """,
            fetch_one=True,
        ) or {}
        degraded_total = int(degraded_row.get("total") or 0)
        degraded_count = int(degraded_row.get("degraded") or 0)
        degraded_rate = round((degraded_count / degraded_total) * 100, 2) if degraded_total else None

        # 4. Snapshot in-memory del último run (poblado por _emit_worker_run_metric)
        try:
            from cron_tasks import _LAST_WORKER_RUN as _lwr
            last_worker_run = dict(_lwr)  # copia defensiva
        except Exception:
            last_worker_run = None

        return {
            "success": True,
            "pending_due": int(pending_row.get("pending_due") or 0),
            "pending_total": int(pending_row.get("pending_total") or 0),
            "oldest_pending_age_seconds": int(pending_row.get("oldest_pending_age_seconds") or 0),
            "failures_24h": int(fail_row.get("failures_24h") or 0),
            "dead_lettered_24h": int(fail_row.get("dead_lettered_24h") or 0),
            "chunks_touched_24h": int(fail_row.get("chunks_touched_24h") or 0),
            "degraded_total_24h": degraded_total,
            "degraded_count_24h": degraded_count,
            "degraded_rate_24h_percent": degraded_rate,
            "last_worker_run": last_worker_run,
        }

    except Exception as e:
        logger.error(f"[P1-5] Error en /chunk-queue-health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pantry-tolerance-health")
def get_pantry_tolerance_health():
    """[P1-4] Visibilidad de fallbacks de `_get_pantry_tolerance_for_user`.

    Cada vez que el helper cae al default por una razón inesperada (DB blip,
    valor no-numérico, override fuera de rango) registra un evento en un ring
    buffer in-memory de cron_tasks. Este endpoint agrega la ventana rolling
    de 24h por `source` para que dashboards detecten degradación silenciosa
    (e.g., users con override 1.30 cayendo a 1.05 sin saberlo).

    Devuelve:
      - total_24h: número total de fallbacks en la ventana.
      - by_source: dict {source: count}.
      - unique_users_24h: usuarios distintos afectados.
      - oldest_event_age_seconds: antigüedad del evento más viejo en buffer.
      - buffer_capacity: tamaño máximo del ring (informativo).
    """
    import time as _p14_time
    try:
        from cron_tasks import (
            _PANTRY_TOLERANCE_FALLBACKS as _events,
            _PANTRY_TOLERANCE_FALLBACK_WINDOW_SECONDS as _window,
            _PANTRY_TOLERANCE_FALLBACK_MAX_RECORDS as _cap,
        )
    except Exception as imp_err:
        logger.warning(f"[P1-4] No se pudo importar buffer de fallbacks: {imp_err}")
        return {
            "success": True,
            "total_24h": 0,
            "by_source": {},
            "unique_users_24h": 0,
            "oldest_event_age_seconds": None,
            "buffer_capacity": 0,
        }

    # Copia defensiva (el buffer puede mutarse mientras leemos).
    snapshot_events = list(_events)
    now = _p14_time.time()
    cutoff = now - _window
    in_window = [e for e in snapshot_events if e[0] >= cutoff]

    by_source: dict = {}
    users: set = set()
    oldest_ts = None
    for ts, source, user_id in in_window:
        by_source[source] = by_source.get(source, 0) + 1
        users.add(user_id)
        if oldest_ts is None or ts < oldest_ts:
            oldest_ts = ts

    return {
        "success": True,
        "total_24h": len(in_window),
        "by_source": by_source,
        "unique_users_24h": len(users),
        "oldest_event_age_seconds": int(now - oldest_ts) if oldest_ts else None,
        "buffer_capacity": int(_cap),
    }


@router.get("/tz-fallback-health")
def get_tz_fallback_health():
    """[P2-1] Visibilidad agregada de fallbacks TZ en `_enqueue_plan_chunk`.

    Antes cada chunk que caía al fallback TZ emitía una línea WARNING; un plan
    de 30 días con `_plan_start_date` corrupto generaba ~8 líneas idénticas
    modulo `week_number`. Ahora `_record_tz_fallback` dedupea por
    `(plan_id, reason)` con TTL 1h: la primera ocurrencia emite WARNING y las
    repeticiones se acumulan en un ring buffer rolling de 24h. Este endpoint
    expone el agregado para dashboards.

    Devuelve:
      - total_24h: número total de fallbacks en la ventana.
      - by_reason: dict {reason: count}.
      - top_plans_24h: lista (limit 10) de planes con más fallbacks
        (señal de plan corrupto persistente).
      - unique_plans_24h, unique_users_24h.
      - oldest_event_age_seconds: antigüedad del evento más viejo.
      - active_dedupe_keys: número de (plan, reason) con WARNING activo.
    """
    import time as _p21_time
    try:
        from cron_tasks import (
            _TZ_FALLBACK_EVENTS as _events,
            _TZ_FALLBACK_WINDOW_SECONDS as _window,
            _TZ_FALLBACK_MAX_RECORDS as _cap,
            _TZ_FALLBACK_DEDUPE_KEYS as _dedupe,
        )
    except Exception as imp_err:
        logger.warning(f"[P2-1] No se pudo importar buffer TZ fallback: {imp_err}")
        return {
            "success": True,
            "total_24h": 0,
            "by_reason": {},
            "top_plans_24h": [],
            "unique_plans_24h": 0,
            "unique_users_24h": 0,
            "oldest_event_age_seconds": None,
            "active_dedupe_keys": 0,
            "buffer_capacity": 0,
        }

    snapshot_events = list(_events)
    now = _p21_time.time()
    cutoff = now - _window
    in_window = [e for e in snapshot_events if e[0] >= cutoff]

    by_reason: dict = {}
    by_plan: dict = {}
    plans: set = set()
    users: set = set()
    oldest_ts = None
    for ts, user_id, plan_id, _wn, reason in in_window:
        by_reason[reason] = by_reason.get(reason, 0) + 1
        by_plan[plan_id] = by_plan.get(plan_id, 0) + 1
        plans.add(plan_id)
        users.add(user_id)
        if oldest_ts is None or ts < oldest_ts:
            oldest_ts = ts

    top_plans = sorted(by_plan.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "success": True,
        "total_24h": len(in_window),
        "by_reason": by_reason,
        "top_plans_24h": [{"plan_id": pid, "count": c} for pid, c in top_plans],
        "unique_plans_24h": len(plans),
        "unique_users_24h": len(users),
        "oldest_event_age_seconds": int(now - oldest_ts) if oldest_ts else None,
        "active_dedupe_keys": len(_dedupe),
        "buffer_capacity": int(_cap),
    }
