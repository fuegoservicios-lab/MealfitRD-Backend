from fastapi import APIRouter, HTTPException, Request
from error_utils import safe_error_detail
import hashlib
import logging
import os
from db_core import execute_sql_query
import json
from typing import Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/system",
    tags=["system"]
)


# ---------------------------------------------------------------------------
# [P2-HEALTH-UID-STRIP · 2026-05-12] Helpers para hashear UUIDs antes de
# devolverlos en endpoints health públicos.
#
# Modo de fallo pre-fix (audit production-readiness 2026-05-12):
#   `/atomic-pool-health` retornaba `last_user_id` (UUID literal) y
#   `/tz-fallback-health` retornaba `top_plans_24h: [{"plan_id": <UUID>,
#   "count": N}, ...]`. Ambos endpoints son públicos (sin auth, sin gate)
#   "útiles para Grafana/k8s probes" según docstring legacy — pero exponen
#   UUIDs enumerables que un atacante o competidor puede recolectar
#   polleando regularmente. UUIDs de usuario/plan no son secretos pero
#   son auth-spray fodder + correlation across sessions.
#
# Mismo gap clase-cerrado por P1-SYSTEM-HEALTH-ADMIN-GATE en
# `/api/system/health`, pero los 5 endpoints siblings quedaron sin gate.
# La fix mínima preserva la utilidad operacional (correlation visual en
# dashboards: "el mismo usuario/plan repite el fallback") sin enumeración:
# SHA-256(uuid)[:12] da 48-bit space — suficiente para detectar patterns
# repetidos, no reversible al UUID original.
#
# Tooltip-anchor: P2-HEALTH-UID-STRIP.
# ---------------------------------------------------------------------------

def _hash_uuid_for_public(value: Optional[str]) -> Optional[str]:
    """Devuelve SHA-256(value)[:12] o None si el input es falsy.

    Determinístico: el mismo UUID siempre hashea al mismo digest, así que
    dashboards pueden contar "user X hit fallback Y veces" sin conocer X.
    """
    if not value:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]

# P1-A4: reuso del helper Bearer-token de routers/plans.py para mantener una
# sola implementación de auth admin (CRON_SECRET). Importar desde el sibling
# router no genera ciclo: plans.py no importa system.py.
from routers.plans import _verify_admin_token, _check_admin_rate_limit  # noqa: E402

@router.get("/health")
def get_system_health(request: Request):
    """
    [P1-SYSTEM-HEALTH-ADMIN-GATE · 2026-05-12] Meta-Dashboard de la
    Inteligencia Autónoma — agregados sobre toda la flota:
      - Nudge response rate global.
      - Distribución de razones de abandono (negocio).
      - Distribución emocional de respuestas (sentiment).
      - Quality score promedio.

    Pre-fix este endpoint era público. No expone PII, pero SÍ expone
    business-intel agregada (engagement, churn signals, sentiment, calidad
    percibida) — un competidor midiendo `/api/system/health` cada minuto
    obtiene un dashboard gratuito de la salud comercial del producto.

    Defensa: gate Bearer `CRON_SECRET` igual que el resto de endpoints
    operacionales (`/admin/chunks/stuck`, `/admin/deploy-lag/check`, etc.).
    Si necesitas un probe público de liveness, usar `/health` (raíz) o
    `/ready` en `app.py` — esos retornan solo `{status: ok}` sin agregados.

    Tooltip-anchor: P1-SYSTEM-HEALTH-ADMIN-GATE.
    Test parser-based: `tests/test_p1_system_health_admin_gate.py`.
    """
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
    metrics = {
        "nudge_effectiveness": {},
        "abandonment_reasons": {},
        "emotional_distribution": {},
        "average_quality_score": 0.0,
        "users_evaluated": 0
    }

    # [P3-FULL-TABLE-SCAN-HEALTH · 2026-05-12] Pre-fix las 4 queries
    # debajo escaneaban tablas completas sin LIMIT — admisible mientras
    # la app era pequeña, pero a escala (100k+ user_profiles, millones
    # de nudge_outcomes) cada hit al endpoint consumía DB CPU significativa.
    # Aunque ahora es admin-only (P1-SYSTEM-HEALTH-ADMIN-GATE), un
    # dashboard polleando cada 30s puede saturar el pool. Knobs:
    #   MEALFIT_SYSTEM_HEALTH_NUDGE_DAYS (default 30): ventana lookback
    #     para nudge_outcomes (no afecta el sentido del agregado — solo
    #     limita ruido histórico estancado).
    #   MEALFIT_SYSTEM_HEALTH_PROFILE_LIMIT (default 10000): tope de
    #     perfiles muestreados para `average_quality_score`. Order BY
    #     `updated_at DESC` para que el muestreo refleje actividad
    #     reciente (perfiles dormidos no distorsionan la métrica).
    try:
        nudge_lookback_days = int(os.environ.get("MEALFIT_SYSTEM_HEALTH_NUDGE_DAYS", "30") or 30)
        if nudge_lookback_days < 1:
            nudge_lookback_days = 30
        if nudge_lookback_days > 365:
            nudge_lookback_days = 365
        profile_sample_limit = int(os.environ.get("MEALFIT_SYSTEM_HEALTH_PROFILE_LIMIT", "10000") or 10000)
        if profile_sample_limit < 100:
            profile_sample_limit = 100
        if profile_sample_limit > 100000:
            profile_sample_limit = 100000

        # 1. Nudge Response Rate Global (ventana rolling, no all-time)
        nudge_stats = execute_sql_query(
            f"""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN responded THEN 1 ELSE 0 END) as responded_count
            FROM nudge_outcomes
            WHERE sent_at > NOW() - INTERVAL '{nudge_lookback_days} days'
            """,
            fetch_one=True
        )
        if nudge_stats and nudge_stats.get("total", 0) > 0:
            total = nudge_stats["total"]
            responded = nudge_stats["responded_count"] or 0
            metrics["nudge_effectiveness"] = {
                "total_sent": total,
                "total_responded": responded,
                "response_rate_percent": round((responded / total) * 100, 2),
                "window_days": nudge_lookback_days,
            }

        # 2. Abandonment Reasons (Gap 2) — mismo lookback rolling.
        reasons = execute_sql_query(
            f"""
            SELECT reason, COUNT(*) as count
            FROM abandoned_meal_reasons
            WHERE created_at > NOW() - INTERVAL '{nudge_lookback_days} days'
            GROUP BY reason ORDER BY count DESC LIMIT 50
            """,
            fetch_all=True
        )
        if reasons:
            metrics["abandonment_reasons"] = {row['reason']: row['count'] for row in reasons}

        # 3. Emotional State Distribution (Gap 4)
        emotions = execute_sql_query(
            f"""
            SELECT response_sentiment, COUNT(*) as count
            FROM nudge_outcomes
            WHERE response_sentiment IS NOT NULL
              AND sent_at > NOW() - INTERVAL '{nudge_lookback_days} days'
            GROUP BY response_sentiment ORDER BY count DESC LIMIT 20
            """,
            fetch_all=True
        )
        if emotions:
            metrics["emotional_distribution"] = {row['response_sentiment']: row['count'] for row in emotions}

        # 4. Average Quality Score (muestreo por perfiles más recientes).
        profiles = execute_sql_query(
            f"""
            SELECT health_profile->>'quality_history' as qh
            FROM user_profiles
            WHERE health_profile->>'quality_history' IS NOT NULL
            ORDER BY updated_at DESC NULLS LAST
            LIMIT {profile_sample_limit}
            """,
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


# ---------------------------------------------------------------------------
# [P3-HEALTH-AGGREGATES-DISCLOSURE-DEFERRED · 2026-05-15] Decisión de producto:
#
# Los 5 endpoints públicos siguientes (`/atomic-pool-health`,
# `/chunk-queue-health`, `/pantry-tolerance-health`, `/tz-fallback-health`,
# `/health/plan-graph`) exponen agregados operacionales SIN auth — fallback
# count, pool availability, backlog del worker, degraded rate. UUIDs ya
# hasheados via `_hash_uuid_for_public` (P2-HEALTH-UID-STRIP), pero los
# números agregados sí son visibles a cualquier IP.
#
# El audit production-readiness 2026-05-15 flageó esto como gap potencial
# (business-intel leak: un competidor polleando cada 5min obtiene la curva
# de salud del sistema). La decisión tomada es **mantenerlos públicos**:
#
#   - Útiles para Grafana / k8s probes / load balancers / UptimeRobot
#     externos que no tienen acceso a `CRON_SECRET`.
#   - Lo que se filtra son métricas operacionales agregadas — NO PII,
#     NO datos de usuario, NO secretos.
#   - El probe público canónico `/health/version` (P2-HEALTHZ-DEEP) ya
#     expone el `drift` y el último heartbeat — coherente con esta política.
#   - Si se necesita reservar la visibilidad granular, gatear con
#     `_verify_admin_token(...)` + ofrecer un endpoint público sintético
#     `/health/lite` con solo `{status: "ok"|"degraded"|"broken"}`. Eso
#     es un follow-up que requiere consenso explícito del equipo.
#
# Análogo al pattern `P3-I18N-DEFERRED`: lo que un auditor confunde con
# deuda es una decisión de producto. Si querés revertir, leé primero
# `~/.claude/projects/.../memory/project_p3_health_aggregates_disclosure_deferred_2026_05_15.md`.
#
# Test ancla: `tests/test_p3_health_aggregates_disclosure_deferred.py`.
# Tooltip-anchor: P3-HEALTH-AGGREGATES-DISCLOSURE-DEFERRED.
# ---------------------------------------------------------------------------


@router.get("/atomic-pool-health")
def get_atomic_pool_health():
    """[P1-4] Salud del `connection_pool` que sustenta
    `update_user_health_profile_atomic`.

    Retorna `{ pool_available, fallback_count, first_at, last_at,
    last_user_hash, strict_mode }`. En producción, alertear cuando:
      - `pool_available=false`: el pool cayó (red, credenciales, pooler agotado).
      - `fallback_count > 0` con `strict_mode=false`: la atomicidad está rota
        para el subset de calls que ya degradaron — lost-update silencioso en
        curso. Recomendación: bumpear `MEALFIT_REQUIRE_ATOMIC_POOL=1` en el
        próximo deploy para forzar fail-fast.

    Sin auth (read-only, útil para load balancers / k8s probes / Grafana).

    [P2-HEALTH-UID-STRIP · 2026-05-12] `last_user_hash` reemplaza el campo
    legacy `last_user_id`: SHA-256(uuid)[:12]. Pre-fix exponía el UUID
    literal del último user que disparó fallback — enumerable polleando
    desde cualquier IP. El hash preserva la utilidad de correlation visual
    (mismo digest = mismo usuario repetido) sin permitir enumeración.
    """
    try:
        from db_profiles import get_atomic_pool_fallback_snapshot
        snapshot = get_atomic_pool_fallback_snapshot()
        # [P2-HEALTH-UID-STRIP] Reemplazar el UUID con su hash truncado
        # ANTES del spread `**snapshot`. La fuente in-memory mantiene el
        # UUID raw para logs locales / debugging server-side.
        raw_uid = snapshot.pop("last_user_id", None)
        snapshot["last_user_hash"] = _hash_uuid_for_public(raw_uid)
        # Status semantics:
        #   - "ok": pool arriba, sin fallbacks acumulados.
        #   - "degraded": pool arriba pero hubo ≥1 fallback histórico (transient).
        #   - "broken": pool ausente AHORA (no atomicidad para nuevas requests).
        if not snapshot["pool_available"]:
            status = "broken"
        elif snapshot["fallback_count"] > 0:
            status = "degraded"
        else:
            status = "ok"
        return {"success": True, "status": status, **snapshot}
    except Exception as e:
        logger.error(f"Error calculando atomic pool health: {e}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


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
        # [P2-3 · 2026-05-08] `meal_plan_id IS NOT NULL` excluye filas
        # huérfanas tras delete del meal_plan padre (FK SET NULL). Health
        # endpoint debe reportar calidad de planes activos, no incluir
        # telemetría stale de planes ya eliminados.
        degraded_row = execute_sql_query(
            """
            SELECT
                COUNT(*)::int AS total,
                COUNT(*) FILTER (WHERE was_degraded IS TRUE)::int AS degraded
            FROM plan_chunk_metrics
            WHERE created_at > NOW() - INTERVAL '24 hours'
              AND meal_plan_id IS NOT NULL
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
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


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
    # [P2-HEALTH-UID-STRIP · 2026-05-12] El endpoint es público (sin auth).
    # `top_plans_24h` originalmente devolvía `plan_id` UUIDs literales —
    # enumerable polleando. Reemplazamos por `plan_hash` (SHA-256[:12])
    # que preserva correlation visual sin enumeración. SRE que necesite
    # el UUID original puede consultar el buffer in-memory desde un shell
    # del backend (no expuesto vía API pública).
    return {
        "success": True,
        "total_24h": len(in_window),
        "by_reason": by_reason,
        "top_plans_24h": [
            {"plan_hash": _hash_uuid_for_public(pid), "count": c}
            for pid, c in top_plans
        ],
        "unique_plans_24h": len(plans),
        "unique_users_24h": len(users),
        "oldest_event_age_seconds": int(now - oldest_ts) if oldest_ts else None,
        "active_dedupe_keys": len(_dedupe),
        "buffer_capacity": int(_cap),
    }


@router.get("/health/plan-graph")
def get_plan_graph_health():
    """[P1-9] Health detallado del grafo LangGraph del orquestador.

    Antes la única señal era el endpoint global `/ready` en `app.py`, que es
    binario (200/503) — útil para readiness probe de Kubernetes pero opaco
    para dashboards: no exponía el contador `_PLAN_GRAPH_BUILD_FAILURES`.

    Una situación común en producción: el build inicial falla (e.g., import
    cyclic transitorio post-deploy), reintenta, eventualmente compila. El
    grafo termina `ready=True` pero `build_failures > 0` indica inestabilidad
    histórica que merece investigación. `/ready` ya da 200 a esa altura;
    este endpoint expone el contador para que Grafana/alerting puedan
    detectarlo y escalar.

    Comportamiento:
      - 200 cuando `ready=True`. Body incluye `build_failures` para tracking.
      - 503 cuando `ready=False`: build aún no exitoso, requests caerán al
        fallback matemático.

    Diferencia con `/ready` (en app.py):
      - `/ready`: readiness probe k8s (binario, sin métricas).
      - `/api/system/health/plan-graph`: dashboard / alerting (con
        `build_failures` accesible siempre).
    """
    try:
        from graph_orchestrator import get_plan_graph_status
        status = get_plan_graph_status()
    except Exception as e:
        logger.error(f"[P1-9] No se pudo obtener plan_graph status: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "ready": False,
                "build_failures": -1,
                "status": "unknown",
                "message": (
                    f"No se pudo importar/consultar el estado del orquestador: "
                    f"{type(e).__name__}: {e}"
                ),
            },
        )

    if status.get("ready"):
        return {"success": True, **status}

    raise HTTPException(status_code=503, detail={"success": False, **status})


class _InvalidatePlanGraphBody(BaseModel):
    """[P1-A4] Body opcional para `POST /admin/plan-graph/invalidate`.

    `reason` se loguea y se persiste como `last_invalidation_reason` en el
    status — útil para auditoría (qué disparó cada invalidación).
    """
    reason: Optional[str] = Field(default=None, max_length=200)


@router.post("/admin/plan-graph/invalidate")
def admin_invalidate_plan_graph(request: Request, body: Optional[_InvalidatePlanGraphBody] = None):
    """[P1-A4] Invalida el grafo LangGraph cacheado del orquestador.

    Antes, `_PLAN_GRAPH` se construía lazy en la primera request y luego se
    reutilizaba para siempre. Si en producción se necesitaba reflejar un
    cambio dinámico (hot-fix de prompt sin redeploy, monkeypatch de un nodo,
    recovery de corrupción detectada upstream), la única vía era reiniciar
    el proceso completo — outage parcial + cold start del worker.

    Ahora este endpoint expone `invalidate_plan_graph()` con auth Bearer
    (mismo patrón que el resto de endpoints `/admin/*` del backend, vía
    `CRON_SECRET`). La próxima request al pipeline reconstruye el grafo
    sobre el estado actual del módulo.

    Auth:
        Header `Authorization: Bearer <CRON_SECRET>`. 401 si falta, 403 si
        no matchea, 503 si `CRON_SECRET` no está seteado en el ambiente
        (fail-secure: admin endpoints no se exponen sin secreto).

    Body (opcional):
        `{"reason": "hotfix prompt review_plan"}` — persistido para audit.

    Responde 200 con el snapshot post-invalidación (mismo shape que
    `GET /api/system/health/plan-graph` ampliado con `invalidations_total`,
    `last_invalidation_ts`, `last_invalidation_reason`).

    Notas operacionales:
      - NO reconstruye inmediatamente: la primera request post-invalidate
        paga la latencia de compile (~<100ms en hardware típico). Si querés
        warm-up post-invalidación, llamá también `warm_plan_graph()` desde
        un script o pegale al endpoint de generación con un payload mínimo.
      - Idempotente: invalidar dos veces seguidas suma 2 al counter pero
        no falla aunque el grafo ya estuviera en None.
    """
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
    try:
        from graph_orchestrator import invalidate_plan_graph
    except Exception as e:
        logger.error(f"[P1-A4] No se pudo importar invalidate_plan_graph: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Orquestador no disponible: {type(e).__name__}: {e}",
        )

    reason = (body.reason if body and body.reason else "admin_endpoint")
    status = invalidate_plan_graph(reason=reason)
    logger.warning(
        f"[P1-A4] plan_graph invalidado vía endpoint admin "
        f"(reason={reason!r}, total={status.get('invalidations_total')})."
    )
    return {"success": True, **status}


@router.post("/admin/deploy-lag/check")
def admin_force_deploy_lag_check(request: Request):
    """[P0-PROD-1-DEPLOY · 2026-05-12] Fuerza la ejecución inmediata del
    detector `_alert_deploy_lag_marker_stale` sin esperar al cron.

    Razón: el cron corre cada `MEALFIT_DEPLOY_LAG_CHECK_INTERVAL_HOURS`
    (default 1h). Cuando el operador acaba de publicar `expected_last_known_pfix`
    en `app_kv_store` tras un `git push` y quiere confirmar que prod ya
    tiene el binario actualizado, este endpoint da feedback inmediato:
      - Si live == expected → ningún alert se inserta y la respuesta lo
        confirma con `drift=False`.
      - Si live ≠ expected → emite `deploy_lag_drift_vs_expected` igual que
        el cron, retorna `drift=True` + ambos markers en la respuesta.

    Auth: `Authorization: Bearer <CRON_SECRET>` (mismo patrón que el resto
    de `/admin/*`). 503 si CRON_SECRET no está seteado.

    Anchor: P0-PROD-1-DEPLOY-FORCE-CHECK.

    Notas operacionales:
      - Endpoint best-effort: si el detector lanza excepción inesperada,
        retornamos 500 con el `type(e).__name__` (no leak del traceback al
        cliente — usar logs server-side para el detalle completo).
      - NO muta nada por sí mismo: solo invoca el detector, que sí puede
        insertar alerts en `system_alerts`. Idempotente: ejecutarlo 2 veces
        seguidas no duplica filas (ON CONFLICT (alert_key) DO UPDATE).
    """
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]
    try:
        from cron_tasks import _alert_deploy_lag_marker_stale, _DEPLOY_LAG_KV_KEY
        from db_core import execute_sql_query
        from app import _LAST_KNOWN_PFIX as live_marker
    except Exception as e:
        logger.error(f"[P0-PROD-1-DEPLOY] Import del detector falló: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Detector no disponible: {type(e).__name__}",
        )

    try:
        _alert_deploy_lag_marker_stale()
    except Exception as e:
        logger.error(f"[P0-PROD-1-DEPLOY] _alert_deploy_lag_marker_stale falló: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detector falló: {type(e).__name__}",
        )

    # Snapshot post-check para retornar al operador sin que tenga que
    # hacer un segundo round-trip a `system_alerts`.
    expected_marker = None
    try:
        row = execute_sql_query(
            "SELECT value FROM app_kv_store WHERE key = %s",
            (_DEPLOY_LAG_KV_KEY,),
            fetch_one=True,
        )
        if row:
            raw = row.get("value") if isinstance(row, dict) else None
            if isinstance(raw, str):
                expected_marker = raw
            elif isinstance(raw, dict):
                cand = raw.get("marker")
                if isinstance(cand, str):
                    expected_marker = cand
    except Exception as e:
        logger.debug(f"[P0-PROD-1-DEPLOY] SELECT KV expected (best-effort): {e}")

    drift = bool(
        expected_marker
        and live_marker
        and expected_marker.strip() != live_marker.strip()
    )
    return {
        "success": True,
        "live_marker": live_marker,
        "expected_marker": expected_marker,
        "drift": drift,
        "message": (
            "Drift detectado: forzar redeploy en EasyPanel." if drift
            else (
                "Sin drift: prod ejecuta la versión publicada en KV."
                if expected_marker else
                "Sin marker esperado publicado en app_kv_store — el detector "
                "solo evaluó staleness por antigüedad."
            )
        ),
    }


# [P1-OBS-1 · 2026-05-12] Anchor: P1-OBS-1-HEALTH-SNAPSHOT.
# Nodos observables expuestos en `watchdog_ticks`. Si añades un watchdog nuevo
# que emite tick en pipeline_metrics, agrega su node aquí Y al test parser-based
# correspondiente — el endpoint es contrato con Grafana/operador.
_HEALTH_SNAPSHOT_WATCHDOG_NODES = (
    "_hardfloor_autoheal_tick",
    "_hot_table_bloat_tick",
    "_pipeline_metrics_silence_check_tick",
)


@router.get("/admin/health-snapshot")
def admin_health_snapshot(request: Request):
    """[P1-OBS-1 · 2026-05-12] Snapshot agregado de salud operacional en una
    sola llamada. Reemplaza ~7 queries SQL manuales que SRE corría tras cada
    deploy para validar prod (drift, alerts abiertos, watchdogs vivos, chunks
    atascados, circuit breakers abiertos).

    Auth: `Authorization: Bearer <CRON_SECRET>` (mismo patrón que el resto
    de `/admin/*`). 503 si CRON_SECRET no está seteado.

    Anchor: P1-OBS-1-HEALTH-SNAPSHOT.

    Retorna (todas las keys siempre presentes — `None` si la sub-query falla):
      - live_marker: `_LAST_KNOWN_PFIX` del binario corriendo.
      - expected_marker: marker publicado en `app_kv_store.expected_last_known_pfix`.
      - drift: bool — True si los markers difieren (igual que /deploy-lag/check).
      - open_alerts: dict `{severity: count}` para `system_alerts.resolved_at IS NULL`.
      - metrics_15min: filas en `pipeline_metrics` con `created_at > NOW() - 15min`.
      - watchdog_ticks: dict `{node: last_created_at_iso}` para nodes observados.
      - stuck_chunks: `plan_chunk_queue` pending/stale con execute_after viejo (>15min).
      - dead_lettered_chunks: filas con `status='dead_lettered'`.
      - open_circuit_breakers: lista de `app_kv_store.key` con `is_open=true`.

    Cada sub-query es best-effort independiente: si una falla, su valor queda
    en `None` (o `[]`/`{}`) y el resto se devuelve. Esto evita que un blip
    transitorio en una tabla (ej. lock conflict) tumbe el endpoint entero.
    """
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)  # [P2-ADMIN-RATE-LIMIT]

    # 1. Drift: reuso de lógica del endpoint /deploy-lag/check (SIN ejecutar
    # el detector — solo lectura del KV + comparación contra _LAST_KNOWN_PFIX).
    live_marker: Optional[str] = None
    expected_marker: Optional[str] = None
    try:
        from app import _LAST_KNOWN_PFIX as _live
        live_marker = _live
    except Exception as e:
        logger.debug(f"[P1-OBS-1] Lectura _LAST_KNOWN_PFIX falló: {e}")

    try:
        from cron_tasks import _DEPLOY_LAG_KV_KEY
        row = execute_sql_query(
            "SELECT value FROM app_kv_store WHERE key = %s",
            (_DEPLOY_LAG_KV_KEY,),
            fetch_one=True,
        )
        if row:
            raw = row.get("value") if isinstance(row, dict) else None
            if isinstance(raw, str):
                expected_marker = raw
            elif isinstance(raw, dict):
                cand = raw.get("marker")
                if isinstance(cand, str):
                    expected_marker = cand
    except Exception as e:
        logger.debug(f"[P1-OBS-1] SELECT KV expected falló: {e}")

    drift = bool(
        expected_marker
        and live_marker
        and expected_marker.strip() != live_marker.strip()
    )

    # 2. Open alerts agrupados por severity.
    open_alerts: dict = {}
    try:
        rows = execute_sql_query(
            """
            SELECT COALESCE(severity, 'unknown') AS severity, COUNT(*)::int AS count
              FROM system_alerts
             WHERE resolved_at IS NULL
             GROUP BY severity
            """,
            fetch_all=True,
        ) or []
        for r in rows:
            open_alerts[str(r.get("severity") or "unknown")] = int(r.get("count") or 0)
    except Exception as e:
        logger.debug(f"[P1-OBS-1] SELECT system_alerts falló: {e}")

    # 3. pipeline_metrics rate en últimos 15 min.
    metrics_15min: Optional[int] = None
    try:
        row = execute_sql_query(
            "SELECT COUNT(*)::int AS n FROM pipeline_metrics WHERE created_at > NOW() - INTERVAL '15 minutes'",
            fetch_one=True,
        ) or {}
        metrics_15min = int(row.get("n") or 0)
    except Exception as e:
        logger.debug(f"[P1-OBS-1] SELECT pipeline_metrics count falló: {e}")

    # 4. Watchdog ticks por node observado.
    watchdog_ticks: dict = {node: None for node in _HEALTH_SNAPSHOT_WATCHDOG_NODES}
    try:
        rows = execute_sql_query(
            """
            SELECT node, MAX(created_at) AS last_at
              FROM pipeline_metrics
             WHERE node = ANY(%s::text[])
             GROUP BY node
            """,
            (list(_HEALTH_SNAPSHOT_WATCHDOG_NODES),),
            fetch_all=True,
        ) or []
        for r in rows:
            node = r.get("node")
            last_at = r.get("last_at")
            if node:
                watchdog_ticks[str(node)] = last_at.isoformat() if last_at else None
    except Exception as e:
        logger.debug(f"[P1-OBS-1] SELECT watchdog ticks falló: {e}")

    # 5. Stuck chunks (pending/stale con execute_after viejo) + dead-lettered.
    stuck_chunks: Optional[int] = None
    dead_lettered_chunks: Optional[int] = None
    try:
        row = execute_sql_query(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE status IN ('pending', 'stale')
                      AND execute_after < NOW() - INTERVAL '15 minutes'
                )::int AS stuck,
                COUNT(*) FILTER (WHERE status = 'dead_lettered')::int AS dead_lettered
              FROM plan_chunk_queue
            """,
            fetch_one=True,
        ) or {}
        stuck_chunks = int(row.get("stuck") or 0)
        dead_lettered_chunks = int(row.get("dead_lettered") or 0)
    except Exception as e:
        logger.debug(f"[P1-OBS-1] SELECT plan_chunk_queue falló: {e}")

    # 6. Circuit breakers abiertos en app_kv_store.
    open_circuit_breakers: list = []
    try:
        rows = execute_sql_query(
            """
            SELECT key
              FROM app_kv_store
             WHERE key LIKE 'llm_circuit_breaker%%'
               AND (value->>'is_open')::boolean IS TRUE
             ORDER BY key
            """,
            fetch_all=True,
        ) or []
        open_circuit_breakers = [str(r.get("key")) for r in rows if r.get("key")]
    except Exception as e:
        logger.debug(f"[P1-OBS-1] SELECT circuit breakers falló: {e}")

    return {
        "success": True,
        "live_marker": live_marker,
        "expected_marker": expected_marker,
        "drift": drift,
        "open_alerts": open_alerts,
        "metrics_15min": metrics_15min,
        "watchdog_ticks": watchdog_ticks,
        "stuck_chunks": stuck_chunks,
        "dead_lettered_chunks": dead_lettered_chunks,
        "open_circuit_breakers": open_circuit_breakers,
    }


@router.get("/admin/crons-status")
def admin_crons_status(request: Request):
    """[P3-CRONS-STATUS-ADMIN · 2026-05-15] Snapshot del estado de crons +
    knobs de kill-switch.

    ANTES: ~25+ crons registrados en `register_plan_chunk_scheduler` cada
    uno con su knob `MEALFIT_<NAME>_ENABLED` (P1-SCHEDULER-1, P2-NEXT-3,
    P3-NEW-D). Si un cron nuevo se añadía sin registrar su knob, no había
    tooling que lo detectara — SRE descubría el gap solo al intentar
    apagar el cron durante un incidente y notar que el env var no
    surtía efecto. Este endpoint enumera los jobs vivos del scheduler +
    sus knobs registrados para que SRE pueda comparar contra el catálogo
    esperado sin grep cross-archivo.

    Auth: `Authorization: Bearer <CRON_SECRET>` (consistente con resto de
    `/admin/*`). 503 si CRON_SECRET no está seteado.

    Retorna:
      - jobs: lista de `{job_id, next_run_time_iso, trigger, coalesce,
        max_instances}` extraídos de `scheduler.get_jobs()`. Vacío si el
        scheduler no está inicializado (HAS_SCHEDULER=False).
      - knobs_registry: snapshot de `_KNOBS_REGISTRY` filtrado a knobs
        cuyo nombre contiene `ENABLED` (kill switches) — permite verificar
        qué crons están apagados sin redeploy.
      - knobs_count_total: cardinalidad total del registry (incluye no-kill).
      - has_scheduler: bool — False indica APScheduler no instalado o
        deshabilitado por env.

    Anchor: P3-CRONS-STATUS-ADMIN.
    """
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)

    jobs_list: list[dict] = []
    has_scheduler = False
    try:
        from app import scheduler, HAS_SCHEDULER
        has_scheduler = bool(HAS_SCHEDULER)
        if scheduler is not None:
            for j in scheduler.get_jobs():
                try:
                    jobs_list.append({
                        "job_id": str(getattr(j, "id", "?")),
                        "next_run_time_iso": (
                            j.next_run_time.isoformat()
                            if getattr(j, "next_run_time", None) else None
                        ),
                        "trigger": str(getattr(j, "trigger", "?")),
                        "coalesce": bool(getattr(j, "coalesce", False)),
                        "max_instances": int(getattr(j, "max_instances", 1)),
                    })
                except Exception as _job_err:
                    logger.debug(
                        f"[P3-CRONS-STATUS-ADMIN] skip job (extract err): {_job_err}"
                    )
    except Exception as e:
        logger.debug(f"[P3-CRONS-STATUS-ADMIN] scheduler import/iterate falló: {e}")

    knobs_kill_switches: dict[str, object] = {}
    knobs_count_total: int = 0
    try:
        from graph_orchestrator import get_knobs_registry_snapshot
        _snap = get_knobs_registry_snapshot() or {}
        knobs_count_total = len(_snap)
        for k, v in _snap.items():
            if "ENABLED" in str(k):
                knobs_kill_switches[str(k)] = v
    except Exception as e:
        logger.debug(f"[P3-CRONS-STATUS-ADMIN] knobs snapshot falló: {e}")

    return {
        "success": True,
        "has_scheduler": has_scheduler,
        "jobs": jobs_list,
        "jobs_count": len(jobs_list),
        "knobs_kill_switches": knobs_kill_switches,
        "knobs_count_total": knobs_count_total,
    }


@router.get("/admin/cost-by-node")
def admin_cost_by_node(request: Request, hours: int = 24):
    """[P1-COST-BY-NODE-ENDPOINT · 2026-05-16] Leaderboard de costo LLM por
    nodo del pipeline, agregando `llm_usage_events` en una ventana temporal.

    Hace consumible la columna `node` populada por P1-COST-INSTRUMENTATION-PHASE2
    sin tener que entrar al SQL Editor cada vez. SRE abre el endpoint tras
    cualquier plan generado y ve qué nodo gasta cuánto en tiempo real.

    Auth: `Authorization: Bearer <CRON_SECRET>` (mismo patrón que el resto
    de `/admin/*`). Rate-limited por `_check_admin_rate_limit`.

    Query params:
      - `hours` (int, default 24, clamp [1, 720]): ventana hacia atrás
        desde NOW(). 720h = 30 días, máximo permitido para evitar full
        table scans no acotados.

    Retorna:
      - `success`: True
      - `window_hours`: el clamp aplicado.
      - `total_calls`, `total_usd`, `total_cached_pct`: agregados globales.
      - `by_node`: lista ordenada por `total_usd` desc:
          `{node, model, calls, total_usd, avg_usd, in_tok, out_tok, cached_tok, cached_pct}`
      - `unattributed`: bool — True si hay calls con `node=NULL` (típicamente
        agent tools, scripts admin, llamadas pre-P1-COST-INSTRUMENTATION-PHASE2).

    Anchor: P1-COST-BY-NODE-ENDPOINT.
    """
    _verify_admin_token(request.headers.get("authorization"))
    _check_admin_rate_limit(request)

    # Clamp del ventana — defensa contra full-scan accidental + spam.
    window_h = max(1, min(720, int(hours) if hours else 24))

    try:
        rows = execute_sql_query(
            """
            SELECT
                COALESCE(node, '(unattributed)') AS node,
                model,
                COUNT(*)::int AS calls,
                COALESCE(SUM(cost_usd_micros), 0) AS cost_micros_sum,
                COALESCE(AVG(cost_usd_micros), 0) AS cost_micros_avg,
                COALESCE(SUM(input_tokens), 0)::bigint AS in_tok,
                COALESCE(SUM(output_tokens), 0)::bigint AS out_tok,
                COALESCE(SUM(cached_tokens), 0)::bigint AS cached_tok
            FROM public.llm_usage_events
            WHERE created_at > NOW() - make_interval(hours => %s)
            GROUP BY node, model
            ORDER BY cost_micros_sum DESC
            """,
            (window_h,),
        )
    except Exception as e:
        logger.warning(f"[P1-COST-BY-NODE-ENDPOINT] query falló: {e}")
        raise HTTPException(status_code=503, detail="DB query falló (best-effort).")

    by_node: list[dict] = []
    total_calls = 0
    total_micros = 0
    total_in_tok = 0
    total_cached_tok = 0
    has_unattributed = False
    for r in (rows or []):
        try:
            cost_sum = int(r.get("cost_micros_sum") or 0)
            cost_avg = int(r.get("cost_micros_avg") or 0)
            in_tok = int(r.get("in_tok") or 0)
            out_tok = int(r.get("out_tok") or 0)
            cached_tok = int(r.get("cached_tok") or 0)
            calls = int(r.get("calls") or 0)
            node_name = str(r.get("node") or "(unattributed)")
            if node_name == "(unattributed)":
                has_unattributed = True
            total_calls += calls
            total_micros += cost_sum
            total_in_tok += in_tok
            total_cached_tok += cached_tok
            by_node.append({
                "node": node_name,
                "model": r.get("model"),
                "calls": calls,
                "total_usd": round(cost_sum / 1_000_000.0, 4),
                "avg_usd": round(cost_avg / 1_000_000.0, 6),
                "in_tok": in_tok,
                "out_tok": out_tok,
                "cached_tok": cached_tok,
                "cached_pct": (
                    round(100.0 * cached_tok / in_tok, 1) if in_tok else 0.0
                ),
            })
        except Exception as _row_err:
            logger.debug(f"[P1-COST-BY-NODE-ENDPOINT] skip row: {_row_err}")

    total_cached_pct = (
        round(100.0 * total_cached_tok / total_in_tok, 1) if total_in_tok else 0.0
    )

    return {
        "success": True,
        "window_hours": window_h,
        "total_calls": total_calls,
        "total_usd": round(total_micros / 1_000_000.0, 4),
        "total_cached_pct": total_cached_pct,
        "unattributed": has_unattributed,
        "by_node": by_node,
    }
