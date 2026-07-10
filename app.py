from fastapi import FastAPI, Request, Response, HTTPException, BackgroundTasks, UploadFile, File, Form, Body, Header, Depends
from error_utils import safe_error_detail
import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional
import json
import traceback
import threading
import time
import sentry_sdk
from knobs import _env_float as _knob_env_float

# [P1-A · 2026-05-08] Marker temporal del proceso. Expuesto vía
# `/health/version` para diagnosticar deployments rezagados (logs Postgres
# mostrando DDL runtime ya consolidado a SSOT migrations indica binary viejo).
_PROCESS_START_ISO = datetime.now(timezone.utc).isoformat()
# Marker textual del último P-fix mergeado en HEAD. Actualizar con cada
# cierre de fix para que `/health/version` permita comparar contra el árbol.
#
# [P3-1 · 2026-05-08] Convención (CLAUDE.md "Convenciones del repo"):
#   - Formato: `Pn-X · YYYY-MM-DD` o `Pn-NEW-X · YYYY-MM-DD`.
#   - Bumpear con CADA cierre de P-fix mergeado a HEAD.
#   - El test `test_p3_1_last_known_pfix_freshness` bloquea formato inválido
#     y fechas anteriores al floor (último audit cerrado).
#   - Si subes el floor del test, sube también el valor aquí — el commit
#     que sube uno sin el otro debería fallar el test en CI.
_LAST_KNOWN_PFIX = "P1-BRAND-PREF-PREP-DISTINCT · 2026-07-10"

# [P1-SENTRY-SAMPLE-COST · 2026-05-12] Sentry sampling driven from env vars
# con default seguro 0.1 (10%). Pre-fix tenía `traces_sample_rate=1.0` y
# `profiles_sample_rate=1.0` → 100% de transacciones + profiling continuo,
# costoso a escala y arriesga throttle de la cuota Sentry (dropping justo
# los errores que necesitas). Validator clamp [0.0, 1.0]; valores fuera de
# rango caen al default. Auto-registrado en `_KNOBS_REGISTRY` → visible en
# `/health/version`. Tooltip-anchor: P1-SENTRY-SAMPLE-COST.
#
# [P3-SENTRY-COST-THRESHOLDS · 2026-05-20] Audit `docs/gaps-audit-2026-05.md`
# C3: defaults actuales (0.1) son adecuados para <100 MAU. THRESHOLDS de
# revisión cuando crucemos escala:
#   - **>500 MAU**: bajar `replaysSessionSampleRate` (frontend, en
#     `frontend/src/main.jsx`) de 0.1 → 0.02 — replays son el output Sentry
#     más caro, satura cuota mensual gratis rápido a 1k+ users.
#   - **>1k MAU**: revisar también traces. Si la cuota Sentry mensual >70%
#     usada, bajar `MEALFIT_SENTRY_TRACES_SAMPLE_RATE` de 0.1 → 0.05.
#   - **Errores genuinos dropeados**: si Sentry empieza a mostrar
#     "events dropped due to throttling" en sus logs, bajar todos los
#     sample rates inmediatamente — los `captureException` reales se
#     pierden cuando la cuota está saturada, y eso es regresión visible.
# Ningún cambio aquí — solo doc. Los 3 knobs ya son ajustables sin redeploy.
_SENTRY_TRACES_SAMPLE_RATE = _knob_env_float(
    "MEALFIT_SENTRY_TRACES_SAMPLE_RATE",
    0.1,
    validator=lambda v: 0.0 <= v <= 1.0,
)
_SENTRY_PROFILES_SAMPLE_RATE = _knob_env_float(
    "MEALFIT_SENTRY_PROFILES_SAMPLE_RATE",
    0.1,
    validator=lambda v: 0.0 <= v <= 1.0,
)

# [P1-SENTRY-PII-SCRUBBING-BACKEND · 2026-05-15] `before_send` +
# `before_breadcrumb` que redactan PII (email, health_profile, plan_data,
# tokens, headers Authorization/Cookie) antes de enviar el event a Sentry.
#
# Pre-fix `sentry_sdk.init` corría sin estos callbacks. Cualquier excepción
# levantada dentro de un endpoint capturaba request body, headers, cookies,
# query string, y locals de la frame automáticamente — incluyendo
# `health_profile` (peso, altura, condiciones médicas), tokens PayPal en
# `/api/subscription/verify`, body completo del chat con prompts del usuario,
# y headers `Authorization` con JWTs. GDPR/HIPAA-relevant para datos de salud
# y risk de leak de credenciales si Sentry se ve comprometido o accedido por
# staff inquieto.
#
# Diseño defensivo:
#   - Cualquier excepción dentro del filtro NO dropea el event (mejor enviar
#     con PII y arreglar el filtro que perder un error genuino).
#   - Match por substring case-insensitive sobre keys (no exact match) para
#     atrapar variantes como `Authorization`, `authorization-bearer`,
#     `x-api-key`.
#   - Redacción a 3 niveles de profundidad (suficiente para `extra`, `contexts`,
#     `request.data` típicos).
#
# Tooltip-anchor: P1-SENTRY-PII-SCRUBBING-BACKEND.
# Test parser-based + funcional: `tests/test_p1_sentry_pii_scrubbing_backend.py`.

_SENSITIVE_KEY_SUBSTRINGS = (
    "password", "secret", "token", "authorization", "cookie",
    "email", "phone", "health_profile", "plan_data", "access_key",
    "api_key", "refresh_token", "credit_card", "card_number",
)


def _is_sensitive_key(key: str) -> bool:
    k = (key or "").lower()
    return any(s in k for s in _SENSITIVE_KEY_SUBSTRINGS)


def _redact_dict_in_place(obj, depth: int = 0) -> None:
    """Reemplaza valor por `[Filtered]` en keys sensibles. Recursivo hasta
    depth=3 para evitar cycles / nested infinito en payloads patológicos."""
    if depth > 3 or not isinstance(obj, dict):
        return
    for k in list(obj.keys()):
        v = obj[k]
        if _is_sensitive_key(k):
            obj[k] = "[Filtered]"
        elif isinstance(v, dict):
            _redact_dict_in_place(v, depth + 1)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _redact_dict_in_place(item, depth + 1)


def _sentry_redact_pii(event, hint):  # type: ignore[no-untyped-def]
    """`before_send` callback de Sentry. Muta + retorna event; nunca dropea."""
    try:
        if not isinstance(event, dict):
            return event
        req = event.get("request")
        if isinstance(req, dict):
            for sub_key in ("data", "headers", "cookies"):
                sub = req.get(sub_key)
                if isinstance(sub, dict):
                    _redact_dict_in_place(sub)
            qs = req.get("query_string")
            if isinstance(qs, str) and any(
                s in qs.lower() for s in ("token=", "secret=", "password=", "key=")
            ):
                req["query_string"] = "[Filtered]"
        extra = event.get("extra")
        if isinstance(extra, dict):
            _redact_dict_in_place(extra)
        ctxs = event.get("contexts")
        if isinstance(ctxs, dict):
            _redact_dict_in_place(ctxs)
        user = event.get("user")
        if isinstance(user, dict):
            for k in ("email", "username", "ip_address"):
                if k in user:
                    user[k] = "[Filtered]"
    except Exception:
        # NUNCA dropear por error del filtro — mejor enviar con PII que
        # perder el error genuino.
        pass
    return event


def _sentry_redact_breadcrumb(crumb, hint):  # type: ignore[no-untyped-def]
    """`before_breadcrumb` callback. Strip tokens/secrets en URLs + data."""
    try:
        if not isinstance(crumb, dict):
            return crumb
        data = crumb.get("data")
        if isinstance(data, dict):
            _redact_dict_in_place(data)
        msg = crumb.get("message")
        if isinstance(msg, str) and "?" in msg:
            q_lower = msg.lower()
            if any(s in q_lower for s in ("token=", "secret=", "password=", "key=")):
                crumb["message"] = msg.split("?", 1)[0] + "?[Filtered]"
    except Exception:
        pass
    return crumb


sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    traces_sample_rate=_SENTRY_TRACES_SAMPLE_RATE,
    profiles_sample_rate=_SENTRY_PROFILES_SAMPLE_RATE,
    before_send=_sentry_redact_pii,
    before_breadcrumb=_sentry_redact_breadcrumb,
)

# Configuración centralizada de logging para todo el backend.
# [H2 / P3-CORRELATION-ID · 2026-05-20] Format incluye `[corr=<8chars>]` —
# permite grep de logs por request para reconstruir el flow completo
# (request → handler → tools → db → bg_task) en un solo filtro.
# Default `-` cuando no hay scope activo (init, cron, shutdown). Ver
# `correlation.py` para diseño completo.
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [corr=%(correlation_id)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# [H2 / P3-CORRELATION-ID] Install filter sobre el root logger.
# DEBE ir DESPUÉS de basicConfig (el handler raíz se crea ahí) y ANTES
# del primer log emitido — los logs hasta este punto no tendrían el
# atributo `correlation_id` y crashearían el format string.
# El filter es idempotente, safe contra re-imports.
from correlation import install_log_filter  # noqa: E402 — orden intencional
install_log_filter()

# Silenciar logs verbosos de httpx (cliente HTTP interno)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from db import (
    connection_pool, async_connection_pool, chat_checkpoint_pool,
    execute_sql_query, execute_sql_write,
    get_or_create_session, save_message, save_message_feedback, insert_like, get_user_likes,
    insert_rejection, get_active_rejections, get_latest_meal_plan, get_user_profile,
    update_user_health_profile, update_user_health_profile_atomic, get_all_user_facts, delete_user_fact,
    save_new_meal_plan_robust,
    log_consumed_meal, get_consumed_meals_today, save_visual_entry, get_session_messages,
    get_user_chat_sessions, get_guest_chat_sessions, get_session_owner, delete_user_agent_sessions,
    delete_single_agent_session, update_session_title,
    check_fact_ownership, upsert_user_profile, migrate_guest_data, log_api_usage, get_monthly_api_usage
)
from agent import (
    swap_meal, chat_with_agent, analyze_preferences_agent,
    generate_chat_title_background, chat_with_agent_stream
)
from ai_helpers import generate_plan_title, expand_recipe_agent
from graph_orchestrator import (
    run_plan_pipeline, warm_plan_graph, is_plan_graph_ready,
    is_plan_graph_ready_with_reason,
    verify_pipeline_metrics_guest_insert,
)
from memory_manager import summarize_and_prune, build_memory_context
from fact_extractor import async_extract_and_save_facts, process_pending_queue_sync
from langgraph.checkpoint.postgres import PostgresSaver
from services import compute_plan_hash, merge_form_data_with_profile
from vision_agent import process_image_with_vision, get_multimodal_embedding
# [P2-1 · 2026-05-08] Helpers compartidos del registry de knobs (extraídos de
# graph_orchestrator). Los 3 knobs `MEALFIT_SCHEDULER_*` de abajo eran raw
# `os.environ.get` y no aparecían en `/health/version` ni en
# `get_knobs_registry_snapshot()`. Migrados a `_env_int`/`_env_bool` aquí.
from knobs import _env_int, _env_bool, is_production

# [P2-NEW-D · 2026-05-08] Knobs de scheduler. Antes `BackgroundScheduler()` se
# instanciaba sin args; default APScheduler = `ThreadPoolExecutor(max_workers=10)`
# + `misfire_grace_time=1s`. Con ~23 cron jobs registrados (1 en app.py + 22 en
# `register_plan_chunk_scheduler`), un burst en minuto 0 podía saturar el pool
# de 10 threads, encolar el 11º+ y — si su próximo schedule llegaba antes del
# drain — APScheduler hacía SKIP silencioso (logging.warning sin métrica). Riesgo
# crítico para `process_plan_chunk_queue`, `_alert_chunk_dual_processing`, etc.
#
# [P0-2 · 2026-05-10] Bump de defaults tras audit del 2026-05-10. system_alerts
# registró 25+ `scheduler_missed_*` en últimas 24h con bursts simultáneos que
# saturaban el pool de 20 threads — 23 jobs disparándose en la misma ventana
# de minuto excedían el grace_time de 60s cuando algún job tardaba >1s en
# tomar conexión DB. Subido a 32 workers + 180s de gracia para absorber el
# pico sin requerir staggering por job (refactor más invasivo). Aún ajustable
# vía env var para sobreprovisionar bajo carga inusual.
_SCHEDULER_MAX_WORKERS = _env_int("MEALFIT_SCHEDULER_MAX_WORKERS", 32)
_SCHEDULER_MISFIRE_GRACE_S = _env_int("MEALFIT_SCHEDULER_MISFIRE_GRACE_S", 180)

# [P2-AUDIT-NEW-3 · 2026-05-12] Ventana de gracia post-startup durante la
# cual los eventos MISSED/ERROR del scheduler NO emiten alerts a
# `system_alerts` (solo logs). Razón: en cada restart, APScheduler dispara
# EVENT_JOB_MISSED para todos los jobs cuyo `next_run_time` ya pasó durante
# el downtime — burst de 15-20 alerts triggered en <1s post-boot que solo
# añaden ruido (los autoheals P0-LIVE-1 / P0-AUDIT-1 las cierran en <5min).
#
# `_APP_START_TIME` se setea en `lifespan()` cuando FastAPI levanta el app.
# El listener compara `time.time() - _APP_START_TIME` contra el grace; si
# está dentro, solo log.info — no UPSERT a system_alerts ni breadcrumb.
#
# Default 5 min (subido de 2 el [P1-D · 2026-05-12]): el audit 2026-05-12
# observó un burst que cruzó la ventana de 2min (cascade abierto >35min
# tras restart porque jobs encolados durante downtime expiraron POSTERIOR
# al grace=2min). 5min cubre deploys rolling estándar del VPS Oracle
# sin ocultar MISSED genuinos (autoheals P0-LIVE-1 / P0-AUDIT-1
# + EVENT_JOB_EXECUTED listener P1-NEW-2 siguen cerrando cualquier MISSED
# real en <5min cuando el job re-ejecuta exitosamente, sin importar el
# grace). Clamp [0, 15] — 0 desactiva totalmente la supresión (back-compat),
# 15min es techo generoso para deploys lentos.
_SCHEDULER_BOOT_GRACE_MIN = max(0, min(_env_int("MEALFIT_SCHEDULER_BOOT_GRACE_MIN", 5), 15))
_SCHEDULER_BOOT_GRACE_S = _SCHEDULER_BOOT_GRACE_MIN * 60
_APP_START_TIME: float = 0.0  # set en lifespan()
# [P2-NEW-F · 2026-05-08] Telemetría: emite a system_alerts cuando un job se
# salta (MISSED) o falla (ERROR). Knob ON por default; `off`/`0`/`false` lo
# desactiva. Se lee fresh en cada invocación del listener (`_is_scheduler_telemetry_enabled`)
# para que el kill switch tome efecto sin restart del worker.
def _is_scheduler_telemetry_enabled() -> bool:
    # [P2-1 · 2026-05-08] `_env_bool` registra en `_KNOBS_REGISTRY`.
    return _env_bool("MEALFIT_SCHEDULER_TELEMETRY_ENABLED", True)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.executors.pool import ThreadPoolExecutor as _APSThreadPoolExecutor
    from apscheduler.events import EVENT_JOB_MISSED, EVENT_JOB_ERROR, EVENT_JOB_EXECUTED
    from proactive_agent import run_proactive_checks
    scheduler = BackgroundScheduler(
        executors={"default": _APSThreadPoolExecutor(_SCHEDULER_MAX_WORKERS)},
        job_defaults={
            "misfire_grace_time": _SCHEDULER_MISFIRE_GRACE_S,
            "coalesce": True,
            "max_instances": 1,
        },
    )
    HAS_SCHEDULER = True
except ImportError as e:
    logger.error(f"⚠️ [APScheduler] Falta instalar apscheduler o dependencias: {e}. El agente proactivo está deshabilitado.")
    HAS_SCHEDULER = False
    scheduler = None
    # [P1-NEW-2 · 2026-05-10] Stub para EVENT_JOB_EXECUTED si APScheduler
    # no está instalado — el listener referencia la constante en
    # `if code == EVENT_JOB_EXECUTED` y necesita un valor válido aunque
    # el path nunca corra (HAS_SCHEDULER=False corto-circuita más arriba).
    EVENT_JOB_MISSED = EVENT_JOB_ERROR = EVENT_JOB_EXECUTED = 0  # type: ignore[assignment]


# [P1-PERF-1 · 2026-05-12] Cache in-memory de job_ids con alerts abiertas.
#
# Por qué existe (audit 2026-05-11):
#   El listener `EVENT_JOB_EXECUTED` (P1-NEW-2) hacía un PATCH REST a
#   `/rest/v1/system_alerts?alert_key=in.(scheduler_missed_<job>,scheduler_error_<job>)
#   &resolved_at=is.null` por CADA job ejecutado, aunque NO existiera alert
#   pendiente. Con 28+ jobs registrados y ejecuciones cada minuto a cada hora,
#   eso significaba ~5000+ PATCH/h, la mayoría 200 OK con 0 rows afectadas.
#   MCP API logs mostraron ~1-3 PATCH/seg sostenido. Desperdicio real:
#   REST quota + DB CPU + RLS check + HTTP roundtrip.
#
# Diseño:
#   - `_SCHEDULER_JOBS_WITH_OPEN_ALERTS`: set[str] de job_ids cuya alert está
#     `resolved_at IS NULL`. Mantenido por el propio listener (add en MISSED/
#     ERROR, discard en EXECUTED post-éxito).
#   - Cold cache: en lifespan startup refrescamos el set con SELECT contra
#     DB. Sin esto, primer ciclo post-boot saltaría PATCHes que SÍ deberían
#     ocurrir (alerts persistentes de pre-restart).
#   - TTL: si pasaron `_OPEN_ALERTS_CACHE_TTL_S` (default 60s) desde el
#     último refresh, hacemos sync best-effort para captar mutaciones
#     fuera del listener (resoluciones manuales, sweeps del cron). Bajo
#     coste: 1 SELECT por minuto.
#   - Race: usar `threading.Lock` porque BackgroundScheduler dispatcha en
#     N threads. El listener es sync; el lock protege add/remove atómico.
#
# Trade-off consciente:
#   El cache es local al proceso. Multi-worker (uvicorn --workers N>1)
#   tendría N caches divergentes — pero la app usa BackgroundScheduler que
#   asume 1 proceso (sino tendrías N copias de cada cron). Si en el futuro
#   se migra a múltiples workers, el cache debería externalizarse a Redis
#   o a app_kv_store.
_SCHEDULER_JOBS_WITH_OPEN_ALERTS: set[str] = set()
_SCHEDULER_OPEN_ALERTS_LOCK = threading.Lock()
_SCHEDULER_OPEN_ALERTS_LAST_REFRESH: float = 0.0

# [P3-SCHEDULER-ALERT-DEDUP · 2026-05-15] Cache TTL 5s para evitar UPSERT
# duplicados desde APScheduler bajo network blip. Key = `alert_key`
# canónico (`scheduler_<event_type>_<job_id>`), value = monotonic time
# del último emit. Lazy: el dict crece sin bound formal — en práctica
# capped por `len(jobs) * 2 event_types`, ~50 entries para 25 jobs.
_SCHEDULER_ALERT_LAST_EMIT: dict[str, float] = {}
_SCHEDULER_ALERT_DEDUP_TTL_S: float = 5.0

# [P1-CASCADE-INLINE · 2026-05-27] Detector inline de cascada en el listener.
# Gap observado en audit 2026-05-27: el cron `_alert_scheduler_cascade_missed`
# corre cada `MEALFIT_SCHEDULER_CASCADE_INTERVAL_MIN` (default 30, floor 5min
# porque "Sub-5min agrava la propia cascada"). Cuando la cascada ocurre entre
# dos ticks del cron, el parent `scheduler_cascade_missed` se emite con hasta
# 30min de delay — SRE ve 18-25 misses huérfanos sin parent durante ese
# intervalo. Live evidence 2026-05-27: 25 distinct `scheduler_missed_*` en 7min
# con threshold default `3`, cero `scheduler_cascade_missed` parent emitido
# todavía. El listener YA corre por cada event MISSED en sub-segundos — el
# costo marginal de un counter in-memory + UPSERT condicional es despreciable
# y NO es "otro cron en el burst" (es 1 INSERT/UPSERT por cada N=5 misses,
# con dedup 120s entre emisiones).
#
# Sliding window de timestamps + dedup mantiene memoria estable; las
# constantes son knobs operacionales con clamps defensivos.
#
# Knobs:
#   MEALFIT_SCHEDULER_CASCADE_INLINE_ENABLED (default True)
#     Kill switch operacional sin redeploy si el inline detector introduce
#     volumen problemático contra `system_alerts`.
#   MEALFIT_SCHEDULER_CASCADE_INLINE_WINDOW_S (default 60, clamp [10, 600])
#     Ventana sliding sobre la que contamos distinct `job_id` misses.
#   MEALFIT_SCHEDULER_CASCADE_INLINE_THRESHOLD (default 5, clamp [3, 50])
#     N distinct jobs misseados dentro de la ventana → emit parent.
#     Default 5 (vs 3 del cron) por margen anti-falso-positivo: el cron
#     mira 1h lookback; el inline mira 60s — necesita ser más conservador
#     para no emitir en bursts triviales de 3 jobs MISSED simultáneos
#     post-restart legítimos (que el boot grace ya suprime parcialmente).
#   MEALFIT_SCHEDULER_CASCADE_INLINE_DEDUP_S (default 120, clamp [30, 1800])
#     Cooldown entre emisiones inline para no spammear si la cascada
#     persiste — el cron de 30min toma el relevo después.
#
# Tooltip-anchor: P1-CASCADE-INLINE.
import collections as _p1_collections
_CASCADE_INLINE_MISS_TIMESTAMPS: "_p1_collections.deque[tuple[float, str]]" = (
    _p1_collections.deque()
)
_CASCADE_INLINE_LOCK = threading.Lock()
_CASCADE_INLINE_LAST_EMIT_AT: float = 0.0


def _cascade_inline_enabled() -> bool:
    """Kill switch operacional (env-driven, sin redeploy). [P2-1-KNOBS-HYGIENE · 2026-06-15] vía
    `_env_bool` (auto-registro en `_KNOBS_REGISTRY` + visible en /health/version), no `os.environ.get` raw."""
    return _env_bool("MEALFIT_SCHEDULER_CASCADE_INLINE_ENABLED", True)


def _cascade_inline_window_s() -> float:
    raw = _env_int("MEALFIT_SCHEDULER_CASCADE_INLINE_WINDOW_S", 60)
    return float(max(10, min(raw, 600)))


def _cascade_inline_threshold() -> int:
    raw = _env_int("MEALFIT_SCHEDULER_CASCADE_INLINE_THRESHOLD", 5)
    return max(3, min(raw, 50))


def _cascade_inline_dedup_s() -> float:
    raw = _env_int("MEALFIT_SCHEDULER_CASCADE_INLINE_DEDUP_S", 120)
    return float(max(30, min(raw, 1800)))


def _maybe_emit_inline_cascade_alert(job_id: str) -> None:
    """[P1-CASCADE-INLINE · 2026-05-27] Append + slide + threshold check
    + UPSERT del parent `scheduler_cascade_missed` directo desde el
    listener. Llamado por `_scheduler_alert_listener` en cada EVENT_JOB_MISSED
    DESPUÉS del boot grace check (no contar misses suprimidos).

    Best-effort: cualquier excepción se loguea sin propagar al scheduler
    (la cascada es señal informativa, no debe pausar workers).

    Defense-in-depth: el cron `_alert_scheduler_cascade_missed` sigue siendo
    el SSOT del parent — el inline solo acelera la detección. Si el inline
    emite spurious, el cron lo confirma/desconfirma en el próximo tick.
    """
    if not _cascade_inline_enabled():
        return
    if connection_pool is None:
        return
    global _CASCADE_INLINE_LAST_EMIT_AT

    try:
        now_mono = time.monotonic()
        window_s = _cascade_inline_window_s()
        threshold = _cascade_inline_threshold()
        dedup_s = _cascade_inline_dedup_s()

        with _CASCADE_INLINE_LOCK:
            # Append este miss + pop los expirados (sliding window).
            _CASCADE_INLINE_MISS_TIMESTAMPS.append((now_mono, str(job_id)))
            while (
                _CASCADE_INLINE_MISS_TIMESTAMPS
                and now_mono - _CASCADE_INLINE_MISS_TIMESTAMPS[0][0] > window_s
            ):
                _CASCADE_INLINE_MISS_TIMESTAMPS.popleft()

            distinct_jobs = {jid for (_ts, jid) in _CASCADE_INLINE_MISS_TIMESTAMPS}
            distinct_count = len(distinct_jobs)

            if distinct_count < threshold:
                return

            # Dedup contra emisiones recientes (cooldown).
            if now_mono - _CASCADE_INLINE_LAST_EMIT_AT < dedup_s:
                logger.debug(
                    f"[P1-CASCADE-INLINE] skip emit (dedup): {distinct_count} "
                    f"distinct jobs in {window_s}s, but last emit "
                    f"{now_mono - _CASCADE_INLINE_LAST_EMIT_AT:.1f}s ago "
                    f"< dedup={dedup_s}s."
                )
                return

            _CASCADE_INLINE_LAST_EMIT_AT = now_mono
            # Snapshot de jobs para metadata (release lock antes del UPSERT
            # para no bloquear el thread del scheduler durante el roundtrip
            # a la DB).
            jobs_snapshot = sorted(distinct_jobs)

        from datetime import datetime as _dt_p1c, timezone as _tz_p1c
        now_utc = _dt_p1c.now(_tz_p1c.utc)
        # [P1-NEON-DB-MIGRATION · 2026-06-12] UPSERT via SQL directo (pool
        # psycopg) — antes PostgREST. Misma semántica: upsert por alert_key
        # con re-apertura (resolved_at=NULL) en cada emisión.
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, triggered_at, resolved_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, NULL)
            ON CONFLICT (alert_key) DO UPDATE
            SET alert_type = EXCLUDED.alert_type,
                severity = EXCLUDED.severity,
                title = EXCLUDED.title,
                message = EXCLUDED.message,
                metadata = EXCLUDED.metadata,
                triggered_at = EXCLUDED.triggered_at,
                resolved_at = NULL
            """,
            (
                "scheduler_cascade_missed",
                "scheduler_cascade",
                "critical",
                f"Cascada de scheduler MISSED detectada inline ({len(jobs_snapshot)} jobs)",
                (
                    f"Listener inline detectó {len(jobs_snapshot)} jobs distintos "
                    f"MISSED en ventana {window_s:.0f}s (threshold={threshold}). "
                    f"Cron `_alert_scheduler_cascade_missed` confirmará en su "
                    f"siguiente tick (interval=MEALFIT_SCHEDULER_CASCADE_INTERVAL_MIN). "
                    f"Posibles causas: thread pool saturado, restart del worker (VPS Oracle), "
                    f"GC pause sostenida."
                ),
                json.dumps({
                    "detected_by": "inline_listener",
                    "distinct_jobs_in_window": len(jobs_snapshot),
                    "window_seconds": window_s,
                    "threshold": threshold,
                    "jobs": jobs_snapshot[:50],  # Cap a 50 para no inflar JSON.
                }, ensure_ascii=False),
                now_utc,
            ),
        )
        logger.warning(
            f"[P1-CASCADE-INLINE] Emit parent scheduler_cascade_missed "
            f"INLINE: {len(jobs_snapshot)} distinct jobs MISSED en {window_s:.0f}s "
            f"(threshold={threshold}). Detector adelanta hasta 30min al cron."
        )
    except Exception as _inline_err:
        logger.warning(
            f"[P1-CASCADE-INLINE] Emit inline cascade falló (best-effort): "
            f"{_inline_err}. El cron `_alert_scheduler_cascade_missed` "
            f"seguirá siendo el SSOT."
        )
_OPEN_ALERTS_CACHE_TTL_S = _env_int("MEALFIT_SCHEDULER_OPEN_ALERTS_CACHE_TTL_S", 60)  # [P2-1-KNOBS-HYGIENE] vía helper; clamp abajo
if _OPEN_ALERTS_CACHE_TTL_S < 15:
    _OPEN_ALERTS_CACHE_TTL_S = 15
if _OPEN_ALERTS_CACHE_TTL_S > 300:
    _OPEN_ALERTS_CACHE_TTL_S = 300


def _refresh_scheduler_open_alerts_cache(force: bool = False) -> int:
    """Sincroniza `_SCHEDULER_JOBS_WITH_OPEN_ALERTS` con la realidad de DB.

    Idempotente. Best-effort: cualquier excepción se loguea sin propagar.
    Retorna el count de jobs en el set post-sync (0 si el pool DB no
    disponible o falló).
    """
    global _SCHEDULER_OPEN_ALERTS_LAST_REFRESH
    if not force:
        if time.time() - _SCHEDULER_OPEN_ALERTS_LAST_REFRESH < _OPEN_ALERTS_CACHE_TTL_S:
            with _SCHEDULER_OPEN_ALERTS_LOCK:
                return len(_SCHEDULER_JOBS_WITH_OPEN_ALERTS)
    if connection_pool is None:
        return 0
    try:
        # SELECT alerts abiertas; parsear job_id; reconstruir el set.
        # No paginamos: típicamente <100 alerts abiertas.
        rows = execute_sql_query(
            """
            SELECT alert_key
            FROM system_alerts
            WHERE resolved_at IS NULL
              AND (alert_key LIKE %s OR alert_key LIKE %s)
            """,
            ("scheduler_missed_%", "scheduler_error_%"),
            fetch_all=True,
        ) or []
        new_set: set[str] = set()
        for r in rows:
            key = (r or {}).get("alert_key") or ""
            if key == "scheduler_cascade_missed":
                continue
            if key.startswith("scheduler_missed_"):
                new_set.add(key[len("scheduler_missed_"):])
            elif key.startswith("scheduler_error_"):
                new_set.add(key[len("scheduler_error_"):])
        with _SCHEDULER_OPEN_ALERTS_LOCK:
            _SCHEDULER_JOBS_WITH_OPEN_ALERTS.clear()
            _SCHEDULER_JOBS_WITH_OPEN_ALERTS.update(new_set)
        _SCHEDULER_OPEN_ALERTS_LAST_REFRESH = time.time()
        return len(new_set)
    except Exception as e:
        logger.debug(f"[P1-PERF-1] refresh cache falló (best-effort): {e}")
        return 0


def _scheduler_alert_listener(event):
    """[P2-NEW-F · 2026-05-08] Listener de eventos APScheduler.

    Emite a `system_alerts` cuando un job se salta (MISSED) o lanza excepción
    (ERROR). Cierra el gap de observabilidad detectado en el audit 2026-05-07:
    23+ crons registrados sin métrica de misfire/error → un job crítico
    (process_plan_chunk_queue, alert_chunk_dual_processing, etc.) podía
    saltarse silenciosamente, dejando solo log warning sin alerta accionable.

    Defensivo: cualquier error del listener se loguea sin crashear el scheduler
    (un fallo de DB no debe pausar el resto de los crons). Idempotente
    vía UPSERT por `alert_key` único: cada nuevo evento del mismo job actualiza
    la fila existente con el `triggered_at` más reciente.

    Knob `MEALFIT_SCHEDULER_TELEMETRY_ENABLED` (default `on`): kill switch
    operacional sin redeploy si la telemetría introduce overhead inesperado.
    """
    if not _is_scheduler_telemetry_enabled():
        return
    try:
        from datetime import datetime, timezone as _tz
        code = getattr(event, "code", None)
        job_id = getattr(event, "job_id", "unknown")
        scheduled_run_time = getattr(event, "scheduled_run_time", None)
        if code == EVENT_JOB_MISSED:
            event_type = "missed"
            severity = "warning"
            title = f"APScheduler job MISSED: {job_id}"
            message = (
                f"Job '{job_id}' skipped — scheduled at {scheduled_run_time}. "
                f"Misfire grace de {_SCHEDULER_MISFIRE_GRACE_S}s superado. "
                f"Posibles causas: thread pool saturado "
                f"({_SCHEDULER_MAX_WORKERS} workers), GC pause, DB blip. "
                f"Revisar pickup lag en logs."
            )
            # [P2-AUDIT-NEW-3 · 2026-05-12] Suprimir MISSED durante boot grace.
            # APScheduler dispara EVENT_JOB_MISSED para todos los jobs cuyo
            # next_run_time pasó durante el downtime del restart — burst de
            # 15-20 alerts en <1s post-boot que solo añaden ruido. Los
            # autoheals (P0-LIVE-1 / P0-AUDIT-1 / EVENT_JOB_EXECUTED listener
            # P1-NEW-2) cierran cualquier MISSED real en <5min cuando el job
            # re-ejecuta. Log INFO sigue presente para auditoría.
            if _APP_START_TIME > 0 and _SCHEDULER_BOOT_GRACE_S > 0:
                _uptime_s = time.time() - _APP_START_TIME
                if _uptime_s < _SCHEDULER_BOOT_GRACE_S:
                    logger.info(
                        f"⏳ [P2-AUDIT-NEW-3] Suprimiendo scheduler_missed_{job_id} "
                        f"durante boot grace (uptime={_uptime_s:.1f}s < "
                        f"grace={_SCHEDULER_BOOT_GRACE_S}s). Knob "
                        f"MEALFIT_SCHEDULER_BOOT_GRACE_MIN={_SCHEDULER_BOOT_GRACE_MIN}."
                    )
                    return
            # [P1-CASCADE-INLINE · 2026-05-27] Counter sub-minuto: si >=N
            # distinct jobs MISSED en ventana corta, emitir el parent
            # `scheduler_cascade_missed` inline sin esperar al cron de 30min.
            # Llamado DESPUÉS del boot grace check para no contar misses
            # suprimidos (que ya son ruido conocido post-restart).
            _maybe_emit_inline_cascade_alert(job_id)
        elif code == EVENT_JOB_ERROR:
            event_type = "error"
            severity = "critical"
            title = f"APScheduler job ERROR: {job_id}"
            exc = getattr(event, "exception", None)
            exc_summary = f"{type(exc).__name__}: {exc}" if exc else "unknown"
            message = (
                f"Job '{job_id}' raised exception during scheduled run at "
                f"{scheduled_run_time}: {exc_summary}"
            )
            # [P2-AUDIT-NEW-3 · 2026-05-12] ERROR durante grace NO se suprime.
            # MISSED post-boot es noise puro (job se reagenda y reintenta);
            # ERROR es siempre una excepción real que merece visibilidad
            # aunque ocurra temprano. Si pre-warming dispara ERROR es señal
            # de que algo está roto del lado del job, NO del scheduler.
        elif code == EVENT_JOB_EXECUTED:
            # [P1-NEW-2 · 2026-05-10 · optimizado P1-PERF-1 2026-05-12]
            # Auto-resolución de alertas `scheduler_missed_<job>` y
            # `scheduler_error_<job>` cuando el mismo job ejecuta exitosamente.
            #
            # Pre P1-PERF-1: hacíamos PATCH REST por CADA EXECUTED (~5000+/h
            # sostenido aunque el UPDATE fuera no-op por 0 rows). Ahora
            # consultamos el cache in-memory; el PATCH solo ocurre cuando
            # SÍ existe una alert pendiente para este job_id (cache hit).
            # Refresh TTL=60s capta cambios out-of-band (resoluciones
            # manuales, sweeps del cron).
            if connection_pool is None:
                return
            # Refresh perezoso si pasó el TTL — barato (~1 SELECT/min).
            _refresh_scheduler_open_alerts_cache(force=False)
            with _SCHEDULER_OPEN_ALERTS_LOCK:
                needs_patch = job_id in _SCHEDULER_JOBS_WITH_OPEN_ALERTS
            if not needs_patch:
                return  # ← skip PATCH no-op
            try:
                # [P1-NEON-DB-MIGRATION · 2026-06-12] UPDATE via SQL directo;
                # el predicado `resolved_at IS NULL` preserva timestamps de
                # alerts ya cerradas manualmente (paridad con P1-NEW-2).
                execute_sql_write(
                    """
                    UPDATE system_alerts
                    SET resolved_at = %s
                    WHERE alert_key = ANY(%s)
                      AND resolved_at IS NULL
                    """,
                    (
                        datetime.now(_tz.utc),
                        [
                            f"scheduler_missed_{job_id}",
                            f"scheduler_error_{job_id}",
                        ],
                    ),
                )
                # PATCH exitoso → remover del cache. Si el job vuelve a
                # MISSED/ERROR, la rama de abajo lo re-añade.
                with _SCHEDULER_OPEN_ALERTS_LOCK:
                    _SCHEDULER_JOBS_WITH_OPEN_ALERTS.discard(job_id)
            except Exception as _resolve_err:
                # No emit warning para no spammear; un fallo de auto-resolve
                # NO debe pausar el scheduler. La alerta original seguirá
                # visible hasta resolución manual o próximo EXECUTED.
                logger.debug(
                    f"[P1-NEW-2] auto-resolve alert para job={job_id} "
                    f"falló (best-effort): {_resolve_err}"
                )
            return
        else:
            return  # otros eventos no nos interesan

        if connection_pool is None:
            return
        alert_key = f"scheduler_{event_type}_{job_id}"

        # [P3-SCHEDULER-ALERT-DEDUP · 2026-05-15] Dedup TTL 5s. ANTES,
        # APScheduler bajo network blip podía emitir el mismo
        # MISSED/ERROR dos veces en <1s (race interna del scheduler);
        # cada uno disparaba un UPSERT contra system_alerts. Aunque
        # idempotente por `alert_key`, son 2 roundtrips REST + 2 UPDATE
        # ON CONFLICT por evento — spam minor pero observable en MCP
        # API logs como `PATCH /rest/v1/system_alerts` duplicado.
        # Ahora cache local con TTL 5s skipea el segundo emit dentro
        # de la ventana.
        _now_mono = time.monotonic()
        _last_emit = _SCHEDULER_ALERT_LAST_EMIT.get(alert_key, 0.0)
        if _now_mono - _last_emit < _SCHEDULER_ALERT_DEDUP_TTL_S:
            logger.debug(
                f"[P3-SCHEDULER-ALERT-DEDUP] skip duplicate emit "
                f"alert_key={alert_key} delta={_now_mono - _last_emit:.2f}s "
                f"< ttl={_SCHEDULER_ALERT_DEDUP_TTL_S}s"
            )
            return
        _SCHEDULER_ALERT_LAST_EMIT[alert_key] = _now_mono

        now_utc = datetime.now(_tz.utc)
        sched_iso = scheduled_run_time.isoformat() if scheduled_run_time else None
        # [P1-NEON-DB-MIGRATION · 2026-06-12] UPSERT via SQL directo (pool
        # psycopg) — antes PostgREST. Idempotente por alert_key único; cada
        # nuevo evento re-abre la alert (resolved_at=NULL) con triggered_at
        # más reciente.
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, triggered_at, resolved_at)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, NULL)
            ON CONFLICT (alert_key) DO UPDATE
            SET alert_type = EXCLUDED.alert_type,
                severity = EXCLUDED.severity,
                title = EXCLUDED.title,
                message = EXCLUDED.message,
                metadata = EXCLUDED.metadata,
                triggered_at = EXCLUDED.triggered_at,
                resolved_at = NULL
            """,
            (
                alert_key,
                "scheduler",
                severity,
                title,
                message,
                json.dumps({
                    "job_id": job_id,
                    "scheduled_run_time": sched_iso,
                    "event_type": event_type,
                    "max_workers": _SCHEDULER_MAX_WORKERS,
                    "misfire_grace_s": _SCHEDULER_MISFIRE_GRACE_S,
                }, ensure_ascii=False),
                now_utc,
            ),
        )
        # [P1-PERF-1] Mantener cache sincrónico: alert recién insertada →
        # job_id en el set. Próximo EXECUTED disparará el PATCH.
        with _SCHEDULER_OPEN_ALERTS_LOCK:
            _SCHEDULER_JOBS_WITH_OPEN_ALERTS.add(job_id)
    except Exception as listener_err:
        logger.warning(
            f"[P2-NEW-F] Listener de scheduler falló para job="
            f"{getattr(event, 'job_id', '?')}: {listener_err}"
        )

async def _hardfloor_autoheal_loop(interval_s: int):
    """[P0-LIVE-1 · 2026-05-11] Loop asyncio independiente del APScheduler.

    Invoca `_resolve_stale_scheduler_alerts` y `_sweep_stale_llm_circuit_breakers`
    cada `interval_s` segundos directamente desde el event loop de FastAPI
    (`asyncio.to_thread` para no bloquear). Garantiza que ambos sweeps corren
    aunque APScheduler esté saturado — cierra el chicken-and-egg detectado
    en el audit 2026-05-11:

      - `scheduler_cascade_missed` (CRITICAL) abierto >35min en prod;
        `_alert_scheduler_cascade_missed` (P0-NEW-2 autoheal) y
        `_resolve_stale_scheduler_alerts` (P0-AUDIT-1, P2-LIVE-1) están AMBOS
        dentro del scheduler. Cuando el pool está saturado el autoheal
        también está MISSED.
      - CB rows `is_open=true` por 4+ días (`gemini-3.1-pro-preview` desde
        2026-05-07). `_sweep_stale_llm_circuit_breakers` (P2-NEW-D) está
        registrado pero pipeline_metrics muestra 0 ticks en 48h.

    Por qué asyncio (no thread):
      - `BackgroundScheduler` corre en su propio ThreadPoolExecutor (32 workers).
        El event loop de FastAPI/uvicorn es independiente.
      - `asyncio.to_thread` despacha el sweep sync a un worker thread del
        event loop default pool — no compite con el pool del scheduler.

    Knobs:
      MEALFIT_HARDFLOOR_AUTOHEAL_ENABLED (default True) — kill switch.
      MEALFIT_HARDFLOOR_AUTOHEAL_INTERVAL_S (default 300, clamp [60, 1800]).

    Tick observable (espejo P3-LIVE-1): emite `_hardfloor_autoheal_tick` a
    pipeline_metrics en cada iteración para confirmar live que el loop
    está vivo independiente del scheduler.

    Tooltip-anchor: P0-LIVE-1-START | gap audit 2026-05-11
    """
    while True:
        scheduler_swept = None
        cb_reset = None
        autoheal_failed = False
        cb_failed = False
        try:
            from cron_tasks import _resolve_stale_scheduler_alerts
            await asyncio.to_thread(_resolve_stale_scheduler_alerts)
            scheduler_swept = "ok"
        except Exception as _autoheal_err:
            autoheal_failed = True
            logger.warning(
                f"[P0-LIVE-1] _resolve_stale_scheduler_alerts falló "
                f"(best-effort): {_autoheal_err}"
            )

        try:
            from cron_tasks import _sweep_stale_llm_circuit_breakers
            cb_reset = await asyncio.to_thread(_sweep_stale_llm_circuit_breakers)
            if cb_reset:
                logger.info(
                    f"[P0-LIVE-1] CB hard-floor sweep reset {cb_reset} stale rows."
                )
        except Exception as _cb_err:
            cb_failed = True
            logger.warning(
                f"[P0-LIVE-1] _sweep_stale_llm_circuit_breakers falló "
                f"(best-effort): {_cb_err}"
            )

        try:
            from cron_tasks import execute_sql_write
            await asyncio.to_thread(
                execute_sql_write,
                """
                INSERT INTO pipeline_metrics
                    (user_id, session_id, node, duration_ms, retries,
                     tokens_estimated, confidence, metadata)
                VALUES (NULL, NULL, %s, 0, 0, 0, 0, %s::jsonb)
                """,
                (
                    "_hardfloor_autoheal_tick",
                    json.dumps({
                        "interval_s": interval_s,
                        "scheduler_swept": scheduler_swept,
                        "cb_reset": cb_reset if isinstance(cb_reset, int) else None,
                        "autoheal_failed": autoheal_failed,
                        "cb_failed": cb_failed,
                    }, ensure_ascii=False),
                ),
            )
        except Exception as _tick_err:
            logger.debug(
                f"[P0-LIVE-1] hard-floor tick emit falló (best-effort): {_tick_err}"
            )

        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            logger.info("🛟 [P0-LIVE-1] Hard-floor autoheal loop cancelado (shutdown).")
            raise


# [P1-SCHEDULER-LEADER-LOCK · 2026-05-27] Leader election para el
# `BackgroundScheduler` in-process. Razón: el scheduler corre dentro del
# proceso uvicorn; si el VPS Oracle/gunicorn arranca con >1 worker, los crons se
# ejecutan N veces (UPSERTs duplicados a `system_alerts`, races de pickup en
# `plan_chunk_queue`, N× hard-floor loops). El repo asume 1 worker (ver
# comentario ~L338) pero NO existe Dockerfile/Procfile versionado que lo fije
# — el worker count vivía sólo en la config del VPS Oracle → asunción no enforced.
#
# Este guard adquiere un advisory lock de Postgres a NIVEL DE SESIÓN sobre una
# conexión session-mode (5432) dedicada que se mantiene abierta toda la vida
# del proceso. Sólo el worker que obtiene el lock arranca el scheduler + el
# hard-floor loop; los demás sirven HTTP normalmente pero NO ejecutan crons.
#
# Forma de DOS-int `pg_try_advisory_lock(classid, objid)` → espacio de locks
# SEPARADO del single-bigint `pg_advisory_xact_lock(hashtextextended(...))`
# que usan los locks de meal_plans/user_history (db_plans.py) → CERO colisión.
# Nivel de sesión (`pg_try_advisory_lock`, NO `_xact_`) → persiste mientras la
# conexión viva; por eso debe ser session-mode 5432 (el Transaction Pooler
# 6543 liberaría el lock al terminar la sentencia).
#
# FAIL-OPEN: si el mecanismo falla por cualquier razón (sin DATABASE_URL,
# error de conexión, etc.) este worker actúa como leader igual — preferimos el
# riesgo conocido de crons duplicados antes que DESACTIVAR todos los crons.
# Knob MEALFIT_SCHEDULER_LEADER_LOCK=0 desactiva el guard (rollback sin
# redeploy → comportamiento pre-fix: cada worker arranca su scheduler).
# Tooltip-anchor: P1-SCHEDULER-LEADER-LOCK.
def _build_session_mode_db_url() -> Optional[str]:
    # [P1-NEON-DB-MIGRATION · 2026-06-12] La resolución del URL session-mode
    # vive en db_core (knob MEALFIT_DB_BACKEND): Neon → NEON_DATABASE_URL
    # (endpoint directo, único backend activo). El mirror local de la
    # heurística ':6543'→':5432' era NO-OP con hostnames Neon y el leader
    # lock habría caído en el pooler transaction-mode (advisory lock de
    # sesión liberado por sentencia → lock inútil).
    try:
        from db_core import DB_SESSION_MODE_URL
        if DB_SESSION_MODE_URL:
            return DB_SESSION_MODE_URL
    except Exception:
        pass
    # Fallback legacy (pools no configurados — e.g. import de psycopg_pool
    # falló pero psycopg directo sí está disponible). [P1-SUPABASE-CLEANUP ·
    # 2026-06-13] Lee el endpoint DIRECTO de Neon (ya es session-mode nativo);
    # el rewrite :6543→:5432 (específico de Supavisor del proveedor anterior)
    # ya no aplica porque el endpoint Neon es directo.
    raw = os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")
    if not raw:
        return None
    return raw.strip().strip("'").strip('"')


def _acquire_scheduler_leader_lock():
    """[P1-SCHEDULER-LEADER-LOCK · 2026-05-27] Devuelve `(is_leader, conn)`.
    `conn` sostiene el lock de sesión (debe cerrarse en shutdown para liberarlo)
    o None. FAIL-OPEN: ante cualquier error retorna `(True, None)` — este worker
    actúa como leader (prefiere crons duplicados a cero crons).

    [P2-PROD-AUDIT-FOLLOWUP · 2026-05-28] LIMITACIÓN ACEPTADA (tooltip-anchor
    P2-LEADER-LOCK-TRIPWIRE): el liderazgo se latchea UNA vez en el startup y
    NO se re-evalúa. Si la conn de sesión muere (blip de red / Supavisor),
    Postgres libera el advisory lock pero este worker SIGUE actuando como
    leader (fail-safe: preferimos crons sin lock a cero crons). Esto NO se
    auto-cura y NO re-elige — es INTENCIONAL: el `Procfile` fija `--workers 1`,
    así que sólo existe UN proceso y el escenario "dos leaders" es imposible en
    la práctica. Este lock es un TRIPWIRE defensivo contra un deploy multi-worker
    accidental (varios procesos → sólo el primero gana el lock), NO un mecanismo
    de elección self-healing. Si algún día se corre con >1 worker de forma
    intencional, AÑADIR aquí un heartbeat periódico que re-asierte el lock (sin
    apilar el contador re-entrante: verificar vía `pg_locks`, no re-llamar
    `pg_try_advisory_lock`) y re-elija ante caída de conn. Mientras `--workers 1`
    sea el contrato, no vale la pena la complejidad adicional."""
    if not _env_bool("MEALFIT_SCHEDULER_LEADER_LOCK", True):
        logger.info(
            "🔓 [P1-SCHEDULER-LEADER-LOCK] Guard desactivado via knob "
            "MEALFIT_SCHEDULER_LEADER_LOCK=0 — este worker arranca el scheduler "
            "sin leader election (comportamiento pre-fix)."
        )
        return True, None
    classid = _env_int(
        "MEALFIT_SCHEDULER_LEADER_LOCK_KEY",
        424242,
        validator=lambda v: -2147483648 <= v <= 2147483647,
    )
    session_url = _build_session_mode_db_url()
    if not session_url:
        logger.warning(
            "🔓 [P1-SCHEDULER-LEADER-LOCK] NEON_DATABASE_URL ausente — fail-open: "
            "este worker actúa como leader (sin lock). Esperado en dev local."
        )
        return True, None
    conn = None
    try:
        import psycopg
        conn = psycopg.connect(
            session_url,
            autocommit=True,
            keepalives=1,
            keepalives_idle=30,
            keepalives_interval=10,
            keepalives_count=5,
            # [P3-PROD-AUDIT-2 · 2026-05-30] connect_timeout para no colgar el
            # lifespan startup hasta el TCP timeout del OS si la DB está
            # inalcanzable durante un deploy (FastAPI no atendería /health ni
            # /ready). Consistente con reconnect_timeout=5 de los pools.
            connect_timeout=5,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s, %s)", (classid, 1))
            row = cur.fetchone()
            got = bool(row[0]) if row else False
        if got:
            logger.info(
                f"👑 [P1-SCHEDULER-LEADER-LOCK] Lock ({classid},1) adquirido — "
                "este worker es el LEADER (ejecuta crons + hard-floor)."
            )
            return True, conn
        # No somos leader: otro worker ya tiene el lock. La conn no sostiene
        # nada → cerrarla y NO arrancar el scheduler en este worker.
        try:
            conn.close()
        except Exception:
            pass
        logger.warning(
            f"🚫 [P1-SCHEDULER-LEADER-LOCK] Lock ({classid},1) ya tomado por otro "
            "worker — este worker NO ejecutará crons/hard-floor (sólo sirve HTTP). "
            "Indica deploy multi-worker; el repo asume 1 worker (ver Procfile)."
        )
        return False, None
    except Exception as _lock_err:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        logger.warning(
            f"🔓 [P1-SCHEDULER-LEADER-LOCK] Adquisición de lock falló "
            f"({type(_lock_err).__name__}: {_lock_err}) — fail-open: este worker "
            "actúa como leader (prefiere crons duplicados a cero crons)."
        )
        return True, None


def _emit_critical_config_alerts() -> None:
    """[P2-CRITICAL-CONFIG-ALERT · 2026-06-15] (gap-audit G7+G10) Al arranque emite/resuelve `system_alerts`
    para configuración crítica mal seteada EN PRODUCCIÓN: motor de precisión OFF (G7) o guards de seguridad
    clínica OFF (G10). Best-effort: NUNCA rompe el arranque. Auto-resuelve los `critical_config` que ya no
    aplican (config corregida + redeploy). En no-prod `get_critical_config_warnings()` retorna []."""
    try:
        from graph_orchestrator import get_critical_config_warnings
        warnings = get_critical_config_warnings()
    except Exception as _cfg_e:
        logger.warning(f"[P2-CRITICAL-CONFIG-ALERT] no pude computar warnings de config: {_cfg_e}")
        return
    from datetime import datetime as _dt_cfg, timezone as _tz_cfg
    now_utc = _dt_cfg.now(_tz_cfg.utc)
    active_keys = [w["alert_key"] for w in warnings]
    try:
        for w in warnings:
            execute_sql_write(
                """
                INSERT INTO system_alerts
                    (alert_key, alert_type, severity, title, message, metadata, triggered_at, resolved_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, NULL)
                ON CONFLICT (alert_key) DO UPDATE
                SET alert_type=EXCLUDED.alert_type, severity=EXCLUDED.severity, title=EXCLUDED.title,
                    message=EXCLUDED.message, metadata=EXCLUDED.metadata, triggered_at=EXCLUDED.triggered_at,
                    resolved_at=NULL
                """,
                (w["alert_key"], "critical_config", w["severity"], w["title"], w["message"],
                 json.dumps({"detected_by": "startup_config_check"}, ensure_ascii=False), now_utc),
            )
            logger.warning(f"🛡 [P2-CRITICAL-CONFIG-ALERT] {w['alert_key']}: {w['title']}")
        # Auto-resolver los `critical_config` abiertos que ya NO aplican (lista vacía → resuelve todos).
        execute_sql_write(
            """
            UPDATE system_alerts SET resolved_at = %s
            WHERE resolved_at IS NULL AND alert_type = 'critical_config'
              AND NOT (alert_key = ANY(%s))
            """,
            (now_utc, active_keys),
        )
        if not warnings:
            logger.info("✅ [P2-CRITICAL-CONFIG-ALERT] Config crítica OK (o entorno no-prod) — sin alertas.")
    except Exception as _emit_e:
        logger.warning(f"[P2-CRITICAL-CONFIG-ALERT] emisión best-effort falló: {_emit_e}")


@asynccontextmanager
async def lifespan(app: FastAPI):

    # [P2-AUDIT-NEW-3 · 2026-05-12] Marca el momento en que el app entra
    # en lifespan startup. El listener APScheduler usa este timestamp +
    # _SCHEDULER_BOOT_GRACE_S para suprimir el burst de
    # `scheduler_missed_*` que APScheduler dispara post-restart (jobs
    # cuyo next_run_time pasó durante el downtime).
    global _APP_START_TIME
    _APP_START_TIME = time.time()
    if _SCHEDULER_BOOT_GRACE_S > 0:
        logger.info(
            f"⏳ [P2-AUDIT-NEW-3] Boot grace activo: scheduler_missed_* "
            f"suprimidos durante los próximos {_SCHEDULER_BOOT_GRACE_S}s "
            f"(knob MEALFIT_SCHEDULER_BOOT_GRACE_MIN={_SCHEDULER_BOOT_GRACE_MIN})."
        )

    if connection_pool:
        connection_pool.open()
    if async_connection_pool:
        await async_connection_pool.open()
    # [P1-CHECKPOINT-POOL-SPLIT · 2026-05-20] Abrir el pool del LangGraph
    # checkpointer en startup junto a los demás. Sin esto, los callsites
    # de `PostgresSaver(chat_checkpoint_pool)` en agent.py fallan con
    # `psycopg_pool.PoolClosed: the pool is not open yet` al primer
    # `chat_graph_app.get_state(...)`.
    if chat_checkpoint_pool:
        chat_checkpoint_pool.open()
        
    # [P2-NEW-E · 2026-05-07] El bloque DDL runtime (CREATE TABLE IF NOT EXISTS
    # + ALTERs + UPDATEs de backfill + índices para 7 tablas) que vivía aquí se
    # consolidó al SSOT `migrations/p2_new_e_consolidate_runtime_ddl.sql`.
    # Mismo patrón estructural que P1-NEW-A 2026-05-08 cerró para índices:
    # cuando la DDL se recrea cada startup desde Python, un cambio de schema
    # vía SQL editor o migration nueva queda invisible al código y el siguiente
    # edit del bloque puede pisarlo en silencio.
    #
    # Lo único que sigue acá es `PostgresSaver(conn).setup()` — esa es API
    # pública de LangGraph (no DDL nuestro), debe correr en cada startup para
    # asegurar que el schema del checkpointer esté sincronizado con la
    # versión instalada de la librería.
    if connection_pool:
        try:
            import psycopg
            # [P1-SUPABASE-CLEANUP · 2026-06-13] Resuelve el URL session-mode de
            # Neon. Pre-fix leía una env var legada que apuntaba a un tenant
            # inexistente y el setup del checkpointer fallaba (capturado) en
            # CADA boot.
            db_uri = _build_session_mode_db_url()
            # autocommit=True requerido por LangGraph PostgresSaver (puede crear
            # índices CONCURRENTLY internamente).
            # [P3-PROD-AUDIT-2 · 2026-05-30] connect_timeout para no colgar el
            # startup hasta el TCP timeout del OS si la DB está inalcanzable.
            with psycopg.connect(db_uri, autocommit=True, connect_timeout=5) as conn:
                PostgresSaver(conn).setup()
            logger.info("🚀 [Postgres] LangGraph Checkpointer setup OK.")
        except Exception as e:
            logger.error(f"⚠️ [Postgres] Error en PostgresSaver(conn).setup(): {e}")
            
    # [P0-NEW-1-AUTOHEAL · 2026-05-11] Startup-time sweep de alerts
    # `scheduler_missed_*`/`scheduler_error_*` huérfanas. Cierra el
    # chicken-and-egg detectado en audit 2026-05-10: el cron periódico
    # `_resolve_stale_scheduler_alerts` (P0-AUDIT-1) está DENTRO del
    # scheduler que limpia — cuando el pool está saturado el sweep
    # también está MISSED y no corre. Resultado observado en prod:
    # alert `scheduler_missed_<uuid>` viva 31.75h pese a TTL=24h.
    #
    # Disparar aquí asegura UNA limpieza por deploy/restart antes de
    # que el scheduler entre en su próximo ciclo de jobs. El cron
    # periódico sigue corriendo como defense-in-depth (cuando NO está
    # MISSED, mantiene la backlog vacía entre deploys).
    #
    # Best-effort: cualquier fallo aquí NO debe abortar el startup
    # (sweep != startup-critical; las alerts solo afectan dashboards).
    try:
        from cron_tasks import _resolve_stale_scheduler_alerts
        _resolve_stale_scheduler_alerts()
        logger.info("🧹 [P0-NEW-1-AUTOHEAL] Startup sweep de scheduler alerts OK.")
    except Exception as _sweep_err:
        logger.warning(
            f"[P0-NEW-1-AUTOHEAL] Startup sweep falló (best-effort): {_sweep_err}"
        )

    # [P0-LIVE-2 · 2026-05-11] Startup-run del CB sweep (espejo P0-NEW-1-AUTOHEAL).
    # Cierra el mismo chicken-and-egg para CB rows stale: el cron periódico
    # `_sweep_stale_llm_circuit_breakers` (P2-NEW-D) está dentro del scheduler.
    # Cuando el pool está saturado, las filas con `is_open=true` viven indefinidamente
    # (audit 2026-05-11: `gemini-3.1-pro-preview` open=true por 4.4 días pese a
    # sweep diseñado para 2h staleness). Disparar aquí garantiza UNA limpieza por
    # deploy/restart ANTES de que APScheduler entre en su ciclo de jobs.
    #
    # El hard-floor asyncio (P0-LIVE-1, abajo) también invoca este sweep cada
    # 5min de forma continua — defense-in-depth.
    try:
        from cron_tasks import _sweep_stale_llm_circuit_breakers
        _cb_reset = _sweep_stale_llm_circuit_breakers()
        logger.info(
            f"🛡️ [P0-LIVE-2] Startup CB sweep OK (reset={_cb_reset} stale rows)."
        )
    except Exception as _cb_sweep_err:
        logger.warning(
            f"[P0-LIVE-2] Startup CB sweep falló (best-effort): {_cb_sweep_err}"
        )

    # [P0-PENDING-PIPELINE-STARTUP-SWEEP · 2026-05-16] Startup sweep de rows
    # `pending_pipeline:*` con `status='generating'`. Razón: cuando el backend
    # se reinicia (kill, crash, deploy), los pipelines en memoria mueren PERO
    # los rows del KV `app_kv_store` sobreviven. Eso deja rows ZOMBI que el
    # guardrail `check_user_has_active_pipeline` (max_age=15min) sigue
    # tratando como activos → bloquea al user con 409 `pipeline_already_running`
    # cuando intenta disparar otro plan.
    #
    # Escenario observado 2026-05-16: user inició plan A → apagó backend
    # mid-pipeline → levantó backend → disparó plan B → 409 (porque el row de
    # A seguía vivo en KV). El cron `_finalize_zombie_partial_plans` corre
    # cada 60min pero con `min_age=24h` — demasiado tarde para este caso.
    #
    # Fix: al startup, DELETE todos los rows `pending_pipeline:%` con
    # `status='generating'`. Cualquiera de esos rows es zombi por definición
    # (el proceso nuevo no tiene esos pipelines en memoria). Si un frontend
    # tenía un poll en flight, recibirá `status='none'` y limpiará su flag
    # local silenciosamente (no doble toast).
    try:
        # [P0-PENDING-PIPELINE-STARTUP-SWEEP fix · 2026-05-16] NO importar
        # `connection_pool` dentro de la función — Python lo trataría como
        # variable local y rompería el `if connection_pool:` de arriba con
        # UnboundLocalError. Reusamos el `connection_pool` ya importado al
        # tope del módulo (línea 181). Solo importamos `execute_sql_write`
        # local (no colisiona con nada).
        from db_core import execute_sql_write as _sweep_execute
        if connection_pool:
            # Pattern como param para evitar interpretación de `%` por psycopg.
            rows = _sweep_execute(
                "DELETE FROM app_kv_store WHERE key LIKE %s "
                "AND value->>'status' = 'generating' RETURNING key",
                ('pending_pipeline:%',),
                returning=True,
            )
            _swept = len(rows) if rows else 0
            logger.info(
                f"🧹 [P0-PENDING-PIPELINE-STARTUP-SWEEP] Sweep OK "
                f"(deleted={_swept} zombie rows from previous instance)."
            )
        else:
            logger.info(
                "[P0-PENDING-PIPELINE-STARTUP-SWEEP] connection_pool no disponible, sweep skipped."
            )
    except Exception as _pp_sweep_err:
        logger.warning(
            f"[P0-PENDING-PIPELINE-STARTUP-SWEEP] Sweep falló (best-effort): {_pp_sweep_err}"
        )

    # [P3-EMBED-CACHE-STARTUP-WARM · 2026-05-16] Pre-warm el semantic cache de
    # master_ingredients en background al startup. Razón: `get_semantic_cache()`
    # (shopping_calculator.py) hace LAZY init en el primer call dentro de
    # `get_shopping_list_delta`. Si Redis está frío (deploy nuevo / TTL expirado /
    # redis ausente), el primer call recalc-shopping-list bloquea ~35 batches ×
    # 3s delay = ~105s + 429s de Gemini → browser fetch timeout → 500 + CORS
    # error al usuario.
    #
    # Escenario observado 2026-05-16 plan 4cc91584: user hizo click en duration
    # → embed cache init disparó en el endpoint, blockeó >100s, frontend timeout.
    #
    # Fix: warming en background thread (NO awaitar para no bloquear lifespan).
    # Si Redis tiene los vectores cacheados → load instantáneo, thread termina
    # en ms. Si Redis vacío → Gemini call demora hasta ~100s pero NO bloquea
    # requests del usuario. La primera request de recalc puede caer al regex
    # fast-path mientras el cache se calienta — degradación graceful documentada
    # por P6-SEMANTIC-SKIP.
    #
    # Best-effort: si la init falla (Gemini quota / red), `_semantic_cache_failed_until`
    # entra en cooldown y los requests caen al fast-path sin error.
    try:
        import threading
        def _warm_semantic_cache_bg():
            try:
                from shopping_calculator import get_semantic_cache
                _cache = get_semantic_cache()
                if _cache is not None:
                    logger.info(
                        "🔥 [P3-EMBED-CACHE-STARTUP-WARM] Semantic cache warmed "
                        f"({len(_cache.get('vectors', []))} vectors) — recalc endpoints unblocked."
                    )
                else:
                    logger.info(
                        "🟡 [P3-EMBED-CACHE-STARTUP-WARM] Semantic cache no-op "
                        "(disabled o cooldown). Recalc usará regex fast-path."
                    )
            except Exception as _warm_err:
                logger.warning(
                    f"[P3-EMBED-CACHE-STARTUP-WARM] Warmer falló (best-effort, "
                    f"recalc fallback al regex): {type(_warm_err).__name__}: {_warm_err}"
                )
        _t = threading.Thread(target=_warm_semantic_cache_bg, name="embed-cache-warmer", daemon=True)
        _t.start()
        logger.info("🔥 [P3-EMBED-CACHE-STARTUP-WARM] Background warmer launched (daemon thread).")
    except Exception as _spawn_err:
        logger.warning(
            f"[P3-EMBED-CACHE-STARTUP-WARM] No se pudo lanzar el warmer "
            f"(best-effort): {_spawn_err}"
        )

    # [P1-SCHEDULER-LEADER-LOCK · 2026-05-27] Leader election ANTES de arrancar
    # los dos loops continuos (hard-floor + APScheduler). Sólo el leader los
    # ejecuta; en deploy multi-worker accidental los demás workers sirven HTTP
    # sin duplicar crons. Fail-open (ver helper). La conn se cierra en shutdown.
    _is_scheduler_leader, _leader_conn = _acquire_scheduler_leader_lock()
    app.state.is_scheduler_leader = _is_scheduler_leader
    app.state.scheduler_leader_conn = _leader_conn

    # [P0-LIVE-1 · 2026-05-11] Hard-floor autoheal asyncio task. Garantiza
    # sweeps de scheduler alerts + CB rows incluso si APScheduler está
    # saturado (corre en el event loop, no en el ThreadPoolExecutor del
    # scheduler). Guardado en app.state para cancelarlo en shutdown.
    if not _is_scheduler_leader:
        # [P1-SCHEDULER-LEADER-LOCK] Worker no-leader: no corre hard-floor.
        app.state.hardfloor_autoheal_task = None
        logger.info(
            "🛟 [P1-SCHEDULER-LEADER-LOCK] Hard-floor NO iniciado en este worker "
            "(no-leader). Sólo el worker leader ejecuta sweeps/crons."
        )
    elif _env_bool("MEALFIT_HARDFLOOR_AUTOHEAL_ENABLED", True):
        _hardfloor_interval = _env_int("MEALFIT_HARDFLOOR_AUTOHEAL_INTERVAL_S", 300)
        _hardfloor_interval = max(60, min(_hardfloor_interval, 1800))
        app.state.hardfloor_autoheal_task = asyncio.create_task(
            _hardfloor_autoheal_loop(_hardfloor_interval)
        )
        logger.info(
            f"🛟 [P0-LIVE-1] Hard-floor autoheal asyncio task iniciado "
            f"(cada {_hardfloor_interval}s, independiente de APScheduler)."
        )
    else:
        app.state.hardfloor_autoheal_task = None
        logger.info(
            "🛟 [P0-LIVE-1] Hard-floor autoheal DESACTIVADO via knob "
            "MEALFIT_HARDFLOOR_AUTOHEAL_ENABLED."
        )

    if not _is_scheduler_leader and HAS_SCHEDULER and scheduler:
        # [P1-SCHEDULER-LEADER-LOCK] Worker no-leader: NO registra ni arranca el
        # scheduler. Las probes de startup posteriores (warm_plan_graph, schema
        # checks) sí corren — este worker sirve HTTP normalmente.
        logger.warning(
            "⏰ [P1-SCHEDULER-LEADER-LOCK] APScheduler NO iniciado en este worker "
            "(no-leader). Crons/chunking los ejecuta el worker leader."
        )

    if _is_scheduler_leader and HAS_SCHEDULER and scheduler:
        # [P0-NEW-2 · 2026-05-10] `run_proactive_checks` también pasa por el
        # wrapper SSOT de jitter. Knob `MEALFIT_SCHEDULER_JITTER_S` controla
        # el spread (default 45s). CronTrigger acepta `jitter` igual que
        # IntervalTrigger.
        from cron_tasks import _add_job_jittered, register_plan_chunk_scheduler
        _add_job_jittered(scheduler, run_proactive_checks, "cron", minute=30)
        # [P2-NEW-C · 2026-05-08] `background_rolling_refill` se movió al SSOT
        # `register_plan_chunk_scheduler` (cron_tasks.py) junto al resto del
        # chunk system. Ya no se registra acá.
        register_plan_chunk_scheduler(scheduler)
        # [P2-CRITICAL-CONFIG-ALERT · 2026-06-15] (gap-audit G7+G10) Tras registrar crons, alerta si la
        # config crítica está mal seteada en producción (motor de precisión OFF / guard de seguridad OFF).
        # Best-effort, no bloquea el arranque (los knobs son estáticos → un check al arranque basta).
        try:
            _emit_critical_config_alerts()
        except Exception as _cfg_alert_e:
            logger.warning(f"[P2-CRITICAL-CONFIG-ALERT] startup check no-op: {_cfg_alert_e}")
        # [P2-NEW-F · 2026-05-08] Registrar listener ANTES de start() para no
        # perder los primeros eventos. Mask combinado MISSED|ERROR. Si la
        # telemetría está desactivada por knob, el listener corto-circuita
        # internamente (no skipeamos el add_listener para no requerir
        # restart al togglear el knob — solo edita la env var y los próximos
        # eventos se procesan según el flag actual).
        #
        # [P1-NEW-2 · 2026-05-10] Mask extendido con EVENT_JOB_EXECUTED para
        # auto-resolver `scheduler_missed_<job>` y `scheduler_error_<job>`
        # cuando el mismo job ejecuta exitosamente después (handler interno
        # en _scheduler_alert_listener).
        try:
            scheduler.add_listener(
                _scheduler_alert_listener,
                EVENT_JOB_MISSED | EVENT_JOB_ERROR | EVENT_JOB_EXECUTED,
            )
            logger.info(
                f"📡 [P2-NEW-F] Listener APScheduler registrado "
                f"(telemetry={'on' if _is_scheduler_telemetry_enabled() else 'off'}, "
                f"workers={_SCHEDULER_MAX_WORKERS}, "
                f"misfire_grace={_SCHEDULER_MISFIRE_GRACE_S}s)."
            )
        except Exception as _listener_err:
            logger.error(
                f"⚠️ [P2-NEW-F] No se pudo registrar listener APScheduler: "
                f"{_listener_err}. Continuando sin telemetría."
            )
        scheduler.start()
        logger.info("⏰ [APScheduler] Tareas proactivas, CRON jobs nocturnos y Background Chunking iniciados.")

        # [P1-PERF-1 · 2026-05-12] Cold-cache de alerts abiertas. Sin este
        # refresh, el primer ciclo post-restart saltaría PATCHes EXECUTED
        # que SÍ deberían ocurrir (alerts persistentes de antes del boot).
        try:
            n = _refresh_scheduler_open_alerts_cache(force=True)
            logger.info(
                f"📥 [P1-PERF-1] Cache `_SCHEDULER_JOBS_WITH_OPEN_ALERTS` "
                f"populado con {n} entradas (cold start)."
            )
        except Exception as _cache_err:
            logger.debug(
                f"[P1-PERF-1] cold cache refresh falló (best-effort): {_cache_err}"
            )
            
    # P1-Q2: warm-up del grafo LangGraph antes de empezar a aceptar tráfico.
    # Mueve la latencia de compile (~ms-cientos-ms) fuera del path de la
    # primera request real. Si el build falla, `warm_plan_graph()` loguea
    # CRITICAL internamente y devuelve False — NO derribamos el startup
    # (las requests caen al fallback matemático vía P0-1, mejor que tener
    # el pod en CrashLoopBackOff). El endpoint `/ready` reflejará el estado
    # real para que el orquestador (Kubernetes / load balancer) decida si
    # enrutar tráfico.
    if warm_plan_graph():
        logger.info("✅ [STARTUP] LangGraph plan_graph pre-compilado.")
    else:
        logger.critical(
            "🚨 [STARTUP] LangGraph plan_graph NO se pudo compilar. "
            "El servidor arranca igual; las requests caerán al fallback "
            "matemático (P0-1) hasta que el problema se resuelva. "
            "Revisa los logs CRITICAL anteriores para el traceback."
        )

    # P1-Q10: Probe de schema de pipeline_metrics.user_id (NULL allowed).
    # La migración del bloque anterior aplica `ALTER COLUMN user_id DROP NOT
    # NULL` idempotente. Este probe verifica el resultado real haciendo un
    # INSERT/DELETE de prueba — si falla, deja `_GUEST_METRICS_ENABLED=False`
    # internamente para que los emitters skipeen guest inserts en lugar de
    # fallar 50× por pipeline. CRITICAL log con remediation steps explícitos
    # si detecta drift entre schema deseado y schema real.
    if verify_pipeline_metrics_guest_insert():
        logger.info("✅ [STARTUP] pipeline_metrics schema OK (guest inserts habilitados).")
    # else: el probe ya logueó CRITICAL con remediation; no spam adicional.

    # [P1-ORQ-5] Observabilidad: contar planes con `CACHE_SCHEMA_VERSION`
    # obsoleta. Antes, bumpear la versión (ej. v1→v2) invalidaba todos los
    # planes pre-deploy de forma silenciosa: `semantic_cache_check_node` los
    # descartaba post-filter pero los planes seguían en `meal_plans`
    # consumiendo slots del vector search → cache hit rate caía sin que
    # operadores pudieran correlacionar con el cambio. Ahora el deploy log
    # muestra explícitamente "N planes con versión obsoleta" para que el
    # equipo decida si necesita correr cleanup manual o ajustar el limit del
    # vector search. NO bloqueante: si el probe falla, el startup continúa
    # (es observabilidad pura, no salud del sistema). NO automático: deletes
    # masivos sobre `meal_plans` perderían historia válida del usuario
    # (cache version solo afecta REUSO, no validez del registro).
    try:
        from db_plans import count_stale_cache_schema_plans
        from graph_orchestrator import CACHE_SCHEMA_VERSION, _LEGACY_CACHE_SCHEMA_VERSION
        _stale_summary = count_stale_cache_schema_plans(CACHE_SCHEMA_VERSION, _LEGACY_CACHE_SCHEMA_VERSION)
        _stale_count = _stale_summary.get("stale_count", 0)
        _total = _stale_summary.get("total", 0)
        if _stale_count > 0:
            logger.warning(
                f"🟠 [P1-ORQ-5] CACHE_SCHEMA_VERSION='{CACHE_SCHEMA_VERSION}' actual; "
                f"{_stale_count}/{_total} planes en `meal_plans` tienen versión "
                f"obsoleta y serán descartados por el cache semántico post-filter "
                f"(distribución: {_stale_summary.get('stale_versions', {})}). "
                f"El cache hit rate puede caer hasta que los planes nuevos los "
                f"reemplacen orgánicamente. Si el bump es intencional y operativo, "
                f"considerar bumpear el `limit` de `search_similar_plan` "
                f"temporalmente (graph_orchestrator.py:~6430) para compensar."
            )
        elif _total > 0:
            logger.info(
                f"✅ [STARTUP] Cache schema OK: {_total} planes alineados con "
                f"CACHE_SCHEMA_VERSION='{CACHE_SCHEMA_VERSION}'."
            )
    except Exception as _stale_err:
        logger.warning(f"⚠️ [P1-ORQ-5] Probe de stale cache schema falló: {_stale_err}")

    # [P1-4] Health-check del connection_pool requerido por
    # `update_user_health_profile_atomic`. El atomic helper degrada
    # silenciosamente a non-atómico si el pool no está; en producción eso
    # significa lost-update bajo concurrencia (signals erosionados de
    # `frictions`, `weight_history`, `reflection_history`,
    # `lifetime_lessons_history`, etc.) que pueden vivir días sin detección.
    #
    # Comportamiento:
    #   - Pool OK → log INFO informativo, continúa.
    #   - Pool NO disponible + `MEALFIT_REQUIRE_ATOMIC_POOL=1` → loguea
    #     CRITICAL y RAISE para que el orquestador (uvicorn/gunicorn/k8s)
    #     reinicie el worker. Producción debe fijar este env var.
    #   - Pool NO disponible + strict OFF (default) → CRITICAL log con
    #     remediation, pero el servidor sigue arriba (preserva dev/scripts
    #     locales sin DATABASE_URL). El counter de
    #     `get_atomic_pool_fallback_snapshot()` empieza a llenarse y el
    #     endpoint `/api/system/atomic-pool-health` queda como fuente de
    #     verdad para alerting.
    try:
        from db_profiles import REQUIRE_ATOMIC_POOL
        from db_core import connection_pool as _pool_check
        if _pool_check is None:
            _msg = (
                "[P1-4/STARTUP] connection_pool=None — "
                "update_user_health_profile_atomic degradará a non-atómico. "
                "Lost-update bajo concurrencia es posible. Verificar "
                "DATABASE_URL/NEON_DATABASE_URL y conectividad al pooler:6543."
            )
            if REQUIRE_ATOMIC_POOL:
                logger.critical(
                    f"🚨 {_msg} MEALFIT_REQUIRE_ATOMIC_POOL=1 → abortando startup."
                )
                raise RuntimeError(_msg)
            logger.critical(
                f"🚨 {_msg} Continuando porque MEALFIT_REQUIRE_ATOMIC_POOL≠1; "
                f"export=1 en producción para fail-fast."
            )
        else:
            logger.info(
                f"✅ [STARTUP] connection_pool inicializado — "
                f"update_user_health_profile_atomic operará en modo atómico real "
                f"(strict={REQUIRE_ATOMIC_POOL})."
            )
    except RuntimeError:
        raise  # propagar para que el process manager reinicie
    except Exception as _pool_probe_err:
        # Probe no debería fallar; si lo hace, NO bloqueamos startup —
        # observabilidad pura. El endpoint /api/system/atomic-pool-health
        # captura el estado real cuando lo consulten.
        logger.warning(
            f"⚠️ [P1-4] Probe de pool falló (no fatal): {_pool_probe_err}"
        )

    logger.info("🚀 [FastAPI] Servidor de MealfitRD IA iniciado con éxito en el puerto 3001.")
    yield

    # [P0-LIVE-1 · 2026-05-11] Cancelar hard-floor autoheal task ANTES de
    # cerrar pools/scheduler. Si `cancel()` deja el task con `await
    # asyncio.sleep(...)`, propaga `CancelledError` y el `try/except` en
    # el loop lo loguea limpio.
    _hardfloor_task = getattr(app.state, "hardfloor_autoheal_task", None)
    if _hardfloor_task is not None and not _hardfloor_task.done():
        _hardfloor_task.cancel()
        try:
            await _hardfloor_task
        except asyncio.CancelledError:
            pass
        except Exception as _cancel_err:
            logger.debug(
                f"[P0-LIVE-1] Cancelación de hard-floor task lanzó: {_cancel_err}"
            )

    if HAS_SCHEDULER and scheduler:
        # [P2-OPS-SHUTDOWN · 2026-05-27] `wait=False`: deja de despachar jobs
        # inmediatamente sin BLOQUEAR el teardown esperando a que terminen los
        # jobs en curso. Pre-fix `scheduler.shutdown()` defaultea a `wait=True`
        # → con `_chunk_worker` mid-LLM-call (multi-minuto) el shutdown podía
        # colgarse más allá del kill-grace del VPS Oracle → SIGKILL. (Los writes
        # del worker son data-safe: rollback de transacción + CHECK I8; lo que
        # se evita acá es el hang del deploy y el trabajo LLM desperdiciado.)
        scheduler.shutdown(wait=False)

    # [P2-OPS-SHUTDOWN · 2026-05-27] Drenar el pool de bg_executor (summarize/
    # facts fire-and-forget) — deja de aceptar submits + descarta encolados sin
    # bloquear. Sin esto el pool quedaba sin teardown explícito.
    try:
        from bg_executor import shutdown_bg_executor
        shutdown_bg_executor(wait=False)
    except Exception as _bg_sd_err:
        logger.debug(f"[P2-OPS-SHUTDOWN] shutdown_bg_executor falló: {_bg_sd_err}")

    # [P1-SCHEDULER-LEADER-LOCK · 2026-05-27] Cerrar la conn que sostiene el
    # advisory lock de sesión → lo libera para que el próximo proceso/worker
    # pueda re-adquirirlo. Best-effort: un fallo aquí no debe romper shutdown.
    _leader_conn = getattr(app.state, "scheduler_leader_conn", None)
    if _leader_conn is not None:
        try:
            _leader_conn.close()
            logger.info(
                "👑 [P1-SCHEDULER-LEADER-LOCK] Leader lock liberado (conn cerrada)."
            )
        except Exception as _llc_err:
            logger.debug(
                f"[P1-SCHEDULER-LEADER-LOCK] cierre de leader conn falló: {_llc_err}"
            )

    if connection_pool:
        connection_pool.close()
        logger.info("🔌 [psycopg] Pool de conexiones cerrado.")
    if async_connection_pool:
        await async_connection_pool.close()
        logger.info("🔌 [psycopg] Pool de conexiones asíncronas cerrado.")
    # [P3-PROD-AUDIT-2 · 2026-05-30] Cerrar también el pool del LangGraph
    # checkpointer (asimetría open/close: se abría en startup @986 pero el
    # teardown inline lo omitía; `close_connection_pool()` que lo cubría es
    # dead code sin callsites). Best-effort: con min_size=0 el impacto es nulo
    # (proceso muere → conns liberadas), pero cerramos por higiene/consistencia.
    if chat_checkpoint_pool:
        try:
            chat_checkpoint_pool.close()
            logger.info("🔌 [psycopg] chat_checkpoint_pool cerrado.")
        except Exception as _ccp_err:
            logger.debug(f"[P3-PROD-AUDIT-2] cierre de chat_checkpoint_pool falló: {_ccp_err}")


# Asegurarnos de que el directorio de uploads exista antes de montar recursos estáticos
os.makedirs("uploads", exist_ok=True)


# [P2-DOCS-GATE · 2026-05-27] Ocultar /docs, /redoc y /openapi.json en
# producción. Pre-fix quedaban públicos → exponían el mapa completo de la API
# (todos los endpoints + schemas request/response) a cualquiera, facilitando
# reconnaissance. En dev se mantienen para DX. Gate por ENVIRONMENT (misma var
# que usa el fail-secure de billing/webhook). Rollback: si necesitas /docs en
# prod temporalmente, dejar ENVIRONMENT != "production". Tooltip-anchor: P2-DOCS-GATE.
_IS_PRODUCTION = is_production()  # [P2-PROD-AUDIT-3] SSOT normalizado (lower+strip)
app = FastAPI(
    lifespan=lifespan,
    docs_url=None if _IS_PRODUCTION else "/docs",
    redoc_url=None if _IS_PRODUCTION else "/redoc",
    openapi_url=None if _IS_PRODUCTION else "/openapi.json",
)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

from routers.billing import router as billing_router, webhooks_router, discount_router
from routers.notifications import router as notifications_router

app.include_router(billing_router)
app.include_router(webhooks_router)
app.include_router(discount_router)
app.include_router(notifications_router)
from routers.plans import router as plans_router
app.include_router(plans_router)
from routers.chat import router as chat_router
app.include_router(chat_router)
from routers.diary import router as diary_router
app.include_router(diary_router)
from routers.system import router as system_router
app.include_router(system_router)
# [LONG-TERM-MEMORY-TOGGLE · 2026-05-13] Endpoints para el toggle de memoria
# a largo plazo (Settings del Dashboard). Migración SSOT:
# migrations/add_long_term_memory_enabled_2026_05_13.sql
from routers.preferences import router as preferences_router
app.include_router(preferences_router)

# [P1-NEON-DB-MIGRATION · 2026-06-12] Datos user-scoped que el frontend
# accedía directo via PostgREST (inventario, perfil, lecturas de planes).
# Post-Neon el cliente no tiene acceso a la DB — todo pasa por aquí.
from routers.user_data import router as user_data_router
app.include_router(user_data_router)
# [P1-FIRST-PARTY-SESSION · 2026-06-16] Cookie de sesión first-party
# (__Host-mf_session) para que iOS PWA conserve la sesión al cerrar la app
# (Neon Auth sirve su cookie third-party → iOS la borra al cerrar).
from routers.auth_session import router as auth_session_router
app.include_router(auth_session_router)
# [P1-SUPERMARKET-DB · 2026-07-02] Supermercado RD artificial: catálogo de
# presentaciones comprables (+variantes de marca) navegable desde el landing
# (/supermercado) y editable con gate admin (Bearer CRON_SECRET).
from routers.supermarket import router as supermarket_router
app.include_router(supermarket_router)
# [P2-HELP-CHATBOT · 2026-07-04] Chatbot de ayuda del menú "Obtener ayuda":
# Q&A de producto sin acceso a datos del usuario (cero tools/DB), fail-cheap
# (flash), quota-exempt (RateLimiter, NO verify_api_quota/log_api_usage).
from routers.help_chat import router as help_chat_router
app.include_router(help_chat_router)

@app.get("/")
@app.get("/health")
def health_check():
    """Liveness probe: el proceso está vivo y atendiendo HTTP. NO valida que
    los componentes downstream (LangGraph, DB, Redis) estén operativos —
    para eso usar `/ready`."""
    return {"status": "ok", "message": "MealfitRD AI Backend is running"}


@app.get("/ready")
def readiness_check():
    """P1-Q2: Readiness probe para orquestadores (Kubernetes, load balancer).

    Devuelve 200 solo si el grafo LangGraph está compilado y listo para
    servir. Si el build inicial falló, `warm_plan_graph()` lo intentó al
    startup y `is_plan_graph_ready()` retorna False; el orquestador NO
    enruta tráfico y reintenta el probe periódicamente. Si el grafo se
    compila exitosamente en una request posterior (vía `_get_plan_graph()`
    lazy), el probe pasa a 200 automáticamente sin restart del pod.

    Diferencia con `/health`:
      - `/health` (liveness): el proceso está vivo. Si falla, K8s reinicia.
      - `/ready`  (readiness): el servicio puede servir requests útiles. Si
                                falla, K8s deja el pod corriendo pero quita
                                del load balancer hasta que se recupere.

    [P3-READY-REASON · 2026-05-12] El 503 ahora incluye `reason` granular
    con el tipo de excepción + mensaje (truncado a 240 chars) + count de
    intentos. Pre-fix devolvía solo `not_compiled` sin pista; el operador
    debía abrir logs separadamente. Formato del reason:
      - `uninitialized`: nunca se intentó (raro — warm_plan_graph debió
        correr en lifespan).
      - `build_failed:<ExcType>:<msg>:<n>`: build crasheó N veces; el
        operador ve inmediato si es TimeoutError (problema DB / red),
        ImportError (deploy roto / dep faltante), KeyError (config), etc.
    Tooltip-anchor: P3-READY-REASON.
    """
    ready, reason = is_plan_graph_ready_with_reason()

    # [P2-READY-DB-CHECK · 2026-05-28] Validación opcional de conectividad a DB.
    # Pre-fix `/ready` solo miraba el grafo LangGraph; un pod con el grafo
    # compilado pero el connection_pool muerto (Supavisor caído, credenciales
    # rotadas) reportaba "ready" y el LB le enrutaba tráfico que fallaba en cada
    # query. TOLERANTE por diseño (evita flapping del LB ante un blip):
    #   - SELECT 1 con timeout corto (2s al tomar conn del pool), best-effort.
    #   - Por DEFAULT solo se REPORTA en el body (`db`), sin cambiar 200/503.
    #   - Con `MEALFIT_READY_REQUIRE_DB=true`, un fallo de DB escala a 503.
    # db_ok: True=ok, False=fallo, None=pool no inicializado (no afirmar fallo).
    # Tooltip-anchor: P2-READY-DB-CHECK.
    db_ok: Optional[bool] = None
    try:
        from knobs import _env_bool
        _require_db = _env_bool("MEALFIT_READY_REQUIRE_DB", False)
    except Exception:
        _require_db = False
    try:
        from db_core import connection_pool as _ready_pool
        if _ready_pool is not None:
            with _ready_pool.connection(timeout=2) as _ready_conn:
                with _ready_conn.cursor() as _ready_cur:
                    _ready_cur.execute("SELECT 1")
                    _ready_cur.fetchone()
            db_ok = True
    except Exception:
        db_ok = False

    _db_blocks = bool(_require_db and db_ok is False)
    if ready and not _db_blocks:
        return {"status": "ready", "plan_graph": "compiled", "db": db_ok}
    if ready and _db_blocks:
        reason = "db_unreachable"

    # [P3-READY-REASON-HASH · 2026-05-15] Hash determinístico de los primeros
    # 8 chars de SHA-256(reason) para correlación cross-fleet. ANTES, dos
    # workers/regiones reportando el mismo `reason` truncado a 240 chars
    # eran indistinguibles en agregados sin SSH a logs. Con el hash, un
    # operador puede grep "reason_hash=abc123de" en su consola unificada y
    # encontrar todos los incidents idénticos. Hash sobre el reason TAL
    # CUAL se reporta (post-trunc); operadores que vean reasons distintos
    # pero hashes iguales aún pueden cross-correlate.
    import hashlib as _hashlib_ready
    reason_hash = None
    if reason:
        try:
            reason_hash = _hashlib_ready.sha256(reason.encode("utf-8")).hexdigest()[:8]
        except Exception:
            reason_hash = None

    raise HTTPException(
        status_code=503,
        detail={
            "status": "not_ready",
            "plan_graph": "compiled" if ready else "not_compiled",
            "db": db_ok,
            "reason": reason,
            "reason_hash": reason_hash,
            "message": (
                "LangGraph plan_graph no está compilado. Las requests al "
                "pipeline de generación caerán al fallback matemático. "
                "Revisa los logs CRITICAL del worker para el traceback "
                "(o el campo `reason` para el último error de build). "
                "`reason_hash` permite correlación cross-fleet."
            ),
        },
    )


@app.get("/health/version")
def health_version():
    """[P1-A · 2026-05-08] Diagnóstico del binary desplegado.

    Expone el commit hash, marker del último P-fix mergeado y un snapshot
    del registry de knobs `MEALFIT_*`. Permite verificar si el deploy en
    producción está rezagado respecto al árbol cuando los logs Postgres
    muestran statements de runtime DDL ya consolidados a SSOT migrations
    (CREATE TABLE system_alerts, ALTER user_profiles ADD quality_alert_at).

    Comparación esperada:
      - `last_known_pfix` debe coincidir con `_LAST_KNOWN_PFIX` en HEAD.
      - `git_sha` (inyectado por el deploy del VPS Oracle via env var) debe
        coincidir con el SHA que el repo apunta como producción.
      - `knobs_count` debe ser ≥ al número de knobs registrados localmente
        (vía `python -c "from graph_orchestrator import get_knobs_registry_snapshot; print(len(get_knobs_registry_snapshot()))"`).

    Si no coincide → redeploy. Si coincide y aún ves DDL en logs Postgres,
    el origen es un cron externo invocando un script `migrate_*.py`.

    [P3-2 · 2026-05-10] Extensiones añadidas para diagnóstico 1-segundo:
      - `process_uptime_s`: segundos desde el arranque del worker.
        Si <60s + cron MISSED en cascada → causa raíz es restart frecuente.
      - `knobs_diff`: dict `{name: {default, value}}` solo para knobs
        cuyo valor activo difiere del default. Huella operacional del deploy.
      - `cron_missed_1h_total`: count de MISSED de scheduler en la última
        hora. Para detalle por job, ver `/admin/cron-health`.

    Para registry completo de knobs, ver `/admin/knobs` (P3-5).

    [P2-HEALTHZ-DEEP · 2026-05-12] 5 keys adicionales para blackbox monitor
    (UptimeRobot/cronitor) que no tiene CRON_SECRET — cierra el modo de
    fallo "binary pre-watchdog corriendo + watchdog interno dormido"
    detectado en audit production-readiness 2026-05-12:
      - `expected_marker`: marker publicado en `app_kv_store.expected_last_known_pfix`.
      - `drift`: bool `(expected_marker != _LAST_KNOWN_PFIX)`. None si KV unreachable.
      - `last_pipeline_metrics_tick_at`: ISO timestamp del último tick de
         `_hardfloor_autoheal_tick` (P0-LIVE-1). Heartbeat del binary.
      - `has_p0_prod_1_gate`: bool — `_is_guest_metrics_enabled` importable
         desde graph_orchestrator. Si False → binary PRE-P0-PROD-1 (errores
         `is_guest` siguen lloviendo).
      - `has_p1_perf_1_cache`: bool — `_SCHEDULER_JOBS_WITH_OPEN_ALERTS` en
         globals de app.py. Si False → binary PRE-P1-PERF-1 (PATCH spam).

    SOP UptimeRobot:
      - URL: `https://<base>/health/version`
      - Method: GET
      - Assertion: `drift = false AND last_pipeline_metrics_tick_at <
        NOW() - 30min` (cron `_hardfloor_autoheal_loop` tickea cada 5 min;
        30 min ventana cubre 6 ticks perdidos antes de alertar).
      - Frecuencia: 5 min (alineado con loop interval).

    No requiere auth: información de diagnóstico no sensible.
    """
    # [P3-2 · 2026-05-10] knobs_diff: cuáles knobs MEALFIT_* tienen value
    # != default. Es la "huella operacional" del proceso — útil para
    # responder "¿qué env vars están activas en este deploy?" sin shell.
    knobs_diff: dict = {}
    try:
        from graph_orchestrator import get_knobs_registry_snapshot
        snap = get_knobs_registry_snapshot()
        knobs_count = len(snap)
        knobs_sample = sorted(snap.keys())[:8]
        for name, info in snap.items():
            try:
                if info.get("value") != info.get("default"):
                    knobs_diff[name] = {
                        "default": info.get("default"),
                        "value": info.get("value"),
                    }
            except Exception:
                # Comparación de tipos exóticos (e.g., numpy floats); skip.
                continue
    except Exception as e:
        knobs_count = -1
        knobs_sample = [f"error:{type(e).__name__}"]

    # [P3-2 · 2026-05-10] process_uptime_s: segundos desde el arranque
    # del worker actual. Si el cron P0-2 reporta MISSED en cascada y el
    # uptime es <60s, la causa raíz es restart frecuente (cold-start
    # procesa el queue de eventos missed), no saturación de pool.
    try:
        from datetime import datetime as _dt, timezone as _tz
        start = _dt.fromisoformat(_PROCESS_START_ISO)
        uptime_s = (_dt.now(_tz.utc) - start).total_seconds()
        process_uptime_s = round(uptime_s, 1)
    except Exception:
        process_uptime_s = -1

    # [P3-2 · 2026-05-10] cron_missed_1h_total: suma de MISSED en última
    # hora desde system_alerts. Detalle por job en /admin/cron-health.
    cron_missed_1h_total = 0
    try:
        if connection_pool is not None:
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            cutoff = _dt.now(_tz.utc) - _td(hours=1)
            # [P1-NEON-DB-MIGRATION · 2026-06-12] count-only via SQL directo
            # (antes PostgREST count='exact' + limit(0)).
            count_row = execute_sql_query(
                """
                SELECT count(*) AS count
                FROM system_alerts
                WHERE alert_type = 'scheduler'
                  AND alert_key LIKE %s
                  AND triggered_at >= %s
                """,
                ("scheduler_missed_%", cutoff),
                fetch_one=True,
            )
            cron_missed_1h_total = int(count_row["count"]) if count_row else 0
    except Exception:
        cron_missed_1h_total = -1

    # [P2-HEALTHZ-DEEP · 2026-05-12] 5 keys adicionales para blackbox monitor
    # externo (UptimeRobot/StatusCake/cronitor). Permite assertion remota
    # SIN auth/CRON_SECRET — cierra el modo de fallo "binary pre-watchdog
    # corriendo + watchdog interno dormido" (audit production-readiness
    # 2026-05-12, sección P2-HEALTHZ-DEEP). Cada lectura es best-effort:
    # cualquier excepción → la key respectiva queda en None/False sin
    # fallar el endpoint completo (debe seguir respondiendo 200 para que
    # los pollers externos distingan "binary alive" de "binary down").
    expected_marker: Optional[str] = None
    drift: Optional[bool] = None
    try:
        if connection_pool is not None:
            kv_row = execute_sql_query(
                "SELECT value FROM app_kv_store WHERE key = %s LIMIT 1",
                ("expected_last_known_pfix",),
                fetch_one=True,
            )
            if kv_row:
                raw = kv_row.get("value")
                # value es jsonb; psycopg deserializa a str/list/dict.
                # Aceptamos str directo o str dentro de jsonb-string ("...").
                expected_marker = raw if isinstance(raw, str) else None
                if expected_marker is not None:
                    drift = (expected_marker != _LAST_KNOWN_PFIX)
    except Exception:
        # Best-effort: dejar expected_marker=None / drift=None.
        # El poller externo debería tratar drift=None como "unknown" (no
        # disparar alert), igual que drift=true (alert).
        pass

    last_pipeline_metrics_tick_at: Optional[str] = None
    try:
        if connection_pool is not None:
            tick_row = execute_sql_query(
                """
                SELECT created_at
                FROM pipeline_metrics
                WHERE node = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                ("_hardfloor_autoheal_tick",),
                fetch_one=True,
            )
            if tick_row and tick_row.get("created_at") is not None:
                _tick_raw = tick_row["created_at"]
                # [P1-NEON-DB-MIGRATION · 2026-06-12] psycopg devuelve
                # datetime (PostgREST devolvía string ISO); preservar el
                # contrato string del poller externo con .isoformat().
                last_pipeline_metrics_tick_at = (
                    _tick_raw.isoformat()
                    if hasattr(_tick_raw, "isoformat")
                    else str(_tick_raw)
                )
    except Exception:
        pass

    # has_p0_prod_1_gate: el gate _is_guest_metrics_enabled
    # (graph_orchestrator.py:10324, P1-Q10/P0-PROD-1) DEBE existir en el
    # binary corriendo. Si False → binary es PRE-P0-PROD-1 → INSERT a
    # pipeline_metrics seguirán crasheando con `is_guest` errors.
    has_p0_prod_1_gate: bool = False
    try:
        from graph_orchestrator import _is_guest_metrics_enabled  # noqa: F401
        has_p0_prod_1_gate = True
    except Exception:
        has_p0_prod_1_gate = False

    # has_p1_perf_1_cache: el cache _SCHEDULER_JOBS_WITH_OPEN_ALERTS
    # (app.py:189, P1-PERF-1) DEBE existir en el módulo cargado. Si
    # False → binary PRE-P1-PERF-1 → spam PATCH /system_alerts cada
    # job EXECUTED.
    has_p1_perf_1_cache: bool = (
        "_SCHEDULER_JOBS_WITH_OPEN_ALERTS" in globals()
    )

    # [P2-EMBED-TELEMETRY · 2026-05-24] Gauge del tamaño actual de los
    # embedding caches bounded (cierre del par natural de P1-EMBEDDING-CACHE-
    # BOUNDED 2026-05-24). Sin estas keys, un operador no puede detectar
    # saturación del cache antes de que el `popitem(last=False)` esté
    # evicting entries útiles continuamente. Si `embedding_cache_size`
    # se queda topado en `embedding_cache_maxsize` durante >24h, vale la
    # pena subir `MEALFIT_EMBEDDING_CACHE_MAXSIZE` para reducir misses.
    #
    # Best-effort: cualquier excepción → keys con `-1` para no romper el
    # endpoint completo (debe seguir respondiendo 200 al poller externo).
    # Tooltip-anchor: P2-EMBED-TELEMETRY.
    embedding_cache_size: int = -1
    pantry_embeddings_cache_size: int = -1
    embedding_cache_maxsize: int = -1
    try:
        from constants import (
            _embedding_cache as _ec,
            _pantry_embeddings_cache as _pec,
            _EMBEDDING_CACHE_MAXSIZE as _ec_max,
        )
        embedding_cache_size = len(_ec)
        pantry_embeddings_cache_size = len(_pec)
        embedding_cache_maxsize = int(_ec_max)
    except Exception:
        pass

    git_sha = os.environ.get("GIT_SHA") or "unknown"
    return {
        "git_sha": git_sha,
        "git_short_sha": git_sha[:7] if git_sha != "unknown" else "unknown",
        "deploy_timestamp": os.environ.get("DEPLOY_TIMESTAMP", "unknown"),
        "process_started_at": _PROCESS_START_ISO,
        "process_uptime_s": process_uptime_s,
        "last_known_pfix": _LAST_KNOWN_PFIX,
        "knobs_count": knobs_count,
        "knobs_sample": knobs_sample,
        # [P2-HEALTH-KNOBS-COUNT · 2026-05-28] Solo el CONTEO de overrides en el
        # endpoint público. Pre-fix exponía `knobs_diff` (nombre+default+value de
        # cada knob tuneado) sin auth → revelaba a anónimos QUÉ defensas están
        # relajadas y a qué valor (p.ej. SHOPPING_COHERENCE_GUARD=warn, rate
        # limits bajados, CB thresholds). El detalle completo sigue en
        # `/admin/knobs`. Tooltip-anchor: P2-HEALTH-KNOBS-COUNT.
        "knobs_overrides_count": len(knobs_diff),
        "cron_missed_1h_total": cron_missed_1h_total,
        # [P2-HEALTHZ-DEEP · 2026-05-12] 5 keys nuevas para blackbox monitor.
        "expected_marker": expected_marker,
        "drift": drift,
        "last_pipeline_metrics_tick_at": last_pipeline_metrics_tick_at,
        "has_p0_prod_1_gate": has_p0_prod_1_gate,
        "has_p1_perf_1_cache": has_p1_perf_1_cache,
        # [P2-EMBED-TELEMETRY · 2026-05-24] gauges de los embedding caches.
        "embedding_cache_size": embedding_cache_size,
        "pantry_embeddings_cache_size": pantry_embeddings_cache_size,
        "embedding_cache_maxsize": embedding_cache_maxsize,
    }


@app.get("/admin/knobs")
def admin_knobs():
    """[P3-5 · 2026-05-10] Registry completo de knobs `MEALFIT_*` activos.

    Devuelve el snapshot completo del `_KNOBS_REGISTRY` para diagnóstico
    operacional. Cada entrada: `{name: {type, default, raw, value, parse_failed}}`.

    Útil para responder "¿qué valor real está usando el proceso?" sin
    ejecutar scripts Python ni leer código. Complementa `/health/version`
    que solo expone el `knobs_diff` (subset relevante).

    No requiere auth: los valores no son secretos (todos vienen de env vars
    `MEALFIT_*` que el operador conoce).
    """
    try:
        from graph_orchestrator import get_knobs_registry_snapshot
        snap = get_knobs_registry_snapshot()
        return {
            "count": len(snap),
            "knobs": snap,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/admin/cron-health")
def admin_cron_health():
    """[P0-2 · 2026-05-10] Diagnóstico rápido del scheduler APScheduler.

    Devuelve para operador (sin auth — info de diagnóstico no sensible):
      - jobs_registered: count + lista de job_ids registrados ahora.
      - missed_last_hour: dict {job_id: count} agregado de `system_alerts`
        con `alert_type='scheduler' AND alert_key LIKE 'scheduler_missed_%'`
        en la última hora. Vacío si no hubo MISSED.
      - cascade_alert: snapshot de la fila `scheduler_cascade_missed` si existe.
      - scheduler_config: defaults activos del pool (workers + misfire_grace).

    Sirve para responder en 1 seg "¿el scheduler está sano?" sin entrar a
    los logs de la plataforma. Cierra el gap diagnóstico identificado en el audit
    2026-05-10 (operador tenía que correr 3 queries SQL para entender
    si era cascada, drift de deploy, o ambas).
    """
    try:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        info: dict = {
            "scheduler_registered": HAS_SCHEDULER and scheduler is not None,
            "scheduler_config": {
                "max_workers": _SCHEDULER_MAX_WORKERS,
                "misfire_grace_s": _SCHEDULER_MISFIRE_GRACE_S,
                "telemetry_enabled": _is_scheduler_telemetry_enabled(),
            },
            "jobs_registered": [],
            "missed_last_hour": {},
            "cascade_alert": None,
        }
        if HAS_SCHEDULER and scheduler is not None:
            try:
                info["jobs_registered"] = sorted(
                    [j.id for j in scheduler.get_jobs()]
                )
            except Exception as e:
                info["jobs_registered_error"] = f"{type(e).__name__}: {e}"

        # MISSED last hour. Reusa el pool psycopg global. Si no está
        # disponible, devuelve dict vacío + flag explícito para diagnóstico.
        if connection_pool is not None:
            try:
                cutoff = _dt.now(_tz.utc) - _td(hours=1)
                # [P1-NEON-DB-MIGRATION · 2026-06-12] fetch + agregación en
                # Python (paridad con el shape PostgREST previo).
                missed_rows = execute_sql_query(
                    """
                    SELECT alert_key, metadata, triggered_at
                    FROM system_alerts
                    WHERE alert_type = 'scheduler'
                      AND alert_key LIKE %s
                      AND triggered_at >= %s
                    LIMIT 500
                    """,
                    ("scheduler_missed_%", cutoff),
                    fetch_all=True,
                ) or []
                counts: dict = {}
                for row in missed_rows:
                    meta = row.get("metadata")
                    job_id = None
                    if isinstance(meta, dict):
                        job_id = meta.get("job_id")
                    if not job_id:
                        ak = row.get("alert_key", "")
                        if ak.startswith("scheduler_missed_"):
                            job_id = ak[len("scheduler_missed_"):]
                    if job_id:
                        counts[job_id] = counts.get(job_id, 0) + 1
                info["missed_last_hour"] = counts
            except Exception as e:
                info["missed_last_hour_error"] = f"{type(e).__name__}: {e}"

            # Última cascada (si existe). Los timestamptz llegan como
            # datetime (FastAPI los serializa a ISO en el response).
            try:
                cascade_row = execute_sql_query(
                    """
                    SELECT alert_key, severity, message, metadata, triggered_at, resolved_at
                    FROM system_alerts
                    WHERE alert_key = %s
                    LIMIT 1
                    """,
                    ("scheduler_cascade_missed",),
                    fetch_one=True,
                )
                info["cascade_alert"] = cascade_row or None
            except Exception as e:
                info["cascade_alert_error"] = f"{type(e).__name__}: {e}"
        else:
            # Señala que el pool psycopg (Neon) no está inicializado. Si un
            # dashboard SRE externo leía la key anterior, actualizarlo a la
            # nueva `db_pool_unavailable`.
            info["db_pool_unavailable"] = True

        return info
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


@app.get("/api/admin/test-proactive")
def api_test_proactive(background_tasks: BackgroundTasks):
    # [P2-1 2026-05-08] Antes existían dos handlers `@app.get("/api/admin/test-proactive")`
    # consecutivos: uno síncrono y este async-via-background. FastAPI registra el
    # último decorador, así que la versión síncrona quedaba sobrescrita y nunca se
    # ejecutaba. Eliminada para que el lector no asuma que existen dos modos.
    import traceback
    def run_push():
        with open("push_log.txt", "w", encoding="utf-8") as f:
            try:
                from test_push import trigger_manual_notification
                import sys, io
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                trigger_manual_notification("Almuerzo", "1:30 PM")
                f.write(sys.stdout.getvalue())
                sys.stdout = old_stdout
            except Exception as e:
                f.write(traceback.format_exc())

    background_tasks.add_task(run_push)
    return {"status": "started", "message": "Task queued"}

from auth import get_verified_user_id, verify_api_quota, clear_session_cookie
from rate_limiter import RateLimiter
from services import _save_plan_and_track_background, _process_swap_rejection_background

# [P2-RATELIMIT-COVERAGE · 2026-05-12] Rate limiters defensivos para endpoints
# que NO van por el paywall (`verify_api_quota`) y NO son admin-only:
#
#  - `_WEBHOOK_FACTS_LIMITER` (10/min/IP): el webhook
#    `/api/webhooks/process-pending-facts` se invoca raras veces (legítimamente
#    1-3/min por DB trigger). Bursts ≥10/min = signal de abuse — atacante
#    intentando flood el HMAC check con UUIDs enumerados antes que el
#    `compare_digest` rechace. Defensa-en-profundidad sobre P0-WEBHOOK-1.
#
#  - `_AUTH_MIGRATE_LIMITER` (5/5min/IP): el migrate guest→user es one-shot
#    en el flujo del usuario (1 vez en su vida). 5 calls cada 5min cubre
#    re-tries legítimos de error transitorio sin permitir brute-force con
#    UUIDs enumerados (vector real: atacante enumera user_id reales y
#    fuerza migrate calls esperando race-conditions con guest sessions).
#
# Defaults conservadores: bloquean abuse sin afectar UX legítima. Pueden
# ser sobrescritos vía env vars MEALFIT_RATELIMIT_WEBHOOK_FACTS_*/MIGRATE_*
# si producción muestra throttle en usuarios reales (improbable).
# Anchor: P2-RATELIMIT-COVERAGE-WEBHOOKS.
_WEBHOOK_FACTS_LIMITER = RateLimiter(max_calls=10, period_seconds=60)
_AUTH_MIGRATE_LIMITER = RateLimiter(max_calls=5, period_seconds=300)
# [P1-ACCOUNT-DELETE-1 · 2026-06-22] Borrado de cuenta = destructivo + irreversible
# → throttle estricto (3/5min) para cortar replay/loop/CSRF accidental.
_ACCOUNT_DELETE_LIMITER = RateLimiter(max_calls=3, period_seconds=300)
# [P2-PRIVACY-SETTINGS · 2026-07-04] Export de datos = query pesada (planes con
# plan_data completo) → mismo throttle estricto que el delete.
_ACCOUNT_EXPORT_LIMITER = RateLimiter(max_calls=3, period_seconds=300)

# [P2-CORS-NARROW · 2026-05-12] Setup CORS para el frontend React.
#
# Pre-fix: `allow_methods=["*"]` y `allow_headers=["*"]` con
# `allow_credentials=True` y 6 origins constraint. Defense-in-depth gap:
# si una futura ruta backend acepta verbos exóticos (CONNECT, TRACE, etc.)
# o si un script third-party intenta usar headers personalizados para
# exfiltrar via fetch + credentials, el wildcard `*` los habilita.
#
# Lista explícita derivada del grep cross-codebase 2026-05-12:
#   - Métodos: solo los 5 verbos REST que el backend define + OPTIONS
#     (preflight). No usamos HEAD/TRACE/CONNECT.
#   - Headers: solo los que el frontend manda en `fetch(...)`:
#     Authorization (JWT), Content-Type (application/json), Accept,
#     X-Requested-With (algunos SDKs lo inyectan automático). NO custom
#     `X-*` headers — si en el futuro se añade alguno, hay que extender
#     esta lista explícitamente.
#
# Si CI falla por CORS preflight rechazado tras añadir un header nuevo,
# es una señal explícita para revisar el threat model antes de añadirlo
# al whitelist (no auto-whitelist via `*`).
#
# Tooltip-anchor: P2-CORS-NARROW.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://mealfitrd.com",
        "https://www.mealfitrd.com"
    ], # Dominios de producción (servidos por nginx en el VPS Oracle)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "X-Requested-With",
        # [H2 / P3-CORRELATION-ID · 2026-05-20] Permite al cliente propagar
        # un correlation ID desde el browser tracing (Sentry browserTracing
        # podría inyectarlo en futuro) o desde scripts SRE/curl manuales.
        # Si el header llega → reusado server-side; si no → generamos uno.
        "X-Correlation-ID",
    ],
    # [H2 / P3-CORRELATION-ID · 2026-05-20] expose_headers permite que el
    # browser JS lea `X-Correlation-ID` de la response — útil para que el
    # frontend lo muestre en reportes de bug ("incluye este ID al reportar").
    expose_headers=["X-Correlation-ID"],
)


# [H2 / P3-CORRELATION-ID · 2026-05-20] Middleware FastAPI que asigna un
# correlation_id por request y lo propaga via ContextVar a TODO el flow
# (handlers, async tasks, asyncio.to_thread, y bg_executor con
# `contextvars.copy_context()`).
#
# Diseño:
#   - Lee `X-Correlation-ID` del request header si el cliente lo provee
#     (8 chars hex max — defensivo contra header injection larga).
#   - Si no hay o es inválido, genera uno nuevo via `new_correlation_id()`.
#   - Setea el contextvar con Token, ejecuta el handler, reset en finally.
#   - Echo del ID en response header → cliente puede citarlo en bug reports.
#   - Logs durante el request automáticamente incluyen `[corr=<id>]` via
#     `CorrelationIdFilter` (ver correlation.py).
#
# Ubicación: `@app.middleware("http")` decora una función que envuelve
# el flow request→response completo. Se registra ANTES de los endpoints
# (este bloque está aquí intencionalmente arriba del primer @app.get).
#
# Tooltip-anchor: P3-CORRELATION-ID.
from correlation import (  # noqa: E402 — orden intencional post-CORS
    new_correlation_id as _new_corr_id,
    set_correlation_id as _set_corr,
    reset_correlation_id as _reset_corr,
)
import re as _re_corr

# Header value sanitization: solo aceptar 1-64 chars de hex/dash/underscore.
# Cualquier otro patrón → rechazar y generar nuevo. Defense contra
# log injection (CRLF en el header), display attacks, y longitud abusiva.
_CORR_ID_HEADER_RE = _re_corr.compile(r"^[A-Za-z0-9_\-]{1,64}$")


@app.middleware("http")
async def _correlation_id_middleware(request, call_next):
    incoming = request.headers.get("X-Correlation-ID") or request.headers.get("x-correlation-id")
    if incoming and _CORR_ID_HEADER_RE.match(incoming):
        cid = incoming
    else:
        cid = _new_corr_id()

    token = _set_corr(cid)
    try:
        response = await call_next(request)
    finally:
        _reset_corr(token)

    # Echo en response — siempre, incluso si el handler levantó excepción
    # (FastAPI ya construyó la response de error en ese caso). El cliente
    # lee este header para reportar bugs.
    try:
        response.headers["X-Correlation-ID"] = cid
    except Exception:
        # Si por alguna razón la response no permite mutar headers
        # (StreamingResponse en estados específicos), ignorar — el log
        # server-side sigue teniendo el ID.
        pass
    return response


# [P2-BODY-SIZE-LIMIT · 2026-05-27] Rechaza requests cuyo `Content-Length`
# excede el cap → previene DoS por agotamiento de memoria (un POST JSON
# gigante a /api/analyze/stream o /api/auth/migrate se buffea entero en RAM
# antes de llegar al handler). Default 25 MiB: por ENCIMA del upload de imagen
# legítimo más grande (20 MB en /api/diary, routers/diary.py:189 lee el file
# completo a memoria) pero acota payloads absurdos (100 MB+). Knob
# `MEALFIT_MAX_REQUEST_BYTES` (clamp [1 MiB, 200 MiB]; 0 desactiva el guard).
# Best-effort: solo aplica cuando el cliente envía `Content-Length` (transfer
# chunked sin él lo limita uvicorn). Registrado como `@app.middleware` DESPUÉS
# del de correlation → queda OUTERMOST → rechaza antes de cualquier otro
# procesamiento. Tooltip-anchor: P2-BODY-SIZE-LIMIT.
_MAX_REQUEST_BYTES = _env_int(
    "MEALFIT_MAX_REQUEST_BYTES",
    25 * 1024 * 1024,
    validator=lambda v: v == 0 or (1024 * 1024 <= v <= 200 * 1024 * 1024),
)


@app.middleware("http")
async def _body_size_limit_middleware(request, call_next):
    if _MAX_REQUEST_BYTES > 0:
        _cl = request.headers.get("content-length")
        if _cl:
            try:
                _cl_int = int(_cl)
            except (TypeError, ValueError):
                _cl_int = None
            if _cl_int is not None and _cl_int > _MAX_REQUEST_BYTES:
                from fastapi.responses import JSONResponse
                logger.warning(
                    f"🛡️ [P2-BODY-SIZE-LIMIT] Rechazado {request.method} "
                    f"{request.url.path}: content-length={_cl_int} > "
                    f"{_MAX_REQUEST_BYTES} bytes."
                )
                return JSONResponse(
                    status_code=413,
                    content={"detail": "El cuerpo de la solicitud es demasiado grande."},
                )
    return await call_next(request)




@app.get("/api/user/credits/{user_id}")
def api_get_user_credits(user_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """Consulta los créditos consumidos en el mes usando api_usage."""
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido. Token inválido o no coincide.")
                
        if not user_id or user_id == "guest":
            return {"credits": 0}
        credits_used = get_monthly_api_usage(user_id)
        return {"credits": credits_used}
    except HTTPException as he:
        # Re-lanzar excepciones HTTP explícitas (ej. 401/403 de Auth)
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/user/credits GET: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

@app.get("/api/user-facts/{user_id}")
def api_get_user_facts(user_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"facts": []}
        facts_data = get_all_user_facts(user_id)
        return {"success": True, "facts": facts_data}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/user-facts GET: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

@app.delete("/api/user-facts/{fact_id}")
def api_delete_user_fact(fact_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="Token de autenticación requerido.")
        
        # Validación IDOR: verificar que el fact pertenece al usuario autenticado
        if not check_fact_ownership(fact_id, verified_user_id):
            raise HTTPException(status_code=403, detail="No tienes permiso para borrar este hecho.")
        
        result = delete_user_fact(fact_id)
        return {"success": True, "message": "Hecho eliminado de la IA."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/user-facts DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

@app.post("/api/account/reset-preferences")
def api_reset_user_preferences(verified_user_id: str = Depends(get_verified_user_id)):
    """Borra preferencias (likes, dislikes), inventario y vacía el health_profile."""
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="Token de autenticación requerido.")
            
        from db_profiles import reset_user_account_preferences
        success = reset_user_account_preferences(verified_user_id)
        
        if success:
            return {"success": True, "message": "Preferencias de la cuenta restablecidas."}
        else:
            raise HTTPException(status_code=500, detail="Error al restablecer las preferencias.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/account/reset-preferences: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))


@app.post("/api/account/delete")
async def api_delete_my_account(
    response: Response,
    data: dict = Body(default={}),
    verified_user_id: Optional[str] = Depends(_ACCOUNT_DELETE_LIMITER),
):
    """[P1-ACCOUNT-DELETE-1 · 2026-06-22] Eliminación de cuenta SELF-SERVICE e
    IRREVERSIBLE desde Ajustes. Reusa el motor determinístico `delete_account_data`
    (P1-PROD-AUDIT-2) que purga TODA la data user-scoped (~36 tablas + checkpoints
    LangGraph + Storage).

    Seguridad / contrato:
      - `verified_user_id` se deriva SOLO del JWT verificado (vía el RateLimiter
        que llama `get_verified_user_id`). NUNCA se acepta un `user_id` del body
        → no IDOR (simétrico de I2 / P0-AGENT-1). El endpoint admin
        `/api/system/admin/account/purge-data` SÍ toma user_id del body pero está
        CRON_SECRET-gated; este NO debe copiar esa forma.
      - Confirmación explícita: el body debe traer `confirm == "ELIMINAR"`.
      - Throttle estricto (`_ACCOUNT_DELETE_LIMITER`, 3/5min).

    Flujo cancel-then-delete (fail-loud):
      1. Si hay suscripción PayPal activa → cancelarla ANTES de borrar (la fila
         user_profiles guarda `paypal_subscription_id`; borrarla sin cancelar =
         cobro fantasma). Un fallo real de PayPal aborta el borrado (502/503).
      2. `delete_account_data(uid, include_profile=True)` → purga determinística.
      3. Invalida la cookie de sesión first-party server-side (belt-and-suspenders
         con el `resetApp()`/signOut del cliente).

    Limitación conocida (documentada): la identidad de Neon Auth NO se borra aún
    (admin API no cableada, ver routers/system.py). Tras eliminar, la data está
    100% borrada y la sesión invalidada; si el usuario vuelve a iniciar sesión con
    el mismo correo obtiene una cuenta NUEVA vacía (re-registro), no la anterior.
    """
    if not verified_user_id or verified_user_id == "guest":
        raise HTTPException(status_code=401, detail="Autenticación requerida.")
    if str((data or {}).get("confirm") or "").strip().upper() != "ELIMINAR":
        raise HTTPException(status_code=400, detail="Confirmación inválida. Escribe ELIMINAR para confirmar.")
    try:
        # 1. Cancelar PayPal ANTES de borrar (cancel-then-delete, fail-loud).
        billing_cancelled = False
        sub_row = await asyncio.to_thread(
            lambda: execute_sql_query(
                "SELECT paypal_subscription_id, subscription_status FROM public.user_profiles WHERE id = %s",
                (verified_user_id,), fetch_one=True,
            )
        )
        sub_id = (sub_row or {}).get("paypal_subscription_id")
        sub_status = str((sub_row or {}).get("subscription_status") or "").upper()
        if sub_id and sub_status not in ("CANCELLED", "INACTIVE", "EXPIRED"):
            from routers.billing import cancel_paypal_subscription_for_user
            # fail-loud: si PayPal rechaza, esto raise 502/503 y ABORTA el borrado
            # → nunca dejamos una sub viva tras "eliminar cuenta".
            await cancel_paypal_subscription_for_user(
                verified_user_id, sub_id, reason="El usuario eliminó su cuenta desde la App."
            )
            billing_cancelled = True

        # 2. Purga determinística de TODA la data user-scoped (motor existente).
        from db_profiles import delete_account_data
        result = await asyncio.to_thread(delete_account_data, verified_user_id, True)

        # 3. Invalidar la sesión first-party server-side.
        clear_session_cookie(response)

        ok = len(result.get("errors") or []) == 0
        logger.info(
            f"[P1-ACCOUNT-DELETE-1] cuenta {verified_user_id} eliminada "
            f"(ok={ok}, billing_cancelled={billing_cancelled}, errors={len(result.get('errors') or [])})"
        )
        return {
            "success": ok,
            "billing_cancelled": billing_cancelled,
            "deleted": result.get("deleted"),
            "errors": result.get("errors"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [P1-ACCOUNT-DELETE-1] Error eliminando cuenta {verified_user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

# [P2-PRIVACY-SETTINGS · 2026-07-04] Tablas incluidas en el export de datos del
# usuario (sección Privacidad de Configuración). Best-effort per-tabla: una tabla
# renombrada/faltante se salta (queda en `skipped`) sin tumbar el export completo.
# Caps de filas por tabla — acotan payload (plan_data puede pesar MBs) sin
# recortar al usuario típico. Si un cap se alcanza, la tabla se lista en
# `truncated` para que el usuario sepa que hay más data que la exportada.
_ACCOUNT_EXPORT_TABLES = (
    ("user_profiles", "id", 1),
    ("meal_plans", "user_id", 50),
    ("user_inventory", "user_id", 2000),
    ("user_depleted_items", "user_id", 2000),
    ("consumed_meals", "user_id", 5000),
    ("user_facts", "user_id", 2000),
    ("user_taste_events", "user_id", 2000),
    ("agent_sessions", "user_id", 200),
    ("agent_messages", "user_id", 5000),
)

# Columnas internas sin valor para el usuario y costosas de serializar
# (vectores pgvector de 1536 floats). Se eliminan de cada fila exportada.
_ACCOUNT_EXPORT_STRIPPED_KEYS = ("embedding",)


@app.get("/api/account/export")
async def api_export_my_account(
    verified_user_id: Optional[str] = Depends(_ACCOUNT_EXPORT_LIMITER),
):
    """[P2-PRIVACY-SETTINGS · 2026-07-04] Export self-service de los datos del
    usuario (sección Privacidad de Configuración) — copia JSON de perfil, planes,
    nevera, comidas registradas, memoria del agente y conversaciones.

    Contrato (espejo de P1-ACCOUNT-DELETE-1):
      - `verified_user_id` SOLO del JWT verificado (vía RateLimiter). NUNCA se
        acepta user_id del cliente → no IDOR (simétrico I2 / P0-AGENT-1).
      - Read-only: cero mutaciones.
      - Quota-exempt: NO `verify_api_quota` ni `log_api_usage` (lección
        P1-NEVERA-QUOTA-EXEMPT — exportar tus datos es un derecho, no un
        consumo de IA; el throttle es `_ACCOUNT_EXPORT_LIMITER` 3/5min).
    """
    if not verified_user_id or verified_user_id == "guest":
        raise HTTPException(status_code=401, detail="Autenticación requerida.")

    def _collect() -> dict:
        payload: dict = {"data": {}, "counts": {}, "truncated": [], "skipped": []}
        for tbl, col, cap in _ACCOUNT_EXPORT_TABLES:
            try:
                rows = execute_sql_query(
                    f"SELECT * FROM {tbl} WHERE {col} = %s LIMIT {int(cap)}",
                    (verified_user_id,),
                    fetch_all=True,
                ) or []
                for row in rows:
                    for key in _ACCOUNT_EXPORT_STRIPPED_KEYS:
                        row.pop(key, None)
                payload["data"][tbl] = rows
                payload["counts"][tbl] = len(rows)
                if cap > 1 and len(rows) >= cap:
                    payload["truncated"].append(tbl)
            except Exception as exc:
                # Best-effort: tabla faltante/renombrada no tumba el export.
                logger.debug(f"[P2-PRIVACY-SETTINGS] export saltó {tbl}: {exc}")
                payload["skipped"].append(tbl)
        return payload

    payload = await asyncio.to_thread(_collect)
    payload["app"] = "MealfitRD"
    payload["format_version"] = 1
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["notes"] = (
        "Copia de tus datos en MealfitRD. Las tablas en `truncated` tienen más "
        "filas que el límite exportado; `skipped` no aplican a tu cuenta o "
        "no existen en esta versión."
    )
    logger.info(
        f"[P2-PRIVACY-SETTINGS] export generado para {verified_user_id} "
        f"(tablas={len(payload['counts'])}, skipped={len(payload['skipped'])})"
    )
    return payload


@app.post("/api/webhooks/process-pending-facts")
def api_webhook_process_pending_facts(
    request: Request,
    data: dict = Body(...),
    authorization: Optional[str] = Header(None),
    _rl: Optional[str] = Depends(_WEBHOOK_FACTS_LIMITER),
):
    """
    Endpoint consumido por el Webhook de la DB (Database Trigger AFTER INSERT en pending_facts_queue).
    Permite procesar asíncronamente y de manera segura la cola de extracción sin depender de demonios en memoria.

    [P0-WEBHOOK-1 · 2026-05-12] Pre-fix: si `WEBHOOK_SECRET` no estaba seteada
    (`if webhook_secret:` sin else), el check de auth se saltaba enteramente
    → cualquier atacante podía invocar `process_pending_queue_sync(user_id_ajeno)`
    con un UUID enumerado. Ahora: en producción, `WEBHOOK_SECRET` faltante
    rechaza con 503; en cualquier ambiente, la comparación usa
    `hmac.compare_digest` (constant-time) para no exponer timing side-channel.
    Anchor: P0-WEBHOOK-1-FAIL-SECURE.

    Nota P1-AUDIT-1 (2026-05-12): el DB trigger que llamaba este endpoint fue
    eliminado y reemplazado por el cron `_drain_pending_facts_queue`. El
    endpoint queda preservado por harm-zero pero podría deprecarse en futuro
    si nadie externo lo consume.
    """
    import hmac
    try:
        # [P0-WEBHOOK-1] Validación fail-secure del secret.
        webhook_secret = os.environ.get("WEBHOOK_SECRET")
        # [P2-PROD-AUDIT-3 · 2026-05-30] SSOT normalizado (lower+strip). Var local
        # renombrada para no sombrear el helper importado `is_production`.
        _is_prod_env = is_production()

        if not webhook_secret:
            if _is_prod_env:
                logger.error(
                    "❌ [P0-WEBHOOK-1] WEBHOOK_SECRET ausente en producción. "
                    "Rechazando invocación (pre-fix saltaba el check entero)."
                )
                raise HTTPException(
                    status_code=503,
                    detail="Webhook secret not configured.",
                )
            logger.warning(
                "⚠️ [DEV] WEBHOOK_SECRET ausente; permitido solo fuera de producción."
            )
        else:
            # Extraer token de múltiples fuentes posibles (headers custom del webhook)
            token = ""
            if authorization and authorization.startswith("Bearer "):
                token = authorization.split(" ", 1)[1]
            elif authorization:
                token = authorization

            custom_header_secret = request.headers.get("X-Webhook-Secret") or ""

            # constant-time compare en ambos slots — evita timing oracle que
            # pudiera distinguir secret válido (mismatch en byte N) de
            # invalidísimo (mismatch en byte 0).
            token_ok = hmac.compare_digest(token or "", webhook_secret)
            header_ok = hmac.compare_digest(custom_header_secret, webhook_secret)
            if not token_ok and not header_ok:
                logger.warning("🔒 Intento no autorizado al Webhook de hechos (Secret inválido).")
                raise HTTPException(status_code=401, detail="Unauthorized webhook invocation")

        # 2. Extraer el Payload del trigger
        # Los webhooks de DB mandan la fila en data["record"] cuando es un trigger INSERT
        record = data.get("record", {})
        user_id = record.get("user_id") or data.get("user_id")

        if not user_id:
            logger.warning("⚠️ Webhook llamado sin parametro user_id.")
            return {"success": False, "message": "Falta user_id"}

        logger.info(f"⚡ [WEBHOOK RECIBIDO] Procesando cola pendiente para user_id: {user_id}")

        # 3. Procesamiento síncrono (garantiza que serverless espere a terminar)
        process_pending_queue_sync(user_id)

        return {"success": True, "message": f"Cola procesada para {user_id}"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [WEBHOOK ERROR]: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

@app.post("/api/auth/migrate")
def api_migrate_guest(
    data: dict = Body(...),
    verified_user_id: str = Depends(get_verified_user_id),
    _rl: Optional[str] = Depends(_AUTH_MIGRATE_LIMITER),
):
    """
    Endpoint invocado post-registro para migrar la metadata acumulada por un 'guest' a su nuevo UUID.
    """
    try:
        session_ids = data.get("session_ids", [])
        session_id = data.get("session_id")
        new_user_id = data.get("user_id")
        current_plan = data.get("current_plan")
        health_profile = data.get("health_profile")
        
        # Validar token
        if not verified_user_id or verified_user_id != new_user_id:
            raise HTTPException(status_code=401, detail="No autorizado o token no coincide con user_id.")
            
        # Homologar session_ids a lista
        if not session_ids and session_id:
            session_ids = [session_id]
        if isinstance(session_ids, str):
            session_ids = [session_ids]
            
        if not session_ids or not new_user_id:
            raise HTTPException(status_code=400, detail="Faltan parámetros (session_ids o user_id).")
            
        
        # 1. Transformar data guest a registrada
        success = migrate_guest_data(session_ids, new_user_id)
        if not success:
            logger.warning(f"⚠️ Aviso: La función de migración base devolvió False, pero continuamos con profile y planes.")
        
        # 2. Upsert health_profile si el frontend lo provee
        if health_profile:
            try:
                profile = get_user_profile(new_user_id)
                # Si el usuario es nuevo, puede no existir su perfil
                if profile:
                    # [P1-2] Atomic write con mutator que MERGEA el payload
                    # del frontend ON TOP del estado existente bajo FOR UPDATE.
                    # Antes era full-overwrite del JSONB column (legacy
                    # `update_user_health_profile`): si entre el `migrate_guest_data`
                    # de arriba y este write, otro path (cron, request paralelo)
                    # poblaba campos derivados del usuario recién migrado, ese
                    # estado se perdía. La migración guest→registered es
                    # típicamente one-shot, pero el doble-click en signup
                    # puede disparar dos llamadas concurrentes a este endpoint;
                    # el atomic helper las serializa.
                    def _migrate_mutator(_hp):
                        _hp.update(health_profile)
                        return None

                    update_user_health_profile_atomic(new_user_id, _migrate_mutator)
                else:
                    upsert_user_profile(new_user_id, health_profile)
            except Exception as e:
                logger.error(f"Error migrando health_profile: {e}")
                
        # 3. Guardar el plan "guest" si existe
        if current_plan:
            existing_plan = get_latest_meal_plan(new_user_id)
            if not existing_plan:
                try:
                    from datetime import datetime
                    # [P1-NEON-DB-MIGRATION · 2026-06-12] El guard chequea el
                    # pool psycopg (save_new_meal_plan_robust escribe via SQL).
                    if connection_pool is not None:
                        calories = current_plan.get("calories", 0)
                        macros = current_plan.get("macros", {})
                        
                        meal_names = []
                        ingredients = []
                        for d in current_plan.get("days", []):
                            for m in d.get("meals", []):
                                if m.get("name"):
                                    meal_names.append(m.get("name"))
                                if m.get("ingredients"):
                                    ingredients.extend(m.get("ingredients"))
                                    
                        insert_data = {
                            "user_id": new_user_id,
                            "plan_data": current_plan,
                            "name": f"Plan Evolutivo - {datetime.now().strftime('%d/%m/%Y')}",
                            "calories": int(calories) if calories else 0,
                            "macros": macros,
                            "meal_names": meal_names,
                            "ingredients": ingredients
                        }
                        save_new_meal_plan_robust(insert_data)
                except Exception as e:
                    logger.error(f"Error migrando current_plan: {e}")
                    
        return {"success": True, "message": "Tu progreso como invitado se ha migrado a tu nueva cuenta."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/auth/migrate: {str(e)}")
        raise HTTPException(status_code=500, detail=safe_error_detail(e))

if __name__ == "__main__":
    # [P2-UVICORN-RELOAD-ENV · 2026-05-12] `reload` controlado por env var.
    # Pre-fix tenía `reload=True` hardcoded. En el VPS Oracle actualmente se
    # arranca via el deploy automatizado (no `python app.py`), pero un futuro script change
    # podría re-introducir este entry point en producción con el flag activo
    # — auto-reload en prod re-importa módulos cada cambio de archivo y
    # rompe cualquier estado in-process (cache de knobs, _SCHEDULER_*, pools).
    # Default `0` (off); setear `UVICORN_RELOAD=1` solo en dev local.
    _reload_flag = os.environ.get("UVICORN_RELOAD", "0").strip().lower() in ("1", "true", "yes", "on")
    uvicorn.run("app:app", host="0.0.0.0", port=3001, reload=_reload_flag)
