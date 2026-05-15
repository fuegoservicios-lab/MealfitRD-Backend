from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile, File, Form, Body, Header, Depends
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
_LAST_KNOWN_PFIX = "P3-SELECT-STAR-AGENT-SESSIONS · 2026-05-15"

# [P1-SENTRY-SAMPLE-COST · 2026-05-12] Sentry sampling driven from env vars
# con default seguro 0.1 (10%). Pre-fix tenía `traces_sample_rate=1.0` y
# `profiles_sample_rate=1.0` → 100% de transacciones + profiling continuo,
# costoso a escala y arriesga throttle de la cuota Sentry (dropping justo
# los errores que necesitas). Validator clamp [0.0, 1.0]; valores fuera de
# rango caen al default. Auto-registrado en `_KNOBS_REGISTRY` → visible en
# `/health/version`. Tooltip-anchor: P1-SENTRY-SAMPLE-COST.
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

# Configuración centralizada de logging para todo el backend
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silenciar logs verbosos de httpx (Supabase client)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from db import (
    connection_pool, async_connection_pool, supabase,
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
from knobs import _env_int, _env_bool

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
# al grace=2min). 5min cubre deploys rolling estándar de Easypanel y
# Nixpacks sin ocultar MISSED genuinos (autoheals P0-LIVE-1 / P0-AUDIT-1
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
_OPEN_ALERTS_CACHE_TTL_S = int(os.environ.get(
    "MEALFIT_SCHEDULER_OPEN_ALERTS_CACHE_TTL_S", "60"
) or 60)
if _OPEN_ALERTS_CACHE_TTL_S < 15:
    _OPEN_ALERTS_CACHE_TTL_S = 15
if _OPEN_ALERTS_CACHE_TTL_S > 300:
    _OPEN_ALERTS_CACHE_TTL_S = 300


def _refresh_scheduler_open_alerts_cache(force: bool = False) -> int:
    """Sincroniza `_SCHEDULER_JOBS_WITH_OPEN_ALERTS` con la realidad de DB.

    Idempotente. Best-effort: cualquier excepción se loguea sin propagar.
    Retorna el count de jobs en el set post-sync (0 si supabase no
    disponible o falló).
    """
    global _SCHEDULER_OPEN_ALERTS_LAST_REFRESH
    if not force:
        if time.time() - _SCHEDULER_OPEN_ALERTS_LAST_REFRESH < _OPEN_ALERTS_CACHE_TTL_S:
            with _SCHEDULER_OPEN_ALERTS_LOCK:
                return len(_SCHEDULER_JOBS_WITH_OPEN_ALERTS)
    if supabase is None:
        return 0
    try:
        # SELECT alerts abiertas; parsear job_id; reconstruir el set.
        # No paginamos: típicamente <100 alerts abiertas.
        res = (
            supabase.table("system_alerts")
            .select("alert_key")
            .is_("resolved_at", "null")
            .or_(
                "alert_key.like.scheduler_missed_%,alert_key.like.scheduler_error_%"
            )
            .execute()
        )
        rows = res.data or []
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
    (un fallo en supabase no debe pausar el resto de los crons). Idempotente
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
            if supabase is None:
                return
            # Refresh perezoso si pasó el TTL — barato (~1 SELECT/min).
            _refresh_scheduler_open_alerts_cache(force=False)
            with _SCHEDULER_OPEN_ALERTS_LOCK:
                needs_patch = job_id in _SCHEDULER_JOBS_WITH_OPEN_ALERTS
            if not needs_patch:
                return  # ← skip PATCH no-op
            try:
                _now_iso = datetime.now(_tz.utc).isoformat()
                supabase.table("system_alerts").update({
                    "resolved_at": _now_iso,
                }).in_(
                    "alert_key",
                    [
                        f"scheduler_missed_{job_id}",
                        f"scheduler_error_{job_id}",
                    ],
                ).is_("resolved_at", "null").execute()
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

        if supabase is None:
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

        now_iso = datetime.now(_tz.utc).isoformat()
        sched_iso = scheduled_run_time.isoformat() if scheduled_run_time else None
        supabase.table("system_alerts").upsert({
            "alert_key": alert_key,
            "alert_type": "scheduler",
            "severity": severity,
            "title": title,
            "message": message,
            "metadata": {
                "job_id": job_id,
                "scheduled_run_time": sched_iso,
                "event_type": event_type,
                "max_workers": _SCHEDULER_MAX_WORKERS,
                "misfire_grace_s": _SCHEDULER_MISFIRE_GRACE_S,
            },
            "triggered_at": now_iso,
            "resolved_at": None,
        }, on_conflict="alert_key").execute()
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
        
    # [P2-NEW-E · 2026-05-07] El bloque DDL runtime (CREATE TABLE IF NOT EXISTS
    # + ALTERs + UPDATEs de backfill + índices para 7 tablas) que vivía aquí se
    # consolidó al SSOT `supabase/migrations/p2_new_e_consolidate_runtime_ddl.sql`.
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
            db_uri = os.environ.get("SUPABASE_DB_URL")
            # autocommit=True requerido por LangGraph PostgresSaver (puede crear
            # índices CONCURRENTLY internamente).
            with psycopg.connect(db_uri, autocommit=True) as conn:
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

    # [P0-LIVE-1 · 2026-05-11] Hard-floor autoheal asyncio task. Garantiza
    # sweeps de scheduler alerts + CB rows incluso si APScheduler está
    # saturado (corre en el event loop, no en el ThreadPoolExecutor del
    # scheduler). Guardado en app.state para cancelarlo en shutdown.
    if _env_bool("MEALFIT_HARDFLOOR_AUTOHEAL_ENABLED", True):
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

    if HAS_SCHEDULER and scheduler:
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
                "DATABASE_URL/SUPABASE_DB_URL y conectividad al pooler:6543."
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
        scheduler.shutdown()
    
    if connection_pool:
        connection_pool.close()
        logger.info("🔌 [psycopg] Pool de conexiones cerrado.")
    if async_connection_pool:
        await async_connection_pool.close()
        logger.info("🔌 [psycopg] Pool de conexiones asíncronas cerrado.")


# Asegurarnos de que el directorio de uploads exista antes de montar recursos estáticos
os.makedirs("uploads", exist_ok=True)


app = FastAPI(lifespan=lifespan)
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
# supabase/migrations/add_long_term_memory_enabled_2026_05_13.sql
from routers.preferences import router as preferences_router
app.include_router(preferences_router)

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
    if ready:
        return {"status": "ready", "plan_graph": "compiled"}

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
            "plan_graph": "not_compiled",
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
      - `git_sha` (inyectado por Nixpacks/EasyPanel via env var) debe
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
        if supabase is not None:
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            cutoff = _dt.now(_tz.utc) - _td(hours=1)
            res = (
                supabase.table("system_alerts")
                .select("alert_key", count="exact")
                .eq("alert_type", "scheduler")
                .like("alert_key", "scheduler_missed_%")
                .gte("triggered_at", cutoff.isoformat())
                .limit(0)
                .execute()
            )
            cron_missed_1h_total = res.count or 0
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
        if supabase is not None:
            kv_res = (
                supabase.table("app_kv_store")
                .select("value")
                .eq("key", "expected_last_known_pfix")
                .limit(1)
                .execute()
            )
            rows = kv_res.data or []
            if rows:
                raw = rows[0].get("value")
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
        if supabase is not None:
            tick_res = (
                supabase.table("pipeline_metrics")
                .select("created_at")
                .eq("node", "_hardfloor_autoheal_tick")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            tick_rows = tick_res.data or []
            if tick_rows:
                last_pipeline_metrics_tick_at = tick_rows[0].get("created_at")
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

    git_sha = os.environ.get("GIT_SHA") or os.environ.get("VERCEL_GIT_COMMIT_SHA") or "unknown"
    return {
        "git_sha": git_sha,
        "git_short_sha": git_sha[:7] if git_sha != "unknown" else "unknown",
        "deploy_timestamp": os.environ.get("DEPLOY_TIMESTAMP", "unknown"),
        "process_started_at": _PROCESS_START_ISO,
        "process_uptime_s": process_uptime_s,
        "last_known_pfix": _LAST_KNOWN_PFIX,
        "knobs_count": knobs_count,
        "knobs_sample": knobs_sample,
        "knobs_diff": knobs_diff,
        "cron_missed_1h_total": cron_missed_1h_total,
        # [P2-HEALTHZ-DEEP · 2026-05-12] 5 keys nuevas para blackbox monitor.
        "expected_marker": expected_marker,
        "drift": drift,
        "last_pipeline_metrics_tick_at": last_pipeline_metrics_tick_at,
        "has_p0_prod_1_gate": has_p0_prod_1_gate,
        "has_p1_perf_1_cache": has_p1_perf_1_cache,
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
    Supabase logs. Cierra el gap diagnóstico identificado en el audit
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

        # MISSED last hour. Reusa el cliente Supabase global. Si no está
        # disponible, devuelve dict vacío + flag explícito para diagnóstico.
        if supabase is not None:
            try:
                cutoff = _dt.now(_tz.utc) - _td(hours=1)
                res = (
                    supabase.table("system_alerts")
                    .select("alert_key,metadata,triggered_at")
                    .eq("alert_type", "scheduler")
                    .like("alert_key", "scheduler_missed_%")
                    .gte("triggered_at", cutoff.isoformat())
                    .limit(500)
                    .execute()
                )
                missed_rows = res.data or []
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

            # Última cascada (si existe).
            try:
                res = (
                    supabase.table("system_alerts")
                    .select("alert_key,severity,message,metadata,triggered_at,resolved_at")
                    .eq("alert_key", "scheduler_cascade_missed")
                    .limit(1)
                    .execute()
                )
                rows = res.data or []
                info["cascade_alert"] = rows[0] if rows else None
            except Exception as e:
                info["cascade_alert_error"] = f"{type(e).__name__}: {e}"
        else:
            info["supabase_unavailable"] = True

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

from auth import get_verified_user_id, verify_api_quota
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
    ], # Añadida la URL de producción de Vercel
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "X-Requested-With",
    ],
)




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

@app.post("/api/webhooks/process-pending-facts")
def api_webhook_process_pending_facts(
    request: Request,
    data: dict = Body(...),
    authorization: Optional[str] = Header(None),
    _rl: Optional[str] = Depends(_WEBHOOK_FACTS_LIMITER),
):
    """
    Endpoint consumido por el Webhook de Supabase (Database Trigger AFTER INSERT en pending_facts_queue).
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
        is_production = os.environ.get("ENVIRONMENT") == "production"

        if not webhook_secret:
            if is_production:
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
            # Extraer token de múltiples fuentes posibles (Supabase custom headers)
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
        # Supabase webhooks mandan la fila en data["record"] cuando es un trigger INSERT
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
                    if supabase:
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
    # Pre-fix tenía `reload=True` hardcoded. En EasyPanel actualmente se
    # arranca via Nixpacks (no `python app.py`), pero un futuro script change
    # podría re-introducir este entry point en producción con el flag activo
    # — auto-reload en prod re-importa módulos cada cambio de archivo y
    # rompe cualquier estado in-process (cache de knobs, _SCHEDULER_*, pools).
    # Default `0` (off); setear `UVICORN_RELOAD=1` solo en dev local.
    _reload_flag = os.environ.get("UVICORN_RELOAD", "0").strip().lower() in ("1", "true", "yes", "on")
    uvicorn.run("app:app", host="0.0.0.0", port=3001, reload=_reload_flag)
