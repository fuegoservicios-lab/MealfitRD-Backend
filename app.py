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
from typing import Optional
import json
import traceback
import threading
import sentry_sdk

sentry_sdk.init(
    dsn=os.environ.get("SENTRY_DSN"),
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
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
    verify_pipeline_metrics_guest_insert,
)
from memory_manager import summarize_and_prune, build_memory_context
from fact_extractor import async_extract_and_save_facts, process_pending_queue_sync
from langgraph.checkpoint.postgres import PostgresSaver
from services import compute_plan_hash, merge_form_data_with_profile
from vision_agent import process_image_with_vision, get_multimodal_embedding

# [P2-NEW-D · 2026-05-08] Knobs de scheduler. Antes `BackgroundScheduler()` se
# instanciaba sin args; default APScheduler = `ThreadPoolExecutor(max_workers=10)`
# + `misfire_grace_time=1s`. Con ~23 cron jobs registrados (1 en app.py + 22 en
# `register_plan_chunk_scheduler`), un burst en minuto 0 podía saturar el pool
# de 10 threads, encolar el 11º+ y — si su próximo schedule llegaba antes del
# drain — APScheduler hacía SKIP silencioso (logging.warning sin métrica). Riesgo
# crítico para `process_plan_chunk_queue`, `_alert_chunk_dual_processing`, etc.
# Defaults conservadores: 20 workers para absorber el burst sin sobre-provisionar;
# 60s de gracia evita skips por GC/lock contention/DB blip transitorio.
_SCHEDULER_MAX_WORKERS = int(os.environ.get("MEALFIT_SCHEDULER_MAX_WORKERS", "20"))
_SCHEDULER_MISFIRE_GRACE_S = int(os.environ.get("MEALFIT_SCHEDULER_MISFIRE_GRACE_S", "60"))
# [P2-NEW-F · 2026-05-08] Telemetría: emite a system_alerts cuando un job se
# salta (MISSED) o falla (ERROR). Knob ON por default; `off`/`0`/`false` lo
# desactiva. Se lee fresh en cada invocación del listener (`_is_scheduler_telemetry_enabled`)
# para que el kill switch tome efecto sin restart del worker.
def _is_scheduler_telemetry_enabled() -> bool:
    return os.environ.get(
        "MEALFIT_SCHEDULER_TELEMETRY_ENABLED", "on"
    ).strip().lower() in ("1", "true", "yes", "on")

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.executors.pool import ThreadPoolExecutor as _APSThreadPoolExecutor
    from apscheduler.events import EVENT_JOB_MISSED, EVENT_JOB_ERROR
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
    EVENT_JOB_MISSED = EVENT_JOB_ERROR = 0  # type: ignore[assignment]


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
        else:
            return  # otros eventos (EXECUTED, etc.) no nos interesan acá

        if supabase is None:
            return
        alert_key = f"scheduler_{event_type}_{job_id}"
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
    except Exception as listener_err:
        logger.warning(
            f"[P2-NEW-F] Listener de scheduler falló para job="
            f"{getattr(event, 'job_id', '?')}: {listener_err}"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    
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
            
    if HAS_SCHEDULER and scheduler:
        scheduler.add_job(run_proactive_checks, "cron", minute=30)
        # [P2-NEW-C · 2026-05-08] `background_rolling_refill` se movió al SSOT
        # `register_plan_chunk_scheduler` (cron_tasks.py) junto al resto del
        # chunk system. Ya no se registra acá.
        from cron_tasks import register_plan_chunk_scheduler
        register_plan_chunk_scheduler(scheduler)
        # [P2-NEW-F · 2026-05-08] Registrar listener ANTES de start() para no
        # perder los primeros eventos. Mask combinado MISSED|ERROR. Si la
        # telemetría está desactivada por knob, el listener corto-circuita
        # internamente (no skipeamos el add_listener para no requerir
        # restart al togglear el knob — solo edita la env var y los próximos
        # eventos se procesan según el flag actual).
        try:
            scheduler.add_listener(
                _scheduler_alert_listener,
                EVENT_JOB_MISSED | EVENT_JOB_ERROR,
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
    """
    if is_plan_graph_ready():
        return {"status": "ready", "plan_graph": "compiled"}
    raise HTTPException(
        status_code=503,
        detail={
            "status": "not_ready",
            "plan_graph": "not_compiled",
            "message": (
                "LangGraph plan_graph no está compilado. Las requests al "
                "pipeline de generación caerán al fallback matemático. "
                "Revisa los logs CRITICAL del worker para el traceback."
            ),
        },
    )

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

# Setup CORS para el frontend React local
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
    allow_methods=["*"],
    allow_headers=["*"],
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
def api_webhook_process_pending_facts(request: Request, data: dict = Body(...), authorization: Optional[str] = Header(None)):
    """
    Endpoint consumido por el Webhook de Supabase (Database Trigger AFTER INSERT en pending_facts_queue).
    Permite procesar asíncronamente y de manera segura la cola de extracción sin depender de demonios en memoria.
    """
    try:
        # 1. Validación de seguridad robusta
        webhook_secret = os.environ.get("WEBHOOK_SECRET")
        if webhook_secret:
            # Extraer token de múltiples fuentes posibles (Supabase custom headers)
            token = None
            if authorization and authorization.startswith("Bearer "):
                token = authorization.split(" ")[1]
            elif authorization:
                token = authorization
                
            custom_header_secret = request.headers.get("X-Webhook-Secret")
            
            if token != webhook_secret and custom_header_secret != webhook_secret:
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
def api_migrate_guest(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
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
    uvicorn.run("app:app", host="0.0.0.0", port=3001, reload=True)
