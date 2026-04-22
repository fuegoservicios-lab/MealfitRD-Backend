from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, UploadFile, File, Form, Body, Header, Depends
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
    update_user_health_profile, get_all_user_facts, delete_user_fact,
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
from graph_orchestrator import run_plan_pipeline
from memory_manager import summarize_and_prune, build_memory_context
from fact_extractor import async_extract_and_save_facts, process_pending_queue_sync
from langgraph.checkpoint.postgres import PostgresSaver
from services import compute_plan_hash, merge_form_data_with_profile
from vision_agent import process_image_with_vision, get_multimodal_embedding

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from proactive_agent import run_proactive_checks
    scheduler = BackgroundScheduler()
    HAS_SCHEDULER = True
except ImportError as e:
    logger.error(f"⚠️ [APScheduler] Falta instalar apscheduler o dependencias: {e}. El agente proactivo está deshabilitado.")
    HAS_SCHEDULER = False
    scheduler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    if connection_pool:
        connection_pool.open()
    if async_connection_pool:
        await async_connection_pool.open()
        
    if connection_pool:
        try:
            import psycopg
            db_uri = os.environ.get("SUPABASE_DB_URL")
            # Setup requires a direct connection with autocommit=True because CREATE INDEX CONCURRENTLY cannot run inside a transaction
            with psycopg.connect(db_uri, autocommit=True) as conn:
                PostgresSaver(conn).setup()
                
                # Crear tabla para Push Subscriptions
                conn.execute("""
                CREATE TABLE IF NOT EXISTS push_subscriptions (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
                    subscription_data JSONB NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """)
                
                # Crear tabla para Nightly Rotation Queue
                conn.execute("""
                CREATE TABLE IF NOT EXISTS nightly_rotation_queue (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
                    status VARCHAR(50) DEFAULT 'pending',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """)
                
                # Asegurar que exista la columna updated_at para tablas antiguas
                conn.execute("""
                ALTER TABLE nightly_rotation_queue
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();
                """)

                # Cola para generación de chunks en background (Background Chunking Just-in-Time)
                conn.execute("""
                CREATE TABLE IF NOT EXISTS plan_chunk_queue (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id UUID REFERENCES user_profiles(id) ON DELETE CASCADE,
                    meal_plan_id UUID NOT NULL REFERENCES meal_plans(id) ON DELETE CASCADE,
                    week_number INT NOT NULL,
                    days_offset INT NOT NULL,
                    days_count INT NOT NULL DEFAULT 3,
                    pipeline_snapshot JSONB NOT NULL,
                    status VARCHAR(50) DEFAULT 'pending',
                    attempts INT DEFAULT 0,
                    execute_after TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """)

                # Migración on-the-fly para la columna execute_after (tablas pre-existentes)
                conn.execute("""
                ALTER TABLE plan_chunk_queue
                ADD COLUMN IF NOT EXISTS execute_after TIMESTAMP WITH TIME ZONE DEFAULT NOW();
                """)

                # Backfill: filas insertadas antes de la migración tienen execute_after=NULL
                # y nunca serían procesadas por `execute_after <= NOW()`. Las marcamos como listas.
                conn.execute("""
                UPDATE plan_chunk_queue
                SET execute_after = NOW()
                WHERE execute_after IS NULL AND status = 'pending';
                """)

                # [GAP A] Métricas de SLA: tier de calidad y lag al hacer pickup
                conn.execute("""
                ALTER TABLE plan_chunk_queue
                ADD COLUMN IF NOT EXISTS quality_tier VARCHAR(20),
                ADD COLUMN IF NOT EXISTS lag_seconds_at_pickup INT,
                ADD COLUMN IF NOT EXISTS escalated_at TIMESTAMP WITH TIME ZONE;
                """)

                # [GAP F] Métricas de aprendizaje inter-chunk (JSON estructurado)
                conn.execute("""
                ALTER TABLE plan_chunk_queue
                ADD COLUMN IF NOT EXISTS learning_metrics JSONB;
                """)

                # [GAP E] Dedup previo de chunks duplicados (meal_plan_id, week_number) antes del UNIQUE
                # Si hay duplicados históricos, conservamos el más reciente (updated_at DESC) y cancelamos
                # los demás. Sin esto, el CREATE UNIQUE INDEX fallaría en tablas pre-existentes.
                try:
                    conn.execute("""
                        UPDATE plan_chunk_queue
                        SET status = 'cancelled', updated_at = NOW()
                        WHERE id IN (
                            SELECT id FROM (
                                SELECT id,
                                       ROW_NUMBER() OVER (
                                           PARTITION BY meal_plan_id, week_number
                                           ORDER BY updated_at DESC, created_at DESC
                                       ) AS rn
                                FROM plan_chunk_queue
                                WHERE status IN ('pending', 'processing', 'stale')
                            ) t
                            WHERE t.rn > 1
                        );
                    """)
                except Exception as _dedup_e:
                    logger.warning(f"[GAP E] No se pudo dedupar chunks previos: {_dedup_e}")

                # [GAP E] UNIQUE parcial: idempotencia a nivel DB. Solo aplica a chunks vivos.
                # chunks 'completed' y 'cancelled' pueden tener duplicados históricos (archivos).
                conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS ux_plan_chunk_queue_live_week
                ON plan_chunk_queue (meal_plan_id, week_number)
                WHERE status IN ('pending', 'processing', 'stale', 'failed');
                """)

                # [GAP G] Tabla de métricas históricas del pipeline de chunks
                conn.execute("""
                CREATE TABLE IF NOT EXISTS plan_chunk_metrics (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    chunk_id UUID,
                    meal_plan_id UUID,
                    user_id UUID,
                    week_number INT,
                    days_count INT,
                    duration_ms INT,
                    quality_tier VARCHAR(20),
                    was_degraded BOOLEAN DEFAULT FALSE,
                    retries INT DEFAULT 0,
                    lag_seconds INT,
                    learning_repeat_pct NUMERIC(5,2),
                    rejection_violations INT DEFAULT 0,
                    allergy_violations INT DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
                """)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_chunk_metrics_recent
                ON plan_chunk_metrics (created_at DESC);
                """)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_chunk_metrics_plan
                ON plan_chunk_metrics (meal_plan_id);
                """)

                # Índice parcial: acelera el polling del worker (status='pending' ordenado por execute_after)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_chunk_queue_pending
                ON plan_chunk_queue (execute_after, created_at)
                WHERE status = 'pending';
                """)

                # Índice para el rescate de zombies (status='processing' con updated_at antiguo)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_chunk_queue_processing
                ON plan_chunk_queue (updated_at)
                WHERE status = 'processing';
                """)

                # [GAP A] Índice para detección rápida de chunks atrasados (stuck)
                conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_chunk_queue_stuck
                ON plan_chunk_queue (execute_after)
                WHERE status IN ('pending', 'stale');
                """)
                
                
            logger.info("🚀 [Postgres] Tablas de LangGraph Checkpointer y Push Subscriptions verificadas/creadas.")
        except Exception as e:
            logger.error(f"⚠️ [Postgres] Error configurando DDL inicial: {e}")
            
    if HAS_SCHEDULER and scheduler:
        scheduler.add_job(run_proactive_checks, "cron", minute=30)
        from cron_tasks import run_nightly_auto_rotation, process_rotation_queue, process_plan_chunk_queue
        scheduler.add_job(run_nightly_auto_rotation, "cron", hour=2, minute=0)
        scheduler.add_job(process_rotation_queue, "interval", minutes=5)
        # max_instances=1 evita solapes si un tick tarda >1 min (ej. por LLM lento).
        # coalesce=True: si se acumulan ticks durante un downtime, solo ejecuta uno al reanudar.
        scheduler.add_job(
            process_plan_chunk_queue,
            "interval",
            minutes=1,
            max_instances=1,
            coalesce=True,
        )
        scheduler.start()
        logger.info("⏰ [APScheduler] Tareas proactivas, CRON jobs nocturnos y Background Chunking iniciados.")
            
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
    return {"status": "ok", "message": "MealfitRD AI Backend is running"}

@app.get("/api/admin/test-proactive")
def api_test_proactive():
    import traceback
    with open("push_log.txt", "w", encoding="utf-8") as f:
        try:
            from test_push import trigger_manual_notification
            import sys, io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            trigger_manual_notification("Almuerzo", "1:30 PM")
            f.write(sys.stdout.getvalue())
            sys.stdout = old_stdout
            return {"status": "success", "message": "Checked log"}
        except Exception as e:
            f.write(traceback.format_exc())
            return {"status": "error", "message": str(e)}

@app.get("/api/admin/test-proactive")
def api_test_proactive(background_tasks: BackgroundTasks):
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

@app.post("/api/cron/nightly-rotation")
def api_trigger_nightly_rotation(
    authorization: Optional[str] = Header(None)
):
    """
    Trigger seguro para Vercel Cron. Dispara la rotación de planes.
    Requiere Bearer token que coincida con CRON_SECRET en el entorno.
    Se ejecuta de forma síncrona para evitar que Vercel mate el proceso background.
    """
    # Validar Bearer token contra CRON_SECRET (Vercel lo inyecta automáticamente)
    cron_secret = os.environ.get("CRON_SECRET")
    if cron_secret:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.replace("Bearer ", "").strip()
        if token != cron_secret:
            raise HTTPException(status_code=403, detail="Invalid cron token")
    else:
        logging.warning("⚠️ CRON_SECRET not set — cron endpoint is unprotected (dev mode)")
    
    from cron_tasks import run_nightly_auto_rotation
    run_nightly_auto_rotation()
    return {"status": "completed", "message": "Nightly rotation queued synchronously"}

@app.post("/api/cron/process-rotation-queue")
def api_trigger_process_rotation_queue(
    authorization: Optional[str] = Header(None)
):
    """
    Trigger para procesar usuarios encolados (para Vercel Cron cada 5 mins).
    Ejecución síncrona para Vercel Serverless.
    """
    cron_secret = os.environ.get("CRON_SECRET")
    if cron_secret:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
        token = authorization.replace("Bearer ", "").strip()
        if token != cron_secret:
            raise HTTPException(status_code=403, detail="Invalid cron token")
            
    from cron_tasks import process_rotation_queue
    process_rotation_queue()
    return {"status": "completed", "message": "Queue processor finished synchronously"}

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
        "https://mealfitrd.com"
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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

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
        raise HTTPException(status_code=500, detail=str(e))

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
                    update_user_health_profile(new_user_id, health_profile)
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
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3001, reload=True)