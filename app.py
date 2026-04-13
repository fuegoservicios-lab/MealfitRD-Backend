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
    connection_pool, supabase,
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
                
            logger.info("🚀 [Postgres] Tablas de LangGraph Checkpointer y Push Subscriptions verificadas/creadas.")
        except Exception as e:
            logger.error(f"⚠️ [Postgres] Error configurando DDL inicial: {e}")
            
    if HAS_SCHEDULER and scheduler:
        scheduler.add_job(run_proactive_checks, "cron", minute=30)
        scheduler.start()
        logger.info("⏰ [APScheduler] Tareas proactivas en segundo plano iniciadas.")
            
    logger.info("🚀 [FastAPI] Servidor de MealfitRD IA iniciado con éxito en el puerto 3001.")
    yield
    
    if HAS_SCHEDULER and scheduler:
        scheduler.shutdown()
    
    if connection_pool:
        connection_pool.close()
        logger.info("🔌 [psycopg] Pool de conexiones cerrado.")


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


@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "MealfitRD AI Backend is running"}

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
        "https://mealfit-rd.vercel.app"
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