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
    update_user_health_profile, get_all_user_facts, delete_user_fact, get_custom_shopping_items,
    get_custom_shopping_items as _get_items, update_custom_shopping_item,
    update_custom_shopping_item_status, delete_custom_shopping_item, clear_all_shopping_items,
    add_custom_shopping_items, uncheck_all_shopping_items, purge_old_shopping_items,
    deduplicate_shopping_items, get_shopping_plan_hash, save_new_meal_plan_robust,
    log_consumed_meal, get_consumed_meals_today, save_visual_entry, get_session_messages,
    get_user_chat_sessions, get_guest_chat_sessions, get_session_owner, delete_user_agent_sessions,
    delete_single_agent_session, update_session_title,
    check_fact_ownership, upsert_user_profile, migrate_guest_data, log_api_usage, get_monthly_api_usage
)
from agent import (
    swap_meal, chat_with_agent, analyze_preferences_agent,
    generate_chat_title_background, generate_auto_shopping_list, chat_with_agent_stream
)
from ai_helpers import generate_plan_title, expand_recipe_agent
from graph_orchestrator import run_plan_pipeline
from memory_manager import summarize_and_prune, build_memory_context
from fact_extractor import async_extract_and_save_facts, process_pending_queue_sync
from langgraph.checkpoint.postgres import PostgresSaver
from services import compute_plan_hash, merge_form_data_with_profile, regenerate_shopping_list_safe
from constants import categorize_shopping_item
from vision_agent import process_image_with_vision, get_multimodal_embedding

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
            logger.info("🚀 [Postgres] Tablas de LangGraph Checkpointer verificadas/creadas.")
        except Exception as e:
            logger.error(f"⚠️ [Postgres] Error configurando checkpointer: {e}")
            
    logger.info("🚀 [FastAPI] Servidor de MealfitRD IA iniciado con éxito en el puerto 3001.")
    yield
    
    if connection_pool:
        connection_pool.close()
        logger.info("🔌 [psycopg] Pool de conexiones cerrado.")


# Asegurarnos de que el directorio de uploads exista antes de montar recursos estáticos
os.makedirs("uploads", exist_ok=True)

import re as _re
def sanitize_shopping_text(text: str, max_length: int = 100) -> str:
    """Escapa y sanitiza inputs en la lista de compras previniendo XSS ciego y control chars destructivos."""
    clean = _re.sub(r'<[^>]+>', '', text).strip()
    clean = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)
    return clean[:max_length]

app = FastAPI(lifespan=lifespan)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "MealfitRD AI Backend is running"}

from auth import get_verified_user_id, verify_api_quota
from rate_limiter import RateLimiter, _shopping_write_limiter, _shopping_autogen_limiter
from services import _preserve_shopping_checkmarks, _save_plan_and_track_background, _process_swap_rejection_background

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

@app.post("/api/analyze")
def api_analyze(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(verify_api_quota)):
    try:
        session_id = data.get("session_id")
        user_id = data.get("user_id") # Para buscar likes (si está logueado)
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        history = []
        likes = []
        taste_profile = ""
        
        if session_id:
            get_or_create_session(session_id)
            # Usar sistema de memoria inteligente (resúmenes + mensajes recientes)
            memory = build_memory_context(session_id)
            history = memory["recent_messages"]
            
        actual_user_id = user_id if user_id and user_id != "guest" else None
        if actual_user_id:
            likes = get_user_likes(actual_user_id)

        # 1. Obtener rechazos activos (últimos 7 días solamente)
        active_rejections = get_active_rejections(user_id=actual_user_id, session_id=session_id)
        rejected_meal_names = [r["meal_name"] for r in active_rejections] if active_rejections else []
            
        # 2. Llamar al Agente Especialista en Preferencias (con rechazos temporales)
        taste_profile = analyze_preferences_agent(likes, history, active_rejections=rejected_meal_names)
            
        # 3. Ejecutar Pipeline Multi-Agente (LangGraph: Generador → Revisor Médico)
        # Pasar el contexto completo (resúmenes + recientes) al pipeline
        result = run_plan_pipeline(data, history, taste_profile, 
                                   memory_context=memory.get("full_context_str", "") if session_id else "")
        
        # 4. Persistir los datos del formulario en user_profiles.health_profile
        if actual_user_id:
            hp_data = {k: v for k, v in data.items() if k not in ['session_id', 'user_id']}
            if hp_data:
                update_user_health_profile(actual_user_id, hp_data)
                logger.info(f"💾 [SYNC] health_profile guardado para user {actual_user_id}")
        
        if session_id:
            goal = data.get('mainGoal', 'Desconocido')
            save_message(session_id, "user", f"Generar plan para mi objetivo: {goal}")
            save_message(session_id, "model", "¡Aquí tienes tu estrategia nutricional personalizada generada analíticamente!")
            
            # 🧠 Background: Resumir y podar mensajes si el historial creció demasiado
            background_tasks.add_task(summarize_and_prune, session_id)
            
        # 👇 NUEVO: Registramos uso de API de Gemini
        if actual_user_id:
            log_api_usage(actual_user_id, "gemini_analyze")
            
        # 👇 NUEVO: Guardar el plan generado y trackear frecuencias en Background
        # Extraer técnicas ANTES del background (dicts son by-reference, el pop debe ser previo)
        selected_techniques = result.pop("_selected_techniques", None)
        if actual_user_id:
            background_tasks.add_task(_save_plan_and_track_background, actual_user_id, result, selected_techniques)

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ [ERROR] Error en /api/analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/recipe/expand")
def api_expand_recipe(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        # Validación opcional de seguridad
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        # data contendrá la receta entera (name, desc, ingredients, recipe, cals)
        if not data.get("recipe") or not data.get("name"):
            raise HTTPException(status_code=400, detail="Faltan datos de la receta para expandir.")
            
        if user_id and user_id != "guest":
            log_api_usage(user_id, "gemini_recipe_expand")
            
        expanded_steps = expand_recipe_agent(data)
        
        # Persistencia Automática en la DB
        if user_id and user_id != "guest":
            from db import get_latest_meal_plan, get_latest_meal_plan_with_id, update_meal_plan_data
            current_plan = get_latest_meal_plan(user_id)
            if current_plan and "days" in current_plan:
                updated = False
                for day in current_plan.get("days", []):
                    for m in day.get("meals", []):
                        if m.get("name") == data.get("name"):
                            m["recipe"] = expanded_steps
                            m["isExpanded"] = True
                            updated = True
                            break
                    if updated: break
                
                if updated:
                    plan_with_id = get_latest_meal_plan_with_id(user_id)
                    if plan_with_id and "id" in plan_with_id:
                        update_meal_plan_data(plan_with_id["id"], current_plan)

        return {"success": True, "expanded_recipe": expanded_steps}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/recipe/expand: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/swap-meal")
def api_swap_meal(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        
        # Guardar en memoria el rechazo para que el Agente de Preferencias aprenda
        session_id = data.get("session_id")
        user_id = data.get("user_id")
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        rejected_meal = data.get("rejected_meal")
        meal_type = data.get("meal_type", "")
        
        # 👇 NUEVO: Mover logueos DB a Background Tasks (incluye fricción silenciosa)
        if rejected_meal:
            background_tasks.add_task(_process_swap_rejection_background, session_id, user_id, rejected_meal, meal_type)
            
        if user_id and user_id != "guest":
            log_api_usage(user_id, "gemini_swap_meal")
            
        result = swap_meal(data)
        return result
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/swap-meal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/like")
def api_like(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
                
        insert_like(data)
        return {"success": True, "message": "Tu like/dislike ha sido guardado exitosamente."}
    except Exception as e:
        return {"error": str(e)}

import httpx

@app.post("/api/subscription/verify")
async def api_verify_subscription(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    try:
        user_id = data.get("user_id")
        subscription_id = data.get("subscriptionID")
        tier = data.get("tier")

        if not user_id or not subscription_id or not tier:
            raise HTTPException(status_code=400, detail="Missing parameters")

        if not verified_user_id or verified_user_id != user_id:
            logger.warning(f"Intento no autorizado de verificar suscripcion: req_user={user_id}, auth_user={verified_user_id}")
            raise HTTPException(status_code=401, detail="No autorizado.")

        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        # Por defecto asumo live, a menos que especifique ambiente testing
        is_sandbox = os.environ.get("ENVIRONMENT") != "production"
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"

        access_token = None
        success = False

        if not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
            # Bypass en desarrollo si no hay llaves aún configuradas (para no bloquearte las pruebas locales iniciales)
            logger.warning("⚠️ No Paypal keys found in backend .env. Bypassing real validation! (SECURITY RISK IF PRODUCTION)")
            success = True
        else:
            # 1. Obtener Token OAuth de PayPal
            async with httpx.AsyncClient() as client:
                auth_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/oauth2/token",
                    auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                    data={"grant_type": "client_credentials"}
                )
                if auth_resp.status_code != 200:
                    logger.error(f"Error auth Paypal: {auth_resp.text}")
                    raise HTTPException(status_code=500, detail="Error de autenticación con proveedor de pagos.")
                
                access_token = auth_resp.json().get("access_token")

                # 2. Verificar el estado de la suscripción
                sub_resp = await client.get(
                    f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}",
                    headers={"Authorization": f"Bearer {access_token}"}
                )
                if sub_resp.status_code != 200:
                    logger.error(f"Error fetching sub: {sub_resp.text}")
                    raise HTTPException(status_code=400, detail="La suscripción no fue recocida por PayPal.")
                
                sub_data = sub_resp.json()
                status = sub_data.get("status")

                if status != "ACTIVE":
                    raise HTTPException(status_code=400, detail=f"Suscripción no válida. Estado actual: {status}")

                success = True

        # 3. Si PayPal dice que todo está bien, actualizamos localmente al usuario.
        if success:
            logger.info(f"✅ Subscripcion Verificada B2B ({subscription_id}) para usuario {user_id}. Asignando tier: {tier}")
            
            # --- PROTECCIÓN CONTRA PAGOS DOBLES (UPGRADES/CROSSGRADES) ---
            # Si el usuario ya tiene una suscripcion activa y es distinta a esta, la cancelamos en PayPal.
            existing_res = supabase.table("user_profiles").select("paypal_subscription_id, subscription_status").eq("id", user_id).execute()
            if existing_res.data and len(existing_res.data) > 0:
                old_sub_id = existing_res.data[0].get("paypal_subscription_id")
                old_status = existing_res.data[0].get("subscription_status")
                
                if old_sub_id and old_sub_id != subscription_id and old_status != "INACTIVE":
                    logger.info(f"🔄 Detectado Upgrade/Cambio. Cancelando suscripción antigua {old_sub_id} en PayPal...")
                    if access_token:
                        async with httpx.AsyncClient() as cancel_client:
                            cancel_resp = await cancel_client.post(
                                f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{old_sub_id}/cancel",
                                headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                                json={"reason": "Upgrade o cambio a una nueva suscripción."}
                            )
                            if cancel_resp.status_code == 204:
                                logger.info(f"✅ Suscripción antigua {old_sub_id} cancelada exitosamente en PayPal.")
                            else:
                                logger.warning(f"⚠️ Error intentando cancelar suscripción antigua {old_sub_id}: {cancel_resp.text}")
            
            # Ejecutamos el update en Supabase.
            res = supabase.table("user_profiles").update({
                "plan_tier": tier,
                "paypal_subscription_id": subscription_id, 
                "subscription_status": "ACTIVE"
            }).eq("id", user_id).execute()
            
            return {"success": True, "message": "Suscripción verificada y plan actualizado B2B."}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error interno en /api/subscription/verify: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/subscription/cancel")
async def api_cancel_subscription(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    """ Endpoint que el cliente llama para cancelar voluntariamente su suscripción """
    try:
        user_id = data.get("user_id")
        
        if not verified_user_id or verified_user_id != user_id:
            logger.warning(f"Intento no autorizado de cancelar suscripcion: req_user={user_id}, auth_user={verified_user_id}")
            raise HTTPException(status_code=401, detail="No autorizado.")
            
        # Obtener el ID de la suscripción de la DB
        res = supabase.table("user_profiles").select("paypal_subscription_id").eq("id", user_id).execute()
        if not res.data or not res.data[0].get("paypal_subscription_id"):
            raise HTTPException(status_code=400, detail="No active subscription found to cancel.")
            
        subscription_id = res.data[0]["paypal_subscription_id"]
        
        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        is_sandbox = os.environ.get("ENVIRONMENT") != "production"
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"
        
        end_date = None
        
        if PAYPAL_CLIENT_ID and PAYPAL_SECRET:
            async with httpx.AsyncClient() as client:
                auth_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/oauth2/token",
                    auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                    data={"grant_type": "client_credentials"}
                )
                if auth_resp.status_code == 200:
                    access_token = auth_resp.json().get("access_token")
                    
                    # Llamada a PayPal para obtener next_billing_time ANTES de cancelar
                    sub_info_resp = await client.get(
                        f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}",
                        headers={"Authorization": f"Bearer {access_token}"}
                    )
                    if sub_info_resp.status_code == 200:
                        billing_info = sub_info_resp.json().get("billing_info", {})
                        end_date = billing_info.get("next_billing_time")
                    
                    # Llamada a PayPal para cancelar la suscripción
                    cancel_resp = await client.post(
                        f"{PAYPAL_API_BASE}/v1/billing/subscriptions/{subscription_id}/cancel",
                        headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                        json={"reason": "El usuario solicitó la cancelación desde la App"}
                    )
                    
                    # 204 No Content indica éxito
                    if cancel_resp.status_code not in [204, 200]:
                        logger.error(f"Error cancelando suscripcion con PayPal: {cancel_resp.text}")
                        # Fallback pasivo continua...
        
        # Graceful Degradation: Dejamos el tier intacto pero marcamos el status como CANCELLED e insertamos el Fin de Ciclo
        logger.info(f"✅ Suscripción {subscription_id} de usuario {user_id} cancelada. Mantendrá acceso hasta {end_date or 'fin de ciclo'}.")
        
        update_payload = {"subscription_status": "CANCELLED"}
        if end_date:
            update_payload["subscription_end_date"] = end_date
            
        supabase.table("user_profiles").update(update_payload).eq("id", user_id).execute()
        
        return {"success": True, "message": "Tu suscripción no se renovará, pero mantendrás tu plan actual hasta el final del ciclo pagado."}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error en /api/subscription/cancel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/webhooks/paypal")
async def api_webhook_paypal(request: Request):
    """ Webhook pasivo que escucha eventos silenciosos de PayPal (Tarjetas denegadas o cancelaciones remotas) """
    try:
        PAYPAL_CLIENT_ID = os.environ.get("PAYPAL_CLIENT_ID")
        PAYPAL_SECRET = os.environ.get("PAYPAL_SECRET")
        PAYPAL_WEBHOOK_ID = os.environ.get("PAYPAL_WEBHOOK_ID")
        is_sandbox = os.environ.get("ENVIRONMENT") != "production"
        PAYPAL_API_BASE = "https://api-m.sandbox.paypal.com" if is_sandbox else "https://api-m.paypal.com"
        
        body = await request.body()
        headers = request.headers
        
        if not PAYPAL_WEBHOOK_ID or not PAYPAL_CLIENT_ID or not PAYPAL_SECRET:
            logger.warning("⚠️ PAYPAL_WEBHOOK_ID o llaves no configuradas. Saltando verificación de firma en webhook.")
            if not is_sandbox:
                raise HTTPException(status_code=400, detail="Missing webhook config")
                
        payload_dict = json.loads(body.decode('utf-8'))
        
        # Validación CRÍTICA: Confirmar usando la API de PayPal que este webhook es legítimo
        if PAYPAL_CLIENT_ID and PAYPAL_SECRET and PAYPAL_WEBHOOK_ID:
            async with httpx.AsyncClient() as client:
                auth_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/oauth2/token",
                    auth=(PAYPAL_CLIENT_ID, PAYPAL_SECRET),
                    data={"grant_type": "client_credentials"}
                )
                if auth_resp.status_code != 200:
                    logger.error("Error autenticando con PayPal en webhook")
                    return {"success": False}
                
                access_token = auth_resp.json().get("access_token")
                
                verify_payload = {
                    "auth_algo": headers.get("paypal-auth-algo"),
                    "cert_url": headers.get("paypal-cert-url"),
                    "transmission_id": headers.get("paypal-transmission-id"),
                    "transmission_sig": headers.get("paypal-transmission-sig"),
                    "transmission_time": headers.get("paypal-transmission-time"),
                    "webhook_id": PAYPAL_WEBHOOK_ID,
                    "webhook_event": payload_dict
                }
                
                verify_resp = await client.post(
                    f"{PAYPAL_API_BASE}/v1/notifications/verify-webhook-signature",
                    headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
                    json=verify_payload
                )
                
                if verify_resp.status_code != 200 or verify_resp.json().get("verification_status") != "SUCCESS":
                    logger.warning(f"🔒 Bloqueado: Firma del Webhook PayPal inválida: {verify_resp.text}")
                    return {"success": False, "message": "Signature mismatch"}
        
        event_type = payload_dict.get("event_type")
        resource = payload_dict.get("resource", {})
        
        logger.info(f"⚡ [WEBHOOK PAYPAL] Evento recibido: {event_type}")
        
        # Eventos que implican remover el acceso al plan
        downgrade_events = [
            "BILLING.SUBSCRIPTION.SUSPENDED", 
            "BILLING.SUBSCRIPTION.EXPIRED",
            "BILLING.SUBSCRIPTION.PAYMENT.FAILED"
        ]
        
        if event_type in downgrade_events:
            subscription_id = resource.get("id")
            if subscription_id:
                logger.info(f"⬇️ Degradando suscripción {subscription_id} en BD debido a {event_type}.")
                supabase.table("user_profiles").update({
                    "plan_tier": "gratis",
                    "subscription_status": "INACTIVE"
                }).eq("paypal_subscription_id", subscription_id).execute()
        elif event_type == "BILLING.SUBSCRIPTION.CANCELLED":
            subscription_id = resource.get("id")
            end_date = resource.get("billing_info", {}).get("next_billing_time")
            
            if subscription_id:
                logger.info(f"ℹ️ Webhook: Suscripción {subscription_id} cancelada remota/silenciosamente. Acceso hasta: {end_date or 'fin de ciclo'}.")
                
                update_payload = {"subscription_status": "CANCELLED"}
                if end_date:
                    update_payload["subscription_end_date"] = end_date
                    
                supabase.table("user_profiles").update(update_payload).eq("paypal_subscription_id", subscription_id).execute()
        
        return {"success": True}

    except Exception as e:
        logger.error(f"❌ Error procesando webhook PayPal: {e}")
        return {"success": False}


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

@app.get("/api/chat/sessions/{user_id}")
def api_get_chat_sessions(user_id: str, session_ids: Optional[str] = None, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        sessions: list = get_user_chat_sessions(user_id) or []
        
        # Siempre leer los session_ids del frontend (localStorage) como capa de seguridad. 
        # Si la BD no tiene la columna user_id, los sessions de arriba regresan vacíos, pero aquí los recuperamos.
        if session_ids:
            guest_sessions = get_guest_chat_sessions(session_ids.split(","))
            if guest_sessions:
                # Merge lists deduplicating by 'id'
                existing_ids = {s["id"] for s in sessions}
                for gs in guest_sessions:
                    if gs["id"] not in existing_ids:
                        sessions.append(gs)
                        
        # Sort again by last_activity descending after merge
        sessions.sort(key=lambda x: x.get("last_activity") or x.get("created_at") or "1970-01-01T00:00:00", reverse=True)
            
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/sessions GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/sessions/{user_id}")
def api_delete_chat_sessions(user_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
            delete_user_agent_sessions(user_id)
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/sessions DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


from pydantic import BaseModel
class RenameSessionReq(BaseModel):
    title: str

@app.put("/api/chat/session/{session_id}")
def api_rename_chat_session(session_id: str, data: RenameSessionReq, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        session_owner = get_session_owner(session_id)
        if session_owner and session_owner != "guest":
            if not verified_user_id or verified_user_id != session_owner:
                raise HTTPException(status_code=403, detail="Prohibido.")
        update_session_title(session_id, data.title)
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/session PUT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history/{session_id}")
def api_get_chat_history(session_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    try:
        # 🛡️ Validación IDOR: Verificar que el session pertenece al usuario autenticado
        session_owner = get_session_owner(session_id)
        if session_owner and session_owner != "guest":
            if not verified_user_id or verified_user_id != session_owner:
                logger.warning(f"🚫 [HISTORY AUTH FAILED] REJECTED. owner={session_owner} != verified={verified_user_id}")
                raise HTTPException(status_code=403, detail="Prohibido. No tienes acceso a esta conversación.")

        messages = get_session_messages(session_id)
        # Ocultar mensajes de sistema como el system_title
        filtered_messages = [m for m in messages if not m.get("content", "").startswith("[SYSTEM_TITLE]")]
        return {"messages": filtered_messages}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/history GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/chat/session/{session_id}")
def api_delete_chat_session(session_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """Elimina una sesión de chat. Requiere autenticación pero sin validación IDOR 
    (RLS desactivado — la auth se maneja aquí)."""
    from db import delete_chat_session
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="Token requerido para eliminar chats.")
        
        success, error_msg = delete_chat_session(session_id)
        if success:
            logger.info(f"🗑️ Chat {session_id} eliminado por usuario {verified_user_id}")
            return {"success": True, "message": "Chat eliminado correctamente."}
        else:
            logger.error(f"❌ Fallo al eliminar chat {session_id}: {error_msg}")
            raise HTTPException(status_code=500, detail=f"Error: {error_msg}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en DELETE chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/message")
def api_save_chat_message(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    session_id = data.get("session_id")
    role = data.get("role")
    content = data.get("content")
    user_id = data.get("user_id", session_id)
    
    # Validación de seguridad IDOR
    if user_id and user_id != "guest" and user_id != session_id:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=401, detail="No autorizado. Token inválido o no coincide.")
            
    if session_id and role and content:
        get_or_create_session(session_id, user_id=user_id if user_id != "guest" else None)
        save_message(session_id, role, content)
        return {"success": True}
    return {"success": False, "error": "Faltan parámetros"}

from fastapi.responses import StreamingResponse
import asyncio

@app.post("/api/chat/feedback")
async def api_chat_feedback(data: dict = Body(...), verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    session_id = data.get("session_id")
    content = data.get("content")
    feedback = data.get("feedback")
    
    if not session_id or not content:
        raise HTTPException(status_code=400, detail="Missing session_id or content")

    success = await asyncio.to_thread(save_message_feedback, session_id, content, feedback)
    if success:
        return {"success": True}
    else:
        raise HTTPException(status_code=500, detail="Error saving feedback")

@app.post("/api/chat/stream")
def api_chat_stream(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: str = Depends(verify_api_quota)):
    try:
        session_id = data.get("session_id", "default_session")
        prompt = data.get("prompt", "")
        user_id = data.get("user_id", session_id)
        current_plan = data.get("current_plan", None)
        form_data = data.get("form_data", None)
        local_date = data.get("local_date", None)
        tz_offset = data.get("tz_offset", None)
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest" and user_id != session_id:
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        logger.info(f"🔍 [DEBUG API CHAT STREAM] session_id={session_id}, user_id={user_id}")
        
        # Operaciones síncronas directas (ya estamos en un threadpool worker)
        get_or_create_session(session_id, user_id=user_id if user_id != "guest" else None)
        save_message(session_id, "user", prompt)
        
        # Handle form_data: merge frontend data with DB health_profile (DRY — shared in services.py)
        form_data = merge_form_data_with_profile(
            user_id if user_id != "guest" and user_id != session_id else "",
            form_data
        )
        
        if not current_plan and user_id and user_id != "guest":
            current_plan = get_latest_meal_plan(user_id)
            
        
        # Iniciar generación del título de inmediato en paralelo
        threading.Thread(
            target=generate_chat_title_background,
            args=(user_id, session_id, prompt),
            daemon=True
        ).start()
        
        def event_generator():
            try:
                for chunk in chat_with_agent_stream(
                    session_id=session_id, 
                    prompt=prompt, 
                    current_plan=current_plan, 
                    user_id=user_id, 
                    form_data=form_data,
                    local_date=local_date,
                    tz_offset=tz_offset
                ):
                    yield chunk
                    
                    # Interceptar el evento 'done' para lanzar background tasks
                    if chunk.startswith("data: "):
                        try:
                            data_obj = json.loads(chunk[len("data: "):].strip())
                            if data_obj.get("type") == "done":
                                response_text = data_obj.get("response", "")
                                if response_text:
                                    save_message(session_id, "model", response_text)
                                    
                                # Lógica Background (resumir, uso de API, embeddings)
                                def bg_tasks():
                                    if user_id and user_id != "guest" and user_id != session_id:
                                        log_api_usage(user_id, "gemini_chat")
                                        
                                    try:
                                        raw_history = get_session_messages(session_id)
                                        recent_history_str = ""
                                        if raw_history:
                                            recent_history_str = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in raw_history[-6:]])
                                        
                                        is_plus = False
                                        if user_id and user_id != "guest":
                                            profile_sync = get_user_profile(user_id)
                                            if profile_sync:
                                                plan_tier_sync = profile_sync.get("plan_tier", "gratis")
                                                is_plus = plan_tier_sync in ["basic", "plus", "admin", "ultra"]
                                                
                                        if is_plus:
                                            async_extract_and_save_facts(user_id, prompt, recent_history_str)
                                            
                                        summarize_and_prune(session_id)
                                    except Exception as inner_e:
                                        logger.error(f"Error en bg tasks: {inner_e}")
                                
                                threading.Thread(target=bg_tasks, daemon=True).start()
                        except Exception as e_json:
                            logger.error(f"Error parseando chunk de fin: {e_json}")
                            
            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def api_chat(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: str = Depends(verify_api_quota)):
    try:
        session_id = data.get("session_id", "default_session")
        prompt = data.get("prompt", "")
        user_id = data.get("user_id", session_id)
        current_plan = data.get("current_plan", None)
        form_data = data.get("form_data", None)
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest" and user_id != session_id:
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        logger.info(f"🔍 [DEBUG API CHAT] session_id={session_id}, user_id={user_id}")
        
        get_or_create_session(session_id, user_id=user_id if user_id != "guest" else None)
        save_message(session_id, "user", prompt)
        
        # Handle form_data: merge frontend data with DB health_profile (DRY — shared in services.py)
        form_data = merge_form_data_with_profile(
            user_id if user_id != "guest" and user_id != session_id else "",
            form_data
        )
        
        if not current_plan and user_id and user_id != "guest":
            current_plan = get_latest_meal_plan(user_id)
        
        response_text, updated_fields, new_plan = chat_with_agent(session_id, prompt, current_plan=current_plan, user_id=user_id, form_data=form_data)
        
        save_message(session_id, "model", response_text)
        
        # 🧠 Background: Resumir y podar mensajes si el historial creció demasiado
        background_tasks.add_task(summarize_and_prune, session_id)
        
        if user_id and user_id != "guest" and user_id != session_id:
            log_api_usage(user_id, "gemini_chat")
        
        # === CONTEXTO PARA HECHOS (Debounce Semántico) ===
        # Obtenemos el historial de la sesión para darle contexto al LLM extractor
        raw_history = get_session_messages(session_id)
        recent_history_str = ""
        if raw_history:
            # Tomar solo los últimos 6 mensajes para contexto rápido
            recent_history_str = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in raw_history[-6:]])
        
        # Verificar tier para usar la Memoria a Largo Plazo
        is_plus = False
        if user_id and user_id != "guest":
            profile = get_user_profile(user_id)
            if profile:
                plan_tier = profile.get("plan_tier", "gratis")
                is_plus = plan_tier in ["basic", "plus", "admin", "ultra"]
        
        if is_plus:
            # 🧠 Background: Extraer hechos y vectorizarlos
            background_tasks.add_task(async_extract_and_save_facts, user_id, prompt, recent_history_str)
        else:
            logger.info("INFO: Memoria a Largo Plazo deshabilitada para usuario Gratis.")
        
        # 🧠 Background: Generar un título si es el primer mensaje
        background_tasks.add_task(generate_chat_title_background, user_id, session_id, prompt)
        
        result = {"response": response_text, "updated_fields": updated_fields}
        if new_plan:
            result["new_plan"] = new_plan
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diary/upload")
async def api_diary_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Form("guest"),
    session_id: str = Form(None),
    tz_offset_mins: int = Form(0),
    verified_user_id: str = Depends(verify_api_quota)
):
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest" and user_id != session_id:
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        actual_user_id = user_id if user_id != "guest" else session_id

        
        MAX_FILE_SIZE = 20 * 1024 * 1024 # 20 MB
        file_bytes = b""
        while chunk := await file.read(1024 * 1024):
            file_bytes += chunk
            if len(file_bytes) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="La imagen es demasiado grande. Máximo 20MB permitidos.")
        
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        unique_filename = f"{actual_user_id}/{uuid.uuid4().hex}.{file_ext}"
        
        image_url = ""
        upload_success = False

        # 1. Intentar subir a Supabase Storage
        if supabase:
            try:
                res = await asyncio.to_thread(
                    supabase.storage.from_("visual_diary_images").upload,
                    path=unique_filename,
                    file=file_bytes,
                    file_options={"content-type": file.content_type}
                )
                image_url = supabase.storage.from_("visual_diary_images").get_public_url(unique_filename)
                upload_success = True
                logger.info(f"☁️ Imagen guardada en Supabase: {image_url}")
            except Exception as sb_err:
                logger.error(f"⚠️ Error subiendo a Supabase (¿Existe el bucket 'visual_diary_images'?): {sb_err}")
                upload_success = False

        # 2. Si no se pudo subir a Supabase, fallar (evitar guardar localmente en la nube)
        if not upload_success:
            logger.error("❌ No se pudo subir la imagen a Supabase. Abortando.")
            raise HTTPException(status_code=500, detail="Error uploading image to cloud storage.")
            
            
        # 3. Procesar imagen con Visión SINCRÓNICAMENTE
        logger.info("\n-------------------------------------------------------------")
        logger.info("📸 [VISION AGENT] Procesando nueva imagen subida...")
        vision_result = await process_image_with_vision(file_bytes)
        
        description = vision_result.get("description", "No se pudo analizar la imagen.")
        is_food = vision_result.get("is_food", False)
        calories = vision_result.get("calories", 0)
        
        if is_food:
            logger.info(f"✅ Descripción generada: '{description}' (Calorías: {calories})")
            
            # --- EVALUACIÓN DETERMINISTA DE CRONONUTRICIÓN (PYTHON) ---
            if calories > 500:
                from datetime import datetime, timedelta
                # Calcular la hora exacta del usuario usando el offset proporcionado por el frontend
                local_time = datetime.utcnow() - timedelta(minutes=tz_offset_mins)
                local_hour = local_time.hour
                
                # Obtener el perfil para ver el tipo de turno
                schedule_type = "standard"
                if user_id != "guest":
                    profile = get_user_profile(user_id)
                    if profile and profile.get("health_profile"):
                        schedule_type = profile["health_profile"].get("scheduleType", "standard")
                
                # Reglas matemáticas duras e inquebrantables
                is_red_alert = False
                if schedule_type == "standard" and (0 <= local_hour < 6):
                    is_red_alert = True
                elif schedule_type == "night_shift" and (14 <= local_hour < 20):  # Ej: 2 PM a 8 PM
                    is_red_alert = True
                
                if is_red_alert:
                    poison_pill = " [⚠️ INSTRUCCIÓN CLÍNICA PRIORITARIA DE SISTEMA: El análisis de esta imagen indica que el usuario está a punto de consumir más de 500 kcal en una hora sumamente crítica para su reloj biológico (madrugada o contra-turno). TIENES LA ORDEN DIRECTA Y OBLIGATORIA de cambiar tu tono y emitir inmediatamente una advertencia y reprimenda constructiva sobre el grave impacto digestivo de comer así a esta hora.]"
                    description += poison_pill
                    logger.warning(f"🚨 [RULE POISONING] Inyectando alerta roja determinista. Hora local: {local_hour}, Calorías: {calories}, Turno: {schedule_type}")
            # ------------------------------------------------------------
            
            if actual_user_id and actual_user_id != session_id:
                log_api_usage(actual_user_id, "gemini_vision")
                
            # 4. Guardar en DB en segundo plano (embedding + insert)
            background_tasks.add_task(
                _save_visual_entry_background,
                actual_user_id, image_url, description
            )
        else:
            logger.info("➡️ La imagen fue ignorada porque no se detectaron alimentos.")
        
        return {
            "success": True, 
            "is_food": is_food,
            "description": description,
            "image_url": image_url
        }
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _save_visual_entry_background(user_id: str, image_url: str, description: str):
    """Background task: genera embedding y guarda en la tabla visual_diary."""
    
    embedding = get_multimodal_embedding(description)
    if embedding:
        logger.info(f"📦 Guardando entrada visual en la DB (Vector 768d)...")
        save_visual_entry(user_id=user_id, image_url=image_url, description=description, embedding=embedding)
        logger.info("✅ ¡Imagen registrada en el Diario Visual con éxito!")
    else:
        logger.warning("⚠️ No se pudo vectorizar la imagen. Abortando guardado.")

@app.post("/api/shopping/auto-generate")
def api_shopping_auto_generate(data: dict = Body(...), verified_user_id: str = Depends(_shopping_autogen_limiter)):
    """Genera y guarda la lista de compras consolidada a partir del plan activo de 3 días.
    Usa cache por hash del plan: si el plan no cambió, retorna la lista existente sin llamar al LLM."""
    try:
        user_id = data.get("user_id")
        force = data.get("force", False)  # Forzar regeneración incluso si el plan no cambió
        days = data.get("days", 7) # Multiplicador de días
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido. Token inválido o no coincide.")
                
        if not user_id or user_id == "guest":
            raise HTTPException(status_code=400, detail="Usuario no válido para auto-generar lista.")
            
        current_plan = get_latest_meal_plan(user_id)
        
        if not current_plan:
            raise HTTPException(status_code=404, detail="No se encontró un plan activo para extraer ingredientes.")
        
        # Calcular hash del plan para detectar cambios (SSoT: compute_plan_hash en services.py)
        plan_hash = compute_plan_hash(current_plan)
        
        # Verificar si el plan ya fue procesado y si la lista sigue vigente
        if not force:
            stored_hash = get_shopping_plan_hash(user_id)
            existing = get_custom_shopping_items(user_id)
            existing_items = existing.get("data", []) if isinstance(existing, dict) else existing
            
            # Bloqueo Automático: Si ya hay items, calculamos si todavía no expira según los días seleccionados
            if existing_items:
                try:
                    from datetime import datetime, timezone
                    created_at_strs = [i.get("created_at") for i in existing_items if i.get("created_at")]
                    if created_at_strs:
                        oldest_str = min(created_at_strs)
                        if oldest_str.endswith("Z"):
                            oldest_str = oldest_str[:-1] + "+00:00"
                        created_dt = datetime.fromisoformat(oldest_str)
                        if created_dt.tzinfo is None:
                            created_dt = created_dt.replace(tzinfo=timezone.utc)
                            
                        days_elapsed = (datetime.now(timezone.utc) - created_dt).days
                        
                        # Si los días transcurridos son menores a los días seleccionados, BLOQUEAR regeneración (así cambie el plan)
                        if days_elapsed < days:
                            logger.info(f"🔒 [LISTA BLOQUEADA] Vigente: {days_elapsed}/{days} días (Ignorando cambios en plan).")
                            return {"success": True, "items": existing_items, "cached": True, "locked": True,
                                    "message": f"Lista bloqueada para preservar tus compras de {days} días."}
                except Exception as e:
                    logger.error(f"Error calculando expiración: {e}")
            
            # Si expiró o falló la verificación de tiempo, validamos por hash clásico
            if stored_hash == plan_hash:
                logger.info(f"✅ [CACHE HIT] Plan sin cambios (hash={plan_hash}). Retornando lista existente.")
                return {"success": True, "items": existing_items, "cached": True, "locked": False,
                        "message": "La lista ya está actualizada con tu plan actual."}
            
        items = generate_auto_shopping_list(current_plan)
        
        if not items:
            return {"success": False, "message": "No se encontraron ingredientes para consolidar en el plan activo."}
            
        existing = _get_items(user_id)
        existing_items = existing.get("data", []) if isinstance(existing, dict) else existing
        
        result = regenerate_shopping_list_safe(user_id, items, existing_items, plan_hash)
        
        if result is not None:
            return {"success": True, "items": items, "cached": False, "message": f"Se auto-generaron y guardaron {len(items)} ingredientes estructurados en tu lista con éxito."}
        else:
            raise HTTPException(status_code=500, detail="Error al intentar guardar los ingredientes en la base de datos.")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ [ERROR] Error en /api/shopping/auto-generate POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/shopping/custom")
def api_add_custom_shopping_item(data: dict = Body(...), verified_user_id: str = Depends(_shopping_write_limiter)):
    """Añade uno o más items a la lista de compras manualmente desde el frontend."""
    try:
        user_id = data.get("user_id")
        items = data.get("items", [])  # Lista de strings: ["Leche", "Pan", "Huevos"]
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido. Token inválido o no coincide.")
                
        if not user_id or user_id == "guest":
            raise HTTPException(status_code=400, detail="Usuario no válido para añadir items.")
            
        if not items:
            raise HTTPException(status_code=400, detail="No se proporcionaron items para añadir.")
        
        MAX_ITEMS = 150
        alert_msg = ""
        if len(items) > MAX_ITEMS:
            items = items[:MAX_ITEMS]
            alert_msg = f" (Alerta: se excedió el límite máximo y solo se añadieron los primeros {MAX_ITEMS} items)"
        
        # Normalizar a JSON struct consistente con ShoppingItemModel
        structured_items = []
        for item_name in items:
            raw = item_name.strip() if isinstance(item_name, str) else ""
            name = sanitize_shopping_text(raw) if raw else ""
            if name:
                cat, emoji = categorize_shopping_item(name)
                structured_items.append({
                    "category": cat,
                    "emoji": emoji,
                    "name": name.capitalize(),
                    "qty": ""
                })
        
        if not structured_items:
            raise HTTPException(status_code=400, detail="Ningún item válido proporcionado.")
            
        result = add_custom_shopping_items(user_id, structured_items, source="manual")
        
        if result is not None:
            # Auto-deduplicar: si el usuario ya tenía "Leche" y añade "leche", se fusionan
            try:
                deduplicate_shopping_items(user_id)
            except Exception:
                pass  # No bloquear la respuesta si falla la dedup
            return {"success": True, "items": result, "message": f"Se añadieron {len(structured_items)} item(s) a tu lista de compras.{alert_msg}"}
        else:
            raise HTTPException(status_code=500, detail="Error al guardar los items en la base de datos.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/shopping/custom/{user_id}")
def api_get_custom_shopping_items(user_id: str, limit: int = 200, offset: int = 0, sort_by: str = "category", sort_order: str = "asc", verified_user_id: str = Depends(get_verified_user_id)):
    """Obtiene los items custom de la lista de compras con paginación y ordenamiento.
    sort_by: category | created_at | display_name | is_checked (default: category)
    sort_order: asc | desc (default: asc)"""
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"items": [], "total": 0}
        result = get_custom_shopping_items(user_id, limit=limit, offset=offset, sort_by=sort_by, sort_order=sort_order)
        return {"items": result.get("data", []), "total": result.get("total", 0)}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom GET: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/shopping/custom/{item_id}")
def api_delete_custom_shopping_item(item_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Elimina un item custom de la lista de compras (con validación IDOR)."""
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="No autorizado. Token requerido para eliminar items.")
        result = delete_custom_shopping_item(item_id, user_id=verified_user_id)
        if result is None or (isinstance(result, list) and len(result) == 0):
            raise HTTPException(status_code=404, detail="Item no encontrado o no pertenece al usuario.")
        return {"success": True, "message": "Item eliminado de la lista."}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/shopping/custom/{item_id}")
def api_update_custom_shopping_item(item_id: str, data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    """Edita campos de un item existente (display_name, qty, category, emoji) con validación IDOR."""
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="No autorizado. Token requerido para editar items.")
        
        updates = {}
        for field in ["display_name", "qty", "category", "emoji"]:
            if field in data:
                val = data[field]
                if isinstance(val, str):
                    updates[field] = sanitize_shopping_text(val)
                else:
                    updates[field] = val
        
        # Si se renombra el item, re-categorizar automáticamente
        if "display_name" in updates and "category" not in data:
            cat, emoji = categorize_shopping_item(updates["display_name"])
            updates["category"] = cat
            updates["emoji"] = emoji
        
        if not updates:
            raise HTTPException(status_code=400, detail="No se proporcionaron campos para actualizar. Campos permitidos: display_name, qty, category, emoji.")
        
        result = update_custom_shopping_item(item_id, updates, user_id=verified_user_id)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Error interno al actualizar el item.")
        if isinstance(result, list) and len(result) == 0:
            raise HTTPException(status_code=404, detail="Item no encontrado o no pertenece al usuario.")
        
        return {"success": True, "message": "Item actualizado.", "updated": updates}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom PUT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/shopping/custom/{item_id}/check")
def api_update_custom_shopping_item_check(item_id: str, data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    """Actualiza el estado de is_checked de un item en la lista de compras (con validación IDOR)."""
    try:
        if not verified_user_id:
            raise HTTPException(status_code=401, detail="No autorizado. Token requerido.")
        is_checked = data.get("is_checked", False)
        result = update_custom_shopping_item_status(item_id, is_checked, user_id=verified_user_id)
        if result is not None:
            return {"success": True, "message": "Estado actualizado"}
        else:
            raise HTTPException(status_code=404, detail="Item no encontrado o no pertenece al usuario")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom PUT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/shopping/custom/clear/{user_id}")
def api_clear_all_shopping_items(user_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Elimina TODOS los items de la lista de compras del usuario."""
    try:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=403, detail="Prohibido.")
        result = clear_all_shopping_items(user_id)
        if result:
            return {"success": True, "message": "Lista de compras vaciada."}
        raise HTTPException(status_code=500, detail="Error al vaciar la lista.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom/clear DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/shopping/custom/uncheck-all/{user_id}")
def api_uncheck_all_shopping_items(user_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Desmarca todos los items de la lista de compras del usuario."""
    try:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=403, detail="Prohibido.")
        result = uncheck_all_shopping_items(user_id)
        if result:
            return {"success": True, "message": "Todos los items desmarcados."}
        raise HTTPException(status_code=500, detail="Error al desmarcar items.")
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom/uncheck-all PUT: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/shopping/custom/deduplicate/{user_id}")
def api_deduplicate_shopping_items(user_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Detecta y fusiona items duplicados en la lista de compras. Suma cantidades cuando es posible."""
    try:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=403, detail="Prohibido.")
        result = deduplicate_shopping_items(user_id)
        return {"success": True, **result}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom/deduplicate POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/shopping/custom/purge/{user_id}")
def api_purge_shopping_items(user_id: str, verified_user_id: str = Depends(get_verified_user_id)):
    """Purga items checked hace más de 30 días y aplica el tope global de 500 items."""
    try:
        if not verified_user_id or verified_user_id != user_id:
            raise HTTPException(status_code=403, detail="Prohibido.")
        result = purge_old_shopping_items(user_id)
        return {"success": True, "message": "Purga completada.", "details": result}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/shopping/custom/purge POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/diary/consumed")
def api_log_consumed_meal(data: dict = Body(...), verified_user_id: str = Depends(get_verified_user_id)):
    """Registra una comida consumida manualmente desde el frontend."""
    try:
        user_id = data.get("user_id")
        meal_name = data.get("meal_name")
        calories = data.get("calories", 0)
        protein = data.get("protein", 0)
        carbs = data.get("carbs", 0)
        healthy_fats = data.get("healthy_fats", 0)
        
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"success": False, "message": "Inicia sesión para registrar comidas."}
            
        log_consumed_meal(user_id, meal_name, int(calories), int(protein), int(carbs), int(healthy_fats))
        
        return {"success": True, "message": "Comida registrada exitosamente."}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed POST: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/diary/consumed/{user_id}")
def api_get_consumed_today(user_id: str, date: Optional[str] = None, tzOffset: Optional[int] = None, verified_user_id: str = Depends(get_verified_user_id)):
    """Obtiene las métricas agregadas de las comidas registradas en el día por la IA."""
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
                
        if not user_id or user_id == "guest":
            return {"meals": [], "totals": {"calories": 0, "protein": 0, "carbs": 0, "healthy_fats": 0}}
        
        meals = get_consumed_meals_today(user_id, date_str=date, tz_offset_mins=tzOffset)
        
        total_cal = sum(m.get("calories", 0) for m in meals)
        total_pro = sum(m.get("protein", 0) for m in meals)
        total_car = sum(m.get("carbs", 0) for m in meals)
        total_fat = sum(m.get("healthy_fats", 0) for m in meals)
        
        return {
            "meals": meals,
            "totals": {
                "calories": total_cal,
                "protein": total_pro,
                "carbs": total_car,
                "healthy_fats": total_fat
            }
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/diary/consumed GET: {str(e)}")
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