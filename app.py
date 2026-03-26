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

from agent import swap_meal, chat_with_agent, analyze_preferences_agent, generate_plan_title
from graph_orchestrator import run_plan_pipeline
from db import get_or_create_session, save_message, save_message_feedback, insert_like, get_user_likes, insert_rejection, get_active_rejections, get_latest_meal_plan, get_user_profile, update_user_health_profile, get_all_user_facts, delete_user_fact, get_custom_shopping_items, delete_custom_shopping_item, log_api_usage, get_monthly_api_usage, connection_pool
from memory_manager import summarize_and_prune, build_memory_context
from fact_extractor import async_extract_and_save_facts, process_pending_queue_sync
from agent import generate_chat_title_background
from langgraph.checkpoint.postgres import PostgresSaver

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

app = FastAPI(lifespan=lifespan)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "MealfitRD AI Backend is running"}

# Dependencia de seguridad para validar token JWT de Supabase
def get_verified_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """Extrae el user_id del token JWT en el header Authorization."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization.split(" ")[1]
    from db import supabase
    if not supabase:
        return None
    try:
        # User auth verification with supabase
        user_res = supabase.auth.get_user(token)
        if user_res and user_res.user:
            return user_res.user.id
    except Exception as e:
        logger.error(f"⚠️ [AUTH] Error validando token: {e}")
    return None

def verify_api_quota(verified_user_id: Optional[str] = Depends(get_verified_user_id)) -> Optional[str]:
    """Dependencia para verificar los límites de uso de la API (Paywall) evitando repetición (DRY)."""
    if verified_user_id:
        credits_used = get_monthly_api_usage(verified_user_id)
        plan_tier = "gratis"
        
        profile = get_user_profile(verified_user_id)
        if profile:
            plan_tier = profile.get("plan_tier", "gratis")
            
        limit = 15 if plan_tier == "gratis" else (100 if plan_tier == "plus" else 999999)
        
        if credits_used >= limit:
            raise HTTPException(status_code=402, detail=f"Límite de créditos alcanzado para tu plan {plan_tier} ({limit}/{limit}). Mejora tu plan para continuar.")
            
    return verified_user_id

# --- Rate limiter ligero (in-memory, sliding window) ---
# Zero dependencias extra. Para multi-worker (gunicorn), cada worker tiene su propio contador,
# lo cual es aceptable (el rate real es N * max_calls). Para rate-limiting distribuido,
# usar Redis + slowapi.
import time as _time
from collections import defaultdict as _defaultdict
from cache_manager import redis_client

class RateLimiter:
    """Sliding-window rate limiter como dependencia FastAPI reutilizable.
    Uso: Depends(RateLimiter(max_calls=10, period_seconds=60))"""
    def __init__(self, max_calls: int = 10, period_seconds: int = 60):
        self.max_calls = max_calls
        self.period = period_seconds
        self._hits: dict = _defaultdict(list)  # user_id → [timestamps]

    def __call__(self, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
        uid = verified_user_id or "anon"
        
        # Opcional: Soporte Redis para Rate Limiting Distribuido (#Mejora 3)
        if redis_client:
            now = _time.time()
            key = f"rl:{self.max_calls}:{self.period}:{uid}"
            window_start = now - self.period
            try:
                pipe = redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start) # Eliminar timestamps viejos
                pipe.zcard(key)                             # Contar peticiones en la ventana
                pipe.zadd(key, {str(now): now})             # Añadir petición actual
                pipe.expire(key, self.period)               # Expirar key para no consumir RAM eterna
                results = pipe.execute()
                
                count = results[1]
                if count >= self.max_calls:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Demasiadas solicitudes. Máximo {self.max_calls} por {self.period}s. Intenta de nuevo en unos segundos."
                    )
                return verified_user_id
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"⚠️ [RATE LIMIT] Error en Redis, cambiando a memoria local transparente: {e}")
                # Hacemos fallback transparente a memoria local

        # --- Fallback Memoria Local ---
        now_mono = _time.monotonic()
        
        # Ocasionalmente (1% de tolerancia) limpiar llaves inactivas globales para evitar fugas de RAM
        import random
        if random.random() < 0.01:
            expired_keys = [
                k for k, timestamps in self._hits.items() 
                if not timestamps or now_mono - timestamps[-1] < now_mono - self.period
            ]
            for k in expired_keys:
                del self._hits[k]

        # Purga timestamps viejos (fuera de la ventana)
        self._hits[uid] = [t for t in self._hits[uid] if now_mono - t < self.period]
        if len(self._hits[uid]) >= self.max_calls:
            raise HTTPException(
                status_code=429,
                detail=f"Demasiadas solicitudes. Máximo {self.max_calls} por {self.period}s. Intenta de nuevo en unos segundos."
            )
        self._hits[uid].append(now_mono)
        return verified_user_id

# Instancias reutilizables para endpoints de shopping list
_shopping_write_limiter = RateLimiter(max_calls=10, period_seconds=60)
_shopping_autogen_limiter = RateLimiter(max_calls=5, period_seconds=60)

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
        if actual_user_id and actual_user_id != "guest":
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
        if actual_user_id and actual_user_id != "guest":
            log_api_usage(actual_user_id, "gemini_analyze")
            
        # 👇 NUEVO: Guardar el plan generado y trackear frecuencias en Background
        # Extraer técnicas ANTES del background (dicts son by-reference, el pop debe ser previo)
        selected_techniques = result.pop("_selected_techniques", None)
        if actual_user_id and actual_user_id != "guest":
            background_tasks.add_task(_save_plan_and_track_background, actual_user_id, result, selected_techniques)

        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"❌ [ERROR] Error en /api/analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _preserve_shopping_checkmarks(existing_items: list, new_items: list):
    """Mantiene activa la casilla de ingredientes ya comprados (is_checked=True) al regenerar el plan."""
    if not existing_items or not new_items:
        return
    import json
    
    def _normalize(text: str) -> str:
        if not text: return ""
        import unicodedata, re
        nfkd = unicodedata.normalize('NFKD', text.lower().strip())
        return re.sub(r'\s+', ' ', ''.join(c for c in nfkd if not unicodedata.combining(c)))

    checked_names = set()
    for old in existing_items:
        if old.get("is_checked"):
            name = old.get("display_name")
            if not name:
                raw = old.get("item_name", "")
                if raw.startswith("{"):
                    try:
                        parsed = json.loads(raw)
                        name = parsed.get("name", raw)
                    except Exception:
                        name = raw
                else:
                    name = raw
            norm = _normalize(name)
            if norm:
                checked_names.add(norm)
                
    if not checked_names:
        return
        
    for new_item in new_items:
        if isinstance(new_item, dict):
            n_name = _normalize(new_item.get("name", ""))
            if n_name in checked_names:
                new_item["is_checked"] = True

def _save_plan_and_track_background(user_id: str, plan_data: dict, selected_techniques: list = None):
    """Background task: Guarda el plan JSON en supabase y trackea frecuencias, O(1)."""
    try:
        from db import supabase, increment_ingredient_frequencies
        from datetime import datetime
        
        # 1. Guardar Plan O(1) Arrays
        if supabase:
            calories = plan_data.get("calories", 0)
            macros = plan_data.get("macros", {})
            
            meal_names = []
            ingredients = []
            raw_ingredients = []
            for d in plan_data.get("days", []):
                for m in d.get("meals", []):
                    if m.get("name"):
                        meal_names.append(m.get("name"))
                    if m.get("ingredients"):
                        ingredients.extend(m.get("ingredients"))
                        raw_ingredients.extend(m.get("ingredients"))
                        
            # Nombre creativo generado por IA (Gemini Flash-Lite)
            plan_name = generate_plan_title(plan_data)
                
            insert_data = {
                "user_id": user_id,
                "plan_data": plan_data,
                "name": plan_name,
                "calories": int(calories) if calories else 0,
                "macros": macros,
                "meal_names": meal_names,
                "ingredients": ingredients
            }
            
            # Añadir técnicas de cocción si están disponibles
            if selected_techniques:
                insert_data["techniques"] = selected_techniques
            
            # 🛡️ Dedup guard: evitar duplicados si otro código path ya guardó el plan
            try:
                recent = supabase.table("meal_plans").select("created_at").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
                if recent.data and len(recent.data) > 0:
                    from datetime import timezone
                    last_saved = datetime.fromisoformat(recent.data[0]["created_at"].replace("Z", "+00:00"))
                    now_utc = datetime.now(timezone.utc)
                    diff_secs = (now_utc - last_saved).total_seconds()
                    if diff_secs < 30:
                        logger.info(f"🛡️ [DEDUP] Plan ya guardado hace {diff_secs:.0f}s para {user_id}. Omitiendo duplicado.")
                        return
            except Exception as dedup_e:
                logger.warning(f"⚠️ [DEDUP] Error en verificación anti-duplicados: {dedup_e}")

            try:
                supabase.table("meal_plans").insert(insert_data).execute()
            except Exception as try_db_e:
                err_msg = str(try_db_e)
                if "meal_names" in err_msg or "techniques" in err_msg or "PGRST205" in err_msg or "Could not find" in err_msg:
                    logger.warning("⚠️ [DB] Faltan columnas optimizadas (meal_names/techniques). Ejecute migration_fast_meals.sql")
                    insert_data.pop("meal_names", None)
                    insert_data.pop("ingredients", None)
                    insert_data.pop("techniques", None)
                    supabase.table("meal_plans").insert(insert_data).execute()
                else:
                    raise try_db_e
            logger.debug(f"💾 [DB BACKGROUND] Plan guardado exitosamente en meal_plans para {user_id}")
            
        # 2. Track Frequencies (solo ingredientes canónicos que existan en los catálogos de variedad)
        if raw_ingredients:
            from constants import normalize_ingredient_for_tracking, GLOBAL_REVERSE_MAP
            # Conjunto de términos base canónicos (ej: "pollo", "platano verde", "aguacate")
            canonical_bases = set(GLOBAL_REVERSE_MAP.values())
            
            normalized = [normalize_ingredient_for_tracking(ing) for ing in raw_ingredients]
            # Filtrar: solo trackear ingredientes que resolvieron a un término base conocido.
            # Esto evita que condimentos/hierbas (cilantro, orégano, ajo) polucionen la tabla.
            canonical = [n for n in normalized if n and n in canonical_bases]
            non_canonical = [n for n in normalized if n and n not in canonical_bases]
            
            if canonical:
                increment_ingredient_frequencies(user_id, canonical)
            
            # 2b. Loguear ingredientes no reconocidos para revisión y expansión del catálogo
            if non_canonical:
                from db import log_unknown_ingredients
                raw_map = {normalize_ingredient_for_tracking(r): r for r in raw_ingredients if r}
                log_unknown_ingredients(user_id, non_canonical, raw_map)
                logger.info(f"🧹 [FREQ TRACKING] {len(non_canonical)} ingredientes no-canónicos logueados para revisión")
                
            logger.info(f"📈 [FREQ TRACKING] Frecuencias actualizadas en background para {user_id} ({len(canonical)} ingredientes canónicos trackeados)")
            
        # 3. Auto-generar lista de compras (background, con cache por hash)
        try:
            import hashlib, json as _json
            all_ingredients = []
            all_supplements = []
            for d in plan_data.get("days", []):
                for m in d.get("meals", []):
                    ing = m.get("ingredients", [])
                    if ing:
                        all_ingredients.extend(ing)
                for s in d.get("supplements") or []:
                    all_supplements.append(s.get("name", ""))
            if all_ingredients:
                plan_hash = hashlib.sha256(_json.dumps({"ingredients": all_ingredients, "supplements": sorted(set(all_supplements)), "version": "v4_multiday"}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
                from db import get_shopping_plan_hash, save_shopping_plan_hash, add_custom_shopping_items, delete_auto_generated_shopping_items, get_custom_shopping_items, get_user_profile
                
                stored_hash = get_shopping_plan_hash(user_id)
                existing = get_custom_shopping_items(user_id)
                existing_items = existing.get("data", []) if isinstance(existing, dict) else existing
                
                # 🔒 REGLA DE TIEMPO PARA BACKGROUND (Evita mutación sutil si ya hay lista vigente)
                locked = False
                if existing_items:
                    try:
                        hp = get_user_profile(user_id)
                        hp_data = hp.get("health_profile", {}) if hp else {}
                        cycle_days = hp_data.get("shopping_cycle", {}).get("duration_days", 7)
                        
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
                            if days_elapsed < cycle_days:
                                locked = True
                                logger.info(f"🔒 [BACKGROUND BLOCKED] Lista auto-generada está vigente: {days_elapsed}/{cycle_days} días. Cancelando regeneración destructiva en background.")
                    except Exception as e:
                        logger.error(f"Error parseando límite de compras en background para {user_id}: {e}")

                if locked:
                    logger.debug(f"✅ [BACKGROUND SKIP] Plan ignorado por la lista de compras actual (Aún válida).")
                elif stored_hash != plan_hash:
                    from agent import generate_auto_shopping_list
                    items = generate_auto_shopping_list(plan_data)
                    if items:
                        # 🔒 Lock per-user: serializa insert→delete→purge→dedup
                        from db import get_user_shopping_lock
                        shopping_lock = get_user_shopping_lock(user_id)
                        with shopping_lock:
                            # 🛡️ Patrón INSERT-FIRST / DELETE-OLD (Crash-Safe)
                            _preserve_shopping_checkmarks(existing_items, items)
                            result = add_custom_shopping_items(user_id, items, source="auto")
                            new_ids = [r.get("id") for r in result if r.get("id")] if result and isinstance(result, list) else []
                            delete_auto_generated_shopping_items(user_id, exclude_ids=new_ids)
                            # 🧹 Auto-purga + deduplicación (consistente con /api/shopping/auto-generate)
                            from db import purge_old_shopping_items, deduplicate_shopping_items
                            purge_old_shopping_items(user_id)
                            deduplicate_shopping_items(user_id)
                            save_shopping_plan_hash(user_id, plan_hash)
                        logger.debug(f"🛒 [BACKGROUND] Lista de compras auto-generada ({len(items)} items) para {user_id}")
                    else:
                        logger.warning(f"⚠️ [BACKGROUND] No se pudieron consolidar ingredientes para {user_id}")
                else:
                    logger.debug(f"✅ [BACKGROUND CACHE HIT] Plan sin cambios, lista de compras ya actualizada para {user_id}")
        except Exception as shop_e:
            logger.error(f"⚠️ [BACKGROUND] Error auto-generando lista de compras: {shop_e}")
            
    except Exception as e:
        logger.error(f"⚠️ [BACKGROUND ERROR] Error asíncrono guardando plan: {e}")


def _process_swap_rejection_background(session_id: str, user_id: str, rejected_meal: str, meal_type: str):
    """Background task: Loguea mensajes y rechazos que expiran en 7 días, asíncronamente."""
    try:
        from db import get_or_create_session, save_message, insert_rejection
        if session_id and rejected_meal:
            get_or_create_session(session_id)
            save_message(session_id, "user", f"Rechacé explícitamente: {rejected_meal}")
        
        # Guardar rechazo TEMPORAL (expira en 7 días)
        if rejected_meal:
            rejection_record = {
                "meal_name": rejected_meal,
                "meal_type": meal_type,
            }
            if user_id and user_id != "guest":
                rejection_record["user_id"] = user_id
            if session_id:
                rejection_record["session_id"] = session_id
            
            insert_rejection(rejection_record)
            logger.debug(f"💾 [DB BACKGROUND] Rechazo temporal guardado para {rejected_meal}")
            
            # Fricción Silenciosa: Validar si la base ya se rechazó 3 veces
            from db import track_meal_friction
            track_meal_friction(user_id, session_id, rejected_meal)
    except Exception as e:
        logger.error(f"⚠️ [BACKGROUND ERROR] Error procesando swap rejection: {e}")

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

@app.get("/api/user/credits/{user_id}")
def api_get_user_credits(user_id: str, verified_user_id: Optional[str] = Depends(get_verified_user_id)):
    """Consulta los créditos consumidos en el mes usando api_usage."""
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido. Token inválido o no coincide.")
                
        from db import get_monthly_api_usage
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
        from db import supabase
        if supabase:
            check = supabase.table("user_facts").select("user_id").eq("id", fact_id).execute()
            if not check.data:
                raise HTTPException(status_code=404, detail="Hecho no encontrado.")
            if check.data[0].get("user_id") != verified_user_id:
                raise HTTPException(status_code=403, detail="Prohibido. Este hecho no te pertenece.")
        
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
    from db import get_guest_chat_sessions, get_user_chat_sessions
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
    from db import supabase
    try:
        if user_id and user_id != "guest":
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=403, detail="Prohibido.")
            if supabase:
                supabase.table("agent_sessions").delete().eq("user_id", user_id).execute()
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/sessions DELETE: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history/{session_id}")
def api_get_chat_history(session_id: str):
    from db import get_session_messages
    try:
        messages = get_session_messages(session_id)
        # Ocultar mensajes de sistema como el system_title
        filtered_messages = [m for m in messages if not m.get("content", "").startswith("[SYSTEM_TITLE]")]
        return {"messages": filtered_messages}
    except Exception as e:
        logger.error(f"❌ [ERROR] Error en /api/chat/history GET: {str(e)}")
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
async def api_chat_stream(background_tasks: BackgroundTasks, data: dict = Body(...), verified_user_id: str = Depends(verify_api_quota)):
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
        
        # Estas operaciones son síncronas pero muy rápidas
        await asyncio.to_thread(get_or_create_session, session_id, user_id=user_id if user_id != "guest" else None)
        await asyncio.to_thread(save_message, session_id, "user", prompt)
        
        # Handle form_data: merge frontend data with DB health_profile
        merged_form_data = form_data or {}
        if user_id and user_id != "guest" and user_id != session_id:
            try:
                profile = await asyncio.to_thread(get_user_profile, user_id)
                if profile:
                    existing_hp = profile.get("health_profile") or {}
                    
                    if existing_hp:
                        # Si hay datos en la DB, los usamos y los actualizamos con lo nuevo del frontend
                        existing_hp.update(merged_form_data)
                        merged_form_data = existing_hp
                    
                    # Sincronizar de vuelta a la DB si el frontend envió algo válido pero la DB estaba vacía o diferente
                    if form_data and not existing_hp:
                        logger.debug(f"🔄 [SYNC STREAM] health_profile vacío, sincronizando...")
                        await asyncio.to_thread(update_user_health_profile, user_id, merged_form_data)
                else:
                    if form_data:
                        try:
                            from db import supabase as sb_client
                            if sb_client:
                                def _upsert_profile():
                                    sb_client.table("user_profiles").upsert({
                                        "id": user_id,
                                        "health_profile": merged_form_data
                                    }).execute()
                                await asyncio.to_thread(_upsert_profile)
                        except Exception as e:
                            logger.error(f"❌ [SYNC STREAM] Error creando perfil: {e}")
            except Exception as e:
                logger.error(f"⚠️ Error cargando health profile en chat: {e}")
                
        form_data = merged_form_data
        
        if not current_plan and user_id and user_id != "guest":
            current_plan = await asyncio.to_thread(get_latest_meal_plan, user_id)
            
        import threading
        from agent import achat_with_agent_stream, generate_chat_title_background
        
        # Iniciar generación del título de inmediato en paralelo a la respuesta del stream (usar OS threads previene bloqueo por el grafo síncrono)
        threading.Thread(
            target=generate_chat_title_background,
            args=(user_id, session_id, prompt),
            daemon=True
        ).start()
        
        async def event_generator():
            try:
                # Obtenemos los chunks del generador de LangGraph
                async for chunk in achat_with_agent_stream(
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
                        import json
                        try:
                            data_obj = json.loads(chunk[len("data: "):].strip())
                            if data_obj.get("type") == "done":
                                response_text = data_obj.get("response", "")
                                if response_text:
                                    # Guardamos la respuesta final en DB  
                                    await asyncio.to_thread(save_message, session_id, "model", response_text)
                                    
                                # Lógica Background (resumir, uso de API, embeddings)
                                def bg_tasks():
                                    if user_id and user_id != "guest" and user_id != session_id:
                                        log_api_usage(user_id, "gemini_chat")
                                        
                                    try:
                                        from db import get_session_messages
                                        raw_history = get_session_messages(session_id)
                                        recent_history_str = ""
                                        if raw_history:
                                            recent_history_str = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in raw_history[-6:]])
                                        
                                        is_plus = False
                                        if user_id and user_id != "guest":
                                            profile_sync = get_user_profile(user_id)
                                            if profile_sync:
                                                plan_tier_sync = profile_sync.get("plan_tier", "gratis")
                                                is_plus = plan_tier_sync in ["plus", "admin", "ultra"]
                                                
                                        if is_plus:
                                            async_extract_and_save_facts(user_id, prompt, recent_history_str)
                                            
                                        summarize_and_prune(session_id)
                                    except Exception as inner_e:
                                        logger.error(f"Error en bg tasks: {inner_e}")
                                
                                await asyncio.to_thread(bg_tasks)
                        except Exception as e_json:
                            logger.error(f"Error parseando chunk de fin: {e_json}")
                            
            except Exception as e:
                import traceback
                traceback.print_exc()
                import json
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except HTTPException:
        raise
    except Exception as e:
        import traceback
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
        
        # Handle form_data: merge frontend data with DB health_profile
        merged_form_data = form_data or {}
        if user_id and user_id != "guest" and user_id != session_id:
            try:
                profile = get_user_profile(user_id)
                if profile:
                    existing_hp = profile.get("health_profile") or {}
                    
                    if existing_hp:
                        existing_hp.update(merged_form_data)
                        merged_form_data = existing_hp
                        
                    if form_data and not existing_hp:
                        logger.debug(f"🔄 [SYNC] health_profile vacío, sincronizando desde formData del frontend...")
                        update_user_health_profile(user_id, merged_form_data)
                else:
                    if form_data:
                        logger.warning(f"⚠️ [SYNC] No existe user_profile para {user_id}, intentando crear...")
                        try:
                            from db import supabase as sb_client
                            if sb_client:
                                sb_client.table("user_profiles").upsert({
                                    "id": user_id,
                                    "health_profile": merged_form_data
                                }).execute()
                                logger.info(f"✅ [SYNC] Perfil creado con health_profile")
                        except Exception as e:
                            logger.error(f"❌ [SYNC] Error creando perfil: {e}")
            except Exception as e:
                logger.error(f"⚠️ Error cargando health profile en chat: {e}")
                
        form_data = merged_form_data
        
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
        from db import get_session_messages
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
                is_plus = plan_tier in ["plus", "admin", "ultra"]
        
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
    verified_user_id: str = Depends(verify_api_quota)
):
    try:
        # Validación de seguridad IDOR
        if user_id and user_id != "guest" and user_id != session_id:
            if not verified_user_id or verified_user_id != user_id:
                raise HTTPException(status_code=401, detail="No autorizado.")
                
        actual_user_id = user_id if user_id != "guest" else session_id

        from vision_agent import process_image_with_vision
        from db import supabase
        
        MAX_FILE_SIZE = 20 * 1024 * 1024 # 20 MB
        file_bytes = b""
        while chunk := await file.read(1024 * 1024):
            file_bytes += chunk
            if len(file_bytes) > MAX_FILE_SIZE:
                raise HTTPException(status_code=413, detail="La imagen es demasiado grande. Máximo 20MB permitidos.")
        
        file_ext = file.filename.split('.')[-1] if '.' in file.filename else 'jpg'
        actual_user_id = user_id if user_id != "guest" else session_id
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
        
        if is_food:
            logger.info(f"✅ Descripción generada: '{description}'")
            
            if actual_user_id and actual_user_id != "guest" and actual_user_id != session_id:
                from db import log_api_usage
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
    from vision_agent import get_multimodal_embedding
    from db import save_visual_entry
    
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
            
        from db import get_latest_meal_plan
        current_plan = get_latest_meal_plan(user_id)
        
        if not current_plan:
            raise HTTPException(status_code=404, detail="No se encontró un plan activo para extraer ingredientes.")
        
        # Calcular hash del plan para detectar cambios (incluye suplementos)
        import hashlib, json as _json
        ingredients_for_hash = []
        supplements_for_hash = []
        for d in current_plan.get("days", []):
            for m in d.get("meals", []):
                ing = m.get("ingredients", [])
                if ing:
                    ingredients_for_hash.extend(ing)
            for s in d.get("supplements") or []:
                supplements_for_hash.append(s.get("name", ""))
        plan_hash = hashlib.sha256(_json.dumps({"ingredients": ingredients_for_hash, "supplements": sorted(set(supplements_for_hash)), "version": "v4_multiday"}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
        
        # Verificar si el plan ya fue procesado y si la lista sigue vigente
        if not force:
            from db import get_shopping_plan_hash
            from db import get_custom_shopping_items
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
            
        from agent import generate_auto_shopping_list
        items = generate_auto_shopping_list(current_plan)
        
        if not items:
            return {"success": False, "message": "No se encontraron ingredientes para consolidar en el plan activo."}
            
        from db import add_custom_shopping_items, delete_auto_generated_shopping_items, save_shopping_plan_hash
        
        # 🔒 Lock per-user: serializa insert→delete→purge→dedup para evitar race conditions
        from db import get_user_shopping_lock
        shopping_lock = get_user_shopping_lock(user_id)
        with shopping_lock:
            # 🛡️ Patrón INSERT-FIRST / DELETE-OLD (Crash-Safe)
            # Insertar los nuevos items ANTES de borrar los viejos.
            # Si el proceso muere entre el INSERT y el DELETE, el usuario tiene duplicados (fácilmente deduplicables),
            # en vez de perder toda su lista (irrecuperable).
            _preserve_shopping_checkmarks(existing_items, items)
            result = add_custom_shopping_items(user_id, items, source="auto")
            
            # Extraer IDs de los items recién insertados para excluirlos del borrado
            new_ids = []
            if result and isinstance(result, list):
                new_ids = [r.get("id") for r in result if r.get("id")]
            
            # Borrar items auto-generados ANTERIORES (excluyendo los que acabamos de insertar)
            delete_auto_generated_shopping_items(user_id, exclude_ids=new_ids)
            
            # 🧹 Auto-purga oportunista: limpiar items checked viejos y enforce tope global
            from db import purge_old_shopping_items
            purge_old_shopping_items(user_id)
            
            # 🔄 Auto-deduplicación: unificar items duplicados automáticamente
            from db import deduplicate_shopping_items
            deduplicate_shopping_items(user_id)
            
            # Guardar hash del plan para cache futuro
            save_shopping_plan_hash(user_id, plan_hash)
        
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
        
        # Sanitización: eliminar HTML/JS tags, control chars, y limitar longitud
        import re as _re
        MAX_ITEM_LENGTH = 100
        
        def _sanitize(text: str) -> str:
            clean = _re.sub(r'<[^>]+>', '', text).strip()
            clean = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)
            return clean[:MAX_ITEM_LENGTH]
        
        # Normalizar a JSON struct consistente con ShoppingItemModel
        from tools import _categorize_item
        structured_items = []
        for item_name in items:
            raw = item_name.strip() if isinstance(item_name, str) else ""
            name = _sanitize(raw) if raw else ""
            if name:
                cat, emoji = _categorize_item(name)
                structured_items.append({
                    "category": cat,
                    "emoji": emoji,
                    "name": name.capitalize(),
                    "qty": ""
                })
        
        if not structured_items:
            raise HTTPException(status_code=400, detail="Ningún item válido proporcionado.")
            
        from db import add_custom_shopping_items
        result = add_custom_shopping_items(user_id, structured_items, source="manual")
        
        if result is not None:
            # Auto-deduplicar: si el usuario ya tenía "Leche" y añade "leche", se fusionan
            try:
                from db import deduplicate_shopping_items
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
        from db import delete_custom_shopping_item
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
        
        # Extraer campos editables del body
        import re as _re
        MAX_FIELD_LENGTH = 100
        
        def _sanitize_field(text: str) -> str:
            clean = _re.sub(r'<[^>]+>', '', text).strip()
            clean = _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', clean)
            return clean[:MAX_FIELD_LENGTH]
        
        updates = {}
        for field in ["display_name", "qty", "category", "emoji"]:
            if field in data:
                val = data[field]
                if isinstance(val, str):
                    updates[field] = _sanitize_field(val)
                else:
                    updates[field] = val
        
        # Si se renombra el item, re-categorizar automáticamente
        if "display_name" in updates and "category" not in data:
            from tools import _categorize_item
            cat, emoji = _categorize_item(updates["display_name"])
            updates["category"] = cat
            updates["emoji"] = emoji
        
        if not updates:
            raise HTTPException(status_code=400, detail="No se proporcionaron campos para actualizar. Campos permitidos: display_name, qty, category, emoji.")
        
        from db import update_custom_shopping_item
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
        from db import update_custom_shopping_item_status
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
        from db import clear_all_shopping_items
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
        from db import uncheck_all_shopping_items
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
        from db import deduplicate_shopping_items
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
        from db import purge_old_shopping_items
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
            
        from db import log_consumed_meal
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
        
        from db import get_consumed_meals_today
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
            
        from db import migrate_guest_data, update_user_health_profile, get_latest_meal_plan, get_user_profile
        from db import supabase
        
        # 1. Transformar data guest a registrada
        success = migrate_guest_data(session_ids, new_user_id)
        if not success:
            logger.warning(f"⚠️ Aviso: La función de migración base devolvió False, pero continuamos con profile y planes.")
        
        # 2. Upsert health_profile si el frontend lo provee
        if health_profile:
            try:
                from db import get_user_profile
                profile = get_user_profile(new_user_id)
                # Si el usuario es nuevo, puede no existir su perfil
                if profile:
                    update_user_health_profile(new_user_id, health_profile)
                else:
                    if supabase:
                        supabase.table("user_profiles").upsert({
                            "id": new_user_id,
                            "health_profile": health_profile
                        }).execute()
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
                        try:
                            supabase.table("meal_plans").insert(insert_data).execute()
                        except Exception as try_db_e:
                            err_msg = str(try_db_e)
                            if "meal_names" in err_msg or "PGRST205" in err_msg or "Could not find" in err_msg:
                                del insert_data["meal_names"]
                                del insert_data["ingredients"]
                                supabase.table("meal_plans").insert(insert_data).execute()
                            else:
                                raise try_db_e
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