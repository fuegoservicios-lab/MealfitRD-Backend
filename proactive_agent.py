import os
import logging
from datetime import datetime, timezone, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI

from db_core import connection_pool, execute_sql_query, execute_sql_write
from db_chat import save_message, get_recent_messages
from db import get_consumed_meals_today, get_user_profile

logger = logging.getLogger(__name__)

from prompts.proactive import PROACTIVE_PROMPT

def get_active_users_for_proactive() -> list:
    """Busca session_ids que pertenezcan a usuarios registrados con actividad reciente."""
    try:
        # Obtenemos sesiones que sí tienen un user_id y han estado activas en los últimos 3 días (72 hrs)
        if not connection_pool: return []
        query = (
            "SELECT DISTINCT ON (user_id) id, user_id "
            "FROM agent_sessions "
            "WHERE user_id IS NOT NULL "
            "AND user_id::text != 'guest' "
            "AND created_at >= NOW() - INTERVAL '3 days' "
            "ORDER BY user_id, created_at DESC"
        )
        res = execute_sql_query(query, fetch_all=True)
        return res if res else []
    except Exception as e:
        logger.error(f"Error fetching active sessions for proactive check: {e}")
        return []

def run_proactive_checks():
    """Esta función será llamada por apscheduler (cron job)."""
    logger.info("⏱️ [CRON] Iniciando verificación proactiva de comidas.")
    
    # --- PHASE 3: JIT Rolling Window Trigger ---
    # check_and_trigger_jit_rolling_windows() # Desactivado: El paso a Micro-Batching usa triggers interactivos vía UI ("Actualizar Platos")

    
    # 1. Obtenemos las horas actuales (asumiendo AST / -04:00 por simplicidad para MVP en RD)
    now_ast = datetime.now(timezone(timedelta(hours=-4)))
    hour = now_ast.hour
    
    meal_to_check = None
    trigger_time_str = ""
    
    if hour == 10:
        meal_to_check = "Desayuno"
        trigger_time_str = "10:30 AM"
    elif hour == 13 or hour == 14:
        meal_to_check = "Almuerzo"
        trigger_time_str = f"{hour if hour == 12 else hour - 12}:30 PM"
    elif hour == 17:
        meal_to_check = "Merienda"
        trigger_time_str = "5:30 PM"
    elif hour == 20 or hour == 21:
        meal_to_check = "Cena"
        trigger_time_str = f"{hour - 12}:30 PM"
    elif hour == 23:
        meal_to_check = "Resumen del día"
        trigger_time_str = "11:00 PM"
        
    if not meal_to_check:
        logger.debug(f"[CRON] Hora actual ({hour}): No hay triggers de comida primaria a esta hora.")
        return
        
    logger.info(f"🔍 [CRON] Verificando registros de {meal_to_check} a las {trigger_time_str}.")
    
    sessions = get_active_users_for_proactive()
    logger.info(f"🔍 [CRON] Encontradas {len(sessions)} sesiones activas para verificar.")
    for s in sessions:
        session_id = str(s.get("id"))
        user_id = str(s.get("user_id"))
        
        try:
            # Regla Anti-Spam: Solo bloquear si ya enviamos un mensaje PROACTIVO (model) en la última hora.
            # Los mensajes del usuario NO bloquean recordatorios — chatear no impide recibir nudges.
            recent = get_recent_messages(session_id, limit=5)
            spam_blocked = False
            if recent:
                for msg in recent:
                    if msg.get("role") != "model":
                        continue  # Solo nos importan mensajes del modelo
                    last_msg_time_str = msg.get("created_at")
                    if last_msg_time_str:
                        if last_msg_time_str.endswith("Z"):
                            last_msg_time_str = last_msg_time_str[:-1] + "+00:00"
                        last_time = datetime.fromisoformat(last_msg_time_str)
                        diff_hours = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
                        if diff_hours < 1:
                            spam_blocked = True
                            logger.info(f"🚫 [CRON] Anti-spam: Usuario {user_id} ya recibió mensaje del modelo hace {diff_hours:.1f}h. Saltando.")
                            break
            
            if spam_blocked:
                continue
            
            # Vemos perfil para checar scheduleType (turno nocturno)
            profile = get_user_profile(user_id)
            if not profile:
                logger.info(f"🚫 [CRON] Usuario {user_id}: sin perfil. Saltando.")
                continue
            
            health = profile.get("health_profile", {})
            schedule = health.get("scheduleType", "standard")
            if schedule == "night_shift" or schedule == "variable":
                logger.info(f"🚫 [CRON] Usuario {user_id}: turno {schedule}. Saltando.")
                continue
                
            # Validar el consumo de HOY
            consumed = get_consumed_meals_today(user_id, date_str=now_ast.strftime("%Y-%m-%d"))
            
            if meal_to_check == "Resumen del día":
                if consumed:
                    logger.info(f"✅ [CRON] Usuario {user_id}: registró comidas hoy. Todo ok para el resumen.")
                    continue
                else:
                    # Enviar mensaje especial: No comió nada
                    logger.info(f"⚠️ [CRON] Usuario {user_id} ({session_id}) no registró NADA. Generando nudge indulgente...")
                    prompt = f"""
Eres tu nutricionista IA. Son las {trigger_time_str} de la noche.
He notado que el paciente no ha registrado NINGUNA comida en todo el día en su diario de Mealfit.
Escríbele un mensaje corto (máximo 2 líneas) muy amistoso e indulgente al estilo WhatsApp preguntándole:
"Veo que no registraste nada hoy, ¿restamos lo de hoy de tu nevera como si lo hubieras cocinado o comiste fuera?"
No uses demasiados emojis. Sé directo, breve y empático.
"""
            else:
                # Checar si la comida objetivo o algo con ese nombre ya se consumió
                already_ate = False
                for m in consumed:
                    mt = m.get("meal_type", "").lower()
                    mn = m.get("meal_name", "").lower()
                    if meal_to_check.lower() in mt or meal_to_check.lower() in mn:
                        already_ate = True
                        break
                        
                if already_ate:
                    logger.info(f"✅ [CRON] Usuario {user_id}: ya registró {meal_to_check}. Todo ok.")
                    continue
                    
                # ESTADO: olvido registrar. Generar mensaje proactivo.
                logger.info(f"⚠️ [CRON] Usuario {user_id} ({session_id}) no registró {meal_to_check}. Generando mensaje...")
                
                diet_types = health.get("dietTypes", ["balanceada"])
                diet_type = diet_types[0] if diet_types else "balanceada"
                goals = ", ".join(health.get("goals", ["mantener de manera saludable"]))
                
                prompt = PROACTIVE_PROMPT.format(
                    missing_meal=meal_to_check,
                    trigger_time=trigger_time_str,
                    diet_type=diet_type,
                    goals=goals
                )
                
            chat_llm = ChatGoogleGenerativeAI(
                model="gemini-3.1-flash-lite-preview", 
                temperature=0.8,
                google_api_key=os.environ.get("GEMINI_API_KEY")
            )
            response = chat_llm.invoke(prompt)
            raw_content = response.content
            if isinstance(raw_content, list):
                content = " ".join([b.get("text", "") for b in raw_content if isinstance(b, dict) and "text" in b]).strip()
            else:
                content = str(raw_content).strip()
            
            if content:
                # Enviar a la base de datos con rol de modelo
                save_message(session_id, "model", content)
                logger.info(f"✅ [CRON] Mensaje proactivo enviado a {session_id} -> '{content[:40]}...'")
                
                # ---------------------------------------------
                # NUEVO: Enviar Web Push Notification a todos los dispositivos del paciente
                # ---------------------------------------------
                from utils_push import send_push_notification
                send_push_notification(
                    user_id=user_id,
                    title="Aviso de tu Nutricionista IA \U0001f9d1\u200d\u2615",
                    body=content,
                    url=f"/dashboard/agent?session_id={session_id}"
                )
                
        except Exception as e:
            logger.error(f"Error procesando proactividad para {session_id}: {e}")

def _trigger_week2_background_generation(user_id, plan_id, existing_plan_data):
    """Generates the next 7 days in the background and appends to the existing plan."""
    from graph_orchestrator import run_plan_pipeline
    from db_profiles import get_user_profile
    from db_plans import update_meal_plan_data
    from db import get_user_likes, get_active_rejections
    from agent import analyze_preferences_agent
    import threading
    
    def _bg_task():
        try:
            logger.info(f"🔄 [JIT BG TASK] Arrancando generación asíncrona de Semana 2 para {user_id}...")
            profile = get_user_profile(user_id) or {}
            health_profile = profile.get("health_profile", {})
            form_data = health_profile.copy()
            form_data["user_id"] = user_id
            
            likes = get_user_likes(user_id)
            rejections = get_active_rejections(user_id)
            rej_names = [r["meal_name"] for r in rejections] if rejections else []
            taste_profile = analyze_preferences_agent(likes, [], active_rejections=rej_names)
            
            result = run_plan_pipeline(form_data, [], taste_profile, memory_context="")
            
            new_days = result.get("days", [])
            existing_days = existing_plan_data.get("days", [])
            
            start_idx = len(existing_days) + 1
            for i, d in enumerate(new_days):
                d["day"] = start_idx + i
                
            existing_plan_data["days"] = existing_days + new_days
            update_meal_plan_data(plan_id, existing_plan_data)
            logger.info(f"✅ [JIT BG TASK] Semana 2 añadida exitosamente al plan {plan_id} de {user_id}")
        except Exception as e:
            logger.error(f"❌ [JIT BG TASK] Error en generación de semana 2 para {user_id}: {e}")

    threading.Thread(target=_bg_task, daemon=True).start()

def check_and_trigger_jit_rolling_windows():
    """
    JIT Rolling Windows Trigger:
    Detects users who are on Day 5 or 6 (i.e. plan generated 4-6 days ago)
    and if their plan has only 7 days, generates Week 2.
    """
    logger.info("⏱️ [CRON JIT] Chequeando ventanas JIT (Rolling Windows) para Semana 2...")
    if not connection_pool: return
    
    try:
        from db_core import execute_sql_query
        query = """
            SELECT id, user_id, plan_data
            FROM meal_plans
            WHERE created_at >= NOW() - INTERVAL '6 days'
            AND created_at <= NOW() - INTERVAL '4 days'
            ORDER BY created_at DESC
        """
        res = execute_sql_query(query, fetch_all=True)
        if not res: return
        
        seen_users = set()
        for row in res:
            uid = row.get("user_id")
            if uid in seen_users: continue
            seen_users.add(uid)
            
            plan_data = row.get("plan_data")
            if not isinstance(plan_data, dict): continue
            
            days = plan_data.get("days", [])
            if len(days) == 7:
                logger.info(f"🔄 [JIT TRIGGER] Usuario {uid} está en Día 5-6. Disparando Fase 3 (Semana 2)...")
                _trigger_week2_background_generation(uid, str(row.get("id")), plan_data)
                
    except Exception as e:
        logger.error(f"⚠️ [JIT CRON] Error en check_and_trigger_jit_rolling_windows: {e}")
