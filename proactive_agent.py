import os
import logging
from datetime import datetime, timezone, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI

from db_core import connection_pool, execute_sql_query
from db_chat import save_message, get_recent_messages
from db import get_consumed_meals_today, get_user_profile

logger = logging.getLogger(__name__)

PROACTIVE_PROMPT = """Eres el Nutricionista IA de MealfitRD. Has notado proactivamente que tu paciente aún no ha registrado su {missing_meal}.
Su zona horaria marca que son pasadas las {trigger_time}.

Contexto del paciente:
- Dieta actual: {diet_type}
- Objetivos: {goals}

Escribe un SOLO mensaje conversacional (corto, máximo 2-3 oraciones) animándolo a no olvidarse de su progreso o preguntándole si ya preparó su {missing_meal}.
¡MUY IMPORTANTE! NO SALUDES CON Hola, el usuario verá este mensaje en la interfaz del chat que ya está abierto. Entra directo al tema como una nota de seguimiento.
Usa un tono amistoso y motivacional, nunca de regaño, ni parezcas un robot asustadizo.
"""

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
    
    # 1. Obtenemos las horas actuales (asumiendo AST / -04:00 por simplicidad para MVP en RD)
    now_ast = datetime.now(timezone(timedelta(hours=-4)))
    hour = now_ast.hour
    
    meal_to_check = None
    trigger_time_str = ""
    
    if hour == 11:
        meal_to_check = "Desayuno"
        trigger_time_str = "11:00 AM"
    elif hour == 16:
        meal_to_check = "Almuerzo"
        trigger_time_str = "4:00 PM"
    elif hour == 18:
        meal_to_check = "Merienda"
        trigger_time_str = "6:00 PM"
    elif hour == 20 or hour == 21:
        meal_to_check = "Cena"
        trigger_time_str = f"{hour - 12}:00 PM"
        
    if not meal_to_check:
        logger.debug(f"[CRON] Hora actual ({hour}): No hay triggers de comida primaria a esta hora.")
        return
        
    logger.info(f"🔍 [CRON] Verificando registros de {meal_to_check} a las {trigger_time_str}.")
    
    sessions = get_active_users_for_proactive()
    for s in sessions:
        session_id = str(s.get("id"))
        user_id = str(s.get("user_id"))
        
        try:
            # Regla de Anti-Spam: Solo enviar si no hemos enviado/recibido nada en las últimas 2 horas
            recent = get_recent_messages(session_id, limit=1)
            if recent:
                last_msg_time_str = recent[0].get("created_at")
                if last_msg_time_str:
                    if last_msg_time_str.endswith("Z"):
                        last_msg_time_str = last_msg_time_str[:-1] + "+00:00"
                    last_time = datetime.fromisoformat(last_msg_time_str)
                    
                    diff_hours = (datetime.now(timezone.utc) - last_time).total_seconds() / 3600
                    if diff_hours < 2:
                        # Si hay actividad menor a 2h, no interrumpimos
                        continue
            
            # Vemos perfil para checar scheduleType (turno nocturno)
            profile = get_user_profile(user_id)
            if not profile: continue
            
            health = profile.get("health_profile", {})
            schedule = health.get("scheduleType", "standard")
            if schedule == "night_shift" or schedule == "variable":
                # MVP simple: saltamos la proactividad estricta horaria si tiene turno raro
                continue
                
            # Validar el consumo de HOY
            consumed = get_consumed_meals_today(user_id, date_str=now_ast.strftime("%Y-%m-%d"))
            
            # Checar si la comida objetivo o algo con ese nombre ya se consumió
            already_ate = False
            for m in consumed:
                # El usuario pudo llamarlo "desayuno" o por el nombre de la comida.
                mt = m.get("meal_type", "").lower()
                mn = m.get("meal_name", "").lower()
                if meal_to_check.lower() in mt or meal_to_check.lower() in mn:
                    already_ate = True
                    break
                    
            if already_ate:
                # Todo en orden, registró la comida.
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
            content = str(response.content).strip()
            
            if content:
                # Enviar a la base de datos con rol de modelo
                save_message(session_id, "model", content)
                logger.info(f"✅ [CRON] Mensaje proactivo enviado a {session_id} -> '{content[:40]}...'")
                
                # ---------------------------------------------
                # NUEVO: Enviar Web Push Notification a todos los dispositivos del paciente
                # ---------------------------------------------
                try:
                    import json
                    from pywebpush import webpush, WebPushException  # type: ignore[import-untyped]
                    
                    vapid_private = os.environ.get("VAPID_PRIVATE_KEY")
                    vapid_claim = os.environ.get("VAPID_CLAIM_EMAIL")
                    
                    if vapid_private and vapid_claim:
                        # Buscar las suscripciones de este usuario en DDBB
                        subs_query = "SELECT subscription_data FROM push_subscriptions WHERE user_id = %s"
                        subs = execute_sql_query(subs_query, (user_id,), fetch_all=True)
                        
                        push_payload = json.dumps({
                            "title": "Aviso de tu Nutricionista IA \U0001f9d1\u200d\u2615",
                            "body": content,
                            "url": f"/dashboard/agent?session_id={session_id}"
                        })
                        
                        for sub_row in subs:
                            sub_info = sub_row['subscription_data']
                            if isinstance(sub_info, str):
                                sub_info = json.loads(sub_info)
                                
                            try:
                                webpush(
                                    subscription_info=sub_info,
                                    data=push_payload,
                                    vapid_private_key=vapid_private,
                                    vapid_claims={"sub": vapid_claim}
                                )
                                logger.info(f"📲 [CRON] Push Notification exitosa al dispositivo del usuario {user_id}")
                            except WebPushException as ex:
                                logger.error(f"❌ [CRON] Error mandando Push al usuario {user_id}: {repr(ex)}")
                    else:
                        logger.warning(f"⚠️ [CRON] No se pueden mandar Web Push Notifications porque faltan llaves VAPID en el entorno.")
                except ImportError:
                    logger.warning("No se ha instalado 'pywebpush'. Las notificaciones nativas a móviles no se enviarán.")
                except Exception as e:
                    logger.error(f"Error despachando Push notification proactiva a {user_id}: {e}")
                
        except Exception as e:
            logger.error(f"Error procesando proactividad para {session_id}: {e}")
