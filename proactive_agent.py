import os
import logging
from datetime import datetime, timezone, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI

from db_core import connection_pool, execute_sql_query, execute_sql_write
from db_chat import save_message, get_recent_messages
from db import get_consumed_meals_today, get_user_profile
from fact_extractor import get_embedding

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

def get_best_nudge_style(user_id: str) -> str:
    """Implementa A/B testing (Epsilon-Greedy) para formatos de mensajes."""
    import random
    styles = ["directo", "sugestivo", "gamificado"]
    try:
        query = """
            SELECT nudge_style, 
                   COUNT(*) as total, 
                   SUM(CASE WHEN meal_logged THEN 1 ELSE 0 END) as successes
            FROM nudge_outcomes 
            WHERE user_id = %s AND nudge_style IS NOT NULL
            GROUP BY nudge_style
        """
        stats = execute_sql_query(query, (user_id,), fetch_all=True)
        
        total_nudges = sum(s['total'] for s in stats) if stats else 0
        
        if total_nudges < 10:
            return random.choice(styles)
            
        best_style = None
        best_rate = -1.0
        for s in stats:
            rate = s['successes'] / float(s['total'])
            if rate > best_rate:
                best_rate = rate
                best_style = s['nudge_style']
                
        if random.random() < 0.1 or not best_style:
            return random.choice(styles)
            
        return best_style
    except Exception as e:
        logger.error(f"Error calculando mejor nudge style: {e}")
        return random.choice(styles)

def log_nudge_outcome(user_id, nudge_type, context_embedding=None, context_summary=None, nudge_content=None, nudge_style=None):
    try:
        if context_embedding and context_summary and nudge_content:
            emb_str = f"[{','.join(map(str, context_embedding))}]"
            query = """INSERT INTO nudge_outcomes 
                       (user_id, nudge_type, sent_at, responded, meal_logged, context_embedding, context_summary, nudge_content, nudge_style) 
                       VALUES (%s, %s, NOW(), false, false, %s, %s, %s, %s)"""
            execute_sql_write(query, (user_id, nudge_type, emb_str, context_summary, nudge_content, nudge_style))
        else:
            query = "INSERT INTO nudge_outcomes (user_id, nudge_type, sent_at, responded, meal_logged, nudge_style) VALUES (%s, %s, NOW(), false, false, %s)"
            execute_sql_write(query, (user_id, nudge_type, nudge_style))
    except Exception as e:
        logger.error(f"Error logging nudge outcome: {e}")

def get_nudge_response_rate(user_id: str, nudge_type: str = None):
    try:
        if nudge_type:
            res = execute_sql_query("SELECT COUNT(*) as total, SUM(CASE WHEN responded THEN 1 ELSE 0 END) as responded_count FROM nudge_outcomes WHERE user_id = %s AND nudge_type = %s", (user_id, nudge_type), fetch_one=True)
        else:
            res = execute_sql_query("SELECT COUNT(*) as total, SUM(CASE WHEN responded THEN 1 ELSE 0 END) as responded_count FROM nudge_outcomes WHERE user_id = %s", (user_id,), fetch_one=True)
            
        if res and res.get("total", 0) > 0:
            return float(res["responded_count"] or 0) / res["total"], res["total"]
    except Exception as e:
        logger.error(f"Error getting nudge response rate: {e}")
    return 1.0, 0

def get_daily_nudge_count(user_id: str) -> int:
    try:
        res = execute_sql_query("SELECT COUNT(*) as total FROM nudge_outcomes WHERE user_id = %s AND DATE(sent_at) = CURRENT_DATE", (user_id,), fetch_one=True)
        return res.get("total", 0) if res else 0
    except Exception as e:
        logger.error(f"Error getting daily nudge count: {e}")
        return 0


def classify_nudge_sentiment(user_reply: str) -> dict:
    import json
    
    prompt = f"""Analiza la siguiente respuesta de un usuario a un recordatorio (nudge) para registrar su comida.
Debes determinar tres cosas:
1. sentiment: El sentimiento principal de la respuesta. Selecciona SOLO UNO de: positive, neutral, annoyed, guilt, motivation, curiosity, frustration, sadness.
2. meal_logged: Booleano (true/false) que indica si en este mensaje el usuario está efectivamente reportando lo que comió o confirmando que ya comió.
3. causal_reason: Si NO comió lo planeado (abandonó la comida), clasifica la razón principal. Selecciona SOLO UNO de: no_time, ate_out, no_ingredients, not_hungry, didnt_like, null (si no aplica o sí comió).

Respuesta del usuario: "{user_reply}"

Devuelve ÚNICAMENTE un JSON válido con las claves "sentiment", "meal_logged" y "causal_reason". No uses bloques markdown."""

    try:
        chat_llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite-preview", 
            temperature=0.1,
            google_api_key=os.environ.get("GEMINI_API_KEY")
        )
        res = chat_llm.invoke(prompt)
        text = str(res.content).strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "").strip()
        elif text.startswith("```"):
            text = text.replace("```", "").strip()
            
        data = json.loads(text)
        return {
            "sentiment": data.get("sentiment", "neutral"),
            "meal_logged": data.get("meal_logged", False),
            "causal_reason": data.get("causal_reason")
        }
    except Exception as e:
        logger.error(f"Error clasificando sentimiento de nudge: {e}")
        return {"sentiment": "neutral", "meal_logged": False, "causal_reason": None}

def handle_nudge_response(user_id: str, content: str):
    try:
        pending = execute_sql_query(
            "SELECT id, nudge_type FROM nudge_outcomes WHERE user_id = %s AND responded = false AND sent_at >= NOW() - INTERVAL '60 minutes' LIMIT 1",
            (user_id,), fetch_one=True
        )
        if pending:
            nudge_id = pending['id']
            nudge_type = pending.get('nudge_type', 'Desconocido')
            classification = classify_nudge_sentiment(content)
            sentiment = classification['sentiment']
            meal_logged = classification['meal_logged']
            causal_reason = classification.get('causal_reason')
            
            try:
                execute_sql_write(
                    "UPDATE nudge_outcomes SET responded = true, response_sentiment = %s, meal_logged = %s WHERE id = %s",
                    (sentiment, meal_logged, nudge_id)
                )
            except Exception as e:
                if "column" in str(e).lower() or "42703" in str(e):
                    try:
                        execute_sql_write("ALTER TABLE nudge_outcomes ADD COLUMN response_sentiment VARCHAR(50)")
                    except Exception: pass
                    try:
                        execute_sql_write("UPDATE nudge_outcomes SET responded = true, response_sentiment = %s, meal_logged = %s WHERE id = %s", (sentiment, meal_logged, nudge_id))
                    except Exception:
                        execute_sql_write("UPDATE nudge_outcomes SET responded = true WHERE id = %s", (nudge_id,))
            
            logger.info(f"✅ Nudge {nudge_id} respondido por {user_id}. Sentiment: {sentiment}, Logged: {meal_logged}, Causal: {causal_reason}")

            # Persist causal reason if meal was abandoned
            if not meal_logged and causal_reason and causal_reason != "null":
                try:
                    execute_sql_write(
                        "INSERT INTO abandoned_meal_reasons (user_id, meal_type, reason) VALUES (%s, %s, %s)",
                        (user_id, nudge_type, causal_reason)
                    )
                    logger.info(f"🧠 Causal reason '{causal_reason}' saved for {nudge_type} (User: {user_id})")
                except Exception as e:
                    logger.error(f"Error persisting causal reason: {e}")
                    
    except Exception as e:
        logger.error(f"Error procesando respuesta al nudge: {e}")

def run_proactive_checks():
    """Esta función será llamada por apscheduler (cron job)."""
    logger.info("⏱️ [CRON] Iniciando verificación proactiva de comidas.")
    
    # --- PHASE 3: JIT Rolling Window Trigger ---
    # check_and_trigger_jit_rolling_windows() # Desactivado: El paso a Micro-Batching usa triggers interactivos vía UI ("Actualizar Platos")

    
    # 1. Obtenemos la hora actual (AST / -04:00)
    now_ast = datetime.now(timezone(timedelta(hours=-4)))
    current_hour_float = now_ast.hour + now_ast.minute / 60.0
    
    sessions = get_active_users_for_proactive()
    logger.info(f"🔍 [CRON] Encontradas {len(sessions)} sesiones activas para verificar (Proactividad Inteligente).")
    
    for s in sessions:
        session_id = str(s.get("id"))
        user_id = str(s.get("user_id"))
        # GAP 3: Nudge Budget (max 2 nudges per day to avoid fatigue)
        daily_nudges = get_daily_nudge_count(user_id)
        if daily_nudges >= 2:
            logger.info(f"🛑 [CRON] Usuario {user_id} ya agotó su presupuesto de nudges hoy ({daily_nudges}/2). Saltando.")
            continue
            
        # Global stats para el tono base
        global_rate, global_total = get_nudge_response_rate(user_id)
        
        base_tone_instruction = "Usa un tono amistoso y motivacional, nunca de regaño, ni parezcas un robot asustadizo."
        send_push = True
        
        if global_total >= 5:
            if global_rate < 0.20:
                logger.info(f"📉 [CRON] Usuario {user_id} tiene response rate muy bajo ({global_rate:.0%}). Cambiando a tono empático.")
                send_push = False
                base_tone_instruction = "El usuario ha estado ignorando notificaciones recientemente. Usa un tono empático, pregúntale si hay algún obstáculo, estrés o falta de tiempo que le impida registrar sus comidas. NO asumas que se le olvidó, asume que podría estar ocupado o desmotivado. Sé muy breve y sin presiones."
            elif global_rate > 0.70:
                logger.info(f"🌟 [CRON] Usuario {user_id} tiene response rate alto ({global_rate:.0%}). Usando tono de refuerzo positivo.")
                base_tone_instruction = "El usuario tiene excelente disciplina. Usa un tono de celebración y refuerzo positivo animándolo a mantener la racha."
        
        meal_to_check = None
        trigger_time_str = ""
        final_tone_instruction = base_tone_instruction
        
        # Resumen del día siempre a las 11 PM
        if now_ast.hour == 23:
            meal_to_check = "Resumen del día"
            trigger_time_str = "11:00 PM"
        else:
            from db_facts import get_avg_meal_hour
            import math
            
            # Horarios default (9AM, 1PM, 4PM, 7:30PM)
            defaults = {
                "Desayuno": 9.0,
                "Almuerzo": 13.0,
                "Merienda": 16.0,
                "Cena": 19.5
            }
            
            for meal, def_hour in defaults.items():
                avg_hr = get_avg_meal_hour(user_id, meal)
                if avg_hr is None:
                    avg_hr = def_hour
                
                # GAP 3: Cadencia por comida
                meal_rate, meal_total = get_nudge_response_rate(user_id, meal)
                delay_hours = 1.5 # Default
                
                if meal_total >= 3:
                    if meal_rate > 0.70:
                        delay_hours = 1.0 # Responde rápido y seguro, mandamos antes
                    elif meal_rate < 0.30:
                        delay_hours = 2.5 # Evitar presión, retrasamos el nudge
                        
                # Nudge dinámico ajustado según historial de adherencia específica
                nudge_hour = avg_hr + delay_hours
                
                # Comparamos si el cron actual (hora entera) coincide con la hora entera del nudge
                if math.floor(current_hour_float) == math.floor(nudge_hour):
                    meal_to_check = meal
                    hours = int(nudge_hour)
                    mins = int((nudge_hour - hours) * 60)
                    am_pm = "AM" if hours < 12 else "PM"
                    display_hr = hours if hours <= 12 else hours - 12
                    if display_hr == 0: display_hr = 12
                    trigger_time_str = f"{display_hr}:{mins:02d} {am_pm}"
                    
                    if meal_rate < 0.30 and meal_total >= 3:
                        final_tone_instruction = "El usuario frecuentemente ignora o abandona esta comida específica. Pregúntale qué está fallando particularmente con esta comida (ej. tiempo, no le gusta, está fuera de casa) sin sonar acusador."
                    break
        
        tone_instruction = final_tone_instruction
        
        if not meal_to_check:
            continue
            
        logger.info(f"🔍 [CRON] Verificando {meal_to_check} para el usuario {user_id} (Nudge Dinámico: {trigger_time_str}).")
        
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
                
            # La evaluación de send_push ya se hizo al inicio del bucle por la Mejora 3
            
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
                
                # --- GAP 3: Embedding-based Nudge Personalization ---
                context_summary = f"Usuario ignoró {meal_to_check} {int((1-meal_rate)*meal_total)} veces de {meal_total} registradas. Tono base: {tone_instruction}"
                context_embedding = None
                proven_strategies_text = ""
                try:
                    context_embedding = get_embedding(context_summary)
                    if context_embedding:
                        emb_str = f"[{','.join(map(str, context_embedding))}]"
                        query = "SELECT * FROM match_successful_nudges(query_embedding => %s, match_threshold => 0.85, match_count => 2)"
                        successful_nudges = execute_sql_query(query, (emb_str,), fetch_all=True)
                        if successful_nudges:
                            proven_strategies_text = "Estrategias previas que funcionaron con usuarios en situaciones emocionales idénticas:\n"
                            for sn in successful_nudges:
                                n_content = sn.get("nudge_content", "")
                                n_sentiment = sn.get("response_sentiment", "motivado")
                                proven_strategies_text += f"- (Éxito: sintió {n_sentiment}): \"{n_content}\"\n"
                            proven_strategies_text += "\nInspírate en estas estrategias para tu respuesta (NO las copies exactas).\n"
                except Exception as emb_err:
                    logger.error(f"Error en embedding de nudge: {emb_err}")
                
                final_tone = (tone_instruction or "") + "\n" + proven_strategies_text
                
                # --- GAP 4: A/B Testing de Formato ---
                nudge_style = get_best_nudge_style(user_id)
                style_instruction = ""
                if nudge_style == "directo":
                    style_instruction = "Haz una pregunta directa y al grano sin rodeos."
                elif nudge_style == "sugestivo":
                    style_instruction = "Haz una sugerencia suave y comprensiva, aportando opciones."
                elif nudge_style == "gamificado":
                    style_instruction = "Usa un tono de reto amistoso, motivando como si fuera un logro a desbloquear."
                
                prompt = PROACTIVE_PROMPT.format(
                    missing_meal=meal_to_check,
                    trigger_time=trigger_time_str,
                    diet_type=diet_type,
                    goals=goals,
                    tone_instruction=final_tone,
                    style_instruction=style_instruction
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
                
                # Intentar pasar embedding si se generó, para GAP 3 y style para GAP 4
                try:
                    log_nudge_outcome(user_id, meal_to_check, context_embedding=locals().get("context_embedding"), context_summary=locals().get("context_summary"), nudge_content=content, nudge_style=locals().get("nudge_style"))
                except Exception as log_err:
                    log_nudge_outcome(user_id, meal_to_check, nudge_style=locals().get("nudge_style"))
                
                if send_push:
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
