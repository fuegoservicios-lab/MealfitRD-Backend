import os
import logging
from datetime import datetime, timezone, timedelta
from langchain_google_genai import ChatGoogleGenerativeAI

from db_core import connection_pool, execute_sql_query, execute_sql_write
from db_chat import save_message, get_recent_messages
from db import get_consumed_meals_today, get_user_profile
from fact_extractor import get_embedding
from knobs import _env_int, _env_float

logger = logging.getLogger(__name__)

from prompts.proactive import PROACTIVE_PROMPT


# [P3-PREVIEW-MODEL-KNOB · 2026-05-12] Knob para overridear el modelo
# Gemini usado por las 2 callsites del proactive agent sin redeploy:
#   - `classify_nudge_sentiment` (analiza respuestas del usuario al nudge).
#   - `_compose_proactive_message` (genera el texto del nudge).
#
# Por qué un knob (no hardcode):
#   El stack actual usa `gemini-3.1-flash-lite` consistentemente
#   (graph_orchestrator `_PRO_MODEL_NAME`, fact_extractor, agent, etc.).
#   Cambiar el default acá rompería la consistencia del stack. Pero los
#   modelos `*-preview` de Google pueden deprecarse/retirarse sin aviso
#   prolongado — el audit 2026-05-11 documentó CB rows stale por el
#   modelo `gemini-3.1-pro-preview` 4.4 días seguidos. Si Google retira
#   el flash-lite-preview, el cron de nudges deja de clasificar
#   sentiment + no envía mensajes hasta el próximo deploy. Knob permite
#   swap inmediato sin redeploy: setear
#   `MEALFIT_PROACTIVE_SENTIMENT_MODEL=gemini-3.1-flash` (stable, sin
#   `-preview`) y reiniciar el worker — el cron retomará operación.
#
# Default = current production model. Cambiar en env vars cuando
# Google publique notice de deprecation.
def _proactive_model_name() -> str:
    return os.environ.get(
        "MEALFIT_PROACTIVE_SENTIMENT_MODEL",
        "gemini-3.1-flash-lite",
    )


# [P2-LLM-TIMEOUT-SWEEP · 2026-05-30] Timeout per-invoke de los 2 constructores
# `ChatGoogleGenerativeAI` del proactive agent: `classify_nudge_sentiment` (148)
# y el compose-nudge del cron `run_proactive_checks` (457). Pre-fix: sin
# `timeout=`. `run_proactive_checks` es SÍNCRONO y corre en el threadpool de
# APScheduler con `max_instances=1`: si Gemini cuelga un socket, el invoke
# bloquea el thread del cron indefinidamente → el slot del job queda tomado y el
# nudge cron NUNCA vuelve a correr (no dispara MISSED ni ERROR — está "running",
# no errored → ningún watchdog lo ve). El budget wall-clock (_max_runtime_s) solo
# se chequea al tope de cada iteración de usuario, no puede abortar un invoke en
# vuelo. El `timeout=` propaga al deadline gRPC → DeadlineExceeded, capturado por
# los `except Exception` existentes (compose por-usuario contenido). Default 20s
# (flash-lite, prompt corto); clamp (0, 120]. Knob auto-registrado.
# Tooltip-anchor: P2-LLM-TIMEOUT-SWEEP.
def _proactive_llm_timeout_s() -> float:
    return _env_float(
        "MEALFIT_PROACTIVE_LLM_TIMEOUT_S",
        20.0,
        validator=lambda v: 0.0 < v <= 120.0,
    )


# [P1-PROACTIVE-TZ · 2026-05-30] Offset (en minutos, UTC→local sumando) de la
# zona horaria dominicana (AST = UTC-4, sin DST). El cron computa `now_ast`
# con `-4h` hardcodeado; este knob mantiene la MISMA constante para el filtro
# de comidas consumidas y la convierte en operacional sin redeploy si DR
# adoptara DST. Convención `getTimezoneOffset()` de JS = +240 para UTC-4 (lo
# que `get_consumed_meals_today` suma a la fecha local para ir a UTC).
# Tooltip-anchor: P1-PROACTIVE-TZ.
def _proactive_tz_offset_min() -> int:
    return _env_int(
        "MEALFIT_PROACTIVE_TZ_OFFSET_MIN",
        240,
        validator=lambda v: 0 <= v <= 720,
    )

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
        # [P2-PROACTIVE-NUDGE-BUDGET-TZ · 2026-05-30] Contar contra el día
        # CALENDARIO AST, no el UTC. Pre-fix: `DATE(sent_at) = CURRENT_DATE`
        # con DB en TimeZone=UTC contaba el día UTC, que rota a las 20:00 AST
        # (=00:00 UTC). Como los nudges se agendan en reloj AST y abarcan un día
        # AST que cruza el límite UTC a las 20:00, hasta 2 nudges diurnos (día
        # UTC D) + 2 vespertinos/Resumen 20:00-23:00 AST (día UTC D+1) = 4 en un
        # mismo día AST, el DOBLE del cap anti-fatiga (>=2). Convertir a AST
        # alinea el conteo con el reloj de agendado. Reusa la conversión
        # 'America/Santo_Domingo' ya usada en get_avg_meal_hour (db_facts.py).
        # Tooltip-anchor: P2-PROACTIVE-NUDGE-BUDGET-TZ.
        res = execute_sql_query(
            "SELECT COUNT(*) as total FROM nudge_outcomes "
            "WHERE user_id = %s "
            "AND (sent_at AT TIME ZONE 'America/Santo_Domingo')::date "
            "= (NOW() AT TIME ZONE 'America/Santo_Domingo')::date",
            (user_id,), fetch_one=True,
        )
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
            model=_proactive_model_name(),
            temperature=0.1,
            google_api_key=os.environ.get("GEMINI_API_KEY"),
            timeout=_proactive_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
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
                logger.error(f"Error actualizando nudge_outcomes id={nudge_id}: {e}")
            
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

    # [P1-PROACTIVE-BUDGET · 2026-05-28] Cota de escala del cron de nudges.
    # El loop hace ~10 queries/usuario (N+1: daily_nudge_count + global rate +
    # 4×(avg_meal_hour + per-meal rate) + perfil/consumed/embedding) y 1 LLM
    # invoke SERIAL cuando dispara nudge. A 1k-10k usuarios activos esto explota
    # a 10k-150k queries + miles de invokes serial por tick, solapándose con el
    # siguiente tick y saturando el pool de DB/threads. Sin cambiar la lógica de
    # decisión, acotamos por tick: usuarios procesados, wall-clock total, y nudges
    # (=invokes LLM). El excedente se atiende en ticks subsecuentes. Knobs con
    # clamps; runtime/nudges = 0 desactiva esa cota. Tooltip-anchor: P1-PROACTIVE-BUDGET.
    _max_users = max(1, min(_env_int("MEALFIT_PROACTIVE_MAX_USERS_PER_TICK", 250), 100000))
    _max_runtime_s = max(0, min(_env_int("MEALFIT_PROACTIVE_MAX_RUNTIME_S", 240), 3600))
    _max_nudges = max(0, min(_env_int("MEALFIT_PROACTIVE_MAX_NUDGES_PER_TICK", 150), 100000))
    _t_start = datetime.now(timezone.utc)
    _nudges_sent = 0
    if len(sessions) > _max_users:
        logger.warning(
            f"⚠️ [P1-PROACTIVE-BUDGET] {len(sessions)} sesiones activas > cap "
            f"{_max_users}/tick. Procesando las primeras {_max_users}; el resto "
            f"se atenderá en ticks subsecuentes."
        )
        sessions = sessions[:_max_users]

    for s in sessions:
        # [P1-PROACTIVE-BUDGET] cotas de wall-clock y de nudges (gasto LLM) por tick
        if _max_runtime_s and (datetime.now(timezone.utc) - _t_start).total_seconds() > _max_runtime_s:
            logger.warning(
                f"⚠️ [P1-PROACTIVE-BUDGET] Wall-clock > {_max_runtime_s}s "
                f"(enviados={_nudges_sent}); abortando resto del tick."
            )
            break
        if _max_nudges and _nudges_sent >= _max_nudges:
            logger.warning(
                f"⚠️ [P1-PROACTIVE-BUDGET] Cap de nudges/tick ({_max_nudges}) "
                f"alcanzado; abortando resto del tick."
            )
            break
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
            # [P1-PROACTIVE-TZ · 2026-05-30] PASAR tz_offset_mins. Pre-fix se
            # pasaba `date_str` SIN offset → `get_consumed_meals_today` caía a
            # su rama `else` (UTC), descartando `date_str` y construyendo la
            # ventana con el día UTC. El "Resumen del día" dispara a
            # now_ast.hour==23 (= 03:00 UTC del día siguiente): a esa hora la
            # ventana UTC `[00:00Z..23:59Z]` del día ya rotado = `[20:00 AST hoy
            # .. 19:59 AST mañana]`, EXCLUYENDO desayuno/almuerzo/cena
            # registrados antes de las 20:00 AST. Un usuario cumplidor caía a
            # `consumed==[]` y recibía el nudge indulgente "no registraste nada,
            # ¿descuento todo de tu nevera?" — falso-positivo NOCTURNO para cada
            # usuario standard. Con el offset, la rama AST-aware usa el día AST
            # correcto. Tooltip-anchor: P1-PROACTIVE-TZ.
            consumed = get_consumed_meals_today(
                user_id,
                date_str=now_ast.strftime("%Y-%m-%d"),
                tz_offset_mins=_proactive_tz_offset_min(),
            )
            
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
                model=_proactive_model_name(),
                temperature=0.8,
                google_api_key=os.environ.get("GEMINI_API_KEY"),
                timeout=_proactive_llm_timeout_s(),  # [P2-LLM-TIMEOUT-SWEEP · 2026-05-30]
            )
            response = chat_llm.invoke(prompt)
            _nudges_sent += 1  # [P1-PROACTIVE-BUDGET] cuenta el gasto LLM real del tick
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

# ============================================================
# [P3-COLDSTART-E2E · 2026-05-29] ⚰️ CÓDIGO MUERTO — NO REACTIVAR SIN MIGRAR
# ------------------------------------------------------------
# Las DOS funciones JIT Rolling Window de abajo
# (`_trigger_week2_background_generation` + `check_and_trigger_jit_rolling_windows`)
# están DESACTIVADAS. Su única invocación está comentada en
# `run_proactive_checks` (línea ~213: "# check_and_trigger_jit_rolling_windows()
# # Desactivado: El paso a Micro-Batching usa triggers interactivos") y NINGÚN
# cron las registra (`register_plan_chunk_scheduler` en cron_tasks.py no las lista).
#
# El disparo REAL del próximo chunk vive 100% en `plan_chunk_queue` +
# `process_plan_chunk_queue` (cron_tasks.py:23122): todos los chunks 2..N se
# encolan al CREAR el plan con un `execute_after` de calendario y el worker
# (cada 1 min, server-side) los levanta cuando llega su fecha. Modelo mental
# único: "todo se dispara por plan_chunk_queue".
#
# Por qué NO se borran: el helper de append (`_apply_week2_append` →
# `update_plan_data_atomic`) tiene cobertura de regresión lost-update activa en
# `test_p1_audit_1_update_meal_plan_data_lostupdate.py`. Borrar las funciones perdería
# ese path de test. Se MARCAN como DEAD para que un mantenedor no las reactive
# por error: reactivarlas junto al worker = DOBLE generación del mismo bloque, y
# el cuerpo exige `len(days) == 7` exacto (incompatible con chunks de 3 días del
# micro-batching actual).
# Tooltip-anchor: P3-COLDSTART-E2E-JIT-DEAD.
# ============================================================
def _trigger_week2_background_generation(user_id, plan_id, existing_plan_data):
    """[⚰️ DEAD CODE — ver banner P3-COLDSTART-E2E-JIT-DEAD arriba] Generates the
    next 7 days in the background and appends to the existing plan.

    NO invocada en producción (micro-batching usa plan_chunk_queue). Conservada
    solo por la cobertura lost-update de `update_plan_data_atomic`."""
    from graph_orchestrator import run_plan_pipeline
    from db_profiles import get_user_profile
    from db_plans import update_plan_data_atomic
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
            # [P1-NEW-9 · 2026-05-11] Atribución del caller para que
            # `_emit_plan_quality_degraded_alert` (graph_orchestrator
            # should_retry, las 5 ramas "end") correlacione el alert
            # al plan_id correcto. Sin estos kwargs, el alert antes
            # quedaba con alert_key="plan_quality_degraded:<user>:no_plan_id"
            # porque `plan_result` de la extensión week-2 no trae `id`
            # (el meal_plan ya existe en DB, este pipeline NO inserta).
            # Resultado pre-fix: SRE veía la alert pero NO sabía cuál
            # plan se extendió degradado, y los alerts colapsaban con
            # alert_key={user_id}:no_plan_id usados para /generate-plan.
            form_data["_caller_target_plan_id"] = plan_id
            form_data["_caller_context"] = "jit_week2"

            likes = get_user_likes(user_id)
            rejections = get_active_rejections(user_id)
            rej_names = [r["meal_name"] for r in rejections] if rejections else []
            taste_profile = analyze_preferences_agent(likes, [], active_rejections=rej_names)

            result = run_plan_pipeline(form_data, [], taste_profile, memory_context="")

            new_days = result.get("days", [])
            if not new_days:
                logger.warning(
                    f"[JIT BG TASK] run_plan_pipeline retornó 0 días para "
                    f"user={user_id} plan={plan_id}. Skip persistencia."
                )
                return

            # [P1-AUDIT-1 · 2026-05-15] Append + persistencia atómica bajo
            # FOR UPDATE row lock (vía `update_plan_data_atomic`). Cierre del
            # follow-up natural documentado en P1-RECALC-LOSTUPDATE
            # (2026-05-14):
            #
            # Pre-fix flow:
            #   t=0  El handler `proactive_agent` recibe `existing_plan_data`
            #        leído por el cron `check_and_trigger_jit_rolling_windows`
            #        (línea ~524, SELECT plano sin lock).
            #   t=1  Mutación in-memory: `existing_plan_data["days"] = existing
            #        _days + new_days` con re-numeración de `day`.
            #   t=2  acquire advisory lock + UPDATE full-overwrite via
            #        `update_meal_plan_data` (P1-NEXT-1).
            #
            # Ventana lost-update entre t=0 y t=2 (puede ser HORAS si
            # `run_plan_pipeline` es lento — la generación LLM puede tomar
            # 30-180s, multiplicando el tamaño de la ventana): si un endpoint
            # hermano muta `plan_data` quirúrgico entre el cron read y
            # nuestro UPDATE, esa mutación se pierde. JIT week-2 es el caso
            # con la ventana MÁS LARGA del sistema (chunk_worker, swap-meal,
            # /recipe/expand son sub-segundo).
            #
            # Fix: `update_plan_data_atomic` re-SELECTea plan_data FRESH bajo
            # FOR UPDATE row lock al momento de persistir, así el callback
            # appendea los new_days a `plan_data_fresh.days` (post-merge de
            # cualquier mutación que ocurrió en la ventana de 30-180s) — el
            # número de días base puede haber cambiado, el re-numerado se
            # hace contra el fresh.
            #
            # Tooltip-anchor: P1-AUDIT-1-JIT-WEEK2-START |
            # test_p1_audit_1_proactive_week2_lostupdate
            def _apply_week2_append(plan_data_fresh: dict) -> dict | bool:
                """Appendea `new_days` a `plan_data_fresh.days` re-numerando
                contra la longitud fresh. Si days no es lista, retorna False
                para abortar UPDATE.
                """
                if not isinstance(plan_data_fresh, dict):
                    return False
                existing_days_fresh = plan_data_fresh.get("days") or []
                if not isinstance(existing_days_fresh, list):
                    existing_days_fresh = []
                start_idx = len(existing_days_fresh) + 1
                for i, d in enumerate(new_days):
                    if isinstance(d, dict):
                        d["day"] = start_idx + i
                plan_data_fresh["days"] = existing_days_fresh + new_days
                return plan_data_fresh

            # [P2-OPEN-1] user_id pasado al helper para que el SELECT/UPDATE
            # filtren `AND user_id = %s` defense-in-depth. plan_id viene del
            # cron `check_and_trigger_jit_rolling_windows` que ya tiene
            # ownership (SELECT con user_id en el row).
            merged = update_plan_data_atomic(
                plan_id, _apply_week2_append, user_id=user_id
            )
            if not merged:
                logger.warning(
                    f"⚠️ [JIT BG TASK] update_plan_data_atomic retornó vacío "
                    f"para plan={plan_id} user={user_id}: el plan desapareció "
                    f"o el filtro user_id no matchea. Week-2 NO persistida."
                )
                return
            logger.info(f"✅ [JIT BG TASK] Semana 2 añadida exitosamente al plan {plan_id} de {user_id}")
        except Exception as e:
            logger.error(f"❌ [JIT BG TASK] Error en generación de semana 2 para {user_id}: {e}")

    threading.Thread(target=_bg_task, daemon=True).start()

def check_and_trigger_jit_rolling_windows():
    """[⚰️ DEAD CODE — ver banner P3-COLDSTART-E2E-JIT-DEAD arriba]
    JIT Rolling Windows Trigger:
    Detects users who are on Day 5 or 6 (i.e. plan generated 4-6 days ago)
    and if their plan has only 7 days, generates Week 2.

    DESACTIVADA: su único callsite está comentado en `run_proactive_checks`.
    El disparo del próximo chunk vive en `process_plan_chunk_queue`
    (cron_tasks.py:23122). No reactivar sin migrar — ver banner.
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
