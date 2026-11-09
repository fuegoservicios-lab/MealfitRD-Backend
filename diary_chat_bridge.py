"""
backend/diary_chat_bridge.py
Puente de sincronización en tiempo real entre el diario de comidas manual / escáner / plan y el Coach conversacional.

Cuando el usuario registra una comida fuera del chat (botón «+ Registrar comida», «+ Registrar en este día»,
escáner de fotos, repetidor o desde el plan), este módulo:
1. Resuelve o inicializa la sesión activa del día del usuario.
2. Inyecta en el hilo conversacional (`agent_messages`) el turno simétrico:
   - Mensaje del usuario (ej: «Registré mi desayuno: Mangú de plátano maduro con pechuga...»)
   - Confirmación del Coach (ej: «Anoté tu **desayuno**: ... (~590 kcal y ~45 g de proteína). Te quedan ~1710 kcal...»)
3. Actualiza el estado del checkpointer de LangGraph si la sesión está cargada.
4. Permite que el Coach conozca en todo momento el contexto, ingredientes y ID_REGISTRO_DIARIO de las comidas manuales.
"""

import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, Any

from db_chat import save_message, get_or_create_session
from db_facts import get_consumed_meals_today
from db import execute_sql_query

logger = logging.getLogger(__name__)

_DEFAULT_TZ_OFFSET_MIN = 240  # UTC-4 (República Dominicana)


def _user_tz_offset(user_id: str) -> int:
    """Obtiene el offset de zona horaria en minutos del usuario o el fallback RD (240)."""
    try:
        from db import user_tz_offset_min
        tz = user_tz_offset_min(user_id)
        return int(tz) if tz is not None else _DEFAULT_TZ_OFFSET_MIN
    except Exception:
        return _DEFAULT_TZ_OFFSET_MIN


def _now_utc() -> datetime:
    """Helper mockeable para tiempo actual UTC."""
    return datetime.now(timezone.utc)


def get_or_resolve_today_session_for_user(user_id: str, session_id: Optional[str] = None) -> str:
    """
    Resuelve el session_id activo para el usuario:
    1. Si se pasa session_id y pertenece al usuario, lo respeta.
    2. Si no, busca la sesión más reciente del usuario en agent_sessions.
    3. Si la sesión más reciente es de hoy en hora local del usuario, la usa.
    4. Si no existe o es de un día anterior, crea una sesión nueva con get_or_create_session.
    """
    if not user_id or user_id == "guest":
        return str(session_id or uuid.uuid4())

    tz_min = _user_tz_offset(user_id)
    now_utc = _now_utc()
    # [BB-044 / P1-CHAT-ROLLOVER-4AM · 2026-11-09] Rollover biológico a las 4:00 AM local
    today_local = (now_utc - timedelta(minutes=tz_min) - timedelta(hours=4)).date()

    if session_id:
        try:
            row = execute_sql_query(
                "SELECT id::text AS id, user_id::text AS user_id FROM public.agent_sessions WHERE id = %s",
                (str(session_id),),
                fetch_one=True,
            )
            if row:
                owner = row.get("user_id")
                if not owner or owner == str(user_id):
                    return str(session_id)
        except Exception as e:
            logger.warning(f"[diary_chat_bridge] Error verificando session_id {session_id}: {e}")

    # Buscar la sesión más reciente del usuario
    try:
        fila = execute_sql_query(
            "SELECT id::text AS id, created_at FROM public.agent_sessions "
            "WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
            (str(user_id),),
            fetch_one=True,
        )
        if fila and fila.get("id"):
            created_at = fila.get("created_at")
            if isinstance(created_at, datetime):
                # Normalizar a offset-aware UTC si es naive
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                dia_sesion = (created_at - timedelta(minutes=tz_min) - timedelta(hours=4)).date()
                if dia_sesion >= today_local:
                    return str(fila["id"])
    except Exception as e:
        logger.warning(f"[diary_chat_bridge] Error buscando última sesión de {user_id}: {e}")

    # Crear una nueva sesión para hoy
    nueva = str(uuid.uuid4())
    try:
        get_or_create_session(nueva, user_id=user_id)
        return nueva
    except Exception as e:
        logger.error(f"❌ [diary_chat_bridge] No se pudo crear sesión nueva para {user_id}: {e}")
        return nueva


_SUPPLEMENT_NAME_KEYWORDS: tuple[str, ...] = (
    "mass gainer", "gainer", "weight gainer", "whey", "proteina en polvo", "proteína en polvo",
    "creatina", "creatine", "bcaa", "caseina", "caseína", "glutamina", "glutamine",
    "pre entreno", "pre workout", "preentreno", "preworkout", "multivitamin", "multivitamínico",
    "colageno", "colágeno", "electrolitos", "electrolytes", "omega 3", "omega3",
    "magnesio", "zinc", "ashwagandha", "melatonina"
)


def _es_suplemento(meal_type: Optional[str], meal_name: str) -> bool:
    """[BB-062] Determina si un registro corresponde a un suplemento por su franja o nombre comercial."""
    _tipo = (meal_type or "").strip().lower()
    if _tipo in ("suplemento", "supplement", "suplementos"):
        return True
    _nombre = (meal_name or "").lower()
    return any(k in _nombre for k in _SUPPLEMENT_NAME_KEYWORDS)


def build_sync_turn_texts(
    meal_name: str,
    meal_type: Optional[str],
    calories: int,
    protein: int,
    days_ago: int = 0,
    user_id: Optional[str] = None,
    meal_id: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Construye el texto del mensaje de usuario y la confirmación del Coach.
    """
    _tipo_orig = (meal_type or "").strip().lower()
    _nombre = (meal_name or "").strip() or "Comida"
    _cal = int(round(float(calories or 0)))
    _prot = int(round(float(protein or 0)))
    _is_sup = _es_suplemento(_tipo_orig, _nombre)
    _tipo = "suplemento" if _is_sup else _tipo_orig

    # --- TEXTO DE USUARIO ---
    if _tipo_orig in ("suplemento", "supplement", "suplementos"):
        if days_ago == 0:
            user_text = f"Registré mi suplemento: {_nombre}"
        elif days_ago == 1:
            user_text = f"Registré mi suplemento de ayer: {_nombre}"
        else:
            user_text = f"Registré mi suplemento de hace {days_ago} días: {_nombre}"
    elif _tipo in ("desayuno", "almuerzo", "cena", "merienda", "snack"):
        if days_ago == 0:
            user_text = f"Registré mi {_tipo}: {_nombre}"
        elif days_ago == 1:
            user_text = f"Registré mi {_tipo} de ayer: {_nombre}"
        else:
            user_text = f"Registré mi {_tipo} de hace {days_ago} días: {_nombre}"
    else:
        if days_ago == 0:
            user_text = f"Registré en mi diario: {_nombre}"
        elif days_ago == 1:
            user_text = f"Registré en mi diario de ayer: {_nombre}"
        else:
            user_text = f"Registré en mi diario de hace {days_ago} días: {_nombre}"

    # --- TEXTO DEL COACH ---
    if _is_sup:
        header = f"Anoté tu **suplemento**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
    elif _tipo in ("desayuno", "almuerzo", "cena", "merienda", "snack"):
        if days_ago == 0:
            header = f"Anoté tu **{_tipo}**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
        elif days_ago == 1:
            header = f"Anoté tu **{_tipo} de ayer**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
        else:
            header = f"Anoté tu **{_tipo} de hace {days_ago} días**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
    elif _tipo == "extra":
        if days_ago == 0:
            header = f"Anoté tu **extra**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
        elif days_ago == 1:
            header = f"Anoté tu **extra de ayer**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
        else:
            header = f"Anoté tu **extra de hace {days_ago} días**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
    elif _tipo == "complemento":
        if days_ago == 0:
            header = f"Anoté tu **complemento**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
        elif days_ago == 1:
            header = f"Anoté tu **complemento de ayer**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
        else:
            header = f"Anoté tu **complemento de hace {days_ago} días**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
    else:
        if days_ago == 0:
            header = f"Anoté tu **comida**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
        elif days_ago == 1:
            header = f"Anoté tu **comida de ayer**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"
        else:
            header = f"Anoté tu **comida de hace {days_ago} días**: {_nombre}; aprox. **{_cal} kcal** y **{_prot} g de proteína**"

    # Cierre / frase de seguimiento
    if days_ago > 0:
        followup = ". No suma a las macros de hoy."
    else:
        # Calcular macros restantes de hoy
        consumed_today = []
        if user_id:
            try:
                _tz = _user_tz_offset(user_id)
                consumed_today = get_consumed_meals_today(user_id, tz_offset_mins=_tz) or []
            except Exception as _ce:
                logger.debug(f"[diary_chat_bridge] Error leyendo comidas de hoy: {_ce}")

        total_cals = sum(int(m.get("calories") or 0) for m in consumed_today)
        total_prot = sum(int(m.get("protein") or 0) for m in consumed_today)

        target_cals = None
        target_prot = None
        if user_id:
            # 1. Intentar resolver metas dinámicas del día (SSOT de nutrición para planes dinámicos y contadores) [BB-063]
            try:
                from tools import _perfil_y_plan_de_hoy
                from coach_day_context import metas_del_dia
                _hp, _plan = _perfil_y_plan_de_hoy(user_id)
                _metas = metas_del_dia(_hp, _plan)
                if _metas:
                    target_cals = _metas.get("kcal")
                    target_prot = _metas.get("protein_g")
            except Exception as _m_err:
                logger.debug(f"[diary_chat_bridge] metas_del_dia fallback: {_m_err}")

            # 2. Fallback a health_profile o meal_plans para compatibilidad con perfiles estáticos o mocks de tests
            if not target_cals:
                try:
                    from db import get_user_profile, get_latest_usable_meal_plan
                    prof = get_user_profile(user_id) or {}
                    hp = prof.get("health_profile") or {}
                    target_cals = hp.get("target_calories")
                    target_prot = hp.get("target_protein")
                    if not target_cals:
                        plan = get_latest_usable_meal_plan(user_id)
                        if plan:
                            target_cals = plan.get("calories")
                            target_prot = plan.get("protein")
                except Exception as _te:
                    logger.debug(f"[diary_chat_bridge] Error leyendo targets: {_te}")

        if _is_sup and not target_cals:
            followup = ". Recuerda que esto no sustituye una comida principal."
        elif target_cals and int(target_cals) > total_cals:
            rem_cal = int(target_cals) - total_cals
            if target_prot and int(target_prot) > total_prot:
                rem_prot = int(target_prot) - total_prot
                followup = f". Te quedan **~{rem_cal} kcal** y **~{rem_prot} g de proteína** para tu meta de hoy."
            elif target_prot and total_prot >= int(target_prot):
                exceso_p = total_prot - int(target_prot)
                if exceso_p > 0:
                    followup = f". Te quedan **~{rem_cal} kcal** para tu meta de hoy; ya superaste la meta de proteína por unos **{exceso_p} g**."
                else:
                    followup = f". Te quedan **~{rem_cal} kcal** para tu meta de hoy; ya cubriste tu meta de proteína."
            else:
                followup = f". Te quedan **~{rem_cal} kcal** para tu meta de hoy."
        else:
            followup = f". Hoy llevas **~{total_cals} kcal** y **~{total_prot} g de proteína** en el diario."

    coach_text = f"{header}{followup}"
    return user_text, coach_text


def _try_update_langgraph_checkpoint(session_id: str, user_text: str, coach_text: str):
    """Actualiza de forma defensiva el checkpointer de LangGraph si la sesión ya existe en memoria/DB."""
    try:
        from agent import _checkpoint_pool, chat_builder
        from langchain_core.messages import HumanMessage, AIMessage
        from langgraph.checkpoint.postgres import PostgresSaver

        if _checkpoint_pool:
            checkpointer = PostgresSaver(_checkpoint_pool)
            graph = chat_builder.compile(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": str(session_id)}}
            state = graph.get_state(config)
            if state and state.values:
                graph.update_state(
                    config,
                    {"messages": [HumanMessage(content=user_text), AIMessage(content=coach_text)]},
                )
                logger.debug(f"✅ [diary_chat_bridge] Checkpoint de LangGraph actualizado para {session_id}")
    except Exception as exc:
        logger.debug(f"[diary_chat_bridge] LangGraph update_state no requerido/omitido: {exc}")


def sync_meal_to_coach_chat(
    *,
    user_id: str,
    meal_id: str,
    meal_name: str,
    meal_type: Optional[str] = None,
    calories: int = 0,
    protein: int = 0,
    carbs: int = 0,
    healthy_fats: int = 0,
    ingredients: Optional[list] = None,
    days_ago: int = 0,
    source: str = "manual",
    session_id: Optional[str] = None,
    background_tasks: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Sincroniza el registro de comida manual al chat del Coach en agent_messages y LangGraph.
    No ejecuta si la comida ya provino del chat (source == 'chat').
    """
    if source == "chat":
        return None
    if not user_id or user_id == "guest":
        return None

    try:
        sid = get_or_resolve_today_session_for_user(user_id, session_id)
        user_text, coach_text = build_sync_turn_texts(
            meal_name=meal_name,
            meal_type=meal_type,
            calories=calories,
            protein=protein,
            days_ago=days_ago,
            user_id=user_id,
            meal_id=meal_id,
        )

        # 1. Guardar mensaje de usuario (process_nudge=False para evitar side-effects de sentiment analysis)
        save_message(sid, "user", user_text, user_id=user_id, process_nudge=False)

        # 2. Guardar confirmación del coach
        save_message(sid, "model", coach_text, user_id=user_id)

        # 3. Actualizar LangGraph checkpoint si el hilo ya existía
        _try_update_langgraph_checkpoint(sid, user_text, coach_text)

        # 4. Encolar summarize_and_prune si hay background_tasks disponible
        if background_tasks:
            try:
                from memory_manager import summarize_and_prune
                background_tasks.add_task(summarize_and_prune, sid)
            except Exception as _se:
                logger.debug(f"[diary_chat_bridge] summarize_and_prune omitido: {_se}")

        logger.info(
            f"🌉 [diary_chat_bridge] Comida '{meal_name}' ({calories} kcal) sincronizada exitosamente al chat {sid}"
        )
        return {
            "session_id": sid,
            "user_text": user_text,
            "coach_text": coach_text,
            "meal_id": meal_id,
        }
    except Exception as e:
        logger.error(f"❌ [diary_chat_bridge] Error sincronizando comida al chat: {e}", exc_info=True)
        return None
