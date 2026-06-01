import logging
from typing import Optional
from datetime import datetime
import json
import unicodedata
import re
import hashlib

# Imports globales locales, movidos al tope para evitar el code smell "Lazy Loading"
from db import (
    supabase,
    increment_ingredient_frequencies,
    check_recent_meal_plan_exists,
    log_unknown_ingredients,
    get_user_profile,
    get_or_create_session,
    save_message,
    insert_rejection,
    track_meal_friction,
    update_user_health_profile,
    update_user_health_profile_atomic,
    upsert_user_profile
)
# [P2-PARTIAL-PLAN-1 · 2026-05-11] Removido `save_new_meal_plan_robust` del
# top-level: el único usuario era el branch del CANCEL out-of-tx, ya
# eliminado de `_save_plan_and_track_background`. `save_partial_plan_get_id`
# (línea ~110) sigue importándolo lazy inside la función — el helper sigue
# vivo en db_plans.py y otros callers (tools.py, app.py) lo usan sin race
# porque pasan `additional_queries=None`.
from db_plans import save_new_meal_plan_atomic

from constants import normalize_ingredient_for_tracking, GLOBAL_REVERSE_MAP, IGNORED_TRACKING_TERMS
from db_inventory import release_meal_reservation

# ⚠️ RESTRICCIÓN ARQUITECTÓNICA: services.py importa agent.py → agent.py NUNCA debe importar services.py.
# Si agent.py necesita lógica de services.py en el futuro, usar lazy import dentro de la función.
from ai_helpers import generate_plan_title

logger = logging.getLogger(__name__)


def compute_plan_hash(plan_data: dict) -> str:
    """Calcula un hash SHA-256 truncado del plan basado en ingredientes y suplementos.
    Fuente única de verdad para detectar si un plan cambió."""
    all_ingredients = []
    all_supplements = []
    for d in plan_data.get("days", []):
        for m in d.get("meals", []):
            ing = m.get("ingredients", [])
            if ing:
                all_ingredients.extend(ing)
        for s in d.get("supplements") or []:
            all_supplements.append(s.get("name", ""))
    return hashlib.sha256(
        json.dumps(
            {"ingredients": all_ingredients, "supplements": sorted(set(all_supplements)), "version": "v7_deterministic_math"},
            sort_keys=True, ensure_ascii=False
        ).encode()
    ).hexdigest()[:16]





def merge_form_data_with_profile(user_id: str, form_data: Optional[dict]) -> dict:
    """
    Merges frontend form_data with the stored health_profile from DB.
    Extracted to avoid DRY violation between /api/chat/stream and /api/chat.
    Returns the merged form_data dict.
    """
    merged = form_data or {}
    if not user_id or user_id == "guest" or user_id == "":
        return merged
    try:
        profile = get_user_profile(user_id)
        if profile:
            existing_hp = profile.get("health_profile") or {}
            if existing_hp:
                non_empty_form = {k: v for k, v in merged.items() if v not in [None, "", [], {}]}
                existing_hp.update(non_empty_form)
                merged = existing_hp
            if form_data and not existing_hp:
                logger.debug(f"🔄 [SYNC] health_profile vacío, sincronizando desde formData del frontend...")
                # [P1-2] Migración a atomic helper. El check `not existing_hp`
                # se hizo PRE-LOCK; entre ese check y este write, otro caller
                # del mismo user_id puede haber poblado hp (race con cron de
                # facts/learning, otro tab del usuario en proceso de
                # registro). Sin atomic, sobrescribiríamos la población
                # concurrente. El mutator hace `hp.update(form_only)` ON TOP
                # del estado fresco bajo FOR UPDATE: si está vacío, equivale a
                # un set; si no, mergea form fields preservando los demás.
                _form_only = {k: v for k, v in (form_data or {}).items() if v not in [None, "", [], {}]}

                def _init_mutator(_hp):
                    if not _form_only:
                        return False  # nada que persistir
                    _hp.update(_form_only)
                    return None

                update_user_health_profile_atomic(user_id, _init_mutator)
        else:
            if form_data:
                logger.warning(f"⚠️ [SYNC] No existe user_profile para {user_id}, intentando crear...")
                try:
                    upsert_user_profile(user_id, merged)
                    logger.info(f"✅ [SYNC] Perfil creado con health_profile")
                except Exception as e:
                    logger.error(f"❌ [SYNC] Error creando perfil: {e}")
    except Exception as e:
        logger.error(f"⚠️ Error cargando health profile en chat: {e}")
    return merged



# [P3-GENCHUNK-SPEED · 2026-06-01] Título determinista (cero LLM) usado como
# placeholder síncrono cuando diferimos la generación del título creativo. Mirror
# del fallback de `ai_helpers.generate_plan_title` (líneas ~147-149) para que el
# nombre placeholder sea razonable si el título creativo nunca llega (worker kill).
def _deterministic_plan_title_placeholder(plan_data: dict) -> str:
    meal_names = [
        m["name"]
        for d in plan_data.get("days", [])
        for m in d.get("meals", [])
        if m.get("name")
    ]
    calories = plan_data.get("calories", 0)
    if not meal_names:
        return f"Plan Evolutivo - {datetime.now().strftime('%d/%m/%Y')}"
    first_meal = meal_names[0]
    short_name = first_meal[:20] + "…" if len(first_meal) > 20 else first_meal
    return f"{short_name} — {calories} kcal"


def _defer_creative_plan_title(plan_id: str, user_id: str, plan_data: dict, placeholder: str) -> None:
    """[P3-GENCHUNK-SPEED · 2026-06-01] Genera el título creativo (LLM Flash-Lite)
    en un thread daemon y lo escribe via UPDATE de UNA columna escalar (`name`),
    fuera del critical path de time-to-plan-visible. Guard `AND name = <placeholder>`
    para NO pisar un rename del usuario (PATCH /name) que haya ocurrido en la
    ventana de ~1-2s. Guard `AND user_id` defense-in-depth (I2). UPDATE escalar →
    exento de advisory lock (I7) y no puede lost-update plan_data del chunk worker.
    Best-effort: si el worker muere antes de completar, el plan queda con el
    placeholder (un nombre válido) — degradación cosmética aceptable."""
    def _bg():
        try:
            creative = generate_plan_title(plan_data)
            if creative and creative != placeholder:
                from db_core import execute_sql_write
                execute_sql_write(
                    "UPDATE meal_plans SET name = %s WHERE id = %s AND user_id = %s AND name = %s",
                    (creative, plan_id, user_id, placeholder),
                )
                logger.info(f"✨ [CHUNK/DEFER-TITLE] Título creativo aplicado a plan {plan_id}")
        except Exception as _e_title:
            logger.warning(f"⚠️ [CHUNK/DEFER-TITLE] Título diferido falló (placeholder se mantiene): {_e_title}")
    import threading
    threading.Thread(target=_bg, name="defer-plan-title", daemon=True).start()


def save_partial_plan_get_id(user_id: str, plan_data: dict, selected_techniques: list = None, total_days_requested: int = 7) -> str:
    """Guarda la Semana 1 de un plan chunked de forma sincrónica y retorna el plan_id UUID.
    Usado exclusivamente por el flujo de Background Chunking para encolar las semanas restantes.

    [P2-PARTIAL-PLAN-1 · 2026-05-11] Removido el lazy import de
    `save_new_meal_plan_robust` — el body real usa `save_new_meal_plan_atomic`
    (línea ~158). El import legacy nunca se referenciaba.
    """
    try:
        # [GAP 3] Limpieza de días huérfanos al regenerar
        if "days" in plan_data and len(plan_data["days"]) > total_days_requested:
            import logging
            logger.warning(f"🧹 [GAP 3] Recortando días huérfanos en partial plan. De {len(plan_data['days'])} a {total_days_requested}")
            plan_data["days"] = plan_data["days"][:total_days_requested]

        calories = plan_data.get("calories", 0)
        macros = plan_data.get("macros", {})

        meal_names = []
        ingredients = []
        for d in plan_data.get("days", []):
            for m in d.get("meals", []):
                if m.get("name"):
                    meal_names.append(m["name"])
                if m.get("ingredients"):
                    ingredients.extend(m["ingredients"])

        # [P3-GENCHUNK-SPEED · 2026-06-01] El título creativo es una llamada LLM
        # bloqueante (Gemini Flash-Lite, hasta 30s en el peor caso) que solo llena
        # la columna cosmética `name` (lista del Historial) — NO es parte del menú
        # de semana-1 que el usuario ve al completar. Este path (chunking) es
        # síncrono en el critical path de time-to-plan-visible. Con el knob ON
        # (default), usamos un placeholder determinista AHORA, persistimos +
        # retornamos el plan_id de inmediato (para encolar semanas 2..N), y
        # generamos el título creativo en background con un UPDATE escalar guardado.
        # Knob MEALFIT_DEFER_PLAN_TITLE=0 revierte al comportamiento síncrono.
        from knobs import _env_bool
        _defer_title = _env_bool("MEALFIT_DEFER_PLAN_TITLE", True)
        if _defer_title:
            plan_name = _deterministic_plan_title_placeholder(plan_data)
        else:
            plan_name = generate_plan_title(plan_data)
        profile_embedding = plan_data.pop("_profile_embedding", None)

        insert_data = {
            "user_id": user_id,
            "plan_data": {**plan_data, "generation_status": "partial", "total_days_requested": total_days_requested},
            "name": plan_name,
            "calories": int(calories) if calories else 0,
            "macros": macros,
            "meal_names": meal_names,
            "ingredients": ingredients,
        }
        if profile_embedding:
            insert_data["profile_embedding"] = profile_embedding
        if selected_techniques:
            insert_data["techniques"] = selected_techniques

        # [P0-2/ATOMIC] INSERT del plan + cancelación de chunks en una sola transacción.
        # Elimina la ventana TOCTOU entre guardar el plan y cancelar los chunks viejos.
        plan_id = save_new_meal_plan_atomic(user_id, insert_data, return_id=True)

        # [P3-GENCHUNK-SPEED · 2026-06-01] Disparar el título creativo en background
        # SOLO tras tener el plan_id (para el UPDATE guardado) y solo si diferimos.
        if _defer_title and plan_id:
            _defer_creative_plan_title(plan_id, user_id, plan_data, plan_name)

        logger.info(f"💾 [CHUNK] Plan parcial (semana 1) guardado para {user_id}, plan_id={plan_id}")
        return plan_id
    except Exception as e:
        logger.error(f"❌ [CHUNK] Error guardando plan parcial para {user_id}: {e}")
        return None


def _persist_plan_persist_failed_alert(user_id: Optional[str], reason: str) -> None:
    """[P2-PLAN-PERSIST-FAILED-ALERT · 2026-05-30] Emite
    `system_alerts.plan_persist_failed:<user_id>` cuando la persistencia de un
    plan ya generado FALLA silenciosamente (INSERT de meal_plans falla por pool
    exhaustion, statement_timeout, la CHECK I8 `meal_plans_complete_requires_days`,
    serialization error, etc.).

    Por qué existe (audit prod-readiness 2026-05-30): el chunking path
    (`_postprocess_pipeline_result`) ignoraba un `save_partial_plan_get_id() ->
    None` → el SSE generator marcaba el KV `complete` con `plan_id_final=None` y
    emitía el evento `complete`: el usuario veía éxito pero el plan NO existía en
    `meal_plans` (historial/dashboard vacíos, weeks 2..N nunca encoladas). Sin
    alerta, el operador no tenía señal de esta falla del path crítico de
    persistencia. El fix del path ahora emite el evento `error` al cliente; esta
    alerta da la visibilidad operacional.

    Best-effort (no propaga). Idempotente por `alert_key` per-user (ON CONFLICT
    bumpea triggered_at). Modelo de resolution: Manual (operador investiga el
    incidente de persistencia). tooltip-anchor: P2-PLAN-PERSIST-FAILED-ALERT —
    row `plan_persist_failed:<>` en backend/docs/system_alerts_resolution_table.md.
    """
    try:
        from db_core import execute_sql_write
        import json as _json
        _uid = str(user_id) if user_id else "unknown"
        execute_sql_write(
            """
            INSERT INTO system_alerts
                (alert_key, alert_type, severity, title, message, metadata, affected_user_ids)
            VALUES (%s, 'plan_persist_failed', 'critical', %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (alert_key) DO UPDATE
            SET triggered_at = NOW(),
                metadata = EXCLUDED.metadata,
                affected_user_ids = EXCLUDED.affected_user_ids,
                resolved_at = NULL
            """,
            (
                f"plan_persist_failed:{_uid}",
                "Persistencia de plan fallida - plan generado NO guardado",
                (
                    "Un plan generado por el pipeline NO pudo persistirse en "
                    "meal_plans (INSERT fallido: pool exhaustion, statement_timeout, "
                    "CHECK constraint I8, serialization, etc). El usuario pudo ver un "
                    "'complete' sin plan real en su historial. Investigar logs del path "
                    f"de persistencia para el usuario afectado. Razon: {reason}."
                ),
                _json.dumps({"reason": reason}, ensure_ascii=False),
                _json.dumps([_uid] if user_id else [], ensure_ascii=False),
            ),
        )
        logger.error(
            f"🛑 [P2-PLAN-PERSIST-FAILED-ALERT] system_alert plan_persist_failed "
            f"emitido user={_uid[:8]} reason={reason}"
        )
    except Exception as _e:
        logger.warning(
            f"[P2-PLAN-PERSIST-FAILED-ALERT] No se pudo persistir el alert "
            f"plan_persist_failed: {_e!r}"
        )


def _save_plan_and_track_background(user_id: str, plan_data: dict, selected_techniques: list = None):
    """Background task para guardar plan y actualizar frecuencias de ingredientes.

    [P2-PARTIAL-PLAN-1 · 2026-05-11] El parámetro `additional_db_queries`
    fue REMOVIDO. Ningún caller en producción lo usaba (verificado vía grep
    cross-codebase: el único caller en `routers/plans.py:1391` solo pasa
    `actual_user_id, result, selected_techniques`). El branch que lo
    consumía hacía `execute_sql_write(...)` para cancelar chunks FUERA de
    la transacción del INSERT — admitido residual TOCTOU: si el chunk
    worker capturaba lock entre el CANCEL y el INSERT, podía persistir
    contra el plan viejo en lugar del nuevo.

    Cierre: consolidar todo a `save_new_meal_plan_atomic` (P0-2/ATOMIC),
    que cancela chunks + libera reservas + INSERT en una SOLA transacción.
    Si en el futuro algún caller necesita queries adicionales atómicas con
    el INSERT, extender `save_new_meal_plan_atomic` con un parámetro
    `additional_queries` que ejecute dentro del mismo `conn.transaction()`,
    NO restaurar el patrón pre-fix de CANCEL out-of-tx.
    """
    try:
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

            # Extraer _profile_embedding si fue inyectado por la caché semántica
            profile_embedding = plan_data.pop("_profile_embedding", None)

            insert_data = {
                "user_id": user_id,
                # [P2-ORCH-10 · 2026-05-28] Estampar generation_status='complete'
                # (simétrico con el path de chunking que estampa 'partial' ~línea 147).
                # Pre-fix el path no-chunking insertaba plan_data SIN el campo →
                # quedaba FUERA del CHECK I8 (meal_plans_complete_requires_days, que
                # exime NULL): un plan full-week con days=[] se mostraba como
                # 'complete' con 0 días vía el default de lectura. Ahora el write
                # entra bajo la invariante I8 (complete ⇒ days>0): si alguna
                # regresión persistiera days vacío, el CHECK lo rechaza en vez de
                # guardar corrupción silenciosa. Tooltip-anchor: P2-ORCH-10.
                "plan_data": {**plan_data, "generation_status": plan_data.get("generation_status", "complete")},
                "name": plan_name,
                "calories": int(calories) if calories else 0,
                "macros": macros,
                "meal_names": meal_names,
                "ingredients": ingredients
            }
            if profile_embedding:
                insert_data["profile_embedding"] = profile_embedding

            # Añadir técnicas de cocción si están disponibles
            if selected_techniques:
                insert_data["techniques"] = selected_techniques


            # 🛡️ Dedup guard: evitar duplicados si otro código path ya guardó el plan
            if check_recent_meal_plan_exists(user_id, max_seconds=30):
                logger.info(f"🛡️ [DEDUP] Plan ya guardado recientemente para {user_id}. Omitiendo duplicado.")
                return

            # [P0-2/ATOMIC + P2-PARTIAL-PLAN-1] Cancelar chunks + liberar
            # reservas + INSERT en una sola transacción. Único path —
            # eliminado el branch pre-P2-PARTIAL-PLAN-1 que hacía CANCEL
            # out-of-tx via execute_sql_write antes de
            # save_new_meal_plan_robust.
            save_new_meal_plan_atomic(user_id, insert_data, return_id=False)
            logger.debug(f"💾 [DB BACKGROUND] Plan guardado exitosamente en meal_plans para {user_id}")
            
        # 2. Track Frequencies (solo ingredientes canónicos que existan en los catálogos de variedad)
        if raw_ingredients:
            # Conjunto de términos base canónicos (ej: "pollo", "platano verde", "aguacate")
            canonical_bases = set(GLOBAL_REVERSE_MAP.values())
            
            normalized = [normalize_ingredient_for_tracking(ing) for ing in raw_ingredients]
            # Filtrar: solo trackear ingredientes que resolvieron a un término base conocido.
            # Esto evita que condimentos/hierbas (cilantro, orégano, ajo) polucionen la tabla.
            canonical = [n for n in normalized if n and n in canonical_bases]

            def _is_ignored(term: str) -> bool:
                """Ignora el término si es exacto o si alguna de sus palabras está en IGNORED_TRACKING_TERMS.
                Cubre compuestos como 'pimienta negra', 'canela en polvo', 'oregano dominicano'.
                """
                if term in IGNORED_TRACKING_TERMS:
                    return True
                return bool(set(term.split()) & IGNORED_TRACKING_TERMS)

            non_canonical = [n for n in normalized if n and n not in canonical_bases and not _is_ignored(n)]
            
            if canonical:
                increment_ingredient_frequencies(user_id, canonical)
            
            # 2b. Loguear ingredientes no reconocidos para revisión y expansión del catálogo
            if non_canonical:
                raw_map = {normalize_ingredient_for_tracking(r): r for r in raw_ingredients if r}
                log_unknown_ingredients(user_id, non_canonical, raw_map)
                logger.info(f"🧹 [FREQ TRACKING] {len(non_canonical)} ingredientes no-canónicos logueados para revisión")
                
            logger.info(f"📈 [FREQ TRACKING] Frecuencias actualizadas en background para {user_id} ({len(canonical)} ingredientes canónicos trackeados)")
            

            
    except Exception as e:
        logger.error(f"⚠️ [BACKGROUND ERROR] Error asíncrono guardando plan: {e}")
        # [P2-PLAN-PERSIST-FAILED-ALERT · 2026-05-30] El save corre como
        # BackgroundTask (post-respuesta): si falla, el usuario ya tiene el plan
        # en pantalla pero NO queda en su historial — falla silenciosa del path de
        # persistencia. Emitir alerta para visibilidad operacional.
        _persist_plan_persist_failed_alert(user_id, f"background_save_failed: {e}")


def _process_swap_rejection_background(session_id: str, user_id: str, rejected_meal: str, meal_type: str):
    """Background task: Loguea mensajes y rechazos que expiran en 7 días, asíncronamente."""
    try:
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
            if user_id and user_id != "guest":
                release_meal_reservation(user_id, rejected_meal)
            
            # Fricción Silenciosa: Validar si la base ya se rechazó 3 veces
            track_meal_friction(user_id, session_id, rejected_meal)
    except Exception as e:
        logger.error(f"⚠️ [BACKGROUND ERROR] Error procesando swap rejection: {e}")
