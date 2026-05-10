from functools import lru_cache as _lru_cache
import json
import threading
import uuid
import unicodedata as _uc
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import logging
logger = logging.getLogger(__name__)
from db_core import supabase, connection_pool, execute_sql_query, execute_sql_write


# ============================================================
# [P1-4] Strict mode + observabilidad del fallback de pool
# ------------------------------------------------------------
# `update_user_health_profile_atomic` (P1-ORQ-1, P1-2) provee atomicidad real
# vía `SELECT ... FOR UPDATE` sobre el `connection_pool`. Si el pool no está
# inicializado (env mal configurado, deploy parcial, error temprano de
# psycopg al startup), el helper SOBREVIVE silenciosamente degradando al
# patrón legacy non-atómico — preservando funcionalidad pero perdiendo
# protección contra lost-update.
#
# Antes de P1-4, este fallback solo emitía un `logger.warning(...)` por cada
# llamada. En producción nadie miraba ese log hasta que el meta-learning
# empezaba a dar señales raras (history truncado, friction strikes
# desaparecidos, weight_history con gaps). El bug podía vivir días sin
# detección.
#
# Ahora:
#   1. Cada fallback incrementa `_POOL_FALLBACK_STATE` (snapshot expuesto vía
#      /api/system/atomic-pool-health) — operadores pueden alertar en >0.
#   2. `MEALFIT_REQUIRE_ATOMIC_POOL=1` (default 0) hace que la primera
#      llamada con pool ausente lance `RuntimeError` en lugar de degradar.
#      Útil en producción: un misconfig se detecta en la PRIMERA request,
#      no tras horas de lost-updates silenciosos.
#   3. El log estructurado (`[P1-4/POOL-FALLBACK] ...`) es agregable por
#      Loki/Grafana.
# ============================================================
_POOL_FALLBACK_STATE_LOCK = threading.Lock()
_POOL_FALLBACK_STATE: Dict[str, Any] = {
    "fallback_count": 0,
    "first_at": None,
    "last_at": None,
    "last_user_id": None,
}


def _env_bool(name: str, default: bool) -> bool:
    """Parser laxo de booleanos. Mismo contrato que `graph_orchestrator._env_bool`
    (1/true/yes/on case-insensitive). Default aplica a env var ausente o vacía.
    """
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Si está activo, `update_user_health_profile_atomic` lanza `RuntimeError` cuando
# `connection_pool is None` en lugar de degradar al fallback non-atómico.
# Usar SOLO en producción; en dev / scripts locales típicamente no hay pool y
# el fallback es la degradación correcta. Default `False` preserva el
# comportamiento histórico.
REQUIRE_ATOMIC_POOL = _env_bool("MEALFIT_REQUIRE_ATOMIC_POOL", False)


def _record_pool_fallback(user_id: str) -> None:
    """[P1-4] Registra una llamada degradada al path non-atómico. Thread-safe
    (puede invocarse desde múltiples workers/threads concurrentes).
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    with _POOL_FALLBACK_STATE_LOCK:
        _POOL_FALLBACK_STATE["fallback_count"] += 1
        if _POOL_FALLBACK_STATE["first_at"] is None:
            _POOL_FALLBACK_STATE["first_at"] = now_iso
        _POOL_FALLBACK_STATE["last_at"] = now_iso
        _POOL_FALLBACK_STATE["last_user_id"] = str(user_id) if user_id else None


def get_atomic_pool_fallback_snapshot() -> Dict[str, Any]:
    """[P1-4] Snapshot inmutable del contador de fallback. Consumido por el
    endpoint `/api/system/atomic-pool-health` y por el health-check al startup.
    """
    with _POOL_FALLBACK_STATE_LOCK:
        return {
            "fallback_count": _POOL_FALLBACK_STATE["fallback_count"],
            "first_at": _POOL_FALLBACK_STATE["first_at"],
            "last_at": _POOL_FALLBACK_STATE["last_at"],
            "last_user_id": _POOL_FALLBACK_STATE["last_user_id"],
            "pool_available": connection_pool is not None,
            "strict_mode": REQUIRE_ATOMIC_POOL,
        }

def upsert_user_profile(user_id: str, health_profile: dict) -> bool:
    """Hace upsert del perfil de usuario y health_profile en user_profiles."""
    if not supabase: return False
    try:
        supabase.table("user_profiles").upsert({
            "id": user_id,
            "health_profile": health_profile
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Error en upsert_user_profile: {e}")
        return False

def get_user_profile(user_id: str):
    """Obtiene el perfil completo del usuario, incluyendo el health_profile."""
    if not supabase: return None
    try:
        from datetime import datetime, timezone
        res = supabase.table("user_profiles").select("*").eq("id", user_id).execute()
        if not res.data:
            return None
            
        profile = res.data[0]
        
        # --- Graceful Degradation Middleware ---
        if profile.get("subscription_status") == "CANCELLED" and profile.get("subscription_end_date"):
            end_date_str = profile.get("subscription_end_date")
            try:
                from constants import safe_fromisoformat
                end_date = safe_fromisoformat(end_date_str)
                if end_date.tzinfo is None:
                    end_date = end_date.replace(tzinfo=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                
                # Si ya cruzamos la hora exacta de terminación, lo degradamos
                if now_utc > end_date:
                    logger.info(f"⬇️ Degradando perfil de {user_id} a 'gratis'. El tiempo de su cancelación (Graceful) terminó.")
                    supabase.table("user_profiles").update({
                        "plan_tier": "gratis",
                        "subscription_status": "INACTIVE"
                    }).eq("id", user_id).execute()
                    
                    profile["plan_tier"] = "gratis"
                    profile["subscription_status"] = "INACTIVE"
            except Exception as d_e:
                logger.error(f"Error parseando fechas en graceful degradation para {user_id}: {d_e}")
        # ---------------------------------------
        
        return profile
    except Exception as e:
        logger.error(f"Error obteniendo perfil: {e}")
        return None

def _invalidate_stale_chunks(user_id: str, reason: str):
    """Marca chunks pendientes como 'stale' para que el worker los re-genere con datos frescos."""
    from db_core import execute_sql_write
    result = execute_sql_write("""
        UPDATE plan_chunk_queue 
        SET status = 'stale', 
            updated_at = NOW()
        WHERE user_id = %s 
        AND status = 'pending'
        RETURNING id, week_number
    """, (user_id,), returning=True)
    
    if result:
        logger.info(f"♻️ [CHUNK INVALIDATION] {len(result)} chunks marcados como 'stale' para {user_id} (razón: {reason})")
        return True
    return False

# ============================================================
# [P1-ORQ-1] Read-Modify-Write atómico de health_profile
# ------------------------------------------------------------
# El patrón previo en `graph_orchestrator.py` para appendear a listas rolling
# del health_profile (`pipeline_score_history`, `reflection_history`,
# `rejection_patterns`) era:
#
#   profile = get_user_profile(user_id)        # SELECT
#   hp = profile.get("health_profile") or {}
#   hp["some_list"].append(new_entry)          # mutación in-memory
#   update_user_health_profile(user_id, hp)    # UPDATE absoluto
#
# Bajo concurrencia del mismo `user_id` (ej. 2 tabs abiertas regenerando, cron
# `_refill_emergency_backup_plan` mientras el usuario también dispara una
# generación), dos writers leen el mismo snapshot, cada uno appendea su
# entrada localmente, y el último UPDATE pisa al primero — pierden silenciosamente
# 1 entry por par concurrente. NO corrompe el JSON, NO rompe entrega del plan,
# pero degrada el meta-learning (`preflight_optimization_node` toma decisiones
# con history truncado → false negatives en detección de drift).
#
# Solución: serializar el read-modify-write con `SELECT … FOR UPDATE` por fila.
# Mismo patrón que `db_plans.update_plan_data_atomic` (P0-2). Sirve cross-process
# (multi-worker Uvicorn) y cross-thread (mismo proceso). El lock se libera al
# COMMIT de la transacción — ~5-10ms para una mutación típica de history list.
# ============================================================
def _resolve_mutator_result(result, hp: dict, *, user_id: str, path_label: str) -> dict:
    """[P3-4 · 2026-05-08] SSOT del contrato `mutator(hp) -> dict | None | False`.

    Interpretación:
      - `dict`: el caller pasó un nuevo state explícito → se persiste tal cual.
      - `None`: el caller mutó `hp` in-place → se persiste `hp`.
      - cualquier otra cosa (str, int, list, True, etc.): probablemente bug del
        caller. Logueamos WARNING claro y caemos al comportamiento "in-place"
        (persistimos `hp`) para no romper el flujo. La invariante es:
        un caller bien-formado nunca dispara este WARNING.

    Note: `False` se filtra ANTES de llegar a este helper (caller decidió abort).
    """
    if isinstance(result, dict):
        return result
    if result is None:
        return hp
    logger.warning(
        f"[P3-4/MUTATOR-CONTRACT] update_user_health_profile_atomic({user_id}, "
        f"path={path_label}): mutator retornó tipo inesperado "
        f"`{type(result).__name__}` (valor truncado: {repr(result)[:100]}). "
        f"Contrato: dict → reemplaza, None → in-place, False → abort. "
        f"Cualquier otro tipo se trata como in-place (preserva `hp` mutado). "
        f"Investigar al caller — probable bug del mutator."
    )
    return hp


def update_user_health_profile_atomic(user_id: str, mutator):
    """[P1-ORQ-1] Read-Modify-Write atómico del health_profile.

    Args:
        user_id: UUID del usuario.
        mutator: callable(hp: dict) -> dict | None | False. Recibe el dict
            actual del health_profile (mutable). Si retorna un dict, lo persiste;
            si retorna None, persiste el dict mutado in-place; si retorna `False`,
            aborta el UPDATE (caller decidió que no había cambios).

    Returns:
        El nuevo health_profile dict tras la mutación, o None si el usuario
        no existe / `connection_pool` no disponible y fallback degradado falla.

    Concurrencia:
        Dos invocaciones simultáneas con el mismo user_id se serializan vía
        `SELECT … FOR UPDATE`. La segunda espera a que la primera commitee
        antes de leer, garantizando que su mutator vea el estado post-primera.
        No hay lost-update.

    Side effects post-commit (preservados de `update_user_health_profile`):
      - Invalidación de chunks pendientes si cambiaron campos críticos
        (goal/budget/allergies/peso ±5kg).
      - Sync inmediato de TZ a `plan_chunk_queue` si cambió `tz_offset_minutes`.
      Ambos corren FUERA de la transacción para no entrelazar locks con
      `_invalidate_stale_chunks` / `_sync_chunk_queue_tz_offsets` que abren sus
      propias conexiones.

    Fallback (degradación graceful — controlado por `MEALFIT_REQUIRE_ATOMIC_POOL`):
        Si `connection_pool` no está disponible (entornos dev sin psycopg
        pool, Supabase REST puro), por default degrada al patrón legacy
        get+update (no atómico): preserva funcionalidad en dev/scripts pero
        deja el bug de lost-update latente.

        [P1-4] En producción, exportar `MEALFIT_REQUIRE_ATOMIC_POOL=1` hace
        que el helper lance `RuntimeError` en la primera llamada con pool
        ausente — fail-fast que evita corrupción silenciosa por misconfig.
        Cada fallback (en modo non-strict) incrementa el counter de
        `get_atomic_pool_fallback_snapshot()` para alerting (Grafana,
        `/api/system/atomic-pool-health`).
    """
    if not connection_pool:
        # [P1-4] Fail-fast en producción: si el operador opt-ó in al
        # strict mode, una request con pool ausente debe romper EN VOZ ALTA
        # antes de que silently corra non-atomic durante horas.
        if REQUIRE_ATOMIC_POOL:
            _record_pool_fallback(user_id)  # contar también para visibility
            raise RuntimeError(
                f"[P1-4/STRICT] connection_pool no disponible y "
                f"MEALFIT_REQUIRE_ATOMIC_POOL=1 — abortando "
                f"update_user_health_profile_atomic({user_id}) en lugar de "
                f"degradar a non-atómico. Verificar inicialización del pool "
                f"en db_core (variables de entorno DATABASE_URL / "
                f"SUPABASE_DB_URL, conectividad a Supabase Pooler:6543)."
            )

        # Modo non-strict (default, dev): registramos + log estructurado +
        # degradamos. Mantiene el contrato pre-P1-4 para no romper scripts y
        # tests locales sin pool.
        _record_pool_fallback(user_id)
        _snapshot = get_atomic_pool_fallback_snapshot()
        logger.warning(
            f"[P1-4/POOL-FALLBACK] connection_pool=None "
            f"user_id={user_id} fallback_count={_snapshot['fallback_count']} "
            f"first_at={_snapshot['first_at']}. Degradando a non-atómico — "
            f"lost-update posible bajo concurrencia. Para fail-fast en prod, "
            f"export MEALFIT_REQUIRE_ATOMIC_POOL=1."
        )
        profile = get_user_profile(user_id)
        if not profile:
            return None
        hp = profile.get("health_profile") or {}
        if not isinstance(hp, dict):
            hp = {}
        result = mutator(hp)
        if result is False:
            return hp
        # [P3-4] SSOT del contrato; loguea WARNING si tipo inesperado.
        new_hp = _resolve_mutator_result(result, hp, user_id=user_id, path_label="fallback")
        update_user_health_profile(user_id, new_hp)
        return new_hp

    from copy import deepcopy
    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    new_hp = None
    invalidation_reasons = []
    tz_changed = False

    with connection_pool.connection() as conn:
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cursor:
                cursor.execute(
                    "SELECT health_profile FROM user_profiles WHERE id = %s FOR UPDATE",
                    (user_id,),
                )
                row = cursor.fetchone()
                if not row:
                    logger.warning(
                        f"[P1-ORQ-1] update_user_health_profile_atomic: "
                        f"user_profiles.{user_id} no existe. Skip."
                    )
                    return None

                old_hp = row["health_profile"] or {}
                if not isinstance(old_hp, dict):
                    old_hp = {}
                snapshot_old = deepcopy(old_hp)

                result = mutator(old_hp)
                if result is False:
                    return old_hp
                # [P3-4] SSOT del contrato; loguea WARNING si tipo inesperado.
                new_hp = _resolve_mutator_result(result, old_hp, user_id=user_id, path_label="atomic")

                # Detectar invalidaciones críticas — mismo conjunto de checks que
                # `update_user_health_profile` (mantener alineado si se añade uno).
                if snapshot_old.get("goal") != new_hp.get("goal"):
                    invalidation_reasons.append("goal_changed")
                if snapshot_old.get("budget") != new_hp.get("budget"):
                    invalidation_reasons.append("budget_changed")
                if set(snapshot_old.get("allergies", []) or []) != set(new_hp.get("allergies", []) or []):
                    invalidation_reasons.append("allergies_changed")
                try:
                    old_w = float(snapshot_old.get("weight", 0) or 0)
                    new_w = float(new_hp.get("weight", 0) or 0)
                    if old_w and new_w and abs(old_w - new_w) >= 5:
                        invalidation_reasons.append("significant_weight_change")
                except (TypeError, ValueError):
                    pass

                _old_tz = snapshot_old.get("tz_offset_minutes")
                if _old_tz is None:
                    _old_tz = snapshot_old.get("tzOffset")
                _new_tz = new_hp.get("tz_offset_minutes")
                if _new_tz is None:
                    _new_tz = new_hp.get("tzOffset")
                try:
                    if _old_tz is not None and _new_tz is not None and int(_old_tz) != int(_new_tz):
                        tz_changed = True
                except (TypeError, ValueError):
                    pass

                cursor.execute(
                    "UPDATE user_profiles SET health_profile = %s::jsonb WHERE id = %s",
                    (Jsonb(new_hp), user_id),
                )
        # Transacción comiteada al salir del `with conn.transaction()`.

    # Side effects post-commit. Usan conexiones independientes; ejecutarlos
    # dentro de la transacción podría entrelazar locks con _invalidate_stale_chunks
    # (que toca plan_chunk_queue) o _sync_chunk_queue_tz_offsets.
    if invalidation_reasons:
        try:
            _invalidate_stale_chunks(user_id, ", ".join(invalidation_reasons))
        except Exception as inv_err:
            logger.warning(
                f"[P1-ORQ-1] _invalidate_stale_chunks falló post-commit para "
                f"{user_id}: {inv_err}"
            )
    if tz_changed:
        try:
            from cron_tasks import _sync_chunk_queue_tz_offsets
            _sync_chunk_queue_tz_offsets(target_user_id=user_id)
        except Exception as sync_e:
            logger.warning(
                f"[P1-ORQ-1] Sync TZ falló post-commit para {user_id}: {sync_e}"
            )

    return new_hp


def update_user_health_profile(user_id: str, health_profile: dict):
    """Sobreescribe el JSONB de health_profile en la base de datos."""
    if not supabase: return None
    try:
        # --- GAP 4: Chunk Invalidation Detector (Conservative) ---
        # [P0-5] También detectamos cambio de tz_offset_minutes para sincronizar chunks
        # encolados con el nuevo offset. Sin este hook, un usuario que viajaba y
        # actualizaba su perfil seguía con chunks que disparaban en la TZ vieja.
        _tz_changed = False
        try:
            old_profile_data = get_user_profile(user_id)
            if old_profile_data and old_profile_data.get('health_profile'):
                old_hp = old_profile_data['health_profile']

                invalidation_reasons = []
                if old_hp.get('goal') != health_profile.get('goal'):
                    invalidation_reasons.append("goal_changed")
                if old_hp.get('budget') != health_profile.get('budget'):
                    invalidation_reasons.append("budget_changed")
                if set(old_hp.get('allergies', [])) != set(health_profile.get('allergies', [])):
                    invalidation_reasons.append("allergies_changed")

                old_w = float(old_hp.get('weight', 0) or 0)
                new_w = float(health_profile.get('weight', 0) or 0)
                if old_w and new_w and abs(old_w - new_w) >= 5:
                    invalidation_reasons.append("significant_weight_change")

                # [P0-5] Detección de cambio de TZ. Acepta tanto `tz_offset_minutes` como
                # `tzOffset` (legacy). Si cambió y el delta supera el threshold, marcamos
                # para disparar el sync inmediato tras persistir.
                _old_tz = old_hp.get('tz_offset_minutes')
                if _old_tz is None:
                    _old_tz = old_hp.get('tzOffset')
                _new_tz = health_profile.get('tz_offset_minutes')
                if _new_tz is None:
                    _new_tz = health_profile.get('tzOffset')
                try:
                    if _old_tz is not None and _new_tz is not None and int(_old_tz) != int(_new_tz):
                        _tz_changed = True
                except (TypeError, ValueError):
                    pass

                if invalidation_reasons:
                    _invalidate_stale_chunks(user_id, ", ".join(invalidation_reasons))
        except Exception as check_e:
            logger.warning(f"⚠️ [CHUNK INVALIDATION] Error checking for critical profile changes: {check_e}")
        # ---------------------------------------------------------

        res = supabase.table("user_profiles").update({
            "health_profile": health_profile
        }).eq("id", user_id).execute()

        # [P0-5] Tras persistir el nuevo perfil, propagar el cambio de TZ a chunks
        # pending/stale del usuario. Lazy import para evitar ciclo (cron_tasks → db).
        # Si el sync falla, no abortamos el update — el cron horario lo arrastrará en
        # la siguiente pasada.
        if _tz_changed:
            try:
                from cron_tasks import _sync_chunk_queue_tz_offsets
                _sync_chunk_queue_tz_offsets(target_user_id=user_id)
            except Exception as sync_e:
                logger.warning(f"⚠️ [P0-5] Sync inmediato de TZ falló para {user_id}: {sync_e}")

        return res.data
    except Exception as e:
        logger.error(f"Error actualizando health_profile: {e}")
        return None

def log_api_usage(user_id: str, endpoint: str = "gemini"):
    """Guarda un registro de uso de la API (consume 1 crédito)."""
    if not user_id or user_id == "guest": return None
    try:
        from db_core import connection_pool
        if connection_pool:
            res = execute_sql_write("INSERT INTO api_usage (user_id, endpoint) VALUES (%s, %s)", (user_id, endpoint))
            return res
        else:
            if not supabase: return None
            res = supabase.table("api_usage").insert({
                "user_id": user_id,
                "endpoint": endpoint
            }).execute()
            return res.data
    except Exception as e:
        logger.error(f"Error registrando api_usage: {e}")
        return None

def get_monthly_api_usage(user_id: str) -> int:
    """Cuenta cuántas llamadas a la API ha hecho el usuario este mes."""
    if not user_id or user_id == "guest": return 0
    from datetime import datetime
    
    try:
        now = datetime.now()
        start_date = datetime(now.year, now.month, 1).isoformat()
        
        from db_core import connection_pool
        if connection_pool:
            res = execute_sql_query("SELECT count(*) as total FROM api_usage WHERE user_id = %s AND created_at >= %s", (user_id, start_date), fetch_one=True)
            if res and 'total' in res:
                return int(res['total'])
            return 0
        else:
            if not supabase: return 0
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    res = supabase.table("api_usage").select("*", count="exact").eq("user_id", user_id).gte("created_at", start_date).execute()
                    return res.count if hasattr(res, 'count') and res.count is not None else 0
                except Exception as inner_e:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(0.5)
                        continue
                    logger.error(f"Error supabase get_monthly_api_usage: {inner_e}")
                    return 0
    except Exception as e:
        logger.error(f"Error obteniendo api_usage mensual: {e}")
        return 0

def migrate_guest_data(session_ids: list, new_user_id: str):
    """
    Migra todos los datos asociados a uno o varios session_ids temporales
    (creados cuando el usuario era invitado) hacia el nuevo UUID del usuario registrado.
    """
    if not supabase or not session_ids or not new_user_id: 
        return False
    
    try:
        # 1. Actualizar agent_sessions para vincular historiales de chat
        supabase.table("agent_sessions").update({"user_id": new_user_id}).in_("id", session_ids).execute()
        
        # 2. Actualizar visual_diary (Vectores/Diario Visual)
        supabase.table("visual_diary").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 3. Actualizar user_facts (Vectores/Memoria a largo plazo)
        supabase.table("user_facts").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 4. Actualizar meal_plans (Planes guardados como guest, si los hubiera)
        supabase.table("meal_plans").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 5. Actualizar consumed_meals (Comidas registradas)
        supabase.table("consumed_meals").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 6. Actualizar pending_facts_queue
        supabase.table("pending_facts_queue").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 7. Actualizar meal_rejections (Ojo: estas usan session_id o user_id)
        # Primero intentamos por session_id (como se guardan usualmente los rechazos guest)
        supabase.table("meal_rejections").update({"user_id": new_user_id}).in_("session_id", session_ids).execute()
        # Y también por user_id por si acaso
        supabase.table("meal_rejections").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 8. Actualizar meal_likes
        supabase.table("meal_likes").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        
        # 9. Actualizar Inventario Físico (Despensa)
        supabase.table("user_inventory").update({"user_id": new_user_id}).in_("user_id", session_ids).execute()
        # 10. Recalcular frecuencias de ingredientes a partir de los planes migrados
        # Sin esto, el usuario registrado parte con freq=0 y pierde el historial de variedad.
        try:
            from constants import normalize_ingredient_for_tracking
            from db_plans import get_ingredient_frequencies_from_plans, increment_ingredient_frequencies
            migrated_ings = get_ingredient_frequencies_from_plans(new_user_id, limit=10)
            if migrated_ings:
                normalized = [normalize_ingredient_for_tracking(i) for i in migrated_ings if i]
                normalized = [n for n in normalized if n]  # Filtrar vacíos
                if normalized:
                    increment_ingredient_frequencies(new_user_id, normalized)
                    logger.info(f"✅ [MIGRACIÓN] Frecuencias recalculadas para {new_user_id} ({len(normalized)} ingredientes)")
        except Exception as freq_e:
            # No bloquear la migración si falla el recálculo de frecuencias
            logger.error(f"⚠️ [MIGRACIÓN] Error recalculando frecuencias (no crítico): {freq_e}")
        
        logger.info(f"✅ Migración exitosa de {session_ids} a UUID {new_user_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error migrando datos de invitado: {e}")
        return False

def reset_user_account_preferences(user_id: str) -> bool:
    """Borra preferencias, rechazos, inventario, planes históricos y limpia el health_profile para un verdadero inicio desde cero."""
    if not supabase or not user_id or user_id == "guest": 
        return False
        
    try:
        from db_core import execute_sql_write
        # 1. Borrar likes
        execute_sql_write("DELETE FROM meal_likes WHERE user_id = %s", (user_id,))
        # 2. Borrar rejections
        execute_sql_write("DELETE FROM meal_rejections WHERE user_id = %s", (user_id,))
        # 3. Borrar inventario
        execute_sql_write("DELETE FROM user_inventory WHERE user_id = %s", (user_id,))
        # 4. Borrar knowledge graph/facts (preferencias aprendidas)
        execute_sql_write("DELETE FROM user_facts WHERE user_id = %s", (user_id,))
        # 5. Borrar frecuencias de ingredientes (para que no rote basado en el pasado)
        execute_sql_write("DELETE FROM ingredient_frequencies WHERE user_id = %s", (user_id,))
        # 6. Borrar planes históricos (esto borra en cascada la plan_chunk_queue)
        execute_sql_write("DELETE FROM meal_plans WHERE user_id = %s", (user_id,))
        # 7. Limpiar health_profile en user_profiles (dejándolo vacío)
        execute_sql_write("UPDATE user_profiles SET health_profile = '{}'::jsonb WHERE id = %s", (user_id,))
        
        logger.info(f"♻️ Preferencias y planes reseteados DESDE CERO con éxito para UUID {user_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error reseteando preferencias para {user_id}: {e}")
        return False
