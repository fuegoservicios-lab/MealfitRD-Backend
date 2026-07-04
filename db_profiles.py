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
# [P1-NEON-DB-MIGRATION · 2026-06-12] `_storage_client` es el placeholder del
# object storage de visual_diary (`_purge_visual_diary_storage`) — pendiente
# de migrar a provider nuevo; todo acceso a datos va por SQL directo.
from db_core import _storage_client, connection_pool, execute_sql_query, execute_sql_write, execute_sql_transaction


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

def _normalize_profile_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """[P1-NEON-DB-MIGRATION · 2026-06-12] Paridad de tipos con PostgREST.

    psycopg devuelve tipos nativos (uuid.UUID, datetime/date, Decimal) donde
    el cliente PostgREST legado devolvía JSON (strings ISO, floats). Los consumers
    del perfil esperan la forma PostgREST — p.ej. `safe_fromisoformat(subscription_end_date)`
    hace slicing de string. Convertimos: uuid→str, datetime/date→ISO string,
    Decimal→float. jsonb (`health_profile`) ya es dict en ambos mundos.
    """
    from datetime import date as _date
    from decimal import Decimal as _Decimal
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, uuid.UUID):
            out[k] = str(v)
        elif isinstance(v, (datetime, _date)):
            out[k] = v.isoformat()
        elif isinstance(v, _Decimal):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def upsert_user_profile(user_id: str, health_profile: dict) -> bool:
    """Hace upsert del perfil de usuario y health_profile en user_profiles."""
    if not connection_pool: return False
    try:
        from psycopg.types.json import Jsonb
        execute_sql_write(
            """
            INSERT INTO user_profiles (id, health_profile)
            VALUES (%s, %s)
            ON CONFLICT (id) DO UPDATE SET health_profile = EXCLUDED.health_profile
            """,
            (user_id, Jsonb(health_profile)),
        )
        return True
    except Exception as e:
        logger.error(f"Error en upsert_user_profile: {e}")
        return False


# [P1-NEON-DB-MIGRATION · 2026-06-12] Cache in-process de user_ids cuyo
# profile row ya fue verificado/creado. Evita un INSERT ON CONFLICT por
# request (solo cuesta uno por usuario por proceso). Se resetea por deploy —
# el ON CONFLICT DO NOTHING hace el re-check idempotente y barato.
_PROFILE_ENSURED_IDS: set = set()
_PROFILE_ENSURED_MAX = 50_000  # backstop de memoria; reset total al superarlo


def ensure_user_profile_exists(user_id: str, email: Optional[str] = None,
                               full_name: Optional[str] = None) -> None:
    """[P1-NEON-DB-MIGRATION · 2026-06-12] Reemplazo app-side del trigger
    `handle_new_user` (vivía sobre `auth.users` en el proveedor anterior; en
    Neon el schema `auth` no existe). El proveedor de Auth sigue creando el
    usuario JWT, pero la fila espejo en `public.user_profiles` la garantiza
    el backend en el primer request autenticado (auth.py::get_verified_user_id).

    Mismo payload que el trigger original: (id, email, full_name, created_at).
    ON CONFLICT (id) DO NOTHING — jamás pisa un profile existente (a
    diferencia del upsert de health_profile). Best-effort: un fallo aquí NO
    debe tumbar la autenticación (el caller ya validó el JWT); se loguea y
    el siguiente request reintenta (el cache solo se llena tras éxito).
    """
    if not user_id or user_id in _PROFILE_ENSURED_IDS:
        return
    if not connection_pool:
        return
    try:
        execute_sql_write(
            """
            INSERT INTO user_profiles (id, email, full_name, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (id) DO NOTHING
            """,
            (user_id, email, full_name),
        )
        if len(_PROFILE_ENSURED_IDS) >= _PROFILE_ENSURED_MAX:
            _PROFILE_ENSURED_IDS.clear()
        _PROFILE_ENSURED_IDS.add(user_id)
    except Exception as e:
        logger.error(
            f"[P1-NEON-DB-MIGRATION] ensure_user_profile_exists falló para "
            f"{user_id}: {type(e).__name__}: {e} (se reintenta en el próximo request)"
        )


def get_user_profile(user_id: str):
    """Obtiene el perfil completo del usuario, incluyendo el health_profile."""
    if not connection_pool: return None
    try:
        from datetime import datetime, timezone
        row = execute_sql_query(
            "SELECT * FROM user_profiles WHERE id = %s LIMIT 1",
            (user_id,),
            fetch_one=True,
        )
        if not row:
            return None

        profile = _normalize_profile_row(row)

        # --- Graceful Degradation Middleware ---
        # [P1-PROD-AUDIT-3 · 2026-05-30] Revocador SSOT del tier pagado tras
        # cancelación. ANTES la condición exigía `subscription_end_date` truthy:
        # cuando PayPal cancela y OMITE `next_billing_time` (común en
        # cancelaciones), el webhook BILLING.SUBSCRIPTION.CANCELLED dejaba
        # `subscription_end_date = NULL` → esta rama NUNCA disparaba → acceso
        # pagado perpetuo sin cobro (el quota gate en auth.py keya solo en
        # plan_tier, que seguía basic/plus/ultra). Contraparte cancel-side del
        # P1-BILLING-REACTIVATE-NOT-CANCELLED. Fix: CANCELLED se evalúa SIEMPRE;
        # con end_date futura se respeta la gracia, sin end_date (o fecha
        # ilegible) se degrada YA — fail-secure: PayPal ya no cobra, no hay base
        # para conceder gracia indefinida.
        if profile.get("subscription_status") == "CANCELLED":
            end_date_str = profile.get("subscription_end_date")
            should_downgrade = False
            if end_date_str:
                try:
                    from constants import safe_fromisoformat
                    end_date = safe_fromisoformat(end_date_str)
                    if end_date.tzinfo is None:
                        end_date = end_date.replace(tzinfo=timezone.utc)
                    now_utc = datetime.now(timezone.utc)
                    # Si ya cruzamos la hora exacta de terminación, lo degradamos
                    if now_utc > end_date:
                        should_downgrade = True
                except Exception as d_e:
                    # Fecha ilegible → fail-secure: degradar (no perpetuar acceso
                    # pagado por data corrupta).
                    logger.error(f"Error parseando fechas en graceful degradation para {user_id}: {d_e}")
                    should_downgrade = True
            else:
                # CANCELLED sin subscription_end_date: degradar ya (ver nota arriba).
                should_downgrade = True

            if should_downgrade:
                logger.info(f"⬇️ Degradando perfil de {user_id} a 'gratis'. Cancelación efectiva (graceful terminado o sin fecha de fin de ciclo).")
                # I2: mutación filtrada por id=user_id. Placeholders nombrados
                # (payload dict, paridad con el update PostgREST legacy).
                execute_sql_write(
                    "UPDATE user_profiles SET plan_tier = %(plan_tier)s, "
                    "subscription_status = %(subscription_status)s "
                    "WHERE id = %(id)s",
                    {  # pyright: ignore[reportArgumentType]
                        "plan_tier": "gratis",
                        "subscription_status": "INACTIVE",
                        "id": user_id,
                    },
                )
                profile["plan_tier"] = "gratis"
                profile["subscription_status"] = "INACTIVE"
        # ---------------------------------------
        
        return profile
    except Exception as e:
        logger.error(f"Error obteniendo perfil: {e}")
        return None


def get_user_plan_tier(user_id: str) -> Optional[str]:
    """[P0-DEEPSEEK-MIGRATION · 2026-06-12] Lookup liviano de `plan_tier`
    para el router de modelos LLM (`llm_provider.resolve_model_for_user`).

    Deliberadamente NO reusa `get_user_profile`: ese helper trae el perfil
    completo Y ejecuta side-effects (downgrade de suscripciones CANCELLED).
    El router de modelos corre en el hot path de CADA llamada LLM (con cache
    TTL upstream) — necesita un SELECT de una columna, sin side-effects.

    Retorna el tier crudo (`gratis`/`basic`/`plus`/`ultra`) o None si el
    perfil no existe (guests / session_ids). El caller normaliza y aplica
    fail-cheap. Excepciones propagan — el caller (`llm_provider.get_user_tier`)
    las captura y degrada a `gratis`.
    """
    if not user_id:
        return None
    # [P1-NEON-DB-MIGRATION · 2026-06-12] SQL directo vía pool (hot path —
    # no abre conexión nueva). Sin try/except a propósito: las excepciones
    # DEBEN propagar (el caller degrada a 'gratis').
    row = execute_sql_query(
        "SELECT plan_tier FROM user_profiles WHERE id = %s LIMIT 1",
        (user_id,),
        fetch_one=True,
    )
    if row:
        return row.get("plan_tier") or "gratis"
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
        pool configurado), por default degrada al patrón legacy
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
                f"en db_core (variables de entorno NEON_DATABASE_URL / "
                f"NEON_DATABASE_URL_POOLED, conectividad al pooler de Neon)."
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
    if not connection_pool: return None
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

        # [P1-NEON-DB-MIGRATION · 2026-06-12] RETURNING id preserva la
        # semántica PostgREST "res.data no-vacío si afectó fila" (los callers
        # solo chequean truthiness; None en error).
        from psycopg.types.json import Jsonb
        res = execute_sql_write(
            "UPDATE user_profiles SET health_profile = %s::jsonb WHERE id = %s RETURNING id",
            (Jsonb(health_profile), user_id),
            returning=True,
        )

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

        return res
    except Exception as e:
        logger.error(f"Error actualizando health_profile: {e}")
        return None

def log_api_usage(user_id: str, endpoint: str = "llm"):
    """Guarda un registro de uso de la API (consume 1 crédito)."""
    if not user_id or user_id == "guest": return None
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Rama fallback PostgREST eliminada
        # — sin pool no hay datasource (los datos viven en Neon).
        from db_core import connection_pool
        if not connection_pool:
            return None
        res = execute_sql_write("INSERT INTO api_usage (user_id, endpoint) VALUES (%s, %s)", (user_id, endpoint))
        return res
    except Exception as e:
        logger.error(f"Error registrando api_usage: {e}")
        return None


# ============================================================
# [P1-COST-INSTRUMENTATION · 2026-05-15] Financial accounting per-LLM-call.
# ------------------------------------------------------------
# `api_usage` (arriba) es count-based para el paywall mensual (gratis=15 /
# basic=50 / plus=200). NO captura tokens, model, ni costo monetario.
# Audit 2026-05-15 estimó ~$0.06-$0.15/plan pero sin observabilidad real.
#
# `llm_usage_events` (migración p1_cost_instrumentation_2026_05_15.sql) y
# las dos funciones de abajo cierran ese gap:
#   - `compute_llm_cost_micros(model, in, out, cached)` → USD * 1e6.
#   - `log_llm_usage_event(...)` → INSERT best-effort post-LLM-success.
#
# Persistencia best-effort: cualquier fallo se silencia (no rompe la
# llamada LLM). Mismo patrón que `_emit_llm_timeout_metric` en
# graph_orchestrator.py.
# ============================================================

# Pricing default (USD por 1M tokens, expresado en MICROS para evitar
# floats). [P0-DEEPSEEK-MIGRATION · 2026-06-12] Basado en pricing oficial
# DeepSeek V4 (api-docs.deepseek.com, consultado 2026-06-12).
# Override sin redeploy via knob `MEALFIT_LLM_PRICING_JSON` (JSON string
# `{"<model_prefix>": {"input": <micros_per_M>, "output": <micros_per_M>,
# "cached": <micros_per_M>}}`).
#
# Match es por prefix-de-modelo (longest-prefix wins) — tolerante a sufijos
# de versión. Si el modelo es desconocido se retorna None y el evento se
# persiste sin costo (operador puede backfillar luego ejecutando SQL con
# tokens × pricing nuevo).
_DEFAULT_LLM_PRICING_MICROS_PER_M: Dict[str, Dict[str, int]] = {
    # DeepSeek V4 (USD/1M tokens):
    #   flash: input $0.14 (cache miss) / $0.0028 (cache hit), output $0.28
    #   pro:   input $0.435 (cache miss) / $0.003625 (cache hit), output $0.87
    # "cached" = rate por token con cache HIT (DeepSeek context caching es
    # automático server-side; el usage del API reporta hit/miss).
    "deepseek-v4-flash": {"input": 140_000, "output": 280_000, "cached": 2_800},
    "deepseek-v4-pro":   {"input": 435_000, "output": 870_000, "cached": 3_625},
    # Aliases legacy del API (deprecan 2026-07-24) — mismo pricing que su
    # equivalente V4. Presentes por si un knob los referencia en transición.
    "deepseek-chat":     {"input": 140_000, "output": 280_000, "cached": 2_800},
    "deepseek-reasoner": {"input": 435_000, "output": 870_000, "cached": 3_625},
}


def _resolve_pricing_table() -> Dict[str, Dict[str, int]]:
    """Combina defaults + override del knob `MEALFIT_LLM_PRICING_JSON`.

    Knob es opcional; si falla parsear, log debug y se usa solo defaults
    (fail-safe: no rompe la instrumentación si alguien typea mal el JSON
    en el VPS Oracle).
    """
    table = dict(_DEFAULT_LLM_PRICING_MICROS_PER_M)
    raw = os.environ.get("MEALFIT_LLM_PRICING_JSON", "").strip()
    if not raw:
        return table
    try:
        override = json.loads(raw)
        if isinstance(override, dict):
            for k, v in override.items():
                if isinstance(v, dict) and "input" in v and "output" in v:
                    table[k] = {
                        "input": int(v["input"]),
                        "output": int(v["output"]),
                        "cached": int(v.get("cached", int(v["input"]) // 4)),
                    }
    except Exception as e:
        logger.debug(f"[P1-COST-INSTRUMENTATION] pricing override parse failed: {e!r}")
    return table


def compute_llm_cost_micros(
    model: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    cached_tokens: Optional[int] = 0,
) -> Optional[int]:
    """Calcula costo en USD*1e6 (micros) para una llamada LLM.

    Match por longest-prefix sobre `model`. Cached tokens se descuentan del
    input (input_billed = input - cached) + cached × cached_rate.
    Retorna None si modelo desconocido o tokens inválidos — la fila se
    persiste igual, operador puede backfillar.
    """
    if not model or input_tokens is None or output_tokens is None:
        return None
    try:
        in_t = max(0, int(input_tokens))
        out_t = max(0, int(output_tokens))
        cached_t = max(0, int(cached_tokens or 0))
    except (TypeError, ValueError):
        return None

    table = _resolve_pricing_table()
    matched_key = None
    for key in table:
        if model.startswith(key):
            if matched_key is None or len(key) > len(matched_key):
                matched_key = key
    if matched_key is None:
        return None

    rates = table[matched_key]
    billable_input = max(0, in_t - cached_t)
    cost = (
        billable_input * rates["input"]
        + cached_t * rates["cached"]
        + out_t * rates["output"]
    )
    return int(cost // 1_000_000)


def log_llm_usage_event(
    *,
    user_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    model: str,
    node: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    cached_tokens: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """[P1-COST-INSTRUMENTATION · 2026-05-15] INSERT best-effort a
    `llm_usage_events` con tokens + cost calculado.

    Kill switch: `MEALFIT_LLM_COST_TRACKING_ENABLED=0` desactiva sin
    redeploy. Cualquier excepción se silencia (no enmascara errores del
    LLM upstream). `user_id`/`plan_id` pueden quedar NULL en esta fase
    inicial — phase 2 los inyectará via contextvar desde el orquestador.
    """
    if not _env_bool("MEALFIT_LLM_COST_TRACKING_ENABLED", True):
        return
    if not model:
        return
    try:
        cost_micros = compute_llm_cost_micros(
            model, input_tokens, output_tokens, cached_tokens or 0
        )
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        from db_core import connection_pool
        if connection_pool:
            execute_sql_write(
                """
                INSERT INTO llm_usage_events
                    (user_id, plan_id, model, node, input_tokens,
                     output_tokens, cached_tokens, cost_usd_micros, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    user_id, plan_id, model, node,
                    input_tokens, output_tokens, cached_tokens,
                    cost_micros, meta_json,
                ),
            )
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Rama fallback PostgREST eliminada:
        # sin pool el evento se pierde (best-effort, mismo contrato que un fallo).
    except Exception as e:
        try:
            logger.debug(
                f"[P1-COST-INSTRUMENTATION] log_llm_usage_event falló "
                f"(best-effort): {e!r}"
            )
        except Exception:
            pass

def get_monthly_api_usage(user_id: str) -> int:
    """Cuenta cuántas llamadas a la API ha hecho el usuario este mes."""
    if not user_id or user_id == "guest": return 0
    from datetime import datetime
    
    try:
        now = datetime.now()
        start_date = datetime(now.year, now.month, 1).isoformat()
        
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Rama fallback PostgREST (con su
        # retry loop específico de red REST) eliminada — pool o nada.
        from db_core import connection_pool
        if not connection_pool:
            return 0
        res = execute_sql_query("SELECT count(*) as total FROM api_usage WHERE user_id = %s AND created_at >= %s", (user_id, start_date), fetch_one=True)
        if res and 'total' in res:
            return int(res['total'])
        return 0
    except Exception as e:
        logger.error(f"Error obteniendo api_usage mensual: {e}")
        return 0

def migrate_guest_data(session_ids: list, new_user_id: str):
    """
    Migra todos los datos asociados a uno o varios session_ids temporales
    (creados cuando el usuario era invitado) hacia el nuevo UUID del usuario registrado.
    """
    if not connection_pool or not session_ids or not new_user_id:
        return False

    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Las 10 updates PostgREST
        # secuenciales ahora corren en UNA transacción (migra todo o nada —
        # antes un fallo a mitad dejaba la migración parcial committeada).
        # Columnas uuid requieren cast %s::uuid[]: psycopg adapta list[str]
        # como text[] y `uuid = ANY(text[])` no resuelve operador.
        # meal_rejections.session_id es TEXT — esa va sin cast.
        ids = list(session_ids)
        execute_sql_transaction([
            # 1. agent_sessions: vincular historiales de chat (filtro por PK id)
            ("UPDATE agent_sessions SET user_id = %s WHERE id = ANY(%s::uuid[])", (new_user_id, ids)),
            # 2. visual_diary (Vectores/Diario Visual)
            ("UPDATE visual_diary SET user_id = %s WHERE user_id = ANY(%s::uuid[])", (new_user_id, ids)),
            # 3. user_facts (Vectores/Memoria a largo plazo)
            ("UPDATE user_facts SET user_id = %s WHERE user_id = ANY(%s::uuid[])", (new_user_id, ids)),
            # 4. meal_plans (planes guardados como guest, si los hubiera)
            ("UPDATE meal_plans SET user_id = %s WHERE user_id = ANY(%s::uuid[])", (new_user_id, ids)),
            # 5. consumed_meals (comidas registradas)
            ("UPDATE consumed_meals SET user_id = %s WHERE user_id = ANY(%s::uuid[])", (new_user_id, ids)),
            # 6. pending_facts_queue
            ("UPDATE pending_facts_queue SET user_id = %s WHERE user_id = ANY(%s::uuid[])", (new_user_id, ids)),
            # 7. meal_rejections: primero por session_id (así se guardan los
            # rechazos guest; columna TEXT)...
            ("UPDATE meal_rejections SET user_id = %s WHERE session_id = ANY(%s)", (new_user_id, ids)),
            # ...y también por user_id por si acaso
            ("UPDATE meal_rejections SET user_id = %s WHERE user_id = ANY(%s::uuid[])", (new_user_id, ids)),
            # 8. meal_likes
            ("UPDATE meal_likes SET user_id = %s WHERE user_id = ANY(%s::uuid[])", (new_user_id, ids)),
            # 9. user_inventory (Inventario Físico / Despensa)
            ("UPDATE user_inventory SET user_id = %s WHERE user_id = ANY(%s::uuid[])", (new_user_id, ids)),
        ])
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
    """Borra preferencias, rechazos, inventario, planes históricos y limpia el health_profile para un verdadero inicio desde cero.

    [P3-RESET-SINGLE-TXN · 2026-05-16] Pre-fix: 7 llamadas secuenciales a
    `execute_sql_write`, cada una hacía `with connection_pool.connection()`
    → 7 connection acquires del pool. En free tier saturado (pool cap ~25,
    visible en logs `couldn't get a connection after 8.00 sec`), cada
    acquire podía tomar hasta 8s → wall-clock 0.4-56s para el reset
    completo → usuario veía el botón "Sí, empezar desde cero" sin reaccionar
    varios segundos.

    Post-fix: TODAS las operaciones bajo UNA conexión + UNA transacción.
    1 acquire del pool, 7 statements, 1 commit. Reducción wall-clock 7x
    en best case, mucho más en pool saturado.

    Atomicidad bonus: si cualquier statement falla, ROLLBACK preserva la
    cuenta consistente. Pre-fix, un fallo en el statement #5 (después de
    borrar #1-4) dejaba la cuenta en estado parcial.
    """
    # [P1-NEON-DB-MIGRATION · 2026-06-12] El object storage ya no es precondición:
    # el reset es 100% SQL; el purge de object storage (best-effort) valida
    # su propio cliente internamente.
    if not user_id or user_id == "guest":
        return False

    if not connection_pool:
        logger.error("❌ reset_user_account_preferences: connection_pool no disponible.")
        return False

    try:
        # [P3-RESET-SINGLE-TXN] Una conexión + una transacción.
        # `with conn.transaction()` garantiza COMMIT en success y
        # ROLLBACK automático en cualquier excepción.
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cursor:
                    # 1. Likes
                    cursor.execute("DELETE FROM meal_likes WHERE user_id = %s", (user_id,))
                    # 2. Rejections
                    cursor.execute("DELETE FROM meal_rejections WHERE user_id = %s", (user_id,))
                    # 3. Inventario
                    cursor.execute("DELETE FROM user_inventory WHERE user_id = %s", (user_id,))
                    # 4. Knowledge graph / facts aprendidos
                    cursor.execute("DELETE FROM user_facts WHERE user_id = %s", (user_id,))
                    # 5. Frecuencias de ingredientes
                    cursor.execute("DELETE FROM ingredient_frequencies WHERE user_id = %s", (user_id,))
                    # 6. Planes históricos (CASCADE plan_chunk_queue)
                    cursor.execute("DELETE FROM meal_plans WHERE user_id = %s", (user_id,))
                    # 7. Limpiar health_profile
                    cursor.execute(
                        "UPDATE user_profiles SET health_profile = '{}'::jsonb WHERE id = %s",
                        (user_id,),
                    )
                    # 8. [P1-PROD-AUDIT-2 · 2026-05-30] Borrar también el diario
                    # visual: alimenta el RAG visual / memoria a largo plazo del
                    # agente, así que dejarlo contradice "verdadero inicio desde
                    # cero" (el agente seguiría "recordando" fotos previas). El
                    # objeto físico en Storage se purga best-effort fuera de la
                    # transacción (Storage no es transaccional con Postgres).
                    cursor.execute("DELETE FROM visual_diary WHERE user_id = %s", (user_id,))
        # [P1-PROD-AUDIT-2] Storage best-effort tras commit DB (no bloquea el reset).
        try:
            _purge_visual_diary_storage(user_id)
        except Exception as _st_e:
            logger.warning(f"[P1-PROD-AUDIT-2] reset: purge Storage visual_diary falló (best-effort): {_st_e}")
        logger.info(f"♻️ Preferencias y planes reseteados DESDE CERO con éxito para UUID {user_id}")
        return True
    except Exception as e:
        logger.error(f"❌ Error reseteando preferencias para {user_id}: {e}")
        return False


# ───────────────────────────────────────────────────────────────────────────
# [P1-PROD-AUDIT-2 · 2026-05-30] Borrado de cuenta DETERMINÍSTICO.
#
# Pre-fix NO existía flujo de borrado programático: el SRE corría
# `auth.admin.delete_user` (manual vía dashboard), que solo cascadea las ~19
# tablas con FK `ON DELETE CASCADE`. Las ~12 tablas con columna `user_id` SIN FK
# (user_facts, visual_diary, weight_log, agent_sessions, ingredient_frequencies,
# learning_experiments, abandoned_meal_reasons, nudge_outcomes, pipeline_metrics,
# plan_chunk_metrics, chunk_deferrals, chunk_lesson_telemetry — varias con
# `user_id` TEXT, por eso nunca pudieron FK-ear a auth.users uuid) quedaban con
# PII huérfana INDEFINIDAMENTE, contradiciendo la Política de Privacidad
# ("eliminar todos los datos asociados / CASCADE sobre todas las tablas").
# Además el objeto físico en Storage `visual_diary_images/{user_id}/...` nunca se
# borraba (Storage no cascadea de auth.users) y los checkpoints LangGraph
# (keyed por `thread_id`, sin columna `user_id`) tampoco.
#
# Este helper borra TODA la data user-scoped de forma determinística, sin
# depender de FKs ni de tipos. Best-effort per-tabla (cada DELETE en su propio
# autocommit) → robusto ante orden de FK y re-ejecutable (borrar filas ya
# cascadeadas es no-op). Pensado para invocarse desde el endpoint admin
# `/api/system/admin/account/purge-data`, ANTES de `auth.admin.delete_user`.
#
# Decisión per-tabla (la que prometía documentar `p1_new_5`):
#   - PII de salud/contenido → DELETE (user_facts, visual_diary, weight_log,
#     consumed_meals, conversation_summaries, agent_messages, agent_sessions,
#     abandoned_meal_reasons, meal_plans).
#   - Telemetría/operacional con user_id → DELETE (no son verdaderamente anónimas:
#     pipeline_metrics, plan_chunk_metrics, learning_experiments, nudge_outcomes,
#     chunk_*, ingredient_frequencies, api_usage, llm_usage_events).
#   - meal_plans_audit (backup forense) → DELETE (contiene plan_data = PII).
# ───────────────────────────────────────────────────────────────────────────
_USER_SCOPED_TABLES_USERID = (
    "abandoned_meal_reasons", "agent_messages", "api_usage", "chunk_deferrals",
    "chunk_lesson_telemetry", "chunk_user_locks", "consumed_meals",
    "conversation_summaries", "custom_shopping_items", "failed_inventory_deductions",
    "ingredient_frequencies", "learning_experiments", "llm_usage_events",
    "meal_likes", "meal_rejections", "nightly_rotation_queue", "nudge_outcomes",
    "pending_facts_queue", "pipeline_metrics", "plan_chunk_metrics",
    "plan_chunk_queue", "push_subscriptions", "shopping_locks",
    "unknown_ingredients", "user_depleted_items", "user_facts", "user_inventory",
    "visual_diary", "water_intake_log", "weight_log", "meal_plans_audit",
    # [P1-ACCOUNT-DELETE-1 · 2026-06-22] Tablas Dreaming (P1-DREAMING-1, creadas
    # 2026-06-13 DESPUÉS de esta lista del 2026-05-30). Cascadean a user_profiles
    # ON DELETE CASCADE, así que con include_profile=True ya se limpiaban; se
    # añaden EXPLÍCITAS para defensa-en-profundidad (la lista es el SSOT del
    # borrado, NO el grafo de FKs — Neon eliminó los FKs a auth.users) y para que
    # include_profile=False (data-erasure conservando cuenta) no las deje huérfanas.
    "user_memory_profile", "dream_work_queue", "dream_consolidation_log",
    # `meal_plans` al final: sus children (plan_chunk_queue, etc.) pueden
    # FK-cascade a él; borrarlas antes evita cualquier orden problemático.
    "meal_plans",
)


def _purge_visual_diary_storage(user_id: str) -> int:
    """Best-effort: borra los objetos del bucket `visual_diary_images/{user_id}/`.
    El object storage de visual_diary no cascadea desde auth.users → sin esto
    las fotos sobreviven a cualquier borrado. Retorna nº de objetos borrados
    (0 si falla / no hay)."""
    if not _storage_client or not user_id:
        return 0
    try:
        bucket = _storage_client.storage.from_("visual_diary_images")
        entries = bucket.list(user_id) or []
        # `_storage_client` es placeholder `None` (object storage de visual_diary
        # pendiente de migrar, ver db_core); el guard `if not _storage_client`
        # arriba hace que pyright narre el cuerpo a Never y marque
        # `for e in entries` como no-iterable. FP — código intencional
        # preservado para cuando vuelva el object storage. Cero efecto runtime.
        paths = [
            f"{user_id}/{e['name']}"
            for e in entries  # pyright: ignore[reportGeneralTypeIssues]
            if isinstance(e, dict) and e.get("name")
        ]
        if paths:
            bucket.remove(paths)
        return len(paths)
    except Exception as e:
        logger.warning(
            f"[P1-PROD-AUDIT-2] purge Storage visual_diary {user_id} falló (best-effort): {e}"
        )
        return 0


def delete_account_data(user_id: str, include_profile: bool = True) -> Dict[str, Any]:
    """Purga determinística de TODA la data user-scoped (ver bloque de doc arriba).

    Args:
        user_id: UUID del usuario.
        include_profile: si True, borra también la fila de `user_profiles`
            (último, porque varias tablas FK a él). False = purge de datos sin
            tocar el perfil (e.g. GDPR data-erasure conservando la cuenta auth).

    Returns:
        dict con `deleted` (counts per tabla), `storage_objects_removed`, `errors`.
    """
    result: Dict[str, Any] = {
        "user_id": user_id,
        "deleted": {},
        "errors": [],
        "storage_objects_removed": 0,
    }
    # [P1-NEON-DB-MIGRATION · 2026-06-12] El object storage queda fuera de la precondición:
    # la purga de datos es 100% SQL; el object storage (best-effort) chequea su cliente.
    if not connection_pool or not user_id or user_id == "guest":
        result["errors"].append("precondición inválida (pool/user_id)")
        return result
    try:
        uuid.UUID(str(user_id))
    except Exception:
        result["errors"].append("user_id no es UUID válido")
        return result

    # 1. Checkpoints LangGraph (keyed por thread_id = agent_sessions.id::text).
    #    DELETE ANTES de borrar agent_sessions (su fuente de thread_ids).
    for ck_tbl in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
        try:
            r = execute_sql_write(
                f"DELETE FROM {ck_tbl} WHERE thread_id IN "
                "(SELECT id::text FROM agent_sessions WHERE user_id = %s) RETURNING thread_id",
                (user_id,), returning=True,
            )
            result["deleted"][ck_tbl] = len(r) if isinstance(r, list) else 0
        except Exception as e:
            result["errors"].append(f"{ck_tbl}: {e}")

    # 2. agent_sessions + 3. todas las tablas user-scoped (best-effort per-tabla).
    for tbl in ("agent_sessions",) + _USER_SCOPED_TABLES_USERID:
        try:
            r = execute_sql_write(
                f"DELETE FROM {tbl} WHERE user_id = %s RETURNING user_id",
                (user_id,), returning=True,
            )
            result["deleted"][tbl] = len(r) if isinstance(r, list) else 0
        except Exception as e:
            result["errors"].append(f"{tbl}: {e}")

    # 4. Storage (best-effort, no transaccional con Postgres).
    result["storage_objects_removed"] = _purge_visual_diary_storage(user_id)

    # 5. user_profiles último (varias tablas FK a él).
    if include_profile:
        try:
            r = execute_sql_write(
                "DELETE FROM user_profiles WHERE id = %s RETURNING id",
                (user_id,), returning=True,
            )
            result["deleted"]["user_profiles"] = len(r) if isinstance(r, list) else 0
        except Exception as e:
            result["errors"].append(f"user_profiles: {e}")

    logger.info(
        f"[P1-PROD-AUDIT-2] delete_account_data({user_id}): "
        f"tablas_con_filas={sum(1 for v in result['deleted'].values() if v)}, "
        f"storage={result['storage_objects_removed']}, errors={len(result['errors'])}"
    )
    return result


# [LONG-TERM-MEMORY-TOGGLE · 2026-05-13]
# Helper para el toggle de memoria a largo plazo desde Settings.
# Migración SSOT: migrations/add_long_term_memory_enabled_2026_05_13.sql
def update_long_term_memory_enabled(user_id: str, enabled: bool) -> bool:
    """Actualiza el flag `long_term_memory_enabled` en user_profiles.

    Devuelve True si el UPDATE afectó una fila. Filtrado por user_id
    (invariante I2: toda mutación a user_profiles requiere user_id explícito).
    """
    if not connection_pool:
        return False
    try:
        # I2: filtro por id=user_id. RETURNING id → "afectó una fila".
        res = execute_sql_write(
            "UPDATE user_profiles SET long_term_memory_enabled = %s WHERE id = %s RETURNING id",
            (bool(enabled), user_id),
            returning=True,
        )
        return bool(res)
    except Exception as e:
        logger.error(f"❌ Error update_long_term_memory_enabled({user_id}, {enabled}): {e}")
        return False


# [P2-AI-TRAINING-CONSENT · 2026-07-04] Consentimiento OPT-IN para uso futuro
# de datos en entrenamiento de modelos propios. Migración SSOT:
# migrations/p2_ai_training_consent_2026_07_04.sql. DEFAULT false (fail-secure).
def update_ai_training_consent(user_id: str, enabled: bool) -> bool:
    """Actualiza el flag `ai_training_consent` en user_profiles.

    Devuelve True si el UPDATE afectó una fila. Filtrado por user_id
    (invariante I2: toda mutación a user_profiles requiere user_id explícito).
    """
    if not connection_pool:
        return False
    try:
        res = execute_sql_write(
            "UPDATE user_profiles SET ai_training_consent = %s WHERE id = %s RETURNING id",
            (bool(enabled), user_id),
            returning=True,
        )
        return bool(res)
    except Exception as e:
        logger.error(f"❌ Error update_ai_training_consent({user_id}, {enabled}): {e}")
        return False


def get_ai_training_consented_user_ids() -> list:
    """[P2-AI-TRAINING-CONSENT · 2026-07-04] **Gate SSOT del corpus de training.**

    TODO pipeline futuro que recolecte/exporte datos de usuarios para entrenar
    modelos propios DEBE obtener su universo de usuarios de esta función —
    NUNCA de un SELECT propio sin el filtro de consentimiento. Fail-secure:
    error de DB → lista vacía (nadie consintió).
    """
    if not connection_pool:
        return []
    try:
        rows = execute_sql_query(
            "SELECT id FROM user_profiles WHERE ai_training_consent IS TRUE",
            fetch_all=True,
        ) or []
        return [str(r["id"]) for r in rows]
    except Exception as e:
        logger.error(f"❌ Error get_ai_training_consented_user_ids: {e}")
        return []


# [P3-WATER-TRACKER · 2026-05-16] Helper para el toggle del tracker de
# hidratación. Migración SSOT: migrations/add_water_tracker_enabled_2026_05_16.sql
def update_water_tracker_enabled(user_id: str, enabled: bool) -> bool:
    """Actualiza el flag `water_tracker_enabled` en user_profiles.

    Devuelve True si el UPDATE afectó una fila. Filtrado por user_id
    (invariante I2). Default TRUE — el toggle se apaga explícitamente
    por el usuario desde Preferencias.
    """
    if not connection_pool:
        return False
    try:
        # I2: filtro por id=user_id (equivalente PostgREST legacy: .eq("id", user_id)).
        # RETURNING id → "afectó una fila".
        res = execute_sql_write(
            "UPDATE user_profiles SET water_tracker_enabled = %s WHERE id = %s RETURNING id",
            (bool(enabled), user_id),
            returning=True,
        )
        return bool(res)
    except Exception as e:
        logger.error(f"❌ Error update_water_tracker_enabled({user_id}, {enabled}): {e}")
        return False


def get_water_tracker_enabled(user_id: str) -> bool:
    """Lee el flag `water_tracker_enabled` para `user_id`. Fail-secure:
    cualquier excepción / fila inexistente → True (default, mismo que la
    columna DB)."""
    if not connection_pool or not user_id:
        return True
    try:
        row = execute_sql_query(
            "SELECT water_tracker_enabled FROM user_profiles WHERE id = %s LIMIT 1",
            (user_id,),
            fetch_one=True,
        )
        if row:
            val = row.get("water_tracker_enabled")
            if val is None:
                return True
            return bool(val)
        return True
    except Exception as e:
        logger.warning(f"[P3-WATER-TRACKER] get_water_tracker_enabled({user_id}) error: {e}")
        return True


# [P3-AGENT-HYDRATION-CONTEXT · 2026-05-27] Helper para que el agente IA
# inyecte hidratación viva al system prompt. Reutiliza la tabla canónica
# `water_intake_log` que ya alimenta /api/plans/water-intake. Filtra por
# `user_id` (invariante I2) + `log_date` (fecha local del cliente).
def get_water_intake_glasses_today(user_id: str, log_date: str) -> int:
    """Lee los vasos consumidos hoy para `user_id`. Retorna 0 si no hay
    fila o si la DB no está disponible. Fail-secure: cualquier error →
    0 (no inventamos un valor que confunda al agente).

    Args:
        user_id: UUID del usuario autenticado.
        log_date: Fecha local del cliente en formato YYYY-MM-DD.

    Returns:
        int: número de vasos registrados hoy (0..14).
    """
    if not connection_pool or not user_id or not log_date:
        return 0
    try:
        row = execute_sql_query(
            "SELECT glasses FROM water_intake_log WHERE user_id = %s AND log_date = %s LIMIT 1",
            (user_id, log_date),
            fetch_one=True,
        )
        if row:
            return int(row.get("glasses") or 0)
        return 0
    except Exception as e:
        logger.warning(
            f"[P3-AGENT-HYDRATION-CONTEXT] get_water_intake_glasses_today"
            f"({user_id}, {log_date}) error: {e}"
        )
        return 0
