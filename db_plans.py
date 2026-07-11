from functools import lru_cache as _lru_cache
import uuid
import unicodedata as _uc
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import logging
logger = logging.getLogger(__name__)
from db_core import connection_pool, execute_sql_query, execute_sql_write
from constants import strip_accents, GLOBAL_REVERSE_MAP
from db_chat import insert_rejection, save_message
from db_profiles import get_user_profile, update_user_health_profile, update_user_health_profile_atomic

# ============================================================
# [P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Tracking de pipelines en curso
# en `app_kv_store` para soportar "deep search style":
#   - El pipeline NO se cancela si el cliente cierra el SSE/pestaña.
#   - El usuario puede volver a la app y ver el plan listo (toast +
#     redirect al dashboard) sin que se haya perdido la generación.
#
# Storage layer: `app_kv_store` (sobrevive restarts del backend, no toca
# `meal_plans` schema). Key = `pending_pipeline:<user_id>` (per-user,
# guardrail "1 pipeline activo por user a la vez"). Value JSON:
#   { started_at, status, plan_id_final, error }
# Status: 'generating' | 'complete' | 'failed' | 'abandoned'.
#
# Guardrail: si el usuario inicia un nuevo plan mientras hay uno
# `generating` < 15 min de edad, el endpoint rechaza con 409 + plan_id
# para que el frontend se conecte al pipeline existente en lugar de
# disparar uno nuevo (evita pagar 2× Gemini por el mismo user).
#
# Limpieza: cron `_finalize_zombie_partial_plans` (cron_tasks.py) limpia
# rows `generating` > 1h con `status='abandoned'`. Ver follow-up.
# ============================================================
_PENDING_PIPELINE_KV_PREFIX = "pending_pipeline:"


def _pending_pipeline_kv_key(user_id: str) -> str:
    return f"{_PENDING_PIPELINE_KV_PREFIX}{user_id}"


def upsert_pending_pipeline(user_id: str, status: str = "generating",
                             plan_id_final: Optional[str] = None,
                             error: Optional[str] = None) -> bool:
    """[P1-DEEP-SEARCH-PIPELINE · 2026-05-15] Upserta el estado del pipeline
    pendiente para `user_id`. Status: generating|complete|failed.

    Best-effort: fallo silencioso. Si la KV no está disponible, el sistema
    sigue funcionando — solo se pierde la capacidad de recovery cross-restart.
    """
    if not user_id:
        return False
    try:
        from datetime import datetime, timezone
        import json as _json
        now_iso = datetime.now(timezone.utc).isoformat()
        payload = {
            "status": status,
            "started_at": now_iso if status == "generating" else None,
            "updated_at": now_iso,
            "plan_id_final": plan_id_final,
            "error": error,
        }
        # Para status != 'generating', preservar `started_at` del row previo si existe.
        if status != "generating":
            try:
                prev = get_pending_pipeline(user_id)
                if prev and prev.get("started_at"):
                    payload["started_at"] = prev["started_at"]
            except Exception:
                pass

        key = _pending_pipeline_kv_key(user_id)
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Fallback PostgREST eliminado:
        # post-Neon el pool es mandatorio (un fallback REST apuntaría a la DB
        # equivocada). Si el pool no está, execute_sql_write lanza y el except
        # de abajo preserva el contrato best-effort (False + warning).
        execute_sql_write(
            """
            INSERT INTO app_kv_store (key, value, updated_at)
            VALUES (%s, %s::jsonb, NOW())
            ON CONFLICT (key) DO UPDATE
              SET value = EXCLUDED.value, updated_at = NOW()
            """,
            (key, _json.dumps(payload)),
        )
        return True
    except Exception as e:
        # [P1-DEEP-SEARCH-DEBUG · 2026-05-15] Elevado de debug→warning para
        # diagnosticar por qué el KV pending_pipeline no se popula bajo carga.
        # Si el log se vuelve ruidoso post-diagnóstico, bajar a info.
        logger.warning(
            f"[P1-DEEP-SEARCH-PIPELINE] upsert_pending_pipeline FAILED "
            f"user_id={user_id} status={status} plan_id_final={plan_id_final} "
            f"error={e!r}"
        )
        return False


def get_pending_pipeline(user_id: str) -> Optional[Dict[str, Any]]:
    """Lee el estado actual del pipeline pendiente del user. None si no existe."""
    if not user_id:
        return None
    try:
        key = _pending_pipeline_kv_key(user_id)
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Fallback PostgREST eliminado (pool mandatorio).
        row = execute_sql_query(
            "SELECT value, updated_at FROM app_kv_store WHERE key = %s",
            (key,), fetch_one=True
        )
        if not row:
            return None
        return row.get("value") or None
    except Exception as e:
        # [P1-DEEP-SEARCH-DEBUG · 2026-05-15] Elevado de debug→warning.
        logger.warning(
            f"[P1-DEEP-SEARCH-PIPELINE] get_pending_pipeline FAILED "
            f"user_id={user_id} error={e!r}"
        )
        return None


def check_user_has_active_pipeline(user_id: str, max_age_min: int = 15) -> Optional[Dict[str, Any]]:
    """[P1-DEEP-SEARCH-PIPELINE GUARDRAIL] Retorna el payload si el user tiene
    un pipeline `status='generating'` con `started_at` < `max_age_min` ago.
    None si no hay activo (o está stale, en cuyo caso un nuevo plan es OK).

    Stale entries son ignoradas (el cron de limpieza eventualmente los marca
    abandoned; pero un user que vuelve no debe ser bloqueado por uno stale).
    """
    payload = get_pending_pipeline(user_id)
    if not payload or payload.get("status") != "generating":
        return None
    try:
        from datetime import datetime, timezone, timedelta
        started_at_str = payload.get("started_at")
        if not started_at_str:
            return None
        started_at = datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - started_at
        if age > timedelta(minutes=max_age_min):
            return None  # stale, ignorar (cron limpiará)
        return payload
    except Exception as e:
        # [P1-DEEP-SEARCH-DEBUG · 2026-05-15] Elevado de debug→warning.
        logger.warning(
            f"[P1-DEEP-SEARCH-PIPELINE] check_user_has_active_pipeline parse error "
            f"user_id={user_id} error={e!r}"
        )
        return None


def clear_pending_pipeline(user_id: str) -> bool:
    """Borra el row de pending pipeline. Usado tras frontend acknowledge
    la finalización (i.e. el usuario aceptó el plan o vio el error)."""
    if not user_id:
        return False
    try:
        key = _pending_pipeline_kv_key(user_id)
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Fallback PostgREST eliminado (pool mandatorio).
        execute_sql_write("DELETE FROM app_kv_store WHERE key = %s", (key,))
        return True
    except Exception as e:
        # [P1-DEEP-SEARCH-DEBUG · 2026-05-15] Elevado de debug→warning.
        logger.warning(
            f"[P1-DEEP-SEARCH-PIPELINE] clear_pending_pipeline FAILED "
            f"user_id={user_id} error={e!r}"
        )
        return False


# ============================================================
# [P1-GUEST-PLAN-RECOVERY · 2026-07-09] Persistencia de planes de GUESTS en KV.
# ------------------------------------------------------------
# Los guests (no logueados) NO persisten en `meal_plans` (FK a user_profiles) →
# si cierran la pestaña mid-generación, el plan se descartaba (nada que recuperar).
# El plan del guest se guarda en `app_kv_store` bajo `guest_plan:<session_id>` con
# TTL vía `expires_at` (limpieza en el cron de zombies). El STATUS del pipeline sigue
# en `pending_pipeline:<session_id>` (reusa upsert/get_pending_pipeline con el
# session_id como key). Read/recovery vía GET /api/plans/guest-plan?session_id=x.
# ============================================================
_GUEST_PLAN_KV_PREFIX = "guest_plan:"


def _guest_plan_kv_key(session_id: str) -> str:
    return f"{_GUEST_PLAN_KV_PREFIX}{session_id}"


def upsert_guest_plan(session_id: str, plan_data: dict) -> bool:
    """[P1-GUEST-PLAN-RECOVERY · 2026-07-09] Guarda el plan_data de un guest en KV para recovery.
    Best-effort: fallo silencioso (la generación no se rompe si la KV no está)."""
    if not session_id or not isinstance(plan_data, dict) or not plan_data.get("days"):
        return False
    try:
        import json as _json
        execute_sql_write(
            """
            INSERT INTO app_kv_store (key, value, updated_at)
            VALUES (%s, %s::jsonb, NOW())
            ON CONFLICT (key) DO UPDATE
              SET value = EXCLUDED.value, updated_at = NOW()
            """,
            (_guest_plan_kv_key(session_id), _json.dumps({"plan_data": plan_data})),
        )
        return True
    except Exception as e:
        logger.warning(f"[P1-GUEST-PLAN-RECOVERY] upsert_guest_plan FAILED session={session_id} error={e!r}")
        return False


def get_guest_plan(session_id: str) -> Optional[Dict[str, Any]]:
    """[P1-GUEST-PLAN-RECOVERY · 2026-07-09] Lee el plan_data guardado de un guest. None si no existe."""
    if not session_id:
        return None
    try:
        row = execute_sql_query(
            "SELECT value FROM app_kv_store WHERE key = %s",
            (_guest_plan_kv_key(session_id),), fetch_one=True
        )
        if not row:
            return None
        val = row.get("value") or {}
        return (val.get("plan_data") if isinstance(val, dict) else None) or None
    except Exception as e:
        logger.warning(f"[P1-GUEST-PLAN-RECOVERY] get_guest_plan FAILED session={session_id} error={e!r}")
        return None


def clear_guest_plan(session_id: str) -> bool:
    """[P1-GUEST-PLAN-RECOVERY · 2026-07-09] Marca el plan de guest como acked tras ack (o al
    regenerar) — el frontend ya no lo necesita para recovery.

    [P2-GUEST-PLAN-FORENSIC-TTL · 2026-07-10] Forensic corr=d57ffe04 (2026-07-10): el DELETE
    inmediato aquí borraba el único rastro del plan minutos después de entregado — cuando el owner
    pidió revisar ESE plan, el KV ya no existía y hubo que reprocesar journalctl línea por línea.
    Ahora SOFT-marca (`acked_at` dentro del jsonb, mismo row) en vez de borrar: `get_guest_plan`
    sigue leyendo `plan_data` del mismo row sin cambios (no filtra por acked_at), preservando la
    ventana de forensics. El hard-delete vive en el cron `_sweep_stale_guest_plans`
    (cron_tasks.py, TTL configurable vía `MEALFIT_GUEST_PLAN_FORENSIC_TTL_HOURS`).
    tooltip-anchor: P2-GUEST-PLAN-FORENSIC-TTL"""
    if not session_id:
        return False
    try:
        execute_sql_write(
            """
            UPDATE app_kv_store
            SET value = value || jsonb_build_object('acked_at', NOW()::text)
            WHERE key = %s
            """,
            (_guest_plan_kv_key(session_id),),
        )
        return True
    except Exception as e:
        logger.warning(f"[P1-GUEST-PLAN-RECOVERY] clear_guest_plan FAILED session={session_id} error={e!r}")
        return False


def check_recent_meal_plan_exists(user_id: str, max_seconds: int = 30) -> bool:
    """Verifica si ya se ha guardado un plan para este usuario recientemente."""
    if not connection_pool: return False
    try:
        from datetime import datetime, timezone
        query = "SELECT created_at FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1"
        res = execute_sql_query(query, (user_id,), fetch_one=True)
        
        if res and "created_at" in res:
            last_saved = res["created_at"]
            if isinstance(last_saved, str):
                from constants import safe_fromisoformat
                last_saved = safe_fromisoformat(last_saved)
            if last_saved.tzinfo is None:
                last_saved = last_saved.replace(tzinfo=timezone.utc)
                
            now_utc = datetime.now(timezone.utc)
            if (now_utc - last_saved).total_seconds() < max_seconds:
                return True
    except Exception as e:
        logger.warning(f"⚠️ [DEDUP] Error en check_recent_meal_plan_exists: {e}")
    return False


def _regen_day_dedup_kv_key(user_id: str, plan_id: str, day_index: int) -> str:
    return f"regen_day_done:{user_id}:{plan_id}:{int(day_index)}"


def mark_regen_day_done(user_id: str, plan_id: str, day_index: int) -> bool:
    """[P2-REGEN-DAY-IDEMPOTENCY · 2026-06-26] (audit 3-flujos P2) Marca en `app_kv_store` que el día
    <day_index> del plan <plan_id> de <user_id> acaba de regenerarse con ÉXITO (se llama post-persist +
    post-cobro). Un retry del cliente tras perder la respuesta HTTP (red caída) lo lee vía
    `check_recent_regen_day` y recibe `already_applied` en vez de re-correr el loop LLM y RE-COBRAR.
    Best-effort: un fallo del KV NO rompe el update (solo se pierde la protección de idempotencia para
    ese retry). tooltip-anchor: P2-REGEN-DAY-IDEMPOTENCY"""
    if not user_id or not plan_id:
        return False
    try:
        import json as _json  # [P1-REGEN-DAY-IDEMPOTENCY-JSONFIX · 2026-06-28] _json era NameError aquí (el import
        # de la línea 54 es local de OTRA función) → la marca de idempotencia fallaba SIEMPRE, y un retry del cliente
        # tras corte de red re-corría el loop LLM y RE-COBRABA. Import local (mismo patrón que el otro callsite).
        key = _regen_day_dedup_kv_key(user_id, plan_id, day_index)
        execute_sql_write(
            """
            INSERT INTO app_kv_store (key, value, updated_at)
            VALUES (%s, %s::jsonb, NOW())
            ON CONFLICT (key) DO UPDATE
              SET value = EXCLUDED.value, updated_at = NOW()
            """,
            (key, _json.dumps({"day_index": int(day_index)})),
        )
        return True
    except Exception as e:
        logger.warning(
            f"[P2-REGEN-DAY-IDEMPOTENCY] mark_regen_day_done FAILED "
            f"user={user_id} plan={plan_id} day={day_index}: {e!r}"
        )
        return False


def check_recent_regen_day(user_id: str, plan_id: str, day_index: int, max_seconds: int = 45) -> bool:
    """[P2-REGEN-DAY-IDEMPOTENCY · 2026-06-26] True si el día <day_index> de <plan_id> se regeneró con
    éxito hace < `max_seconds` (marca puesta por `mark_regen_day_done`). Usado al inicio de
    /regenerate-day para cortar un retry duplicado ANTES del loop LLM + cobro. FAIL-OPEN: cualquier
    error o `max_seconds<=0` → False (el update procede normal). tooltip-anchor: P2-REGEN-DAY-IDEMPOTENCY"""
    if not user_id or not plan_id or max_seconds <= 0:
        return False
    try:
        from datetime import datetime, timezone
        key = _regen_day_dedup_kv_key(user_id, plan_id, day_index)
        row = execute_sql_query(
            "SELECT updated_at FROM app_kv_store WHERE key = %s",
            (key,), fetch_one=True
        )
        if not row or not row.get("updated_at"):
            return False
        ts = row["updated_at"]
        if isinstance(ts, str):
            from constants import safe_fromisoformat
            ts = safe_fromisoformat(ts)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds() < max_seconds
    except Exception as e:
        logger.warning(
            f"[P2-REGEN-DAY-IDEMPOTENCY] check_recent_regen_day FAILED "
            f"user={user_id} plan={plan_id} day={day_index}: {e!r}"
        )
        return False


def check_meal_plan_generated_today(user_id: str) -> bool:
    """Valida si el último plan generado por el usuario se realizó el día actual."""
    if not connection_pool: return False
    try:
        from datetime import datetime, timezone
        query = "SELECT created_at FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1"
        res = execute_sql_query(query, (user_id,), fetch_one=True)
        
        if res and "created_at" in res:
            last_saved = res["created_at"]
            if isinstance(last_saved, str):
                from constants import safe_fromisoformat
                last_saved = safe_fromisoformat(last_saved)
            if last_saved.tzinfo is None:
                last_saved = last_saved.replace(tzinfo=timezone.utc)
            
            now_utc = datetime.now(timezone.utc)
            if last_saved.date() == now_utc.date():
                return True
    except Exception as e:
        logger.error(f"Error comprobando si plan fue hoy: {e}")
    return False

# Columnas que son text[] en PostgreSQL (no jsonb) — constante de módulo compartida
_MEAL_PLAN_ARRAY_COLS = {"meal_names", "ingredients", "techniques"}


# [P1-5] Lock advisory unificado para serializar operaciones sobre un meal_plan.
# Antes coexistían dos invocaciones de `pg_advisory_xact_lock` con hash functions
# diferentes (`hashtext` en routers/plans.py vs `hashtextextended` en cron_tasks.py)
# y keys diferentes, así que dos call sites que SÍ querían colisionar (e.g., dos
# catchups concurrentes con dos invocaciones desincronizadas) podían no hacerlo.
# Este helper estandariza:
#   - Función de hash: `hashtextextended(text, 0)` (seed fijo, bigint estable).
#   - Espacio de keys: `meal_plan:<purpose>:<plan_id>`.
# El parámetro `purpose` permite múltiples namespaces sobre el mismo plan_id sin
# colisionar accidentalmente entre sí: dos `purpose='catchup'` se serializan,
# dos `purpose='tz_resync'` se serializan, pero un catchup y un tz_resync NO
# se bloquean (es lo deseado: operan sobre filas distintas).
#
# Uso:
#     with conn.cursor() as cur:
#         acquire_meal_plan_advisory_lock(cur, plan_id, purpose='catchup')
#         # ... resto de la transacción protegida
# El lock se libera automáticamente al cerrar la transacción (xact_lock).
_MEAL_PLAN_LOCK_PURPOSES = {
    "general",
    "catchup",        # routers/plans.py: serializar inserción de chunks por shift_plan
    "tz_resync",      # cron_tasks.py: serializar resync de _tz_offset_snapshot
    "anchor_recovery",  # reservado para futuros recovery flows
}


# [P1-HIST-AUDIT-7 · 2026-05-09] Lock advisory PER-USER para serializar
# mutators del Historial (restore/delete/rename). Diseño hermano de
# `acquire_meal_plan_advisory_lock` pero con namespace de keys por
# user_id, no por plan_id — los tres endpoints operan sobre LISTAS de
# planes del mismo user (e.g., restore necesita resolver `target` =
# fila más reciente, que cambia si dos restores corren a la vez).
#
# Bug original (audit historial 2026-05-08):
#   Doble-click "Reactivar" desde dos tabs sobre planes distintos:
#   ambos endpoints resolvían el MISMO `target` (latest), ambos
#   sobreescribían — el último gana, el primero perdía silenciosamente.
#   Mismo riesgo en restore vs delete vs rename simultáneo del mismo
#   user (race en el SELECT inicial vs el UPDATE/DELETE final).
#
# Sin esto, los tres endpoints son no-op tras el primer SELECT pero
# read-modify-write a nivel global del Historial NO es atómico.
# `pg_advisory_xact_lock` per-user serializa los tres mutators a nivel
# de toda la lista del Historial del usuario; otros users no se
# bloquean entre sí.
_USER_HISTORY_LOCK_PURPOSE = "history_mutator"


def acquire_user_history_advisory_lock(cursor, user_id) -> None:
    """[P1-HIST-AUDIT-7 · 2026-05-09] Adquiere lock advisory transaccional
    PER-USER para serializar mutators del Historial.

    `cursor` debe estar dentro de una transacción abierta; el lock se
    libera al COMMIT/ROLLBACK. Si otro caller con el mismo user_id ya
    posee el lock (otro restore/delete/rename del mismo Historial),
    esta llamada bloquea hasta liberación.

    Args:
        cursor: psycopg cursor activo dentro de transacción.
        user_id: UUID del user (str o UUID).
    """
    key = f"user:{_USER_HISTORY_LOCK_PURPOSE}:{user_id}"
    cursor.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s::text, 0))",
        (key,),
    )


def acquire_meal_plan_advisory_lock(cursor, meal_plan_id, purpose: str = "general") -> None:
    """[P1-5] Adquiere advisory lock transaccional por (meal_plan_id, purpose).

    `cursor` debe estar dentro de una transacción abierta; el lock se libera
    cuando la transacción termina (commit o rollback). Si otro caller con la
    misma key ya posee el lock, esta llamada bloquea hasta liberación.

    Args:
        cursor: psycopg cursor activo dentro de transacción.
        meal_plan_id: UUID del meal_plan (str o UUID).
        purpose: namespace del lock (ver `_MEAL_PLAN_LOCK_PURPOSES`). Strings
            arbitrarios funcionan pero un valor desconocido emite un warning para
            captar typos en code review (no es bloqueante: el lock se adquiere
            igual y la coherencia depende de que todos los call sites usen la
            misma cadena).
    """
    if purpose not in _MEAL_PLAN_LOCK_PURPOSES:
        logger.warning(
            f"[P1-5] acquire_meal_plan_advisory_lock recibió purpose desconocido "
            f"{purpose!r}. ¿Typo? Valores conocidos: {sorted(_MEAL_PLAN_LOCK_PURPOSES)}"
        )
    key = f"meal_plan:{purpose}:{meal_plan_id}"
    cursor.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended(%s::text, 0))",
        (key,),
    )


def set_meal_plan_for_update_timeouts(cursor) -> None:
    """[P1-LOCK-1 · 2026-05-10] Setea SET LOCAL lock_timeout +
    statement_timeout antes de un `SELECT … FOR UPDATE` sobre `meal_plans`.

    Bug observado (auditoría 2026-05-10, logs Postgres prod):
        Una transacción esperó 92.5s un `AccessExclusiveLock` sobre tuple
        de `meal_plans` antes de ser cancelada por `statement_timeout`
        de la sesión. Sin `lock_timeout` local explícito, el default
        Postgres (lock_timeout=0 = infinito) deja al caller esperando
        indefinidamente; el `statement_timeout` de Supavisor (60s+) es
        la única red de seguridad y opera sobre la tx completa, no sobre
        el lock specific. Resultado: spinner de UX colgado por minutos
        cuando dos callers contienden por el mismo plan.

    Patrón de uso:
        with conn.cursor() as cur:
            set_meal_plan_for_update_timeouts(cur)
            cur.execute("SELECT … FROM meal_plans WHERE … FOR UPDATE", ...)
            ...

    Knobs (auto-registrados en `_KNOBS_REGISTRY` vía `_env_int`):
        MEALFIT_PLAN_FOR_UPDATE_LOCK_TIMEOUT_MS  (default 5000)
            Máximo tiempo esperando adquirir el row lock. Si excede, psycopg
            propaga `LockNotAvailable` (SQLSTATE 55P03); el caller debe
            decidir si re-intentar (idempotente) o fallar al usuario. 5s =
            margen sobre worst-case T1 worker merge típico (~2s) y < umbral
            de UX para spinner de `/shift-plan` (10s aceptable).
        MEALFIT_PLAN_FOR_UPDATE_STMT_TIMEOUT_MS  (default 30000)
            Máximo tiempo de toda la transacción. Último gate: si la tx
            mantiene el lock más de 30s tras adquirirlo, hay I/O lento
            (LLM/HTTP) DENTRO del bloque transaccional — síntoma a
            investigar (follow-up pendiente: auditar que `/shift-plan` y
            workers no hagan llamadas externas mientras sostienen el lock).
        MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS  (default 60000)
            [P0-PERSIST-TXN-IDLE · 2026-07-10] Override per-tx del
            idle_in_transaction_session_timeout de sesión (15s,
            P1-DB-STMT-TIMEOUT en db_core). Los mutators bajo FOR UPDATE
            son CPU-only por contrato (P2-MUTATOR-PURITY) pero un stretch
            CPU legítimo (T1 merge + finalize parity ~10-20s) cuenta como
            "idle" para el server: sin este override la sesión moría a los
            15s (chunk 4 del plan 72c8b965, 2026-07-06). El propio diseño
            de db_core bendice los `SET LOCAL` per-tx como override. Valor
            0 ⇒ NO se emite el SET (en Postgres 0 = deshabilitado/infinito
            — nunca queremos eso; queda vivo el default de sesión de 15s).

    Best-effort: si `SET LOCAL` falla (Postgres viejo, permisos), log
    debug y la transacción continúa sin timeouts locales. NO propaga la
    excepción — el comportamiento previo (esperar indefinidamente) es
    estrictamente menos seguro que cualquier timeout configurado, pero
    debe ser indistinguible del fallo de configuración para no romper la
    funcionalidad existente.
    """
    # Import lazy para evitar ciclo db_plans -> knobs -> ... durante module init.
    from knobs import _env_int

    lock_to_ms = _env_int("MEALFIT_PLAN_FOR_UPDATE_LOCK_TIMEOUT_MS", 5000)
    stmt_to_ms = _env_int("MEALFIT_PLAN_FOR_UPDATE_STMT_TIMEOUT_MS", 30000)
    idle_txn_ms = _env_int("MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS", 60000)
    try:
        cursor.execute(f"SET LOCAL lock_timeout = '{int(lock_to_ms)}ms'")
        cursor.execute(f"SET LOCAL statement_timeout = '{int(stmt_to_ms)}ms'")
        if idle_txn_ms > 0:
            cursor.execute(
                f"SET LOCAL idle_in_transaction_session_timeout = '{int(idle_txn_ms)}ms'"
            )
    except Exception as e:
        logger.debug(
            f"[P1-LOCK-1] No se pudo setear lock_timeout/statement_timeout "
            f"en meal_plans FOR UPDATE: {e}"
        )


def update_plan_data_atomic(
    plan_id: str,
    mutator,
    lock_timeout_ms: Optional[int] = None,
    *,
    user_id: Optional[str] = None,
) -> dict:
    """[P0-2] Read-Modify-Write atómico de meal_plans.plan_data.

    Aplica `mutator(plan_data)` (que MUTA o RETORNA un nuevo dict) dentro de un
    `SELECT … FOR UPDATE` que serializa concurrentes contra la misma fila. Es la
    forma segura de tocar keys acumulativas como _consecutive_zero_log_chunks,
    _last_chunk_learning, _recent_chunk_lessons, _critical_lessons_permanent y
    _recovery_exhausted_chunks: dos chunks que escriben simultáneamente NO se
    sobrescriben — el segundo ve el plan_data ya actualizado por el primero.

    [P2-PROD-AUDIT-FOLLOWUP · 2026-05-28] CONTRATO DEL MUTATOR (tooltip-anchor
    P2-MUTATOR-PURITY): el `mutator` corre DENTRO del `SELECT … FOR UPDATE`,
    reteniendo el row-lock + una conexión del pool sync (max 60) durante toda
    su ejecución. DEBE ser una transformación pura, CPU-only, sobre el dict:
    NADA de IO, llamadas LLM, sleeps, ni —crítico— re-entrada al pool
    (`with connection_pool.connection()`) ni otra llamada DB. Un mutator que
    haga IO o re-entre al pool retiene el slot + el lock por su duración; con
    suficientes callers re-entrantes concurrentes se agota el pool sync →
    deadlock/starvation. Si necesitas datos externos, resuélvelos ANTES de
    llamar a `update_plan_data_atomic` y pásalos por closure al mutator.

    Args:
        plan_id: UUID del meal_plan a modificar.
        mutator: callable(plan_data: dict) -> dict | None. Si retorna un dict
            nuevo se persiste; si retorna None se persiste el dict original
            (asume mutación in-place). Si retorna `False` literal, se aborta el
            UPDATE (caller decidió que no había cambios).
        lock_timeout_ms: timeout para adquirir el FOR UPDATE; default toma
            CHUNK_LEARNING_LOCK_TIMEOUT_MS de constants. Si excede el timeout,
            la función propaga la excepción de psycopg para que el caller
            decida re-encolar / saltar.
        user_id: [P2-OPEN-1 · 2026-05-11] Si presente, el SELECT y UPDATE
            incluyen `AND user_id = %s`. Defense-in-depth contra invariante
            I2 (CLAUDE.md): aunque todos los callers actuales resuelven
            ownership upstream (plan_id viene de plan_chunk_queue claimeado,
            que ya filtró user_id), un refactor futuro que reordene la
            resolución puede abrir IDOR silente sin este filtro local.
            Patrón espejo de `update_meal_plan_data` (P1-NEW-3). Si `None`,
            la función emite `logger.warning("[I2-MISS] ...")` con el caller
            inferido del stack para que SRE detecte callers no-migrados —
            el UPDATE procede igual (back-compat con callers viejos).

    Returns:
        El dict plan_data resultante tras la mutación. Si la fila no existe
        (o no pertenece al user_id si fue provisto), retorna {} y NO ejecuta
        UPDATE.
    """
    if not connection_pool:
        raise RuntimeError("db connection_pool is not available for atomic plan_data update.")

    if lock_timeout_ms is None:
        from constants import CHUNK_LEARNING_LOCK_TIMEOUT_MS
        lock_timeout_ms = int(CHUNK_LEARNING_LOCK_TIMEOUT_MS)

    # [P2-OPEN-1 · 2026-05-11] Detección de callers no-migrados. Cuando un
    # caller omite `user_id`, loggeamos a WARNING con el call frame inmediato
    # (filename:lineno + nombre de la función) para que SRE vea la migración
    # pendiente en logs sin tener que grep el código. NO bloqueamos el flujo
    # — el UPDATE procede sin el filtro (back-compat). Test parser-based
    # `test_p2_open_1_update_plan_data_atomic_user_id` enforza que TODOS los
    # callers conocidos pasen `user_id`, así que en estado verde el log no
    # debería aparecer en prod salvo callers nuevos no auditados.
    if user_id is None:
        try:
            import inspect as _inspect_p2o1
            _frame = _inspect_p2o1.stack()[1]
            _caller_loc = f"{_frame.filename}:{_frame.lineno} ({_frame.function})"
        except Exception:
            _caller_loc = "<stack unavailable>"
        logger.warning(
            f"[I2-MISS] update_plan_data_atomic invocado sin `user_id` desde "
            f"{_caller_loc}. Defense-in-depth I2 (CLAUDE.md) se omite — UPDATE "
            f"procede sin filtro AND user_id = %s. Migrar el caller pasando "
            f"`user_id=<owner>` (kwarg) cierra el gap. Plan_id={plan_id}."
        )

    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    with connection_pool.connection() as conn:
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cursor:
                try:
                    cursor.execute(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms'")  # pyright: ignore[reportArgumentType, reportCallIssue]  # psycopg LiteralString FP
                except Exception as set_err:
                    logger.debug(f"[P0-2] No se pudo setear lock_timeout en update_plan_data_atomic: {set_err}")
                # [P0-PERSIST-TXN-IDLE-ATOMIC · 2026-07-10] Los mutators de esta función corren
                # DENTRO del FOR UPDATE y algunos (regen-day: micros recompute + panel + rebuild
                # inline de listas) tienen tramos CPU-bound largos SIN queries → la conexión queda
                # idle-in-transaction y el SET de sesión del pool (MEALFIT_DB_IDLE_IN_TXN_TIMEOUT_MS
                # = 15s, db_core) la mata: `terminating connection due to idle-in-transaction
                # timeout` → HTTP 500 con TODO el trabajo del update perdido (incidente en vivo
                # 2026-07-10 15:19 en /regenerate-day, misma clase que P0-PERSIST-TXN-IDLE del
                # INSERT). Mismo remedio que el T1 del chunk (set_meal_plan_for_update_timeouts):
                # SET LOCAL con presupuesto amplio SOLO para esta transacción (60s default, knob
                # compartido MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS).
                try:
                    _idle_ms_atomic = _env_int("MEALFIT_PLAN_FOR_UPDATE_IDLE_TXN_TIMEOUT_MS", 60000)
                    if _idle_ms_atomic > 0:
                        cursor.execute(f"SET LOCAL idle_in_transaction_session_timeout = '{int(_idle_ms_atomic)}ms'")  # pyright: ignore[reportArgumentType, reportCallIssue]  # psycopg LiteralString FP
                except Exception as set_err:
                    logger.debug(f"[P0-PERSIST-TXN-IDLE-ATOMIC] no se pudo setear idle timeout: {set_err}")

                # [P2-OPEN-1] SELECT con filtro user_id si presente. Bajo
                # `FOR UPDATE` el row se locka; si user_id no matchea, no hay
                # fila → caemos al early return abajo.
                if user_id is not None:
                    cursor.execute(
                        "SELECT plan_data FROM meal_plans WHERE id = %s AND user_id = %s FOR UPDATE",
                        (plan_id, user_id),
                    )
                else:
                    cursor.execute(
                        "SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE",
                        (plan_id,),
                    )
                row = cursor.fetchone()
                if not row:
                    logger.warning(
                        f"[P0-2] update_plan_data_atomic: meal_plan {plan_id} no existe "
                        f"(probablemente cancelado por save_new_meal_plan_atomic, o "
                        f"user_id={user_id!r} no coincide con el owner). Skip."
                    )
                    return {}

                current = row["plan_data"] or {}
                if not isinstance(current, dict):
                    current = {}

                result = mutator(current)
                if result is False:
                    return current

                new_data = result if isinstance(result, dict) else current
                # [P2-OPEN-1] UPDATE con `AND user_id = %s` defense-in-depth.
                # Aunque ya verificamos ownership en el SELECT bajo FOR UPDATE
                # (mismo lock que cubre el UPDATE), el filtro repetido ancla
                # la invariante I2 para refactors futuros que separen el SELECT
                # del UPDATE.
                if user_id is not None:
                    cursor.execute(
                        "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s AND user_id = %s",
                        (Jsonb(new_data), plan_id, user_id),
                    )
                else:
                    cursor.execute(
                        "UPDATE meal_plans SET plan_data = %s::jsonb WHERE id = %s",
                        (Jsonb(new_data), plan_id),
                    )
                return new_data


def _ensure_grocery_start_date(plan_data: dict) -> dict:
    """[P0-1-RECOVERY/C] Garantiza que plan_data tenga grocery_start_date persistido.

    El pipeline LLM no siempre lo incluye; cuando falta, downstream:
      - El cron de recovery lo necesitaba para calcular si el plan sigue activo
        (ahora cubierto con COALESCE → created_at, pero éste es el fix at the source).
      - routers/plans.py:175 retorna 'Falta fecha de inicio.' y el shift-renewal falla.
      - cron_tasks.py:5118 hace backfill on-demand al procesar el primer chunk, pero
        si el chunk falla antes de llegar al worker el plan queda con NULL para siempre.

    Esta función mute el dict in-place (también lo retorna) y asume que el caller
    está en el momento de inserción. Si plan_data ya tiene un valor truthy lo respeta;
    si no, persiste hoy en formato ISO date (YYYY-MM-DD) en UTC.
    """
    if not isinstance(plan_data, dict):
        return plan_data
    existing = plan_data.get("grocery_start_date")
    if existing:
        return plan_data
    from datetime import datetime, timezone
    # [GROCERY-START-DATE-TIMESTAMP-FIX 2026-05-06] Persistir timestamp ISO
    # completo (no solo date). Antes guardábamos `datetime.now(utc).date().isoformat()`
    # = "2026-05-06" sin TZ. JavaScript `new Date("2026-05-06")` interpreta eso
    # como UTC midnight, convertido a TZ del usuario produce un día ANTES
    # (en TZ negativas como UTC-4): local 5-may 20:00 → setHours = 5-may 00:00 →
    # `daysSinceCreation = 1` → shift dispara → recorta día 1 del plan recién
    # generado, dejándolo con N-1 días. Persistir el timestamp completo deja
    # que el frontend convierta a local correctamente sin off-by-one.
    plan_data["grocery_start_date"] = datetime.now(timezone.utc).isoformat()
    logger.info(
        "[P0-1-RECOVERY/C] grocery_start_date ausente al insertar meal_plan; "
        f"persistido fallback={plan_data['grocery_start_date']} (timestamp ISO completo)."
    )
    return plan_data


def _inherit_lifetime_lessons_from_prior_plan(cursor, user_id: str) -> Optional[Dict[str, Any]]:
    """[P0-1] Lee lecciones del plan más reciente del usuario para sembrar el plan nuevo.

    Devuelve {"history": [...], "summary": {...}, "from_plan_id": "<uuid>"} o None.
    Filtra el historial por LIFETIME_LESSONS_WINDOW_DAYS (60d por defecto) y recomputa
    el summary si el filtro descartó entradas. Si `cursor` es None, abre su propia
    consulta vía execute_sql_query (path no-transaccional, p.ej. save_new_meal_plan_robust).
    Cualquier excepción se loguea y retorna None — la herencia es best-effort, no
    debe bloquear el INSERT del plan nuevo.
    """
    try:
        from constants import LIFETIME_LESSONS_WINDOW_DAYS, safe_fromisoformat
        from datetime import datetime, timezone, timedelta

        if cursor is not None:
            cursor.execute(
                "SELECT id, plan_data FROM meal_plans "
                "WHERE user_id = %s "
                "ORDER BY created_at DESC "
                "LIMIT 1",
                (user_id,),
            )
            row = cursor.fetchone()
        else:
            row = execute_sql_query(
                "SELECT id, plan_data FROM meal_plans "
                "WHERE user_id = %s "
                "ORDER BY created_at DESC "
                "LIMIT 1",
                (user_id,),
                fetch_one=True,
            )
        if not row:
            return None

        prior_plan_data = row.get("plan_data") if isinstance(row, dict) else None
        if not isinstance(prior_plan_data, dict):
            return None

        history_raw = prior_plan_data.get("_lifetime_lessons_history")
        summary_raw = prior_plan_data.get("_lifetime_lessons_summary")
        if not isinstance(history_raw, list):
            history_raw = []
        if not isinstance(summary_raw, dict):
            summary_raw = {}

        if not history_raw and not summary_raw:
            return None

        cutoff = (datetime.now(timezone.utc) - timedelta(days=LIFETIME_LESSONS_WINDOW_DAYS))
        history_filtered: List[Dict[str, Any]] = []
        for lesson in history_raw:
            if not isinstance(lesson, dict):
                continue
            ts = lesson.get("timestamp")
            if not ts:
                continue
            try:
                ts_dt = safe_fromisoformat(ts) if isinstance(ts, str) else ts
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                if ts_dt >= cutoff:
                    history_filtered.append(lesson)
            except Exception:
                continue

        if not history_filtered and not summary_raw:
            return None

        # [P1-5] Recomputamos siempre, no solo cuando se filtró historial. Razón:
        # `summary_raw` puede venir de una versión anterior del schema sin los campos
        # `top_repeated_meal_names` y `permanent_meal_blocklist`. Forzar el recompute
        # garantiza que el plan nuevo siempre reciba un summary completo, independiente
        # de cuándo fue generado el plan anterior. El costo es despreciable (dict ops
        # sobre history filtered de ≤60 entradas máximo).
        # [P1-4] Aplicamos el mismo decay temporal que el worker (cron_tasks._chunk_worker
        # ~14210). Sin esto, la herencia cross-plan trata como iguales lecciones del
        # primer chunk del plan previo (~8 semanas atrás) y del último (~1 semana
        # atrás): el plan nuevo arranca con un summary cuyo ranking es arbitrario
        # (por inserción del set) en vez de priorizar recencia.
        from cron_tasks import (
            compute_lifetime_lesson_weight as _p14_weight,
            _derive_learning_provenance as _p17_provenance,
            LIFETIME_LESSON_PROXY_WEIGHT_FACTOR as _P17_FACTOR,
        )
        from constants import LIFETIME_LESSON_MIN_WEIGHT as _P14_MIN_W
        from datetime import datetime as _p14_datetime
        _p14_now = _p14_datetime.now(timezone.utc)

        rej_weights: dict = {}
        base_weights: dict = {}
        meal_chunk_counts: dict = {}  # [P1-5] meal_name → set(chunk)
        meal_weights: dict = {}       # [P1-4] meal_name → suma de pesos (ranking por recencia)
        total_rej = 0
        total_alg = 0
        # [P1-7] Tracking de provenance para persistir ratio en el summary heredado.
        _p17_user_logs_count = 0
        _p17_proxy_count = 0
        for l in history_filtered:
            total_rej += int(l.get("rejection_violations") or 0)
            total_alg += int(l.get("allergy_violations") or 0)
            _w = _p14_weight(l, now=_p14_now)
            if _w < float(_P14_MIN_W):
                # Lección demasiado vieja para influir en el ranking heredado.
                # `total_rej` y `total_alg` SÍ se acumulan independientemente del
                # decay (representan totales históricos brutos para telemetría).
                continue
            # [P1-7] Aplicar factor por provenance simétricamente al worker recompute.
            # Sin esto, el plan nuevo heredaba un summary donde lessons de proxy/synthesis
            # del plan previo competían igualadas con user_logs por los caps de top-N,
            # diluyendo señal real al inicializar el aprendizaje cross-plan.
            _prov = _p17_provenance(l)
            if _prov == "user_logs":
                _p17_user_logs_count += 1
            else:
                _p17_proxy_count += 1
            _w *= 1.0 if _prov == "user_logs" else float(_P17_FACTOR)
            _ch = l.get("chunk")
            for rj in (l.get("rejected_meals_that_reappeared") or []):
                rej_weights[str(rj)] = rej_weights.get(str(rj), 0.0) + _w
            for rb_entry in (l.get("repeated_bases") or []):
                if isinstance(rb_entry, dict):
                    for b in (rb_entry.get("bases") or []):
                        base_weights[str(b)] = base_weights.get(str(b), 0.0) + _w
                else:
                    base_weights[str(rb_entry)] = base_weights.get(str(rb_entry), 0.0) + _w
            for rm in (l.get("repeated_meal_names") or []):
                if rm:
                    _mk = str(rm)
                    meal_chunk_counts.setdefault(_mk, set()).add(_ch)
                    meal_weights[_mk] = meal_weights.get(_mk, 0.0) + _w

        # [P1-4] Ranking por peso descendente (recencia primero), luego truncar.
        all_repeated_meals = sorted(
            meal_weights.keys(),
            key=lambda m: (-meal_weights[m], -len(meal_chunk_counts[m])),
        )
        # [P1-7] Ratio proxy persistido para telemetría cross-plan.
        _p17_total = _p17_user_logs_count + _p17_proxy_count
        summary_filtered = {
            "total_rejection_violations": total_rej,
            "total_allergy_violations": total_alg,
            "top_rejection_hits": [
                k for k, _ in sorted(rej_weights.items(), key=lambda kv: -kv[1])
            ][:20],
            "top_repeated_bases": [
                k for k, _ in sorted(base_weights.items(), key=lambda kv: -kv[1])
            ][:20],
            # [P1-5] Mantener consistencia con el recompute en cron_tasks.py.
            "top_repeated_meal_names": all_repeated_meals[:30],
            "permanent_meal_blocklist": sorted(
                [m for m in all_repeated_meals if len(meal_chunk_counts[m]) >= 2],
                key=lambda m: -meal_weights[m],
            )[:50],
            "_lifetime_window_days": LIFETIME_LESSONS_WINDOW_DAYS,
            "_lifetime_proxy_ratio": (
                round(_p17_proxy_count / _p17_total, 3) if _p17_total > 0 else 0.0
            ),
            "_lifetime_user_logs_count": _p17_user_logs_count,
            "_lifetime_proxy_count": _p17_proxy_count,
        }

        from_plan_id = str(row.get("id")) if row.get("id") else None
        return {
            "history": history_filtered,
            "summary": summary_filtered,
            "from_plan_id": from_plan_id,
        }
    except Exception as e:
        logger.warning(f"⚠️ [P0-1/INHERIT] Error leyendo lecciones del plan previo de {user_id}: {e}")
        return None


def _apply_inherited_lifetime_lessons(user_id: str, insert_data: dict, cursor=None) -> None:
    """[P0-1] Inyecta lecciones heredadas en `insert_data['plan_data']` si faltan.

    Mutación in-place. Best-effort: si algo falla o no hay plan previo, deja el
    plan_data como estaba y los chunks empezarán con aprendizaje vacío.
    """
    plan_data_dict = insert_data.get("plan_data")
    if not isinstance(plan_data_dict, dict):
        return
    already_has_history = bool(plan_data_dict.get("_lifetime_lessons_history"))
    already_has_summary = bool(plan_data_dict.get("_lifetime_lessons_summary"))
    if already_has_history and already_has_summary:
        return
    inherited = _inherit_lifetime_lessons_from_prior_plan(cursor, user_id)
    if not inherited:
        return
    if not already_has_history and inherited.get("history"):
        plan_data_dict["_lifetime_lessons_history"] = inherited["history"]
    if not already_has_summary and inherited.get("summary"):
        plan_data_dict["_lifetime_lessons_summary"] = inherited["summary"]
    if inherited.get("from_plan_id"):
        plan_data_dict["_lifetime_lessons_inherited_from"] = inherited["from_plan_id"]
    logger.info(
        f"🧠 [P0-1/INHERIT] {user_id}: heredadas "
        f"{len(inherited.get('history') or [])} lecciones "
        f"(bases={len((inherited.get('summary') or {}).get('top_repeated_bases') or [])}, "
        f"hits={len((inherited.get('summary') or {}).get('top_rejection_hits') or [])}) "
        f"de plan {inherited.get('from_plan_id')}."
    )


def _finalize_plan_data_for_insert(data: dict, *, surface: str = "pre-INSERT") -> None:
    """[P0-PERSIST-TXN-IDLE · 2026-07-10] Pases de normalización/finalize de
    `plan_data` previos a CUALQUIER INSERT de meal_plans (mutación in-place).

    Vivían inline en `_build_meal_plan_insert_sql`; extraídos para poder
    ejecutarlos ANTES de abrir la transacción del path atomic: son CPU-bound
    (~10-25s en planes reales — fuzzy matching + coherence stack + motor de
    macros all-4) y corriendo dentro del BEGIN dejaban la conexión
    idle-in-transaction hasta que el SET de sesión de db_core
    (`MEALFIT_DB_IDLE_IN_TXN_TIMEOUT_MS`=15s, P1-DB-STMT-TIMEOUT) la mataba:
    `terminating connection due to idle-in-transaction timeout` → plan generado
    PERDIDO tras 13 min de pipeline (forensic corr=d2bc0bcc 2026-07-10; también
    Jul 09 sync ×2 user 99a02318 — el bug precede al band closer, cada pase
    añadido desde 2026-06-28 acercaba la ventana al umbral).

    [P0-BAND-PRE-REVIEW · 2026-07-10] `surface` etiqueta los logs con el punto de
    ejecución real: además del INSERT, el chain corre en la cola de assemble
    (pre-shopping/pre-review, vía `apply_plan_quality_finalize_chain`) y en el
    merge T1 del chunk worker (semanas 2+). Kw-only con default → los callers y
    monkeypatches existentes (1 arg posicional) siguen intactos.

    Idempotente (cada pase interno lo es) y fail-safe (nunca bloquea el INSERT).
    """
    # [P0-1-RECOVERY/C] Defensa centralizada: cualquier path que use este helper para
    # insertar a meal_plans tendrá grocery_start_date garantizado. Evita reintroducir
    # el bug donde el pipeline LLM omitía el campo y sólo el backfill en runtime
    # (cron_tasks.py:5118) lo poblaba a costa de chunks que fallaban antes.
    if "plan_data" in data:
        data["plan_data"] = _ensure_grocery_start_date(data["plan_data"])
        # [P1-COHERENCE-FINALIZE · 2026-06-28] Escudo defensivo: aplica el coherence stack (slice-grams/leaf-cap/quantize)
        # a `days` antes de CUALQUIER INSERT — cubre los paths que saltan assemble_plan_node (partial/rechazado-pero-
        # entregado/SSE-fallback) y persistían "1¼ lonjas de queso" crudo. Idempotente (no-op si assemble ya corrió →
        # band 1.00 intacto), fail-safe (nunca bloquea el INSERT). Import LAZY: graph_orchestrator importa db_plans (ciclo).
        try:
            _pd = data["plan_data"]
            if isinstance(_pd, dict) and isinstance(_pd.get("days"), list):
                from graph_orchestrator import finalize_plan_data_coherence as _fpc
                # [P1-CHUNK-FINALIZE-PARITY · 2026-07-07] deriva target de grasa del plan → el shield
                # pre-INSERT corre relevel + cheese-final en planes que saltan assemble (partial/
                # rechazado-pero-entregado/SSE-fallback), no solo el slice/quantize.
                # [P0-BAND-PRE-REVIEW · 2026-07-10] deriva TAMBIÉN main_goal + target_macros (los 3
                # numéricos) del propio plan_data — paridad P1-CHUNK-GAINMUSCLE-PARITY: antes solo el
                # chunk T1 los pasaba a mano; ahora el chunk delega en este chain y los paths que
                # saltan assemble ganan el refill de bulk que nunca tuvieron.
                _tf_ins = None
                _tm_ins = None
                try:
                    import re as _re_ins
                    _mac_ins = _pd.get("macros") or {}
                    _vals_ins = {}
                    for _ks_i, _kd_i in (("protein", "protein_g"), ("carbs", "carbs_g"), ("fats", "fats_g")):
                        _mm_ins = _re_ins.search(r"(\d+(?:\.\d+)?)", str(_mac_ins.get(_ks_i)))
                        _vals_ins[_kd_i] = float(_mm_ins.group(1)) if _mm_ins else None
                    _tf_ins = _vals_ins.get("fats_g")
                    if all(_v is not None for _v in _vals_ins.values()):
                        _tm_ins = _vals_ins
                except Exception:
                    _tf_ins = None
                    _tm_ins = None
                _n, _summ = _fpc(_pd["days"], target_fats=_tf_ins,
                                 main_goal=_pd.get("main_goal"), target_macros=_tm_ins)
                if _n:
                    logger.info(f"🧩 [P1-COHERENCE-FINALIZE] {surface} aplicó coherencia a un plan no-finalizado ({_summ}).")
                # [P1-PROTEIN-BAND-POST-FINALIZE · 2026-07-09] El truth-up de _fpc recomputa la proteína HONESTA
                # → puede re-exponer drift de proteína (día bajo el piso / sobre el techo) que ningún closer de
                # assemble re-encuadra. Re-escala porciones proteína-dominantes EXISTENTES (sin ingredientes
                # nuevos → sin riesgo de alérgeno). Corre ANTES de _csd/_rbs para que el band re-medido refleje
                # el estado corregido. Fail-safe. Import LAZY (mismo ciclo que finalize).
                try:
                    from graph_orchestrator import reconcile_protein_band_post_finalize as _rpb
                    _rpb(_pd)
                except Exception as _rpb_e:
                    logger.debug(f"[P1-PROTEIN-BAND-POST-FINALIZE] pre-INSERT no-op: {type(_rpb_e).__name__}: {_rpb_e}")
                # [P1-RECIPE-VISIBLE-DEFECTS · 2026-07-11] re-resuelve el placeholder 'ingrediente
                # alternativo' del sanitizador con la proteína REAL de ingredients (capturas vivas
                # del owner: "mezcla la ingrediente alternativo con..."). Corre ANTES del sweep de
                # paridad: con la mención restaurada, el sweep ya no pega su línea mecánica encima.
                try:
                    from graph_orchestrator import resolve_alt_ingredient_placeholders as _raip
                    _raip(_pd.get("days"))
                except Exception as _raip_e:
                    logger.debug(f"[P1-RECIPE-VISIBLE-DEFECTS] pre-INSERT no-op: {type(_raip_e).__name__}: {_raip_e}")
                # [P0-2-PROTEIN-STEP-PARITY · 2026-07-10] (recipe plausibility roadmap) sweep universal:
                # cierra proteínas físicamente presentes en ingredients (añadidas por CUALQUIER closer
                # aguas arriba, incl. el pase de proteína de arriba) que ningún paso menciona — evidencia
                # visual "40g de camarones cocido" sin rastro en Mise en Place/Fuego/Montaje. Corre tras el
                # pase de proteína (barre su estado final). Fail-safe. Import LAZY (mismo ciclo que finalize).
                try:
                    from graph_orchestrator import ensure_protein_step_parity as _epsp
                    _epsp(_pd)
                except Exception as _epsp_e:
                    logger.debug(f"[P0-2-PROTEIN-STEP-PARITY] pre-INSERT no-op: {type(_epsp_e).__name__}: {_epsp_e}")
                # [P0-1-FINAL-BAND-CLOSER · 2026-07-10] el pase de proteína de arriba SOLO re-encuadra
                # proteína — cuando el macro fuera de banda es carbs/fats/kcal nada lo corregía (forensic
                # corr=d57ffe04). Motor SSOT all-4 ya testeado (apply_update_macro_engine), solo wiring.
                # Corre DESPUÉS del pase de proteína (protein ya estable) y ANTES del re-check de banda.
                try:
                    from graph_orchestrator import reconcile_all_macros_band_post_finalize as _ramb
                    _ramb(_pd)
                except Exception as _ramb_e:
                    logger.debug(f"[P0-1-FINAL-BAND-CLOSER] pre-INSERT no-op: {type(_ramb_e).__name__}: {_ramb_e}")
                # [P1-3-POLISH-REFIRE · 2026-07-10] (recipe plausibility roadmap) el countable-polish de _fpc
                # (arriba) corrió ANTES de los dos pases de reconciliación de banda de arriba, que mutan
                # cantidades de ingredientes (rebalance/refine 5g) SIN re-pulir el display — plan 564d6e4e
                # (banda 1.00) salió con "1.25 cdas de mantequilla de maní", "0.5 toronja". Re-fire tras la
                # ÚLTIMA mutación real de cantidades. Fail-safe. Import LAZY (mismo ciclo que finalize).
                try:
                    from graph_orchestrator import refire_display_polish_post_finalize as _rdp
                    _rdp(_pd)
                except Exception as _rdp_e:
                    logger.debug(f"[P1-3-POLISH-REFIRE] pre-INSERT no-op: {type(_rdp_e).__name__}: {_rdp_e}")
                # [P2-1-CONDIMENT-PORTION-SANITY · 2026-07-10] (recipe plausibility roadmap) cap por-
                # porción de ajo/cebolla — evidencia "7 dientes de ajo" para 1 persona. Fail-safe.
                try:
                    from graph_orchestrator import cap_condiments_per_portion as _ccpp
                    _ccpp(_pd)
                except Exception as _ccpp_e:
                    logger.debug(f"[P2-1-CONDIMENT-PORTION-SANITY] pre-INSERT no-op: {type(_ccpp_e).__name__}: {_ccpp_e}")
                # [P2-2-BIGFRUIT-COUNT-RECONCILE · 2026-07-10] (recipe plausibility roadmap) anota
                # fracción de la fruta completa en líneas "1 lechosa (200g)" — display-only. Fail-safe.
                try:
                    from graph_orchestrator import annotate_bigfruit_fractional_hint as _abfh
                    _abfh(_pd)
                except Exception as _abfh_e:
                    logger.debug(f"[P2-2-BIGFRUIT-COUNT-RECONCILE] pre-INSERT no-op: {type(_abfh_e).__name__}: {_abfh_e}")
                # [P3-2-INGREDIENT-COUNT-AGREEMENT · 2026-07-10] (recipe plausibility roadmap) "1 naranjas"
                # → "1 naranja" — concordancia número-sustantivo en líneas de ingrediente. Fail-safe.
                try:
                    from graph_orchestrator import fix_ingredient_count_agreement as _fica
                    _fica(_pd)
                except Exception as _fica_e:
                    logger.debug(f"[P3-2-INGREDIENT-COUNT-AGREEMENT] pre-INSERT no-op: {type(_fica_e).__name__}: {_fica_e}")
                # [P0-1-PAIRING-PLAUSIBILITY-GATE · 2026-07-10] (recipe plausibility roadmap) fase 1
                # warn-only: detecta combinaciones implausibles (mantequilla de maní + tubérculo hervido/
                # gajos cítricos/queso salado) sobre el estado ENTREGADO. Telemetría pura, no muta el plan
                # todavía. Fail-safe. Import LAZY (mismo ciclo que finalize).
                try:
                    from graph_orchestrator import detect_pairing_plausibility_violations as _dppv
                    _dppv(_pd)
                except Exception as _dppv_e:
                    logger.debug(f"[P0-1-PAIRING-PLAUSIBILITY-GATE] pre-INSERT no-op: {type(_dppv_e).__name__}: {_dppv_e}")
                # [P2-3-BATCH-ARITHMETIC-CHECK · 2026-07-10] (recipe plausibility roadmap) warn-only:
                # "forma 6 tortitas ... sirve 3" — batch de cocción que no coincide con lo servido.
                # Telemetría pura, no muta el plan. Fail-safe.
                try:
                    from graph_orchestrator import detect_batch_arithmetic_mismatch as _dbam
                    _dbam(_pd)
                except Exception as _dbam_e:
                    logger.debug(f"[P2-3-BATCH-ARITHMETIC-CHECK] pre-INSERT no-op: {type(_dbam_e).__name__}: {_dbam_e}")
                # [P1-BAND-DEGRADED-STALE-CLEAR · 2026-07-08] El gate de banda del pipeline marcó _quality_degraded
                # ANTES de este finalize (que acaba de recortar grasas/re-truthear macros) → un flag low_band_* puede
                # ser un FALSO POSITIVO de timing. Re-evalúa sobre el estado ENTREGADO y limpia el banner si ya está
                # en banda. CLEAR-ONLY (nunca marca), fail-safe. Import LAZY (mismo ciclo que finalize).
                try:
                    from graph_orchestrator import clear_stale_low_band_degraded as _csd
                    _csd(_pd)
                except Exception as _csd_e:
                    logger.debug(f"[P1-BAND-DEGRADED-STALE-CLEAR] pre-INSERT no-op: {type(_csd_e).__name__}: {_csd_e}")
                # [P1-BAND-SCORE-POST-FINALIZE · 2026-07-08] La métrica `clinical_band_score` persistida
                # hasta acá era la medida DENTRO del pipeline, pre-finalize (stale para TODO plan, no solo
                # los flagged degradado que _csd ya cubre). Refresca sobre el estado ENTREGADO — corre
                # SIEMPRE, independiente de si _csd limpió algo. CLEAR-ONLY del flag sigue siendo de _csd.
                try:
                    from graph_orchestrator import refresh_clinical_band_score_post_finalize as _rbs
                    _rbs(_pd)
                except Exception as _rbs_e:
                    logger.debug(f"[P1-BAND-SCORE-POST-FINALIZE] pre-INSERT no-op: {type(_rbs_e).__name__}: {_rbs_e}")
        except Exception as _fce:
            logger.warning(f"[P1-COHERENCE-FINALIZE] {surface} no-op: {type(_fce).__name__}: {_fce}")


def apply_plan_quality_finalize_chain(plan_data: dict, *, surface: str = "quality-chain") -> None:
    """[P0-BAND-PRE-REVIEW · 2026-07-10] Adapter público del chain de calidad/banda.

    Delegación pura a `_finalize_plan_data_for_insert({"plan_data": plan_data})` —
    el SSOT del ORDEN de pases vive allí (fpc → protein-band → step-parity →
    all-4-band-closer → polish-refire → condimentos → bigfruit → count-agreement →
    detectores pairing/batch → stale-clear → band-score refresh). Seguro porque
    `_ensure_grocery_start_date` muta in-place y retorna el MISMO dict (documentado
    en su docstring) → el caller conserva todas las mutaciones.

    Surfaces (además del INSERT, que lo invoca directo):
      1. Cola de `assemble_plan_node` ANTES de construir la lista de compras y del
         review → el gate de banda y el P2-BAND-SCORE-GATE miden el estado CERRADO
         (mata los retries por banda —que escalaban day_generator a PRO, driver #1
         del gasto LLM— y el banner falso "precisión de las calorías" que viajaba
         en el payload SSE pre-shield).
      2. Merge T1 del chunk worker (semanas 2+) → paridad: los pases del shield
         cubren el 100% de los días del plan, no solo el primer chunk.

    Idempotente y fail-safe (hereda ambas garantías del shield). Mutación in-place;
    retorna None. tooltip-anchor: P0-BAND-PRE-REVIEW
    """
    if not isinstance(plan_data, dict):
        return
    _finalize_plan_data_for_insert({"plan_data": plan_data}, surface=surface)


def _build_meal_plan_insert_sql(data: dict, with_returning: bool = False,
                                skip_plan_data_finalize: bool = False):
    """Construye la SQL de INSERT para meal_plans a partir de un dict.

    Retorna (sql, vals) listos para pasar a cursor.execute.

    [P0-PERSIST-TXN-IDLE · 2026-07-10] `skip_plan_data_finalize=True` lo pasa
    `save_new_meal_plan_atomic`, que ejecuta `_finalize_plan_data_for_insert`
    ANTES de abrir su transacción (los pases son CPU-bound ~10-25s y dentro del
    BEGIN disparaban el idle-in-transaction timeout de sesión → INSERT muerto,
    plan perdido). El default False conserva el escudo central para
    `save_new_meal_plan_robust` y cualquier call site futuro — esos construyen
    FUERA de transacción, donde el costo CPU es inocuo.
    """
    from psycopg.types.json import Jsonb
    if not skip_plan_data_finalize:
        _finalize_plan_data_for_insert(data)

    # [P0-1/CENTRAL] Último escudo para herencia cross-plan de lifetime lessons.
    # save_new_meal_plan_atomic y save_new_meal_plan_robust ya invocan
    # _apply_inherited_lifetime_lessons explícitamente (con el cursor óptimo:
    # transaccional para atomic, independiente para robust). Este bloque cubre:
    #   1) call sites futuros que inserten vía _build_meal_plan_insert_sql sin
    #      pasar por los helpers (única forma documentada de insertar plans en
    #      producción — verificado en audit P0-1).
    #   2) refactors que extraigan parte del flujo y olviden replicar la herencia.
    # _apply_inherited_lifetime_lessons es idempotente: si plan_data ya tiene
    # _lifetime_lessons_history Y _lifetime_lessons_summary, retorna sin trabajo.
    # Si entra al fallback Y rellena algo, emitimos warning con el user_id para
    # detectar el call site infractor sin romper el INSERT.
    _user_id = data.get("user_id")
    _plan_data_dict = data.get("plan_data") if isinstance(data.get("plan_data"), dict) else None
    if _user_id and _plan_data_dict is not None:
        _had_history_before = bool(_plan_data_dict.get("_lifetime_lessons_history"))
        _had_summary_before = bool(_plan_data_dict.get("_lifetime_lessons_summary"))
        if not (_had_history_before and _had_summary_before):
            _apply_inherited_lifetime_lessons(_user_id, data, cursor=None)
            _filled_history = (
                bool(_plan_data_dict.get("_lifetime_lessons_history"))
                and not _had_history_before
            )
            _filled_summary = (
                bool(_plan_data_dict.get("_lifetime_lessons_summary"))
                and not _had_summary_before
            )
            if _filled_history or _filled_summary:
                logger.warning(
                    "[P0-1/CENTRAL] _build_meal_plan_insert_sql aplicó herencia "
                    f"fallback para user_id={_user_id} "
                    f"(history={_filled_history}, summary={_filled_summary}). "
                    "El call site no invocó _apply_inherited_lifetime_lessons "
                    "antes; preferir save_new_meal_plan_atomic/_robust como "
                    "entrada para que la herencia use el cursor óptimo."
                )

    cols = list(data.keys())
    vals = []
    for col, v in zip(cols, data.values()):
        if col in _MEAL_PLAN_ARRAY_COLS:
            vals.append(v)
        elif col == "profile_embedding" and isinstance(v, list):
            vals.append(str(v))
        elif isinstance(v, (dict, list)):
            vals.append(Jsonb(v))
        else:
            vals.append(v)
    placeholders = ", ".join(["%s"] * len(cols))
    col_str = ", ".join(cols)
    sql = f"INSERT INTO meal_plans ({col_str}) VALUES ({placeholders})"
    if with_returning:
        sql += " RETURNING id"
    return sql, vals


def save_new_meal_plan_atomic(user_id: str, insert_data: dict, return_id: bool = False):
    """Inserta nuevo plan nutricional Y cancela chunks vivos del usuario en una sola transacción.

    Garantiza que no queden chunks huérfanos de planes anteriores que puedan re-encolarse
    contra el nuevo plan (vía recovery crons o re-pickup del worker).
    Si return_id=True retorna el UUID del plan insertado (str). Si return_id=False retorna True.
    Lanza excepción en caso de error.
    """
    if not connection_pool:
        raise RuntimeError("db connection_pool is not available.")

    import copy
    from psycopg.rows import dict_row

    def _run(data: dict):
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor(row_factory=dict_row) as cursor:
                    # [P0-3] Cancelar TODOS los estados que puedan re-disparar generación contra
                    # el plan nuevo. Antes solo se cubrían 'pending' y 'processing', dejando
                    # huérfanos:
                    #   - 'failed' → _recover_failed_chunks_for_long_plans los re-encola.
                    #   - 'pending_user_action' → _recover_pantry_paused_chunks puede levantarlos.
                    #   - 'stale' → el worker los re-pickea al refrescar pantry.
                    # Atómico con el INSERT: si algo falla después, el UPDATE se revierte.

                    # [P0-4] Liberar reservas ANTES de cancelar para evitar phantom reserved_quantity.
                    cursor.execute(
                        "SELECT id FROM plan_chunk_queue "
                        "WHERE user_id = %s "
                        "AND status IN ('pending', 'processing', 'stale', 'pending_user_action', 'failed')",
                        (user_id,)
                    )
                    chunks_to_cancel = cursor.fetchall()
                    from db_inventory import release_chunk_reservations
                    for _ctc in chunks_to_cancel:
                        release_chunk_reservations(user_id, str(_ctc["id"]))

                    cursor.execute(
                        "UPDATE plan_chunk_queue SET status = 'cancelled', updated_at = NOW() "
                        "WHERE user_id = %s "
                        "AND status IN ('pending', 'processing', 'stale', 'pending_user_action', 'failed') "
                        "RETURNING id",
                        (user_id,)
                    )
                    cancelled_rows = cursor.fetchall()

                    # [P0-1/INHERIT] Heredar lecciones del plan previo si el plan nuevo no
                    # las trae explícitas. Sin esto, cambiar de 7d → 15d → 30d reseteaba el
                    # aprendizaje del usuario: cada plan partía de cero pese a que el
                    # _lifetime_lessons_history del plan previo seguía vivo en meal_plans.
                    _apply_inherited_lifetime_lessons(user_id, data, cursor=cursor)

                    # [P0-PERSIST-TXN-IDLE · 2026-07-10] skip: los pases finalize ya
                    # corrieron FUERA de esta transacción (ver pre-finalize abajo).
                    # Re-correrlos acá dejaría la conexión idle ~10-25s bajo el BEGIN
                    # → el SET de sesión de 15s (P1-DB-STMT-TIMEOUT) mata el INSERT.
                    sql, vals = _build_meal_plan_insert_sql(
                        data, with_returning=True, skip_plan_data_finalize=True
                    )
                    cursor.execute(sql, vals)  # pyright: ignore[reportArgumentType]  # psycopg LiteralString FP (sql dinámico)
                    row = cursor.fetchone()
                    plan_id = str(row["id"]) if row else None
        return plan_id, len(cancelled_rows)

    safe_data = copy.deepcopy(insert_data)
    # [P0-PERSIST-TXN-IDLE · 2026-07-10] Finaliza plan_data ANTES de abrir la
    # transacción: coherence stack + protein band + all-4 band closer + clear/
    # refresh son CPU-bound (~10-25s con fuzzy matching) y dentro del BEGIN
    # dejaban la conexión idle-in-transaction > 15s (SET de sesión de db_core,
    # P1-DB-STMT-TIMEOUT) → Neon mataba la sesión y el INSERT fallaba con el
    # plan YA generado (pérdida total: corr=d2bc0bcc 2026-07-10, sync ×2
    # 2026-07-09). El retry por columnas ausentes (except abajo) re-entra a
    # _run con el MISMO safe_data ya finalizado — no se re-paga el costo.
    _finalize_plan_data_for_insert(safe_data)
    try:
        plan_id, n_cancelled = _run(safe_data)
    except Exception as e:
        err_msg = str(e)
        if "column" in err_msg and any(c in err_msg for c in ("meal_names", "techniques", "ingredients")):
            logger.warning(f"⚠️ [DB/ATOMIC] Columnas optimizadas ausentes, reintentando sin ellas: {err_msg[:120]}")
            safe_data.pop("meal_names", None)
            safe_data.pop("ingredients", None)
            safe_data.pop("techniques", None)
            plan_id, n_cancelled = _run(safe_data)
        else:
            raise

    if n_cancelled > 0:
        logger.info(f"✅ [P0-3/ATOMIC] {n_cancelled} chunk(s) huérfano(s) cancelados (incl. failed/stale/pending_user_action) al crear plan {plan_id} para {user_id}")
    return plan_id if return_id else True


def save_new_meal_plan_robust(insert_data: dict, additional_queries: Optional[List[Tuple[str, tuple]]] = None, return_id: bool = False):
    """Guarda un nuevo plan nutricional con fallback por si faltan columnas optimizadas.

    Si return_id=True, añade RETURNING id y devuelve el UUID del plan insertado (str).
    Si return_id=False (default), devuelve True al éxito como antes.
    """
    if not connection_pool: return None if return_id else False
    # [P-TYPING-1] deepcopy fuera del try: garantiza `safe_data` ligado antes del
    # except (que lo usa). Un fallo de deepcopy se propaga idéntico al path
    # `else: raise try_db_e` previo (era inalcanzable: deepcopy de dict no falla).
    import copy
    safe_data = copy.deepcopy(insert_data)
    try:

        # [P0-1/INHERIT] Misma herencia cross-plan que el path atomic, pero sin
        # cursor compartido: el helper abre su propio SELECT vía execute_sql_query.
        _user_id = safe_data.get("user_id")
        if _user_id:
            _apply_inherited_lifetime_lessons(_user_id, safe_data, cursor=None)

        query, vals = _build_meal_plan_insert_sql(safe_data, with_returning=return_id)

        if additional_queries:
            from db_core import execute_sql_transaction
            queries_to_run = [(query, tuple(vals))] + additional_queries
            execute_sql_transaction(queries_to_run)
            return True
        else:
            if return_id:
                result = execute_sql_write(query, tuple(vals), returning=True)
                return str(result[0]["id"]) if result else None
            execute_sql_write(query, tuple(vals))
            return True
    except Exception as try_db_e:
        err_msg = str(try_db_e)
        if "column" in err_msg and ("meal_names" in err_msg or "techniques" in err_msg or "ingredients" in err_msg):
            logger.warning(f"⚠️ [DB] Error con columnas optimizadas ({err_msg[:120]}). Guardando sin ellas.")
            safe_data.pop("meal_names", None)
            safe_data.pop("ingredients", None)
            safe_data.pop("techniques", None)
            query, vals = _build_meal_plan_insert_sql(safe_data, with_returning=return_id)
            try:
                if additional_queries:
                    from db_core import execute_sql_transaction
                    queries_to_run = [(query, tuple(vals))] + additional_queries
                    execute_sql_transaction(queries_to_run)
                    return True
                else:
                    if return_id:
                        result = execute_sql_write(query, tuple(vals), returning=True)
                        return str(result[0]["id"]) if result else None
                    execute_sql_write(query, tuple(vals))
                    return True
            except Exception as e2:
                raise e2
        else:
            raise try_db_e

def get_latest_meal_plan(user_id: str):
    """Obtiene el JSON del plan de comidas más reciente del usuario."""
    # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo.
    if not connection_pool: return None
    try:
        row = execute_sql_query(
            "SELECT plan_data FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1",
            (user_id,), fetch_one=True
        )
        if row:
            return row.get("plan_data")
        return None
    except Exception as e:
        logger.error(f"Error obteniendo plan actual: {e}")
        return None

def get_recent_plans(user_id: str, days: int = 14) -> list:
    """Obtiene los JSON de los planes recientes dentro del rango de días especificado."""
    # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo.
    if not connection_pool: return []
    try:
        from datetime import datetime, timezone, timedelta
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        rows = execute_sql_query(
            "SELECT plan_data FROM meal_plans WHERE user_id = %s AND created_at >= %s ORDER BY created_at DESC",
            (user_id, cutoff_date), fetch_all=True
        )
        if rows:
            return [row.get("plan_data") for row in rows if row.get("plan_data")]
        return []
    except Exception as e:
        logger.error(f"Error obteniendo planes recientes: {e}")
        return []

def get_recent_meals_from_plans(user_id: str, days: int = 5):
    """Obtiene una lista de nombres de comidas de los planes recientes para evitar repeticiones."""
    # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo.
    if not connection_pool: return []
    try:
        rows = execute_sql_query(
            "SELECT plan_data, meal_names FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, days), fetch_all=True
        )
        meals = set() # 👈 Usar un Set evita enviar nombres duplicados al LLM y ahorra tokens
        if rows:
            for row in rows:
                meal_names_sql = row.get("meal_names")
                if meal_names_sql:
                    # 🚀 Fast Path O(1)
                    for n in meal_names_sql:
                        meals.add(n)
                else:
                    # 🐢 Slow Path O(N)
                    plan_data = row.get("plan_data", {})
                    if isinstance(plan_data, dict):
                         for day in plan_data.get("days", []):
                             for meal in day.get("meals", []):
                                 meal_name = meal.get("name")
                                 if meal_name:
                                     meals.add(meal_name)
                         if "meals" in plan_data:
                             for meal in plan_data.get("meals", []):
                                 meal_name = meal.get("name")
                                 if meal_name:
                                     meals.add(meal_name)
        return list(meals)
    except Exception as e:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Fallback PGRST205 (columna
        # meal_names ausente) eliminado: la columna existe en el schema Neon.
        logger.error(f"Error obteniendo comidas recientes: {e}")
        return []

def get_recent_techniques(user_id: str, limit: int = 5) -> list:
    """Obtiene las técnicas de cocción usadas en planes recientes desde la columna `techniques` (text[]).
    Retorna una lista de tuplas (technique, created_at) para que el caller pueda aplicar decaimiento temporal.
    Ejemplo: [('Horneado Saludable', '2026-03-18T...'), ('Al Vapor', '2026-03-15T...')]
    """
    # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo. `created_at::text`
    # preserva el contrato string ISO que el caller parsea (graph_orchestrator
    # `_select_techniques` hace `.endswith("Z")` + fromisoformat sobre el valor).
    if not connection_pool or not user_id or user_id == "guest": return []
    try:
        rows = execute_sql_query(
            "SELECT techniques, created_at::text AS created_at FROM meal_plans "
            "WHERE user_id = %s ORDER BY meal_plans.created_at DESC LIMIT %s",
            (user_id, limit), fetch_all=True
        )
        # Retornar lista de tuplas CON duplicados y timestamps para decaimiento temporal.
        techniques = []
        if rows:
            for row in rows:
                techs = row.get("techniques")
                created_at = row.get("created_at", "")
                if techs and isinstance(techs, list):
                    for t in techs:
                        if t:
                            techniques.append((t, created_at))
        return techniques
    except Exception as e:
        logger.error(f"Error obteniendo técnicas recientes: {e}")
        return []

def get_ingredient_frequencies_from_plans(user_id: str, limit: int = 5) -> list:
    """Extrae los ingredientes crudos directamente del JSON o de la columna optimizada si existe.
    Retorna una lista plana de strings de ingredientes."""
    # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo.
    if not connection_pool or not user_id: return []
    try:
        rows = execute_sql_query(
            "SELECT plan_data, ingredients FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit), fetch_all=True
        )
        all_ingredients = []
        if rows:
            for row in rows:
                ingredients_sql = row.get("ingredients")
                if ingredients_sql:
                    # 🚀 Fast Path O(1)
                    all_ingredients.extend(ingredients_sql)
                else:
                    # 🐢 Slow Path O(N)
                    plan_data = row.get("plan_data", {})
                    if isinstance(plan_data, dict):
                        for day in plan_data.get("days", []):
                            for meal in day.get("meals", []):
                                ingredients = meal.get("ingredients", [])
                                if isinstance(ingredients, list):
                                    all_ingredients.extend(ingredients)
        return all_ingredients
    except Exception as e:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Fallback PGRST205 (columna
        # ingredients ausente) eliminado: la columna existe en el schema Neon.
        logger.error(f"Error extrayendo ingredientes de planes: {e}")
        return []

def get_latest_meal_plan_with_id(user_id: str):
    """Obtiene el plan más reciente del usuario incluyendo su ID para poder actualizarlo."""
    try:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Fallback PostgREST eliminado (pool
        # mandatorio post-Neon). Si el pool falta, execute_sql_query lanza y el
        # except de abajo preserva el contrato (None + error log).
        res = execute_sql_query("SELECT id, plan_data, created_at FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1", (user_id,), fetch_one=True)
        if res:
            return res
        return None
    except Exception as e:
        logger.error(f"Error obteniendo plan con ID: {e}")
        return None

def update_meal_plan_data(plan_id: str, new_plan_data: dict, user_id: Optional[str] = None):
    """[P1-NEW-3 · 2026-05-10] Actualiza el plan_data JSONB de un plan
    filtrando por `(id, user_id)` cuando se provee `user_id`.

    [P1-NEXT-1 · 2026-05-11] El UPDATE full-overwrite se ejecuta dentro
    de una transacción que adquiere `acquire_meal_plan_advisory_lock(
    cursor, plan_id, purpose='general')` ANTES del UPDATE. Mismo
    `purpose='general'` que `_chunk_worker` T1/T2, `api_shift_plan` y
    `_background_shift_plan_for_user` — garantiza serialización
    inter-worker contra lost-update (invariante I7 de CLAUDE.md).

    Antes de P1-NEXT-1, este helper hacía `execute_sql_write` plano
    sin lock: los 4 callsites producción (`/recipe/expand`,
    `/recalculate-shopping-list`, `proactive_agent` JIT week-2,
    `tools.modify_single_meal`) entraban en race read-modify-write
    con `_chunk_worker` T2 que también full-overwrite plan_data tras
    cada chunk. Síntoma: una mutación bonafide se perdía silenciosamente
    si el cron commiteaba después. El test P1-NEW-C no lo detectaba
    porque scaneaba el patrón literal `UPDATE … SET plan_data = %s::jsonb`
    en `routers/plans.py`, mientras el helper usa el adapter `Jsonb(...)`
    en `db_plans.py` — fuera del scope del test.

    Antes el helper hacía `UPDATE … WHERE id = %s` sin ownership check —
    delegaba 100% al caller la responsabilidad de validar ownership. Un
    callsite futuro que olvidara el check (como ocurrió originalmente en
    `/restock`, cerrado en P0-NEW-1) reintroducía IDOR sobre meal_plans.

    Contrato post-P1-NEW-3:
      - `user_id` es OPCIONAL solo por compatibilidad mientras migramos
        los 4 callsites históricos. Tests parser-based enforzan que cada
        callsite producción lo pase.
      - Si `user_id` se pasa: WHERE filtra por `(id, user_id)`. Si la
        fila no existe o no pertenece, retorna `0`/`[]` (mismo retorno
        que un plan_id inexistente — el caller decide cómo manejarlo).
      - Si `user_id` es `None`: comportamiento legacy (solo `WHERE id`).
        DEPRECATED — emite warning de log para forzar migración.

    Retorna:
      - `True` si el UPDATE se ejecutó (path psycopg con connection_pool).
      - `None` si la operación falló (pool no disponible, error SQL).
        [P1-NEON-DB-MIGRATION · 2026-06-12] El path legado via PostgREST (que
        retornaba `list`) fue eliminado — pool mandatorio post-Neon.

    Tooltip-anchor: P1-NEXT-1-LOCK-START | test_p1_next_1_update_meal_plan_data_holds_advisory_lock
    """
    try:
        if user_id is None:
            logger.warning(
                "[P1-NEW-3] update_meal_plan_data llamado sin user_id "
                f"para plan_id={plan_id}. Defense-in-depth a DB-level "
                f"degrada a legacy WHERE id. Migrar el callsite para "
                f"pasar user_id."
            )
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Fallback via PostgREST (dev/local
        # sin connection_pool) eliminado: PostgREST no soporta advisory locks
        # (violaba I7) y apuntaría a la DB equivocada. Pool
        # mandatorio — fail-loud; el except preserva el contrato None.
        if not connection_pool:
            raise RuntimeError("db connection_pool is not available (mandatorio post-Neon).")
        from psycopg.types.json import Jsonb
        # [P1-NEXT-1 · 2026-05-11] El lock + UPDATE viven en la MISMA
        # transacción. `pg_advisory_xact_lock` (vía
        # `acquire_meal_plan_advisory_lock`) se libera al commit/rollback.
        # Mismo purpose='general' que _chunk_worker T1/T2 y api_shift_plan
        # → todos los writers full-overwrite del mismo plan se serializan.
        with connection_pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cursor:
                    acquire_meal_plan_advisory_lock(cursor, plan_id, purpose="general")
                    if user_id is None:
                        cursor.execute(
                            "UPDATE meal_plans SET plan_data = %s WHERE id = %s",
                            (Jsonb(new_plan_data), plan_id),
                        )
                    else:
                        # Path nuevo con ownership a DB-level. Mismo patrón
                        # que P0-HIST-IDOR-1/2 cerraron con `AND user_id = %s`
                        # defense-in-depth (invariante I2).
                        cursor.execute(
                            "UPDATE meal_plans SET plan_data = %s WHERE id = %s AND user_id = %s",
                            (Jsonb(new_plan_data), plan_id, user_id),
                        )
        return True
    except Exception as e:
        logger.error(f"Error actualizando plan_data: {e}")
        return None

def track_meal_friction(user_id: str, session_id: str, rejected_meal: str):
    """
    Memoria Conductual: Trackea cuántas veces el usuario rechaza platos con la misma proteína base.
    Al tercer rechazo (strike 3), inserta el ingrediente en rechazos temporales y notifica proactivamente.
    """
    if not user_id or user_id == "guest" or not rejected_meal: return False
    
    from constants import DOMINICAN_PROTEINS
    # GLOBAL_REVERSE_MAP is imported globally
    
    # Usar el mapa pre-computado a nivel de módulo (O(1)) en vez de reconstruirlo por llamada.
    # Crear versión sin acentos para matching robusto
    # (el LLM no siempre preserva tildes: "platano" vs "plátano")
    accent_safe_map = GLOBAL_REVERSE_MAP
    
    base_ingredient = None
    lower_meal = strip_accents(rejected_meal.lower())
    
    # Resolver sinónimos con n-gramas (trigrams → bigrams → unigrams)
    # para detectar multipalabra como "carne molida" → "res", "queso de freír" → "queso de freír"
    words = [w.strip(".,;:!?()\"'") for w in lower_meal.split()]
    for n in range(min(3, len(words)), 0, -1):  # 3-grams primero, luego 2, luego 1
        if base_ingredient:
            break
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i:i+n])
            if ngram in accent_safe_map:
                base_ingredient = accent_safe_map[ngram].capitalize()
                break
    
    # Fallback: búsqueda directa por nombre de proteína base
    if not base_ingredient:
        for p in DOMINICAN_PROTEINS:
            if p.lower() in lower_meal:
                base_ingredient = p
                break
            
    if not base_ingredient: return False
    
    # --- MÉTODO ATÓMICO (RPC) para evitar Race Conditions ---
    # Si dos requests de swap llegan simultáneos, el FOR UPDATE en PostgreSQL
    # serializa las escrituras garantizando que ambos incrementos se registren.
    # [P1-NEON-DB-MIGRATION · 2026-06-12] RPC via SQL directo (antes PostgREST).
    try:
        rpc_row = execute_sql_query(
            "SELECT public.increment_friction_rpc(%s::uuid, %s) AS result",
            (user_id, base_ingredient),
            fetch_one=True,
        )
        rpc_val = rpc_row.get("result") if rpc_row else None

        # El RPC retorna el conteo PRE-RESET (ej: 3 si alcanzó el umbral)
        new_count = rpc_val if isinstance(rpc_val, int) else 0
        
        if new_count >= 3:
            logger.info(f"🛑 [FRICCIÓN SILENCIOSA] 3 strikes para {base_ingredient}. Auto-bloqueando ingrediente (vía RPC atómico).")
            
            rejection_record = {
                "meal_name": base_ingredient,
                "meal_type": "Ingrediente Fricción",
                "user_id": user_id,
                "session_id": session_id if session_id else None
            }
            insert_rejection(rejection_record)
            
            if session_id:
                msg = f"He notado que últimamente has estado evitando opciones con **{base_ingredient}**, así que lo he sacado de tu radar y guardado en tus rechazos temporales por unas semanas para asegurar variedad. 🤖"
                save_message(session_id, "model", msg)
            return True
        
        return False
        
    except Exception as rpc_e:
        # [P1-NEON-DB-MIGRATION · 2026-06-12] Detección PGRST202 ("RPC no
        # desplegado") eliminada — la función existe en el schema Neon.
        logger.error(f"⚠️ [FRICCIÓN] Error en RPC atómico, usando fallback: {rpc_e}")
    
    # --- FALLBACK ATÓMICO (P1-2): si el RPC no está disponible, usamos el
    # advisory lock + FOR UPDATE de `update_user_health_profile_atomic` para
    # garantizar que el contador de friction se incrementa SIEMPRE bajo lock.
    # Antes era un read-modify-write desnudo: dos clicks de "swap" simultáneos
    # del mismo user_id leían el mismo counter, ambos lo subían a (n+1)
    # localmente, y el último UPDATE pisaba al primero — perdiendo 1 strike.
    # Con 3 strikes para auto-bloquear, perder strikes silenciosamente
    # significa que el ingrediente NUNCA se bloquea aunque el usuario lo
    # rechace 3+ veces. El SQL directo sobre Neon ya no es vulnerable a esa
    # race condition.
    decision_box = {"reset": False, "current_count": 0}

    def _friction_mutator(_hp):
        _frictions = dict(_hp.get("frictions", {}) or {})
        _new_count = _frictions.get(base_ingredient, 0) + 1
        decision_box["current_count"] = _new_count
        if _new_count >= 3:
            decision_box["reset"] = True
            _frictions[base_ingredient] = 0
        else:
            _frictions[base_ingredient] = _new_count
        _hp["frictions"] = _frictions
        return None

    new_hp = update_user_health_profile_atomic(user_id, _friction_mutator)
    if new_hp is None:
        # Pool no disponible o user no existe; el atomic helper degradó al
        # legacy non-atómico (loguea su propio warning) — preservamos el
        # contrato de retorno False y no insertamos rejection.
        return False

    if decision_box["reset"]:
        logger.info(
            f"🛑 [FRICCIÓN SILENCIOSA] 3 strikes para {base_ingredient}. "
            f"Auto-bloqueando ingrediente (fallback atómico)."
        )
        rejection_record = {
            "meal_name": base_ingredient,
            "meal_type": "Ingrediente Fricción",
            "user_id": user_id,
            "session_id": session_id if session_id else None,
        }
        insert_rejection(rejection_record)
        if session_id:
            msg = f"He notado que últimamente has estado evitando opciones con **{base_ingredient}**, así que lo he sacado de tu radar y guardado en tus rechazos temporales por unas semanas para asegurar variedad. 🤖"
            save_message(session_id, "model", msg)
        return True

    return False

def log_unknown_ingredients(user_id: str, unknown_ings: list, raw_map: Optional[dict] = None):
    """Loguea ingredientes que el LLM genera pero que el sistema de sinónimos no reconoce.
    Se guardan en la tabla `unknown_ingredients` para revisión periódica y expansión del catálogo.
    Usa el RPC atómico `log_unknown_ingredient_rpc` (incrementa occurrences en conflicto).
    """
    # [P1-NEON-DB-MIGRATION · 2026-06-12] RPC via SQL directo (antes PostgREST).
    # Fallbacks eliminados: detección PGRST202/PGRST205 ("RPC/tabla no
    # desplegados") y upsert directo a `unknown_ingredients` — la función y la
    # tabla existen en el schema Neon. Best-effort: el except loguea y retorna.
    if not connection_pool or not user_id or user_id == "guest" or not unknown_ings:
        return

    try:
        for ing in unknown_ings[:20]:  # Cap a 20 por plan para no saturar
            raw_text = raw_map.get(ing, "") if raw_map else ""
            execute_sql_query(
                "SELECT public.log_unknown_ingredient_rpc(%s::uuid, %s, %s)",
                (user_id, ing, raw_text or None),
                fetch_one=True,
            )

        logger.info(f"📝 [UNKNOWN ING] {len(unknown_ings)} ingredientes no reconocidos logueados para revisión")
    except Exception as e:
        logger.error(f"⚠️ [UNKNOWN ING] Error logueando ingredientes desconocidos: {e}")

def increment_ingredient_frequencies(user_id: str, ingredients: list[str]):
    """Incrementa la frecuencia histórica de los ingredientes consumidos por un usuario.
    Usa el RPC atómico `increment_ingredient_frequencies_rpc` (ON CONFLICT
    incrementa count), robusto ante Race Conditions.
    """
    # [P1-NEON-DB-MIGRATION · 2026-06-12] RPC via SQL directo (antes PostgREST).
    # Fallback clásico Select+Upsert eliminado: era dev-only, tenía race
    # condition documentada (lost update) y la función existe en el schema
    # Neon. Best-effort: el except loguea y retorna.
    if not connection_pool or not user_id or user_id == "guest": return

    try:
        from collections import Counter

        # strip_accents is imported globally

        normalized_ings = [strip_accents(i.lower()).strip() for i in ingredients if i]
        if not normalized_ings: return

        incoming_counts = Counter(normalized_ings)
        ingredients_list = list(incoming_counts.keys())
        counts_list = list(incoming_counts.values())

        # Método atómico (RPC) para evitar Race Conditions. Casts explícitos:
        # psycopg adapta list[str]/list[int] a arrays pero el cast fija el tipo
        # exacto que la función espera (text[], integer[]).
        execute_sql_query(
            "SELECT public.increment_ingredient_frequencies_rpc(%s::uuid, %s::text[], %s::int[])",
            (user_id, ingredients_list, counts_list),
            fetch_one=True,
        )
        logger.info(f"✅ [DB] Frecuencia atómica (RPC) incrementada para {user_id} ({len(ingredients_list)} items)")

    except Exception as e:
        logger.error(f"⚠️ [DB] Error incrementando frecuecia de ingredientes: {e}")

def get_user_ingredient_frequencies(user_id: str, days_limit: int = 60) -> dict:
    """Retorna un diccionario {ingrediente_normalizado: conteo_decaimiento} de la DB.
    Implementa Decaimiento Temporal Continuo Matemático: count * (decay ^ dias_transcurridos).
    Se amplía days_limit a 60 días por defecto para dar margen al decaimiento suave.

    [P1-4] El factor de decay viene de constants.INGREDIENT_FATIGUE_DECAY_FACTOR para
    estar alineado con cron_tasks.calculate_ingredient_fatigue. Antes ambos usaban 0.9
    hardcoded por separado — riesgo de drift si alguien tuneaba uno y olvidaba el otro.
    """
    # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo. `last_used::text`
    # preserva el contrato string ISO que el parse de abajo (safe_fromisoformat) espera.
    if not connection_pool or not user_id or user_id == "guest": return {}
    try:
        from datetime import datetime, timedelta, timezone
        from constants import INGREDIENT_FATIGUE_DECAY_FACTOR as decay_factor

        now = datetime.now(timezone.utc)
        cutoff_date = now - timedelta(days=days_limit)

        rows = execute_sql_query(
            "SELECT ingredient, count, last_used::text AS last_used "
            "FROM ingredient_frequencies WHERE user_id = %s AND last_used >= %s",
            (user_id, cutoff_date), fetch_all=True
        )

        if not rows:
            return {}

        freq_dict = {}

        for row in rows:
            ingredient = row["ingredient"]
            count = row["count"]
            last_used_str = row.get("last_used")
            
            if not last_used_str:
                freq_dict[ingredient] = count
                continue
                
            try:
                # Parse robusto para last_used asumiendo formato ISO de la DB
                from constants import safe_fromisoformat
                last_used_dt = safe_fromisoformat(last_used_str)
                if last_used_dt.tzinfo is None:
                    last_used_dt = last_used_dt.replace(tzinfo=timezone.utc)
                        
                days_elapsed = max(0, (now - last_used_dt).days)
                
                # Fórmula decaimiento matemático: count * (decay_factor ^ days_elapsed)
                decayed_count = count * (decay_factor ** days_elapsed)
                
                freq_dict[ingredient] = round(decayed_count, 2)
            except Exception as parse_e:
                logger.error(f"⚠️ [DB] Error parseando fecha {last_used_str}: {parse_e}")
                freq_dict[ingredient] = count
                
        return freq_dict
    except Exception as e:
        logger.error(f"⚠️ [DB] Error obteniendo diccionario de frecuencias: {e}")
        return {}

def search_similar_plan(query_embedding: list, threshold: float = 0.98, limit: int = 1):
    """Busca planes similares usando búsqueda vectorial (similitud coseno) a través de la RPC."""
    # [P1-NEON-DB-MIGRATION · 2026-06-12] PostgREST → SQL directo. El vector se
    # serializa como literal text (`str(list)` — mismo patrón que el INSERT de
    # `profile_embedding` en _build_meal_plan_insert_sql) y se castea a
    # `extensions.vector` (schema donde vive pgvector en el dump restaurado).
    # `id/user_id::text` preservan el contrato string que PostgREST devolvía.
    if not connection_pool: return []
    try:
        rows = execute_sql_query(
            """
            SELECT id::text AS id, user_id::text AS user_id, plan_data, similarity
            FROM public.match_similar_plan(%s::extensions.vector, %s::float8, %s::int)
            """,
            (str(query_embedding), threshold, limit),
            fetch_all=True,
        )
        return rows or []
    except Exception as e:
        logger.error(f"Error buscando planes similares: {e}")
        return []


def count_stale_cache_schema_plans(current_version: str, legacy_version: str) -> dict:
    """[P1-ORQ-5] Cuenta planes en `meal_plans` con `_cache_schema_version`
    distinto al actual.

    Usado por el startup hook (`app.py` lifespan) para emitir warning con el
    conteo cuando un deploy bumpea `CACHE_SCHEMA_VERSION` y deja N planes
    legacy en la tabla. Sin este probe, operadores veían un drop súbito en
    el cache hit rate (planes legacy pasan el vector search pero son
    descartados por `semantic_cache_check_node` post-filter) sin manera de
    correlacionarlo con el cambio de schema.

    Args:
        current_version: la versión actual de `CACHE_SCHEMA_VERSION`.
        legacy_version: la versión legacy asumida para planes pre-fix sin
            el flag (típicamente "v1").

    Returns:
        Dict con:
          - `stale_count`: total de planes con versión != current_version
          - `stale_versions`: dict version → count de cada versión obsoleta
          - `total`: total de planes en la tabla
        Vacío `{}` si el pool no está disponible o el query falló.

    Best-effort: cualquier excepción se loguea como warning y devuelve {}.
    El startup NO debe fallar si este probe falla — es observabilidad pura.
    """
    # [P1-NEON-DB-MIGRATION · 2026-06-12] El count migra de PostgREST
    # (count="exact") a SQL directo — el GROUP BY de abajo ya iba por pool.
    if not connection_pool:
        return {}
    try:
        # COUNT exact total para contexto.
        total_row = execute_sql_query(
            "SELECT COUNT(*) AS total FROM meal_plans", fetch_one=True
        )
        total = int(total_row.get("total") or 0) if total_row else 0
        if total == 0:
            return {"stale_count": 0, "stale_versions": {}, "total": 0}

        # Bucket por versión via GROUP BY sobre extracción de JSONB.
        # `COALESCE(... ->> '_cache_schema_version', legacy)` aplica la misma
        # convención que `_is_cached_plan_schema_compatible` (línea ~6324 de
        # graph_orchestrator): planes pre-fix sin la key se cuentan como
        # legacy_version.
        rows = execute_sql_query(
            """
            SELECT COALESCE(plan_data->>'_cache_schema_version', %s) AS schema_v,
                   COUNT(*) AS cnt
            FROM meal_plans
            GROUP BY schema_v
            """,
            (legacy_version,),
            fetch_all=True,
        )
        if not rows:
            return {"stale_count": 0, "stale_versions": {}, "total": total}

        stale_versions: dict[str, int] = {}
        stale_count = 0
        for row in rows:
            v = row.get("schema_v") or legacy_version
            cnt = int(row.get("cnt") or 0)
            if v != current_version:
                stale_versions[v] = cnt
                stale_count += cnt
        return {
            "stale_count": stale_count,
            "stale_versions": stale_versions,
            "total": total,
        }
    except Exception as e:
        logger.warning(f"⚠️ [P1-ORQ-5] Error contando planes con schema obsoleto: {e}")
        return {}
