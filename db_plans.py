from functools import lru_cache as _lru_cache
import uuid
import unicodedata as _uc
from typing import Optional, List, Dict, Any, Tuple, Union
import os
import logging
logger = logging.getLogger(__name__)
from db_core import supabase, connection_pool, execute_sql_query, execute_sql_write
from constants import strip_accents, GLOBAL_REVERSE_MAP
from db_chat import insert_rejection, save_message
from db_profiles import get_user_profile, update_user_health_profile

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


def update_plan_data_atomic(
    plan_id: str,
    mutator,
    lock_timeout_ms: int = None,
) -> dict:
    """[P0-2] Read-Modify-Write atómico de meal_plans.plan_data.

    Aplica `mutator(plan_data)` (que MUTA o RETORNA un nuevo dict) dentro de un
    `SELECT … FOR UPDATE` que serializa concurrentes contra la misma fila. Es la
    forma segura de tocar keys acumulativas como _consecutive_zero_log_chunks,
    _last_chunk_learning, _recent_chunk_lessons, _critical_lessons_permanent y
    _recovery_exhausted_chunks: dos chunks que escriben simultáneamente NO se
    sobrescriben — el segundo ve el plan_data ya actualizado por el primero.

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

    Returns:
        El dict plan_data resultante tras la mutación. Si la fila no existe,
        retorna {} y NO ejecuta UPDATE.
    """
    if not connection_pool:
        raise RuntimeError("db connection_pool is not available for atomic plan_data update.")

    if lock_timeout_ms is None:
        from constants import CHUNK_LEARNING_LOCK_TIMEOUT_MS
        lock_timeout_ms = int(CHUNK_LEARNING_LOCK_TIMEOUT_MS)

    from psycopg.rows import dict_row
    from psycopg.types.json import Jsonb

    with connection_pool.connection() as conn:
        with conn.transaction():
            with conn.cursor(row_factory=dict_row) as cursor:
                try:
                    cursor.execute(f"SET LOCAL lock_timeout = '{int(lock_timeout_ms)}ms'")
                except Exception as set_err:
                    logger.debug(f"[P0-2] No se pudo setear lock_timeout en update_plan_data_atomic: {set_err}")

                cursor.execute(
                    "SELECT plan_data FROM meal_plans WHERE id = %s FOR UPDATE",
                    (plan_id,),
                )
                row = cursor.fetchone()
                if not row:
                    logger.warning(
                        f"[P0-2] update_plan_data_atomic: meal_plan {plan_id} no existe "
                        f"(probablemente cancelado por save_new_meal_plan_atomic). Skip."
                    )
                    return {}

                current = row["plan_data"] or {}
                if not isinstance(current, dict):
                    current = {}

                result = mutator(current)
                if result is False:
                    return current

                new_data = result if isinstance(result, dict) else current
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
    plan_data["grocery_start_date"] = datetime.now(timezone.utc).date().isoformat()
    logger.info(
        "[P0-1-RECOVERY/C] grocery_start_date ausente al insertar meal_plan; "
        f"persistido fallback={plan_data['grocery_start_date']}."
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


def _build_meal_plan_insert_sql(data: dict, with_returning: bool = False):
    """Construye la SQL de INSERT para meal_plans a partir de un dict.

    Retorna (sql, vals) listos para pasar a cursor.execute.
    """
    from psycopg.types.json import Jsonb
    # [P0-1-RECOVERY/C] Defensa centralizada: cualquier path que use este helper para
    # insertar a meal_plans tendrá grocery_start_date garantizado. Evita reintroducir
    # el bug donde el pipeline LLM omitía el campo y sólo el backfill en runtime
    # (cron_tasks.py:5118) lo poblaba a costa de chunks que fallaban antes.
    if "plan_data" in data:
        data["plan_data"] = _ensure_grocery_start_date(data["plan_data"])

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

                    sql, vals = _build_meal_plan_insert_sql(data, with_returning=True)
                    cursor.execute(sql, vals)
                    row = cursor.fetchone()
                    plan_id = str(row["id"]) if row else None
        return plan_id, len(cancelled_rows)

    safe_data = copy.deepcopy(insert_data)
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


def save_new_meal_plan_robust(insert_data: dict, additional_queries: List[Tuple[str, tuple]] = None, return_id: bool = False):
    """Guarda un nuevo plan nutricional con fallback por si faltan columnas optimizadas.

    Si return_id=True, añade RETURNING id y devuelve el UUID del plan insertado (str).
    Si return_id=False (default), devuelve True al éxito como antes.
    """
    if not connection_pool: return None if return_id else False
    try:
        import copy
        safe_data = copy.deepcopy(insert_data)

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
            result = execute_sql_write(query, tuple(vals), returning=return_id)
            if return_id:
                return str(result[0]["id"]) if result else None
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
                    result = execute_sql_write(query, tuple(vals), returning=return_id)
                    if return_id:
                        return str(result[0]["id"]) if result else None
                    return True
            except Exception as e2:
                raise e2
        else:
            raise try_db_e

def get_latest_meal_plan(user_id: str):
    """Obtiene el JSON del plan de comidas más reciente del usuario."""
    if not supabase: return None
    try:
        res = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        if res.data and len(res.data) > 0:
            return res.data[0].get("plan_data")
        return None
    except Exception as e:
        logger.error(f"Error obteniendo plan actual: {e}")
        return None

def get_recent_plans(user_id: str, days: int = 14) -> list:
    """Obtiene los JSON de los planes recientes dentro del rango de días especificado."""
    if not supabase: return []
    try:
        from datetime import datetime, timezone, timedelta
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        res = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).gte("created_at", cutoff_date).order("created_at", desc=True).execute()
        if res.data:
            return [row.get("plan_data") for row in res.data if row.get("plan_data")]
        return []
    except Exception as e:
        logger.error(f"Error obteniendo planes recientes: {e}")
        return []

def get_recent_meals_from_plans(user_id: str, days: int = 5):
    """Obtiene una lista de nombres de comidas de los planes recientes para evitar repeticiones."""
    if not supabase: return []
    try:
        res = supabase.table("meal_plans").select("plan_data, meal_names").eq("user_id", user_id).order("created_at", desc=True).limit(days).execute()
        meals = set() # 👈 Usar un Set evita enviar nombres duplicados al LLM y ahorra tokens
        if res.data:
            for row in res.data:
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
        error_msg = str(e)
        if "meal_names" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            try:
                logger.warning("⚠️ [DB] Columna meal_names ausente en GET, usando fallback O(N)...")
                res_fb = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(days).execute()
                meals_fb = set()
                if res_fb.data:
                    for row in res_fb.data:
                        plan_data = row.get("plan_data", {})
                        if isinstance(plan_data, dict):
                             for day in plan_data.get("days", []):
                                 for meal in day.get("meals", []):
                                     if meal.get("name"): meals_fb.add(meal.get("name"))
                             if "meals" in plan_data:
                                 for meal in plan_data.get("meals", []):
                                     if meal.get("name"): meals_fb.add(meal.get("name"))
                return list(meals_fb)
            except Exception as e2:
                logger.error(f"Error obteniendo comidas recientes (fallback): {e2}")
                return []
                
        logger.error(f"Error obteniendo comidas recientes: {e}")
        return []

def get_recent_techniques(user_id: str, limit: int = 5) -> list:
    """Obtiene las técnicas de cocción usadas en planes recientes desde la columna `techniques` (text[]).
    Retorna una lista de tuplas (technique, created_at) para que el caller pueda aplicar decaimiento temporal.
    Ejemplo: [('Horneado Saludable', '2026-03-18T...'), ('Al Vapor', '2026-03-15T...')]
    """
    if not supabase or not user_id or user_id == "guest": return []
    try:
        res = supabase.table("meal_plans").select("techniques, created_at").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        # Retornar lista de tuplas CON duplicados y timestamps para decaimiento temporal.
        techniques = []
        if res.data:
            for row in res.data:
                techs = row.get("techniques")
                created_at = row.get("created_at", "")
                if techs and isinstance(techs, list):
                    for t in techs:
                        if t:
                            techniques.append((t, created_at))
        return techniques
    except Exception as e:
        error_msg = str(e)
        if "techniques" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            # La columna aún no existe en la DB → retornar vacío silenciosamente
            return []
        logger.error(f"Error obteniendo técnicas recientes: {e}")
        return []

def get_ingredient_frequencies_from_plans(user_id: str, limit: int = 5) -> list:
    """Extrae los ingredientes crudos directamente del JSON o de la columna optimizada si existe.
    Retorna una lista plana de strings de ingredientes."""
    if not supabase or not user_id: return []
    try:
        res = supabase.table("meal_plans").select("plan_data, ingredients").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
        all_ingredients = []
        if res.data:
            for row in res.data:
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
        error_msg = str(e)
        if "ingredients" in error_msg or "PGRST205" in error_msg or "Could not find" in error_msg:
            try:
                logger.warning("⚠️ [DB] Columna ingredients ausente en GET, usando fallback O(N)...")
                res_fb = supabase.table("meal_plans").select("plan_data").eq("user_id", user_id).order("created_at", desc=True).limit(limit).execute()
                all_ings_fb = []
                if res_fb.data:
                    for row in res_fb.data:
                        plan_data = row.get("plan_data", {})
                        if isinstance(plan_data, dict):
                            for day in plan_data.get("days", []):
                                for meal in day.get("meals", []):
                                    ings = meal.get("ingredients", [])
                                    if isinstance(ings, list):
                                        all_ings_fb.extend(ings)
                return all_ings_fb
            except Exception as e2:
                logger.error(f"Error extrayendo ingredientes de planes (fallback): {e2}")
                return []
                
        logger.error(f"Error extrayendo ingredientes de planes: {e}")
        return []

def get_latest_meal_plan_with_id(user_id: str):
    """Obtiene el plan más reciente del usuario incluyendo su ID para poder actualizarlo."""
    try:
        if connection_pool:
            res = execute_sql_query("SELECT id, plan_data, created_at FROM meal_plans WHERE user_id = %s ORDER BY created_at DESC LIMIT 1", (user_id,), fetch_one=True)
            if res:
                return res
            return None
        else:
            if not supabase: return None
            res = supabase.table("meal_plans").select("id, plan_data, created_at").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
            if res.data and len(res.data) > 0:
                return res.data[0]
            return None
    except Exception as e:
        logger.error(f"Error obteniendo plan con ID: {e}")
        return None

def update_meal_plan_data(plan_id: str, new_plan_data: dict):
    """Actualiza el plan_data JSONB de un plan existente por su ID."""
    try:
        if connection_pool:
            from psycopg.types.json import Jsonb
            execute_sql_write("UPDATE meal_plans SET plan_data = %s WHERE id = %s", (Jsonb(new_plan_data), plan_id))
            return True
        else:
            if not supabase: return None
            res = supabase.table("meal_plans").update({"plan_data": new_plan_data}).eq("id", plan_id).execute()
            return res.data
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
    try:
        rpc_result = supabase.rpc("increment_friction_rpc", {
            "p_user_id": user_id,
            "p_ingredient": base_ingredient
        }).execute()
        
        # El RPC retorna el conteo PRE-RESET (ej: 3 si alcanzó el umbral)
        new_count = rpc_result.data if isinstance(rpc_result.data, int) else 0
        
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
        error_msg = str(rpc_e)
        if "Could not find the function" in error_msg or "PGRST202" in error_msg:
            # RPC aún no desplegado → fallback al método clásico (read-modify-write)
            pass
        else:
            logger.error(f"⚠️ [FRICCIÓN] Error en RPC atómico, usando fallback: {rpc_e}")
    
    # --- FALLBACK CLÁSICO (read-modify-write, vulnerable a race condition) ---
    # ⚠️ En producción, desplegar rpc_increment_friction.sql en Supabase para eliminar esto.
    profile = get_user_profile(user_id)
    if not profile: return False
    
    hp = profile.get("health_profile") or {}
    frictions = hp.get("frictions", {})
    
    current_count = frictions.get(base_ingredient, 0) + 1
    
    if current_count >= 3:
        logger.info(f"🛑 [FRICCIÓN SILENCIOSA] 3 strikes para {base_ingredient}. Auto-bloqueando ingrediente (fallback).")
        
        rejection_record = {
            "meal_name": base_ingredient,
            "meal_type": "Ingrediente Fricción",
            "user_id": user_id,
            "session_id": session_id if session_id else None
        }
        insert_rejection(rejection_record)
        
        frictions[base_ingredient] = 0
        hp["frictions"] = frictions
        update_user_health_profile(user_id, hp)
        
        if session_id:
            msg = f"He notado que últimamente has estado evitando opciones con **{base_ingredient}**, así que lo he sacado de tu radar y guardado en tus rechazos temporales por unas semanas para asegurar variedad. 🤖"
            save_message(session_id, "model", msg)
        return True
    else:
        frictions[base_ingredient] = current_count
        hp["frictions"] = frictions
        update_user_health_profile(user_id, hp)
        return False

def log_unknown_ingredients(user_id: str, unknown_ings: list, raw_map: dict = None):
    """Loguea ingredientes que el LLM genera pero que el sistema de sinónimos no reconoce.
    Se guardan en la tabla `unknown_ingredients` para revisión periódica y expansión del catálogo.
    Usa RPC atómico con fallback a upsert clásico.
    """
    if not supabase or not user_id or user_id == "guest" or not unknown_ings:
        return
    
    try:
        for ing in unknown_ings[:20]:  # Cap a 20 por plan para no saturar
            raw_text = raw_map.get(ing, "") if raw_map else ""
            try:
                # Intentar RPC atómico
                supabase.rpc("log_unknown_ingredient_rpc", {
                    "p_user_id": user_id,
                    "p_ingredient": ing,
                    "p_raw_text": raw_text or None
                }).execute()
            except Exception as rpc_e:
                err = str(rpc_e)
                if "Could not find the function" in err or "PGRST202" in err or "unknown_ingredients" in err:
                    # RPC o tabla no desplegados aún → silenciar
                    return
                # Fallback: upsert directo
                try:
                    from datetime import datetime, timezone
                    supabase.table("unknown_ingredients").upsert({
                        "user_id": user_id,
                        "ingredient": ing,
                        "raw_text": raw_text or None,
                        "occurrences": 1,
                        "last_seen": datetime.now(timezone.utc).isoformat()
                    }, on_conflict="user_id,ingredient").execute()
                except Exception as fb_e:
                    if "unknown_ingredients" in str(fb_e) or "PGRST205" in str(fb_e):
                        return  # Tabla no existe aún → silenciar
                    logger.error(f"⚠️ [UNKNOWN ING] Error en fallback: {fb_e}")
                    return
        
        logger.info(f"📝 [UNKNOWN ING] {len(unknown_ings)} ingredientes no reconocidos logueados para revisión")
    except Exception as e:
        logger.error(f"⚠️ [UNKNOWN ING] Error logueando ingredientes desconocidos: {e}")

def increment_ingredient_frequencies(user_id: str, ingredients: list[str]):
    """Incrementa la frecuencia histórica de los ingredientes consumidos por un usuario.
    Intenta usar un RPC atómico O(1) robusto ante Race Conditions,
    con fallback al viejo método Select+Upsert si la función SQL no se ha creado.
    """
    if not supabase or not user_id or user_id == "guest": return
    
    try:
        from collections import Counter
        from datetime import datetime, timezone
        
        # strip_accents is imported globally
            
        normalized_ings = [strip_accents(i.lower()).strip() for i in ingredients if i]
        if not normalized_ings: return
        
        incoming_counts = Counter(normalized_ings)
        ingredients_list = list(incoming_counts.keys())
        counts_list = list(incoming_counts.values())
        
        # 1. Intentar método atómico (RPC) para evitar Race Conditions
        try:
            supabase.rpc("increment_ingredient_frequencies_rpc", {
                "p_user_id": user_id,
                "p_ingredients": ingredients_list,
                "p_counts": counts_list
            }).execute()
            logger.info(f"✅ [DB] Frecuencia atómica (RPC) incrementada para {user_id} ({len(ingredients_list)} items)")
            return
        except Exception as rpc_e:
            error_msg = str(rpc_e)
            if "Could not find the function" in error_msg or "PGRST202" in error_msg:
                # El usuario aún no corre el código SQL en Supabase, pasamos al fallback silenciosamente
                pass
            else:
                logger.warning(f"⚠️ [DB] Aviso de RPC, recurriendo a fallback... Detalles: {rpc_e}")

        # 2. Fallback clásico: Leer estado actual y luego hacer upsert
        # ⚠️ RACE CONDITION: Si dos requests concurrentes leen el mismo count antes de que
        # cualquiera escriba, un incremento se pierde (lost update).
        # En producción, desplegar el RPC `increment_ingredient_frequencies_rpc` en Supabase
        # para garantizar atomicidad. Este fallback solo existe para desarrollo local.
        res = supabase.table("ingredient_frequencies").select("ingredient, count").eq("user_id", user_id).execute()
        current_map = {row["ingredient"]: row["count"] for row in res.data} if res.data else {}
        
        upsert_rows = []
        now_str = datetime.now(timezone.utc).isoformat()
        
        for ing, inc_val in incoming_counts.items():
            new_val = current_map.get(ing, 0) + inc_val
            upsert_rows.append({
                "user_id": user_id,
                "ingredient": ing,
                "count": new_val,
                "last_used": now_str
            })
            
        if upsert_rows:
            supabase.table("ingredient_frequencies").upsert(upsert_rows).execute()
            logger.info(f"✅ [DB] Frecuencia (Fallback Clásico) incrementada para {user_id} ({len(upsert_rows)} items)")
            
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
    if not supabase or not user_id or user_id == "guest": return {}
    try:
        from datetime import datetime, timedelta, timezone
        from constants import INGREDIENT_FATIGUE_DECAY_FACTOR as decay_factor

        now = datetime.now(timezone.utc)
        cutoff_date = (now - timedelta(days=days_limit)).isoformat()

        res = supabase.table("ingredient_frequencies").select("ingredient, count, last_used").eq("user_id", user_id).gte("last_used", cutoff_date).execute()

        if not res.data:
            return {}

        freq_dict = {}

        for row in res.data:
            ingredient = row["ingredient"]
            count = row["count"]
            last_used_str = row.get("last_used")
            
            if not last_used_str:
                freq_dict[ingredient] = count
                continue
                
            try:
                # Parse robusto para last_used asumiendo formato de Supabase
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
    if not supabase: return []
    try:
        res = supabase.rpc("match_similar_plan", {
            "query_embedding": query_embedding,
            "match_threshold": threshold,
            "match_count": limit
        }).execute()
        return res.data
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
        Vacío `{}` si Supabase no está disponible o el query falló.

    Best-effort: cualquier excepción se loguea como warning y devuelve {}.
    El startup NO debe fallar si este probe falla — es observabilidad pura.
    """
    if not supabase:
        return {}
    try:
        # COUNT exact total para contexto.
        total_res = supabase.table("meal_plans").select("id", count="exact").limit(1).execute()
        total = total_res.count or 0
        if total == 0:
            return {"stale_count": 0, "stale_versions": {}, "total": 0}

        # Bucket por versión via SQL crudo: el supabase-py no expone GROUP BY
        # sobre extracciones de JSONB de forma fluida, y el RPC sería overkill.
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
