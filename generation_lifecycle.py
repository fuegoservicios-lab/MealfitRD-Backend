"""[P1-ARQ25-F1-LIFECYCLE · 2026-09-02] Lifecycle único de generación (roadmap 2.5, Fase 1).

SSOT de la Fase 1 del roadmap «Núcleo único»
(docs/superpowers/plans/2026-08-29-bioboros-v22-v24-roadmap-maestro.md §5):

  · I9  idempotencia de la solicitud   → `create_or_replay_run`
  · I10 fencing                        → `FENCING_TOKEN_COLUMN = "attempts"` (ya era el CAS del worker)
  · I12 revisión monotónica            → trigger `meal_plans_bump_revision_trg` (migración arq25_f1)
  · I19 autoridad única                → el Bloque 1 es un chunk (`INITIAL_CHUNK_KIND`) de
                                         `plan_chunk_queue`, ejecutado por `_chunk_worker`
  · H5  disponibilidad honesta         → `derive_availability` jamás devuelve PLAN_READY sin días

Diseño (decisión registrada en el roadmap §18, 2026-09-01/02):

  * `plan_chunk_queue` se EVOLUCIONA, no se duplica. El Bloque 1 entra como fila con
    `chunk_kind='initial'`, `week_number=1`, `days_offset=0`. Antes de encolarla se crea un
    PLACEHOLDER en `meal_plans` (`generation_status='generating'`, `days=[]`): el pickup usa
    `meal_plan_id NOT IN (...)`, y un `NULL` ahí anula el predicado para TODA la cola, así que
    la fila necesita un plan_id real desde el nacimiento (I1: nace de un INSERT backend).
  * El worker, al ver `chunk_kind='initial'`, delega en `run_initial_chunk` (aquí) que
    reutiliza EL MISMO postprocess que el SSE legacy (`_postprocess_pipeline_result`) con
    `existing_plan_id` para RELLENAR el placeholder en vez de insertar otro plan.
  * `run_status` y `availability` se DERIVAN de columnas existentes (`derive_run_status`,
    `derive_availability`). Almacenarlos sería la sexta fuente de verdad del lifecycle.
  * El progreso al cliente viaja por `app_kv_store` (`run_progress:<run_id>`), publicado por
    el worker con throttle; el SSE `/generation-runs/{id}/events` sólo lo TAILEA. Si el
    cliente se cae, el worker no se entera y la verdad sigue en DB.

Knobs:
  MEALFIT_INITIAL_VIA_QUEUE            (False)  — el interruptor de la Fase 1. OFF ⇒ el
                                                  endpoint nuevo responde 404 y nada cambia.
  MEALFIT_INITIAL_CHUNK_MAX_ATTEMPTS   (2)      — reintentos del chunk 0 antes de dead-letter.
  MEALFIT_RUN_PROGRESS_THROTTLE_S      (1.0)    — mínimo entre escrituras de progreso al KV.
  MEALFIT_SHUTDOWN_DRAIN_S             (90)     — espera cooperativa del worker en SIGTERM.

Sin DDL en runtime. Sin LLM dentro de transacción (I11): `run_plan_pipeline` corre fuera de
cualquier `conn.transaction()`; las escrituras van en llamadas cortas separadas.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from knobs import _env_bool, _env_float, _env_int

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes / knobs
# ---------------------------------------------------------------------------

#: Valor de `plan_chunk_queue.chunk_kind` del Bloque 1 encolado. Los valores previos
#: (`initial_plan`, `rolling_refill`, `catchup`) conservan su semántica; éste es NUEVO.
INITIAL_CHUNK_KIND = "initial"

#: I10 — el token de fencing del worker. El commit del chunk es
#: `UPDATE plan_chunk_queue SET ... WHERE id = %s AND attempts = %s AND status = 'processing'`
#: (`cron_tasks._cas_update_chunk_status`, T2 commit). El zombie rescue incrementa
#: `attempts` al re-encolar, así que un worker desplazado hace un UPDATE de 0 filas.
#: No se añade `lease_token`: sería un segundo token para la misma pregunta.
FENCING_TOKEN_COLUMN = "attempts"

#: Prefijo del KV de progreso (tabla `app_kv_store`).
RUN_PROGRESS_KV_PREFIX = "run_progress:"

#: Estados derivados del run (roadmap §5.2). Derivados, nunca persistidos.
RUN_PENDING = "PENDING"
RUN_RUNNING = "RUNNING"
RUN_WAITING_RETRY = "WAITING_RETRY"
RUN_WAITING_USER = "WAITING_USER"
RUN_PAUSED = "PAUSED"
RUN_COMPLETED = "COMPLETED"
RUN_FAILED = "FAILED"
RUN_CANCELLED = "CANCELLED"

AVAIL_NONE = "NONE"
AVAIL_PREVIEW_READY = "PREVIEW_READY"
AVAIL_PLAN_READY = "PLAN_READY"

#: Estado del placeholder mientras el chunk 0 no ha rellenado el plan.
PLACEHOLDER_GENERATION_STATUS = "generating"


def initial_via_queue_enabled(user_id: str | None = None) -> bool:
    """Interruptor de la Fase 1. Se lee en cada llamada (rollback sin redeploy).

    Dos capas, por el canary del roadmap (§14.1: «cuenta del dueño → usuarios de test → knob
    global»): `MEALFIT_INITIAL_VIA_QUEUE=true` enciende para todos; si está apagado,
    `MEALFIT_INITIAL_VIA_QUEUE_USERS` (uuids separados por coma) enciende SOLO para esos
    usuarios. El frontend puede llevar su flag encendido para todos: a quien no esté en la
    lista el endpoint le responde 404 y cae al SSE legacy en el mismo intento.
    """
    if _env_bool("MEALFIT_INITIAL_VIA_QUEUE", False):
        return True
    allow = os.environ.get("MEALFIT_INITIAL_VIA_QUEUE_USERS", "") or ""
    if user_id and allow.strip():
        allowed = {u.strip().lower() for u in allow.split(",") if u.strip()}
        return str(user_id).strip().lower() in allowed
    return False


def initial_chunk_max_attempts() -> int:
    return _env_int("MEALFIT_INITIAL_CHUNK_MAX_ATTEMPTS", 2, validator=lambda v: 1 <= v <= 5)


def shutdown_drain_seconds() -> int:
    return _env_int("MEALFIT_SHUTDOWN_DRAIN_S", 90, validator=lambda v: 0 <= v <= 600)


def preview_ready_days() -> int:
    """Decisión #1 del roadmap §16: hoy PREVIEW_READY = PLAN_CHUNK_SIZE (3) por construcción.

    Se expone como función (no constante) para que cambiar la decisión sea un knob y no
    una búsqueda por el código. `MEALFIT_PREVIEW_READY_DAYS` ausente ⇒ PLAN_CHUNK_SIZE.
    """
    from constants import PLAN_CHUNK_SIZE
    return _env_int("MEALFIT_PREVIEW_READY_DAYS", int(PLAN_CHUNK_SIZE), validator=lambda v: 1 <= v <= 30)


def worker_execution_id() -> str:
    """`claimed_by`: identifica una EJECUCIÓN concreta del worker, no el proceso reutilizable."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# I9 — idempotencia de la solicitud
# ---------------------------------------------------------------------------

class RunFingerprintConflict(Exception):
    """Misma `idempotency_key`, cuerpo distinto. El caller responde 409."""

    def __init__(self, run_id: str, existing_fingerprint: str):
        super().__init__(f"idempotency_key reutilizada con otro cuerpo (run {run_id})")
        self.run_id = run_id
        self.existing_fingerprint = existing_fingerprint


_FINGERPRINT_VOLATILE_KEYS = frozenset({
    "idempotency_key", "session_id", "tzOffset", "_client_ts", "_nonce",
})


def request_fingerprint(data: dict) -> str:
    """sha256 del cuerpo canónico. Excluye claves volátiles y las internas (`_*`)."""
    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: _clean(v) for k, v in sorted(obj.items())
                if k not in _FINGERPRINT_VOLATILE_KEYS and not str(k).startswith("_")
            }
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj
    canon = json.dumps(_clean(data or {}), sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def create_or_replay_run(
    *,
    user_id: str,
    idempotency_key: str,
    fingerprint: str,
    requested_days: int,
    market_country: Optional[str],
    locale: Optional[str],
    input_snapshot: dict,
    correlation_id: Optional[str] = None,
    policy: Optional[dict] = None,
) -> tuple[dict, bool]:
    """Crea el run o devuelve el existente (misma clave + mismo cuerpo).

    Returns: (run_row, created). Lanza `RunFingerprintConflict` si la clave existe con
    otro fingerprint. Una sola sentencia atómica: `INSERT ... ON CONFLICT DO NOTHING
    RETURNING`; si no devuelve fila, se lee la existente. La clave NO caduca al terminar.
    """
    from db_core import execute_sql_query, execute_sql_write
    from psycopg.types.json import Jsonb

    # [P1-ARQ25-F2-PLANPOLICY · 2026-09-02] requested/effective/relaxations/hash viajan en el
    # mismo INSERT (§6.4): el run nace explicable. Sin política (knob off) quedan los defaults.
    _pol = policy if isinstance(policy, dict) else {}
    rows = execute_sql_write(
        """
        INSERT INTO plan_generation_runs
            (user_id, idempotency_key, request_fingerprint, requested_days,
             market_country, locale, input_snapshot, correlation_id,
             requested_policy, effective_policy, relaxations, policy_hash,
             policy_schema_version, engine_versions)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id, idempotency_key) DO NOTHING
        RETURNING id, user_id, plan_id, idempotency_key, request_fingerprint,
                  requested_days, cancel_requested_at, created_at, completed_at
        """,
        (user_id, idempotency_key, fingerprint, int(requested_days),
         market_country, locale, Jsonb(input_snapshot or {}), correlation_id,
         Jsonb(_pol.get("requested") or {}), Jsonb(_pol.get("effective") or {}),
         Jsonb(_pol.get("relaxations") or []), _pol.get("policy_hash"),
         int(_pol.get("schema_version") or 0),
         Jsonb({"policy_compiler": _pol.get("compiler_version")} if _pol else {})),
        returning=True,
    )
    if rows:
        return dict(rows[0]), True

    existing = execute_sql_query(
        """
        SELECT id, user_id, plan_id, idempotency_key, request_fingerprint,
               requested_days, cancel_requested_at, created_at, completed_at,
               error_code, error_redacted
        FROM plan_generation_runs
        WHERE user_id = %s AND idempotency_key = %s
        """,
        (user_id, idempotency_key), fetch_one=True,
    )
    if not existing:  # carrera improbable: INSERT no devolvió y SELECT no ve → reintentar una vez
        raise RuntimeError("plan_generation_runs: INSERT sin RETURNING y SELECT vacío")
    if existing.get("request_fingerprint") != fingerprint:
        raise RunFingerprintConflict(str(existing["id"]), str(existing.get("request_fingerprint")))
    return dict(existing), False


def attach_plan_to_run(run_id: str, plan_id: str, user_id: str) -> None:
    from db_core import execute_sql_write
    execute_sql_write(
        "UPDATE plan_generation_runs SET plan_id = %s, updated_at = NOW() "
        "WHERE id = %s AND user_id = %s",
        (plan_id, run_id, user_id),
    )
    # I2: la mutación de meal_plans filtra user_id.
    execute_sql_write(
        "UPDATE meal_plans SET run_id = %s WHERE id = %s AND user_id = %s",
        (run_id, plan_id, user_id),
    )


def mark_run_error(run_id: str, code: str, detail: str | None = None, *, completed: bool = False) -> None:
    from db_core import execute_sql_write
    execute_sql_write(
        "UPDATE plan_generation_runs SET error_code = %s, error_redacted = %s, updated_at = NOW(), "
        "completed_at = CASE WHEN %s THEN NOW() ELSE completed_at END WHERE id = %s",
        (code[:64], (detail or "")[:240] or None, bool(completed), run_id),
    )


def clear_run_error(run_id: str) -> None:
    from db_core import execute_sql_write
    execute_sql_write(
        "UPDATE plan_generation_runs SET error_code = NULL, error_redacted = NULL, updated_at = NOW() WHERE id = %s",
        (run_id,),
    )


def mark_run_completed(run_id: str) -> None:
    from db_core import execute_sql_write
    execute_sql_write(
        "UPDATE plan_generation_runs SET completed_at = COALESCE(completed_at, NOW()), "
        "error_code = NULL, error_redacted = NULL, updated_at = NOW() WHERE id = %s",
        (run_id,),
    )


def request_run_cancel(run_id: str, user_id: str) -> dict:
    """Cancelación cooperativa: marca el run y cancela los chunks que AÚN no corren.

    Un chunk `processing` no se toca aquí (su worker es el dueño): `run_initial_chunk`
    consulta `cancel_requested_at` antes de persistir y aborta por sí mismo.
    """
    from db_core import execute_sql_write
    execute_sql_write(
        "UPDATE plan_generation_runs SET cancel_requested_at = COALESCE(cancel_requested_at, NOW()), "
        "updated_at = NOW() WHERE id = %s AND user_id = %s",
        (run_id, user_id),
    )
    rows = execute_sql_write(
        "UPDATE plan_chunk_queue SET status = 'cancelled', updated_at = NOW() "
        "WHERE run_id = %s AND user_id = %s AND status IN ('pending', 'stale', 'failed', 'pending_user_action') "
        "RETURNING id",
        (run_id, user_id), returning=True,
    ) or []
    return {"run_id": run_id, "cancelled_chunks": len(rows)}


def run_cancel_requested(run_id: str) -> bool:
    from db_core import execute_sql_query
    row = execute_sql_query(
        "SELECT cancel_requested_at FROM plan_generation_runs WHERE id = %s", (run_id,), fetch_one=True,
    )
    return bool(row and row.get("cancel_requested_at"))


# ---------------------------------------------------------------------------
# Estados derivados (§5.2) — funciones puras, testeables sin DB
# ---------------------------------------------------------------------------

_LIVE_CHUNK_STATUSES = ("pending", "processing", "stale")


def derive_availability(plan_data: dict | None, requested_days: int | None, *, preview_days: int | None = None) -> str:
    """NONE → PREVIEW_READY → PLAN_READY, derivado de los días REALES del plan.

    H5: `PLAN_READY` exige `days` no vacío aunque `generation_status` diga `complete`;
    el CHECK I8 lo garantiza en DB y esta función lo garantiza en la respuesta.
    """
    pd = plan_data if isinstance(plan_data, dict) else {}
    days = pd.get("days") if isinstance(pd.get("days"), list) else []
    n = len(days)
    if n <= 0:
        return AVAIL_NONE
    pv = preview_days if preview_days is not None else preview_ready_days()
    req = int(requested_days or pd.get("total_days_requested") or 0)
    status = str(pd.get("generation_status") or "")
    if req > 0 and n >= req:
        return AVAIL_PLAN_READY
    if status in ("complete", "complete_partial") and n > 0:
        # complete_partial: la cola murió; lo que hay es lo que habrá.
        return AVAIL_PLAN_READY
    if n >= pv:
        return AVAIL_PREVIEW_READY
    return AVAIL_NONE


def derive_run_status(
    *,
    plan_data: dict | None,
    chunk_rows: list[dict] | None,
    run_row: dict | None,
    plan_mode: str | None = None,
) -> str:
    """Estado del run derivado de `plan_chunk_queue` + `meal_plans` + `plan_generation_runs`.

    Orden de precedencia (el primero que aplica gana):
      CANCELLED  run.cancel_requested_at y ningún chunk vivo
      PAUSED     plan_mode='tracking' (H1) o generation_status paused_by_user
      WAITING_USER  algún chunk pending_user_action (H3) y ninguno corriendo
      RUNNING    algún chunk processing, o pending/stale ya vencido
      WAITING_RETRY  algún chunk failed no dead-lettered, o pending con execute_after futuro
      FAILED     chunk dead-lettered / generation_status failed / run.error sin días
      COMPLETED  disponibilidad PLAN_READY y ningún chunk vivo
      PENDING    lo demás (encolado, nadie lo ha tomado)
    """
    pd = plan_data if isinstance(plan_data, dict) else {}
    chunks = [c for c in (chunk_rows or []) if isinstance(c, dict)]
    run = run_row if isinstance(run_row, dict) else {}
    gen_status = str(pd.get("generation_status") or "")
    now = datetime.now(timezone.utc)

    def _st(c: dict) -> str:
        return str(c.get("status") or "")

    def _due(c: dict) -> bool:
        ea = c.get("execute_after")
        if ea is None:
            return True
        if isinstance(ea, str):
            try:
                ea = datetime.fromisoformat(ea.replace("Z", "+00:00"))
            except Exception:
                return True
        if getattr(ea, "tzinfo", None) is None:
            ea = ea.replace(tzinfo=timezone.utc)
        return ea <= now

    live = [c for c in chunks if _st(c) in _LIVE_CHUNK_STATUSES]
    processing = [c for c in chunks if _st(c) == "processing"]
    waiting_user = [c for c in chunks if _st(c) == "pending_user_action"]
    failed_retryable = [c for c in chunks if _st(c) == "failed" and not c.get("dead_lettered_at")]
    dead = [c for c in chunks if c.get("dead_lettered_at")]

    if run.get("cancel_requested_at") and not live:
        return RUN_CANCELLED
    if plan_mode == "tracking" or gen_status == "paused_by_user":
        return RUN_PAUSED
    if waiting_user and not processing:
        return RUN_WAITING_USER
    if processing or any(_due(c) for c in live):
        return RUN_RUNNING
    if failed_retryable or live:  # live aquí = pending/stale con execute_after futuro
        return RUN_WAITING_RETRY
    if dead or gen_status == "failed" or (run.get("error_code") and not pd.get("days")):
        return RUN_FAILED
    avail = derive_availability(pd, run.get("requested_days") or pd.get("total_days_requested"))
    if avail == AVAIL_PLAN_READY and not live:
        return RUN_COMPLETED
    if not chunks and not pd.get("days") and gen_status in ("", PLACEHOLDER_GENERATION_STATUS):
        return RUN_PENDING
    if avail != AVAIL_NONE and not live:
        # Hay días pero no todos y no queda cola: la generación terminó como pudo.
        return RUN_COMPLETED if gen_status in ("complete", "complete_partial") else RUN_FAILED
    return RUN_PENDING


# ---------------------------------------------------------------------------
# Progreso (KV) — el worker publica, el SSE tailea
# ---------------------------------------------------------------------------

def _progress_key(run_id: str) -> str:
    return f"{RUN_PROGRESS_KV_PREFIX}{run_id}"


class RunProgressPublisher:
    """Callback compatible con `progress_callback` de `run_plan_pipeline`.

    Escribe en `app_kv_store` con throttle (`MEALFIT_RUN_PROGRESS_THROTTLE_S`, 1 s) el
    ÚLTIMO evento + un contador monótono `seq`, para que el tail SSE deduplique.
    Fail-open: un error de DB no interrumpe la generación.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self._seq = 0
        self._last_write = 0.0
        self._lock = threading.Lock()
        self._pending: dict | None = None
        self._throttle = _env_float("MEALFIT_RUN_PROGRESS_THROTTLE_S", 1.0, validator=lambda v: 0.0 <= v <= 30.0)

    def __call__(self, event: dict) -> None:
        try:
            if not isinstance(event, dict):
                return
            with self._lock:
                self._seq += 1
                self._pending = {"seq": self._seq, "event": event}
                now = time.monotonic()
                if now - self._last_write < self._throttle:
                    return
                self._last_write = now
                payload = self._pending
                self._pending = None
            _write_progress(self.run_id, payload)
        except Exception as e:  # fail-open
            logger.debug(f"[ARQ25-F1/PROGRESS] publish no-op: {type(e).__name__}: {e}")

    def flush(self, final_event: dict | None = None) -> None:
        try:
            with self._lock:
                if final_event is not None:
                    self._seq += 1
                    self._pending = {"seq": self._seq, "event": final_event}
                payload = self._pending
                self._pending = None
            if payload is not None:
                _write_progress(self.run_id, payload)
        except Exception as e:
            logger.debug(f"[ARQ25-F1/PROGRESS] flush no-op: {type(e).__name__}: {e}")


def _write_progress(run_id: str, payload: dict) -> None:
    from db_core import execute_sql_write
    from psycopg.types.json import Jsonb
    doc = {**payload, "updated_at": datetime.now(timezone.utc).isoformat()}
    execute_sql_write(
        "INSERT INTO app_kv_store (key, value, updated_at) VALUES (%s, %s, NOW()) "
        "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = NOW()",
        (_progress_key(run_id), Jsonb(doc)),
    )


def read_run_progress(run_id: str) -> dict | None:
    from db_core import execute_sql_query
    row = execute_sql_query("SELECT value FROM app_kv_store WHERE key = %s", (_progress_key(run_id),), fetch_one=True)
    if not row:
        return None
    val = row.get("value")
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            return None
    return val if isinstance(val, dict) else None


def clear_run_progress(run_id: str) -> None:
    from db_core import execute_sql_write
    execute_sql_write("DELETE FROM app_kv_store WHERE key = %s", (_progress_key(run_id),))


# ---------------------------------------------------------------------------
# Snapshot durable del run (lo que lee GET /generation-runs/{id} y el tail SSE)
# ---------------------------------------------------------------------------

def load_run_snapshot(run_id: str, user_id: str, *, include_plan: bool = False) -> dict | None:
    """Estado derivado + progreso. `None` si el run no existe o no es del usuario (→ 404)."""
    from db_core import execute_sql_query

    run = execute_sql_query(
        "SELECT id, user_id, plan_id, requested_days, cancel_requested_at, created_at, "
        "completed_at, error_code, error_redacted FROM plan_generation_runs "
        "WHERE id = %s AND user_id = %s",
        (run_id, user_id), fetch_one=True,
    )
    if not run:
        return None
    plan_id = str(run["plan_id"]) if run.get("plan_id") else None
    plan_row = None
    plan_data: dict = {}
    revision = None
    if plan_id:
        plan_row = execute_sql_query(
            "SELECT plan_data, revision, created_at FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, user_id), fetch_one=True,
        )
        if plan_row:
            pd = plan_row.get("plan_data")
            if isinstance(pd, str):
                try:
                    pd = json.loads(pd)
                except Exception:
                    pd = {}
            plan_data = pd if isinstance(pd, dict) else {}
            revision = plan_row.get("revision")
    chunks = []
    if plan_id:
        chunks = execute_sql_query(
            "SELECT id, week_number, chunk_kind, status, attempts, execute_after, dead_lettered_at, "
            "dead_letter_reason, updated_at FROM plan_chunk_queue WHERE meal_plan_id = %s "
            "ORDER BY week_number ASC",
            (plan_id,), fetch_all=True,
        ) or []
    pm = execute_sql_query("SELECT plan_mode FROM user_profiles WHERE id = %s", (user_id,), fetch_one=True) or {}
    status = derive_run_status(plan_data=plan_data, chunk_rows=chunks, run_row=run, plan_mode=pm.get("plan_mode"))
    availability = derive_availability(plan_data, run.get("requested_days"))
    days = plan_data.get("days") if isinstance(plan_data.get("days"), list) else []
    snap: dict[str, Any] = {
        "run_id": str(run["id"]),
        "plan_id": plan_id,
        "status": status,
        "availability": availability,
        "requested_days": run.get("requested_days"),
        "days_generated": len(days),
        "revision": revision,
        "generation_status": plan_data.get("generation_status"),
        "error_code": run.get("error_code"),
        "error_message": run.get("error_redacted"),
        "cancel_requested": bool(run.get("cancel_requested_at")),
        "chunks": [
            {
                "id": str(c["id"]), "week_number": c.get("week_number"), "chunk_kind": c.get("chunk_kind"),
                "status": c.get("status"), "attempts": c.get("attempts"),
                "execute_after": c.get("execute_after").isoformat() if hasattr(c.get("execute_after"), "isoformat") else c.get("execute_after"),
                "dead_letter_reason": c.get("dead_letter_reason"),
            } for c in chunks
        ],
        "progress": read_run_progress(str(run["id"])),
        "created_at": run.get("created_at").isoformat() if hasattr(run.get("created_at"), "isoformat") else run.get("created_at"),
        "completed_at": run.get("completed_at").isoformat() if hasattr(run.get("completed_at"), "isoformat") else run.get("completed_at"),
    }
    if include_plan and plan_id and availability != AVAIL_NONE:
        snap["plan"] = {**plan_data, "id": plan_id, "revision": revision}
    return snap


# ---------------------------------------------------------------------------
# Encolado del Bloque 1 (endpoint) — placeholder + chunk 0 + despertar al worker
# ---------------------------------------------------------------------------

def _chunk_input_hash_f3(form_data: dict) -> str:
    """[P1-ARQ25-F3-HORIZON] `input_hash` = huella del formulario + hash de `_blueprint_slice`."""
    fp = request_fingerprint(form_data or {})
    try:
        from horizon import chunk_input_hash
        return chunk_input_hash(fp, (form_data or {}).get("_blueprint_slice"))
    except Exception:
        return fp


def create_placeholder_plan_and_enqueue_initial(
    *,
    user_id: str,
    run_id: str,
    snapshot: dict,
    total_days_requested: int,
    days_count: int,
) -> tuple[str, str | None]:
    """Crea el placeholder (`save_new_meal_plan_atomic`: cancela chunks vivos del usuario en
    la MISMA transacción, como un plan nuevo de hoy) y encola el chunk 0. Devuelve
    (plan_id, chunk_id)."""
    from db_plans import save_new_meal_plan_atomic
    from cron_tasks import _enqueue_plan_chunk
    from db_core import execute_sql_write

    placeholder = {
        "user_id": user_id,
        "plan_data": {
            "days": [],
            "generation_status": PLACEHOLDER_GENERATION_STATUS,
            "total_days_requested": int(total_days_requested),
            "_run_id": run_id,
            "_placeholder_created_at": datetime.now(timezone.utc).isoformat(),
        },
        "name": "Plan en preparación",
        "calories": 0,
        "macros": {},
        "meal_names": [],
        "ingredients": [],
    }
    plan_id = save_new_meal_plan_atomic(user_id, placeholder, return_id=True)
    if not plan_id:
        raise RuntimeError("placeholder meal_plans INSERT no devolvió id")
    plan_id = str(plan_id)
    attach_plan_to_run(run_id, plan_id, user_id)

    chunk_snapshot = {**snapshot, "_run_id": run_id, "_initial": True}
    _enqueue_plan_chunk(user_id, plan_id, 1, 0, int(days_count), chunk_snapshot, chunk_kind=INITIAL_CHUNK_KIND)
    # `execute_after = NOW()`: `_enqueue_plan_chunk` programa los chunks con un margen
    # (medido 2026-09-02: +60 s), pensado para los bloques 2..N. El chunk 0 es la
    # generación que el usuario está MIRANDO: el wake del worker (`wake_chunk_worker`)
    # no sirve de nada si el pickup lo descarta por `execute_after <= NOW()`.
    rows = execute_sql_write(
        "UPDATE plan_chunk_queue SET run_id = %s, input_hash = %s, execute_after = NOW() "
        "WHERE meal_plan_id = %s AND week_number = 1 AND chunk_kind = %s AND status = 'pending' "
        "RETURNING id",
        # [P1-ARQ25-F3-HORIZON · 2026-09-02] huella del formulario + hash de la rebanada (§6.5).
        (run_id, _chunk_input_hash_f3(chunk_snapshot.get("form_data") or {}), plan_id, INITIAL_CHUNK_KIND),
        returning=True,
    ) or []
    chunk_id = str(rows[0]["id"]) if rows else None
    try:
        from cron_tasks import wake_chunk_worker
        wake_chunk_worker(reason=f"initial:{plan_id[:8]}")
    except Exception as e:  # sin líder local el tick del minuto lo recoge igual
        logger.debug(f"[ARQ25-F1] wake_chunk_worker no-op: {type(e).__name__}: {e}")
    return plan_id, chunk_id


# ---------------------------------------------------------------------------
# Ejecución del chunk 0 dentro del worker
# ---------------------------------------------------------------------------

def _run_background_tasks(bt) -> None:
    """Ejecuta en el hilo del worker lo que el postprocess encoló en `BackgroundTasks`
    (fuera de FastAPI no hay quien las corra)."""
    import asyncio
    import inspect
    for t in list(getattr(bt, "tasks", []) or []):
        try:
            fn = getattr(t, "func", None)
            if fn is None:
                continue
            args = getattr(t, "args", ()) or ()
            kwargs = getattr(t, "kwargs", {}) or {}
            if inspect.iscoroutinefunction(fn):
                asyncio.run(fn(*args, **kwargs))
            else:
                fn(*args, **kwargs)
        except Exception as e:
            logger.warning(f"[ARQ25-F1] background task {getattr(t, 'func', None)} falló: {type(e).__name__}: {e}")


def _placeholder_already_filled(plan_id: str, user_id: str) -> bool:
    """[P1-ARQ25-F1-CLOSE] True si el plan ya salió del estado placeholder con días persistidos."""
    try:
        from db_core import execute_sql_query
        row = execute_sql_query(
            "SELECT plan_data->>'generation_status' AS st, "
            "jsonb_array_length(coalesce(plan_data->'days', '[]'::jsonb)) AS n "
            "FROM meal_plans WHERE id = %s AND user_id = %s",
            (plan_id, user_id), fetch_one=True,
        )
    except Exception as e:  # sin DB ⇒ conducta previa (regenerar)
        logger.warning(f"[ARQ25-F1] _placeholder_already_filled no-op: {e}")
        return False
    if not row:
        return False
    return str(row.get("st") or "") != PLACEHOLDER_GENERATION_STATUS and int(row.get("n") or 0) > 0


def _emit_lifecycle_metric(name: str, user_id, meta: dict | None = None) -> None:
    """[P1-ARQ25-F1-CLOSE] §13.2 métricas de durabilidad (`fencing_rejected`, …) en
    `pipeline_metrics`, best-effort: nunca rompe el worker."""
    try:
        from db_core import execute_sql_write
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, "
            "tokens_estimated, confidence, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
            (user_id, "__lifecycle__", f"arq25_{name}", 0, 0, 0, None, json.dumps(meta or {}, default=str)),
        )
    except Exception as e:
        logger.debug(f"[ARQ25-F1] métrica {name} no persistida: {e}")


def run_initial_chunk(*, task: dict, snap: dict, form_data: dict, pickup_attempts: int) -> None:
    """Cuerpo del Bloque 1 cuando lo ejecuta `_chunk_worker` (chunk_kind='initial').

    Reproduce el orden del SSE legacy: pipeline → guard de fallback → validación de
    nevera → postprocess (que RELLENA el placeholder vía `existing_plan_id`) → CAS de
    commit. Nunca lanza hacia el handler genérico del worker: cada rama termina en un
    CAS explícito (`_cas_update_chunk_status`) o en `completed`.
    """
    from fastapi import BackgroundTasks
    from cron_tasks import _cas_update_chunk_status
    from db_core import execute_sql_write
    from graph_orchestrator import run_plan_pipeline

    task_id = task["id"]
    user_id = str(task["user_id"])
    plan_id = str(task["meal_plan_id"])
    run_id = str(snap.get("_run_id") or "")
    raw_data = snap.get("raw_data") or {}
    history = snap.get("history") or []
    taste_profile = snap.get("taste_profile") or ""
    memory_ctx = snap.get("memory_context") or ""
    rejected_meal_names = snap.get("rejected_meal_names") or []
    total_days_requested = int(snap.get("totalDays") or raw_data.get("totalDays") or 3)
    use_chunking = bool(snap.get("use_chunking"))
    plan_start_date = str(form_data.get("_plan_start_date") or "")
    tz_offset_mins = int(form_data.get("tz_offset_minutes") or 0)
    session_id = snap.get("session_id")
    # [P1-ARQ25-F3-HORIZON · 2026-09-02] el flag `enforce` se recalcula al ejecutar (canary/knob
    # se leen en cada llamada); la rebanada del snapshot es inmutable.
    if isinstance(form_data.get("_blueprint_slice"), dict):
        try:
            from horizon import policy_enforced as _policy_enforced_f3
            form_data["_policy_enforced"] = _policy_enforced_f3(user_id)
        except Exception:
            form_data["_policy_enforced"] = False
    max_attempts = initial_chunk_max_attempts()
    exec_id = worker_execution_id()

    try:
        execute_sql_write(
            "UPDATE plan_chunk_queue SET claimed_by = %s WHERE id = %s AND attempts = %s AND status = 'processing'",
            (exec_id, task_id, int(pickup_attempts)),
        )
    except Exception:
        pass

    publisher = RunProgressPublisher(run_id) if run_id else None
    if publisher:
        publisher({"event": "status", "data": {"phase": "started", "attempt": int(pickup_attempts) + 1}})

    # [P1-ARQ25-F1-CLOSE · 2026-09-02] §13.2 «crash después de commit y antes de encolar».
    # Si el proceso murió tras `fill_placeholder_meal_plan_atomic` y antes del CAS `completed`,
    # el zombie rescue devuelve el chunk a `pending` con attempts+1 y este worker lo reclama:
    # el placeholder YA tiene días. Sin este guard volvíamos a correr el pipeline (gasto LLM
    # duplicado) y el fill devolvía None (`status != generating`) ⇒ `persist_failed` en bucle
    # hasta dead-letter. Con días persistidos, el trabajo está hecho: se cierra el CAS.
    if _placeholder_already_filled(plan_id, user_id):
        logger.warning(f"[ARQ25-F1] chunk 0 {str(task_id)[:8]} plan={plan_id[:8]}: placeholder ya rellenado "
                       f"(crash post-commit); CAS completed sin regenerar.")
        ok = _cas_update_chunk_status(task_id, int(pickup_attempts), "completed")
        if ok and run_id:
            clear_run_error(run_id)
            if not snap.get("use_chunking", True):
                mark_run_completed(run_id)
        elif not ok:
            _emit_lifecycle_metric("fencing_rejected", user_id, {"plan_id": plan_id, "site": "post_commit_replay"})
        if publisher:
            publisher.flush({"event": "complete", "data": {"plan_id": plan_id, "replayed": True}})
        return

    def _fail(code: str, msg: str, *, terminal: bool) -> None:
        exhausted = terminal or (int(pickup_attempts) + 1 >= max_attempts)
        extra = {"attempts": "attempts + 1"}
        if exhausted:
            extra["dead_lettered_at"] = "NOW()"
            extra["dead_letter_reason"] = "'initial_" + code.replace("'", "") + "'"
            new_status = "failed"
        else:
            extra["execute_after"] = "NOW() + INTERVAL '30 seconds'"
            new_status = "pending"
        ok = _cas_update_chunk_status(task_id, int(pickup_attempts), new_status, extra_set_clauses=extra)
        if not ok:
            logger.warning(f"[ARQ25-F1] chunk 0 {str(task_id)[:8]} desplazado (CAS 0 filas) al fallar: {code}")
            _emit_lifecycle_metric("fencing_rejected", user_id, {"plan_id": plan_id, "site": "fail", "code": code})
            return
        if run_id:
            mark_run_error(run_id, code, msg, completed=exhausted)
        if exhausted:
            # El placeholder no puede quedarse en 'generating' para siempre.
            try:
                execute_sql_write(
                    "UPDATE meal_plans SET plan_data = jsonb_set(plan_data, '{generation_status}', '\"failed\"') "
                    "WHERE id = %s AND user_id = %s AND plan_data->>'generation_status' = %s",
                    (plan_id, user_id, PLACEHOLDER_GENERATION_STATUS),
                )
            except Exception as e:
                logger.warning(f"[ARQ25-F1] placeholder→failed no-op: {e}")
            try:
                from db_plans import upsert_pending_pipeline
                upsert_pending_pipeline(user_id, status="failed", error=code[:200])
            except Exception:
                pass
        if publisher:
            publisher.flush({"event": "error", "data": {"code": code, "message": msg, "terminal": exhausted}})

    # 1) Pipeline (fuera de toda transacción — I11)
    try:
        form_data["_caller_target_plan_id"] = plan_id
        form_data["_caller_context"] = "chunk_worker:initial"
        result = run_plan_pipeline(form_data, history, taste_profile, memory_ctx, publisher, None)
    except Exception as e:
        logger.exception(f"[ARQ25-F1] pipeline del chunk 0 falló plan={plan_id[:8]}: {e}")
        _fail("pipeline_error", str(e)[:200], terminal=False)
        return

    if not isinstance(result, dict) or not result.get("days"):
        _fail("empty_result", "el pipeline no devolvió días", terminal=False)
        return

    # 2) Guard de fallback (mismo criterio que el SSE legacy)
    if result.get("_is_fallback") and not result.get("_partial_repair"):
        code = "critical_restriction" if result.get("_critical_rejection") else "llm_unavailable_fallback"
        msg = result.get("_review_disclaimer") or (
            "No pudimos generar un plan que respete tus restricciones declaradas."
            if result.get("_critical_rejection") else "El generador no estuvo disponible; inténtalo de nuevo."
        )
        _fail(code, str(msg)[:200], terminal=bool(result.get("_critical_rejection")))
        return

    # 3) Cancelación cooperativa antes de persistir
    if run_id and run_cancel_requested(run_id):
        _cas_update_chunk_status(task_id, int(pickup_attempts), "cancelled")
        if publisher:
            publisher.flush({"event": "error", "data": {"code": "cancelled", "message": "Generación cancelada.", "terminal": True}})
        return

    # 4) Validación de nevera + postprocess (RELLENA el placeholder)
    from routers.plans import (
        _postprocess_pipeline_result,
        _resolve_live_pantry,
        _run_pantry_validation_for_initial_chunk,
    )
    bt = BackgroundTasks()
    try:
        result = _run_pantry_validation_for_initial_chunk(
            result=result, pipeline_data=form_data, history=history, taste_profile=taste_profile,
            memory_ctx=memory_ctx, background_tasks=bt, actual_user_id=user_id,
            pantry_ingredients=_resolve_live_pantry(user_id, raw_data),
            transport_label="P0-2 QUEUE", update_reason=raw_data.get("update_reason"),
        )
        result = _postprocess_pipeline_result(
            result=result, actual_user_id=user_id, session_id=session_id, data=raw_data,
            taste_profile=taste_profile, memory_ctx=memory_ctx, rejected_meal_names=rejected_meal_names,
            total_days_requested=total_days_requested, use_chunking=use_chunking,
            background_tasks=bt, plan_start_date=plan_start_date, tz_offset_mins=tz_offset_mins,
            transport_label="queue", existing_plan_id=plan_id,
        )
    except Exception as e:
        logger.exception(f"[ARQ25-F1] postprocess del chunk 0 falló plan={plan_id[:8]}: {e}")
        _fail("postprocess_error", str(e)[:200], terminal=False)
        return
    finally:
        _run_background_tasks(bt)

    if result.get("_persist_failed"):
        _fail("persist_failed", "no se pudo escribir el plan", terminal=False)
        return

    # 5) Commit con fencing (I10): attempts es el token.
    out_hash = hashlib.sha256(json.dumps(result.get("days", []), sort_keys=True, default=str).encode("utf-8")).hexdigest()
    ok = _cas_update_chunk_status(
        task_id, int(pickup_attempts), "completed",
        extra_set_clauses={"output_hash": "'" + out_hash + "'", "learning_persisted_at": "NOW()"},
    )
    if not ok:
        _emit_lifecycle_metric("fencing_rejected", user_id, {"plan_id": plan_id, "site": "commit"})
        logger.warning(f"[ARQ25-F1] chunk 0 {str(task_id)[:8]} completó pero el CAS devolvió 0 filas (desplazado). Plan ya persistido; el otro worker verá el placeholder relleno.")
        return
    if run_id:
        if not use_chunking:
            mark_run_completed(run_id)
        else:
            clear_run_error(run_id)  # el run sigue vivo (chunks 2..N); limpia el error de intentos previos
            # Los chunks 2..N los encoló `_enqueue_remaining_chunks` (postprocess legacy) sin
            # run_id: se estampan aquí para que `plan_generation_runs` ↔ cola quede enlazado
            # (medido en el primer run real: chunks 2..8 con run_id NULL).
            try:
                execute_sql_write(
                    "UPDATE plan_chunk_queue SET run_id = %s WHERE meal_plan_id = %s AND user_id = %s AND run_id IS NULL",
                    (run_id, plan_id, user_id),
                )
            except Exception as e:
                logger.debug(f"[ARQ25-F1] run_id en chunks 2..N no-op: {type(e).__name__}: {e}")
    try:
        from db_plans import upsert_pending_pipeline
        upsert_pending_pipeline(user_id, status="complete", plan_id_final=plan_id)
    except Exception:
        pass
    if publisher:
        publisher.flush({"event": "complete", "data": {"plan_id": plan_id, "days": len(result.get("days") or [])}})
    logger.info(f"✅ [ARQ25-F1] Bloque 1 persistido vía cola: plan={plan_id[:8]} run={run_id[:8]} días={len(result.get('days') or [])} chunked={use_chunking}")
