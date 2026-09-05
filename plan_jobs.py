"""[P1-ARQ25-F5-PLAN-JOBS · 2026-09-04] Outbox `plan_jobs` en producción (Fase 5, capa V2.2).

Motor SSOT del protocolo de proyecciones asíncronas del roadmap 2.5 (§5.1 «plan_jobs», §5.3
«protocolo de worker», §5.7 «consumidores»). La tabla existe desde la Fase 1
(`migrations/arq25_f1_lifecycle_expand_2026_09_02.sql`); la Fase 3 ya encola `shopping_projection`;
esta fase pone el WORKER y el primer consumidor, `display_i18n`.

Contrato (mismo protocolo que `_chunk_worker`, escrito para una tabla genérica):

  1. `claim_plan_jobs`: transacción corta, `FOR UPDATE SKIP LOCKED` sobre `status IN ('pending','failed')
     AND execute_after <= NOW()`; `attempts += 1`, `status='processing'`, `claimed_by`, `heartbeat_at`.
     **`attempts` es el token de fencing.**
  2. El consumidor corre FUERA de la DB (I11) y comprueba que `meal_plans.revision` sigue siendo la del
     job (I13): si cambió, el job termina `stale` y se re-encola para la revisión vigente.
  3. `finish_plan_job`: UPDATE fenced por `(id, claimed_by, attempts, status='processing')`. `failed`
     reintenta con backoff exponencial; al agotar `MEALFIT_PLAN_JOBS_MAX_ATTEMPTS` pasa a `dead`
     (`dead_lettered_at`, error redactado). 0 filas afectadas ⇒ `fencing_rejected` (métrica).
  4. `reclaim_stale_processing`: un `processing` sin heartbeat desde hace `MEALFIT_PLAN_JOBS_HEARTBEAT_STALE_S`
     vuelve a `failed` (o `dead` si agotó intentos). Cubre el deploy que mató al worker a mitad.

Semántica at-least-once: cada consumidor es idempotente (`enrich_plan_display` ya lo es: lock KV +
`jsonb_set` por comida + ownership `AND user_id`). Los `job_type` sin consumidor registrado NO se
reclaman (quedan `pending` intactos: `shopping_projection` hasta su rebanada).

Knobs (todos `MEALFIT_PLAN_JOBS_*`, registrados vía `knobs._env_*`):
  - `MEALFIT_PLAN_JOBS_ENABLED` (False): interruptor maestro del worker y del encolado.
  - `MEALFIT_PLAN_JOBS_DISPLAY_I18N` (True): consumidor de traducciones; apagado ⇒ el disparador vuelve
    al hilo legacy de `schedule_plan_display_enrichment`.
  - `MEALFIT_PLAN_JOBS_BATCH` (10), `MEALFIT_PLAN_JOBS_MAX_ATTEMPTS` (5),
    `MEALFIT_PLAN_JOBS_BACKOFF_BASE_S` (60), `MEALFIT_PLAN_JOBS_HEARTBEAT_STALE_S` (600),
    `MEALFIT_PLAN_JOBS_WORKER_INTERVAL_S` (60).

Fail-open total: ninguna función de este módulo lanza hacia el request path; el encolado que falla
devuelve None y el disparador cae al camino legacy.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from knobs import _env_bool, _env_int

logger = logging.getLogger(__name__)

JOB_TYPE_DISPLAY_I18N = "display_i18n"
JOB_TYPE_SHOPPING_PROJECTION = "shopping_projection"  # lo encola la Fase 3 (horizon.enqueue_shopping_projection_job)
JOB_STATUSES = ("pending", "processing", "done", "failed", "dead", "stale")
WORKER_JOB_ID = "process_plan_jobs"
METRIC_NODE = "plan_jobs"
_MAX_BACKOFF_S = 6 * 60 * 60


# ----------------------------------------------------------------------------- knobs
def _clamp(v: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except Exception:
        return lo


def plan_jobs_enabled() -> bool:
    return _env_bool("MEALFIT_PLAN_JOBS_ENABLED", False)


def consumer_enabled(job_type: str) -> bool:
    return _env_bool(f"MEALFIT_PLAN_JOBS_{str(job_type).upper()}", True)


def worker_batch() -> int:
    return _clamp(_env_int("MEALFIT_PLAN_JOBS_BATCH", 10), 1, 100)


def max_attempts() -> int:
    return _clamp(_env_int("MEALFIT_PLAN_JOBS_MAX_ATTEMPTS", 5), 1, 20)


def backoff_base_s() -> int:
    return _clamp(_env_int("MEALFIT_PLAN_JOBS_BACKOFF_BASE_S", 60), 5, 3600)


def heartbeat_stale_s() -> int:
    return _clamp(_env_int("MEALFIT_PLAN_JOBS_HEARTBEAT_STALE_S", 600), 60, 6 * 3600)


def worker_interval_s() -> int:
    return _clamp(_env_int("MEALFIT_PLAN_JOBS_WORKER_INTERVAL_S", 60), 15, 600)


def backoff_seconds(attempts: int) -> int:
    """Backoff exponencial con base `MEALFIT_PLAN_JOBS_BACKOFF_BASE_S`, tope 6 h. attempts=1 ⇒ base."""
    n = max(1, int(attempts or 1))
    return int(min(_MAX_BACKOFF_S, backoff_base_s() * (2 ** (n - 1))))


# ----------------------------------------------------------------------------- helpers puros
def _is_uuid(v: Any) -> bool:
    try:
        uuid.UUID(str(v))
        return True
    except Exception:
        return False


def _worker_identity() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


def display_i18n_dedup_key(plan_id: str, revision: Optional[int], locale: str, day_indices: Optional[list] = None) -> str:
    days = "all" if not day_indices else ",".join(str(int(d)) for d in sorted({int(d) for d in day_indices}))
    return f"{JOB_TYPE_DISPLAY_I18N}:{plan_id}:{int(revision or 0)}:{locale}:{days}"


# Vocabulario de `enrich_plan_display(...)["skipped"]` → veredicto del job. Lo que no hay que hacer
# es `done` (idempotencia: no hay nada que reintentar); lo transitorio es `failed` (reintento con
# backoff); `partial_loss` (un lote perdido tras fallo de invoke) TAMBIÉN reintenta: el lote que sí
# se escribió no se toca (jsonb_set por comida) y el perdido se recupera en el siguiente intento.
# Vocabulario completo = literales de `enrich_plan_display` + los valores de `last_skip_reason`
# (verificado contra producción 2026-09-05: `no_valid_meals` = todo ya traducido, NO un fallo).
_DONE_SKIPS = frozenset({"no_meals", "no_valid_meals", "no_days", "knob_off", "locale", "not_found"})
_RETRY_SKIPS = frozenset({
    "circuit_breaker_open", "dedupe_inprocess", "dedupe_locked", "exception", "partial_loss",
    "invocation_budget_exhausted", "json_parse_error", "llm_exception", "persist_stale_mismatch",
})


def verdict_for_display_result(result: Any) -> tuple[str, Optional[str]]:
    """(status, error_code) para el resultado de `enrich_plan_display`. Puro, sin DB."""
    if not isinstance(result, dict):
        return "failed", "bad_result"
    skipped = result.get("skipped")
    if skipped in _DONE_SKIPS:
        return "done", None
    if skipped in _RETRY_SKIPS:
        return "failed", str(skipped)
    if skipped:
        return "failed", str(skipped)[:64]
    return "done", None


# ----------------------------------------------------------------------------- DB: encolar
def current_plan_revision(plan_id: str) -> Optional[int]:
    try:
        from db import execute_sql_query
        row = execute_sql_query("SELECT revision FROM meal_plans WHERE id = %s", (plan_id,), fetch_one=True)
        if row and row.get("revision") is not None:
            return int(row["revision"])
    except Exception as e:
        logger.debug(f"[ARQ25-F5] current_plan_revision no disponible plan={plan_id}: {e!r}")
    return None


def enqueue_plan_job(
    job_type: str,
    plan_id: str,
    user_id: str,
    *,
    plan_revision: Optional[int],
    dedup_key: str,
    payload: Optional[dict] = None,
    execute_after: Optional[datetime] = None,
    wake: bool = True,
) -> Optional[str]:
    """INSERT idempotente por `dedup_key` (ON CONFLICT DO NOTHING). Devuelve el id nuevo, o None si ya
    existía (o si falló: fail-open). Despierta al worker si insertó."""
    if not plan_id or not _is_uuid(plan_id) or not _is_uuid(user_id):
        return None
    try:
        from db import execute_sql_write
        from psycopg.types.json import Jsonb
        rows = execute_sql_write(
            "INSERT INTO plan_jobs (job_type, plan_id, user_id, plan_revision, dedup_key, payload, execute_after) "
            "VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, NOW())) "
            "ON CONFLICT (dedup_key) DO NOTHING RETURNING id",
            (job_type, plan_id, user_id, plan_revision, dedup_key, Jsonb(payload or {}), execute_after),
            returning=True,
        ) or []
        job_id = str(rows[0]["id"]) if rows else None
        if job_id and wake:
            wake_plan_jobs_worker(f"enqueue:{job_type}")
        return job_id
    except Exception as e:
        logger.warning(f"[ARQ25-F5] enqueue_plan_job falló (fail-open) type={job_type} plan={plan_id}: {e!r}")
        return None


def maybe_enqueue_display_i18n(plan_id: str, user_id: str, locale: str, day_indices: Optional[list] = None) -> bool:
    """Puerta del disparador de traducciones: True ⇒ la cola se hace cargo (insertado O ya en cola);
    False ⇒ el llamador sigue por el hilo legacy. Guests (sin UUID de usuario) siempre False."""
    if not plan_jobs_enabled() or not consumer_enabled(JOB_TYPE_DISPLAY_I18N):
        return False
    if not _is_uuid(plan_id) or not _is_uuid(user_id) or not locale:
        return False
    try:
        revision = current_plan_revision(plan_id)
        days = None if day_indices is None else sorted({int(d) for d in day_indices})
        key = display_i18n_dedup_key(plan_id, revision, locale, days)
        payload = {"locale": str(locale), "day_indices": days, "schema_version": 1}
        job_id = enqueue_plan_job(
            JOB_TYPE_DISPLAY_I18N, plan_id, user_id,
            plan_revision=revision, dedup_key=key, payload=payload,
        )
        if job_id:
            logger.info(f"[ARQ25-F5] display_i18n encolado job={job_id} plan={plan_id} rev={revision} locale={locale} days={days}")
            return True
        # None puede ser «ya existía» (dedup) o «falló»: distinguir mirando la fila.
        from db import execute_sql_query
        row = execute_sql_query("SELECT status FROM plan_jobs WHERE dedup_key = %s", (key,), fetch_one=True)
        if row and row.get("status") in ("pending", "processing", "failed"):
            return True  # ya en cola: no duplicar por el hilo
        return False
    except Exception as e:
        logger.debug(f"[ARQ25-F5] maybe_enqueue_display_i18n cae a legacy: {e!r}")
        return False


# ----------------------------------------------------------------------------- DB: worker
def wake_plan_jobs_worker(reason: str = "") -> bool:
    """Adelanta el próximo tick de `process_plan_jobs` a AHORA (paridad con `wake_chunk_worker`)."""
    try:
        import cron_tasks as _ct
        sched = getattr(_ct, "_SCHEDULER_REF", None)
        if sched is None:
            return False
        job = sched.get_job(WORKER_JOB_ID)
        if job is None:
            return False
        job.modify(next_run_time=datetime.now(timezone.utc))
        return True
    except Exception as e:
        logger.debug(f"[ARQ25-F5] wake_plan_jobs_worker no-op: {type(e).__name__}: {e}")
        return False


CLAIM_SQL = """
WITH candidates AS (
    SELECT j.id FROM plan_jobs j
    WHERE j.status IN ('pending', 'failed')
      AND j.execute_after <= NOW()
      AND j.job_type = ANY(%s)
    ORDER BY j.execute_after ASC, j.created_at ASC
    LIMIT %s
    FOR UPDATE SKIP LOCKED
)
UPDATE plan_jobs p
SET status = 'processing',
    attempts = p.attempts + 1,
    claimed_by = %s,
    heartbeat_at = NOW(),
    updated_at = NOW()
FROM candidates c
WHERE p.id = c.id
RETURNING p.id, p.job_type, p.plan_id, p.user_id, p.plan_revision, p.dedup_key, p.payload,
          p.attempts, p.claimed_by, p.created_at
"""


def claim_plan_jobs(limit: int, claimed_by: str, job_types: list[str]) -> list[dict]:
    if not job_types:
        return []
    from db import execute_sql_write
    rows = execute_sql_write(CLAIM_SQL, (list(job_types), int(limit), claimed_by), returning=True) or []
    return [dict(r) for r in rows]


def heartbeat_plan_job(job_id: str, claimed_by: str, attempts: int) -> bool:
    try:
        from db import execute_sql_write
        rows = execute_sql_write(
            "UPDATE plan_jobs SET heartbeat_at = NOW(), updated_at = NOW() "
            "WHERE id = %s AND claimed_by = %s AND attempts = %s AND status = 'processing' RETURNING id",
            (job_id, claimed_by, int(attempts)), returning=True,
        ) or []
        return bool(rows)
    except Exception:
        return False


RECLAIM_SQL = """
UPDATE plan_jobs
SET status = CASE WHEN attempts >= %s THEN 'dead' ELSE 'failed' END,
    dead_lettered_at = CASE WHEN attempts >= %s THEN NOW() ELSE dead_lettered_at END,
    error_code = 'heartbeat_stale',
    error_redacted = 'worker sin heartbeat (reclaim)',
    execute_after = NOW(),
    updated_at = NOW()
WHERE status = 'processing'
  AND heartbeat_at < NOW() - (%s * INTERVAL '1 second')
RETURNING id, status
"""


def reclaim_stale_processing() -> int:
    """Un `processing` sin heartbeat vuelve a la cola (o muere si agotó intentos)."""
    try:
        from db import execute_sql_write
        m = max_attempts()
        rows = execute_sql_write(RECLAIM_SQL, (m, m, heartbeat_stale_s()), returning=True) or []
        if rows:
            logger.warning(f"[ARQ25-F5] reclaim: {len(rows)} job(s) processing sin heartbeat → {sorted({r['status'] for r in rows})}")
        return len(rows)
    except Exception as e:
        logger.debug(f"[ARQ25-F5] reclaim_stale_processing falló: {e!r}")
        return 0


FINISH_SQL = """
UPDATE plan_jobs
SET status = %s,
    execute_after = CASE WHEN %s = 'failed' THEN NOW() + (%s * INTERVAL '1 second') ELSE execute_after END,
    processed_at = CASE WHEN %s IN ('done', 'stale') THEN NOW() ELSE processed_at END,
    dead_lettered_at = CASE WHEN %s = 'dead' THEN NOW() ELSE dead_lettered_at END,
    error_code = %s,
    error_redacted = %s,
    payload = payload || %s,
    updated_at = NOW()
WHERE id = %s AND claimed_by = %s AND attempts = %s AND status = 'processing'
RETURNING id
"""


def finish_plan_job(
    job: dict,
    status: str,
    *,
    error_code: Optional[str] = None,
    error_redacted: Optional[str] = None,
    result: Optional[dict] = None,
) -> bool:
    """Commit fenced. `failed` con `attempts >= max_attempts()` se convierte en `dead`.
    Devuelve False si el fencing rechazó (otro worker reclamó la fila): `fencing_rejected`."""
    assert status in ("done", "failed", "dead", "stale"), status
    attempts = int(job.get("attempts") or 0)
    if status == "failed" and attempts >= max_attempts():
        status = "dead"
    backoff = backoff_seconds(attempts) if status == "failed" else 0
    redacted = (error_redacted or "")[:240] or None
    try:
        from db import execute_sql_write
        from psycopg.types.json import Jsonb
        rows = execute_sql_write(
            FINISH_SQL,
            (status, status, backoff, status, status, error_code, redacted,
             Jsonb({"result": result or {}, "finished_status": status}), str(job["id"]), job.get("claimed_by"), attempts),
            returning=True,
        ) or []
        ok = bool(rows)
        if not ok:
            logger.warning(f"[ARQ25-F5] fencing_rejected job={job.get('id')} attempts={attempts} status→{status}")
        if status == "dead":
            logger.error(f"[ARQ25-F5] job DEAD type={job.get('job_type')} plan={job.get('plan_id')} attempts={attempts} error={error_code}")
        return ok
    except Exception as e:
        logger.warning(f"[ARQ25-F5] finish_plan_job falló job={job.get('id')}: {e!r}")
        return False


# ----------------------------------------------------------------------------- consumidores
def _consume_display_i18n(job: dict) -> tuple[str, Optional[str], dict]:
    """(status, error_code, result). Comprueba la revisión vigente ANTES de escribir (I13)."""
    payload = job.get("payload") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    locale = str(payload.get("locale") or "")
    day_indices = payload.get("day_indices")
    plan_id = str(job.get("plan_id") or "")
    user_id = str(job.get("user_id") or "")
    if not locale or not plan_id or not user_id:
        return "dead", "bad_payload", {"payload": payload}
    job_rev = job.get("plan_revision")
    cur = current_plan_revision(plan_id)
    if cur is None:
        return "done", None, {"reason": "plan_gone"}  # plan borrado: nada que traducir (CASCADE lo limpiará)
    if job_rev is not None and int(job_rev) != int(cur):
        # la proyección ya no corresponde: stale + re-encolar para la revisión vigente
        key = display_i18n_dedup_key(plan_id, cur, locale, day_indices)
        new_id = enqueue_plan_job(
            JOB_TYPE_DISPLAY_I18N, plan_id, user_id, plan_revision=cur, dedup_key=key,
            payload={"locale": locale, "day_indices": day_indices, "schema_version": 1, "requeued_from": str(job.get("id"))},
            wake=False,
        )
        return "stale", "revision_changed", {"job_revision": job_rev, "current_revision": cur, "requeued_job_id": new_id}
    from plan_display_i18n import enrich_plan_display
    result = enrich_plan_display(plan_id, user_id, locale, day_indices=day_indices)
    status, error_code = verdict_for_display_result(result)
    return status, error_code, {"enrich": result if isinstance(result, dict) else str(result)[:200]}


# ----------------------------------------------------------------------------- consumidor: shopping_projection
# [P1-ARQ25-F5-SHOPPING-PROJECTION · 2026-09-04] Las listas 7/15/30 como PROYECCIÓN (roadmap §5.7):
# la Fase 3 encola `shopping_projection` con las ventanas de la política (principal + top-ups de
# frescos); este consumidor las materializa con el MISMO agregador que `/recalculate-shopping-list`
# (paquetes, precios, Nevera descontada) por revisión, y guarda el read model en `payload.result`
# del job (no en `plan_data`: escribir ahí bumpea `revision` y haría stale a la propia proyección).
# La lista síncrona no cambia. Lo lee `GET /api/plans/{plan_id}/projections`.
_PROJECTION_ROW_KEYS = ("name", "display_string", "display_name_en", "category", "display_category", "market_qty",
                        "market_unit", "base_qty", "base_unit", "is_perishable", "is_staple", "package_grams")
_PROJECTION_MAX_ITEMS = 150


def _duration_for_days(days: int) -> str:
    d = int(days or 0)
    return "weekly" if d <= 7 else ("biweekly" if d <= 15 else "monthly")


def _compact_row(row: dict) -> dict:
    out = {k: row.get(k) for k in _PROJECTION_ROW_KEYS if row.get(k) is not None}
    cost = row.get("estimated_cost", row.get("cost_rd"))
    if isinstance(cost, (int, float)):
        out["cost_rd"] = round(float(cost), 2)
    return out


def _load_plan_for_projection(plan_id: str, user_id: str) -> tuple:
    from db import execute_sql_query
    row = execute_sql_query("SELECT plan_data, revision FROM meal_plans WHERE id = %s AND user_id = %s",
                            (plan_id, user_id), fetch_one=True)
    if not row:
        return None, None
    pd = row.get("plan_data")
    if isinstance(pd, str):
        try:
            pd = json.loads(pd)
        except Exception:
            pd = None
    rev = int(row["revision"]) if row.get("revision") is not None else None
    return (pd if isinstance(pd, dict) else None), rev


def build_shopping_projection(plan_data: dict, user_id: str, windows: list, *, revision: Optional[int],
                              policy_hash: Optional[str] = None, max_items: int = _PROJECTION_MAX_ITEMS) -> dict:
    """Read model: una lista por ventana. Principal = ciclo completo (multiplicador de ciclo, como el
    recálculo); `fresh_only` = solo perecederos de ese top-up. Determinista, cero LLM."""
    import shopping_calculator as sc
    try:
        hm = float(plan_data.get("calc_household_multiplier") or 1.0)
    except (TypeError, ValueError):
        hm = 1.0
    hm = max(0.5, min(10.0, hm))
    inv, cons = sc.fetch_inventory_and_consumed_for_plan(user_id, plan_data, is_new_plan=True)
    try:
        trip = sc.active_trip_window_days(plan_data)
    except Exception:
        trip = None
    out_windows = []
    for w in windows or []:
        if not isinstance(w, dict):
            continue
        days = int(w.get("days") or 0)
        if days <= 0:
            continue
        fresh_only = bool(w.get("fresh_only"))
        mult = hm if fresh_only else hm * float(sc.cycle_qty_multiplier(_duration_for_days(days)) or 1.0)
        rows = sc.get_shopping_list_delta(
            user_id, plan_data, is_new_plan=True, structured=True, multiplier=mult,
            inventory_override=inv, consumed_override=cons, cycle_days=days, window_days=trip,
        ) or []
        rows = [r for r in rows if isinstance(r, dict)]
        if fresh_only:
            rows = [r for r in rows if r.get("is_perishable")]
        items = [_compact_row(r) for r in rows][:max_items]
        cost = round(sum(float(r.get("estimated_cost") or r.get("cost_rd") or 0) for r in rows), 2)
        out_windows.append({
            "kind": str(w.get("kind") or ("fresh_topup" if fresh_only else "main")),
            "start_day": int(w.get("start_day") or 0), "end_day": int(w.get("end_day") or days),
            "days": days, "cycle_days": int(w.get("cycle_days") or days), "fresh_only": fresh_only,
            "item_count": len(items), "cost_rd": cost, "items": items,
        })
    return {
        "schema_version": 1, "revision": revision, "policy_hash": policy_hash,
        "household_multiplier": hm, "computed_at": datetime.now(timezone.utc).isoformat(),
        "windows": out_windows,
    }


def _consume_shopping_projection(job: dict) -> tuple[str, Optional[str], dict]:
    payload = job.get("payload") or {}
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    windows = payload.get("windows") or []
    plan_id = str(job.get("plan_id") or "")
    user_id = str(job.get("user_id") or "")
    if not windows or not plan_id or not user_id:
        return "dead", "bad_payload", {"payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else []}
    plan_data, cur = _load_plan_for_projection(plan_id, user_id)
    if plan_data is None:
        return "done", None, {"reason": "plan_gone"}
    job_rev = job.get("plan_revision")
    if job_rev is not None and cur is not None and int(job_rev) != int(cur):
        suffix = str(job.get("dedup_key") or "").rsplit(":", 1)[-1] or "policy"
        new_id = enqueue_plan_job(
            JOB_TYPE_SHOPPING_PROJECTION, plan_id, user_id, plan_revision=cur,
            dedup_key=f"{JOB_TYPE_SHOPPING_PROJECTION}:{plan_id}:{cur}:{suffix}",
            payload={k: v for k, v in payload.items() if k != "result"} | {"requeued_from": str(job.get("id"))}, wake=False,
        )
        return "stale", "revision_changed", {"job_revision": job_rev, "current_revision": cur, "requeued_job_id": new_id}
    projection = build_shopping_projection(plan_data, user_id, windows, revision=cur, policy_hash=payload.get("policy_hash"))
    if not projection.get("windows"):
        return "failed", "empty_projection", {"windows_in": len(windows)}
    return "done", None, {"projection": projection}


def classify_projection_jobs(current_revision: Optional[int], jobs: list) -> dict:
    """Estado UI de la proyección a partir de los jobs del plan (más reciente primero). Puro."""
    cur = int(current_revision or 0)

    def _proj(j):
        pl = j.get("payload") or {}
        if isinstance(pl, str):
            try:
                pl = json.loads(pl)
            except Exception:
                pl = {}
        return ((pl.get("result") or {}).get("projection")) if isinstance(pl, dict) else None

    for j in jobs:
        if j.get("status") == "done" and int(j.get("plan_revision") or -1) == cur and _proj(j):
            return {"status": "ready", "revision": cur, "job_id": str(j.get("id")), "projection": _proj(j)}
    for j in jobs:
        if j.get("status") in ("pending", "processing"):
            return {"status": "pending", "revision": cur, "job_id": str(j.get("id")), "attempts": int(j.get("attempts") or 0)}
    for j in jobs:
        if j.get("status") == "failed":
            return {"status": "failed", "revision": cur, "job_id": str(j.get("id")), "attempts": int(j.get("attempts") or 0),
                    "error_code": j.get("error_code"), "retrying": True}
    for j in jobs:
        if j.get("status") == "done" and _proj(j):
            return {"status": "stale", "revision": cur, "projection_revision": j.get("plan_revision"),
                    "job_id": str(j.get("id")), "projection": _proj(j)}
    for j in jobs:
        if j.get("status") == "dead":
            return {"status": "failed", "revision": cur, "job_id": str(j.get("id")), "error_code": j.get("error_code"), "retrying": False}
    return {"status": "none", "revision": cur}


CONSUMERS: dict[str, Callable[[dict], tuple[str, Optional[str], dict]]] = {
    JOB_TYPE_DISPLAY_I18N: _consume_display_i18n,
    JOB_TYPE_SHOPPING_PROJECTION: _consume_shopping_projection,
}


def enabled_consumers() -> list[str]:
    return [t for t in CONSUMERS if consumer_enabled(t)]


def _emit_metric(job: dict, status: str, duration_ms: int, error_code: Optional[str]) -> None:
    try:
        from db import execute_sql_write
        from psycopg.types.json import Jsonb
        meta = {
            "job_type": job.get("job_type"), "status": status, "attempts": int(job.get("attempts") or 0),
            "plan_id": str(job.get("plan_id") or ""), "plan_revision": job.get("plan_revision"),
            "error_code": error_code, "claimed_by": job.get("claimed_by"),
        }
        execute_sql_write(
            "INSERT INTO pipeline_metrics (user_id, session_id, node, duration_ms, retries, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (job.get("user_id"), None, METRIC_NODE, int(duration_ms), int(job.get("attempts") or 0), Jsonb(meta)),
        )
    except Exception as e:
        logger.debug(f"[ARQ25-F5] pipeline_metrics falló (best-effort): {e!r}")


def process_plan_jobs() -> dict:
    """Un tick del worker: reclaim de zombis + claim de un lote + consumo + commit fenced. Nunca lanza."""
    summary: dict[str, Any] = {"claimed": 0, "done": 0, "failed": 0, "dead": 0, "stale": 0, "fencing_rejected": 0, "reclaimed": 0}
    if not plan_jobs_enabled():
        summary["skipped"] = "knob_off"
        return summary
    try:
        summary["reclaimed"] = reclaim_stale_processing()
        types = enabled_consumers()
        if not types:
            summary["skipped"] = "no_consumers"
            return summary
        me = _worker_identity()
        jobs = claim_plan_jobs(worker_batch(), me, types)
        summary["claimed"] = len(jobs)
        for job in jobs:
            t0 = time.monotonic()
            consumer = CONSUMERS.get(str(job.get("job_type")))
            try:
                status, error_code, result = consumer(job) if consumer else ("dead", "no_consumer", {})
            except Exception as e:  # el consumidor es fail-open; esto es el cinturón
                status, error_code, result = "failed", f"exception:{type(e).__name__}", {"error": str(e)[:200]}
            duration_ms = int((time.monotonic() - t0) * 1000)
            ok = finish_plan_job(job, status, error_code=error_code, error_redacted=(result or {}).get("error"), result=result)
            final = status if status != "failed" or int(job.get("attempts") or 0) < max_attempts() else "dead"
            if ok:
                summary[final] = summary.get(final, 0) + 1
            else:
                summary["fencing_rejected"] += 1
            _emit_metric(job, final if ok else "fencing_rejected", duration_ms, error_code)
        if jobs:
            logger.info(f"[ARQ25-F5] plan_jobs tick: {summary}")
    except Exception as e:
        logger.warning(f"[ARQ25-F5] process_plan_jobs falló (tick perdido): {e!r}")
        summary["error"] = str(e)[:120]
    return summary


__all__ = [
    "JOB_TYPE_DISPLAY_I18N", "JOB_TYPE_SHOPPING_PROJECTION", "JOB_STATUSES", "WORKER_JOB_ID", "METRIC_NODE",
    "plan_jobs_enabled", "consumer_enabled", "worker_batch", "max_attempts", "backoff_base_s",
    "heartbeat_stale_s", "worker_interval_s", "backoff_seconds", "display_i18n_dedup_key",
    "verdict_for_display_result", "current_plan_revision", "enqueue_plan_job", "maybe_enqueue_display_i18n",
    "wake_plan_jobs_worker", "claim_plan_jobs", "heartbeat_plan_job", "reclaim_stale_processing",
    "finish_plan_job", "CONSUMERS", "enabled_consumers", "process_plan_jobs",
    "build_shopping_projection", "classify_projection_jobs",
]
