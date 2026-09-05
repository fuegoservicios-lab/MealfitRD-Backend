"""[P1-ARQ25-F5-REPROJECTION · 2026-09-05 · extraído P3-SHOPPING-PROJECTION-PKG] Re-encolar la proyección cuando
la LISTA cambia (roadmap §5.7). La revisión del plan sube con CUALQUIER escritura de `plan_data` (también con
cada recálculo, que corre en cada visita al Dashboard), así que la revisión sola dispararía una proyección de
~1 min por visita. La huella de la lista semanal + hogar + días fuente decide: misma huella que el último job
⇒ la proyección sería idéntica ⇒ no se encola.

Las primitivas del outbox (`enqueue_plan_job`, `current_plan_revision`, knobs) se toman de `plan_jobs` en
tiempo de llamada (import perezoso: `plan_jobs` importa este paquete al cargar, y los tests parchean allí).
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_FINGERPRINT_ROW_KEYS = ("name", "base_qty", "base_unit", "market_qty", "market_unit")


def shopping_list_fingerprint(plan_data: dict) -> str:
    rows = plan_data.get("aggregated_shopping_list_weekly") if isinstance(plan_data, dict) else None
    parts = []
    for r in (rows or []):
        if isinstance(r, dict):
            parts.append(tuple(str(r.get(k)) for k in _FINGERPRINT_ROW_KEYS))
    parts.sort()
    days = 0
    if isinstance(plan_data, dict):
        days = len(plan_data.get("days") or []) + len(plan_data.get("_archived_days") or [])
    raw = json.dumps({"rows": parts, "hm": (plan_data or {}).get("calc_household_multiplier"), "days": days},
                     ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def enqueue_shopping_reprojection(plan_id: str, user_id: str, *, reason: str, plan_data: Optional[dict] = None) -> Optional[str]:
    """Encola `shopping_projection` para la revisión vigente si la lista cambió desde el último job.
    Fail-open: None si la cola está apagada, el plan no tiene política (pre-F3), la huella no cambió o
    algo falla. Llamada desde los commits de recálculo, swap, regeneración de día y relleno de bloques."""
    import plan_jobs as pj
    if not pj.plan_jobs_enabled() or not pj.consumer_enabled(pj.JOB_TYPE_SHOPPING_PROJECTION):
        return None
    if not pj._is_uuid(plan_id) or not pj._is_uuid(user_id):
        return None
    try:
        import horizon
        if not horizon.shopping_projection_jobs_enabled():
            return None
        if plan_data is None:
            plan_data, rev = pj._load_plan_for_projection(plan_id, user_id)
        else:
            rev = pj.current_plan_revision(plan_id)
        if not isinstance(plan_data, dict):
            return None
        eff = horizon.effective_policy_for_plan(plan_data)
        if not eff:
            return None  # sin política no hay ventanas: planes anteriores a la Fase 3
        fp = shopping_list_fingerprint(plan_data)
        from db import execute_sql_query
        last = execute_sql_query(
            "SELECT status, payload->>'list_fingerprint' AS fp FROM plan_jobs "
            "WHERE plan_id = %s AND user_id = %s AND job_type = %s AND status IN ('pending', 'processing', 'failed', 'done') "
            "ORDER BY created_at DESC LIMIT 1",
            (plan_id, user_id, pj.JOB_TYPE_SHOPPING_PROJECTION), fetch_one=True,
        )
        if last and last.get("fp") == fp:
            return None  # la lista no cambió: la proyección sería idéntica
        total = 0
        try:
            total = int(plan_data.get("total_days_requested") or 0)
        except (TypeError, ValueError):
            total = 0
        if total <= 0:
            try:
                from shopping_calculator import shopping_source_days
                total = len(shopping_source_days(plan_data) or [])
            except Exception:
                total = 0
        total = total or len(plan_data.get("days") or []) or 1
        windows = horizon.shopping_projection_windows(eff, total)
        payload = {
            "schema_version": horizon.BLUEPRINT_SCHEMA_VERSION, "policy_hash": eff.get("policy_hash"),
            "total_days": int(total), "windows": windows,
            "freezer_mode": str(((eff.get("shopping") or {}).get("freezer_mode")) or "limited"),
            "list_fingerprint": fp, "reason": str(reason or "")[:40],
        }
        key = f"{pj.JOB_TYPE_SHOPPING_PROJECTION}:{plan_id}:{int(rev or 0)}:{fp[:12]}"
        jid = pj.enqueue_plan_job(pj.JOB_TYPE_SHOPPING_PROJECTION, plan_id, user_id, plan_revision=rev, dedup_key=key, payload=payload)
        if jid:
            logger.info(f"[ARQ25-F5] reprojection encolada job={jid} plan={plan_id} rev={rev} reason={reason} fp={fp[:12]}")
        return jid
    except Exception as e:
        logger.debug(f"[ARQ25-F5] enqueue_shopping_reprojection no encolada ({reason}): {e!r}")
        return None


__all__ = ["shopping_list_fingerprint", "enqueue_shopping_reprojection", "_FINGERPRINT_ROW_KEYS"]
