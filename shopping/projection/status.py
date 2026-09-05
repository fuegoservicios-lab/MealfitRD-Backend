"""[P1-ARQ25-F5-SHOPPING-PROJECTION · 2026-09-04 · extraído P3-SHOPPING-PROJECTION-PKG] Estado UI de la
proyección a partir de los jobs del plan (más reciente primero). Puro; lo consume
`GET /api/plans/{plan_id}/projections` y, en el Dashboard, `ShoppingProjectionStatus`."""
from __future__ import annotations

import json
from typing import Optional


def classify_projection_jobs(current_revision: Optional[int], jobs: list) -> dict:
    """none / pending / ready / failed(retrying|dead) / stale (con la proyección vieja)."""
    cur = int(current_revision or 0)

    def _proj(j):
        pl = j.get("payload") or {}
        if isinstance(pl, str):
            try:
                pl = json.loads(pl)
            except Exception:
                pl = {}
        return ((pl.get("result") or {}).get("projection")) if isinstance(pl, dict) else None

    def _rev(j):
        # los jobs de la Fase 3 nacieron con plan_revision NULL: vale la revisión que la proyección declara
        if j.get("plan_revision") is not None:
            return int(j.get("plan_revision"))
        pr = _proj(j) or {}
        return int(pr.get("revision")) if pr.get("revision") is not None else -1

    for j in jobs:
        if j.get("status") == "done" and _rev(j) == cur and _proj(j):
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
            return {"status": "stale", "revision": cur, "projection_revision": _rev(j),
                    "job_id": str(j.get("id")), "projection": _proj(j)}
    for j in jobs:
        if j.get("status") == "dead":
            return {"status": "failed", "revision": cur, "job_id": str(j.get("id")), "error_code": j.get("error_code"), "retrying": False}
    return {"status": "none", "revision": cur}


__all__ = ["classify_projection_jobs"]
