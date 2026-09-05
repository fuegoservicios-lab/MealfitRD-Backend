"""[P3-SHOPPING-PROJECTION-PKG · 2026-09-05] Proyección de compras (Fase 5 del roadmap 2.5).

- `read_model`: `build_shopping_projection` — una lista por ventana (7/15/30) con el agregador del recálculo.
- `reprojection`: `shopping_list_fingerprint` + `enqueue_shopping_reprojection` — re-encolar solo si la lista cambió.
- `status`: `classify_projection_jobs` — none/pending/ready/failed/stale a partir de los jobs.

El outbox (claim/commit/consumidores) sigue en `plan_jobs.py`, que re-exporta estos nombres.
"""
from .read_model import build_shopping_projection
from .reprojection import enqueue_shopping_reprojection, shopping_list_fingerprint
from .status import classify_projection_jobs

__all__ = ["build_shopping_projection", "enqueue_shopping_reprojection", "shopping_list_fingerprint", "classify_projection_jobs"]
