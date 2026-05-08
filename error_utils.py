"""[P3-1 · 2026-05-07] Helpers de manejo seguro de errores.

Wrapper único usado por todos los routers para evitar leak de detalles
internos (Postgres exception text, schema names, SQL fragments) en el
campo `detail` de HTTPException — que viaja al cliente HTTP.

Knob `MEALFIT_LEAK_DB_ERRORS`:
- "true": expone str(exc) (modo dev / debugging local)
- cualquier otro valor (default): mensaje genérico + correlation_id

El caller debe seguir loggeando exc por su cuenta. Este helper sólo
controla el surface HTTP y emite un log paralelo con el correlation_id
para que SRE pueda correlacionar el ref del cliente con el stack trace.
"""

from __future__ import annotations

import logging
import os
import uuid

logger = logging.getLogger(__name__)


def _leak_enabled() -> bool:
    return os.environ.get("MEALFIT_LEAK_DB_ERRORS", "false").strip().lower() == "true"


def safe_error_detail(exc: BaseException, *, context: str = "") -> str:
    if _leak_enabled():
        return str(exc)
    cid = uuid.uuid4().hex[:8]
    logger.error(
        f"[SAFE-ERR cid={cid}] {context or 'unhandled'}: "
        f"{type(exc).__name__}: {exc}"
    )
    return f"Internal server error (ref: {cid})"
